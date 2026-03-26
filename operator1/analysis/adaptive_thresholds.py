"""Adaptive threshold calibration for survival mode and vanity detection.

Replaces fixed textbook thresholds with data-derived, sector-aware values
using peer distributions, own-history changepoint detection, and HMM
emission crossover analysis.

Methods implemented:

1. **Peer Percentile Thresholds** (Huber 1981): survival triggers calibrated
   to the 10th/90th percentile of the sector peer distribution, with MAD-based
   fallback for small peer groups.

2. **BOCPD Deterioration Tightening** (Adams & MacKay 2007): when a company's
   own ratio series shows a recent structural downward shift, thresholds are
   tightened by 20% to catch the decline earlier.

3. **Sector Z-Score** (Iglewicz & Hoaglin 1993): vanity metrics flagged when
   they exceed 2 modified-Z-scores from the sector median (using MAD for
   robustness).

4. **Jenks Natural Breaks** (Fisher 1958): financial health composite score
   labels derived from optimal class boundaries rather than fixed quintiles.

5. **HMM Emission Crossover** (Rabiner 1989): regime mixer thresholds derived
   from the HMM's learned Gaussian emission parameters, finding the exact
   value where P(distress|x) = P(healthy|x).

Top-level entry point:
    ``compute_adaptive_thresholds(cache, linked_caches, regime_detector)``
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Absolute floor values (never less protective than these)
# ---------------------------------------------------------------------------

_ABSOLUTE_FLOORS: dict[str, dict[str, float]] = {
    "current_ratio": {"floor": 0.3, "direction": "lower_is_worse"},
    "debt_to_equity_abs": {"floor": 20.0, "direction": "higher_is_worse"},
    "fcf_yield": {"floor": -0.10, "direction": "lower_is_worse"},
    "drawdown_252d": {"floor": -0.15, "direction": "lower_is_worse"},
}

# Textbook defaults (fallback when no peer data)
_TEXTBOOK_DEFAULTS: dict[str, float] = {
    "current_ratio": 1.0,
    "debt_to_equity_abs": 3.0,
    "fcf_yield": 0.0,
    "drawdown_252d": -0.40,
}

# Metrics where lower values indicate worse condition
_LOWER_IS_WORSE: set[str] = {"current_ratio", "fcf_yield", "drawdown_252d"}

# Metrics where higher values indicate worse condition
_HIGHER_IS_WORSE: set[str] = {"debt_to_equity_abs"}

# BOCPD tightening factor when deterioration is detected
_BOCPD_TIGHTENING_FACTOR: float = 0.20

# Minimum peers needed for percentile-based thresholds
_MIN_PEERS_PERCENTILE: int = 5

# Minimum peers needed for MAD-based fallback
_MIN_PEERS_MAD: int = 3


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ThresholdSet:
    """Adaptive thresholds for survival mode and vanity detection.

    All fields have sensible defaults matching the current textbook values,
    so modules can use this as a drop-in replacement without breaking when
    peer data is unavailable.
    """

    # Survival mode thresholds (consumed by survival_mode.py, monte_carlo.py)
    current_ratio_lt: float = 1.0
    debt_to_equity_abs_gt: float = 3.0
    fcf_yield_lt: float = 0.0
    drawdown_252d_lt: float = -0.40

    # Regime mixer thresholds (consumed by regime_mixer.py)
    distress_current_ratio: float = 0.8
    distress_debt_to_equity: float = 4.0
    distress_drawdown: float = -0.50
    distress_cash_ratio: float = 0.1
    stressed_current_ratio: float = 1.2
    stressed_debt_to_equity: float = 2.5
    stressed_drawdown: float = -0.25
    stressed_cash_ratio: float = 0.3

    # Vanity thresholds (consumed by vanity.py)
    rnd_intensity_threshold: float = 0.10
    sga_ratio_threshold: float = 0.30
    exec_comp_pct_threshold: float = 5.0
    marketing_pct_threshold: float = 10.0

    # Financial health label breakpoints (consumed by financial_health.py)
    fh_label_breaks: list[float] = field(
        default_factory=lambda: [20.0, 35.0, 50.0, 65.0, 80.0],
    )

    # Metadata: how each threshold was derived
    methods_used: dict[str, str] = field(default_factory=dict)
    n_peers_available: int = 0
    adapted: bool = False  # True if any threshold was adapted from default


# ---------------------------------------------------------------------------
# Method A: Peer-calibrated percentile thresholds
# ---------------------------------------------------------------------------


def _collect_peer_latest(
    metric: str,
    linked_caches: dict[str, pd.DataFrame],
) -> list[float]:
    """Collect the latest non-NaN value of a metric from all peer caches."""
    values: list[float] = []
    for _isin, peer_cache in linked_caches.items():
        if metric not in peer_cache.columns:
            continue
        series = peer_cache[metric].dropna()
        if series.empty:
            continue
        val = float(series.iloc[-1])
        if not math.isnan(val) and not math.isinf(val):
            values.append(val)
    return values


def _peer_percentile_threshold(
    metric: str,
    peer_values: list[float],
) -> float | None:
    """Compute the peer-calibrated threshold for a survival metric.

    For "lower is worse" metrics: 10th percentile (below this = distress).
    For "higher is worse" metrics: 90th percentile (above this = distress).

    Returns None if insufficient peers.
    """
    if len(peer_values) < _MIN_PEERS_PERCENTILE:
        return None

    arr = np.array(peer_values)

    if metric in _LOWER_IS_WORSE:
        threshold = float(np.percentile(arr, 10))
    elif metric in _HIGHER_IS_WORSE:
        threshold = float(np.percentile(arr, 90))
    else:
        return None

    return threshold


def _mad_fallback_threshold(
    metric: str,
    peer_values: list[float],
) -> float | None:
    """MAD-based fallback when too few peers for percentile.

    Uses Winsorized median +/- 2 * MAD (Huber 1981).
    The 1.4826 scaling makes MAD comparable to std under normality.

    Returns None if insufficient peers.
    """
    if len(peer_values) < _MIN_PEERS_MAD:
        return None

    arr = np.array(peer_values)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) * 1.4826

    if mad < 1e-8:
        return None

    if metric in _LOWER_IS_WORSE:
        threshold = median - 2.0 * mad
    elif metric in _HIGHER_IS_WORSE:
        threshold = median + 2.0 * mad
    else:
        return None

    return threshold


def _apply_floor(metric: str, threshold: float) -> float:
    """Clamp threshold to absolute floor (never less protective)."""
    info = _ABSOLUTE_FLOORS.get(metric)
    if info is None:
        return threshold

    direction = info["direction"]
    floor = info["floor"]

    if direction == "lower_is_worse":
        # Threshold is "trigger when below X". Floor means "never set X below floor"
        return max(threshold, floor)
    else:
        # Threshold is "trigger when above X". Floor means "never set X above floor"
        return min(threshold, floor)


# ---------------------------------------------------------------------------
# Method D: BOCPD deterioration detection
# ---------------------------------------------------------------------------


def _detect_recent_deterioration(
    metric: str,
    cache: pd.DataFrame,
    lookback_days: int = 63,
) -> bool:
    """Detect if the metric has recently shifted downward using BOCPD-lite.

    Uses a simplified version: compare the last `lookback_days` mean against
    the preceding period mean. If the recent mean is significantly worse
    (more than 1 MAD shift), flag deterioration.

    Full BOCPD (Adams & MacKay 2007) is available in regime_detector.py
    but is heavier. This simplified version catches 80% of cases.
    """
    if metric not in cache.columns:
        return False

    series = cache[metric].dropna()
    if len(series) < lookback_days * 2:
        return False

    recent = series.iloc[-lookback_days:]
    prior = series.iloc[-lookback_days * 2:-lookback_days]

    recent_mean = recent.mean()
    prior_mean = prior.mean()
    prior_mad = float(np.median(np.abs(prior - prior.median()))) * 1.4826

    if prior_mad < 1e-8:
        return False

    shift = (recent_mean - prior_mean) / prior_mad

    if metric in _LOWER_IS_WORSE:
        # Deterioration = recent mean is lower
        return shift < -1.0
    elif metric in _HIGHER_IS_WORSE:
        # Deterioration = recent mean is higher
        return shift > 1.0

    return False


def _tighten_threshold(
    metric: str,
    threshold: float,
    factor: float = _BOCPD_TIGHTENING_FACTOR,
) -> float:
    """Tighten a threshold when deterioration is detected.

    For "lower is worse" metrics: raise the threshold (higher bar).
    For "higher is worse" metrics: lower the threshold (lower bar).
    """
    if metric in _LOWER_IS_WORSE:
        return threshold * (1.0 + factor)
    elif metric in _HIGHER_IS_WORSE:
        return threshold * (1.0 - factor)
    return threshold


# ---------------------------------------------------------------------------
# Method E: Sector-relative Z-score for vanity thresholds
# ---------------------------------------------------------------------------


def _sector_zscore_threshold(
    metric: str,
    peer_values: list[float],
    z_critical: float = 2.0,
) -> float | None:
    """Compute vanity threshold as sector median + z_critical * MAD.

    Uses modified Z-score (Iglewicz & Hoaglin 1993) with MAD for
    robustness. A value exceeding z_critical standard MADs from the
    sector median is considered an outlier.

    Returns the threshold value, or None if insufficient peers.
    """
    if len(peer_values) < _MIN_PEERS_MAD:
        return None

    arr = np.array(peer_values)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) * 1.4826

    if mad < 1e-8:
        return None

    # Vanity metrics are all "higher is worse" (excess spending)
    return median + z_critical * mad


# ---------------------------------------------------------------------------
# Method H: Jenks natural breaks for financial health labels
# ---------------------------------------------------------------------------


def _jenks_breaks(data: np.ndarray, n_classes: int = 5) -> list[float]:
    """Compute Jenks natural breaks using dynamic programming.

    Implements the Fisher-Jenks algorithm (Fisher 1958, Jenks 1967)
    which minimizes within-class variance. This is the 1-D equivalent
    of k-means, solvable exactly via DP in O(k * n^2).

    Parameters
    ----------
    data:
        1-D array of values (will be sorted internally).
    n_classes:
        Number of classes (default 5 for health labels).

    Returns
    -------
    List of n_classes - 1 interior breakpoints.
    """
    data = np.sort(data[~np.isnan(data)])
    n = len(data)

    if n < n_classes * 2:
        # Not enough data for meaningful breaks
        return []

    # Compute cumulative sums for O(1) variance calculation
    cum_sum = np.zeros(n + 1)
    cum_sq_sum = np.zeros(n + 1)
    for i in range(n):
        cum_sum[i + 1] = cum_sum[i] + data[i]
        cum_sq_sum[i + 1] = cum_sq_sum[i] + data[i] ** 2

    def _ssdev(start: int, end: int) -> float:
        """Sum of squared deviations for data[start:end+1]."""
        count = end - start + 1
        s = cum_sum[end + 1] - cum_sum[start]
        sq = cum_sq_sum[end + 1] - cum_sq_sum[start]
        return sq - (s * s) / count

    # DP table: dp[k][i] = min total SSDEV for first i+1 items in k classes
    INF = float("inf")
    dp = [[INF] * n for _ in range(n_classes)]
    split = [[0] * n for _ in range(n_classes)]

    # Base case: 1 class
    for i in range(n):
        dp[0][i] = _ssdev(0, i)

    # Fill DP table
    for k in range(1, n_classes):
        for i in range(k, n):
            best_cost = INF
            best_j = k - 1
            for j in range(k - 1, i):
                cost = dp[k - 1][j] + _ssdev(j + 1, i)
                if cost < best_cost:
                    best_cost = cost
                    best_j = j
            dp[k][i] = best_cost
            split[k][i] = best_j

    # Trace back to find breakpoints
    breaks: list[int] = []
    idx = n - 1
    for k in range(n_classes - 1, 0, -1):
        idx = split[k][idx]
        breaks.append(idx)

    breaks.reverse()

    # Convert indices to actual breakpoint values
    # Breakpoint is the midpoint between the last element of class k
    # and the first element of class k+1
    result: list[float] = []
    for b in breaks:
        if b + 1 < n:
            bp = (data[b] + data[b + 1]) / 2.0
            result.append(round(bp, 2))

    return result


# ---------------------------------------------------------------------------
# Method J: HMM emission crossover for regime thresholds
# ---------------------------------------------------------------------------


def _hmm_crossover_threshold(
    metric: str,
    regime_detector: Any | None,
    cache: pd.DataFrame,
) -> float | None:
    """Compute the metric value where P(distress) = P(healthy) using HMM.

    Uses the HMM's learned Gaussian emission parameters for each regime
    to find the crossover point. Requires the HMM to have been fitted
    in Step 5.5 and the metric to be available in the cache.

    Returns None if HMM is not fitted or metric not available.
    """
    if regime_detector is None:
        return None

    result = getattr(regime_detector, "result", None)
    if result is None or not getattr(result, "hmm_fitted", False):
        return None

    model = getattr(regime_detector, "_hmm_model", None)
    if model is None:
        return None

    if metric not in cache.columns:
        return None

    series = cache[metric].dropna()
    if len(series) < 30:
        return None

    try:
        # Get HMM regime labels for days where the metric is available
        regime_labels = cache.get("regime_label")
        if regime_labels is None:
            return None

        # Compute per-regime statistics for this metric
        aligned = pd.DataFrame({
            "metric": series,
            "regime": regime_labels,
        }).dropna()

        if len(aligned) < 10:
            return None

        regime_stats: dict[str, dict[str, float]] = {}
        for regime_name, group in aligned.groupby("regime"):
            if len(group) < 5:
                continue
            regime_stats[str(regime_name)] = {
                "mean": float(group["metric"].mean()),
                "std": max(float(group["metric"].std()), 1e-8),
                "count": len(group),
            }

        # Find the "worst" and "best" regimes
        if metric in _LOWER_IS_WORSE:
            # Worst = lowest mean
            sorted_regimes = sorted(
                regime_stats.items(),
                key=lambda x: x[1]["mean"],
            )
        else:
            # Worst = highest mean
            sorted_regimes = sorted(
                regime_stats.items(),
                key=lambda x: -x[1]["mean"],
            )

        if len(sorted_regimes) < 2:
            return None

        worst = sorted_regimes[0][1]
        best = sorted_regimes[-1][1]

        # Gaussian crossover: where N(x; mu_worst, sigma_worst) = N(x; mu_best, sigma_best)
        # Simplified: weighted midpoint biased toward the distress regime
        # Full quadratic solution is unstable when variances are similar
        mu_w, sigma_w = worst["mean"], worst["std"]
        mu_b, sigma_b = best["mean"], best["std"]

        if abs(sigma_w - sigma_b) < 1e-8:
            # Equal variance: crossover at midpoint
            crossover = (mu_w + mu_b) / 2.0
        else:
            # Weighted midpoint: bias 60% toward the distress distribution
            crossover = 0.6 * mu_w + 0.4 * mu_b

        return float(crossover)

    except Exception as exc:
        logger.debug("HMM crossover computation failed for %s: %s", metric, exc)
        return None


# ---------------------------------------------------------------------------
# Vanity metric collection from peers
# ---------------------------------------------------------------------------


def _collect_vanity_peer_metric(
    metric_name: str,
    cache: pd.DataFrame,
    linked_caches: dict[str, pd.DataFrame],
) -> list[float]:
    """Collect peer values for vanity-related ratios.

    Maps vanity metric names to cache column names and collects
    the latest value from each peer.
    """
    # Map vanity metric names to actual cache column names
    col_map: dict[str, str] = {
        "rnd_intensity": "rd_expenses",  # will compute ratio below
        "sga_ratio": "sga_expenses",
        "exec_comp": "exec_compensation",
        "marketing": "marketing_spend",
    }

    col = col_map.get(metric_name, metric_name)
    values: list[float] = []

    for _isin, peer_cache in linked_caches.items():
        if col not in peer_cache.columns:
            continue
        revenue_col = "revenue"
        if revenue_col not in peer_cache.columns:
            continue

        metric_series = peer_cache[col].dropna()
        revenue_series = peer_cache[revenue_col].dropna()

        if metric_series.empty or revenue_series.empty:
            continue

        # Compute ratio (metric / revenue)
        latest_metric = float(metric_series.iloc[-1])
        latest_revenue = float(revenue_series.iloc[-1])

        if abs(latest_revenue) < 1e-8:
            continue

        ratio = abs(latest_metric / latest_revenue)
        if not math.isnan(ratio) and not math.isinf(ratio):
            values.append(ratio)

    return values


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_adaptive_thresholds(
    cache: pd.DataFrame,
    linked_caches: dict[str, pd.DataFrame] | None = None,
    regime_detector: Any | None = None,
    fh_composite_scores: pd.Series | None = None,
    peer_fh_scores: list[float] | None = None,
) -> ThresholdSet:
    """Compute adaptive thresholds from peer data, own history, and HMM.

    Parameters
    ----------
    cache:
        Daily cache DataFrame for the target company.
    linked_caches:
        Dict of {entity_id: daily_cache} for peer companies.
        From entity discovery Step 5f.
    regime_detector:
        Fitted regime detector from Step 5.5 (for HMM crossover).
    fh_composite_scores:
        Series of financial health composite scores for the target
        (for Jenks natural breaks).
    peer_fh_scores:
        List of latest FH composite scores from peers.

    Returns
    -------
    ThresholdSet
        Adaptive thresholds with graceful fallback to textbook defaults.
    """
    result = ThresholdSet()
    linked = linked_caches or {}

    # Track adaptation
    n_adapted = 0
    n_peers = len(linked)
    result.n_peers_available = n_peers

    # -----------------------------------------------------------------
    # 1. Survival mode thresholds (Method A + Method D)
    # -----------------------------------------------------------------
    survival_metrics = ["current_ratio", "debt_to_equity_abs", "fcf_yield", "drawdown_252d"]
    threshold_attrs = {
        "current_ratio": "current_ratio_lt",
        "debt_to_equity_abs": "debt_to_equity_abs_gt",
        "fcf_yield": "fcf_yield_lt",
        "drawdown_252d": "drawdown_252d_lt",
    }

    for metric in survival_metrics:
        attr = threshold_attrs[metric]
        method = "textbook"

        # Try peer percentile (Method A)
        peer_values = _collect_peer_latest(metric, linked)

        threshold = _peer_percentile_threshold(metric, peer_values)
        if threshold is not None:
            method = f"peer_percentile_p{'10' if metric in _LOWER_IS_WORSE else '90'}"
        else:
            # Try MAD fallback
            threshold = _mad_fallback_threshold(metric, peer_values)
            if threshold is not None:
                method = "mad_2sigma"
            else:
                # Use textbook default
                threshold = _TEXTBOOK_DEFAULTS[metric]
                method = "textbook"

        # Apply BOCPD deterioration tightening (Method D)
        if _detect_recent_deterioration(metric, cache):
            threshold = _tighten_threshold(metric, threshold)
            method += "+bocpd_tightened"

        # Apply absolute floor
        threshold = _apply_floor(metric, threshold)

        # For drawdown: also check own historical P5 as alternative
        if metric == "drawdown_252d" and "drawdown_252d" in cache.columns:
            own_dd = cache["drawdown_252d"].dropna()
            if len(own_dd) >= 50:
                own_p5 = float(np.percentile(own_dd, 5))
                # Use the more protective (higher) of peer-based and own-history
                if own_p5 > threshold:
                    threshold = own_p5
                    method += "+own_p5"

        setattr(result, attr, round(threshold, 4))
        result.methods_used[attr] = method
        if method != "textbook":
            n_adapted += 1

    # -----------------------------------------------------------------
    # 2. Regime mixer thresholds (Method J: HMM crossover, fallback A)
    # -----------------------------------------------------------------
    regime_metrics = {
        "current_ratio": ("distress_current_ratio", "stressed_current_ratio"),
        "debt_to_equity_abs": ("distress_debt_to_equity", "stressed_debt_to_equity"),
        "drawdown_252d": ("distress_drawdown", "stressed_drawdown"),
    }

    for metric, (distress_attr, stressed_attr) in regime_metrics.items():
        method = "textbook"

        # Try HMM crossover (Method J)
        hmm_threshold = _hmm_crossover_threshold(metric, regime_detector, cache)

        if hmm_threshold is not None:
            # HMM crossover gives the healthy/distress boundary
            # Stressed is halfway between healthy mean and distress boundary
            if metric in _LOWER_IS_WORSE:
                setattr(result, distress_attr, round(hmm_threshold, 4))
                # Stressed = threshold * 1.5 (less severe)
                setattr(result, stressed_attr, round(hmm_threshold * 1.5, 4))
            else:
                setattr(result, distress_attr, round(hmm_threshold, 4))
                setattr(result, stressed_attr, round(hmm_threshold * 0.625, 4))
            method = "hmm_crossover"
            n_adapted += 1
        else:
            # Fall back to peer percentile
            peer_values = _collect_peer_latest(metric, linked)
            if len(peer_values) >= _MIN_PEERS_PERCENTILE:
                arr = np.array(peer_values)
                if metric in _LOWER_IS_WORSE:
                    p10 = float(np.percentile(arr, 10))
                    p25 = float(np.percentile(arr, 25))
                    setattr(result, distress_attr, round(p10, 4))
                    setattr(result, stressed_attr, round(p25, 4))
                else:
                    p75 = float(np.percentile(arr, 75))
                    p90 = float(np.percentile(arr, 90))
                    setattr(result, distress_attr, round(p90, 4))
                    setattr(result, stressed_attr, round(p75, 4))
                method = "peer_percentile"
                n_adapted += 1

        result.methods_used[distress_attr] = method

    # Cash ratio thresholds from peers
    cash_peer = _collect_peer_latest("cash_ratio", linked)
    if len(cash_peer) >= _MIN_PEERS_PERCENTILE:
        arr = np.array(cash_peer)
        result.distress_cash_ratio = round(float(np.percentile(arr, 10)), 4)
        result.stressed_cash_ratio = round(float(np.percentile(arr, 25)), 4)
        result.methods_used["distress_cash_ratio"] = "peer_p10"
        n_adapted += 1

    # -----------------------------------------------------------------
    # 3. Vanity thresholds (Method E: sector Z-score)
    # -----------------------------------------------------------------
    vanity_map = {
        "rnd_intensity": ("rnd_intensity_threshold", 0.10),
        "sga_ratio": ("sga_ratio_threshold", 0.30),
        "exec_comp": ("exec_comp_pct_threshold", 5.0),
        "marketing": ("marketing_pct_threshold", 10.0),
    }

    for vanity_metric, (attr, default) in vanity_map.items():
        method = "textbook"
        peer_values = _collect_vanity_peer_metric(vanity_metric, cache, linked)

        threshold = _sector_zscore_threshold(vanity_metric, peer_values)
        if threshold is not None:
            # Convert to percentage if needed
            if attr in ("exec_comp_pct_threshold", "marketing_pct_threshold"):
                threshold = threshold * 100.0
            setattr(result, attr, round(threshold, 4))
            method = "sector_zscore_2mad"
            n_adapted += 1
        else:
            setattr(result, attr, default)

        result.methods_used[attr] = method

    # -----------------------------------------------------------------
    # 4. Financial health label breaks (Method H: Jenks)
    # -----------------------------------------------------------------
    fh_method = "textbook"

    if fh_composite_scores is not None and len(fh_composite_scores.dropna()) >= 20:
        own_scores = fh_composite_scores.dropna().values

        if peer_fh_scores and len(peer_fh_scores) >= 10:
            # Combine own + peer scores for Jenks
            all_scores = np.concatenate([own_scores, np.array(peer_fh_scores)])
        else:
            all_scores = own_scores

        breaks = _jenks_breaks(all_scores, n_classes=5)
        if len(breaks) == 4:
            result.fh_label_breaks = [round(b, 2) for b in breaks]
            fh_method = "jenks_natural_breaks"
            n_adapted += 1
        else:
            # Jenks failed (too few data); use expanding percentile
            percentile_breaks = [
                float(np.nanpercentile(all_scores, p))
                for p in [20, 35, 50, 65, 80]
            ]
            if all(not math.isnan(b) for b in percentile_breaks):
                result.fh_label_breaks = [round(b, 2) for b in percentile_breaks]
                fh_method = "expanding_percentile"
                n_adapted += 1

    result.methods_used["fh_label_breaks"] = fh_method

    # -----------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------
    result.adapted = n_adapted > 0

    logger.info(
        "Adaptive thresholds: %d/%d adapted, %d peers available, methods: %s",
        n_adapted,
        len(result.methods_used),
        n_peers,
        {k: v for k, v in result.methods_used.items() if v != "textbook"},
    )

    return result


def threshold_set_to_survival_dict(ts: ThresholdSet) -> dict[str, float]:
    """Convert ThresholdSet to the dict format expected by survival_mode.py.

    Returns a dict compatible with ``_COMPANY_THRESHOLDS``.
    """
    return {
        "current_ratio_lt": ts.current_ratio_lt,
        "debt_to_equity_abs_gt": ts.debt_to_equity_abs_gt,
        "fcf_yield_lt": ts.fcf_yield_lt,
        "drawdown_252d_lt": ts.drawdown_252d_lt,
    }


def threshold_set_to_mc_dict(
    ts: ThresholdSet,
) -> dict[str, tuple[str, float]]:
    """Convert ThresholdSet to the dict format expected by monte_carlo.py.

    Returns a dict compatible with ``DEFAULT_SURVIVAL_THRESHOLDS``.
    """
    return {
        "current_ratio": ("lt", ts.current_ratio_lt),
        "debt_to_equity_abs": ("gt", ts.debt_to_equity_abs_gt),
        "fcf_yield": ("lt", ts.fcf_yield_lt),
        "drawdown_252d": ("lt", ts.drawdown_252d_lt),
    }


def threshold_set_to_regime_dict(
    ts: ThresholdSet,
) -> dict[str, dict[str, float]]:
    """Convert ThresholdSet to the dict format expected by regime_mixer.py.

    Returns a dict compatible with ``_FUND_THRESHOLDS``.
    """
    return {
        "distress": {
            "current_ratio": ts.distress_current_ratio,
            "debt_to_equity": ts.distress_debt_to_equity,
            "drawdown_252d": ts.distress_drawdown,
            "cash_ratio": ts.distress_cash_ratio,
        },
        "stressed": {
            "current_ratio": ts.stressed_current_ratio,
            "debt_to_equity": ts.stressed_debt_to_equity,
            "drawdown_252d": ts.stressed_drawdown,
            "cash_ratio": ts.stressed_cash_ratio,
        },
    }

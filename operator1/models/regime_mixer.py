"""Soft regime switching and dual regime layer classification.

Implements:
1. Dual regime layers (market regime + fundamental regime) per Sec K.1
2. Soft switching via probability-weighted prediction blending per Sec K.2
3. Regime-weighted training windows per Sec L.1

Spec refs: Sec K.1, K.2, K.3, K.4, L.1
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
# Fundamental regime classifier
# ---------------------------------------------------------------------------

# Thresholds for fundamental regime classification
_FUND_THRESHOLDS = {
    "distress": {
        "current_ratio": 1.0,
        "debt_to_equity": 3.0,
        "drawdown_252d": -0.40,
    },
    "stressed": {
        "current_ratio": 1.5,
        "debt_to_equity": 2.0,
        "drawdown_252d": -0.20,
    },
}


@dataclass
class DualRegimeResult:
    """Container for dual-layer regime classification."""

    # Market regime (from HMM/GMM: bull, bear, high_vol, low_vol)
    market_regime_labels: pd.Series | None = None
    market_regime_probs: pd.DataFrame | None = None

    # Fundamental regime (from liquidity/solvency: healthy, stressed, distress)
    fund_regime_labels: pd.Series | None = None
    fund_regime_probs: pd.DataFrame | None = None

    # Combined soft weights for prediction blending
    # {regime_name: weight} per day -> stored as DataFrame
    blended_weights: pd.DataFrame | None = None

    fitted: bool = False
    error: str | None = None


def classify_fundamental_regime(
    cache: pd.DataFrame,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Classify each day into a fundamental regime based on financial ratios.

    Regimes:
    - ``healthy``: all ratios within safe bounds
    - ``stressed``: at least one ratio in warning zone
    - ``distress``: at least one ratio in critical zone (survival mode)

    Parameters
    ----------
    cache:
        Daily cache DataFrame with derived financial ratios.
    thresholds:
        Optional adaptive thresholds dict with "distress" and "stressed"
        sub-dicts. If None, uses the module-level ``_FUND_THRESHOLDS``.
        Format: ``{"distress": {"current_ratio": 0.8, ...}, "stressed": {...}}``

    Returns
    -------
    (labels, probabilities)
        ``labels`` is a Series of regime strings.
        ``probabilities`` is a DataFrame with columns
        [healthy, stressed, distress], each in [0, 1].
    """
    t = thresholds if thresholds is not None else _FUND_THRESHOLDS

    n = len(cache)
    labels = pd.Series("healthy", index=cache.index, dtype="object")
    probs = pd.DataFrame(
        {"healthy": 1.0, "stressed": 0.0, "distress": 0.0},
        index=cache.index,
    )

    # Score each dimension, tracking how many indicators are actually checked
    distress_score = pd.Series(0.0, index=cache.index)
    stress_score = pd.Series(0.0, index=cache.index)
    n_indicators = 0

    # Current ratio
    if "current_ratio" in cache.columns:
        cr = cache["current_ratio"]
        distress_score += (cr < t["distress"]["current_ratio"]).astype(float)
        stress_score += (
            (cr >= t["distress"]["current_ratio"])
            & (cr < t["stressed"]["current_ratio"])
        ).astype(float) * 0.5
        n_indicators += 1

    # Debt-to-equity
    for de_col in ("debt_to_equity_abs", "debt_to_equity"):
        if de_col in cache.columns:
            de = cache[de_col].abs()
            distress_score += (de > t["distress"]["debt_to_equity"]).astype(float)
            stress_score += (
                (de > t["stressed"]["debt_to_equity"])
                & (de <= t["distress"]["debt_to_equity"])
            ).astype(float) * 0.5
            n_indicators += 1
            break

    # Drawdown (uses equity_drawdown proxy when resolved via private company mode)
    if "drawdown_252d" in cache.columns:
        dd = cache["drawdown_252d"]
        distress_score += (dd < t["distress"]["drawdown_252d"]).astype(float)
        stress_score += (
            (dd >= t["distress"]["drawdown_252d"])
            & (dd < t["stressed"]["drawdown_252d"])
        ).astype(float) * 0.5
        n_indicators += 1

    # FCF yield
    if "fcf_yield" in cache.columns:
        distress_score += (cache["fcf_yield"] < 0).astype(float)
        n_indicators += 1

    # Cash ratio
    if "cash_ratio" in cache.columns:
        _cr_distress = t.get("distress", {}).get("cash_ratio", 0.1)
        _cr_stressed = t.get("stressed", {}).get("cash_ratio", 0.3)
        distress_score += (cache["cash_ratio"] < _cr_distress).astype(float) * 0.5
        stress_score += (
            (cache["cash_ratio"] >= _cr_distress) & (cache["cash_ratio"] < _cr_stressed)
        ).astype(float) * 0.3
        n_indicators += 1

    # Normalize scores to probabilities using the actual number of indicators
    # checked, not a hardcoded constant.  This prevents bias toward "healthy"
    # when some indicators are unavailable (e.g. private company mode before
    # proxy resolution, or markets with sparse data).
    max_possible = max(float(n_indicators), 1.0)
    distress_prob = (distress_score / max_possible).clip(0, 1)
    stress_prob = (stress_score / max_possible).clip(0, 1)
    healthy_prob = (1.0 - distress_prob - stress_prob).clip(0, 1)

    # Renormalize
    total = healthy_prob + stress_prob + distress_prob + 1e-10
    probs["healthy"] = healthy_prob / total
    probs["stressed"] = stress_prob / total
    probs["distress"] = distress_prob / total

    # Hard labels from max probability
    labels = probs.idxmax(axis=1)

    return labels, probs


def compute_dual_regimes(
    cache: pd.DataFrame,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> DualRegimeResult:
    """Compute dual regime layers (market + fundamental).

    The market regime comes from existing ``regime_hmm`` or
    ``regime_label`` columns in the cache (set by the regime detector).
    The fundamental regime is computed here from financial ratios.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with derived financial ratios and
        regime columns from the regime detector.
    thresholds:
        Optional adaptive thresholds for fundamental regime
        classification. Format: ``{"distress": {...}, "stressed": {...}}``.
        If None, uses module-level defaults.

    Returns
    -------
    DualRegimeResult with both regime layers and blended weights.
    """
    result = DualRegimeResult()

    if cache is None or cache.empty:
        result.error = "No data for dual regime classification"
        return result

    try:
        # Market regime: use existing columns
        if "regime_label" in cache.columns:
            result.market_regime_labels = cache["regime_label"].copy()
        elif "regime_hmm" in cache.columns:
            result.market_regime_labels = cache["regime_hmm"].astype(str)

        # Build market regime probability matrix if HMM probabilities exist
        market_regimes = ["bull", "bear", "high_vol", "low_vol"]
        hmm_prob_cols = [f"regime_hmm_prob_{i}" for i in range(4)]
        existing_prob_cols = [c for c in hmm_prob_cols if c in cache.columns]

        if existing_prob_cols:
            result.market_regime_probs = cache[existing_prob_cols].copy()
            result.market_regime_probs.columns = market_regimes[: len(existing_prob_cols)]
        elif result.market_regime_labels is not None:
            # Convert hard labels to one-hot probabilities
            result.market_regime_probs = pd.get_dummies(
                result.market_regime_labels, dtype=float,
            )

        # Fundamental regime
        result.fund_regime_labels, result.fund_regime_probs = (
            classify_fundamental_regime(cache, thresholds=thresholds)
        )

        # Store in cache for downstream use
        cache["regime_fundamental"] = result.fund_regime_labels

        # Compute blended weights
        result.blended_weights = _compute_blended_weights(
            result.market_regime_probs,
            result.fund_regime_probs,
        )

        result.fitted = True
        logger.info(
            "Dual regime classification complete: market=%s, fundamental=%s",
            result.market_regime_labels.value_counts().to_dict()
            if result.market_regime_labels is not None else "N/A",
            result.fund_regime_labels.value_counts().to_dict(),
        )

    except Exception as exc:
        result.error = str(exc)
        logger.warning("Dual regime classification failed: %s", exc)

    return result


def _compute_blended_weights(
    market_probs: pd.DataFrame | None,
    fund_probs: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Compute blended prediction weights from both regime layers.

    Each combined state (market_regime x fund_regime) gets a weight
    proportional to the product of its probabilities.
    """
    if fund_probs is None:
        return market_probs
    if market_probs is None:
        return fund_probs

    # Align indices
    common_idx = market_probs.index.intersection(fund_probs.index)
    mp = market_probs.loc[common_idx]
    fp = fund_probs.loc[common_idx]

    # Create combined weight columns
    cols: dict[str, pd.Series] = {}
    for mc in mp.columns:
        for fc in fp.columns:
            combined_name = f"{mc}_{fc}"
            cols[combined_name] = mp[mc] * fp[fc]

    blended = pd.DataFrame(cols, index=common_idx)

    # Normalize rows to sum to 1
    row_sums = blended.sum(axis=1).clip(lower=1e-10)
    blended = blended.div(row_sums, axis=0)

    return blended


# ---------------------------------------------------------------------------
# Soft regime-weighted predictions
# ---------------------------------------------------------------------------


def soft_regime_predict(
    predictions_per_regime: dict[str, float],
    regime_probabilities: dict[str, float],
) -> float:
    """Blend predictions from multiple regimes using soft switching.

    pred = Sum_r p(r|t) * pred_r

    Parameters
    ----------
    predictions_per_regime:
        {regime_name: point_forecast} for each regime-specific model.
    regime_probabilities:
        {regime_name: probability} for the current day.

    Returns
    -------
    Blended point forecast.
    """
    total_weight = 0.0
    blended = 0.0

    for regime, prob in regime_probabilities.items():
        if regime in predictions_per_regime and prob > 0:
            pred = predictions_per_regime[regime]
            if not math.isnan(pred):
                blended += prob * pred
                total_weight += prob

    if total_weight > 0:
        return blended / total_weight
    return float("nan")


# ---------------------------------------------------------------------------
# Regime-weighted training windows (Sec L.1)
# ---------------------------------------------------------------------------


def compute_regime_training_weights(
    cache: pd.DataFrame,
    current_day_idx: int,
    *,
    half_life_days: int = 126,
    regime_col: str = "regime_label",
) -> np.ndarray:
    """Compute regime-weighted training sample weights.

    w(tau) ~ exp(-delta_t / half_life) * similarity(regime(tau), regime(t))

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    current_day_idx:
        Index position of the current day in the cache.
    half_life_days:
        Exponential decay half-life in trading days.
    regime_col:
        Column name for regime labels.

    Returns
    -------
    Array of weights for days 0..current_day_idx, normalized to sum to 1.
    """
    n = current_day_idx + 1
    weights = np.ones(n, dtype=float)

    # Time decay: exponential
    decay_rate = math.log(2) / half_life_days
    for i in range(n):
        delta_t = current_day_idx - i
        weights[i] = math.exp(-decay_rate * delta_t)

    # Regime similarity bonus
    if regime_col in cache.columns:
        current_regime = cache[regime_col].iloc[current_day_idx]
        if pd.notna(current_regime):
            regimes = cache[regime_col].iloc[:n]
            similarity = (regimes == current_regime).astype(float)
            # Same regime gets 2x weight, different regime gets 1x
            weights *= (1.0 + similarity)

    # Also check fundamental regime if available
    fund_col = "regime_fundamental"
    if fund_col in cache.columns:
        current_fund = cache[fund_col].iloc[current_day_idx]
        if pd.notna(current_fund):
            fund_regimes = cache[fund_col].iloc[:n]
            fund_similarity = (fund_regimes == current_fund).astype(float)
            weights *= (1.0 + 0.5 * fund_similarity)

    # Normalize
    total = weights.sum()
    if total > 0:
        weights /= total

    return weights

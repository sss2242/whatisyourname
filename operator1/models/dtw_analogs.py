"""G1 -- Dynamic Time Warping (DTW) for historical analog discovery.

Finds past periods in the 2-year cache where the multi-variable pattern
was most similar to the current state.  Uses those historical outcomes
as empirical priors for prediction -- more robust than regime matching
alone because it considers the *shape* of the trajectory, not just a
regime label.

**Example insight:**
    "The last time this company had this debt ratio + this volatility +
    this macro environment was March 2020.  In the 30 days following
    that analog period, the stock declined 12% before recovering."

**How it works:**

1. Extract a recent window of multi-variable features (e.g., last 21 days).
2. Slide a window of the same length across the full history.
3. Compute DTW distance between the current window and each historical
   window.
4. Select the K closest analogs.
5. Examine what happened *after* each analog period (empirical outcomes).
6. Use those outcomes as priors for the forecast.

Falls back gracefully if ``dtaidistance`` or ``tslearn`` is not installed
-- uses Euclidean distance as a simpler alternative.

Top-level entry points:
    ``find_historical_analogs(cache, ...)``
    ``analog_based_forecast(cache, ...)``

Spec refs: Phase G enhancement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_QUERY_WINDOW: int = 21  # days in the query pattern
DEFAULT_K_ANALOGS: int = 5  # number of closest analogs to return
DEFAULT_FORECAST_HORIZON: int = 21  # days to look ahead from each analog
MIN_HISTORY_FOR_DTW: int = 100  # minimum cache rows needed


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class HistoricalAnalog:
    """A single historical analog period."""

    start_idx: int = 0
    end_idx: int = 0
    start_date: str = ""
    end_date: str = ""
    dtw_distance: float = float("inf")
    # What happened AFTER this analog period
    outcome_return: float = 0.0  # cumulative return in next N days
    outcome_max_drawdown: float = 0.0
    outcome_volatility: float = 0.0
    outcome_regime_at_end: str = ""
    narrative: str = ""


@dataclass
class DTWAnalogResult:
    """Collection of historical analog matches."""

    analogs: list[HistoricalAnalog] = field(default_factory=list)
    query_window_days: int = 0
    forecast_horizon_days: int = 0
    variables_used: list[str] = field(default_factory=list)
    method: str = ""  # "dtw", "euclidean"
    available: bool = False

    # Empirical forecast derived from analogs
    empirical_return_mean: float = 0.0
    empirical_return_median: float = 0.0
    empirical_return_p5: float = 0.0
    empirical_return_p95: float = 0.0
    empirical_drawdown_worst: float = 0.0


# ---------------------------------------------------------------------------
# DTW distance computation
# ---------------------------------------------------------------------------


def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Compute DTW distance between two multivariate sequences.

    Tries ``dtaidistance`` first, then ``tslearn``, then falls back to
    simple Euclidean distance.

    Parameters
    ----------
    seq_a, seq_b:
        2-D arrays of shape (time_steps, n_features).

    Returns
    -------
    Scalar distance (lower = more similar).
    """
    # Try dtaidistance (fast C implementation)
    try:
        from dtaidistance import dtw as dtw_lib  # type: ignore[import-untyped]

        # dtaidistance works on 1-D; for multivariate, sum per-feature DTW
        total_dist = 0.0
        for f in range(seq_a.shape[1]):
            d = dtw_lib.distance_fast(
                seq_a[:, f].astype(np.double),
                seq_b[:, f].astype(np.double),
            )
            total_dist += d
        return total_dist
    except (ImportError, Exception):
        pass

    # Try tslearn
    try:
        from tslearn.metrics import dtw as tslearn_dtw  # type: ignore[import-untyped]

        return float(tslearn_dtw(seq_a, seq_b))
    except (ImportError, Exception):
        pass

    # Fallback: simple Euclidean distance on aligned sequences
    min_len = min(len(seq_a), len(seq_b))
    a = seq_a[:min_len]
    b = seq_b[:min_len]
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _normalise_features(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalise each feature column.

    Returns (normalised, means, stds).
    """
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0)
    stds[stds < 1e-8] = 1.0
    normalised = (data - means) / stds
    np.nan_to_num(normalised, copy=False)
    return normalised, means, stds


# ---------------------------------------------------------------------------
# Core: Find historical analogs
# ---------------------------------------------------------------------------


def find_historical_analogs(
    cache: pd.DataFrame,
    variables: list[str] | None = None,
    *,
    query_window: int = DEFAULT_QUERY_WINDOW,
    k: int = DEFAULT_K_ANALOGS,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    min_gap: int = 10,
) -> DTWAnalogResult:
    """Find the K closest historical analogs to the current state.

    Parameters
    ----------
    cache:
        Full 2-year daily cache (DatetimeIndex).
    variables:
        Feature columns to use for DTW matching.  If None, uses a
        default set of key financial variables.
    query_window:
        Number of recent days to use as the query pattern.
    k:
        Number of closest analogs to return.
    forecast_horizon:
        Number of days to look ahead from each analog for outcomes.
    min_gap:
        Minimum gap (in days) between analog end and present, to avoid
        matching the query with itself.

    Returns
    -------
    DTWAnalogResult with ranked analogs and empirical forecast.
    """
    result = DTWAnalogResult(
        query_window_days=query_window,
        forecast_horizon_days=forecast_horizon,
    )

    if len(cache) < MIN_HISTORY_FOR_DTW:
        # Log once per distinct cache length to avoid spamming during
        # forward pass warmup (called hundreds of times with growing cache).
        _cache_len = len(cache)
        if not hasattr(find_historical_analogs, "_warned_lengths"):
            find_historical_analogs._warned_lengths = set()
        if _cache_len not in find_historical_analogs._warned_lengths:
            logger.warning("Insufficient data for DTW analogs (%d < %d)", _cache_len, MIN_HISTORY_FOR_DTW)
            find_historical_analogs._warned_lengths.add(_cache_len)
        return result

    # Select variables
    if variables is None:
        candidates = [
            "close", "return_1d", "volatility_21d", "volume",
            "debt_to_equity_abs", "current_ratio", "cash_ratio",
            "free_cash_flow_ttm_asof", "gross_margin", "operating_margin",
        ]
        variables = [v for v in candidates if v in cache.columns]

    if len(variables) < 2:
        logger.warning("Fewer than 2 DTW variables available")
        return result

    result.variables_used = variables

    # Extract and normalise feature matrix
    feature_matrix = cache[variables].values.astype(np.float64)
    normalised, _, _ = _normalise_features(feature_matrix)

    n_rows = len(normalised)
    if n_rows < query_window + min_gap + forecast_horizon:
        logger.warning("Cache too short for DTW with query_window=%d", query_window)
        return result

    # Query: the most recent `query_window` days
    query = normalised[-query_window:]

    # Slide across history and compute distances
    candidates: list[tuple[float, int, int]] = []  # (distance, start_idx, end_idx)
    search_end = n_rows - query_window - min_gap

    for start in range(0, search_end):
        end = start + query_window
        candidate = normalised[start:end]
        dist = _dtw_distance(query, candidate)
        candidates.append((dist, start, end))

    # Sort by distance (ascending)
    candidates.sort(key=lambda x: x[0])

    # Detect method used
    try:
        from dtaidistance import dtw  # type: ignore[import-untyped]  # noqa: F401
        result.method = "dtw_dtaidistance"
    except ImportError:
        try:
            from tslearn.metrics import dtw  # type: ignore[import-untyped]  # noqa: F401
            result.method = "dtw_tslearn"
        except ImportError:
            result.method = "euclidean_fallback"

    # Extract top-K analogs with outcomes
    has_close = "close" in cache.columns
    has_regime = "regime_label" in cache.columns
    returns_list: list[float] = []

    for dist, start, end in candidates[:k]:
        analog = HistoricalAnalog(
            start_idx=start,
            end_idx=end,
            dtw_distance=dist,
        )

        # Dates
        if hasattr(cache.index, "strftime"):
            analog.start_date = str(cache.index[start])[:10]
            analog.end_date = str(cache.index[end - 1])[:10]

        # Outcome: what happened in the next forecast_horizon days after this analog
        outcome_start = end
        outcome_end = min(end + forecast_horizon, n_rows)

        if has_close and outcome_end > outcome_start:
            prices = cache["close"].iloc[outcome_start:outcome_end].dropna()
            if len(prices) > 1:
                analog.outcome_return = float(prices.iloc[-1] / prices.iloc[0] - 1)
                running_max = prices.cummax()
                drawdowns = prices / running_max - 1
                analog.outcome_max_drawdown = float(drawdowns.min())
                if "return_1d" in cache.columns:
                    rets = cache["return_1d"].iloc[outcome_start:outcome_end].dropna()
                    if len(rets) > 1:
                        analog.outcome_volatility = float(rets.std() * np.sqrt(252))

                returns_list.append(analog.outcome_return)

        if has_regime and outcome_end - 1 < n_rows:
            analog.outcome_regime_at_end = str(cache["regime_label"].iloc[min(outcome_end - 1, n_rows - 1)])

        # Build narrative
        ret_pct = analog.outcome_return * 100
        direction = "gained" if ret_pct > 0 else "lost"
        analog.narrative = (
            f"Analog from {analog.start_date} to {analog.end_date} "
            f"(DTW distance: {dist:.2f}). In the {forecast_horizon} days "
            f"following, the stock {direction} {abs(ret_pct):.1f}% with "
            f"max drawdown of {analog.outcome_max_drawdown * 100:.1f}%."
        )

        result.analogs.append(analog)

    # Empirical forecast from analog outcomes
    if returns_list:
        arr = np.array(returns_list)
        result.empirical_return_mean = float(np.mean(arr))
        result.empirical_return_median = float(np.median(arr))
        result.empirical_return_p5 = float(np.percentile(arr, 5))
        result.empirical_return_p95 = float(np.percentile(arr, 95))
        result.empirical_drawdown_worst = float(
            min(a.outcome_max_drawdown for a in result.analogs)
        )

    result.available = len(result.analogs) > 0

    logger.info(
        "DTW analogs: found %d matches (method=%s), "
        "empirical return: mean=%.2f%%, median=%.2f%%, range=[%.2f%%, %.2f%%]",
        len(result.analogs),
        result.method,
        result.empirical_return_mean * 100,
        result.empirical_return_median * 100,
        result.empirical_return_p5 * 100,
        result.empirical_return_p95 * 100,
    )

    return result


def format_analogs_for_profile(dtw_result: DTWAnalogResult) -> dict[str, Any]:
    """Format DTW analog results for the company profile JSON.

    Returns a dict suitable for ``company_profile["historical_analogs"]``.
    """
    if not dtw_result.available:
        return {"available": False}

    return {
        "available": True,
        "method": dtw_result.method,
        "query_window_days": dtw_result.query_window_days,
        "forecast_horizon_days": dtw_result.forecast_horizon_days,
        "variables_used": dtw_result.variables_used,
        "n_analogs": len(dtw_result.analogs),
        "empirical_forecast": {
            "return_mean_pct": round(dtw_result.empirical_return_mean * 100, 2),
            "return_median_pct": round(dtw_result.empirical_return_median * 100, 2),
            "return_p5_pct": round(dtw_result.empirical_return_p5 * 100, 2),
            "return_p95_pct": round(dtw_result.empirical_return_p95 * 100, 2),
            "worst_drawdown_pct": round(dtw_result.empirical_drawdown_worst * 100, 2),
        },
        "analogs": [
            {
                "period": f"{a.start_date} to {a.end_date}",
                "dtw_distance": round(a.dtw_distance, 3),
                "outcome_return_pct": round(a.outcome_return * 100, 2),
                "outcome_max_drawdown_pct": round(a.outcome_max_drawdown * 100, 2),
                "narrative": a.narrative,
            }
            for a in dtw_result.analogs
        ],
    }

"""Shared helpers for the Hedge Fund Analysis pipeline.

Provides utilities for extracting quarterly/annual data from raw
statement DataFrames, loading HF config weights, and safe math.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "hedge_fund_weights.yml"
_config_cache: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_hf_config(*, reload: bool = False) -> dict[str, Any]:
    """Load hedge fund weights from config/hedge_fund_weights.yml."""
    global _config_cache
    if _config_cache is not None and not reload:
        return _config_cache
    try:
        import yaml
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
                _config_cache = yaml.safe_load(fh) or {}
        else:
            logger.debug("HF config not found at %s, using defaults", _CONFIG_PATH)
            _config_cache = {}
    except Exception as exc:
        logger.debug("Failed to load HF config: %s", exc)
        _config_cache = {}
    return _config_cache


def get_hf_weight(dotted_path: str, default: Any = None) -> Any:
    """Get a nested value from HF config using dot notation."""
    cfg = load_hf_config()
    keys = dotted_path.split(".")
    current = cfg
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


# ---------------------------------------------------------------------------
# Raw statement DataFrame extractors
# ---------------------------------------------------------------------------

def extract_quarterly_series(
    df: pd.DataFrame,
    column: str,
    n_periods: int = 8,
    date_col: str = "report_date",
) -> pd.Series:
    """Extract a time-ordered Series of quarterly values from a raw DF.

    Parameters
    ----------
    df:
        Raw statement DataFrame (income_df, balance_df, or cashflow_df)
        with one row per filing period.
    column:
        Column name to extract (e.g. ``"revenue"``, ``"total_assets"``).
    n_periods:
        Maximum number of most recent periods to return.
    date_col:
        Date column to use for ordering.

    Returns
    -------
    pd.Series indexed by report_date with the column values.
    Empty Series if column not found or DF is empty.
    """
    if df is None or df.empty or column not in df.columns:
        return pd.Series(dtype=float)

    work = df.copy()
    if date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col])
        work = work.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
        work = work.set_index(date_col)
    elif work.index.name == date_col or hasattr(work.index, "date"):
        work = work.sort_index()
    else:
        # No date column -- just use row order
        pass

    series = work[column].dropna()
    if len(series) > n_periods:
        series = series.iloc[-n_periods:]
    return series.astype(float)


def extract_latest_value(
    df: pd.DataFrame,
    column: str,
    date_col: str = "report_date",
) -> float | None:
    """Extract the most recent non-NaN value of a column from a raw DF."""
    series = extract_quarterly_series(df, column, n_periods=1, date_col=date_col)
    if series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    return float(val)


def extract_qoq_changes(
    df: pd.DataFrame,
    column: str,
    n_periods: int = 8,
    date_col: str = "report_date",
) -> pd.Series:
    """Extract quarter-over-quarter percentage changes.

    Returns a Series of pct_change values (e.g. 0.05 = +5% Q/Q).
    First value is NaN (no prior quarter).
    """
    series = extract_quarterly_series(df, column, n_periods=n_periods + 1, date_col=date_col)
    if len(series) < 2:
        return pd.Series(dtype=float)
    changes = series.pct_change().iloc[1:]  # drop first NaN
    return changes


def extract_yoy_changes(
    df: pd.DataFrame,
    column: str,
    n_periods: int = 8,
    date_col: str = "report_date",
) -> pd.Series:
    """Extract year-over-year changes (4-period lag for quarterly data).

    Returns pct_change with lag=4 (i.e. same quarter last year).
    """
    series = extract_quarterly_series(df, column, n_periods=n_periods + 4, date_col=date_col)
    if len(series) < 5:
        return pd.Series(dtype=float)
    yoy = series.pct_change(periods=4).dropna()
    return yoy


def compute_rolling_slope(series: pd.Series, window: int = 8) -> float | None:
    """Compute the OLS slope of the last ``window`` values.

    Returns the slope coefficient (positive = trending up).
    None if insufficient data.
    """
    if series is None or len(series) < max(3, window // 2):
        return None
    recent = series.dropna().iloc[-window:]
    if len(recent) < 3:
        return None
    x = np.arange(len(recent), dtype=float)
    y = recent.values.astype(float)
    # Simple OLS: slope = cov(x,y) / var(x)
    try:
        slope = np.polyfit(x, y, 1)[0]
        return float(slope) if np.isfinite(slope) else None
    except (np.linalg.LinAlgError, ValueError):
        return None


def safe_divide(
    numerator: float | None,
    denominator: float | None,
    default: float | None = None,
    epsilon: float = 1e-9,
) -> float | None:
    """Safe division with NaN/zero/tiny protection."""
    if numerator is None or denominator is None:
        return default
    try:
        n, d = float(numerator), float(denominator)
        if not np.isfinite(n) or not np.isfinite(d):
            return default
        if abs(d) < epsilon:
            return default
        result = n / d
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def normalize_score(value: float, floor: float = 0.0, ceiling: float = 100.0) -> float:
    """Clamp a score to [floor, ceiling]."""
    if not np.isfinite(value):
        return (floor + ceiling) / 2
    return max(floor, min(ceiling, value))


def get_cache_latest(cache: pd.DataFrame, column: str) -> float | None:
    """Get the latest non-NaN value from the daily cache."""
    if cache is None or cache.empty or column not in cache.columns:
        return None
    series = cache[column].dropna()
    if series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    return float(val)

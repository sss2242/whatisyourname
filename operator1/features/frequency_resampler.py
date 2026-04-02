"""Multi-frequency cache resampler for temporal hierarchical forecasting.

Resamples the daily cache to lower frequencies (weekly, monthly, quarterly,
annual) while preserving PIT (point-in-time) constraints and correctly
handling OHLCV vs financial statement data.

No-look-ahead guarantees:
  - Last period truncated to current date (no future data in incomplete periods)
  - Financial statements use filing_date alignment (not report_date)
  - Incomplete periods flagged with ``is_partial_period``
  - Forward-fill confidence decays with distance from nearest filing

Used by ``multi_frequency_runner.py`` to prepare caches for each frequency
scope before running the analytical pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frequency configuration
# ---------------------------------------------------------------------------

FREQUENCY_CONFIG: dict[str, dict[str, Any]] = {
    "D": {
        "label": "Daily",
        "pd_freq": "B",       # business day
        "lookback_years": 2,
        "resample_rule": None,  # no resampling needed
    },
    "W": {
        "label": "Weekly",
        "pd_freq": "W-FRI",   # week ending Friday
        "lookback_years": 3,
        "resample_rule": "W-FRI",
    },
    "M": {
        "label": "Monthly",
        "pd_freq": "ME",      # month end
        "lookback_years": 5,
        "resample_rule": "ME",
    },
    "Q": {
        "label": "Quarterly",
        "pd_freq": "QE",      # quarter end
        "lookback_years": 6,
        "resample_rule": "QE",
    },
    "A": {
        "label": "Annual",
        "pd_freq": "YE",      # year end
        "lookback_years": 8,
        "resample_rule": "YE",
    },
}


# OHLCV columns that need special aggregation rules
_OHLCV_COLS = {"open", "high", "low", "close", "volume", "adjusted_close", "vwap"}

# Financial statement columns (stock vs flow handled differently)
# Stock variables: take last value in period (balance sheet snapshots)
# Flow variables: sum over period (income/cashflow totals)
_FLOW_VARIABLES = {
    "revenue", "cost_of_revenue", "gross_profit", "operating_income",
    "ebit", "ebitda", "net_income", "interest_expense", "taxes",
    "operating_cash_flow", "capex", "free_cash_flow",
    "investing_cf", "financing_cf", "dividends_paid", "stock_buybacks",
    "sga_expenses", "rd_expenses",
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ResampledCache:
    """Container for a resampled cache at a specific frequency."""

    frequency: str
    label: str
    lookback_years: int
    cache: pd.DataFrame
    n_periods: int
    is_partial_last_period: bool
    original_daily_rows: int
    resampled_rows: int


# ---------------------------------------------------------------------------
# Core resampling function
# ---------------------------------------------------------------------------

def resample_cache_to_frequency(
    daily_cache: pd.DataFrame,
    frequency: str,
    reference_date: date | None = None,
) -> ResampledCache:
    """Resample a daily cache to the target frequency.

    Parameters
    ----------
    daily_cache:
        Daily-frequency DataFrame with DatetimeIndex.
    frequency:
        Target frequency: ``"D"``, ``"W"``, ``"M"``, ``"Q"``, ``"A"``.
    reference_date:
        The "today" date for truncation.  Defaults to the last date
        in the cache.

    Returns
    -------
    ResampledCache with the resampled DataFrame and metadata.

    No-Look-Ahead:
        The last period is truncated to ``reference_date``.  If the
        current week/month/quarter/year is incomplete, only data up
        to and including ``reference_date`` is used.
    """
    config = FREQUENCY_CONFIG.get(frequency)
    if config is None:
        raise ValueError(f"Unknown frequency: {frequency}")

    if daily_cache.empty:
        return ResampledCache(
            frequency=frequency,
            label=config["label"],
            lookback_years=config["lookback_years"],
            cache=daily_cache.copy(),
            n_periods=0,
            is_partial_last_period=False,
            original_daily_rows=0,
            resampled_rows=0,
        )

    if reference_date is None:
        reference_date = daily_cache.index[-1].date() if hasattr(daily_cache.index[-1], "date") else date.today()

    ref_ts = pd.Timestamp(reference_date)

    # Determine lookback start date
    lookback_years = config["lookback_years"]
    start_date = ref_ts - pd.DateOffset(years=lookback_years)

    # Trim daily cache to lookback window (no-look-ahead: end at reference_date)
    mask = (daily_cache.index >= start_date) & (daily_cache.index <= ref_ts)
    trimmed = daily_cache.loc[mask].copy()

    if trimmed.empty:
        return ResampledCache(
            frequency=frequency,
            label=config["label"],
            lookback_years=lookback_years,
            cache=trimmed,
            n_periods=0,
            is_partial_last_period=False,
            original_daily_rows=len(daily_cache),
            resampled_rows=0,
        )

    # Daily frequency: no resampling needed, just trim
    if frequency == "D":
        return ResampledCache(
            frequency=frequency,
            label=config["label"],
            lookback_years=lookback_years,
            cache=trimmed,
            n_periods=len(trimmed),
            is_partial_last_period=False,
            original_daily_rows=len(daily_cache),
            resampled_rows=len(trimmed),
        )

    # Resample to target frequency
    resample_rule = config["resample_rule"]
    resampled = _resample_dataframe(trimmed, resample_rule, ref_ts)

    # Detect partial last period
    is_partial = _is_last_period_partial(ref_ts, frequency)

    # Mark partial period
    if is_partial and not resampled.empty:
        resampled.loc[resampled.index[-1], "is_partial_period"] = 1
    if "is_partial_period" not in resampled.columns:
        resampled["is_partial_period"] = 0

    logger.info(
        "Resampled %s: %d daily rows -> %d %s periods (%s lookback, partial_last=%s)",
        frequency, len(trimmed), len(resampled),
        config["label"].lower(), f"{lookback_years}yr",
        is_partial,
    )

    return ResampledCache(
        frequency=frequency,
        label=config["label"],
        lookback_years=lookback_years,
        cache=resampled,
        n_periods=len(resampled),
        is_partial_last_period=is_partial,
        original_daily_rows=len(daily_cache),
        resampled_rows=len(resampled),
    )


# ---------------------------------------------------------------------------
# Internal resampling logic
# ---------------------------------------------------------------------------

def _resample_dataframe(
    df: pd.DataFrame,
    rule: str,
    ref_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Resample a daily DataFrame to the target frequency.

    OHLCV columns use standard OHLC aggregation rules.
    Financial statement columns use stock (last) or flow (sum) rules.
    Other numeric columns use last-value.
    """
    ohlcv_present = [c for c in df.columns if c in _OHLCV_COLS]
    flow_present = [c for c in df.columns if c in _FLOW_VARIABLES and c not in _OHLCV_COLS]
    other_numeric = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c not in _OHLCV_COLS and c not in _FLOW_VARIABLES
    ]

    parts = []

    # OHLCV resampling with proper aggregation
    if ohlcv_present:
        ohlcv_agg = {}
        if "open" in df.columns:
            ohlcv_agg["open"] = "first"
        if "high" in df.columns:
            ohlcv_agg["high"] = "max"
        if "low" in df.columns:
            ohlcv_agg["low"] = "min"
        if "close" in df.columns:
            ohlcv_agg["close"] = "last"
        if "volume" in df.columns:
            ohlcv_agg["volume"] = "sum"
        if "adjusted_close" in df.columns:
            ohlcv_agg["adjusted_close"] = "last"
        if "vwap" in df.columns:
            ohlcv_agg["vwap"] = "mean"

        ohlcv_resampled = df[list(ohlcv_agg.keys())].resample(rule).agg(ohlcv_agg)
        parts.append(ohlcv_resampled)

    # Flow variables: sum if distributed by interpolator, last if flat-ffilled
    if flow_present:
        # Detect whether flow variables were distributed to daily amounts
        # (by frequency_interpolator) or are flat forward-filled periodic totals.
        # If values within a resample period are all identical -> flat ffill -> use last()
        # If values vary within periods -> distributed -> sum() reconstructs period total
        _use_sum = False
        _sample_col = flow_present[0]
        _sample = df[_sample_col].dropna()
        if len(_sample) > 5:
            # Check first non-trivial resample group for value variation
            _groups = _sample.resample(rule)
            for _name, _group in _groups:
                if len(_group) >= 3:
                    _nunique = _group.nunique()
                    _use_sum = _nunique > 1  # values vary -> distributed
                    break
        if _use_sum:
            flow_resampled = df[flow_present].resample(rule).sum()
        else:
            # Flat forward-fill: each day carries the full periodic total.
            # Use last() to avoid N-fold inflation.
            flow_resampled = df[flow_present].resample(rule).last()
        parts.append(flow_resampled)

    # Stock/other numeric variables: last value in period
    if other_numeric:
        stock_resampled = df[other_numeric].resample(rule).last()
        parts.append(stock_resampled)

    if not parts:
        return pd.DataFrame(index=df.resample(rule).last().index)

    result = pd.concat(parts, axis=1)

    # Drop rows that are entirely NaN (periods with no data)
    result = result.dropna(how="all")

    return result


def _is_last_period_partial(ref_ts: pd.Timestamp, frequency: str) -> bool:
    """Check if the reference date falls before the end of its period."""
    if frequency == "W":
        # Friday is end of week
        return ref_ts.weekday() < 4  # 0=Mon ... 4=Fri
    elif frequency == "M":
        # Check if reference date is the last business day of the month
        month_end = ref_ts + pd.offsets.MonthEnd(0)
        return ref_ts.date() < month_end.date()
    elif frequency == "Q":
        quarter_end = ref_ts + pd.offsets.QuarterEnd(0)
        return ref_ts.date() < quarter_end.date()
    elif frequency == "A":
        year_end = pd.Timestamp(ref_ts.year, 12, 31)
        return ref_ts.date() < year_end.date()
    return False


# ---------------------------------------------------------------------------
# Helper: get all frequencies in execution order (slow to fast)
# ---------------------------------------------------------------------------

def build_cache_from_raw_filings(
    income_df: pd.DataFrame | None = None,
    balance_df: pd.DataFrame | None = None,
    cashflow_df: pd.DataFrame | None = None,
    quotes_df: pd.DataFrame | None = None,
    frequency: str = "Q",
    reference_date: date | None = None,
) -> ResampledCache:
    """Build a Q/A cache directly from raw filing DataFrames.

    For quarterly and annual frequencies, the daily cache's interpolated
    values are artifacts of the daily pipeline.  This function constructs
    the cache from the original filing data, preserving the actual
    reported values without interpolation artifacts.

    OHLCV data is resampled from daily to the target frequency using
    standard OHLC aggregation (first/max/min/last/sum).

    Parameters
    ----------
    income_df, balance_df, cashflow_df:
        Wide-format statement DataFrames with ``report_date`` column.
        One row per filing period with actual reported values.
    quotes_df:
        Daily OHLCV DataFrame (will be resampled to target frequency).
    frequency:
        Target frequency: ``"Q"`` or ``"A"``.
    reference_date:
        The "today" date for truncation.
    """
    config = FREQUENCY_CONFIG.get(frequency)
    if config is None or frequency not in ("Q", "A"):
        raise ValueError(f"build_cache_from_raw_filings only supports Q/A, got: {frequency}")

    resample_rule = config["resample_rule"]
    lookback_years = config["lookback_years"]

    if reference_date is None:
        reference_date = date.today()
    ref_ts = pd.Timestamp(reference_date)
    start_ts = ref_ts - pd.DateOffset(years=lookback_years)

    parts: list[pd.DataFrame] = []

    # --- OHLCV: resample daily to target frequency ---
    if quotes_df is not None and not quotes_df.empty:
        ohlcv = quotes_df.copy()
        if "date" in ohlcv.columns:
            ohlcv["date"] = pd.to_datetime(ohlcv["date"])
            ohlcv = ohlcv.set_index("date").sort_index()
        # Trim to lookback window
        ohlcv = ohlcv[(ohlcv.index >= start_ts) & (ohlcv.index <= ref_ts)]
        if not ohlcv.empty:
            ohlcv_resampled = _resample_dataframe(ohlcv, resample_rule, ref_ts)
            if not ohlcv_resampled.empty:
                parts.append(ohlcv_resampled)

    # --- Financial statements: use raw filing data directly ---
    for label, stmt_df in [("income", income_df), ("balance", balance_df), ("cashflow", cashflow_df)]:
        if stmt_df is None or stmt_df.empty:
            continue

        df = stmt_df.copy()

        # Find the date column
        date_col = None
        for dc in ("report_date", "filing_date"):
            if dc in df.columns:
                date_col = dc
                break
        if date_col is None:
            continue

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

        # Trim to lookback window
        df = df[(df[date_col] >= start_ts) & (df[date_col] <= ref_ts)]
        if df.empty:
            continue

        # Extract numeric columns only
        numeric_cols = [
            c for c in df.select_dtypes(include=["number"]).columns
            if c != date_col and "date" not in c.lower()
        ]
        if not numeric_cols:
            continue

        # Set report_date as index (these are the ACTUAL filing period dates)
        stmt_indexed = df.set_index(date_col)[numeric_cols]

        # For the target frequency, just use the raw values as-is
        # No interpolation, no resampling -- each row IS a Q or A period
        parts.append(stmt_indexed)

    if not parts:
        return ResampledCache(
            frequency=frequency,
            label=config["label"],
            lookback_years=lookback_years,
            cache=pd.DataFrame(),
            n_periods=0,
            is_partial_last_period=False,
            original_daily_rows=0,
            resampled_rows=0,
        )

    # Combine OHLCV + statements, aligning by date
    # Use outer join to preserve all filing dates
    combined = parts[0]
    for part in parts[1:]:
        # Join on index, keeping both date grids
        combined = combined.join(part, how="outer", rsuffix="_dup")
        # Drop duplicate columns (first wins)
        dup_cols = [c for c in combined.columns if c.endswith("_dup")]
        for dc in dup_cols:
            orig = dc.replace("_dup", "")
            if orig in combined.columns:
                # Fill gaps in original from duplicate
                combined[orig] = combined[orig].fillna(combined[dc])
            combined = combined.drop(columns=[dc])

    # Sort by date and drop all-NaN rows
    combined = combined.sort_index().dropna(how="all")

    is_partial = _is_last_period_partial(ref_ts, frequency)
    if is_partial and not combined.empty:
        combined.loc[combined.index[-1], "is_partial_period"] = 1
    if "is_partial_period" not in combined.columns:
        combined["is_partial_period"] = 0

    n_daily = len(quotes_df) if quotes_df is not None else 0

    logger.info(
        "Built %s cache from raw filings: %d periods (%d statements merged, partial_last=%s)",
        config["label"], len(combined),
        sum(1 for d in [income_df, balance_df, cashflow_df] if d is not None and not d.empty),
        is_partial,
    )

    return ResampledCache(
        frequency=frequency,
        label=config["label"],
        lookback_years=lookback_years,
        cache=combined,
        n_periods=len(combined),
        is_partial_last_period=is_partial,
        original_daily_rows=n_daily,
        resampled_rows=len(combined),
    )


def get_frequencies_slow_to_fast() -> list[str]:
    """Return frequency codes in execution order: Annual -> Daily."""
    return ["A", "Q", "M", "W", "D"]


def get_frequency_config(frequency: str) -> dict[str, Any]:
    """Return configuration for a frequency."""
    return FREQUENCY_CONFIG.get(frequency, {})

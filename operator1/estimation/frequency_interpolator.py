"""Frequency-aware interpolation of periodic filings to daily cache.

Financial filings arrive periodically -- quarterly, semi-annually, or
annually.  The naive approach (flat forward-fill) creates step functions
where 90-365 days have identical values.  This module creates smooth
daily trajectories that respect the nature of each variable:

**Stock variables** (balance sheet snapshots):
    Linear interpolation between filing values.  Total assets don't jump
    on filing day -- they change gradually as the company operates.

**Flow variables** (income/cash flow period totals):
    Distribute the period total across the business days in the period.
    $100M quarterly revenue becomes ~$1.1M/day across 63 business days.

Top-level entry point:
    ``interpolate_statement_to_daily(stmt_df, daily_index, market_id)``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variable type classification
# ---------------------------------------------------------------------------

# Stock variables: point-in-time balance sheet snapshots.
# These are interpolated linearly between filings.
STOCK_VARIABLES: frozenset[str] = frozenset({
    "total_assets", "total_liabilities", "total_equity",
    "current_assets", "current_liabilities",
    "cash_and_equivalents", "short_term_debt", "long_term_debt",
    "total_debt", "total_debt_asof",
    "receivables", "inventory", "payables",
    "retained_earnings", "goodwill", "intangible_assets",
    "shares_outstanding", "net_debt",
})

# Flow variables: cumulative totals over the reporting period.
# These are distributed across the business days in the period.
FLOW_VARIABLES: frozenset[str] = frozenset({
    "revenue", "cost_of_revenue", "gross_profit",
    "operating_income", "ebit", "ebitda", "net_income",
    "interest_expense", "taxes",
    "operating_cash_flow", "capex", "free_cash_flow",
    "investing_cf", "financing_cf", "dividends_paid",
    "stock_buybacks", "sga_expenses", "rd_expenses",
})

# Filing frequency configs (days between filings, used for confidence)
_FREQ_CONFIG: dict[str, dict[str, Any]] = {
    "quarterly": {"period_days": 90, "confidence_base": 0.85},
    "semiannual": {"period_days": 180, "confidence_base": 0.70},
    "annual": {"period_days": 365, "confidence_base": 0.50},
    "unknown": {"period_days": 180, "confidence_base": 0.40},
}


# ---------------------------------------------------------------------------
# Frequency detection helper
# ---------------------------------------------------------------------------

def detect_frequency_from_dates(filing_dates: list[pd.Timestamp]) -> str:
    """Detect filing frequency from a list of report/filing dates.

    Returns one of: "quarterly", "semiannual", "annual", "unknown".
    """
    if len(filing_dates) < 2:
        return "unknown"

    dates = sorted(filing_dates)
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    median_gap = float(np.median(gaps))

    if median_gap < 120:
        return "quarterly"
    elif median_gap < 250:
        return "semiannual"
    elif median_gap < 500:
        return "annual"
    return "unknown"


def classify_variable(col_name: str) -> str:
    """Classify a column as 'stock', 'flow', or 'unknown'.

    Parameters
    ----------
    col_name:
        Canonical column name.

    Returns
    -------
    "stock", "flow", or "unknown"
    """
    if col_name in STOCK_VARIABLES:
        return "stock"
    if col_name in FLOW_VARIABLES:
        return "flow"
    return "unknown"


# ---------------------------------------------------------------------------
# Core interpolation functions
# ---------------------------------------------------------------------------

def _interpolate_stock(
    filing_values: pd.Series,
    daily_index: pd.DatetimeIndex,
) -> pd.Series:
    """Linear interpolation for stock (balance sheet) variables.

    Between two filings, the value transitions linearly.
    Before the first filing and after the last: flat extrapolation.

    Parameters
    ----------
    filing_values:
        Series indexed by filing/report dates with the periodic values.
    daily_index:
        Target daily DatetimeIndex to produce values for.

    Returns
    -------
    Daily series with linearly interpolated values.
    """
    if filing_values.empty or filing_values.dropna().empty:
        return pd.Series(np.nan, index=daily_index, dtype=float)

    clean = filing_values.dropna().sort_index()

    if len(clean) == 1:
        # Single filing: flat fill everywhere
        return pd.Series(float(clean.iloc[0]), index=daily_index, dtype=float)

    # Place filing values on the daily index, then interpolate
    combined = pd.Series(np.nan, index=daily_index, dtype=float)

    for dt, val in clean.items():
        # Find the closest daily index date to each filing date
        if dt in combined.index:
            combined.loc[dt] = val
        else:
            # Find nearest date in the daily index
            idx_pos = combined.index.searchsorted(dt)
            if idx_pos < len(combined.index):
                combined.iloc[idx_pos] = val
            elif len(combined.index) > 0:
                combined.iloc[-1] = val

    # Interpolate linearly between placed values
    combined = combined.interpolate(method="time", limit_direction="both")
    # Fill edges with flat extrapolation
    combined = combined.ffill().bfill()

    return combined


def _distribute_flow(
    filing_values: pd.Series,
    filing_dates: list[pd.Timestamp],
    daily_index: pd.DatetimeIndex,
) -> pd.Series:
    """Distribute flow variable period totals across business days.

    Each quarterly/annual total is spread evenly across the business
    days in that period.  The result sums back to the original total
    when aggregated over the period.

    Parameters
    ----------
    filing_values:
        Series indexed by report dates with the period totals.
    filing_dates:
        Sorted list of report dates (period end dates).
    daily_index:
        Target daily DatetimeIndex (business days).

    Returns
    -------
    Daily series with the period total distributed across days.
    """
    if filing_values.empty or filing_values.dropna().empty:
        return pd.Series(np.nan, index=daily_index, dtype=float)

    clean = filing_values.dropna().sort_index()
    sorted_dates = sorted(clean.index)

    result = pd.Series(np.nan, index=daily_index, dtype=float)

    for i, period_end in enumerate(sorted_dates):
        # Determine period start: day after previous period end,
        # or earliest date in the index for the first period.
        if i == 0:
            # First period: estimate start based on frequency
            if len(sorted_dates) >= 2:
                gap = (sorted_dates[1] - sorted_dates[0]).days
            else:
                gap = 90  # default quarterly
            period_start = period_end - pd.Timedelta(days=gap)
        else:
            period_start = sorted_dates[i - 1] + pd.Timedelta(days=1)

        # Get business days in this period
        period_mask = (daily_index > period_start) & (daily_index <= period_end)
        n_bdays = period_mask.sum()

        if n_bdays == 0:
            continue

        # Distribute the total evenly across business days
        period_total = float(clean.iloc[i])
        daily_amount = period_total / n_bdays
        result.loc[period_mask] = daily_amount

    # For days after the last filing: carry forward the last daily rate
    last_date = sorted_dates[-1]
    after_mask = daily_index > last_date
    if after_mask.any() and result.notna().any():
        last_rate = result.dropna().iloc[-1]
        result.loc[after_mask] = last_rate

    # For days before the first period: use the first daily rate
    first_period_start = sorted_dates[0]
    if len(sorted_dates) >= 2:
        gap = (sorted_dates[1] - sorted_dates[0]).days
    else:
        gap = 90
    estimated_start = first_period_start - pd.Timedelta(days=gap)
    before_mask = (daily_index <= estimated_start) & result.isna()
    if before_mask.any() and result.notna().any():
        first_rate = result.dropna().iloc[0]
        result.loc[before_mask] = first_rate

    # Fill any remaining NaN gaps (between estimated period start and
    # actual first value) with the nearest non-null rate
    result = result.ffill().bfill()

    return result


def _compute_interpolation_confidence(
    daily_index: pd.DatetimeIndex,
    filing_dates: list[pd.Timestamp],
    frequency: str,
    variable_type: str,
) -> pd.Series:
    """Compute per-day interpolation confidence.

    Higher confidence for days closer to a filing.
    Lower confidence for annual filers and flow variables.

    Parameters
    ----------
    daily_index:
        Target daily index.
    filing_dates:
        Sorted list of filing dates.
    frequency:
        "quarterly", "semiannual", "annual", "unknown"
    variable_type:
        "stock" or "flow"

    Returns
    -------
    Series of confidence scores in [0, 1].
    """
    if not filing_dates:
        return pd.Series(0.3, index=daily_index, dtype=float)

    config = _FREQ_CONFIG.get(frequency, _FREQ_CONFIG["unknown"])
    base_conf = config["confidence_base"]
    max_gap_days = config["period_days"] // 2  # half-period is max distance

    # Compute distance to nearest filing for each daily date
    filing_ts = pd.DatetimeIndex(sorted(filing_dates))
    distances = pd.Series(0, index=daily_index, dtype=float)

    for i, day in enumerate(daily_index):
        # Binary search for nearest filing
        pos = filing_ts.searchsorted(day)
        candidates = []
        if pos > 0:
            candidates.append(abs((day - filing_ts[pos - 1]).days))
        if pos < len(filing_ts):
            candidates.append(abs((day - filing_ts[pos]).days))
        distances.iloc[i] = min(candidates) if candidates else max_gap_days

    # Decay confidence with distance from nearest filing
    distance_decay = (1.0 - (distances / max(max_gap_days, 1)) * 0.5).clip(0.3, 1.0)

    # Flow variables are slightly less reliable than stock variables
    type_mult = 1.0 if variable_type == "stock" else 0.85

    confidence = (base_conf * distance_decay * type_mult).clip(0.1, 0.95)
    return confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def interpolate_statement_to_daily(
    stmt_indexed: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    market_id: str = "",
    frequency: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interpolate periodic financial statement data to daily frequency.

    Replaces the naive flat forward-fill with frequency-aware
    interpolation that respects the nature of each variable:
    - Stock variables: linear interpolation between filings
    - Flow variables: distribute period totals across business days
    - Unknown variables: linear interpolation (conservative default)

    Parameters
    ----------
    stmt_indexed:
        Financial statement DataFrame indexed by report_date with
        one row per filing period and columns for each financial field.
    daily_index:
        Target daily DatetimeIndex (business days) to interpolate onto.
    market_id:
        Market identifier (used for frequency lookup if not detected).
    frequency:
        Override filing frequency ("quarterly", "semiannual", "annual").
        If None, detected from filing dates.

    Returns
    -------
    (interpolated_df, confidence_df)
        interpolated_df: Daily values with smooth trajectories.
        confidence_df: Per-column confidence scores for each daily value.
    """
    if stmt_indexed.empty:
        return (
            pd.DataFrame(index=daily_index),
            pd.DataFrame(index=daily_index),
        )

    # Detect filing frequency
    filing_dates = sorted(stmt_indexed.index.tolist())
    if frequency is None:
        frequency = detect_frequency_from_dates(filing_dates)
    logger.info(
        "Interpolating %d filing periods (%s frequency) to %d daily rows",
        len(filing_dates), frequency, len(daily_index),
    )

    result = pd.DataFrame(index=daily_index)
    confidence = pd.DataFrame(index=daily_index)

    for col in stmt_indexed.columns:
        col_values = stmt_indexed[col].dropna()
        if col_values.empty:
            result[col] = np.nan
            confidence[col] = 0.0
            continue

        var_type = classify_variable(col)

        if var_type == "flow":
            result[col] = _distribute_flow(
                col_values, filing_dates, daily_index,
            )
        else:
            # Stock or unknown: linear interpolation
            result[col] = _interpolate_stock(
                col_values, daily_index,
            )

        confidence[col] = _compute_interpolation_confidence(
            daily_index, filing_dates, frequency,
            variable_type=var_type if var_type != "unknown" else "stock",
        )

    n_stock = sum(1 for c in stmt_indexed.columns if classify_variable(c) == "stock")
    n_flow = sum(1 for c in stmt_indexed.columns if classify_variable(c) == "flow")
    n_other = len(stmt_indexed.columns) - n_stock - n_flow
    logger.info(
        "  Interpolated: %d stock (linear), %d flow (distributed), %d other",
        n_stock, n_flow, n_other,
    )

    return result, confidence

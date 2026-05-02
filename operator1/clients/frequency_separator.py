"""Separate mixed-frequency financial statement DataFrames by period type.

PIT clients (US EDGAR, DART, etc.) often return both annual and quarterly
filings in a single DataFrame.  Naively merging these produces flow variable
mismatches -- annual totals divided by quarterly totals yield 3-4x inflated
ratios (F5 bug: 319% gross margin, 14302% operating margin).

This module:
1. Splits a mixed-frequency DF into per-frequency groups
2. Backfills missing columns from lower-frequency groups using revenue-ratio
   scaling so flow variables are at the correct period scale
3. Builds a frequency-priority cache where quarterly > semi-annual > annual
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from operator1.estimation.frequency_interpolator import (
    FLOW_VARIABLES,
    STOCK_VARIABLES,
    classify_variable,
    detect_frequency_from_dates,
)

logger = logging.getLogger(__name__)

# Priority order: prefer highest frequency
_FREQ_PRIORITY = ["quarterly", "semiannual", "annual"]


def separate_by_period_type(
    stmt_df: pd.DataFrame,
    date_col: str = "report_date",
) -> dict[str, pd.DataFrame]:
    """Split a mixed-frequency statement DF into per-frequency groups.

    Parameters
    ----------
    stmt_df:
        Financial statement DataFrame.  May contain a ``period_type``
        column (values: "annual", "quarterly", "semiannual") set by
        the PIT client.  If absent, frequency is inferred from date gaps.
    date_col:
        Name of the date column for gap-based inference.

    Returns
    -------
    Dict keyed by frequency label ("quarterly", "semiannual", "annual").
    Each value is a DataFrame containing only rows of that frequency.
    Empty frequencies are omitted.
    """
    if stmt_df is None or stmt_df.empty:
        return {}

    # Path 1: explicit period_type column
    if "period_type" in stmt_df.columns:
        groups: dict[str, pd.DataFrame] = {}
        for pt in stmt_df["period_type"].dropna().unique():
            pt_str = str(pt).lower().strip()
            # Normalize aliases
            if pt_str in ("10-q", "q", "quarter"):
                pt_str = "quarterly"
            elif pt_str in ("10-k", "a", "annual", "yearly"):
                pt_str = "annual"
            elif pt_str in ("h", "half", "semi", "semiannual", "semi-annual"):
                pt_str = "semiannual"
            mask = stmt_df["period_type"].astype(str).str.lower().str.strip().isin(
                _period_type_aliases(pt_str)
            )
            sub = stmt_df[mask].copy()
            if not sub.empty:
                groups[pt_str] = sub
        if groups:
            _log_separation(groups)
            return groups

    # Path 2: infer from date gaps
    if date_col not in stmt_df.columns:
        return {"unknown": stmt_df.copy()}

    stmt_df = stmt_df.copy()
    stmt_df[date_col] = pd.to_datetime(stmt_df[date_col], errors="coerce")
    stmt_df = stmt_df.dropna(subset=[date_col]).sort_values(date_col)

    if len(stmt_df) < 2:
        freq = detect_frequency_from_dates(stmt_df[date_col].tolist())
        return {freq: stmt_df}

    # Compute gap from each row to the next
    dates = stmt_df[date_col].values
    gaps = np.diff(dates).astype("timedelta64[D]").astype(int)

    labels = []
    labels.append(_classify_gap(gaps[0] if len(gaps) > 0 else 90))
    for g in gaps:
        labels.append(_classify_gap(g))

    stmt_df["_inferred_freq"] = labels
    groups = {}
    for freq_label in stmt_df["_inferred_freq"].unique():
        sub = stmt_df[stmt_df["_inferred_freq"] == freq_label].drop(
            columns=["_inferred_freq"]
        )
        if not sub.empty:
            groups[freq_label] = sub

    _log_separation(groups)
    return groups


def backfill_from_lower_frequency(
    high_freq_df: pd.DataFrame,
    low_freq_df: pd.DataFrame,
    date_col: str = "report_date",
) -> pd.DataFrame:
    """Fill missing columns in high-freq DF from low-freq DF with scaling.

    For FLOW variables (revenue, gross_profit, etc.), scales the lower-
    frequency total proportionally using revenue as a proxy for seasonal
    distribution.  For example:

        Q4_gross_profit = annual_gross_profit * (Q4_revenue / annual_revenue)

    For STOCK variables (total_assets, etc.), uses the lower-frequency
    value directly (balance sheet snapshots are frequency-independent).

    Parameters
    ----------
    high_freq_df:
        Higher-frequency wide-format DF (e.g., quarterly).
    low_freq_df:
        Lower-frequency wide-format DF (e.g., annual).
    date_col:
        Date column name.

    Returns
    -------
    high_freq_df with missing columns backfilled from low_freq_df.
    """
    if low_freq_df is None or low_freq_df.empty:
        return high_freq_df
    if high_freq_df is None or high_freq_df.empty:
        return low_freq_df

    result = high_freq_df.copy()

    # Identify columns that need backfill from the lower-frequency DF:
    # 1. Columns only in low-freq (not in high-freq at all)
    # 2. Columns in high-freq but all-NaN (reported only at lower freq)
    high_cols = set(result.columns)
    low_only_cols: list[str] = []

    for c in low_freq_df.columns:
        if c == date_col or "date" in c.lower() or c == "period_type":
            continue
        if c not in high_cols:
            # Column only exists in lower frequency
            low_only_cols.append(c)
        elif c in result.columns and result[c].isna().all():
            # Column exists in high-freq but is all NaN
            low_only_cols.append(c)

    if not low_only_cols:
        return result

    # Build revenue ratio for flow variable scaling
    revenue_ratio = _compute_revenue_ratio(result, low_freq_df, date_col)

    for col in low_only_cols:
        if col not in low_freq_df.columns:
            continue

        var_type = classify_variable(col)
        low_values = low_freq_df[[date_col, col]].dropna(subset=[col]).copy()
        if low_values.empty:
            continue

        if var_type == "flow" and revenue_ratio is not None:
            # Scale the low-freq total by the revenue ratio
            scaled = _scale_flow_by_revenue_ratio(
                low_values, revenue_ratio, col, date_col,
            )
            if scaled is not None:
                result = _merge_column(result, scaled, col, date_col)
                logger.debug(
                    "Backfilled flow column '%s' from lower freq "
                    "(revenue-ratio scaled)",
                    col,
                )
                continue

        # Stock variables or fallback: use low-freq value directly
        # For each high-freq date, find the nearest low-freq value
        result = _merge_column(result, low_values, col, date_col)
        logger.debug(
            "Backfilled %s column '%s' from lower freq (direct)",
            var_type, col,
        )

    n_filled = len(low_only_cols)
    if n_filled > 0:
        logger.info(
            "Backfilled %d columns from lower-frequency data", n_filled,
        )

    return result


def build_highest_frequency_statement(
    freq_groups: dict[str, pd.DataFrame],
    date_col: str = "report_date",
) -> pd.DataFrame:
    """Build a single statement DF using highest-frequency data first.

    Prefers quarterly over semi-annual over annual.  Missing columns
    are backfilled from lower frequencies with proper scaling.

    Parameters
    ----------
    freq_groups:
        Dict from ``separate_by_period_type()``.
    date_col:
        Date column name.

    Returns
    -------
    Single DF at the highest available frequency with missing columns
    scaled from lower frequencies.
    """
    if not freq_groups:
        return pd.DataFrame()

    # Sort by priority
    sorted_freqs = sorted(
        freq_groups.keys(),
        key=lambda f: _FREQ_PRIORITY.index(f) if f in _FREQ_PRIORITY else 99,
    )

    # Start with highest frequency
    result = freq_groups[sorted_freqs[0]].copy()

    # Backfill from each lower frequency
    for freq in sorted_freqs[1:]:
        result = backfill_from_lower_frequency(
            result, freq_groups[freq], date_col,
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _period_type_aliases(normalized: str) -> set[str]:
    """Return all string variants that map to a normalized period type."""
    aliases = {
        "quarterly": {"quarterly", "10-q", "q", "quarter"},
        "annual": {"annual", "10-k", "a", "yearly", "fy"},
        "semiannual": {"semiannual", "semi-annual", "h", "half", "semi"},
    }
    return aliases.get(normalized, {normalized})


def _classify_gap(gap_days: int) -> str:
    """Classify a date gap into a frequency label."""
    if gap_days < 120:
        return "quarterly"
    elif gap_days < 250:
        return "semiannual"
    elif gap_days < 500:
        return "annual"
    return "annual"


def _compute_revenue_ratio(
    high_freq_df: pd.DataFrame,
    low_freq_df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame | None:
    """Compute ratio of high-freq revenue to low-freq revenue per period.

    Returns a DF with columns [date_col, 'revenue_ratio'] where
    revenue_ratio = high_freq_revenue / annual_revenue for the
    corresponding fiscal year.
    """
    if "revenue" not in high_freq_df.columns or "revenue" not in low_freq_df.columns:
        return None

    hf = high_freq_df[[date_col, "revenue"]].dropna(subset=["revenue"]).copy()
    lf = low_freq_df[[date_col, "revenue"]].dropna(subset=["revenue"]).copy()

    if hf.empty or lf.empty:
        return None

    hf[date_col] = pd.to_datetime(hf[date_col])
    lf[date_col] = pd.to_datetime(lf[date_col])
    lf = lf.sort_values(date_col)

    # For each high-freq date, find the matching low-freq period
    # (the low-freq date that is >= the high-freq date's fiscal year end)
    ratios = []
    for _, row in hf.iterrows():
        hf_date = row[date_col]
        hf_rev = row["revenue"]
        if pd.isna(hf_rev) or hf_rev == 0:
            continue

        # Find the annual revenue that covers this quarter
        # (annual report_date >= this quarter's report_date, same fiscal year)
        year = hf_date.year
        annual_mask = (lf[date_col].dt.year == year)
        if not annual_mask.any():
            # Try previous year's annual report
            annual_mask = (lf[date_col].dt.year == year - 1)
        annual_rows = lf[annual_mask]

        if annual_rows.empty:
            continue

        annual_rev = float(annual_rows.iloc[-1]["revenue"])
        if annual_rev == 0 or pd.isna(annual_rev):
            continue

        ratio = hf_rev / annual_rev
        # Sanity: ratio should be between 0 and 1 for quarterly
        ratio = max(0.01, min(1.0, ratio))
        ratios.append({date_col: hf_date, "revenue_ratio": ratio})

    if not ratios:
        return None

    return pd.DataFrame(ratios)


def _scale_flow_by_revenue_ratio(
    low_values: pd.DataFrame,
    revenue_ratio: pd.DataFrame,
    col: str,
    date_col: str,
) -> pd.DataFrame | None:
    """Scale a low-freq flow variable by the revenue ratio.

    For each high-freq period, the scaled value = low_freq_total * revenue_ratio.
    """
    if revenue_ratio.empty:
        return None

    result_rows = []
    for _, rr_row in revenue_ratio.iterrows():
        hf_date = rr_row[date_col]
        ratio = rr_row["revenue_ratio"]

        # Find the annual value for this period
        year = hf_date.year
        lv = low_values.copy()
        lv[date_col] = pd.to_datetime(lv[date_col])
        annual_mask = (lv[date_col].dt.year == year)
        if not annual_mask.any():
            annual_mask = (lv[date_col].dt.year == year - 1)
        matched = lv[annual_mask]

        if matched.empty:
            continue

        annual_val = float(matched.iloc[-1][col])
        if pd.isna(annual_val):
            continue

        scaled_val = annual_val * ratio
        result_rows.append({date_col: hf_date, col: scaled_val})

    if not result_rows:
        return None

    return pd.DataFrame(result_rows)


def _merge_column(
    target: pd.DataFrame,
    source: pd.DataFrame,
    col: str,
    date_col: str,
) -> pd.DataFrame:
    """Merge a single column from source into target, filling NaN only."""
    src = source[[date_col, col]].copy()
    src[date_col] = pd.to_datetime(src[date_col])
    target = target.copy()
    target[date_col] = pd.to_datetime(target[date_col])

    # Build a lookup from source dates to values
    src_map = dict(zip(src[date_col], src[col]))

    if col not in target.columns:
        target[col] = np.nan

    for idx in target.index:
        if pd.isna(target.at[idx, col]):
            d = target.at[idx, date_col]
            if d in src_map and not pd.isna(src_map[d]):
                target.at[idx, col] = src_map[d]

    return target


def _log_separation(groups: dict[str, pd.DataFrame]) -> None:
    """Log the frequency separation result."""
    parts = [f"{k}: {len(v)} rows" for k, v in groups.items()]
    logger.info("Frequency separation: %s", ", ".join(parts))

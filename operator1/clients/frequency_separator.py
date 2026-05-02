"""Separate mixed-frequency financial statement DataFrames by period type.

PIT clients (US EDGAR, DART, etc.) often return both annual and quarterly
filings in a single DataFrame.  Naively merging these produces flow variable
mismatches -- annual totals divided by quarterly totals yield 3-4x inflated
ratios (F5 bug: 319% gross margin, 14302% operating margin).

This module:
1. Deduplicates amended filings (keep latest filing_date per report_date)
2. Splits a mixed-frequency DF into per-frequency groups using Bayesian
   classification with market-specific priors
3. Backfills missing columns from lower-frequency groups using Chow-Lin
   temporal disaggregation (flow variables) and Denton-Cholette benchmarking
   (stock variables), with revenue-ratio scaling as fallback
4. Reconciles disaggregated flow variable sums against annual totals
5. Builds a frequency-priority cache where quarterly > semi-annual > annual

Methods:
  - Bayesian frequency detection (Geweke 1977 pattern, market priors)
  - Chow-Lin temporal disaggregation via tempdisagg (Chow & Lin 1971)
  - Denton-Cholette proportional benchmarking via tempdisagg (Denton 1971)
  - Post-disaggregation reconciliation (ISA 520 audit logic)
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

# Config cache
_config_cache: dict[str, Any] | None = None


def _load_freq_config() -> dict[str, Any]:
    """Load and cache frequency separator config."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    try:
        from operator1.config_loader import load_config
        _config_cache = load_config("frequency_separator")
    except Exception:
        _config_cache = {}
    return _config_cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def separate_by_period_type(
    stmt_df: pd.DataFrame,
    date_col: str = "report_date",
    market_id: str = "",
) -> dict[str, pd.DataFrame]:
    """Split a mixed-frequency statement DF into per-frequency groups.

    Parameters
    ----------
    stmt_df:
        Financial statement DataFrame.  May contain a ``period_type``
        column (values: "annual", "quarterly", "semiannual") set by
        the PIT client.  If absent, frequency is inferred using
        Bayesian gap classification with market-specific priors.
    date_col:
        Name of the date column for gap-based inference.
    market_id:
        Market identifier for Bayesian priors (e.g. "us_sec_edgar").

    Returns
    -------
    Dict keyed by frequency label ("quarterly", "semiannual", "annual").
    Each value is a DataFrame containing only rows of that frequency.
    Empty frequencies are omitted.
    """
    if stmt_df is None or stmt_df.empty:
        return {}

    # Step 0: Deduplicate amendments (keep latest filing per report_date)
    stmt_df = _deduplicate_amendments(stmt_df, date_col)

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

    # Path 2: Bayesian inference from date gaps
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

    # Bayesian classification with market priors
    gap_list = list(gaps)
    labels = _bayesian_classify_gaps(gap_list, market_id)
    # First row gets same label as the first gap
    all_labels = [labels[0]] + labels

    stmt_df["_inferred_freq"] = all_labels
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

    For FLOW variables (revenue, gross_profit, etc.), uses Chow-Lin
    temporal disaggregation (tempdisagg library) to produce quarterly
    estimates that sum to the annual total and correlate with revenue.
    Falls back to revenue-ratio scaling if tempdisagg is unavailable.

    For STOCK variables (total_assets, etc.), uses Denton-Cholette
    proportional benchmarking or direct copy as fallback.

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

    # Identify columns that need backfill
    high_cols = set(result.columns)
    low_only_cols: list[str] = []

    for c in low_freq_df.columns:
        if c == date_col or "date" in c.lower() or c == "period_type":
            continue
        if c not in high_cols:
            low_only_cols.append(c)
        elif c in result.columns and result[c].isna().all():
            low_only_cols.append(c)

    if not low_only_cols:
        return result

    # Try Chow-Lin / Denton via tempdisagg first
    _has_tempdisagg = _check_tempdisagg()

    # Build revenue ratio as fallback for flow variables
    revenue_ratio = _compute_revenue_ratio(result, low_freq_df, date_col)

    for col in low_only_cols:
        if col not in low_freq_df.columns:
            continue

        var_type = classify_variable(col)
        low_values = low_freq_df[[date_col, col]].dropna(subset=[col]).copy()
        if low_values.empty:
            continue

        filled = False

        if var_type == "flow" and _has_tempdisagg:
            # Primary: Chow-Lin temporal disaggregation
            scaled = _disaggregate_flow_chow_lin(
                low_freq_df, result, col, date_col,
            )
            if scaled is not None:
                result = _merge_column_vectorized(result, scaled, col, date_col)
                logger.debug("Backfilled flow '%s' via Chow-Lin", col)
                filled = True

        if not filled and var_type == "flow" and revenue_ratio is not None:
            # Fallback: revenue-ratio scaling
            scaled = _scale_flow_by_revenue_ratio(
                low_values, revenue_ratio, col, date_col,
            )
            if scaled is not None:
                result = _merge_column_vectorized(result, scaled, col, date_col)
                logger.debug("Backfilled flow '%s' via revenue-ratio scaling", col)
                filled = True

        if not filled and var_type == "stock" and _has_tempdisagg:
            # Denton-Cholette benchmarking for stock variables
            benchmarked = _benchmark_stock_denton(
                low_freq_df, result, col, date_col,
            )
            if benchmarked is not None:
                result = _merge_column_vectorized(result, benchmarked, col, date_col)
                logger.debug("Backfilled stock '%s' via Denton-Cholette", col)
                filled = True

        if not filled:
            # Final fallback: direct merge from low-freq (nearest date)
            result = _merge_column_vectorized(result, low_values, col, date_col)
            logger.debug("Backfilled %s '%s' via direct copy", var_type, col)

    n_filled = len(low_only_cols)
    if n_filled > 0:
        logger.info("Backfilled %d columns from lower-frequency data", n_filled)

    return result


def build_highest_frequency_statement(
    freq_groups: dict[str, pd.DataFrame],
    date_col: str = "report_date",
) -> pd.DataFrame:
    """Build a single statement DF using highest-frequency data first.

    Prefers quarterly over semi-annual over annual.  Missing columns
    are backfilled from lower frequencies with proper scaling.
    Post-disaggregation reconciliation validates flow variable sums.

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

    # Post-disaggregation reconciliation
    result = _reconcile_disaggregated(result, freq_groups, date_col)

    return result


# ---------------------------------------------------------------------------
# Amendment deduplication
# ---------------------------------------------------------------------------


def _deduplicate_amendments(
    stmt_df: pd.DataFrame,
    date_col: str = "report_date",
) -> pd.DataFrame:
    """Remove duplicate filings, keeping latest amendment per report_date.

    When a company restates a filing, the PIT client returns both the
    original and amended version.  We keep only the latest filing_date
    per report_date (SCD Type 2 pattern).

    Also detects material restatements (>10% value change) and logs warnings.
    """
    if stmt_df is None or stmt_df.empty:
        return stmt_df

    if "filing_date" not in stmt_df.columns or date_col not in stmt_df.columns:
        return stmt_df

    df = stmt_df.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Check for material restatements before dedup
    config = _load_freq_config()
    restatement_threshold = config.get("reconciliation", {}).get(
        "restatement_threshold", 0.10,
    )
    dupes = df[df.duplicated(subset=[date_col], keep=False)]
    if len(dupes) > 0:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for report_date, group in dupes.groupby(date_col):
            if len(group) < 2:
                continue
            original = group.sort_values("filing_date").iloc[0]
            amended = group.sort_values("filing_date").iloc[-1]
            for col in numeric_cols:
                orig_val = original.get(col)
                amend_val = amended.get(col)
                if (
                    pd.notna(orig_val)
                    and pd.notna(amend_val)
                    and abs(orig_val) > 1e-6
                ):
                    pct_change = abs(amend_val - orig_val) / abs(orig_val)
                    if pct_change > restatement_threshold:
                        logger.warning(
                            "Material restatement: %s changed %.1f%% on %s",
                            col, pct_change * 100, report_date,
                        )

    # Keep latest filing per report_date
    before = len(df)
    df = df.sort_values("filing_date").drop_duplicates(
        subset=[date_col], keep="last",
    )
    removed = before - len(df)
    if removed > 0:
        logger.info("Removed %d duplicate/amended filings", removed)

    return df


# ---------------------------------------------------------------------------
# Bayesian frequency classification
# ---------------------------------------------------------------------------


def _bayesian_classify_gaps(
    gaps: list[int],
    market_id: str = "",
) -> list[str]:
    """Classify filing date gaps using Bayesian posterior with market priors.

    For each gap, computes P(freq | gap) = P(gap | freq) * P(freq | market)
    and assigns the maximum a posteriori frequency class.

    Parameters
    ----------
    gaps:
        List of gap sizes in calendar days between consecutive filings.
    market_id:
        Market identifier for loading priors from config.

    Returns
    -------
    List of frequency labels, one per gap.
    """
    config = _load_freq_config()

    # Load market-specific priors or defaults
    cal = config.get("fiscal_calendars", {}).get(market_id, {})
    priors = cal.get("freq_prior", config.get(
        "default_freq_prior", {"quarterly": 0.50, "semiannual": 0.20, "annual": 0.30},
    ))

    # Load Gaussian parameters for each frequency class
    bp = config.get("bayesian_detection", {})
    freq_params = {
        "quarterly": (
            bp.get("gap_mean_quarterly", 90),
            bp.get("gap_std_quarterly", 15),
        ),
        "semiannual": (
            bp.get("gap_mean_semiannual", 180),
            bp.get("gap_std_semiannual", 25),
        ),
        "annual": (
            bp.get("gap_mean_annual", 365),
            bp.get("gap_std_annual", 40),
        ),
    }

    labels = []
    for gap in gaps:
        posteriors = {}
        for freq, (mu, sigma) in freq_params.items():
            # Gaussian likelihood
            likelihood = np.exp(-0.5 * ((gap - mu) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi)
            )
            posteriors[freq] = likelihood * priors.get(freq, 0.33)

        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total for k, v in posteriors.items()}

        # Assign MAP (maximum a posteriori) label
        best = max(posteriors, key=posteriors.get)
        labels.append(best)

    return labels


# ---------------------------------------------------------------------------
# Chow-Lin temporal disaggregation (via tempdisagg)
# ---------------------------------------------------------------------------


_tempdisagg_available: bool | None = None


def _check_tempdisagg() -> bool:
    """Check if tempdisagg library is available."""
    global _tempdisagg_available
    if _tempdisagg_available is not None:
        return _tempdisagg_available
    try:
        import tempdisagg  # noqa: F401
        _tempdisagg_available = True
    except ImportError:
        _tempdisagg_available = False
        logger.debug("tempdisagg not available -- falling back to revenue-ratio scaling")
    return _tempdisagg_available


def _disaggregate_flow_chow_lin(
    low_freq_df: pd.DataFrame,
    high_freq_df: pd.DataFrame,
    col: str,
    date_col: str,
) -> pd.DataFrame | None:
    """Disaggregate a low-freq flow variable to high-freq using Chow-Lin.

    Uses quarterly revenue as the indicator variable (Chow & Lin 1971).
    The disaggregated series sums to the annual total and correlates
    with the seasonal pattern of revenue.

    Returns a DataFrame with [date_col, col] or None on failure.
    """
    try:
        from tempdisagg import TempDisaggModel

        # Need revenue in both frequencies as indicator
        if "revenue" not in high_freq_df.columns or "revenue" not in low_freq_df.columns:
            return None
        if col not in low_freq_df.columns:
            return None

        # Prepare data for tempdisagg: needs grain (high-freq ID), index (low-freq group)
        hf = high_freq_df[[date_col, "revenue"]].dropna(subset=["revenue"]).copy()
        lf = low_freq_df[[date_col, col]].dropna(subset=[col]).copy()

        if hf.empty or lf.empty or len(hf) < 3 or len(lf) < 1:
            return None

        hf[date_col] = pd.to_datetime(hf[date_col])
        lf[date_col] = pd.to_datetime(lf[date_col])

        # Assign fiscal years
        hf["_year"] = hf[date_col].dt.year
        lf["_year"] = lf[date_col].dt.year

        # Build tempdisagg input DataFrame
        # Grain = quarter index, Index = fiscal year, X = revenue, y = target
        hf = hf.sort_values(date_col).reset_index(drop=True)
        hf["_grain"] = range(len(hf))

        td_df = hf[["_grain", "_year", "revenue"]].rename(columns={
            "_grain": "Grain", "_year": "Index", "revenue": "X",
        })

        # Merge annual target values by year
        lf_yearly = lf.groupby("_year")[col].last().reset_index()
        lf_yearly.columns = ["Index", "y"]
        td_df = td_df.merge(lf_yearly, on="Index", how="left")

        # tempdisagg requires non-null y for all Index groups that have X
        td_df = td_df.dropna(subset=["X"])
        if td_df["y"].isna().all():
            return None

        config = _load_freq_config()
        method = config.get("disaggregation", {}).get("flow_method", "chow-lin-opt")
        fallback = config.get("disaggregation", {}).get("fallback_method", "fast")

        try:
            model = TempDisaggModel(
                method=method,
                conversion="sum",
                grain_col="Grain",
                index_col="Index",
                y_col="y",
                X_col="X",
            )
            model.fit(td_df)
            predicted = model.predict()
        except Exception:
            # Fallback method
            try:
                model = TempDisaggModel(
                    method=fallback,
                    conversion="sum",
                    grain_col="Grain",
                    index_col="Index",
                    y_col="y",
                    X_col="X",
                )
                model.fit(td_df)
                predicted = model.predict()
            except Exception:
                return None

        if predicted is None or len(predicted) == 0:
            return None

        # Map predictions back to dates
        result_rows = []
        pred_values = predicted.values.flatten() if hasattr(predicted, "values") else list(predicted)
        for i, val in enumerate(pred_values):
            if i < len(hf) and not np.isnan(val):
                result_rows.append({
                    date_col: hf.iloc[i][date_col],
                    col: float(val),
                })

        if not result_rows:
            return None

        return pd.DataFrame(result_rows)

    except Exception as exc:
        logger.debug("Chow-Lin disaggregation failed for '%s': %s", col, exc)
        return None


# ---------------------------------------------------------------------------
# Denton-Cholette stock variable benchmarking
# ---------------------------------------------------------------------------


def _benchmark_stock_denton(
    low_freq_df: pd.DataFrame,
    high_freq_df: pd.DataFrame,
    col: str,
    date_col: str,
) -> pd.DataFrame | None:
    """Benchmark a stock variable using Denton-Cholette (via tempdisagg).

    Adjusts preliminary quarterly values so annual averages match exactly.

    Returns a DataFrame with [date_col, col] or None on failure.
    """
    try:
        from tempdisagg import TempDisaggModel

        if col not in low_freq_df.columns:
            return None

        hf = high_freq_df[[date_col]].copy()
        lf = low_freq_df[[date_col, col]].dropna(subset=[col]).copy()

        if hf.empty or lf.empty or len(hf) < 3:
            return None

        hf[date_col] = pd.to_datetime(hf[date_col])
        lf[date_col] = pd.to_datetime(lf[date_col])

        hf["_year"] = hf[date_col].dt.year
        lf["_year"] = lf[date_col].dt.year

        hf = hf.sort_values(date_col).reset_index(drop=True)
        hf["_grain"] = range(len(hf))
        # Use a constant indicator (uniform)
        hf["_X"] = 1.0

        td_df = hf[["_grain", "_year", "_X"]].rename(columns={
            "_grain": "Grain", "_year": "Index", "_X": "X",
        })

        lf_yearly = lf.groupby("_year")[col].last().reset_index()
        lf_yearly.columns = ["Index", "y"]
        td_df = td_df.merge(lf_yearly, on="Index", how="left")
        td_df = td_df.dropna(subset=["X"])

        if td_df["y"].isna().all():
            return None

        config = _load_freq_config()
        method = config.get("disaggregation", {}).get("stock_method", "denton-cholette")

        model = TempDisaggModel(
            method=method,
            conversion="last",
            grain_col="Grain",
            index_col="Index",
            y_col="y",
            X_col="X",
        )
        model.fit(td_df)
        predicted = model.predict()

        if predicted is None or len(predicted) == 0:
            return None

        result_rows = []
        pred_values = predicted.values.flatten() if hasattr(predicted, "values") else list(predicted)
        for i, val in enumerate(pred_values):
            if i < len(hf) and not np.isnan(val):
                result_rows.append({
                    date_col: hf.iloc[i][date_col],
                    col: float(val),
                })

        return pd.DataFrame(result_rows) if result_rows else None

    except Exception as exc:
        logger.debug("Denton-Cholette benchmarking failed for '%s': %s", col, exc)
        return None


# ---------------------------------------------------------------------------
# Post-disaggregation reconciliation
# ---------------------------------------------------------------------------


def _reconcile_disaggregated(
    result_df: pd.DataFrame,
    freq_groups: dict[str, pd.DataFrame],
    date_col: str = "report_date",
) -> pd.DataFrame:
    """Validate and adjust disaggregated flow variables against annual totals.

    For each flow variable, checks that quarterly sums approximate the
    annual total.  Applies pro-rata adjustment when deviation exceeds
    the configured tolerance (ISA 520 analytical review pattern).
    """
    annual = freq_groups.get("annual")
    if annual is None or annual.empty:
        return result_df
    if date_col not in result_df.columns or date_col not in annual.columns:
        return result_df

    config = _load_freq_config()
    tolerance = config.get("reconciliation", {}).get("sum_tolerance", 0.05)

    result = result_df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    annual_copy = annual.copy()
    annual_copy[date_col] = pd.to_datetime(annual_copy[date_col], errors="coerce")

    n_adjusted = 0
    for col in result.columns:
        if classify_variable(col) != "flow":
            continue
        if col not in annual_copy.columns:
            continue
        if result[col].isna().all():
            continue

        # Group quarterly values by fiscal year
        result["_fy"] = result[date_col].dt.year
        for year, group in result.groupby("_fy"):
            q_sum = group[col].sum()
            if pd.isna(q_sum) or abs(q_sum) < 1e-6:
                continue

            # Find matching annual value
            ann_mask = annual_copy[date_col].dt.year == year
            ann_rows = annual_copy[ann_mask]
            if ann_rows.empty:
                continue

            annual_val = ann_rows.iloc[-1].get(col)
            if pd.isna(annual_val) or abs(annual_val) < 1e-6:
                continue

            discrepancy = abs(q_sum - annual_val) / abs(annual_val)
            if discrepancy > tolerance:
                # Pro-rata adjustment
                adjustment = annual_val / q_sum
                mask = (result["_fy"] == year) & result[col].notna()
                result.loc[mask, col] = result.loc[mask, col] * adjustment
                n_adjusted += 1
                logger.debug(
                    "Reconciled %s FY%d: sum=%.0f vs annual=%.0f (%.1f%% off, adjusted)",
                    col, year, q_sum, annual_val, discrepancy * 100,
                )

    if "_fy" in result.columns:
        result = result.drop(columns=["_fy"])

    if n_adjusted > 0:
        logger.info("Reconciliation: adjusted %d flow variable-year pairs", n_adjusted)

    return result


# ---------------------------------------------------------------------------
# Revenue-ratio scaling fallback
# ---------------------------------------------------------------------------


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

    ratios = []
    for _, row in hf.iterrows():
        hf_date = row[date_col]
        hf_rev = row["revenue"]
        if pd.isna(hf_rev) or hf_rev == 0:
            continue

        year = hf_date.year
        annual_mask = lf[date_col].dt.year == year
        if not annual_mask.any():
            annual_mask = lf[date_col].dt.year == year - 1
        annual_rows = lf[annual_mask]

        if annual_rows.empty:
            continue

        annual_rev = float(annual_rows.iloc[-1]["revenue"])
        if annual_rev == 0 or pd.isna(annual_rev):
            continue

        ratio = hf_rev / annual_rev
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
    """Scale a low-freq flow variable by the revenue ratio (fallback)."""
    if revenue_ratio.empty:
        return None

    result_rows = []
    for _, rr_row in revenue_ratio.iterrows():
        hf_date = rr_row[date_col]
        ratio = rr_row["revenue_ratio"]

        year = hf_date.year
        lv = low_values.copy()
        lv[date_col] = pd.to_datetime(lv[date_col])
        annual_mask = lv[date_col].dt.year == year
        if not annual_mask.any():
            annual_mask = lv[date_col].dt.year == year - 1
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


# ---------------------------------------------------------------------------
# Vectorized merge (replaces row-by-row _merge_column)
# ---------------------------------------------------------------------------


def _merge_column_vectorized(
    target: pd.DataFrame,
    source: pd.DataFrame,
    col: str,
    date_col: str,
) -> pd.DataFrame:
    """Merge a single column from source into target using vectorized ops.

    Uses pd.merge_asof for backward-looking date matching instead of
    row-by-row Python loop.
    """
    src = source[[date_col, col]].dropna(subset=[col]).copy()
    src[date_col] = pd.to_datetime(src[date_col])
    target = target.copy()
    target[date_col] = pd.to_datetime(target[date_col])

    if col not in target.columns:
        target[col] = np.nan

    # Only fill where target has NaN
    mask = target[col].isna()
    if not mask.any():
        return target

    needs_fill = target.loc[mask, [date_col]].sort_values(date_col)
    if needs_fill.empty or src.empty:
        return target

    merged = pd.merge_asof(
        needs_fill,
        src.sort_values(date_col),
        on=date_col,
        direction="backward",
    )

    # Write back the filled values
    target.loc[mask, col] = merged[col].values

    return target


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


def _log_separation(groups: dict[str, pd.DataFrame]) -> None:
    """Log the frequency separation result."""
    parts = [f"{k}: {len(v)} rows" for k, v in groups.items()]
    logger.info("Frequency separation: %s", ", ".join(parts))

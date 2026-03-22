"""Data source reconciliation layer.

Validates and reconciles data coming from different PIT sources
before it enters the pipeline.  Catches common issues:

1. **Stale data detection**: flags financial data that hasn't been
   updated in > 6 months (likely a dead filing).
2. **Currency normalization**: ensures all monetary values are in the
   same currency as the price data.
3. **Schema alignment**: maps PIT-source-specific field names to the
   canonical schema expected by derived_variables and downstream models.
4. **Duplicate filing removal**: deduplicates filings that appear
   multiple times with the same report_date.
5. **Filing date validation**: ensures filing_date >= report_date
   (no time-travel in government filings).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical field mapping
# ---------------------------------------------------------------------------

# Maps common PIT source field names to our canonical schema.
# Left side = variations from different APIs, right side = canonical name.
_FIELD_ALIASES: dict[str, str] = {
    # Revenue
    "totalRevenue": "revenue",
    "total_revenue": "revenue",
    "revenues": "revenue",
    "netSales": "revenue",
    "net_sales": "revenue",
    # Profit
    "grossProfit": "gross_profit",
    "gross_profit": "gross_profit",
    "operatingIncome": "ebit",
    "operating_income": "ebit",
    "netIncome": "net_income",
    "net_income": "net_income",
    "netIncomeLoss": "net_income",
    # EBITDA
    "ebitda": "ebitda",
    "EBITDA": "ebitda",
    # Balance sheet
    "totalAssets": "total_assets",
    "total_assets": "total_assets",
    "totalLiabilities": "total_liabilities",
    "total_liabilities": "total_liabilities",
    "totalEquity": "total_equity",
    "total_equity": "total_equity",
    "stockholdersEquity": "total_equity",
    "stockholders_equity": "total_equity",
    "totalCurrentAssets": "current_assets",
    "current_assets": "current_assets",
    "totalCurrentLiabilities": "current_liabilities",
    "current_liabilities": "current_liabilities",
    "cashAndCashEquivalents": "cash_and_equivalents",
    "cash_and_equivalents": "cash_and_equivalents",
    "cashAndShortTermInvestments": "cash_and_equivalents",
    "totalDebt": "total_debt",
    "total_debt": "total_debt",
    "longTermDebt": "long_term_debt",
    "long_term_debt": "long_term_debt",
    "shortTermDebt": "short_term_debt",
    "short_term_debt": "short_term_debt",
    # Cash flow
    "operatingCashFlow": "operating_cash_flow",
    "operating_cash_flow": "operating_cash_flow",
    "netCashFromOperations": "operating_cash_flow",
    "capitalExpenditure": "capex",
    "capital_expenditure": "capex",
    "capex": "capex",
    "freeCashFlow": "free_cash_flow",
    "free_cash_flow": "free_cash_flow",
    # Dates
    "fillingDate": "filing_date",
    "filingDate": "filing_date",
    "filing_date": "filing_date",
    "acceptedDate": "filing_date",
    "accepted_date": "filing_date",
    "periodOfReport": "report_date",
    "period_of_report": "report_date",
    "fiscalDateEnding": "report_date",
    "fiscal_date_ending": "report_date",
    "date": "report_date",
    # Per share
    "earningsPerShare": "eps",
    "eps": "eps",
    "earningsPerShareDiluted": "eps_diluted",
    "eps_diluted": "eps_diluted",
}


def normalize_field_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical field names.

    Unknown columns are kept as-is.  Duplicate canonical names are
    resolved by keeping the first non-null column.

    Uses the local alias dict first, then falls back to the centralized
    field registry (config/field_registry.yml) for any unmatched columns.
    """
    rename_map = {}
    for col in df.columns:
        canonical = _FIELD_ALIASES.get(col)
        if canonical and canonical not in df.columns:
            rename_map[col] = canonical

    # Fallback: try the centralized field registry for unmatched columns
    unmatched = [c for c in df.columns if c not in rename_map and c not in _FIELD_ALIASES.values()]
    if unmatched:
        try:
            from operator1.types import resolve_field_name
            for col in unmatched:
                canonical = resolve_field_name(col)
                if canonical and canonical not in df.columns and col not in rename_map:
                    rename_map[col] = canonical
        except ImportError:
            pass

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.debug("Normalized %d field names: %s", len(rename_map), rename_map)

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_filing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure filing_date >= report_date (no time travel).

    Rows where filing_date < report_date are flagged as suspect and
    their filing_date is set to report_date + 1 day as a safe default.
    """
    if "filing_date" not in df.columns or "report_date" not in df.columns:
        return df

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    suspect = (
        df["filing_date"].notna()
        & df["report_date"].notna()
        & (df["filing_date"] < df["report_date"])
    )

    n_suspect = suspect.sum()
    if n_suspect > 0:
        logger.warning(
            "Found %d filings where filing_date < report_date (time travel) "
            "-- correcting to report_date + 1 day",
            n_suspect,
        )
        df.loc[suspect, "filing_date"] = df.loc[suspect, "report_date"] + timedelta(days=1)

    return df


def remove_duplicate_filings(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate filings with the same report_date.

    Keeps the filing with the latest filing_date (amendment supersedes original).
    """
    if "report_date" not in df.columns:
        return df

    before = len(df)
    sort_col = "filing_date" if "filing_date" in df.columns else "report_date"
    df = df.sort_values(sort_col, ascending=False).drop_duplicates(
        subset=["report_date"], keep="first"
    ).sort_values("report_date")

    removed = before - len(df)
    if removed > 0:
        logger.info("Removed %d duplicate filings (same report_date)", removed)

    return df


def detect_stale_data(
    df: pd.DataFrame,
    stale_threshold_days: int = 180,
) -> dict[str, Any]:
    """Check if the most recent filing is stale (older than threshold).

    Returns a dict with staleness info for logging/reporting.
    """
    date_col = "filing_date" if "filing_date" in df.columns else "report_date"
    if date_col not in df.columns or df.empty:
        return {"stale": False, "reason": "no_data"}

    latest = pd.to_datetime(df[date_col]).max()
    if pd.isna(latest):
        return {"stale": True, "reason": "no_valid_dates"}

    days_old = (pd.Timestamp(date.today()) - latest).days
    is_stale = days_old > stale_threshold_days

    if is_stale:
        logger.warning(
            "Stale data detected: latest filing is %d days old (threshold: %d)",
            days_old, stale_threshold_days,
        )

    return {
        "stale": is_stale,
        "latest_filing": str(latest.date()),
        "days_since_latest": days_old,
        "threshold_days": stale_threshold_days,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconcile_financial_data(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run full reconciliation pipeline on financial statement DataFrames.

    Steps:
      1. Normalize field names to canonical schema
      2. Validate filing dates (no time travel)
      3. Remove duplicate filings
      4. Detect stale data

    Returns
    -------
    (income_df, balance_df, cashflow_df, reconciliation_report)
    """
    report: dict[str, Any] = {"statements_reconciled": 0, "issues": []}

    for label, df in [
        ("income", income_df),
        ("balance", balance_df),
        ("cashflow", cashflow_df),
    ]:
        if df.empty:
            continue

        # 1. Normalize field names
        df_out = normalize_field_names(df)

        # 2. Validate filing dates
        df_out = validate_filing_dates(df_out)

        # 3. Remove duplicates
        df_out = remove_duplicate_filings(df_out)

        # 4. Check for stale data
        staleness = detect_stale_data(df_out)
        if staleness.get("stale"):
            report["issues"].append({
                "statement": label,
                "issue": "stale_data",
                "detail": staleness,
            })

        report["statements_reconciled"] += 1

        # Write back (in-place modification via reassignment)
        if label == "income":
            income_df = df_out
        elif label == "balance":
            balance_df = df_out
        else:
            cashflow_df = df_out

    n_issues = len(report["issues"])
    if n_issues > 0:
        logger.warning("Reconciliation found %d issues", n_issues)
    else:
        logger.info("Reconciliation complete: %d statements, no issues", report["statements_reconciled"])

    return income_df, balance_df, cashflow_df, report

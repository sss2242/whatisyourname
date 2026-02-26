"""Step D -- As-of daily cache builder and direct field storage.

Builds a daily-frequency DataFrame for each entity by:
  1. Generating a business-day date index for the 2-year window.
  2. Starting with price data (OHLCV from PIT provider).
  3. Applying as-of joins for periodic financial statements -- for each
     day ``t``, the latest row with ``report_date <= t`` is attached.
  4. Merging static profile fields as constant columns.
  5. Adding ``is_missing_<field>`` companion flags for every column.

**Critical invariant**: no look-ahead is allowed.  If any statement row
has ``report_date > t`` when applied to day ``t``, the pipeline raises
``LookAheadError``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import CACHE_DIR, DATE_START, DATE_END
from operator1.config_loader import get_global_config
from operator1.steps.data_extraction import EntityData, ExtractionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LookAheadError(Exception):
    """Raised when a financial statement leaks future data into a past day."""

    def __init__(self, entity: str, statement: str, day: str, report_date: str) -> None:
        self.entity = entity
        self.statement = statement
        self.day = day
        self.report_date = report_date
        super().__init__(
            f"Look-ahead violation for {entity}: {statement} report_date "
            f"{report_date} applied to day {day}"
        )


# ---------------------------------------------------------------------------
# Column schema (Sec 8 direct fields)
# ---------------------------------------------------------------------------

# Profile fields -- stored as static columns
PROFILE_FIELDS = (
    "isin", "ticker", "exchange", "currency", "country",
    "sector", "industry", "sub_industry",
)

# Quote / price columns expected from PIT provider
QUOTE_FIELDS = (
    "open", "high", "low", "close", "volume",
    "adjusted_close", "vwap", "market_cap", "shares_outstanding",
)

# Financial-statement decision variables (Sec 8)
STATEMENT_FIELDS = (
    "revenue", "cost_of_revenue", "gross_profit",
    "operating_income", "ebit", "ebitda", "net_income",
    "interest_expense", "taxes",
    "total_assets", "total_liabilities", "total_equity",
    "current_assets", "current_liabilities",
    "cash_and_equivalents", "short_term_debt", "long_term_debt",
    "total_debt", "retained_earnings",
    "goodwill", "intangible_assets",
    "receivables", "inventory", "payables",
    "operating_cash_flow", "capex", "free_cash_flow",
    "investing_cf", "financing_cf", "dividends_paid",
    "stock_buybacks",
    "sga_expenses", "rd_expenses",
    "eps", "eps_diluted",
)

# Macro indicator columns persisted in the daily cache so that
# downstream models (survival timeline, walk-forward) can consume them
# as features without re-fetching.
MACRO_INDICATOR_FIELDS = (
    "gdp_growth", "gdp_current_usd",
    "inflation_rate_yoy", "inflation_rate_daily_equivalent",
    "real_interest_rate", "lending_interest_rate",
    "unemployment_rate",
    "official_exchange_rate_lcu_per_usd",
    "credit_spread", "yield_curve_slope", "fx_volatility",
    "real_return_1d",
)

# Company protection / survival score columns persisted alongside the
# cache so walk-forward and survival timeline can use them as features.
PROTECTION_SCORE_FIELDS = (
    "company_survival_mode_flag",
    "country_survival_mode_flag",
    "country_protected_flag",
    "fuzzy_protection_degree",
    "fuzzy_sector_score",
    "fuzzy_economic_score",
    "fuzzy_policy_score",
    "sector_strategicness",
)


# ---------------------------------------------------------------------------
# Date index
# ---------------------------------------------------------------------------


def build_date_index(
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Generate a complete business-day index for the 2-year window.

    Parameters
    ----------
    start, end:
        Override start/end dates (mainly for testing).  Defaults to
        ``DATE_START`` / ``DATE_END`` from constants.

    Returns
    -------
    pd.DatetimeIndex
        Business-day frequency index.
    """
    s = pd.Timestamp(start or DATE_START)
    e = pd.Timestamp(end or DATE_END)
    return pd.bdate_range(start=s, end=e, name="date")


# ---------------------------------------------------------------------------
# As-of join helpers
# ---------------------------------------------------------------------------


def _validate_no_lookahead(
    daily: pd.DataFrame,
    stmt_df: pd.DataFrame,
    entity_id: str,
    stmt_name: str,
) -> None:
    """Assert that no statement row is applied to a day earlier than its report_date.

    We check by merging: for any row in ``daily`` where a statement was
    joined, the ``_report_date`` column must be ``<= date``.
    """
    if "_report_date" not in daily.columns:
        return
    violations = daily.loc[
        daily["_report_date"].notna()
        & (daily["_report_date"] > daily.index)
    ]
    if not violations.empty:
        bad = violations.iloc[0]
        raise LookAheadError(
            entity=entity_id,
            statement=stmt_name,
            day=str(bad.name.date()),
            report_date=str(bad["_report_date"].date()),
        )


def _asof_merge_statement(
    daily_index: pd.DatetimeIndex,
    stmt_df: pd.DataFrame,
    entity_id: str,
    stmt_name: str,
    fields: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """As-of merge a periodic statement onto a daily date index.

    For each business day ``t``, we attach the latest statement row
    whose ``report_date <= t``.

    Parameters
    ----------
    daily_index:
        The business-day DatetimeIndex to align to.
    stmt_df:
        Periodic statement DataFrame.  Must contain a ``report_date``
        column (datetime).
    entity_id:
        Used for error messages.
    stmt_name:
        E.g. ``"income_statement"``, used for error messages.
    fields:
        Subset of columns to keep from the statement.  If *None*, all
        non-date columns are kept.

    Returns
    -------
    pd.DataFrame
        Indexed by ``daily_index`` with statement columns filled via
        as-of logic.
    """
    if stmt_df.empty:
        logger.debug("Empty %s for %s -- returning nulls", stmt_name, entity_id)
        result = pd.DataFrame(index=daily_index)
        if fields:
            for f in fields:
                result[f] = np.nan
        return result

    # Ensure report_date is datetime
    df = stmt_df.copy()
    if "report_date" not in df.columns:
        # Try common alternatives
        for alt in ("date", "fillingDate", "acceptedDate"):
            if alt in df.columns:
                df = df.rename(columns={alt: "report_date"})
                break
    if "report_date" not in df.columns:
        logger.warning(
            "No report_date column in %s for %s -- returning nulls",
            stmt_name, entity_id,
        )
        result = pd.DataFrame(index=daily_index)
        if fields:
            for f in fields:
                result[f] = np.nan
        return result

    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"]).sort_values("report_date")

    # Normalise column names to snake_case where possible
    rename_map = _build_column_rename_map(df.columns.tolist())
    if rename_map:
        df = df.rename(columns=rename_map)

    # Select fields
    keep_cols = ["report_date"]
    if fields:
        keep_cols += [f for f in fields if f in df.columns]
    else:
        keep_cols += [c for c in df.columns if c != "report_date"]
    df = df[keep_cols].drop_duplicates(subset=["report_date"], keep="last")

    # Build daily frame for merge_asof
    daily_frame = pd.DataFrame({"date": daily_index})
    daily_frame["date"] = pd.to_datetime(daily_frame["date"])

    merged = pd.merge_asof(
        daily_frame.sort_values("date"),
        df.rename(columns={"report_date": "_report_date"}).sort_values("_report_date"),
        left_on="date",
        right_on="_report_date",
        direction="backward",
    )

    # Validate no look-ahead
    merged = merged.set_index("date")
    _validate_no_lookahead(merged, df, entity_id, stmt_name)

    # Drop the helper column
    merged = merged.drop(columns=["_report_date"], errors="ignore")

    return merged


def _build_column_rename_map(columns: list[str]) -> dict[str, str]:
    """Build a mapping from camelCase / odd API names to snake_case decision vars.

    Only returns mappings for columns that match known statement fields.
    Covers PIT provider field names across all supported markets.
    """
    # Common API -> canonical mappings
    known: dict[str, str] = {
        "totalRevenue": "revenue",
        "grossProfit": "gross_profit",
        "operatingIncome": "ebit",
        "ebitda": "ebitda",
        "netIncome": "net_income",
        "interestExpense": "interest_expense",
        "incomeTaxExpense": "taxes",
        "totalAssets": "total_assets",
        "totalLiabilities": "total_liabilities",
        # Some data sources may use "totalLiab" as an abbreviation;
        # kept here as a safety net for column normalisation.
        "totalLiab": "total_liabilities",
        "totalStockholdersEquity": "total_equity",
        "totalStockholderEquity": "total_equity",  # alternate spelling variant
        "totalEquity": "total_equity",
        "totalCurrentAssets": "current_assets",
        "currentAssets": "current_assets",
        "totalCurrentLiabilities": "current_liabilities",
        "currentLiabilities": "current_liabilities",
        "cashAndCashEquivalents": "cash_and_equivalents",
        "cashAndShortTermInvestments": "cash_and_equivalents",
        "cash": "cash_and_equivalents",  # EOD short name
        "shortTermDebt": "short_term_debt",
        "longTermDebt": "long_term_debt",
        "longTermDebtTotal": "long_term_debt",  # EOD variant
        "netReceivables": "receivables",
        "accountsReceivable": "receivables",
        "operatingCashFlow": "operating_cash_flow",
        "totalCashFromOperatingActivities": "operating_cash_flow",  # EOD
        "capitalExpenditure": "capex",
        "capitalExpenditures": "capex",  # EOD variant
        "investingCashFlow": "investing_cf",
        "investingActivitiesCashflow": "investing_cf",
        "totalCashflowsFromInvestingActivities": "investing_cf",  # EOD
        "financingCashFlow": "financing_cf",
        "financingActivitiesCashflow": "financing_cf",
        "totalCashFromFinancingActivities": "financing_cf",  # EOD
        "dividendsPaid": "dividends_paid",
        "paymentOfDividends": "dividends_paid",  # EOD variant
        "freeCashFlow": "free_cash_flow",
        "commonStockRepurchased": "stock_buybacks",
        "stockBuybacks": "stock_buybacks",
        "totalDebt": "total_debt",
        "sgaExpenses": "sga_expenses",
        "sellingGeneralAndAdministrativeExpenses": "sga_expenses",
        "sga_expense": "sga_expenses",   # alias from older client output
        "rdExpenses": "rd_expenses",
        "researchAndDevelopmentExpenses": "rd_expenses",
        "research_and_development": "rd_expenses",  # alias from older client output
        # J-Quants V2 wide-format column names (jp_jquants_wrapper.py)
        "operating_cashflow": "operating_cash_flow",
        "investing_cashflow": "investing_cf",
        "financing_cashflow": "financing_cf",
        "pretax_income": "ebit",
        "eps": "eps",
        "eps_basic": "eps",  # canonical translator outputs eps_basic
        "epsDiluted": "eps_diluted",
        "sharesOutstanding": "shares_outstanding",
        "marketCap": "market_cap",
        "marketCapitalization": "market_cap",  # EOD variant
        "adjustedClose": "adjusted_close",
    }
    return {k: v for k, v in known.items() if k in columns and k != v}


# ---------------------------------------------------------------------------
# Price data alignment
# ---------------------------------------------------------------------------


def _align_price_data(
    daily_index: pd.DatetimeIndex,
    price_df: pd.DataFrame,
    entity_id: str,
) -> pd.DataFrame:
    """Align price data (OHLCV from PIT provider) to the daily index.

    Forward-fills the last known price for non-trading days that fall on
    business days (e.g. holidays).
    """
    if price_df.empty:
        logger.warning("No price data for %s -- columns will be null", entity_id)
        result = pd.DataFrame(index=daily_index)
        for f in QUOTE_FIELDS:
            result[f] = np.nan
        return result

    df = price_df.copy()

    # Normalise column names
    rename_map = _build_column_rename_map(df.columns.tolist())
    if rename_map:
        df = df.rename(columns=rename_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()

    # Reindex to daily and forward-fill
    df = df.reindex(daily_index)
    df = df.ffill()

    # Keep only known quote columns (plus any extras that exist)
    return df


# ---------------------------------------------------------------------------
# Profile fields
# ---------------------------------------------------------------------------


def _add_profile_columns(
    daily: pd.DataFrame,
    profile: dict[str, Any],
) -> pd.DataFrame:
    """Add static profile fields as constant columns."""
    for f in PROFILE_FIELDS:
        daily[f] = profile.get(f)
    return daily


# ---------------------------------------------------------------------------
# Missing-data flags
# ---------------------------------------------------------------------------


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """For every non-flag column, add ``is_missing_<col>`` companion.

    Flags are 1 where the original value is NaN/None, 0 otherwise.
    """
    flag_cols: dict[str, pd.Series] = {}
    for col in df.columns:
        if col.startswith("is_missing_") or col.startswith("invalid_math_"):
            continue
        flag_name = f"is_missing_{col}"
        if flag_name not in df.columns:
            flag_cols[flag_name] = df[col].isna().astype(int)
    if flag_cols:
        flags_df = pd.DataFrame(flag_cols, index=df.index)
        df = pd.concat([df, flags_df], axis=1)
    return df


# ---------------------------------------------------------------------------
# Single entity cache builder
# ---------------------------------------------------------------------------


def build_entity_daily_cache(
    entity: EntityData,
    daily_index: pd.DatetimeIndex,
    use_ohlcv_prices: bool = False,
) -> pd.DataFrame:
    """Build a single daily DataFrame for one entity.

    Parameters
    ----------
    entity:
        Raw ``EntityData`` from the extraction phase.
    daily_index:
        Business-day date index.
    use_ohlcv_prices:
        If True, use ``entity.ohlcv`` as the price source (target).
        Otherwise use ``entity.quotes`` (linked entities).

    Returns
    -------
    pd.DataFrame
        Daily-frequency DataFrame with all direct fields and missing flags.
    """
    isin = entity.isin

    # 1. Price data
    price_source = entity.ohlcv if use_ohlcv_prices else entity.quotes
    daily = _align_price_data(daily_index, price_source, isin)

    # 2. Financial statements -- as-of merge
    for stmt_name, stmt_df in [
        ("income_statement", entity.income_statement),
        ("balance_sheet", entity.balance_sheet),
        ("cashflow_statement", entity.cashflow_statement),
    ]:
        stmt_daily = _asof_merge_statement(
            daily_index, stmt_df, isin, stmt_name,
            fields=STATEMENT_FIELDS,
        )
        # Merge, avoiding duplicate columns
        new_cols = [c for c in stmt_daily.columns if c not in daily.columns]
        if new_cols:
            daily = daily.join(stmt_daily[new_cols])

    # 3. Profile fields
    daily = _add_profile_columns(daily, entity.profile)

    # 4. Ensure all expected columns exist
    for f in QUOTE_FIELDS:
        if f not in daily.columns:
            daily[f] = np.nan
    for f in STATEMENT_FIELDS:
        if f not in daily.columns:
            daily[f] = np.nan

    # 4b. Ensure macro indicator placeholders exist (filled later by
    # enrich_cache_with_indicators) so downstream code can always
    # reference them without KeyError.
    for f in MACRO_INDICATOR_FIELDS:
        if f not in daily.columns:
            daily[f] = np.nan
    for f in PROTECTION_SCORE_FIELDS:
        if f not in daily.columns:
            daily[f] = np.nan

    # 5. Missing-data flags
    daily = add_missing_flags(daily)

    return daily


# ---------------------------------------------------------------------------
# Indicator enrichment (Phase 1 -- survival time-series)
# ---------------------------------------------------------------------------


def enrich_cache_with_indicators(
    daily: pd.DataFrame,
    macro_aligned: pd.DataFrame | None = None,
    survival_flags: pd.DataFrame | None = None,
    fuzzy_result_series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge macro indicators and protection scores into the daily cache.

    This is the Phase 1 entry point that ensures every model-consumable
    feature is persisted in the 2-year daily Parquet cache.

    Parameters
    ----------
    daily:
        The entity daily cache (output of ``build_entity_daily_cache``).
    macro_aligned:
        Output of ``align_macro_to_daily()`` -- contains GDP, inflation,
        unemployment, rates, FX vol columns aligned to daily frequency.
    survival_flags:
        DataFrame with ``company_survival_mode_flag``,
        ``country_survival_mode_flag``, ``country_protected_flag`` columns
        (output of ``compute_survival_flags``).
    fuzzy_result_series:
        DataFrame with fuzzy protection columns:
        ``fuzzy_protection_degree``, ``fuzzy_sector_score``,
        ``fuzzy_economic_score``, ``fuzzy_policy_score``.

    Returns
    -------
    pd.DataFrame
        Enriched daily cache with all indicator columns populated.
    """
    result = daily.copy()

    # --- Macro indicators ---
    if macro_aligned is not None and not macro_aligned.empty:
        for col in MACRO_INDICATOR_FIELDS:
            if col in macro_aligned.columns:
                result[col] = macro_aligned[col].reindex(result.index)
        logger.info(
            "Enriched cache with %d macro indicator columns",
            sum(1 for c in MACRO_INDICATOR_FIELDS if c in macro_aligned.columns),
        )

    # --- Survival flags ---
    if survival_flags is not None and not survival_flags.empty:
        for col in ("company_survival_mode_flag", "country_survival_mode_flag",
                     "country_protected_flag"):
            if col in survival_flags.columns:
                result[col] = survival_flags[col].reindex(result.index)

    # --- Fuzzy protection scores ---
    if fuzzy_result_series is not None and not fuzzy_result_series.empty:
        fuzzy_cols = (
            "fuzzy_protection_degree", "fuzzy_sector_score",
            "fuzzy_economic_score", "fuzzy_policy_score",
        )
        for col in fuzzy_cols:
            if col in fuzzy_result_series.columns:
                result[col] = fuzzy_result_series[col].reindex(result.index)

    # --- Sector strategicness (static per entity, from profile) ---
    if "sector" in result.columns:
        try:
            from operator1.analysis.fuzzy_protection import _sector_membership
            sector_vals = result["sector"].fillna("")
            result["sector_strategicness"] = sector_vals.apply(
                lambda s: _sector_membership(s) if isinstance(s, str) else 0.1,
            )
        except ImportError:
            pass

    # Refresh missing-data flags for newly added columns
    result = add_missing_flags(result)

    logger.info(
        "Cache enrichment complete: %d total columns",
        len(result.columns),
    )
    return result


# ---------------------------------------------------------------------------
# Full pipeline builder
# ---------------------------------------------------------------------------


@dataclass
class CacheResult:
    """Output of the full cache-building step."""

    target_daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    linked_daily: dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)


def build_all_caches(
    extraction: ExtractionResult,
    force_rebuild: bool | None = None,
) -> CacheResult:
    """Build daily caches for the target and all linked entities.

    Parameters
    ----------
    extraction:
        Raw data from ``extract_all_data()``.
    force_rebuild:
        Override for FORCE_REBUILD config flag.

    Returns
    -------
    CacheResult
        Contains daily DataFrames for target and each linked entity.
    """
    cfg = get_global_config()
    if force_rebuild is None:
        force_rebuild = cfg.get("FORCE_REBUILD", False)

    result = CacheResult()
    daily_index = build_date_index()

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------
    target_cache_path = Path(CACHE_DIR) / "target_company_daily.parquet"
    if not force_rebuild and target_cache_path.exists():
        logger.info("Loading target daily cache from disk ...")
        result.target_daily = pd.read_parquet(target_cache_path)
    else:
        logger.info("Building target daily cache (%s) ...", extraction.target.isin)
        try:
            result.target_daily = build_entity_daily_cache(
                extraction.target, daily_index, use_ohlcv_prices=True,
            )
        except LookAheadError as exc:
            logger.error("FATAL: %s", exc)
            result.errors.append({
                "entity": extraction.target.isin,
                "module": "cache_builder",
                "error": str(exc),
            })
            raise

    # ------------------------------------------------------------------
    # Linked entities
    # ------------------------------------------------------------------
    linked_cache_path = Path(CACHE_DIR) / "linked_entities_daily.parquet"
    if not force_rebuild and linked_cache_path.exists():
        logger.info("Loading linked entities daily cache from disk ...")
        combined = pd.read_parquet(linked_cache_path)
        for isin in combined["isin"].unique():
            result.linked_daily[isin] = combined[combined["isin"] == isin].copy()
    else:
        total = len(extraction.linked)
        for idx, (isin, entity) in enumerate(extraction.linked.items(), 1):
            logger.info(
                "Building linked entity cache %d/%d: %s ...", idx, total, isin,
            )
            try:
                result.linked_daily[isin] = build_entity_daily_cache(
                    entity, daily_index, use_ohlcv_prices=False,
                )
            except LookAheadError as exc:
                logger.error("FATAL: %s", exc)
                result.errors.append({
                    "entity": isin,
                    "module": "cache_builder",
                    "error": str(exc),
                })
                raise
            except Exception as exc:
                logger.warning("Cache build failed for %s: %s", isin, exc)
                result.errors.append({
                    "entity": isin,
                    "module": "cache_builder",
                    "error": str(exc),
                })

    logger.info(
        "Cache build complete: target=%s, linked=%d, errors=%d",
        bool(not result.target_daily.empty),
        len(result.linked_daily),
        len(result.errors),
    )

    return result


def persist_caches(result: CacheResult) -> None:
    """Write daily caches to disk as Parquet.

    Outputs:
    - ``cache/target_company_daily.parquet``
    - ``cache/linked_entities_daily.parquet``
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Target
    if not result.target_daily.empty:
        path = Path(CACHE_DIR) / "target_company_daily.parquet"
        result.target_daily.to_parquet(path)
        logger.info("Target daily cache saved: %s (%d rows)", path, len(result.target_daily))

    # Linked -- concatenate all into a single Parquet
    if result.linked_daily:
        frames = []
        for isin, df in result.linked_daily.items():
            frame = df.copy()
            # Ensure isin column is present for partitioning
            if "isin" not in frame.columns:
                frame["isin"] = isin
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=False)
        path = Path(CACHE_DIR) / "linked_entities_daily.parquet"
        combined.to_parquet(path)
        logger.info(
            "Linked entities daily cache saved: %s (%d rows, %d entities)",
            path, len(combined), len(result.linked_daily),
        )

"""Shared type definitions, column registries, and field registry for the Operator 1 pipeline.

Extracted from legacy modules (cache_builder, data_extraction, verify_identifiers)
to provide a single import location for types used across the codebase.

The field registry (config/field_registry.yml) is the single source of truth
for canonical field names and their aliases across all regional APIs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verified target (from verify_identifiers.py)
# ---------------------------------------------------------------------------

@dataclass
class VerifiedTarget:
    """Container for a successfully verified target company.

    All fields are populated during the verification step and
    remain immutable for the rest of the pipeline.
    """

    isin: str
    ticker: str
    name: str
    country: str          # ISO-2 from PIT provider profile
    sector: str
    industry: str
    sub_industry: str | None
    fmp_symbol: str       # kept for backward compat; same as ticker in new arch
    currency: str
    exchange: str
    raw_profile: dict[str, Any] = field(default_factory=dict, repr=False)


class VerificationError(Exception):
    """Raised when identifier verification fails."""

    def __init__(self, source: str, detail: str) -> None:
        self.source = source
        self.detail = detail
        super().__init__(f"[{source}] Verification failed: {detail}")


# ---------------------------------------------------------------------------
# Entity data containers (from data_extraction.py)
# ---------------------------------------------------------------------------

@dataclass
class EntityData:
    """Raw data container for a single entity."""

    isin: str
    profile: dict[str, Any] = field(default_factory=dict)
    quotes: pd.DataFrame = field(default_factory=pd.DataFrame)
    income_statement: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow_statement: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Target-only extras
    peers: list[str] = field(default_factory=list)
    supply_chain: list[dict[str, Any]] = field(default_factory=list)
    executives: list[dict[str, Any]] = field(default_factory=list)

    # OHLCV price data (from PIT provider)
    ohlcv: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ExtractionResult:
    """Container for all extracted data."""

    target: EntityData = field(default_factory=lambda: EntityData(isin=""))
    linked: dict[str, EntityData] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache exceptions (from cache_builder.py)
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
# Column registries (from cache_builder.py)
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

# Macro indicator columns persisted in the daily cache
MACRO_INDICATOR_FIELDS = (
    "gdp_growth", "gdp_current_usd",
    "inflation_rate_yoy", "inflation_rate_daily_equivalent",
    "real_interest_rate", "lending_interest_rate",
    "unemployment_rate",
    "official_exchange_rate_lcu_per_usd",
    "credit_spread", "yield_curve_slope", "fx_volatility",
    "real_return_1d",
)

# Company protection / survival score columns
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

# Conflict / geopolitical risk columns
CONFLICT_RISK_FIELDS = (
    "country_conflict_flag",
    "company_conflict_flag",
    "conflict_intensity_score",
    "sanctions_flag",
    "fragile_state_flag",
    "conflict_type",
    "supply_chain_risk_score",
    "revenue_exposure_score",
    "competitive_advantage_score",
)


# ---------------------------------------------------------------------------
# Field registry loader (config/field_registry.yml)
# ---------------------------------------------------------------------------

_field_registry_cache: dict[str, Any] | None = None
_alias_to_canonical_cache: dict[str, str] | None = None


def _load_field_registry() -> dict[str, Any]:
    """Load and cache the field registry from config/field_registry.yml."""
    global _field_registry_cache
    if _field_registry_cache is not None:
        return _field_registry_cache

    try:
        import yaml
        registry_path = Path(__file__).resolve().parent.parent / "config" / "field_registry.yml"
        if not registry_path.exists():
            logger.debug("Field registry not found at %s", registry_path)
            _field_registry_cache = {}
            return _field_registry_cache

        with open(registry_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        _field_registry_cache = data
        return data
    except Exception as exc:
        logger.debug("Failed to load field registry: %s", exc)
        _field_registry_cache = {}
        return _field_registry_cache


def get_alias_map() -> dict[str, str]:
    """Build a flat alias -> canonical_name mapping from the field registry.

    Returns a dict where keys are all known aliases (including the
    canonical name itself) and values are the canonical field name.
    This replaces the scattered alias dictionaries in canonical_translator,
    data_reconciliation, and cache_builder.
    """
    global _alias_to_canonical_cache
    if _alias_to_canonical_cache is not None:
        return _alias_to_canonical_cache

    registry = _load_field_registry()
    alias_map: dict[str, str] = {}

    for canonical_name, info in registry.items():
        if not isinstance(info, dict):
            continue
        # The canonical name maps to itself
        alias_map[canonical_name] = canonical_name
        # All aliases map to the canonical name
        for alias in info.get("aliases", []):
            alias_str = str(alias)
            if alias_str not in alias_map:
                alias_map[alias_str] = canonical_name

    _alias_to_canonical_cache = alias_map
    logger.debug("Field registry loaded: %d aliases -> %d canonical fields",
                 len(alias_map), len(registry))
    return alias_map


def get_field_info(canonical_name: str) -> dict[str, Any]:
    """Get metadata for a canonical field (statement_type, variable_type, tier).

    Returns empty dict if the field is not in the registry.
    """
    registry = _load_field_registry()
    info = registry.get(canonical_name, {})
    return info if isinstance(info, dict) else {}


def resolve_field_name(raw_name: str) -> str:
    """Resolve a raw field name to its canonical equivalent using the registry.

    Returns the canonical name if found, or empty string if not matched.
    This is a faster alternative to the 3-module lookup chain.
    """
    alias_map = get_alias_map()

    # Exact match
    if raw_name in alias_map:
        return alias_map[raw_name]

    # Case-insensitive match
    raw_lower = raw_name.lower().replace(" ", "_")
    for alias, canonical in alias_map.items():
        if alias.lower().replace(" ", "_") == raw_lower:
            return canonical

    return ""

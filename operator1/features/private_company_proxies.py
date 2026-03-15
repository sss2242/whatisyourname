"""Private company proxy variables -- substitutes for OHLCV-derived metrics.

When a company has no publicly traded shares (private, government entity,
cooperative, pre-IPO startup), OHLCV price data is unavailable.  This module
computes proxy variables from financial statement data that serve the same
analytical role as price-derived metrics.

Proxy mapping:
    close          -> total_equity (book value as value proxy)
    return_1d      -> equity_change_rate (QoQ equity change, interpolated daily)
    volatility_21d -> financial_volatility (rolling std of equity change rate)
    drawdown_252d  -> equity_drawdown (max decline in equity from peak)
    volume         -> revenue_velocity (rolling revenue change)
    market_cap     -> total_equity (book value proxy)

Extended proxy mapping (new):
    open           -> equity_value (same as close for private)
    high           -> equity_value (same as close for private)
    low            -> equity_value (same as close for private)
    return_5d      -> revenue_momentum_5d (5-period revenue change)
    return_21d     -> revenue_momentum_21d (21-period revenue change)
    volatility_63d -> earnings_volatility (rolling std of net_income change)

Top-level entry points:
    ``compute_private_company_proxies(cache)`` -- adds proxy columns
    ``resolve_proxies(cache)`` -- writes proxy values INTO standard column
        names so downstream modules work without any code changes.

The proxy columns are added alongside the standard column names so that
downstream models (regime detection, forecasting, Monte Carlo) can consume
them without modification when operating in private company mode.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Mapping from standard OHLCV column names to their private company proxies.
# Used by downstream models to determine which variable to target.
PROXY_MAP: dict[str, str] = {
    "close": "equity_value",
    "open": "equity_value",
    "high": "equity_value",
    "low": "equity_value",
    "return_1d": "equity_change_rate",
    "volatility_21d": "financial_volatility",
    "drawdown_252d": "equity_drawdown",
    "volume": "revenue_velocity",
    "market_cap": "equity_value",
}

# Extended proxies that provide additional analytical signals.
EXTENDED_PROXY_MAP: dict[str, str] = {
    "return_5d": "revenue_momentum_5d",
    "return_21d": "revenue_momentum_21d",
    "volatility_63d": "earnings_volatility",
}

# Private mode forecast targets (replaces price-based targets)
PRIVATE_FORECAST_TARGETS: list[str] = [
    "total_equity",
    "revenue",
    "net_income",
    "operating_cash_flow",
    "free_cash_flow",
    "total_debt",
    "current_ratio",
    "debt_to_equity_abs",
    "cash_ratio",
    "net_margin",
    "roe",
    "roa",
]

# Private mode tier variables (replaces price-based tier variables)
PRIVATE_TIER_VARIABLES: dict[str, list[str]] = {
    "tier1": ["cash_ratio", "current_ratio", "quick_ratio"],
    "tier2": ["debt_to_equity_abs", "net_debt_to_ebitda"],
    "tier3": ["financial_volatility", "equity_drawdown"],
    "tier4": ["net_margin", "roe", "roa"],
    "tier5": ["revenue_growth_yoy", "earnings_growth_yoy"],
}

# Confidence scores for each proxy -- quantifies how trustworthy the proxy
# is relative to the real OHLCV column it replaces.  Higher = more reliable.
PROXY_CONFIDENCE: dict[str, float] = {
    "equity_value": 0.85,             # directly from balance sheet
    "equity_change_rate": 0.55,       # interpolated between quarters
    "financial_volatility": 0.40,     # derived from interpolated data
    "equity_drawdown": 0.75,          # straightforward peak-decline
    "revenue_velocity": 0.65,         # direct from income statement
    "enterprise_value_proxy": 0.70,   # composite of 3 balance sheet items
    "implied_pe_proxy": 0.60,         # depends on net_income accuracy
    "cash_burn_rate": 0.80,           # direct from cash flow statement
    "debt_service_coverage": 0.75,    # direct from CF + interest
    "revenue_momentum_5d": 0.50,      # interpolated short-term
    "revenue_momentum_21d": 0.55,     # interpolated medium-term
    "earnings_volatility": 0.45,      # derived from interpolated NI
    "balance_sheet_leverage_change": 0.65,  # direct ratio change
}


def is_private_company(cache: pd.DataFrame) -> bool:
    """Detect whether the cache represents a private company (no OHLCV).

    Returns True if the cache lacks usable OHLCV price data, meaning
    the company is either private, delisted, or the OHLCV fetch failed.

    Parameters
    ----------
    cache:
        Daily cache DataFrame from the pipeline.
    """
    if "close" not in cache.columns:
        return True
    if cache["close"].notna().sum() < 5:
        return True
    return False


def compute_private_company_proxies(cache: pd.DataFrame) -> pd.DataFrame:
    """Compute proxy variables for a private company (no OHLCV data).

    Adds proxy columns that serve the same analytical role as OHLCV-derived
    metrics.  These proxies enable regime detection, forecasting, Monte Carlo,
    and other temporal models to run on financial statement data alone.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with financial statement columns
        (total_equity, revenue, net_income, etc.) but NO close/OHLCV data.

    Returns
    -------
    cache with additional proxy columns added.
    """
    if not is_private_company(cache):
        logger.debug("Cache has OHLCV data -- skipping private company proxies")
        return cache

    logger.info("Private company mode: computing proxy variables from financial statements")

    # --- Equity value (proxy for close/market_cap) ---
    if "total_equity" in cache.columns:
        cache["equity_value"] = cache["total_equity"].copy()
        logger.info("  equity_value: from total_equity (%d non-NaN)",
                     cache["equity_value"].notna().sum())
    elif "total_assets" in cache.columns and "total_liabilities" in cache.columns:
        cache["equity_value"] = cache["total_assets"] - cache["total_liabilities"]
        logger.info("  equity_value: from total_assets - total_liabilities")
    else:
        logger.warning("  equity_value: cannot compute (no equity or assets/liabilities)")

    # --- Equity change rate (proxy for return_1d) ---
    # The raw equity value is forward-filled from quarterly filings, so
    # simple pct_change() would produce 0 for ~90% of days and a spike
    # at quarter boundaries.  Instead, we interpolate between quarterly
    # values to produce a smooth daily rate of change that temporal
    # models (regime detection, Monte Carlo, forecasting) can consume.
    if "equity_value" in cache.columns and cache["equity_value"].notna().sum() >= 2:
        ev = cache["equity_value"]
        # Detect quarter transitions (where value changes)
        ev_shifted = ev.shift(1)
        is_new = ev.notna() & (ev != ev_shifted) & ev_shifted.notna()
        # Get the distinct quarterly values at their transition points
        quarterly_vals = ev.where(is_new | (ev.index == ev.first_valid_index()))
        # Interpolate linearly between quarters to get smooth daily values
        ev_smooth = quarterly_vals.interpolate(method="time")
        # Fill any remaining NaN at the edges
        ev_smooth = ev_smooth.ffill().bfill()
        # Daily change rate from the smoothly interpolated series
        cache["equity_change_rate"] = ev_smooth.pct_change().fillna(0.0)
        # Clip extreme values
        cache["equity_change_rate"] = cache["equity_change_rate"].clip(-0.1, 0.1)
        n_nonzero = (cache["equity_change_rate"].abs() > 1e-8).sum()
        logger.info("  equity_change_rate: %d non-zero values (smooth interpolation)", n_nonzero)
    else:
        cache["equity_change_rate"] = 0.0
        logger.warning("  equity_change_rate: defaulting to 0 (insufficient equity data)")

    # --- Financial volatility (proxy for volatility_21d) ---
    if "equity_change_rate" in cache.columns:
        ecr = cache["equity_change_rate"]
        cache["financial_volatility"] = ecr.rolling(
            window=63, min_periods=10
        ).std() * np.sqrt(252)
        cache["financial_volatility"] = cache["financial_volatility"].fillna(
            ecr.expanding(min_periods=5).std() * np.sqrt(252)
        )
        logger.info("  financial_volatility: %d non-NaN values",
                     cache["financial_volatility"].notna().sum())

    # --- Equity drawdown (proxy for drawdown_252d) ---
    if "equity_value" in cache.columns and cache["equity_value"].notna().sum() >= 5:
        ev = cache["equity_value"]
        rolling_max = ev.expanding().max()
        cache["equity_drawdown"] = (ev - rolling_max) / rolling_max.replace(0, np.nan)
        cache["equity_drawdown"] = cache["equity_drawdown"].fillna(0.0)
        logger.info("  equity_drawdown: min=%.3f", cache["equity_drawdown"].min())
    else:
        cache["equity_drawdown"] = 0.0

    # --- Revenue velocity (proxy for volume) ---
    if "revenue" in cache.columns and cache["revenue"].notna().sum() >= 2:
        cache["revenue_velocity"] = cache["revenue"].pct_change().fillna(0.0)
        cache["revenue_velocity"] = cache["revenue_velocity"].clip(-1.0, 1.0)
        logger.info("  revenue_velocity: computed")
    else:
        cache["revenue_velocity"] = 0.0

    # ------------------------------------------------------------------
    # Extended proxies (new)
    # ------------------------------------------------------------------

    # --- Enterprise value proxy (equity + debt - cash) ---
    _ev_components = []
    if "equity_value" in cache.columns:
        _ev_components.append(cache["equity_value"])
    if "total_debt" in cache.columns:
        _ev_components.append(cache["total_debt"])
    elif "total_debt_asof" in cache.columns:
        _ev_components.append(cache["total_debt_asof"])
    if _ev_components:
        ev_proxy = _ev_components[0].copy()
        for c in _ev_components[1:]:
            ev_proxy = ev_proxy.add(c, fill_value=0)
        if "cash_and_equivalents" in cache.columns:
            ev_proxy = ev_proxy.sub(cache["cash_and_equivalents"], fill_value=0)
        cache["enterprise_value_proxy"] = ev_proxy
        logger.info("  enterprise_value_proxy: computed")

    # --- Implied P/E proxy (equity_value / net_income) ---
    if "equity_value" in cache.columns and "net_income" in cache.columns:
        ni = cache["net_income"]
        # Avoid division by zero or very small values
        safe_ni = ni.where(ni.abs() > 1e-6, other=np.nan)
        cache["implied_pe_proxy"] = cache["equity_value"] / safe_ni
        # Clip extreme values (P/E ratios above 200 or negative are not useful)
        cache["implied_pe_proxy"] = cache["implied_pe_proxy"].clip(-200, 200)
        logger.info("  implied_pe_proxy: computed")

    # --- Cash burn rate (when OCF is negative: -OCF / cash) ---
    if "operating_cash_flow" in cache.columns and "cash_and_equivalents" in cache.columns:
        ocf = cache["operating_cash_flow"]
        cash = cache["cash_and_equivalents"]
        safe_cash = cash.where(cash.abs() > 1e-6, other=np.nan)
        # Burn rate is positive when company is burning cash (negative OCF)
        cache["cash_burn_rate"] = (-ocf / safe_cash).clip(-5, 5).fillna(0.0)
        logger.info("  cash_burn_rate: computed")

    # --- Debt service coverage (OCF / interest_expense) ---
    if "operating_cash_flow" in cache.columns and "interest_expense" in cache.columns:
        ie = cache["interest_expense"]
        safe_ie = ie.where(ie.abs() > 1e-6, other=np.nan)
        cache["debt_service_coverage"] = (cache["operating_cash_flow"] / safe_ie).clip(-50, 50)
        logger.info("  debt_service_coverage: computed")

    # --- Revenue momentum 5d and 21d (rolling revenue change) ---
    if "revenue" in cache.columns and cache["revenue"].notna().sum() >= 2:
        rev = cache["revenue"]
        # Detect transitions and interpolate (same approach as equity_change_rate)
        rev_shifted = rev.shift(1)
        is_new_rev = rev.notna() & (rev != rev_shifted) & rev_shifted.notna()
        quarterly_rev = rev.where(is_new_rev | (rev.index == rev.first_valid_index()))
        rev_smooth = quarterly_rev.interpolate(method="time").ffill().bfill()
        rev_pct = rev_smooth.pct_change().fillna(0.0).clip(-0.1, 0.1)
        cache["revenue_momentum_5d"] = rev_pct.rolling(5, min_periods=1).mean()
        cache["revenue_momentum_21d"] = rev_pct.rolling(21, min_periods=1).mean()
        logger.info("  revenue_momentum_5d/21d: computed")

    # --- Earnings volatility (rolling std of net_income change) ---
    if "net_income" in cache.columns and cache["net_income"].notna().sum() >= 2:
        ni = cache["net_income"]
        ni_shifted = ni.shift(1)
        is_new_ni = ni.notna() & (ni != ni_shifted) & ni_shifted.notna()
        quarterly_ni = ni.where(is_new_ni | (ni.index == ni.first_valid_index()))
        ni_smooth = quarterly_ni.interpolate(method="time").ffill().bfill()
        ni_pct = ni_smooth.pct_change().fillna(0.0).clip(-0.5, 0.5)
        cache["earnings_volatility"] = ni_pct.rolling(
            window=63, min_periods=10
        ).std() * np.sqrt(252)
        cache["earnings_volatility"] = cache["earnings_volatility"].fillna(
            ni_pct.expanding(min_periods=5).std() * np.sqrt(252)
        )
        logger.info("  earnings_volatility: computed")

    # --- Balance sheet leverage change ---
    if "debt_to_equity_abs" in cache.columns:
        cache["balance_sheet_leverage_change"] = (
            cache["debt_to_equity_abs"].pct_change().fillna(0.0).clip(-1.0, 1.0)
        )
        logger.info("  balance_sheet_leverage_change: computed")

    # --- Flag this cache as private company mode ---
    cache["is_private_company"] = True

    # --- Store proxy confidence scores ---
    for proxy_col, conf in PROXY_CONFIDENCE.items():
        if proxy_col in cache.columns:
            cache[f"proxy_confidence_{proxy_col}"] = conf

    n_proxies = sum(1 for col in PROXY_MAP.values() if col in cache.columns)
    n_extended = sum(
        1 for col in list(EXTENDED_PROXY_MAP.values()) + [
            "enterprise_value_proxy", "implied_pe_proxy", "cash_burn_rate",
            "debt_service_coverage", "balance_sheet_leverage_change",
        ]
        if col in cache.columns
    )
    logger.info(
        "Private company proxies: %d core + %d extended proxy columns computed",
        n_proxies, n_extended,
    )

    return cache


def resolve_proxies(cache: pd.DataFrame) -> pd.DataFrame:
    """Write proxy values into standard OHLCV column names.

    After calling this function, ``cache["close"]`` contains
    ``equity_value``, ``cache["return_1d"]`` contains
    ``equity_change_rate``, etc.  Downstream models work unchanged
    because they find the standard column names populated.

    This function is idempotent -- calling it multiple times has no
    additional effect.  It only writes into columns that are either
    missing or entirely NaN.

    Parameters
    ----------
    cache:
        Daily cache DataFrame, after ``compute_private_company_proxies()``
        has been called.

    Returns
    -------
    The same cache DataFrame with standard columns populated from proxies.
    """
    if not is_private_company(cache):
        return cache

    resolved_count = 0

    # Core proxy map
    for standard, proxy in PROXY_MAP.items():
        if proxy not in cache.columns:
            continue
        if standard not in cache.columns:
            cache[standard] = cache[proxy]
            resolved_count += 1
        elif cache[standard].isna().all():
            cache[standard] = cache[proxy]
            resolved_count += 1

    # Extended proxy map
    for standard, proxy in EXTENDED_PROXY_MAP.items():
        if proxy not in cache.columns:
            continue
        if standard not in cache.columns:
            cache[standard] = cache[proxy]
            resolved_count += 1
        elif cache[standard].isna().all():
            cache[standard] = cache[proxy]
            resolved_count += 1

    if resolved_count > 0:
        logger.info(
            "Proxy resolution: %d standard columns populated from proxies",
            resolved_count,
        )

    return cache


def get_proxy_variable(standard_name: str, cache: pd.DataFrame) -> str:
    """Return the appropriate variable name for the current mode.

    If the cache is in private company mode (no OHLCV), returns the proxy
    variable name.  Otherwise returns the standard OHLCV variable name.

    Parameters
    ----------
    standard_name:
        Standard variable name (e.g. "close", "return_1d").
    cache:
        Daily cache DataFrame.

    Returns
    -------
    The variable name to use (either standard or proxy).
    """
    if is_private_company(cache) and standard_name in PROXY_MAP:
        proxy = PROXY_MAP[standard_name]
        if proxy in cache.columns:
            return proxy
    if is_private_company(cache) and standard_name in EXTENDED_PROXY_MAP:
        proxy = EXTENDED_PROXY_MAP[standard_name]
        if proxy in cache.columns:
            return proxy
    return standard_name


def get_proxy_confidence(column_name: str) -> float:
    """Return the confidence score for a proxy column.

    Returns 1.0 for non-proxy columns (real OHLCV data).

    Parameters
    ----------
    column_name:
        Column name (either a proxy name or a standard name).
    """
    return PROXY_CONFIDENCE.get(column_name, 1.0)


def get_forecast_targets(cache: pd.DataFrame) -> list[str]:
    """Return the list of forecast target variables for the current mode.

    In public company mode, returns the standard price + financial targets.
    In private company mode, returns only financial statement targets.
    """
    if is_private_company(cache):
        return [v for v in PRIVATE_FORECAST_TARGETS if v in cache.columns]
    # Public mode: let the forecasting module use its default target selection
    return []


def get_tier_variables(cache: pd.DataFrame) -> dict[str, list[str]]:
    """Return tier variable mapping for forward pass / burnout.

    In private company mode, returns financial statement-based tier variables.
    In public mode, returns empty dict (use default).
    """
    if not is_private_company(cache):
        return {}

    result = {}
    for tier, variables in PRIVATE_TIER_VARIABLES.items():
        available = [v for v in variables if v in cache.columns]
        if available:
            result[tier] = available
    return result

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

Top-level entry point:
    ``compute_private_company_proxies(cache)``

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
    "return_1d": "equity_change_rate",
    "volatility_21d": "financial_volatility",
    "drawdown_252d": "equity_drawdown",
    "volume": "revenue_velocity",
    "market_cap": "equity_value",
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

    # --- Flag this cache as private company mode ---
    cache["is_private_company"] = True

    n_proxies = sum(1 for col in PROXY_MAP.values() if col in cache.columns)
    logger.info(
        "Private company proxies: %d/%d proxy columns computed",
        n_proxies, len(PROXY_MAP),
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
    return standard_name


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

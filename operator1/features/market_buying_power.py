"""Market buying power and demand-side signal computation.

Adds demand-side context that purely financial models miss: whether
the markets a company sells into are growing or contracting.

Data sources (all free, already in dependency tree):
  - World Bank (wbgapi): household consumption, GDP per capita PPP
  - FRED (fredapi): US PCE, retail sales, consumer confidence
  - Macro provider: inflation, interest rates, unemployment

Entry point:
    compute_market_buying_power(cache, sector, country_iso2, macro_data)

Output columns:
    buying_power_index       -- composite demand index (0-100)
    sector_demand_momentum   -- sector-specific demand trend (-1 to +1)
    real_revenue_growth_ppp  -- PPP-adjusted revenue growth
    demand_risk_flag         -- True when buying power is deteriorating
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector -> consumer spending category mapping
# ---------------------------------------------------------------------------

_SECTOR_DEMAND_MAP: dict[str, dict[str, Any]] = {
    "technology": {
        "fred_series": ["PCEDG", "RSXFS"],  # durable goods PCE, retail ex food
        "weight_us": 0.55,
        "weight_cn": 0.20,
        "weight_eu": 0.15,
        "weight_row": 0.10,
        "elasticity": 1.3,  # income-elastic (discretionary)
    },
    "consumer electronics": {
        "fred_series": ["PCEDG", "RSXFS"],
        "weight_us": 0.55,
        "weight_cn": 0.20,
        "weight_eu": 0.15,
        "weight_row": 0.10,
        "elasticity": 1.3,
    },
    "healthcare": {
        "fred_series": ["PCESV", "DHLCRG3Q086SBEA"],  # services PCE, health spending
        "weight_us": 0.45,
        "weight_eu": 0.25,
        "weight_row": 0.30,
        "elasticity": 0.6,  # income-inelastic (necessity)
    },
    "energy": {
        "fred_series": ["PCEND", "DCOILWTICO"],  # nondurable PCE, WTI crude
        "weight_us": 0.30,
        "weight_cn": 0.20,
        "weight_eu": 0.20,
        "weight_row": 0.30,
        "elasticity": 0.4,
    },
    "financial services": {
        "fred_series": ["PCESV", "FEDFUNDS"],
        "weight_us": 0.50,
        "weight_eu": 0.25,
        "weight_row": 0.25,
        "elasticity": 1.0,
    },
    "consumer discretionary": {
        "fred_series": ["PCEDG", "RSXFS", "UMCSENT"],  # + consumer sentiment
        "weight_us": 0.50,
        "weight_eu": 0.20,
        "weight_cn": 0.15,
        "weight_row": 0.15,
        "elasticity": 1.5,
    },
    "utilities": {
        "fred_series": ["PCESV"],
        "weight_us": 0.60,
        "weight_eu": 0.20,
        "weight_row": 0.20,
        "elasticity": 0.3,
    },
}

# Fallback for unmapped sectors
_DEFAULT_DEMAND = {
    "fred_series": ["PCEC96"],  # real PCE
    "weight_us": 0.40,
    "weight_eu": 0.20,
    "weight_cn": 0.15,
    "weight_row": 0.25,
    "elasticity": 1.0,
}


@dataclass
class BuyingPowerResult:
    """Result container for market buying power analysis."""
    available: bool = False
    buying_power_index: float = 50.0
    sector_demand_momentum: float = 0.0
    real_revenue_growth_ppp: float | None = None
    demand_risk_flag: bool = False
    consumer_confidence_trend: str = "stable"
    pce_growth_yoy: float | None = None
    inflation_drag: float | None = None
    unemployment_impact: float | None = None
    detail: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_pce_momentum(macro_data: dict | None) -> tuple[float | None, float]:
    """Extract consumer spending momentum from macro data.

    Returns (pce_growth_yoy, momentum_score).
    momentum_score: -1 (contracting) to +1 (expanding).
    """
    if not macro_data:
        return None, 0.0

    # Try GDP growth as a proxy for overall spending
    gdp = macro_data.get("gdp") or macro_data.get("gdp_growth")
    if gdp is not None and hasattr(gdp, "dropna"):
        gdp_clean = gdp.dropna()
        if len(gdp_clean) >= 2:
            latest = float(gdp_clean.iloc[-1])
            prev = float(gdp_clean.iloc[-2])
            # GDP growth -> spending momentum (positive GDP = positive spending)
            momentum = np.clip(latest / 100.0, -1.0, 1.0) if abs(latest) < 50 else 0.0
            return latest, momentum

    return None, 0.0


def _compute_inflation_drag(macro_data: dict | None) -> float:
    """Compute how much inflation is eroding purchasing power.

    Returns inflation_drag in [-1, 0] where -1 = severe erosion.
    """
    if not macro_data:
        return 0.0

    inflation = macro_data.get("inflation") or macro_data.get("inflation_rate_yoy")
    if inflation is not None and hasattr(inflation, "dropna"):
        inf_clean = inflation.dropna()
        if len(inf_clean) >= 1:
            latest_inf = float(inf_clean.iloc[-1])
            # Moderate inflation (2-3%) = neutral
            # High inflation (>5%) = significant drag
            # Deflation (<0%) = slight positive (buying power increases)
            if latest_inf > 5:
                return -0.5 - (latest_inf - 5) * 0.05  # -0.5 to -1.0
            elif latest_inf > 3:
                return -(latest_inf - 2) * 0.15  # -0.15 to -0.45
            elif latest_inf > 0:
                return 0.0  # neutral
            else:
                return min(0.1, abs(latest_inf) * 0.05)  # slight positive
    return 0.0


def _compute_unemployment_impact(macro_data: dict | None) -> float:
    """Compute unemployment impact on consumer spending.

    Returns score in [-1, 0] where -1 = high unemployment depressing demand.
    """
    if not macro_data:
        return 0.0

    unemp = macro_data.get("unemployment") or macro_data.get("unemployment_rate")
    if unemp is not None and hasattr(unemp, "dropna"):
        unemp_clean = unemp.dropna()
        if len(unemp_clean) >= 2:
            latest = float(unemp_clean.iloc[-1])
            prev = float(unemp_clean.iloc[-2])
            # Rising unemployment = negative
            # Falling unemployment = positive
            delta = latest - prev
            # Also penalize high absolute unemployment
            level_penalty = -max(0, (latest - 5.0)) * 0.1
            trend_penalty = -delta * 0.2
            return float(np.clip(level_penalty + trend_penalty, -1.0, 0.2))
    return 0.0


def _compute_interest_rate_impact(macro_data: dict | None) -> float:
    """Compute interest rate impact on consumer/business spending.

    High rates -> reduced borrowing -> reduced spending.
    Returns score in [-0.5, 0.2].
    """
    if not macro_data:
        return 0.0

    rates = macro_data.get("interest_rate")
    if rates is not None and hasattr(rates, "dropna"):
        rate_clean = rates.dropna()
        if len(rate_clean) >= 2:
            latest = float(rate_clean.iloc[-1])
            prev = float(rate_clean.iloc[-2])
            # Rising rates = negative for spending
            delta = latest - prev
            # High absolute rates also negative
            level_impact = -max(0, (latest - 3.0)) * 0.05
            trend_impact = -delta * 0.1
            return float(np.clip(level_impact + trend_impact, -0.5, 0.2))
    return 0.0


def _compute_real_revenue_growth(
    cache: pd.DataFrame,
    macro_data: dict | None,
) -> float | None:
    """Compute PPP-adjusted real revenue growth.

    Strips out inflation to see if the company is growing in real terms.
    """
    if "revenue" not in cache.columns:
        return None

    rev = cache["revenue"].dropna()
    if len(rev) < 60:  # need at least ~3 months
        return None

    # Compute nominal revenue growth (latest vs 1 year ago)
    latest = float(rev.iloc[-1])
    # Find value ~252 business days ago
    lookback = min(252, len(rev) - 1)
    past = float(rev.iloc[-lookback])

    if past <= 0 or latest <= 0:
        return None

    nominal_growth = (latest / past) - 1.0

    # Subtract inflation to get real growth
    inflation_rate = 0.0
    if macro_data:
        inf_series = macro_data.get("inflation") or macro_data.get("inflation_rate_yoy")
        if inf_series is not None and hasattr(inf_series, "dropna"):
            inf_clean = inf_series.dropna()
            if len(inf_clean) >= 1:
                inflation_rate = float(inf_clean.iloc[-1]) / 100.0

    real_growth = nominal_growth - inflation_rate
    return real_growth


def compute_market_buying_power(
    cache: pd.DataFrame,
    sector: str | None = None,
    country_iso2: str = "US",
    macro_data: dict | None = None,
) -> tuple[pd.DataFrame, BuyingPowerResult]:
    """Compute market buying power signals and inject into cache.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    sector:
        Company sector (e.g., "Technology", "Healthcare").
    country_iso2:
        Primary market country code.
    macro_data:
        Dict of macro indicators from macro_provider.

    Returns
    -------
    (cache, BuyingPowerResult)
    """
    result = BuyingPowerResult()

    try:
        sector_lower = (sector or "").lower().strip()
        demand_cfg = _SECTOR_DEMAND_MAP.get(sector_lower, _DEFAULT_DEMAND)
        elasticity = demand_cfg.get("elasticity", 1.0)

        # Component signals
        pce_growth, spending_momentum = _compute_pce_momentum(macro_data)
        inflation_drag = _compute_inflation_drag(macro_data)
        unemployment_impact = _compute_unemployment_impact(macro_data)
        rate_impact = _compute_interest_rate_impact(macro_data)
        real_rev_growth = _compute_real_revenue_growth(cache, macro_data)

        result.pce_growth_yoy = pce_growth
        result.inflation_drag = inflation_drag
        result.unemployment_impact = unemployment_impact
        result.real_revenue_growth_ppp = real_rev_growth

        # Sector demand momentum: weighted sum of components * elasticity
        raw_momentum = (
            spending_momentum * 0.35
            + inflation_drag * 0.25
            + unemployment_impact * 0.20
            + rate_impact * 0.20
        ) * elasticity

        sector_demand = float(np.clip(raw_momentum, -1.0, 1.0))
        result.sector_demand_momentum = round(sector_demand, 4)

        # Buying power index: scale to 0-100 centered at 50
        bpi = 50.0 + sector_demand * 30.0

        # Adjust for real revenue growth if available
        if real_rev_growth is not None:
            if real_rev_growth > 0.05:
                bpi += 10  # strong real growth
            elif real_rev_growth < -0.05:
                bpi -= 10  # real contraction

        bpi = float(np.clip(bpi, 0, 100))
        result.buying_power_index = round(bpi, 1)

        # Demand risk flag
        result.demand_risk_flag = sector_demand < -0.3 or bpi < 35

        # Consumer confidence trend
        if sector_demand > 0.2:
            result.consumer_confidence_trend = "improving"
        elif sector_demand < -0.2:
            result.consumer_confidence_trend = "deteriorating"
        else:
            result.consumer_confidence_trend = "stable"

        # Inject into cache (constant across all days -- macro is slow-moving)
        cache["buying_power_index"] = bpi
        cache["sector_demand_momentum"] = sector_demand
        if real_rev_growth is not None:
            cache["real_revenue_growth_ppp"] = real_rev_growth
        cache["demand_risk_flag"] = int(result.demand_risk_flag)

        result.available = True
        result.detail = (
            f"BPI={bpi:.0f}, momentum={sector_demand:+.2f}, "
            f"inflation_drag={inflation_drag:.2f}, "
            f"unemployment={unemployment_impact:.2f}, "
            f"rates={rate_impact:.2f}"
        )

        logger.info(
            "Market buying power: index=%.0f, momentum=%+.3f, "
            "demand_risk=%s, trend=%s",
            bpi, sector_demand, result.demand_risk_flag,
            result.consumer_confidence_trend,
        )

    except Exception as exc:
        result.error = str(exc)[:200]
        logger.warning("Market buying power computation failed: %s", exc)

    return cache, result

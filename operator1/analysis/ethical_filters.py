"""Ethical filter computation for Operator 1.

Implements the four ethical/analysis filters described in the spec
(The_Apps_core_idea.pdf Section G):

1. **Purchasing Power** -- inflation-adjusted real return vs nominal.
2. **Solvency** -- debt-to-equity fragility assessment.
3. **Gharar** -- volatility-based speculation vs calculated risk.
4. **Cash is King** -- free cash flow yield quality.

Each filter produces a verdict string and supporting metrics that are
injected into the company profile and displayed in the final report.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Purchasing Power filter
# ---------------------------------------------------------------------------

def compute_purchasing_power_filter(
    cache: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate whether the investment beat inflation over the cache period.

    Spec reference: Section G filter 1.

    Returns
    -------
    dict with keys: verdict, nominal_return, real_return, inflation_impact.
    """
    # Use close price if available; fall back to equity_value proxy
    # (private company mode resolves equity_value into close, but
    # this fallback handles cases where resolution hasn't been called).
    close = cache.get("close")
    if (close is None or close.dropna().empty or len(close.dropna()) < 2):
        close = cache.get("equity_value")
    if close is None or close.dropna().empty or len(close.dropna()) < 2:
        return {
            "available": False,
            "verdict": "UNAVAILABLE",
            "nominal_return": None,
            "real_return": None,
            "inflation_impact": None,
        }

    first_valid = close.dropna().iloc[0]
    last_valid = close.dropna().iloc[-1]
    nominal_return = (last_valid / first_valid) - 1.0 if first_valid != 0 else None

    # Real return (cumulative) from daily real returns if available
    real_return_1d = cache.get("real_return_1d")
    if real_return_1d is not None and real_return_1d.notna().sum() > 0:
        real_return = float(real_return_1d.dropna().sum())
    else:
        # Fallback: use nominal (no inflation data)
        real_return = nominal_return

    if nominal_return is None:
        verdict = "UNAVAILABLE"
    elif nominal_return > 0 and real_return is not None and real_return > 0:
        verdict = "PASS"
    elif nominal_return > 0 and real_return is not None and real_return <= 0:
        verdict = "FAIL - Nominal gain, real loss"
    else:
        verdict = "FAIL - Negative returns"

    inflation_impact = (
        (nominal_return - real_return)
        if nominal_return is not None and real_return is not None
        else None
    )

    return {
        "available": True,
        "verdict": verdict,
        "nominal_return": _safe(nominal_return),
        "real_return": _safe(real_return),
        "inflation_impact": _safe(inflation_impact),
    }


# ---------------------------------------------------------------------------
# 2. Solvency filter (Riba / debt-to-equity)
# ---------------------------------------------------------------------------

def compute_solvency_filter(
    cache: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate debt-to-equity fragility.

    Spec reference: Section G filter 2.

    Thresholds:
      < 1.0  -> PASS - Conservative
      < 2.0  -> PASS - Stable
      < 3.0  -> WARNING - Elevated
      >= 3.0 -> FAIL - Fragile

    Returns
    -------
    dict with keys: verdict, debt_to_equity, threshold, interpretation.
    """
    # Prefer the absolute version for threshold comparison
    d2e_col = "debt_to_equity_abs" if "debt_to_equity_abs" in cache.columns else "debt_to_equity"
    d2e_series = cache.get(d2e_col)

    if d2e_series is None or d2e_series.dropna().empty:
        return {
            "available": False,
            "verdict": "UNAVAILABLE",
            "debt_to_equity": None,
            "threshold": 3.0,
            "interpretation": "Debt-to-equity data not available.",
        }

    d2e = float(d2e_series.dropna().iloc[-1])

    if d2e < 1.0:
        verdict = "PASS - Conservative"
        interpretation = (
            "Low leverage. The company is conservatively financed and "
            "resilient to interest rate shocks."
        )
    elif d2e < 2.0:
        verdict = "PASS - Stable"
        interpretation = (
            "Moderate leverage within normal range. The company can "
            "service its debt under typical conditions."
        )
    elif d2e < 3.0:
        verdict = "WARNING - Elevated"
        interpretation = (
            "Leverage is elevated. A prolonged downturn or rate hike "
            "could strain the balance sheet."
        )
    else:
        verdict = "FAIL - Fragile"
        interpretation = (
            "Dangerously high leverage (D/E >= 3.0). The company is "
            "vulnerable to bankruptcy in a recession or rate spike."
        )

    return {
        "available": True,
        "verdict": verdict,
        "debt_to_equity": _safe(d2e),
        "threshold": 3.0,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# 3. Gharar filter (volatility / speculation)
# ---------------------------------------------------------------------------

def compute_gharar_filter(
    cache: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate volatility-based speculation risk.

    Spec reference: Section G filter 3.

    Thresholds (21-day annualised volatility):
      < 0.15  -> LOW - Stable (stability score 9)
      < 0.25  -> MODERATE - Calculated Risk (score 7)
      < 0.40  -> HIGH - Speculation (score 4)
      >= 0.40 -> EXTREME - Gambling (score 2)

    Returns
    -------
    dict with keys: verdict, volatility_21d, stability_score, interpretation.
    """
    # Use volatility_21d if available; fall back to financial_volatility
    # proxy (private company mode resolves financial_volatility into
    # volatility_21d, but this fallback handles unresolved cases).
    vol_series = cache.get("volatility_21d")
    if vol_series is None or vol_series.dropna().empty:
        vol_series = cache.get("financial_volatility")
    if vol_series is None or vol_series.dropna().empty:
        return {
            "available": False,
            "verdict": "UNAVAILABLE",
            "volatility_21d": None,
            "stability_score": None,
            "interpretation": "Volatility data not available.",
        }

    vol = float(vol_series.dropna().iloc[-1])

    if vol < 0.15:
        verdict = "LOW - Stable"
        stability_score = 9
        interpretation = (
            "Low volatility indicates a stable price pattern. This "
            "is a calculated investment, not speculation."
        )
    elif vol < 0.25:
        verdict = "MODERATE - Calculated Risk"
        stability_score = 7
        interpretation = (
            "Moderate volatility. Price swings are present but within "
            "a range typical of the asset class."
        )
    elif vol < 0.40:
        verdict = "HIGH - Speculation"
        stability_score = 4
        interpretation = (
            "High volatility. Large price swings make predictions "
            "unreliable. This resembles speculation more than investment."
        )
    else:
        verdict = "EXTREME - Gambling"
        stability_score = 2
        interpretation = (
            "Extreme volatility. The stock behaves like a lottery "
            "ticket. Any prediction is as likely to be wrong as right."
        )

    return {
        "available": True,
        "verdict": verdict,
        "volatility_21d": _safe(vol),
        "stability_score": stability_score,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# 4. Cash is King filter (FCF yield)
# ---------------------------------------------------------------------------

def compute_cash_is_king_filter(
    cache: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate free cash flow yield quality.

    Spec reference: Section G filter 4.

    Thresholds (FCF yield = FCF TTM / market cap):
      > 0.05 -> PASS - Strong
      > 0.02 -> PASS - Healthy
      > 0.00 -> WARNING - Weak
      <= 0   -> FAIL - Burning Cash

    Returns
    -------
    dict with keys: verdict, fcf_yield, fcf_margin, interpretation.
    """
    fcf_yield_series = cache.get("fcf_yield")
    if fcf_yield_series is None or fcf_yield_series.dropna().empty:
        return {
            "available": False,
            "verdict": "UNAVAILABLE",
            "fcf_yield": None,
            "fcf_margin": None,
            "interpretation": "Free cash flow yield data not available.",
        }

    fcf_yield = float(fcf_yield_series.dropna().iloc[-1])

    # FCF margin (optional enrichment)
    fcf_margin = None
    fcf_margin_series = cache.get("fcf_margin")
    if fcf_margin_series is not None and fcf_margin_series.notna().any():
        fcf_margin = float(fcf_margin_series.dropna().iloc[-1])

    if fcf_yield > 0.05:
        verdict = "PASS - Strong"
        interpretation = (
            "Strong free cash flow generation. The company produces "
            "real cash relative to its valuation -- a sign of a "
            "genuine, productive business."
        )
    elif fcf_yield > 0.02:
        verdict = "PASS - Healthy"
        interpretation = (
            "Healthy cash flow generation. Profit translates into "
            "actual cash, supporting dividends and reinvestment."
        )
    elif fcf_yield > 0.0:
        verdict = "WARNING - Weak"
        interpretation = (
            "Cash generation is positive but thin. Accounting profits "
            "may overstate the real economic value being created."
        )
    else:
        verdict = "FAIL - Burning Cash"
        interpretation = (
            "Negative free cash flow yield. The company is consuming "
            "more cash than it generates. 'Profit is an opinion, "
            "but cash is a fact.'"
        )

    return {
        "available": True,
        "verdict": verdict,
        "fcf_yield": _safe(fcf_yield),
        "fcf_margin": _safe(fcf_margin),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Combined filter runner
# ---------------------------------------------------------------------------

def compute_all_ethical_filters(
    cache: pd.DataFrame,
) -> dict[str, Any]:
    """Run all four ethical filters and return a combined result dict.

    Returns
    -------
    dict with keys: purchasing_power, solvency, gharar, cash_is_king.
    """
    logger.info("Computing ethical filters...")

    result = {
        "purchasing_power": compute_purchasing_power_filter(cache),
        "solvency": compute_solvency_filter(cache),
        "gharar": compute_gharar_filter(cache),
        "cash_is_king": compute_cash_is_king_filter(cache),
    }

    # Log verdicts
    for name, filt in result.items():
        logger.info("  %s: %s", name, filt.get("verdict", "N/A"))

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val: float | None) -> float | None:
    """Return None for NaN/Inf, else the float value."""
    if val is None:
        return None
    try:
        if np.isnan(val) or np.isinf(val):
            return None
    except (TypeError, ValueError):
        pass
    return float(val)

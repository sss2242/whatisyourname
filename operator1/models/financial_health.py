"""Financial Health Scoring -- daily composite scores injected into cache.

Computes per-day financial health scores across the five survival tiers
so that downstream temporal models (forecasting, forward pass, burn-out)
automatically learn from them.  Each score is a 0-100 normalized value
written as a daily column in the cache DataFrame.

**Scores produced (all daily columns):**

- ``fh_liquidity_score``     -- Tier 1: cash, FCF, operating CF
- ``fh_solvency_score``      -- Tier 2: debt ratios, interest coverage
- ``fh_stability_score``     -- Tier 3: volatility, drawdown
- ``fh_profitability_score`` -- Tier 4: margins
- ``fh_growth_score``        -- Tier 5: revenue trend, valuation
- ``fh_composite_score``     -- Weighted blend of tiers 1-5
- ``fh_composite_label``     -- Categorical: Critical / Weak / Fair / Strong / Excellent
- ``fh_altman_z_score``      -- Altman Z-Score (bankruptcy predictor)
- ``fh_altman_z_zone``       -- safe / grey / distress
- ``fh_beneish_m_score``     -- Beneish M-Score (manipulation detector)
- ``fh_beneish_flag``        -- 1 if likely manipulator, 0 otherwise
- ``fh_runway_months``       -- Months of cash runway at current burn

The composite uses hierarchy weights when provided, so a company in
survival mode will have its composite dominated by liquidity/solvency,
exactly matching how the temporal models should prioritize learning.

The Altman Z-Score and Liquidity Runway strengthen Tier 1-2 survival
signals.  The Beneish M-Score serves the ethical filter mission
(Sec 13): if earnings are manipulated, downstream analysis is unreliable.

Top-level entry point:
    ``compute_financial_health(cache, hierarchy_weights=None)``

Spec refs: Sec 17 (temporal learning), Sec C.3-C.4 (hierarchy weights)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-10  # avoid division by zero

# Default equal weights across 5 tiers (matches normal regime)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "tier1": 0.20,
    "tier2": 0.20,
    "tier3": 0.20,
    "tier4": 0.20,
    "tier5": 0.20,
}

# Composite label thresholds
_LABEL_THRESHOLDS: list[tuple[float, str]] = [
    (20.0, "Critical"),
    (40.0, "Weak"),
    (60.0, "Fair"),
    (80.0, "Strong"),
    (100.1, "Excellent"),
]

# Rolling window for trend-based scoring (business days)
_TREND_WINDOW: int = 63  # ~3 months

# Altman Z-Score coefficients (Altman 1968)
_Z_COEFF = {
    "x1_working_capital_ta": 1.2,
    "x2_retained_earnings_ta": 1.4,
    "x3_ebit_ta": 3.3,
    "x4_market_cap_tl": 0.6,
    "x5_revenue_ta": 1.0,
}
_Z_SAFE_THRESHOLD = 2.99
_Z_DISTRESS_THRESHOLD = 1.81

# Beneish M-Score coefficients (Beneish 1999)
_M_INTERCEPT = -4.84
_M_COEFFS = {
    "DSRI": 0.920,
    "GMI": 0.528,
    "AQI": 0.404,
    "SGI": 0.892,
    "DEPI": 0.115,
    "SGAI": -0.172,
    "TATA": 4.679,
    "LVGI": -0.327,
}
_M_THRESHOLD = -1.78


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class AltmanZResult:
    """Altman Z-Score computation result."""
    z_score_series: pd.Series | None = None
    latest_z_score: float | None = None
    zone: str = "unknown"
    components: dict[str, float] = field(default_factory=dict)
    available: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available, "error": self.error,
            "latest_z_score": self.latest_z_score, "zone": self.zone,
            "components": self.components,
        }


@dataclass
class BeneishMResult:
    """Beneish M-Score computation result."""
    m_score: float | None = None
    likely_manipulator: bool = False
    verdict: str = "unknown"
    components: dict[str, float] = field(default_factory=dict)
    available: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available, "error": self.error,
            "m_score": self.m_score, "likely_manipulator": self.likely_manipulator,
            "verdict": self.verdict, "components": self.components,
        }


@dataclass
class LiquidityRunwayResult:
    """Liquidity runway estimation result."""
    months_of_runway: float | None = None
    verdict: str = "unknown"
    cash_available: float | None = None
    monthly_burn_rate: float | None = None
    available: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available, "error": self.error,
            "months_of_runway": self.months_of_runway, "verdict": self.verdict,
            "cash_available": self.cash_available,
            "monthly_burn_rate": self.monthly_burn_rate,
        }


@dataclass
class FinancialHealthResult:
    """Summary statistics from the financial health computation."""

    columns_added: list[str] = field(default_factory=list)
    latest_composite: float = float("nan")
    latest_label: str = "Unknown"
    mean_composite: float = float("nan")
    tier_means: dict[str, float] = field(default_factory=dict)
    n_days_scored: int = 0
    # Extended models from PR integration
    altman_z: AltmanZResult = field(default_factory=AltmanZResult)
    beneish_m: BeneishMResult = field(default_factory=BeneishMResult)
    liquidity_runway: LiquidityRunwayResult = field(default_factory=LiquidityRunwayResult)


# ---------------------------------------------------------------------------
# Individual tier scoring helpers
# ---------------------------------------------------------------------------


def _normalize_series(
    s: pd.Series,
    *,
    lower: float | None = None,
    upper: float | None = None,
    invert: bool = False,
) -> pd.Series:
    """Normalize a series to 0-100 using rolling percentile rank.

    Parameters
    ----------
    s : pd.Series
        Raw metric series.
    lower, upper : float, optional
        If provided, clip the series before normalizing. Useful for
        known-range metrics like ratios.
    invert : bool
        If True, higher raw values map to *lower* scores (e.g. debt ratios,
        volatility).

    Returns
    -------
    pd.Series
        Normalized 0-100 score (NaN where input is NaN).
    """
    if s.isna().all():
        return pd.Series(np.nan, index=s.index)

    work = s.copy()
    if lower is not None or upper is not None:
        work = work.clip(lower=lower, upper=upper)

    # Use expanding percentile rank for a stable, non-leaking normalization
    ranked = work.expanding(min_periods=1).rank(pct=True) * 100.0

    if invert:
        ranked = 100.0 - ranked

    return ranked


def _score_liquidity(cache: pd.DataFrame) -> pd.Series:
    """Tier 1 -- Liquidity & Cash score.

    Looks at: cash_ratio, free_cash_flow_ttm (or operating_cash_flow),
    cash_and_equivalents.
    """
    components: list[pd.Series] = []

    if "cash_ratio" in cache.columns:
        components.append(
            _normalize_series(cache["cash_ratio"], lower=0, upper=10)
        )

    if "free_cash_flow_ttm_asof" in cache.columns:
        components.append(
            _normalize_series(cache["free_cash_flow_ttm_asof"])
        )
    elif "operating_cash_flow" in cache.columns:
        components.append(
            _normalize_series(cache["operating_cash_flow"])
        )

    if "cash_and_equivalents" in cache.columns:
        components.append(
            _normalize_series(cache["cash_and_equivalents"])
        )

    if not components:
        return pd.Series(np.nan, index=cache.index, name="fh_liquidity_score")

    score = pd.concat(components, axis=1).mean(axis=1)
    score.name = "fh_liquidity_score"
    return score


def _score_solvency(cache: pd.DataFrame) -> pd.Series:
    """Tier 2 -- Debt & Solvency score.

    Looks at: debt_to_equity (inverted), net_debt_to_ebitda (inverted),
    interest_coverage, current_ratio.
    """
    components: list[pd.Series] = []

    # derived_variables.py creates debt_to_equity_abs (not debt_to_equity)
    _de_col = next(
        (c for c in ("debt_to_equity_abs", "debt_to_equity_signed", "debt_to_equity")
         if c in cache.columns),
        None,
    )
    if _de_col is not None:
        components.append(
            _normalize_series(cache[_de_col], invert=True)
        )

    if "net_debt_to_ebitda" in cache.columns:
        components.append(
            _normalize_series(cache["net_debt_to_ebitda"], invert=True)
        )

    if "interest_coverage" in cache.columns:
        components.append(
            _normalize_series(cache["interest_coverage"], lower=0, upper=50)
        )

    if "current_ratio" in cache.columns:
        components.append(
            _normalize_series(cache["current_ratio"], lower=0, upper=10)
        )

    if not components:
        return pd.Series(np.nan, index=cache.index, name="fh_solvency_score")

    score = pd.concat(components, axis=1).mean(axis=1)
    score.name = "fh_solvency_score"
    return score


def _score_stability(cache: pd.DataFrame) -> pd.Series:
    """Tier 3 -- Market Stability score.

    Looks at: volatility_21d (inverted), drawdown_252d (inverted), volume.
    """
    components: list[pd.Series] = []

    if "volatility_21d" in cache.columns:
        components.append(
            _normalize_series(cache["volatility_21d"], invert=True)
        )

    if "drawdown_252d" in cache.columns:
        # Drawdown is typically negative; more negative = worse
        components.append(
            _normalize_series(cache["drawdown_252d"])
        )

    if "volume" in cache.columns:
        components.append(
            _normalize_series(cache["volume"])
        )

    if not components:
        return pd.Series(np.nan, index=cache.index, name="fh_stability_score")

    score = pd.concat(components, axis=1).mean(axis=1)
    score.name = "fh_stability_score"
    return score


def _score_profitability(cache: pd.DataFrame) -> pd.Series:
    """Tier 4 -- Profitability score.

    Looks at: gross_margin, operating_margin, net_margin.
    """
    components: list[pd.Series] = []

    for col in ("gross_margin", "operating_margin", "net_margin"):
        if col in cache.columns:
            components.append(
                _normalize_series(cache[col])
            )

    if not components:
        return pd.Series(np.nan, index=cache.index, name="fh_profitability_score")

    score = pd.concat(components, axis=1).mean(axis=1)
    score.name = "fh_profitability_score"
    return score


def _score_growth(cache: pd.DataFrame) -> pd.Series:
    """Tier 5 -- Growth & Valuation score.

    Looks at: revenue trend (rolling % change), pe_ratio (inverted --
    lower PE = cheaper = higher score), ev_to_ebitda (inverted).
    """
    components: list[pd.Series] = []

    # Revenue growth trend
    if "revenue" in cache.columns:
        rev = cache["revenue"]
        rev_growth = rev.pct_change(periods=_TREND_WINDOW)
        components.append(_normalize_series(rev_growth))

    if "pe_ratio_calc" in cache.columns:
        # Low PE -> potentially undervalued -> higher score
        pe = cache["pe_ratio_calc"].clip(lower=0, upper=200)
        components.append(_normalize_series(pe, invert=True))

    if "ev_to_ebitda" in cache.columns:
        ev = cache["ev_to_ebitda"].clip(lower=0, upper=100)
        components.append(_normalize_series(ev, invert=True))

    if not components:
        return pd.Series(np.nan, index=cache.index, name="fh_growth_score")

    score = pd.concat(components, axis=1).mean(axis=1)
    score.name = "fh_growth_score"
    return score


def _composite_label(score: float) -> str:
    """Map a composite score to a categorical label."""
    if np.isnan(score):
        return "Unknown"
    for threshold, label in _LABEL_THRESHOLDS:
        if score < threshold:
            return label
    return "Excellent"


# ---------------------------------------------------------------------------
# Extended models: Altman Z-Score, Beneish M-Score, Liquidity Runway
# ---------------------------------------------------------------------------


def compute_altman_z_score(df: pd.DataFrame) -> AltmanZResult:
    """Compute daily Altman Z-Score series (bankruptcy predictor).

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    where X1..X5 are balance-sheet ratios.  Zones: safe (>2.99),
    grey (1.81-2.99), distress (<1.81).
    """
    result = AltmanZResult()

    total_assets = df.get("total_assets")
    if total_assets is None or total_assets.notna().sum() == 0:
        result.error = "total_assets not available"
        return result

    ta = total_assets.replace(0, np.nan)
    tl = df.get("total_liabilities", pd.Series(np.nan, index=df.index)).replace(0, np.nan)

    current_assets = df.get("current_assets", pd.Series(np.nan, index=df.index))
    current_liabilities = df.get("current_liabilities", pd.Series(np.nan, index=df.index))
    x1 = (current_assets - current_liabilities) / ta

    retained_earnings = df.get("retained_earnings", pd.Series(np.nan, index=df.index))
    x2 = retained_earnings / ta

    ebit = df.get("ebit", df.get("operating_income", df.get("ebitda", pd.Series(np.nan, index=df.index))))
    x3 = ebit / ta

    market_cap = df.get("market_cap", pd.Series(np.nan, index=df.index))
    x4 = market_cap / tl

    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    x5 = revenue / ta

    z = (
        _Z_COEFF["x1_working_capital_ta"] * x1
        + _Z_COEFF["x2_retained_earnings_ta"] * x2
        + _Z_COEFF["x3_ebit_ta"] * x3
        + _Z_COEFF["x4_market_cap_tl"] * x4
        + _Z_COEFF["x5_revenue_ta"] * x5
    )

    result.z_score_series = z
    result.available = z.notna().any()

    if result.available:
        latest = z.dropna().iloc[-1] if z.notna().any() else None
        result.latest_z_score = float(latest) if latest is not None else None

        if result.latest_z_score is not None:
            if result.latest_z_score >= _Z_SAFE_THRESHOLD:
                result.zone = "safe"
            elif result.latest_z_score <= _Z_DISTRESS_THRESHOLD:
                result.zone = "distress"
            else:
                result.zone = "grey"

        for label, series in [
            ("x1_working_capital_ta", x1), ("x2_retained_earnings_ta", x2),
            ("x3_ebit_ta", x3), ("x4_market_cap_tl", x4), ("x5_revenue_ta", x5),
        ]:
            val = series.dropna().iloc[-1] if series.notna().any() else None
            result.components[label] = float(val) if val is not None else None

    logger.info("Altman Z-Score: %.2f (%s)", result.latest_z_score or 0.0, result.zone)
    return result


def compute_beneish_m_score(df: pd.DataFrame) -> BeneishMResult:
    """Compute Beneish M-Score for earnings manipulation detection.

    Uses period-over-period changes in financial ratios.  Score > -1.78
    suggests earnings manipulation.  Serves the ethical filter mission.
    """
    result = BeneishMResult()

    revenue = df.get("revenue")
    if revenue is None or revenue.notna().sum() < 2:
        result.error = "Insufficient revenue data for M-Score"
        return result

    rev_clean = revenue.dropna()
    rev_changes = rev_clean.diff().abs()
    period_breaks = rev_changes[rev_changes > _EPS].index

    if len(period_breaks) < 1:
        result.error = "No distinct financial periods detected"
        return result

    break_point = period_breaks[-1]
    prior_mask = df.index < break_point
    current_mask = df.index >= break_point

    if prior_mask.sum() == 0 or current_mask.sum() == 0:
        result.error = "Cannot split data into prior/current periods"
        return result

    def _pv(col: str, mask: pd.Series) -> float | None:
        c = df.get(col)
        if c is None:
            return None
        vals = c.loc[mask].dropna()
        return float(vals.mean()) if len(vals) > 0 else None

    def _sr(num: float | None, denom: float | None) -> float | None:
        if num is None or denom is None or abs(denom) < _EPS:
            return None
        return num / denom

    components: dict[str, float | None] = {}

    recv_c, recv_p = _pv("receivables", current_mask), _pv("receivables", prior_mask)
    rev_c, rev_p = _pv("revenue", current_mask), _pv("revenue", prior_mask)
    components["DSRI"] = _sr(_sr(recv_c, rev_c), _sr(recv_p, rev_p)) if _sr(recv_c, rev_c) and _sr(recv_p, rev_p) else None

    gp_c, gp_p = _pv("gross_profit", current_mask), _pv("gross_profit", prior_mask)
    gm_c, gm_p = _sr(gp_c, rev_c), _sr(gp_p, rev_p)
    components["GMI"] = _sr(gm_p, gm_c) if gm_c and gm_p else None

    ca_c, ca_p = _pv("current_assets", current_mask), _pv("current_assets", prior_mask)
    ta_c, ta_p = _pv("total_assets", current_mask), _pv("total_assets", prior_mask)
    aqi_c = 1.0 - (_sr(ca_c, ta_c) or 0.0) if ta_c else None
    aqi_p = 1.0 - (_sr(ca_p, ta_p) or 0.0) if ta_p else None
    components["AQI"] = _sr(aqi_c, aqi_p) if aqi_c is not None and aqi_p is not None else None

    components["SGI"] = _sr(rev_c, rev_p)

    ebit_c, ebit_p = _pv("ebit", current_mask), _pv("ebit", prior_mask)
    ebitda_c, ebitda_p = _pv("ebitda", current_mask), _pv("ebitda", prior_mask)
    dep_c = (ebitda_c - ebit_c) if ebitda_c is not None and ebit_c is not None else None
    dep_p = (ebitda_p - ebit_p) if ebitda_p is not None and ebit_p is not None else None
    dep_rate_c = _sr(dep_c, (dep_c + (ta_c or 0))) if dep_c is not None else None
    dep_rate_p = _sr(dep_p, (dep_p + (ta_p or 0))) if dep_p is not None else None
    components["DEPI"] = _sr(dep_rate_p, dep_rate_c) if dep_rate_c and dep_rate_p else None

    sga_c = (rev_c - ebit_c) if rev_c is not None and ebit_c is not None else None
    sga_p = (rev_p - ebit_p) if rev_p is not None and ebit_p is not None else None
    components["SGAI"] = _sr(_sr(sga_c, rev_c), _sr(sga_p, rev_p)) if _sr(sga_c, rev_c) and _sr(sga_p, rev_p) else None

    ni_c = _pv("net_income", current_mask)
    ocf_c = _pv("operating_cash_flow", current_mask)
    if ni_c is not None and ocf_c is not None and ta_c is not None and abs(ta_c) > _EPS:
        components["TATA"] = (ni_c - ocf_c) / ta_c
    else:
        components["TATA"] = None

    tl_c, tl_p = _pv("total_liabilities", current_mask), _pv("total_liabilities", prior_mask)
    components["LVGI"] = _sr(_sr(tl_c, ta_c), _sr(tl_p, ta_p)) if _sr(tl_c, ta_c) and _sr(tl_p, ta_p) else None

    m_score = _M_INTERCEPT
    n_available = 0
    for key, coeff in _M_COEFFS.items():
        val = components.get(key)
        if val is not None and np.isfinite(val):
            m_score += coeff * val
            n_available += 1

    if n_available < 4:
        result.error = f"Only {n_available}/8 M-Score components available"
        return result

    result.m_score = float(m_score)
    result.available = True
    result.components = {k: float(v) if v is not None else None for k, v in components.items()}
    result.likely_manipulator = m_score > _M_THRESHOLD

    if m_score > -1.78:
        result.verdict = "likely"
    elif m_score > -2.22:
        result.verdict = "possible"
    else:
        result.verdict = "unlikely"

    logger.info("Beneish M-Score: %.2f (%s manipulation)", result.m_score, result.verdict)
    return result


def compute_liquidity_runway(df: pd.DataFrame) -> LiquidityRunwayResult:
    """Estimate months of cash runway at current burn rate.

    Answers: "If revenue stopped today, how many months can this company
    survive on its current cash reserves at the current spending rate?"
    """
    result = LiquidityRunwayResult()

    cash = df.get("cash_and_equivalents")
    if cash is None or cash.notna().sum() == 0:
        result.error = "cash_and_equivalents not available"
        return result

    latest_cash = cash.dropna().iloc[-1]
    result.cash_available = float(latest_cash)

    ocf = df.get("operating_cash_flow")
    if ocf is not None and ocf.notna().sum() > 0:
        latest_ocf = float(ocf.dropna().iloc[-1])

        if latest_ocf < 0:
            monthly_burn = abs(latest_ocf) / 12.0
            result.monthly_burn_rate = float(monthly_burn)
        else:
            capex = df.get("capex")
            if capex is not None and capex.notna().sum() > 0:
                latest_capex = float(capex.dropna().iloc[-1])
                net_outflow = abs(latest_capex) - latest_ocf
                if net_outflow > 0:
                    result.monthly_burn_rate = float(net_outflow / 12.0)
                else:
                    result.monthly_burn_rate = 0.0
            else:
                result.monthly_burn_rate = 0.0
    else:
        result.error = "operating_cash_flow not available for burn rate"
        return result

    result.available = True

    if result.monthly_burn_rate is not None and result.monthly_burn_rate > _EPS:
        result.months_of_runway = float(latest_cash / result.monthly_burn_rate)
    elif result.monthly_burn_rate == 0.0:
        result.months_of_runway = float("inf")
    else:
        result.months_of_runway = None

    runway = result.months_of_runway
    if runway is None:
        result.verdict = "unknown"
    elif runway == float("inf") or runway > 24:
        result.verdict = "strong"
    elif runway > 12:
        result.verdict = "adequate"
    elif runway > 6:
        result.verdict = "tight"
    else:
        result.verdict = "critical"

    logger.info(
        "Liquidity runway: %.1f months (%s)",
        result.months_of_runway if result.months_of_runway is not None and result.months_of_runway != float("inf") else -1,
        result.verdict,
    )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_financial_health(
    cache: pd.DataFrame,
    hierarchy_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, FinancialHealthResult]:
    """Compute daily financial health scores and inject into the cache.

    This function MUST be called before temporal models (Step 6) so that
    forecasting, forward pass, and burn-out automatically learn from the
    health scores as additional daily features.

    Runs the 5-tier scoring system (liquidity, solvency, stability,
    profitability, growth) plus three extended models:
    - Altman Z-Score (bankruptcy prediction, strengthens Tier 1-2)
    - Beneish M-Score (earnings manipulation, ethical filter)
    - Liquidity Runway (months of cash, strengthens Tier 1)

    Parameters
    ----------
    cache : pd.DataFrame
        The daily cache DataFrame (DatetimeIndex) with derived features
        from Steps 4-5.
    hierarchy_weights : dict, optional
        Tier weights for the composite score.  Keys: ``tier1`` .. ``tier5``.
        If None, uses equal weights (20% each).  When the company is in
        survival mode, these weights shift toward liquidity/solvency,
        making the composite reflect survival priorities.

    Returns
    -------
    (cache, result)
        The cache with new ``fh_*`` columns appended, and a
        ``FinancialHealthResult`` summary.
    """
    logger.info("Computing financial health scores...")

    weights = dict(_DEFAULT_WEIGHTS)
    if hierarchy_weights:
        for k, v in hierarchy_weights.items():
            if k in weights:
                weights[k] = float(v)
        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

    result = FinancialHealthResult()

    # Compute tier scores
    tier_scores: dict[str, pd.Series] = {
        "tier1": _score_liquidity(cache),
        "tier2": _score_solvency(cache),
        "tier3": _score_stability(cache),
        "tier4": _score_profitability(cache),
        "tier5": _score_growth(cache),
    }

    col_names = {
        "tier1": "fh_liquidity_score",
        "tier2": "fh_solvency_score",
        "tier3": "fh_stability_score",
        "tier4": "fh_profitability_score",
        "tier5": "fh_growth_score",
    }

    # Inject tier scores into cache
    for tier_key, score_series in tier_scores.items():
        col = col_names[tier_key]
        cache[col] = score_series
        result.columns_added.append(col)
        mean_val = float(score_series.mean()) if not score_series.isna().all() else float("nan")
        result.tier_means[col] = mean_val

    # Compute weighted composite
    composite_parts: list[pd.Series] = []
    for tier_key in ("tier1", "tier2", "tier3", "tier4", "tier5"):
        w = weights[tier_key]
        s = tier_scores[tier_key]
        if not s.isna().all() and w > 0:
            composite_parts.append(s * w)

    if composite_parts:
        # Sum weighted components; re-normalize by actual weight coverage
        composite_df = pd.concat(composite_parts, axis=1)
        # For each row, compute weighted sum / sum of weights of non-NaN tiers
        weight_vals = []
        for tier_key in ("tier1", "tier2", "tier3", "tier4", "tier5"):
            s = tier_scores[tier_key]
            if not s.isna().all() and weights[tier_key] > 0:
                weight_vals.append(weights[tier_key])

        raw_sum = composite_df.sum(axis=1)
        # Track which tiers have data per row for proper normalization
        coverage_mask = pd.concat(
            [tier_scores[tk].notna().astype(float) * weights[tk]
             for tk in ("tier1", "tier2", "tier3", "tier4", "tier5")
             if not tier_scores[tk].isna().all() and weights[tk] > 0],
            axis=1,
        )
        weight_coverage = coverage_mask.sum(axis=1).replace(0, np.nan)
        composite = raw_sum / weight_coverage
    else:
        composite = pd.Series(np.nan, index=cache.index)

    composite = composite.clip(0, 100)
    cache["fh_composite_score"] = composite
    result.columns_added.append("fh_composite_score")

    # Label
    cache["fh_composite_label"] = composite.apply(_composite_label)
    result.columns_added.append("fh_composite_label")

    # Also add rate-of-change for temporal models to learn trends
    if not composite.isna().all():
        cache["fh_composite_delta_5d"] = composite.diff(5)
        cache["fh_composite_delta_21d"] = composite.diff(21)
        result.columns_added.extend(["fh_composite_delta_5d", "fh_composite_delta_21d"])

    # Summary stats
    result.n_days_scored = int(composite.notna().sum())
    result.mean_composite = float(composite.mean()) if result.n_days_scored > 0 else float("nan")
    if result.n_days_scored > 0:
        result.latest_composite = float(composite.iloc[-1]) if not np.isnan(composite.iloc[-1]) else float("nan")
        result.latest_label = _composite_label(result.latest_composite)

    # ------------------------------------------------------------------
    # Extended models: inject as daily cache columns for temporal learning
    # ------------------------------------------------------------------

    # Altman Z-Score -- bankruptcy predictor (strengthens Tier 1-2 signals)
    try:
        z_result = compute_altman_z_score(cache)
        result.altman_z = z_result
        if z_result.available and z_result.z_score_series is not None:
            cache["fh_altman_z_score"] = z_result.z_score_series
            zone_map = {True: "safe", False: "grey"}  # placeholder
            cache["fh_altman_z_zone"] = z_result.z_score_series.apply(
                lambda v: "safe" if v >= _Z_SAFE_THRESHOLD
                else ("distress" if v <= _Z_DISTRESS_THRESHOLD else "grey")
                if not np.isnan(v) else "unknown"
            )
            result.columns_added.extend(["fh_altman_z_score", "fh_altman_z_zone"])
    except Exception as exc:
        logger.warning("Altman Z-Score failed: %s", exc)

    # Beneish M-Score -- earnings manipulation (ethical filter signal)
    try:
        m_result = compute_beneish_m_score(cache)
        result.beneish_m = m_result
        if m_result.available and m_result.m_score is not None:
            cache["fh_beneish_m_score"] = m_result.m_score
            cache["fh_beneish_flag"] = 1.0 if m_result.likely_manipulator else 0.0
            result.columns_added.extend(["fh_beneish_m_score", "fh_beneish_flag"])
    except Exception as exc:
        logger.warning("Beneish M-Score failed: %s", exc)

    # Liquidity Runway -- months of cash survival (strengthens Tier 1)
    try:
        runway_result = compute_liquidity_runway(cache)
        result.liquidity_runway = runway_result
        if runway_result.available and runway_result.months_of_runway is not None:
            months = runway_result.months_of_runway
            cache["fh_runway_months"] = months if months != float("inf") else 999.0
            result.columns_added.append("fh_runway_months")
    except Exception as exc:
        logger.warning("Liquidity runway failed: %s", exc)

    logger.info(
        "Financial health: %d days scored, composite=%.1f (%s), cols=%d",
        result.n_days_scored,
        result.latest_composite,
        result.latest_label,
        len(result.columns_added),
    )

    return cache, result

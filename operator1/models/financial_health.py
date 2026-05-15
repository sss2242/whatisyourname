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

from operator1.scoring_weights import get_weight

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-10  # avoid division by zero

# ---------------------------------------------------------------------------
# Adaptive valuation caps (replaces fixed PE=200, EV/EBITDA=100)
# ---------------------------------------------------------------------------

# Absolute floor caps: never clip below these values, even with sparse data.
# These are the minimum "extreme" thresholds that preserve enough range for
# the expanding percentile rank normalization to produce meaningful scores.
_PE_CAP_FLOOR: float = 50.0
_EV_CAP_FLOOR: float = 30.0

# Absolute ceiling caps: sanity bounds beyond which values are economically
# meaningless (negative earnings approaching zero produce PE -> infinity).
_PE_CAP_CEILING: float = 500.0
_EV_CAP_CEILING: float = 200.0

# Minimum observations before switching from textbook defaults to adaptive
_MIN_OBS_ADAPTIVE: int = 20

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
def _z_safe_threshold() -> float:
    try:
        from operator1.scoring_weights import get_weight
        return float(get_weight("financial_health.altman_safe_zone", 2.99))
    except Exception:
        return 2.99


def _z_distress_threshold() -> float:
    try:
        from operator1.scoring_weights import get_weight
        return float(get_weight("financial_health.altman_distress_zone", 1.81))
    except Exception:
        return 1.81


_Z_SAFE_THRESHOLD = 2.99      # kept for backward compat; use _z_safe_threshold()
_Z_DISTRESS_THRESHOLD = 1.81  # kept for backward compat; use _z_distress_threshold()

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


def _compute_adaptive_valuation_cap(
    series: pd.Series,
    *,
    floor: float,
    ceiling: float,
    textbook_default: float,
    percentile: float = 0.995,
) -> float:
    """Compute a data-driven upper clip for valuation ratios (PE, EV/EBITDA).

    Replaces fixed caps (PE=200, EV/EBITDA=100) with an adaptive threshold
    derived from the company's own expanding distribution.  Three methods
    are tried in order; the final cap is the consensus of whichever succeed.

    **Method 1 -- Log-Normal P99.5 (Aitchison & Brown 1957; Limpert 2001):**
    PE and EV/EBITDA distributions are well-known to be approximately
    log-normal (positively skewed, bounded below by zero).  Fit mu and
    sigma in log-space, then cap at exp(mu + z * sigma) where z is the
    normal quantile for the target percentile.

    **Method 2 -- Tukey Extreme Fence (Tukey 1977):**
    Upper fence = Q3 + 3 * IQR.  Non-parametric, makes no distributional
    assumptions.  The k=3 multiplier identifies extreme outliers (vs k=1.5
    for mild).

    **Method 3 -- MAD-Based Cap (Iglewicz & Hoaglin 1993):**
    Cap = median + 3.5 * MAD * 1.4826.  Uses the Median Absolute Deviation
    scaled by 1.4826 to be comparable to standard deviation under normality.
    The 3.5 threshold is the standard recommendation from the NIST
    Engineering Statistics Handbook.

    The final cap is the *median* of the three method outputs (robust to
    any single method failing or producing an outlier estimate), clamped
    to [floor, ceiling].

    Parameters
    ----------
    series:
        Raw positive-valued ratio series (e.g., pe_ratio_calc). NaN and
        non-positive values are excluded before computation.
    floor:
        Minimum cap (prevents over-capping with sparse data).
    ceiling:
        Maximum cap (absolute sanity bound).
    textbook_default:
        Fallback cap when insufficient data for adaptive methods.
    percentile:
        Target percentile for the log-normal method (default 0.995).

    Returns
    -------
    float
        The adaptive upper clip value, in [floor, ceiling].
    """
    from scipy.stats import norm as _norm

    # Filter to positive, finite values only
    clean = series.dropna()
    clean = clean[(clean > 0) & np.isfinite(clean)]

    if len(clean) < _MIN_OBS_ADAPTIVE:
        return min(max(textbook_default, floor), ceiling)

    candidates: list[float] = []

    # --- Method 1: Log-Normal P99.5 (Aitchison & Brown 1957) ---
    # PE/EV distributions are approximately log-normal: log(PE) ~ N(mu, sigma^2).
    # The percentile in the original space is exp(mu + z * sigma).
    try:
        log_vals = np.log(clean.values)
        mu_ln = float(np.mean(log_vals))
        sigma_ln = float(np.std(log_vals, ddof=1))
        if sigma_ln > 1e-8:
            z_score = float(_norm.ppf(percentile))
            lognormal_cap = float(np.exp(mu_ln + z_score * sigma_ln))
            candidates.append(lognormal_cap)
    except Exception:
        pass

    # --- Method 2: Tukey Extreme Fence (Tukey 1977) ---
    # Upper fence = Q3 + k * IQR, with k=3 for extreme outliers.
    try:
        q1 = float(np.percentile(clean.values, 25))
        q3 = float(np.percentile(clean.values, 75))
        iqr = q3 - q1
        if iqr > 1e-8:
            tukey_cap = q3 + 3.0 * iqr
            candidates.append(tukey_cap)
    except Exception:
        pass

    # --- Method 3: MAD-Based Cap (Iglewicz & Hoaglin 1993) ---
    # Cap = median + 3.5 * MAD * 1.4826 (scaled MAD approximates std
    # under normality; 3.5 is the NIST recommendation for outlier flagging).
    try:
        median_val = float(np.median(clean.values))
        mad = float(np.median(np.abs(clean.values - median_val))) * 1.4826
        if mad > 1e-8:
            mad_cap = median_val + 3.5 * mad
            candidates.append(mad_cap)
    except Exception:
        pass

    if not candidates:
        return min(max(textbook_default, floor), ceiling)

    # Consensus: take the median of available method outputs.
    # This is robust to any single method producing an extreme estimate.
    consensus = float(np.median(candidates))

    # Clamp to [floor, ceiling]
    cap = min(max(consensus, floor), ceiling)

    logger.debug(
        "Adaptive valuation cap: methods=%s, consensus=%.1f, final=%.1f "
        "(floor=%.1f, ceiling=%.1f, n=%d)",
        [round(c, 1) for c in candidates], consensus, cap, floor, ceiling, len(clean),
    )

    return cap


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


# ---------------------------------------------------------------------------
# Cross-sectional scoring (Batch B -- sector reference ranges)
# ---------------------------------------------------------------------------

_SECTOR_RANGES: dict | None = None


def _load_sector_ranges() -> dict:
    """Load sector reference ranges from config/sector_reference_ranges.yml."""
    global _SECTOR_RANGES
    if _SECTOR_RANGES is not None:
        return _SECTOR_RANGES
    try:
        from operator1.config_loader import load_config
        _SECTOR_RANGES = load_config("sector_reference_ranges")
    except Exception:
        _SECTOR_RANGES = {}
    return _SECTOR_RANGES


def _get_sector_range(sector: str, variable: str) -> list[float] | None:
    """Get [p10, p25, median, p75, p90] for a variable in a sector."""
    ranges = _load_sector_ranges()
    if not ranges:
        return None
    sector_key = sector.lower().replace(" ", "_") if sector else "_default"
    # Try exact match, then substring, then default
    section = ranges.get(sector_key)
    if section is None:
        for key in ranges:
            if key != "_default" and (key in sector_key or sector_key in key):
                section = ranges[key]
                break
    if section is None:
        section = ranges.get("_default", {})
    return section.get(variable)


def _cross_sectional_score(
    value: float,
    sector_range: list[float],
    higher_is_better: bool = True,
) -> float:
    """Score 0-100 based on sector reference range.

    Uses linear interpolation between [p10, p25, median, p75, p90]
    breakpoints. Value at sector median = 50, at p90 = 90, etc.
    """
    if value is None or np.isnan(value):
        return np.nan
    bp = list(sector_range)
    if len(bp) != 5:
        return np.nan
    if not higher_is_better:
        bp = bp[::-1]
    score_anchors = [10.0, 25.0, 50.0, 75.0, 90.0]
    # Within range: linear interpolation
    for i in range(len(bp) - 1):
        lo, hi = min(bp[i], bp[i + 1]), max(bp[i], bp[i + 1])
        if lo <= value <= hi:
            if abs(bp[i + 1] - bp[i]) < 1e-12:
                return score_anchors[i]
            frac = (value - bp[i]) / (bp[i + 1] - bp[i])
            return score_anchors[i] + frac * (score_anchors[i + 1] - score_anchors[i])
    # Below p10
    if (higher_is_better and value < bp[0]) or (not higher_is_better and value > bp[0]):
        return max(0.0, 5.0)
    # Above p90
    return min(100.0, 95.0)


def _hybrid_score_series(
    s: pd.Series,
    sector: str,
    variable: str,
    higher_is_better: bool = True,
    cross_weight: float = 0.7,
) -> pd.Series:
    """Compute hybrid score: 70% cross-sectional + 30% self-history trend.

    When sector reference range is available, uses cross-sectional scoring
    anchored to sector peers. Falls back to pure self-percentile when
    no sector range exists.
    """
    sr = _get_sector_range(sector, variable)
    if sr is None:
        # No sector range: pure self-percentile (original behavior)
        return _normalize_series(s, invert=not higher_is_better)

    # Cross-sectional score
    cross = s.apply(lambda v: _cross_sectional_score(v, sr, higher_is_better))
    # Self-history trend score (existing method)
    trend = _normalize_series(s, invert=not higher_is_better)
    # Weighted combination
    return cross_weight * cross + (1 - cross_weight) * trend


def _normalize_series(
    s: pd.Series,
    *,
    lower: float | None = None,
    upper: float | None = None,
    invert: bool = False,
    halflife: int | None = None,
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
    halflife : int, optional
        When set, uses exponentially weighted percentile rank where recent
        observations are weighted more heavily (half-life in business days).
        This prevents long healthy periods from diluting recent deterioration.
        Typical values: 63 (quarterly filer), 126 (semi-annual), 252 (annual).
        When None (default), uses the original expanding percentile rank.

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

    if halflife is not None and halflife > 0:
        # Exponentially weighted percentile rank: recent values matter more.
        # For each day, compute weighted rank where weights decay with age.
        ranked = _ewm_percentile_rank(work, halflife=halflife) * 100.0
    else:
        # Original expanding percentile rank (stable, equal-weight)
        ranked = work.expanding(min_periods=1).rank(pct=True) * 100.0

    if invert:
        ranked = 100.0 - ranked

    return ranked


def _ewm_percentile_rank(s: pd.Series, halflife: int = 63) -> pd.Series:
    """Exponentially weighted percentile rank.

    For each day t, computes what fraction of historical values (weighted
    by recency) are below the current value. Recent observations contribute
    more to the rank, making the score more responsive to deterioration.

    Complexity: O(n * min(n, 2*halflife)) -- capped window for efficiency.
    """
    values = s.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)
    decay = np.log(2) / max(halflife, 1)
    # Cap lookback to 4 half-lives (97% of total weight) for performance
    max_lookback = min(4 * halflife, n)

    for i in range(1, n):
        if np.isnan(values[i]):
            continue
        start = max(0, i - max_lookback)
        window = values[start:i + 1]
        valid_mask = ~np.isnan(window)
        if valid_mask.sum() < 2:
            continue
        valid_vals = window[valid_mask]
        ages = np.arange(len(window))[valid_mask]
        # Weights: newest (index=len-1) has weight 1.0, older decays
        weights = np.exp(-decay * (len(window) - 1 - ages))
        # Weighted rank: sum of weights where value <= current
        current_val = values[i]
        below_weight = np.sum(weights[valid_vals <= current_val])
        total_weight = np.sum(weights)
        result[i] = below_weight / max(total_weight, _EPS)

    return pd.Series(result, index=s.index)


def _score_liquidity(cache: pd.DataFrame, sector: str = "") -> pd.Series:
    """Tier 1 -- Liquidity & Cash score.

    Looks at: cash_ratio, free_cash_flow_ttm (or operating_cash_flow),
    cash_and_equivalents.

    When ``sector`` is provided and sector reference ranges exist,
    uses 70% cross-sectional + 30% self-history hybrid scoring.
    """
    components: list[pd.Series] = []

    if "cash_ratio" in cache.columns:
        components.append(
            _hybrid_score_series(cache["cash_ratio"], sector, "current_ratio", higher_is_better=True)
            if sector else _normalize_series(cache["cash_ratio"], lower=0, upper=10)
        )

    if "free_cash_flow_ttm_asof" in cache.columns:
        components.append(
            _hybrid_score_series(cache["free_cash_flow_ttm_asof"], sector, "fcf_yield", higher_is_better=True)
            if sector else _normalize_series(cache["free_cash_flow_ttm_asof"])
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


def _score_solvency(cache: pd.DataFrame, sector: str = "") -> pd.Series:
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


def _score_stability(cache: pd.DataFrame, sector: str = "") -> pd.Series:
    """Tier 3 -- Market Stability score.

    Looks at: volatility_21d (inverted), drawdown_252d (inverted), volume.
    Falls back to private company proxies (financial_volatility,
    equity_drawdown, revenue_velocity) when OHLCV-derived columns are
    missing.
    """
    components: list[pd.Series] = []

    # Volatility: prefer volatility_21d, fall back to financial_volatility
    for vol_col in ("volatility_21d", "financial_volatility"):
        if vol_col in cache.columns and cache[vol_col].notna().any():
            components.append(
                _normalize_series(cache[vol_col], invert=True)
            )
            break

    # Drawdown: prefer drawdown_252d, fall back to equity_drawdown
    for dd_col in ("drawdown_252d", "equity_drawdown"):
        if dd_col in cache.columns and cache[dd_col].notna().any():
            components.append(
                _normalize_series(cache[dd_col])
            )
            break

    # Volume: prefer volume, fall back to revenue_velocity
    for vol_col in ("volume", "revenue_velocity"):
        if vol_col in cache.columns and cache[vol_col].notna().any():
            components.append(
                _normalize_series(cache[vol_col])
            )
            break

    if not components:
        return pd.Series(np.nan, index=cache.index, name="fh_stability_score")

    score = pd.concat(components, axis=1).mean(axis=1)
    score.name = "fh_stability_score"
    return score


def _score_profitability(cache: pd.DataFrame, sector: str = "") -> pd.Series:
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


def _score_growth(cache: pd.DataFrame, sector: str = "") -> pd.Series:
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
        # Low PE -> potentially undervalued -> higher score.
        # Adaptive cap replaces the fixed 200 using three expert methods:
        # log-normal P99.5 (Aitchison & Brown 1957), Tukey extreme fence
        # (Tukey 1977), MAD-based cap (Iglewicz & Hoaglin 1993).
        pe_cap = _compute_adaptive_valuation_cap(
            cache["pe_ratio_calc"],
            floor=_PE_CAP_FLOOR,
            ceiling=_PE_CAP_CEILING,
            textbook_default=200.0,
        )
        pe = cache["pe_ratio_calc"].clip(lower=0, upper=pe_cap)
        components.append(_normalize_series(pe, invert=True))

    if "ev_to_ebitda" in cache.columns:
        # Low EV/EBITDA -> potentially undervalued -> higher score.
        # Same adaptive cap logic as PE above.
        ev_cap = _compute_adaptive_valuation_cap(
            cache["ev_to_ebitda"],
            floor=_EV_CAP_FLOOR,
            ceiling=_EV_CAP_CEILING,
            textbook_default=100.0,
        )
        ev = cache["ev_to_ebitda"].clip(lower=0, upper=ev_cap)
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


def compute_altman_z_score(df: pd.DataFrame, freq: str = "D") -> AltmanZResult:
    """Compute Altman Z-Score series (bankruptcy predictor, frequency-aware).

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    X1 (WC/TA) and X2 (RE/TA) are STOCK/STOCK -- correct at any freq.
    X3 (EBIT/TA) and X5 (Revenue/TA) are FLOW/STOCK -- need annualization
    at Q/A/S.  At D, ebit/revenue are daily rates -> x3/x5 are ~1000x
    too small.  We annualize flow variables before dividing by TA.
    X4 (MVE/TL) is MARKET/STOCK -- correct at any freq.
    """
    freq = freq.upper() if freq else "D"
    # D/W/M = 1.0 (no annualization -- mixed scale on daily cache).
    # Q/A/S = annualize to annual scale for correct Altman Z.
    _annualize = {"D": 1.0, "W": 1.0, "M": 1.0, "Q": 4.0, "S": 2.0, "A": 1.0}
    _mult = _annualize.get(freq, 1.0)

    result = AltmanZResult()

    total_assets = df.get("total_assets")
    if total_assets is None or total_assets.notna().sum() == 0:
        result.error = "total_assets not available"
        return result

    ta = total_assets.replace(0, np.nan)
    tl = df.get("total_liabilities", pd.Series(np.nan, index=df.index)).replace(0, np.nan)

    # X1: Working Capital / TA (STOCK/STOCK -- OK at any freq)
    current_assets = df.get("current_assets", pd.Series(np.nan, index=df.index))
    current_liabilities = df.get("current_liabilities", pd.Series(np.nan, index=df.index))
    x1 = (current_assets - current_liabilities) / ta

    # X2: Retained Earnings / TA (STOCK/STOCK -- OK at any freq)
    retained_earnings = df.get("retained_earnings", pd.Series(np.nan, index=df.index))
    x2 = retained_earnings / ta

    # X3: EBIT / TA (FLOW/STOCK -- annualize EBIT at Q/A/S)
    ebit = df.get("ebit", df.get("operating_income", df.get("ebitda", pd.Series(np.nan, index=df.index))))
    x3 = (ebit * _mult) / ta

    # X4: Market Cap / TL (MARKET/STOCK -- OK at any freq)
    market_cap = df.get("market_cap", pd.Series(np.nan, index=df.index))
    x4 = market_cap / tl

    # X5: Revenue / TA (FLOW/STOCK -- annualize Revenue at Q/A/S)
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    x5 = (revenue * _mult) / ta

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
    elif m_score > get_weight("financial_health.beneish_threshold", -2.22):
        result.verdict = "possible"
    else:
        result.verdict = "unlikely"

    logger.info("Beneish M-Score: %.2f (%s manipulation)", result.m_score, result.verdict)
    return result


def compute_liquidity_runway(df: pd.DataFrame, freq: str = "D") -> LiquidityRunwayResult:
    """Estimate months of cash runway at current burn rate (frequency-aware).

    Answers: "If revenue stopped today, how many months can this company
    survive on its current cash reserves at the current spending rate?"

    OCF is a flow variable: at Q = quarterly total, at A = annual total,
    at D = daily rate.  Monthly burn = |OCF| / months_in_period.
    At Q: |OCF_q| / 3.  At A: |OCF_a| / 12.  At D: |OCF_daily| * 30.
    """
    freq = freq.upper() if freq else "D"
    # Months in one filing period at each frequency.
    # D=12.0 (backward compatible: assumes OCF in daily cache is annual-scale
    # from forward-fill of quarterly/annual filing).  Correct runway comes
    # from Q/A pipeline results after Stage 2.F fusion.
    _months_in_period = {"A": 12.0, "S": 6.0, "Q": 3.0, "M": 1.0, "W": 1.0, "D": 12.0}
    _mip = _months_in_period.get(freq, 12.0)

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
            # OCF is negative: compute monthly burn from period OCF
            monthly_burn = abs(latest_ocf) / max(_mip, 0.01)
            result.monthly_burn_rate = float(monthly_burn)
        else:
            capex = df.get("capex")
            if capex is not None and capex.notna().sum() > 0:
                latest_capex = float(capex.dropna().iloc[-1])
                net_outflow = abs(latest_capex) - latest_ocf
                if net_outflow > 0:
                    result.monthly_burn_rate = float(net_outflow / max(_mip, 0.01))
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
    freq: str = "D",
    sector: str = "",
) -> tuple[pd.DataFrame, FinancialHealthResult]:
    """Compute financial health scores and inject into the cache.

    Runs the 5-tier scoring system (liquidity, solvency, stability,
    profitability, growth) plus extended models (Altman Z, Beneish M,
    runway).

    Parameters
    ----------
    cache : pd.DataFrame
        Cache DataFrame with derived features.
    hierarchy_weights : dict, optional
        Tier weights for the composite.
    freq : str
        Data frequency (D/W/M/Q/A/S).  At D/W/M, Tier 5 (Growth)
        is scored with reduced confidence because PE, EV/EBITDA are
        distorted by flow-variable interpolation.  Altman Z components
        x3 (EBIT/TA) and x5 (Revenue/TA) are also unreliable at D.
        At Q/A/S, all tiers and Altman Z components are fully valid.
    sector : str
        Company sector for sector-aware scoring adjustments.

    Returns
    -------
    (cache, result)
        The cache with ``fh_*`` columns, and ``FinancialHealthResult``.
    """
    freq = freq.upper() if freq else "D"
    _native_ratio_freqs = {"Q", "A", "S"}
    logger.info("Computing financial health scores (freq=%s)...", freq)

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
    # Tier 5 (Growth) uses PE, EV/EBITDA, revenue_growth -- PE and EV/EBITDA
    # are MARKET/FLOW ratios that are distorted on interpolated daily data.
    # At D/W/M: still compute T5 but its values will be based on whatever
    # PE/EV values are in the cache (may be corrected by prior fusion step).
    # At Q/A/S: T5 is fully reliable (native-scale ratios).
    tier_scores: dict[str, pd.Series] = {
        "tier1": _score_liquidity(cache, sector=sector),
        "tier2": _score_solvency(cache, sector=sector),
        "tier3": _score_stability(cache, sector=sector),
        "tier4": _score_profitability(cache, sector=sector),
        "tier5": _score_growth(cache, sector=sector),
    }
    if freq not in _native_ratio_freqs:
        logger.debug(
            "FH T5 (Growth) at freq=%s may use distorted PE/EV values",
            freq,
        )

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

    # ------------------------------------------------------------------
    # Financial health calibration adjustments
    # ------------------------------------------------------------------
    # Fix: mega-cap companies with high debt but excellent debt serviceability
    # (e.g., Apple: D/E=1.7 but interest coverage=29x, cash > debt)
    # should not be penalized as heavily as distressed companies.

    # Adjustment 1: Debt serviceability override
    # When interest coverage > 10x AND OCF can repay all debt within 3 years,
    # cap the solvency penalty at -10 instead of the full penalty.
    if "interest_coverage" in cache.columns and "total_debt" in cache.columns:
        ic = cache["interest_coverage"].fillna(0)
        ocf = cache.get("operating_cash_flow", pd.Series(0, index=cache.index)).fillna(0)
        debt = cache["total_debt"].fillna(0)
        well_serviced = (ic > 10) & (ocf * 3 > debt) & (debt > 0)
        if well_serviced.any():
            # Boost composite where debt is well-serviced but solvency score is low
            solvency = tier_scores.get("tier2", pd.Series(50, index=cache.index))
            solvency_penalty = (50 - solvency).clip(lower=0)  # how much solvency is below average
            # Recover up to 60% of the solvency penalty
            recovery = solvency_penalty * 0.6 * well_serviced.astype(float) * weights.get("tier2", 0.2)
            composite = composite + recovery

    # Adjustment 2: Cash reserves bonus
    # When cash + short-term investments > total debt, add a bonus.
    if "cash_and_equivalents" in cache.columns and "total_debt" in cache.columns:
        cash = cache["cash_and_equivalents"].fillna(0)
        debt = cache["total_debt"].fillna(0)
        net_cash_positive = (cash > debt) & (debt > 0)
        if net_cash_positive.any():
            # Bonus proportional to net cash / debt ratio, capped at +8
            cash_ratio_to_debt = (cash / debt.clip(lower=1)).clip(upper=3.0)
            bonus = (cash_ratio_to_debt - 1.0).clip(lower=0) * 4.0 * net_cash_positive.astype(float)
            composite = composite + bonus.clip(upper=8.0)

    # ------------------------------------------------------------------
    # Adjustment 3: Sector-aware baseline floors (2026-05-09 fix)
    # ------------------------------------------------------------------
    # Technology companies (Apple, Google, Microsoft) structurally operate
    # with current_ratio < 1.0 (negative working capital model), high D/E
    # (stock buybacks funded by cheap debt), and thin net margins relative
    # to gross margins (massive R&D spend).  The expanding percentile rank
    # self-referentially scores these as "bad" because the company has
    # always operated this way.
    #
    # Fix: when a sector has known structural norms that differ from
    # textbook ideals, apply a floor boost so fundamentally healthy
    # mega-caps don't score 28/100 ("Weak").
    #
    # The boost is proportional to evidence of health (gross margin > 30%
    # and positive FCF) to avoid rescuing truly distressed tech companies.
    _SECTOR_FLOORS: dict[str, dict[str, float]] = {
        "technology": {
            "profitability_floor": 40.0,  # tech with high gross margins
            "solvency_floor": 35.0,       # tech uses leverage for buybacks
            "gross_margin_gate": 0.30,     # only apply if gross margin > 30%
        },
        "financial services": {
            "solvency_floor": 40.0,       # banks are inherently leveraged
            "liquidity_floor": 35.0,      # banks operate with low current ratio
            "gross_margin_gate": 0.0,     # not applicable for banks
        },
        "communication services": {
            "profitability_floor": 35.0,
            "solvency_floor": 30.0,
            "gross_margin_gate": 0.25,
        },
    }
    # Normalize sector label (SIC "Electronic Computers" -> "technology")
    # so that sector-aware floors match correctly.
    try:
        from operator1.sector_mapper import normalize_sector
        sector_lower = normalize_sector(sector) if sector else ""
    except ImportError:
        sector_lower = sector.lower().strip() if sector else ""
    _sector_config = None
    for _sk, _sv in _SECTOR_FLOORS.items():
        if _sk in sector_lower:
            _sector_config = _sv
            break

    if _sector_config is not None:
        _gm_gate = _sector_config.get("gross_margin_gate", 0.30)
        _has_gm = "gross_margin" in cache.columns and cache["gross_margin"].notna().any()
        _gm_ok = True  # default pass if no margin data
        if _has_gm and _gm_gate > 0:
            _latest_gm = float(cache["gross_margin"].dropna().iloc[-1]) if cache["gross_margin"].notna().any() else 0
            _gm_ok = _latest_gm >= _gm_gate
        _has_positive_fcf = (
            "free_cash_flow" in cache.columns
            and cache["free_cash_flow"].notna().any()
            and float(cache["free_cash_flow"].dropna().iloc[-1]) > 0
        )

        if _gm_ok:
            _boost_applied = False
            # Profitability floor: tech with 77% gross margins should not score 11/100
            _prof_floor = _sector_config.get("profitability_floor", 0)
            if _prof_floor > 0 and "tier4" in tier_scores:
                _prof = tier_scores["tier4"]
                _prof_deficit = (_prof_floor - _prof).clip(lower=0)
                _prof_boost = _prof_deficit * weights.get("tier4", 0.2)
                composite = composite + _prof_boost
                if _prof_deficit.max() > 0:
                    _boost_applied = True

            # Solvency floor: buyback-funded leverage should not dominate
            _solv_floor = _sector_config.get("solvency_floor", 0)
            if _solv_floor > 0 and "tier2" in tier_scores:
                _solv = tier_scores["tier2"]
                _solv_deficit = (_solv_floor - _solv).clip(lower=0)
                _solv_boost = _solv_deficit * weights.get("tier2", 0.2)
                composite = composite + _solv_boost
                if _solv_deficit.max() > 0:
                    _boost_applied = True

            # Liquidity floor
            _liq_floor = _sector_config.get("liquidity_floor", 0)
            if _liq_floor > 0 and "tier1" in tier_scores:
                _liq = tier_scores["tier1"]
                _liq_deficit = (_liq_floor - _liq).clip(lower=0)
                _liq_boost = _liq_deficit * weights.get("tier1", 0.2)
                composite = composite + _liq_boost
                if _liq_deficit.max() > 0:
                    _boost_applied = True

            # Extra bonus for positive FCF (confirms the model is working)
            if _has_positive_fcf and _boost_applied:
                composite = composite + 5.0

            if _boost_applied:
                logger.info(
                    "FH sector adjustment (%s): composite boosted "
                    "(gross_margin_ok=%s, positive_fcf=%s)",
                    sector, _gm_ok, _has_positive_fcf,
                )

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
        z_result = compute_altman_z_score(cache, freq=freq)
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

    # Merton Distance-to-Default (Merton 1974) -- structural credit risk
    # Models equity as a call option on firm assets. DD measures how many
    # standard deviations the asset value is above the default point.
    # DD > 5 = extremely safe, DD < 2 = distress zone.
    try:
        _has_mcap = "close" in cache.columns and "shares_outstanding" in cache.columns
        _has_vol = "volatility_21d" in cache.columns
        _has_debt = "total_debt" in cache.columns
        if _has_debt and _has_vol:
            _total_debt = cache["total_debt"].fillna(0)
            _equity_vol = cache["volatility_21d"].fillna(0.01) * np.sqrt(252)  # annualize
            # Estimate market cap: close * shares_outstanding, or use total_equity as proxy
            if _has_mcap:
                _mcap = (cache["close"] * cache["shares_outstanding"]).fillna(0)
            elif "market_cap" in cache.columns:
                _mcap = cache["market_cap"].fillna(0)
            elif "total_equity" in cache.columns:
                _mcap = cache["total_equity"].fillna(0).clip(lower=1)
            else:
                _mcap = pd.Series(0, index=cache.index)

            _V = _mcap + _total_debt  # asset value proxy
            _D = _total_debt.clip(lower=1)  # default point (avoid div by zero)
            _equity_fraction = (_V - _D) / _V.clip(lower=1)
            _sigma_V = _equity_vol * _V / (_V - _D).clip(lower=1)  # asset volatility
            _r = 0.04  # risk-free rate approximation
            _T = 1.0   # 1-year horizon

            _dd_num = np.log(_V / _D) + (_r - 0.5 * _sigma_V ** 2) * _T
            _dd_den = _sigma_V * np.sqrt(_T)
            _dd = _dd_num / _dd_den.clip(lower=0.001)
            _dd = _dd.clip(-10, 20)  # bound extreme values

            # Probability of default: PD = N(-DD) using standard normal CDF
            from scipy.stats import norm
            _pd = norm.cdf(-_dd)

            cache["fh_merton_dd"] = _dd
            cache["fh_merton_pd"] = _pd
            result.columns_added.extend(["fh_merton_dd", "fh_merton_pd"])

            # Boost composite for extremely safe companies (DD > 5)
            _dd_bonus = (_dd - 5.0).clip(lower=0, upper=5) * 2.0  # up to +10 bonus
            _dd_valid = _dd.notna() & (_total_debt > 0)
            composite = composite + _dd_bonus * _dd_valid.astype(float)
            composite = composite.clip(0, 100)
            cache["fh_composite_score"] = composite

            latest_dd = float(_dd.iloc[-1]) if _dd.notna().any() else float("nan")
            logger.info(
                "Merton DD: latest=%.2f, PD=%.6f, DD>5 bonus applied",
                latest_dd, float(_pd.iloc[-1]) if _pd.notna().any() else 0,
            )
    except Exception as exc:
        logger.warning("Merton Distance-to-Default failed: %s", exc)

    # Liquidity Runway -- months of cash survival (strengthens Tier 1)
    try:
        runway_result = compute_liquidity_runway(cache, freq=freq)
        result.liquidity_runway = runway_result
        if runway_result.available and runway_result.months_of_runway is not None:
            months = runway_result.months_of_runway
            cache["fh_runway_months"] = months if months != float("inf") else 999.0
            result.columns_added.append("fh_runway_months")
    except Exception as exc:
        logger.warning("Liquidity runway failed: %s", exc)

    # ------------------------------------------------------------------
    # Ensemble Distress Prediction (Altman Z + Ohlson O + Zmijewski + Merton)
    # Stacked ensemble with inverse-Brier weighting. Uses survival flag as
    # proxy label for calibration.
    # ------------------------------------------------------------------
    try:
        _distress_models: dict[str, pd.Series] = {}

        # Model 1: Altman Z -> probability via logistic transform
        if "fh_altman_z_score" in cache.columns and cache["fh_altman_z_score"].notna().any():
            _z = cache["fh_altman_z_score"]
            # P(distress) = 1 / (1 + exp(Z - 1.81))  -- centered at grey zone
            _distress_models["altman_z"] = 1.0 / (1.0 + np.exp(_z - 1.81))

        # Model 2: Ohlson O-Score (Ohlson 1980, frds pattern)
        # O = -1.32 - 0.407*ln(TA) + 6.03*(TL/TA) - 1.43*(WC/TA)
        #     + 0.076*(CL/CA) - 1.72*OENEG - 2.37*(NI/TA) - 1.83*(FFO/TL)
        #     + 0.285*INTWO - 0.521*CHIN
        _ta = cache.get("total_assets")
        _tl = cache.get("total_liabilities")
        _ni = cache.get("net_income")
        _ca = cache.get("current_assets")
        _cl = cache.get("current_liabilities")
        _ocf = cache.get("operating_cash_flow")
        if all(x is not None for x in [_ta, _tl, _ni]):
            _ta_f = _ta.astype(float).clip(lower=_EPS)
            _tl_f = _tl.astype(float).fillna(0)
            _ni_f = _ni.astype(float).fillna(0)
            _wc = (_ca.astype(float).fillna(0) - _cl.astype(float).fillna(0)) if _ca is not None and _cl is not None else pd.Series(0, index=cache.index)
            _ffo = _ocf.astype(float).fillna(0) if _ocf is not None else pd.Series(0, index=cache.index)
            _cl_ca = (_cl.astype(float).fillna(1) / _ca.astype(float).clip(lower=_EPS)) if _ca is not None and _cl is not None else pd.Series(0, index=cache.index)
            _oeneg = (_tl_f > _ta_f).astype(float)
            _ni_shifted = _ni_f.shift(252).fillna(_ni_f)
            _chin = ((_ni_f - _ni_shifted) / (_ni_f.abs() + _ni_shifted.abs()).clip(lower=_EPS))

            _o_score = (
                -1.32
                - 0.407 * np.log(_ta_f.clip(lower=1))
                + 6.03 * (_tl_f / _ta_f)
                - 1.43 * (_wc / _ta_f)
                + 0.076 * _cl_ca
                - 1.72 * _oeneg
                - 2.37 * (_ni_f / _ta_f)
                - 1.83 * (_ffo / _tl_f.clip(lower=_EPS))
                + 0.285 * ((_ni_f < 0).astype(float) & (_ni_shifted < 0).astype(float)).astype(float)
                - 0.521 * _chin
            )
            _distress_models["ohlson_o"] = 1.0 / (1.0 + np.exp(-_o_score))

        # Model 3: Zmijewski Score (Zmijewski 1984, probit)
        if all(x is not None for x in [_ta, _tl, _ni]):
            _zmij = -4.336 - 4.513 * (_ni_f / _ta_f) + 5.679 * (_tl_f / _ta_f)
            if _ca is not None and _cl is not None:
                _zmij = _zmij + 0.004 * (_ca.astype(float).fillna(0) / _cl.astype(float).clip(lower=_EPS))
            from scipy.stats import norm as _norm
            _distress_models["zmijewski"] = pd.Series(_norm.cdf(_zmij.values), index=cache.index)

        # Model 4: Merton PD (already computed above)
        if "fh_merton_pd" in cache.columns and cache["fh_merton_pd"].notna().any():
            _distress_models["merton_pd"] = cache["fh_merton_pd"]

        if len(_distress_models) >= 2:
            _stacked = pd.DataFrame(_distress_models)
            # Equal weighting (no labeled default data for Brier scoring)
            _ensemble = _stacked.mean(axis=1)

            # Label mapping for profile
            def _distress_label(p: float) -> str:
                if p > 0.5:
                    return "critical"
                if p > 0.3:
                    return "warning"
                if p > 0.15:
                    return "watch"
                return "safe"

            cache["fh_ensemble_distress_prob"] = _ensemble.clip(0, 1)
            cache["fh_ensemble_distress_label"] = _ensemble.apply(_distress_label)
            result.columns_added.extend(["fh_ensemble_distress_prob", "fh_ensemble_distress_label"])
            logger.info(
                "Ensemble distress: %d models stacked, latest P(distress)=%.3f",
                len(_distress_models), float(_ensemble.iloc[-1]) if _ensemble.notna().any() else 0,
            )
    except Exception as exc:
        logger.debug("Ensemble distress prediction failed: %s", exc)

    # ------------------------------------------------------------------
    # CVaR-Weighted Composite (Rockafellar & Uryasev 2000)
    # More sensitive to the weakest tier than weighted average.
    # ------------------------------------------------------------------
    try:
        _tier_scores = pd.DataFrame({
            "t1": cache.get("fh_liquidity_score", pd.Series(dtype=float)),
            "t2": cache.get("fh_solvency_score", pd.Series(dtype=float)),
            "t3": cache.get("fh_stability_score", pd.Series(dtype=float)),
            "t4": cache.get("fh_profitability_score", pd.Series(dtype=float)),
            "t5": cache.get("fh_growth_score", pd.Series(dtype=float)),
        })
        if _tier_scores.notna().sum().sum() > 0:
            _alpha = 0.30  # tail level: average of worst 30% of tiers

            def _daily_cvar(row: pd.Series) -> float:
                scores = row.dropna().values
                if len(scores) < 2:
                    return float(np.nanmean(scores)) if len(scores) > 0 else float("nan")
                var_alpha = np.percentile(scores, _alpha * 100)
                tail = scores[scores <= var_alpha]
                return float(np.mean(tail)) if len(tail) > 0 else float(var_alpha)

            cache["fh_cvar_composite"] = _tier_scores.apply(_daily_cvar, axis=1)
            result.columns_added.append("fh_cvar_composite")
            logger.info(
                "CVaR composite: latest=%.1f (alpha=%.0f%%)",
                float(cache["fh_cvar_composite"].iloc[-1]) if cache["fh_cvar_composite"].notna().any() else 0,
                _alpha * 100,
            )
    except Exception as exc:
        logger.debug("CVaR composite failed: %s", exc)

    logger.info(
        "Financial health: %d days scored, composite=%.1f (%s), cols=%d",
        result.n_days_scored,
        result.latest_composite,
        result.latest_label,
        len(result.columns_added),
    )

    return cache, result

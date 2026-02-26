"""T4.2 -- Vanity scoring (v2): capital allocation quality assessment.

Measures wasteful or misallocated spending as a percentage of revenue,
composed of five independently testable proxy components that use
fields reliably available from PIT APIs and derived variables.

**V2 Components (proxy-based, always computable):**

1. **R&D vs Growth Mismatch**: high R&D spend + negative revenue growth
   for 4+ consecutive quarters -> vanity R&D.
2. **SGA Bloat v2**: company SGA-to-revenue ratio ranked via peer
   percentile; top-20th-percentile bloat while profitability declines.
3. **Capital Misallocation**: debt rising while liquidity declining,
   or paying dividends while solvency score is critical.
4. **Competitive Decay**: game-theory competitive pressure high +
   peer ranking declining -> losing ground behind a facade.
5. **Sentiment-Reality Gap**: positive news sentiment while financial
   health composite is declining -> vanity signaling.

**Legacy Components (retained for backward compatibility):**

6. Executive compensation excess (when data available)
7. Buyback waste (when data available)
8. Marketing excess during survival (when data available)

``vanity_score = weighted_composite(components, config_weights)``
``vanity_percentage`` retained for backward compatibility.

Returns null (not 0) when revenue is null/zero.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import EPSILON

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration (overridden by config/survival_hierarchy.yml)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "rnd_mismatch": 0.15,
    "sga_bloat": 0.25,
    "capital_misallocation": 0.30,
    "competitive_decay": 0.15,
    "sentiment_gap": 0.15,
}

_DEFAULT_LABELS: list[tuple[float, str]] = [
    (20.0, "Disciplined"),
    (40.0, "Moderate"),
    (70.0, "Wasteful"),
    (100.0, "Reckless"),
]

_TREND_WINDOW_SHORT: int = 21
_TREND_WINDOW_LONG: int = 63


# ---------------------------------------------------------------------------
# Helper: normalize a raw score to 0-100
# ---------------------------------------------------------------------------

def _normalize_score(series: pd.Series, lower: float = 0.0, upper: float = 100.0) -> pd.Series:
    """Clip and normalize a score series to [0, 100]."""
    return series.clip(lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# Component 1: R&D vs Growth Mismatch
# ---------------------------------------------------------------------------

def _rnd_growth_mismatch(df: pd.DataFrame) -> pd.Series:
    """Detect high R&D spend coupled with negative revenue growth.

    Score logic:
    - Compute R&D intensity = R&D / revenue.
    - Compute revenue growth (rolling 252-day pct change).
    - If R&D intensity > 10% AND revenue growth < 0, score scales
      with the magnitude of the mismatch.
    - Max score 100 when R&D is >20% of revenue with deeply negative
      growth.

    Returns 0-100 score; NaN where inputs missing.
    """
    rnd = df.get("rd_expenses", pd.Series(np.nan, index=df.index))
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))

    # R&D intensity
    with np.errstate(divide="ignore", invalid="ignore"):
        rnd_intensity = rnd / revenue
        rnd_intensity = rnd_intensity.replace([np.inf, -np.inf], np.nan)

    # Revenue growth (252-day trailing pct change, approximating annual)
    rev_growth = revenue.pct_change(periods=min(252, max(1, len(revenue) - 1)), fill_method=None)
    rev_growth = rev_growth.replace([np.inf, -np.inf], np.nan)

    # Score: mismatch penalty when R&D intensity > 10% and growth < 0
    rnd_threshold = 0.10
    intensity_excess = (rnd_intensity - rnd_threshold).clip(lower=0)

    # Growth penalty: deeper negative growth = higher score
    growth_penalty = (-rev_growth).clip(lower=0)

    # Raw score: product of intensity excess and growth penalty, scaled
    raw_score = (intensity_excess * growth_penalty) * 1000.0  # scale factor
    raw_score = _normalize_score(raw_score)

    # Null where inputs are missing
    inputs_ok = rnd.notna() & revenue.notna() & (revenue.abs() > EPSILON)
    return raw_score.where(inputs_ok, other=np.nan)


# ---------------------------------------------------------------------------
# Component 2: SGA Bloat v2 (peer-ranked)
# ---------------------------------------------------------------------------

def _sga_bloat_v2(df: pd.DataFrame) -> pd.Series:
    """Compute SGA bloat using peer-relative comparison.

    Score logic:
    - Compute company SGA-to-revenue ratio.
    - Compare against industry median (if available) or use absolute
      thresholds.
    - Penalize further if operating margin is declining while SGA is
      elevated.

    Returns 0-100 score; NaN where inputs missing.
    """
    sga = df.get("sga_expenses", pd.Series(np.nan, index=df.index))
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    op_margin = df.get("operating_margin",
                       pd.Series(np.nan, index=df.index))

    # SGA ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        sga_ratio = sga / revenue
        sga_ratio = sga_ratio.replace([np.inf, -np.inf], np.nan)

    # Industry median SGA ratio (may or may not be available)
    industry_sga = df.get(
        "industry_median_sga_ratio",
        df.get("industry_peers_median_sga_ratio",
               pd.Series(np.nan, index=df.index)),
    )

    # Base score: how much above the benchmark
    if industry_sga.notna().any():
        excess_ratio = (sga_ratio / industry_sga.clip(lower=EPSILON) - 1.0).clip(lower=0)
    else:
        # Fallback: absolute threshold (SGA > 30% of revenue is high)
        excess_ratio = (sga_ratio - 0.30).clip(lower=0)

    base_score = excess_ratio * 100.0

    # Margin decline amplifier: if operating margin is declining (21d),
    # the bloat is worse
    margin_delta = op_margin - op_margin.shift(21)
    margin_penalty = (-margin_delta).clip(lower=0) * 200.0  # amplifier
    margin_penalty = margin_penalty.fillna(0)

    raw_score = _normalize_score(base_score + margin_penalty)

    inputs_ok = sga.notna() & revenue.notna() & (revenue.abs() > EPSILON)
    return raw_score.where(inputs_ok, other=np.nan)


# ---------------------------------------------------------------------------
# Component 3: Capital Misallocation
# ---------------------------------------------------------------------------

def _capital_misallocation(df: pd.DataFrame) -> pd.Series:
    """Detect capital misallocation from balance sheet signals.

    Score logic:
    - Debt trending up while liquidity score declining.
    - Paying dividends while solvency is critical (< 30).
    - Net debt to EBITDA rising above 4x without revenue growth.

    Returns 0-100 score; NaN where inputs missing.
    """
    total_debt = df.get("total_debt_asof",
                        pd.Series(np.nan, index=df.index))
    liq_score = df.get("fh_liquidity_score",
                       pd.Series(np.nan, index=df.index))
    solv_score = df.get("fh_solvency_score",
                        pd.Series(np.nan, index=df.index))
    dividends = df.get("dividends_paid",
                       pd.Series(np.nan, index=df.index))
    nd_ebitda = df.get("net_debt_to_ebitda",
                       pd.Series(np.nan, index=df.index))
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))

    score = pd.Series(0.0, index=df.index)
    has_data = pd.Series(False, index=df.index)

    # Signal 1: Debt rising + liquidity declining
    if total_debt.notna().any() and liq_score.notna().any():
        debt_delta = total_debt.pct_change(21, fill_method=None).clip(-1, 1)
        liq_delta = liq_score - liq_score.shift(21)
        # Debt up AND liquidity down
        signal1 = (debt_delta.clip(lower=0) * 50) + ((-liq_delta).clip(lower=0) * 0.5)
        score = score + signal1.fillna(0)
        has_data = has_data | (total_debt.notna() & liq_score.notna())

    # Signal 2: Paying dividends while solvency is critical
    if dividends.notna().any() and solv_score.notna().any():
        div_positive = dividends.abs() > EPSILON
        solv_critical = solv_score < 30.0
        signal2 = pd.Series(0.0, index=df.index)
        signal2 = signal2.where(~(div_positive & solv_critical), other=40.0)
        score = score + signal2
        has_data = has_data | (dividends.notna() & solv_score.notna())

    # Signal 3: Excessive leverage without growth
    if nd_ebitda.notna().any():
        leverage_excess = (nd_ebitda - 4.0).clip(lower=0) * 10.0
        rev_growth = revenue.pct_change(252, fill_method=None).clip(-1, 1).fillna(0)
        # Penalize only if no revenue growth
        growth_offset = rev_growth.clip(lower=0) * 100.0
        signal3 = (leverage_excess - growth_offset).clip(lower=0)
        score = score + signal3.fillna(0)
        has_data = has_data | nd_ebitda.notna()

    raw_score = _normalize_score(score)
    return raw_score.where(has_data, other=np.nan)


# ---------------------------------------------------------------------------
# Component 4: Competitive Decay
# ---------------------------------------------------------------------------

def _competitive_decay(df: pd.DataFrame) -> pd.Series:
    """Detect competitive decay from game theory and peer signals.

    Score logic:
    - Use competitive_pressure_index (from game_theory.py) if available.
    - Combine with peer_composite_rank decline.
    - Higher score = losing competitive ground while maintaining facade.

    Returns 0-100 score; NaN where inputs missing.
    """
    pressure = df.get("competitive_pressure_index",
                      pd.Series(np.nan, index=df.index))
    peer_rank = df.get("peer_composite_rank",
                       pd.Series(np.nan, index=df.index))
    op_margin = df.get("operating_margin",
                       pd.Series(np.nan, index=df.index))

    score = pd.Series(0.0, index=df.index)
    has_data = pd.Series(False, index=df.index)

    # Competitive pressure contribution (already 0-1 typically)
    if pressure.notna().any():
        score = score + (pressure * 50.0).fillna(0)
        has_data = has_data | pressure.notna()

    # Peer rank decline (rank dropping = losing position)
    if peer_rank.notna().any():
        rank_delta = peer_rank - peer_rank.shift(21)
        # Negative delta = rank dropping (worse)
        rank_penalty = (-rank_delta).clip(lower=0) * 2.0
        score = score + rank_penalty.fillna(0)
        has_data = has_data | peer_rank.notna()

    # Margin compression amplifier
    if op_margin.notna().any():
        margin_delta = op_margin - op_margin.shift(21)
        margin_squeeze = (-margin_delta).clip(lower=0) * 100.0
        score = score + margin_squeeze.fillna(0)
        has_data = has_data | op_margin.notna()

    raw_score = _normalize_score(score)
    return raw_score.where(has_data, other=np.nan)


# ---------------------------------------------------------------------------
# Component 5: Sentiment-Reality Gap
# ---------------------------------------------------------------------------

def _sentiment_reality_gap(df: pd.DataFrame) -> pd.Series:
    """Detect gap between positive sentiment and declining fundamentals.

    Score logic:
    - If news_sentiment_score > 0.3 (positive) AND fh_composite_score
      is declining, the gap signals vanity.
    - Larger gap = higher score.

    Returns 0-100 score; NaN where inputs missing.
    """
    sentiment = df.get("news_sentiment_score",
                       pd.Series(np.nan, index=df.index))
    fh_composite = df.get("fh_composite_score",
                          pd.Series(np.nan, index=df.index))

    has_data = sentiment.notna() & fh_composite.notna()

    # Financial health trend (21-day delta)
    fh_delta = fh_composite - fh_composite.shift(21)

    # Gap: positive sentiment + declining health
    sentiment_positive = sentiment.clip(lower=0)  # only positive sentiment
    health_decline = (-fh_delta).clip(lower=0)  # magnitude of decline

    raw_score = sentiment_positive * health_decline * 5.0
    raw_score = _normalize_score(raw_score)

    return raw_score.where(has_data, other=np.nan)


# ---------------------------------------------------------------------------
# Legacy Component: Executive compensation excess (when data available)
# ---------------------------------------------------------------------------

def _exec_comp_excess(
    df: pd.DataFrame,
    threshold_pct: float = 5.0,
) -> pd.Series:
    """Compute excess executive compensation.

    If ``exec_compensation > threshold_pct% * net_income``, the excess
    above the threshold is returned.  Otherwise 0.

    Returns NaN where either input is missing.
    """
    exec_comp = df.get("exec_compensation", pd.Series(np.nan, index=df.index))
    net_income = df.get("net_income", pd.Series(np.nan, index=df.index))

    threshold = net_income.abs() * (threshold_pct / 100.0)
    excess = exec_comp - threshold
    result = excess.clip(lower=0)
    result = result.where(exec_comp.notna() & net_income.notna(), other=np.nan)

    return result


# ---------------------------------------------------------------------------
# Legacy Component: Buyback waste (when data available)
# ---------------------------------------------------------------------------

def _buyback_waste(df: pd.DataFrame) -> pd.Series:
    """Compute buyback waste.

    If the company is buying back stock (``share_buybacks > 0``) while
    ``free_cash_flow_ttm_asof < 0``, the full buyback amount is waste.

    Returns NaN where inputs are missing.
    """
    buybacks = df.get("share_buybacks", pd.Series(np.nan, index=df.index))
    fcf_ttm = df.get(
        "free_cash_flow_ttm_asof",
        df.get("free_cash_flow", pd.Series(np.nan, index=df.index)),
    )

    waste = buybacks.where(
        buybacks.notna() & fcf_ttm.notna() & (buybacks > 0) & (fcf_ttm < 0),
        other=0.0,
    )
    waste = waste.where(buybacks.notna() | fcf_ttm.notna(), other=np.nan)

    return waste


# ---------------------------------------------------------------------------
# Legacy Component: Marketing excess during survival
# ---------------------------------------------------------------------------

def _marketing_excess(
    df: pd.DataFrame,
    threshold_pct: float = 10.0,
) -> pd.Series:
    """Compute marketing excess during survival mode.

    If ``marketing_spend > threshold_pct% * revenue`` AND the company
    is in survival mode, the excess above the threshold is waste.

    Returns NaN where inputs are missing.
    """
    marketing = df.get("marketing_expense", pd.Series(np.nan, index=df.index))
    revenue = df.get("revenue", pd.Series(np.nan, index=df.index))
    survival = df.get(
        "company_survival_mode_flag",
        pd.Series(0, index=df.index),
    )

    threshold = revenue * (threshold_pct / 100.0)
    excess = (marketing - threshold).clip(lower=0)

    result = excess.where(survival == 1, other=0.0)
    result = result.where(
        marketing.notna() & revenue.notna(), other=np.nan,
    )

    return result


# ---------------------------------------------------------------------------
# Vanity label assignment
# ---------------------------------------------------------------------------

def _assign_label(score: float, labels: list[tuple[float, str]]) -> str:
    """Map a vanity score to a categorical label."""
    if np.isnan(score):
        return "Unknown"
    for threshold, label in labels:
        if score <= threshold:
            return label
    return labels[-1][1] if labels else "Unknown"


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

def _compute_trend(
    series: pd.Series,
    short_window: int = _TREND_WINDOW_SHORT,
    long_window: int = _TREND_WINDOW_LONG,
) -> pd.Series:
    """Classify vanity score trend as rising / stable / falling.

    Uses short-term vs long-term rolling mean crossover.
    """
    short_ma = series.rolling(window=short_window, min_periods=max(1, short_window // 2)).mean()
    long_ma = series.rolling(window=long_window, min_periods=max(1, long_window // 2)).mean()

    delta = short_ma - long_ma
    threshold = 3.0  # points difference to count as a trend

    trend = pd.Series("stable", index=series.index)
    trend = trend.where(~(delta > threshold), other="rising")
    trend = trend.where(~(delta < -threshold), other="falling")
    trend = trend.where(series.notna(), other=np.nan)

    return trend


# ---------------------------------------------------------------------------
# Public API: Legacy vanity_percentage (backward compatible)
# ---------------------------------------------------------------------------

def compute_vanity_percentage(
    df: pd.DataFrame,
    exec_comp_threshold: float = 5.0,
    sga_bloat_factor: float = 1.2,
    marketing_threshold: float = 10.0,
) -> pd.DataFrame:
    """Compute legacy vanity percentage and attach to the feature table.

    Retained for backward compatibility.  New code should use
    ``compute_vanity_score`` instead.

    Parameters
    ----------
    df:
        Daily feature table with derived variables and survival flags.
    exec_comp_threshold:
        Percentage of net income above which exec comp is excessive.
    sga_bloat_factor:
        Multiplier above industry median SGA ratio.
    marketing_threshold:
        Percentage of revenue above which marketing is excessive
        during survival mode.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with legacy vanity columns.
    """
    result = df.copy()

    # Compute legacy components
    result["vanity_exec_comp_excess"] = _exec_comp_excess(
        result, exec_comp_threshold,
    )
    result["vanity_sga_bloat"] = _sga_bloat_v2(result)
    result["vanity_buyback_waste"] = _buyback_waste(result)
    result["vanity_marketing_excess"] = _marketing_excess(
        result, marketing_threshold,
    )

    # Total vanity spend
    components = [
        "vanity_exec_comp_excess",
        "vanity_sga_bloat",
        "vanity_buyback_waste",
        "vanity_marketing_excess",
    ]
    total = pd.Series(0.0, index=result.index)
    any_component = pd.Series(False, index=result.index)
    for comp in components:
        vals = result[comp]
        any_component = any_component | vals.notna()
        total = total + vals.fillna(0)

    result["vanity_total_spend"] = total.where(any_component, other=np.nan)

    revenue = result.get("revenue", pd.Series(np.nan, index=result.index))
    denom_ok = revenue.notna() & (revenue.abs() > EPSILON)

    with np.errstate(divide="ignore", invalid="ignore"):
        vp = (result["vanity_total_spend"] / revenue * 100)
        vp = vp.replace([np.inf, -np.inf], np.nan)

    vp = vp.clip(lower=0, upper=100)
    vp = vp.where(denom_ok, other=np.nan)

    result["vanity_percentage"] = vp
    result["is_missing_vanity_percentage"] = vp.isna().astype(int)

    triggered = (vp.notna() & (vp > 0)).sum()
    logger.info(
        "Vanity percentage (legacy): %d / %d days with non-zero vanity (max=%.1f%%)",
        triggered, len(vp),
        vp.max() if vp.notna().any() else 0.0,
    )

    return result


# ---------------------------------------------------------------------------
# Public API: Vanity Score v2 (new composite)
# ---------------------------------------------------------------------------

def compute_vanity_score(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute vanity score v2 and attach to the feature table.

    This is the primary vanity entry point.  It computes five proxy-based
    components that are reliably available, plus the legacy
    vanity_percentage for backward compatibility.

    Parameters
    ----------
    df:
        Daily feature table with derived variables, financial health
        scores, peer rankings, and survival flags.
    config:
        Optional vanity_v2 configuration dict.  If None, uses defaults.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with:
        - ``vanity_rnd_mismatch`` (0-100)
        - ``vanity_sga_bloat_v2`` (0-100)
        - ``vanity_capital_misallocation`` (0-100)
        - ``vanity_competitive_decay`` (0-100)
        - ``vanity_sentiment_gap`` (0-100)
        - ``vanity_score`` (weighted composite, 0-100)
        - ``vanity_score_21d`` (rolling 21-day mean)
        - ``vanity_trend`` (rising / stable / falling)
        - ``vanity_label`` (Disciplined / Moderate / Wasteful / Reckless)
        - Plus all legacy vanity columns.
    """
    cfg = config or {}
    weights = cfg.get("weights", _DEFAULT_WEIGHTS)
    labels = cfg.get("labels", _DEFAULT_LABELS)
    short_window = cfg.get("trend_window_short", _TREND_WINDOW_SHORT)
    long_window = cfg.get("trend_window_long", _TREND_WINDOW_LONG)

    # First compute legacy vanity for backward compatibility
    result = compute_vanity_percentage(df)

    # Compute v2 components
    result["vanity_rnd_mismatch"] = _rnd_growth_mismatch(result)
    result["vanity_sga_bloat_v2"] = _sga_bloat_v2(result)
    result["vanity_capital_misallocation"] = _capital_misallocation(result)
    result["vanity_competitive_decay"] = _competitive_decay(result)
    result["vanity_sentiment_gap"] = _sentiment_reality_gap(result)

    # Weighted composite score
    v2_components = {
        "vanity_rnd_mismatch": weights.get("rnd_mismatch", 0.15),
        "vanity_sga_bloat_v2": weights.get("sga_bloat", 0.25),
        "vanity_capital_misallocation": weights.get("capital_misallocation", 0.30),
        "vanity_competitive_decay": weights.get("competitive_decay", 0.15),
        "vanity_sentiment_gap": weights.get("sentiment_gap", 0.15),
    }

    composite = pd.Series(0.0, index=result.index)
    total_weight = pd.Series(0.0, index=result.index)
    any_v2 = pd.Series(False, index=result.index)

    for col, w in v2_components.items():
        vals = result[col]
        has_vals = vals.notna()
        any_v2 = any_v2 | has_vals
        composite = composite + (vals.fillna(0) * w)
        total_weight += w * has_vals.astype(float)

    # Re-normalize by actual weight used (to handle missing components)
    with np.errstate(divide="ignore", invalid="ignore"):
        composite = composite / total_weight.replace(0, np.nan)
        composite = composite.replace([np.inf, -np.inf], np.nan)

    composite = _normalize_score(composite)
    result["vanity_score"] = composite.where(any_v2, other=np.nan)

    # Rolling average
    result["vanity_score_21d"] = result["vanity_score"].rolling(
        window=short_window, min_periods=max(1, short_window // 2),
    ).mean()

    # Trend detection
    result["vanity_trend"] = _compute_trend(
        result["vanity_score"], short_window, long_window,
    )

    # Label assignment
    result["vanity_label"] = result["vanity_score"].apply(
        lambda s: _assign_label(s, labels),
    )

    # Logging
    score_col = result["vanity_score"]
    n_available = score_col.notna().sum()
    n_total = len(score_col)
    mean_score = score_col.mean() if n_available > 0 else 0.0
    logger.info(
        "Vanity score v2: %d / %d days scored (mean=%.1f, label=%s)",
        n_available, n_total, mean_score,
        result["vanity_label"].iloc[-1] if n_total > 0 else "N/A",
    )

    return result

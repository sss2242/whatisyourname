"""Shared dataclasses for the Hedge Fund Analysis pipeline.

Every HF module returns a typed result dataclass.  The orchestrator
collects them into ``HedgeFundResult`` which is stored in the profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sf(val: Any) -> float | None:
    """Safe float for JSON serialization (None for NaN/Inf/None)."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _score_label(score: float) -> str:
    """Convert a 0-100 score to a human label."""
    if score >= 80:
        return "excellent"
    if score >= 60:
        return "good"
    if score >= 40:
        return "fair"
    if score >= 20:
        return "weak"
    return "critical"


def _risk_label(score: float) -> str:
    """Convert a 0-100 risk score to a label (higher = worse)."""
    if score >= 70:
        return "high_risk"
    if score >= 40:
        return "elevated"
    if score >= 20:
        return "moderate"
    return "low_risk"


# ---------------------------------------------------------------------------
# Tier 1: Earnings Forensics
# ---------------------------------------------------------------------------

@dataclass
class FCFQualityResult:
    """FCF Quality Score -- measures whether earnings are backed by cash."""

    available: bool = False
    score: float = 50.0             # 0-100, higher = better quality
    label: str = "fair"
    ocf_ni_ratio_8q: float | None = None
    accruals_component: float | None = None
    fcf_trend_slope: float | None = None
    capex_volatility: float | None = None
    degradation_flag: bool = False   # True if score dropped >15pts in 2Q
    narrative: str = ""
    error: str = ""


@dataclass
class AccrualsForensicResult:
    """Accruals forensic analysis -- detects earnings manipulation."""

    available: bool = False
    red_flag_score: float = 50.0    # 0-100, higher = worse
    label: str = "moderate"
    sloan_accruals: float | None = None
    modified_jones_discretionary: float | None = None
    ni_ocf_divergence_trend: float | None = None
    working_capital_anomaly: float | None = None
    cash_conversion_efficiency: float | None = None
    cce_trend: str = "stable"       # improving, stable, declining
    narrative: str = ""
    error: str = ""


@dataclass
class SmoothingResult:
    """Earnings smoothing detection."""

    available: bool = False
    smoothing_index: float = 50.0   # 0-100, higher = more smoothing
    label: str = "moderate"
    beneish_probability: float | None = None
    earnings_vol_ratio: float | None = None  # NI_vol / OCF_vol
    benford_deviation: float | None = None
    sequential_surprise_pattern: float | None = None
    restatement_probability: float | None = None
    narrative: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Tier 2: Cash Flow Stress Test
# ---------------------------------------------------------------------------

@dataclass
class DividendBurnResult:
    """Dividend sustainability risk assessment."""

    available: bool = False
    risk_score: float = 50.0        # 0-100, higher = more risk
    label: str = "moderate"
    dividend_fcf_coverage: float | None = None  # dividends / FCF
    debt_service_coverage: float | None = None
    working_capital_drain: bool = False
    months_to_cut_estimate: float | None = None
    buyback_funded_by_debt: bool = False
    narrative: str = ""
    error: str = ""


@dataclass
class ReturnSpreadResult:
    """CROA vs ROIC spread -- cash returns vs accounting returns."""

    available: bool = False
    croa: float | None = None       # Cash Return on Assets
    roic: float | None = None       # Return on Invested Capital
    spread_bps: float | None = None  # CROA - ROIC in basis points
    spread_trend: str = "stable"    # widening, stable, narrowing
    quality_label: str = "neutral"  # healthy, neutral, concerning
    # Layer 4 enhancement: 5-factor DuPont decomposition (4.5A)
    dupont_tax_burden: float | None = None       # NI / EBT
    dupont_interest_burden: float | None = None   # EBT / EBIT
    dupont_asset_turnover: float | None = None    # Revenue / Total Assets
    dupont_equity_multiplier: float | None = None  # TA / Equity
    dupont_quality_driver: str = ""               # margin / leverage / turnover
    narrative: str = ""
    error: str = ""


@dataclass
class OperatingLeverageResult:
    """Operating and financial leverage analysis."""

    available: bool = False
    dol: float | None = None        # Degree of Operating Leverage
    dfl: float | None = None        # Degree of Financial Leverage
    dtl: float | None = None        # Degree of Total Leverage (DOL * DFL)
    margin_elasticity: float | None = None  # slope of margin vs revenue
    earnings_sensitivity: str = "moderate"  # low, moderate, high, extreme
    narrative: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Tier 3: Balance Sheet Risk
# ---------------------------------------------------------------------------

@dataclass
class OBSRiskResult:
    """Off-balance-sheet risk assessment."""

    available: bool = False
    risk_score: float = 50.0        # 0-100, higher = more hidden risk
    label: str = "moderate"
    goodwill_to_assets: float | None = None
    intangibles_to_assets: float | None = None
    sga_anomaly_score: float | None = None
    keyword_matches: int = 0        # OBS keywords found in filings
    narrative: str = ""
    error: str = ""


@dataclass
class AssetQualityResult:
    """Asset quality deterioration detection."""

    available: bool = False
    deterioration_score: float = 50.0  # 0-100, higher = worse
    label: str = "stable"
    dso: float | None = None        # Days Sales Outstanding
    dso_change_pct: float | None = None
    inventory_days: float | None = None
    inventory_days_change_pct: float | None = None
    capitalization_ratio: float | None = None
    capitalization_change_pct: float | None = None
    narrative: str = ""
    error: str = ""


@dataclass
class StressScenario:
    """Single leverage stress scenario output."""

    name: str = ""
    revenue_shock_pct: float = 0.0
    margin_shock_bps: float = 0.0
    rate_shock_bps: float = 0.0
    debt_to_ebitda: float | None = None
    interest_coverage: float | None = None
    cash_runway_months: float | None = None
    covenant_breach: bool = False


@dataclass
class LeverageStressResult:
    """3-scenario leverage stress test."""

    available: bool = False
    base_case: StressScenario = field(default_factory=StressScenario)
    revenue_miss: StressScenario = field(default_factory=StressScenario)
    systemic_crisis: StressScenario = field(default_factory=StressScenario)
    refinancing_risk: bool = False
    # Multi-frequency merge metadata
    freq_results: dict = field(default_factory=dict)   # per-freq results {"A": {...}, "Q": {...}}
    merge_method: str = ""                             # "worst_case_envelope" or "single_freq"
    freq_divergence: float = 0.0                       # do frequencies agree on risk?
    narrative: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Tier 4: Inflection Detection
# ---------------------------------------------------------------------------

@dataclass
class MomentumCompositeResult:
    """Multi-dimensional fundamental momentum."""

    available: bool = False
    score: float = 50.0             # 0-100
    label: str = "neutral"
    revenue_acceleration: float | None = None
    margin_trend_slope: float | None = None
    fcf_conversion_trend: float | None = None
    roic_trajectory: float | None = None
    price_momentum_divergence: float | None = None  # fundamental vs price
    price_momentum_63d: float | None = None          # 63-day price return
    price_momentum_252d: float | None = None         # 252-day price return
    price_fundamental_divergence: bool = False        # True when price and fundamentals disagree
    divergence_direction: str = ""                    # "price_leading" or "fundamentals_leading"
    inflection_detected: bool = False
    # Layer 4 enhancement: regime-conditional momentum (4.10A)
    regime_adjusted_momentum: float | None = None
    momentum_regime_bias: str = ""  # which component drives score in this regime
    # Multi-frequency merge metadata
    freq_scores: dict = field(default_factory=dict)    # per-freq scores {"A": 45, "Q": 62}
    merge_method: str = ""                             # "weighted_regression" or "single_freq"
    freq_agreement: float = 1.0                        # do frequencies agree on direction? (0-1)
    narrative: str = ""
    error: str = ""


@dataclass
class GrowthQualityResult:
    """Growth decomposition -- organic vs inorganic."""

    available: bool = False
    score: float = 50.0             # 0-100
    label: str = "mixed"
    organic_growth_pct: float | None = None
    inorganic_growth_pct: float | None = None  # M&A-driven
    organic_fraction: float | None = None  # organic / total
    margin_adjusted_growth: float | None = None
    incremental_roic: float | None = None  # ROIC on new capital
    narrative: str = ""
    error: str = ""


@dataclass
class SurpriseResult:
    """Earnings surprise probability estimation."""

    available: bool = False
    p_beat: float = 0.33            # probability of beating
    p_miss: float = 0.33            # probability of missing
    p_inline: float = 0.34          # probability of inline
    expected_direction: str = "neutral"  # beat, miss, neutral
    magnitude_estimate_pct: float | None = None
    days_to_next_filing: int | None = None
    narrative: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Tier 5: Valuation Engine
# ---------------------------------------------------------------------------

@dataclass
class DCFResult:
    """Probabilistic DCF valuation."""

    available: bool = False
    intrinsic_p10: float | None = None
    intrinsic_p25: float | None = None
    intrinsic_p50: float | None = None  # median intrinsic value
    intrinsic_p75: float | None = None
    intrinsic_p90: float | None = None
    current_price: float | None = None
    upside_pct: float | None = None     # (p50 - current) / current
    risk_reward_ratio: float | None = None  # upside / downside
    n_simulations: int = 0
    implied_growth_rate: float | None = None  # Gap 5: reverse DCF implied growth
    reliable: bool | None = None              # Gap 5: sanity gate (DCF/price ratio check)
    warning: str = ""                         # Gap 5: sanity gate warning message
    # Layer 4 enhancement: market-implied growth + RIV (4.13A/B)
    growth_gap: float | None = None           # implied minus actual growth
    priced_for_perfection_flag: bool = False   # growth_gap > 2x actual
    riv_intrinsic: float | None = None        # residual income valuation
    riv_excess_return: float | None = None    # RI / book value
    riv_vs_dcf_divergence: float | None = None  # riv - dcf difference
    # Multi-frequency merge metadata
    freq_intrinsics: dict = field(default_factory=dict)  # per-freq P50 {"A": 185.0, "Q": 192.0}
    merge_method: str = ""                               # "inverse_variance" or "single_freq"
    freq_divergence: float = 0.0                         # how much do frequencies disagree?
    narrative: str = ""
    error: str = ""


@dataclass
class ValuationQualityResult:
    """Valuation vs quality matrix positioning."""

    available: bool = False
    quality_score: float = 50.0     # 0-100, composite of Tiers 1-3
    valuation_percentile: float | None = None  # vs sector peers
    quadrant: str = "fair"          # undervalued, fair, overvalued, value_trap
    mispricing_pct: float | None = None
    narrative: str = ""
    error: str = ""


@dataclass
class PEGCompositeResult:
    """Quality-adjusted PEG + FCF yield spread."""

    available: bool = False
    peg_raw: float | None = None
    peg_adjusted: float | None = None   # adjusted by quality multiplier
    fcf_yield_pct: float | None = None
    debt_cost_pct: float | None = None
    fcf_spread_bps: float | None = None  # FCF yield - debt cost
    cheap_flag: bool = False
    narrative: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Integration Layer
# ---------------------------------------------------------------------------

@dataclass
class TierScore:
    """Score for one of the 5 HF tiers."""

    tier_name: str = ""
    score: float = 50.0             # 0-100
    label: str = "fair"
    trend: str = "stable"           # improving, stable, degrading
    risk_flag: bool = False         # True if score < 30
    opportunity_flag: bool = False  # True if score > 70 and valuation cheap
    key_narrative: str = ""
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class ThesisScorecard:
    """5-tier Investment Thesis Scorecard."""

    available: bool = False
    earnings_quality: TierScore = field(default_factory=lambda: TierScore(tier_name="earnings_quality"))
    cash_flow: TierScore = field(default_factory=lambda: TierScore(tier_name="cash_flow"))
    balance_sheet: TierScore = field(default_factory=lambda: TierScore(tier_name="balance_sheet"))
    inflection: TierScore = field(default_factory=lambda: TierScore(tier_name="inflection"))
    valuation: TierScore = field(default_factory=lambda: TierScore(tier_name="valuation"))
    investment_grade: str = "C"     # A+ to D
    conviction: int = 5             # 0-10
    catalyst: str = ""              # next expected event
    risk_reward: str = "balanced"   # favorable, balanced, unfavorable
    data_sufficiency: float = 1.0   # fraction of metrics computed from real data (0-1)
    n_defaulted: int = 0            # number of metrics that hit defaults (score=50)


@dataclass
class PositionSignalResult:
    """Actionable position signal with sizing guidance."""

    available: bool = False
    signal: float = 0.0             # -1 (strong sell) to +1 (strong buy)
    label: str = "hold"             # strong_buy, buy, hold, sell, strong_sell
    conviction: float = 0.0         # 0-1
    alpha_base: float = 0.0
    quality_multiplier: float = 1.0
    survival_multiplier: float = 1.0
    decay_multiplier: float = 1.0
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    risk_reward_ratio: float | None = None
    # Momentum clamp: prevents fighting strong price trends
    momentum_override: bool = False
    momentum_override_reason: str = ""
    # Layer 4 enhancement: Kelly criterion sizing (4.17A)
    kelly_fraction: float | None = None       # full Kelly optimal bet size
    half_kelly_size: float | None = None      # conservative half-Kelly
    kelly_edge: float | None = None           # p*b - q (is there an edge?)
    narrative: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Master Result
# ---------------------------------------------------------------------------

@dataclass
class HedgeFundResult:
    """Combined result from the entire HF analysis pipeline."""

    available: bool = False

    # Tier 1: Earnings Forensics
    fcf_quality: FCFQualityResult = field(default_factory=FCFQualityResult)
    accruals_forensic: AccrualsForensicResult = field(default_factory=AccrualsForensicResult)
    smoothing: SmoothingResult = field(default_factory=SmoothingResult)

    # Tier 2: Cash Flow Stress Test
    dividend_burn: DividendBurnResult = field(default_factory=DividendBurnResult)
    return_spread: ReturnSpreadResult = field(default_factory=ReturnSpreadResult)
    operating_leverage: OperatingLeverageResult = field(default_factory=OperatingLeverageResult)

    # Tier 3: Balance Sheet Risk
    obs_risk: OBSRiskResult = field(default_factory=OBSRiskResult)
    asset_quality: AssetQualityResult = field(default_factory=AssetQualityResult)
    leverage_stress: LeverageStressResult = field(default_factory=LeverageStressResult)

    # Tier 4: Inflection Detection
    momentum: MomentumCompositeResult = field(default_factory=MomentumCompositeResult)
    growth_quality: GrowthQualityResult = field(default_factory=GrowthQualityResult)
    earnings_surprise: SurpriseResult = field(default_factory=SurpriseResult)

    # Tier 5: Valuation Engine
    dcf: DCFResult = field(default_factory=DCFResult)
    valuation_quality: ValuationQualityResult = field(default_factory=ValuationQualityResult)
    peg_composite: PEGCompositeResult = field(default_factory=PEGCompositeResult)

    # Integration
    scorecard: ThesisScorecard = field(default_factory=ThesisScorecard)
    position: PositionSignalResult = field(default_factory=PositionSignalResult)

    # Advanced methods (15 additional techniques)
    advanced: dict[str, Any] = field(default_factory=dict)

    # Cross-pipeline fusion (8 methods combining HF + multi-freq)
    fusion: dict[str, Any] = field(default_factory=dict)

    # Data readiness assessment (computed before HF runs)
    data_readiness: dict[str, Any] = field(default_factory=dict)

    def to_profile_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for profile storage."""
        from dataclasses import asdict

        def _clean(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(i) for i in obj]
            if isinstance(obj, float):
                return _sf(obj)
            return obj

        raw = asdict(self)
        return _clean(raw)

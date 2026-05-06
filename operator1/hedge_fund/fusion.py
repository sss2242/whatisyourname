"""Cross-Pipeline Insight Fusion -- 8 methods for combining HF + Multi-Frequency results.

Transforms 80+ raw analytical outputs into a single coherent action
recommendation with conviction, timing, and risk context.

Methods:
  1. Anomaly-Driven Priority Routing (override on critical signals)
  2. Cross-Frequency Disagreement Signals (meta-signal from agreement)
  3. Regime-Conditioned Signal Weighting (per-regime IC table)
  4. Meta-Ensemble Stacking (combine both position signals)
  5. Catalyst-Driven Temporal Weighting (event-proximity modulation)
  6. Bottom-Up Early Warning Propagation (fast -> slow override)
  7. Multi-Frequency IC Attribution (data-driven freq assignment)
  8. Bayesian Belief Network (simplified probabilistic inference)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from operator1.hedge_fund.helpers import normalize_score, safe_divide

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """Output of the cross-pipeline fusion layer."""

    available: bool = False

    # Final fused signal (-1 to +1)
    fused_signal: float = 0.0
    fused_label: str = "hold"          # strong_buy/buy/hold/sell/strong_sell
    fused_conviction: float = 0.0      # 0-1

    # Method outputs
    anomaly_overrides: list[dict] = field(default_factory=list)
    anomaly_frozen: bool = False       # True if anomaly override freezes signal
    freq_disagreement_score: float = 0.0   # 0-1 (0 = full agreement)
    regime_adjusted_weights: dict = field(default_factory=dict)
    meta_ensemble_agreement: float = 0.0   # -1 to +1 (positive = agree)
    catalyst_proximity_days: int | None = None
    catalyst_weight_regime: str = "normal"  # pre_catalyst/catalyst_zone/post_catalyst/normal
    early_warnings: list[str] = field(default_factory=list)
    freq_ic_attribution: dict = field(default_factory=dict)
    belief_network_posterior: float = 0.0  # 0-1 posterior probability of positive return

    # Phase 9: HRP signal combination (Lopez de Prado 2016)
    hrp_weights: dict = field(default_factory=dict)
    signal_clustering: dict = field(default_factory=dict)
    # Phase 9: Brier calibration (Brier 1950)
    calibrated_conviction: float | None = None
    overconfidence_flag: bool = False
    calibration_reliability: float | None = None

    # Action summary
    action: str = ""                   # human-readable action sentence
    primary_risk: str = ""             # top risk factor
    next_catalyst: str = ""            # next expected event


# ---------------------------------------------------------------------------
# Method 1: Anomaly-Driven Priority Routing
# ---------------------------------------------------------------------------

_ANOMALY_RULES = [
    # (check_key, check_path, threshold, comparison, override_action, severity)
    ("covenant_breach", "leverage_stress.base_case.covenant_breach", True, "eq", "set_sell", "critical"),
    ("covenant_breach_stress", "leverage_stress.revenue_miss.covenant_breach", True, "eq", "cap_hold", "high"),
    ("forensic_cf_flags", "forensic_cashflow.risk_score", 60, "gt", "freeze_signal", "high"),
    ("torpedo_risk", "earnings_torpedo.torpedo_risk", 75, "gt", "cap_hold", "high"),
    ("vol_inverted", "garch_vol_term_structure.inverted", True, "eq", "widen_stops", "medium"),
    ("negative_ebitda", "leverage_stress.base_case.debt_to_ebitda", None, "is_none_breach", "set_sell", "critical"),
    ("dividend_burn_extreme", "dividend_burn.risk_score", 85, "gt", "flag_risk", "medium"),
    ("fcf_quality_critical", "fcf_quality.score", 20, "lt", "flag_risk", "medium"),
]


def _get_nested(data: dict, path: str) -> Any:
    """Navigate nested dicts/objects via dot-separated path."""
    current = data
    for key in path.split("."):
        if isinstance(current, dict):
            current = current.get(key)
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return None
        if current is None:
            return None
    return current


def compute_anomaly_routing(hf_profile: dict) -> tuple[list[dict], bool]:
    """Check all anomaly rules and return overrides + freeze flag."""
    overrides = []
    frozen = False

    for rule_name, path, threshold, comparison, action, severity in _ANOMALY_RULES:
        value = _get_nested(hf_profile, path)
        triggered = False

        if comparison == "eq" and value == threshold:
            triggered = True
        elif comparison == "gt" and value is not None and isinstance(value, (int, float)) and value > threshold:
            triggered = True
        elif comparison == "lt" and value is not None and isinstance(value, (int, float)) and value < threshold:
            triggered = True
        elif comparison == "is_none_breach" and value is None:
            # debt_to_ebitda = None means negative EBITDA = breach
            breach = _get_nested(hf_profile, "leverage_stress.base_case.covenant_breach")
            if breach:
                triggered = True

        if triggered:
            overrides.append({
                "rule": rule_name,
                "action": action,
                "severity": severity,
                "value": str(value)[:50] if value is not None else "N/A",
            })
            if action == "freeze_signal":
                frozen = True

    return overrides, frozen


# ---------------------------------------------------------------------------
# Method 2: Cross-Frequency Disagreement Signals
# ---------------------------------------------------------------------------

def compute_freq_disagreement(
    multi_frequency_result: Any = None,
    hf_profile: dict | None = None,
    filing_age_days: int = 0,
) -> float:
    """Compute a 0-1 disagreement score across frequencies.

    0 = all frequencies agree, 1 = maximum disagreement.
    """
    if multi_frequency_result is None:
        return 0.0

    try:
        # Get regime consensus
        rc = None
        if hasattr(multi_frequency_result, "regime_consensus"):
            rc = multi_frequency_result.regime_consensus
        elif isinstance(multi_frequency_result, dict):
            rc = multi_frequency_result.get("regime_consensus")

        if rc is None:
            return 0.0

        agreement = getattr(rc, "agreement_ratio", None)
        if isinstance(rc, dict):
            agreement = rc.get("agreement_ratio")

        if agreement is not None:
            disagreement = 1.0 - float(agreement)

            # Filing staleness amplifier: stale data = more uncertainty
            if filing_age_days > 60:
                staleness_boost = min(0.3, (filing_age_days - 60) / 300)
                disagreement = min(1.0, disagreement + staleness_boost)

            return round(disagreement, 4)
    except Exception as exc:
        logger.debug("Freq disagreement failed: %s", exc)

    return 0.0


# ---------------------------------------------------------------------------
# Method 3: Regime-Conditioned Signal Weighting
# ---------------------------------------------------------------------------

# Default IC estimates per metric per regime (empirical priors)
_REGIME_IC_TABLE: dict[str, dict[str, float]] = {
    "fcf_quality":        {"bull": 0.06, "bear": 0.12, "high_vol": 0.05, "low_vol": 0.08, "survival": 0.15},
    "accruals_forensic":  {"bull": 0.04, "bear": 0.09, "high_vol": 0.03, "low_vol": 0.05, "survival": 0.11},
    "smoothing":          {"bull": 0.03, "bear": 0.08, "high_vol": 0.04, "low_vol": 0.04, "survival": 0.09},
    "dividend_burn":      {"bull": 0.02, "bear": 0.10, "high_vol": 0.06, "low_vol": 0.03, "survival": 0.13},
    "return_spread":      {"bull": 0.05, "bear": 0.07, "high_vol": 0.04, "low_vol": 0.06, "survival": 0.08},
    "operating_leverage":  {"bull": 0.07, "bear": 0.11, "high_vol": 0.09, "low_vol": 0.05, "survival": 0.06},
    "obs_risk":           {"bull": 0.02, "bear": 0.06, "high_vol": 0.04, "low_vol": 0.03, "survival": 0.08},
    "asset_quality":      {"bull": 0.03, "bear": 0.08, "high_vol": 0.05, "low_vol": 0.04, "survival": 0.10},
    "leverage_stress":    {"bull": 0.02, "bear": 0.12, "high_vol": 0.08, "low_vol": 0.03, "survival": 0.15},
    "momentum":           {"bull": 0.11, "bear": 0.03, "high_vol": 0.02, "low_vol": 0.10, "survival": 0.01},
    "growth_quality":     {"bull": 0.08, "bear": 0.05, "high_vol": 0.03, "low_vol": 0.07, "survival": 0.02},
    "earnings_surprise":  {"bull": 0.09, "bear": 0.06, "high_vol": 0.04, "low_vol": 0.08, "survival": 0.03},
    "dcf":                {"bull": 0.05, "bear": 0.09, "high_vol": 0.04, "low_vol": 0.07, "survival": 0.06},
    "valuation_quality":  {"bull": 0.06, "bear": 0.10, "high_vol": 0.05, "low_vol": 0.08, "survival": 0.07},
    "peg_composite":      {"bull": 0.04, "bear": 0.07, "high_vol": 0.03, "low_vol": 0.06, "survival": 0.04},
}


def compute_regime_weights(current_regime: str) -> dict[str, float]:
    """Return per-metric weights adjusted for the current regime.

    Regime labels: bull, bear, high_vol, low_vol, survival, unknown.
    """
    regime_key = current_regime.lower().replace(" ", "_")
    # Map regime labels to our table keys
    regime_map = {
        "bull": "bull", "bear": "bear", "high_vol": "high_vol",
        "high_volatility": "high_vol", "low_vol": "low_vol",
        "low_volatility": "low_vol", "company_survival": "survival",
        "extreme_survival": "survival", "modified_survival": "survival",
        "normal": "low_vol",  # normal ~ low_vol regime
    }
    mapped = regime_map.get(regime_key, "low_vol")

    weights = {}
    total = 0.0
    for metric, ic_by_regime in _REGIME_IC_TABLE.items():
        ic = ic_by_regime.get(mapped, 0.05)
        weights[metric] = ic
        total += ic

    # Normalize to sum to 1
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


# ---------------------------------------------------------------------------
# Method 4: Meta-Ensemble Stacking
# ---------------------------------------------------------------------------

def compute_meta_ensemble(
    main_signal: float,
    hf_signal: float,
    main_ic: float = 0.05,
    hf_ic: float = 0.05,
) -> tuple[float, float]:
    """Stack the main pipeline signal with HF signal.

    Returns (fused_signal, agreement_score).
    Agreement: positive when both signals point the same direction.
    """
    # IC-weighted combination
    total_ic = abs(main_ic) + abs(hf_ic)
    if total_ic < 1e-8:
        fused = (main_signal + hf_signal) / 2
    else:
        w_main = abs(main_ic) / total_ic
        w_hf = abs(hf_ic) / total_ic
        fused = w_main * main_signal + w_hf * hf_signal

    # Agreement score: +1 when both agree, -1 when they disagree
    if abs(main_signal) < 0.05 or abs(hf_signal) < 0.05:
        agreement = 0.0
    else:
        agreement = np.sign(main_signal) * np.sign(hf_signal)
        # Scale by confidence
        agreement *= min(abs(main_signal), abs(hf_signal))

    # When disagreeing, reduce conviction
    if agreement < 0:
        fused *= 0.5  # halve signal on disagreement

    return (max(-1.0, min(1.0, fused)), float(agreement))


# ---------------------------------------------------------------------------
# Method 5: Catalyst-Driven Temporal Weighting
# ---------------------------------------------------------------------------

def compute_catalyst_weight_regime(
    days_to_catalyst: int | None,
    days_since_last_catalyst: int | None = None,
) -> str:
    """Classify the current position in the catalyst cycle.

    Returns regime label that modulates which HF tiers get priority.
    """
    if days_to_catalyst is not None:
        if days_to_catalyst <= 10:
            return "catalyst_zone"     # imminent: vol + positioning dominate
        elif days_to_catalyst <= 30:
            return "pre_catalyst"      # approaching: momentum + drift dominate
    if days_since_last_catalyst is not None:
        if days_since_last_catalyst <= 5:
            return "post_catalyst"     # PEAD + surprise reaction dominate
    return "normal"                    # fundamentals dominate


_CATALYST_TIER_WEIGHTS = {
    "normal":        {"T1": 0.25, "T2": 0.20, "T3": 0.20, "T4": 0.20, "T5": 0.15},
    "pre_catalyst":  {"T1": 0.15, "T2": 0.15, "T3": 0.15, "T4": 0.35, "T5": 0.20},
    "catalyst_zone": {"T1": 0.10, "T2": 0.10, "T3": 0.10, "T4": 0.30, "T5": 0.40},
    "post_catalyst": {"T1": 0.10, "T2": 0.15, "T3": 0.10, "T4": 0.45, "T5": 0.20},
}


def get_catalyst_tier_weights(regime: str) -> dict[str, float]:
    """Return tier weights modulated by catalyst proximity."""
    return _CATALYST_TIER_WEIGHTS.get(regime, _CATALYST_TIER_WEIGHTS["normal"])


# ---------------------------------------------------------------------------
# Method 6: Bottom-Up Early Warning Propagation
# ---------------------------------------------------------------------------

def compute_early_warnings(
    hf_profile: dict,
    cache_latest: dict | None = None,
) -> list[str]:
    """Check fast signals that should override slow assessments.

    Returns list of warning strings for signals that propagate bottom-up.
    """
    warnings = []

    # Daily/weekly signals that should override quarterly
    adv = hf_profile.get("advanced", {})

    # Vol term structure inverted (daily signal)
    vol_ts = adv.get("garch_vol_term_structure", {})
    if vol_ts.get("inverted"):
        warnings.append("Vol term structure INVERTED -- large move imminent (daily signal overrides weekly calm)")

    # OU reversion break (daily signal)
    ou = adv.get("ou_mean_reversion", {})
    if ou.get("available") and ou.get("mean_reverting"):
        dev = ou.get("current_deviation_pct", 0)
        hl = ou.get("half_life_days", 999)
        if abs(dev) > 15 and hl < 10:
            warnings.append(f"OU reversion: {dev:+.1f}% deviation, half-life {hl:.0f}d -- snap-back expected")

    # VRP tail event (daily signal)
    vrp = adv.get("vrp_proxy", {})
    if vrp.get("regime") == "tail_event":
        warnings.append("VRP: realized vol exceeding forecast -- tail event active")

    # Torpedo risk (weekly signal)
    torpedo = adv.get("earnings_torpedo", {})
    if torpedo.get("torpedo_risk", 0) > 50:
        warnings.append(f"Earnings torpedo: {torpedo['torpedo_risk']}% risk -- high PE + deceleration")

    # Forensic CF flags (quarterly signal but high urgency)
    forensic = hf_profile.get("forensic_cashflow", adv.get("forensic_cashflow", {}))
    if isinstance(forensic, dict) and forensic.get("risk_score", 0) > 50:
        n_flags = len(forensic.get("flags", []))
        warnings.append(f"Cash flow forensics: {n_flags} manipulation indicators detected")

    return warnings


# ---------------------------------------------------------------------------
# Method 7: Multi-Frequency IC Attribution
# ---------------------------------------------------------------------------

# Default natural frequency for each HF metric (empirical assignment)
_METRIC_NATURAL_FREQ: dict[str, str] = {
    "fcf_quality": "Q",
    "accruals_forensic": "Q",
    "smoothing": "A",
    "dividend_burn": "Q",
    "return_spread": "Q",
    "operating_leverage": "Q",
    "obs_risk": "A",
    "asset_quality": "Q",
    "leverage_stress": "Q",
    "momentum": "M",
    "growth_quality": "Q",
    "earnings_surprise": "Q",
    "dcf": "A",
    "valuation_quality": "Q",
    "peg_composite": "Q",
}


def compute_freq_ic_attribution(
    signal_ic_result: Any = None,
) -> dict[str, dict[str, float]]:
    """For each HF metric, estimate which frequency provides highest IC.

    Returns: {metric_name: {freq: ic_estimate}}.
    Falls back to hardcoded natural frequencies when IC data unavailable.
    """
    attribution = {}

    # If signal_ic is available, use actual IC data
    if signal_ic_result is not None and getattr(signal_ic_result, "available", False):
        ic_scores = getattr(signal_ic_result, "ic_scores", {})
        if isinstance(ic_scores, dict):
            for metric in _METRIC_NATURAL_FREQ:
                # Check if signal_ic has IC for related cache columns
                # Map HF metrics to cache column names they derive from
                metric_to_cols = {
                    "fcf_quality": ["accruals", "operating_cash_flow_ttm"],
                    "momentum": ["return_5d", "return_21d"],
                    "earnings_surprise": ["sue_score", "pead_signal"],
                }
                related_cols = metric_to_cols.get(metric, [])
                max_ic = 0.0
                for col in related_cols:
                    ic = ic_scores.get(col, {})
                    if isinstance(ic, (int, float)):
                        max_ic = max(max_ic, abs(float(ic)))
                if max_ic > 0:
                    attribution[metric] = {"data_ic": round(max_ic, 4)}

    # Fill defaults for metrics without data-driven IC
    for metric, nat_freq in _METRIC_NATURAL_FREQ.items():
        if metric not in attribution:
            attribution[metric] = {"natural_freq": nat_freq, "data_ic": None}

    return attribution


# ---------------------------------------------------------------------------
# Method 8: Simplified Bayesian Belief Network
# ---------------------------------------------------------------------------

def compute_belief_posterior(
    hf_profile: dict,
    regime: str = "normal",
    freq_disagreement: float = 0.0,
) -> float:
    """Simplified Bayesian posterior for positive forward return.

    Uses a naive Bayes approximation with conditional independence.
    P(return > 0 | signals) proportional to product of likelihoods.
    """
    try:
        # Prior based on regime
        regime_priors = {
            "bull": 0.60, "bear": 0.35, "high_vol": 0.45,
            "low_vol": 0.55, "survival": 0.25, "normal": 0.52,
        }
        regime_key = regime.lower().replace(" ", "_")
        prior = regime_priors.get(regime_key, 0.50)

        # Likelihood contributions from HF signals
        # Each signal nudges the probability up or down
        log_lr = 0.0  # log likelihood ratio

        # Scorecard grade
        sc = hf_profile.get("scorecard", {})
        grade = sc.get("investment_grade", "C")
        grade_lr = {"A+": 0.8, "A": 0.6, "B+": 0.3, "B": 0.1,
                     "C+": -0.05, "C": -0.15, "D": -0.5, "F": -0.8}
        log_lr += grade_lr.get(grade, 0.0)

        # Position signal direction
        pos = hf_profile.get("position", {})
        sig = pos.get("signal", 0)
        log_lr += sig * 2.0  # scale position signal as evidence

        # Momentum
        mom = hf_profile.get("momentum", {})
        mom_score = mom.get("score", 50)
        log_lr += (mom_score - 50) / 100  # positive momentum = positive evidence

        # Valuation
        vq = hf_profile.get("valuation_quality", {})
        quadrant = vq.get("quadrant", "fair")
        quad_lr = {"undervalued": 0.4, "fair": 0.0, "overvalued": -0.3, "value_trap": -0.5}
        log_lr += quad_lr.get(quadrant, 0.0)

        # Frequency disagreement reduces confidence toward 0.5
        uncertainty_penalty = freq_disagreement * 0.5  # max 0.5 penalty

        # Apply Bayes: posterior = prior * LR / (prior * LR + (1-prior))
        lr = np.exp(log_lr)
        posterior = (prior * lr) / (prior * lr + (1 - prior))

        # Apply uncertainty
        posterior = posterior * (1 - uncertainty_penalty) + 0.5 * uncertainty_penalty

        return round(max(0.01, min(0.99, float(posterior))), 4)

    except Exception as exc:
        logger.debug("Belief posterior failed: %s", exc)
        return 0.50


# ---------------------------------------------------------------------------
# Master fusion function
# ---------------------------------------------------------------------------

def run_fusion(
    hf_profile: dict,
    main_position_signal: float = 0.0,
    multi_frequency_result: Any = None,
    signal_ic_result: Any = None,
    filing_calendar_result: Any = None,
    survival_controller: Any = None,
    cache: Any = None,
) -> FusionResult:
    """Run all 8 fusion methods and produce the final fused output.

    Parameters
    ----------
    hf_profile:
        Dict from HedgeFundResult.to_profile_dict().
    main_position_signal:
        Position signal from the main pipeline (-1 to +1).
    multi_frequency_result:
        Result from multi-frequency runner.
    signal_ic_result:
        Result from signal IC module.
    filing_calendar_result:
        Result from filing calendar module.
    survival_controller:
        USS controller for regime info.
    """
    result = FusionResult()

    try:
        hf_signal = hf_profile.get("position", {}).get("signal", 0.0)

        # Method 1: Anomaly routing
        result.anomaly_overrides, result.anomaly_frozen = compute_anomaly_routing(hf_profile)
        if result.anomaly_overrides:
            logger.info("Fusion: %d anomaly overrides detected", len(result.anomaly_overrides))

        # Method 2: Cross-frequency disagreement
        filing_age = 0
        if filing_calendar_result is not None:
            filing_age = getattr(filing_calendar_result, "latest_filing_age_days", 0)
        result.freq_disagreement_score = compute_freq_disagreement(
            multi_frequency_result, hf_profile, filing_age,
        )

        # Method 3: Regime-conditioned weighting
        current_regime = "normal"
        if survival_controller is not None:
            current_regime = getattr(survival_controller, "current_regime", "normal")
        result.regime_adjusted_weights = compute_regime_weights(current_regime)

        # Method 4: Meta-ensemble
        main_ic = 0.05
        hf_ic = 0.05
        if signal_ic_result is not None and getattr(signal_ic_result, "available", False):
            main_ic = abs(getattr(signal_ic_result, "best_ic", 0.05))
            hf_ic = max(0.01, main_ic * 0.8)  # HF IC estimated as ~80% of best signal IC

        fused_signal, agreement = compute_meta_ensemble(
            main_position_signal, hf_signal, main_ic, hf_ic,
        )
        result.meta_ensemble_agreement = agreement

        # Method 5: Catalyst timing
        days_to = None
        days_since = None
        if filing_calendar_result is not None:
            nf = getattr(filing_calendar_result, "next_expected_filing", None)
            if isinstance(nf, dict) and nf.get("available"):
                days_to = nf.get("days_until")
            days_since = filing_age  # rough proxy

        result.catalyst_proximity_days = days_to
        result.catalyst_weight_regime = compute_catalyst_weight_regime(days_to, days_since)
        catalyst_weights = get_catalyst_tier_weights(result.catalyst_weight_regime)

        # Method 6: Early warnings
        result.early_warnings = compute_early_warnings(hf_profile)

        # Method 7: Frequency IC attribution
        result.freq_ic_attribution = compute_freq_ic_attribution(signal_ic_result)

        # Method 8: Bayesian posterior
        result.belief_network_posterior = compute_belief_posterior(
            hf_profile, current_regime, result.freq_disagreement_score,
        )

        # --- Final signal synthesis ---

        # Start with meta-ensemble signal
        final = fused_signal

        # Apply Bayesian posterior as conviction modifier
        bayesian_conviction = abs(result.belief_network_posterior - 0.5) * 2  # 0-1 range
        final *= (0.5 + 0.5 * bayesian_conviction)  # scale signal by Bayesian confidence

        # Apply frequency disagreement penalty
        final *= (1.0 - result.freq_disagreement_score * 0.5)

        # Apply anomaly overrides
        for override in result.anomaly_overrides:
            action = override["action"]
            if action == "set_sell" and final > -0.3:
                final = -0.5
            elif action == "cap_hold" and final > 0.2:
                final = 0.0
            elif action == "freeze_signal":
                final = 0.0
                result.anomaly_frozen = True
            elif action == "widen_stops":
                pass  # doesn't change signal, affects risk management

        result.fused_signal = max(-1.0, min(1.0, final))

        # Label
        if result.fused_signal > 0.5:
            result.fused_label = "strong_buy"
        elif result.fused_signal > 0.2:
            result.fused_label = "buy"
        elif result.fused_signal > -0.2:
            result.fused_label = "hold"
        elif result.fused_signal > -0.5:
            result.fused_label = "sell"
        else:
            result.fused_label = "strong_sell"

        # Conviction (0-1): driven by agreement + Bayesian + IC
        result.fused_conviction = min(1.0, max(0.0,
            0.3 * bayesian_conviction
            + 0.3 * (1.0 - result.freq_disagreement_score)
            + 0.2 * abs(agreement)
            + 0.2 * abs(final)
        ))

        # Action sentence
        sc = hf_profile.get("scorecard", {})
        grade = sc.get("investment_grade", "?")
        result.action = (
            f"{result.fused_label.replace('_', ' ').upper()} "
            f"(signal {result.fused_signal:+.2f}, "
            f"conviction {result.fused_conviction:.0%}, "
            f"grade {grade})"
        )

        # Primary risk
        if result.anomaly_overrides:
            result.primary_risk = result.anomaly_overrides[0].get("rule", "anomaly detected")
        elif result.early_warnings:
            result.primary_risk = result.early_warnings[0][:80]
        else:
            result.primary_risk = "no critical risks detected"

        # Next catalyst
        if result.catalyst_proximity_days is not None:
            result.next_catalyst = f"Filing expected in ~{result.catalyst_proximity_days} days"
        else:
            result.next_catalyst = "No upcoming catalyst detected"

        result.available = True

    except Exception as exc:
        logger.warning("Fusion failed: %s", exc)
        result.fused_signal = 0.0
        result.fused_label = "hold"

    return result

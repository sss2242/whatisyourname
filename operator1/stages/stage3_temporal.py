"""Stage 3: Temporal Analysis -- regime detection, causality, patterns, synergies.

Sub-stages:
  3.1  Regime detection (HMM + GMM + PELT + BCP + ChangeFinder)
  3.2  Dual regime mixer
  3.3  Granger causality (PCMCI) + feature pruning
  3.4  Transfer entropy
  3.5  Cycle decomposition (EMD/FFT)
  3.6  Pattern detection (candlestick + Matrix Profile)
  3.7  Economic planes + pre-forecasting synergies
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operator1.pipeline_state import PipelineState

logger = logging.getLogger("operator1.stages.stage3")


def _init_extra_vars(state: PipelineState) -> None:
    """Build the extra_vars list from cache columns.

    Mirrors main.py Step 6 _extra_vars construction.
    """
    cache = state.cache
    if cache is None or cache.empty:
        state.extra_vars = []
        return

    _linked_prefixes = (
        "competitors_", "suppliers_", "customers_",
        "financial_institutions_", "sector_peers_", "industry_peers_",
        "rel_", "valuation_premium_",
    )
    _hmm_lookahead_cols = {
        "survival_intensity",
        "regime_confidence",
        "regime_transition_prob",
    }
    state.extra_vars = [
        c for c in cache.columns
        if (c.startswith("fh_") or c.startswith("sentiment_")
            or c.startswith("peer_") or c.startswith("macro_")
            or c.startswith("inst_")
            or c.startswith("buying_power_") or c.startswith("catalyst_")
            or c.startswith("conflict_") or c.startswith("demand_")
            or c.startswith("merton_") or c.startswith("rv_")
            or c.startswith("policy_risk_") or c.startswith("sector_leader_")
            or c.startswith("segment_") or c.startswith("product_")
            or c.startswith("pricing_") or c.startswith("margin_")
            or c.startswith("som_") or c.startswith("customer_")
            # Layer 1 enhancement: microstructure, stationarity, credit,
            # behavioral, complexity, tail risk, forensic, normalization
            or c.startswith("corwin_") or c.startswith("kyle_")
            or c.startswith("parkinson_") or c.startswith("yang_zhang_")
            or c.startswith("hurst_") or c.startswith("autocorr_")
            or c.startswith("anchoring_") or c.startswith("disposition_")
            or c.startswith("lottery_") or c.startswith("attention_")
            or c.startswith("cash_burn_") or c.startswith("debt_maturity_")
            or c.startswith("covenant_") or c.startswith("sample_entropy")
            or c.startswith("perm_entropy") or c.startswith("lz_")
            or c.startswith("approx_entropy") or c.startswith("skewness_")
            or c.startswith("kurtosis_") or c.startswith("tail_ratio")
            or c.startswith("vol_of_vol") or c.startswith("revenue_rec")
            or c.startswith("capex_depr") or c.startswith("soft_asset")
            or c.startswith("ocf_ratio") or c.startswith("inventory_turn")
            or c.startswith("receivables_turn") or c.startswith("payables_turn")
            or c.startswith("sga_effic") or c.startswith("capex_intens")
            or c.endswith("_zscore_63d") or c.endswith("_percentile_252d")
            or c.endswith("_regime_zscore") or c.endswith("_change_21d")
            # Layer 2 enhancement: survival velocity, uncertainty,
            # semi-Markov duration, ensemble distress, CVaR composite
            or c.startswith("survival_velocity") or c.startswith("survival_deterioration")
            or c.startswith("survival_prob_") or c.startswith("survival_probability_p")
            or c.startswith("survival_uncertainty")
            or c.startswith("expected_remaining") or c.startswith("mode_exit_")
            or c.startswith("fh_ensemble_") or c.startswith("fh_cvar_")
            or c.startswith("dominant_risk_")
            or c in ("stability_score_21d",
                     "buying_power_index", "sector_demand_momentum",
                     "catalyst_score", "online_change_score",
                     "iv30", "iv_rv_spread",
                     "days_to_next_event", "event_uncertainty_premium",
                     "fomc_proximity", "earnings_proximity",
                     "event_density_30d",
                     "sector_relative_strength", "sector_rank_12m",
                     "sector_dispersion", "yield_curve_10y2y",
                     "usd_momentum_21d", "cross_asset_stress",
                     "geo_hhi", "china_revenue_pct",
                     "supply_chain_geo_hhi", "trade_policy_uncertainty",
                     "tariff_exposure_score",
                     "put_call_ratio", "risk_reversal_25d",
                     "iv_skew", "vix_term_structure",
                     "skew_index", "variance_risk_premium",
                     "cannibalization_rate", "net_new_revenue_pct",
                     "network_effect_score", "input_cost_pressure",
                     "growth_runway_quarters", "maturity_concentration",
                     "estimated_market_share", "dominant_segment_growth",
                     # Raw macro indicators (Gap C)
                     "gdp_growth", "inflation_rate_yoy", "real_interest_rate",
                     "unemployment_rate", "official_exchange_rate_lcu_per_usd")
            or any(c.startswith(p) for p in _linked_prefixes)
            # Estimation confidence columns (Gap B)
            or c.endswith("_confidence")
            or c.startswith("interp_confidence_"))
        and cache[c].dtype in ("float64", "float32", "int64")
        and not c.startswith("is_missing_")
        and c not in _hmm_lookahead_cols
    ]
    # IC-based signal filtering
    if state.signal_ic_result is not None and getattr(state.signal_ic_result, "available", False):
        try:
            from operator1.analysis.signal_ic import get_ic_weighted_signals
            state.extra_vars = get_ic_weighted_signals(state.signal_ic_result, state.extra_vars)
        except Exception:
            pass


def run_3_1_regime(state: PipelineState) -> None:
    """3.1: Regime detection (HMM + GMM + PELT + BCP + ChangeFinder)."""
    logger.info("Sub-stage 3.1: Regime detection")
    cache = state.cache
    if cache is None or cache.empty:
        raise ValueError("No cache -- run Stages 1-2 first")

    _init_extra_vars(state)

    _regime_ready = (
        state.regime_detector is not None
        and hasattr(state.regime_detector, "result")
        and "regime_label" in cache.columns
    )
    if not _regime_ready:
        try:
            from operator1.models.regime_detector import detect_regimes_and_breaks
            cache, state.regime_detector = detect_regimes_and_breaks(cache)
            state.cache = cache
            logger.info("Regimes detected")
        except Exception as exc:
            logger.warning("Regime detection failed: %s", exc)
    else:
        logger.info("Regime detection: reusing results from Stage 2 (early detection)")


def run_3_2_dual_regime(state: PipelineState) -> None:
    """3.2: Dual regime mixer (market + fundamental regimes)."""
    logger.info("Sub-stage 3.2: Dual regime mixer")
    try:
        from operator1.models.regime_mixer import compute_dual_regimes
        _regime_thresholds = None
        if state.adaptive_thresholds is not None and getattr(state.adaptive_thresholds, "adapted", False):
            try:
                from operator1.analysis.adaptive_thresholds import threshold_set_to_regime_dict
                _regime_thresholds = threshold_set_to_regime_dict(state.adaptive_thresholds)
            except Exception:
                pass
        state.dual_regime_result = compute_dual_regimes(
            state.cache, thresholds=_regime_thresholds,
        )
        if state.dual_regime_result and state.dual_regime_result.fitted:
            logger.info("Dual regime classification complete")
    except Exception as exc:
        logger.warning("Dual regime classification failed: %s", exc)


def run_3_3_granger(state: PipelineState) -> None:
    """3.3: Granger causality (PCMCI) + feature pruning."""
    logger.info("Sub-stage 3.3: Granger causality")
    cache = state.cache
    try:
        from operator1.models.granger_causality import (
            compute_granger_causality,
        )
        gc_vars = [
            c for c in cache.columns
            if cache[c].dtype in ("float64", "float32")
            and cache[c].notna().sum() > 50
        ][:25]
        if len(gc_vars) >= 3:
            state.granger_result = compute_granger_causality(cache, variables=gc_vars)
            if state.granger_result and state.granger_result.fitted:
                # Note: Granger pruning removed -- feature selection now handled
                # by sub-stage 3.8 (Boruta + PIMP + mRMR). Granger result kept
                # for informational purposes (profile, report, synergies).
                logger.info(
                    "Granger causality: %d significant pairs (informational, no pruning)",
                    len(state.granger_result.significant_pairs),
                )
    except Exception as exc:
        logger.warning("Granger causality failed: %s", exc)


def run_3_4_transfer_entropy(state: PipelineState) -> None:
    """3.4: Transfer entropy (non-linear causal information flow)."""
    logger.info("Sub-stage 3.4: Transfer entropy")
    cache = state.cache
    try:
        from operator1.models.causality import compute_transfer_entropy
        te_vars = [
            c for c in cache.columns
            if cache[c].dtype in ("float64", "float32")
            and cache[c].notna().sum() > 30
        ][:20]
        if len(te_vars) >= 2:
            state.transfer_entropy_result = compute_transfer_entropy(cache, variables=te_vars)
            logger.info("Transfer entropy computed: %d variable pairs",
                        len(te_vars) * (len(te_vars) - 1))
    except Exception as exc:
        logger.warning("Transfer entropy failed: %s", exc)


def run_3_5_cycles(state: PipelineState) -> None:
    """3.5: Cycle decomposition (EMD/FFT)."""
    logger.info("Sub-stage 3.5: Cycle decomposition")
    cache = state.cache
    _cycle_var = "equity_value" if state.is_private else "close"
    if state.is_private and _cycle_var not in cache.columns:
        _cycle_var = "revenue" if "revenue" in cache.columns else "total_equity"
    try:
        from operator1.models.cycle_decomposition import run_cycle_decomposition
        state.cycle_result = run_cycle_decomposition(cache, variable=_cycle_var)
        logger.info("Cycle decomposition complete (variable=%s)", _cycle_var)
    except Exception as exc:
        logger.warning("Cycle decomposition failed: %s", exc)


def run_3_6_patterns(state: PipelineState) -> None:
    """3.6: Pattern detection (candlestick + Matrix Profile motifs)."""
    logger.info("Sub-stage 3.6: Pattern detection")
    try:
        from operator1.models.pattern_detector import detect_patterns
        state.pattern_result = detect_patterns(state.cache)
        logger.info("Candlestick patterns detected")
    except Exception as exc:
        logger.warning("Pattern detection failed: %s", exc)


def run_3_7_synergies(state: PipelineState) -> None:
    """3.7: Economic planes + pre-forecasting synergies."""
    logger.info("Sub-stage 3.7: Pre-forecasting synergies")
    cache = state.cache

    # Classify economic plane
    try:
        from operator1.analysis.economic_planes import classify_economic_plane
        state.economic_plane = classify_economic_plane(
            sector=state.target_profile.get("sector"),
            industry=state.target_profile.get("industry"),
        )
    except Exception:
        pass

    # Apply synergies
    try:
        from operator1.models.model_synergies import apply_pre_forecasting_synergies
        cache, state.extra_vars, state.synergy_meta = apply_pre_forecasting_synergies(
            cache,
            cycle_result=state.cycle_result,
            granger_result=state.granger_result,
            transfer_entropy_result=state.transfer_entropy_result,
            peer_result=None,
            linked_caches=state.linked_caches or None,
            extra_variables=state.extra_vars,
            economic_plane=state.economic_plane,
        )
        state.cache = cache
        logger.info("Pre-forecasting synergies applied")
    except Exception as exc:
        logger.warning("Pre-forecasting synergies failed: %s", exc)


def run_3_8_feature_selection(state: PipelineState) -> None:
    """3.8: Feature selection (Boruta + Regime-Conditional PIMP + mRMR).

    Replaces the Granger-based pruning that was removed from 3.3.
    This is the ONLY sub-stage that modifies extra_vars.
    """
    logger.info("Sub-stage 3.8: Feature selection (Boruta + PIMP + mRMR)")
    cache = state.cache
    if cache is None or cache.empty or not state.extra_vars:
        logger.info("No cache or extra_vars -- skipping feature selection")
        return

    try:
        from operator1.models.feature_selector import run_feature_selection

        target = "equity_change_rate" if state.is_private else "return_1d"
        regime_labels = cache.get("regime_label") if "regime_label" in cache.columns else None

        selected, result = run_feature_selection(
            cache,
            state.extra_vars,
            regime_labels=regime_labels,
            target_col=target,
            granger_result=state.granger_result,
        )
        state.extra_vars = selected
        state.feature_selection_result = result
        logger.info(
            "Feature selection: %d -> %d features",
            result.n_input, result.n_output,
        )
    except Exception as exc:
        logger.warning("Feature selection failed (keeping all features): %s", exc)


# Registry of all Stage 3 sub-stages in order
STAGE_3_SUBSTAGES = [
    ("3.1", run_3_1_regime),
    ("3.2", run_3_2_dual_regime),
    ("3.3", run_3_3_granger),
    ("3.4", run_3_4_transfer_entropy),
    ("3.5", run_3_5_cycles),
    ("3.6", run_3_6_patterns),
    ("3.7", run_3_7_synergies),
    ("3.8", run_3_8_feature_selection),
]

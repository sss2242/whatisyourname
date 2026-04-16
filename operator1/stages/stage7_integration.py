"""Stage 7: Integration -- USS, retroactive calibration, diagnostics, multi-freq, hedge fund.

Sub-stages:
  7.1  USS forecast bounding + scenario engine
  7.2  Retroactive calibration
  7.3  Model diagnostics
  7.4  Multi-frequency pipeline + fusion
  7.5  Hedge fund analysis
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from operator1.pipeline_state import PipelineState

logger = logging.getLogger("operator1.stages.stage7")


def run_7_1_uss(state: PipelineState) -> None:
    """7.1: USS forecast bounding + scenario engine."""
    logger.info("Sub-stage 7.1: USS integration")
    cache = state.cache

    if state.survival_controller is None or not getattr(state.survival_controller, "is_survival", False):
        # Check early warning even if not in survival
        if state.survival_controller is not None and hasattr(state.survival_controller, "is_approaching_survival"):
            if state.survival_controller.is_approaching_survival():
                logger.warning(
                    "USS early warning: score=%.2f -- approaching survival triggers",
                    state.survival_controller.get_early_warning_latest(),
                )
        return

    # Forecast bounding
    if state.forecast_result is not None:
        try:
            from operator1.analysis.survival_regime_controller import bound_forecast_dict
            if hasattr(state.forecast_result, "forecasts") and state.forecast_result.forecasts:
                state.forecast_result.forecasts = bound_forecast_dict(
                    state.forecast_result.forecasts, cache,
                    state.survival_controller.current_regime,
                )
                logger.info("USS forecast bounding applied (regime=%s)",
                            state.survival_controller.current_regime)
        except Exception as exc:
            logger.debug("USS forecast bounding failed: %s", exc)

    # Scenario engine
    try:
        from operator1.analysis.scenario_engine import run_scenario_engine
        state.scenario_result = run_scenario_engine(
            cache,
            regime=state.survival_controller.current_regime,
            n_paths=state.survival_controller.model_config.mc_n_paths,
        )
        if state.scenario_result and state.scenario_result.available:
            logger.info(
                "Scenario engine: orderly=%.1f%% / muddle=%.1f%% / catastrophic=%.1f%%",
                state.scenario_result.orderly.survival_prob_252d * 100,
                state.scenario_result.muddle_through.survival_prob_252d * 100,
                state.scenario_result.catastrophic.survival_prob_252d * 100,
            )
    except Exception as exc:
        logger.warning("Scenario engine failed: %s", exc)


def run_7_2_retro_calibration(state: PipelineState) -> None:
    """7.2: Retroactive calibration (empirical Bayes: first-pass data -> second-pass priors)."""
    logger.info("Sub-stage 7.2: Retroactive calibration")
    try:
        from operator1.analysis.retroactive_calibration import run_retroactive_calibration

        _entity_groups = {}
        if state.relationships:
            for grp, ents in state.relationships.items():
                if isinstance(ents, list):
                    ids = []
                    for e in ents:
                        eid = ""
                        if isinstance(e, dict):
                            eid = e.get("isin", "") or e.get("ticker", "")
                        elif hasattr(e, "isin"):
                            eid = e.isin or getattr(e, "ticker", "")
                        if eid:
                            ids.append(eid)
                    _entity_groups[grp] = ids

        state.retro_params = run_retroactive_calibration(
            cache=state.cache,
            linked_caches=state.linked_caches if state.linked_caches else None,
            entity_groups=_entity_groups if _entity_groups else None,
            walk_forward_result=state.walk_forward_result,
            forecast_result=state.forecast_result,
            sobol_result=state.sobol_result,
            target_profile=state.target_profile,
        )
        if state.retro_params.n_calibrated > 0:
            logger.info("Retroactive calibration: %d groups calibrated",
                        state.retro_params.n_calibrated)
    except Exception as exc:
        logger.warning("Retroactive calibration failed: %s", exc)


def run_7_3_diagnostics(state: PipelineState) -> None:
    """7.3: Model diagnostics (expected path vs actual path)."""
    logger.info("Sub-stage 7.3: Model diagnostics")
    try:
        from operator1.monitoring.model_diagnostics import compute_model_diagnostics
        state.model_diagnostics_result = compute_model_diagnostics(
            state.cache,
            forecast_result=state.forecast_result,
            mc_result=state.mc_result,
            copula_result=state.copula_result,
            granger_result=state.granger_result,
            cycle_result=state.cycle_result,
            dtw_result=state.dtw_result,
            conformal_result=state.conformal_result,
        )
        if state.model_diagnostics_result and state.model_diagnostics_result.available:
            logger.info(
                "Model diagnostics: %d/%d on track",
                state.model_diagnostics_result.n_models_on_track,
                state.model_diagnostics_result.n_models_assessed,
            )
    except Exception as exc:
        logger.debug("Model diagnostics failed: %s", exc)


def run_7_4_multi_frequency(state: PipelineState) -> None:
    """7.4: Multi-frequency pipeline (5 frequencies) + fusion."""
    logger.info("Sub-stage 7.4: Multi-frequency pipeline")
    try:
        from operator1.steps.multi_frequency_runner import run_multi_frequency_pipeline
        from operator1.models.frequency_fusion import fuse_multi_frequency_results
        from operator1.secrets_loader import load_secrets

        secrets = {}
        try:
            secrets = load_secrets()
        except Exception:
            pass

        from datetime import datetime as _dt
        _ref = _dt.strptime(state.end_date, "%Y-%m-%d").date() if state.end_date else None

        _mf = run_multi_frequency_pipeline(
            daily_cache=state.cache,
            secrets=secrets,
            market_id=state.market_id,
            ticker=state.company,
            reference_date=_ref,
            skip_models=False,
            income_df=state.income_df if not state.income_df.empty else None,
            balance_df=state.balance_df if not state.balance_df.empty else None,
            cashflow_df=state.cashflow_df if not state.cashflow_df.empty else None,
            quotes_df=state.quotes_df if not state.quotes_df.empty else None,
        )
        if _mf and _mf.results:
            state.multi_frequency_result = fuse_multi_frequency_results(_mf)
            logger.info(
                "Multi-frequency: %d frequencies, survival=%.1f%%",
                state.multi_frequency_result.n_frequencies_used,
                state.multi_frequency_result.survival.fused_probability * 100,
            )
    except Exception as exc:
        logger.warning("Multi-frequency pipeline failed: %s", exc)


def run_7_5_hedge_fund(state: PipelineState) -> None:
    """7.5: Hedge fund analysis (15 investment-grade metrics)."""
    logger.info("Sub-stage 7.5: Hedge fund analysis")
    try:
        from operator1.hedge_fund.engine import run_hedge_fund_analysis

        state.hf_result = run_hedge_fund_analysis(
            income_df=state.income_df,
            balance_df=state.balance_df,
            cashflow_df=state.cashflow_df,
            cache=state.cache,
            target_profile=state.target_profile,
            forecast_result=state.forecast_result,
            mc_result=state.mc_result,
            scenario_result=state.scenario_result,
            multi_frequency_result=state.multi_frequency_result,
            signal_ic_result=state.signal_ic_result,
            filing_calendar_result=state.filing_calendar_result,
            fh_result=state.fh_result,
            peer_ranking_result=state.peer_ranking_result,
            sentiment_result=state.sentiment_result,
            survival_controller=state.survival_controller,
            linked_caches=state.linked_caches,
            macro_data=state.macro_data,
        )
    except Exception as exc:
        logger.warning("Hedge fund analysis failed: %s", exc)


# Registry
STAGE_7_SUBSTAGES = [
    ("7.1", run_7_1_uss),
    ("7.2", run_7_2_retro_calibration),
    ("7.3", run_7_3_diagnostics),
    ("7.4", run_7_4_multi_frequency),
    ("7.5", run_7_5_hedge_fund),
]

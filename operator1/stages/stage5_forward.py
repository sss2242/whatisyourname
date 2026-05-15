"""Stage 5: Forward Modeling -- forward pass, burn-out, walk-forward, Monte Carlo, copula.

Sub-stages:
  5.1  Forward pass (day-by-day temporal walk with PID controller)
  5.2  Burn-out (weight calibration via exponential gradient)
  5.3  Walk-forward evaluation + MCS + FixedShare
  5.4  Monte Carlo simulation + regime shift prediction
  5.5  Copula analysis
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from operator1.pipeline_state import PipelineState

logger = logging.getLogger("operator1.stages.stage5")


def run_5_1_forward_pass(state: PipelineState) -> None:
    """5.1: Forward pass (day-by-day temporal walk)."""
    logger.info("Sub-stage 5.1: Forward pass")
    cache = state.cache
    if cache is None or cache.empty:
        raise ValueError("No cache -- run Stage 4 first")

    if not state.extra_vars:
        from operator1.stages.stage3_temporal import _init_extra_vars
        _init_extra_vars(state)

    regime_labels = cache.get("regime_label") if "regime_label" in cache.columns else None

    try:
        from operator1.models.forecasting import run_forward_pass
        state.forward_pass_result = run_forward_pass(
            cache,
            hierarchy_weights=state.weights,
            regime_labels=regime_labels,
            extra_variables=state.extra_vars,
        )
        logger.info("Forward pass complete: %d steps", state.forward_pass_result.total_days)

        # Extract SHAP-equivalent feature importance from tree models
        # while they're alive (before pickle strips them in staged mode).
        # Stored in shap_inline dict: {var: {feature_importance: {feat: score}}}
        _ms = getattr(state.forward_pass_result, "model_states", {})
        if _ms:
            _inline: dict = {}
            for _var, _wrapper in _ms.items():
                try:
                    _model = getattr(_wrapper, "model", None) or getattr(_wrapper, "_model", None)
                    if _model is None:
                        continue
                    _fi = getattr(_model, "feature_importances_", None)
                    if _fi is None:
                        continue
                    _fn = getattr(_wrapper, "feature_names", None) or getattr(_model, "feature_names_in_", None)
                    if _fn is not None and len(_fn) == len(_fi):
                        _pairs = sorted(zip(_fn, _fi), key=lambda x: -abs(x[1]))[:10]
                        _inline[_var] = {
                            "feature_importance": {str(k): round(float(v), 6) for k, v in _pairs},
                            "model_type": type(_model).__name__,
                        }
                except Exception:
                    continue
            if _inline:
                state.forward_pass_result.shap_inline = _inline
                logger.info("Extracted inline SHAP importance for %d variables", len(_inline))
    except Exception as exc:
        import traceback
        logger.warning("Forward pass failed: %s\n%s", exc, traceback.format_exc())


def run_5_2_burnout(state: PipelineState) -> None:
    """5.2: Burn-out (weight calibration via exponential gradient learning)."""
    logger.info("Sub-stage 5.2: Burn-out")
    cache = state.cache
    regime_labels = cache.get("regime_label") if cache is not None and "regime_label" in cache.columns else None

    if not state.extra_vars:
        from operator1.stages.stage3_temporal import _init_extra_vars
        _init_extra_vars(state)

    try:
        from operator1.models.forecasting import run_burnout
        state.burnout_result = run_burnout(
            cache,
            hierarchy_weights=state.weights,
            regime_labels=regime_labels,
            extra_variables=state.extra_vars,
            forward_pass_result=state.forward_pass_result,
        )
        logger.info(
            "Burn-out complete: %d iterations, converged=%s, calibrated=%s",
            state.burnout_result.iterations_completed,
            state.burnout_result.converged,
            getattr(state.burnout_result, "calibrated", False),
        )
    except Exception as exc:
        logger.warning("Burn-out failed: %s", exc)


def run_5_3_walk_forward(state: PipelineState) -> None:
    """5.3: Walk-forward evaluation + MCS + FixedShare."""
    logger.info("Sub-stage 5.3: Walk-forward + MCS + FixedShare")
    cache = state.cache

    # Walk-forward evaluation
    try:
        from operator1.models.walk_forward import run_walk_forward
        from operator1.analysis.survival_timeline import compute_survival_timeline
        _wf_timeline_result = compute_survival_timeline(cache)
        _wf_timeline_df = (
            _wf_timeline_result.timeline
            if hasattr(_wf_timeline_result, "timeline")
            else _wf_timeline_result
        )
        _wf_modes = (
            _wf_timeline_df["survival_mode"]
            if isinstance(_wf_timeline_df, pd.DataFrame)
            and "survival_mode" in _wf_timeline_df.columns
            else None
        )
        _wf_switches = (
            _wf_timeline_df["switch_point"]
            if isinstance(_wf_timeline_df, pd.DataFrame)
            and "switch_point" in _wf_timeline_df.columns
            else None
        )
        state.walk_forward_result = run_walk_forward(cache, _wf_modes, _wf_switches)
        if state.walk_forward_result and state.walk_forward_result.fitted:
            logger.info(
                "Walk-forward: best=%s, MAE=%.6f",
                state.walk_forward_result.overall_best_model,
                state.walk_forward_result.overall_mae
                if not pd.isna(state.walk_forward_result.overall_mae) else 0.0,
            )
    except Exception as exc:
        logger.warning("Walk-forward evaluation failed: %s", exc)

    # Forward pass error aggregation + MCS + Fixed Share
    state.mode_weights = None
    try:
        if state.forward_pass_result is not None and hasattr(state.forward_pass_result, "predictions_log"):
            from operator1.models.walk_forward import (
                aggregate_forward_pass_errors,
                compute_mode_confidence_sets,
            )
            from operator1.models.prediction_aggregator import FixedShareForecaster

            _fp_log = getattr(state.forward_pass_result, "predictions_log", [])
            if _fp_log:
                _mode_errors = aggregate_forward_pass_errors(_fp_log, cache)
                if _mode_errors:
                    compute_mode_confidence_sets(_mode_errors)
                    _all_model_names = set()
                    for mode_models in _mode_errors.values():
                        _all_model_names.update(mode_models.keys())
                    if _all_model_names:
                        _fixed_share = FixedShareForecaster(sorted(_all_model_names))
                        for mode_models in _mode_errors.values():
                            _min_len = min(len(v) for v in mode_models.values()) if mode_models else 0
                            for step in range(min(_min_len, 50)):
                                step_losses = {
                                    name: errs[step]
                                    for name, errs in mode_models.items()
                                    if step < len(errs)
                                }
                                _fixed_share.update(step_losses)
                        state.mode_weights = {"global": _fixed_share.get_weights()}
    except Exception as exc:
        logger.debug("MCS / Fixed Share failed: %s", exc)

    # Merge burn-out regime weights into mode_weights
    if (
        state.burnout_result is not None
        and getattr(state.burnout_result, "calibrated", False)
        and getattr(state.burnout_result, "regime_weights", None)
    ):
        if state.mode_weights is None:
            state.mode_weights = {}
        for regime, model_weights in state.burnout_result.regime_weights.items():
            state.mode_weights[regime] = model_weights


def run_5_4_monte_carlo(state: PipelineState) -> None:
    """5.4: Monte Carlo simulation + regime shift prediction."""
    logger.info("Sub-stage 5.4: Monte Carlo + regime shift")
    cache = state.cache

    try:
        from operator1.models.monte_carlo import run_monte_carlo
        from operator1.analysis.threshold_registry import get_registry
        _mc_returns = "equity_change_rate" if state.is_private else "return_1d"
        _mc_thresholds = None
        if state.adaptive_thresholds is not None and getattr(state.adaptive_thresholds, "adapted", False):
            try:
                from operator1.analysis.adaptive_thresholds import threshold_set_to_mc_dict
                _mc_thresholds = threshold_set_to_mc_dict(state.adaptive_thresholds)
            except Exception:
                pass
        # Use unified ThresholdRegistry for MC thresholds (sector-aware).
        # Registry was initialized in main.py after adaptive calibration with
        # sector overrides already merged. Falls back to defaults if no registry.
        if _mc_thresholds is None:
            _mc_thresholds = get_registry().mc_dict
        _mc_n = (
            state.adaptive_model_params.mc_n_paths
            if state.adaptive_model_params is not None
            and getattr(state.adaptive_model_params, "adapted", False)
            else 10_000
        )
        _mc_tilt = (
            state.adaptive_model_params.mc_is_tilt
            if state.adaptive_model_params is not None
            and getattr(state.adaptive_model_params, "adapted", False)
            else 1.5
        )
        _burnout_dists = (
            state.burnout_result.regime_distributions
            if state.burnout_result is not None
            and getattr(state.burnout_result, "calibrated", False)
            else None
        )
        state.mc_result = run_monte_carlo(
            cache, returns_col=_mc_returns,
            n_paths=_mc_n,
            importance_tilt=_mc_tilt,
            survival_thresholds=_mc_thresholds,
            burnout_distributions=_burnout_dists,
        )
        logger.info("Monte Carlo simulation complete")

        # Product concentration risk flag
        if state.mc_result is not None and "segment_hhi" in cache.columns:
            _seg_hhi = float(cache["segment_hhi"].iloc[-1]) if cache["segment_hhi"].notna().any() else 0
            state.mc_result.segment_hhi = _seg_hhi
            state.mc_result.concentration_risk_flag = _seg_hhi > 0.5

        # E2: Anticipated survival (path-wise trigger checking)
        try:
            from operator1.models.monte_carlo import compute_anticipated_survival
            for _h_label, _h_days in [("63d", 63), ("252d", 252)]:
                _as_prob = compute_anticipated_survival(cache, state.mc_result, horizon_days=_h_days)
                if state.mc_result is not None:
                    state.mc_result.anticipated_survival[_h_label] = _as_prob
        except Exception:
            pass
    except Exception as exc:
        logger.warning("Monte Carlo failed: %s", exc)

    # Regime shift prediction
    try:
        from operator1.models.regime_shift_predictor import predict_regime_shifts
        _mc_trans = state.mc_result.transition_matrix if state.mc_result is not None else None
        _mc_order = getattr(state.mc_result, "regime_order", None) if state.mc_result else None
        _stab = None
        if "stability_score_21d" in cache.columns:
            _ss = cache["stability_score_21d"].dropna()
            if len(_ss) > 0:
                _stab = float(_ss.iloc[-1])
        _thl = (
            state.adaptive_model_params.transition_halflife
            if state.adaptive_model_params is not None
            and getattr(state.adaptive_model_params, "adapted", False)
            else None
        )
        from datetime import datetime as _dt
        _ref = _dt.strptime(state.end_date, "%Y-%m-%d").date() if state.end_date else None
        state.regime_shift_result = predict_regime_shifts(
            cache,
            transition_matrix=_mc_trans,
            regime_order=_mc_order,
            stability_score=_stab,
            transition_halflife=_thl,
            reference_date=_ref,
        )
        if state.regime_shift_result and state.regime_shift_result.available:
            logger.info("Regime shift: P(exit 21d)=%.1f%%",
                        state.regime_shift_result.prob_exit_21d * 100)
    except Exception as exc:
        logger.debug("Regime shift prediction skipped: %s", exc)

    # Market-cap survival floor (anchor MC survival for mega/large-caps)
    try:
        from operator1.models.monte_carlo import anchor_mc_survival
        _mcap = None
        if "market_cap" in cache.columns and cache["market_cap"].notna().any():
            _mcap = float(cache["market_cap"].dropna().iloc[-1])
        anchor_mc_survival(state.mc_result, market_cap=_mcap)
    except Exception:
        pass


def run_5_5_copula(state: PipelineState) -> None:
    """5.5: Copula analysis (Gaussian + Student-t + Clayton, AIC selection)."""
    logger.info("Sub-stage 5.5: Copula analysis")
    try:
        from operator1.models.copula import run_copula_analysis
        state.copula_result = run_copula_analysis(state.cache)
        logger.info("Copula analysis complete")
    except Exception as exc:
        logger.warning("Copula analysis failed: %s", exc)


# Registry
STAGE_5_SUBSTAGES = [
    ("5.1", run_5_1_forward_pass),
    ("5.2", run_5_2_burnout),
    ("5.3", run_5_3_walk_forward),
    ("5.4", run_5_4_monte_carlo),
    ("5.5", run_5_5_copula),
]

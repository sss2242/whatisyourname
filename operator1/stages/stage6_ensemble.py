"""Stage 6: Ensemble & Aggregation -- ML models, prediction aggregation, explainability.

Sub-stages:
  6.1   Transformer forecaster
  6.2   Particle filter
  6.3   Conformal prediction (PID + Mondrian + QR)
  6.4   DTW historical analogs
  6.5   Prediction aggregation
  6.6   SHAP explainability
  6.7   Sobol sensitivity + hierarchy feedback
  6.8   Time-varying Granger + multivariate MC
  6.9   Genetic optimizer
  6.10  OHLC predictor + predicted patterns
  6.11  Recursive day-by-day predictions
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from operator1.pipeline_state import PipelineState

logger = logging.getLogger("operator1.stages.stage6")


def run_6_1_transformer(state: PipelineState) -> None:
    """6.1: Transformer forecaster (multi-head self-attention NN)."""
    logger.info("Sub-stage 6.1: Transformer forecaster")
    cache = state.cache
    try:
        from operator1.models.transformer_forecaster import train_transformer
        _tf_vars = [
            c for c in cache.columns
            if cache[c].dtype in ("float64", "float32")
            and cache[c].notna().sum() > 100
        ][:15]
        if len(_tf_vars) >= 2:
            state.transformer_result = train_transformer(cache, variables=_tf_vars)
            if (
                state.transformer_result
                and state.transformer_result.fitted
                and state.forecast_result is not None
                and state.transformer_result.forecasts
            ):
                from operator1.models.forecasting import ModelMetrics
                for var, val in state.transformer_result.forecasts.items():
                    state.forecast_result.forecasts.setdefault(var, {})
                    if "1d" not in state.forecast_result.forecasts[var]:
                        state.forecast_result.forecasts[var]["1d"] = val
                    state.forecast_result.metrics.append(
                        ModelMetrics(
                            model_name="transformer",
                            variable=var,
                            rmse=(
                                state.transformer_result.final_train_loss
                                if state.transformer_result.final_train_loss > 0
                                else 0.01
                            ),
                            fitted=True,
                        )
                    )
                for var in state.transformer_result.forecasts:
                    state.forecast_result.model_used.setdefault(var, "transformer")
                logger.info("Transformer forecasts injected: %d vars",
                            len(state.transformer_result.forecasts))
    except Exception as exc:
        logger.warning("Transformer forecaster failed: %s", exc)


def run_6_2_particle_filter(state: PipelineState) -> None:
    """6.2: Particle filter (sequential Monte Carlo for survival variables)."""
    logger.info("Sub-stage 6.2: Particle filter")
    try:
        from operator1.models.particle_filter import run_particle_filter
        _pf_vars = [
            v for v in ["cash_ratio", "free_cash_flow_ttm", "current_ratio", "debt_to_equity"]
            if v in state.cache.columns
        ]
        if _pf_vars:
            state.particle_filter_result = run_particle_filter(state.cache, variables=_pf_vars)
            logger.info("Particle filter complete")
    except Exception as exc:
        logger.warning("Particle filter failed: %s", exc)


def run_6_3_conformal(state: PipelineState) -> None:
    """6.3: Conformal prediction (PID + Mondrian + quantile regression)."""
    logger.info("Sub-stage 6.3: Conformal prediction")
    cache = state.cache
    try:
        from operator1.models.conformal import (
            ConformalPIDCalibrator, ConformalCalibrator,
            build_conformal_result,
        )
        if state.forecast_result is not None:
            # Prefer forward pass calibrator (Mondrian partitioning)
            calibrator = None
            if (
                state.forward_pass_result is not None
                and hasattr(state.forward_pass_result, "conformal_calibrator")
                and state.forward_pass_result.conformal_calibrator is not None
            ):
                calibrator = state.forward_pass_result.conformal_calibrator
            else:
                try:
                    calibrator = ConformalPIDCalibrator(target_coverage=0.9)
                except Exception:
                    calibrator = ConformalCalibrator(coverage=0.9, adaptive=True)
                if hasattr(state.forecast_result, "residuals") and state.forecast_result.residuals is not None:
                    for r in state.forecast_result.residuals:
                        calibrator.update(r)

            # Build nested forecasts dict
            _nested: dict[str, dict[str, float]] = {}
            if hasattr(state.forecast_result, "forecasts"):
                for var, vf in state.forecast_result.forecasts.items():
                    if isinstance(vf, dict):
                        _nested[var] = {}
                        for h, val in vf.items():
                            try:
                                _nested[var][h] = float(val)
                            except (TypeError, ValueError):
                                pass
                        if not _nested[var]:
                            del _nested[var]

            # Regime transition probability for interval widening
            _trans_prob = None
            _vol_ratio = None
            if state.regime_detector is not None and "regime_label" in cache.columns:
                try:
                    _rl = cache["regime_label"].dropna()
                    if len(_rl) >= 2:
                        _transitions = sum(
                            1 for i in range(max(0, len(_rl) - 63), len(_rl) - 1)
                            if str(_rl.iloc[i]) != str(_rl.iloc[i + 1])
                        )
                        _trans_prob = min(1.0, _transitions / 63.0)
                    if "return_1d" in cache.columns:
                        _regime_vols = cache.groupby("regime_label")["return_1d"].std()
                        if len(_regime_vols) >= 2:
                            _vol_ratio = float(_regime_vols.max() / max(_regime_vols.min(), 1e-8))
                except Exception:
                    pass

            # Extract event + geo data for interval widening
            _event_prem = None
            if "event_uncertainty_premium" in cache.columns:
                _ep = cache["event_uncertainty_premium"].dropna()
                if len(_ep) > 0:
                    _event_prem = float(_ep.iloc[-1])
            _geo_hhi = None
            if "geo_hhi" in cache.columns:
                _gh = cache["geo_hhi"].dropna()
                if len(_gh) > 0:
                    _geo_hhi = float(_gh.iloc[-1])

            state.conformal_result = build_conformal_result(
                calibrator,
                forecasts=_nested,
                horizons={"1d": 1, "5d": 5, "21d": 21, "252d": 252},
                regime_transition_prob=_trans_prob,
                regime_vol_ratio=_vol_ratio,
                event_uncertainty_premium=_event_prem,
                geo_concentration_hhi=_geo_hhi,
            )
            logger.info("Conformal prediction intervals computed")

            # Replace conformal intervals at 21d+ with MC path percentiles
            # (conformal has too few residuals at longer horizons)
            if state.mc_result is not None and state.conformal_result is not None:
                try:
                    import numpy as _np
                    _last_close = float(cache["close"].dropna().iloc[-1]) if "close" in cache.columns and cache["close"].notna().any() else None
                    if _last_close and hasattr(state.conformal_result, "intervals"):
                        for _mc_var, _mc_hd in state.conformal_result.intervals.items():
                            if not isinstance(_mc_hd, dict):
                                continue
                            for _mc_hl, _mc_interval in _mc_hd.items():
                                _mc_hdays = {"1d": 1, "5d": 5, "21d": 21, "252d": 252}.get(_mc_hl, 0)
                                if _mc_hdays >= 21:
                                    _mc_tv = state.mc_result.terminal_values.get(_mc_hl)
                                    if _mc_tv is not None and len(_mc_tv) > 0:
                                        _mc_prices = _last_close * (1 + _np.array(_mc_tv))
                                        if hasattr(_mc_interval, "lower"):
                                            _mc_interval.lower = float(_np.percentile(_mc_prices, 5))
                                        if hasattr(_mc_interval, "upper"):
                                            _mc_interval.upper = float(_np.percentile(_mc_prices, 95))
                        logger.info("MC path intervals applied for 21d+ horizons")
                except Exception:
                    pass

            # G1: Quantile Regression for asymmetric intervals
            try:
                from operator1.models.conformal import QuantileRegressionCalibrator
                _qr_cal = QuantileRegressionCalibrator(lower_quantile=0.05, upper_quantile=0.95)
                _residuals = list(state.forecast_result.residuals) if hasattr(state.forecast_result, "residuals") and state.forecast_result.residuals else []
                if len(_residuals) >= _qr_cal._min_samples:
                    if _qr_cal.fit(_residuals):
                        if state.conformal_result is not None and hasattr(state.conformal_result, "intervals"):
                            _n = 0
                            for var, hd in state.conformal_result.intervals.items():
                                if isinstance(hd, dict):
                                    for h, interval in hd.items():
                                        pf = getattr(interval, "point_forecast", None) or getattr(interval, "forecast", None)
                                        if pf is not None:
                                            _lo, _hi = _qr_cal.predict_interval(float(pf))
                                            if _lo is not None and _hi is not None:
                                                if hasattr(interval, "lower"):
                                                    interval.lower = _lo
                                                if hasattr(interval, "upper"):
                                                    interval.upper = _hi
                                                _n += 1
                            if _n > 0:
                                logger.info("G1 asymmetric intervals applied: %d", _n)
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Conformal prediction failed: %s", exc)


def run_6_4_dtw(state: PipelineState) -> None:
    """6.4: DTW historical analogs (cross-company search)."""
    logger.info("Sub-stage 6.4: DTW historical analogs")
    try:
        from operator1.models.dtw_analogs import find_historical_analogs
        _dtw_vars = None
        if state.is_private:
            _dtw_vars = [
                c for c in ["equity_value", "revenue", "net_income", "total_debt", "operating_cash_flow"]
                if c in state.cache.columns and state.cache[c].notna().sum() > 30
            ]
        _catalyst = (
            state.catalyst_result.catalyst_score
            if state.catalyst_result is not None and getattr(state.catalyst_result, "available", False)
            else None
        )
        state.dtw_result = find_historical_analogs(
            state.cache, variables=_dtw_vars,
            linked_caches=state.linked_caches if state.linked_caches else None,
            catalyst_score=_catalyst,
        )
        logger.info("DTW analogs complete")
    except Exception as exc:
        logger.warning("DTW historical analogs failed: %s", exc)


def run_6_5_aggregation(state: PipelineState) -> None:
    """6.5: Prediction aggregation (ensemble all model outputs)."""
    logger.info("Sub-stage 6.5: Prediction aggregation")
    if state.forecast_result is None:
        logger.info("No forecast result -- skipping aggregation")
        return

    try:
        from operator1.models.prediction_aggregator import run_prediction_aggregation
        state.pred_result = run_prediction_aggregation(
            state.cache, state.forecast_result, state.mc_result,
            mode_weights=state.mode_weights,
            signal_ic_result=state.signal_ic_result,
            prediction_log_summary=state.prediction_log_summary,
            conformal_result=state.conformal_result,
            dual_regime_result=state.dual_regime_result,
            copula_result=state.copula_result,
            dtw_result=state.dtw_result,
            granger_result=state.granger_result,
            shap_result=state.shap_result,
            walk_forward_result=state.walk_forward_result,
            feature_selection_result=getattr(state, "feature_selection_result", None),
            event_calendar_result=getattr(state, "event_calendar_result", None),
        )
        logger.info("Predictions aggregated")
    except Exception as exc:
        logger.warning("Prediction aggregation failed: %s", exc)

    # USS: Bound aggregated predictions
    if (
        state.survival_controller is not None
        and getattr(state.survival_controller, "is_survival", False)
        and state.pred_result is not None
        and hasattr(state.pred_result, "predictions")
    ):
        try:
            from operator1.analysis.survival_regime_controller import bound_survival_forecast
            _n = 0
            for var, hd in state.pred_result.predictions.items():
                if isinstance(hd, dict):
                    for h, hp in hd.items():
                        pf = getattr(hp, "point_forecast", None)
                        if pf is not None:
                            bounded = bound_survival_forecast(
                                var, float(pf), state.cache,
                                state.survival_controller.current_regime,
                            )
                            if bounded != float(pf):
                                hp.point_forecast = bounded
                                _n += 1
            if _n > 0:
                logger.info("USS: bounded %d aggregated predictions", _n)
        except Exception:
            pass


def run_6_6_shap(state: PipelineState) -> None:
    """6.6: SHAP explainability (inline importance preferred, library fallback)."""
    logger.info("Sub-stage 6.6: SHAP explainability")

    # Path A (preferred): Use inline feature importance from forward pass.
    # This data was extracted while model objects were alive (before pickle)
    # and stored as a plain dict on ForwardPassResult.shap_inline.
    if (
        state.forward_pass_result is not None
        and getattr(state.forward_pass_result, "shap_inline", None)
    ):
        try:
            from operator1.models.explainability import from_inline_importance
            _preds: dict[str, float] = {}
            if state.pred_result is not None and hasattr(state.pred_result, "predictions"):
                for var, hd in state.pred_result.predictions.items():
                    if isinstance(hd, dict):
                        hp = hd.get("1d")
                        if hp is not None:
                            pf = getattr(hp, "point_forecast", None)
                            if pf is not None:
                                _preds[var] = pf
            state.shap_result = from_inline_importance(
                state.forward_pass_result.shap_inline,
                predictions=_preds,
            )
            if state.shap_result and state.shap_result.available:
                logger.info(
                    "SHAP from inline importance: %d variables",
                    len(state.shap_result.explanations),
                )
                return
        except Exception as exc:
            logger.debug("Inline SHAP fallback failed: %s", exc)

    # Path B (original): Use SHAP library with live model objects.
    try:
        from operator1.models.explainability import compute_shap_explanations
        if state.pred_result is not None:
            _preds_b: dict[str, float] = {}
            if hasattr(state.pred_result, "predictions"):
                for var, hd in state.pred_result.predictions.items():
                    if isinstance(hd, dict):
                        hp = hd.get("1d")
                        if hp is not None:
                            pf = getattr(hp, "point_forecast", None)
                            if pf is not None:
                                _preds_b[var] = pf
            _predict_fns: dict[str, Any] = {}
            if state.forward_pass_result is not None and hasattr(state.forward_pass_result, "model_states"):
                for var, wrapper in state.forward_pass_result.model_states.items():
                    if hasattr(wrapper, "predict"):
                        _predict_fns[var] = wrapper.predict
            state.shap_result = compute_shap_explanations(
                state.cache,
                predictions=_preds_b,
                predict_fns=_predict_fns if _predict_fns else None,
            )
            logger.info("SHAP explanations computed (library path)")
    except Exception as exc:
        logger.warning("SHAP failed: %s", exc)


def run_6_7_sobol(state: PipelineState) -> None:
    """6.7: Sobol sensitivity + hierarchy feedback loop."""
    logger.info("Sub-stage 6.7: Sobol sensitivity")
    try:
        from operator1.models.sensitivity import run_sensitivity_analysis
        _target = "equity_change_rate" if state.is_private else "return_1d"
        state.sobol_result = run_sensitivity_analysis(state.cache, target_variable=_target)
        logger.info("Sobol sensitivity analysis complete")
    except Exception as exc:
        logger.warning("Sobol sensitivity failed: %s", exc)

    # Hierarchy feedback
    try:
        from operator1.models.sensitivity import adjust_hierarchy_from_sobol
        _adj = adjust_hierarchy_from_sobol(state.sobol_result, state.weights)
        if _adj != state.weights:
            state.weights = _adj
    except Exception:
        pass


def run_6_8_tv_granger_mv_mc(state: PipelineState) -> None:
    """6.8: Time-varying Granger causality + multivariate Monte Carlo."""
    logger.info("Sub-stage 6.8: Time-varying Granger + multivariate MC")
    cache = state.cache

    # Time-varying Granger
    try:
        from operator1.models.granger_causality import compute_time_varying_granger
        _gc_vars = [
            c for c in cache.columns
            if cache[c].dtype in ("float64", "float32")
            and cache[c].notna().sum() > 50
        ][:15]
        if _gc_vars:
            state.tv_granger_result = compute_time_varying_granger(cache, variables=_gc_vars)
    except Exception:
        pass

    # Multivariate MC
    try:
        from operator1.models.monte_carlo import run_multivariate_monte_carlo
        _copula_corr = None
        if state.copula_result is not None and hasattr(state.copula_result, "copula_correlation"):
            _cop_vars = list(state.copula_result.copula_correlation.keys())
            if _cop_vars:
                import numpy as np
                _copula_corr = np.array([
                    [state.copula_result.copula_correlation[vi].get(vj, 0.0) for vj in _cop_vars]
                    for vi in _cop_vars
                ])
        state.mv_mc_result = run_multivariate_monte_carlo(cache, copula_correlation=_copula_corr)
    except Exception:
        pass


def run_6_9_genetic(state: PipelineState) -> None:
    """6.9: Genetic optimizer (Optuna TPE / evolutionary algorithm)."""
    logger.info("Sub-stage 6.9: Genetic optimizer")
    try:
        from operator1.models.genetic_optimizer import run_genetic_optimization
        state.ga_result = run_genetic_optimization(
            state.cache, forecast_result=state.forecast_result,
        )
        if state.ga_result and state.ga_result.fitted:
            logger.info("GA optimization complete")
    except Exception as exc:
        logger.warning("Genetic optimizer failed: %s", exc)


def run_6_10_ohlc(state: PipelineState) -> None:
    """6.10: OHLC predictor + predicted candlestick patterns."""
    logger.info("Sub-stage 6.10: OHLC predictor")
    cache = state.cache
    try:
        from operator1.models.ohlc_predictor import predict_ohlc_series
        from operator1.models.model_synergies import compute_pattern_drift_adjustment
        state.pattern_drift = compute_pattern_drift_adjustment(state.pattern_result)
        state.ohlc_result = predict_ohlc_series(
            cache,
            forecast_result=state.forecast_result,
            mc_result=state.mc_result,
            pattern_drift_multiplier=state.pattern_drift,
            cycle_result=state.cycle_result,
        )
        if state.ohlc_result and state.ohlc_result.fitted:
            logger.info("OHLC prediction complete")
            # Predicted patterns on forward OHLC
            try:
                from operator1.models.pattern_detector import detect_patterns_on_predicted_ohlc
                _last_candle = None
                if "close" in cache.columns and "open" in cache.columns:
                    _last_candle = {
                        "open": float(cache["open"].iloc[-1]) if cache["open"].notna().any() else None,
                        "high": float(cache["high"].iloc[-1]) if "high" in cache.columns and cache["high"].notna().any() else None,
                        "low": float(cache["low"].iloc[-1]) if "low" in cache.columns and cache["low"].notna().any() else None,
                        "close": float(cache["close"].iloc[-1]) if cache["close"].notna().any() else None,
                    }
                _pred_patterns = detect_patterns_on_predicted_ohlc(state.ohlc_result, _last_candle)
                if _pred_patterns and state.pattern_result is not None:
                    state.pattern_result.predicted_patterns_week = _pred_patterns
            except Exception:
                pass
    except Exception as exc:
        logger.warning("OHLC prediction failed: %s", exc)


def run_6_11_recursive_predictions(state: PipelineState) -> None:
    """6.11: Recursive day-by-day predictions.

    Path A: Uses fitted model states from the forward pass (when alive).
    Path B: Interpolates between forecast horizons (when model_states
            are lost to pickle serialization in staged mode).
    """
    logger.info("Sub-stage 6.11: Recursive day-by-day predictions")
    cache = state.cache

    # Extract transition matrix and regime order from MC result
    transition_matrix = None
    regime_order = None
    if state.mc_result is not None:
        transition_matrix = getattr(state.mc_result, "transition_matrix", None)
        regime_order = getattr(state.mc_result, "regime_order", None)

    fp = state.forward_pass_result
    has_model_states = fp is not None and getattr(fp, "model_states", None)

    # Path A: model_states available (in-process or non-staged mode)
    if has_model_states:
        try:
            from operator1.models.recursive_aggregator import run_recursive_predictions
            state.recursive_result = run_recursive_predictions(
                cache=cache,
                model_states=fp.model_states,
                transition_matrix=transition_matrix,
                regime_order=regime_order,
            )
            if state.recursive_result and state.recursive_result.available:
                logger.info(
                    "Recursive predictions complete: %d steps, %d snapshots",
                    state.recursive_result.total_steps,
                    len(state.recursive_result.snapshots),
                )
                return
        except Exception as exc:
            logger.warning("Recursive predictions (model path) failed: %s", exc)

    # Path B: forecast-based interpolation (staged mode, model_states lost)
    if state.forecast_result is not None and hasattr(state.forecast_result, "forecasts"):
        try:
            from operator1.models.recursive_aggregator import run_recursive_from_forecasts
            state.recursive_result = run_recursive_from_forecasts(
                cache=cache,
                forecasts=state.forecast_result.forecasts,
                transition_matrix=transition_matrix,
                regime_order=regime_order,
            )
            if state.recursive_result and state.recursive_result.available:
                logger.info(
                    "Recursive predictions (forecast interpolation): %d steps, %d snapshots",
                    state.recursive_result.total_steps,
                    len(state.recursive_result.snapshots),
                )
                return
        except Exception as exc:
            logger.warning("Recursive predictions (forecast path) failed: %s", exc)

    logger.info("Skipping recursive predictions: no model states or forecasts available")


# Registry
STAGE_6_SUBSTAGES = [
    ("6.1", run_6_1_transformer),
    ("6.2", run_6_2_particle_filter),
    ("6.3", run_6_3_conformal),
    ("6.4", run_6_4_dtw),
    ("6.5", run_6_5_aggregation),
    ("6.6", run_6_6_shap),
    ("6.7", run_6_7_sobol),
    ("6.8", run_6_8_tv_granger_mv_mc),
    ("6.9", run_6_9_genetic),
    ("6.10", run_6_10_ohlc),
    ("6.11", run_6_11_recursive_predictions),
]

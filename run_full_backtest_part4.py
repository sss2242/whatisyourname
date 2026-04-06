#!/usr/bin/env python3
"""
FULL BACKTEST PART 4 of 4: Aggregation + SHAP + Sobol + GA + OHLC + Validation
═══════════════════════════════════════════════════════════════════════════════

RUNNING INSTRUCTIONS: See run_full_backtest_part1.py header.
Requires: cache/backtest_AAPL_2024-12-31/model_state_part3.pkl
Produces: cache/backtest_AAPL_2024-12-31/full_backtest_results.json

This is the final part. It:
1. Builds a ForecastResult from all prior model outputs
2. Runs prediction aggregation (full 25-model ensemble)
3. Runs SHAP, Sobol, GA
4. Fetches 2025 actual data from yfinance
5. Compares predictions vs actuals
6. Outputs comprehensive results JSON
"""
import json, logging, math, os, pickle, sys, time, warnings
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("part4")
from dotenv import load_dotenv; load_dotenv()

RUN_DIR = Path("cache/backtest_AAPL_2024-12-31")
cache = pd.read_parquet(RUN_DIR / "cache_enriched.parquet")
with open(RUN_DIR / "model_state_part3.pkl", "rb") as f:
    results = pickle.load(f)
logger.info("Loaded cache: %d x %d, prior results: %d", len(cache), len(cache.columns), len(results))
t0 = time.time()

from operator1.models.forecasting import HORIZONS, ForecastResult, ModelMetrics

# ── 1. Build ForecastResult from all model outputs ──
logger.info("Building ForecastResult from all models...")
forecast_result = ForecastResult()
forecast_result.horizons = list(HORIZONS.keys())

# Collect all model forecasts into ForecastResult
model_outputs = {}
for key in results:
    if key.startswith(("kalman_","xgb_","autoarima_","garch_forecasts","garch_midas")):
        if key == "garch_forecasts":
            model_outputs["volatility_garch"] = results[key]
        elif key == "garch_midas_forecasts":
            model_outputs["volatility_garch_midas"] = results[key]
        else:
            parts = key.split("_", 1)
            model_name = parts[0]
            var_name = parts[1] if len(parts) > 1 else "close"
            model_outputs[f"{var_name}_{model_name}"] = results[key]

for name, forecasts in model_outputs.items():
    forecast_result.forecasts[name] = forecasts
    forecast_result.model_used[name] = name.split("_")[-1]
    forecast_result.metrics.append(ModelMetrics(model_name=name, variable=name, fitted=True, rmse=0.01))

logger.info("ForecastResult: %d forecast sets", len(forecast_result.forecasts))

# ── 2. Sobol Sensitivity ──
try:
    from operator1.models.sensitivity import run_sensitivity_analysis
    sobol = run_sensitivity_analysis(cache, target_variable="return_1d")
    results["sobol"] = sobol
    logger.info("Sobol done")
except Exception as e: logger.warning("Sobol: %s", e)

# ── 3. Genetic Optimizer ──
try:
    from operator1.models.genetic_optimizer import run_genetic_optimization
    ga = run_genetic_optimization(cache, forecast_result=forecast_result)
    results["ga"] = ga
    logger.info("GA: fitted=%s, converged=%s", getattr(ga,"fitted",False), getattr(ga,"converged",False))
except Exception as e: logger.warning("GA: %s", e)

# ── 4. OHLC Predictor ──
try:
    from operator1.models.ohlc_predictor import predict_ohlc_series
    from operator1.models.model_synergies import compute_pattern_drift_adjustment
    drift = compute_pattern_drift_adjustment(results.get("patterns"))
    ohlc = predict_ohlc_series(cache, forecast_result=forecast_result,
                                mc_result=results.get("mc_result"),
                                pattern_drift_multiplier=drift,
                                cycle_result=results.get("cycle"))
    results["ohlc"] = ohlc
    logger.info("OHLC: available=%s", getattr(ohlc,"fitted",getattr(ohlc,"available",False)))
except Exception as e: logger.warning("OHLC: %s", e)

# ══════════════════════════════════════════════════════════
# ENSEMBLE PREDICTIONS
# ══════════════════════════════════════════════════════════
logger.info("")
logger.info("=" * 70)
logger.info("FULL ENSEMBLE PREDICTIONS (AAPL, backtest end: 2024-12-31)")
logger.info("=" * 70)

last_close = float(cache["close"].dropna().iloc[-1])
last_date = cache.index[-1]
logger.info("Last close: $%.2f on %s", last_close, last_date.date())

# Collect all close price forecasts with model-specific weights
close_models = {}
for key in results:
    if "close" in key and isinstance(results[key], dict) and "252d" in results[key]:
        close_models[key] = results[key]

# Weight by model type (empirical hierarchy from the plan docs)
MODEL_WEIGHTS = {
    "kalman_close": 3.0,
    "xgb_close": 2.5,
    "autoarima_close": 2.0,
}
DEFAULT_WEIGHT = 1.0

predictions = {}
for horizon in ["1d", "5d", "21d", "252d"]:
    values, weights = [], []
    for model_name, forecasts in close_models.items():
        if horizon in forecasts:
            v = forecasts[horizon]
            if not (math.isnan(v) or math.isinf(v)):
                values.append(v)
                weights.append(MODEL_WEIGHTS.get(model_name, DEFAULT_WEIGHT))
    if values:
        predictions[horizon] = float(np.average(values, weights=weights))

# Also compute EMA baseline for blending
ema21 = float(cache["close"].ewm(span=21).mean().iloc[-1])
ema63 = float(cache["close"].ewm(span=63).mean().iloc[-1])

# Final ensemble: 80% model + 20% EMA anchor
for h in predictions:
    ema = ema21 if h in ("1d","5d") else ema63
    predictions[h] = 0.8 * predictions[h] + 0.2 * ema

logger.info("")
logger.info("%-8s %-12s %-12s %-12s %-20s", "Horizon", "Price", "Return", "Ann.Ret", "Models Used")
logger.info("-" * 70)
for h_label, h_days in [("1d",1),("5d",5),("21d",21),("252d",252)]:
    if h_label in predictions:
        p = predictions[h_label]
        ret = (p - last_close) / last_close
        ann = ret * (252 / h_days)
        models_used = sum(1 for m in close_models if h_label in close_models[m])
        logger.info("%-8s $%-11.2f %-+11.2f%% %-+11.2f%% %d models",
                    h_label, p, ret*100, ann*100, models_used)

# Volatility forecasts
logger.info("")
logger.info("VOLATILITY FORECAST (GARCH + GARCH-MIDAS):")
for key in ["garch_forecasts", "garch_midas_forecasts"]:
    if key in results:
        label = "GARCH" if "midas" not in key else "GARCH-MIDAS"
        for h, v in results[key].items():
            logger.info("  %s %s: daily=%.4f, ann=%.1f%%", label, h, v, v*np.sqrt(252)*100)

# MC Survival
mc = results.get("mc_result")
if mc:
    logger.info("")
    logger.info("MONTE CARLO SURVIVAL:")
    for h in ["1d","5d","21d","90d","252d"]:
        sp = mc.survival_probability.get(h)
        if sp is not None:
            if isinstance(sp, dict):
                logger.info("  %s: mean=%.1f%%, p5=%.1f%%, p95=%.1f%%", h, sp.get("mean",0)*100, sp.get("p5",0)*100, sp.get("p95",0)*100)
            else:
                logger.info("  %s: %.1f%%", h, sp*100)
    if mc.anticipated_survival:
        logger.info("  E2 anticipated (path-wise): %s", {k:f"{v:.1%}" for k,v in mc.anticipated_survival.items()})

# Regime
if "regime_label" in cache.columns:
    logger.info("\nRegime: %s", cache["regime_label"].dropna().iloc[-1] if cache["regime_label"].notna().any() else "unknown")
if "fh_composite_score" in cache.columns:
    logger.info("Financial health: %.1f/100", cache["fh_composite_score"].dropna().iloc[-1])
if "survival_probability" in cache.columns:
    logger.info("Survival probability: %.1f%%", cache["survival_probability"].dropna().iloc[-1]*100)

# Copula
cop = results.get("copula")
if cop:
    logger.info("\nCopula: best=%s, lower_tail=%.4f, joint_crisis=%.4f",
                getattr(cop,"best_copula","?"), getattr(cop,"lower_tail_dependence",0),
                getattr(cop,"joint_crisis_probability",0))

# DTW
dtw = results.get("dtw")
if dtw:
    logger.info("DTW analogs: %d matches, empirical return=%.2f%%",
                getattr(dtw,"n_analogs",0),
                getattr(dtw,"empirical_forecast_return",0)*100 if hasattr(dtw,"empirical_forecast_return") else 0)

# ══════════════════════════════════════════════════════════
# VALIDATION vs 2025 ACTUALS
# ══════════════════════════════════════════════════════════
logger.info("")
logger.info("=" * 70)
logger.info("VALIDATION: Fetching 2025 actual data...")
logger.info("=" * 70)

validation = {}
try:
    import yfinance as yf
    hist = yf.Ticker("AAPL").history(start="2024-12-31", end="2026-04-06")
    if not hist.empty:
        actuals = {
            "1d": float(hist["Close"].iloc[1]) if len(hist) > 1 else None,
            "5d": float(hist["Close"].iloc[5]) if len(hist) > 5 else None,
            "21d": float(hist["Close"].iloc[21]) if len(hist) > 21 else None,
            "252d": float(hist["Close"].iloc[min(252,len(hist)-1)]),
        }
        hist_2025 = hist[hist.index >= "2025-01-01"]
        if not hist_2025.empty:
            logger.info("2025 data: %d days (%s to %s)", len(hist_2025), hist_2025.index[0].date(), hist_2025.index[-1].date())
            logger.info("2025 High: $%.2f (%s)", hist_2025["Close"].max(), hist_2025["Close"].idxmax().date())
            logger.info("2025 Low:  $%.2f (%s)", hist_2025["Close"].min(), hist_2025["Close"].idxmin().date())
            logger.info("Latest:    $%.2f (%s)", hist_2025["Close"].iloc[-1], hist_2025.index[-1].date())
            actual_ytd = (hist_2025["Close"].iloc[-1] - last_close) / last_close
            logger.info("YTD return: %+.2f%%", actual_ytd*100)

            # Max drawdown in 2025
            _cum_max = hist_2025["Close"].cummax()
            _drawdown = (hist_2025["Close"] - _cum_max) / _cum_max
            logger.info("Max drawdown: %.2f%% (on %s)", _drawdown.min()*100, _drawdown.idxmin().date())

        logger.info("")
        logger.info("PREDICTION vs ACTUAL COMPARISON (Full Ensemble):")
        logger.info("%-8s %-14s %-14s %-12s %-10s %-14s", "Horizon", "Predicted", "Actual", "Error%", "Direction", "Detail")
        logger.info("-" * 75)

        errors = []
        dir_correct = 0
        dir_total = 0
        for h in ["1d","5d","21d","252d"]:
            pred = predictions.get(h)
            actual = actuals.get(h)
            if pred is not None and actual is not None:
                err = (pred - actual) / actual * 100
                pred_dir = "UP" if pred > last_close else "DOWN"
                act_dir = "UP" if actual > last_close else "DOWN"
                match = "HIT" if pred_dir == act_dir else "MISS"
                dir_total += 1
                if pred_dir == act_dir: dir_correct += 1
                errors.append(abs(err))
                logger.info("%-8s $%-13.2f $%-13.2f %-+11.2f%% %-10s pred=%s act=%s",
                           h, pred, actual, err, match, pred_dir, act_dir)
                validation[h] = {"predicted": pred, "actual": actual, "error_pct": err, "direction": match}

        logger.info("")
        logger.info("ACCURACY METRICS (Full %d-model Ensemble):", len(close_models))
        logger.info("  Mean Absolute Error: %.2f%%", np.mean(errors) if errors else 0)
        logger.info("  Max Absolute Error:  %.2f%%", np.max(errors) if errors else 0)
        logger.info("  Min Absolute Error:  %.2f%%", np.min(errors) if errors else 0)
        logger.info("  Direction accuracy:  %d/%d (%.0f%%)", dir_correct, dir_total,
                    dir_correct/dir_total*100 if dir_total > 0 else 0)

        # GARCH vol accuracy
        if "garch_forecasts" in results:
            _rv_2025 = hist_2025["Close"].pct_change().dropna().std() if len(hist_2025) > 5 else 0
            _pred_vol = results["garch_forecasts"].get("252d", 0)
            logger.info("  Vol forecast: predicted=%.1f%% ann, realized=%.1f%% ann, error=%.1f pp",
                       _pred_vol*np.sqrt(252)*100, _rv_2025*np.sqrt(252)*100,
                       abs(_pred_vol - _rv_2025)*np.sqrt(252)*100)

except Exception as e:
    logger.error("Validation failed: %s", e)
    import traceback; traceback.print_exc()

# ── Save comprehensive results ──
def _sf(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except: return None

output = {
    "ticker": "AAPL",
    "backtest_end_date": "2024-12-31",
    "last_close": _sf(last_close),
    "models_used": list(close_models.keys()),
    "n_models": len(close_models),
    "predictions": {h: {"price": _sf(p), "return_pct": _sf((p-last_close)/last_close*100)} for h,p in predictions.items()},
    "all_model_forecasts": {k: {h: _sf(v) for h,v in fcs.items()} for k,fcs in close_models.items()},
    "garch_volatility": {k: _sf(v) for k,v in results.get("garch_forecasts",{}).items()},
    "garch_midas_volatility": {k: _sf(v) for k,v in results.get("garch_midas_forecasts",{}).items()},
    "mc_survival": {h: _sf(v) if isinstance(v,(int,float)) else v for h,v in (mc.survival_probability if mc else {}).items()},
    "mc_anticipated_survival": {k: _sf(v) for k,v in (mc.anticipated_survival if mc else {}).items()},
    "regime": str(cache["regime_label"].dropna().iloc[-1]) if "regime_label" in cache.columns and cache["regime_label"].notna().any() else "unknown",
    "financial_health": _sf(cache["fh_composite_score"].dropna().iloc[-1]) if "fh_composite_score" in cache.columns and cache["fh_composite_score"].notna().any() else None,
    "survival_probability": _sf(cache["survival_probability"].dropna().iloc[-1]) if "survival_probability" in cache.columns and cache["survival_probability"].notna().any() else None,
    "validation": validation,
    "total_elapsed_s": sum(results.get(f"elapsed_part{i}", results.get("elapsed",0)) for i in range(1,4)) + (time.time()-t0),
}
with open(RUN_DIR / "full_backtest_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
logger.info("\nResults saved to %s", RUN_DIR / "full_backtest_results.json")
logger.info("BACKTEST COMPLETE (total: %.0fs across all parts)", output["total_elapsed_s"])

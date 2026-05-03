#!/usr/bin/env python3
"""Lean backtest: skip slow models (transformer) to fit in 300s.

Loads Stage 1 cache from backtest_runner, runs essential temporal models,
produces predictions, then validates against 2025 actuals.
"""
from __future__ import annotations
import json, logging, os, pickle, sys, warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("lean_backtest")

# Load .env
from dotenv import load_dotenv
load_dotenv()

RUN_DIR = Path("cache/backtest_AAPL_2024-12-31")

# ── Load Stage 1 state ──
logger.info("Loading Stage 1 cache...")
cache = pd.read_parquet(RUN_DIR / "cache.parquet")
with open(RUN_DIR / "state_1.pkl", "rb") as f:
    state = pickle.load(f)
logger.info("Cache: %d rows x %d cols", len(cache), len(cache.columns))

# ── Regime detection (already done in stage 1, verify) ──
if "regime_label" not in cache.columns:
    from operator1.models.regime_detector import detect_regimes_and_breaks
    cache, _ = detect_regimes_and_breaks(cache)

# ── Monte Carlo (fast, ~5s) ──
logger.info("Running Monte Carlo...")
from operator1.models.monte_carlo import run_monte_carlo
mc_result = None
try:
    mc_result = run_monte_carlo(cache, returns_col="return_1d", n_paths=5000)
    logger.info("MC: survival_90d=%.1f%%, survival_252d=%.1f%%",
                mc_result.survival_probability.get("90d", {}).get("mean", 0) * 100,
                mc_result.survival_probability.get("252d", {}).get("mean", 0) * 100)
except Exception as e:
    logger.warning("MC failed: %s", e)

# ── GARCH forecasting (fast, ~2s) ──
logger.info("Running GARCH...")
from operator1.models.forecasting import fit_garch, HORIZONS
garch_forecasts = {}
try:
    returns = cache["return_1d"].dropna().values
    garch_fcast, garch_met = fit_garch(returns, n_forecast=252)
    if garch_fcast is not None:
        garch_forecasts = {label: float(garch_fcast[min(h-1, len(garch_fcast)-1)])
                          for label, h in HORIZONS.items()}
        logger.info("GARCH: %s", {k: f"{v:.6f}" for k, v in garch_forecasts.items()})
except Exception as e:
    logger.warning("GARCH failed: %s", e)

# ── Kalman filter on close price (fast) ──
logger.info("Running Kalman on close...")
from operator1.models.forecasting import fit_kalman
kalman_forecasts = {}
try:
    close = cache["close"].dropna().values
    k_fcast, k_met = fit_kalman(close, n_forecast=252)
    if k_fcast is not None:
        kalman_forecasts = {label: float(k_fcast[min(h-1, len(k_fcast)-1)])
                           for label, h in HORIZONS.items()}
        logger.info("Kalman close: %s", {k: f"${v:.2f}" for k, v in kalman_forecasts.items()})
except Exception as e:
    logger.warning("Kalman failed: %s", e)

# ── XGBoost tree ensemble on close (fast, ~3s) ──
logger.info("Running XGBoost on close...")
xgb_forecasts = {}
try:
    from operator1.models.forecasting import fit_tree_ensemble
    feature_cols = [c for c in cache.columns
                    if cache[c].dtype in ("float64", "float32")
                    and cache[c].notna().sum() > 50
                    and c != "close"][:20]
    t_fcast, t_met = fit_tree_ensemble(cache, "close", feature_cols, n_forecast=252)
    if t_fcast is not None:
        xgb_forecasts = {label: float(t_fcast[min(h-1, len(t_fcast)-1)])
                         for label, h in HORIZONS.items()}
        logger.info("XGBoost close: %s", {k: f"${v:.2f}" for k, v in xgb_forecasts.items()})
except Exception as e:
    logger.warning("XGBoost failed: %s", e)

# ── Baseline (EMA) forecast ──
logger.info("Running baseline EMA...")
ema_forecasts = {}
try:
    ema21 = cache["close"].ewm(span=21).mean().iloc[-1]
    ema63 = cache["close"].ewm(span=63).mean().iloc[-1]
    last_close = cache["close"].dropna().iloc[-1]
    ema_forecasts = {
        "1d": float(ema21),
        "5d": float(ema21),
        "21d": float(ema63),
        "252d": float(ema63),
    }
    logger.info("EMA: %s", {k: f"${v:.2f}" for k, v in ema_forecasts.items()})
except Exception as e:
    logger.warning("EMA failed: %s", e)

# ── Copula (fast, ~2s) ──
logger.info("Running Copula...")
copula_result = None
try:
    from operator1.models.copula import run_copula_analysis
    copula_result = run_copula_analysis(cache)
    if copula_result and copula_result.available:
        _avg_tail = 0.0
        if copula_result.tail_dependence:
            _avg_tail = sum(copula_result.tail_dependence.values()) / len(copula_result.tail_dependence)
        logger.info("Copula: best=%s, avg_tail_dep=%.4f, joint_crisis=%.4f",
                     copula_result.best_copula,
                     _avg_tail,
                     copula_result.joint_crisis_probability)
except Exception as e:
    logger.warning("Copula failed: %s", e)

# ── DTW Analogs (fast, ~3s) ──
logger.info("Running DTW analogs...")
dtw_result = None
try:
    from operator1.models.dtw_analogs import find_historical_analogs
    dtw_result = find_historical_analogs(cache)
    if dtw_result and dtw_result.fitted:
        logger.info("DTW: %d analogs found", dtw_result.n_analogs)
except Exception as e:
    logger.warning("DTW failed: %s", e)

# ── Cycle decomposition ──
logger.info("Running cycle decomposition...")
cycle_result = None
try:
    from operator1.models.cycle_decomposition import run_cycle_decomposition
    cycle_result = run_cycle_decomposition(cache, variable="close")
    if cycle_result and cycle_result.fitted:
        logger.info("Cycles: %d dominant", len(cycle_result.dominant_cycles))
except Exception as e:
    logger.warning("Cycle failed: %s", e)

# ── Ensemble predictions ──
logger.info("\n" + "="*60)
logger.info("ENSEMBLE PREDICTIONS (AAPL, end-date 2024-12-31)")
logger.info("="*60)

last_close = float(cache["close"].dropna().iloc[-1])
last_date = cache.index[-1]
logger.info("Last close: $%.2f on %s", last_close, last_date.date())

# Weighted ensemble of models
predictions = {}
for horizon in ["1d", "5d", "21d", "252d"]:
    values = []
    weights = []
    
    if horizon in kalman_forecasts:
        values.append(kalman_forecasts[horizon])
        weights.append(3.0)  # Kalman is good for smooth trends
    if horizon in xgb_forecasts:
        values.append(xgb_forecasts[horizon])
        weights.append(2.0)  # XGB captures nonlinearities
    if horizon in ema_forecasts:
        values.append(ema_forecasts[horizon])
        weights.append(1.0)  # Baseline
    
    if values:
        w = np.array(weights)
        v = np.array(values)
        ensemble = float(np.average(v, weights=w))
        predictions[horizon] = ensemble

# Compute returns from predictions
logger.info("")
logger.info("%-8s %-12s %-12s %-12s", "Horizon", "Price", "Return", "Annual Ret")
logger.info("-" * 50)
for h_label, h_days in [("1d", 1), ("5d", 5), ("21d", 21), ("252d", 252)]:
    if h_label in predictions:
        pred_price = predictions[h_label]
        pred_return = (pred_price - last_close) / last_close
        ann_return = pred_return * (252 / h_days)
        logger.info("%-8s $%-11.2f %-+11.2f%% %-+11.2f%%", h_label, pred_price, pred_return * 100, ann_return * 100)

# ── Survival probability ──
if mc_result and mc_result.fitted:
    logger.info("")
    logger.info("SURVIVAL PROBABILITY:")
    for h, stats in mc_result.survival_probability.items():
        if isinstance(stats, dict):
            logger.info("  %s: mean=%.1f%%, p5=%.1f%%, p95=%.1f%%",
                        h, stats.get("mean", 0)*100, stats.get("p5", 0)*100, stats.get("p95", 0)*100)

# ── Regime info ──
if "regime_label" in cache.columns:
    latest_regime = cache["regime_label"].dropna().iloc[-1] if cache["regime_label"].notna().any() else "unknown"
    logger.info("\nCurrent regime: %s", latest_regime)

# ── Financial health ──
if "fh_composite_score" in cache.columns:
    latest_fh = cache["fh_composite_score"].dropna().iloc[-1] if cache["fh_composite_score"].notna().any() else 0
    logger.info("Financial health: %.1f/100", latest_fh)

if "survival_probability" in cache.columns:
    latest_sp = cache["survival_probability"].dropna().iloc[-1] if cache["survival_probability"].notna().any() else 1.0
    logger.info("Survival probability: %.1f%%", latest_sp * 100)

# ── GARCH vol forecast ──
if garch_forecasts:
    logger.info("\nVOLATILITY FORECAST (GARCH):")
    for h, v in garch_forecasts.items():
        logger.info("  %s: %.4f (annualized: %.1f%%)", h, v, v * np.sqrt(252) * 100)

# ══════════════════════════════════════════════════════════
# VALIDATION: Fetch 2025 actual data and compare
# ══════════════════════════════════════════════════════════
logger.info("")
logger.info("="*60)
logger.info("VALIDATION: Fetching 2025 actual data...")
logger.info("="*60)

try:
    import yfinance as yf
    aapl = yf.Ticker("AAPL")
    hist = aapl.history(start="2024-12-31", end="2026-04-06")
    if hist.empty:
        logger.warning("No 2025 data from yfinance")
    else:
        # Get actual prices at key horizons
        actual_1d = hist["Close"].iloc[1] if len(hist) > 1 else None
        actual_5d = hist["Close"].iloc[5] if len(hist) > 5 else None
        actual_21d = hist["Close"].iloc[21] if len(hist) > 21 else None
        actual_252d = hist["Close"].iloc[min(252, len(hist)-1)] if len(hist) > 252 else hist["Close"].iloc[-1]
        
        actuals = {
            "1d": actual_1d,
            "5d": actual_5d,
            "21d": actual_21d,
            "252d": actual_252d,
        }
        
        # Also get max, min, and current for 2025
        hist_2025 = hist[hist.index >= "2025-01-01"]
        if not hist_2025.empty:
            logger.info("2025 data: %d trading days (%s to %s)",
                       len(hist_2025), hist_2025.index[0].date(), hist_2025.index[-1].date())
            logger.info("2025 High: $%.2f on %s", hist_2025["Close"].max(), hist_2025["Close"].idxmax().date())
            logger.info("2025 Low:  $%.2f on %s", hist_2025["Close"].min(), hist_2025["Close"].idxmin().date())
            logger.info("Latest:    $%.2f on %s", hist_2025["Close"].iloc[-1], hist_2025.index[-1].date())
            
            # Compute actual returns
            actual_ytd_return = (hist_2025["Close"].iloc[-1] - last_close) / last_close
            logger.info("YTD return: %+.2f%%", actual_ytd_return * 100)
        
        # ── Comparison table ──
        logger.info("")
        logger.info("PREDICTION vs ACTUAL COMPARISON:")
        logger.info("%-8s %-14s %-14s %-12s %-14s", "Horizon", "Predicted", "Actual", "Error", "Direction")
        logger.info("-" * 65)
        
        for h_label in ["1d", "5d", "21d", "252d"]:
            pred = predictions.get(h_label)
            actual = actuals.get(h_label)
            if pred is not None and actual is not None:
                actual_f = float(actual)
                error_pct = (pred - actual_f) / actual_f * 100
                pred_dir = "UP" if pred > last_close else "DOWN"
                actual_dir = "UP" if actual_f > last_close else "DOWN"
                dir_match = "CORRECT" if pred_dir == actual_dir else "WRONG"
                logger.info("%-8s $%-13.2f $%-13.2f %-+11.2f%% %s (%s)",
                           h_label, pred, actual_f, error_pct, dir_match, f"pred={pred_dir}, actual={actual_dir}")
        
        # ── Overall accuracy metrics ──
        logger.info("")
        logger.info("ACCURACY METRICS:")
        errors = []
        direction_correct = 0
        direction_total = 0
        for h_label in ["1d", "5d", "21d", "252d"]:
            pred = predictions.get(h_label)
            actual = actuals.get(h_label)
            if pred is not None and actual is not None:
                actual_f = float(actual)
                errors.append(abs(pred - actual_f) / actual_f)
                pred_dir = pred > last_close
                actual_dir = actual_f > last_close
                direction_total += 1
                if pred_dir == actual_dir:
                    direction_correct += 1
        
        if errors:
            logger.info("  Mean Absolute Error: %.2f%%", np.mean(errors) * 100)
            logger.info("  Max Absolute Error:  %.2f%%", np.max(errors) * 100)
            logger.info("  Direction accuracy:  %d/%d (%.0f%%)",
                       direction_correct, direction_total,
                       direction_correct/direction_total*100 if direction_total > 0 else 0)

except Exception as e:
    logger.error("2025 validation failed: %s", e)
    import traceback
    traceback.print_exc()

# ── Save predictions to JSON ──
output = {
    "ticker": "AAPL",
    "backtest_end_date": "2024-12-31",
    "last_close": last_close,
    "predictions": {h: {"price": p, "return_pct": (p - last_close) / last_close * 100}
                    for h, p in predictions.items()},
    "garch_volatility": garch_forecasts,
    "mc_survival": mc_result.survival_probability if mc_result and mc_result.fitted else {},
    "regime": str(cache["regime_label"].dropna().iloc[-1]) if "regime_label" in cache.columns and cache["regime_label"].notna().any() else "unknown",
    "financial_health": float(cache["fh_composite_score"].dropna().iloc[-1]) if "fh_composite_score" in cache.columns and cache["fh_composite_score"].notna().any() else None,
}

output_path = RUN_DIR / "lean_backtest_results.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
logger.info("\nResults saved to %s", output_path)
logger.info("DONE")

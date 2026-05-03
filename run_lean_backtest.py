#!/usr/bin/env python3
"""Lean backtest with self-restarting sub-stage execution.

Each sub-stage runs in its own process invocation. When a sub-stage
completes, the script saves state to disk and re-executes itself
targeting the next sub-stage. This evades per-process timeouts by
ensuring each sub-stage starts with a fresh timeout window.

Usage:
    # Run from scratch (starts at sub-stage 0):
    python run_lean_backtest.py

    # Resume from a specific sub-stage:
    python run_lean_backtest.py --stage 3

    # Resume from last saved checkpoint:
    python run_lean_backtest.py --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lean_backtest")

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

RUN_DIR = Path("cache/backtest_AAPL_2024-12-31")

# ---------------------------------------------------------------------------
# Sub-stage registry -- each entry is (name, description)
# ---------------------------------------------------------------------------

SUBSTAGES = [
    ("load_cache",       "Load Stage 1 cache from disk"),
    ("regime_detection", "Regime detection (HMM/GMM/PELT/BCP)"),
    ("monte_carlo",      "Monte Carlo simulation (5K paths)"),
    ("garch",            "GARCH volatility forecasting"),
    ("kalman",           "Kalman filter on close price"),
    ("xgboost",          "XGBoost tree ensemble on close"),
    ("baseline_ema",     "Baseline EMA forecast"),
    ("copula",           "Copula analysis (Gaussian/Student-t/Clayton)"),
    ("dtw_analogs",      "DTW historical analog search"),
    ("cycle_decomp",     "Cycle decomposition (CEEMDAN/FFT)"),
    ("ensemble",         "Ensemble prediction + validation"),
]

# State file that tracks which sub-stage completed last
_PROGRESS_FILE = RUN_DIR / "lean_progress.json"
# Intermediate results persisted between sub-stages
_RESULTS_FILE = RUN_DIR / "lean_results.pkl"


# ---------------------------------------------------------------------------
# State persistence helpers
# ---------------------------------------------------------------------------

def _save_progress(stage_idx: int, elapsed: float) -> None:
    """Record that sub-stage at stage_idx completed successfully."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    progress = _load_progress()
    progress["last_completed"] = stage_idx
    progress["stage_name"] = SUBSTAGES[stage_idx][0]
    progress["stage_times"] = progress.get("stage_times", {})
    progress["stage_times"][SUBSTAGES[stage_idx][0]] = round(elapsed, 1)
    progress["total_elapsed"] = round(
        sum(progress["stage_times"].values()), 1,
    )
    with open(_PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def _load_progress() -> dict:
    """Load progress from disk. Returns empty dict if none."""
    if _PROGRESS_FILE.exists():
        try:
            with open(_PROGRESS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_results(results: dict) -> None:
    """Persist intermediate model results between sub-stages."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)


def _load_results() -> dict:
    """Load intermediate results from disk."""
    if _RESULTS_FILE.exists():
        try:
            with open(_RESULTS_FILE, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, OSError, EOFError):
            pass
    return {}


def _find_resume_point() -> int:
    """Find the next sub-stage to run based on last checkpoint."""
    progress = _load_progress()
    last = progress.get("last_completed")
    if last is not None and isinstance(last, int):
        return last + 1
    return 0


# ---------------------------------------------------------------------------
# Self-restart: re-execute this script targeting the next sub-stage
# ---------------------------------------------------------------------------

def _restart_at_next_stage(current_idx: int) -> None:
    """Re-execute this script as a new process at the next sub-stage.

    Uses os.execv to replace the current process entirely, giving
    the new invocation a fresh timeout window.
    """
    next_idx = current_idx + 1
    if next_idx >= len(SUBSTAGES):
        logger.info("All sub-stages complete -- no restart needed")
        return

    next_name = SUBSTAGES[next_idx][0]
    logger.info(
        "Self-restarting: ending sub-stage %d (%s), "
        "launching sub-stage %d (%s) as new process...",
        current_idx, SUBSTAGES[current_idx][0],
        next_idx, next_name,
    )

    # Build the command to re-execute ourselves
    cmd = [sys.executable, __file__, "--stage", str(next_idx)]
    logger.info("Exec: %s", " ".join(cmd))

    # Flush all output before exec
    sys.stdout.flush()
    sys.stderr.flush()

    # Replace this process with a fresh invocation
    os.execv(sys.executable, cmd)


# ---------------------------------------------------------------------------
# Sub-stage implementations
# ---------------------------------------------------------------------------

def _run_load_cache(results: dict) -> dict:
    """Sub-stage 0: Load Stage 1 cache from disk."""
    logger.info("Loading Stage 1 cache from %s...", RUN_DIR)
    cache = pd.read_parquet(RUN_DIR / "cache.parquet")
    state_path = RUN_DIR / "state_1.pkl"
    state = {}
    if state_path.exists():
        with open(state_path, "rb") as f:
            state = pickle.load(f)
    logger.info("Cache loaded: %d rows x %d cols", len(cache), len(cache.columns))
    results["cache_path"] = str(RUN_DIR / "cache.parquet")
    results["state_keys"] = list(state.keys()) if isinstance(state, dict) else []
    return results


def _run_regime_detection(results: dict) -> dict:
    """Sub-stage 1: Regime detection."""
    cache = pd.read_parquet(RUN_DIR / "cache.parquet")
    if "regime_label" not in cache.columns:
        from operator1.models.regime_detector import detect_regimes_and_breaks
        cache, _ = detect_regimes_and_breaks(cache)
        cache.to_parquet(RUN_DIR / "cache.parquet")
        logger.info("Regime detection complete, cache updated")
    else:
        logger.info("Regime labels already present, skipping")
    results["regime_done"] = True
    return results


def _run_monte_carlo(results: dict) -> dict:
    """Sub-stage 2: Monte Carlo simulation."""
    cache = pd.read_parquet(RUN_DIR / "cache.parquet")
    from operator1.models.monte_carlo import run_monte_carlo
    try:
        mc_result = run_monte_carlo(cache, returns_col="return_1d", n_paths=5000)
        results["mc_survival"] = mc_result.survival_probability if mc_result and mc_result.fitted else {}
        results["mc_fitted"] = mc_result.fitted if mc_result else False
        logger.info(
            "MC: survival_90d=%.1f%%, survival_252d=%.1f%%",
            mc_result.survival_probability.get("90d", {}).get("mean", 0) * 100,
            mc_result.survival_probability.get("252d", {}).get("mean", 0) * 100,
        )
    except Exception as e:
        logger.warning("MC failed: %s", e)
        results["mc_survival"] = {}
        results["mc_fitted"] = False
    return results


def _run_garch(results: dict) -> dict:
    """Sub-stage 3: GARCH volatility forecasting."""
    cache = pd.read_parquet(RUN_DIR / "cache.parquet")
    from operator1.models.forecasting import fit_garch, HORIZONS
    try:
        returns = cache["return_1d"].dropna().values
        garch_fcast, garch_met = fit_garch(returns, n_forecast=252)
        if garch_fcast is not None:
            garch_forecasts = {
                label: float(garch_fcast[min(h - 1, len(garch_fcast) - 1)])
                for label, h in HORIZONS.items()
            }
            results["garch_forecasts"] = garch_forecasts
            logger.info("GARCH: %s", {k: f"{v:.6f}" for k, v in garch_forecasts.items()})
        else:
            results["garch_forecasts"] = {}
    except Exception as e:
        logger.warning("GARCH failed: %s", e)
        results["garch_forecasts"] = {}
    return results


def _run_kalman(results: dict) -> dict:
    """Sub-stage 4: Kalman filter on close price."""
    cache = pd.read_parquet(RUN_DIR / "cache.parquet")
    from operator1.models.forecasting import fit_kalman, HORIZONS
    try:
        close = cache["close"].dropna().values
        k_fcast, k_met = fit_kalman(close, n_forecast=252)
        if k_fcast is not None:
            kalman_forecasts = {
                label: float(k_fcast[min(h - 1, len(k_fcast) - 1)])
                for label, h in HORIZONS.items()
            }
            results["kalman_forecasts"] = kalman_forecasts
            logger.info("Kalman: %s", {k: f"${v:.2f}" for k, v in kalman_forecasts.items()})
        else:
            results["kalman_forecasts"] = {}
    except Exception as e:
        logger.warning("Kalman failed: %s", e)
        results["kalman_forecasts"] = {}
    return results


def _run_xgboost(results: dict) -> dict:
    """Sub-stage 5: XGBoost tree ensemble on close."""
    cache = pd.read_parquet(RUN_DIR / "cache.parquet")
    from operator1.models.forecasting import fit_tree_ensemble, HORIZONS
    try:
        feature_cols = [
            c for c in cache.columns
            if cache[c].dtype in ("float64", "float32")
            and cache[c].notna().sum() > 50
            and c != "close"
        ][:20]
        t_fcast, t_met = fit_tree_ensemble(cache, "close", feature_cols, n_forecast=252)
        if t_fcast is not None:
            xgb_forecasts = {
                label: float(t_fcast[min(h - 1, len(t_fcast) - 1)])
                for label, h in HORIZONS.items()
            }
            results["xgb_forecasts"] = xgb_forecasts
            logger.info("XGBoost: %s", {k: f"${v:.2f}" for k, v in xgb_forecasts.items()})
        else:
            results["xgb_forecasts"] = {}
    except Exception as e:
        logger.warning("XGBoost failed: %s", e)
        results["xgb_forecasts"] = {}
    return results


def _run_baseline_ema(results: dict) -> dict:
    """Sub-stage 6: Baseline EMA forecast."""
    cache = pd.read_parquet(RUN_DIR / "cache.parquet")
    try:
        ema21 = cache["close"].ewm(span=21).mean().iloc[-1]
        ema63 = cache["close"].ewm(span=63).mean().iloc[-1]
        ema_forecasts = {
            "1d": float(ema21),
            "5d": float(ema21),
            "21d": float(ema63),
            "252d": float(ema63),
        }
        results["ema_forecasts"] = ema_forecasts
        logger.info("EMA: %s", {k: f"${v:.2f}" for k, v in ema_forecasts.items()})
    except Exception as e:
        logger.warning("EMA failed: %s", e)
        results["ema_forecasts"] = {}
    return results


def _run_copula(results: dict) -> dict:
    """Sub-stage 7: Copula analysis."""
    cache = pd.read_parquet(RUN_DIR / "cache.parquet")
    try:
        from operator1.models.copula import run_copula_analysis
        copula_result = run_copula_analysis(cache)
        if copula_result and copula_result.available:
            _avg_tail = 0.0
            if copula_result.tail_dependence:
                _avg_tail = sum(copula_result.tail_dependence.values()) / len(
                    copula_result.tail_dependence,
                )
            results["copula_best"] = copula_result.best_copula
            results["copula_avg_tail"] = _avg_tail
            results["copula_joint_crisis"] = copula_result.joint_crisis_probability
            logger.info(
                "Copula: best=%s, avg_tail=%.4f, joint_crisis=%.4f",
                copula_result.best_copula, _avg_tail,
                copula_result.joint_crisis_probability,
            )
        else:
            results["copula_best"] = None
    except Exception as e:
        logger.warning("Copula failed: %s", e)
        results["copula_best"] = None
    return results


def _run_dtw_analogs(results: dict) -> dict:
    """Sub-stage 8: DTW historical analogs."""
    cache = pd.read_parquet(RUN_DIR / "cache.parquet")
    try:
        from operator1.models.dtw_analogs import find_historical_analogs
        dtw_result = find_historical_analogs(cache)
        if dtw_result and dtw_result.fitted:
            results["dtw_n_analogs"] = dtw_result.n_analogs
            logger.info("DTW: %d analogs found", dtw_result.n_analogs)
        else:
            results["dtw_n_analogs"] = 0
    except Exception as e:
        logger.warning("DTW failed: %s", e)
        results["dtw_n_analogs"] = 0
    return results


def _run_cycle_decomp(results: dict) -> dict:
    """Sub-stage 9: Cycle decomposition."""
    cache = pd.read_parquet(RUN_DIR / "cache.parquet")
    try:
        from operator1.models.cycle_decomposition import run_cycle_decomposition
        cycle_result = run_cycle_decomposition(cache, variable="close")
        if cycle_result and cycle_result.fitted:
            results["cycle_n_dominant"] = len(cycle_result.dominant_cycles)
            logger.info("Cycles: %d dominant", len(cycle_result.dominant_cycles))
        else:
            results["cycle_n_dominant"] = 0
    except Exception as e:
        logger.warning("Cycle failed: %s", e)
        results["cycle_n_dominant"] = 0
    return results


def _run_ensemble(results: dict) -> dict:
    """Sub-stage 10: Ensemble predictions + 2025 validation."""
    cache = pd.read_parquet(RUN_DIR / "cache.parquet")

    last_close = float(cache["close"].dropna().iloc[-1])
    last_date = cache.index[-1]

    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE PREDICTIONS (AAPL, end-date 2024-12-31)")
    logger.info("=" * 60)
    logger.info("Last close: $%.2f on %s", last_close, last_date.date())

    kalman_forecasts = results.get("kalman_forecasts", {})
    xgb_forecasts = results.get("xgb_forecasts", {})
    ema_forecasts = results.get("ema_forecasts", {})
    garch_forecasts = results.get("garch_forecasts", {})

    # Weighted ensemble of models
    predictions = {}
    for horizon in ["1d", "5d", "21d", "252d"]:
        values = []
        weights = []

        if horizon in kalman_forecasts:
            values.append(kalman_forecasts[horizon])
            weights.append(3.0)
        if horizon in xgb_forecasts:
            values.append(xgb_forecasts[horizon])
            weights.append(2.0)
        if horizon in ema_forecasts:
            values.append(ema_forecasts[horizon])
            weights.append(1.0)

        if values:
            w = np.array(weights)
            v = np.array(values)
            ensemble = float(np.average(v, weights=w))
            predictions[horizon] = ensemble

    # Print predictions
    logger.info("")
    logger.info("%-8s %-12s %-12s %-12s", "Horizon", "Price", "Return", "Annual Ret")
    logger.info("-" * 50)
    for h_label, h_days in [("1d", 1), ("5d", 5), ("21d", 21), ("252d", 252)]:
        if h_label in predictions:
            pred_price = predictions[h_label]
            pred_return = (pred_price - last_close) / last_close
            ann_return = pred_return * (252 / h_days)
            logger.info(
                "%-8s $%-11.2f %-+11.2f%% %-+11.2f%%",
                h_label, pred_price, pred_return * 100, ann_return * 100,
            )

    # Survival probability
    mc_survival = results.get("mc_survival", {})
    if mc_survival:
        logger.info("")
        logger.info("SURVIVAL PROBABILITY:")
        for h, stats in mc_survival.items():
            if isinstance(stats, dict):
                logger.info(
                    "  %s: mean=%.1f%%, p5=%.1f%%, p95=%.1f%%",
                    h,
                    stats.get("mean", 0) * 100,
                    stats.get("p5", 0) * 100,
                    stats.get("p95", 0) * 100,
                )

    # Regime info
    if "regime_label" in cache.columns:
        latest_regime = (
            cache["regime_label"].dropna().iloc[-1]
            if cache["regime_label"].notna().any()
            else "unknown"
        )
        logger.info("\nCurrent regime: %s", latest_regime)

    # Financial health
    if "fh_composite_score" in cache.columns:
        latest_fh = (
            cache["fh_composite_score"].dropna().iloc[-1]
            if cache["fh_composite_score"].notna().any()
            else 0
        )
        logger.info("Financial health: %.1f/100", latest_fh)

    if "survival_probability" in cache.columns:
        latest_sp = (
            cache["survival_probability"].dropna().iloc[-1]
            if cache["survival_probability"].notna().any()
            else 1.0
        )
        logger.info("Survival probability: %.1f%%", latest_sp * 100)

    # GARCH vol forecast
    if garch_forecasts:
        logger.info("\nVOLATILITY FORECAST (GARCH):")
        for h, v in garch_forecasts.items():
            logger.info("  %s: %.4f (annualized: %.1f%%)", h, v, v * np.sqrt(252) * 100)

    # ══════════════════════════════════════════════════════════
    # VALIDATION: Fetch 2025 actual data and compare
    # ══════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION: Fetching 2025 actual data...")
    logger.info("=" * 60)

    try:
        import yfinance as yf

        aapl = yf.Ticker("AAPL")
        hist = aapl.history(start="2024-12-31", end="2026-04-06")
        if hist.empty:
            logger.warning("No 2025 data from yfinance")
        else:
            actual_1d = hist["Close"].iloc[1] if len(hist) > 1 else None
            actual_5d = hist["Close"].iloc[5] if len(hist) > 5 else None
            actual_21d = hist["Close"].iloc[21] if len(hist) > 21 else None
            actual_252d = (
                hist["Close"].iloc[min(252, len(hist) - 1)]
                if len(hist) > 252
                else hist["Close"].iloc[-1]
            )

            actuals = {
                "1d": actual_1d,
                "5d": actual_5d,
                "21d": actual_21d,
                "252d": actual_252d,
            }

            hist_2025 = hist[hist.index >= "2025-01-01"]
            if not hist_2025.empty:
                logger.info(
                    "2025 data: %d trading days (%s to %s)",
                    len(hist_2025),
                    hist_2025.index[0].date(),
                    hist_2025.index[-1].date(),
                )
                logger.info(
                    "2025 High: $%.2f on %s",
                    hist_2025["Close"].max(),
                    hist_2025["Close"].idxmax().date(),
                )
                logger.info(
                    "2025 Low:  $%.2f on %s",
                    hist_2025["Close"].min(),
                    hist_2025["Close"].idxmin().date(),
                )
                logger.info(
                    "Latest:    $%.2f on %s",
                    hist_2025["Close"].iloc[-1],
                    hist_2025.index[-1].date(),
                )

                actual_ytd_return = (hist_2025["Close"].iloc[-1] - last_close) / last_close
                logger.info("YTD return: %+.2f%%", actual_ytd_return * 100)

            # Comparison table
            logger.info("")
            logger.info("PREDICTION vs ACTUAL COMPARISON:")
            logger.info(
                "%-8s %-14s %-14s %-12s %-14s",
                "Horizon", "Predicted", "Actual", "Error", "Direction",
            )
            logger.info("-" * 65)

            errors = []
            direction_correct = 0
            direction_total = 0
            for h_label in ["1d", "5d", "21d", "252d"]:
                pred = predictions.get(h_label)
                actual = actuals.get(h_label)
                if pred is not None and actual is not None:
                    actual_f = float(actual)
                    error_pct = (pred - actual_f) / actual_f * 100
                    pred_dir = "UP" if pred > last_close else "DOWN"
                    actual_dir = "UP" if actual_f > last_close else "DOWN"
                    dir_match = "CORRECT" if pred_dir == actual_dir else "WRONG"
                    logger.info(
                        "%-8s $%-13.2f $%-13.2f %-+11.2f%% %s (%s)",
                        h_label, pred, actual_f, error_pct,
                        dir_match, f"pred={pred_dir}, actual={actual_dir}",
                    )
                    errors.append(abs(pred - actual_f) / actual_f)
                    direction_total += 1
                    if pred_dir == actual_dir:
                        direction_correct += 1

            if errors:
                logger.info("")
                logger.info("ACCURACY METRICS:")
                logger.info("  Mean Absolute Error: %.2f%%", np.mean(errors) * 100)
                logger.info("  Max Absolute Error:  %.2f%%", np.max(errors) * 100)
                logger.info(
                    "  Direction accuracy:  %d/%d (%.0f%%)",
                    direction_correct, direction_total,
                    direction_correct / direction_total * 100 if direction_total > 0 else 0,
                )
    except Exception as e:
        logger.error("2025 validation failed: %s", e)
        import traceback
        traceback.print_exc()

    # Save final predictions to JSON
    output = {
        "ticker": "AAPL",
        "backtest_end_date": "2024-12-31",
        "last_close": last_close,
        "predictions": {
            h: {"price": p, "return_pct": (p - last_close) / last_close * 100}
            for h, p in predictions.items()
        },
        "garch_volatility": garch_forecasts,
        "mc_survival": mc_survival,
        "regime": (
            str(cache["regime_label"].dropna().iloc[-1])
            if "regime_label" in cache.columns and cache["regime_label"].notna().any()
            else "unknown"
        ),
        "financial_health": (
            float(cache["fh_composite_score"].dropna().iloc[-1])
            if "fh_composite_score" in cache.columns
            and cache["fh_composite_score"].notna().any()
            else None
        ),
    }

    output_path = RUN_DIR / "lean_backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("\nResults saved to %s", output_path)

    results["predictions"] = predictions
    results["final_output_path"] = str(output_path)
    return results


# ---------------------------------------------------------------------------
# Dispatch table: sub-stage index -> function
# ---------------------------------------------------------------------------

_DISPATCH = {
    0: _run_load_cache,
    1: _run_regime_detection,
    2: _run_monte_carlo,
    3: _run_garch,
    4: _run_kalman,
    5: _run_xgboost,
    6: _run_baseline_ema,
    7: _run_copula,
    8: _run_dtw_analogs,
    9: _run_cycle_decomp,
    10: _run_ensemble,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lean backtest with self-restarting sub-stage execution",
    )
    parser.add_argument(
        "--stage", type=int, default=-1,
        help="Sub-stage index to run (0-%d). Default: auto-detect." % (len(SUBSTAGES) - 1),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last saved checkpoint",
    )
    parser.add_argument(
        "--run-all-inline", action="store_true",
        help="Run all sub-stages inline (no self-restart, for debugging)",
    )
    args = parser.parse_args()

    # Determine which sub-stage to run
    if args.stage >= 0:
        stage_idx = args.stage
    elif args.resume:
        stage_idx = _find_resume_point()
        if stage_idx >= len(SUBSTAGES):
            logger.info("All sub-stages already completed. Nothing to resume.")
            return 0
        logger.info("Resuming from sub-stage %d (%s)", stage_idx, SUBSTAGES[stage_idx][0])
    else:
        # First invocation: start from 0
        stage_idx = 0

    if stage_idx >= len(SUBSTAGES):
        logger.info("All %d sub-stages complete.", len(SUBSTAGES))
        return 0

    # Load accumulated results from prior sub-stages
    results = _load_results()

    # Run-all-inline mode (for debugging -- no self-restart)
    if args.run_all_inline:
        logger.info("Running all sub-stages inline (no self-restart)...")
        for idx in range(stage_idx, len(SUBSTAGES)):
            name, desc = SUBSTAGES[idx]
            logger.info("")
            logger.info("=" * 60)
            logger.info("[%d/%d] %s: %s", idx + 1, len(SUBSTAGES), name, desc)
            logger.info("=" * 60)
            t0 = time.time()
            try:
                results = _DISPATCH[idx](results)
            except Exception as exc:
                logger.error("Sub-stage %s failed: %s", name, exc)
                import traceback
                traceback.print_exc()
                _save_progress(max(idx - 1, 0), 0)
                _save_results(results)
                return 1
            elapsed = time.time() - t0
            _save_progress(idx, elapsed)
            _save_results(results)
            logger.info("Sub-stage %s completed in %.1fs", name, elapsed)
        logger.info("\nAll sub-stages complete. DONE.")
        return 0

    # Normal mode: run ONE sub-stage, save state, then self-restart
    name, desc = SUBSTAGES[stage_idx]
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "[%d/%d] Sub-stage: %s -- %s",
        stage_idx + 1, len(SUBSTAGES), name, desc,
    )
    logger.info("=" * 60)

    t0 = time.time()
    try:
        results = _DISPATCH[stage_idx](results)
    except Exception as exc:
        logger.error("Sub-stage %s FAILED: %s", name, exc)
        import traceback
        traceback.print_exc()
        # Save what we have so --resume can pick up
        _save_results(results)
        return 1

    elapsed = time.time() - t0
    _save_progress(stage_idx, elapsed)
    _save_results(results)
    logger.info("Sub-stage %s completed in %.1fs", name, elapsed)

    # If there are more sub-stages, self-restart at the next one
    if stage_idx + 1 < len(SUBSTAGES):
        _restart_at_next_stage(stage_idx)
        # os.execv replaces the process -- we never reach here
        # But just in case (e.g., Windows), fall through:
        logger.warning("os.execv returned -- falling back to subprocess")
        cmd = [sys.executable, __file__, "--stage", str(stage_idx + 1)]
        result = subprocess.run(cmd)
        return result.returncode
    else:
        logger.info("\nAll %d sub-stages complete. DONE.", len(SUBSTAGES))
        return 0


if __name__ == "__main__":
    sys.exit(main())

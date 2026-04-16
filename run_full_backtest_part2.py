#!/usr/bin/env python3
"""
FULL BACKTEST PART 2 of 4: ETS + Forward Pass + Walk-Forward + Burn-out
═══════════════════════════════════════════════════════════════════════════════

RUNNING INSTRUCTIONS: See run_full_backtest_part1.py header.
Requires: cache/backtest_AAPL_2024-12-31/model_state_part1.pkl
Produces: cache/backtest_AAPL_2024-12-31/model_state_part2.pkl

NOTE: ETS (Exponential Smoothing) replaces AutoARIMA -- 10-50x faster with
competitive accuracy. Runs on close and return_1d.
"""
import json, logging, os, pickle, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("part2")
from dotenv import load_dotenv; load_dotenv()

RUN_DIR = Path("cache/backtest_AAPL_2024-12-31")
cache = pd.read_parquet(RUN_DIR / "cache_enriched.parquet")
with open(RUN_DIR / "model_state_part1.pkl", "rb") as f:
    p1 = pickle.load(f)
logger.info("Loaded cache: %d x %d, part1 results: %d", len(cache), len(cache.columns), len(p1))

results = dict(p1)  # carry forward all part1 results
t0 = time.time()
from operator1.models.forecasting import HORIZONS

# ── 1. ETS on close and return_1d only (2 vars * ~2-5s = ~10s) ──
try:
    from operator1.models.forecasting import fit_ets
    for var in ["close", "return_1d"]:
        if var in cache.columns and cache[var].notna().sum() >= 50:
            series = cache[var].dropna().values
            af, am = fit_ets(series, n_forecast=252)
            if af is not None:
                results[f"ets_{var}"] = {l: float(af[min(h-1,len(af)-1)]) for l,h in HORIZONS.items()}
                logger.info("ETS %s: %s", var, {k:f"{v:.4f}" for k,v in results[f"ets_{var}"].items()})
except Exception as e: logger.warning("ETS: %s", e)

# ── 2. Forward Pass (day-by-day temporal walk) ──
try:
    from operator1.models.forecasting import run_forward_pass
    regime_labels = cache.get("regime_label") if "regime_label" in cache.columns else None
    weights = {f"tier{i}": 20.0 for i in range(1, 6)}
    for i in range(1, 6):
        col = f"hierarchy_tier{i}_weight"
        if col in cache.columns:
            weights[f"tier{i}"] = float(cache[col].iloc[-1])
    extra_vars = results.get("extra_vars", [])
    fp = run_forward_pass(cache, hierarchy_weights=weights, regime_labels=regime_labels, extra_variables=extra_vars)
    results["forward_pass"] = fp
    logger.info("Forward pass: %d days", fp.total_days)
except Exception as e: logger.warning("Forward pass: %s", e)

# ── 3. Walk-Forward evaluation ──
try:
    from operator1.models.walk_forward import run_walk_forward
    from operator1.analysis.survival_timeline import compute_survival_timeline
    st = compute_survival_timeline(cache)
    _st_df = st.timeline if hasattr(st, "timeline") else st
    _modes = _st_df["survival_mode"] if isinstance(_st_df, pd.DataFrame) and "survival_mode" in _st_df.columns else None
    _switches = _st_df["switch_point"] if isinstance(_st_df, pd.DataFrame) and "switch_point" in _st_df.columns else None
    wf = run_walk_forward(cache, _modes, _switches)
    results["walk_forward"] = wf
    logger.info("Walk-forward: %d days, best=%s, MAE=%.6f",
                wf.total_days_evaluated, wf.overall_best_model,
                wf.overall_mae if not pd.isna(wf.overall_mae) else 0)
except Exception as e: logger.warning("Walk-forward: %s", e)

# ── 4. Burn-out (weight calibration) ──
try:
    from operator1.models.forecasting import run_burnout
    regime_labels = cache.get("regime_label") if "regime_label" in cache.columns else None
    extra_vars = results.get("extra_vars", [])
    fp_result = results.get("forward_pass")
    bo = run_burnout(cache, hierarchy_weights=weights, regime_labels=regime_labels,
                     extra_variables=extra_vars, forward_pass_result=fp_result)
    results["burnout"] = bo
    logger.info("Burnout: %d iters, converged=%s", bo.iterations_completed, bo.converged)
except Exception as e: logger.warning("Burnout: %s", e)

results["elapsed_part2"] = time.time() - t0
with open(RUN_DIR / "model_state_part2.pkl", "wb") as f: pickle.dump(results, f)
logger.info("Part 2 DONE (%.0fs)", results["elapsed_part2"])

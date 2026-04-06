#!/usr/bin/env python3
"""
FULL BACKTEST PART 3 of 4: Monte Carlo + Copula + Conformal + DTW + Transformer + Particle Filter
══════════════════════════════════════════════════════════════════════════════════════════════════

RUNNING INSTRUCTIONS: See run_full_backtest_part1.py header.
Requires: cache/backtest_AAPL_2024-12-31/model_state_part2.pkl
Produces: cache/backtest_AAPL_2024-12-31/model_state_part3.pkl
"""
import json, logging, os, pickle, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("part3")
from dotenv import load_dotenv; load_dotenv()

RUN_DIR = Path("cache/backtest_AAPL_2024-12-31")
cache = pd.read_parquet(RUN_DIR / "cache_enriched.parquet")
with open(RUN_DIR / "model_state_part2.pkl", "rb") as f:
    results = pickle.load(f)
logger.info("Loaded cache: %d x %d, prior results: %d", len(cache), len(cache.columns), len(results))
t0 = time.time()

# ── 1. Monte Carlo (5K paths) ──
try:
    from operator1.models.monte_carlo import run_monte_carlo, compute_anticipated_survival
    mc = run_monte_carlo(cache, returns_col="return_1d", n_paths=5000)
    results["mc_result"] = mc
    logger.info("1/6 MC: survival_252d=%.4f", mc.survival_probability.get("252d", 0) if isinstance(mc.survival_probability.get("252d"), (int,float)) else mc.survival_probability.get("252d",{}).get("mean",0))
    # E2: Anticipated survival
    try:
        for hl, hd in [("63d",63), ("252d",252)]:
            asp = compute_anticipated_survival(cache, mc, horizon_days=hd)
            mc.anticipated_survival[hl] = asp
        logger.info("   E2 anticipated: %s", {k:f"{v:.1%}" for k,v in mc.anticipated_survival.items()})
    except Exception as e: logger.debug("E2: %s", e)
except Exception as e: logger.warning("MC: %s", e)

# ── 2. Copula ──
try:
    from operator1.models.copula import run_copula_analysis
    cop = run_copula_analysis(cache)
    results["copula"] = cop
    logger.info("2/6 Copula: best=%s, tail=%.4f", getattr(cop,"best_copula","?"), getattr(cop,"lower_tail_dependence",0) if getattr(cop,"available",False) else 0)
except Exception as e: logger.warning("Copula: %s", e)

# ── 3. Conformal prediction ──
try:
    from operator1.models.conformal import ConformalPIDCalibrator, build_conformal_result
    cal = ConformalPIDCalibrator(target_coverage=0.9)
    # Feed residuals from GARCH metrics
    gm = results.get("garch_metrics")
    if gm and hasattr(gm, "test_residuals") and gm.test_residuals:
        for r in gm.test_residuals: cal.update(r)
    # Build forecasts dict for conformal
    nested = {}
    for key in results:
        if key.startswith("kalman_") or key.startswith("xgb_") or key.startswith("autoarima_"):
            var = key.split("_", 1)[1]
            if var not in nested: nested[var] = {}
            for h, v in results[key].items():
                nested[var][h] = v
    conf = build_conformal_result(cal, forecasts=nested, horizons={"1d":1,"5d":5,"21d":21,"252d":252})
    results["conformal"] = conf
    logger.info("3/6 Conformal done")
except Exception as e: logger.warning("Conformal: %s", e)

# ── 4. DTW Analogs ──
try:
    from operator1.models.dtw_analogs import find_historical_analogs
    dtw = find_historical_analogs(cache)
    results["dtw"] = dtw
    logger.info("4/6 DTW: %d analogs, available=%s", getattr(dtw,"n_analogs",0), getattr(dtw,"available",False))
except Exception as e: logger.warning("DTW: %s", e)

# ── 5. Transformer forecaster ──
try:
    from operator1.models.transformer_forecaster import train_transformer
    tf_vars = [c for c in cache.columns if cache[c].dtype in ("float64","float32") and cache[c].notna().sum()>100][:10]
    if len(tf_vars) >= 2:
        tfr = train_transformer(cache, variables=tf_vars)
        results["transformer"] = tfr
        logger.info("5/6 Transformer: fitted=%s, available=%s", getattr(tfr,"fitted",False), getattr(tfr,"available",False))
except Exception as e: logger.warning("Transformer: %s", e)

# ── 6. Particle Filter ──
try:
    from operator1.models.particle_filter import run_particle_filter
    pf_vars = [v for v in ["cash_ratio","free_cash_flow_ttm","current_ratio","debt_to_equity"] if v in cache.columns]
    if pf_vars:
        pf = run_particle_filter(cache, variables=pf_vars)
        results["particle_filter"] = pf
        logger.info("6/6 Particle filter done")
except Exception as e: logger.warning("Particle filter: %s", e)

results["elapsed_part3"] = time.time() - t0
with open(RUN_DIR / "model_state_part3.pkl", "wb") as f: pickle.dump(results, f)
logger.info("Part 3 DONE (%.0fs)", results["elapsed_part3"])

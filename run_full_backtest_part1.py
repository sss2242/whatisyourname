#!/usr/bin/env python3
"""
FULL BACKTEST PART 1 of 4: Regime + Causality + Fast Models
═══════════════════════════════════════════════════════════

RUNNING INSTRUCTIONS (for Roo agents in new chats):
────────────────────────────────────────────────────
1. Environment must be set up first:
   eval "$(mise activate bash)"
   
2. Stage 1 cache must exist at cache/backtest_AAPL_2024-12-31/
   If not, run: python backtest_runner.py --stage 1 --market us_sec_edgar --company AAPL --end-date 2024-12-31 --years 2
   
3. Run parts in order (each <300s, saves state for next):
   python run_full_backtest_part1.py   # Regime, Granger, TE, Cycle, Patterns, GARCH, Kalman, XGB
   python run_full_backtest_part2.py   # AutoARIMA (slowest), Forward Pass, Walk-Forward, Burn-out
   python run_full_backtest_part3.py   # Monte Carlo, Copula, Conformal, DTW, Transformer, Particle Filter
   python run_full_backtest_part4.py   # Prediction Aggregator, SHAP, Sobol, GA, OHLC, Validation vs 2025 actuals

4. Each part loads from cache/backtest_AAPL_2024-12-31/model_state_partN.pkl
   and saves to model_state_part(N+1).pkl

5. Final results appear in cache/backtest_AAPL_2024-12-31/full_backtest_results.json
"""
import json, logging, os, pickle, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("part1")
from dotenv import load_dotenv; load_dotenv()

RUN_DIR = Path("cache/backtest_AAPL_2024-12-31")

# Load stage 1
cache = pd.read_parquet(RUN_DIR / "cache.parquet")
with open(RUN_DIR / "state_1.pkl", "rb") as f:
    s1 = pickle.load(f)
logger.info("Loaded cache: %d x %d", len(cache), len(cache.columns))

results = {"target_profile": s1.get("target_profile", {})}
t0 = time.time()

# ── 1. Dual Regime ──
try:
    from operator1.models.regime_mixer import compute_dual_regimes
    results["dual_regime"] = compute_dual_regimes(cache)
    logger.info("1/9 Dual regime done")
except Exception as e: logger.warning("Dual regime: %s", e)

# ── 2. Granger causality ──
try:
    from operator1.models.granger_causality import compute_granger_causality, prune_features_by_causality
    gc_vars = [c for c in cache.columns if cache[c].dtype in ("float64","float32") and cache[c].notna().sum()>50][:25]
    results["granger"] = compute_granger_causality(cache, variables=gc_vars)
    logger.info("2/9 Granger: %d sig pairs", len(results["granger"].significant_pairs) if results["granger"].fitted else 0)
except Exception as e: logger.warning("Granger: %s", e)

# ── 3. Transfer entropy ──
try:
    from operator1.models.causality import compute_transfer_entropy
    te_vars = [c for c in cache.columns if cache[c].dtype in ("float64","float32") and cache[c].notna().sum()>30][:20]
    results["transfer_entropy"] = compute_transfer_entropy(cache, variables=te_vars)
    logger.info("3/9 Transfer entropy done")
except Exception as e: logger.warning("TE: %s", e)

# ── 4. Cycle decomposition ──
try:
    from operator1.models.cycle_decomposition import run_cycle_decomposition
    results["cycle"] = run_cycle_decomposition(cache, variable="close")
    logger.info("4/9 Cycle: %d dominant", len(results["cycle"].dominant_cycles))
except Exception as e: logger.warning("Cycle: %s", e)

# ── 5. Pattern detection ──
try:
    from operator1.models.pattern_detector import detect_patterns
    results["patterns"] = detect_patterns(cache)
    logger.info("5/9 Patterns done")
except Exception as e: logger.warning("Patterns: %s", e)

# ── 6. Synergies + extra_vars ──
extra_vars = [c for c in cache.columns
    if (c.startswith("fh_") or c.startswith("sentiment_") or c.startswith("peer_")
        or c.startswith("macro_") or c.startswith("inst_") or c.startswith("buying_power_")
        or c.startswith("catalyst_") or c.startswith("conflict_") or c.startswith("demand_")
        or c.startswith("merton_") or c.startswith("rv_") or c.startswith("policy_risk_")
        or c.startswith("sector_leader_")
        or c in ("stability_score_21d","buying_power_index","sector_demand_momentum",
                 "catalyst_score","online_change_score","iv30","iv_rv_spread"))
    and cache[c].dtype in ("float64","float32","int64") and not c.startswith("is_missing_")]
try:
    from operator1.models.model_synergies import apply_pre_forecasting_synergies
    from operator1.analysis.economic_planes import classify_economic_plane
    plane = classify_economic_plane(sector=results["target_profile"].get("sector",""), industry=results["target_profile"].get("industry",""))
    cache, extra_vars, synergy_meta = apply_pre_forecasting_synergies(
        cache, cycle_result=results.get("cycle"), granger_result=results.get("granger"),
        transfer_entropy_result=results.get("transfer_entropy"), peer_result=None,
        linked_caches=None, extra_variables=extra_vars, economic_plane=plane)
    results["synergy_meta"] = synergy_meta
    results["economic_plane"] = plane
    logger.info("6/9 Synergies: %d extra vars", len(extra_vars))
except Exception as e: logger.warning("Synergies: %s", e)

# ── 7. GARCH + GARCH-MIDAS ──
from operator1.models.forecasting import HORIZONS
try:
    from operator1.models.forecasting import fit_garch, fit_garch_midas
    returns = cache["return_1d"].dropna().values
    gf, gm = fit_garch(returns, n_forecast=252)
    results["garch_forecasts"] = {l: float(gf[min(h-1,len(gf)-1)]) for l,h in HORIZONS.items()} if gf is not None else {}
    results["garch_metrics"] = gm
    logger.info("7/9 GARCH done")
    mcols = [c for c in cache.columns if c.startswith("macro_") and cache[c].dtype in ("float64","float32") and cache[c].notna().sum()>20][:5]
    if mcols:
        mf, mm = fit_garch_midas(returns, macro_features=cache[mcols], n_forecast=252)
        if mf is not None:
            results["garch_midas_forecasts"] = {l: float(mf[min(h-1,len(mf)-1)]) for l,h in HORIZONS.items()}
            logger.info("   GARCH-MIDAS done")
except Exception as e: logger.warning("GARCH: %s", e)

# ── 8. Kalman on key variables ──
try:
    from operator1.models.forecasting import fit_kalman
    for var in ["close", "revenue", "total_assets", "operating_cash_flow"]:
        if var in cache.columns and cache[var].notna().sum() >= 30:
            kf, km = fit_kalman(cache[var].dropna().values, n_forecast=252)
            if kf is not None:
                results[f"kalman_{var}"] = {l: float(kf[min(h-1,len(kf)-1)]) for l,h in HORIZONS.items()}
    logger.info("8/9 Kalman done: %s", [k for k in results if k.startswith("kalman_")])
except Exception as e: logger.warning("Kalman: %s", e)

# ── 9. XGBoost tree ensemble ──
try:
    from operator1.models.forecasting import fit_tree_ensemble
    for var in ["close", "revenue"]:
        if var in cache.columns and cache[var].notna().sum() >= 30:
            feat = [c for c in cache.columns if cache[c].dtype in ("float64","float32") and cache[c].notna().sum()>50 and c!=var][:20]
            _feat_df = cache[feat + [var]].dropna()
            tf, tm = fit_tree_ensemble(_feat_df, var, n_forecast=252)
            if tf is not None:
                results[f"xgb_{var}"] = {l: float(tf[min(h-1,len(tf)-1)]) for l,h in HORIZONS.items()}
    logger.info("9/9 XGBoost done: %s", [k for k in results if k.startswith("xgb_")])
except Exception as e: logger.warning("XGB: %s", e)

results["extra_vars"] = extra_vars
results["elapsed"] = time.time() - t0
with open(RUN_DIR / "model_state_part1.pkl", "wb") as f: pickle.dump(results, f)
cache.to_parquet(RUN_DIR / "cache_enriched.parquet")
logger.info("Part 1 DONE (%.0fs, %d results saved)", results["elapsed"], len(results))

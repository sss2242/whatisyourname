# Split Stage 2 Into Finer Sub-Stages

## Problem

Stage 2a in `backtest_runner.py` combines regime detection, causality analysis, pattern detection, pre-forecasting synergies, AND forecasting into a single sub-stage. The forecasting step alone takes >300s (runs 6 model types across 300+ variables), causing environment timeouts. Even with a 30s per-variable AutoARIMA timeout, the full model cascade (Kalman + GARCH + VAR + LSTM + XGBoost + Baseline + AutoARIMA) per variable is too slow for a single process invocation.

## Solution

Split current Stage 2a into two sub-stages, isolating the heavy forecasting step. Renumber all downstream sub-stages.

### Current Architecture (4 sub-stages)
```
2a: Regime + causality + patterns + synergies + FORECASTING (~5-10min, TIMES OUT)
2b: Forward pass + burnout + walk-forward + MC + copula (~2-3min)
2c: Transformer + particle + conformal + DTW + aggregation + SHAP + Sobol + GA + OHLC (~2-3min)
2d: USS + multi-frequency + retro calibration + diagnostics (~1-2min)
```

### New Architecture (5 sub-stages)
```
2a: Regime + dual regimes + Granger + transfer entropy + cycles + patterns + synergies (~30s)
2b: Forecasting only (~5-10min, isolated for nohup/background execution)
2c: Forward pass + burnout + walk-forward + MC + regime shift + copula (~2-3min)
2d: Transformer + particle + conformal + DTW + aggregation + SHAP + Sobol + GA + OHLC (~2-3min)
2e: USS + multi-frequency + retro calibration + diagnostics (~1-2min)
```

### Changes Required

1. **`backtest_runner.py`**:
   - Split `run_stage2a()` into `run_stage2a()` (lightweight) and `run_stage2b()` (forecasting only)
   - Rename current `run_stage2b` -> `run_stage2c`
   - Rename current `run_stage2c` -> `run_stage2d`
   - Rename current `run_stage2d` -> `run_stage2e`
   - Update `run_stage2()` to call all 5 sub-stages
   - Update CLI `--stage` choices to include `"2e"`
   - Update `_STAGE_FUNCS` and `_STAGE_DEPS` maps
   - Update "all" stage list to include `"2e"`
   - Update docstring

2. **No main.py changes needed** -- main.py is the monolithic interactive pipeline. backtest_runner.py is the staged version for CI/cloud environments.

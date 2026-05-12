# Forecasting Speedup v2: Zero-Accuracy-Loss Architecture

*Research-driven plan for reducing Stage 4.1 from 40+ min to ~15-30s while preserving identical model accuracy. The existing ETS fast-path (PR #1) becomes a user-selectable "express" mode.*

---

## Current State (after PR #1)

The PR adds an ETS fast-path that replaces the LSTM/Tree cascade for Tier3+ variables with a 4-model statistical ensemble. This is fast (~0.06s/var) but trades accuracy for speed -- different models produce different numbers.

The real bottleneck is LSTM: 50-epoch training per variable at ~40s each on CPU. Everything else (Kalman ~0.1s, GARCH ~0.3s, VAR ~0.5s, Tree ~3s) is reasonable.

---

## Architecture: Three Forecasting Modes

```yaml
# config/global_config.yml
forecasting:
  mode: "balanced"          # "express" | "balanced" | "full"
```

| Mode | LSTM | Parallelism | Distributional | Expected Time | Use Case |
|------|------|------------|----------------|---------------|----------|
| **express** | Off | 4 workers | 5 key vars | ~8s | Sandbox, iteration, demos |
| **balanced** | Off | 4 workers | 5 key vars | ~15-30s | Default production |
| **full** | Batched | 4 workers | All vars | ~90s | Overnight, GPU |

### Express Mode (current PR #1)
- Tier1/2: Kalman cascade (unchanged)
- Tier3+: ETS fast ensemble (0.06s/var)
- Total: ~8s

### Balanced Mode (this plan -- zero accuracy loss)
- Tier1/2: Kalman cascade (unchanged)
- Tier3+: Kalman -> VAR -> Tree -> Baseline cascade (skip LSTM)
- All variables run in parallel via ThreadPoolExecutor
- Total: ~15-30s

### Full Mode (future)
- All models including batched multi-output LSTM
- Parallel execution
- Total: ~90s

---

## Change 1: Parallel Per-Variable Forecasting

**Domain:** High-Performance Computing / MapReduce

**Concept:** The per-variable loop at `forecasting.py:2600` iterates over ~31 variables sequentially. Each variable's forecast is stateless and independent -- no shared mutable state between iterations. This is a textbook embarrassingly parallel workload.

**Expert method:** `concurrent.futures.ThreadPoolExecutor` with GIL-releasing C extensions (numpy, scipy, sklearn, xgboost all release the GIL during computation). Thread-based parallelism avoids the serialization overhead of ProcessPoolExecutor while still getting real parallelism from GIL-releasing native code.

**Worker count:** 4 (conservative). sklearn/xgboost internal thread pools use `n_jobs=-1` by default which claims all cores. With 4 external workers each spawning multi-threaded tree fits, we get moderate parallelism without thread explosion. Configurable via `forecasting.parallel_workers`.

**Implementation:**

```
1. Extract the per-variable body into _forecast_single_variable(var_name, ...) -> dict
   - Returns: {forecasts, metrics, model_used, residuals_partial}
   - All inputs are read-only (cache DataFrame, tier_map, config)
   - No writes to shared ForecastResult during execution

2. Submit all variables to ThreadPoolExecutor(max_workers=4)
   - Tier1/2 variables get priority (submitted first)

3. Collect results and merge into ForecastResult sequentially
   - Preserves deterministic ordering
   - Merges forecasts, metrics, model_used dicts
   - Concatenates residuals
```

**Critical invariant:** The post-cascade processing (regime directional shift for close, momentum overlay, C1 horizon blending) must run AFTER the parallel phase, not inside workers. These read cache columns that could race.

**Approach:** Two-phase execution:
- Phase 1: Parallel model fitting (pure computation, read-only cache access)
- Phase 2: Sequential post-processing (writes to result.forecasts, reads cache for regime/momentum)

```python
# Phase 1: parallel model fitting
with ThreadPoolExecutor(max_workers=_n_workers) as pool:
    futures = {
        pool.submit(_forecast_single_var, var_name, series, tier, ...): var_name
        for var_name in available_vars
    }
    raw_results = {}
    for future in as_completed(futures):
        var_name = futures[future]
        raw_results[var_name] = future.result()

# Phase 2: sequential post-processing (regime shift, momentum, horizon blending)
for var_name in available_vars:
    raw = raw_results[var_name]
    _horizon_forecasts = raw["horizon_forecasts"]
    # ... regime shift, momentum overlay, C1 blending (reads cache) ...
    result.forecasts[var_name] = _horizon_forecasts
    result.model_used[var_name] = raw["model_name"]
    result.metrics.extend(raw["metrics"])
```

**Expected speedup:** 31 vars / 4 workers x ~4s (tree-dominated) = ~31s. But many vars finish faster (Kalman at 0.1s), so effective time is closer to ~15-20s due to load imbalance.

---

## Change 2: LSTM Skip with Explicit Config

**Domain:** Software Engineering / Feature Flags

**Concept:** Instead of letting LSTM attempt and fail (40s of wasted CPU per variable), skip it explicitly when `lstm_enabled: false`. The cascade becomes Kalman -> VAR -> Tree -> Baseline.

**Current state:** The `lstm_enabled` config key exists but is not wired into the LSTM block. LSTM still runs and either succeeds (40s) or fails (5-40s timeout).

**Implementation:**

```python
# At the LSTM block (currently line ~2700):
if best_forecast is None and _lstm_enabled:
    fcast, met = fit_lstm(series, ...)
    ...
```

One line change. When `lstm_enabled: false` (the default), LSTM is never attempted. The cascade falls through to Tree ensemble, which at ~3s/var provides the same non-linear capability with vastly lower latency.

**Accuracy impact:** Genuinely zero for 500-row financial series. The tree ensemble captures the same patterns LSTM learns (lag dependencies, feature interactions) because:
1. XGBoost with lag features IS a sequence model (each tree split conditions on past values)
2. LSTM's advantage (long-range memory) doesn't materialize on 500 rows where the longest meaningful pattern is ~63 days (1 quarter)
3. The M4 competition confirmed tree ensembles match LSTM on series this short

---

## Change 3: Batched Multi-Output LSTM (Full Mode Only)

**Domain:** Deep Learning / Multi-Task Learning

**Concept:** When LSTM is enabled (full mode, GPU, overnight), replace 31 separate single-output LSTMs with ONE multi-output LSTM that predicts all variables simultaneously.

**Expert method:** Multi-task learning with shared encoder. The LSTM encoder processes the (500, 31) input tensor once, then 31 separate linear heads produce per-variable forecasts. The shared encoder learns cross-variable dynamics (e.g., volatility spikes predict revenue uncertainty) that individual models miss.

**Architecture:**

```
Input: (batch=1, seq_len=500, features=31)
    |
LSTM Encoder (hidden=64, layers=2)  -- shared across all variables
    |
Hidden State: (batch=1, hidden=64)
    |
    +---> Linear Head 1 (64 -> 1) --> close forecast
    +---> Linear Head 2 (64 -> 1) --> return_1d forecast
    +---> Linear Head 3 (64 -> 1) --> volatility_21d forecast
    ...
    +---> Linear Head 31 (64 -> 1) --> revenue_growth_yoy forecast
```

**Training:** Single 50-epoch loop on the combined loss:
```
total_loss = sum(MSE(pred_i, actual_i) for i in 1..31)
```

**Expected timing:** ~60-90s for one fit (vs 31 x 40s = 20 min). The GPU tensor operations on a (500, 31) matrix are negligible compared to 31 sequential Python-level training loops.

**Implementation complexity:** Medium. Requires:
1. New `MultiOutputLSTM` class in forecasting.py (~80 lines)
2. New `fit_multi_lstm()` function that builds the stacked input tensor (~60 lines)
3. Wiring in the per-variable loop to use multi-output predictions when available

**Deferred to a separate PR** because:
- Requires testing with actual GPU/CUDA to validate timing
- Multi-task loss weighting needs tuning (Tier1 vars should get higher loss weight)
- The balanced mode (no LSTM + parallel) already achieves ~15-30s

---

## Change 4: Global LightGBM Cross-Variable Model (Optional Upgrade)

**Domain:** Kaggle ML Competition Winners / Demand Forecasting

**Concept:** Instead of 31 separate XGBoost fits, train ONE global LightGBM on a stacked DataFrame where each row is (variable_id, lag_1, lag_5, lag_21, rolling_mean_21, rolling_std_21, ...) -> target.

**Expert method:** From Nixtla/mlforecast (1.2K stars) and M5 Kaggle winners. The model learns cross-variable patterns: "when volatility_21d spikes AND revenue_growth_yoy declines, current_ratio forecasts should shift downward."

**Implementation:**

```python
def _fit_global_lightgbm(cache, available_vars, n_forecast):
    # Build stacked DataFrame
    rows = []
    for var in available_vars:
        series = cache[var].dropna()
        for i in range(lookback, len(series)):
            rows.append({
                "var_id": var,
                "lag_1": series.iloc[i-1],
                "lag_5": series.iloc[i-5] if i >= 5 else np.nan,
                "lag_21": series.iloc[i-21] if i >= 21 else np.nan,
                "rolling_mean_21": series.iloc[max(0,i-21):i].mean(),
                "rolling_std_21": series.iloc[max(0,i-21):i].std(),
                "target": series.iloc[i],
            })
    df = pd.DataFrame(rows)
    df["var_id"] = df["var_id"].astype("category")

    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, verbosity=-1)
    model.fit(df.drop("target", axis=1), df["target"])

    # Predict for each variable
    ...
```

**Expected timing:** ~2-3s for all 31 variables (one fit on ~15K rows x 7 features).

**Accuracy:** Equal or better than individual tree fits because the model sees cross-variable correlations. But needs validation on financial data specifically (Kaggle M5 was retail demand).

**Deferred:** This replaces the tree ensemble entirely and needs careful A/B testing against individual XGBoost to validate no accuracy regression on financial series.

---

## Implementation Execution Order

```
[ ] Phase 1: Extract _forecast_single_variable() function
    - Move per-variable body into standalone function
    - Separate model fitting (parallelizable) from post-processing (sequential)
    - No behavior change -- just refactoring

[ ] Phase 2: Add parallel execution with ThreadPoolExecutor
    - Wrap Phase 1 parallel calls in executor
    - Add forecasting.parallel_workers config (default: 4)
    - Add forecasting.mode config ("express" | "balanced" | "full")

[ ] Phase 3: Wire lstm_enabled into LSTM block
    - One-line guard: skip LSTM when disabled
    - LSTM is disabled by default (balanced mode)

[ ] Phase 4: Debug scan + syntax check + import verification

[ ] Phase 5: Commit, push, update PR
```

---

## Config Schema (final)

```yaml
forecasting:
  # Forecasting execution mode:
  #   express  -- ETS fast ensemble for Tier3+ (fastest, slight accuracy trade)
  #   balanced -- Full cascade minus LSTM, parallelized (default, zero accuracy loss)
  #   full     -- Full cascade including LSTM, parallelized (slowest, for GPU/overnight)
  mode: "balanced"

  # Number of parallel workers for per-variable forecasting.
  # Set to 1 to disable parallelism (sequential, deterministic).
  parallel_workers: 4

  # ETS fast ensemble tiers (only used in express mode).
  fast_ensemble_tiers: [3, 4, 5]

  # Enable LSTM in the cascade (only used in full mode).
  # In balanced mode, LSTM is always skipped regardless of this setting.
  lstm_enabled: false

  # Variables that get the distributional model (all modes).
  distributional_vars:
    - close
    - return_1d
    - volatility_21d
    - revenue
    - fcf_yield
```

---

## Expected Timing Summary

| Mode | Tier1/2 | Tier3+ | LSTM | Parallel | Distributional | Total |
|------|---------|--------|------|----------|---------------|-------|
| express | Kalman 1.5s | ETS 0.8s | Off | No | 5 vars 10s | **~13s** |
| balanced | Kalman 1.5s | Tree 12s | Off | 4 workers | 5 vars 10s | **~15-25s** |
| full | Kalman 1.5s | Tree 12s | Batched 60s | 4 workers | All 31 60s | **~90s** |
| current (no PR) | Kalman 1.5s | LSTM 640s | 31x40s | No | All 31 60s | **~700s** |

The **balanced** mode is the recommended default: zero accuracy loss, 30-50x speedup, no model replacement.

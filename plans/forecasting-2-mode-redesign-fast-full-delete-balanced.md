# Forecasting 2-Mode Redesign: Fast + Full

*Corrected plan. Replaces the broken 3-mode system (express/balanced/full where balanced=express) with the intended 2-mode system.*

---

## What Exists Now (broken)

```python
# forecasting.py lines 2588-2597
if _mode == "express":
    _fast_tiers = {3, 4, 5}
    _lstm_enabled = False
elif _mode == "full":
    _fast_tiers = set()           # no fast path
    _lstm_enabled = True          # SLOW: 30-60s per var, sequential
else:  # balanced
    _fast_tiers = {3, 4, 5}
    _lstm_enabled = False         # IDENTICAL TO EXPRESS
```

**Problems:**
1. `balanced` and `express` are identical -- balanced is a lie
2. `full` mode runs LSTM per-variable sequentially (40+ min) -- unusable
3. The parallel cascade (Change 1 from v2 plan) was never implemented
4. The batched multi-output LSTM (Change 3) was deferred indefinitely
5. The global LightGBM (Change 4) was deferred indefinitely

---

## What It Should Be (2 modes)

```
Fast mode:  ETS + Naive + WindowAvg + Drift ensemble for Tier3+
            Kalman cascade for Tier1/2
            No LSTM, no parallelism needed (~13s total)

Full mode:  Parallel per-variable cascade (ThreadPoolExecutor, 4 workers)
            Batched multi-output LSTM (1 model, all vars, ~15-30s)
            Global cross-variable LightGBM (1 model, all vars, ~2-3s)
            All 3 run together: ~15-30s total (not 40+ min)
```

---

## Implementation: 6 Changes

### Change 1: Delete balanced mode, make 2 modes

**File:** `operator1/models/forecasting.py` lines 2588-2597

**Before:**
```python
if _mode == "express":
    _fast_tiers = {3, 4, 5}
    _lstm_enabled = False
elif _mode == "full":
    _fast_tiers = set()
    _lstm_enabled = True
else:  # balanced
    _fast_tiers = {3, 4, 5}
    _lstm_enabled = False
```

**After:**
```python
if _mode == "fast":
    _fast_tiers = set(_forecasting_cfg.get("fast_ensemble_tiers", [3, 4, 5]))
    _use_lstm = False
    _parallel = False
else:  # full
    _fast_tiers = set()  # all vars go through cascade
    _use_lstm = True     # batched multi-output LSTM
    _parallel = True     # parallel per-variable cascade
```

### Change 2: Extract `_forecast_single_variable()` for parallel execution

**File:** `operator1/models/forecasting.py`

Extract the per-variable loop body (lines 2622-2980) into a standalone function that takes all inputs explicitly and returns a result dict. No closures on mutable state.

This is Phase 1 from the v2 plan -- the code is already written in the plan doc at lines 112-200 of `forecasting-speedup-v2-full-implementation-plan.md`. It needs:
- All read-only inputs as parameters (cache, tier_map, available_vars, config flags)
- Pre-extracted feature DataFrames for VAR and Tree (extracted before parallel phase)
- Returns `dict` with forecasts, metrics, model_used, model_flags
- NO access to `result` ForecastResult object (merged by main thread after)

### Change 3: Add parallel execution (ThreadPoolExecutor)

**File:** `operator1/models/forecasting.py`

In full mode, run `_forecast_single_variable()` in parallel:

```python
if _parallel:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=_n_workers) as pool:
        futures = {
            pool.submit(_forecast_single_variable, var_name, ...): var_name
            for var_name in available_vars
        }
        raw_results = {}
        for future in as_completed(futures):
            raw_results[futures[future]] = future.result()
else:
    # Fast mode: sequential (already fast enough at ~13s)
    raw_results = {
        var_name: _forecast_single_variable(var_name, ...)
        for var_name in available_vars
    }

# Phase 2: sequential post-processing for ALL modes
for var_name in available_vars:
    raw = raw_results[var_name]
    _horizon_forecasts = raw["horizon_forecasts"]
    # regime shift, momentum overlay, C1 blending (reads cache)
    ...
    result.forecasts[var_name] = _horizon_forecasts
    result.model_used[var_name] = raw["model_name"]
```

### Change 4: Batched multi-output LSTM

**File:** `operator1/models/forecasting.py` (new class + function)

Add `MultiOutputLSTM` class (~80 lines) and `fit_multi_lstm()` function (~60 lines):

```python
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=31):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(output_size)])

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # last time step
        return torch.cat([head(last_hidden) for head in self.heads], dim=1)

def fit_multi_lstm(cache, target_vars, n_forecast=252, epochs=50):
    # Build (seq_len, n_features) tensor from all variables
    # Train single model with combined loss
    # Return per-variable forecasts extracted from output columns
```

In full mode, this runs ONCE for all vars (~15-30s) instead of 31 times (~20 min).

### Change 5: Global cross-variable LightGBM

**File:** `operator1/models/forecasting.py` (new function)

```python
def _fit_global_lightgbm(cache, available_vars, n_forecast):
    # Build stacked DataFrame: (var_id, lag_1, lag_5, lag_21, rolling_mean, rolling_std) -> target
    # One lgb.train() call (~2-3s for 31 vars x 500 rows = 15,500 rows)
    # Return per-variable forecasts
```

This replaces the per-variable tree ensemble in full mode.

### Change 6: Update config + dashboard

**File:** `config/global_config.yml`

```yaml
forecasting:
  mode: "fast"           # "fast" | "full"  (delete express/balanced)
  parallel_workers: 4    # full mode only
  fast_ensemble_tiers: [3, 4, 5]
  distributional_vars: [close, return_1d, volatility_21d, revenue, fcf_yield]
```

**File:** `dashboard.py`

Update the forecasting mode toggle from 3 options to 2:
```python
ui.select(
    options={
        "fast": "Fast (~13s, ETS ensemble, no LSTM)",
        "full": "Full (~30s, parallel cascade + batched LSTM)",
    },
    ...
)
```

---

## Expected Timing

| Mode | Tier1/2 | Tier3+ | LSTM | LightGBM | Parallel | Total |
|------|---------|--------|------|----------|----------|-------|
| **fast** | Kalman 1.5s | ETS 0.8s | Off | Off | No | **~13s** |
| **full** | Kalman 1.5s | Tree+LSTM | Batched 15-30s | Global 2-3s | 4 workers | **~15-30s** |
| old full | Kalman 1.5s | Tree+LSTM | Per-var 20min | Per-var 90s | No | **~40min** |

---

## Files Changed

| # | File | What |
|---|------|------|
| 1 | `operator1/models/forecasting.py` | Delete balanced, extract per-var function, add parallel, add MultiOutputLSTM, add global LightGBM |
| 2 | `config/global_config.yml` | 2 modes (fast/full), remove balanced/express |
| 3 | `dashboard.py` | Update toggle to 2 options |
| 4 | `run_backtest_staged.py` | No change (sub-stage 4.1 is the same) |

---

## Execution Order

```
[ ] 1. Delete balanced mode, rename express->fast in forecasting.py + config + dashboard
[ ] 2. Extract _forecast_single_variable() (Phase 1 from v2 plan)
[ ] 3. Add ThreadPoolExecutor parallel execution for full mode
[ ] 4. Add MultiOutputLSTM class + fit_multi_lstm()
[ ] 5. Add _fit_global_lightgbm()
[ ] 6. Wire batched LSTM + global LightGBM into full mode cascade
[ ] 7. Update config + dashboard toggle
[ ] 8. Syntax check + import verification
[ ] 9. Debug scan
[ ] 10. Commit + push
```

---

## Backward Compatibility

- `mode: "balanced"` in old configs will fall through to `else` branch = full mode (safe)
- `mode: "express"` in old configs will match `"fast"` via alias (add `if _mode in ("fast", "express"):`)
- `lstm_enabled` config key kept for backward compat but ignored (LSTM always runs in full mode via batched path)

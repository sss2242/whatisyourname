# Forecasting 2-Mode Redesign: Full Implementation Plan with Exact Code References

*From debug scan of all inputs, outputs, closures, downstream consumers, and model name dependencies.*

---

## Debug Scan Summary

### Primary edit target
- `operator1/models/forecasting.py` (5,268 lines)
  - Mode logic: lines 2580-2614 (DELETE balanced, rename express->fast)
  - Per-variable loop: lines 2622-2980 (EXTRACT into `_forecast_single_variable()`)
  - LSTM block: line 2724 (REPLACE per-variable with batched)
  - Post-processing: lines 2818-2980 (KEEP sequential, runs after parallel phase)
  - Distributional batch: lines 2986-3029 (KEEP, already parallelized)
  - fit_lstm(): line 1131 (KEEP for backward compat, add `fit_batched_lstm()` separately)
  - _fit_fast_ensemble(): line 2161 (KEEP for fast mode)

### Config edit
- `config/global_config.yml` -- change `mode: "full"` to `mode: "fast"`, remove `balanced`/`express` refs

### Dashboard edit
- `dashboard.py` lines 962-971 -- update toggle from 3 options to 2 (fast/full)

### Files NOT edited (verified compatible)
All downstream consumers use generic dict/list access on `ForecastResult`. The new model names (`batched_lstm`, `global_lightgbm`) will flow through safely because:

---

## Backward Compatibility: Model Name Consumers

**CRITICAL:** 4 files check specific model name strings. The new model names must be ADDED to these lookups, not replace existing ones.

### prediction_aggregator.py (4 locations)
- **L742-748:** `_MODEL_VARIABLE_AFFINITY` dict -- maps model names to preferred variables. ADD:
  ```python
  "batched_lstm": ["close", "return_1d", "volatility_21d"],
  "global_lightgbm": ["close", "return_1d", "fcf_yield", "current_ratio"],
  ```
- **L1439-1454:** Regime-based weight adjustment -- matches `"kalman"`, `"tree"`, `"garch"` in model name. ADD `"lstm"` check to also match `"batched_lstm"` (already works via `"lstm" in name_lower`).
- **L2399-2406:** Hurst-based weight dicts. ADD `"batched_lstm"` and `"global_lightgbm"` entries with same weights as their non-batched equivalents.

### genetic_optimizer.py (2 locations)
- **L26:** `MODEL_NAMES` list. ADD `"batched_lstm"`, `"global_lightgbm"`.
- **L433-437:** Tier-specific initial weights. ADD entries for new model names.

### profile_builder.py (1 location)
- **L862-866:** Model name to human label mapping. ADD:
  ```python
  "batched_lstm": "Batched LSTM",
  "global_lightgbm": "Global LightGBM",
  ```

### model_diagnostics.py (checks model_used generically)
- No changes needed -- uses `in` operator for matching.

---

## Files with "balanced" string (NOT forecasting-related -- SAFE)

The debug scan found `"balanced"` in:
- `survival_mode.py` -- "balanced" refers to survival mode blend, not forecasting
- `llm_base.py`, `llm_factory.py`, `openrouter.py` -- "balanced" is an LLM model tier name
- `hedge_fund/types.py` -- "balanced" is a portfolio allocation type

These are UNRELATED to the forecasting mode. Do NOT change them.

---

## Implementation: 8 Changes (ordered by execution)

### Change 1: Delete balanced mode, rename to fast/full

**File:** `operator1/models/forecasting.py` lines 2580-2614

**Delete:**
```python
if _mode == "express":
    _fast_tiers: set[int] = set(_forecasting_cfg.get("fast_ensemble_tiers", [3, 4, 5]))
    _lstm_enabled: bool = False
elif _mode == "full":
    _fast_tiers = set()
    _lstm_enabled = _forecasting_cfg.get("lstm_enabled", True)
else:  # balanced (default)
    _fast_tiers = set(_forecasting_cfg.get("fast_ensemble_tiers", [3, 4, 5]))
    _lstm_enabled = False
```

**Replace with:**
```python
# Two modes: fast (ETS ensemble, no LSTM) and full (parallel cascade + batched LSTM + global LightGBM)
if _mode in ("fast", "express"):  # express kept as alias for backward compat
    _fast_tiers: set[int] = set(_forecasting_cfg.get("fast_ensemble_tiers", [3, 4, 5]))
    _use_batched_lstm: bool = False
    _use_global_lgbm: bool = False
    _parallel: bool = False
else:  # full (default), also catches old "balanced" config values
    _fast_tiers = set()  # all vars go through cascade
    _use_batched_lstm = True
    _use_global_lgbm = True
    _parallel = True
```

Also update the comment at line 2582 and the log message at line 2610-2614.

### Change 2: Add `fit_batched_lstm()` function

**File:** `operator1/models/forecasting.py` -- add BEFORE `run_forecasting()` (line ~2240)

~80 lines. Batch-across-series pattern from Nixtla neuralforecast:
1. Per-series z-normalization
2. Stack into tensor `(n_vars, seq_len, 1)`
3. Single LSTM training loop with ReduceLROnPlateau + early stopping (patience=3)
4. Extract per-variable forecasts from batch output
5. Inverse-transform using stored mean/std

Returns: `dict[str, np.ndarray]` (var_name -> forecast array)

### Change 3: Add `_fit_global_lightgbm()` function

**File:** `operator1/models/forecasting.py` -- add BEFORE `run_forecasting()` (line ~2240)

~60 lines. Global stacked DataFrame pattern from Nixtla mlforecast:
1. Build stacked DataFrame with `var_id` categorical + lag features (1, 5, 21, 63) + rolling stats
2. One `lgb.LGBMRegressor.fit()` call
3. Predict for each variable
4. Return `dict[str, float]` (var_name -> point forecast)

### Change 4: Extract `_forecast_single_variable()` from loop body

**File:** `operator1/models/forecasting.py` -- extract lines 2622-2812

The per-variable body (model fitting only, NOT post-processing) becomes a standalone function:

**Parameters (all read-only):**
- `var_name`, `series`, `tier`, `tier_num`, `max_horizon`
- `fast_tiers`, `fast_path_excluded`, `use_batched_lstm`, `distributional_vars`
- `random_state`
- Pre-extracted: `multivariate_df`, `feature_df`, `regime_data`

**Returns:** `dict` with `best_forecast`, `best_model_name`, `best_metrics`, `metrics_list`, `model_flags`

**Critical: Post-processing (lines 2818-2980) stays OUTSIDE this function.** It reads cache columns (regime, momentum, anchoring) and must run sequentially.

### Change 5: Add parallel execution for full mode

**File:** `operator1/models/forecasting.py` -- replace the per-variable loop

```python
# Pre-extract all data before parallel phase (thread safety)
pre_extracted = {}
for var_name in available_vars:
    pre_extracted[var_name] = {
        "series": _extract_series(var_name),
        "tier": _get_tier_for_variable(var_name, tier_map),
        "multivariate_df": _extract_multivariate(var_name) if ...,
        "feature_df": _extract_features(var_name, model_type="tree") if ...,
        "regime_data": _extract_regime_data() if ...,
    }

if _parallel:
    from joblib import Parallel, delayed
    raw_results = dict(zip(
        available_vars,
        Parallel(n_jobs=_n_workers, backend="threading", prefer="threads")(
            delayed(_forecast_single_variable)(
                var_name, pre_extracted[var_name], ...
            )
            for var_name in available_vars
        )
    ))
else:
    raw_results = {
        var_name: _forecast_single_variable(var_name, pre_extracted[var_name], ...)
        for var_name in available_vars
    }

# Phase 2: sequential post-processing (regime shift, momentum, etc.)
for var_name in available_vars:
    raw = raw_results[var_name]
    # ... existing post-processing code (lines 2818-2980) ...
```

### Change 6: Wire batched LSTM into full mode

**File:** `operator1/models/forecasting.py` -- add AFTER parallel cascade, BEFORE post-processing

```python
# In full mode: run batched LSTM on all variables, merge with cascade results
if _use_batched_lstm:
    try:
        lstm_forecasts = fit_batched_lstm(cache, list(available_vars))
        for var_name, lstm_fcast in lstm_forecasts.items():
            if var_name in raw_results:
                # Add LSTM as an additional model result
                raw_results[var_name]["lstm_forecast"] = lstm_fcast
                raw_results[var_name]["metrics_list"].append(
                    ModelMetrics(model_name="batched_lstm", variable=var_name, fitted=True)
                )
    except Exception as exc:
        logger.warning("Batched LSTM failed (continuing with cascade): %s", exc)
```

### Change 7: Wire global LightGBM into full mode

**File:** `operator1/models/forecasting.py` -- add AFTER batched LSTM, BEFORE post-processing

```python
if _use_global_lgbm:
    try:
        lgbm_forecasts, _ = _fit_global_lightgbm(cache, list(available_vars))
        for var_name, lgbm_val in lgbm_forecasts.items():
            if var_name in raw_results:
                raw_results[var_name]["lgbm_forecast"] = lgbm_val
                raw_results[var_name]["metrics_list"].append(
                    ModelMetrics(model_name="global_lightgbm", variable=var_name, fitted=True)
                )
    except Exception as exc:
        logger.warning("Global LightGBM failed (continuing with cascade): %s", exc)
```

### Change 8: Update config + dashboard

**File:** `config/global_config.yml`

```yaml
forecasting:
  mode: "fast"           # "fast" (ETS ensemble) | "full" (parallel + batched LSTM + global LightGBM)
  parallel_workers: 4    # workers for full mode parallel cascade
  fast_ensemble_tiers: [3, 4, 5]
  distributional_vars: [close, return_1d, volatility_21d, revenue, fcf_yield]
```

**File:** `dashboard.py` -- update the forecasting mode toggle

```python
ui.select(
    options={
        "fast": "Fast (~13s, ETS ensemble)",
        "full": "Full (~30s, parallel + batched LSTM)",
    },
    ...
)
```

---

## Downstream Consumer Updates (4 files, ~20 lines total)

### prediction_aggregator.py
Add `"batched_lstm"` and `"global_lightgbm"` to:
- L742: `_MODEL_VARIABLE_AFFINITY` dict
- L2399-2406: Hurst weight dicts

### genetic_optimizer.py
Add to:
- L26: `MODEL_NAMES` list
- L433-437: Tier-specific initial weights

### profile_builder.py
Add to:
- L862-866: Model name to human label mapping

---

## Execution Checklist

```
[ ] 1. Change 1: Delete balanced, rename modes (forecasting.py L2580-2614)
[ ] 2. Change 2: Add fit_batched_lstm() function (~80 lines)
[ ] 3. Change 3: Add _fit_global_lightgbm() function (~60 lines)
[ ] 4. Change 4: Extract _forecast_single_variable() from loop body
[ ] 5. Change 5: Add parallel execution with pre-extraction
[ ] 6. Change 6: Wire batched LSTM into full mode
[ ] 7. Change 7: Wire global LightGBM into full mode
[ ] 8. Change 8: Update config + dashboard (2 options, not 3)
[ ] 9. Update 4 downstream consumers with new model names
[ ] 10. Syntax check + import verification
[ ] 11. Debug scan
[ ] 12. Commit + push
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Batched LSTM fails on some data shapes | Try/except with fallback to per-variable cascade results |
| Global LightGBM fails | Try/except with fallback to per-variable tree results |
| joblib not available | Fallback to sequential execution (already in plan) |
| Old "balanced"/"express" config values | Alias handling: "express" -> "fast", "balanced" -> "full" |
| Thread safety in parallel cascade | Pre-extract all data before parallel phase, pure worker function |
| Model name not recognized by consumers | Added to all 4 consumer files |

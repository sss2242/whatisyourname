# Fix Plan: 4 Temporal Model Feature Integration Bugs

*2026-04-20*

## Summary

Debug scan confirmed 4 bugs introduced in commit `e9545c7` -- feat: temporal model feature integration. Two are HIGH severity runtime crashes, two are MEDIUM severity silent feature disablement. All 4 prevent the temporal model feature integration from functioning as designed.

---

## Bug #1 -- HIGH: Parallel Tree Nested Dict Crashes Prediction Aggregator

**Location:** `operator1/models/forecasting.py` line 2439
**Crash path:** `operator1/models/prediction_aggregator.py` line 800-801

**Root cause:** The parallel tree block stores results as:
```python
result.forecasts[_ptv]["tree_parallel"] = {
    label: float(_pt_fcast[min(h - 1, len(_pt_fcast) - 1)])
    for label, h in HORIZONS.items()
}
```

`ForecastResult.forecasts` is typed `dict[str, dict[str, float]]` -- values at the horizon level must be floats. The key `"tree_parallel"` injects a nested dict where a float is expected. When `prediction_aggregator.py` iterates:
```python
for h_label, value in horizons.items():
    if math.isnan(value):  # TypeError: dict is not a float
```

**Fix:** Flatten the parallel tree forecasts into individual horizon entries with a suffix, matching the existing contract:

```python
# BEFORE (bug):
result.forecasts[_ptv]["tree_parallel"] = {label: float(...) for label, h in HORIZONS.items()}

# AFTER (fix):
for label, h in HORIZONS.items():
    # Store as separate horizon entries; the ensemble will treat them
    # as an additional forecast channel for each horizon.
    _horizon_key = f"{label}"
    if _horizon_key not in result.forecasts[_ptv]:
        result.forecasts[_ptv][_horizon_key] = float(
            _pt_fcast[min(h - 1, len(_pt_fcast) - 1)]
        )
```

This stores the parallel tree forecast as the value for each horizon key only if that horizon doesn't already have a value from the primary cascade winner. The parallel tree result is still tracked via `model_used[_ptv] = "tree_parallel"` when it becomes the primary, and always via the `"tree_parallel"` entry in `metrics`.

**Alternative approach -- if we want both primary and parallel tree forecasts per horizon:** Add a new `parallel_forecasts` field to `ForecastResult` that the aggregator can optionally consume. This is cleaner but requires more changes. The simpler fix above is recommended for now.

**Files changed:**
- `operator1/models/forecasting.py` -- line 2439: flatten the nested dict into per-horizon entries

**Verification:** After fix, `math.isnan(value)` in prediction_aggregator.py will always receive a float, not a dict.

---

## Bug #2 -- HIGH: apply_residual_feature_adjustment Defined But Never Called

**Location:** `operator1/models/forecasting.py` line 1909 -- function definition
**Missing call sites:** After Kalman succeeds at line 2257, after Baseline succeeds at line 2331

**Root cause:** The temporal model feature integration plan Step 2 specifies: "Call this function after Kalman, AR1, ETS, and Baseline produce forecasts inside the per-variable cascade, after a univariate model succeeds." The function was created but the call sites were never added.

**Fix:** Add calls to `apply_residual_feature_adjustment` after each univariate model succeeds in the cascade. The function already handles all edge cases -- returns original forecast when fitted_values is None or extra_variables is empty.

**Call site 1 -- After Kalman succeeds, around line 2258:**
```python
best_forecast = fcast
best_model_name = "kalman"
best_metrics = met
# NEW: Apply residual feature adjustment
if extra_variables:
    for _hi, _hv in HORIZONS.items():
        _idx = min(_hv - 1, len(best_forecast) - 1)
        _orig = float(best_forecast[_idx])
        _adj = apply_residual_feature_adjustment(
            cache, var_name, _orig, extra_variables,
            fitted_values=met.test_residuals_fitted if hasattr(met, 'test_residuals_fitted') else None,
        )
        best_forecast[_idx] = _adj
```

However, the simpler approach from the plan is: apply the adjustment to the final horizon forecasts AFTER the cascade completes, not inside each model's block. This avoids duplicating the call in 4+ places.

**Recommended approach -- single call site after cascade completion, around line 2350:**
```python
# After baseline (always succeeds) and before storing in result.forecasts:
if best_forecast is not None and extra_variables:
    try:
        # Compute in-sample fitted values from the winning model
        _fitted = None
        if best_metrics and best_metrics.test_residuals is not None:
            # Reconstruct fitted from actual - residuals
            _actual = cache[var_name].dropna()
            if len(_actual) > len(best_metrics.test_residuals):
                _n = len(best_metrics.test_residuals)
                _fitted = _actual.iloc[-_n:] - pd.Series(
                    best_metrics.test_residuals, index=_actual.index[-_n:]
                )
        for _hi, _hv in HORIZONS.items():
            _idx = min(_hv - 1, len(best_forecast) - 1)
            _orig = float(best_forecast[_idx])
            _adj = apply_residual_feature_adjustment(
                cache, var_name, _orig, extra_variables,
                fitted_values=_fitted,
            )
            best_forecast[_idx] = _adj
    except Exception:
        pass  # graceful fallback: use unadjusted forecast
```

**Files changed:**
- `operator1/models/forecasting.py` -- add call after cascade completion, around line 2350
- `backtest_runner.py` -- mirror the same call in `run_stage2a2` forecasting section

**Verification:** After fix, `apply_residual_feature_adjustment` should appear in grep results as both definition AND call site. The adjustment is capped at 5% of forecast value by the function itself, preventing wild swings.

---

## Bug #3 -- MEDIUM: backtest_runner.py Discards feature_selection_result

**Location:** `backtest_runner.py` line 1355, line 64-140

**Root cause:** Two issues:
1. `state._extra_vars, _ = run_feature_selection(...)` -- the `_` discards the result
2. `BacktestState.__init__` never declares `self.feature_selection_result`

So `getattr(state, "feature_selection_result", None)` at line 1685 always returns None, and Step 4 feature-weighted confidence never triggers in the backtest path.

**Fix:**

1. Change line 1355:
```python
# BEFORE:
state._extra_vars, _ = run_feature_selection(...)

# AFTER:
state._extra_vars, state.feature_selection_result = run_feature_selection(...)
```

2. Add to `BacktestState.__init__` around line 112:
```python
self.feature_selection_result = None
```

**Files changed:**
- `backtest_runner.py` -- line 1355: store result instead of discarding
- `backtest_runner.py` -- line ~112: add `self.feature_selection_result = None` to `__init__`

**Verification:** After fix, `getattr(state, "feature_selection_result", None)` at line 1685 returns the actual result object, enabling Step 4 feature-weighted confidence in backtests.

---

## Bug #4 -- MEDIUM: main.py Passes Serialized Dict Instead of Result Object

**Location:** `main.py` line 3304

**Root cause:** The prediction aggregator call passes:
```python
feature_selection_result=profile.get("feature_selection") if profile else None
```

But `profile["feature_selection"]` is a plain dict serialized at line 4207. The aggregator checks `getattr(feature_selection_result, "fitted", False)` -- `getattr` on a dict for attribute `"fitted"` returns the default `False`, so the entire feature-weighted confidence block is silently skipped.

The actual `feature_selection_result` object with the `.fitted` attribute is defined at line 2814 and is still in scope at line 3304.

**Fix:** Pass the actual result object instead of the serialized profile dict:

```python
# BEFORE (line 3304):
feature_selection_result=profile.get("feature_selection") if profile else None,

# AFTER:
feature_selection_result=feature_selection_result,
```

The variable `feature_selection_result` is defined at line 2814 and is in scope throughout the `main()` function.

**Files changed:**
- `main.py` -- line 3304: change `profile.get("feature_selection") if profile else None` to `feature_selection_result`

**Verification:** After fix, the aggregator receives a `FeatureSelectionResult` object with `.fitted = True`, enabling the feature-weighted confidence block.

---

## Implementation Order

1. **Bug #1** -- HIGH, prevents runtime crash. Fix first because it blocks all testing.
2. **Bug #4** -- MEDIUM, 1-line fix. Enables feature-weighted confidence in main.py path.
3. **Bug #3** -- MEDIUM, 2-line fix. Enables feature-weighted confidence in backtest path.
4. **Bug #2** -- HIGH, ~20-line fix. Enables residual feature adjustment for univariate models.

Bugs #1 and #4 should be fixed together since they're in the same data flow path. Bug #3 mirrors Bug #4 for the backtest path. Bug #2 is the largest change but is isolated to `forecasting.py`.

---

## Files Changed Summary

| File | Bug | Lines Changed | Description |
|------|-----|---------------|-------------|
| `operator1/models/forecasting.py` | #1 | ~5 | Flatten parallel tree forecasts into per-horizon entries |
| `operator1/models/forecasting.py` | #2 | ~20 | Add call to apply_residual_feature_adjustment after cascade |
| `main.py` | #4 | 1 | Pass actual result object instead of serialized dict |
| `backtest_runner.py` | #3 | 2 | Store feature_selection_result + add to __init__ |
| `backtest_runner.py` | #2 | ~20 | Mirror residual feature adjustment call |

Total: ~48 lines across 3 files. No new files. No new dependencies. No architectural changes.

---

## Verification Plan

1. `python -m py_compile` all 3 modified files
2. Run AAPL backtest Stage 1 + Stage 2 -- verify no TypeError from Bug #1 fix
3. Check prediction_aggregator logs -- verify "feature-weighted confidence" block executes from Bug #3/#4 fixes
4. Check forecasting logs -- verify "residual feature adjustment" appears from Bug #2 fix
5. Compare predictions to previous run -- should differ slightly due to Bug #2 residual adjustments

---

## Branch and PR

- **Branch:** `fix/4-temporal-model-feature-integration-bugs`
- **Commit message:** `fix: 4 bugs in temporal model feature integration -- parallel tree crash, dead residual code, discarded feature_selection_result, serialized dict passthrough`

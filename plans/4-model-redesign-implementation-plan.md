# 4-Model Redesign: Full Implementation Plan

## Debug Scan Summary

### Files to Edit (6 files)

| # | File | Lines | Changes |
|---|------|-------|---------|
| 1 | `operator1/models/forecasting.py` | 4,610 | Add `shap_inline` field to `ForwardPassResult`, extract feature importance after forward pass loop |
| 2 | `operator1/stages/stage6_ensemble.py` | 497 | Modify `run_6_6_shap()` to use `shap_inline` fallback, modify `run_6_11_recursive_predictions()` to use forecast interpolation fallback |
| 3 | `operator1/models/recursive_aggregator.py` | 575 | Add `run_recursive_predictions_from_forecasts()` fallback function |
| 4 | `operator1/models/copula.py` | 418 | Add empirical copula + Kendall tau fallback when parametric fitting fails |
| 5 | `operator1/models/sensitivity.py` | 331 | Add Morris screening as primary method before Sobol |
| 6 | `operator1/models/explainability.py` | 510 | Add `from_inline_importance()` factory to build `SHAPResult` from forward pass data |

### Connection Map

```
Forward Pass (5.1)
  |-- model_bank: {var: [BaseModelWrapper, ...]}
  |-- TreeWrapper._model_obj: XGBRegressor/GBM/RF (has .feature_importances_)
  |-- TreeWrapper._feature_cols: list[str] (feature names)
  |-- KalmanWrapper: no importances (state-space, not feature-based)
  |-- LSTMWrapper: no importances (NN, needs gradient computation)
  |-- BaselineWrapper: no importances (EMA)
  |-- OUTPUT: ForwardPassResult
  |     |-- .model_states: {var: BaseModelWrapper}  << LOST to pickle between stages
  |     |-- .predictions_log: list[dict]             << survives pickle
  |     |-- .conformal_calibrator: Any               << survives pickle
  |     +-- .shap_inline: dict  << NEW: survives pickle (plain dict)

Stage 6.6 SHAP
  |-- INPUT: state.forward_pass_result.model_states  << EMPTY after pickle
  |-- INPUT: state.pred_result.predictions            << EXISTS
  |-- CURRENT: tries model_states.predict -> fails -> no SHAP
  |-- FIX: check .shap_inline first -> build SHAPResult from it
  |-- OUTPUT: state.shap_result -> consumed by:
  |     |-- stage 6.5 prediction_aggregator (step 8: SHAP attachment)
  |     |-- profile_builder -> profile["extended_models"]["shap_explanations"]
  |     +-- report_generator section 19

Stage 6.11 Recursive Predictions
  |-- INPUT: state.forward_pass_result.model_states  << EMPTY after pickle
  |-- INPUT: state.forecast_result.forecasts          << EXISTS {var: {horizon: value}}
  |-- INPUT: state.mc_result.transition_matrix        << EXISTS
  |-- CURRENT: checks model_states empty -> returns available=False
  |-- FIX: when model_states empty, use forecast_result interpolation
  |-- OUTPUT: state.recursive_result -> consumed by:
  |     |-- profile_builder -> profile["extended_models"]["recursive_predictions"]
  |     +-- report_generator "Recursive Predictions" section

Stage 5.5 Copula
  |-- INPUT: state.cache (daily DataFrame)
  |-- CURRENT: selects variables -> dropna -> requires >= 30 rows
  |-- ISSUE: with mixed-freq data, parametric copula (copulae lib) fails
  |--         on insufficient overlapping observations
  |-- FIX: add empirical copula (rank-based) + Kendall tau fallback
  |-- OUTPUT: state.copula_result -> consumed by:
  |     |-- stage 6.5 prediction_aggregator (step 5: band widening)
  |     |-- stage 6.8 multivariate MC (copula correlation)
  |     +-- profile_builder -> profile["extended_models"]["copula"]

Stage 6.7 Sobol
  |-- INPUT: state.cache, target="return_1d"
  |-- CURRENT: Saltelli sampling needs N*(2D+2) samples
  |--         with 502 rows and 30+ features, Saltelli needs ~1,920 samples -> fails
  |--         falls back to permutation importance (works but slower)
  |-- FIX: add Morris screening (needs r*(p+1) ~ 310 samples, fits 502)
  |-- OUTPUT: state.sobol_result -> consumed by:
  |     |-- adjust_hierarchy_from_sobol() (hierarchy weight nudging)
  |     +-- profile_builder -> profile["extended_models"]["sobol_sensitivity"]
```

---

## Fix 9: SHAP Inline Feature Importance

### Step 1: Add `shap_inline` field to `ForwardPassResult`
**File:** `operator1/models/forecasting.py` line 3695
```python
# Add after conformal_calibrator field:
shap_inline: dict[str, dict[str, Any]] = field(default_factory=dict)
```

### Step 2: Extract feature importance at end of forward pass
**File:** `operator1/models/forecasting.py` after line 4115 (after `result.model_states = ...`)
- Loop through `model_bank` items
- For TreeWrapper: check `._model_obj` for `feature_importances_` + `._feature_cols`
- For LinearRegression fallback: check `.coef_`
- Store as `result.shap_inline[var] = {top_drivers: [...], method: "tree_importance"}`

### Step 3: Add `from_inline_importance()` factory to SHAPResult
**File:** `operator1/models/explainability.py` after line 100
- Converts the inline dict to proper SHAPResult with PredictionExplanation objects

### Step 4: Modify stage 6.6 to use inline fallback
**File:** `operator1/stages/stage6_ensemble.py` `run_6_6_shap()` (line 306)
- Check `state.forward_pass_result.shap_inline` first
- If non-empty, use `from_inline_importance()` to build SHAPResult
- Otherwise fall through to existing SHAP library path

---

## Fix 10: Recursive Predictions Fallback

### Step 1: Add forecast-based fallback to `recursive_aggregator.py`
**File:** `operator1/models/recursive_aggregator.py` 
- Add `run_recursive_from_forecasts()` function
- Takes: `forecast_result.forecasts`, `mc_result.transition_matrix`, `cache`
- Interpolates between known horizons (1d/5d/21d/252d)
- Applies sqrt(t) confidence decay
- Returns same `RecursivePredictionResult` type

### Step 2: Modify stage 6.11 dispatcher
**File:** `operator1/stages/stage6_ensemble.py` `run_6_11_recursive_predictions()` (line 443)
- When `model_states` is empty, call `run_recursive_from_forecasts()` instead
- Pass `state.forecast_result` and `state.mc_result`

---

## Fix 11: Copula Empirical Fallback

### Step 1: Add empirical copula + Kendall tau fallback
**File:** `operator1/models/copula.py`
- Add `_fit_empirical_copula()` function using rank-based pseudo-observations
- Add `_fit_kendall_tau_pairwise()` function for very sparse data (10-30 rows)
- Modify `_run_copula_impl()` to try parametric first, fall back to empirical

### Architecture:
```
_run_copula_impl():
  1. Select variables + filter data (existing logic)
  2. If >= 30 rows of overlap:
     a. Try parametric (existing: Gaussian + Student-t + Clayton AIC)
     b. If parametric fails: try empirical copula (rank-based)
  3. If 10-30 rows:
     a. Kendall tau pairwise + empirical tail dependence
  4. If < 10 rows:
     a. Return CopulaResult(available=False)
```

---

## Fix 12: Sobol Morris Screening

### Step 1: Add `_try_morris()` function
**File:** `operator1/models/sensitivity.py`
- Uses `SALib.sample.morris` + `SALib.analyze.morris`
- Needs only `r*(p+1)` samples (r=15, p=30 -> 465, fits 502 rows)
- Returns `mu_star` (ranking metric) mapped to first_order format

### Step 2: Modify cascade in `_run_sensitivity_impl()`
**File:** `operator1/models/sensitivity.py` line 202
- Change order: Morris first -> Sobol on top-5 -> permutation fallback
```
1. Try Morris screening (works with 502 rows)
2. If Morris works: optionally run Sobol on top-5 features only (12*N samples)
3. If Morris fails: fall back to permutation importance (existing)
```

---

## Execution Order

1. **Fix 9** (SHAP): `forecasting.py` -> `explainability.py` -> `stage6_ensemble.py`
2. **Fix 10** (Recursive): `recursive_aggregator.py` -> `stage6_ensemble.py`
3. **Fix 11** (Copula): `copula.py` only
4. **Fix 12** (Sobol): `sensitivity.py` only

Fixes 11 and 12 are self-contained in their respective model files. Fixes 9 and 10 span multiple files but the stage dispatcher changes are minimal.

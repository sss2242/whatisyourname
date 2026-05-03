# Fix Guide: Remaining Issues from Deep Scan v3

## Simple Fixes (1-line changes)

### Issue 8: Conformal residuals attribute name mismatch

**File:** `operator1/models/forecasting.py:3704-3705`
**Severity:** Medium -- prevents conformal calibrator residuals from surviving pickle
**Type:** Simple attribute name typo

The `ConformalPIDCalibrator` stores its scores in `self._scores` (private attribute, dict of lists). The `ForwardPassResult.__getstate__` method checks `hasattr(cal, "scores")` (public), which always returns False.

**Fix:**
```python
# Line 3704-3705: Change
if cal is not None and hasattr(cal, "scores"):
    state["_conformal_residuals"] = list(cal.scores)
# To
if cal is not None and hasattr(cal, "_scores"):
    state["_conformal_residuals"] = [
        s for bucket in cal._scores.values() for s in bucket
    ]
```

Note: `cal._scores` is a dict of `{bucket_key: [float]}`, so we flatten all buckets into a single list. The downstream conformal rebuilder in stage 6.3 feeds these back via `calibrator.update(residual)` which routes to the global bucket.

---

## Complex Fixes (require architectural consideration)

### Issue 9: SHAP empty because model_states lost to pickle

**File:** `operator1/stages/stage6_ensemble.py` (run_6_6_shap) + `operator1/models/forecasting.py` (forward pass)
**Severity:** Low-Medium -- SHAP explanations are informational, not used for predictions

**Problem:** SHAP needs fitted model objects (tree models or predict functions) from `forward_pass_result.model_states`. These are stripped by `__getstate__` because sklearn/torch models contain C-extension references.

**Two approaches:**

**Approach A: Compute SHAP during forward pass (in-process)**
Move SHAP computation inside `run_forward_pass()` where models are still in memory. Store SHAP values as a picklable dict on ForwardPassResult:
```python
# Inside forward pass, after all predictions:
shap_summary = {}
for var, wrapper in model_wrappers.items():
    if hasattr(wrapper, 'model') and hasattr(wrapper.model, 'feature_importances_'):
        shap_summary[var] = dict(zip(feature_names, wrapper.model.feature_importances_))
result.shap_inline = shap_summary  # picklable dict
```
Then stage 6.6 reads `forward_pass_result.shap_inline` instead of trying to run SHAP from scratch.

**Approach B: Serialize model coefficients (not full objects)**
Extract picklable coefficients during `__getstate__`:
```python
def __getstate__(self):
    state = self.__dict__.copy()
    # Convert model_states to coefficient-only representation
    coefs = {}
    for var, wrapper in self.model_states.items():
        if hasattr(wrapper, 'model') and hasattr(wrapper.model, 'feature_importances_'):
            coefs[var] = {'importances': list(wrapper.model.feature_importances_)}
    state['model_states'] = {}
    state['_model_coefficients'] = coefs
    return state
```

**Recommendation:** Approach A is simpler and more reliable. Feature importances from tree models are a good proxy for SHAP values.

---

### Issue 10: Recursive predictions empty because model_states lost

**File:** `operator1/stages/stage6_ensemble.py` (run_6_11_recursive_predictions)
**Severity:** Low -- recursive predictions are supplementary

**Problem:** Same root cause as SHAP. The recursive aggregator needs fitted model wrappers to produce day-by-day autoregressive predictions.

**Fix approach:** The recursive aggregator should fall back to using the `forecast_result` directly (which IS preserved) when `model_states` is empty. Instead of calling `wrapper.predict(features)` for each day, use the forecast's point predictions with a simple linear decay model:
```python
# If model_states empty, fall back to forecast-based recursive
if not model_states:
    # Use forecast_result point predictions with confidence decay
    for day in range(1, n_days + 1):
        decay = 0.95 ** day  # 5% confidence decay per step
        ...
```

This is a moderate refactor of `recursive_aggregator.py`.

---

### Non-issues (no fix needed)

**Config keys not in global_config.yml (42 keys):** These are module-specific config keys accessed via `cfg.get("key", default)` with fallback defaults. They work correctly without yml entries. Examples: `archetypes`, `covenant_threshold`, `credit_spread_pct`. These come from per-module config files (`crisis_archetypes.yml`, `country_protection_rules.yml`) not global_config.

**Copula fitted=False:** Data-dependent. AAPL has limited multivariate overlap for Student-t/Clayton fitting. Not fixable without more data.

**Sobol fitted=False:** Saltelli sampling needs N*(2D+2) rows. With 502 rows and many features, data is insufficient. Not fixable without reducing feature count or increasing data window.

---

## Implementation Priority

```
Issue 8 (_scores typo):     Simple, 3-line change, medium impact
Issue 9 (SHAP in-process):  Complex, ~15 lines, low-medium impact  
Issue 10 (recursive fallback): Complex, ~20 lines, low impact
```

Recommend fixing Issue 8 now, deferring Issues 9-10 to a future PR.

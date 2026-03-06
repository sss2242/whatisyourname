# Deep Codebase Audit: Wiring Bugs, Dead Code, and Duplication

## Critical Bugs Found

### BUG 1: Prediction Aggregator Not Receiving Sibling Module Results (WIRING)

**Severity: HIGH** -- Confirmed by the AAPL run profile JSON

At [`main.py:1616-1618`](main.py:1616), the prediction aggregator is called with only 3 arguments:

```python
pred_result = run_prediction_aggregation(
    cache, forecast_result, mc_result,
)
```

But [`run_prediction_aggregation()`](operator1/models/prediction_aggregator.py:1671) accepts 7 additional optional keyword arguments that main.py never passes:

```python
def run_prediction_aggregation(
    cache, forecast_result, mc_result,
    *,
    conformal_result=None,     # NEVER PASSED
    dual_regime_result=None,   # NEVER PASSED
    copula_result=None,        # NEVER PASSED
    dtw_result=None,           # NEVER PASSED
    granger_result=None,       # NEVER PASSED
    shap_result=None,          # NEVER PASSED
    walk_forward_result=None,  # NEVER PASSED
)
```

All 7 of these results ARE computed in main.py (lines 1356-1365) and stored in local variables, but they are never wired to the aggregator. Instead, copula tail adjustment is done AFTER the aggregation as a separate post-hoc patch (lines 1621-1636), which is fragile and only partially implements what the aggregator could do internally.

**Fix**: Pass all available results to `run_prediction_aggregation()`:
```python
pred_result = run_prediction_aggregation(
    cache, forecast_result, mc_result,
    conformal_result=conformal_result,
    dual_regime_result=dual_regime_result,
    copula_result=copula_result,
    dtw_result=dtw_result,
    granger_result=granger_result,
    shap_result=shap_result,
    walk_forward_result=forward_pass_result,
)
```

### BUG 2: Duplicate `compute_granger_causality` Function (DUPLICATION)

**Severity: MEDIUM** -- Tests use the wrong one

Two different implementations exist:
- [`operator1/models/causality.py:56`](operator1/models/causality.py:56) -- older version, with `significance: float` parameter
- [`operator1/models/granger_causality.py:51`](operator1/models/granger_causality.py:51) -- newer version, with `significance_level: float` parameter and `GrangerResult` dataclass

**main.py imports from `granger_causality`** (the new one) at line 1432.
**Tests import from `causality`** (the old one) at test_phase6_regime.py lines 479-655.

This means tests are testing the OLD implementation while the pipeline runs the NEW one. If the new one has bugs, the tests wouldn't catch them.

**Fix**: Delete the Granger function from `causality.py` (keep only `compute_transfer_entropy` there), update tests to import from `granger_causality.py`.

### BUG 3: Conformal Result Computed AFTER Aggregation (ORDERING)

**Severity: MEDIUM**

At [`main.py:1616`](main.py:1616), `run_prediction_aggregation()` runs first. Then at line 1640, conformal prediction is computed. But the aggregator has a `conformal_result` parameter that expects conformal data DURING aggregation to replace Gaussian uncertainty bands.

The current order:
1. Run prediction aggregation (no conformal)
2. Compute conformal intervals (too late)
3. Store conformal in profile (just for display, not used in predictions)

**Fix**: Move conformal calibration BEFORE the prediction aggregation call, or at minimum ensure the forward pass's conformal diagnostics are fed to the aggregator.

### BUG 4: `forecast_result` Type Mismatch Between Models and Profile Builder (TYPE)

**Severity: LOW**

`run_forecasting()` returns a `ForecastResult` dataclass. But `build_company_profile()` expects `forecast_result` to be a `dict`. In main.py, there's implicit conversion happening somewhere, but it's fragile.

Looking at line 1792 of main.py, `forecast_result` is passed directly to `build_company_profile`. Let me verify the profile builder handles both types.

### BUG 5: `forward_pass_result` PID Summary Not Reliably Extracted (WIRING)

**Severity: LOW**

At [`main.py:1801-1805`](main.py:1801):
```python
pid_summary=(
    getattr(forward_pass_result, "pid_summary", None)
    if forward_pass_result is not None
    else None
),
```

But `forward_pass_result.pid_summary` is set in [`forecasting.py:2844`](operator1/models/forecasting.py:2844) as:
```python
result.pid_summary = pid_summary.to_dict()
```

The `ForwardPassResult` dataclass doesn't have `pid_summary` as a declared field -- it's dynamically attached via `setattr`. This means `getattr` works but static type checkers would flag it, and it's easy to miss.

---

## Dead Code / Legacy References

### LEGACY 1: `fmp_client` Parameters

Files with legacy FMP (Financial Modeling Prep) parameters that are never used:
- [`operator1/steps/data_extraction.py:265`](operator1/steps/data_extraction.py:265) -- `fmp_client` docstring
- [`operator1/steps/verify_identifiers.py:57`](operator1/steps/verify_identifiers.py:57) -- `fmp_client: Any = None` parameter

These are kept for "backward compatibility" but FMP was removed from the pipeline. They add confusion.

### LEGACY 2: `causality.py` Granger Code

The Granger causality code in [`causality.py`](operator1/models/causality.py) is a legacy version. The canonical one is in [`granger_causality.py`](operator1/models/granger_causality.py). The `causality.py` file should only contain `compute_transfer_entropy` and `prune_weak_relationships`.

---

## Code Duplication

### DUP 1: Granger Causality (described above)

Two implementations, different signatures, different return types.

### DUP 2: `_build_column_rename_map` Pattern

The column rename map in [`cache_builder.py:270-340`](operator1/steps/cache_builder.py:270) duplicates field mappings that also exist in:
- [`canonical_translator.py`](operator1/clients/canonical_translator.py) -- the primary field mapping module
- [`us_edgar.py`](operator1/clients/us_edgar.py) -- EDGAR-specific XBRL concept mapping

Three places map camelCase column names to snake_case, and they could diverge.

### DUP 3: Minimum Observation Constants

Multiple files define their own minimum observation thresholds:
- [`forecasting.py`](operator1/models/forecasting.py): `_MIN_OBS_KALMAN=30, _MIN_OBS_GARCH=60, _MIN_OBS_VAR=50, _MIN_OBS_LSTM=100, _MIN_OBS_TREE=30`
- [`regime_detector.py`](operator1/models/regime_detector.py): `_MIN_OBS_HMM=60, _MIN_OBS_GMM=30, _MIN_OBS_PELT=30`
- [`transformer_forecaster.py`](operator1/models/transformer_forecaster.py): `_MIN_OBS_TRANSFORMER=100`
- [`estimator.py`](operator1/estimation/estimator.py): `_MIN_OBSERVED_FOR_MODEL=10`

These should be centralized or at least documented together.

---

## Cache Handling Issues

### CACHE 1: In-Place Mutation in main.py

main.py mutates the `cache` DataFrame in-place throughout the pipeline:
- Line 879: `cache = compute_derived_variables(cache)`
- Line 887: `cache["company_survival_mode_flag"] = ...`
- Line 934-964: Multiple `cache["col"] = ...` assignments

This means if any module holds a reference to the cache from before a mutation, it sees stale data. The forward pass at line 1476 gets the cache AFTER all mutations, which is correct, but intermediate modules like the regime detector (line 1274) get the cache BEFORE financial health scores are computed (line 978).

### CACHE 2: No Cache Schema Validation

There's no validation that the cache has the expected columns before passing it to models. Each model silently handles missing columns differently (some return empty results, some crash). A schema check at pipeline entry would catch misconfigurations early.

---

## Prioritized Fix Order

1. **BUG 1** (aggregator wiring) -- Single-line fix with massive impact
2. **BUG 2** (duplicate Granger) -- Delete old code, fix test imports
3. **BUG 3** (conformal ordering) -- Reorder conformal before aggregation
4. **LEGACY 1-2** (FMP + old Granger) -- Clean dead code
5. **DUP 2-3** (column maps + constants) -- Centralize for maintainability
6. **CACHE 1-2** (mutation + validation) -- Defensive improvements

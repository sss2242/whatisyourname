# Fix Plan: Pipeline Wiring Issues W1-W5 and Warnings

## Summary

The audit found 5 wiring issues, 2 performance warnings, and 3 dependency deprecations.
This plan addresses all of them in priority order, grouped by risk and complexity.

---

## Priority 1: Fix Wiring Issues (W1-W3) -- Feature Degradation

These three issues silently degrade features that were designed to work together.

### Fix W1 + W2: Wire walk-forward into the pipeline

**Problem:** `run_walk_forward()` from `walk_forward.py` is never called.
Instead, `forward_pass_result` (wrong type) is passed where `WalkForwardResult`
is expected, making recency-weighted RMSE permanently NaN.

**Plan:**
1. In `main.py` Step 6, after `run_forward_pass()`, add a call to
   `run_walk_forward()` from `operator1/models/walk_forward.py`
2. Pass the `WalkForwardResult` (not `ForwardPassResult`) as
   `walk_forward_result` to `run_prediction_aggregation()`
3. Keep passing `forward_pass_result` separately for PID summary extraction
4. Add the walk-forward leaderboard results to the profile

**Files to change:**
- `main.py` (add import and call around line 1590-1600, fix line 1728)

**Risk:** Low. Walk-forward is self-contained and returns a clean dataclass.
The prediction aggregator already has full duck-typed handling for `WalkForwardResult`.

---

### Fix W3: Add `residuals` to `ForecastResult` and populate it

**Problem:** `ForecastResult` has no `residuals` attribute. The conformal
calibrator never receives calibration data, producing uncalibrated intervals.

**Plan:**
1. Add `residuals: list[float] | None = None` field to `ForecastResult` dataclass
   in `operator1/models/forecasting.py`
2. In `run_forecasting()`, collect validation residuals from each model's
   train/test split and store them in `forecast_result.residuals`
3. The existing code in `main.py:1686-1688` already correctly feeds residuals
   to the conformal calibrator -- it just needs the data to exist

**Files to change:**
- `operator1/models/forecasting.py` (add field to dataclass, populate in run_forecasting)

**Risk:** Low. Adding a field to a dataclass is backward-compatible. Tests
already pass with the field absent.

---

## Priority 2: Fix Wiring Issues (W4-W5) -- Wasted Computation

### Fix W4: Wire `burnout_result` into the profile

**Problem:** `run_burnout()` runs but its output is never stored in the profile
or used by downstream modules.

**Plan:**
1. In `main.py` Step 7 (profile building), add `burnout_result` to the
   `extended_models` section of the profile using `_available_dict()`
2. Include key metrics: `iterations_completed`, `converged`, `final_rmse`

**Files to change:**
- `main.py` (add ~5 lines in the profile injection block around line 1958-2064)

**Risk:** Very low. Pure additive change.

---

### Fix W5: Store full transfer entropy results in profile

**Problem:** Transfer entropy results are stored as just `{"available": True}`
instead of including the actual pairwise entropy scores.

**Plan:**
1. In `main.py` line 1962-1963, replace the hard-coded dict with
   `_available_dict(transfer_entropy_result)` -- same pattern used by copula,
   SHAP, DTW, and all other extended models

**Files to change:**
- `main.py` (change 1 line around line 1962)

**Risk:** Very low. Just using the existing helper function.

---

## Priority 3: Fix Performance Warnings

### Fix P1: DataFrame fragmentation in `estimator.py:542`

**Problem:** `_store_estimation_results()` is called per-variable in a loop,
each time doing `df[new_cols.columns] = new_cols` which triggers pandas
PerformanceWarning about high fragmentation.

**Plan:**
1. Refactor `run_estimation()` to collect all estimation result DataFrames
   in a list during the variable loop
2. After the loop, use a single `pd.concat([df] + all_results, axis=1)` call
   to merge everything at once
3. Alternatively, change `_store_estimation_results()` to return a dict of
   Series instead of inserting into the DataFrame, then batch-insert at the end

**Files to change:**
- `operator1/estimation/estimator.py` (refactor the estimation loop and
  `_store_estimation_results()`)

**Risk:** Medium. Requires careful refactoring of the estimation loop. Must
verify that in-loop references to previously estimated variables still work.

---

### Fix P2: NaN divide in copula correlation

**Problem:** `np.corrcoef` produces NaN when a column has zero standard
deviation (constant values) after the PIT transform.

**Plan:**
1. In `_fit_gaussian_copula()` in `copula.py:59-74`, before calling
   `np.corrcoef`, check each column's standard deviation
2. Drop columns with zero variance (or replace with small noise)
3. Return identity matrix for dropped dimensions

**Files to change:**
- `operator1/models/copula.py` (add ~5 lines of variance check before line 73)

**Risk:** Low. Localized change within a single function.

---

## Priority 4: Track Dependency Deprecations (No Code Changes)

These are upstream library issues that don't require immediate code changes:

| Warning | Source | Action |
|---------|--------|--------|
| `verbose` FutureWarning | statsmodels 0.14.6 | Monitor; already passing `verbose=False` |
| edgar.files.* deprecated | edgartools 5.19.1 | Update imports when v6.0 releases |
| pkg_resources deprecated | pykrx 1.0.51 | Wait for upstream fix |

---

## Implementation Order

```
1. W5: transfer_entropy profile fix     (1 line change, ~2 min)
2. W4: burnout_result profile wiring    (5 lines, ~5 min)
3. W3: ForecastResult.residuals         (15 lines, ~15 min)
4. W1+W2: walk-forward wiring           (20 lines, ~20 min)
5. P2: copula zero-variance guard       (5 lines, ~5 min)
6. P1: estimator defragmentation        (30 lines, ~30 min)
```

Each fix is independent and can be tested individually.

# Pipeline Bug Fixes and Wiring Corrections

## Context

A full pipeline audit identified **11 bugs** and **3 code duplications** across the Operator 1 codebase. This plan organizes fixes into 4 phases, ordered by impact and risk.

---

## Phase 1: Critical Wiring Fixes (Step 6 model ordering + dead features)

These are silent data-loss bugs where models produce results that never reach downstream consumers.

### 1.1 Reorder conformal/DTW/SHAP before prediction aggregation

**Bug 7** -- The prediction aggregator at `main.py:1616` receives `conformal_result=None`, `dtw_result=None`, `shap_result=None` because these models are computed after it runs.

**Changes in `main.py`:**
- Move the conformal prediction block (lines 1634-1657) to run BEFORE `run_prediction_aggregation`
- Move DTW analogs block (lines 1659-1665) to run BEFORE `run_prediction_aggregation`
- Move SHAP explainability block (lines 1667-1686) to run BEFORE `run_prediction_aggregation`
- Keep prediction aggregation as the final step that consumes all model results

**Dependency chain to respect:**
```
forecasting -> forward_pass -> burnout -> monte_carlo -> copula
                                                           |
transformer -> particle_filter                             |
                                                           v
conformal (needs forecast_result.residuals) -------> prediction_aggregation
DTW (needs cache only) ----------------------------> prediction_aggregation
SHAP (needs pred_result... circular!) -------------> prediction_aggregation
```

**Note on SHAP:** SHAP currently depends on `pred_result` (line 1670-1683), creating a circular dependency. The fix should split this into two passes:
1. Run prediction aggregation without SHAP
2. Compute SHAP from the initial predictions
3. Attach SHAP to the profile (it doesn't need to feed back into aggregation)

**Revised order:**
```
forecasting -> forward_pass -> burnout -> monte_carlo -> copula -> transformer -> particle_filter -> conformal -> DTW -> prediction_aggregation -> SHAP -> sobol -> GA -> OHLC
```

### 1.2 Wire up `compute_macro_quadrant`

**Bug 8** -- `macro_quadrant_result` is never computed.

**Changes in `main.py`:**
- After macro data fetch (around line 860), add:
```python
if macro_data:
    from operator1.features.macro_quadrant import compute_macro_quadrant
    cache, macro_quadrant_result = compute_macro_quadrant(cache, macro_data=macro_data)
```
- This goes in Step 4a after the macro data fetch, before the validation summary

### 1.3 Build `MacroDataset` from raw macro dict

**Bug 9** -- `macro_dataset` is never constructed.

**Changes in `main.py`:**
- After macro data fetch, build the dataset:
```python
if macro_data:
    from operator1.steps.macro_mapping import build_macro_dataset
    macro_dataset = build_macro_dataset(macro_data, country_iso2=market_info.country_code)
```
- Check if `build_macro_dataset` exists in `macro_mapping.py`; if not, create a simple constructor
- Pass `macro_dataset` to `compute_macro_quadrant` instead of raw dict

---

## Phase 2: PIT Safety and Merge Consistency

### 2.1 Replace inline cache building with `cache_builder.py`

**Bug 10 + Duplication D1** -- `main.py:760-832` reimplements cache building without look-ahead protection.

**Changes:**
- Refactor `main.py` Step 4 to call `build_entity_daily_cache()` from `operator1/steps/cache_builder.py`
- This requires constructing an `EntityData` object from the fetched data
- If the `EntityData` interface is too rigid, create a lighter wrapper function in `cache_builder.py` that accepts the raw DataFrames directly
- The key benefit is getting `LookAheadError` checks and `is_missing_*` flags for free

### 2.2 Align linked entity merge strategy

**Bug 11 + Duplication D2** -- Linked entity cache uses `filing_date` first and simpler reindex.

**Changes in `main.py` `_fetch_linked_entity()` (line 1090):**
- Change line 1115 from `"filing_date" if "filing_date" in ...` to `"report_date" if "report_date" in ...` to match the target strategy
- Change line 1124 from `reindex(method="ffill")` to the union+ffill+reindex pattern used for the target:
```python
combined_idx = _ent_cache.index.union(_si.index).sort_values()
_sa = _si.reindex(combined_idx).ffill()
_sa = _sa.reindex(_ent_cache.index)
```
- Ideally, extract a shared `_merge_statement_to_cache()` helper used by both target and linked entity code

---

## Phase 3: Test and Deprecation Fixes

### 3.1 Fix Banxico test mock

**Bug 1** -- Test mocks `get_series_data()` but API uses `get()`.

**Changes in `tests/test_macro_banxico.py`:**
- Change `mock_api.get_series_data.return_value` to `mock_api.get.return_value`
- Update the mock return value to match the actual API format:
```python
mock_api.get.return_value = [
    {
        "idSerie": "SF61745",
        "datos": [
            {"fecha": "01/01/2023", "dato": "11.25"},
            {"fecha": "01/02/2023", "dato": "11.25"},
            {"fecha": "01/03/2023", "dato": "11.00"},
        ],
    }
]
```

### 3.2 Replace `datetime.utcnow()`

**Bug 2** -- Deprecated in Python 3.12, will be removed.

**Changes:**
- `operator1/report/profile_builder.py:1100`: Replace `datetime.utcnow().isoformat() + "Z"` with `datetime.now(datetime.UTC).isoformat() + "Z"`
- `operator1/report/report_generator.py:2531`: Replace `datetime.utcnow().isoformat()` with `datetime.now(datetime.UTC).isoformat()`
- Add `from datetime import datetime, UTC` or use `datetime.timezone.utc` for Python 3.10 compatibility

### 3.3 Fix statsmodels verbose warning

**Bug 6** -- FutureWarning from `grangercausalitytests`.

**Changes in `operator1/models/granger_causality.py`:**
- Find the call to `grangercausalitytests()` and add `verbose=False`

---

## Phase 4: Performance Improvements

### 4.1 Fix DataFrame fragmentation in `derived_variables.py`

**Bug 3** -- 84 PerformanceWarnings from repeated single-column inserts.

**Changes in `operator1/features/derived_variables.py` around lines 540-571:**
- Collect all technical indicator columns in a dict first:
```python
new_cols = {}
new_cols["sma_50"] = close.rolling(window=50, min_periods=10).mean()
new_cols["is_missing_sma_50"] = new_cols["sma_50"].isna().astype(int)
# ... all other indicators ...
df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
```
- Apply same pattern to all `_compute_*` stage functions that repeatedly assign to `df["col"]`

### 4.2 Fix DataFrame fragmentation in `estimator.py`

**Bug 4** -- 32 PerformanceWarnings from line 542.

**Changes in `operator1/estimation/estimator.py` around line 542:**
- The current code `df[new_cols.columns] = new_cols` is already batched but triggers the warning because the DataFrame was already fragmented from earlier inserts
- Review the calling code that builds `cols_dict` and ensure all columns for a variable are collected before assignment
- Add `df = df.copy()` after the estimation loop to defragment

---

## Verification Plan

After all fixes:

1. Run full test suite in stages (same as audit):
   - Stage 1: `test_phase1_smoke.py test_phase2_ingestion.py test_phase3_features.py`
   - Stage 2: `test_phase4_analysis.py test_phase5_estimation.py test_split_estimator.py test_enriched_survival_timeline.py test_survival_timeline_walkforward.py test_financial_health.py`
   - Stage 3: `test_phase6_forecasting.py test_phase6_monte_carlo.py test_phase6_prediction_aggregator.py test_phase6_regime.py`
   - Stage 4: `test_phase7_report.py test_advanced_modules.py test_bug_regression.py test_phase_b_ethical_filters.py test_phase_c_modules.py test_three_features.py`
   - Stage 5: LLM + translator + equity tests
   - Stage 6: Macro + OHLCV + PIT tests

2. Confirm zero PerformanceWarnings from Phases 4.1/4.2
3. Confirm zero DeprecationWarnings from Phase 3.2
4. Confirm `test_macro_banxico::test_fetch_with_mock` passes (Phase 3.1)
5. Confirm macro_quadrant section appears in profile output (Phase 1.2)

---

## Files Modified Summary

| File | Phase | Changes |
|------|-------|---------|
| `main.py` | 1.1, 1.2, 1.3, 2.1, 2.2 | Reorder Step 6 models, wire macro_quadrant, build MacroDataset, refactor cache building, fix linked merge |
| `tests/test_macro_banxico.py` | 3.1 | Fix mock method name and return format |
| `operator1/report/profile_builder.py` | 3.2 | Replace datetime.utcnow() |
| `operator1/report/report_generator.py` | 3.2 | Replace datetime.utcnow() |
| `operator1/models/granger_causality.py` | 3.3 | Add verbose=False |
| `operator1/features/derived_variables.py` | 4.1 | Batch column assignments |
| `operator1/estimation/estimator.py` | 4.2 | Defragment DataFrame after estimation |

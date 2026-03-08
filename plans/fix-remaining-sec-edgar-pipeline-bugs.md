# Fix Remaining SEC EDGAR Pipeline Bugs

3 code bugs + 2 external issues identified from the Apple Inc (AAPL) pipeline run on 2026-03-07.

---

## Bug 1: Estimation Shape Mismatch (HIGH)

**Error:** `Shape of passed values is (1256, 16), indices imply (1256, 17)`
**Location:** `operator1/estimation/estimator.py` around line 547
**Impact:** Estimation engine fails; pipeline falls back to raw unimputed data

### Root Cause

The estimator builds a list of variable names to estimate, then constructs a DataFrame from the results. If one variable is skipped (e.g. all-NaN, or classification error), the result array has fewer columns than the schema expects.

Line 547 has a fallback that assigns columns one at a time, but the initial construction still fails.

### Fix Plan

1. Read `estimator.py` lines 530-560 to understand the exact DataFrame construction
2. Add a guard that filters the column list to match the actual result array shape before DataFrame construction
3. Add a debug log showing which variable was skipped (the delta between expected 17 and actual 16)
4. Ensure the fallback path at line 547 works correctly by testing with a mock that returns fewer columns

### Files Changed
- `operator1/estimation/estimator.py`

---

## Bug 6: ConformalCalibrator Missing `update` Method (HIGH)

**Error:** `'ConformalCalibrator' object has no attribute 'update'`
**Location:** `main.py` calls `.update()` on the calibrator; `operator1/models/conformal.py` has `add_score()` and `update_adaptive()` but no `update()`
**Impact:** Conformal prediction intervals are unavailable

### Root Cause

The `ConformalCalibrator` interface was refactored to use `add_score()` for adding residuals and `update_adaptive()` for adaptive adjustments. But the caller in `main.py` still calls the old `.update()` method.

### Fix Plan

1. Read the `ConformalCalibrator` class in `conformal.py` (lines 86-300) to understand the current API
2. Read the caller in `main.py` to see what arguments are passed to `.update()`
3. **Option A (preferred):** Add an `update()` method to `ConformalCalibrator` that delegates to `add_score()` -- this maintains backward compatibility
4. **Option B:** Update the caller in `main.py` to use `add_score()` directly
5. Add a test that verifies the calibrator can be used end-to-end

### Files Changed
- `operator1/models/conformal.py` (add `update` alias)
- OR `main.py` (update caller)

---

## Bug 8: 4 NaN Financial Variables from SEC EDGAR (MEDIUM)

**Error:** `4 variables had 0 non-NaN observations: ['cash_ratio', 'net_debt_to_ebitda', 'interest_coverage', 'current_ratio']`
**Location:** `operator1/clients/us_edgar.py` (field extraction) + `operator1/features/derived_variables.py` (computation)
**Impact:** Key financial health ratios are missing from the report

### Root Cause

These 4 variables are *derived* (computed from source fields):
- `cash_ratio` = `cash_and_equivalents` / `current_liabilities`
- `net_debt_to_ebitda` = `net_debt` / `ebitda`
- `interest_coverage` = `ebit` / `interest_expense`
- `current_ratio` = `current_assets` / `current_liabilities`

If the source fields (`current_assets`, `current_liabilities`, `interest_expense`, `ebitda`) are NaN, the derived variables will be NaN. The SEC EDGAR wrapper has mappings for these XBRL concepts but the `edgartools` library may return different field names than expected.

### Fix Plan

1. Inspect the actual XBRL concepts returned by `edgartools` for Apple's 10-Q filings
2. Compare against the concept mapping in `us_edgar.py` (lines 51-62, 890-924, 1001)
3. Add debug logging in the canonical translator to show which concepts matched and which didn't
4. If concepts are present but names differ, update the mapping
5. If concepts are genuinely absent (Apple may not report some fields), add estimation fallbacks (e.g. `ebitda = operating_income + depreciation`)
6. Test with the AAPL fixture to verify all 4 variables are populated

### Files Changed
- `operator1/clients/us_edgar.py` (concept mapping updates)
- `operator1/clients/canonical_translator.py` (debug logging)
- `operator1/features/derived_variables.py` (fallback computations)

---

## External Issue 9: Stale Data (LOW)

**Warning:** `STALE DATA: Latest filing is 705 days old`

Not a code bug -- likely a cached result from an old run. The pipeline already warns about this. No code change needed unless we want to add a user-facing prompt asking whether to continue with stale data.

---

## External Issue 10: Triple HTTP Requests (MEDIUM)

**Observation:** Every SEC EDGAR filing request is made 3 times

This is a behavior of the `edgartools` library (v5.19.1), not our code. The library initializes 3 internal client instances (3 `set_identity` calls visible in the log). Each filing request routes through all 3 clients.

### Mitigation Options
- Upgrade `edgartools` to a newer version that may have fixed this
- Cache at the application level (our `http_utils.py` disk cache) to avoid redundant network calls
- File an issue with the `edgartools` maintainer

---

## Implementation Order

```
Bug 6 (conformal update)   -- quick fix, 1 line alias
Bug 1 (estimation shape)   -- medium fix, DataFrame guard
Bug 8 (NaN financials)     -- investigation + mapping update
```

Bug 6 is the fastest to fix (add a method alias). Bug 1 needs careful DataFrame handling. Bug 8 requires investigation into what `edgartools` actually returns for Apple's XBRL data.

# Plan: Fix Remaining Pipeline Bugs -- Second Pass (B12-B17)

Bugs found during the second debug pass of all three pipeline run logs. These are separate from the 10 bugs already fixed in PR #1.

---

## Bug Inventory

| # | Bug | Severity | Affected Runs | Files |
|---|-----|----------|---------------|-------|
| B12 | Estimation shape mismatch crashes estimator | HIGH | SEC EDGAR | `operator1/estimation/estimator.py` |
| B13 | Stale filing detection uses wrong date source | MEDIUM | SEC EDGAR | `operator1/features/filing_calendar.py` |
| B15 | 4 Tier 1-2 financial ratios all-NaN for AAPL | HIGH | Both | `operator1/clients/us_edgar.py` |
| B16 | FRED macro missing 4/5 indicators for US | MEDIUM | SEC EDGAR | `operator1/clients/macro_fredapi.py`, `main.py` |
| B17 | pywt not installed -- wavelet decomposition always skipped | LOW | Both | `requirements/stage2-ml.txt` |

---

## B12: Estimation Shape Mismatch

### Problem
SEC EDGAR run at line 2482:
```
Estimation failed (continuing with raw data): Shape of passed values is (1256, 16), indices imply (1256, 17)
```
The estimation engine crashes entirely, so the pipeline falls back to raw (un-imputed) data. This means financial ratios computed from missing data will all be NaN.

### Root Cause
The `run_estimation()` function in [`operator1/estimation/estimator.py:709`](operator1/estimation/estimator.py:709) builds a result DataFrame from the estimation output. During the column assembly phase, it creates column arrays and index arrays separately. When a variable's estimation produces a different number of output columns than expected (e.g., one variable produces 6 companion columns instead of 7, or a column name collision causes dedup to drop one), the shapes mismatch.

This is likely triggered when:
- A variable has the same name as an existing column in the DataFrame
- The `_source` or `_confidence` companion column collides with an existing column
- The number of estimable variables changes between Phase 2 (classification) and Phase 3 (imputation)

### Fix
1. Read [`estimator.py`](operator1/estimation/estimator.py) around line 709-800 to find the exact DataFrame construction
2. Add a try/except around the DataFrame construction that logs the exact column names and shapes on failure
3. Use `pd.concat(axis=1)` to assemble estimation columns instead of constructing from arrays (more robust to shape mismatches)
4. Add a guard that checks column count matches before constructing the DataFrame

### Files to modify
- [`operator1/estimation/estimator.py`](operator1/estimation/estimator.py) -- Fix DataFrame assembly, add shape validation

---

## B13: Stale Filing Detection Uses Wrong Date

### Problem
SEC EDGAR run shows:
```
STALE DATA: Latest filing is 705 days old (threshold: 130 days for us_sec_edgar)
```
Apple files quarterly, so the latest filing should be less than 90 days old, not 705 days.

### Root Cause
[`filing_calendar.py:131`](operator1/features/filing_calendar.py:131) calls `_detect_filing_dates_from_cache(cache)` which detects filing dates from value changes in the cache DataFrame. The detected dates may correspond to `report_date` (fiscal period end) rather than `filing_date` (when the filing was published).

For Apple's Q4 FY2024 (period ending Sep 2024), the report_date would be 2024-09-28, which is ~18 months before March 2026. But the filing_date would be ~Nov 2024 (40 days after period end). The staleness check at line 146 computes `today - latest_detected_date`, which gives 705 days if using the period-end date from an old filing instead of the actual filing date.

### Fix
1. Read `_detect_filing_dates_from_cache()` to understand how it identifies filing dates
2. If the cache has a `filing_date` column, use that instead of detecting from value changes
3. If filing_date is not available, at least look for `report_date` changes and add the market-specific deadline_days as a buffer
4. Add logging to show which dates were detected so the staleness warning is more informative

### Files to modify
- [`operator1/features/filing_calendar.py`](operator1/features/filing_calendar.py) -- Improve filing date detection logic

---

## B15: 4 Tier 1-2 Variables All-NaN for AAPL

### Problem
SEC EDGAR run shows:
```
Forecasting: 4 variables had 0 non-NaN observations: cash_ratio, net_debt_to_ebitda, interest_coverage, current_ratio
```
These are Tier 1-2 survival variables (the most critical for survival analysis).

### Root Cause
The canonical translator already maps `us-gaap:LiabilitiesCurrent`, `us-gaap:InterestExpense`, and `us-gaap:AssetsCurrent` (confirmed in the search results). The issue is upstream:

1. **edgartools may not return these concepts**: Apple's XBRL filings may use different concept names like `us-gaap:InterestExpenseDebtExcludingAmortization` or `us-gaap:InterestPaidNet` instead of the standard `us-gaap:InterestExpense`
2. **The pivot step may drop them**: If the data comes through in long format, some concepts might be filtered out during the pivot
3. **The as-of merge may not forward-fill them**: If these fields appear in the income statement but the merge only forward-fills balance sheet fields

### Fix
1. Add more XBRL concept aliases to the US-GAAP mapping in [`canonical_translator.py`](operator1/clients/canonical_translator.py):
   - `us-gaap:InterestExpenseDebtExcludingAmortization` -> `interest_expense`
   - `us-gaap:InterestPaidNet` -> `interest_expense`
   - `us-gaap:OtherLiabilitiesCurrent` -> (combine with `current_liabilities` if main is missing)
2. Add debug logging in [`us_edgar.py`](operator1/clients/us_edgar.py) to show which concepts were found vs expected
3. After the canonical translation, log which key fields are present and which are missing

### Files to modify
- [`operator1/clients/canonical_translator.py`](operator1/clients/canonical_translator.py) -- Add more XBRL concept aliases for US-GAAP
- [`operator1/clients/us_edgar.py`](operator1/clients/us_edgar.py) -- Add debug logging for concept coverage

---

## B16: FRED Macro Missing 4/5 Indicators for US

### Problem
SEC EDGAR run shows 4 MISSING indicators (GDP, inflation, unemployment, currency) with only interest_rate (FEDFUNDS) succeeding.

### Root Cause
The FRED API code at [`macro_fredapi.py:121-129`](operator1/clients/macro_fredapi.py:121) catches exceptions per-series with `logger.debug()` (not warning), so failures are invisible at INFO level. The series IDs look correct (`A191RL1Q225SBEA`, `CPIAUCSL`, `UNRATE`, `DTWEXBGS`), so the failure is likely:

1. **FRED API key tier limitation**: Some series require specific API permissions
2. **FRED series discontinued or renamed**: Series IDs can change over time
3. **Network timeout**: Individual series fetches may time out silently
4. **Date range issue**: Some quarterly/annual series may not have data in the recent date range

### Fix
1. Change `logger.debug` to `logger.warning` for FRED fetch failures (line 129) so they're visible in pipeline logs
2. Add per-indicator fallback series IDs (e.g., if `A191RL1Q225SBEA` fails, try `GDPC1` for real GDP)
3. Add a try-with-fallback pattern:
   ```python
   _FRED_SERIES_WITH_FALLBACK = {
       "gdp_growth": ["A191RL1Q225SBEA", "GDPC1"],
       "inflation_rate_yoy": ["CPIAUCSL", "CPILFESL"],
       ...
   }
   ```
4. Log the actual FRED error message to diagnose whether it's auth, not-found, or timeout

### Files to modify
- [`operator1/clients/macro_fredapi.py`](operator1/clients/macro_fredapi.py) -- Add warning-level logging, fallback series IDs

---

## B17: pywt Not Installed

### Problem
Both runs show:
```
pywt not installed, skipping wavelet decomposition
```
The wavelet transform in cycle decomposition is always skipped because PyWavelets is not in the dependency list.

### Fix
Add `PyWavelets>=1.7` to [`requirements/stage2-ml.txt`](requirements/stage2-ml.txt) and [`requirements.txt`](requirements.txt).

### Files to modify
- [`requirements/stage2-ml.txt`](requirements/stage2-ml.txt) -- Add `PyWavelets>=1.7`
- [`requirements.txt`](requirements.txt) -- Add `PyWavelets>=1.7` in the ML section

---

## Execution Order

```
B15 (Tier 1-2 all-NaN)        -- highest impact on analysis quality
B12 (estimation shape crash)   -- second highest, causes entire estimation skip
B16 (FRED macro missing)       -- improves macro data coverage
B13 (stale filing detection)   -- false positive staleness warnings
B17 (pywt missing)             -- low priority, add to requirements
```

## Todo List

```
[ ] B15: Add more XBRL concept aliases for US-GAAP, add concept coverage logging
[ ] B12: Fix estimation DataFrame assembly, add shape validation with try/except
[ ] B16: Upgrade FRED logging to warning level, add fallback series IDs
[ ] B13: Improve filing date detection to prefer filing_date over report_date
[ ] B17: Add PyWavelets to stage2-ml and main requirements
[ ] Run tests
[ ] Push and update PR
```

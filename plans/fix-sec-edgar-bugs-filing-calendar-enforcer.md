# Fix SEC EDGAR Pipeline Bugs + Filing Calendar Data Quality Enforcer

## Overview

4 code bugs from the SEC EDGAR pipeline run + a new filing calendar enforcement system that marks stale data in the cache, lowers model confidence during data gaps, and uses LLM-parsed filing dates as supplementary calendar input.

---

## Phase 1: Quick Bug Fixes

### Fix 1A: ConformalCalibrator missing `update()` method

**File:** `operator1/models/conformal.py`
**Error:** `'ConformalCalibrator' object has no attribute 'update'`
**Fix:** Add `update()` as an alias for `add_score()` on `ConformalCalibrator` class

```python
def update(self, variable: str, residual: float) -> None:
    """Alias for add_score -- backward compatibility with main.py caller."""
    self.add_score(variable, residual)
```

### Fix 1B: Estimation shape mismatch

**File:** `operator1/estimation/estimator.py`
**Error:** `Shape of passed values is (1256, 16), indices imply (1256, 17)`
**Fix:** Before DataFrame construction, filter the column list to match the actual result array width. Add a debug log identifying which variable was skipped.

```python
# Before constructing the result DataFrame:
if result_array.shape[1] != len(expected_columns):
    logger.warning(
        "Estimation column mismatch: expected %d, got %d. "
        "Skipped variables: %s",
        len(expected_columns), result_array.shape[1],
        set(expected_columns) - set(actual_columns),
    )
    expected_columns = actual_columns  # align to what was produced
```

### Fix 1C: Triple HTTP requests in edgartools

**File:** `operator1/clients/us_edgar.py`
**Fix:** Add a per-run URL memo cache that intercepts duplicate requests:

```python
_url_memo: dict[str, Any] = {}

def _cached_filing_fetch(url: str) -> Any:
    if url in _url_memo:
        return _url_memo[url]
    result = _original_fetch(url)
    _url_memo[url] = result
    return result
```

### Fix 1D: SEC EDGAR NaN financial variables

**File:** `operator1/clients/us_edgar.py`, `operator1/features/derived_variables.py`
**Fix:**
1. Add debug logging in `us_edgar.py` to show which XBRL concepts were found vs expected
2. Add fallback computations in `derived_variables.py`:
   - `ebitda = operating_income + depreciation_amortization` (if `ebitda` is NaN)
   - `interest_coverage = operating_income / interest_expense` (if `ebit` unavailable)
3. Check if edgartools returns `AssetsCurrent` as a different concept name

---

## Phase 2: Filing Calendar as Active Data Quality Enforcer

### 2A: Freshness scoring in the cache

**File:** `operator1/features/filing_calendar.py`

After analyzing the filing schedule, inject a `filing_freshness` column into the daily cache:

```
filing_freshness = 1.0   on the day a filing was published
                  decay   linearly toward 0 as days pass beyond expected frequency
                  0.0     when data is older than 2x the expected filing interval
```

For a quarterly filer (expected every 90 days):
- Days 0-90 after filing: freshness = 1.0 (within expected window)
- Days 91-180: freshness decays linearly from 1.0 to 0.0
- Days 180+: freshness = 0.0 (severely stale)

New columns added to cache:
- `filing_freshness`: 0.0 to 1.0 continuous freshness score
- `filing_gap_flag`: 1 if current day is beyond expected filing window
- `days_since_last_filing`: integer days since the most recent filing

### 2B: LLM-parsed filing dates as supplementary input

**File:** `operator1/features/filing_calendar.py`

Some PIT wrappers only provide `report_date` (fiscal period end) but not `filing_date` (when the filing was actually published). The LLM filing extractor (`llm_filing_extractor.py`) can parse filing dates from PDF filings.

Add a function that:
1. Checks if `filing_date` column exists in the financial data
2. If not, checks if the LLM filing extractor produced filing dates for this company
3. If LLM dates are available, merge them into the calendar analysis
4. Falls back to using `report_date + market-specific lag` as an estimate

```python
def _infer_filing_dates(
    financials_df: pd.DataFrame,
    market_id: str,
    llm_filing_dates: dict[str, str] | None = None,
) -> pd.Series:
    """Infer filing publication dates from available sources.
    
    Priority:
    1. filing_date column (if present in the data)
    2. LLM-parsed filing dates (from PDF extraction)
    3. report_date + market-specific lag estimate
    """
```

Market-specific filing lag estimates:
- US (SEC EDGAR): 40 days for 10-K, 35 days for 10-Q
- Japan (J-Quants): 45 days for annual, 30 days for quarterly
- Korea (DART): 90 days for annual, 45 days for quarterly
- UK (Companies House): 270 days for annual (UK has a very long deadline)
- Brazil (CVM): 90 days for annual
- Taiwan (MOPS): 60 days for annual

### 2C: Estimation engine integration

**File:** `operator1/estimation/estimator.py`

Use `filing_freshness` to weight the estimation:
- When `filing_freshness` is high (1.0): treat forward-filled financial data as observed
- When `filing_freshness` is low (< 0.5): treat forward-filled data as partially missing
  - Lower the `_confidence` score for estimated values in stale periods
  - The estimation engine already produces `{var}_confidence` -- reduce it by `filing_freshness`

```python
# After estimation, adjust confidence by freshness
if "filing_freshness" in cache.columns:
    for var in estimated_vars:
        conf_col = f"{var}_confidence"
        if conf_col in cache.columns:
            cache[conf_col] = cache[conf_col] * cache["filing_freshness"]
```

### 2D: Model confidence weighting

**File:** `operator1/models/forecasting.py`

When building training data for temporal models, weight recent observations by `filing_freshness`:
- Fresh data (filing_freshness = 1.0): full weight
- Stale data (filing_freshness < 0.5): reduced weight
- Very stale data (filing_freshness = 0.0): near-zero weight (still included to avoid data holes)

This naturally makes models learn less from stale forward-filled periods.

### 2E: Report visualization

**File:** `operator1/report/report_generator.py`

Add stale-region shading to price and financial health charts:
- Light red/orange background shading on chart areas where `filing_gap_flag = 1`
- Legend entry: "Data gap -- forward-filled from prior filing"
- Filing calendar section in the report shows expected vs actual filing timeline

---

## Phase 3: Wire into pipeline

**File:** `main.py`

Current flow:
```
Step 5d: filing_calendar -> analyze_filing_calendar() -> result stored in profile
```

Updated flow:
```
Step 5d: filing_calendar -> analyze_filing_calendar() -> result stored in profile
Step 5d.1: inject_filing_freshness(cache, filing_calendar_result, market_id)
           -> adds filing_freshness, filing_gap_flag, days_since_last_filing to cache
Step 5d.2: if llm_filing_dates available, merge into calendar
Step 5e (estimation): confidence adjusted by filing_freshness
Step 6 (forecasting): sample weights adjusted by filing_freshness
```

---

## Implementation Order

```
Phase 1A: ConformalCalibrator update() alias        -- 5 min
Phase 1B: Estimation shape guard                    -- 15 min  
Phase 1C: Triple HTTP memo cache                    -- 15 min
Phase 1D: SEC EDGAR NaN field investigation + fix   -- 30 min
Phase 2A: Filing freshness scoring                  -- 30 min
Phase 2B: LLM filing date integration               -- 20 min
Phase 2C: Estimation confidence adjustment           -- 15 min
Phase 2D: Model weight adjustment                    -- 15 min
Phase 2E: Report stale-region visualization          -- 20 min
Phase 3:  Pipeline wiring                            -- 15 min
```

---

## Files Changed Summary

| File | Phase | Changes |
|------|-------|---------|
| `operator1/models/conformal.py` | 1A | Add `update()` alias |
| `operator1/estimation/estimator.py` | 1B, 2C | Shape guard + freshness-weighted confidence |
| `operator1/clients/us_edgar.py` | 1C, 1D | Memo cache + debug logging for XBRL concepts |
| `operator1/features/derived_variables.py` | 1D | Fallback computations for derived ratios |
| `operator1/features/filing_calendar.py` | 2A, 2B | Freshness scoring + LLM filing date integration |
| `operator1/models/forecasting.py` | 2D | Freshness-weighted training samples |
| `operator1/report/report_generator.py` | 2E | Stale-region chart shading |
| `main.py` | 3 | Wire freshness injection after filing calendar |

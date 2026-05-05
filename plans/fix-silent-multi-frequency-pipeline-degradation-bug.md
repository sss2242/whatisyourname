# Fix: Silent Multi-Frequency Pipeline Degradation Bug

## Problem Statement

The multi-frequency pipeline (Stage 7.4) runs without proper frequency-specific source data and produces results that look valid but are semantically wrong. When raw statement DataFrames (income_df, balance_df, cashflow_df) are empty or incomplete -- because Stage 1 data fetch failed, timed out, or was rate-limited -- the MF pipeline silently falls back to resampling the daily cache instead of using actual filing data. No warning is logged, no flag is set, and the profile/report presents these degraded results as if they were computed from real frequency-specific data.

## Root Cause Analysis

### The Silent Fallback Path

In `run_7_4_0_resample_prep()` at `operator1/stages/stage7_integration.py:195-226`:

```python
_has_raw = any(df is not None and not df.empty for df in [income_df, balance_df, cashflow_df])

for freq in frequencies:  # ["A", "Q", "M", "W", "D"]
    if freq in ("Q", "A", "W", "M", "S") and _has_raw:
        resampled = build_cache_from_raw_filings(...)     # CORRECT path
    else:
        resampled = resample_cache_to_frequency(cache, ...) # SILENT FALLBACK
```

When `_has_raw` is False, ALL non-daily frequencies take the fallback branch.

### What the Fallback Actually Produces

`resample_cache_to_frequency()` resamples the daily cache (504 rows of forward-filled interpolated data) to lower frequencies:

- **Flow variables** (revenue, net_income): The daily cache has the same forward-filled quarterly total on every row. Resampling detects flat values and uses `last()`, giving the quarterly total as if it were the annual total. Revenue growth computes as 0% because every period shows the same value.
- **Stock variables** (total_assets, total_debt): The daily cache forward-fills between filings. Resampling takes `last()`, giving the most recent value in each period. This is approximately correct for stock variables but loses actual filing-date granularity.
- **OHLCV**: Correctly aggregated (open=first, high=max, low=min, close=last, volume=sum). This is the only part that works properly.

### Why It Doesn't Crash

1. `resample_cache_to_frequency()` always returns a valid `ResampledCache` even with meaningless data
2. The `n_periods < 3` guard only filters by count, not by data quality (Q gives ~8 periods, M gives ~24, W gives ~104 -- all pass)
3. `run_single_frequency_pipeline()` wraps every step in try/except with graceful fallback
4. The stage runner has graceful degradation for non-critical sub-stages (7.4.x are non-critical)

### Impact

The MF fusion results in the profile (`profile["multi_frequency"]`) contain:
- **Regime consensus**: Based on OHLCV-only data at each frequency (missing fundamental context)
- **Survival probability**: Based on forward-filled financial ratios that don't change across periods (always shows same survival state)
- **Predictions**: Based on resampled interpolated data (circular: daily -> annual -> predict, where annual is just aggregated daily)
- **Cross-frequency momentum**: Direction agreement is artificially high because all frequencies see the same forward-filled financial data

## Fix Design

### Fix 1: Log WARNING when falling back to resampled daily cache

**File:** `operator1/stages/stage7_integration.py`
**Location:** `run_7_4_0_resample_prep()`, around line 195

Add a clear WARNING log when `_has_raw` is False, explaining that MF results will be degraded. Also log per-frequency when the fallback path is taken.

### Fix 2: Add `data_source` field to `ResampledCache`

**File:** `operator1/features/frequency_resampler.py`
**Location:** `ResampledCache` dataclass

Add a `data_source: str` field with values:
- `"raw_filings"` -- built from actual filing data via `build_cache_from_raw_filings()`
- `"resampled_daily"` -- resampled from the daily cache (degraded)
- `"daily_native"` -- daily frequency (always correct)

This lets downstream consumers know whether the data is trustworthy.

### Fix 3: Tag degraded results in the MF fusion output

**File:** `operator1/stages/stage7_integration.py`
**Location:** `_run_7_4_single_freq()` and `run_7_4_6_fusion()`

Pass the `data_source` tag through to `FrequencyResult` so the fusion layer knows which frequencies have degraded data. The fusion output should include a `degraded_frequencies` list in the profile.

### Fix 4: Skip MF financial analysis on degraded frequencies

**File:** `operator1/steps/multi_frequency_runner.py`
**Location:** `run_single_frequency_pipeline()`

When the resampled cache has `data_source == "resampled_daily"`, skip financial-statement-dependent analysis (survival mode, financial health, hedge fund) and only run OHLCV-based analysis (regime detection from price data, volatility, technical patterns). This prevents the pipeline from producing fake financial ratios at non-daily frequencies.

### Fix 5: Profile disclosure

**File:** `operator1/stages/stage7_integration.py` (profile injection in main.py Step 7)

When MF results are degraded, add a `data_quality_warning` to `profile["multi_frequency"]` that the report generator can render as a caveat.

## Files to Modify

| File | Changes |
|------|---------|
| `operator1/features/frequency_resampler.py` | Add `data_source` field to `ResampledCache` |
| `operator1/stages/stage7_integration.py` | WARNING logs, pass `data_source` through, tag degraded results |
| `operator1/steps/multi_frequency_runner.py` | Skip financial analysis on degraded caches |
| `operator1/pipeline_state.py` | Persist `data_source` in MF cache metadata |
| `operator1/models/frequency_fusion.py` | Include `degraded_frequencies` in fusion output |

## Implementation Order

1. Add `data_source` field to `ResampledCache` dataclass
2. Set `data_source` correctly in both `build_cache_from_raw_filings()` and `resample_cache_to_frequency()`
3. Persist `data_source` in PipelineState MF cache save/load
4. Add WARNING logs in `run_7_4_0_resample_prep()` when falling back
5. Pass `data_source` to `FrequencyResult` in `run_single_frequency_pipeline()`
6. Skip financial analysis on degraded caches in `run_single_frequency_pipeline()`
7. Include `degraded_frequencies` in fusion output
8. Add profile disclosure warning

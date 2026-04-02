# Fix 7 Debug Scan Findings in Multi-Frequency Pipeline

## Overview

The debug scan of the last commit (multi-frequency forecasting, `a324a0e`) found 7 issues across 6 files. All issues are in the new multi-frequency pipeline code -- the existing 50 analytical modules are clean.

## Fix Plan

### F1. Flow variable inflation in frequency_resampler.py (DATA BUG)

**File:** `operator1/features/frequency_resampler.py`
**Line:** 262-263

**Current code:**
```python
# Flow variables: sum over period
if flow_present:
    flow_resampled = df[flow_present].resample(rule).sum()
```

**Problem:** When flat ffill was used (single-filing companies), daily values are the FULL periodic total. Summing multiplies by days-in-period.

**Fix:** Detect whether flow variables were distributed (interpolated) or flat-ffilled by checking for value changes within a period. If all values in a period are identical (flat ffill), use `.last()` instead of `.sum()`. Implementation:

```python
if flow_present:
    # Check if flow variables were distributed to daily or flat-ffilled
    # If all values in a resample period are identical, use last() (undistributed)
    # If values vary within periods, use sum() (distributed by interpolator)
    sample_col = flow_present[0]
    sample_series = df[sample_col].dropna()
    if len(sample_series) > 1:
        # Count unique values in first resample period
        first_period = sample_series.resample(rule).nunique().iloc[0] if len(sample_series) > 0 else 1
        if first_period <= 1:
            # Flat ffill -- use last value (NOT sum)
            flow_resampled = df[flow_present].resample(rule).last()
        else:
            # Distributed by interpolator -- sum reconstructs period total
            flow_resampled = df[flow_present].resample(rule).sum()
    else:
        flow_resampled = df[flow_present].resample(rule).last()
    parts.append(flow_resampled)
```

**Risk:** Low. Uses the same `.last()` aggregation that stock variables already use. No new dependencies.

---

### F2. Multi-frequency results NOT rendered in reports (MISSING WIRING)

**File:** `operator1/report/report_generator.py`

**Fix:** Add a `_build_multi_frequency_section()` builder function and wire it as section 2005 in `TIER_SECTIONS` for Premium tier.

**Section content:**
- Regime consensus table (per-frequency regime + agreement ratio)
- Survival fusion (harmonic mean + per-frequency probabilities + weakest link)
- Cross-frequency predictions table (top 10, with contributing frequencies)
- Frequency disagreement interpretation text

**Wire in `_build_fallback_report()` section_builders dict:**
```python
2005: ("21.11. Multi-Frequency Analysis", _build_multi_frequency_section(profile)),
```

**Wire in `TIER_SECTIONS`:**
- Add `2005` to Premium set
- Optionally add to Pro set (lighter version)

**Risk:** None. New section builder only, no changes to existing logic.

---

### F3. `multi_frequency` key missing from profile schema (MISSING DECLARATION)

**File:** `operator1/report/profile_schema.py`
**Line:** 53-77

**Fix:** Add `"multi_frequency"` to `OPTIONAL_PROFILE_KEYS` set.

```python
OPTIONAL_PROFILE_KEYS: set[str] = {
    ...
    "multi_frequency",
}
```

**Risk:** None. Purely declarative.

---

### F4. `walk_forward_mae` never populated (DEAD CODE PATH)

**File:** `operator1/steps/multi_frequency_runner.py`
**Line:** 247-260

**Fix:** After forecasting succeeds, run a lightweight walk-forward evaluation to populate `walk_forward_mae`. Only for frequencies with enough data (n_periods >= 30).

```python
# After forecasting block (line 260):
if resampled.n_periods >= 30 and forecast_result is not None:
    try:
        from operator1.models.walk_forward import run_walk_forward
        from operator1.analysis.survival_timeline import compute_survival_timeline
        _wf_tl = compute_survival_timeline(cache)
        _wf_modes = (
            _wf_tl.timeline["survival_mode"]
            if hasattr(_wf_tl, "timeline") and "survival_mode" in _wf_tl.timeline.columns
            else None
        )
        _wf_switches = (
            _wf_tl.timeline["switch_point"]
            if hasattr(_wf_tl, "timeline") and "switch_point" in _wf_tl.timeline.columns
            else None
        )
        wf_result = run_walk_forward(cache, _wf_modes, _wf_switches)
        if wf_result and wf_result.fitted and not pd.isna(wf_result.overall_mae):
            walk_forward_mae = wf_result.overall_mae
    except Exception:
        pass
```

**Risk:** Low. Walk-forward is fast (~100ms on monthly data). Wrapped in try/except.

---

### F5. Remove dead `model_metrics_summary` field (DEAD FIELD)

**File:** `operator1/steps/multi_frequency_runner.py`
**Line:** 77

**Fix:** Remove the field from the `FrequencyResult` dataclass and the assignment at line 307.

**Risk:** None. Field was never read.

---

### F6. Multi-frequency NOT in backtest_runner.py (MISSING WIRING)

**File:** `backtest_runner.py`

**Fix:** Add multi-frequency pipeline call at the end of `run_stage2()` (after all temporal models). Store result in `BacktestState`. In `run_stage3()`, inject into profile.

In `BacktestState.__init__()`:
```python
self.multi_frequency_result = None
```

At end of `run_stage2()`:
```python
# Multi-frequency forecasting
try:
    from operator1.steps.multi_frequency_runner import run_multi_frequency_pipeline
    from operator1.models.frequency_fusion import fuse_multi_frequency_results
    mf = run_multi_frequency_pipeline(
        daily_cache=cache, secrets=state._secrets,
        market_id=state.market_id, ticker=state.company,
        reference_date=datetime.strptime(state.end_date, "%Y-%m-%d").date(),
    )
    if mf and mf.results:
        state.multi_frequency_result = fuse_multi_frequency_results(mf)
except Exception as exc:
    logger.warning("Multi-frequency failed: %s", exc)
```

In `run_stage3()` profile injection:
```python
if state.multi_frequency_result is not None and hasattr(state.multi_frequency_result, "available") and state.multi_frequency_result.available:
    profile["multi_frequency"] = state.multi_frequency_result.to_profile_dict()
else:
    profile["multi_frequency"] = {"available": False}
```

**Risk:** Low. Same pattern as main.py. Wrapped in try/except.

---

### F7. Remove dead prior_* column injection (DEAD FEATURE)

**File:** `operator1/steps/multi_frequency_runner.py`
**Line:** 331-333

**Fix:** Remove the three `cache[f"prior_{ctx.frequency}_*"]` assignments from `_apply_prior_context()`. Keep the frequency disagreement warning log (useful for debugging).

The function becomes:
```python
def _apply_prior_context(cache, ctx, current_freq):
    # Log frequency disagreements
    if "company_survival_mode_flag" in cache.columns:
        current_survival = cache["company_survival_mode_flag"].iloc[-1] if len(cache) > 0 else 0
        if current_survival == 0 and ctx.survival_probability_latest < 0.5:
            logger.warning(...)
```

**Risk:** None. Columns were never consumed.

---

## Execution Order

1. **F1** (data bug -- highest priority, prevents wrong data)
2. **F3** (schema -- quick, 1 line)
3. **F5** (dead field -- quick, clean up)
4. **F7** (dead feature -- quick, clean up)
5. **F4** (populate walk_forward_mae -- enables F2 to show accuracy)
6. **F2** (report section -- largest change, new function)
7. **F6** (backtest integration -- last, depends on F1-F5 being clean)

## Files Modified

| File | Changes |
|------|---------|
| `operator1/features/frequency_resampler.py` | F1: Smart flow variable aggregation |
| `operator1/report/report_generator.py` | F2: New `_build_multi_frequency_section()` + section 2005 |
| `operator1/report/profile_schema.py` | F3: Add `multi_frequency` to optional keys |
| `operator1/steps/multi_frequency_runner.py` | F4: Populate walk_forward_mae, F5: Remove dead field, F7: Remove dead columns |
| `backtest_runner.py` | F6: Add multi-frequency to Stage 2/3 |

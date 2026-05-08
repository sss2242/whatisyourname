# Fix Guide: Remaining Frequency-First Pipeline Limitations

6 items to fix so every frequency works correctly end-to-end across all entry points.

---

## Fix 1: main.py inline flow -- survival + FH re-run after MF pipeline

**Problem:** `main.py` computes survival (Step 5) and FH (Step 5d) with D-freq BEFORE the MF pipeline (Step 6.7). After MF fusion forward-fills correct ratios, survival and FH are NOT recomputed. The profile gets stale survival_probability and fh_composite_score.

**File:** `main.py` -- after the multi-frequency pipeline step (around line 2760)

**Fix:** After `run_multi_frequency_pipeline()` or the staged runner completes, add:

```python
# Post-MF-fusion recalibration: re-run survival + FH with correct Q/A ratios
if multi_frequency_result is not None and multi_frequency_result.available:
    try:
        cache["company_survival_mode_flag"] = compute_company_survival_flag(cache, freq="Q")
        cache["survival_probability"] = compute_survival_probability(cache)
        cache = compute_hierarchy_weights(cache)
        cache, fh_result = compute_financial_health(cache, hierarchy_weights=weights, freq="Q")
        logger.info("Post-MF recalibration: survival=%.3f, FH=%.1f",
                     float(cache["survival_probability"].dropna().iloc[-1]),
                     fh_result.latest_composite)
    except Exception as exc:
        logger.debug("Post-MF recalibration failed: %s", exc)
```

**Lines changed:** ~12

---

## Fix 2: _CURRENT_FREQ thread safety

**Problem:** `_CURRENT_FREQ` is a module-level global in `derived_variables.py`. Not thread-safe.

**Fix:** Replace global with a threading.local object:

```python
import threading
_freq_context = threading.local()

def _get_freq() -> str:
    return getattr(_freq_context, 'freq', 'D')

def compute_derived_variables(df, freq="D"):
    _freq_context.freq = freq.upper() if freq else "D"
    ...
```

And replace all `_CURRENT_FREQ` reads with `_get_freq()`.

**Files:** `operator1/features/derived_variables.py`
**Lines changed:** ~25 (replace 13 `_CURRENT_FREQ` reads with `_get_freq()` + add threading import + context class)

---

## Fix 3: FH re-run after Stage 2.F fusion

**Problem:** Stage 2.F re-runs survival but NOT financial health. FH T5 uses PE/EV which are now correct, but `fh_growth_score` and `fh_composite_score` still reflect pre-fusion values.

**File:** `operator1/stages/stage2_freq_pipeline.py` in `run_2_F_fusion()` -- after the survival re-run

**Fix:** Add FH recomputation:

```python
# Also re-run financial health with correct Q/A ratios
try:
    from operator1.models.financial_health import compute_financial_health
    cache, _fh = compute_financial_health(cache, freq="Q")
    state.fh_result = _fh
    logger.info("Post-fusion FH re-run: composite=%.1f", _fh.latest_composite)
except Exception as exc:
    logger.debug("Post-fusion FH re-run failed: %s", exc)
```

**Lines changed:** ~8

---

## Fix 4: Pre-flight validation for Stage 2.0

**Problem:** Stage 2.0 (resample prep) needs raw statement DataFrames but the staged runner's `_SUBSTAGE_REQUIREMENTS` doesn't validate this.

**File:** `operator1/stages/runner.py` -- `_SUBSTAGE_REQUIREMENTS` dict

**Fix:** Add entry:

```python
_SUBSTAGE_REQUIREMENTS: dict[str, list[str]] = {
    "2.0": ["cache"],  # needs cache + raw DFs (checked inside resample prep)
    "2.1": ["income_df"],
    ...
}
```

**Lines changed:** 1

---

## Fix 5: HF engine cache-dependent values verified correct

**Problem:** HF engine reads PE, fcf_yield from cache for valuation modules. Need to verify these are correct after Stage 2.F forward-fill.

**Status:** Already handled. HF runs at Stage 7.5 (after 2.F) in run_backtest_staged.py. In backtest_runner.py, HF is called inside `run_stage3()` which runs AFTER `run_stage2()`. In main.py, HF reads from cache which has MF fusion results. No code change needed -- just verification.

**Fix:** No code change. Add a verification log in HF engine:

```python
# In hedge_fund/engine.py, before using PE/fcf_yield from cache:
_pe = get_cache_latest(cache, "pe_ratio_calc")
if _pe is not None and _pe > 200:
    logger.warning("HF: PE=%.1f seems distorted (>200), check freq pipeline", _pe)
```

**Lines changed:** ~5 (optional verification log)

---

## Fix 6: Beneish M-Score period-over-period changes

**Problem:** Beneish M-Score computes YoY changes from forward-filled daily values. At D freq, these changes detect filing transitions (cliff changes) rather than true quarterly deltas.

**File:** `operator1/models/financial_health.py` in `compute_beneish_m_score()`

**Fix:** Add freq parameter and use freq-appropriate shift for period-over-period changes:

```python
def compute_beneish_m_score(df, freq="D"):
    _yoy_shift = {"A": 1, "S": 2, "Q": 4, "M": 12, "W": 52, "D": 252}.get(freq, 252)
    # Use _yoy_shift instead of hardcoded shift(252) for DSRI, GMI, etc.
```

At Q freq: `shift(4)` compares current quarter to 4 quarters ago (true YoY).
At D freq: `shift(252)` compares current day to 252 days ago (may cross filing boundaries unevenly).

**Lines changed:** ~10

---

## Execution Order

1. Fix 2 (thread safety) -- smallest change, prevents future bugs
2. Fix 3 (FH re-run after 2.F) -- 8 lines in stage2_freq_pipeline.py
3. Fix 4 (pre-flight validation) -- 1 line
4. Fix 6 (Beneish freq-aware) -- 10 lines
5. Fix 1 (main.py post-MF recalibration) -- 12 lines
6. Fix 5 (verification log) -- 5 lines, optional

Total: ~61 lines across 5 files.

---

## Per-Frequency Verification Checklist

After all fixes, every frequency should produce correct results:

| Check | D | W | M | Q | S | A |
|-------|---|---|---|---|---|---|
| Returns/vol/drawdown computed | Y | Y | Y | Y | Y | Y |
| Stock/stock ratios correct | Y | Y | Y | Y | Y | Y |
| Flow-based ratios correct | via 2.F | via 2.F | via 2.F | native | native | native |
| Survival triggers appropriate | 8 D-triggers | 6 W-triggers | 4 M-triggers | 9 Q-triggers | 9 S-triggers | 8 A-triggers |
| survival_probability correct | post-2.F re-run | post-2.F re-run | post-2.F re-run | native | native | native |
| FH composite correct | post-2.F re-run | post-2.F re-run | post-2.F re-run | native | native | native |
| Altman Z x3/x5 correct | mult=1 backward | mult=1 | mult=1 | mult=4 | mult=2 | mult=1 |
| Runway months correct | /12 backward | /1 | /1 | /3 | /6 | /12 |
| DSO/DIO/DPO period days | 90 approx | 90 | 90 | 90 | 180 | 365 |
| TTM values correct | _rolling_4q_ttm | same | same | rolling(4).sum | rolling(2).sum | =annual |
| Technical indicators | native | native | skip | skip | skip | skip |
| Institutional triggers | native | native | skip | skip | skip | skip |
| Merton DD trigger | native | native | skip | skip | skip | skip |
| HF ratios from cache | post-2.F | post-2.F | post-2.F | native | native | native |
| Beneish M-Score correct | shift(252) | shift(52) | shift(12) | shift(4) | shift(2) | shift(1) |

# Frequency-Aware Model Parameter Scaling

## Problem

The multi-frequency pipeline runs 25+ temporal models at 5 frequencies (A/Q/M/W/D), but all models use daily-calibrated parameters. At lower frequencies, this causes:
- Monte Carlo simulating 252 weekly steps (5 years) instead of 52 (1 year)
- Forward pass warmup of 60 exceeding total data length (6 annual rows)
- DTW/Transformer lookback of 60 exceeding data at A/Q
- Horizon labels like "21d" meaningless at weekly frequency (it's actually 21 weeks)

## Architecture: Frequency Context Object

Create a single `FrequencyParams` dataclass that maps frequency to correctly scaled parameters. Pass it through the MF pipeline so every model receives frequency-appropriate values.

```
FREQUENCY_SCALING = {
    "D": {"steps_per_year": 252, "steps_per_quarter": 63, "steps_per_month": 21},
    "W": {"steps_per_year": 52,  "steps_per_quarter": 13, "steps_per_month": 4},
    "M": {"steps_per_year": 12,  "steps_per_quarter": 3,  "steps_per_month": 1},
    "Q": {"steps_per_year": 4,   "steps_per_quarter": 1,  "steps_per_month": 0.33},
    "A": {"steps_per_year": 1,   "steps_per_quarter": 0.25, "steps_per_month": 0.08},
}
```

---

## Fix F1: Monte Carlo horizon scaling

**File:** `operator1/models/monte_carlo.py`
**Lines:** 58-63 (DEFAULT_HORIZONS), 1055-1057, 1428-1429

**Current:**
```python
DEFAULT_HORIZONS = {"1d": 1, "5d": 5, "21d": 21, "252d": 252}
```

**Fix:** Accept an optional `frequency` parameter. When provided, scale horizons:
```python
def _scale_horizons(frequency: str) -> dict:
    spy = FREQUENCY_SCALING[frequency]["steps_per_year"]
    return {
        "1p": 1,                          # 1 period
        "5p": min(5, spy // 4),           # ~1 week equivalent
        "1q": spy // 4,                   # 1 quarter equivalent
        "1y": spy,                        # 1 year equivalent
    }
```

Also scale `_FILING_INTERVALS` and `horizon_steps` in `run_multivariate_monte_carlo`.

---

## Fix F2: Forward pass warmup scaling

**File:** `operator1/models/forecasting.py`
**Line:** 3509

**Current:**
```python
warmup_days: int = 60,
```

**Fix:** Cap warmup to a fraction of available data:
```python
# Inside run_forward_pass, after receiving cache:
effective_warmup = min(warmup_days, max(3, len(cache) // 3))
```

This ensures:
- Annual (6 rows): warmup = 2, leaving 4 prediction steps
- Quarterly (15 rows): warmup = 5, leaving 10 steps
- Monthly (61 rows): warmup = 20, leaving 41 steps
- Weekly (158 rows): warmup = 52, leaving 106 steps
- Daily (502 rows): warmup = 60 (unchanged)

---

## Fix F3: Walk-forward window scaling

**File:** `operator1/models/walk_forward.py`

**Fix:** Same pattern as forward pass -- cap evaluation window to `len(cache) // 3`:
```python
eval_start = max(warmup, len(cache) - max(len(cache) // 3, 10))
```

---

## Fix F4: DTW lookback scaling

**File:** `operator1/models/dtw_analogs.py`

**Current:** `window=63` hardcoded default

**Fix:** `window = min(63, max(5, len(cache) // 4))`

---

## Fix F5: Transformer lookback scaling

**File:** `operator1/models/transformer_forecaster.py`
**Line:** 3162 (lookback=60), 420 (horizons hardcoded)

**Fix:**
```python
lookback = min(lookback, max(5, len(cache) // 3))
# Scale horizons to frequency
if horizons is None:
    spy = len(cache)  # approximate steps_per_year from data
    horizons = {"1p": 1, "short": max(1, spy//12), "mid": max(1, spy//4), "long": spy}
```

---

## Fix F6: Prediction aggregator horizon awareness

**File:** `operator1/models/prediction_aggregator.py`

**Fix:** The aggregator uses `HORIZONS` dict from forecasting.py. When called from MF pipeline, the horizons should match what the frequency-specific forecasting produced. This is already handled by the fusion engine (frequency_fusion.py) which maps per-frequency predictions to calendar horizons. No change needed here -- the per-frequency aggregator just needs to receive the right horizon labels from forecasting.

---

## Fix F7: Regime shift predictor scaling

**File:** `operator1/models/regime_shift_predictor.py`

**Fix:** Accept `steps_per_year` parameter to convert transition probabilities from "per-step" to "per-day" basis for reporting.

---

## Fix F8: Backtest runner MF frequency parameter passing

**File:** `backtest_runner.py` (functions `_bt_mf_run_freq`, `run_stage2a2`, `run_stage2b`, `run_stage2c`)

**Fix:** When creating per-frequency `BacktestState`, set a `._frequency` attribute and pass it through to:
- `run_monte_carlo(... frequency=freq_state._frequency)`
- `run_forward_pass(... warmup_days=scaled_warmup)`
- `train_transformer(... lookback=scaled_lookback)`
- `find_historical_analogs(... window=scaled_window)`

---

## Execution Order

1. **F2** (forward pass warmup cap) -- highest impact, simplest change (1 line)
2. **F1** (MC horizon scaling) -- high impact, moderate complexity
3. **F3** (walk-forward window) -- same pattern as F2
4. **F4** (DTW lookback) -- same pattern
5. **F5** (Transformer lookback) -- same pattern
6. **F7** (regime shift scaling) -- moderate
7. **F8** (backtest runner wiring) -- connects everything
8. **F6** (prediction aggregator) -- may not need changes

## Approach

The simplest approach that doesn't break existing daily behavior: **cap all window/warmup parameters to `min(default, len(cache) // 3)`** at the start of each model function. This requires no new parameters, no new data structures, and automatically adapts to whatever frequency the cache happens to be. Models that need 60 warmup days but only have 6 annual rows will get warmup=2 and run with reduced accuracy (expected and correct).

For Monte Carlo horizons specifically, we need the frequency context to produce meaningful calendar-time labels. This is the one place where a `frequency` parameter is genuinely needed.

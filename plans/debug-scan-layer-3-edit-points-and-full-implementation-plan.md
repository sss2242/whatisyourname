# Debug Scan: Layer 3 Edit Points, Inputs, Outputs & Full Implementation Plan

*Systematic scan of every file affected by the Layer 3 temporal model enhancements*

---

## 1. Files That Will Be DIRECTLY EDITED

### 1a. `operator1/models/monte_carlo.py` (~1,694 lines)

**Enhancements:** 3.10A (jump-diffusion), 3.10B (antithetic variates)

**Edit point 3.10A -- inside `simulate_return_paths()` at line 559:**
After `dist = regime_distributions[r_idx]` (line 561), add jump component. The current code generates `z = rng.normal(tilted_mean, tilted_std)`. Add Poisson jump: `jump = rng.poisson(lambda_jump * dt) * rng.normal(jump_mu, jump_sigma)`, then `z += jump`.

**Edit point 3.10B -- before the `for i in range(n_paths)` loop at line 552:**
Generate antithetic pairs: `n_half = n_paths // 2`, generate `n_half` paths, then mirror.

**New parameters for `simulate_return_paths()`:**
- `jump_lambda: float = 0.0` (jump intensity, 0 = no jumps = backward compatible)
- `jump_mean: float = 0.0`
- `jump_std: float = 0.0`
- `antithetic: bool = False`

**New parameters for `run_monte_carlo()`:**
- `jump_params: dict | None = None` (from cache `jump_spike_flag` statistics)
- `antithetic: bool = True` (default on for variance reduction)

**Called from:**
- [`operator1/stages/stage5_forward.py:163`](operator1/stages/stage5_forward.py:163) -- `run_5_4_monte_carlo()`
- [`main.py` Step 6l](main.py) -- MC simulation
- [`backtest_runner.py`](backtest_runner.py) -- Stage 2 MC

**Outlets:**
- `MonteCarloResult.survival_probability` -> profile, MF fusion, triage card
- `MonteCarloResult.terminal_values` -> conformal interval override at 21d+
- `MonteCarloResult.regime_distributions` -> HF DCF, scenario engine
- NEW: `MonteCarloResult.jump_intensity_lambda` -> profile (informational)

---

### 1b. `operator1/models/forecasting.py` (~4,610 lines)

**Enhancement:** 3.6C (regime-conditional forecasting wrapper)

**Edit point -- inside `run_forecasting()` at line 1972:**
The current cascade tries models in order. Add a `RegimeConditionalWrapper` that partitions training data by `regime_label` before calling each model. This is a wrapper AROUND the existing cascade, not a replacement.

**Implementation approach:** Add helper function `_partition_by_regime(cache, regime_col)` before `run_forecasting()`. Inside the forecasting loop, when `regime_label` is available in cache, train on regime-filtered data and blend predictions by regime probability.

**Called from:**
- [`operator1/stages/stage4_forecasting.py:22`](operator1/stages/stage4_forecasting.py:22) -- `run_4_1_forecasting()`
- [`operator1/steps/multi_frequency_runner.py`](operator1/steps/multi_frequency_runner.py) -- per-frequency forecasting
- [`main.py` Step 6h](main.py) -- primary forecasting

**Outlets:**
- `ForecastResult.forecasts` -> prediction aggregator, OHLC predictor, conformal
- `ForecastResult.metrics` -> GA optimizer, prediction aggregator weights
- `ForecastResult.residuals` -> conformal calibrator

---

### 1c. `operator1/models/conformal.py` (~1,029 lines)

**Enhancement:** 3.12A (CQR asymmetric intervals)

**Edit point -- inside `build_conformal_result()` at line 378:**
After computing standard conformal intervals, add CQR path: fit `GradientBoostingRegressor(loss="quantile", alpha=0.05)` and `alpha=0.95` on the calibration data, then conformally calibrate.

**New function:** `_cqr_intervals(calibration_X, calibration_y, forecast_X, target_coverage)` returning asymmetric `(lower, upper)` per variable per horizon.

**Called from:**
- [`operator1/stages/stage6_ensemble.py:88`](operator1/stages/stage6_ensemble.py:88) -- `run_6_3_conformal()`

**Outlets:**
- `ConformalResult.intervals` -> prediction aggregator (uncertainty bands)
- NEW: intervals are now asymmetric (wider downside during distress)

---

### 1d. `operator1/models/walk_forward.py` (~704 lines)

**Enhancement:** 3.8A (adaptive conformal retraining)

**Edit point -- inside `run_walk_forward()` at line 366:**
After the prediction loop computes per-day errors, add a coverage check: if trailing 21-day coverage < target - 5%, trigger retraining (in addition to existing switch-point-based retraining).

**New logic (after existing retrain check):**
```python
if _trailing_coverage(day_errors, window=21) < target_coverage - 0.05:
    retrain_flag = True
```

**Called from:**
- [`operator1/stages/stage5_forward.py:80`](operator1/stages/stage5_forward.py:80) -- `run_5_3_walk_forward()`

**Outlets:**
- `WalkForwardResult.retrain_dates` -> profile (now includes coverage-triggered retrains)
- `WalkForwardResult.mode_scores` -> prediction aggregator

---

### 1e. `operator1/models/prediction_aggregator.py` (~2,253 lines)

**Enhancement:** 3.20A (BOA online aggregation)

**Edit point -- inside `run_prediction_aggregation()` at line 2216:**
Replace or supplement the FixedShare aggregation with BOA (Bernstein Online Aggregation). Add `_boa_aggregate()` helper that maintains per-expert cumulative loss and adaptive learning rate.

**Called from:**
- [`operator1/stages/stage6_ensemble.py:250`](operator1/stages/stage6_ensemble.py:250) -- `run_6_5_aggregation()`

**Outlets:**
- `PredictionAggregatorResult.predictions` -> profile, report, backtest validation
- NEW: `PredictionAggregatorResult.metadata["aggregation_method"]` will include "boa" when used

---

## 2. Files That Will Be INDIRECTLY AFFECTED

### 2a. `operator1/stages/stage5_forward.py`

**Edit point at line ~163 (`run_5_4_monte_carlo`):**
Pass jump params from cache:
```python
jump_params = None
if "jump_spike_flag" in cache.columns:
    spikes = cache["jump_spike_flag"].fillna(0)
    if spikes.sum() > 0:
        jump_params = {
            "lambda": float(spikes.mean() * 252),
            "mean": float(cache["return_1d"].loc[spikes==1].mean()),
            "std": float(cache["return_1d"].loc[spikes==1].std()),
        }
```

### 2b. `operator1/stages/stage4_forecasting.py`

**Edit point at line ~22 (`run_4_1_forecasting`):**
Pass `regime_label` column to `run_forecasting()` for regime-conditional training. The column already exists in cache from Stage 3.1.

### 2c. `operator1/report/profile_builder.py`

**Update MC section** to include `jump_intensity_lambda`.
**Update predictions section** to note asymmetric intervals.

### 2d. `config/scoring_weights.yml`

No changes needed -- jump parameters are calibrated from data, not config.

---

## 3. COMPLETE INPUT/OUTPUT/OUTLET MAP

### 3.1 New Variables -> Downstream Consumers

| New Variable/Output | Producer | Direct Consumers |
|---|---|---|
| `jump_intensity_lambda` | monte_carlo.py | Profile (how jumpy is this stock?) |
| Jump-diffusion paths | monte_carlo.py | Survival probability (more realistic crisis scenarios) |
| Antithetic variance reduction | monte_carlo.py | All MC consumers (tighter confidence for same n_paths) |
| Regime-conditional forecasts | forecasting.py | Prediction aggregator (regime-blended predictions) |
| CQR asymmetric intervals | conformal.py | Prediction aggregator (wider downside in distress) |
| Coverage-triggered retrains | walk_forward.py | Profile (retrain_dates), prediction aggregator |
| BOA-weighted ensemble | prediction_aggregator.py | Profile, report |

### 3.2 Input Dependencies

| Enhancement | Required Inputs | Available? |
|---|---|---|
| 3.10A Jump-diffusion | `jump_spike_flag`, `return_1d` from Layer 1 Stage 16 | Yes (PR #1) |
| 3.10B Antithetic | None (pure variance reduction) | N/A |
| 3.6C Regime-conditional | `regime_label` from Stage 3.1 | Yes (existing) |
| 3.12A CQR | Calibration residuals from forward pass | Yes (existing) |
| 3.8A Adaptive retrain | Trailing coverage from walk-forward errors | Yes (computed in-loop) |
| 3.20A BOA | Expert predictions from forecast_result | Yes (existing) |

---

## 4. SUB-STAGE CONSIDERATIONS

Unlike Layer 2 (which runs in Stage 1), Layer 3 enhancements are IN the staged pipeline:

| Enhancement | Sub-Stage | Checkpoint | Impact |
|---|---|---|---|
| 3.10A/B Jump MC | 5.4 (`run_5_4_monte_carlo`) | `state.save("5.4")` | MC result object changes shape (backward compatible) |
| 3.6C Regime-conditional | 4.1 (`run_4_1_forecasting`) | `state.save("4.1")` | ForecastResult structure unchanged |
| 3.12A CQR | 6.3 (`run_6_3_conformal`) | `state.save("6.3")` | ConformalResult intervals become asymmetric |
| 3.8A Adaptive retrain | 5.3 (`run_5_3_walk_forward`) | `state.save("5.3")` | WalkForwardResult retrain_dates has more entries |
| 3.20A BOA | 6.5 (`run_6_5_aggregation`) | `state.save("6.5")` | PredictionAggregatorResult weights change |

All enhancements are **backward compatible** -- they add optional parameters with defaults that preserve current behavior. Existing checkpoints remain loadable.

---

## 5. RISK ANALYSIS

| Risk | File | Mitigation |
|---|---|---|
| Jump-diffusion overshoots | monte_carlo.py | Cap jump size at 3*daily_std; lambda capped at 50 (max ~1 jump/5 days) |
| Antithetic breaks importance sampling | monte_carlo.py | Only use antithetic when `importance_tilt == 0`; skip for tilted paths |
| Regime-conditional has too few points per regime | forecasting.py | Minimum 30 days per regime; fall back to full-data model |
| CQR quantile GBM doesn't converge | conformal.py | Fall back to standard symmetric conformal |
| BOA weights collapse to single expert | prediction_aggregator.py | Floor weight at 1/n_experts * 0.1 (never zero) |
| Existing checkpoint incompatibility | PipelineState | All new params have defaults; old checkpoints load fine |

---

## 6. COMPLETE FILE EDIT LIST

| # | File | Lines | Edit Type | Enhancement | New Lines |
|---|------|-------|-----------|-------------|-----------|
| 1 | `operator1/models/monte_carlo.py` | 1,694 | MODERATE | 3.10A + 3.10B | ~40 |
| 2 | `operator1/models/forecasting.py` | 4,610 | MODERATE | 3.6C | ~45 |
| 3 | `operator1/models/conformal.py` | 1,029 | MODERATE | 3.12A | ~45 |
| 4 | `operator1/models/walk_forward.py` | 704 | MINOR | 3.8A | ~20 |
| 5 | `operator1/models/prediction_aggregator.py` | 2,253 | MODERATE | 3.20A | ~45 |
| 6 | `operator1/stages/stage5_forward.py` | ~285 | MINOR | Wire jump params | ~10 |
| 7 | `operator1/stages/stage4_forecasting.py` | 54 | MINOR | Pass regime_label | ~5 |
| 8 | `operator1/report/profile_builder.py` | ~1,285 | MINOR | New MC fields | ~5 |
| **Total** | | | | | **~215 lines** |

---

## 7. IMPLEMENTATION PHASES

### Phase 1: MC Enhancements (3.10A + 3.10B) -- ~40 lines

**Step 1.1:** Add `jump_lambda`, `jump_mean`, `jump_std`, `antithetic` params to `simulate_return_paths()`
**Step 1.2:** Inside the inner loop (line 559-584), add Poisson jump after regime-conditioned draw
**Step 1.3:** Wrap path generation with antithetic logic (n_half paths + mirrored)
**Step 1.4:** Add `jump_params` to `run_monte_carlo()` signature, calibrate from cache
**Step 1.5:** Wire in `stage5_forward.py` -- pass jump params from cache columns
**Step 1.6:** Add `jump_intensity_lambda` to MonteCarloResult

### Phase 2: Regime-Conditional Forecasting (3.6C) -- ~45 lines

**Step 2.1:** Add `_partition_by_regime()` helper before `run_forecasting()`
**Step 2.2:** Inside model cascade, when `regime_label` available, train per-regime and blend
**Step 2.3:** Wire in `stage4_forecasting.py` -- pass regime_label from cache

### Phase 3: CQR Intervals (3.12A) -- ~45 lines

**Step 3.1:** Add `_cqr_intervals()` function to conformal.py
**Step 3.2:** Inside `build_conformal_result()`, try CQR first, fall back to standard
**Step 3.3:** Ensure prediction_aggregator handles asymmetric intervals

### Phase 4: Walk-Forward + BOA (3.8A + 3.20A) -- ~65 lines

**Step 4.1:** Add `_trailing_coverage()` helper to walk_forward.py
**Step 4.2:** Add coverage-based retrain trigger in the walk loop
**Step 4.3:** Add `_boa_aggregate()` to prediction_aggregator.py
**Step 4.4:** Integrate BOA as alternative/supplement to FixedShare

### Phase 5: Profile + Wiring -- ~10 lines

**Step 5.1:** Update profile_builder.py MC section with jump params
**Step 5.2:** Update stage5_forward.py and stage4_forecasting.py wiring

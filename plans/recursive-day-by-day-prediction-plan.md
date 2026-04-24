# Recursive Day-by-Day Prediction Aggregation Plan

*Created: 2026-04-24*

## Problem Statement

The current prediction pipeline produces forecasts at 4 independent horizons (1d, 5d, 21d, 252d). Each horizon is predicted separately using the same models but extrapolated further into the future. This causes accuracy to degrade rapidly at longer horizons because:

1. **Independent extrapolation**: The 21d forecast is a single 21-step-ahead jump, accumulating all model uncertainty at once.
2. **No compounding of information**: The rich 1d prediction (which benefits from the full ensemble, regime detection, conformal calibration, etc.) is not used as input for the 2d prediction, and so on.
3. **Error amplification**: Model errors at each horizon are independent -- there is no self-correction mechanism between day 1 and day 21.

## Proposed Solution: Recursive 1d Aggregation

Instead of predicting each horizon independently, we:

1. **Maximize 1d accuracy** using the full ensemble (Kalman, GARCH, VAR, LSTM, Tree, ETS, baseline) with all available signals (regime, momentum, conformal, copula, DTW, SHAP, Granger).
2. **Append the predicted day** to the cache as a synthetic row.
3. **Re-derive key features** (return_1d, volatility_21d rolling update, etc.) from the extended cache.
4. **Predict the next day** using the updated cache, recursively for N days.
5. **Extract multi-horizon forecasts** from the recursive chain: day 5 = the 5th recursive prediction, day 21 = the 21st, day 252 = the 252nd.

This approach mirrors how markets actually evolve: each day depends on the previous day, not on a single long-range extrapolation.

## Architecture

### New Module: `operator1/models/recursive_aggregator.py`

```
run_recursive_predictions(
    cache,                    # Full daily cache up to reference_date
    fitted_models,            # Dict of {variable: [BaseModelWrapper, ...]} from forward pass
    ensemble_weights,         # Per-variable per-model weights from GA/inverse-RMSE
    horizon_days=252,         # Max recursive steps
    regime_detector=None,     # For re-classifying regime on synthetic days
    conformal_calibrator=None,# For interval estimation at each step
    mc_transition_matrix=None,# For regime evolution
    momentum_blend=0.3,       # Momentum overlay weight (from P1 fix)
    snapshot_days=[1,5,21,63,252], # Which days to extract as horizon predictions
) -> RecursivePredictionResult
```

### Step-by-Step Algorithm

For each recursive day `t` in `[1, 2, ..., horizon_days]`:

**Step 1: Predict close(t)**
- Each fitted model predicts 1 step ahead from the current (possibly extended) cache.
- Weight predictions using ensemble weights (from GA optimizer or inverse-RMSE).
- Apply momentum overlay (P1 fix: 70% model + 30% momentum).
- Apply regime-conditional weight adjustment if dual_regime weights available.
- Result: `predicted_close(t)`.

**Step 2: Predict auxiliary variables**
- For each non-close variable that models were fitted on (return_1d, volatility_21d, etc.), predict 1 step ahead.
- Use the same ensemble mechanism.

**Step 3: Build synthetic cache row**
- `close(t) = predicted_close(t)`
- `return_1d(t) = (close(t) - close(t-1)) / close(t-1)`
- `log_return_1d(t) = ln(close(t) / close(t-1))`
- `volatility_21d(t)` = rolling std of last 21 return_1d values (including synthetic ones)
- `drawdown_252d(t)` = updated from rolling max
- Financial statement columns: carry forward (ffill) from last actual value
- Regime label: evolve via MC transition matrix or carry forward

**Step 4: Append to extended cache**
- The synthetic row is appended to the working cache.
- Models that support online updates (Kalman, LSTM) get the new observation.

**Step 5: Snapshot if needed**
- If `t` is in `snapshot_days`, capture the current prediction as a horizon forecast.
- Store: point forecast, uncertainty band (from recursive conformal), regime at that point.

**Step 6: Uncertainty propagation**
- Conformal interval widens at each step: `interval(t) = interval(1) * sqrt(t)` (base), but capped by the conformal PID calibrator.
- Alternative: track the empirical distribution of recursive residuals and use that for bands.

### Uncertainty Quantification

The recursive approach naturally builds uncertainty through compounding:

1. **Per-step conformal**: Each 1d prediction gets a conformal interval.
2. **Cumulative uncertainty**: For day `t`, the total uncertainty is the convolution of `t` individual 1d uncertainties.
3. **Practical approximation**: `total_width(t) = width_1d * sqrt(t) * regime_factor(t)`
4. **Regime-aware widening**: If the MC transition matrix predicts a regime change before day `t`, widen further.

### Integration with Existing Pipeline

The recursive aggregator runs **after** the existing prediction pipeline (Step 6r) as an alternative aggregation path. It does NOT replace the existing independent-horizon predictions -- it provides a second set of predictions that can be compared or blended.

**Wiring in main.py / stage6_ensemble.py:**
```
# After step 6r (prediction_aggregator)
recursive_result = run_recursive_predictions(
    cache=cache,
    fitted_models=forward_pass_result.model_states,
    ensemble_weights=ga_result.best_weights if ga_result else None,
    regime_detector=regime_detector,
    conformal_calibrator=conformal_calibrator,
    mc_transition_matrix=mc_result.transition_matrix if mc_result else None,
)
```

**Profile storage:**
```python
profile["recursive_predictions"] = recursive_result.to_dict()
```

### Key Design Decisions

1. **Feature re-derivation is lightweight**: Only update rolling features (returns, volatility, drawdown) from the synthetic close. Do NOT re-run survival mode, financial health, or other heavy computations on synthetic data.

2. **Financial statement columns freeze**: After the last actual filing, all statement-derived columns (revenue, total_assets, etc.) stay constant during recursion. These change only at filing dates, not daily.

3. **Regime evolution**: Use the HMM transition matrix to probabilistically evolve the regime label. On each synthetic day, sample the next regime from `P(regime(t+1) | regime(t))`.

4. **Model re-fitting is NOT done**: The fitted models from the forward pass are reused as-is. Only online-update models (Kalman) get new observations. This keeps recursion fast (~100ms per step, ~25s for 252 days).

5. **Drift correction**: Every `k` steps (configurable, default 5), compare the recursive trajectory against the most recent actual data trend and apply a small correction to prevent runaway drift.

### Output: RecursivePredictionResult

```python
@dataclass
class RecursiveSnapshot:
    day: int                    # Recursive step number
    date: date                  # Predicted date
    predictions: dict           # {variable: point_forecast}
    uncertainty: dict           # {variable: (lower, upper)}
    regime: str                 # Predicted regime at this day
    cumulative_confidence: float # Decaying confidence (1.0 at day 1)

@dataclass
class RecursivePredictionResult:
    snapshots: dict[str, RecursiveSnapshot]  # Keyed by horizon label ("1d", "5d", ...)
    full_trajectory: pd.DataFrame            # All 252 days of recursive predictions
    method: str = "recursive_1d"
    confidence_decay_rate: float = 0.0       # Empirical per-step decay
```

## Expected Accuracy Improvement

| Horizon | Current Method | Recursive Method | Why Better |
|---------|---------------|-----------------|------------|
| 1d | ~1.3% error | ~1.0% error (same) | No change -- already 1d optimized |
| 5d | ~2.0% error | ~1.5% error | 5 recursive 1d steps compound less error than 1 direct 5d jump |
| 21d | ~9.0% error | ~4-5% error | 21 small steps vs 1 large jump; regime transitions handled per-day |
| 252d | ~25%+ error | ~15-18% error | Still compounds, but each step is well-calibrated |

## Implementation Phases

### Phase 1: Core recursive engine (this PR)
- `recursive_aggregator.py` with `run_recursive_predictions()`
- Synthetic row builder with lightweight feature re-derivation
- Snapshot extraction at configurable days
- Basic uncertainty propagation (sqrt-t scaling)

### Phase 2: Wire into pipeline
- Add as sub-stage 6.12 in `stage6_ensemble.py`
- Store in profile as `recursive_predictions`
- Add to report section (Premium tier)

### Phase 3: Backtest validation
- Compare recursive vs independent predictions against AAPL 2025 actuals
- Measure improvement at each horizon
- Tune drift correction interval

## Files Changed

| File | Change |
|------|--------|
| `operator1/models/recursive_aggregator.py` | **NEW** -- Core recursive prediction engine |
| `operator1/stages/stage6_ensemble.py` | Add sub-stage 6.12 for recursive predictions |
| `operator1/report/profile_builder.py` | Add `recursive_predictions` profile key |
| `plans/recursive-day-by-day-prediction-plan.md` | **NEW** -- This plan |

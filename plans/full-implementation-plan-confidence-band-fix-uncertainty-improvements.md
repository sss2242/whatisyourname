# Full Implementation Plan: Confidence Band Fix + Uncertainty Improvements

## Scan Results Summary

Debug scan mapped **35 interval construction points**, **30 conformal calibrator references**, **2 IV30 injection points**, **5 anchoring_52w_high consumers**, and **0 vol_of_vol consumers** (feature exists but not used for interval construction). The edit surface is concentrated in 2 files.

---

## Files to Edit

| File | Lines | Edit Points | Description |
|------|-------|-------------|-------------|
| [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py) | 2,945 | 5 edits | Main interval construction + ensemble disagreement + IV anchoring + floor + ATH scaling |
| [`operator1/models/conformal.py`](operator1/models/conformal.py) | 1,029 | 1 edit | PID anti-windup clamp |

**No other files need editing.** All inputs (iv30, anchoring_52w_high, vol_of_vol_21d, volatility_21d, model forecasts) are already in the cache or available as function parameters.

---

## Edit 1: Always Re-Center Intervals (Bug Fix -- CRITICAL)

**File:** [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)
**Location:** Lines 2676-2689 (conformal path) and line 2693 (RMSE fallback path)

**Problem:** Conformal intervals are centered on the raw forward-pass forecast. After ensemble blending shifts the point forecast by $10+, the bounds are NOT re-centered. Result: lower_bound > point_forecast.

**Current code flow (lines 2650-2700):**
```
1. Compute ensemble point forecast (point = $239.30)
2. Get conformal bounds (lower=$243.74, upper=$244.63) -- centered on raw forecast
3. IF conformal: re-center on ensemble point -- WORKS
4. IF survival_adjusted: re-center again -- overwrites step 3 correctly
5. IF RMSE fallback: bounds from compute_uncertainty_bands(point, rmse, h) -- WORKS
```

The bug is that Step 3 fires correctly BUT the conformal bounds are produced from a calibrator trained on 1-step residuals producing a $0.45 half-width. This is too narrow. The re-centering is correct -- the problem is the half-width itself.

**Fix:** After computing the half-width from ANY source (conformal or RMSE), enforce a minimum floor and blend with ensemble disagreement:

```python
# AFTER line 2689 (end of conformal path) and AFTER line 2700 (end of RMSE fallback):

# --- FIX 1: Minimum half-width floor ---
# A $0.45 band on a $250 stock is 0.18% -- absurdly narrow for 90% coverage.
# Floor: 1% of price * sqrt(horizon) ensures meaningful intervals.
_MIN_PCT = 0.01
_min_hw = abs(point) * _MIN_PCT * math.sqrt(max(horizon_days, 1))
current_hw = (upper - lower) / 2.0
if current_hw < _min_hw:
    lower = point - _min_hw
    upper = point + _min_hw
    interval_source += "+floor"
```

---

## Edit 2: Ensemble Disagreement Width (Lakshminarayanan 2017)

**File:** [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)
**Location:** New function + integration at line ~2705 (after interval construction, before copula widening)

**New function (add around line 930):**

```python
def compute_ensemble_disagreement_width(
    forecasts: dict[str, dict[str, float]],
    weights: dict[str, float],
    variable: str,
    horizon: str,
    point_forecast: float,
) -> float:
    """Epistemic uncertainty from model disagreement (Lakshminarayanan 2017).
    
    When models agree, returns 0. When models disagree, returns the
    weighted standard deviation of their forecasts -- a measure of
    how uncertain the ENSEMBLE is about the point forecast.
    """
    model_preds = []
    model_weights = []
    for model_name, w in weights.items():
        if w <= 0:
            continue
        # Get this model's forecast for this variable at this horizon
        var_forecasts = forecasts.get(variable, {})
        if isinstance(var_forecasts, dict) and horizon in var_forecasts:
            pred = var_forecasts[horizon]
            if pred is not None and not math.isnan(pred):
                model_preds.append(pred)
                model_weights.append(w)
    
    if len(model_preds) < 2:
        return 0.0
    
    preds = np.array(model_preds)
    wts = np.array(model_weights)
    wts = wts / wts.sum()  # normalize
    
    weighted_var = np.sum(wts * (preds - point_forecast) ** 2)
    return float(np.sqrt(weighted_var))
```

**Integration (after interval floor, before copula widening at line ~2709):**

```python
# --- FIX 2: Ensemble disagreement widening ---
# When models disagree by more than the interval width,
# the disagreement IS the uncertainty signal.
_disagree_width = compute_ensemble_disagreement_width(
    forecasts=aggregated,  # per-model forecasts dict
    weights=base_weights,
    variable=var_name,
    horizon=h_label,
    point_forecast=point,
)
if _disagree_width > 0:
    current_hw = (upper - lower) / 2.0
    # Blend: max of current and disagreement (take the wider)
    blended_hw = max(current_hw, _disagree_width * Z_SCORE_90)
    if blended_hw > current_hw:
        lower = point - blended_hw
        upper = point + blended_hw
        interval_source += "+disagreement"
```

**Input:** `aggregated` dict (already available at line 2812), `base_weights` dict (already computed at line 2429)

**Note:** The `aggregated` dict currently maps `{variable: {horizon: point_forecast}}`. To get per-model forecasts, we need to also pass `forecast_result.forecasts` which maps `{variable: {horizon: forecast_value}}`. This is the per-model cascade winner, not per-model individual. For true disagreement, we need the individual model forecasts BEFORE the cascade selects a winner.

**Alternative (simpler):** Use `forecast_result.metrics` to get per-model RMSE spread as a proxy for disagreement:

```python
_model_rmses = [m.rmse for m in forecast_result.metrics if m.variable == var_name and m.fitted and m.rmse > 0]
if len(_model_rmses) >= 2:
    _rmse_spread = max(_model_rmses) - min(_model_rmses)
    _disagree_width = _rmse_spread * math.sqrt(max(horizon_days, 1))
```

---

## Edit 3: IV-Anchored Interval Blending (Hull 2018)

**File:** [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)
**Location:** After Edit 2, before copula widening (line ~2710)

**Input:** `cache["iv30"]` -- already injected at [`main.py:1052`](main.py:1052) and [`backtest_runner.py:327`](backtest_runner.py:327)

```python
# --- FIX 3: IV-anchored interval blending ---
# Options-implied volatility is the market's forward-looking
# consensus on uncertainty. Blend with model-derived intervals
# when IV data is available.
if "iv30" in cache.columns and var_name == "close":
    _iv_series = cache["iv30"].dropna()
    if len(_iv_series) > 0:
        _iv30_val = float(_iv_series.iloc[-1])
        if _iv30_val > 0 and not math.isnan(_iv30_val):
            _iv_daily = _iv30_val / math.sqrt(252)
            _last_px = last_value if last_value and not math.isnan(last_value) else point
            _iv_hw = z_score * _last_px * _iv_daily * math.sqrt(max(horizon_days, 1))
            
            current_hw = (upper - lower) / 2.0
            # Blend: 60% IV (forward-looking), 40% model (backward-looking)
            _blended_hw = 0.60 * _iv_hw + 0.40 * current_hw
            if _blended_hw > current_hw:
                lower = point - _blended_hw
                upper = point + _blended_hw
                interval_source += "+iv"
```

---

## Edit 4: ATH Volatility Scaling (Bouchaud 2002)

**File:** [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)
**Location:** After Edit 3, before copula widening (line ~2715)

**Inputs:** `cache["anchoring_52w_high"]`, `cache["vol_of_vol_21d"]`, `cache["volatility_21d"]` -- all already in cache from derived_variables

```python
# --- FIX 4: ATH volatility scaling ---
# Near all-time highs, realized vol underestimates future vol
# because the trailing window only saw the uptrend. Scale by
# vol-of-vol ratio (how unstable recent volatility has been).
if "anchoring_52w_high" in cache.columns:
    _ath_s = cache["anchoring_52w_high"].dropna()
    if len(_ath_s) > 0:
        _ath_v = float(_ath_s.iloc[-1])
        if _ath_v > 0.90:
            _vov = 0.0
            _vol = 0.02  # default
            if "vol_of_vol_21d" in cache.columns:
                _vov_s = cache["vol_of_vol_21d"].dropna()
                if len(_vov_s) > 0:
                    _vov = float(_vov_s.iloc[-1])
            if "volatility_21d" in cache.columns:
                _vol_s = cache["volatility_21d"].dropna()
                if len(_vol_s) > 0:
                    _vol = max(0.001, float(_vol_s.iloc[-1]))
            
            _vov_ratio = max(1.0, _vov / _vol) if _vov > 0 else 1.0
            _ath_scale = 1.0 + (_ath_v - 0.90) * _vov_ratio
            current_hw = (upper - lower) / 2.0
            lower = point - current_hw * _ath_scale
            upper = point + current_hw * _ath_scale
            interval_source += "+ath_scale"
```

---

## Edit 5: PID Anti-Windup Clamp

**File:** [`operator1/models/conformal.py`](operator1/models/conformal.py)
**Location:** Inside `ConformalPIDCalibrator` class, in the PID update method

**Find the integral accumulation line** (inside the `update` or `_pid_update` method):

```python
# Current (no clamp):
self._integral += error

# Fixed (with anti-windup):
self._integral += error
# Anti-windup: clamp integral to prevent stuck-narrow bands
# (from simple-pid library, 901 stars)
_MAX_INTEGRAL = 5.0 * max(self._target_half_width, 0.01)
self._integral = max(-_MAX_INTEGRAL, min(_MAX_INTEGRAL, self._integral))
```

---

## Edit 6: Final Invariant Guard

**File:** [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)
**Location:** Just before constructing the `HorizonPrediction` object (line ~2800)

```python
# --- INVARIANT: lower <= point <= upper ---
# This must ALWAYS hold regardless of which interval path fired.
if not math.isnan(point) and not math.isnan(lower) and not math.isnan(upper):
    if lower > point:
        hw = (upper - lower) / 2.0
        lower = point - hw
        upper = point + hw
    if upper < point:
        hw = (upper - lower) / 2.0
        lower = point - hw
        upper = point + hw
```

---

## Execution Order

```
1. Edit 5 (conformal.py PID anti-windup) -- prevents the calibrator from getting stuck
2. Edit 1 (aggregator minimum floor) -- prevents absurdly narrow bands
3. Edit 6 (aggregator invariant guard) -- guarantees lower <= point <= upper
4. Edit 2 (aggregator ensemble disagreement) -- captures model uncertainty
5. Edit 3 (aggregator IV blending) -- forward-looking market consensus
6. Edit 4 (aggregator ATH scaling) -- near-extreme position widening
```

Edits 1-3-6 fix the bug. Edits 2-4-5 improve quality. Edit order matters: floor and invariant must be last in the chain so they catch any upstream issues.

---

## Checklist

- [ ] Edit 5: PID anti-windup in conformal.py
- [ ] Edit 1: Minimum half-width floor in prediction_aggregator.py
- [ ] Edit 6: Invariant guard (lower <= point <= upper) in prediction_aggregator.py
- [ ] Edit 2: Ensemble disagreement width in prediction_aggregator.py
- [ ] Edit 3: IV-anchored blending in prediction_aggregator.py
- [ ] Edit 4: ATH volatility scaling in prediction_aggregator.py
- [ ] Verify: `python -c "from operator1.models.prediction_aggregator import run_prediction_aggregation; print('OK')"`
- [ ] Verify: `python -c "from operator1.models.conformal import ConformalPIDCalibrator; print('OK')"`
- [ ] Create branch, commit, push, open PR #3

## Consumer Impact

**Zero API changes.** All edits modify internal interval computation only. The `HorizonPrediction` dataclass output is unchanged (`point_forecast`, `lower_ci`, `upper_ci`, `confidence`). All downstream consumers (profile_builder, report_generator, dashboard, backtest_runner validation) read the same fields.

## Expected Results on AAPL Backtest

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| 1d half-width | $0.45 (0.18%) | ~$3.50 (1.4%) via floor + IV |
| 1d lower_bound | $243.74 (above point!) | ~$235.80 (below point) |
| 1d upper_bound | $244.63 | ~$243.10 |
| 1d actual_in_bounds | false | likely true ($242.53 in $235-243 range) |
| 21d half-width | ~$2.00 | ~$15-20 (IV-scaled + ATH) |
| lower <= point <= upper | VIOLATED | GUARANTEED |

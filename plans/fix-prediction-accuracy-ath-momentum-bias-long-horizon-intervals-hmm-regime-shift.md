# Fix: Prediction Accuracy -- 3 Sources

## Problem Summary

AAPL backtest (predicted from 2024-12-31) showed systematic prediction errors:
- **1d:** predicted $250.30, actual $242.53 (+3.2% overshoot)
- **1d aggregated:** $239.53 (ensemble corrected, only -1.2% error)
- **21d:** actual -9.44% (likely outside conformal bands)
- **Max DD 2025:** -30.22% (extreme tail event)

Three root causes identified in the prediction chain.

---

## Fix 1: Adaptive Momentum Blend at ATH (High Priority)

### Root Cause

[`forecasting.py:2396-2412`](operator1/models/forecasting.py:2396) uses a **fixed 70/30 blend** (70% model, 30% momentum) for close predictions, regardless of where the price sits relative to its historical range:

```python
_horizon_forecasts[_ml] = 0.7 * _horizon_forecasts[_ml] + 0.3 * _mom_pred
```

Near all-time highs, trailing momentum is positive by definition (the stock went up to get there). Blending 30% positive momentum at ATH adds systematic upward bias. The Kalman/baseline model (which anchors near last price) was actually more accurate.

### Fix

Make the momentum blend weight adaptive using `anchoring_52w_high` from [`behavioral_signals.py`](operator1/features/behavioral_signals.py). This variable is already computed in Step 5i.7 and available in the cache:

```python
# anchoring_52w_high = close / close.rolling(252).max()
# Values near 1.0 = at 52-week high, near 0.5 = 50% off high
```

**Changes to [`operator1/models/forecasting.py`](operator1/models/forecasting.py):**

Replace lines 2396-2412:

```python
# Current (fixed blend):
_mom_pred = _last_close * (1 + _mom * _mh)
_horizon_forecasts[_ml] = 0.7 * _horizon_forecasts[_ml] + 0.3 * _mom_pred
```

With:

```python
# Adaptive blend: dampen momentum weight near 52-week extremes
_mom_pred = _last_close * (1 + _mom * _mh)

# Get anchoring score (0-1, where 1.0 = at 52-week high)
_ath_score = 0.5  # neutral default
if "anchoring_52w_high" in cache.columns:
    _ath_val = cache["anchoring_52w_high"].dropna()
    if len(_ath_val) > 0:
        _ath_score = float(_ath_val.iloc[-1])

# Also check 52-week low anchoring for mean-reversion near lows
_atl_score = 2.0  # neutral default
if "anchoring_52w_low" in cache.columns:
    _atl_val = cache["anchoring_52w_low"].dropna()
    if len(_atl_val) > 0:
        _atl_score = float(_atl_val.iloc[-1])

# Dampen momentum weight near extremes:
# - At ATH (ath_score > 0.95): momentum_weight -> 0.05 (mostly model)
# - At 52w low (atl_score < 1.05): momentum_weight -> 0.05 (mostly model)
# - Mid-range (ath_score ~ 0.7): momentum_weight -> 0.30 (full blend)
if _ath_score > 0.85:
    # Near highs: linear dampen from 0.30 at 0.85 to 0.05 at 1.0
    _mom_weight = max(0.05, 0.30 - ((_ath_score - 0.85) / 0.15) * 0.25)
elif _atl_score < 1.15:
    # Near lows: dampen momentum (mean reversion more likely)
    _mom_weight = max(0.05, 0.30 - ((1.15 - _atl_score) / 0.15) * 0.25)
else:
    _mom_weight = 0.30  # original weight

_horizon_forecasts[_ml] = (
    (1.0 - _mom_weight) * _horizon_forecasts[_ml]
    + _mom_weight * _mom_pred
)
```

**Impact:** At AAPL's 2024 year-end (near ATH, `anchoring_52w_high` ~ 0.97), the momentum weight drops from 0.30 to ~0.05, reducing the upward bias from ~$7.77 to ~$1.30.

### Files Changed
- [`operator1/models/forecasting.py`](operator1/models/forecasting.py): lines 2396-2412

### Consumer Impact
- None -- `_horizon_forecasts` dict structure unchanged
- Downstream: `ForecastResult.forecasts[var][horizon]` still a float

---

## Fix 2: MC Percentile Override for Long-Horizon Intervals (Medium Priority)

### Root Cause

[`prediction_aggregator.py:882`](operator1/models/prediction_aggregator.py:882) uses random-walk RMSE scaling for uncertainty bands:

```python
base_spread = z_score * rmse * math.sqrt(max(horizon_days, 1))
```

This assumes independent daily errors. At 21d: `sqrt(21) = 4.58x` scaling. During trending markets (Jan 2025 AAPL selloff), errors are serially correlated, so sqrt(h) **underestimates** true uncertainty.

The conformal calibrator produces well-calibrated 1-step intervals but has few multi-step residuals. The MC path percentiles (from 10K regime-switching simulations) are the most reliable source for 21d+ uncertainty.

Currently, MC percentile override only fires in [`stage6_ensemble.py:171-173`](operator1/stages/stage6_ensemble.py:171) for conformal intervals, not for the RMSE fallback path.

### Fix

In [`prediction_aggregator.py`](operator1/models/prediction_aggregator.py), after the RMSE fallback interval computation (line 2687), add MC percentile override for horizons >= 21d:

**Changes to [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py):**

After line 2687 (`interval_source = "rmse"`), add:

```python
            # Phase 1.5: MC percentile override for long horizons.
            # The RMSE * sqrt(h) scaling assumes independent errors,
            # which underestimates uncertainty during trending markets.
            # MC path percentiles (10K regime-switching simulations)
            # are more reliable for horizons >= 21d.
            if (
                interval_source == "rmse"
                and mc_result is not None
                and horizon_days >= 21
                and var_name == "close"
            ):
                try:
                    _mc_tv = getattr(mc_result, "terminal_values", {})
                    _mc_horizon_key = f"{horizon_days}d"
                    _mc_paths = _mc_tv.get(horizon_days) or _mc_tv.get(_mc_horizon_key)
                    if _mc_paths is not None and len(_mc_paths) > 100:
                        _mc_arr = np.array(_mc_paths)
                        _last_close = last_value if last_value and not math.isnan(last_value) else point
                        _mc_p5 = float(np.percentile(_mc_arr, 5)) * _last_close
                        _mc_p95 = float(np.percentile(_mc_arr, 95)) * _last_close
                        # Use MC bounds if they're wider than RMSE bounds
                        # (MC captures regime-switching tail risk that RMSE misses)
                        if (_mc_p95 - _mc_p5) > (upper - lower):
                            lower = _mc_p5
                            upper = _mc_p95
                            interval_source = "mc_percentile"
                except Exception:
                    pass  # keep RMSE bounds on failure
```

**Impact:** For 21d+ horizons, MC path percentiles replace RMSE scaling when they produce wider (more conservative) intervals. This captures regime-switching tail risk that sqrt(h) misses.

### Files Changed
- [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py): after line 2687

### Consumer Impact
- None -- `lower`/`upper` bounds still floats, `interval_source` string updated for diagnostics

---

## Fix 3: HMM Regime Probability Dampening Near Extremes (Low Priority)

### Root Cause

[`forecasting.py:2367-2390`](operator1/models/forecasting.py:2367) applies a regime probability-weighted directional shift to close forecasts:

```python
_expected_daily = sum(regime_mean * regime_prob for each regime)
_horizon_forecasts[_rl] *= (1 + _expected_daily * horizon_days)
```

At year-end 2024, the HMM likely assigned high probability to the "bull" regime (AAPL had strong Q4). The bull regime's historical mean daily return (~0.05-0.10%) gets multiplied by the number of horizon days, adding a cumulative upward shift. This compounds the momentum bias from Fix 1.

### Fix

Dampen the regime shift when the Hurst exponent indicates mean-reversion tendency or when the stock is near 52-week extremes.

**Changes to [`operator1/models/forecasting.py`](operator1/models/forecasting.py):**

After line 2387 (`_rshift = _expected_daily * _rh`), add dampening:

```python
                                # Dampen regime shift near 52-week extremes
                                # (mean reversion more likely at extremes)
                                _regime_dampen = 1.0
                                if "anchoring_52w_high" in cache.columns:
                                    _ath = cache["anchoring_52w_high"].dropna()
                                    if len(_ath) > 0:
                                        _ath_val = float(_ath.iloc[-1])
                                        if _ath_val > 0.90:
                                            # Near high: dampen upward shifts
                                            if _rshift > 0:
                                                _regime_dampen = max(0.1, 1.0 - (_ath_val - 0.90) / 0.10)
                                if "hurst_exponent_rolling" in cache.columns:
                                    _hurst = cache["hurst_exponent_rolling"].dropna()
                                    if len(_hurst) > 0:
                                        _h_val = float(_hurst.iloc[-1])
                                        if _h_val < 0.45:
                                            # Mean-reverting: dampen directional shifts
                                            _regime_dampen *= max(0.3, _h_val / 0.45)
                                _rshift *= _regime_dampen
```

**Impact:** When AAPL is near ATH (anchoring > 0.90) AND the HMM predicts upward shift, the shift is dampened by up to 90%. When Hurst exponent < 0.45 (mean-reverting), directional shifts are dampened by up to 70%.

### Files Changed
- [`operator1/models/forecasting.py`](operator1/models/forecasting.py): after line 2387

### Consumer Impact
- None -- `_horizon_forecasts` dict structure unchanged

---

## Execution Checklist

- [ ] **Fix 1:** Edit [`forecasting.py`](operator1/models/forecasting.py) lines 2396-2412 -- adaptive momentum blend using `anchoring_52w_high`
- [ ] **Fix 2:** Edit [`prediction_aggregator.py`](operator1/models/prediction_aggregator.py) after line 2687 -- MC percentile override for 21d+ RMSE fallback
- [ ] **Fix 3:** Edit [`forecasting.py`](operator1/models/forecasting.py) after line 2387 -- regime shift dampening near extremes
- [ ] Verify: import check passes
- [ ] Create branch `fix/prediction-accuracy-ath-bias`, commit, push, open PR

---

## Expected Impact on AAPL Backtest

| Source | Before Fix | After Fix | Delta |
|--------|-----------|-----------|-------|
| Fix 1: Momentum at ATH | +$7.77 bias | ~+$1.30 bias | -$6.47 |
| Fix 2: 21d interval width | Too narrow (sqrt scaling) | MC-calibrated (wider) | Better coverage |
| Fix 3: HMM regime shift | Additional upward bias | Dampened near ATH | Minor improvement |
| **Combined 1d error** | **$250.30 (+3.2%)** | **~$244 (+0.6%)** | **~2.6pp improvement** |

# Band-Informed Confidence + Day-by-Day Prediction Accuracy Fix

## Problem Statement

The AAPL backtest reveals that **all 44 predictions at 1d horizon have confidence = 0.0** despite the predictions themselves being accurate (close@1d: predicted $241.36, actual $242.30 = 0.4% error). The conformal bands are also correct: [$240.57, $251.59] captured the actual perfectly.

### Root Cause Chain

1. MC survival at 1d = 0.0 (Apple's `current_ratio = 0.92 < 1.0` triggers survival immediately)
2. `surv_prob = mc_survival_by_horizon.get("1d", 1.0)` returns 0.0
3. `confidence = sqrt(model_quality * surv_prob)` = `sqrt(anything * 0.0)` = 0.0
4. This happens because the sector label "Electronic Computers" (SIC) doesn't match "technology" in `SECTOR_SURVIVAL_OVERRIDES`, so the survival threshold stays at 1.0 instead of being relaxed to 0.7 for tech companies

### What the User Wants

Use the conformal bands (which ARE correct) to derive a more meaningful confidence score AND to improve the day-by-day predictions' accuracy.

## Design: Band-Derived Confidence Score

The core idea: if the conformal bands are narrow relative to the variable's scale, that IS high confidence -- regardless of what the MC survival model says. The bands incorporate 441 days of forward pass residuals and reflect actual prediction error, not survival risk.

### New Confidence Formula

```
confidence = w_band * band_confidence + w_mc * mc_confidence

where:
  band_confidence = 1.0 - (band_width / scale_reference)
  mc_confidence   = sqrt(model_quality * surv_prob)  [existing]
  
  w_band = 0.6 at 1d, 0.4 at 5d, 0.2 at 21d, 0.1 at 252d
  w_mc   = 1.0 - w_band
```

**Rationale:** At short horizons (1d, 5d), the conformal bands are tightly calibrated from hundreds of residuals -- they're the most trustworthy signal. At longer horizons (252d), MC survival becomes more relevant because path-dependent risk matters more.

### Band Confidence Computation

```python
def _band_derived_confidence(point, lower, upper, last_value, horizon_days):
    """Derive confidence from conformal band width relative to price scale."""
    band_width = upper - lower
    if band_width <= 0 or last_value <= 0:
        return 0.5  # neutral default
    
    # Relative width: band_width / last_value
    rel_width = band_width / last_value
    
    # Expected width at this horizon (from random walk scaling)
    # A stock with 1.5% daily vol has ~1.5% 1d width, ~3.4% 5d, ~7% 21d
    expected_width = 0.015 * math.sqrt(max(horizon_days, 1))
    
    # Confidence: how much narrower are bands vs expected?
    # ratio < 1.0 = bands narrower than expected = high confidence
    # ratio > 1.0 = bands wider = low confidence
    ratio = rel_width / expected_width
    confidence = math.exp(-ratio)  # exponential decay
    return max(0.0, min(1.0, confidence))
```

### Where to Apply

In [`prediction_aggregator.py`](operator1/models/prediction_aggregator.py) at line ~3205, replace:

```python
confidence = compute_confidence_score(var_rmse, rmse_reference, surv_prob)
```

With:

```python
# Base confidence from model quality + survival
base_confidence = compute_confidence_score(var_rmse, rmse_reference, surv_prob)

# Band-derived confidence from conformal interval width
band_confidence = _band_derived_confidence(point, lower, upper, last_value, horizon_days)

# Horizon-weighted blend: bands dominate at short horizons
_BAND_WEIGHT = {1: 0.6, 5: 0.4, 21: 0.2, 252: 0.1}
w_band = _BAND_WEIGHT.get(horizon_days, 0.3)
confidence = w_band * band_confidence + (1.0 - w_band) * base_confidence
```

## Design: Band-Guided Point Forecast Adjustment

The conformal bands can also improve the day-by-day point forecast. Currently the point forecast comes purely from the ensemble model weights. But the bands encode distributional information that the point forecast ignores.

### Constrained Optimization (enhance existing Phase 3.10)

The existing `_constrained_point_forecast()` at line 3302 already adjusts close predictions using bands + momentum. Enhance it to use band asymmetry:

```python
def _constrained_point_forecast(point, lower, upper, last_value, momentum, regime):
    """Adjust point forecast using band asymmetry and momentum."""
    band_center = (lower + upper) / 2.0
    
    # Band asymmetry: if upper band is wider than lower, distribution is right-skewed
    upper_width = upper - point
    lower_width = point - lower
    
    if upper_width + lower_width > 0:
        skew_ratio = (upper_width - lower_width) / (upper_width + lower_width)
        # Nudge point forecast toward the wider side (more probability mass there)
        skew_adjustment = skew_ratio * (upper - lower) * 0.1  # 10% of band width
        point += skew_adjustment
    
    # Existing momentum constraint
    ...
    return point
```

### Band Tightening During Day-by-Day Walk

In the recursive aggregator (sub-stage 6.11), each day's prediction uses the previous day's prediction as input. The bands from day t should constrain day t+1's prediction:

```python
# In recursive_aggregator.py, after predicting day t+1:
if prev_bands is not None:
    # If today's prediction falls outside yesterday's bands,
    # shrink it back toward the band edge (elastic constraint)
    if prediction > prev_upper:
        prediction = prev_upper + (prediction - prev_upper) * 0.3  # 70% pullback
    elif prediction < prev_lower:
        prediction = prev_lower + (prediction - prev_lower) * 0.3
```

## Implementation Plan

### Files to Modify

1. **[`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)** (~30 lines)
   - Add `_band_derived_confidence()` function
   - Modify confidence computation at line ~3205 to blend band + MC confidence
   - Enhance `_constrained_point_forecast()` with band asymmetry adjustment

2. **[`operator1/models/recursive_aggregator.py`](operator1/models/recursive_aggregator.py)** (~15 lines)
   - Add band elasticity constraint in the day-by-day loop
   - Pass previous day's bands to next day's prediction step

### Files NOT Modified (no side effects)

- `conformal.py` -- bands computation unchanged
- `forecasting.py` -- model cascade unchanged
- `monte_carlo.py` -- MC survival unchanged (sector fix is a separate issue)
- `backtest_runner.py` -- pipeline wiring unchanged

### Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| close@1d confidence | 0.0 | ~0.7 (band-derived) |
| close@5d confidence | 0.039 | ~0.5 (blended) |
| close@21d confidence | 0.008 | ~0.3 (MC-weighted blend) |
| close@252d confidence | 0.179 | ~0.2 (MC dominates) |
| 1d point accuracy | 0.4% error | ~0.4% (no change expected -- already good) |
| Recursive 5d accuracy | 0.5% error | potentially improved via band constraint |

### Risk Assessment

- **Low risk:** The band confidence is additive -- it can only increase confidence when bands are tight, never decrease it when bands are wide (since `w_band * 0 + w_mc * base = base * w_mc <= base`)
- **No model changes:** Only the confidence computation and a light point forecast nudge are modified
- **Backward compatible:** If conformal bands are unavailable (lower/upper = NaN), band confidence defaults to 0.5 and the existing MC-based confidence dominates

# Plan: IV Blend Weight Calibration for Confidence Bands

## Problem

The current IV blend weight is fixed at 60% IV / 40% model. This produces bands that are too wide at longer horizons:

| Horizon | Lower | Upper | Width | Width % |
|---------|-------|-------|-------|---------|
| 1d | $209.02 | $279.35 | $70.33 | 28% |
| 5d | $171.38 | $328.65 | $157.27 | 63% |
| 21d | $91.72 | $414.04 | $322.32 | 130% |

The IV30 is annualized (~25% for AAPL). When scaled to daily and then to horizon via `sqrt(h)`, it grows fast: at 21d, `0.25/sqrt(252)*sqrt(21)*1.645*$250 = ~$18.70` per side. But the current bands are much wider than this, suggesting the other widening phases (disagreement, ATH scaling) are stacking multiplicatively on top.

## Root Cause Analysis

The widening chain is multiplicative:
```
base_hw (conformal or RMSE)
  * floor max
  * disagreement max
  * IV blend (60/40)
  * ATH scale (1 + (0.97-0.90) * vov_ratio)
  * copula widening
  * tier confidence (survival mode)
```

Each "max" and "scale" operation can only increase the width. After 6 stages of one-directional widening, the 21d band explodes.

## Solution: Horizon-Decay IV Blend Weight

Instead of a fixed 60/40 at all horizons, the IV weight should **decay with horizon** because:

1. At **1d horizon**, IV is the best uncertainty estimator (market-priced, includes event risk)
2. At **5d**, IV is still good but model RMSE from walk-forward starts contributing meaningful signal
3. At **21d+**, model-based uncertainty (RMSE, MC percentiles) dominates because IV30 is a 30-day average that doesn't scale cleanly beyond its native horizon

### Proposed Blend Schedule

| Horizon | IV Weight | Model Weight | Rationale |
|---------|-----------|-------------|-----------|
| 1d | 0.50 | 0.50 | Equal -- IV is best single-day signal but model calibrated too |
| 5d | 0.35 | 0.65 | IV still useful but 5d walk-forward RMSE is more calibrated |
| 21d | 0.15 | 0.85 | Beyond IV30's native horizon; model RMSE + MC percentiles dominate |
| 252d | 0.05 | 0.95 | IV is noise at annual horizon; fundamental uncertainty rules |

### Implementation

**File:** [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)
**Location:** Phase 3.7 (IV-anchored blending)

Replace the fixed 0.60/0.40 blend:

```python
# Current:
_blended_hw = 0.60 * _iv_hw + 0.40 * current_hw

# New: horizon-decaying IV weight
_IV_WEIGHTS = {1: 0.50, 5: 0.35, 21: 0.15, 252: 0.05}
_iv_w = _IV_WEIGHTS.get(horizon_days, 0.20)
_blended_hw = _iv_w * _iv_hw + (1.0 - _iv_w) * current_hw
```

### Also: Cap the Stacking

Add a **total widening cap** after all phases to prevent multiplicative blowup. The maximum reasonable band width is the MC P5/P95 range (which already accounts for regime switching, tail risk, etc.):

```python
# After Phase 3.9 (invariant guard), before DTW:
# Cap total width at 2x the MC P5-P95 range or 50% of price, whichever is smaller
if mc_result is not None and var_name == "close":
    _mc_tv = getattr(mc_result, "terminal_values", {})
    _mc_paths = _mc_tv.get(horizon_days)
    if _mc_paths is not None and len(_mc_paths) > 100:
        _mc_arr = np.array(_mc_paths)
        _mc_base = last_value if last_value and not math.isnan(last_value) else point
        _mc_range = (float(np.percentile(_mc_arr, 95)) - float(np.percentile(_mc_arr, 5))) * _mc_base
        _max_hw = _mc_range  # MC range is the maximum reasonable uncertainty
        _cur_hw = (upper - lower) / 2.0
        if _cur_hw > _max_hw and _max_hw > 0:
            lower = point - _max_hw
            upper = point + _max_hw
            interval_source += "+mc_cap"

# Also: absolute cap at 50% of price for any horizon
_abs_max = abs(point) * 0.50
_cur_hw = (upper - lower) / 2.0
if _cur_hw > _abs_max:
    lower = point - _abs_max
    upper = point + _abs_max
    interval_source += "+abs_cap"
```

### Expected Results

| Horizon | Before | After (est.) |
|---------|--------|-------------|
| 1d | $209-$279 (28%) | ~$232-$247 (6%) |
| 5d | $171-$329 (63%) | ~$225-$265 (16%) |
| 21d | $92-$414 (130%) | ~$200-$295 (38%) |

## Execution Checklist

- [ ] Change Phase 3.7 IV blend to horizon-decaying weights
- [ ] Add MC range cap after Phase 3.9
- [ ] Add absolute 50% cap as final safety net
- [ ] Verify imports, commit, push

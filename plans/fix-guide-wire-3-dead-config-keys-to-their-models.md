# Fix Guide: Wire 3 Dead Config Keys to Their Models

*Generated from debug scan on 2026-05-08*

## Summary

The debug scan of 278 Python files found 3 config keys in `config/scoring_weights.yml` that are never consumed by the models they configure. The models use hardcoded values instead. Wiring these keys makes the parameters tweakable from the dashboard Scoring Weights panel without code changes.

**Severity:** Low -- models work correctly with hardcoded values. This is a configurability improvement, not a bug fix.

---

## Issue 1: `dcf_calibration` -- HF DCF Model Uses Inline WACC/Growth Params

**Config key:** `dcf_calibration` in `config/scoring_weights.yml`
**Model file:** `operator1/hedge_fund/engine.py` (inside `_compute_dcf()`)

### Current State

The DCF Monte Carlo valuation in the HF engine samples WACC from `N(0.09, 0.01)` and terminal growth from a fixed range. These are hardcoded inline:

```python
# Inside _compute_dcf():
wacc = np.random.normal(0.09, 0.01)  # hardcoded
terminal_growth = ...                  # inline calculation
```

The `dcf_calibration` key in `scoring_weights.yml` has configurable values but `engine.py` never imports `get_weight()`.

### Fix

| Step | Action | File | Lines |
|------|--------|------|-------|
| 1 | Import `get_weight` at top of file | `operator1/hedge_fund/engine.py` | ~1 line |
| 2 | Replace hardcoded WACC mean/std with `get_weight('dcf_calibration.wacc_mean', 0.09)` and `get_weight('dcf_calibration.wacc_std', 0.01)` | `operator1/hedge_fund/engine.py` inside `_compute_dcf()` | ~4 lines |
| 3 | Replace hardcoded terminal growth bounds with `get_weight('dcf_calibration.terminal_growth_min/max', ...)` | Same function | ~2 lines |
| 4 | Add `wt_dcf` tab to dashboard weights panel | `dashboard.py` inside `render_scoring_weights()` | ~15 lines |

### Config Structure (already in yml)

```yaml
dcf_calibration:
  wacc_mean: 0.09
  wacc_std: 0.01
  terminal_growth_min: 0.02
  terminal_growth_max: 0.04
  n_simulations: 10000
  explicit_years: 5
```

---

## Issue 2: `geographic_metrics` -- Product Metrics Uses Hardcoded Thresholds

**Config key:** `geographic_metrics` in `config/scoring_weights.yml`
**Model file:** `operator1/features/product_metrics.py` (inside `compute_geographic_metrics()`)

### Current State

Geographic concentration and China revenue exposure thresholds are hardcoded:

```python
# Inside compute_geographic_metrics():
china_rev += abs(value) * 0.5   # hardcoded weight
# HHI computation uses raw values with no configurable threshold
```

The `geographic_metrics` key in `scoring_weights.yml` exists but `product_metrics.py` never imports `get_weight()`.

### Fix

| Step | Action | File | Lines |
|------|--------|------|-------|
| 1 | Import `get_weight` at top of file | `operator1/features/product_metrics.py` | ~1 line |
| 2 | Replace hardcoded China weight with `get_weight('geographic_metrics.china_partial_match_weight', 0.5)` | Inside `compute_geographic_metrics()` | ~2 lines |
| 3 | Add configurable HHI concentration threshold: `get_weight('geographic_metrics.hhi_concentration_threshold', 0.25)` | Same function | ~2 lines |
| 4 | Add `wt_geo` tab to dashboard weights panel (optional) | `dashboard.py` | ~10 lines |

### Config Structure (already in yml)

```yaml
geographic_metrics:
  china_partial_match_weight: 0.5
  hhi_concentration_threshold: 0.25
  tariff_exposure_base_score: 0.3
```

---

## Issue 3: `jump_diffusion` -- Monte Carlo Estimates Jump Params from Data

**Config key:** `jump_diffusion` in `config/scoring_weights.yml`
**Model file:** `operator1/models/monte_carlo.py` (inside jump-diffusion path)

### Current State

The Merton jump-diffusion model estimates lambda (jump intensity), mu_j (jump mean), and sigma_j (jump vol) from the return distribution's tail analysis at runtime. These are data-derived, not hardcoded constants.

However, the config key could serve as **floor/ceiling bounds** or **override values** for when data estimation produces unreasonable results (e.g., insufficient tail data).

### Fix

| Step | Action | File | Lines |
|------|--------|------|-------|
| 1 | Import `get_weight` at top of file | `operator1/models/monte_carlo.py` | ~1 line |
| 2 | After data-derived estimation, apply config bounds: `lambda_jump = max(get_weight('jump_diffusion.lambda_min', 0.01), min(lambda_jump, get_weight('jump_diffusion.lambda_max', 2.0)))` | Inside jump-diffusion estimation | ~6 lines |
| 3 | Add fallback: if data estimation fails, use config defaults | Same area | ~4 lines |
| 4 | Add `wt_jump` tab to dashboard weights panel (optional) | `dashboard.py` | ~10 lines |

### Config Structure (already in yml)

```yaml
jump_diffusion:
  lambda_min: 0.01
  lambda_max: 2.0
  mu_jump_default: -0.05
  sigma_jump_default: 0.10
  antithetic: true
```

---

## Implementation Order

1. **Issue 1 (DCF):** Highest impact -- WACC assumption is the single biggest driver of intrinsic value. Making it configurable lets users calibrate for different risk environments.
2. **Issue 3 (Jump Diffusion):** Medium impact -- bounds prevent unreasonable jump parameters from distorting MC survival probabilities.
3. **Issue 2 (Geographic):** Lowest impact -- China revenue threshold is a minor classification detail.

## Total Effort

~50 lines of code changes across 3 model files + ~35 lines for optional dashboard tabs. No new dependencies. No architectural changes.

## Testing

Each fix can be verified by:
1. Editing the value in `config/scoring_weights.yml`
2. Running the pipeline and confirming the model uses the new value (check log output)
3. Reverting the config and confirming the default behavior is unchanged

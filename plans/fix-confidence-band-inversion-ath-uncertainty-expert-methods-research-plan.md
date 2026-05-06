# Fix: Confidence Band Inversion + ATH Uncertainty

## Two Problems Identified

**Problem 1 -- Inverted confidence bands**: Lower bound ($243.74) above point forecast ($239.30). Conformal half-width ($0.45) from 1-step residuals is too narrow and not re-centered after ensemble shift.

**Problem 2 -- No uncertainty expression at ATH**: Every model pushes upward near all-time highs. The system picks a direction instead of expressing "we genuinely don't know."

---

## Research: Expert Methods by Domain

### From Bayesian Statistics: Predictive Posterior Intervals (Gelman et al. 2013)

The current approach computes point forecast first, then bolts on intervals separately. The Bayesian approach is to compute the **full posterior predictive distribution** and derive both point and interval from it.

**Method:** Instead of `point = ensemble_weighted_mean; interval = conformal_half_width`, compute:
```
posterior_samples = [model_i.sample_prediction() for each model, weighted by RMSE]
point = median(posterior_samples)
lower = quantile(posterior_samples, 0.05)
upper = quantile(posterior_samples, 0.95)
```

This guarantees `lower <= point <= upper` by construction. No re-centering needed. Each model contributes samples proportional to its inverse RMSE weight. Models that disagree produce wider intervals naturally.

**Reference:** Gelman, Carlin, Stern, Dunson, Vehtari & Rubin. "Bayesian Data Analysis" (3rd ed, 2013), Ch. 14: Posterior predictive checking.

### From Quantitative Finance: Garman-Klass ATH Volatility Scaling (Garman & Klass 1980 + Bouchaud 2002)

Near ATH, realized volatility underestimates future volatility because the trailing window only saw the uptrend. Garman-Klass intraday vol and the **volatility-of-volatility** (vol-of-vol) are better forward-looking indicators.

**Method:** When `anchoring_52w_high > 0.90`:
```
vol_scale = 1.0 + (anchoring_52w_high - 0.90) * vol_of_vol_21d / volatility_21d
adjusted_half_width = base_half_width * vol_scale
```

This widens bands proportionally to how close to ATH AND how unstable recent volatility has been. A stock at ATH with stable low vol gets moderate widening; a stock at ATH with vol-of-vol spiking gets aggressive widening.

**Reference:** Bouchaud, Potters & Meyer. "Apparent multifractality in financial time series" (2002). Shows that vol estimation error increases near distribution extremes.

### From Control Theory: Integral Windup Guard (Astrom & Hagglund 2006)

The PID controller for conformal coverage can experience **integral windup** -- when the conformal calibrator has been in a long "too narrow" regime, the integral term accumulates error but can't correct fast enough during a regime change. This is why the half-width stays at $0.45 even when the ensemble disagrees by $10+.

**Method:** Add anti-windup clamping to the conformal PID:
```
# Current: integral accumulates without bound
self.integral += error

# Fixed: clamp integral term
MAX_INTEGRAL = 3.0 * target_half_width
self.integral = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self.integral + error))
```

**Reference:** Astrom & Hagglund. "Advanced PID Control" (2006), Ch. 6: Anti-windup. The standard technique in industrial control.

### From Ensemble Learning: Disagreement-Aware Intervals (Lakshminarayanan et al. 2017)

The current system uses inverse-RMSE weighting which ignores **model disagreement** as an uncertainty signal. When 6 models produce forecasts ranging from $230 to $260, the disagreement itself tells us the true uncertainty is high, regardless of each model's individual RMSE.

**Method -- Deep Ensemble Uncertainty:**
```
model_forecasts = [kalman, garch, var, lstm, tree, ets, transformer]
ensemble_mean = weighted_mean(model_forecasts)
ensemble_var = weighted_variance(model_forecasts)  # THIS IS THE KEY
epistemic_uncertainty = sqrt(ensemble_var)  # model disagreement

# Aleatoric uncertainty: from each model's own confidence
aleatoric_uncertainty = mean(model_rmse)

# Total uncertainty
total_uncertainty = sqrt(epistemic^2 + aleatoric^2)
half_width = z_score * total_uncertainty
```

When all models agree (low epistemic), bands are narrow. When models disagree (high epistemic), bands widen regardless of individual RMSE scores.

**Reference:** Lakshminarayanan, Pritzel & Blundell. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (NeurIPS 2017). Extended to non-neural ensembles by Pearce et al. (2020).

### From Robust Statistics: Hodges-Lehmann Estimator (1963)

The current ensemble uses weighted mean, which is sensitive to outlier forecasts. Near ATH, one model predicting a crash ($220) and another predicting continuation ($260) shouldn't average to $240 -- the median of pairwise means is more robust.

**Method:**
```
# Current: weighted mean
point = sum(w_i * forecast_i)

# Robust: Hodges-Lehmann (median of pairwise means)
pairwise_means = [(f_i + f_j) / 2 for all i < j]
point = median(pairwise_means)
```

This naturally dampens extreme forecasts without throwing them away entirely. It's the standard robust estimator in non-parametric statistics.

**Reference:** Hodges & Lehmann. "Estimates of Location Based on Rank Tests" (Annals of Mathematical Statistics, 1963).

### From Market Microstructure: Implied Volatility Surface Anchoring (Hull 2018)

The system already fetches options-implied volatility (IV30) but doesn't use it for interval construction. ATM implied vol is the market's consensus on future uncertainty -- it naturally widens before earnings and when institutional positioning shifts.

**Method:**
```
# Current: intervals from model RMSE (backward-looking)
half_width = z_score * rmse * sqrt(h)

# Enhanced: blend with IV (forward-looking)
iv_daily = iv30 / sqrt(252)
iv_half_width = z_score * last_close * iv_daily * sqrt(h)

# Blend: 60% IV (forward-looking), 40% model RMSE (backward-looking)
blended_half_width = 0.60 * iv_half_width + 0.40 * rmse_half_width
```

This anchors intervals to the market's own assessment of uncertainty, which already incorporates ATH effects, upcoming events, and institutional positioning.

**Reference:** Hull. "Options, Futures, and Other Derivatives" (10th ed, 2018), Ch. 20: The Black-Scholes-Merton model for implied vol surface construction.

---

## Implementation Plan

### Fix A: Always Re-Center Intervals (Bug Fix)

**File:** [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)
**Location:** After all ensemble adjustments (B1/B2/B3/C2/C3), before interval construction

**Change:** Move interval construction AFTER the final point forecast is determined, not before. Currently intervals are computed during the per-variable loop, but ensemble blending shifts the point afterward.

```python
# AFTER all adjustments are complete:
final_half_width = max(conformal_half_width, ensemble_disagreement_width)
lower = final_point - final_half_width
upper = final_point + final_half_width
```

### Fix B: Ensemble Disagreement-Aware Intervals (Lakshminarayanan 2017)

**File:** [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)
**Location:** New function `compute_ensemble_disagreement()`

**Change:** After computing weighted point forecast, compute weighted variance across models:

```python
def compute_ensemble_disagreement(
    forecasts: dict[str, float],
    weights: dict[str, float],
    point_forecast: float,
) -> float:
    """Epistemic uncertainty from model disagreement."""
    weighted_sq_diff = sum(
        w * (f - point_forecast) ** 2
        for model, f in forecasts.items()
        if (w := weights.get(model, 0)) > 0
    )
    return math.sqrt(weighted_sq_diff)
```

### Fix C: IV-Anchored Intervals (Hull 2018)

**File:** [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)
**Location:** In `compute_uncertainty_bands()` and the conformal fallback path

**Change:** When `iv30` is available in the cache, blend IV-derived half-width with model-derived half-width:

```python
iv30 = cache.get("iv30")
if iv30 is not None and not math.isnan(float(iv30.iloc[-1])):
    iv_daily = float(iv30.iloc[-1]) / math.sqrt(252)
    iv_half_width = z_score * last_close * iv_daily * math.sqrt(horizon_days)
    half_width = 0.60 * iv_half_width + 0.40 * model_half_width
```

### Fix D: Minimum Half-Width Floor (Practical Guard)

**File:** [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)
**Location:** After all half-width computation

**Change:** Enforce a minimum half-width as a percentage of the price:

```python
# Minimum half-width: 1% of price for 1d, scaling with sqrt(h)
MIN_PCT = 0.01
min_half_width = abs(point_forecast) * MIN_PCT * math.sqrt(max(horizon_days, 1))
half_width = max(half_width, min_half_width)
```

This ensures the $0.45 half-width on a $250 stock never happens again -- the floor would be `$250 * 0.01 * 1.0 = $2.50` at 1d.

### Fix E: ATH Volatility Scaling (Bouchaud 2002)

**File:** [`operator1/models/prediction_aggregator.py`](operator1/models/prediction_aggregator.py)
**Location:** After half-width computation, before final bounds

**Change:**

```python
if "anchoring_52w_high" in cache.columns:
    ath = float(cache["anchoring_52w_high"].dropna().iloc[-1])
    if ath > 0.90:
        # Near ATH: scale by vol-of-vol ratio
        vov = float(cache.get("vol_of_vol_21d", pd.Series(0)).dropna().iloc[-1]) if "vol_of_vol_21d" in cache.columns else 0
        vol = float(cache.get("volatility_21d", pd.Series(0.02)).dropna().iloc[-1])
        vov_ratio = vov / vol if vol > 1e-8 else 1.0
        ath_scale = 1.0 + (ath - 0.90) * max(1.0, vov_ratio)
        half_width *= ath_scale
```

---

## Execution Checklist

- [ ] **Fix A**: Re-center intervals after ALL ensemble adjustments (bug fix, highest priority)
- [ ] **Fix B**: Add `compute_ensemble_disagreement()` and blend into half-width
- [ ] **Fix C**: IV-anchored interval blending when iv30 available
- [ ] **Fix D**: Minimum half-width floor (1% of price * sqrt(h))
- [ ] **Fix E**: ATH volatility scaling using vol-of-vol
- [ ] Verify: confidence bands always satisfy `lower <= point <= upper`
- [ ] Create branch, commit, push, open PR

## Priority Order

1. **Fix A** (bug fix) -- re-centering must happen for any interval to make sense
2. **Fix D** (practical guard) -- prevents absurdly narrow bands regardless of method
3. **Fix B** (ensemble disagreement) -- captures the most important missing signal
4. **Fix C** (IV anchoring) -- leverages market-priced forward uncertainty
5. **Fix E** (ATH scaling) -- fine-tuning for extreme price positions

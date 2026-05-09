# GitHub Research: Multi-Frequency Model Implementations

*Source code analysis of 8 open-source projects for methods applicable to our 37 freq-blind modules.*

---

## 1. Hierarchical Forecast Reconciliation

### Nixtla/hierarchicalforecast (742 stars)

**File:** `hierarchicalforecast/methods.py` (2,203 lines, 55 functions)

**MinTrace implementation** (our target for Phase 4):

The core reconciliation formula is at class `MinTrace` (L1244):

```
P_MinT = (S'W_h S)^{-1} S'W_h^{-1}
```

Where S is the summing matrix and W_h is the covariance of coherency errors.

**6 covariance estimators** (L1278):
- `ols` -- identity matrix (no error info, simplest)
- `wls_struct` -- diagonal with structural weights (number of bottom-level series)
- `wls_var` -- diagonal with in-sample forecast variance
- `mint_cov` -- full sample covariance of in-sample residuals
- `mint_shrink` -- Schaefer-Strimmer shrunk covariance (RECOMMENDED -- handles small samples well)
- `emint` -- expectation-maximization MinTrace (newest, handles missing residuals)

**Key implementation detail:** The `_get_PW_matrices()` method at L1293 uses NumPy's `linalg.solve` instead of matrix inversion for numerical stability. Ridge regularization (`mint_shr_ridge=2e-8`) prevents singular covariance matrices.

**For our use:** We have 5 frequencies (A, Q, M, W, D) producing forecasts. The summing matrix S encodes: Annual = sum of 4 Quarterly = sum of 12 Monthly. MinTrace with `mint_shrink` is the safest choice since we have few observations at A/Q frequencies.

**Dependencies:** Only numpy. No new packages needed.

### sktime (9,754 stars)

**File:** `sktime/forecasting/reconcile.py` (464 lines)

Wraps the reconciliation into the sktime forecaster API. Key method `_get_error_covariance_matrix()` at L334:

```python
def _get_error_covariance_matrix(self, shrink=False, diag_only=False):
    residuals = self.residuals_
    W = np.cov(residuals.T)
    if shrink:
        # Schaefer-Strimmer shrinkage
        ...
    if diag_only:
        W = np.diag(np.diag(W))
    return W
```

**For our use:** The Schaefer-Strimmer shrinkage implementation is more robust than Nixtla's for small samples. We could use this directly since sktime is not a dependency, but the pattern is simple enough to implement inline.

---

## 2. GARCH Temporal Aggregation

### bashtage/arch (1,515 stars)

**File:** `arch/univariate/volatility.py` (3,812 lines, 191 functions)

The `arch` library does NOT implement Drost-Nijman temporal aggregation natively. However, the `VolatilityProcess` base class (L202) provides the foundation:

- `forecast()` method produces h-step-ahead variance forecasts
- Multi-step forecasts use the analytical recursion: `sigma^2_{t+h} = omega + (alpha + beta) * sigma^2_{t+h-1}`
- For GARCH(1,1), the multi-step forecast converges to: `sigma^2_infty = omega / (1 - alpha - beta)`

**Key insight from arch source:** The library's `FixedVariance` (L3426) and `EGARCH` (L1779) classes show how to handle non-standard variance processes. The pattern for temporal aggregation would be:

```python
# Drost-Nijman aggregation for GARCH(1,1) daily -> weekly (m=5)
alpha_agg = alpha * (1 - beta**m) / (1 - beta)
beta_agg = beta**m
omega_agg = omega * (1 - beta**m) / (1 - beta)
```

**For our use:** Instead of implementing Drost-Nijman aggregation (complex for GARCH(p,o,q)), we should fit GARCH directly at the target frequency with `arch.arch_model()`. The only change needed is the annualization step: replace `vol * sqrt(252)` with `vol * sqrt(periods_per_year)`.

---

## 3. Discrete-Time Survival Analysis

### tomer1812/pydts (25 stars)

**File:** `src/pydts/fitters.py` (796 lines, 27 functions)

**TwoStagesFitter** (L224) -- the most relevant for our use:

Stage 1: Estimate baseline hazard alpha_j(t) via MLE
Stage 2: Fit Cox PH for covariate effects beta_j

The discrete hazard function uses complementary log-log link:

```python
def _hazard_transformation(self, a):
    return np.log(-np.log(1 - a))

def _hazard_inverse_transformation(self, a):
    return 1 - np.exp(-np.exp(a))
```

**Data expansion:** The key insight is that discrete-time survival requires "data expansion" -- each observation at risk at time t generates a row (L61). For quarterly data with 8 observations, this produces ~36 expanded rows (8+7+6+...+1), which is sufficient for fitting.

**For our use:** Our current `compute_cox_survival_score()` in `survival_mode.py` uses `lifelines.CoxPHFitter` which assumes continuous time. At quarterly frequency (8 observations), the continuous-time assumption breaks down. Switching to a discrete-time model with complementary log-log link would be more appropriate.

**Dependencies:** `pydts` uses `lifelines.CoxPHFitter` internally -- we already have this.

---

## 4. PyTorch Forecasting / Temporal Fusion Transformer

### sktime/pytorch-forecasting (4,884 stars)

The TFT implementation handles multi-horizon forecasting natively with quantile loss:

```python
# Quantile loss at different levels
QuantileLoss(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
```

**Key pattern for frequency adaptation:** The TFT uses `time_varying_known_reals` and `time_varying_unknown_reals` with explicit `max_encoder_length` and `max_prediction_length`. At quarterly frequency, these would be:
- `max_encoder_length = 8` (2 years of quarterly data)
- `max_prediction_length = 4` (1 year ahead)

**For our use:** Our `transformer_forecaster.py` uses fixed `lookback` from adaptive_windows but doesn't adapt the loss function. At quarterly frequency, switching from MSE to quantile loss would better capture the distribution.

---

## 5. Conformal Prediction for Time Series

### MAPIE (scikit-learn compatible, already in our deps)

MAPIE's `MapieTimeSeriesRegressor` handles non-exchangeable time series with:
- **EnbPI** (Ensemble Batch Prediction Intervals): sliding window residuals
- **ACI** (Adaptive Conformal Inference): online adjustment for distribution shift

**Key implementation pattern:**

```python
# Adaptive alpha based on coverage error
alpha_t = alpha_{t-1} + gamma * (alpha - indicator(y_t in C_t))
```

This is essentially what our `ConformalPIDCalibrator` already does. The difference is MAPIE uses a simpler gamma (fixed learning rate) while we use full PID. Our approach is more sophisticated.

**For frequency adaptation:** The residual window should scale with frequency. At quarterly frequency, use `cv_block_size=1` (each quarter is one block) instead of the default rolling window.

---

## 6. Multi-Frequency Financial Analysis Patterns

### No single dominant open-source project exists for multi-frequency financial analysis.

However, several patterns emerge from the examined codebases:

**Pattern A: Frequency as a first-class parameter (from sktime)**

sktime's design principle: every estimator/transformer accepts a `freq` parameter via the `ForecastingHorizon` object. The frequency propagates through the entire pipeline automatically.

```python
fh = ForecastingHorizon([1, 2, 3, 4], freq="QS")  # quarterly
forecaster.fit(y_train, fh=fh)
```

**For our use:** Our MF pipeline already does this partially via the `_freq_context` thread-local. The gap is that individual models don't read it.

**Pattern B: Frequency-indexed constants (from statsforecast/Nixtla)**

Nixtla's `statsforecast` uses `season_length` parameter that adapts all internal computations:

```python
AutoETS(season_length=4)   # quarterly
AutoETS(season_length=12)  # monthly
AutoETS(season_length=252) # daily
```

**For our use:** This is exactly what our proposed `ANNUALIZATION_FACTOR[freq]` map does.

**Pattern C: Model routing by data characteristics (from AutoML/FLAML)**

FLAML and Auto-sklearn route models based on dataset meta-features (n_samples, n_features, task type). The pattern:

```python
if n_samples < 30:
    use_models = ["naive", "ets", "theta"]
elif n_samples < 100:
    use_models = ["arima", "ets", "lightgbm"]
else:
    use_models = ["lightgbm", "lstm", "tft"]
```

**For our use:** Our forecasting cascade already falls back, but it's accidental (try LSTM, catch exception, try tree, catch, try baseline). Making it explicit via our proposed `MODEL_FREQ_COMPATIBILITY` table is the right approach.

---

## 7. Hurst Exponent and Volatility Scaling

No major dedicated library, but the pattern is well-established in quantitative finance:

```python
# R/S method (from various quant repos)
def hurst_exponent(series, max_k=None):
    n = len(series)
    max_k = max_k or n // 2
    RS = []
    for k in range(10, max_k):
        subseries = np.array_split(series, n // k)
        rs_values = []
        for ss in subseries:
            mean = np.mean(ss)
            cumdev = np.cumsum(ss - mean)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(ss, ddof=1)
            if S > 0:
                rs_values.append(R / S)
        if rs_values:
            RS.append((k, np.mean(rs_values)))
    log_k = np.log([r[0] for r in RS])
    log_rs = np.log([r[1] for r in RS])
    H = np.polyfit(log_k, log_rs, 1)[0]
    return H
```

**For our use:** Our `derived_variables.py` already computes `hurst_exponent` at Stage 19. We just need to use it: `vol_scaled = vol * (T ** H)` instead of `vol * sqrt(T)`.

---

## Summary: What to Adopt from Each Project

| Project | Stars | Method | Adopt? | Priority |
|---------|-------|--------|--------|----------|
| Nixtla/hierarchicalforecast | 742 | MinTrace reconciliation with mint_shrink | YES -- new module | Phase 4 |
| bashtage/arch | 1,515 | Fit GARCH at target freq + correct annualization | YES -- fix annualization | Phase 1 |
| tomer1812/pydts | 25 | Discrete-time survival with cloglog link | YES -- replace Cox at Q/A freq | Phase 5 |
| sktime/pytorch-forecasting | 4,884 | Quantile loss at low frequencies | MAYBE -- low priority | Phase 5 |
| MAPIE | (in deps) | Block CV for conformal at low freq | YES -- change cv_block_size | Phase 2 |
| sktime | 9,754 | Freq as first-class parameter pattern | YES -- FrequencyContext design | Phase 1 |
| statsforecast/Nixtla | (in deps) | season_length parameter pattern | YES -- ANNUALIZATION_FACTOR map | Phase 1 |

---

## Recommended Implementation Order

1. **Phase 1:** `freq_constants.py` + replace 252 in 14 modules (from arch and statsforecast patterns)
2. **Phase 2:** Model routing table + conformal block CV (from FLAML and MAPIE patterns)
3. **Phase 3:** Hurst-based vol scaling T^H (from quantitative finance standard)
4. **Phase 4:** MinTrace hierarchical reconciliation (from Nixtla, ~200 lines)
5. **Phase 5:** Discrete-time survival at Q/A + quantile loss for NN models (from pydts and pytorch-forecasting)

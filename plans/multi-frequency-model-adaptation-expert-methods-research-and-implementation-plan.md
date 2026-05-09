# Multi-Frequency Model Adaptation: Expert Methods Research

*The core problem: 73/82 analytical modules assume daily frequency. When the MF pipeline runs them at Q/A/M/W, they produce distorted results because of hardcoded 252-day annualization, fixed rolling windows, and daily-only statistical assumptions.*

*This plan draws from quantitative finance, signal processing, econometrics, and systems engineering to identify the right approach for each model class.*

---

## The Fundamental Problem

Financial time series at different frequencies are not just resampled versions of each other. They have fundamentally different statistical properties:

| Property | Daily | Weekly | Monthly | Quarterly | Annual |
|----------|-------|--------|---------|-----------|--------|
| Distribution | Leptokurtic (fat tails) | Less kurtotic | Approaching normal | Near-normal | Normal |
| Autocorrelation | Weak (EMH) | Moderate | Moderate | Strong | Strong |
| Volatility clustering | Strong (GARCH) | Moderate | Weak | Negligible | None |
| Mean reversion | Weak | Moderate | Strong | Strong | Very strong |
| Predictability | Low (noise-dominated) | Moderate | Higher | Highest | High |
| Noise/signal ratio | Very high | High | Moderate | Low | Very low |

A model designed for daily data that's naively applied to quarterly data will mischaracterize the statistical regime.

---

## Approach 1: Frequency Context Object (Systems Engineering)

**Source:** Microservices context propagation pattern (Google SRE, distributed tracing)

Instead of passing `freq` as a string parameter to every function, create a **FrequencyContext** object that carries all frequency-derived constants and is available via thread-local storage (already partially implemented via `_freq_context`).

```python
@dataclass
class FrequencyContext:
    freq: str                    # "D", "W", "M", "Q", "S", "A"
    periods_per_year: int        # 252, 52, 12, 4, 2, 1
    vol_annualization: float     # sqrt(periods_per_year)
    rolling_month: int           # 21, 4, 1, 0.33, ...
    rolling_quarter: int         # 63, 13, 3, 1, ...
    rolling_year: int            # 252, 52, 12, 4, 2, 1
    min_obs_for_garch: int       # 100, 30, 12, 4, 2, 1
    min_obs_for_hmm: int         # 50, 20, 8, 4, 2, 1
    min_obs_for_lstm: int        # 100, 30, 12, skip, skip, skip
    should_skip: set[str]        # model names to skip at this freq
    distribution_family: str     # "student_t", "student_t", "normal", ...
```

Every model calls `get_freq_context()` and uses it for all frequency-dependent decisions. No more scattered `252` literals.

---

## Approach 2: Temporal Aggregation Theory (Econometrics)

**Source:** Working (1960), Tiao (1972), Marcellino (1999) -- "temporal aggregation bias"

When aggregating a high-frequency process to low frequency, statistical parameters transform in specific ways:

### Volatility Scaling (Mandelbrot 1963, Cont 2001)

Daily vol does NOT scale as `sqrt(T)` for financial returns (violates i.i.d. assumption). The correct scaling depends on the Hurst exponent:

```
sigma_T = sigma_1 * T^H
```

Where H is the Hurst exponent (H=0.5 for random walk, H>0.5 for trending, H<0.5 for mean-reverting).

**Fix for monte_carlo.py, forecasting.py, copula.py:** Instead of `vol * sqrt(252)`, compute the empirical scaling factor from the data using the Hurst exponent already computed by `derived_variables.py`:

```python
H = cache.get("hurst_exponent", 0.5)  # from derived_variables stage 19
vol_scaled = vol_daily * (periods_per_year ** H)
```

### GARCH Temporal Aggregation (Drost & Nijman 1993)

A GARCH(1,1) at daily frequency does NOT aggregate to GARCH(1,1) at weekly frequency. The aggregated process is a weak GARCH with modified parameters:

```
alpha_agg = sum(alpha * beta^(i-1) * (1 - beta^m) / (1 - beta)) for i=1..m
beta_agg = beta^m
```

Where m is the aggregation factor (5 for daily->weekly, 63 for daily->quarterly).

**Fix for forecasting.py GARCH path:** When running at non-daily frequency, either:
1. Fit GARCH directly to the low-freq returns (correct but fewer observations)
2. Fit GARCH at daily, then apply Drost-Nijman aggregation formulas

Option 1 is simpler and already happens naturally when the MF pipeline passes the Q/A cache. The issue is the annualization step after fitting.

### VAR Temporal Aggregation (Lutkepolh 1987)

A VAR(p) at daily frequency aggregates to a VAR(p*) at lower frequency with different lag structure and coefficient matrix. The correct approach is to fit VAR directly at the target frequency (which the MF pipeline already does by passing the frequency-appropriate cache).

**No code change needed** -- VAR already fits on whatever data it receives. The only issue is the `min_obs >= 50` threshold which should scale with frequency.

---

## Approach 3: Scale-Free Feature Engineering (Signal Processing)

**Source:** Fourier analysis, wavelet theory, normalized indicators

Instead of making every model frequency-aware, make the FEATURES frequency-invariant. If all features are normalized/dimensionless, models don't need to know the frequency.

### Approach 3a: Fractional Returns Instead of Absolute Returns

Replace `return_1d` with `log_return` which is already frequency-invariant (log returns aggregate by summation).

Already partially done -- `log_return_1d` exists in `derived_variables.py`. The issue is that many downstream models use `return_1d` (arithmetic) instead.

### Approach 3b: Z-Score Normalization Per Frequency

`feature_normalization.py` already computes z-scores but uses hardcoded 63-day and 252-day windows. Fix: replace with `rolling_quarter` and `rolling_year` from FrequencyContext.

### Approach 3c: Rank-Based Features (Cont & de Larrard 2013)

Convert all features to percentile ranks within their frequency-appropriate history. A percentile rank is inherently frequency-invariant -- "90th percentile of quarterly PE" has the same meaning regardless of whether the underlying data is daily or quarterly.

Already partially done in `feature_normalization.py` (percentile_252d). Extend to all features.

---

## Approach 4: Model Selection Per Frequency (Machine Learning)

**Source:** AutoML, model routing (Brazdil et al. 2003)

Instead of running ALL models at ALL frequencies, define a routing table:

```python
MODEL_FREQ_COMPATIBILITY = {
    # Model             D     W     M     Q     A
    "garch":          [True, True, True, False, False],
    "hmm_4regime":    [True, True, True, False, False],
    "kalman":         [True, True, True, True,  True ],
    "lstm":           [True, True, False,False, False],
    "tree_ensemble":  [True, True, True, True,  False],
    "var":            [True, True, False,False, False],
    "ar1":            [True, True, True, True,  True ],
    "ets":            [True, True, True, True,  True ],
    "baseline":       [True, True, True, True,  True ],
    "copula":         [True, True, True, False, False],
    "dtw":            [True, True, False,False, False],
    "pattern":        [True, False,False,False, False],
    "particle_filter":[True, True, True, True,  False],
    "transformer":    [True, True, False,False, False],
    "monte_carlo":    [True, True, True, True,  True ],  # but params change
    "conformal":      [True, True, True, True,  True ],
}
```

The forecasting cascade already handles this via `min_obs` guards that cause models to fall back. But it's accidental -- making it explicit via a routing table is cleaner and avoids wasting time trying to fit models that will certainly fail.

---

## Approach 5: Hierarchical Reconciliation (Supply Chain Forecasting)

**Source:** Hyndman & Athanasopoulos (2021), Wickramasuriya et al. (2019) -- "optimal forecast reconciliation"

Instead of treating each frequency's forecast independently, enforce coherence: the annual forecast must equal the sum of quarterly forecasts, which must equal the sum of monthly forecasts, etc.

The frequency fusion module (`frequency_fusion.py`) does a weighted average of forecasts across frequencies but does NOT enforce this additive coherence constraint.

**Method:** MinT (Minimum Trace) reconciliation:
1. Generate base forecasts at each frequency independently (already done)
2. Compute the reconciliation matrix W that minimizes trace of the forecast error covariance
3. Apply: `reconciled = S * (S'WS)^{-1} * S'W * base_forecasts`

Where S is the summing matrix encoding the hierarchical relationships.

This would be a new module `operator1/models/hierarchical_reconciliation.py` called after frequency fusion.

---

## Approach 6: Frequency-Specific Loss Functions (Deep Learning)

**Source:** Temporal Fusion Transformer (Lim et al. 2021), N-BEATS (Oreshkin et al. 2020)

For the neural network models (LSTM, Transformer), the loss function should adapt to frequency:

- **Daily:** MSE on 1-step-ahead (standard)
- **Weekly:** Multi-step MSE with exponential decay weighting
- **Monthly:** Quantile loss (distribution matters more than point forecast at low freq)
- **Quarterly:** Pinball loss with asymmetric penalties (underestimation of revenue is worse than overestimation)

---

## Approach 7: Survival Analysis Frequency Adaptation (Biostatistics)

**Source:** Kalbfleisch & Prentice (2002), grouped survival data

The Cox PH model in `survival_mode.py` is fitted on daily data. At quarterly frequency, the correct approach is **discrete-time survival analysis** (complementary log-log link) instead of the continuous-time Cox model:

```
P(T = t | T >= t, X) = 1 - exp(-exp(beta'X + gamma_t))
```

This is the grouped-data version of the Cox model and is appropriate when events are only observed at discrete intervals (filing dates).

---

## Implementation Priority

### Phase 1: FrequencyContext + 252 Replacement (Highest ROI)

Create `freq_constants.py` with the FrequencyContext object and replace all hardcoded 252 in the 14 priority-1 modules. This fixes the most visible distortion (wrong annualization) with minimal risk.

**~14 files, ~50 lines per file = ~700 lines**

### Phase 2: Model Routing Table + Skip Guards

Add the `MODEL_FREQ_COMPATIBILITY` routing table to the multi-frequency runner and skip incompatible models. This eliminates wasted computation and misleading results.

**~3 files, ~100 lines**

### Phase 3: Hurst-Based Vol Scaling

Replace `sqrt(T)` vol scaling with `T^H` scaling using the Hurst exponent already in the cache. This fixes the most common statistical error in multi-frequency finance.

**~5 files, ~30 lines**

### Phase 4: Hierarchical Reconciliation (New Module)

Add MinT reconciliation after frequency fusion to enforce additive coherence across frequencies.

**~1 new file, ~200 lines**

### Phase 5: Frequency-Specific Loss Functions for NN Models

Adapt LSTM and Transformer loss functions per frequency.

**~2 files, ~50 lines**

---

## Expected Impact

| Improvement | Before | After |
|-------------|--------|-------|
| Q/A pipeline accuracy | Distorted by 252 annualization | Correct Q/A-native scaling |
| Model compatibility | All models attempted at all freqs (many fail) | Only compatible models run per freq |
| Vol scaling | sqrt(252) everywhere | T^H (Hurst-corrected) |
| Forecast coherence | Independent per-freq forecasts | Hierarchically reconciled |
| Wasted computation | ~40% of model runs fail at low freqs | Near-zero failed runs |

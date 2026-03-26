# Tier 3 Adaptive Parameters: Expert Methods and Formulas

Replacing 42 hardcoded constants across 15 files with data-derived values. These are individually lower impact but collectively affect many modules.

---

## 11. Rolling Window Sizes (12+ modules)

### Current State

| File | Constant | Value | Used For |
|------|----------|-------|----------|
| `derived_variables.py:169` | volatility window | 21 | `volatility_21d` |
| `derived_variables.py:173` | drawdown window | 252 | `drawdown_252d` |
| `walk_forward.py:67` | `_ERROR_ROLLING_WINDOW` | 63 | Error tracking |
| `survival_timeline.py:63` | `_STABILITY_WINDOW` | 21 | Mode stability |
| `survival_timeline.py:622` | `_TRANSITION_WINDOW` | 42 | Transition probs |
| `vanity.py:64-65` | `_TREND_WINDOW_SHORT/LONG` | 21/63 | Vanity trends |
| `forecasting.py:71` | `_BURNOUT_WINDOW` | 126 | Burn-out retraining |
| `genetic_optimizer.py:197` | `validation_window` | 63 | GA evaluation |
| `financial_health.py:73` | `_TREND_WINDOW` | 63 | FH trend scoring |

### Why Fixed Windows Are Suboptimal

A company that files quarterly (every ~63 business days) should use different analysis windows than one that files annually (~252 days). Using a 21-day volatility window for an annual filer means measuring noise in interpolated data, not real volatility.

### Proposed Method: Filing-Frequency-Anchored Windows

**Method AB: Harmonic Window Sizing** (Novel -- based on Nyquist-Shannon)

**Theory**: The Nyquist sampling theorem says you need at least 2 samples per cycle to detect a pattern. If filings arrive every F days, meaningful financial analysis needs windows that are integer multiples of F.

**Formula**:
```
base_window = filing_period_days  # quarterly=63, semi=126, annual=252
short_window = max(base_window // 3, 5)      # ~1/3 of filing period
medium_window = base_window                    # 1 filing period
long_window = base_window * 2                  # 2 filing periods
trend_window = base_window * 4                 # 4 filing periods (2 years for quarterly)
```

For quarterly filers: short=21, medium=63, long=126, trend=252 (matches current defaults).
For annual filers: short=84, medium=252, long=504, trend=1008 (current 21/63 is too short).

**Data source**: `filing_calendar_result.detected_frequency` from Step 4c.

**Implementation**:
```python
def compute_adaptive_windows(filing_frequency):
    period_map = {"quarterly": 63, "semiannual": 126, "annual": 252, "unknown": 63}
    base = period_map.get(filing_frequency, 63)
    return {
        "short": max(base // 3, 5),
        "medium": base,
        "long": base * 2,
        "trend": base * 4,
        "volatility": max(base // 3, 5),
        "drawdown": base * 4,
        "stability": max(base // 3, 5),
        "error_rolling": base,
        "burnout": base * 2,
        "validation": base,
    }
```

**Academic basis**: Nyquist-Shannon sampling theorem applied to financial time series. Oppenheim & Willsky (1997) "Signals and Systems." The filing period IS the fundamental frequency of financial data.

---

## 12. Neural Network Hyperparameters

### Current State

| File | Constants | Values |
|------|-----------|--------|
| `transformer_forecaster.py:37-45` | d_model, n_heads, epochs, lr, dropout, lookback | 32, 4, 60, 0.001, 0.1, 21 |
| `forecasting.py:74-79` | LSTM hidden, layers, lookback, epochs, lr | 32, 1, 21, 50, 0.001 |
| `gain_imputer.py:28-33` | epochs, hidden, hint_rate, alpha | 100, 32, 0.9, 10.0 |
| `vae_imputer.py:51-55` | epochs, hidden, kl_weight | 80, 32, 0.5 |

### Proposed Methods

#### Method AC: Data-Size-Proportional Architecture (Kaplan et al. 2020)

**Theory**: Neural scaling laws (Kaplan et al. 2020, "Scaling Laws for Neural Language Models") show that optimal model size scales as a power law of dataset size. For small financial datasets, smaller models generalize better.

**Formula** (adapted from scaling laws for tabular data):
```
d_model = max(16, min(128, int(8 * log2(n_eff))))
n_heads = max(1, d_model // 8)
hidden_dim = max(16, min(64, int(4 * sqrt(n_features * n_eff / 100))))
n_layers = 1 if n_eff < 200 else 2
```

For n_eff=50, n_features=10: d_model=45->48 (round to 8), hidden=14->16, n_layers=1
For n_eff=500, n_features=20: d_model=72->72, hidden=40, n_layers=2

**Academic basis**: Kaplan et al. (2020, arXiv:2001.08361). Extended by Bahri et al. (2024) for tabular data scaling.

#### Method AD: Learning Rate from Loss Landscape (Smith 2017)

**Theory**: The optimal learning rate can be found by the "LR range test" (Smith 2017): train for a few iterations with exponentially increasing LR, find the LR where loss decreases fastest (steepest negative slope).

For production use, a simpler formula based on batch size:
```
lr = 0.001 * sqrt(batch_size / 32)  # linear scaling rule (Goyal et al. 2017)
```

**Epochs from early stopping**: Instead of fixed epochs, use patience-based early stopping (already implemented in transformer_forecaster.py:287, but the fixed `patience=10` and `epochs=60` should be:
```
max_epochs = max(30, int(n_eff / batch_size * 3))  # 3 passes through data
patience = max(5, max_epochs // 6)
```

#### Method AE: Dropout from Effective Sample Size (Srivastava et al. 2014)

**Theory**: Dropout rate should increase with the ratio of model parameters to training samples (higher dropout for more overfit-prone setups).

**Formula** (Baldi & Sadowski 2013):
```
optimal_dropout = 1 - sqrt(n_eff / (n_eff + n_params))
```

For n_eff=100, n_params=1024: dropout = 1 - sqrt(100/1124) = 0.70 (high -- limited data)
For n_eff=5000, n_params=1024: dropout = 1 - sqrt(5000/6024) = 0.09 (low -- plenty of data)

**Academic basis**: Srivastava et al. (2014, JMLR) "Dropout: A Simple Way to Prevent Overfitting." Baldi & Sadowski (2013) optimal dropout analysis.

### Recommended: Method AC (scaling-law architecture) + Method AD (early stopping) + Method AE (adaptive dropout)

All use `n_eff` from the Kish effective sample size (already computed in Tier 2).

---

## 13. Particle Filter Noise Scales

### Current State
```python
_STATE_NOISE_SCALE: float = 0.02  # 2% of state
_OBS_NOISE_SCALE: float = 0.05   # 5% of state
```

### Proposed Method

#### Method AF: Innovation-Based Noise Estimation (Mehra 1970)

**Theory**: The process and observation noise scales can be estimated from the innovation sequence (prediction errors). If the filter is correctly specified, innovations should be white noise. Mehra (1970) showed how to back out Q (process noise) and R (observation noise) from the innovation covariance.

**Formula** (simplified for univariate):
```
innovation = actual[t] - predicted[t]
C_innov = cov(innovation sequence)

R_hat = C_innov - H * P_predicted * H'  # observation noise
Q_hat = P_filtered - (I - K*H) * P_predicted  # process noise
```

**Simplified practical version**:
```python
def estimate_noise_scales(series):
    diff = series.diff().dropna()
    obs_noise = float(diff.std())  # observation noise ~ std of changes
    
    # Process noise = fraction of observation noise
    # (process is smoother than observations)
    state_noise = obs_noise * 0.4  # typical ratio from financial data
    
    # Normalize to fraction of state magnitude
    state_scale = state_noise / max(abs(series.mean()), 1e-8)
    obs_scale = obs_noise / max(abs(series.mean()), 1e-8)
    
    return max(0.001, min(0.20, state_scale)), max(0.005, min(0.50, obs_scale))
```

**Academic basis**: Mehra (1970, IEEE TAC) "On the Identification of Variances and Adaptive Kalman Filtering." Also used in the adaptive Kalman literature (Sage & Husa 1969).

---

## 14. Candlestick Pattern Thresholds

### Current State
```python
_BODY_THRESHOLD = 0.3   # body/range ratio for "real-bodied"
_DOJI_THRESHOLD = 0.1   # body/range ratio for doji
```

### Proposed Method

#### Method AG: Adaptive Thresholds from Recent Body Distribution

**Theory**: Candlestick patterns are relative to "what's normal for this stock." A tech stock with large daily ranges has different body/range ratios than a utility. Use the recent empirical distribution of body/range ratios.

**Formula**:
```
body_ratios = |close - open| / (high - low) for last 63 days
doji_threshold = percentile(body_ratios, 10)   # bottom 10% are doji-like
body_threshold = percentile(body_ratios, 50)    # median is "normal body"
```

This adapts automatically to the stock's volatility regime. During high-volatility periods, the thresholds adjust upward.

**Academic basis**: Bulkowski (2008) "Encyclopedia of Candlestick Charts" -- pattern definitions should be relative to recent price action, not absolute.

---

## 15. Fuzzy Protection Membership Functions

### Current State
```python
# fuzzy_protection.py:142-147
if ratio >= 0.005: score = high
if ratio >= 0.001: score = medium
if ratio >= 0.0001: score = low
# trimf breakpoints: [0.35, 0.55, 0.75]
```

### Proposed Method

#### Method AH: Data-Driven Fuzzy Membership via Fuzzy C-Means (Bezdek 1981)

**Theory**: Instead of hand-crafted triangular membership functions, use Fuzzy C-Means clustering on the actual market_cap/GDP ratios across all companies in the dataset to find natural membership function boundaries.

**Formula**:
```
# FCM with 3 clusters on market_cap_to_gdp ratios
centroids = fcm(ratios, n_clusters=3)
# Sort: low, medium, high
# Use centroids as peaks of triangular MFs
# Crossover points at midpoints between adjacent centroids
```

**Fallback**: When insufficient data, use log-scale percentiles of the GDP ratio distribution:
```
low_peak = percentile(log_ratios, 25)
med_peak = percentile(log_ratios, 50)
high_peak = percentile(log_ratios, 75)
```

**Academic basis**: Bezdek (1981) "Pattern Recognition with Fuzzy Objective Function Algorithms." FCM is the standard for learning fuzzy membership functions from data.

---

## 16. Frequency Interpolator Confidence

### Current State
```python
distance_decay = (1.0 - distances/max_gap * 0.5).clip(0.3, 1.0)
confidence = (base_conf * distance_decay * type_mult).clip(0.1, 0.95)
```

### Proposed Method

#### Method AI: Gaussian Process Posterior Variance (Rasmussen & Williams 2006)

**Theory**: The GP posterior variance gives a principled uncertainty measure at any interpolation point. Near observed data, variance is low (high confidence). Far from data, variance increases. This naturally produces the desired confidence decay without arbitrary clipping.

**Formula**:
```
sigma_posterior^2(x*) = k(x*, x*) - k(x*, X) * K^-1 * k(X, x*)
confidence = 1 - sigma_posterior / sigma_prior
```

The missing_data_estimator.py already uses GP regression -- this reuses the same kernel.

**Simpler practical version** (no GP needed):
```python
def confidence_from_distance(days_from_nearest_filing, filing_period):
    # Exponential decay centered on filing dates
    # Half-life = 1/4 of filing period (25% decay per quarter for quarterly filers)
    halflife = filing_period / 4
    decay = math.exp(-0.693 * days_from_nearest_filing / halflife)
    return max(0.10, min(0.95, decay))
```

**Academic basis**: Rasmussen & Williams (2006) "Gaussian Processes for Machine Learning." The posterior variance is the theoretically correct interpolation uncertainty.

---

## 17. Model Synergy Constants

### Current State
```python
# model_synergies.py:513
peer_threshold_factor = 0.8 / 1.2  # lower/upper adjustment
# model_synergies.py:711
plane_weight_boost = 1.0 + 0.3 * delta
```

### Proposed Method

#### Method AJ: Walk-Forward Validated Synergy Weights

**Theory**: Instead of fixed 0.8/1.2 and 0.3 boost factors, use the walk-forward evaluation results to determine how much each synergy actually improved predictions.

**Formula**:
```
# During walk-forward, track MAE with and without each synergy
synergy_improvement = (mae_without - mae_with) / mae_without
boost_factor = 1.0 + synergy_improvement * 2.0  # scale to [1.0, 3.0]
```

If a synergy improved MAE by 5%, the boost is 1.10.
If it worsened MAE (negative improvement), the boost is clamped to 1.0 (no harm).

**Data source**: `walk_forward_result.mode_scores` already tracks per-model performance.

---

## 18. Stale Data Threshold

### Current State
```python
stale_threshold_days: int = 180  # data_reconciliation.py
```

### Proposed Method

#### Method AK: Filing-Frequency-Proportional Staleness (trivial)

```python
stale_threshold = {
    "quarterly": 120,   # 2x quarterly period
    "semiannual": 210,  # ~1.7x semi-annual period
    "annual": 400,      # ~1.6x annual period
    "unknown": 180,     # current default
}[filing_frequency]
```

**Data source**: `filing_calendar_result.detected_frequency` from Step 4c.

---

## 19. Entity Discovery Scoring

### Current State
```python
# entity_discovery.py:93
score += int(ratio * 30)  # name similarity weight = 30 points
MATCH_SCORE_THRESHOLD: int = 70
```

### Proposed Method

#### Method AL: Market-Specific Name Similarity Calibration

**Theory**: Name matching accuracy varies by market. Japanese companies often have romanized names that differ significantly from their official English names. Chinese companies have pinyin vs English name mismatches. Calibrate the similarity weight by market.

**Formula**:
```python
NAME_WEIGHT_BY_MARKET = {
    "us_sec_edgar": 30,    # English names, high match quality
    "uk_companies_house": 30,
    "jp_jquants": 20,      # Romanized Japanese, lower confidence
    "kr_dart": 25,         # Korean names, moderate
    "cn_sse": 15,          # Chinese pinyin, lowest confidence
    "tw_mops": 20,
    "br_cvm": 25,
    "default": 25,
}
# Compensate by increasing ticker exact match weight for low-name-confidence markets
TICKER_WEIGHT_BY_MARKET = {
    "cn_sse": 50,  # ticker codes are the primary identifier
    "tw_mops": 45,
    "default": 40,
}
```

**Match threshold**: Instead of fixed 70, use the market's typical match quality:
```
threshold = 50 + (10 * name_weight / 30)  # scales 50-60 for low, 70-80 for high
```

---

## 20. SIX Proxy Blend Weights

### Current State
```python
blended_pe = result.implied_pe * 0.65 + ebo_pe * 0.35
```

### Proposed Method

#### Method AM: Model Confidence Blend (Bayesian Model Averaging Lite)

**Theory**: Blend proxy models by their estimation confidence rather than fixed weights. Each proxy method in `six_derived_proxies.py` has a different confidence level based on data availability.

**Formula**:
```python
def blend_by_confidence(estimates, confidences):
    # Normalize confidences to sum to 1
    w = np.array(confidences) / sum(confidences)
    return np.dot(estimates, w)
```

Where confidence comes from:
- Dividend-based PE: high confidence if 10+ years of dividends
- EBO residual income: confidence from the model's R-squared
- Merton-based PE: confidence from equity volatility estimation quality

**Data source**: Already tracked in `SIXProxyResult` fields.

---

## Summary: Selected Methods Per Constant Group

| # | Constant Group | Primary Method | Complexity | New Deps |
|---|---------------|---------------|------------|----------|
| 11 | Rolling windows (9 constants) | AB: Nyquist filing-frequency anchoring | Low | None |
| 12 | NN hyperparams (16 constants) | AC+AD+AE: Scaling laws + early stop + adaptive dropout | Medium | None |
| 13 | Particle filter noise (2 constants) | AF: Innovation-based estimation (Mehra 1970) | Low | None |
| 14 | Candlestick thresholds (2 constants) | AG: Recent body distribution percentiles | Low | None |
| 15 | Fuzzy membership (4 constants) | AH: Fuzzy C-Means or log-percentile | Medium | None (scikit-fuzzy already installed) |
| 16 | Interpolation confidence (3 constants) | AI: Exponential decay with filing-period half-life | Low | None |
| 17 | Synergy constants (3 constants) | AJ: Walk-forward validated improvement | Low | None |
| 18 | Stale threshold (1 constant) | AK: Filing-frequency proportional | Trivial | None |
| 19 | Entity scoring (2 constants) | AL: Market-specific name confidence | Low | None |
| 20 | SIX blend weights (1 constant) | AM: Model confidence blend | Low | None |

### New Dependencies: None

### Academic References
- Oppenheim & Willsky (1997) -- Nyquist-Shannon for financial windows
- Kaplan et al. (2020) -- Neural scaling laws
- Smith (2017) -- LR range test
- Srivastava et al. (2014) -- Dropout theory
- Baldi & Sadowski (2013) -- Optimal dropout rate
- Mehra (1970) -- Innovation-based noise estimation
- Bulkowski (2008) -- Adaptive candlestick thresholds
- Bezdek (1981) -- Fuzzy C-Means for membership functions
- Rasmussen & Williams (2006) -- GP posterior variance

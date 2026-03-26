# Category C Adaptive Parameters: Expert Methods and Formulas

62 "edge case" parameters that were initially assessed as marginal improvement. After deeper analysis, I've reclassified them into 3 subcategories.

---

## Reclassification

After careful review, the 62 findings break down as:

| Subcategory | Count | Verdict |
|-------------|-------|---------|
| **C-Actionable**: genuinely should be adaptive | 24 | Implement |
| **C-Structural**: fixed by design (definitions, not parameters) | 22 | No action -- these are structural constants |
| **C-Trivial**: data validation bounds, not analytical parameters | 16 | No action -- these are safety guards |

---

## C-Structural (22 findings, No Action Needed)

These are **definitions**, not tunable parameters:

- **Holder percentage validation** (`0.01 < pct < 100`): data integrity check, not a threshold. A percentage below 0.01% or above 100% is corrupt data.
- **Health runway labels** (`> 24 months`, `> 12 months`): human-readable categories. "More than 24 months of cash" IS the definition of "strong runway" -- this is not a statistical threshold.
- **Cache TTL** (`age_hours < 24`): infrastructure timing, correctly fixed at 24 hours.
- **Filing period boundaries** (`< 120`, `< 250`, `< 500` days): these are the filing frequency DEFINITIONS. 120 days = boundary between quarterly and semi-annual. These define what "quarterly" means -- they're not tunable.
- **Cycle period names** (`18-25` days = monthly, `55-70` = quarterly): calendar-aligned definitions.

---

## C-Trivial (16 findings, No Action Needed)

These are **data validation guards**, not analytical parameters:

- **Feature clipping** in `private_company_proxies.py`: `.clip(-0.1, 0.1)` for equity_change_rate. This prevents absurd values from interpolation errors (a company's equity doesn't change by 10% daily). These are safety caps, not analytical thresholds.
- **PDF parser noise filtering**: `noise_count >= 3` in fuzzy_pdf_parser.py. Data cleaning, not financial analysis.
- **Filing discoverer token refresh**: `< 300` (5 min) for BMV token. API infrastructure.
- **LLM extraction scoring**: `number_count > 10`, `score += number_count * 0.1`. PDF content classification, not financial parameters.

---

## C-Actionable (24 findings, Should Be Adaptive)

### C1. Conflict Risk Event Score Breakpoints (6 findings in conflict_risk.py)

**Current State:**
```python
if events_30d < 10:
    event_score = events_30d / 20.0
elif events_30d < 50:
    event_score = 0.5 + (events_30d - 10) / 80.0

if news_mentions < 10:
    news_score = news_mentions / 20.0

if ratio > 1.5:  # escalation threshold
    trend = "escalating"
```

**Why adaptive**: The "significance" of 10 events or 50 events varies dramatically by country. Colombia might have 50 low-intensity events in a normal month; Japan having 1 event is alarming.

**Proposed Method:**

#### Method BD: Country-Relative Event Scoring (Gleditsch et al. 2002)

**Theory**: Score events relative to the country's baseline event rate. Use UCDP historical data to establish the country's normal event frequency, then score deviations.

**Formula**:
```
baseline_rate = historical_monthly_events(country, last_5_years)
z_events = (events_30d - baseline_rate) / max(sqrt(baseline_rate), 1)
event_score = sigmoid(z_events)
```

This is a Poisson Z-score: events are count data, and `sqrt(lambda)` is the standard deviation of a Poisson distribution.

For Colombia (baseline=40/month): 50 events -> z=(50-40)/sqrt(40) = 1.58 -> mild
For Japan (baseline=0/month): 1 event -> z=(1-0)/1 = 1.0 -> notable

**Simpler fallback** (when no historical data): Use log-scaling with population-adjusted rates.

**Academic basis**: Gleditsch et al. (2002) "Armed Conflict 1946-2001: A New Dataset." The UCDP/PRIO conflict intensity scale already uses per-capita normalization. Applying Poisson Z-scores is the standard for count data anomaly detection (Woodroofe 1982).

---

### C2. Regime Mixer Indicator Score Multipliers (4 findings in regime_mixer.py)

**Current State:**
```python
).astype(float) * 0.5   # stressed score multiplier (repeated 3x)
).astype(float) * 0.3   # cash_ratio stressed multiplier
```

**Why adaptive**: The 0.5 and 0.3 multipliers control the relative weight of "being in the stressed zone" vs "being in the distress zone." A value of 0.5 means the stressed signal counts half as much as distress. But this should depend on how predictive the stressed zone actually is of future outcomes.

**Proposed Method:**

#### Method BE: Discriminative Power Weighting (Engelmann et al. 2003)

**Theory**: Weight each indicator's contribution by its Accuracy Ratio (AR) -- how well it discriminates between companies that later entered distress and those that didn't. This is the Gini coefficient of the indicator's ROC curve.

**Formula**:
```
ar_i = 2 * AUC(indicator_i, future_distress) - 1
stressed_weight_i = ar_i * 0.5  # stressed gets AR-scaled weight
distress_weight_i = 1.0          # distress always gets full weight
```

**Data source**: Can be computed from the company's own historical data if sufficient distress episodes exist. Otherwise, use the Sobol first-order index for each indicator as a proxy for discriminative power.

**Academic basis**: Engelmann, Hayden & Tasche (2003) "Testing Rating Accuracy." The Accuracy Ratio is the standard credit risk metric for discriminative power. Used by Basel II/III for IRB model validation.

---

### C3. OHLC Predictor Noise Factors (4 findings in ohlc_predictor.py)

**Current State:**
```python
gap = rng.normal(0, volatility * 0.3) * prev_close          # open gap noise
intraday_range = abs(prev_close * volatility * 1.5)          # (covered in Tier 2)
predicted_high = body_high + abs(rng.normal(0, intraday_range * 0.5))  # high noise
predicted_low = body_low - abs(rng.normal(0, intraday_range * 0.5))    # low noise
noise_scale = sigma_daily * survival_mult * (1.0 + 0.02 * day_i)       # MC horizon scaling
```

**Proposed Method:**

#### Method BF: Parkinson (1980) + Yang-Zhang (2000) OHLC Noise Decomposition

**Theory**: The Parkinson estimator separates close-to-open (gap) variance from intraday (high-low) variance. Yang-Zhang further decomposes into overnight and trading-hour components.

**Formulas**:
```
sigma_overnight^2 = var(log(open_t / close_{t-1}))     # gap noise
sigma_intraday^2 = var(log(high_t / low_t)) / (4 * log(2))  # Parkinson intraday

gap_factor = sigma_overnight / sigma_close     # replaces 0.3
high_low_factor = sigma_intraday / sigma_close  # replaces 0.5
```

**Data source**: OHLC data already in the cache. Compute from the last 63 trading days.

**Academic basis**: Parkinson (1980, Journal of Business) "The Extreme Value Method for Estimating the Variance of the Rate of Return." Yang & Zhang (2000, Journal of Business) "Drift Independent Volatility Estimation."

---

### C4. Prediction Aggregator Confidence Score (3 findings)

**Current State:**
```python
base_spread = abs(point_forecast) * 0.10           # L627
noise_std = max(abs(base_val) * 0.02, 0.001)       # L1521
lower_ci = close_val * 0.99                         # L1640
```

**Proposed Method:**

#### Method BG: Bootstrap Prediction Interval (Efron 1979)

**Theory**: Instead of fixed percentage spreads, use the bootstrap distribution of the model ensemble predictions. The spread comes from the actual disagreement between models, not an assumed percentage.

**Formula**:
```
base_spread = percentile(model_predictions, 95) - percentile(model_predictions, 5)
noise_std = std(model_predictions) / sqrt(n_models)
```

**Data source**: `forecast_result.forecasts` contains per-model predictions. The spread between them IS the prediction uncertainty.

**Academic basis**: Efron (1979, Annals of Statistics) "Bootstrap Methods: Another Look at the Jackknife." The model ensemble spread is a nonparametric estimate of prediction uncertainty.

---

### C5. Conflict Intensity Composite Weights (4 findings in conflict_risk.py)

**Current State:**
The composite intensity is computed as:
```
intensity = 0.40 * event_score + 0.20 * fatality_score + 0.25 * flag_score + 0.15 * news_score
```

**Proposed Method:**

#### Method BH: Principal Component Weighting (Hotelling 1933)

**Theory**: The weights should reflect how much each component contributes to the overall variance of conflict intensity across countries. PCA's first component loadings give the variance-maximizing weights.

**Formula**:
```
X = [event_scores, fatality_scores, flag_scores, news_scores] across all countries
pca = PCA(n_components=1).fit(X)
weights = abs(pca.components_[0]) / sum(abs(pca.components_[0]))
```

**Fallback**: When only analyzing one country, use the current fixed weights (can't compute cross-country PCA from a single observation).

**Academic basis**: Hotelling (1933, Journal of Educational Psychology). PCA for composite index construction is standard in economics (e.g., the Human Development Index uses PCA-derived weights).

---

### C6. Monte Carlo Horizon Noise Scaling (3 findings)

**Current State:**
```python
noise_scale = sigma_daily * survival_mult * (1.0 + 0.02 * day_i)  # ohlc_predictor.py:225
```

The `0.02` is a fixed per-day noise growth factor.

**Proposed Method:**

#### Method BI: Hurst Exponent-Based Scaling (Hurst 1951)

**Theory**: The noise growth rate should depend on whether the series is mean-reverting (H < 0.5), random walk (H = 0.5), or trending (H > 0.5). For a random walk, noise scales as sqrt(t). For a trending series, noise scales as t^H.

**Formula**:
```
H = hurst_exponent(return_series, method="rs")
noise_growth = (1.0 + (2*H - 1.0) / 252) ^ day_i

# H=0.5 (random walk): growth = (1 + 0/252)^day = 1.0 (constant)
# H=0.7 (trending): growth = (1 + 0.4/252)^day = exponential
# H=0.3 (mean-reverting): growth = (1 - 0.4/252)^day = decaying
```

**Data source**: Return series in the cache. Hurst exponent computable via R/S analysis in ~5 lines of numpy.

**Academic basis**: Hurst (1951, Transactions ASCE) "Long-term Storage Capacity of Reservoirs." Mandelbrot & Wallis (1969) formalized R/S analysis. Lo (1991, Econometrica) "Long-term Memory in Stock Market Prices" introduced the modified R/S statistic.

---

## Summary: Selected Methods

| Group | Count | Method | Academic Basis |
|-------|-------|--------|----------------|
| C1 | 6 | BD: Poisson Z-score country-relative events | Gleditsch 2002, Woodroofe 1982 |
| C2 | 4 | BE: Accuracy Ratio discriminative power | Engelmann et al. 2003 |
| C3 | 4 | BF: Parkinson/Yang-Zhang OHLC decomposition | Parkinson 1980, Yang-Zhang 2000 |
| C4 | 3 | BG: Bootstrap prediction spread | Efron 1979 |
| C5 | 4 | BH: PCA composite weights | Hotelling 1933 |
| C6 | 3 | BI: Hurst exponent noise scaling | Hurst 1951, Lo 1991 |
| **Total actionable** | **24** | | |
| C-Structural | 22 | No action (definitions, not parameters) | |
| C-Trivial | 16 | No action (safety guards, not analysis) | |

### New Dependencies: None
All methods use numpy/scipy already installed.

### Academic References
- Gleditsch et al. (2002) -- UCDP conflict data
- Woodroofe (1982) -- Poisson anomaly detection
- Engelmann, Hayden & Tasche (2003) -- Accuracy Ratio for credit risk
- Parkinson (1980) -- Extreme value variance estimator
- Yang & Zhang (2000) -- OHLC volatility decomposition
- Efron (1979) -- Bootstrap methods
- Hotelling (1933) -- PCA for composite indices
- Hurst (1951) -- Long-term memory / R/S analysis
- Lo (1991) -- Modified R/S statistic
- Mandelbrot & Wallis (1969) -- R/S analysis formalization

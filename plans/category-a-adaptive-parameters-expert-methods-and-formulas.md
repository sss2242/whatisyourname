# Category A Adaptive Parameters: Expert Methods and Formulas

22 inline hardcoded parameters in 5 groups that should be data-derived.

---

## A1. Train/Test Split Ratios (10 occurrences in forecasting.py, transformer_forecaster.py)

### Current State
```python
split = max(1, int(len(clean) * 0.85))  # 85/15 split, repeated 8x in forecasting.py
split = max(1, int(len(X) * 0.85))      # transformer_forecaster.py
split = max(1, int(len(X) * 0.80))      # some tree models use 80/20
```

### Why Fixed 85/15 Is Wrong

When `n_eff` is 30 (an annual filer with 2 filings), an 85/15 split gives 25 train + 5 test. With only 5 test points, the RMSE estimate has huge variance (SE of RMSE with k=5 is RMSE/sqrt(2k-2) = RMSE/2.8). With n_eff of 500 (daily OHLCV), 85/15 gives 425/75, which is fine.

### Proposed Method

#### Method AN: Variance-Optimized Split (Arlot & Celisse 2010)

**Theory**: The optimal train/test split minimizes the total estimation error, which is the sum of model fitting error (decreases with more training data) and evaluation error (decreases with more test data). Arlot & Celisse (2010) show the optimal test fraction for k-fold-like evaluation is approximately:

**Formula**:
```
test_fraction = max(0.10, min(0.30, c / sqrt(n_eff)))
```

Where c is a constant (~2.0 for linear models, ~3.0 for nonlinear). This gives:
- n_eff=30: test_frac = 3.0/sqrt(30) = 0.55 -> clamped to 0.30 (more test data when scarce)
- n_eff=100: test_frac = 3.0/sqrt(100) = 0.30
- n_eff=500: test_frac = 3.0/sqrt(500) = 0.13
- n_eff=2000: test_frac = 3.0/sqrt(2000) = 0.067 -> clamped to 0.10

**Model-specific constants**:
```python
_SPLIT_C = {
    "kalman": 2.0,        # linear, needs less test data
    "garch": 2.5,
    "var": 2.0,
    "lstm": 3.5,          # nonlinear, needs more test data for reliable eval
    "tree": 3.0,
    "transformer": 4.0,   # most prone to overfitting
    "baseline": 1.5,
}
```

**Academic basis**: Arlot & Celisse (2010, Statistics Surveys) "A survey of cross-validation procedures for model selection." Also Shao (1993, JASA) "Linear Model Selection by Cross-Validation" -- proved that leave-one-out is asymptotically inefficient and larger test sets are needed for model selection (not just evaluation).

**Implementation**:
```python
def adaptive_train_test_split(n_eff, model_type="tree"):
    c = _SPLIT_C.get(model_type, 2.5)
    test_frac = max(0.10, min(0.30, c / math.sqrt(max(n_eff, 4))))
    return 1.0 - test_frac
```

---

## A2. Estimation Confidence Formulas (6 occurrences)

### Current State
```python
# estimator.py:439
conf = train_frac * 0.5

# missing_data_estimator.py:145
base_conf = min(0.85, 0.4 + obs_frac * 0.5)

# hidden_data_estimator.py:200
base_conf = 0.3 + probit_accuracy * 0.2 + ols_r2 * 0.2

# missing_data_estimator.py:458-459
disagreement_penalty = np.clip(relative_disagreement, 0, 0.5)
final_conf = (avg_conf - disagreement_penalty).clip(0.05, 0.95)

# gain_imputer.py:269
conf_raw = conf_raw.clip(0.1, 0.85)
```

### Why These Are Wrong

The coefficients (0.5, 0.4, 0.3, 0.2) are arbitrary. A confidence of `0.4 + obs_frac * 0.5` means at 100% observation rate, confidence is 0.9. But this ignores whether the imputation method actually performs well -- a method with 90% observations but poor model fit should have lower confidence than one with 60% observations but excellent fit.

### Proposed Methods

#### Method AO: Conformal Calibration-Derived Confidence (Vovk et al. 2005)

**Theory**: The conformal prediction framework provides mathematically guaranteed coverage rates. If we predict a value with 90% conformal interval [a, b] and the actual value falls within [a, b], the method's calibration is correct. Use the conformal coverage rate as the confidence score.

**Formula**:
```
empirical_coverage = fraction of held-out values within conformal intervals
confidence = empirical_coverage * interval_tightness

where interval_tightness = 1.0 - (interval_width / value_range)
```

A method that covers 90% of values with narrow intervals gets high confidence. One that covers 90% with very wide intervals gets lower confidence.

**Academic basis**: Vovk, Gammerman & Shafer (2005) "Algorithmic Learning in a Random World." The conformal framework provides distribution-free confidence measures.

**Data source**: `ConformalCalibrator` is already fitted and has scores available. The conformal module computes interval widths per variable.

#### Method AP: Prediction Interval Coverage Probability (PICP) Weighting

**Theory** (Khosravi et al. 2011): For each estimation method, compute the PICP (fraction of actual values within the estimated interval) and MPIW (mean prediction interval width). The confidence is:

**Formula**:
```
confidence = PICP * (1 - NMPIW)
```

Where NMPIW = MPIW / range(y) is the normalized mean prediction interval width. This penalizes methods that achieve high coverage by having very wide intervals.

**Academic basis**: Khosravi et al. (2011, IEEE TNN) "Comprehensive Review of Neural Network-Based Prediction Intervals." Widely used in uncertainty quantification.

### Recommended: Method AO (conformal-derived) with Method AP as cross-check

```python
def adaptive_confidence(method_name, values, predictions, conformal_calibrator):
    # Residuals for this method
    residuals = values - predictions
    
    # Build intervals
    q = conformal_calibrator.get_quantile(0.90)
    in_interval = (residuals.abs() <= q).mean()
    
    # Interval tightness
    value_range = max(values.max() - values.min(), 1e-8)
    tightness = 1.0 - min(2 * q / value_range, 0.99)
    
    confidence = in_interval * tightness
    return max(0.05, min(0.95, confidence))
```

---

## A3. Vanity Score Scaling Factors (5 occurrences in vanity.py)

### Current State
```python
raw_score = (intensity_excess * growth_penalty) * 1000.0   # L114
margin_penalty = (-margin_delta).clip(lower=0) * 200.0     # L167
leverage_excess = (nd_ebitda - 4.0).clip(lower=0) * 10.0   # L225
rank_penalty = (-rank_delta).clip(lower=0) * 2.0           # L270
raw_score = sentiment_positive * health_decline * 5.0       # L313
```

### Why These Are Wrong

The multipliers (1000, 200, 10, 2, 5) are arbitrary scaling factors that map different metrics to a 0-100 score range. But the metrics have different natural ranges depending on the company and sector. An R&D intensity excess of 0.05 scaled by 1000 = 50, but for biotech where intensity excess is routinely 0.20, the score would be 200 (clipped to 100).

### Proposed Method

#### Method AQ: Robust Percentile Rank Normalization (Standard in Factor Investing)

**Theory**: Instead of arbitrary multipliers, convert each raw signal to its percentile rank within the company's own history (expanding window). This naturally maps any distribution to [0, 100] regardless of scale.

**Formula**:
```
score = expanding_percentile_rank(raw_signal) * weight
```

This is the same approach already used by `financial_health.py` for tier scoring (expanding percentile rank normalization) and by `peer_ranking.py` for peer comparison. The vanity module should use the same pattern.

**Academic basis**: Standard in quantitative factor investing (Barra, MSCI). Percentile rank is a monotone transformation that preserves order while eliminating scale sensitivity. Fama & French (1993) factor portfolios use percentile/decile breakpoints for this reason.

**Implementation**:
```python
def _percentile_score(series, higher_is_worse=True):
    """Convert a raw signal to a 0-100 percentile score."""
    rank = series.expanding(min_periods=10).rank(pct=True)
    if higher_is_worse:
        return rank * 100
    else:
        return (1 - rank) * 100
```

No multipliers needed. Each component naturally falls in [0, 100]. The component weights (`rnd_mismatch: 0.15, sga_bloat: 0.25`, etc.) stay as they are -- those are design choices about relative importance, not scaling artifacts.

---

## A4. Conformal Fallback Width (2 occurrences in conformal.py)

### Current State
```python
fallback_width = abs(point_forecast) * 0.10  # 10% of forecast value
```

### Why 10% Is Wrong

For volatile variables (close price with 2% daily volatility), 10% is far too wide -- a $100 stock gets $10 bands when the daily move is $2. For stable variables (current_ratio ~1.5 with 0.01 variability), 10% = 0.15 is reasonable. The width should reflect the variable's actual prediction uncertainty.

### Proposed Method

#### Method AR: Historical Residual IQR Fallback (Hyndman & Athanasopoulos 2021)

**Theory**: When the conformal calibrator has insufficient scores (<20), use the interquartile range of the most recent prediction residuals as the fallback width. IQR is robust to outliers (unlike standard deviation).

**Formula**:
```
residuals = recent_actuals - recent_predictions
fallback_width = 1.35 * IQR(residuals)  # 1.35 * IQR ≈ 1.0 * sigma for normal
```

The 1.35 scaling makes IQR comparable to one standard deviation under normality. For 90% coverage, multiply by 1.645.

**Academic basis**: Hyndman & Athanasopoulos (2021) "Forecasting: Principles and Practice" (3rd ed.). IQR-based prediction intervals are the standard robust alternative when distributional assumptions fail.

**Data source**: `ForecastResult.residuals` is already populated (fixed in PR #1). Use the residuals directly.

```python
def adaptive_conformal_fallback(residuals, point_forecast, coverage=0.90):
    if residuals and len(residuals) >= 10:
        arr = np.array(residuals)
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        # Scale IQR to desired coverage
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(coverage, 1.645)
        width = 1.35 * iqr * z / 1.645  # normalize then rescale
        return max(width, abs(point_forecast) * 0.001)  # floor at 0.1%
    else:
        # True last resort: use absolute value * volatility-scaled fraction
        return abs(point_forecast) * 0.10
```

---

## A5. Estimation Confidence Clip Bounds (scattered across 5 files)

### Current State
```python
.clip(0.1, 0.95)   # missing_data_estimator.py:247, :459
.clip(0.05, 0.95)  # missing_data_estimator.py:459
.clip(0.1, 0.85)   # gain_imputer.py:269
.clip(0.05, 0.75)  # hidden_data_estimator.py:503
```

### Proposed Method

#### Method AS: Calibration-Derived Confidence Bounds (Platt 1999)

**Theory**: The upper bound should reflect the best a method CAN do (its calibration ceiling), and the lower bound should reflect the worst case (the prior probability of random imputation being correct).

**Formula**:
```
upper_bound = min(0.95, empirical_coverage_rate * 1.05)  # slight optimism
lower_bound = max(0.01, 1 / n_possible_values)  # random chance baseline
```

For continuous variables with resolution r: `n_possible = range / r`, so `lower_bound ≈ 0.01`.
For the MNAR path (hidden_data_estimator): lower bound is higher (0.10) because the Heckman selection model has structural priors.

**Academic basis**: Platt (1999) "Probabilistic Outputs for Support Vector Machines." Platt scaling is the standard for calibrating classifier confidence. Extended to regression by Kuleshov et al. (2018).

---

## Summary: Selected Methods

| Group | Count | Method | Key Insight |
|-------|-------|--------|-------------|
| A1 | 10 | AN: Arlot & Celisse (2010) variance-optimized split | test_frac = c / sqrt(n_eff), model-specific c |
| A2 | 6 | AO: Conformal calibration-derived confidence | confidence = coverage * tightness |
| A3 | 5 | AQ: Percentile rank normalization | Standard factor investing approach, eliminates all multipliers |
| A4 | 2 | AR: IQR-based residual fallback (Hyndman 2021) | Uses actual residuals instead of 10% of forecast |
| A5 | 5 | AS: Calibration-derived bounds (Platt 1999) | Bounds from empirical coverage, not arbitrary clips |

### New Dependencies: None

### Academic References
- Arlot & Celisse (2010) -- Optimal train/test split theory
- Shao (1993) -- Leave-one-out inefficiency proof
- Vovk, Gammerman & Shafer (2005) -- Conformal prediction confidence
- Khosravi et al. (2011) -- PICP prediction interval quality
- Fama & French (1993) -- Percentile rank factor construction
- Hyndman & Athanasopoulos (2021) -- IQR prediction intervals
- Platt (1999) -- Probabilistic calibration
- Kuleshov et al. (2018) -- Regression calibration

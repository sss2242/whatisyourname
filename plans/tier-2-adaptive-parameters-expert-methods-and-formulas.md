# Tier 2 Adaptive Parameters: Expert Methods and Formulas

Replacing 35 hardcoded model parameters across 10 files with data-derived values.

---

## 5. Model Minimum Observation Counts

### Current State

| File | Constant | Value | Used For |
|------|----------|-------|----------|
| `forecasting.py:63` | `_MIN_OBS_KALMAN` | 30 | Kalman filter |
| `forecasting.py:64` | `_MIN_OBS_GARCH` | 60 | GARCH volatility |
| `forecasting.py:65` | `_MIN_OBS_VAR` | 50 | Vector autoregression |
| `forecasting.py:66` | `_MIN_OBS_LSTM` | 100 | LSTM neural network |
| `forecasting.py:67` | `_MIN_OBS_TREE` | 30 | Tree ensemble |
| `regime_detector.py:65` | `_MIN_OBS_HMM` | 60 | Hidden Markov Model |
| `regime_detector.py:66` | `_MIN_OBS_GMM` | 30 | Gaussian Mixture Model |
| `regime_detector.py:67` | `_MIN_OBS_PELT` | 30 | PELT changepoint detection |
| `transformer_forecaster.py:36` | `_MIN_OBS_TRANSFORMER` | 100 | Transformer model |
| `estimator.py:110` | `_MIN_OBSERVED_FOR_MODEL` | 10 | Estimation engine |

### Why These Are Wrong

A company filing semi-annually has 4 distinct data points over 2 years, interpolated to ~500 daily rows. The 500 "observations" are largely synthetic -- 496 are interpolated, 4 are real. Using 30+ "observations" from 2 real filings to fit a Kalman filter gives false confidence.

### Proposed Methods

#### Method L: Effective Degrees of Freedom (EDoF)

**Theory**: The actual information content of a time series depends on the number of independent observations, not the row count. For interpolated financial data, EDoF is bounded by the number of original filings times the number of independent variables per filing.

**Formula** (Satterthwaite 1946):

```
EDoF = n_unique_filings * sqrt(n_variables_observed_per_filing)
```

For a company with 8 quarterly filings and 15 observed variables each:
```
EDoF = 8 * sqrt(15) = 8 * 3.87 = 31
```

For a company with 2 annual filings and 8 variables:
```
EDoF = 2 * sqrt(8) = 2 * 2.83 = 5.66
```

**Adaptive minimum**:
```python
def adaptive_min_obs(model_type, cache, metric):
    # Count unique filings (where interpolated values transition)
    n_filings = count_unique_filings(cache, metric)
    n_vars = count_observed_variables(cache)
    edof = n_filings * math.sqrt(n_vars)
    
    # Model-specific multiplier (more complex models need more data)
    multipliers = {
        'kalman': 3,      # needs ~3x EDoF for stable state estimation
        'garch': 5,       # needs ~5x for stable conditional variance
        'var': 4,         # needs ~4x per included variable
        'lstm': 8,        # neural nets need ~8x for generalization
        'tree': 3,        # trees are more sample-efficient
        'hmm': 5,         # HMM needs ~5x per regime
        'transformer': 10, # transformers need more data
        'estimator': 2,   # estimation needs ~2x for basic inference
    }
    
    required = edof * multipliers.get(model_type, 3)
    
    # Floor: never below model's theoretical minimum
    floors = {
        'kalman': 10, 'garch': 20, 'var': 15, 'lstm': 30,
        'tree': 10, 'hmm': 20, 'transformer': 40, 'estimator': 5,
    }
    
    return max(int(required), floors.get(model_type, 10))
```

**Academic basis**: Satterthwaite (1946) effective degrees of freedom approximation. Extended by Hastie, Tibshirani & Friedman (2009, "Elements of Statistical Learning" Ch. 7) for model complexity vs sample size trade-offs.

#### Method M: Interpolation-Confidence-Weighted Sample Count

**Theory**: Weight each daily observation by the interpolation confidence from `frequency_interpolator.py`. Days near a filing have confidence ~0.95, days far from any filing have confidence ~0.3. The effective sample size is the sum of confidence weights.

**Formula** (Kish 1965):

```
n_effective = (sum(w_i))^2 / sum(w_i^2)
```

Where `w_i` is the interpolation confidence for day i.

**Academic basis**: Kish (1965) "Survey Sampling" effective sample size. Standard in survey statistics for weighted samples. Directly applicable here because interpolation confidence IS the weighting of information content per observation.

```python
def confidence_weighted_n_eff(cache, metric):
    conf_col = f"interp_confidence_{metric}"
    if conf_col not in cache.columns:
        return len(cache)  # no confidence data -> assume all real
    
    w = cache[conf_col].fillna(0.5).values
    sum_w = np.sum(w)
    sum_w2 = np.sum(w ** 2)
    
    if sum_w2 < 1e-10:
        return 0
    
    return (sum_w ** 2) / sum_w2
```

### Recommended: Method M (Kish n_eff) with Method L floors

Use confidence-weighted effective sample size when interpolation confidence data is available (it is, from Step 4). Fall back to EDoF estimation when confidence columns are missing.

---

## 6. Cox/Sigmoid Blend Weight

### Current State
```python
# main.py:1247
cache["survival_probability"] = 0.4 * _sig + 0.6 * _cox
```

### Proposed Methods

#### Method N: Inverse-Variance Weighting (Cochrane 1954)

**Theory**: Weight each model by the inverse of its prediction variance. The model with lower variance (more precise) gets higher weight. This is the minimum-variance unbiased estimator.

**Formula**:
```
w_sig = (1/var_sig) / (1/var_sig + 1/var_cox)
w_cox = (1/var_cox) / (1/var_sig + 1/var_cox)
blend = w_sig * sig + w_cox * cox
```

Where `var_sig` and `var_cox` are estimated from the rolling prediction variance of each model on the training window.

**Academic basis**: Cochrane (1954) "The combination of estimates from different experiments." Foundation of meta-analysis. Used by the prediction aggregator already for ensemble weighting (inverse-RMSE) -- this extends the same principle to the survival blend.

#### Method O: Bayesian Model Averaging (BMA)

**Theory**: Compute posterior model probabilities using the marginal likelihood of each model given the data. The BIC approximation avoids expensive integration:

**Formula** (Schwarz 1978):
```
BIC_model = -2 * log_likelihood + k * log(n)
w_model = exp(-0.5 * delta_BIC) / sum_j(exp(-0.5 * delta_BIC_j))
```

Where `delta_BIC = BIC_model - BIC_best`.

**Academic basis**: Hoeting et al. (1999, Statistical Science) "Bayesian Model Averaging: A Tutorial." Raftery (1995) "Bayesian Model Selection."

**Pros**: Theoretically optimal model combination. Accounts for model complexity.
**Cons**: Requires computing log-likelihood for both models. Cox PH has a well-defined partial likelihood; sigmoid survival is harder.

### Recommended: Method N (inverse-variance weighting)

Simpler, robust, and uses the same principle already applied elsewhere in the pipeline. Compute rolling variance of each model's predicted survival probability against actual outcomes over the training window.

```python
def adaptive_blend_weights(sig_series, cox_series, cache, lookback=126):
    # Use the recent survival mode flags as "ground truth"
    actual = cache.get("company_survival_mode_flag", pd.Series(0, index=cache.index))
    
    # Rolling prediction variance for each model
    sig_var = (sig_series - actual).rolling(lookback).var().iloc[-1]
    cox_var = (cox_series - actual).rolling(lookback).var().iloc[-1]
    
    if sig_var < 1e-10 and cox_var < 1e-10:
        return 0.5, 0.5  # both perfect, equal weight
    
    w_sig = (1 / max(sig_var, 1e-10))
    w_cox = (1 / max(cox_var, 1e-10))
    total = w_sig + w_cox
    
    return w_sig / total, w_cox / total
```

---

## 7. PID Controller Gains

### Current State
```python
DEFAULT_KP: float = 0.5    # proportional
DEFAULT_KI: float = 0.1    # integral
DEFAULT_KD: float = 0.2    # derivative
```

### Proposed Methods

#### Method P: Ziegler-Nichols (1942) Auto-Tuning

**Theory**: The classic PID tuning method. Increase Kp until the system oscillates (find the "ultimate gain" Ku and "ultimate period" Tu), then set gains:

| Controller | Kp | Ki | Kd |
|-----------|-----|-----|-----|
| PID | 0.6*Ku | 1.2*Ku/Tu | 0.075*Ku*Tu |

**Implementation**: During the warmup phase of the forward pass, record the prediction error series. Find the frequency of the dominant oscillation using FFT. Ku is the gain at which the error oscillation neither grows nor decays.

**Academic basis**: Ziegler & Nichols (1942) "Optimum Settings for Automatic Controllers." Still the most widely used method after 80+ years.

#### Method Q: Lambda Tuning (Dahlin 1968)

**Theory**: Instead of oscillation-based tuning, specify a desired closed-loop time constant lambda (how fast the controller should respond). For financial time series, lambda should be proportional to the data's autocorrelation half-life.

**Formula**:
```
tau = autocorrelation_halflife(error_series)
lambda_target = 2 * tau  # desired response time = 2x natural decay

Kp = tau / (lambda_target * K_process)
Ki = Kp / tau
Kd = Kp * tau / 4
```

Where `K_process` is estimated from the steady-state error magnitude.

**Academic basis**: Dahlin (1968) "Designing and tuning digital controllers." Preferred over Ziegler-Nichols for processes that should not oscillate (financial time series definitely should not).

#### Method R: ITAE Optimal (Graham & Lathrop 1953)

**Theory**: Minimize the Integral of Time-weighted Absolute Error. This penalizes long-lasting errors more than brief spikes -- exactly what we want for financial prediction (a model that's persistently wrong is worse than one that occasionally spikes).

**Formula** (for a first-order-plus-dead-time model):
```
Kp = 0.965 / (K_process) * (theta/tau)^(-0.855)
Ki = Kp / (tau * 0.796 * (theta/tau)^(-0.147))
Kd = Kp * tau * 0.308 * (theta/tau)^0.929
```

Where theta is the dead time (filing lag) and tau is the process time constant.

**Academic basis**: Graham & Lathrop (1953). Tabulated in O'Dwyer (2009) "Handbook of PI and PID Controller Tuning Rules." Used in process control when oscillation is unacceptable.

### Recommended: Method Q (Lambda tuning) with per-variable autocorrelation

Lambda tuning is the best fit because:
1. It explicitly avoids oscillation (critical for financial systems)
2. The autocorrelation half-life is trivially computable from the cache
3. Different variables (close price vs current_ratio) have very different dynamics

```python
def compute_pid_gains(error_series, variable_name):
    # Compute autocorrelation half-life
    acf = error_series.autocorr(lag=1)
    if abs(acf) < 0.01:
        tau = 1.0  # no autocorrelation, fast response
    else:
        tau = -1.0 / math.log(abs(acf))  # half-life in days
    
    lambda_target = max(2.0 * tau, 3.0)  # at least 3 days response time
    k_process = max(abs(error_series.mean()), 0.01)
    
    kp = min(tau / (lambda_target * k_process), 2.0)
    ki = min(kp / tau, 0.5) if tau > 0 else 0.1
    kd = min(kp * tau / 4.0, 1.0)
    
    return kp, ki, kd
```

---

## 8. Contagion and Graph Risk Probabilities

### Current State
```python
DEFAULT_CONTAGION_PROB: float = 0.3
DEFAULT_PARTICIPATION_RATE: float = 0.25
CROWDING_THRESHOLD: float = 0.6
```

### Proposed Methods

#### Method S: Copula-Derived Edge Contagion (Joe 2014)

**Theory**: The contagion probability between two entities should reflect their actual statistical dependence, not a flat 0.3. The copula module already computes tail dependence -- use the lower tail dependence coefficient lambda_L as the contagion probability.

**Formula**:
```
contagion_prob(A, B) = lambda_L(copula(returns_A, returns_B))
```

For a Student-t copula with nu degrees of freedom and correlation rho:
```
lambda_L = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
```

**Data source**: The copula module (`run_copula_analysis`) already fits copulas and computes tail dependence. Currently only used for uncertainty band widening -- this reuses it for contagion.

**Academic basis**: Joe (2014) "Dependence Modeling with Copulas." The tail dependence coefficient is the theoretically correct measure of joint crash probability.

#### Method T: Amihud-Derived Participation Rate (Amihud 2002)

**Theory**: The participation rate (what fraction of daily volume a large holder can trade without moving the market) depends on the stock's liquidity, measured by the Amihud illiquidity ratio.

**Formula**:
```
amihud = mean(|return_i| / dollar_volume_i)
participation_rate = 1 / (1 + 10 * amihud)
```

Highly liquid stocks (low Amihud): participation_rate -> 0.5 (50% of volume)
Illiquid stocks (high Amihud): participation_rate -> 0.05 (5% of volume)

**Academic basis**: Amihud (2002, JFM) "Illiquidity and Stock Returns." The Amihud ratio is already computed in `six_derived_proxies.py` and `institutional_flow.py`. The `ownership_contagion.py` module already references it but doesn't use it to set the participation rate.

#### Method U: Empirical Crowding Threshold (Khandani & Lo 2011)

**Theory**: Instead of a fixed 0.6 crowding threshold, compute it as the percentile of the historical crowding score distribution where drawdowns significantly increase.

**Formula**:
```
# For each historical day, compute crowding_score and forward_drawdown
# Find the crowding_score percentile that maximizes the Youden index
# (sensitivity + specificity - 1) for predicting drawdown > 2*sigma

threshold = optimal_youden_cutoff(crowding_scores, drawdowns > 2 * sigma)
```

**Academic basis**: Youden (1950, Cancer) index for optimal binary classifier threshold. Applied to financial crowding by Khandani & Lo (2011).

### Recommended: Method S (copula tail dependence) for contagion, Method T (Amihud) for participation rate

Both methods reuse already-computed data in the pipeline. No new computations needed.

---

## 9. Prediction Aggregator Constants

### Current State

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_SURVIVAL_RISK_MULTIPLIER` | 2.0 | Band widening during distress |
| `INTRADAY_LOW_FACTOR` | 1.5 | Low estimation |
| `_TRANSITION_BLEND_HALFLIFE` | 5 days | Regime transition smoothing |
| `DEFAULT_VOLATILITY` | 0.02 | Fallback when no vol data |

### Proposed Methods

#### Method V: Regime-Conditional Volatility Multiplier

**Theory**: The survival risk multiplier should scale with how much more volatile the company becomes during distress. Compute the ratio of distress-regime volatility to normal-regime volatility from the HMM.

**Formula**:
```
risk_mult = sigma_distress / sigma_normal
```

If the HMM found that volatility triples during bear regimes (sigma_bear = 0.06 vs sigma_normal = 0.02), the risk multiplier should be 3.0, not the fixed 2.0.

**Data source**: HMM emission parameters from `regime_detector.result`.

#### Method W: Garman-Klass Intraday Range Estimator (1980)

**Theory**: The intraday low factor should be derived from the stock's actual intraday range, not a fixed 1.5. The Garman-Klass estimator uses open/high/low/close to estimate true volatility more efficiently than close-to-close.

**Formula**:
```
sigma_GK = sqrt(0.5 * (log(H/L))^2 - (2*log(2) - 1) * (log(C/O))^2)
intraday_low_factor = sigma_GK / sigma_close * 1.645  # 90% CI
```

**Academic basis**: Garman & Klass (1980, Journal of Business) "On the Estimation of Security Price Volatilities from Historical Data." More efficient than close-to-close (5x lower variance).

#### Method X: Regime Duration Half-Life

**Theory**: The transition blend half-life should match the average speed at which regime transitions complete. If the company typically takes 10 days to transition between regimes, the half-life should be 5 (half of the transition period).

**Formula**:
```
transition_half_life = median(transition_durations) / 2
```

Where `transition_durations` comes from the enriched survival timeline (days_in_mode at each switch_point).

**Data source**: `enriched_timeline_result.switch_points` already tracks mode transitions.

### Recommended: All three methods (V, W, X)

Each directly uses data already available in the pipeline:
- V uses HMM emission parameters
- W uses OHLC data in the cache
- X uses switch points from the survival timeline

---

## 10. Monte Carlo Simulation Parameters

### Current State
```python
DEFAULT_N_PATHS: int = 10_000
DEFAULT_IS_TILT: float = 1.5
_MIN_OBS_PER_REGIME: int = 10
DEFAULT_N_BOOTSTRAP: int = 1000
_TRANSITION_SMOOTHING: float = 1.0
```

### Proposed Methods

#### Method Y: Precision-Targeted Path Count (Glasserman 2003)

**Theory**: The number of MC paths should achieve a target precision for the survival probability estimate. For a proportion p estimated from n paths, the standard error is sqrt(p*(1-p)/n). For 1% standard error on a 95% survival probability:

**Formula**:
```
n_paths = p * (1 - p) / (target_se)^2

# For p=0.95, target_se=0.01:
n_paths = 0.95 * 0.05 / 0.0001 = 475

# For p=0.50, target_se=0.01:
n_paths = 0.50 * 0.50 / 0.0001 = 2,500
```

**Adaptive formula**:
```python
def adaptive_n_paths(preliminary_survival_prob, target_se=0.005):
    p = max(0.01, min(0.99, preliminary_survival_prob))
    n = p * (1 - p) / (target_se ** 2)
    return max(1000, min(50000, int(n)))
```

Run a quick preliminary MC with 500 paths to estimate p, then scale up.

**Academic basis**: Glasserman (2003) "Monte Carlo Methods in Financial Engineering" Ch. 2. Standard sample size calculation for proportion estimation.

#### Method Z: Adaptive Importance Sampling Tilt (Bucklew 2004)

**Theory**: The IS tilt should be proportional to the rarity of the event being estimated. For very safe companies (survival probability > 99%), you need aggressive tilting to get enough tail samples. For distressed companies (survival probability ~50%), minimal tilting is needed.

**Formula** (exponential tilting):
```
optimal_tilt = -log(target_event_probability) / sigma

# For p_event = 0.05 (5% of paths breach survival): tilt = 3.0 / sigma
# For p_event = 0.30 (30% breach): tilt = 1.2 / sigma
```

**Academic basis**: Bucklew (2004) "Introduction to Rare Event Simulation." Siegmund (1976) importance sampling for rare events.

#### Method AA: Minimum Observations Per Regime via Anderson-Darling

**Theory**: The minimum observations per regime should ensure the fitted Gaussian passes a normality test (since the MC assumes Gaussian regime distributions). Use the Anderson-Darling critical value.

**Formula**:
```
min_obs = max(8, 5 + 2 * n_params)  # heuristic: 5 + 2 per distribution parameter
```

For a Gaussian with 2 parameters (mean, std): min_obs = 9.
For a Student-t with 3 parameters: min_obs = 11.

The fixed 10 is actually close to optimal for Gaussian, but should scale with the distribution family used.

### Recommended: Method Y (precision-targeted paths) + Method Z (adaptive tilt)

Both are cheap to compute and directly improve simulation quality:

```python
def adaptive_mc_params(cache, preliminary_survival=None):
    # Estimate preliminary survival from sigmoid
    if preliminary_survival is None:
        surv = cache.get("survival_probability")
        preliminary_survival = float(surv.iloc[-1]) if surv is not None else 0.9
    
    # Precision-targeted path count
    p = max(0.01, min(0.99, preliminary_survival))
    target_se = 0.005  # 0.5% standard error
    n_paths = int(p * (1 - p) / (target_se ** 2))
    n_paths = max(1000, min(50000, n_paths))
    
    # Adaptive IS tilt
    p_tail = 1 - preliminary_survival
    sigma = cache.get("volatility_21d", pd.Series(0.02)).dropna().iloc[-1] if "volatility_21d" in cache.columns else 0.02
    if p_tail > 0.01:
        tilt = min(-math.log(p_tail) / max(sigma * math.sqrt(252), 0.01), 5.0)
    else:
        tilt = 3.0  # aggressive tilt for very safe companies
    
    return n_paths, tilt
```

---

## Summary: Selected Methods Per Constant Group

| # | Constant Group | Primary Method | Data Source | New Deps |
|---|---------------|---------------|-------------|----------|
| 5 | Min observations | M: Kish effective sample size from interp confidence | `interp_confidence_*` columns | None |
| 6 | Cox/sigmoid blend | N: Inverse-variance weighting (Cochrane 1954) | Rolling prediction variance vs actual | None |
| 7 | PID gains | Q: Lambda tuning (Dahlin 1968) from autocorrelation half-life | Error series autocorrelation | None |
| 8a | Contagion prob | S: Copula tail dependence (Joe 2014) | `copula_result.tail_dependence` | None |
| 8b | Participation rate | T: Amihud illiquidity ratio | Already computed in cache | None |
| 8c | Crowding threshold | U: Youden optimal cutoff | Historical crowding vs drawdowns | None |
| 9a | Survival risk mult | V: sigma_distress / sigma_normal from HMM | `regime_detector.result` | None |
| 9b | Intraday low factor | W: Garman-Klass estimator | OHLC in cache | None |
| 9c | Transition half-life | X: Median transition duration / 2 | `enriched_timeline_result` | None |
| 9d | Default volatility | Always use actual -- never fallback | Cache `volatility_21d` | None |
| 10a | MC n_paths | Y: Precision-targeted (Glasserman 2003) | Preliminary survival probability | None |
| 10b | MC IS tilt | Z: Adaptive exponential tilting (Bucklew 2004) | Tail probability + volatility | None |
| 10c | Min obs per regime | AA: 5 + 2 * n_distribution_params | Distribution family | None |

### New Dependencies: None
All methods use data already computed by the pipeline. No new packages needed.

### Academic References
- Satterthwaite (1946) -- Effective degrees of freedom
- Kish (1965) -- Effective sample size for weighted data
- Cochrane (1954) -- Inverse-variance combination
- Hoeting et al. (1999) -- Bayesian Model Averaging
- Ziegler & Nichols (1942) -- Classic PID tuning
- Dahlin (1968) -- Lambda tuning for non-oscillating systems
- Graham & Lathrop (1953) -- ITAE optimal PID
- Joe (2014) -- Copula tail dependence
- Amihud (2002) -- Illiquidity ratio
- Khandani & Lo (2011) -- Crowded trades
- Garman & Klass (1980) -- Intraday volatility estimation
- Glasserman (2003) -- Monte Carlo precision targeting
- Bucklew (2004) -- Importance sampling for rare events

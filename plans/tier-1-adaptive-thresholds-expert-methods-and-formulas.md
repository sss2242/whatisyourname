# Tier 1 Adaptive Thresholds: Expert Methods and Formulas

Replacing 18 hardcoded constants across 4 areas with data-derived, sector-aware formulas.

---

## 1. Survival Mode Thresholds

### Current State
Three files duplicate the same fixed thresholds:
- `survival_mode.py:46-51` -- `_COMPANY_THRESHOLDS`
- `monte_carlo.py:66-71` -- `DEFAULT_SURVIVAL_THRESHOLDS`
- `regime_mixer.py:29-42` -- `_FUND_THRESHOLDS`

```python
current_ratio_lt: 1.0
debt_to_equity_abs_gt: 3.0
fcf_yield_lt: 0.0
drawdown_252d_lt: -0.40
```

### Why These Are Wrong as Fixed Values

- **current_ratio = 1.0**: Software companies routinely operate at 0.7-0.9 (negative working capital model -- think Apple, Microsoft). Banks operate at 0.05-0.15 (fractional reserves). A fixed 1.0 permanently flags healthy tech and finance companies as distressed.
- **debt_to_equity = 3.0**: Utilities average 1.5-2.5 (capital intensive, regulated returns). REITs average 2.0-4.0 (leverage is the business model). Banks average 8-15x (deposits are liabilities). A fixed 3.0 misses distressed industrials and false-flags normal financials.
- **fcf_yield = 0.0**: Growth companies (biotech, early-stage tech) burn cash for years. A negative FCF yield is structural, not distress. Meanwhile, a mature utility with FCF yield of 0.1% is effectively in trouble despite being "above zero."
- **drawdown = -40%**: Crypto-adjacent and small-cap stocks regularly draw down 40%+ in normal corrections. Large-cap defensives drawing down 40% is genuinely unprecedented.

### Proposed Methods (4 approaches, pick best for each metric)

#### Method A: Conditional Percentile Thresholds (Peer-Calibrated)

**Theory**: The threshold should be the boundary where the company transitions from "normal for its sector" to "outlier distress." This is precisely the empirical quantile of the peer distribution.

**Formula**:
```
threshold(metric, sector) = Q_alpha(peer_distribution(metric, sector))
```

Where `Q_alpha` is the alpha-quantile:
- For "lower is worse" metrics (current_ratio, fcf_yield): `alpha = 0.10` (10th percentile)
- For "higher is worse" metrics (debt_to_equity): `alpha = 0.90` (90th percentile)
- For drawdown: `alpha = 0.05` (5th percentile of own historical distribution)

**Data source**: `linked_caches` from entity discovery (competitor caches contain current_ratio, D/E, etc.). Already computed and available at Step 5h.

**Fallback**: When fewer than 5 peers are available, use the Winsorized mean +/- 2 MAD (Median Absolute Deviation) as a robust alternative:
```
threshold = median(peer_values) - 2 * MAD(peer_values)
```

**Academic basis**: Robust statistics (Huber 1981, Hampel et al. 1986). MAD is the most breakdown-resistant measure of scale (50% breakdown point vs 0% for standard deviation).

**Pros**: Simple, interpretable, adapts to sector norms.
**Cons**: Requires peer data. Peer group may be too small for niche companies.

#### Method B: Merton Distance-to-Default (For debt_to_equity and current_ratio)

**Theory**: Instead of a fixed threshold, compute the Merton (1974) distance-to-default (DD) -- the number of standard deviations the company's asset value is from the default barrier. DD < 1.5 is typically considered distressed.

**Formula**:
```
DD = (ln(V/D) + (mu - 0.5*sigma^2)*T) / (sigma * sqrt(T))
```

Where:
- V = market cap + total debt (proxy for asset value)
- D = total debt (default barrier)
- mu = expected asset return (from CAPM or historical)
- sigma = asset volatility (from equity volatility via Merton's relation)
- T = time horizon (1 year)

**Conversion to adaptive threshold**:
```
distress_flag = (DD < DD_critical)
DD_critical = max(1.5, Q_10(sector_DD_distribution))
```

**Academic basis**: Merton (1974) structural credit risk model. Used by Moody's KMV (now part of Moody's Analytics) for Expected Default Frequency. The `six_derived_proxies.py` module already implements a Merton model (line 996) for Swiss companies -- this would generalize it.

**Pros**: Theoretically grounded, captures the joint effect of leverage and volatility.
**Cons**: Requires market cap (not available for private companies). Assumes lognormal asset dynamics.

#### Method C: Isolation Forest Anomaly Score (Unsupervised)

**Theory**: Instead of threshold-per-variable, treat the 4-dimensional point (current_ratio, D/E, fcf_yield, drawdown) as a single observation and detect anomalies using Isolation Forest (Liu et al. 2008).

**Formula**:
```
anomaly_score = IsolationForest(contamination=0.10).decision_function(X)
distress_flag = anomaly_score < 0
```

Where X is the matrix of all 4 metrics for the company over time plus its peers.

**Academic basis**: Liu, Ting & Zhou (2008) "Isolation Forest." Used extensively in fraud detection and financial anomaly detection. Advantage: captures non-linear interactions between variables (e.g., high D/E is fine WITH high current ratio, but not with low current ratio).

**Pros**: Captures multi-variable interactions. No per-variable threshold needed.
**Cons**: Less interpretable than per-variable thresholds. Requires sklearn (already installed).

#### Method D: Bayesian Online Changepoint Detection (BOCPD)

**Theory**: Instead of comparing against a static threshold, detect when a variable has undergone a structural shift downward. A company with current_ratio declining from 2.0 to 1.2 is in distress even though 1.2 > 1.0 (the fixed threshold). BOCPD detects the regime change.

**Formula**: Adams & MacKay (2007):
```
P(r_t | x_{1:t}) = sum over r_{t-1} of [P(x_t | r_t) * P(r_t | r_{t-1}) * P(r_{t-1} | x_{1:t-1})]
```

Where r_t is the run length (time since last changepoint). The `regime_detector.py` module already implements BCP (Bayesian Change Point) detection -- this would extend it to financial ratios, not just returns.

**Pros**: Captures deterioration dynamics, not just absolute levels. Works for any sector.
**Cons**: More complex. Already partially implemented in the pipeline.

### Recommended Approach: Hybrid (A + D)

Use **Method A** (peer percentile) as the primary threshold, then apply **Method D** (BOCPD) as a secondary trigger for deterioration detection. This catches both "worse than peers" and "deteriorating from own baseline":

```python
def compute_adaptive_threshold(metric, cache, linked_caches, sector):
    # Primary: peer-calibrated percentile
    peer_values = collect_peer_latest(metric, linked_caches)
    if len(peer_values) >= 5:
        if metric in LOWER_IS_WORSE:
            threshold = np.percentile(peer_values, 10)
        else:
            threshold = np.percentile(peer_values, 90)
    else:
        # Fallback: Winsorized mean +/- 2*MAD
        threshold = np.median(peer_values) - 2 * median_abs_deviation(peer_values)
    
    # Secondary: own-history deterioration via BOCPD
    own_series = cache[metric].dropna()
    changepoints = detect_changepoints_bcp(own_series)
    if changepoints and most_recent_is_downward(changepoints):
        # Tighten threshold by 20% when company is deteriorating
        if metric in LOWER_IS_WORSE:
            threshold *= 1.20  # raise the bar
        else:
            threshold *= 0.80  # lower the bar
    
    # Floor: never less protective than textbook minimums
    threshold = apply_minimum_floor(metric, threshold)
    return threshold
```

**Minimum floors** (absolute lower bounds regardless of peer data):
- current_ratio: never below 0.3 (even for banks)
- debt_to_equity: never above 20.0 (even for REITs)
- fcf_yield: always at least -0.10 (allow some burn)
- drawdown: never above -0.15 (always flag 15%+ drops)

---

## 2. Vanity Score Thresholds

### Current State
```python
rnd_threshold = 0.10          # R&D intensity > 10%
sga_ratio threshold = 0.30    # SGA > 30% of revenue
exec_comp_threshold = 5.0%    # exec comp > 5% of net income
marketing_threshold = 10.0%   # marketing > 10% of revenue
```

### Proposed Methods

#### Method E: Sector-Relative Z-Score

**Theory**: Express each metric as standard deviations from the sector median. A Z-score > 2.0 means the company is an outlier in its sector.

**Formula**:
```
z_metric = (company_value - sector_median) / sector_MAD
vanity_trigger = z_metric > z_critical
```

Where:
- `sector_median` and `sector_MAD` are computed from peer caches
- `z_critical = 2.0` (standard outlier threshold, adjustable)

**Academic basis**: Modified Z-score using MAD instead of standard deviation for robustness against outliers (Iglewicz & Hoaglin 1993). The "1.4826 * MAD" scaling makes it comparable to standard deviation under normality.

```python
def sector_relative_threshold(metric, company_value, peer_values):
    if len(peer_values) < 3:
        return STATIC_FALLBACK[metric]
    med = np.median(peer_values)
    mad = 1.4826 * np.median(np.abs(peer_values - med))
    if mad < 1e-8:
        return STATIC_FALLBACK[metric]
    z = (company_value - med) / mad
    return med + 2.0 * mad  # threshold at z=2
```

#### Method F: Benford-Adjusted Industry Norms (For R&D and SGA)

**Theory**: Amiram, Bozanic & Rouen (2015) showed that companies manipulating financial statements deviate from Benford's Law in their reported figures. Rather than flagging R&D > 10% as vanity, check whether the R&D reporting pattern deviates from what's expected given the company's sector and size.

**Formula**:
```
expected_rnd_intensity = f(sector, log_revenue, growth_stage)
excess = actual_rnd - expected_rnd_intensity
vanity_signal = excess > 2 * sector_MAD(rnd_intensity)
```

Where `f()` is a simple linear model fitted on peers:
```
expected_rnd = beta0 + beta1 * log(revenue) + beta2 * growth_rate
```

**Academic basis**: Amiram, Bozanic & Rouen (2015, TAR). The idea is that R&D spending should be predictable from company characteristics. Deviations suggest either genuine innovation or capital misallocation.

**Pros**: Accounts for company size and growth stage.
**Cons**: Needs enough peers to fit the regression. Fallback to Z-score when < 5 peers.

### Recommended: Method E (Z-Score) with Method F regression when peers are plentiful

```python
def vanity_threshold(metric, cache, linked_caches):
    peer_values = collect_peer_metric(metric, linked_caches)
    if len(peer_values) >= 10:
        # Enough peers for regression-based expected value
        expected = regression_expected(metric, cache, linked_caches)
        mad = sector_mad(metric, peer_values)
        return expected + 2.0 * mad
    elif len(peer_values) >= 3:
        # Z-score approach
        med = np.median(peer_values)
        mad = 1.4826 * np.median(np.abs(peer_values - med))
        return med + 2.0 * mad
    else:
        return STATIC_FALLBACK[metric]
```

---

## 3. Financial Health Label Thresholds

### Current State
```python
_LABEL_THRESHOLDS = [
    (20.0, "Critical"),
    (35.0, "Weak"),
    (50.0, "Fair"),
    (65.0, "Good"),
    (80.0, "Strong"),
]
```

### Proposed Methods

#### Method G: Gaussian Mixture Model Clustering

**Theory**: Instead of fixed quintile breakpoints, let the data tell you where the natural clusters are. A GMM with 5 components will find the natural break points in the composite score distribution.

**Formula**:
```
gmm = GaussianMixture(n_components=5).fit(fh_composite_scores)
boundaries = sorted centroids between adjacent components
```

**Academic basis**: Fraley & Raftery (2002, JASA). Model-based clustering is the statistically proper way to find natural breakpoints in continuous distributions.

**Pros**: Finds natural clusters in the data. Different sectors will get different breakpoints.
**Cons**: Needs enough historical scores. GMM is already in the pipeline (regime_detector).

#### Method H: Jenks Natural Breaks (Fisher-Jenks)

**Theory**: The Fisher-Jenks optimization (1977) minimizes within-class variance while maximizing between-class variance. It finds the optimal breakpoints for categorizing a continuous variable into k classes. Used extensively in cartography (GIS) for choropleth maps.

**Formula**:
```
minimize sum_k(sum_i_in_k (x_i - mean_k)^2)
```

This is the 1-D equivalent of k-means, solvable exactly via dynamic programming in O(k*n^2).

**Academic basis**: Fisher (1958), Jenks (1967). The `jenkspy` library implements this efficiently.

**Pros**: Optimal breakpoints by definition (minimum variance within groups). Deterministic (no random initialization like GMM).
**Cons**: Requires a distribution of scores. Needs `jenkspy` package (~50KB) or a custom DP implementation.

#### Method I: Expanding Percentile Ranks (Already Partially Used)

**Theory**: The financial health module already uses expanding percentile rank normalization for individual tier scores. Extend this to the composite score labels: instead of fixed [20, 35, 50, 65, 80], use the expanding 20th/35th/50th/65th/80th percentiles of the company's own composite score history.

**Formula**:
```
label_thresholds[i] = Q_p(composite_score_history_to_day_t)
where p = [0.20, 0.35, 0.50, 0.65, 0.80]
```

**Pros**: Zero new dependencies. Self-calibrating. First-day scores use the fixed defaults; as history accumulates, thresholds adapt.
**Cons**: Company-specific only (not peer-relative). A consistently mediocre company will never get a "Critical" label.

### Recommended: Method H (Jenks) when peers available, Method I (expanding percentile) as fallback

```python
def adaptive_health_labels(composite_scores, peer_scores=None):
    if peer_scores is not None and len(peer_scores) >= 20:
        # Jenks natural breaks on combined own + peer scores
        all_scores = np.concatenate([composite_scores, peer_scores])
        breaks = jenkspy.jenks_breaks(all_scores, n_classes=5)
        return breaks[1:-1]  # 4 interior breakpoints
    else:
        # Expanding percentile on own history
        return [
            np.nanpercentile(composite_scores, p)
            for p in [20, 35, 50, 65, 80]
        ]
```

---

## 4. Regime Mixer Fundamental Thresholds

### Current State
```python
_FUND_THRESHOLDS = {
    "distress": {"current_ratio": 0.8, "debt_to_equity": 4.0, "drawdown_252d": -0.50},
    "stressed": {"current_ratio": 1.2, "debt_to_equity": 2.5, "drawdown_252d": -0.25},
}
```

Plus inline: `cash_ratio < 0.1` (distress), `cash_ratio < 0.3` (stress).

### Proposed Methods

#### Method J: Hidden Markov Model Emission Parameters (Self-Calibrating)

**Theory**: The regime_mixer already classifies into healthy/stressed/distress regimes. Instead of fixed thresholds, use the HMM emission parameters directly. The HMM learns the distribution of each variable under each regime during fitting. The natural boundary between regimes is where the posterior probability of regime membership crosses 0.5.

**Formula**:
```
P(distress | x) = N(x; mu_distress, sigma_distress) * pi_distress / sum_k(...)
threshold = x where P(distress | x) = 0.5
```

Solving for the crossover:
```
threshold = (mu_healthy * sigma_distress^2 - mu_distress * sigma_healthy^2 
             + sigma_healthy * sigma_distress * sqrt(delta)) 
            / (sigma_distress^2 - sigma_healthy^2)
```
where delta accounts for the log-prior ratio.

**Academic basis**: Gaussian HMM crossover points (Rabiner 1989). The regime_detector already fits HMMs -- this reuses those parameters.

**Pros**: Fully data-driven. Automatically adapts to each company's risk profile.
**Cons**: Requires the HMM to have already been fitted (available after Step 5.5).

#### Method K: Copula-Based Joint Distress Probability

**Theory**: Instead of per-variable thresholds, compute the joint probability of being in the distress region of the multivariate distribution using the fitted copula. The copula module already estimates tail dependence.

**Formula**:
```
P(distress) = C(u1, u2, u3, u4) 
where ui = P(Xi < xi) for each metric
```

Use the copula's joint CDF to find the contour where P(joint_distress) = 0.10.

**Academic basis**: Joe (2014) "Dependence Modeling with Copulas." The copula module already fits Student-t and Clayton copulas. This extends their use from uncertainty bands to regime classification.

**Pros**: Captures correlations between distress indicators (current_ratio and D/E move together during stress).
**Cons**: Requires copula fitting (available after Step 6m). Higher computational cost.

### Recommended: Method J (HMM crossover) for regime thresholds, falling back to Method A (peer percentile) when HMM didn't fit

The HMM is already fitted in Step 5.5 and its parameters are stored in `regime_detector.result`. The crossover calculation adds ~10 lines of code.

---

## Summary: Selected Methods Per Constant

| Constant | Primary Method | Fallback | New Deps |
|----------|---------------|----------|----------|
| `current_ratio_lt` | A: Peer P10 + D: BOCPD deterioration | Winsorized mean - 2*MAD, then textbook 1.0 | None |
| `debt_to_equity_gt` | A: Peer P90 + D: BOCPD | Winsorized mean + 2*MAD, then textbook 3.0 | None |
| `fcf_yield_lt` | A: Peer P10 | MAD-based, then textbook 0.0 | None |
| `drawdown_lt` | Own historical P5 + D: BOCPD | textbook -0.40 | None |
| `rnd_threshold` | E: Sector Z-score (z>2) | F: Regression-expected + 2*MAD when 10+ peers | None |
| `sga_threshold` | E: Sector Z-score | Static 0.30 when < 3 peers | None |
| `exec_comp_threshold` | E: Sector Z-score | Static 5.0% | None |
| `marketing_threshold` | E: Sector Z-score | Static 10.0% | None |
| `fh_label_thresholds` | H: Jenks natural breaks on peer+own scores | I: Expanding percentile on own history | `jenkspy` (~50KB) or custom DP |
| `regime_mixer thresholds` | J: HMM emission crossover points | A: Peer percentile then static | None |
| `cash_ratio distress/stress` | A: Peer P10 / P25 | Static 0.1 / 0.3 | None |

### New Dependencies
- `jenkspy` (optional, ~50KB) for Jenks natural breaks. Can be replaced with a 30-line custom implementation.

### Implementation Location
A new module `operator1/analysis/adaptive_thresholds.py` that:
1. Consumes `linked_caches`, `regime_detector.result`, and `cache`
2. Produces a `ThresholdSet` dataclass with all 18 adaptive thresholds
3. Called once in main.py after Step 5h (peer ranking) and before survival mode detection
4. Result passed to `compute_company_survival_flag()`, `run_monte_carlo()`, `compute_dual_regimes()`, and `compute_vanity_score()` via their existing `thresholds` parameters
5. Falls back to current static values when insufficient data is available

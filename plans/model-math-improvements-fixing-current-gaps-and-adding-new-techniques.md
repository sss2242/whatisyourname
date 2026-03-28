# Model Math Improvements: Fixing Current Gaps and Adding New Techniques

Based on the V3 backtest findings (new modules added zero predictive value due to data gaps and disconnected wiring) and a deep review of the 45-module analysis map.

This plan has two parts:
- **Part A**: Fix the 8 concrete problems exposed by the backtest
- **Part B**: Add 12 new mathematical techniques that would genuinely improve predictions

---

## Part A: Fix the 8 Backtest Problems

### A1. Feed buying_power + catalyst INTO forecasting models

**Problem**: BPI and catalyst_score are computed and stored in cache columns but NOT included in `_extra_vars` fed to temporal models. They are invisible to GARCH, VAR, LSTM, and the prediction aggregator.

**Fix**: In main.py Step 6, add `buying_power_index`, `sector_demand_momentum`, `catalyst_score` to `_extra_vars` list. These columns already exist in the cache -- they just need to be whitelisted.

**Impact**: VAR and LSTM can now learn from demand-side signals. If BPI is declining while price is rising, the model learns this divergence is unsustainable.

### A2. Compute derived ratios BEFORE estimation

**Problem**: `interest_coverage`, `cash_ratio`, `gross_margin`, etc. are NaN because they require derived variables that are computed AFTER estimation. But the FH calibration checks `interest_coverage > 10` which is always NaN at that point.

**Fix**: Move a lightweight derived-ratio pass (just the basic ratios, not returns/technicals) to Step 4a.6, right before estimation. This way:
- Step 4a.6: Compute basic ratios (current_ratio, interest_coverage, margins) from raw statement data
- Step 4b: Estimation fills remaining gaps using these ratios as features
- Step 5: Full derived variables (returns, technicals) from the enriched cache

### A3. Fix product_catalysts news article passing

**Problem**: `sentiment_result.articles` doesn't exist -- the SentimentResult dataclass stores article data differently (it's computed from scores, not stored as raw articles).

**Fix**: Modify news_sentiment.py to expose `_fetched_articles` as a public attribute, or have the catalyst detector fetch its own news independently (more robust).

### A4. Fix MC terminal_values population

**Problem**: `max_drawdown_distribution` is empty because `terminal_values` dict is never populated in the MC simulation loop.

**Fix**: In monte_carlo.py, after each horizon simulation, store the terminal price ratios: `result.terminal_values[h_label] = terminal_prices / start_price`. The array exists in the simulation loop but isn't being saved to the result.

### A5. Handle single-regime GARCH gracefully

**Problem**: HMM classified 495 of 501 days as "high_vol" and 1 day as "bull". The regime-switching GARCH can't differentiate with only 1 observation in the second regime.

**Fix**: Add a minimum regime diversity check. If any regime has fewer than 20 observations, merge it into its nearest neighbor (by mean return). If only 1 effective regime remains, fall back to single-regime GARCH.

### A6. Derive R&D from SGA for Apple-type companies

**Problem**: Product catalyst's R&D acceleration signal was 0.0 because Apple bundles R&D into SGA. SEC filings DO contain R&D as a separate line item in the notes, but the canonical translator maps it to `sga_expenses`.

**Fix**: In the canonical translator, map `ResearchAndDevelopmentExpense` (US-GAAP XBRL tag) to `rd_expenses` as a separate field. The edgartools library already extracts it -- the translator just doesn't have the mapping.

### A7. FH calibration: use TTM values, not daily-ffilled

**Problem**: `interest_coverage = ebit / interest_expense`. Both values in the cache are forward-filled quarterly values, meaning they're the SAME value repeated for 63 days. The calibration check sees non-NaN values but they don't represent daily economics.

**Fix**: Use `_ttm` (trailing twelve month) versions of these ratios when available. The derived_variables module already computes TTM for some variables -- extend to interest_coverage_ttm and cash_ratio_ttm.

### A8. Expand estimation identity fill

**Problem**: The estimator's Phase 1 only has 4 accounting identities. Adding more would cascade-fill the NaN variables:
- `gross_profit = revenue - cost_of_revenue`
- `operating_income = gross_profit - sga_expenses` (approximation)
- `interest_coverage = ebit / interest_expense`
- `cash_ratio = cash_and_equivalents / current_liabilities`
- `net_margin = net_income / revenue`

**Fix**: Add these 5 derived identities to the estimator's Phase 1 loop. They're not pure accounting identities (they're ratios), but they can be computed deterministically when both components are present.

---

## Part B: 12 New Mathematical Techniques

### B1. Merton Distance-to-Default Model

**What**: The Merton (1974) structural credit model treats equity as a call option on firm assets. Distance-to-Default (DD) measures how many standard deviations the firm's asset value is above the default point.

**Why**: The current survival model uses threshold triggers (current_ratio < 1, etc.). Merton DD provides a continuous, market-implied default probability that incorporates both asset volatility and leverage.

**Formula**: DD = (ln(V/D) + (mu - 0.5*sigma_V^2)*T) / (sigma_V * sqrt(T)), where V = market_cap + total_debt (asset value proxy), D = total_debt (default point), sigma_V = equity_vol * (V / (V-D)).

**Module**: Add to `financial_health.py` as `compute_merton_dd()`. Output: `fh_merton_dd`, `fh_merton_pd` (probability of default).

**Free dependencies**: None needed (uses numpy only).

### B2. Fama-French Factor Decomposition

**What**: Decompose returns into systematic factors: market (MKT-RF), size (SMB), value (HML), profitability (RMW), investment (CMA). The residual is alpha.

**Why**: The current pipeline treats returns as a single time series. Factor decomposition separates company-specific risk from market-wide risk. If 80% of a stock's movement is explained by the market factor, the forecasting models should focus on the 20% idiosyncratic component.

**Data source**: Kenneth French's data library (free, no key, CSV download).

**Module**: New `operator1/features/factor_decomposition.py`. Output: `factor_alpha_1y`, `factor_beta_market`, `factor_beta_smb`, `factor_beta_hml`, `factor_exposure_pct`, `idiosyncratic_vol`.

### B3. Ornstein-Uhlenbeck Mean Reversion Speed

**What**: Fit an OU process to financial ratios to estimate the mean-reversion half-life. Ratios like current_ratio and debt_to_equity are mean-reverting by nature -- companies that drift too far from their normal level tend to correct.

**Why**: The current GARCH and Kalman models treat volatility as mean-reverting but don't model financial ratios the same way. OU half-life tells you: "this company's current_ratio is 2 std below its mean, and it typically takes 45 business days to revert halfway." This directly improves drawdown and survival probability forecasts.

**Formula**: dX = theta * (mu - X) * dt + sigma * dW. Estimate theta via MLE: theta = -ln(rho) / dt where rho = autocorrelation at lag 1.

**Module**: Add to `adaptive_model_params.py` as `compute_ou_reversion_speed()`. Output: per-variable `ou_halflife`, `ou_equilibrium`, `ou_current_deviation`.

### B4. Entropy-Based Market State Classification

**What**: Use Shannon entropy of the return distribution over rolling windows to classify market states. Low entropy = concentrated distribution = trending market. High entropy = uniform distribution = choppy/uncertain market.

**Why**: HMM regime detection uses parametric Gaussian assumptions. Entropy is non-parametric and captures distributional shape changes that HMM misses (fat tails, skewness shifts).

**Formula**: H = -sum(p_i * log(p_i)) over discretized return bins. Normalize by log(n_bins) for comparability.

**Module**: Add to `regime_detector.py` as `compute_entropy_regimes()`. Output: `entropy_21d`, `entropy_regime` (low/medium/high), `entropy_trend`.

### B5. Benford's Law Forensic Analysis

**What**: Check if the leading digits of financial statement line items follow Benford's Law. Deviations suggest potential data manipulation or accounting irregularities.

**Why**: The Beneish M-Score detects earnings manipulation via ratio changes. Benford's Law catches a different class of fraud: fabricated numbers. Real financial data follows Benford's distribution naturally; manipulated numbers don't.

**Test**: Chi-squared goodness-of-fit of first digits against Benford's expected distribution [30.1%, 17.6%, 12.5%, 9.7%, 7.9%, 6.7%, 5.8%, 5.1%, 4.6%].

**Module**: Add to `ethical_filters.py` as `compute_benford_test()`. Output: `benford_chi2`, `benford_p_value`, `benford_flag` (True if p < 0.05).

### B6. Topological Data Analysis (Persistent Homology)

**What**: Use persistent homology (via the `ripser` package already installed) to detect topological features in the financial time series -- loops, voids, and higher-dimensional structure that linear methods miss.

**Why**: Market crashes often create "holes" in the return-volume phase space that persist for weeks before the actual price decline. TDA can detect these structural changes before they manifest in price.

**Module**: New `operator1/models/topological_risk.py`. Uses `ripser` (already in stage2-ml). Output: `tda_persistence_score`, `tda_structural_anomaly`, `tda_regime_complexity`.

### B7. Wasserstein Distance for Distribution Drift

**What**: Measure the Wasserstein (Earth Mover's) distance between the current return distribution and historical reference distributions. A large distance indicates the return process has structurally changed.

**Why**: GARCH assumes stationarity within each regime. Wasserstein distance detects WHEN the distribution has shifted enough that the GARCH parameters are stale, triggering a refit.

**Formula**: W_1(P, Q) = integral of |F_P(x) - F_Q(x)| dx.

**Module**: Add to `regime_detector.py` as `compute_distribution_drift()`. Output: `wasserstein_drift_21d`, `distribution_shift_flag`.

### B8. Kelly Criterion Position Sizing

**What**: Compute the Kelly-optimal fraction of capital to allocate based on the forecast edge and variance.

**Why**: The pipeline produces forecasts and survival probabilities but doesn't translate these into a position size recommendation. Kelly criterion (f* = (p*b - q) / b where p=win prob, b=win/loss ratio, q=1-p) gives the mathematically optimal allocation.

**Module**: Add to `prediction_aggregator.py` as `compute_kelly_fraction()`. Output: `kelly_fraction`, `kelly_edge`, `half_kelly` (conservative version).

### B9. Spectral Risk Measures (Distortion Risk)

**What**: Replace VaR/CVaR with spectral risk measures that weight tail losses by a user-specified risk aversion function.

**Why**: CVaR weights all tail losses equally. Spectral measures let you express "I care 3x more about the worst 1% than the worst 5%." This produces more realistic risk-adjusted survival probabilities.

**Formula**: rho(X) = integral of q_alpha(X) * phi(alpha) d_alpha where phi is the distortion function.

**Module**: Add to `monte_carlo.py` as `compute_spectral_risk()`. Output: `spectral_risk_conservative`, `spectral_risk_moderate`.

### B10. Robust Covariance via Minimum Covariance Determinant

**What**: Replace sample covariance (used in copula, VAR, and graph risk) with the Minimum Covariance Determinant estimator from scikit-learn. MCD is robust to up to 50% outliers.

**Why**: A single earnings surprise day can corrupt the sample covariance matrix, causing copula, VAR, and correlation-based modules to produce distorted results for weeks afterward.

**Module**: Apply within `copula.py`, `forecasting.py` (VAR), and `graph_risk.py`. Use `sklearn.covariance.MinCovDet` (already installed) instead of `np.cov`.

### B11. Hawkes Process for Event Clustering

**What**: Model the arrival rate of extreme events (large drawdowns, earnings surprises, regime switches) as a self-exciting Hawkes process. Past events increase the probability of future events.

**Why**: Financial crises cluster. The 2008 Lehman collapse triggered Bear Stearns, AIG, etc. in rapid succession. The current MC simulation assumes regime switches are Markovian (memoryless). Hawkes process captures event clustering, producing more realistic tail simulations.

**Module**: New `operator1/models/event_clustering.py`. Output: `hawkes_intensity`, `cluster_probability_7d`, `event_half_life`.

### B12. Conformal E-Values for Sequential Testing

**What**: Replace p-value-based hypothesis testing in Granger causality with e-values (Vovk 2021). E-values are valid under optional stopping and continuous monitoring, unlike p-values which require fixed sample sizes.

**Why**: The pipeline runs Granger tests on rolling windows, violating the fixed-sample-size assumption. E-values remain valid under this sequential testing regime. This prevents false causal links from being detected in short windows.

**Module**: Add to `granger_causality.py` as an alternative to the F-test. Use the `e-value = likelihood_ratio` formulation from Ramdas et al. (2023).

---

## Implementation Priority

| Priority | Items | Rationale |
|----------|-------|-----------|
| P0 (critical fixes) | A1-A5 | These fix the backtest failures -- new modules are wired but not producing value |
| P1 (high impact) | A6-A8, B1, B2, B3 | Fill data gaps + Merton DD and factor decomposition are standard institutional tools |
| P2 (medium impact) | B4, B5, B7, B10 | Entropy regimes, Benford forensics, distribution drift, robust covariance |
| P3 (advanced) | B6, B8, B9, B11, B12 | TDA, Kelly, spectral risk, Hawkes, e-values -- powerful but complex |

---

## Dependency Impact

All P0 and P1 items use packages already installed. P2-P3 items:
- B6 (TDA): `ripser` already in stage2-ml
- B10 (MCD): `sklearn` already installed
- B11 (Hawkes): needs `tick` library (~5MB) or manual implementation
- B12 (E-values): manual implementation (no library needed)

No new dependencies required for P0-P2.

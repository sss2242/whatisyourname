# Expert Methods for Closing Prediction Error Gaps

*2026-04-20*

For each error pattern, I've cataloged what works in practice -- both the methods everyone knows and the ones that actually move the needle at top quant firms.

---

## Error #1: Upward Bias (Kalman predicts near last close, misses -3% to -9% drop)

### Popular Methods

**A. Momentum Factor Overlay (Jegadeesh & Titman 1993)**
The textbook answer. Blend a momentum signal (12-1 month return) into the prediction. Every quant shop runs some version of this. The problem: momentum is well-known and already partially priced, so the alpha is thin for liquid mega-caps like AAPL.

**B. ARIMAX / Kalman with Exogenous Regressors (Harvey 1989)**
Add macro/sector features as control inputs to the state-space model. We already implemented this via the residual regression (Bug #2 fix). But it only shifts the prediction by the capped 5% adjustment.

**C. Ensemble with Directional Models**
Run a classification model (direction: up/down) alongside the regression model (magnitude), then combine. If the classifier says "down" with 70% confidence, scale the regression prediction downward. XGBoost and LightGBM are the standard classifiers.

### Unpopular but Effective Methods

**D. Post-Earnings Announcement Drift (PEAD, Bernard & Thomas 1989)**
AAPL's January drop was partially driven by Q4 earnings uncertainty. PEAD is one of the most robust anomalies in finance -- stocks drift in the direction of their earnings surprise for 60+ trading days. The pipeline has an `earnings_surprise_probability` module in the HF engine, but it doesn't flow back to price predictions. Connecting the HF earnings probability to the price forecast would capture this.

**E. Conditional Density Estimation (Coulombe 2024 -- "The macroeconomy as a random forest")**
Instead of predicting E[price], predict the full conditional distribution P(price | features). When the distribution is skewed left (as it typically is at all-time highs with elevated IV), the median prediction will be below the mean prediction. The pipeline's Monte Carlo already does this implicitly -- the MC median path should be used instead of the mean.

**F. Anchoring Adjustment (Tversky & Kahneman 1974, applied to financial models by De Bondt 1993)**
The Kalman filter anchors to the last price. But the "fair" anchor might be the 200-day moving average, the sector median PE x trailing EPS, or the DCF midpoint. Blending multiple anchors reduces directional bias. The HF DCF module produces an intrinsic value ($33.05 for AAPL -- clearly wrong due to NaN ratios, but once fixed it provides a fundamental anchor).

**G. Regime-Conditional Prediction Shift (Ang & Timmermann 2012)**
Instead of one prediction, produce regime-specific predictions and weight them by regime probabilities. If P(bear) = 30%, shift the ensemble prediction down by 30% * expected_bear_drawdown. The HMM regime probabilities are already in the cache (`regime_hmm_prob_*`) but not used for prediction adjustment.

---

## Error #2: Conformal Intervals Too Wide at 21d (-$60 to $559)

### Popular Methods

**A. Split Conformal by Horizon (Lei et al. 2018)**
Maintain separate calibrators per horizon. Standard approach, straightforward to implement. The current single-pool design is the bug.

**B. Locally Weighted Conformal (Papadopoulos 2008)**
Weight recent residuals more than old ones (exponential decay). Already partially implemented via the adaptive flag in `ConformalCalibrator`, but the decay rate isn't tuned to the horizon.

### Unpopular but Effective Methods

**C. Conformal Prediction with Kernel Density Estimation (Vovk 2005 "Algorithmic Learning in a Random World")**
Instead of using quantiles of the residual distribution, estimate the full residual density and compute intervals from the density. This is particularly effective when the residual distribution is multimodal (e.g., pre- vs post-earnings), which quantile-based methods handle poorly.

**D. Predictive Intervals from MC Paths Directly (Glasserman 2003)**
Skip the conformal framework entirely for longer horizons. The Monte Carlo simulation already produces 10,000 price paths. The P5/P95 of terminal values at each horizon IS a prediction interval with approximately correct coverage, and it naturally incorporates regime switching, fat tails, and non-stationarity. For AAPL at 21d, this would produce a tighter, more realistic interval than the conformal approach.

**E. Horizon-Aware Variance Scaling (Christoffersen & Diebold 2006)**
Scale the 1d prediction interval to longer horizons using sqrt(h) for random walks or a calibrated scaling factor learned from the data. If the 1d interval is $24.41 wide ($225.84 to $274.66 = $48.82 total), the 21d interval should be approximately $48.82 * sqrt(21/1) = $223.79 wide, not $619 wide. The sqrt(h) scaling naturally prevents degenerate widening.

**F. Conformalized Quantile Regression (Romano et al. 2019)**
Train a quantile regression model (we already have `QuantileRegressionCalibrator` in conformal.py) to produce the 5th and 95th quantile directly, then calibrate the coverage using conformal adjustment. This produces asymmetric intervals (wider on the downside for stocks at ATH) and naturally adapts to the horizon.

---

## Error #3: MC Survival 252d = 12% (Mega-cap treated like small-cap)

### Popular Methods

**A. Market-Cap Quintile Calibration**
Bucket stocks by market cap, compute survival statistics per bucket from historical data (CRSP/Compustat), use bucket-specific thresholds. Standard approach in credit risk modeling.

**B. Peer Percentile Thresholds (already in the pipeline)**
The `adaptive_thresholds.py` module does this, but it failed because SEC rate limiting blocked peer discovery. Fix: pre-cache sector peer lists or use a different peer resolution path (yfinance sector peers, Fama-French industry portfolios).

### Unpopular but Effective Methods

**C. Implied Survival from Options (Carr & Wu 2011)**
If a stock has listed options, the out-of-the-money put skew directly encodes the market's assessment of crash probability. The difference between 25-delta put IV and ATM IV is a market-implied survival probability. The pipeline's options_signals module already computes `risk_reversal_25d` and `iv_skew` -- these can be converted to an implied survival probability and blended with the MC estimate.

**D. Distance-to-Default (Merton 1974, KMV Model)**
The HF advanced methods already compute Merton default probability, but it's not fed back to the MC survival thresholds. If Merton says P(default 1yr) = 0.01% (as it would for AAPL), the MC survival should be anchored near 99.99%, not 12%.

**E. Survival Bayesian Updating (Kaplan-Meier with Prior)**
Instead of simulating from scratch, start with a prior survival curve based on the company's quality metrics (Altman Z, Piotroski F-Score, market cap percentile) and update it with the MC simulation paths. The prior prevents the MC from producing absurd results when the simulation inputs are noisy.

**F. Structural Breaks in Survival Thresholds (Bai & Perron 2003)**
The fixed -40% drawdown threshold should be replaced with a market-regime-conditional threshold. During normal markets, a -40% drawdown for a mega-cap is catastrophic. During 2022 (bear market), AAPL's drawdown was -31% and it survived fine. The threshold should widen during broad market stress periods.

---

## Error #4: HF Strong Sell Due to NaN Ratios

### Popular Methods

**A. Expand XBRL Tag Coverage**
Map more non-standard tags. Apple uses extended taxonomy entries that differ from the standard US-GAAP concepts. The SEC's full taxonomy has ~16,000 concepts; we map ~45.

**B. Fallback to Computed Values**
When `current_assets` is NaN but `total_assets` and `non_current_assets` are available, compute `current_assets = total_assets - non_current_assets`. The estimation engine's identity fill phase does some of this but doesn't cover all useful identities.

### Unpopular but Effective Methods

**C. SEC CompanyFacts API Fallback (sec-edgar-api)**
When edgartools XBRL parsing misses a concept, fall back to the SEC's `companyfacts` endpoint which returns ALL reported XBRL facts for a CIK as flat JSON. The `sec-edgar-api` library (already installed) provides direct access. Search for the concept name across all reported facts.

**D. Multi-Source Reconciliation for Critical Ratios**
For the 7 NaN ratios, attempt extraction from THREE sources: (1) XBRL structured data (current), (2) SEC companyfacts API (flat fact lookup), (3) 10-K text extraction via LLM (already built in `llm_filing_extractor.py`). Take the first non-NaN value. This is how Bloomberg and FactSet handle it internally -- multi-source with priority cascading.

**E. Bayesian Imputation with Sector Priors (Rubin 1976)**
When a ratio is NaN, impute from the sector distribution. AAPL's current_ratio is missing, but the Technology sector median current_ratio is ~1.4. Use a Bayesian prior: `imputed = sector_median * 0.5 + company_estimate * 0.5`. The `miceforest` MICE imputation in the estimation engine could handle this if given cross-sectional sector data.

**F. Score Normalization by Available Data (Fama & French 2008)**
Instead of scoring on 9 Piotroski factors and getting 2/9, score on the 4 factors that ARE available and normalize: 2/4 = 50% = score of 4.5/9 equivalent. This prevents NaN factors from dragging the score toward distress.

---

## Error #5: HMM Labels 495/502 Days as High-Vol

### Popular Methods

**A. BIC/AIC Model Selection for State Count**
Fit HMMs with 2, 3, 4, 5, 6 states, pick the one with lowest BIC. For 502 observations, 2-3 states usually wins.

**B. Sticky HMM (Fox et al. 2011)**
Add a self-transition bias to the HMM so states persist longer. This prevents the degenerate case where one state absorbs everything because switching costs are too low.

### Unpopular but Effective Methods

**C. Spectral Clustering on Return Distribution (Cont 2001)**
Instead of HMM (which assumes Gaussian emissions), cluster the return distribution directly using kernel density estimation. Identify modes of the return density as natural regimes. This handles multimodal, heavy-tailed distributions better than Gaussian HMM.

**D. Change Point Detection as Regime Boundaries (Killick 2012 -- already in the pipeline)**
Use PELT/BCP change points to segment the time series, then characterize each segment as a regime. The pipeline already does PELT (detected 2 breaks for AAPL), but the results aren't used to OVERRIDE the HMM when the HMM produces degenerate results. A simple heuristic: if any HMM state has <5% of observations, fall back to PELT-based regime labeling.

**E. Online Regime Detection (Bardet & Kengne 2014)**
Replace the batch HMM (fits all 502 days at once, look-ahead risk) with an online detector that updates beliefs one day at a time. ChangeFinder is already in the pipeline (`online_change_score`), but its output isn't used for regime labeling -- only as a feature. Promoting ChangeFinder scores to regime labels would eliminate the look-ahead issue entirely.

**F. Volatility Regime from Realized Measures (Bollerslev et al. 2016)**
Instead of HMM on (return, volatility), classify regimes directly from realized volatility quintiles: Q1=low-vol, Q2-Q3=normal, Q4=elevated, Q5=high-vol. This is deterministic, interpretable, and can't produce degenerate states. Renaissance Technologies reportedly uses a variant of this.

---

## Recommended Implementation Priority

Based on impact per line of code:

| Priority | Method | Error Fixed | Lines | Impact |
|----------|--------|-------------|-------|--------|
| 1 | SEC CompanyFacts API fallback for NaN ratios (D4) | Error #4 | ~40 | **CRITICAL** -- fixes the cascade of NaN -> wrong HF signal |
| 2 | Score normalization by available data (F4) | Error #4 | ~20 | **HIGH** -- Piotroski/Altman become meaningful even with partial data |
| 3 | MC intervals instead of conformal for 21d+ (D2) | Error #2 | ~25 | **HIGH** -- immediately produces realistic intervals |
| 4 | Merton default -> MC survival anchor (D3) | Error #3 | ~15 | **HIGH** -- prevents absurd survival estimates for quality companies |
| 5 | Regime probability-weighted prediction shift (G1) | Error #1 | ~20 | **MEDIUM** -- reduces directional bias |
| 6 | PELT fallback when HMM degenerate (D5) | Error #5 | ~15 | **MEDIUM** -- fixes 495/502 high_vol labeling |
| 7 | PEAD earnings drift -> price prediction (D1) | Error #1 | ~30 | **MEDIUM** -- captures the strongest known anomaly |
| 8 | Market-cap quintile survival calibration (A3) | Error #3 | ~40 | **MEDIUM** -- right-sizes thresholds by company size |

Total: ~205 lines across 6-8 files. Methods 1-4 together would fix the two biggest errors (NaN cascade and degenerate intervals/survival) with ~100 lines.

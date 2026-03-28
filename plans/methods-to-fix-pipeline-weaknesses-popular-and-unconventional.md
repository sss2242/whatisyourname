# Methods to Fix Pipeline Weaknesses -- Popular and Unconventional

Based on the AAPL backtest (2023-2024 data predicting 2025), these are the specific failures observed and candidate solutions ranging from textbook approaches to obscure-but-effective techniques.

---

## Problem 1: Close Price Prediction Off by $45 (predicted $197.67, actual $242.53)

The Kalman filter (which won the GA tournament) used stale filing data (634 days old) as its state, producing a price anchored to mid-2023 fundamentals rather than Dec 2024 market reality.

### Popular Methods

**A. Ensemble with yfinance real-time close (trivial fix)**
The pipeline fetches OHLCV from yfinance but the forecasting models don't anchor to the last observed close. The Kalman state should be initialized from the most recent OHLCV close, not from the financial statement merge.

**B. Exponential Moving Average (EMA) as baseline anchor**
Replace last-value carry-forward baseline with 21-day EMA. This is standard in quantitative finance and already partially implemented in the walk-forward module.

**C. ARIMA/SARIMA with exogenous variables (ARIMAX)**
statsforecast AutoARIMA is already available (`fit_autoarima()` in forecasting.py). Wire it as a first-class model in the forecast chain rather than a standalone function.

### Unconventional Methods

**D. Implied Forward Price from Options Term Structure**
Extract the risk-neutral expected price from put-call parity across option chains. CBOE data is available via yfinance `options` attribute. The options market prices in information that backward-looking models miss.

Formula: `F = S * e^(r*T)` adjusted by skew = `C - P + K*e^(-r*T)` where C/P are call/put prices at strike K.

No one does this in a filing-based pipeline. It bridges the gap between PIT fundamental data and market expectations.

**E. Anchored Diffusion Process (Heston 1993 variant)**
Instead of forecasting the price level, forecast the stochastic variance process and derive the price distribution. The current GARCH does variance but doesn't integrate it back into a price distribution properly. Heston's closed-form solution gives you the entire forward price density, not just a point estimate.

**F. Peer Consensus Anchor**
For whale companies, the close price is well-covered by analysts. Use the median of peer company price-to-earnings ratios multiplied by the target's earnings as an anchor. This is basically "where should Apple trade if it's priced like its peers?" -- a fundamentals-implied price.

---

## Problem 2: Volatility Underestimated by 2x (predicted 0.0103, actual 0.0199)

Single-regime GARCH mean-reverts to unconditional variance, missing regime transitions.

### Popular Methods

**A. Markov-Switching GARCH (already implemented, needs tuning)**
The regime-switching GARCH code exists but the HMM only had 2 effective regimes in the AAPL data. Increase to 3-4 regimes and use the transition probability matrix to weight the forecast.

**B. Realized Volatility Estimators (Andersen & Bollerslev 1998)**
Use 5-minute or daily OHLC data to compute realized variance directly, rather than fitting a parametric GARCH model. The Garman-Klass estimator already exists in adaptive_model_params.py -- feed it into the forecast chain.

**C. HAR-RV Model (Corsi 2009)**
Heterogeneous Autoregressive model of Realized Volatility. Regress current RV on lagged daily, weekly, and monthly RV. Captures the long-memory property of volatility that GARCH misses. Simple linear regression, massively outperforms GARCH in empirical tests.

Implementation: 3 features (RV_1d, RV_5d, RV_21d) -> LinearRegression -> forecast.

### Unconventional Methods

**D. VIX Term Structure Decomposition**
CBOE VIX and VIX futures data is available via yfinance. The VIX term structure (VIX vs VIX3M vs VIX6M) contains forward-looking volatility expectations that GARCH cannot see. When the term structure is in contango (VIX < VIX3M), the market expects volatility to rise -- GARCH would miss this.

**E. Rough Volatility (Gatheral, Jaisson & Rosenbaum 2018)**
Volatility is not a diffusion process (smooth) but a rough process with Hurst exponent H ~ 0.1. The `roughvol` or fractional Brownian motion framework captures this. We already compute the Hurst exponent in adaptive_model_params.py -- use it to parametrize a rough Bergomi model instead of GARCH.

This is cutting-edge quantitative finance. Very few open-source pipelines implement it.

**F. Cross-Sectional Volatility Spillover (Diebold & Yilmaz 2012)**
If linked entities are available, compute the volatility spillover index: how much of Apple's volatility is explained by its supply chain's volatility? This connectedness measure would have shown rising spillover from TSMC/Samsung in late 2024 (chip supply concerns), predicting higher AAPL volatility.

---

## Problem 3: Max Drawdown Severely Underestimated (-3.4% vs -16.2%)

The Kalman-smoothed trend extrapolation doesn't capture tail risk.

### Popular Methods

**A. Monte Carlo Drawdown Distribution (already implemented)**
`run_monte_carlo()` already computes `max_drawdown_distribution` with median/P10/P90. The issue is the profile builder uses the Kalman point forecast instead. Wire the MC P10 drawdown into the prediction output.

**B. Historical Drawdown Quantiles**
Compute the empirical distribution of max drawdowns over rolling 252-day windows in the 2-year history. Report the P10/P25/P50 quantiles. No model needed -- pure empirical distribution.

**C. Expected Shortfall (CVaR) Approach**
Compute the expected loss in the worst 5% of scenarios (ES_5%). This is the standard regulatory risk measure (Basel III). Use the return distribution from the full sample.

### Unconventional Methods

**D. Optimal Stopping Theory (Shiryaev 2007)**
Model drawdown as a sequential detection problem: "what is the probability that we are currently in a drawdown that will exceed X%?" The Cusum-Shiryaev statistic gives a real-time posterior probability of being in a deepening drawdown. This would have flagged the early stages of the 2025 decline.

**E. Record Theory (Wergen 2013)**
Apply the theory of records (record highs and record lows) from extreme value statistics. The waiting time between new highs follows a predictable distribution. If the time since last all-time-high is lengthening, the probability of a deep drawdown increases exponentially. Apple hit ATH in late December 2024 -- record theory would have flagged the elevated drawdown risk from the compressed waiting time.

**F. Copula-Based Joint Drawdown**
If we have peer data (Samsung, MSFT, GOOGL), compute the joint drawdown probability using the fitted copula (Clayton for lower tail). When all tech peers are simultaneously declining, the probability of a deep AAPL drawdown is much higher than the marginal estimate. The copula module already exists -- wire its tail dependence into the drawdown forecast.

---

## Problem 4: Entity Discovery Fails with Free LLM (rate limited, empty responses)

### Popular Methods

**A. Cached Entity Graphs (the Wikipedia approach)**
Pre-build entity relationship graphs offline for the top 500 companies per exchange. Store in YAML/JSON. Skip LLM entirely for these companies. Update quarterly.

**B. SEC EDGAR Co-Filing Analysis**
Companies that file 13F reports holding the same stocks are likely in the same sector. Build a co-filing graph from EDGAR data alone -- no LLM needed. This is how academic researchers discover industry clusters.

**C. SIC/NAICS Code Peer Groups**
SEC filings include SIC codes. Companies sharing the same 4-digit SIC code are industry peers. EDGAR provides a full-text search API that can filter by SIC. Already partially implemented in the peer fallback.

### Unconventional Methods

**D. GLEIF Relationship Graph Traversal**
GLEIF has relationship data beyond parent-subsidiary: "fund management", "branch", "international branch", "ultimate accounting consolidation." Traverse 2 hops in the GLEIF graph to find related entities without any LLM. Apple -> Apple Operations Europe -> its auditor/law firm -> other clients of the same auditor. This reveals hidden structural connections.

**E. Patent Citation Network (USPTO PAIR)**
Companies that cite each other's patents are technologically related. USPTO PAIR API is free. Build a patent citation graph for the target company and extract the top cited/citing entities. These are your real competitors and technology suppliers -- more accurate than LLM guesses.

**F. Supply Chain Extraction from 10-K MD&A (Filing-Based, No LLM)**
SEC 10-K filings contain a "Risk Factors" section that names specific suppliers, customers, and competitive threats. Use regex patterns to extract company names from these sections:
- "our principal suppliers include" 
- "significant customers include"
- "we compete with"

This is deterministic, free, PIT-compliant, and doesn't need an LLM.

---

## Problem 5: Financial Health Score Too Conservative (48.7 "Fair" for Apple)

### Popular Methods

**A. Sector-Relative Scoring**
Score each ratio against the sector median, not absolute thresholds. Apple's D/E of 1.7 is average for tech but terrible for utilities. The adaptive thresholds module partially does this via Peer Percentile, but it needs peer data that entity discovery didn't provide.

**B. Piotroski F-Score (Piotroski 2000)**
9-point binary score (positive/negative on each of 9 fundamentals). Apple would score 7-8/9. Simple, well-validated in academic literature, less sensitive to extreme ratio values than the current percentile-based scoring.

### Unconventional Methods

**C. Merton Distance-to-Default (Merton 1974)**
Model the company's equity as a call option on its assets. The "distance to default" (DD) is the number of standard deviations between the current asset value and the default point. Apple's DD would be extremely high (~8-10), correctly identifying it as one of the safest companies in existence. This is how credit rating agencies actually think.

Implementation: `DD = (ln(V/D) + (r - 0.5*sigma^2)*T) / (sigma * sqrt(T))` where V = market cap + debt, D = total debt, sigma = asset volatility.

**D. CreditGrades Model (Finger et al. 2002)**
Extension of Merton that adds recovery rate estimation and a stochastic default barrier. Produces a credit spread and a survival probability that is much more calibrated than the current sigmoid-based approach.

**E. Altman Z-Score with Sector Adjustment (Altman 2013 revision)**
The classic Z-Score uses coefficients calibrated on 1960s manufacturing firms. Altman published revised coefficients for different sectors in 2013. Using tech-specific coefficients would give Apple a much higher score. Already have Z-Score in the pipeline -- just need sector-adjusted coefficients.

---

## Problem 6: OHLC Year Prediction Explodes to $157K (exponential compounding bug)

### Popular Methods

**A. Cap Cumulative Returns**
The OHLC predictor compounds daily returns for 252 steps without any mean-reversion or cap. Add a simple constraint: cumulative return over 252 days cannot exceed the historical maximum annual return for the sector (tech: ~100%).

**B. Use Log Returns with Drift Correction**
Compound log-returns instead of simple returns. Log-returns are additive and naturally constrain the distribution. Add a drift term equal to the risk-free rate for the mean path.

### Unconventional Methods

**C. Reflecting Barrier Process**
Model the price as a geometric Brownian motion with reflecting barriers at the analyst consensus price range. When the simulated price exceeds 2x or drops below 0.5x the starting price, reflect it back. This prevents runaway paths while preserving the realistic variance.

**D. Anti-Persistence Adjustment (Mandelbrot & Van Ness 1968)**
Financial returns exhibit anti-persistence (Hurst exponent < 0.5) at longer horizons. Large up-moves are followed by corrections. Inject negative autocorrelation into the multi-step OHLC propagation proportional to the deviation from the starting price. This naturally dampens the exponential explosion.

---

## Recommended Priority

| # | Fix | Method | Impact | Complexity |
|---|-----|--------|--------|------------|
| 1 | Entity discovery | F: 10-K regex extraction (no LLM) + A: cached graphs | Critical | Medium |
| 2 | Close price anchor | A: Initialize Kalman from last OHLCV close | Critical | Low |
| 3 | Volatility | C: HAR-RV model | High | Low |
| 4 | Max drawdown | A: Wire MC drawdown distribution + B: Historical quantiles | High | Low |
| 5 | OHLC explosion | A: Cap cumulative returns + B: Log returns | High | Low |
| 6 | Financial health | C: Merton distance-to-default | Medium | Medium |
| 7 | Volatility (advanced) | D: VIX term structure | Medium | Medium |
| 8 | Entity discovery (advanced) | E: Patent citation network | Low | High |

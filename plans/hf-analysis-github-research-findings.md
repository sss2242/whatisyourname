# HF Analysis -- GitHub Research Findings for 30 Proposed Methods

*Research date: 2026-05-06. Searched via GitHub Code Search API across public repositories.*

Each method below includes: best GitHub implementations found, key implementation patterns worth adopting, specific code techniques, and recommendations for our implementation.

---

## Domain 1: Advanced Credit Analysis

### 1.1 Merton KMV Default Term Structure

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `k-bargaoui/LongRangeDefaultProbas` | `merton.py` | Econometric approach using fractional Brownian motions for long-range default probabilities |
| `charlesrambo/risk_management` | `merton_model.py` | Clean risk management implementation |
| `DiegoLozoya1511/StockMarketRiskAnalysis` | `models.py` | Altman Z + Merton combined model |

**Key implementation pattern from `k-bargaoui`:**
- Uses a `CalibrationModel` base class with iterative Newton-Raphson to solve for asset value and volatility simultaneously
- `d1(x, sigma_A, current_time, mu)` and `d2(x, sigma_A, current_time, mu)` accept a time parameter, making term structure trivial: just iterate over T = [1, 2, 3, 5]
- Uses `scipy.optimize` for the `inversed_formula` calibration step
- The `mu` (drift) parameter enables different term structure shapes

**Recommendation for our implementation:**
- Adopt the parameterized-T approach: our existing Merton DD in [`derived_variables.py`](operator1/features/derived_variables.py) uses fixed T=1. Simply loop over T = [1, 2, 3, 5] with the same d1/d2 formula
- Use `scipy.optimize.brentq` for root-finding (faster than Newton-Raphson for 1D)
- Key insight: the drift `mu` should come from our `regime_distributions` (per-regime expected return from MC), not a fixed risk-free rate. This makes the term structure regime-aware

### 1.2 Credit Migration Matrix

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `open-risk/transitionMatrix` | `empirical_transition_matrix.py` | Apache 2.0 licensed, dedicated library for credit transition matrices |
| `KilianVoillaume/Financial-Projects` | `Credit_rating_transition_matrix_heatmap.py` | Visual credit migration analysis |
| `Akshat111111/Hedging-of-Financial-Derivatives` | `credit_rating_migration.py` | Practical derivative hedging context |

**Key implementation pattern from `open-risk/transitionMatrix`:**
- Uses the **Aalen-Johansen estimator** for empirical transition matrices from duration data
- Handles the conversion from datetime timestamps to continuous time floats via `datetime_to_float()`
- Supports both cohort-based (annual snapshots) and duration-based (continuous observation) estimation
- Built on pandas DataFrames with state columns as strings

**Recommendation for our implementation:**
- We don't need the full `transitionMatrix` library -- our state space is simpler (4 states: strong/adequate/weak/distress from `fh_altman_z_zone` history)
- Use the cohort approach: for each quarter, classify the company into a state based on Z-score zone, then count transitions between consecutive quarters
- Build a 4x4 matrix, normalize rows to get probabilities. With 8 quarters of history, we get 7 transitions -- enough for a rough matrix
- Add Laplace smoothing (add 0.01 to all cells) to avoid zero-probability transitions

### 1.3 Options-Implied Default Probability

**Best repos found:** Limited direct matches. The Carr & Wu (2011) approach is primarily implemented in academic Matlab code, not open-source Python.

**Alternative approach from research:**
- Instead of the full Carr-Wu model, use a simpler proxy: **deep OTM put spread as credit spread proxy**
- `put_price(K_low) - put_price(K_high)` for strikes well below current price approximates the probability of the stock falling below K_low
- This data is already available from our `options_signal_result` (yfinance options chain)

**Recommendation for our implementation:**
- Extract deep OTM put prices (strike < 70% of current price) from the existing options chain data
- `implied_pd = max_put_price / (strike_distance * exp(-r*T))` gives an approximate risk-neutral default probability
- Compare with Merton structural PD for model agreement signal
- No new dependencies needed -- uses existing yfinance options data from [`options_signals.py`](operator1/features/options_signals.py)

---

## Domain 2: Earnings Forensics

### 2.1 Dechow F-Score

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `Waguy02/Financial-Statements-Fraud-Detection` | `logistic_dechow.py` | Full ML pipeline using Dechow features, trained on SEC enforcement data |
| `hkalager/ML_AccountingFraud` | `MLFraud_module/__init__.py` | Academic-grade fraud detection with 50 financial features including RSST accruals, soft_assets, issuance |
| `TY-Cheng/reporting-risk-cascade` | `public_peer_comparison.py` | Corporate reporting risk measurement |

**Key implementation pattern from `hkalager/ML_AccountingFraud`:**
- Uses 50 financial features from Compustat, mapped to standard GAAP fields
- Key Dechow features already identifiable: `rsst` (RSST accruals), `dreceivables`, `dinventory`, `soft_assets`, `chcs` (change in cash sales), `droa`, `issuance`
- Uses `kpss` stationarity test from statsmodels
- The `adjust_serial` flag handles serial fraud (same company flagged across multiple years)

**Key implementation pattern from `Waguy02`:**
- Implements the Dechow probit as a PyTorch logistic regression -- overkill for us since we have the exact coefficients from the paper
- The preprocessing module `sec_financial_preprocessing_quarterly_dechow.py` maps SEC XBRL fields to the 8 Dechow variables

**Recommendation for our implementation:**
- Use the **exact probit coefficients** from Dechow et al. (2011) Table 7: `F = -7.893 + 0.790*RSST + 2.518*dREC + 1.191*dINV + 1.979*SOFT + ...`
- We already have most inputs: `accruals` (RSST proxy), `receivables` change, `inventory` change, `soft_asset_ratio` from derived_variables Stage 24
- Apply `scipy.stats.norm.cdf(F)` for probit probability
- No ML training needed -- the published coefficients are the model

### 2.2 Real Activities Manipulation (Roychowdhury 2006)

**Best repos found:** No direct Python implementation found. This is primarily implemented in Stata/SAS in academic settings.

**Implementation approach from the paper:**
- Three OLS regressions, each estimating "normal" levels, then computing residuals as "abnormal":
  1. `CFO/Assets = a0*(1/Assets) + a1*(Sales/Assets) + a2*(dSales/Assets) + e` -- residual = abnormal CFO
  2. `PROD/Assets = b0*(1/Assets) + b1*(Sales/Assets) + b2*(dSales/Assets) + b3*(dSales_lag/Assets) + e` -- residual = abnormal production
  3. `DISC/Assets = c0*(1/Assets) + c1*(Sales_lag/Assets) + e` -- residual = abnormal discretionary expenses

**Recommendation for our implementation:**
- Use `sklearn.linear_model.LinearRegression` for each of the 3 regressions
- Requires at least 8 quarterly observations (our standard lookback)
- Cross-sectional estimation (across peer companies) is ideal but we can use time-series per company
- Composite: `real_manipulation_score = normalize(abs(abnormal_cfo) + abs(abnormal_prod) + abs(abnormal_disc))`
- Negative abnormal CFO + positive abnormal production = overproduction manipulation
- Negative abnormal discretionary = cost cutting to inflate earnings

### 2.3 Beneish M-Score Extended

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `GabNoaro/financial-analysis` | `Beneish_M-Score.py` | Clean implementation with all 8 components |
| `raihanpka/satria-ml` | `beneish_m_score.py` | Production ML service implementation |
| `parkminhyung/python-code-for-finance` | `Beneish_M_Score.py` | Finance teaching implementation |

**Key implementation pattern from `GabNoaro`:**
- Computes all 8 indices: DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA
- Uses financial data from FMP API via pandas DataFrames
- Clean separation: each index computed as a separate ratio from consecutive years

**Recommendation for our implementation:**
- We already have M-8 in [`financial_health.py`](operator1/models/financial_health.py). Add the M-5 variant (5-variable subset that works better with limited data):
  `M5 = -6.065 + 0.823*DSRI + 0.906*GMI + 0.593*AQI + 0.717*SGI + 0.107*DEPI`
- Add probability conversion: `P(manipulator) = 1 / (1 + exp(-M))`
- Compare M5 vs M8 agreement as a robustness signal

---

## Domain 3: Valuation Engine

### 3.1 Stochastic DCF

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `Shreyansh100704/Agentic-Equity-Research-System` | `dcf_monte_carlo.py` | 10K-iteration MC DCF with stochastic WACC drawn from distribution |
| `Holzkopfblob/SOTP-Monte-Carlo-DCF-Model` | `monte_carlo_engine.py` | Vectorized numpy SOTP MC -- no Python loops over simulations |
| `Abelian-Analysis/Agentic-Investing-Framework` | `monte_carlo.py` | Another agentic DCF MC framework |

**Key implementation pattern from `Shreyansh100704`:**
- `DCFAssumptions` dataclass with `wacc_mean` AND `wacc_std` (typically 150bps std)
- `terminal_growth_mean` AND `terminal_growth_std` -- both stochastic
- Revenue growth sampled from N(growth_mean, growth_std) per simulation
- EBITDA margin sampled from N(margin_mean, margin_std)
- All 10K iterations vectorized via numpy, no Python loops
- Produces P10/P25/P50/P75/P90 distribution of intrinsic values

**Key implementation pattern from `Holzkopfblob` (SOTP MC):**
- `MonteCarloEngine` with segment-level DCFs -- each segment has its own growth/margin distribution
- Uses `scipy.stats.qmc` (quasi-random sampling) for better coverage of tail scenarios
- Computes `valuation_quality_score`, `implied_roic`, `prob_value_destruction`, `tv_ev_ratio`
- Fully vectorized: "The entire hot-path is numpy-vectorised -- no Python-level iteration over simulations"

**Recommendation for our implementation:**
- Upgrade `_compute_dcf()` to sample WACC from `N(wacc_mean, 0.015)` per simulation (currently flat)
- Add negative correlation between growth and WACC: when growth falls, spread widens. Use `np.random.multivariate_normal` with `cov = [[g_var, -0.3*g_std*w_std], [-0.3*g_std*w_std, w_var]]`
- Adopt the `valuation_quality_score` concept from Holzkopfblob
- For SOTP: use segment-specific multiples when `seg_result` has 2+ segments

### 3.2 EVA Decomposition

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `scoofy/wxStocks` | `CFA_functions.py` | CFA Level 2 implementation with `economic_value_added(NOPAT, cost_of_capital, total_capital)` |
| `Fincept-Corporation/FinceptTerminal` | `base_models.py` | Full terminal with EVA, MVA, residual income models |

**Key implementation pattern from `scoofy/wxStocks`:**
- Simple, correct: `EVA = NOPAT - (C_percent * TC)`
- Also implements `market_value_added = market_value - book_value`
- And `residual_income` with proper discounting: `V_0 = B_0 + sum(RI_t / (1+r)^t)`

**Recommendation for our implementation:**
- `NOPAT = operating_income * (1 - tax_rate)` where tax_rate estimated from `taxes / (net_income + taxes)`
- `IC = total_assets - current_liabilities` (simplified invested capital)
- `EVA = NOPAT - WACC * IC`
- Track `eva_spread = ROIC - WACC` (positive = value creation)
- Track `eva_momentum = EVA_current - EVA_4q_ago`

### 3.3 Sum of Parts (SOTP)

**Key implementation from `Holzkopfblob/SOTP-Monte-Carlo-DCF-Model`:**
- Full segment-level Monte Carlo with per-segment `SegmentConfig`: revenue growth distribution, margin distribution, capex ratio, depreciation
- `compute_segment_ev()` function values each segment independently
- `compute_corporate_costs_pv()` subtracts unallocated corporate overhead
- Equity bridge: `total_ev - net_debt = equity_value`
- `prob_value_destruction` per segment -- identifies segments destroying value

**Recommendation for our implementation:**
- When `seg_result` has 2+ segments: value each using segment revenue * sector EV/Revenue multiple
- Sector multiples from peer data (or hardcoded industry benchmarks as fallback)
- `sotp_value = sum(segment_values) - corporate_overhead`
- `conglomerate_discount = 1 - (market_cap / sotp_value)`

### 3.4 Reverse DCF

**Key implementation from `wangzhe3224/valueinvest`:**
- Uses binary search (bisection) to find the growth rate where `DCF_value == current_price`
- `_calculate_ev(fcf, g1, g2, g_term, r)` projects FCF forward 10 years in 2 stages
- `implied_growth_1_5` and `implied_growth_6_10` separated (stage 1 and 2)
- Returns `converged` boolean and `confidence` based on whether solution is in [0%, 30%] range

**Recommendation for our implementation:**
- Use `scipy.optimize.brentq` to solve `DCF(g) - market_cap = 0` for g in [-0.20, 0.50]
- `expectations_gap = implied_growth - actual_revenue_growth_yoy`
- Large positive gap = "priced for perfection"
- Large negative gap = "priced for failure" (potential value opportunity)

---

## Domain 4: Risk Management

### 4.1 Component CVaR

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `skfolio/skfolio` | `skfolio_risk.py` | Comprehensive risk measures library (11K+ stars) |
| `Fincept-Corporation/FinceptTerminal` | `skfolio_risk.py` | Uses skfolio for risk decomposition |

**Recommendation for our implementation:**
- Component CVaR = `CVaR_i = w_i * E[r_i | portfolio_return < VaR]`
- Use the historical simulation approach: identify worst 5% of days, compute average contribution of each risk factor on those days
- Risk factors: return_1d, revenue_growth_change, margin_change, leverage_change, macro_quadrant_shift
- `component_cvar[factor] = mean(factor_contribution | portfolio_return < VaR_5%)`
- No new dependencies -- pure numpy/pandas

### 4.2 CDaR (Conditional Drawdown-at-Risk)

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `skfolio/skfolio` | `_measures.py` | `cdar()` function -- CDaR is simply CVaR applied to drawdowns |
| `PyPortfolio/PyPortfolioOpt` | `efficient_cdar.py` | Full CDaR optimization frontier (Chekhlov 2005 formulation) |

**Key insight from `skfolio`:**
```python
def cdar(drawdowns, beta=0.95):
    return cvar(returns=drawdowns, beta=beta)
```
CDaR is literally CVaR computed on the drawdown series instead of the return series. Elegant.

**Recommendation for our implementation:**
- Compute drawdown series from MC paths: `dd = cummax - cumulative_return`
- `cdar_95 = np.percentile(max_drawdowns_per_path, 95)`
- `cdar_adjusted_kelly = kelly_fraction * (target_max_dd / cdar_95)` -- scales Kelly by drawdown tolerance
- Uses existing MC paths from `mc_result.terminal_values`

### 4.3 Risk Parity / Budgeting

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `skfolio/skfolio` | `risk_parity_variance.py` | Risk parity with multiple risk measures |
| `ArturSepp/OptimalPortfolios` | `risk_budgeting_pyrb_vs_scipy.py` | Comparison of `pyrb` vs scipy for risk budgeting |

**Recommendation for our implementation:**
- Not portfolio-level (we analyze single stocks), so adapt to **risk factor budgeting**
- Allocate "analytical attention" (computational budget) proportional to each risk factor's contribution
- Uses Component CVaR output from 4.1
- `risk_budget[factor] = component_cvar[factor] / total_cvar`
- Per-regime budgets: survival regime concentrates budget on liquidity/solvency factors

---

## Domain 5: Capital Cycle & Competitive Positioning

### 5.1 Capital Cycle Classification

**Limited direct implementations found.** The Marathon Asset Management framework is proprietary.

**Recommendation for our implementation:**
- Classification from CapEx/Revenue trend (slope) + Gross Margin trend (slope) over 8Q:
  - **Invest**: CapEx/Rev rising, margin stable/rising (spending to grow)
  - **Harvest**: CapEx/Rev peaking/flat, margin rising (reaping benefits)
  - **Restructure**: CapEx/Rev falling, margin falling (cutting losses)
  - **Rebuild**: CapEx/Rev troughing, margin recovering (turnaround)
- Use `compute_rolling_slope()` from existing helpers
- Confirm with ROIC trajectory: invest phase should show declining then improving ROIC

### 5.2 Economic Moat Quantification

**Repos found were LLM-based (prompt engineering).** No quantitative financial-data moat scoring found.

**Recommendation for our implementation:**
- Build from financial fundamentals (no LLM needed):
  - **Pricing Power** (30%): gross margin stability over 8Q (low std = pricing power). `score = max(0, 100 - std(gross_margin) * 1000)`
  - **Switching Costs** (25%): revenue retention (low revenue volatility vs peers). `score = max(0, 100 * (1 - cv(revenue) / peer_cv(revenue)))`
  - **Cost Advantage** (25%): operating margin vs peer median. `score = percentile_rank(op_margin, peer_margins) * 100`
  - **Network Effects** (20%): revenue acceleration (2nd derivative positive). `score = 100 if revenue_accel > 0 else 50 * (1 + revenue_accel)`
- Composite: weighted average. Track `moat_trend` via rolling 4Q slope.

### 5.3 Sustainable Growth Rate

**Repos found:**
| Repo | File | Description |
|------|------|-------------|
| `Fincept-Corporation/FinceptTerminal` | `base_models.py` | `SGR = ROE * (1 - payout_ratio)` |
| `Alethevia/Stock_Deepseeker` | `growth.py` | Growth factor analysis |

**Recommendation:** Straightforward formula. `SGR = ROE * (1 - dividends_paid / net_income)`. Compare to actual revenue growth. Gap > 5% = funding concern flag.

---

## Domain 6: Multi-Frequency Expansion

### 6.1-6.3 FCF Quality / Accruals / Growth Quality MF

No specific multi-frequency variants found on GitHub. Our multi-freq merge patterns from [`multi_freq_metrics.py`](operator1/hedge_fund/multi_freq_metrics.py) (weighted regression, inverse-variance, worst-case envelope) are already ahead of what's publicly available.

### 6.4 Frequency Coherence Test

**Recommendation:** Before merging, compute Spearman rank correlation of component scores across frequencies. If rho < 0.3, flag disagreement and use single-freq (quarterly preferred). This prevents harmful averaging when frequencies give contradictory signals.

---

## Domain 7: Macro Sensitivity

### 7.1 Multi-Factor Exposure

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `ArturSepp/QuantInvestStrats` | `factor_model.py` | Production-grade `LinearModel` with EWM loadings, factor attribution, rolling R-squared, residual analysis |
| `liangdabiao/autogen-financial-analysis` | `factor_models.py` | Multi-agent financial analysis with factor decomposition |

**Key implementation pattern from `ArturSepp/QuantInvestStrats`:**
- `LinearModel` dataclass with `x` (factors), `y` (asset returns), `loadings` (time-varying)
- Uses EWM (exponentially weighted) regression for time-varying betas -- adapts to changing factor exposures
- `get_factor_alpha()` computes unexplained return after removing factor contributions
- `get_model_ewm_r2()` tracks model fit over time
- `compute_agg_factor_exposures()` for portfolio-level aggregation

**Recommendation for our implementation:**
- Use 63-day rolling OLS of `return_1d ~ yield_curve_change + usd_momentum + sector_return + vix_change`
- Factor data already in cache from cross_asset_signals
- Output: `factor_betas` dict, `factor_r_squared`, `dominant_factor`, `macro_vulnerability = sum(abs(betas))`
- EWM variant (halflife=21) for more responsive estimates

### 7.2 Equity Duration

**Key implementation from `Fincept-Corporation/FinceptTerminal`:**
- Simple empirical approach: regress stock returns on interest rate changes
- `equity_duration = -beta(return, d_yield) / (1 + yield)` -- analogous to bond duration

**Recommendation:** Compute from existing data. `duration_beta = rolling_cov(return_1d, yield_curve_change, 252) / rolling_var(yield_curve_change, 252)`. Growth stocks: positive duration (hurt by rate hikes). Banks: negative.

### 7.3-7.4 Commodity / FX Exposure

**Limited implementations found.** Straightforward rolling regression approach.

**Recommendation:** Same pattern as 7.1 but with commodity ETF returns (USO, GLD) and currency returns as factors. Already have cross_asset_signals infrastructure.

---

## Domain 8: Governance & ESG

### 8.1 Governance Score

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `Kjain02/AIproject` | `governance_score.py` | LLM-based governance scoring (Azure OpenAI) |
| `DrewThomas09/RCM` | `board_governance.py` | Data-driven governance scoring |

**Key insight from `Kjain02`:** Uses LLM to score 7 governance dimensions: board independence, executive compensation, shareholder rights, audit quality, ethical practices, succession planning, stakeholder engagement. Returns scores 0-100 per dimension.

**Recommendation for our implementation:**
- We don't want LLM dependency for this. Build from available financial data:
  - **Insider alignment** (40%): already have `insider_alignment` in advanced_methods -- net insider buying trend
  - **Ownership concentration** (30%): `inst_crowding_risk` from institutional_flow -- high concentration = governance risk
  - **Capital discipline** (30%): vanity score (inverted) -- low vanity = good governance
- Composite: weighted average. All inputs already exist in pipeline.

### 8.2 Capital Allocation Quality

**Recommendation:** Enhance existing vanity `capital_misallocation` component:
- **Buyback timing**: compute correlation of buyback quarters (from cashflow_df) vs PE percentile. Buying when cheap = good timing.
- **Dividend consistency**: coefficient of variation of dividend payments over 8Q.
- **Reinvestment spread**: ROIC - WACC. Positive = creating value with reinvested capital.

### 8.3 Accounting Conservatism

**No Python implementations found** for Khan & Watts C-Score specifically.

**Recommendation:** Use simplified Basu (1997) asymmetric timeliness:
- `conservatism = corr(negative_returns, earnings_change) - corr(positive_returns, earnings_change)`
- Conservative accounting: bad news reflected in earnings faster than good news
- `conservatism_index`: negative returns should have higher R-squared with earnings changes than positive returns
- Implement as rolling 252-day window

---

## Domain 9: Fusion & Signal Quality

### 9.1 HRP Signal Combination

**Best repos found:**
| Repo | File | Description |
|------|------|-------------|
| `bsdz/yabte` | `hierarchical_risk_parity.py` | Clean 3-function HRP: `_getIVP`, `_getClusterVar`, `_getRecBipart` directly from Lopez de Prado (2016) |
| `shreejitverma/Dynamic-Portfolio-Optimization` | `risk_parity.py` | HRP for portfolio optimization |

**Key implementation from `bsdz/yabte` (directly from the paper):**
```python
def hrp(corr, sigma):
    dist = ((1 - corr) / 2.0) ** 0.5  # correlation distance
    link = linkage(dist, method='single')
    sortIx = to_tree(link).pre_order()  # quasi-diagonalization
    return _getRecBipart(cov, sortIx)  # recursive bisection
```
Three steps: (1) hierarchical clustering on correlation distance, (2) quasi-diagonalization, (3) recursive bisection for weights.

**Recommendation for our implementation:**
- Apply HRP to SIGNAL correlation matrix (not asset returns)
- Build correlation matrix of individual HF metric scores (FCF quality, accruals, momentum, etc.)
- HRP assigns lower weight to correlated signals, higher weight to uncorrelated ones
- Uses `scipy.cluster.hierarchy.linkage` + `to_tree` -- already installed
- Replace equal-weight signal combination in fusion with HRP weights

### 9.2 Brier Score Calibration

**Key implementations found use Platt scaling (logistic calibration).**

**Recommendation:**
- Track historical conviction vs actual outcome: did conviction=8 predictions actually hit 80% of the time?
- `brier_score = mean((predicted_probability - actual_outcome)^2)`
- If `brier_score > 0.25`, apply Platt scaling: fit logistic `P_calibrated = 1 / (1 + exp(-a*P_raw - b))` on historical data
- Uses `prediction_log_summary` (already in pipeline) as training data

### 9.3 Adaptive Scorecard Weights

**No direct implementation found** for rolling IC-based weight optimization in HF scoring context.

**Recommendation:**
- For each tier score (T1-T5), compute rolling 252-day Spearman IC vs forward 21-day return
- `IC_t = spearman_corr(tier_score_t, return_21d_forward_t)` over rolling window
- Normalize ICs: `weight_i = max(0, IC_i) / sum(max(0, IC_j))`
- Fallback to fixed weights when < 252 days of history
- This makes the scorecard self-correcting: tiers that actually predict returns get more weight

---

## Summary: Key Takeaways Across All 30 Methods

### Libraries Already Installed That We Should Leverage More

| Library | Used For | Currently Used? |
|---------|----------|-----------------|
| `scipy.optimize.brentq` | Root-finding for Merton term structure, reverse DCF | No (uses Newton-Raphson) |
| `scipy.cluster.hierarchy` | HRP signal clustering | No |
| `scipy.stats.norm.cdf` | Probit probability for Dechow F-Score | Yes (for Merton DD) |
| `sklearn.linear_model` | Factor exposure regressions, Roychowdhury normal models | Yes (for forecasting) |
| `numpy.random.multivariate_normal` | Correlated growth/WACC sampling in stochastic DCF | No |

### Implementation Patterns Worth Adopting

1. **Vectorized MC from `Holzkopfblob`**: "No Python-level iteration over simulations." Our DCF MC should be fully vectorized.
2. **Parameterized-T Merton from `k-bargaoui`**: d1/d2 with explicit time parameter enables term structure with zero refactoring.
3. **CDaR = CVaR on drawdowns from `skfolio`**: Beautifully simple. One line reuses existing CVaR infrastructure.
4. **HRP 3-function pattern from Lopez de Prado via `bsdz/yabte`**: Cluster -> quasi-diag -> recursive bisection. Clean, tested, ~40 lines.
5. **EWM rolling factor model from `ArturSepp`**: Time-varying betas that adapt to changing regimes.
6. **Stochastic WACC dataclass from `Shreyansh100704`**: `wacc_mean` + `wacc_std` as explicit parameters, not fixed constants.

### Methods That Can Be Implemented in Under 30 Lines Each

| Method | Lines | Reason |
|--------|-------|--------|
| 3.2 EVA | ~15 | `EVA = NOPAT - WACC * IC`, three lookups |
| 5.3 SGR | ~10 | `SGR = ROE * (1 - payout)`, one formula |
| 4.2 CDaR | ~10 | `cvar(drawdowns, 0.95)` on MC paths |
| 8.1 Governance | ~20 | Weighted average of 3 existing scores |
| 9.2 Brier | ~25 | Historical accuracy tracking + logistic calibration |

### Methods That Need More Careful Implementation (>100 lines)

| Method | Lines | Reason |
|--------|-------|--------|
| 2.2 Real Activities Manipulation | ~150 | 3 separate OLS regressions + abnormal residual computation |
| 3.1 Stochastic DCF | ~120 | Multivariate sampling + vectorized projection |
| 3.3 SOTP | ~130 | Per-segment valuation with sector multiples |
| 7.1 Factor Exposure | ~100 | Rolling OLS + EWM variant + attribution |
| 9.1 HRP | ~80 | Clustering + quasi-diag + recursive bisection on signal correlation |

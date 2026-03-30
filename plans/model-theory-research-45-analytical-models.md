# Model Theory Research -- 45 Analytical Models

Deep theoretical analysis of every analytical model in the Operator 1 pipeline. For each model: foundational theory, the specific method implemented, non-standard conditions and adaptations that serve the pipeline's survival-analysis objectives, and improvement opportunities.

---

## Layer 1: Feature Engineering

### Model 1 -- Derived Variables (`derived_variables.py`)

**Foundational Theory.** Financial ratio analysis traces to Benjamin Graham and David Dodd's *Security Analysis* (1934). The core idea: raw financial statement numbers are meaningless in isolation; ratios normalize them for cross-company and cross-time comparison. The `safe_ratio` abstraction enforces a numerical safety discipline from Sec 15 of the spec: when a denominator is null, zero, or smaller than epsilon (1e-9), the result is null and companion flags (`is_missing_*`, `invalid_math_*`) record why.

**TTM (Trailing Twelve Months).** The `_rolling_4q_ttm` helper implements a non-trivial computation on as-of-aligned data. Standard TTM just sums four quarterly values. The challenge here is that the daily cache carries forward-filled quarterly values (e.g., Q1 revenue repeats daily until Q2 is filed). The algorithm detects quarter transitions (where the value changes), extracts the distinct quarterly values, applies a rolling 4-quarter sum, and broadcasts back to the daily index. When fewer than 4 quarters are available, it scales proportionally (e.g., 2 quarters * 2). This is a non-standard approach -- most implementations assume quarterly data is already aggregated. The pipeline's approach handles the as-of-join reality where you have daily observations but quarterly underlying data.

**Technical Indicators.** ADX (Average Directional Index, Welles Wilder 1978) measures trend strength regardless of direction. OBV (On-Balance Volume, Joe Granville 1963) uses volume flow to predict price changes. Bollinger Band width (John Bollinger 1983) measures volatility via the spread between upper and lower bands. MACD histogram (Gerald Appel 1979) captures momentum divergence. These are computed via the `ta` library. The non-standard aspect: these technical indicators are injected into a *fundamental analysis* pipeline. The pipeline treats them not as trading signals but as features for temporal models (LSTM, XGB, transformers) that learn from both fundamental and technical patterns.

**Non-standard adaptation for our objectives.** The epsilon-guarded `safe_ratio` with companion flags means downstream models always know whether a value is real or the result of a degenerate computation. This prevents silent propagation of infinity or NaN through the survival detection chain. Every ratio feeds survival thresholds (current_ratio < 1.0, debt_to_equity_abs > 3.0, etc.) so numerical garbage here would produce false survival triggers.

---

### Model 2 -- Conflict Risk Assessment (`conflict_risk.py`)

**Foundational Theory.** Geopolitical risk quantification draws from political science (Caldara and Iacoviello's Geopolitical Risk Index, 2022) and conflict studies (Uppsala Conflict Data Program). The core insight is that armed conflict, sanctions, and state fragility create existential risks that pure financial metrics miss entirely. A company with perfect balance sheet ratios in a war zone faces risks that current_ratio cannot capture.

**Methods.** The module combines four data sources into a weighted intensity score:

1. **World Bank FCS (Fragile and Conflict-affected Situations) List** -- 31 countries classified as fragile/conflict-affected. Static list, updated annually by the World Bank. Binary flag.

2. **OFAC/EU Sanctions Lists** -- 13 countries under major international sanctions (Iran, North Korea, Russia, etc.). Sanctions create legal risk (compliance violations), financial risk (blocked transactions), and supply chain risk.

3. **UCDP GED (Georeferenced Event Dataset)** -- Academic-grade conflict event data from Uppsala University. Provides event counts, fatality counts, and conflict type classification (low_intensity, civil_war, interstate_war). The trend computation (escalating/stable/de-escalating) compares 30-day vs 90-day event counts.

4. **GDELT (Global Database of Events, Language, and Tone)** -- Real-time news monitoring. Provides conflict mention counts and average tone (negative tone correlates with conflict severity). Rate-limited to 5 req/sec.

**Weighted intensity formula:** `0.40 * event_score + 0.20 * fatality_score + 0.25 * flag_score + 0.15 * news_score`. The 40% weight on events (rather than fatalities) is deliberate: many conflicts are economically devastating without mass casualties (sanctions, blockades, infrastructure destruction).

**Non-standard adaptation.** The linked entity conflict propagation (`assess_linked_entity_conflict`) is unique. It checks each linked entity's country against conflict lists, then classifies risk by relationship type: suppliers in war zones create supply chain risk, customers in conflict zones create revenue exposure risk, and competitors in conflict zones create competitive advantage opportunities. This multi-hop propagation of geopolitical risk through the corporate graph is rare in production systems.

---

### Model 3 -- Filing Calendar (`filing_calendar.py`)

**Foundational Theory.** Information freshness in financial analysis. The Efficient Market Hypothesis (Fama 1970) assumes information is promptly incorporated into prices, but regulatory filings have structural delays. Quarterly filers publish with 45-60 day lags; annual filers with 90+ day lags. This module quantifies the information gap.

**Methods.** Frequency detection uses median filing gap analysis: < 120 days = quarterly, 120-250 days = semiannual, 250-500 days = annual. Coverage ratio = actual_filings / expected_filings over a 2-year window. Staleness detection uses market-specific thresholds (US quarterly: 100 days, UK semi-annual: 210 days, EU annual: 400 days). Gap detection identifies periods where an expected filing is missing.

**Non-standard adaptation.** The `inject_filing_freshness` function writes a daily `filing_freshness` column that decays from 1.0 (filing day) toward 0.0 as time passes since the last filing. This provides temporal models with a continuous signal about how trustworthy today's financial data is -- fresh filings have high confidence, stale data has low confidence. This is particularly important for the Frequency Interpolator which uses filing distance to modulate interpolation confidence.

---

### Model 4 -- Linked Aggregates (`linked_aggregates.py`)

**Foundational Theory.** Cross-sectional analysis and relative valuation (Damodaran, *Investment Valuation*, 2012). A company's financial metrics are most meaningful when compared to peers. The linked aggregates module computes group-level statistics (mean, median) across entity relationship groups (competitors, suppliers, customers, financial institutions, sector peers).

**Methods.** For each entity group, compute mean and median of key variables (return_1d, volatility_21d, pe_ratio_calc, current_ratio, etc.) across group members, aligned to the daily index. The `compute_relative_metrics` function (now wired in our PR) computes:

- **Relative strength** = company return - sector average return (momentum relative to peers)
- **Valuation premium** = company PE - industry median PE (over/undervaluation signal)
- **Relative volatility** = company vol / sector average vol (risk relative to peers)

**Non-standard adaptation.** The linked aggregate columns (e.g., `competitors_avg_return_1d`) are injected into the cache and passed as `_extra_vars` to temporal models. This means the LSTM, VAR, and XGB models learn from cross-entity signals -- a competitor's declining returns may predict the target's future performance. This cross-entity feature injection is uncommon in standard time series forecasting.

---

### Model 5 -- Macro Alignment (`macro_alignment.py`)

**Foundational Theory.** Macro-financial linkage (Bernanke and Gertler 1989, the financial accelerator). Corporate survival depends not just on internal metrics but on the macroeconomic environment: GDP growth, inflation, interest rates, unemployment, and exchange rates. The challenge: macro data is published at annual or quarterly frequency while the pipeline operates daily.

**Methods.** As-of join: for each daily row, the latest available macro value with `year <= day's year` is attached. This prevents look-ahead bias. The module also computes `inflation_rate_daily_equivalent = inflation_rate_yoy / 365` and `real_return_1d = return_1d - inflation_rate_daily_equivalent`, providing a purchasing-power-adjusted return metric.

**Non-standard adaptation.** All macro variables get `is_missing_*` companion columns, consistent with the pipeline's philosophy that the model should always know what data is real vs absent. The real_return_1d computation adjusts stock returns for inflation -- this is standard in academic finance (Fisher 1930, the Fisher equation) but rare in automated analysis pipelines.

---

### Model 6 -- Macro Quadrant (`macro_quadrant.py`)

**Foundational Theory.** Macroeconomic regime classification draws from the Goldilocks framework and business cycle theory (Burns and Mitchell 1946). The four quadrants map GDP growth (above/below trend) against inflation (above/below target):

| | Low Inflation | High Inflation |
|---|---|---|
| **High Growth** | Goldilocks | Overheating |
| **Low Growth** | Disinflation | Stagflation |

**Non-standard adaptation.** The quadrant classification is injected as a daily cache column and fed to temporal models. In the survival framework, stagflation (low growth + high inflation) triggers country survival mode because companies face simultaneous revenue pressure (low growth) and cost pressure (high inflation). The stability score modulates how much weight downstream models give to macro signals.

---

### Model 7 -- News Sentiment (`news_sentiment.py`)

**Foundational Theory.** Textual sentiment analysis of financial news (Loughran and McDonald 2011, financial-domain sentiment lexicons; Tetlock 2007, media content predicting stock prices). The core insight: news sentiment contains forward-looking information not yet reflected in financial statements.

**Methods.** Three-tier approach:

1. **LLM scoring (preferred)**: Send article titles/snippets to Gemini/Claude with a financial sentiment prompt. Returns -1.0 to +1.0 per article. Highest quality but API-cost-limited.

2. **VADER (fallback)**: Valence Aware Dictionary and sEntiment Reasoner (Hutto and Gilbert 2014). Rule-based sentiment analyzer that handles: negation ("not good" is negative), intensity modifiers ("very good" > "good"), punctuation ("good!!!" > "good"), and capitalization ("GOOD" > "good").

3. **Keyword matching (final fallback)**: Simple positive/negative word counting with financial-domain word lists.

**Non-standard adaptation.** The sentiment is computed as a daily time series (`sentiment_score`, `sentiment_momentum`) injected into the cache. Temporal models can learn from sentiment-return lead-lag relationships. The pipeline also computes `sentiment_gap` (vanity module) measuring the divergence between sentiment and actual financial performance -- a large gap may indicate hype or hidden problems.

---

### Model 8 -- Peer Ranking (`peer_ranking.py`)

**Foundational Theory.** Percentile ranking within a reference group (Tobin 1958, relative performance evaluation). A company's current_ratio of 1.5 means little in isolation -- it could be excellent (in capital-intensive industries) or mediocre (in asset-light software). Peer percentile ranking normalizes each metric relative to the company's competitive context.

**Non-standard adaptation.** The peer ranking feeds the adaptive threshold calibration (Model 18a). Instead of using fixed textbook thresholds (current_ratio < 1.0 = distress), the adaptive system calibrates thresholds to the P10/P90 of the peer distribution. A company in a sector where current_ratio of 0.8 is normal should not be flagged as distressed -- the peer percentile context prevents false positives.

---

### Model 9 -- Private Company Proxies (`private_company_proxies.py`)

**Foundational Theory.** Private company valuation (Damodaran 2009, *The Dark Side of Valuation*). Private companies lack market prices (no OHLCV data), which breaks most financial analysis models that depend on close prices, returns, and volatility. The solution: construct proxy variables from available balance sheet and income statement data.

**Methods.** Three proxy variables: equity_value = total_equity (book equity as liquidation value), equity_change_rate = daily change in equity value (proxy for return_1d), financial_volatility = rolling std of equity_change_rate (proxy for volatility_21d).

**Non-standard adaptation.** The `resolve_proxies` function transparently writes these proxy values into the standard column names (close, return_1d, volatility_21d) so all downstream models work without modification. Most analysis platforms simply refuse to analyze private companies.

---

### Model 10 -- SIX Derived Proxies (`six_derived_proxies.py`)

**Foundational Theory.** For SIX-listed Swiss companies, the available data is limited to dividend history and capital events. This module reconstructs 22 canonical financial fields using: Kalman Filter (Kalman 1960), Merton Model (Merton 1974), Unscented Kalman Filter (Julier and Uhlmann 1997), PELT (Killick et al. 2012), L1 Trend Filtering (Kim et al. 2009), and Edwards-Bell-Ohlson Model (Ohlson 1995).

**Non-standard adaptation.** This module essentially performs "financial statement reconstruction" from minimal data -- a research-grade capability. The 3,848-line implementation is the largest single module in the pipeline. It's unique to markets (like SIX) where structured XBRL filings aren't available but dividend/capital history is.

---

## Layer 2: Analysis Modules

### Model 11 -- Survival Mode (`survival_mode.py`)

**Foundational Theory.** Corporate distress prediction: Altman Z-Score (1968), Ohlson O-Score (1980), Merton's Distance-to-Default (1974). The pipeline takes a hybrid approach: rule-based triggers (immediate, interpretable) combined with Cox Proportional Hazards (data-driven, calibrated to the company's own history).

**Methods.** (1) Rule-based survival flag: OR of 6 conditions. (2) Sigmoid survival probability: continuous 0-1 score. (3) Cox PH survival score via lifelines (Cox 1972): semi-parametric proportional hazards model learning hazard ratios from the company's own distress episodes. (4) Blended probability: 0.4 * sigmoid + 0.6 * cox.

**Non-standard adaptation.** Most survival models are cross-sectional (trained on populations of firms). The pipeline trains Cox PH on the company's own time series, treating each day as an observation and survival_mode_flag as the event indicator. The trade-off is that companies with few distress episodes produce poorly calibrated Cox models -- the sigmoid fallback mitigates this.

---

### Model 12 -- Hierarchy Weights (`hierarchy_weights.py`)

**Foundational Theory.** Hierarchical risk budgeting (Roncalli 2013). Four regimes: normal (equal 20% weights), company_survival (50/30/15/4/1), modified_survival (40/35/20/4/1), extreme_survival (60/30/10/0/0).

**Non-standard adaptation.** The Sobol sensitivity feedback loop adjusts these weights toward data-driven importance after temporal models run. The vanity adjustment further shifts weight from tiers 4-5 to tier 1 when the company shows signs of capital misallocation.

---

### Model 13 -- Survival Timeline (`survival_timeline.py`)

**Foundational Theory.** Finite state machines applied to financial regime classification. The module bridges rule-based survival flags (deterministic) and HMM regime labels (probabilistic). Six base modes combined with four market regimes produce 11 combined states with continuous survival intensity (0.0 to 1.0).

**Non-standard adaptation.** The regime_transition_prob column provides temporal models with a "regime change risk" signal distinct from volatility. The 30-entry `_COMBINED_STATE_MAP` codifies domain knowledge about how financial and market regimes interact.

---

### Model 14 -- Fuzzy Protection (`fuzzy_protection.py`)

**Foundational Theory.** Mamdani fuzzy inference (Mamdani and Assilian 1975). Three input variables (sector strategicness, economic significance, policy responsiveness) with 11 interaction rules. Defuzzification via centroid method produces a continuous protection degree in 0 to 1.

**Non-standard adaptation.** The scikit-fuzzy Mamdani engine captures the nuanced reality of government support that binary classification misses. A company with protection_degree >= 0.5 in a country under crisis gets classified into "protected" modes, changing hierarchy weights and Monte Carlo survival thresholds.

---

### Model 15 -- Ethical Filters (`ethical_filters.py`)

**Foundational Theory.** Islamic finance screening (AAOIFI Sharia Standard No. 21). Four filters: Purchasing Power, Solvency, Gharar, Cash is King.

**Non-standard adaptation.** Beyond ethical screening, these serve as quality signals. A company failing the "Cash is King" filter (net income much higher than operating cash flow) likely has aggressive accrual accounting -- the same signal the Beneish M-Score detects.

---

### Model 16 -- Economic Planes (`economic_planes.py`)

**Foundational Theory.** Sector classification into five planes: Real Economy, Financial, Technology, Resources, Services.

**Non-standard adaptation.** The plane classification adjusts model weights per sector. Financial companies get higher GARCH weight; technology companies get higher LSTM/transformer weight; resources companies get higher cycle decomposition weight.

---

### Model 17 -- Vanity Score (`vanity.py`)

**Foundational Theory.** Capital allocation quality assessment (Buffett's "return on retained capital" framework). Five components: R&D Mismatch (15%), SGA Bloat (25%), Capital Misallocation (30%), Competitive Decay (15%), Sentiment Gap (15%).

**Non-standard adaptation.** High vanity companies in survival mode get even more weight on tier 1 (liquidity) because wasteful management is less likely to navigate a crisis effectively.

---

### Model 18a -- Adaptive Thresholds (`adaptive_thresholds.py`)

**Foundational Theory.** Five calibration methods: (A) Peer Percentile with MAD (Huber 1981), (D) BOCPD Deterioration Tightening (Adams and MacKay 2007), (E) Sector Z-Score (Iglewicz and Hoaglin 1993), (H) Jenks Natural Breaks (Fisher 1958), (J) HMM Emission Crossover (Rabiner 1989).

---

### Model 18b -- Adaptive Model Parameters (`adaptive_model_params.py`)

**Foundational Theory.** Ten calibration methods: Kish Effective Sample Size (1965), Inverse-Variance Blending (Cochrane 1954), Lambda PID Tuning (Dahlin 1968), Copula Tail Contagion (Joe 2014), Amihud Participation Rate (2002), Regime Risk Multiplier, Garman-Klass Factor (1980), Transition Half-Life, Precision-Targeted MC (Glasserman 2003).

---

### Model 18c -- Adaptive Windows (`adaptive_windows.py`)

**Foundational Theory.** Nyquist-Shannon Anchored Windows, Scaling-Law NN Architecture (Kaplan et al. 2020), Innovation-Based Particle Noise (Mehra 1970), Distribution-Based Pattern Thresholds (Bulkowski 2008).

---

## Layer 3: Temporal Models

### Model 19 -- Regime Detector (`regime_detector.py`)

**Foundational Theory.** Five methods: HMM (Baum et al. 1970), GMM (Dempster et al. 1977), PELT (Killick et al. 2012), BCP (Adams and MacKay 2007), ChangeFinder (Yamanishi and Takeuchi 2002).

---

### Model 20 -- Regime Mixer (`regime_mixer.py`)

**Foundational Theory.** Dual-regime classification separates market regimes from fundamental regimes with soft blended weights.

---

### Model 21 -- Granger Causality (`granger_causality.py`)

**Foundational Theory.** Granger causality (1969) upgraded with PCMCI (Runge et al. 2019) via tigramite. Time-varying rolling-window causal discovery.

---

### Model 22 -- Transfer Entropy (`causality.py`)

**Foundational Theory.** Transfer entropy (Schreiber 2000): information-theoretic measure of directed information flow capturing nonlinear dependencies.

---

### Model 23 -- Model Synergies (`model_synergies.py`)

**Foundational Theory.** Feature engineering and model cooperation: cycle phase injection, unified causal network from Granger + TE, peer-adjusted survival thresholds, plane-aware model weighting.

---

### Model 24 -- Forecasting (`forecasting.py`)

**Foundational Theory.** Seven model types: Kalman (1960), GARCH(1,1) (Bollerslev 1986), VAR (Sims 1980), LSTM (Hochreiter and Schmidhuber 1997), Tree ensembles (Breiman 2001, Chen and Guestrin 2016), AutoARIMA via statsforecast, Dynamic Factor Model.

---

### Model 25 -- Walk Forward (`walk_forward.py`)

**Foundational Theory.** Walk-forward analysis (Pardo 2008) with mode-conditioned evaluation and Model Confidence Sets (Hansen et al. 2011).

---

### Model 26 -- Monte Carlo (`monte_carlo.py`)

**Foundational Theory.** Regime-aware path generation with importance sampling (Glasserman 2003). Multivariate MC jointly simulates financial ratios via copula correlation.

---

### Model 27 -- PID Controller (`pid_controller.py`)

**Foundational Theory.** PID control (Ziegler and Nichols 1942) with Lambda tuning (Dahlin 1968) for adaptive learning rates.

---

### Model 28 -- Transformer Forecaster (`transformer_forecaster.py`)

**Foundational Theory.** Temporal Transformer (Vaswani et al. 2017) with Kaplan scaling law sizing.

---

### Model 29 -- Genetic Optimizer (`genetic_optimizer.py`)

**Foundational Theory.** Optuna TPE (Bergstra et al. 2011) and Genetic Algorithm (Holland 1975) with per-regime weight optimization.

---

### Model 30 -- OHLC Predictor (`ohlc_predictor.py`)

**Foundational Theory.** Next-day candlestick prediction combining point forecasts with cycle-aware volatility envelopes.

---

### Model 31 -- Conformal Prediction (`conformal.py`)

**Foundational Theory.** Distribution-free prediction intervals (Vovk et al. 2005) with PID-controlled coverage (Angelopoulos et al. 2023) and Mondrian partitioning per survival mode.

---

### Model 32 -- Copula (`copula.py`)

**Foundational Theory.** Sklar's Theorem (1959). Three families: Gaussian, Student-t, Clayton (1978) with AIC-based model selection.

---

### Model 33 -- Particle Filter (`particle_filter.py`)

**Foundational Theory.** Sequential Monte Carlo (Gordon et al. 1993) tracking survival-related variables with Mehra-calibrated noise.

---

### Model 34 -- Cycle Decomposition (`cycle_decomposition.py`)

**Foundational Theory.** EMD/CEEMDAN (Wu and Huang 2009) preferred over FFT (Cooley and Tukey 1965) for non-stationary financial data.

---

### Model 35 -- Pattern Detector (`pattern_detector.py`)

**Foundational Theory.** Candlestick patterns (Nison 1991) with Matrix Profile motif/discord discovery (Yeh et al. 2016, stumpy).

---

### Model 36 -- SHAP Explainability (`explainability.py`)

**Foundational Theory.** Shapley values (Shapley 1953) via TreeExplainer and KernelExplainer (Lundberg and Lee 2017).

---

### Model 37 -- Sobol Sensitivity (`sensitivity.py`)

**Foundational Theory.** Sobol variance-based global sensitivity analysis (Sobol 1993) with Saltelli sampling (2010). Feedback loop adjusts hierarchy weights.

---

### Model 38 -- Financial Health (`financial_health.py`)

**Foundational Theory.** Altman Z-Score (1968), Beneish M-Score (1999), adaptive PE/EV caps via Log-Normal P99.5 (Aitchison and Brown 1957), Tukey Extreme Fence (1977), MAD-Based Cap (Iglewicz and Hoaglin 1993).

---

### Model 39 -- Graph Risk (`graph_risk.py`)

**Foundational Theory.** CoVaR (Adrian and Brunnermeier 2016), SRISK (Brownlees and Engle 2017), edge-weighted contagion via SIR-like cascade.

---

### Model 40 -- Game Theory (`game_theory.py`)

**Foundational Theory.** Cournot (1838), Bertrand (1883), Stackelberg (1934) competition models with CR4 market structure classification.

---

### Model 41 -- DTW Analogs (`dtw_analogs.py`)

**Foundational Theory.** Dynamic Time Warping (Sakoe and Chiba 1978) with cross-company analog search using peer caches weighted at 0.7x.

---

### Model 42 -- Prediction Aggregator (`prediction_aggregator.py`)

**Foundational Theory.** Fixed Share Forecaster (Herbster and Warmuth 1998) for online model weighting with Model Confidence Sets filtering (Hansen et al. 2011).

---

### Model 43 -- Ownership Contagion (`ownership_contagion.py`)

**Foundational Theory.** MHHI (O'Brien and Salop 2000), crowding score (Khandani and Lo 2011), liquidation days (Kyle 1985).

---

### Model 44 -- Estimator (`estimator.py`)

**Foundational Theory.** Three-phase: deterministic identity fill, Rubin's MAR/MNAR classification (1976), MAR path via MICE (van Buuren and Groothuis-Oudshoorn 2011) + GP (Rasmussen and Williams 2006) + Matrix Completion (Mazumder et al. 2010), MNAR path via Heckman Selection (1979) + Pattern-Mixture (Little 1993) + GAIN (Yoon et al. 2018).

---

### Model 45 -- Retroactive Calibration (`retroactive_calibration.py`)

**Foundational Theory.** Empirical Bayes (Robbins 1956). Two-pass architecture: first pass uses defaults, retroactive calibration adjusts parameters based on results.

---

## Theory-Derived Improvement Edit Map

Specific code edits derived from the theoretical analysis, organized by priority.

### Priority 1 -- Correctness Fixes

| # | File | Line | What to Change | Why (Theory) |
|---|------|------|----------------|--------------|
| I1 | `operator1/models/monte_carlo.py` | ~158-199 | `estimate_regime_distributions` uses Normal distribution for all regimes. Add Student-t fitting when `scipy.stats.t` is available -- use Jarque-Bera test to choose Normal vs t per regime. | MC assumes Normal regime distributions, but financial returns have fat tails (Mandelbrot 1963). During crisis regimes, the Normal assumption underestimates tail probabilities by 10-100x. Student-t with low df captures this. The survival probability in "bear" regimes is likely overestimated by the current Gaussian assumption. |
| I2 | `operator1/analysis/survival_mode.py` | ~259 | Cox PH uses `duration = range(1, len)` as a synthetic duration column. Replace with actual "time since last survival event" using run-length encoding of the survival flag. | Cox PH theory (Cox 1972) requires proper duration data. The current synthetic duration treats every row as an independent observation at increasing time, which violates the PH assumption. Proper duration = days since last survival entry creates valid survival intervals. |
| I3 | `operator1/features/derived_variables.py` | ~376 | `enterprise_value` uses `market_cap.fillna(0) + total_debt.fillna(0) - cash.fillna(0)`. The fillna(0) masks missing data as zero values. Should only compute EV when at least market_cap is non-null. | The fillna(0) approach treats missing market_cap as zero, producing a negative EV (= 0 + debt - cash) that gets passed to ev_to_ebitda. This creates invalid negative EV/EBITDA ratios that confuse the financial health scoring. |

### Priority 2 -- Statistical Improvements

| # | File | Line | What to Change | Why (Theory) |
|---|------|------|----------------|--------------|
| I4 | `operator1/models/copula.py` | ~54-62 | `_to_uniform_marginals` uses empirical CDF via rankdata. Add kernel-smoothed CDF option for small samples (n < 100) using `scipy.stats.gaussian_kde`. | For small samples, the empirical CDF has step discontinuities that create artifacts in copula fitting (Fermanian 2005). A kernel-smoothed CDF produces smoother marginals. This matters when only 200-300 daily observations are available. |
| I5 | `operator1/models/forecasting.py` | ~3427-3433 | Forward pass error uses simple squared error. Add Huber loss option: `huber_loss = delta^2 * (abs(error)/delta - 0.5) if abs(error) > delta else 0.5*error^2`. | Squared error gives disproportionate weight to outliers (fat-tailed financial returns). Huber loss (Huber 1964) is quadratic for small errors and linear for large errors, making the forward pass robust to occasional extreme observations. The delta threshold can be set from the MAD of recent errors. |
| I6 | `operator1/models/graph_risk.py` | ~47-48 | `DEFAULT_CONTAGION_PROB = 0.3` is a fixed constant. Replace with the copula tail dependence coefficient from Model 32 when available. | The current SIR-like contagion model uses a fixed infection probability. Theory (Allen and Gale 2000, financial contagion) says contagion probability should be proportional to the strength of financial linkages. The copula lower tail dependence directly measures this -- two entities with lambda_L = 0.4 have stronger crisis co-movement than those with lambda_L = 0.1. |
| I7 | `operator1/models/regime_detector.py` | ~65 | `DEFAULT_PELT_PENALTY = 10.0` is fixed. Add BIC-based penalty selection: `penalty = 2 * n_features * log(n_obs)` for automatic tuning. | PELT with a fixed penalty either over-segments (too many breaks) or under-segments (misses real breaks). The BIC-based penalty (Schwarz 1978) automatically balances model complexity against fit, adapting to the data length and dimensionality. |
| I8 | `operator1/models/particle_filter.py` | ~state_transition | Particle state transition uses a random walk. Add an AR(1) state transition: `x_t = phi * x_{t-1} + noise` where phi is estimated from the data via OLS. | Financial ratios (current_ratio, debt_to_equity) are mean-reverting, not random walks. An AR(1) transition captures mean reversion -- a company with temporarily low current_ratio tends to recover. The random walk assumption produces overly diffuse particle swarms after many time steps. |

### Priority 3 -- Feature Enhancements

| # | File | Line | What to Change | Why (Theory) |
|---|------|------|----------------|--------------|
| I9 | `operator1/features/derived_variables.py` | ~168-170 | Volatility uses simple rolling std. Add Exponentially Weighted Moving Average (EWMA) volatility as a companion column: `ewma_vol = return_1d.ewm(span=21).std()`. | EWMA volatility (RiskMetrics, JP Morgan 1996) gives more weight to recent observations, adapting faster to volatility regime changes. Simple rolling std lags the true volatility by half the window size. Both should be available for temporal models to choose from. |
| I10 | `operator1/analysis/survival_mode.py` | ~166-206 | `compute_survival_probability` uses max of individual sigmoid signals. Add a geometric mean option: `prob = (prod(1 + signals))^(1/n) - 1`. | The max aggregation means a single trigger dominates. When multiple triggers are mildly breached simultaneously (current_ratio = 1.05, debt_to_equity = 2.8, fcf_yield = -0.01 -- all near thresholds), the max function only reflects the worst single trigger. A geometric mean captures the compounding risk of multiple near-breaches. |
| I11 | `operator1/models/monte_carlo.py` | ~transition_matrix | Transition matrix uses simple frequency counting with Laplace smoothing. Add Bayesian estimation via Dirichlet prior: `posterior = Dirichlet(counts + alpha)` for uncertainty-aware transitions. | With 500 daily observations and 4 regimes, some transitions are observed only 2-3 times. Frequency-based estimates are unreliable. A Dirichlet prior (conjugate to multinomial) produces posterior transition probabilities with proper uncertainty quantification. The posterior mean is `(count + alpha) / (total + K*alpha)` which naturally shrinks rare transitions toward the uniform prior. |
| I12 | `operator1/features/news_sentiment.py` | ~VADER section | Add temporal decay to sentiment aggregation: `daily_sentiment = sum(score_i * exp(-age_i / 3.0)) / sum(exp(-age_i / 3.0))`. | The current implementation weights all articles equally regardless of age. A 7-day-old article should contribute less than today's article. Exponential decay with a 3-day half-life captures the rapid information incorporation observed in efficient markets (Tetlock et al. 2008). |
| I13 | `operator1/models/walk_forward.py` | ~mode_errors | Add Diebold-Mariano test (Diebold and Mariano 1995) alongside MCS for pairwise model comparison. | MCS identifies the set of best models, but doesn't tell you which specific model significantly beats which other. DM test provides pairwise p-values: "Kalman is significantly better than LSTM in survival mode with p=0.03". This complements MCS with actionable pairwise comparisons. |
| I14 | `operator1/models/conformal.py` | ~intervals | Add asymmetric conformal intervals using signed residuals: separate upper and lower quantiles. | Current intervals are symmetric (point +/- quantile). Financial distributions are skewed -- downside risk is typically larger than upside potential. Asymmetric intervals using separate lower and upper nonconformity scores (Romano et al. 2019, Conformalized Quantile Regression) would produce wider lower bounds and tighter upper bounds, better reflecting the asymmetric risk profile. |
| I15 | `operator1/models/game_theory.py` | ~cournot | Add Bertrand-Edgeworth capacity-constrained pricing model alongside pure Bertrand. | Pure Bertrand predicts marginal-cost pricing (zero profit) which rarely occurs. Bertrand-Edgeworth (Edgeworth 1897) with capacity constraints produces mixed-strategy equilibria with positive margins, better matching real oligopoly behavior. The capacity constraint can be proxied from capital expenditure or asset utilization ratios. |

### Priority 4 -- Wiring / Integration

| # | File | Line | What to Change | Why |
|---|------|------|----------------|-----|
| I16 | `main.py` | ~Step 5g | Add `compute_relative_metrics` call after linked aggregates merge. | **DONE in PR #1.** |
| I17 | `operator1/models/forecasting.py` | ~3433 | Consume `_burnout_sample_weight` in forward pass error. | **DONE in PR #1.** |
| I18 | `main.py` | ~Step 6 | Pass `_adaptive_tier3.windows` to `run_forecasting()` so it can use filing-frequency-anchored windows instead of hardcoded 21/63/126/252. | The adaptive windows module computes data-derived window sizes but `run_forecasting` still uses fixed constants. The connection point exists (Tier 3 params are computed in Step 5k.2) but the values aren't passed through. |
| I19 | `main.py` | ~Step 6p | Pass `enriched_timeline_result.timeline["survival_mode"]` to `ConformalPIDCalibrator` as the Mondrian partition variable. | The Mondrian partitioning is implemented in the calibrator but the survival_mode Series needs to be explicitly passed from the enriched timeline. Currently, conformal intervals are the same width regardless of survival mode. |
| I20 | `operator1/models/prediction_aggregator.py` | ~ensemble | Add the `rel_strength_vs_sector` column (now wired via PR #1) to the prediction adjustment: when relative strength is strongly negative, widen uncertainty bands. | Peer underperformance is a forward-looking signal of additional risk not captured by the company's own time series. The linked aggregates data is now in the cache but the aggregator doesn't consume it for uncertainty adjustment. |

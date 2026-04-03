# Scoring Weights Panel -- All Models (2026-04-03)

Every weight, score, and calibration parameter across all 50 analytical models in Operator 1, organized into two panels:

- **PANEL A:** Input Variables (from data) + Tweakable Parameters (constants you can change)
- **PANEL B:** Output Variables (computed by formulas -- read-only, derived from Panel A)

---

# PANEL A: INPUTS + TWEAKABLE PARAMETERS

These are the knobs you can turn. Each row shows what the parameter controls, its default value, where to change it, and what happens when you change it.

---

## A1. Global Pipeline Config (`config/global_config.yml`)

| Parameter | Default | Range | What It Controls | Effect of Changing |
|-----------|---------|-------|-----------------|-------------------|
| `timeout_s` | 30 | 5-120 | HTTP request timeout | Higher = more tolerant of slow APIs, lower = faster failure |
| `max_retries` | 5 | 1-10 | HTTP retry count | Higher = more resilient to transient failures |
| `backoff_factor` | 2.0 | 1.0-5.0 | Exponential backoff multiplier | Higher = longer waits between retries |
| `rate_limit_calls_per_second` | 2.0 | 0.1-10.0 | Max sustained request rate per host | Lower = safer for free tiers, higher = faster data fetch |
| `search_budget_per_group` | 15 | 5-30 | Max search calls per entity group | Higher = finds more linked entities, costs more API calls |
| `search_budget_global` | 80 | 20-200 | Max total search calls | Higher = broader entity discovery |
| `cross_region_discovery` | false | true/false | Search all 25 PIT clients for entities | true = finds cross-border entities, much slower |
| `estimation_imputer` | "split" | "split" / "bayesian_ridge" / "vae" | Imputation algorithm | "split" = best (MAR/MNAR aware), "bayesian_ridge" = simple, "vae" = neural |
| `llm_provider` | "" (auto) | "gemini" / "claude" / "openrouter" / "" | LLM for reports + entity discovery | "" = auto-detect from available keys |
| `llm_model` | "" (auto) | See model list | Specific LLM model | "" = best available for provider |
| `filing_extraction_max_filings` | 8 | 1-20 | Max PDFs to extract per ticker | Higher = more financial data, slower + more LLM calls |
| `filing_extraction_stage_pause_s` | 15 | 5-60 | Seconds between extraction stages | Higher = less likely to hit rate limits |

### Prediction Contract (`config/global_config.yml` -> `prediction_contract`)

| Parameter | Default | Range | What It Controls |
|-----------|---------|-------|-----------------|
| `primary_target` | "return_5d" | Any cache column | Main prediction variable |
| `secondary_targets` | ["return_21d", "return_1d"] | List of columns | Additional prediction horizons |
| `evaluation_metric` | "spearman_ic" | "spearman_ic" / "rmse" / "mae" | How prediction quality is scored |
| `ic_inclusion_threshold` | 0.02 | 0.0-0.10 | Min |IC| to include signal in ensemble |
| `icir_inclusion_threshold` | 0.3 | 0.0-1.0 | Min |ICIR| to include signal |
| `signal_decay.slow` | 0.003 | 0.001-0.01 | Decay rate for fundamentals (per day) |
| `signal_decay.medium` | 0.05 | 0.01-0.15 | Decay rate for momentum signals (per day) |
| `signal_decay.fast` | 0.30 | 0.10-0.50 | Decay rate for technical signals (per day) |

---

## A2. Survival Mode Thresholds

**Location:** `operator1/analysis/survival_mode.py` + `config/country_survival_rules.yml`

### Company Survival Triggers (tweakable defaults, overridden by adaptive calibration)

| Trigger Variable | Default Threshold | Tweakable? | Adaptive Override Source | Safety Floor |
|-----------------|-------------------|-----------|------------------------|-------------|
| `current_ratio` | < 1.0 | Yes | Peer P10 (Huber 1981) | 0.3 |
| `debt_to_equity_abs` | > 3.0 | Yes | Peer P90 (Huber 1981) | 1.0 |
| `fcf_yield` | < 0.0 | Yes | Peer P10 (Huber 1981) | -0.5 |
| `drawdown_252d` | < -0.40 | Yes | BOCPD tightening (Adams & MacKay 2007) | -0.80 |
| `conflict_intensity_score` | > 0.70 | Yes | Static (no peer data for geopolitics) | 0.50 |
| `sanctions_flag` | == 1 | No | Binary, not calibratable | -- |
| `inst_flow_momentum` | < -0.15 | Yes | Static | -0.50 |

### Survival Probability Blend Weights

| Weight | Default | Tweakable? | Adaptive Override |
|--------|---------|-----------|------------------|
| `sigmoid_weight` | 0.4 | Yes (in code) | Cochrane inverse-variance blend |
| `cox_weight` | 0.6 | Yes (in code) | Cochrane inverse-variance blend |

---

## A3. Hierarchy Tier Weight Presets

**Location:** `operator1/analysis/hierarchy_weights.py` + `config/survival_hierarchy.yml`

| Regime | T1 (Liquidity) | T2 (Solvency) | T3 (Stability) | T4 (Profit) | T5 (Growth) | Tweakable? |
|--------|---------------|---------------|----------------|------------|------------|-----------|
| `normal` | 20 | 20 | 20 | 20 | 20 | Yes |
| `company_survival` | 50 | 30 | 15 | 4 | 1 | Yes |
| `modified_survival` | 40 | 35 | 20 | 4 | 1 | Yes |
| `extreme_survival` | 60 | 30 | 10 | 0 | 0 | Yes |

**Vanity adjustment amount:** 5% shift from T4/T5 to T1 (tweakable)

---

## A4. Financial Health Scoring Parameters

**Location:** `operator1/models/financial_health.py`

| Parameter | Default | Tweakable? | What It Controls |
|-----------|---------|-----------|-----------------|
| Tier weights | From hierarchy (Panel A3) | Yes (via hierarchy) | How much each tier contributes to composite |
| Altman Z-Score coefficients | 1.2, 1.4, 3.3, 0.6, 0.999 | No (from 1968 paper) | Bankruptcy prediction formula |
| Altman safe zone threshold | 2.99 | Yes | Above this = safe |
| Altman distress zone threshold | 1.81 | Yes | Below this = distress |
| Beneish M-Score threshold | -2.22 | Yes | Above this = likely manipulator |
| PE cap method | 3-method consensus | Yes (select method) | How extreme PE values are capped |
| EV/EBITDA cap method | 3-method consensus | Yes (select method) | How extreme EV values are capped |

---

## A5. Conflict Risk Scoring Weights

**Location:** `operator1/features/conflict_risk.py`

| Component | Weight | Tweakable? | What It Scores |
|-----------|--------|-----------|---------------|
| Event score | 0.40 | Yes | Armed conflict events count |
| Fatality score | 0.20 | Yes | Conflict fatalities |
| Flag score | 0.25 | Yes | Static risk classification |
| News score | 0.15 | Yes | Real-time news conflict mentions |

### Linked Entity Conflict Propagation Weights

| Group | Supply Chain Weight | Revenue Weight | Competitive Weight | Tweakable? |
|-------|--------------------|--------------|--------------------|-----------|
| Suppliers | 1.0 | 0.3 | 0.0 | Yes |
| Customers | 0.3 | 1.0 | 0.0 | Yes |
| Competitors | 0.0 | 0.0 | 1.0 | Yes |
| Financial institutions | 0.5 | 0.2 | 0.0 | Yes |
| Logistics | 0.8 | 0.1 | 0.0 | Yes |

---

## A6. Fuzzy Protection Parameters

**Location:** `operator1/analysis/fuzzy_protection.py`

| Parameter | Tweakable? | Default | What It Controls |
|-----------|-----------|---------|-----------------|
| Number of fuzzy rules | Yes | 11 | Inference rule count |
| Membership function shapes | Yes | Trapezoidal | Input variable fuzzification |
| Defuzzification method | Yes | Centroid | How output is converted from fuzzy to crisp |
| Strategic sector list | Yes | defense, energy, banking, telecom | Which sectors get high protection score |
| GDP significance threshold | Yes | 0.001 (0.1% of GDP) | market_cap / GDP threshold for economic significance |

---

## A7. Vanity Component Weights

**Location:** `operator1/analysis/vanity.py`

| Component | Weight | Tweakable? |
|-----------|--------|-----------|
| R&D Mismatch | 15% | Yes |
| SGA Bloat | 25% | Yes |
| Capital Misallocation | 30% | Yes |
| Competitive Decay | 15% | Yes |
| Sentiment Gap | 15% | Yes |

**Label thresholds:** Disciplined=20, Moderate=40, Wasteful=70, Reckless=100 (tweakable)

---

## A8. Plane-Aware Model Weight Adjustments

**Location:** `operator1/models/model_synergies.py`

All adjustments are multipliers on the base weight of 1.0. Each cell is tweakable.

| Model | Supply | Manufacturing | Consumption | Logistics | Finance |
|-------|--------|---------------|-------------|-----------|---------|
| Forecasting | 1.0 | **1.4** | 1.1 | 1.1 | 1.0 |
| Monte Carlo | **1.3** | 1.0 | 1.0 | **1.5** | **1.3** |
| Transformer | 0.8 | 1.0 | **1.4** | 0.9 | 1.0 |
| Cycle Decomposition | **1.8** | 1.2 | 1.0 | 1.2 | 0.7 |
| Pattern Detector | 0.7 | 1.0 | **1.5** | 0.8 | 0.8 |
| Copula | 1.2 | 0.8 | 1.0 | 1.0 | **1.8** |
| Particle Filter | **1.4** | 1.0 | 0.7 | **1.4** | 1.0 |
| DTW Analogs | 1.0 | 1.1 | **1.3** | 1.0 | 1.0 |
| Granger Causality | 1.0 | **1.5** | 0.8 | 1.0 | **1.5** |
| Transfer Entropy | 1.0 | **1.3** | 1.0 | 1.0 | **1.4** |

---

## A9. Graph Risk Edge Weights

**Location:** `operator1/models/graph_risk.py`

| Relationship Type | Default Weight | Tweakable? | What It Models |
|------------------|---------------|-----------|----------------|
| Parent companies | 2.8 | Yes | Parent distress -> subsidiary |
| Subsidiaries | 2.3 | Yes | Subsidiary distress -> parent |
| Suppliers | 1.2 | Yes | Supply chain disruption |
| Financial institutions | 1.3 | Yes | Credit channel contagion |
| Customers | 1.1 | Yes | Revenue channel disruption |
| Competitors | 1.0 | Yes | Market share competition |
| Logistics | 0.8 | Yes | Logistics disruption |
| Regulators | 0.5 | Yes | Regulatory change |

---

## A10. Monte Carlo Simulation Parameters

**Location:** `operator1/models/monte_carlo.py` + adaptive_model_params

| Parameter | Default | Tweakable? | Adaptive Override |
|-----------|---------|-----------|------------------|
| `n_paths` | 10,000 | Yes | Precision-targeted (Glasserman 2003) |
| `importance_tilt` | 1.5 | Yes | Adaptive from survival regime |
| `horizon_days` | [90, 252] | Yes | USS horizon compression |
| `correlation_override` (survival) | None / 0.85 / 0.90 | Yes | USS Dimension 4 |
| `copula_type_override` (survival) | None / Clayton | Yes | USS Dimension 4 |

### Scenario Engine Parameters (USS)

| Scenario | Revenue Shift | Daily Drift | Returns Used | Tweakable? |
|----------|-------------|------------|-------------|-----------|
| Orderly | -15% | -0.1%/day | P25-P75 | Yes |
| Muddle Through | 0% | 0%/day | Full | Yes |
| Catastrophic | -40% | -0.3%/day | Bottom P10 | Yes |

---

## A11. Ensemble Aggregation Parameters

**Location:** `operator1/models/prediction_aggregator.py`

| Parameter | Default | Tweakable? | What It Controls |
|-----------|---------|-----------|-----------------|
| `MIN_RMSE_FOR_WEIGHTING` | 1e-10 | Yes | Floor to avoid div-by-zero |
| `Z_SCORE_90` | 1.645 | Yes | CI width for 90% confidence |
| `DEFAULT_SURVIVAL_RISK_MULTIPLIER` | 2.0 | Yes | How much survival mode widens CIs |
| `_TRANSITION_BLEND_HALFLIFE` | 21 days | Yes | Speed of mode transition blending |
| FixedShare `share` parameter | 0.05 | Yes | How fast model weights can shift |
| FixedShare `eta` (learning rate) | 0.1 | Yes | Responsiveness to loss updates |

### Model-Regime Affinity (tweakable)

| Model | Bull | Bear | High Vol | Low Vol |
|-------|------|------|----------|---------|
| Kalman | 1.0 | 0.6 | 0.5 | 1.2 |
| GARCH | 0.7 | 1.0 | 1.3 | 0.6 |
| VAR | 1.0 | 0.8 | 0.7 | 1.1 |
| LSTM | 0.9 | 0.9 | 0.8 | 1.0 |
| Tree (RF/GBM/XGB) | 0.8 | 1.1 | 1.2 | 0.9 |
| Baseline | 1.0 | 1.0 | 1.0 | 1.0 |

---

## A12. Conformal Prediction Parameters

**Location:** `operator1/models/conformal.py`

| Parameter | Default | Tweakable? | What It Controls |
|-----------|---------|-----------|-----------------|
| `target_coverage` | 0.90 | Yes | Desired coverage level (90%) |
| PID Kp (proportional) | 0.01 | Yes | Coverage correction speed |
| PID Ki (integral) | 0.001 | Yes | Steady-state coverage error fix |
| PID Kd (derivative) | 0.005 | Yes | Oscillation damping |
| Mondrian partitioning | Per survival mode | Yes | Separate calibration per mode |
| Copula tail amplification | 0.5 | Yes | How much copula widens CIs |

---

## A13. Frequency Fusion Horizon Weights

**Location:** `operator1/models/frequency_fusion.py`

Each cell is the contribution weight of that frequency to that horizon (tweakable).

| Horizon | Daily | Weekly | Monthly | Quarterly | Annual |
|---------|-------|--------|---------|-----------|--------|
| 1d | **1.00** | -- | -- | -- | -- |
| 5d | **0.70** | 0.30 | -- | -- | -- |
| 1w | 0.40 | **0.60** | -- | -- | -- |
| 21d | 0.30 | 0.40 | **0.30** | -- | -- |
| 1m | -- | 0.30 | **0.70** | -- | -- |
| 3m | -- | 0.15 | 0.35 | **0.50** | -- |
| 6m | -- | -- | 0.25 | **0.50** | 0.25 |
| 1y | -- | -- | 0.15 | 0.35 | **0.50** |
| 2y | -- | -- | -- | 0.30 | **0.70** |

---

## A14. USS Dimension Parameters

**Location:** `operator1/analysis/survival_regime_controller.py`

### Dimension 2: Model Switching (tweakable per regime)

| Parameter | Normal | Company Survival | Extreme Survival |
|-----------|--------|-----------------|-----------------|
| Kalman noise multiplier | 1x | 3x | 5x |
| LSTM lookback (days) | 60 | 10 | 5 |
| MC paths | 10K | 20K | 30K |
| Tree max depth | 10 | 5 | 3 |

### Dimension 3: Horizon Compression (tweakable)

| Regime | Available Horizons |
|--------|-------------------|
| Normal | 1d, 5d, 21d, 252d |
| Company Survival | 1d, 5d, 21d |
| Extreme Survival | 1d, 5d |

### Dimension 5: Forecast Bounding (tweakable multipliers)

| Bound | Company Survival | Extreme Survival |
|-------|-----------------|-----------------|
| Revenue cap | Last actual * 1.0 | Last actual * 0.85 |
| Debt floor | Current level * 1.0 | Current * 1.15 |
| Cash bound | Max burn rate * 1.0 | Max burn * 1.5 |

---

## A15. Forecasting Model Minimum Data Requirements

**Location:** `operator1/models/forecasting.py`

| Model | Min Rows | Tweakable? | Notes |
|-------|----------|-----------|-------|
| Kalman | 30 | Yes | Lower = more fits, possibly unstable |
| GARCH | 50 | Yes | Needs enough for vol clustering |
| VAR | 50 | Yes | Multivariate, needs more data |
| AR(1) | 20 | Yes | VAR fallback |
| LSTM | 200 | Yes | Neural net needs training data |
| GBM/LR | 50 | Yes | LSTM fallback |
| Tree (RF/GBM/XGB) | 30 | Yes | Feature-based, robust to small data |
| Baseline (EMA) | 1 | No | Always available |
| AutoARIMA | 30 | Yes | statsforecast |
| DynamicFactor | 50 | Yes | Multi-variable DFM |

---

## A16. Input Variables (from data -- not tweakable)

These come from PIT API wrappers and cannot be adjusted. They are the raw data that everything else computes from.

### OHLCV (from PIT wrapper or yfinance)
`open`, `high`, `low`, `close`, `volume`, `adjusted_close`, `vwap`, `market_cap`, `shares_outstanding`

### Financial Statements (from PIT wrapper, 34 fields)
`revenue`, `cost_of_revenue`, `gross_profit`, `operating_income`, `ebit`, `ebitda`, `net_income`, `interest_expense`, `taxes`, `total_assets`, `total_liabilities`, `total_equity`, `current_assets`, `current_liabilities`, `cash_and_equivalents`, `short_term_debt`, `long_term_debt`, `total_debt`, `retained_earnings`, `goodwill`, `intangible_assets`, `receivables`, `inventory`, `payables`, `operating_cash_flow`, `capex`, `free_cash_flow`, `investing_cf`, `financing_cf`, `dividends_paid`, `stock_buybacks`, `sga_expenses`, `rd_expenses`, `eps`, `eps_diluted`

### Macro Data (from FRED/BCB/Banxico/SDMX/wbgapi)
`gdp_growth`, `inflation_rate_yoy`, `real_interest_rate`, `unemployment_rate`, `official_exchange_rate_lcu_per_usd`

### Profile Data (from PIT wrapper)
`name`, `ticker`, `country`, `sector`, `industry`, `sub_industry`, `exchange`, `currency`, `isin`

### News Articles (from gnews/feedparser)
Article titles + snippets (text, not numeric)

### Holder Data (from PIT wrapper/GLEIF)
`holder_name`, `shares_held`, `percentage`, `holder_type`, `filing_date`

---

# PANEL B: OUTPUTS (Formula-Derived -- Read-Only)

These are computed from Panel A inputs + tweakable parameters. You cannot set them directly -- they change when you change the inputs or parameters above.

---

## B1. Derived Variables (from `compute_derived_variables()`)

| Output Variable | Formula | Input Variables |
|----------------|---------|-----------------|
| `return_1d` | `close.pct_change()` | close |
| `log_return_1d` | `ln(close / close.shift(1))` | close |
| `return_5d` / `return_21d` | `close.pct_change(5)` / `(21)` | close |
| `volatility_21d` / `63d` | `return_1d.rolling(21).std()` | return_1d |
| `drawdown_252d` | `(close - close.rolling(252).max()) / max` | close |
| `current_ratio` | `current_assets / current_liabilities` | current_assets, current_liabilities |
| `debt_to_equity_abs` | `total_debt / abs(total_equity)` | total_debt, total_equity |
| `cash_ratio` | `cash / current_liabilities` | cash_and_equivalents, current_liabilities |
| `interest_coverage` | `ebit / interest_expense` | ebit, interest_expense |
| `gross_margin` | `gross_profit / revenue` | gross_profit, revenue |
| `operating_margin` | `operating_income / revenue` | operating_income, revenue |
| `net_margin` | `net_income / revenue` | net_income, revenue |
| `fcf_yield` | `free_cash_flow / market_cap` | free_cash_flow, market_cap |
| `pe_ratio_calc` | `close / eps_diluted` | close, eps_diluted |
| `ev_to_ebitda` | `(market_cap + total_debt - cash) / ebitda` | market_cap, total_debt, cash, ebitda |
| `beta_252d` | `cov(return, benchmark) / var(benchmark)` | return_1d, benchmark_return_1d |
| `revenue_ttm` | `revenue.rolling(4Q).sum()` | revenue |
| `adx_14` | `ta.trend.ADXIndicator` | high, low, close |
| `obv` | `ta.volume.OnBalanceVolumeIndicator` | close, volume |
| `bb_width` | `ta.volatility.BollingerBands` | close |
| `macd_histogram` | `ta.trend.MACD` | close |

---

## B2. Survival & Protection Outputs

| Output | Formula | Depends On |
|--------|---------|-----------|
| `company_survival_mode_flag` | OR of 7 triggers (Panel A2) | current_ratio, debt_to_equity_abs, fcf_yield, drawdown_252d, conflict, sanctions, inst_flow |
| `country_survival_mode_flag` | OR of 4 macro triggers | credit_spread, fx_volatility, unemployment, yield_curve |
| `country_protected_flag` | OR of 3 conditions | sector, market_cap/GDP, rate_cut_detected |
| `survival_probability` | 0.4*sigmoid + 0.6*cox | Distance to each threshold + Cox PH hazard |
| `cox_survival_score` | Cox PH fitted hazard | current_ratio, debt_to_equity, fcf_yield, drawdown |
| `fuzzy_protection_degree` | Mamdani defuzzification | sector, market_cap/GDP, rate_cut |
| `hierarchy_tier{1-5}_weight` | Regime lookup table (Panel A3) | survival flags + protected flag |
| `survival_regime` | 4-state classification | company_flag, country_flag, protected_flag |

---

## B3. Financial Health Outputs

| Output | Formula | Depends On |
|--------|---------|-----------|
| `fh_liquidity_score` | Expanding percentile rank | cash_ratio, current_ratio, free_cash_flow |
| `fh_solvency_score` | Expanding percentile rank | debt_to_equity, interest_coverage |
| `fh_stability_score` | Expanding percentile rank | volatility_21d, drawdown_252d |
| `fh_profitability_score` | Expanding percentile rank | gross_margin, operating_margin, net_margin |
| `fh_growth_score` | Expanding percentile rank | revenue_growth, pe_ratio, ev_to_ebitda |
| `fh_composite_score` | Weighted sum (tier weights) | All 5 tier scores + hierarchy weights |
| `fh_composite_label` | Jenks Natural Breaks | fh_composite_score distribution |
| `fh_altman_z_score` | Z = 1.2*WC/TA + 1.4*RE/TA + ... | working_capital, retained_earnings, ebit, market_cap, revenue, total_assets, total_liabilities |
| `fh_altman_z_zone` | Threshold comparison | altman_z_score vs 2.99/1.81 |
| `fh_beneish_m_score` | 8-factor formula | DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA |
| `fh_runway_months` | `cash / abs(monthly_burn)` | cash_and_equivalents, operating_cash_flow |

---

## B4. Conflict & Geopolitical Outputs

| Output | Formula | Depends On |
|--------|---------|-----------|
| `conflict_intensity_score` | 0.40*event + 0.20*fatality + 0.25*flag + 0.15*news | UCDP events, fatalities, static lists, GDELT |
| `country_conflict_flag` | intensity > 0.3 or in war list | conflict_intensity_score |
| `sanctions_flag` | In OFAC/EU sanctions list | country_code |
| `supply_chain_risk_score` | Max across supplier countries | linked entity countries vs conflict lists |
| `revenue_exposure_score` | Max across customer countries | linked entity countries vs conflict lists |

---

## B5. Regime Detection Outputs

| Output | Formula | Depends On |
|--------|---------|-----------|
| `regime_hmm` | 4-state Gaussian HMM | return_1d, volatility_21d |
| `regime_gmm` | BIC-optimized GMM clustering | return_1d, volatility_21d |
| `regime_label` | Majority vote across methods | regime_hmm, regime_gmm, PELT, BCP |
| `structural_break` | PELT change points | return_1d |
| `online_change_score` | ChangeFinder SDAR | return_1d (past only, no look-ahead) |
| `regime_state` | Enriched timeline mapping | survival_mode x market_regime |
| `survival_intensity` | Continuous 0-1 | survival_mode, regime_label |
| `stability_score_21d` | Rolling 21d same-mode fraction | survival_mode |

---

## B6. Forecasting Outputs

| Output | Formula | Depends On |
|--------|---------|-----------|
| `forecast[var][horizon]` | First-fit model in cascade | cache columns, extra_vars |
| `model_used[var]` | Which model won | RMSE comparison |
| `residuals[var]` | Predicted - actual (validation) | model predictions, actuals |

---

## B7. Ensemble Aggregation Outputs

| Output | Formula | Depends On |
|--------|---------|-----------|
| `ensemble_weights` | 1/RMSE normalized | Per-model RMSE |
| `regime_blended_weights` | Sum(prob * affinity * base_weight) | Regime probs, affinity table, base weights |
| `survival_aware_weights` | Exp blend during transitions | Mode weights, days_in_mode, halflife |
| `point_forecast` | Weighted average of model forecasts | Ensemble weights, model forecasts |
| `lower_ci` / `upper_ci` | forecast +/- adjusted_spread | RMSE, z_score, horizon_days, survival_prob |
| `confidence_score` | 0.7*model_quality + 0.3*survival | RMSE/reference, survival_probability |
| `module_contributions` | Weight fraction per model | Ensemble weights |

---

## B8. Monte Carlo Outputs

| Output | Formula | Depends On |
|--------|---------|-----------|
| `survival_probability` (MC) | Fraction of paths not triggering | n_paths, regime distributions, transition matrix, thresholds |
| `terminal_values` | Path endpoints | Return distributions, regime switching |
| `transition_matrix` | HMM state transitions | regime_hmm fitted parameters |
| `regime_distributions` | Per-regime (mu, sigma) | Historical returns grouped by regime |

---

## B9. Advanced Model Outputs

| Output | Formula | Depends On |
|--------|---------|-----------|
| `copula_correlation` | Fitted copula params | Uniform marginals of selected variables |
| `tail_dependence` | From best-fit copula | Gaussian/Student-t/Clayton AIC selection |
| `joint_crisis_probability` | Copula simulation | tail_dependence, crisis thresholds |
| `granger_significant_pairs` | PCMCI or F-test p < 0.05 | Up to 25 float variables |
| `transfer_entropy_pairs` | kNN conditional entropy | Up to 20 float variables |
| `dominant_cycles` | CEEMDAN IMFs or FFT peaks | close (or cycle variable) |
| `shap_values[var]` | TreeExplainer or KernelExplainer | Fitted models, cache features |
| `sobol_S1[var]` / `sobol_ST[var]` | Saltelli sampling | All features vs target variable |
| `dtw_matches` | DTW distance ranking | Recent window vs historical windows |
| `particle_filter_states` | Sequential MC posterior mean | Survival trigger variables |
| `conformal_intervals` | PID-calibrated quantiles | Residuals, target coverage |
| `pattern_result` | Body/shadow ratio rules + stumpy | open, high, low, close |
| `ohlc_predictions` | Forecast + MC + pattern + cycle | All temporal model outputs |

---

## B10. Report & Profile Outputs

| Output | Depends On |
|--------|-----------|
| `company_profile.json` | All Panel B outputs aggregated |
| `basic_report.md` | 5 profile sections |
| `pro_report.md` | 18 profile sections + 9 charts |
| `premium_report.md` | 27 profile sections + 9 charts |
| `triage_card.md` | USS scenario results (distress only) |
| `position_signal` (-1 to +1) | return forecast * IC confidence * survival multiplier |

---

## Quick Reference: What to Change for Common Goals

| Goal | Tweak These Parameters |
|------|----------------------|
| **Less aggressive survival triggering** | Raise thresholds in Panel A2 (current_ratio < 0.8 instead of 1.0) |
| **More weight on profitability** | Increase T4 weight in Panel A3 normal regime |
| **Better predictions for financial companies** | Increase copula/granger weights in Panel A8 finance column |
| **Wider confidence intervals** | Increase Z_SCORE in Panel A11, or survival_risk_multiplier |
| **Faster model adaptation** | Increase FixedShare share parameter in Panel A11 |
| **More MC paths for precision** | Increase n_paths in Panel A10 (costs more compute) |
| **Include weaker signals** | Lower ic_inclusion_threshold in global_config prediction_contract |
| **Prioritize short-term predictions** | Increase daily/weekly weights in Panel A13 |
| **Disable survival forecast bounding** | Set USS Dimension 5 multipliers to 1.0 / None |
| **Run lighter pipeline** | Reduce filing_extraction_max_filings, search_budget_global |

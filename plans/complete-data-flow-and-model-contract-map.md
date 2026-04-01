# Complete Data Flow and Model Contract Map

*Last updated: 2026-04-01*

Every model in the Operator 1 pipeline, from raw cache construction through
report generation, with expected vs actual inputs, outputs, and operations.

**Legend:**
- OK = expected matches actual
- FIXED = was broken, now fixed in this PR
- NOTE = minor observation

---

## Phase A: Cache Construction

### A1. Canonical Translator -- `pivot_to_canonical_wide()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | Long-format DataFrame with `canonical_name`, `value`, `report_date`, `filing_date` columns | Same -- PIT clients output canonical long format | OK |
| **Output** | Wide-format DataFrame: one row per `report_date`, columns = canonical field names (revenue, total_assets, etc.) | Same -- pivot + merge filing_date from latest filing per report_date | OK |
| **Operation** | Pivot long to wide using `report_date` as row key, `canonical_name` as column key, `value` as cell value | Same | OK |

### A2. Cache Builder -- inline in `main.py` Step 4

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `quotes_df` (OHLCV), wide `income_df`, `balance_df`, `cashflow_df` | Same | OK |
| **Output** | Daily cache DataFrame with DatetimeIndex (OHLCV spine) + forward-filled financial statement columns | Same -- uses index union + ffill for as-of join | OK |
| **Operation** | 1. Set OHLCV as the daily spine. 2. For each statement: sort by date, dedup by date, union indices with cache, ffill, reindex to daily. 3. First-statement-wins for duplicate column names. | Same | OK |
| **Note** | The as-of join correctly uses `report_date` (not `filing_date`) as the merge key. PIT constraint is satisfied: report_date <= filing_date always. | Verified in data_reconciliation.py | OK |

### A3. Data Reconciliation -- `reconcile_financial_data()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | Raw `income_df`, `balance_df`, `cashflow_df` from PIT client | Same | OK |
| **Output** | Cleaned DataFrames + reconciliation_report dict | Same | OK |
| **Operation** | 1. Normalize field names to canonical schema. 2. Validate filing_date >= report_date (fix time-travel). 3. Remove duplicate filings (keep latest amendment). 4. Detect stale data (>180 days). | Same -- all 4 steps implemented | OK |

---

## Phase B: Macro Data and Quadrant

### B1. Macro Provider -- `fetch_macro()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `country_code`, `secrets`, `years` | Same | OK |
| **Output** | Dict of `{indicator_name: pd.Series}` with indicators: gdp, inflation, interest_rate, unemployment, currency | Same -- routes to per-region macro API (FRED, BCB, Banxico, etc.) | OK |
| **Operation** | Fetch 5 standard macro indicators from the appropriate regional API for the given country | Same | OK |

### B2. Macro Mapping -- `fetch_macro_data()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `country_iso2`, `macro_raw` dict | Same | OK |
| **Output** | `MacroDataset` object with `.indicators` dict and `.missing` list | Same | OK |
| **Operation** | Wrap raw macro dict into structured MacroDataset container for downstream modules | Same | OK |

### B3. Macro Quadrant -- `compute_macro_quadrant()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `cache` DataFrame, `macro_data` (MacroDataset) | Same | OK |
| **Output** | cache (+ macro_quadrant columns), `macro_quadrant_result` with `latest_quadrant` and `stability_score` | Same | OK |
| **Operation** | Classify macro environment into quadrant based on GDP growth vs inflation trends | Same | OK |

---

## Phase C: Feature Engineering

### C1. Derived Variables -- `compute_derived_variables()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache DataFrame with `close`, financial statement columns | Same | OK |
| **Output** | cache + ~50 derived columns: `return_1d`, `log_return_1d`, `volatility_21d`, `drawdown_252d`, `current_ratio`, `debt_to_equity_abs`, `fcf_yield`, `cash_ratio`, `gross_margin`, `pe_ratio_calc`, `beta_252d`, `debt_service_coverage`, etc. + `is_missing_*` and `invalid_math_*` flags | Same | OK |
| **Operation** | 1. Returns and risk from close price. 2. Solvency ratios (debt_to_equity, net_debt, debt_service_coverage). 3. Liquidity ratios (current_ratio, cash_ratio). 4. Profitability (margins). 5. TTM computations. 6. **Technical indicators via `ta` library: ADX, OBV, BB width, MACD histogram**. 7. **Beta vs market benchmark** (`beta_252d` from `config/market_benchmarks.yml`, per-market index). All use `safe_ratio()` to handle division by zero. | Same | **ENHANCED** |

### C2. Survival Mode -- `compute_company_survival_flag()` + `compute_cox_survival_score()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache with `current_ratio`, `debt_to_equity_abs`, `fcf_yield`, `drawdown_252d` | Same | OK |
| **Output** | `cache["company_survival_mode_flag"]` -- integer Series: 1=survival, 0=normal. `cache["survival_probability"]` -- continuous 0-1. **NEW**: `cache["cox_survival_score"]` -- Cox PH data-driven hazard score | Same | OK |
| **Operation** | OR of 4 conditions + sigmoid probability + **Cox PH hazard via `lifelines.CoxPHFitter`** (blended 0.4*sigmoid + 0.6*cox) | Same -- thresholds loaded from config. Cox PH learns hazard ratios from the company's own distress episodes | **ENHANCED** |

### C3. Hierarchy Weights -- `compute_hierarchy_weights()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache with survival_mode_flag | Same | OK |
| **Output** | cache (+ `hierarchy_tier1_weight` through `hierarchy_tier5_weight`), `weights` dict | Same | OK |
| **Operation** | Compute per-tier weights based on survival mode. In survival: tiers 1-2 get higher weight. In normal: equal weights. | Same | OK |

### C4. Fuzzy Protection -- `compute_fuzzy_protection()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `sector` string, `gdp` float (optional) | Same | OK |
| **Output** | cache + `fuzzy_protection_degree` (0-1), `fuzzy_sector_score`, `fuzzy_protection_label` | Same | OK |
| **Operation** | Fuzzy logic: compute sector strategicness membership, economic significance (market_cap/GDP), policy responsiveness (rate cuts). **Aggregate via scikit-fuzzy Mamdani rule engine with 11 interaction rules** (fallback: fuzzy OR max). | Same | **ENHANCED** |

### C5. Financial Health -- `compute_financial_health()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `hierarchy_weights` dict | Same | OK |
| **Output** | cache + `fh_liquidity_score`, `fh_solvency_score`, `fh_stability_score`, `fh_profitability_score`, `fh_growth_score`, `fh_composite_score`, `fh_composite_label`, `fh_altman_z_score`, `fh_altman_z_zone`, `fh_beneish_m_score`, `fh_beneish_flag`, `fh_runway_months`; `fh_result` (FinancialHealthResult) | Same | OK |
| **Operation** | 1. Per-tier scoring (expanding percentile rank normalization). 2. Weighted composite (uses hierarchy_weights). 3. Altman Z-Score (5 coefficients from 1968 paper). 4. Beneish M-Score (8 coefficients from 1999 paper). 5. Liquidity runway (cash / monthly burn rate). 6. **Adaptive PE/EV caps** (Log-Normal P99.5 + Tukey Extreme Fence + MAD consensus, replacing fixed 200/100 caps). | Same | **ENHANCED** |

### C6. Entity Discovery -- `discover_linked_entities()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `target_profile`, `gemini_client` (LLM), `pit_client`, `secrets` | Same | OK |
| **Output** | `relationships` dict: `{group_name: [entity_list]}` (competitors, suppliers, customers, etc.) | Same | OK |
| **Operation** | Ask LLM to identify linked entities (competitors, suppliers, customers, financial institutions) for the target company | Same | OK |

### C7. Linked Aggregates -- `compute_linked_aggregates()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `target_daily` cache, `linked_daily` dict of entity caches, `entity_groups` dict | Same | OK |
| **Output** | `linked_agg_df` DataFrame with columns like `competitors_avg_return_1d`, `suppliers_median_volatility_21d`, etc. | Same -- merged into main cache | OK |
| **Operation** | For each entity group, compute mean/median of key variables across group members, aligned to daily index | Same | OK |

### C8. Peer Ranking -- `compute_peer_ranking()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `linked_caches` dict | Same | OK |
| **Output** | cache + `peer_*` columns, `peer_ranking_result` | Same | OK |
| **Operation** | Compute percentile rank of target vs peers for key variables | Same | OK |

### C9. News Sentiment -- `compute_news_sentiment()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `gemini_client` (LLM, optional), `symbol` string | Same | OK |
| **Output** | cache + `sentiment_*` columns, `sentiment_result` | Same | OK |
| **Operation** | 1. Fetch news via gnews/feedparser. 2. Score sentiment via LLM (if available) or **VADER** (handles negation, intensity, context) or keyword fallback. 3. Write daily sentiment score to cache. | Same | **ENHANCED** |

### C10. Filing Calendar -- `analyze_filing_calendar()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `market_id` string | Same | OK |
| **Output** | `filing_calendar_result` with `expected_frequency`, `detected_frequency`, `expected_filings_2yr`, `actual_filings_2yr`, `coverage_ratio`, `latest_filing_age_days`, `is_stale`, `stale_threshold_days`, `gaps` list | Same | OK |
| **Operation** | 1. Detect filing frequency from cache date patterns (quarterly, semi-annual, annual). 2. Count expected vs actual filings in 2-year window. 3. Compute coverage ratio. 4. Detect staleness (latest filing age vs market-specific threshold). 5. Identify filing gaps. | Same | OK |
| **Profile** | Stored in `profile["filing_calendar"]` with all fields | Same | OK |

### C11. Graph Risk -- `compute_graph_risk_metrics()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `target_isin` (or ticker), `relationships` dict from entity discovery | Same | OK |
| **Output** | `graph_risk_result` with `n_nodes`, `target_degree_centrality`, network topology metrics | Same | OK |
| **Operation** | Build network graph from entity relationships. Compute degree centrality, PageRank, **edge-weighted contagion** (revenue/supply exposure), **CoVaR** (conditional value-at-risk per linked entity), **SRISK** (capital shortfall under stress). Identify systemically important connections. | Same | **ENHANCED** |
| **Profile** | Stored via `_available_dict(graph_risk_result)` | Same | OK |

### C12. Game Theory -- `analyze_competitive_dynamics()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `target_cache`, `competitor_caches` dict of linked entity caches, `target_name` | Same -- now passes `competitor_caches=linked_caches` | OK |
| **Output** | `game_theory_result` with Cournot/Stackelberg analysis, `competitive_pressure`, `market_structure` | Same | OK |
| **Operation** | Cournot quantity game, Bertrand price game, Stackelberg leadership, CR4 market structure, competitive pressure index | Same | OK |
| **Profile** | Stored via `_available_dict(game_theory_result)` | Same | OK |

### C13. Economic Planes -- `classify_economic_plane()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `sector` string, `industry` string | Same | OK |
| **Output** | Dict with `primary_plane` (one of 5 planes), `secondary_planes` list | Same | OK |
| **Operation** | Map sector/industry to one of 5 economic planes from the Sudoku framework: Real Economy, Financial, Technology, Resources, Services. Used for plane-aware model weighting in pre-forecasting synergies. | Same | OK |
| **Profile** | Stored in `profile["economic_plane"]` | Same | OK |

### C14. OHLCV Provider (fallback) -- `fetch_ohlcv()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `ticker` string, `market_id` string | Same | OK |
| **Output** | `quotes_df` DataFrame with date, open, high, low, close, volume columns | Same | OK |
| **Operation** | When the PIT filing API does not provide price data (most don't -- SEC EDGAR, DART, etc.), fetch OHLCV from a free-tier source: yfinance (global fallback), or per-region wrappers (pykrx for Korea, baostock for China, twstock for Taiwan, nselib for India). | Same | OK |
| **Note** | OHLCV source is tracked separately in `profile["meta"]["ohlcv_source"]` | Same | OK |

---

## Phase D: Estimation Engine

### D1. Estimator -- `run_estimation()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache DataFrame, `imputer_method` string ("split", "bayesian_ridge", or "vae") | Same | OK |
| **Output** | Augmented cache with per-variable columns: `{var}_observed`, `{var}_estimated`, `{var}_final`, `{var}_source`, `{var}_confidence`, `{var}_missingness_type`, `{var}_estimation_method`, `{var}_sensitivity_lower/upper`; `EstimationCoverage` result | Same | OK |
| **Operation** | **Phase 1:** Deterministic accounting identity fill (total_assets = total_liabilities + total_equity, etc., 5 iterations max). **Phase 2:** Classify NaN as MAR or MNAR. **Phase 3a (MAR):** **miceforest LightGBM MICE** (preferred, non-linear) or sklearn BayesianRidge MICE (fallback) + Gaussian Process + Matrix Completion ensemble. **Phase 3b (MNAR):** Heckman Selection + Pattern-Mixture + GAIN ensemble. Observed values are NEVER overwritten. | Same -- all 3 phases implemented | **ENHANCED** |
| **Note** | PerformanceWarning during in-loop column insertion is now suppressed (P1 fix). DataFrame defragmented via `.copy()` before return. | FIXED | OK |

---

## Phase D-extra: Adaptive Parameter Calibration

Three modules that replace fixed constants across the pipeline with data-derived values. Wired in main.py Steps 5j, 5k, and 5k.2 -- after peer data is available but before temporal models consume parameters.

### Dx1. Adaptive Thresholds -- `compute_adaptive_thresholds()` (Tier 1)

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `cache`, `linked_caches` (peer data from Step 5f), `regime_detector` (from Step 5.5), `fh_composite_scores` | Same | OK |
| **Output** | `ThresholdSet` with calibrated survival thresholds, regime mixer thresholds, vanity thresholds, FH label breakpoints | Same | OK |
| **Location** | `operator1/analysis/adaptive_thresholds.py` (837 lines) | | |
| **Operation** | 5 methods: (A) **Peer Percentile** (Huber 1981) -- survival triggers at P10/P90 of sector peer distribution, MAD-based fallback for small groups. (D) **BOCPD Deterioration Tightening** (Adams & MacKay 2007) -- when own-history shows structural downward shift, thresholds tighten by 20%. (E) **Sector Z-Score** (Iglewicz & Hoaglin 1993) -- vanity metrics flagged at 2 modified-Z-scores from sector median. (H) **Jenks Natural Breaks** (Fisher 1958) -- FH composite labels from optimal class boundaries via DP. (J) **HMM Emission Crossover** (Rabiner 1989) -- regime mixer thresholds from HMM Gaussian crossover point. | Same | OK |
| **Downstream** | Consumed by `compute_company_survival_flag()` (recalibrated), `compute_survival_probability()` (recalibrated), `compute_hierarchy_weights()` (re-run), `run_monte_carlo()` (via `threshold_set_to_mc_dict()`), `compute_dual_regimes()` (via `threshold_set_to_regime_dict()`) | Same | OK |
| **Fallback** | All thresholds have textbook defaults; adapted only when sufficient peer data or HMM results are available. Absolute floors prevent over-adaptation. | Same | OK |

### Dx2. Adaptive Model Parameters -- Tier 2 constants

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `cache`, `regime_detector`, `enriched_timeline_result`, Cox/sigmoid series | Same | OK |
| **Output** | `AdaptiveModelParams` with: blend weights, risk multiplier, GK factor, transition halflife, MC params, participation rate, PID gains, contagion probs, train splits, confidence bounds | Same | OK |
| **Location** | `operator1/analysis/adaptive_model_params.py` (1186 lines) | | |
| **Operation** | 10+ methods: (5) **Kish Effective Sample Size** (Kish 1965) -- true information content from interpolation weights. (6) **Inverse-Variance Blending** (Cochrane 1954) -- Cox/sigmoid blend from prediction variance. (7) **Lambda PID Tuning** (Dahlin 1968) -- PID gains from error ACF half-life. (8a) **Copula Tail Contagion** (Joe 2014) -- edge-specific contagion from tail dependence. (8b) **Amihud Participation Rate** (Amihud 2002) -- liquidation rate from illiquidity. (9a) **Regime Risk Multiplier** -- HMM volatility ratio. (9b) **Garman-Klass Factor** (1980) -- intraday low from OHLC volatility. (9c) **Transition Half-Life** -- from enriched timeline switch durations. (10) **Precision-Targeted MC** (Glasserman 2003) -- path count for target SE. Also includes Category A (train splits, confidence bounds, percentile scoring) and Category C (conflict scoring, OHLC noise, Hurst exponent, bootstrap spread). | Same | OK |
| **Downstream** | Consumed by `survival_probability` (re-blended), `run_monte_carlo()` (n_paths, is_tilt), `run_forward_pass()` (PID gains), graph_risk (contagion probs), ownership_contagion (participation rate) | Same | OK |

### Dx3. Adaptive Windows & Hyperparameters -- Tier 3 constants

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `filing_calendar_result.detected_frequency`, `cache`, Kish n_eff | Same | OK |
| **Output** | `AdaptiveTier3Params` with: filing-anchored windows, NN hyperparams, particle noise, pattern thresholds, stale threshold, entity scoring weights | Same | OK |
| **Location** | `operator1/analysis/adaptive_windows.py` (553 lines) | | |
| **Operation** | 7 methods: (11) **Filing-Frequency-Anchored Windows** (Nyquist-Shannon) -- all rolling windows are integer multiples of filing period (short=base/3, medium=base, long=2*base, trend=4*base). (12) **Scaling-Law NN Architecture** (Kaplan 2020) -- d_model/hidden_dim proportional to n_eff. (13) **Innovation-Based Particle Noise** (Mehra 1970) -- state/obs noise from difference series std. (14) **Distribution-Based Pattern Thresholds** (Bulkowski 2008) -- doji=P10, body=P50 of body/range ratio. (16) **Confidence Decay Half-Life** -- proportional to filing period. (18) **Filing-Frequency Staleness** -- 2x filing period. (19) **Market-Specific Entity Scoring** -- CJK markets get higher ticker weight, lower name weight. | Same | OK |
| **Downstream** | Windows consumed by derived_variables, forecasting, forward_pass, burnout. NN params consumed by transformer_forecaster. Pattern thresholds consumed by pattern_detector. Staleness consumed by filing_calendar. | Same | OK |

---

## Phase E: Enriched Survival Timeline

### E1. Early Regime Detection -- `run_early_regime_detection()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache with `return_1d` and `volatility_21d` | Same | OK |
| **Output** | cache + `regime_hmm`, `regime_gmm`, `regime_label`, `structural_break`, `breakpoint_method`, `regime_hmm_prob_*` columns; `early_regime_result` with `detector` object | Same | OK |
| **Operation** | 1. **HMM** (4-regime Gaussian on returns+volatility). 2. **GMM** (unsupervised clustering). 3. **PELT** (structural break detection). 4. **BCP** (Bayesian change point). 5. **NEW: ChangeFinder** online change point detection (real-time SDAR, no look-ahead). Each wrapped in try/except with graceful fallback. | Same | **ENHANCED** |

### E2. Enriched Survival Timeline -- `compute_enriched_survival_timeline()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `regime_labels`, `regime_confidence` | Same | OK |
| **Output** | `enriched_timeline_result` with `.timeline` DataFrame containing: `regime_state`, `survival_intensity`, `regime_confidence`, `regime_switch`, `regime_transition_prob`, `survival_mode`, `survival_mode_code`, `switch_point`, `days_in_mode`, `stability_score_21d`, `market_regime` | Same -- merged into cache via reindex | OK |
| **Operation** | 1. Base survival timeline (6-mode classification). 2. Merge with HMM regime labels. 3. Compute survival intensity (continuous 0-1 metric). 4. Add transition probabilities and switch detection. | Same | OK |

---

## Phase F: Temporal Modeling

### F1. Regime Detector -- `detect_regimes_and_breaks()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache | Same -- SKIP if early detection already ran in Step 5.5 | OK |
| **Output** | cache + regime columns, `regime_detector` object | Same | OK |
| **Operation** | Same as E1. Guarded: only runs if `regime_label` not already in cache. | Same | OK |

### F2. Dual Regime Mixer -- `compute_dual_regimes()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache (with `regime_label` or `regime_hmm` columns) | Same | OK |
| **Output** | `dual_regime_result` with `market_regime_labels`, `fund_regime_labels`, `blended_weights` | Same | OK |
| **Operation** | 1. Extract market regime from existing HMM columns. 2. Classify fundamental regime from financial ratios (healthy/stressed/distress). 3. Compute blended weights for soft switching. | Same | OK |

### F3. Granger Causality -- `compute_granger_causality()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `variables` list (up to 25 float columns with >50 non-NaN obs) | Same | OK |
| **Output** | `granger_result` with `causality_matrix`, `significant_pairs`, `retained_variables`, `pruned_variables`, `network_density` | Same | OK |
| **Operation** | **PCMCI via tigramite** (preferred, handles autocorrelation and confounders) or pairwise Granger F-tests (fallback). Prune variables with no causal links. **NEW: `compute_time_varying_granger()` for rolling-window temporal causal graph.** Result used to prune `_extra_vars` list fed to temporal models. | Same | **ENHANCED** |
| **Downstream** | `prune_features_by_causality()` removes non-causal variables from `_extra_vars` | Same | OK |

### F4. Transfer Entropy -- `compute_transfer_entropy()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `variables` list (up to 20 float columns with >30 non-NaN obs) | Same | OK |
| **Output** | `transfer_entropy_result` with pairwise entropy scores | Same | OK |
| **Profile storage** | Should store full result via `_available_dict()` | Was storing only `{"available": True}` | FIXED |

### F5. Cycle Decomposition -- `run_cycle_decomposition()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `variable="close"` | Same | OK |
| **Output** | `cycle_result` with `dominant_cycles` list (period, amplitude, phase) | Same | OK |
| **Operation** | **EMD (CEEMDAN) via EMD-signal** (preferred, adaptive for non-stationary data) or FFT-based spectral analysis (fallback) to identify dominant periodicities | Same | **ENHANCED** |
| **Downstream** | Fed to `inject_cycle_phase_features()` (Synergy B) to add `cycle_phase_*` columns to cache | Same | OK |

### F6. Pattern Detector -- `detect_patterns()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache (needs `open`, `high`, `low`, `close`) | Same | OK |
| **Output** | `pattern_result` with detected candlestick patterns (doji, hammer, engulfing, etc.), **motifs** (recurring patterns via stumpy Matrix Profile), **discords** (anomalies) | Same | **ENHANCED** |
| **Downstream** | Fed to `compute_pattern_drift_adjustment()` (Synergy C) for OHLC predictor drift | Same | OK |

### F7. Pre-Forecasting Synergies -- `apply_pre_forecasting_synergies()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `cycle_result`, `granger_result`, `transfer_entropy_result`, `linked_caches`, `extra_variables`, `economic_plane` | Same | OK |
| **Output** | cache (+ synergy features), `_extra_vars` (updated), `_synergy_meta` dict | Same | OK |
| **Operation** | 1. Inject cycle phase features (Synergy B). 2. Build unified causal network from Granger + TE (Synergy D). 3. Compute peer-adjusted survival thresholds (Synergy G). 4. Apply plane-aware model weights. | Same | OK |

### F8. Forecasting -- `run_forecasting()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `extra_variables` list | Same | OK |
| **Output** | cache + forecast columns, `ForecastResult` with `.forecasts` (per-var, per-horizon point forecasts), `.metrics` (per-model RMSE/MAE), `.model_used`, `.residuals` | `.residuals` was None (never populated) | FIXED |
| **Operation** | For each variable, try models in order: 1. Kalman (local-level state-space). 2. GARCH (conditional volatility). 3. VAR (multivariate, AR(1) fallback). 4. LSTM (PyTorch, GBM/LR fallback). 5. Tree ensemble (RF/GBM/XGB). 6. Baseline (last-value or EMA). First model that fits successfully wins. **NEW standalone functions**: `fit_autoarima()` via statsforecast (100x faster) and `fit_dynamic_factor()` (multi-variable DFM). | Same | **ENHANCED** |

### F9. Forward Pass -- `run_forward_pass()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `hierarchy_weights`, `regime_labels`, `extra_variables` | Same | OK |
| **Output** | `ForwardPassResult` with `errors_by_tier`, `errors_by_regime`, `model_states`, `predictions_log`, `total_days`, `pid_summary` | Same | OK |
| **Operation** | Day-by-day temporal walk: for each day t, predict t+1 using warmup models, compare to actual, update PID controller for adaptive learning rate adjustment. | Same | OK |

### F10. Walk-Forward -- `run_walk_forward()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `daily_cache`, `SurvivalTimelineResult` | Same (after fix) | FIXED |
| **Output** | `WalkForwardResult` with `day_errors`, `mode_scores`, `best_model_by_mode`, `retrain_dates`, `overall_best_model`, `overall_mae` | Was never called; `ForwardPassResult` was passed instead | FIXED |
| **Operation** | 1. Walk day-by-day using 4 model types (baseline, EMA, linear trend, mean reversion). 2. Track per-model per-survival-mode error. 3. Retrain at switch points. 4. Build mode-conditioned leaderboard. **NEW**: `aggregate_forward_pass_errors()` extracts per-model per-mode errors from forward pass. `compute_mode_confidence_sets()` via `arch.bootstrap.MCS` identifies statistically equivalent models per mode. | Was dead code -- now wired into pipeline | **ENHANCED** |

### F11. Burn-Out -- `run_burnout()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `hierarchy_weights`, `regime_labels`, `extra_variables` | Same | OK |
| **Output** | `BurnoutResult` with `iterations_completed`, `converged`, `best_rmse_by_tier` | Same | OK |
| **Profile storage** | Should appear in profile `extended_models` | Was not stored in profile | FIXED |
| **Operation** | Intensive re-training on recent data with convergence detection. Each iteration: reset to burnout_window, run forward pass with higher learning rates, measure accuracy, early-stop if no improvement. | Same | OK |

### F12. Monte Carlo -- `run_monte_carlo()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache (needs `return_1d`, `regime_label`) | Same | OK |
| **Output** | `MonteCarloResult` with `survival_probability` per horizon, `regime_distributions`, `transition_matrix`, `terminal_values` | Same | OK |
| **Operation** | 1. Estimate per-regime return distributions. 2. Build regime transition matrix. 3. Simulate 10,000 paths with regime switching. 4. Apply importance sampling for tail events. 5. Compute survival probability (fraction of paths not triggering survival thresholds). **NEW**: `run_multivariate_monte_carlo()` jointly simulates (return, delta_current_ratio, delta_fcf_yield, delta_debt_to_equity) using copula correlation structure, checking survival triggers on simulated ratios directly. | Same | **ENHANCED** |

### F13. Copula -- `run_copula_analysis()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `variables` (auto-selected if None) | Same | OK |
| **Output** | `CopulaResult` with `copula_correlation`, `tail_dependence`, `joint_crisis_probability` | Same | OK |
| **Operation** | 1. Transform to uniform marginals (PIT). 2. Fit **Gaussian, Student-t, and Clayton copulas** (via `copulae` library). 3. **Select best by AIC**. 4. Estimate lower tail dependence from best-fitting copula. 5. Estimate joint crisis probability. | Same | **ENHANCED** |
| **Note** | Zero-variance columns now handled with noise injection (P2 fix) | FIXED | OK |

### F14. Transformer -- `train_transformer()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `variables` (up to 15 float columns with >100 non-NaN obs) | Same | OK |
| **Output** | `TransformerResult` with `forecasts`, `feature_importance`, `train_loss_history` | Same -- forecasts injected into `forecast_result` | OK |
| **Operation** | 1. Build sliding window sequences. 2. Train TemporalTransformer (multi-head self-attention + positional encoding). 3. Produce 1-step forecasts for each variable. | Same | OK |

### F15. Particle Filter -- `run_particle_filter()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `variables` (survival-related: cash_ratio, free_cash_flow_ttm, current_ratio, debt_to_equity) | Same | OK |
| **Output** | `ParticleFilterResult` with `filtered_states`, `particles_final`, `weights_final`, `percentiles` | Same | OK |
| **Operation** | Sequential Monte Carlo: maintain particle swarm, propagate via random walk, update weights via Gaussian likelihood, systematic resample when ESS drops. | Same | OK |

### F16. Conformal Prediction -- `build_conformal_result()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `ConformalCalibrator` (fed with residuals), `forecasts` dict, `horizons` dict | Calibrator was getting no residuals (always empty) | FIXED |
| **Output** | `ConformalResult` with distribution-free prediction intervals per variable per horizon | Same | OK |
| **Operation** | 1. **ConformalPIDCalibrator** (preferred) with PID-controlled coverage (Angelopoulos 2023) + hierarchical Mondrian per-survival-mode partitioning, or standard ConformalCalibrator (fallback). 2. Computes conformal quantile at target coverage. 3. Applies PID-adaptive adjustment for non-stationarity. 4. Builds intervals: forecast +/- conformal quantile. | Same -- now receives residuals from ForecastResult | **ENHANCED** |

### F17. DTW Analogs -- `find_historical_analogs()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache | Same | OK |
| **Output** | `DTWAnalogResult` with historical pattern matches and empirical forecasts | Same | OK |
| **Operation** | 1. Extract recent window of close prices. 2. DTW distance search over historical windows **+ cross-company search using linked_caches** (peer analogs weighted 0.7x). 3. Rank analog matches. 4. Derive empirical forecast from what happened after each analog. | Same | **ENHANCED** |

### F18. Prediction Aggregator -- `run_prediction_aggregation()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `forecast_result`, `mc_result`, `conformal_result`, `dual_regime_result`, `copula_result`, `dtw_result`, `granger_result`, `shap_result`, `walk_forward_result` | `walk_forward_result` was receiving `ForwardPassResult` (wrong type) | FIXED |
| **Output** | `PredictionAggregatorResult` with `predictions` (per-var, per-horizon `HorizonPrediction`), `technical_alpha` mask, `metadata` | Same | OK |
| **Operation** | 1. Compute inverse-RMSE ensemble weights **or FixedShareForecaster weights filtered by Model Confidence Sets**. 2. Apply regime blending (soft switching from dual_regime_result). 3. Aggregate point forecasts. 4. Build uncertainty bands (ConformalPIDCalibrator with Mondrian preferred, RMSE fallback). 5. Widen bands using copula tail dependence. 6. Add DTW analog forecasts. 7. Propagate Granger/PCMCI causal adjustments. 8. Attach SHAP explanations. 9. Apply Technical Alpha mask (hide OHLC except Low). 10. Recency-weighted RMSE from walk-forward + MCS-filtered mode-conditioned weights. | Step 10 was dead (always NaN) -- now active | **ENHANCED** |

### F19. SHAP Explainability -- `compute_shap_explanations()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `predictions` dict, `tree_models` dict (fitted model objects), `predict_fns` dict | Same -- now extracts `predict_fns` from `forward_pass_result.model_states` | OK |
| **Output** | `SHAPResult` with per-variable feature importance, top drivers, narratives | Same | OK |
| **Operation** | 1. Try TreeExplainer with tree_models. 2. Fall back to KernelExplainer with predict_fns. | Same | OK |

### F20. Sobol Sensitivity -- `run_sensitivity_analysis()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `target_variable="return_1d"` | Same | OK |
| **Output** | `SobolResult` with first-order and total-order sensitivity indices | Same | OK |
| **Operation** | Sobol variance-based global sensitivity analysis using Saltelli sampling | Same | OK |

### F21. Genetic Optimizer -- `run_genetic_optimization()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `forecast_result` | Same | OK |
| **Output** | `GAResult` with `best_weights`, `tier_weights`, `fitness_history`, `converged` | Same | OK |
| **Operation** | **Optuna TPE** tried first (faster convergence). Fallback: 1. Initialize population via Dirichlet distribution. 2. Evaluate fitness as -RMSE of weighted ensemble. 3. Tournament selection, crossover, mutation. 4. Elite carryover. 5. Converge after N generations. **NEW**: per-regime GA weight optimization. | Same | **ENHANCED** |

### F22. OHLC Predictor -- `predict_ohlc_series()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `forecast_result`, `mc_result`, `pattern_drift_multiplier` | Same | OK |
| **Output** | `OHLCResult` with next-day OHLC predictions | Same | OK |
| **Operation** | Combine forecast + Monte Carlo + pattern drift to predict next-day Open, High, Low, Close | Same | OK |

### F23. Regime Shift Predictor -- `predict_regime_shifts()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `transition_matrix` (from MC), `regime_order`, `stability_score`, `transition_halflife` (from adaptive params), `reference_date` | Same | OK |
| **Output** | `RegimeShiftResult` with `prob_exit_21d`, `prob_exit_252d`, `expected_days_to_shift`, `most_probable_next_regime`, `current_regime`, `transition_matrix_used` | Same | OK |
| **Location** | `operator1/models/regime_shift_predictor.py` (351 lines) | | |
| **Operation** | Uses HMM transition matrix from Monte Carlo to predict when the current regime is likely to change and to which regime. Computes geometric CDF of regime exit. Adjusts by stability_score and transition_halflife from adaptive params. | Same | OK |
| **Profile** | Stored in `profile["predicted_regime_shifts"]` via `result.to_dict()` | Same | OK |

### F24. Model Diagnostics -- `compute_model_diagnostics()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `forecast_result`, `mc_result`, `copula_result`, `granger_result`, `cycle_result`, `dtw_result`, `conformal_result` | Same | OK |
| **Output** | `ModelDiagnosticsResult` with `n_models_assessed`, `n_models_on_track`, `overall_robustness`, per-model `expected_path` vs `actual_path` comparison | Same | OK |
| **Location** | `operator1/monitoring/model_diagnostics.py` (875 lines) | | |
| **Operation** | For each of 10 models (Kalman, GARCH, VAR, LSTM, Tree, MC, Copula, Granger, Cycle, DTW, Conformal), pre-computes what it SHOULD produce based on data characteristics, then compares against what it actually produced. Produces per-model robustness ratings (on_track/degraded/failed). | Same | OK |
| **Profile** | Stored in `profile["model_diagnostics"]` via `result.to_dict()` | Same | OK |
| **Report** | Rendered in sections 19.95-19.98 (Unified Survival System + Scenario Analysis + Model Diagnostics) in Premium tier | Same | OK |

---

## Phase F-extra: Modules Called Indirectly

### Fx1. PID Controller -- `compute_pid_adjustment()` (called inside forward pass)

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | Per-tier prediction error history | Same -- receives errors from forward pass loop | OK |
| **Output** | Adaptive learning rate multiplier per tier | Same | OK |
| **Operation** | PID (Proportional-Integral-Derivative) control loop that adjusts model learning rates based on prediction error trends. Prevents overshoot and oscillation during the forward pass. | Same | OK |

### Fx2. Macro Alignment -- `align_yearly_series_to_daily()` (called inside macro_quadrant)

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | Yearly or quarterly macro Series, daily DatetimeIndex | Same -- called from macro_quadrant.py | OK |
| **Output** | Daily-frequency Series with forward-filled macro values | Same | OK |
| **Operation** | Reindex a low-frequency macro series onto a daily business day index using forward-fill. Handles timezone and frequency mismatches. | Same | OK |

### Fx3. Ethical Filters -- `compute_all_ethical_filters()` (called inside profile_builder)

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache DataFrame | Same -- called from profile_builder.py line 762 | OK |
| **Output** | Ethical filter results: manipulation signals, quality flags | Same | OK |
| **Operation** | Scan financial data for signs of earnings manipulation, aggressive accounting, or data quality red flags. Complements Beneish M-Score from financial_health. | Same | OK |

---

## Phase F-extra-2: Geopolitical Risk Assessment

### Fx4. Conflict Risk Assessment -- `assess_conflict_risk()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `country_iso2` (ISO-2 code from VerifiedTarget), optional `company_name` | Same | OK |
| **Output** | `ConflictRiskResult` with `country_conflict_flag`, `company_conflict_flag`, `conflict_intensity_score` (0-1), `sanctions_flag`, `fragile_state_flag`, `conflict_type`, `recent_events_30d/90d`, `recent_fatalities_30d`, `conflict_trend`, `news_conflict_mentions_7d`, `news_conflict_tone`, `data_sources_used`, `confidence` | Same | OK |
| **Operation** | 1. Check static lists (World Bank FCS: 31 countries, OFAC/EU sanctions: 13 countries, active wars: 9 countries). 2. Fetch UCDP GED events (free, no key -- currently returns 401, falls back gracefully). 3. Query GDELT news (free, no key, 5s rate limiting). 4. Compute weighted intensity score (40% events + 20% fatalities + 25% flags + 15% news). 5. Set country/company conflict flags. | Same | OK |
| **Location** | `operator1/features/conflict_risk.py` | | |
| **Data sources** | UCDP GED API (auth required since 2025), GDELT (rate-limited), static lists (always available) | | |

### Fx5. Linked Entity Conflict Propagation -- `assess_linked_entity_conflict()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `linked_entities` dict from entity discovery (`{group_name: [entity_dict, ...]}`), `target_conflict` (ConflictRiskResult) | Same | OK |
| **Output** | Dict with `supply_chain_risk_score` (0-1), `revenue_exposure_score` (0-1), `competitive_advantage_score` (0-1), `linked_entities_in_conflict` (list), `linked_conflict_summary` (str) | Same | OK |
| **Operation** | For each linked entity group: check entity country against static conflict lists. Supplier/logistics in war zone -> supply chain risk. Customer in conflict -> revenue risk. Competitor in conflict -> competitive advantage. Financial institution -> credit risk (0.5x weight). Max severity per category. | Same | OK |
| **Location** | `operator1/features/conflict_risk.py` | | |

### Fx6. Conflict Risk Cache Injection -- `inject_conflict_risk_into_cache()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `cache` DataFrame, `ConflictRiskResult`, optional `linked_conflict` dict | Same | OK |
| **Output** | cache + columns: `country_conflict_flag`, `company_conflict_flag`, `conflict_intensity_score`, `sanctions_flag`, `fragile_state_flag`, `conflict_type`. Optional: `supply_chain_risk_score`, `revenue_exposure_score`, `competitive_advantage_score` | Same | OK |
| **Operation** | Set daily columns from conflict assessment result. All values are constant across the daily index (conflict status doesn't change day-to-day within a pipeline run). | Same | OK |
| **Downstream** | Consumed by `compute_company_survival_flag()` in `survival_mode.py` (conflict flag + sanctions flag are new survival triggers). Consumed by `_build_conflict_risk_profile_section()` in `profile_builder.py`. | Same | OK |

### Fx6b. Supply Chain Stress -- `compute_supply_chain_stress()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `conflict_result`, `linked_caches`, `relationships` | Same | OK |
| **Output** | Dict with `supply_chain_stress_flag`, `supply_chain_stress_score` (0-1), `stress_sources` list | Same | OK |
| **Operation** | Combines geopolitical risk + supplier financial health into one unified supply chain stress signal. Injected into cache as `supply_chain_stress_flag` and `supply_chain_stress_score`. | Same | OK |
| **Profile** | Stored in `profile["supply_chain_stress"]` | Same | OK |

### Fx6c. Predicted Next Filing Date -- `predict_next_filing_date()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `filing_calendar_result`, `reference_date` | Same | OK |
| **Output** | Dict with `predicted_date`, `confidence`, `method` | Same | OK |
| **Location** | `operator1/features/filing_calendar.py` | | |
| **Operation** | Extrapolates next expected filing date from detected filing frequency and last filing date. Uses median inter-filing gap + market-specific calendar adjustments. | Same | OK |
| **Profile** | Stored in `profile["filing_calendar"]["next_expected_filing"]` | Same | OK |

### Fx6d. Predicted OHLC Patterns -- `detect_patterns_on_predicted_ohlc()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `ohlc_result` (predicted candles from OHLC predictor), `last_candle` (latest actual) | Same | OK |
| **Output** | List of pattern dicts with `pattern_name`, `date`, `confidence` for next week's predicted candles | Same | OK |
| **Location** | `operator1/models/pattern_detector.py` | | |
| **Operation** | Runs candlestick pattern detection on the predicted OHLC series (from OHLC predictor). Detects doji, hammer, engulfing, etc. on forward-looking candles. Results stored in `pattern_result.predicted_patterns_week`. | Same | OK |
| **Profile** | Stored in `profile["extended_models"]["candlestick_patterns"]` via `_available_dict()` | Same | OK |

---

## Phase F-extra-3: Filing Discovery for Tier 2 Markets

### Fx7. BSE India Filing Discoverer -- `BSEFilingDiscoverer.discover_filings()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | BSE scrip code (e.g. `500325`), `years` lookback | Same | OK |
| **Output** | `FilingDiscovery` with list of `FilingMetadata` objects (title, filing_date, report_date, document_url, filing_type) | Same | OK |
| **Operation** | Query `api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w` with `strCat=Result`. Parse subject lines for fiscal period dates. Classify as annual/quarterly/interim. Construct PDF URLs from attachment UUIDs. | Same | OK |
| **Location** | `operator1/clients/filing_discoverer.py` | | |
| **Rate limit** | No documented limit; Referer header required | | |

### Fx8. BSE Filing PDF Download -- `BSEFilingDiscoverer.download_filing()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `FilingMetadata` with `document_url` | Same | OK |
| **Output** | Raw PDF bytes (validated: first 4 bytes must be `%PDF`) | Same | OK |
| **Operation** | HTTP GET to `bseindia.com/xml-data/corpfiling/AttachLive/{uuid}.pdf`. Validates PDF magic bytes. Rejects non-PDF content. | Same -- tested with Reliance 4.7MB PDF | OK |

### Fx9. ASX Filing Discoverer -- `ASXFilingDiscoverer.discover_filings()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | ASX ticker (e.g. `BHP`), `years` lookback | Same | OK |
| **Output** | `FilingDiscovery` with announcement metadata (headline, date, documentKey, filing_type) | Same | OK |
| **Operation** | Query `asx.api.markitdigital.com/asx-research/1.0/companies/{ticker}/announcements`. Filter by announcement type (PERIODIC REPORTS, ANNUAL REPORT) or headline keywords. | Same | OK |
| **Note** | Document download returns 404 for the MarkitDigital endpoint. Discovery-only for now. | Same | OK |

### Fx10. Filing Extraction Pipeline -- `try_filing_extraction()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `ticker`, `market_id`, `statement_type` (income/balance/cashflow), optional `llm_client` | Same | OK |
| **Output** | Canonical long-format DataFrame (columns: canonical_name, value, report_date, filing_date) filtered to the requested statement type | Same | OK |
| **Operation** | 1. Check per-ticker extraction cache (avoids redundant calls). 2. Get discoverer from registry. 3. Discover filings. 4. Download PDFs. 5. Extract via LLMFilingExtractor. 6. Cache combined result. 7. Filter by statement_type using canonical field name sets. | Same | OK |
| **Caching** | Module-level `_extraction_cache` keyed by `market_id:ticker`. First call does full pipeline; subsequent calls (income/balance/cashflow) return cached+filtered result. | Same | OK |
| **Location** | `operator1/clients/filing_discoverer.py` | | |
| **Wired in** | `operator1/clients/in_bse.py` -- tried before yfinance fallback in `_fetch_financials()` | Same | OK |

---

## Unwired Modules -- Investigation Results

### W6: `vanity.py` -- WIRED

**Status:** FIXED
**Location:** `operator1/analysis/vanity.py` (637 lines)
**Tests:** `tests/test_vanity_v2.py` (dedicated), referenced in `test_phase4_analysis.py`, `test_phase7_report.py`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache with derived vars, fh_* scores, sentiment (optional), peer_ranking (optional) | Same -- wired in main.py Step 5d after financial health | OK |
| **Output** | cache + `vanity_score`, `vanity_label`, `vanity_trend`, 5 component columns | Same | OK |
| **Operation** | 5-component composite: R&D mismatch, SGA bloat, capital misallocation, competitive decay, sentiment gap | Same | OK |

### W7: `supplement.py` -- WIRED

**Status:** FIXED
**Location:** `operator1/clients/supplement.py` (549 lines)
**Tests:** `tests/test_supplement_and_translator.py`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `profile` dict, `ticker`, `market_id` | Same -- wired in main.py after Step 2 profile fetch | OK |
| **Output** | Enriched profile with sector, industry, identifiers filled from OpenFIGI/regional APIs | Same | OK |
| **Operation** | OpenFIGI lookup + per-region enrichers (Euronext, JPX, TWSE, B3, Santiago) | Same | OK |

### Genuinely Unwired (Planned Features / Legacy)

| Module | Location | Lines | Tests | Status | Assessment |
|--------|----------|-------|-------|--------|------------|
| `portfolio_analysis.py` | `features/` | 283 | None | **Removed** | Intentionally deleted. User portfolio context is out of scope (Operator 1 analyzes companies, not portfolios). Institutional overlap (Part 2) blocked on data -- free PIT APIs don't provide 13F/holder data. |
| `llm_filing_extractor.py` | `clients/` | 571 | Used in `live_helpers.py` + `filing_discoverer.py` | **Wired** | Now integrated via `try_filing_extraction()` in `filing_discoverer.py`. BSE India discoverer downloads PDFs and feeds them to `extract_from_pdf()`. ASX discoverer provides announcement metadata (PDF download TBD). |
| `data_extraction.py` | `steps/` | 392 | Referenced by `cache_builder.py` for types | Not wired | **Legacy code.** Superseded by `main.py` inline extraction logic. Only used for `EntityData` type import by `cache_builder.py`. |
| `verify_identifiers.py` | `steps/` | 129 | None direct | Not wired | **Legacy code.** Superseded by `main.py` inline verification. Only imported by `data_extraction.py` for `VerifiedTarget` type. |

---

## Phase G: Profile Building

### G1. Profile Builder -- `build_company_profile()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `target_profile`, cache, `linked_aggregates`, all model results (regime, forecast, MC, prediction, estimation, graph_risk, game_theory, fuzzy, financial_health, sentiment, peer_ranking, macro_quadrant) | Same -- now also includes burnout_result, transfer_entropy_result, and conflict_risk | OK |
| **Output** | `profile` dict with all sections: identity, financials, survival, models, extended_models, conflict_risk, meta | Same -- `conflict_risk` section added via `_build_conflict_risk_profile_section(cache)` | OK |
| **Operation** | Assemble all model outputs into a single JSON-serializable profile dict. Conflict risk is built from cache columns: `country_conflict_flag`, `conflict_intensity_score`, `sanctions_flag`, `fragile_state_flag`, `conflict_type`, plus optional linked entity scores. | Same | OK |

---

## Phase H: Report Generation

### H1. Report Generator -- `generate_all_reports()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `profile` dict, `gemini_client` (LLM), cache, `output_dir` | Same | OK |
| **Output** | Basic + Pro + Premium markdown reports, optional PDF, embedded chart images | Same | OK |
| **Operation** | 1. Extract key metrics from profile. 2. Generate narrative via LLM (or template fallback). 3. Generate charts (now for Pro + Premium tiers). 4. Embed chart `![](charts/filename.png)` references into markdown. 5. Three tiers with increasing detail. 6. New section 19.5: Geopolitical & Conflict Risk (Pro + Premium). 7. New Chart 9: Conflict Risk Dashboard (intensity gauge + status indicators). | Same | OK |
| **Chart embedding** | `_embed_charts_in_markdown()` maps 9 chart filenames to section headings and inserts image tags after matching headings. Charts generated for Pro and Premium tiers (was Premium-only). | Same | OK |
| **Section 19.5** | `_build_geopolitical_risk_section(profile)` renders status badge (CRITICAL/ELEVATED/MODERATE/LOW), risk factor table, UCDP event counts, GDELT news tone, linked entity conflict exposure table, investment implications narrative. Included in `TIER_SECTIONS` for Pro (section 195) and Premium. | Same | OK |

---

## Summary of All Fixes Applied

| Issue | Location | What was wrong | What was fixed |
|-------|----------|----------------|----------------|
| W1 | main.py:1728 | `ForwardPassResult` passed as `walk_forward_result` | Now passes actual `WalkForwardResult` |
| W2 | main.py | `run_walk_forward()` never called | Now called after burn-out phase |
| W3 | forecasting.py | `ForecastResult.residuals` did not exist | Added field + populated from validation RMSE |
| W4 | main.py | `burnout_result` not in profile | Now stored in `extended_models` |
| W5 | main.py:1962 | `transfer_entropy_result` stored as `{"available": True}` | Now uses `_available_dict()` for full data |
| P1 | estimator.py:542 | PerformanceWarning from fragmented DataFrame | Warning suppressed; defragmentation already occurs via `.copy()` |
| P2 | copula.py:69-74 | NaN from `np.corrcoef` on zero-variance columns | Zero-variance guard + NaN fallback to identity |
| W6 | vanity.py | Profile builder expects vanity columns but `compute_vanity_scores()` never called | **FIXED** -- wired `compute_vanity_score(cache)` in main.py Step 5d after financial health |
| W7 | supplement.py | Profile enrichment for non-US markets never called | **FIXED** -- wired `enrich_profile()` in main.py after Step 2 profile fetch |
| W8 | main.py:1109 | `analyze_competitive_dynamics()` never receives `linked_caches` -- always returns "monopoly" with 0 competitors | **FIXED** -- now passes `competitor_caches=linked_caches` |
| W9 | main.py:1758 | `compute_shap_explanations()` never receives `tree_models` or `predict_fns` -- explanations always empty | **FIXED** -- now extracts `predict_fns` from `forward_pass_result.model_states` |
| P3 | derived_variables.py:662 | PerformanceWarning from 39 column-by-column insertions across compute stages | Added `result.copy()` defragmentation before return |
| S2 | genetic_optimizer.py:182 | GA uses EWM/shifted proxy predictions instead of actual per-model forecast arrays | **IMPROVED** -- GA now seeds half the population with inverse-RMSE-informed weights from `forecast_result.metrics`, giving the optimizer a head start from the real model performance |
| S14 | forecasting.py:1682 | Conformal calibrator receives synthetic `+/-RMSE` pairs (~12-14 points) instead of actual validation residuals (hundreds) | **FIXED** -- Added `test_residuals` field to `ModelMetrics`; Kalman, LSTM, tree, and baseline wrappers now store actual residuals; collection logic prefers real residuals with synthetic fallback |
| S15 | news_sentiment.py:1 | Docstring says "Fetches stock news from FMP (1 API call)" but FMP was removed. Actual source is GNews/RSS. `_fetch_news_alpha_vantage()` at line 190 is dead code (redirects to gnews). | **FIXED** -- docstring updated |
| S16 | constants.py:8-9 | `DATE_END = date.today()` and `DATE_START` computed at import time, not at call time. In a long-running process or multi-day library use, the date window is stale. | **FIXED** -- added `get_date_window()` function; module-level constants kept for backward compatibility |
| B3 | financial_health.py:379,383 | PE ratio capped at fixed 200, EV/EBITDA capped at fixed 100 regardless of sector/distribution | **FIXED** -- adaptive caps via 3-method consensus: Log-Normal P99.5 (Aitchison & Brown 1957), Tukey Extreme Fence (Tukey 1977), MAD-Based Cap (Iglewicz & Hoaglin 1993). Floor/ceiling bounds preserved. |
| R1 | report_generator.py:2214-2353 | 140 lines of unreachable dead code after `return` in `_build_economic_position()`, duplicating `_build_appendix()` methodology | **FIXED** -- dead code deleted, file reduced from 4151 to 4034 lines |
| R2 | report_generator.py:2932 | `_build_economic_position()` existed but was not wired in `_section_builders` -- economic plane data invisible in reports | **FIXED** -- added as section 75 ("Economic Position & Industry Classification") in PRO and PREMIUM tiers |
| R3 | report_generator.py:4049 | LLM validation logged "Attempting to append missing section" but never actually appended -- complete no-op | **FIXED** -- implemented auto-patching: missing sections now appended from fallback template builders with keyword matching |
| R4 | report_generator.py:1141,1159,1190 | Section 9 (Predictions) used wrong profile keys for conformal (`conformal_intervals` vs `extended_models.conformal_prediction`), SHAP, and DTW analogs -- all always empty | **FIXED** -- changed to read from `extended_models` sub-dict where data actually lives |
| R5 | report_generator.py:2577 | Stale `geopolitical_risk` fallback key never produced by any module | **FIXED** -- removed dead fallback |
| R6 | profile_builder.py:675 | `_build_historical_section()` crashes on `cache=None` (None guard placed after `.columns` access) | **FIXED** -- moved None guard before `.columns` access |
| R7 | test_phase7_report.py:222,679,708 | 17 tests failing due to obsolete `output_path` parameter and stale file existence assertions | **FIXED** -- removed `output_path`, updated assertions to test returned dict directly (34/34 pass) |

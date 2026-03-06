# Complete Data Flow and Model Contract Map

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
| **Output** | cache + ~25 derived columns: `return_1d`, `log_return_1d`, `volatility_21d`, `drawdown_252d`, `current_ratio`, `debt_to_equity_abs`, `fcf_yield`, `cash_ratio`, `gross_margin`, `pe_ratio_calc`, etc. + `is_missing_*` and `invalid_math_*` flags | Same | OK |
| **Operation** | 1. Returns and risk from close price. 2. Solvency ratios (debt_to_equity, net_debt). 3. Liquidity ratios (current_ratio, cash_ratio). 4. Profitability (margins). 5. TTM computations. All use `safe_ratio()` to handle division by zero. | Same | OK |

### C2. Survival Mode -- `compute_company_survival_flag()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache with `current_ratio`, `debt_to_equity_abs`, `fcf_yield`, `drawdown_252d` | Same | OK |
| **Output** | `cache["company_survival_mode_flag"]` -- integer Series: 1=survival, 0=normal | Same | OK |
| **Operation** | OR of 4 conditions: current_ratio<1.0, debt_to_equity_abs>3.0, fcf_yield<0, drawdown_252d<-0.40 | Same -- thresholds loaded from config | OK |

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
| **Operation** | Fuzzy logic: compute sector strategicness membership, economic significance (market_cap/GDP), policy responsiveness (rate cuts). Aggregate via fuzzy OR (max). | Same | OK |

### C5. Financial Health -- `compute_financial_health()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `hierarchy_weights` dict | Same | OK |
| **Output** | cache + `fh_liquidity_score`, `fh_solvency_score`, `fh_stability_score`, `fh_profitability_score`, `fh_growth_score`, `fh_composite_score`, `fh_composite_label`, `fh_altman_z_score`, `fh_altman_z_zone`, `fh_beneish_m_score`, `fh_beneish_flag`, `fh_runway_months`; `fh_result` (FinancialHealthResult) | Same | OK |
| **Operation** | 1. Per-tier scoring (expanding percentile rank normalization). 2. Weighted composite (uses hierarchy_weights). 3. Altman Z-Score (5 coefficients from 1968 paper). 4. Beneish M-Score (8 coefficients from 1999 paper). 5. Liquidity runway (cash / monthly burn rate). | Same | OK |

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
| **Operation** | 1. Fetch news via gnews/feedparser. 2. Score sentiment via LLM (if available) or keyword fallback. 3. Write daily sentiment score to cache. | Same | OK |

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
| **Operation** | Build network graph from entity relationships. Compute degree centrality, betweenness centrality, clustering coefficient for the target node. Identify systemically important connections. | Same | OK |
| **Profile** | Stored via `_available_dict(graph_risk_result)` | Same | OK |

### C12. Game Theory -- `analyze_competitive_dynamics()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `target_cache`, `competitor_caches` dict of linked entity caches, `target_name` | `competitor_caches` is NEVER passed from main.py (always None) | W8 |
| **Output** | `game_theory_result` with Cournot/Stackelberg analysis, `competitive_pressure`, `market_structure` | Always returns "monopoly" with 0 competitors | W8 |
| **Operation** | Cournot quantity game, Bertrand price game, Stackelberg leadership, CR4 market structure, competitive pressure index | Only executes the n=0 path (no competitors) | W8 |
| **Profile** | Stored via `_available_dict(game_theory_result)` | Stored but always shows monopoly | W8 |
| **Fix** | Pass `competitor_caches=linked_caches` in main.py line 1109 | | |

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
| **Operation** | **Phase 1:** Deterministic accounting identity fill (total_assets = total_liabilities + total_equity, etc., 5 iterations max). **Phase 2:** Classify NaN as MAR or MNAR. **Phase 3a (MAR):** MICE + Gaussian Process + Matrix Completion ensemble. **Phase 3b (MNAR):** Heckman Selection + Pattern-Mixture + GAIN ensemble. Observed values are NEVER overwritten. | Same -- all 3 phases implemented | OK |
| **Note** | PerformanceWarning during in-loop column insertion is now suppressed (P1 fix). DataFrame defragmented via `.copy()` before return. | FIXED | OK |

---

## Phase E: Enriched Survival Timeline

### E1. Early Regime Detection -- `run_early_regime_detection()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache with `return_1d` and `volatility_21d` | Same | OK |
| **Output** | cache + `regime_hmm`, `regime_gmm`, `regime_label`, `structural_break`, `breakpoint_method`, `regime_hmm_prob_*` columns; `early_regime_result` with `detector` object | Same | OK |
| **Operation** | 1. **HMM** (4-regime Gaussian on returns+volatility). 2. **GMM** (unsupervised clustering). 3. **PELT** (structural break detection). 4. **BCP** (Bayesian change point). Each wrapped in try/except with graceful fallback. | Same | OK |

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
| **Operation** | Pairwise Granger F-tests across all variable pairs. Prune variables with no causal links. Result used to prune `_extra_vars` list fed to temporal models. | Same | OK |
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
| **Operation** | FFT-based spectral analysis to identify dominant periodicities in the close price series | Same | OK |
| **Downstream** | Fed to `inject_cycle_phase_features()` (Synergy B) to add `cycle_phase_*` columns to cache | Same | OK |

### F6. Pattern Detector -- `detect_patterns()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache (needs `open`, `high`, `low`, `close`) | Same | OK |
| **Output** | `pattern_result` with detected candlestick patterns (doji, hammer, engulfing, etc.) | Same | OK |
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
| **Operation** | For each variable, try models in order: 1. Kalman (local-level state-space). 2. GARCH (conditional volatility). 3. VAR (multivariate, AR(1) fallback). 4. LSTM (PyTorch, GBM/LR fallback). 5. Tree ensemble (RF/GBM/XGB). 6. Baseline (last-value or EMA). First model that fits successfully wins. | Same | OK |

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
| **Operation** | 1. Walk day-by-day using 4 model types (baseline, EMA, linear trend, mean reversion). 2. Track per-model per-survival-mode error. 3. Retrain at switch points. 4. Build mode-conditioned leaderboard. | Was dead code -- now wired into pipeline | FIXED |

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
| **Operation** | 1. Estimate per-regime return distributions. 2. Build regime transition matrix. 3. Simulate 10,000 paths with regime switching. 4. Apply importance sampling for tail events. 5. Compute survival probability (fraction of paths not triggering survival thresholds). | Same | OK |

### F13. Copula -- `run_copula_analysis()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `variables` (auto-selected if None) | Same | OK |
| **Output** | `CopulaResult` with `copula_correlation`, `tail_dependence`, `joint_crisis_probability` | Same | OK |
| **Operation** | 1. Transform to uniform marginals (PIT). 2. Fit Gaussian copula (normal transform + correlation). 3. Estimate lower tail dependence for each pair. 4. Estimate joint crisis probability. | Same | OK |
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
| **Operation** | 1. Calibrator ingests residuals from model validation. 2. Computes conformal quantile at target coverage. 3. Applies adaptive adjustment for non-stationarity. 4. Builds intervals: forecast +/- conformal quantile. | Same -- now receives residuals from ForecastResult | FIXED |

### F17. DTW Analogs -- `find_historical_analogs()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache | Same | OK |
| **Output** | `DTWAnalogResult` with historical pattern matches and empirical forecasts | Same | OK |
| **Operation** | 1. Extract recent window of close prices. 2. DTW distance search over historical windows. 3. Rank analog matches. 4. Derive empirical forecast from what happened after each analog. | Same | OK |

### F18. Prediction Aggregator -- `run_prediction_aggregation()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `forecast_result`, `mc_result`, `conformal_result`, `dual_regime_result`, `copula_result`, `dtw_result`, `granger_result`, `shap_result`, `walk_forward_result` | `walk_forward_result` was receiving `ForwardPassResult` (wrong type) | FIXED |
| **Output** | `PredictionAggregatorResult` with `predictions` (per-var, per-horizon `HorizonPrediction`), `technical_alpha` mask, `metadata` | Same | OK |
| **Operation** | 1. Compute inverse-RMSE ensemble weights. 2. Apply regime blending (soft switching from dual_regime_result). 3. Aggregate point forecasts. 4. Build uncertainty bands (conformal preferred, RMSE fallback). 5. Widen bands using copula tail dependence. 6. Add DTW analog forecasts. 7. Propagate Granger causal adjustments. 8. Attach SHAP explanations. 9. Apply Technical Alpha mask (hide OHLC except Low). 10. Recency-weighted RMSE from walk-forward. | Step 10 was dead (always NaN) -- now active | FIXED |

### F19. SHAP Explainability -- `compute_shap_explanations()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `predictions` dict, `tree_models` dict (fitted model objects), `predict_fns` dict | `tree_models` and `predict_fns` are NEVER passed from main.py (both None -> empty dict) | W9 |
| **Output** | `SHAPResult` with per-variable feature importance, top drivers, narratives | Always returns `available=False`, empty `explanations` dict | W9 |
| **Operation** | 1. Try TreeExplainer with tree_models (skipped -- no models). 2. Fall back to KernelExplainer with predict_fns (skipped -- no functions). 3. Neither path runs. | Neither explainer path executes. | W9 |
| **Fix** | Pass fitted tree models from `forecast_result.model_states` or `forward_pass_result.model_states` to `tree_models` parameter | | |

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
| **Operation** | 1. Initialize population via Dirichlet distribution. 2. Evaluate fitness as -RMSE of weighted ensemble. 3. Tournament selection, crossover, mutation. 4. Elite carryover. 5. Converge after N generations. | Same | OK |

### F22. OHLC Predictor -- `predict_ohlc_series()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache, `forecast_result`, `mc_result`, `pattern_drift_multiplier` | Same | OK |
| **Output** | `OHLCResult` with next-day OHLC predictions | Same | OK |
| **Operation** | Combine forecast + Monte Carlo + pattern drift to predict next-day Open, High, Low, Close | Same | OK |

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

## Unwired Modules -- Investigation Results

### W6: `vanity.py` -- SHOULD BE WIRED (profile builder expects it)

**Severity:** Medium (wiring gap -- module is complete, tested, but never called)
**Location:** `operator1/analysis/vanity.py` (637 lines)
**Tests:** `tests/test_vanity_v2.py` (dedicated), referenced in `test_phase4_analysis.py`, `test_phase7_report.py`
**Problem:** The profile builder at `profile_builder.py:333` has `_build_vanity_section(cache)` that reads columns:
`vanity_score`, `vanity_label`, `vanity_trend`, `vanity_score_21d`,
`vanity_rnd_mismatch`, `vanity_sga_bloat_v2`, `vanity_capital_misallocation`,
`vanity_competitive_decay`, `vanity_sentiment_gap`.

But `compute_vanity_scores()` from `vanity.py` is NEVER called in `main.py` to populate these columns.
The vanity section in every profile will always show as unavailable.

**Fix:** Add a call to `compute_vanity_scores(cache)` in `main.py` Step 5 (feature engineering),
after financial health and before entity discovery. Requires: cache with derived variables,
financial health scores, sentiment scores (optional), peer ranking (optional).

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | cache with derived vars, fh_* scores, sentiment (optional), peer_ranking (optional) | Never called | NOT WIRED |
| **Output** | cache + `vanity_score`, `vanity_label`, `vanity_trend`, 5 component columns | Never computed | NOT WIRED |
| **Operation** | 5-component composite: R&D mismatch, SGA bloat, capital misallocation, competitive decay, sentiment gap | Never runs | NOT WIRED |

### W7: `supplement.py` -- SHOULD BE WIRED (fills profile gaps for non-US markets)

**Severity:** Medium (wiring gap -- enrichment available but never called)
**Location:** `operator1/clients/supplement.py` (549 lines)
**Tests:** `tests/test_supplement_and_translator.py`
**Problem:** For non-US markets (EU, Japan, Taiwan, Brazil, Chile), PIT filing APIs often return
profiles missing sector, industry, and market classification. `supplement.py` provides
`enrich_profile()` which fills these gaps via OpenFIGI (free, no key) and regional APIs.
This function is never called from `main.py` even though the pipeline would benefit from it
for every non-US company.

**Fix:** Add a call to `enrich_profile()` in `main.py` after Step 2 (profile fetch),
passing the market_id and ticker. This is a safe additive enrichment (never overwrites
existing data).

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `profile` dict, `ticker`, `market_id` | Never called | NOT WIRED |
| **Output** | Enriched profile with sector, industry, identifiers filled from OpenFIGI/regional APIs | Profiles stay incomplete for non-US markets | NOT WIRED |
| **Operation** | OpenFIGI lookup + per-region enrichers (Euronext, JPX, TWSE, B3, Santiago) | Never runs | NOT WIRED |

### Genuinely Unwired (Planned Features / Legacy)

| Module | Location | Lines | Tests | Status | Assessment |
|--------|----------|-------|-------|--------|------------|
| `portfolio_analysis.py` | `features/` | 283 | None | Not wired | **Planned feature.** Needs institutional holder data most PIT APIs don't provide. Awaiting data source. |
| `llm_filing_extractor.py` | `clients/` | 571 | Used in `live_helpers.py` only | Not wired | **Planned feature.** LLM-based PDF/HTML/iXBRL extraction for raw-filing markets (AU, HK, SG, SA, etc.). Awaiting integration. |
| `data_extraction.py` | `steps/` | 392 | Referenced by `cache_builder.py` for types | Not wired | **Legacy code.** Superseded by `main.py` inline extraction logic. Only used for `EntityData` type import by `cache_builder.py`. |
| `verify_identifiers.py` | `steps/` | 129 | None direct | Not wired | **Legacy code.** Superseded by `main.py` inline verification. Only imported by `data_extraction.py` for `VerifiedTarget` type. |

---

## Phase G: Profile Building

### G1. Profile Builder -- `build_company_profile()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `target_profile`, cache, `linked_aggregates`, all model results (regime, forecast, MC, prediction, estimation, graph_risk, game_theory, fuzzy, financial_health, sentiment, peer_ranking, macro_quadrant) | Same -- now also includes burnout_result and full transfer_entropy_result | FIXED |
| **Output** | `profile` dict with all sections: identity, financials, survival, models, extended_models, meta | Same | OK |
| **Operation** | Assemble all model outputs into a single JSON-serializable profile dict | Same | OK |

---

## Phase H: Report Generation

### H1. Report Generator -- `generate_all_reports()`

| | Expected | Actual | Status |
|---|----------|--------|--------|
| **Input** | `profile` dict, `gemini_client` (LLM), cache, `output_dir` | Same | OK |
| **Output** | Basic + Pro + Premium markdown reports, optional PDF | Same | OK |
| **Operation** | 1. Extract key metrics from profile. 2. Generate narrative via LLM (or template fallback). 3. Include charts, survival analysis, model diagnostics. 4. Three tiers with increasing detail. | Same | OK |

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
| W6 | vanity.py | Profile builder expects vanity columns but `compute_vanity_scores()` never called | **NOT YET FIXED** -- needs wiring in main.py Step 5 |
| W7 | supplement.py | Profile enrichment for non-US markets never called | **NOT YET FIXED** -- needs wiring in main.py after Step 2 |
| W8 | main.py:1109 | `analyze_competitive_dynamics()` never receives `linked_caches` -- always returns "monopoly" with 0 competitors | **NOT YET FIXED** -- pass `competitor_caches=linked_caches` |
| W9 | main.py:1758 | `compute_shap_explanations()` never receives `tree_models` or `predict_fns` -- explanations always empty | **NOT YET FIXED** -- pass fitted model objects from forecasting |
| P3 | derived_variables.py:662 | PerformanceWarning from 39 column-by-column insertions across compute stages | Added `result.copy()` defragmentation before return |
| S2 | genetic_optimizer.py:182 | GA uses EWM/shifted proxy predictions instead of actual per-model forecast arrays | **DESIGN LIMITATION** -- comment at line 183 acknowledges "in production, these would come from stored per-model predictions" |

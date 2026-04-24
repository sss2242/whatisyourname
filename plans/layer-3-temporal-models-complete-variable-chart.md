# Layer 3: Temporal Models -- Complete Variable Chart

Every variable produced by the 25 Layer 3 modules in `operator1/models/`. These are the statistical and ML models that consume the ~372-column enriched cache and produce forecasts, uncertainty bands, regime classifications, and ensemble predictions. Most outputs are result objects stored in the profile, not cache columns.

---

## 3.1 Regime Detector

**File:** `operator1/models/regime_detector.py` (1,103 lines)
**Pipeline step:** Step 5.5 / Step 6a

### Cache Variables

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `regime_hmm` | 4-regime Gaussian HMM on return_1d + volatility_21d (hmmlearn) | Integer (0-3) | Regime label mapping, HMM probabilities |
| 2 | `regime_hmm_prob_0` | HMM posterior probability for regime 0 | Continuous (0-1) | Forecasting regime shift, extra vars |
| 3 | `regime_hmm_prob_1` | HMM posterior probability for regime 1 | Continuous (0-1) | Forecasting regime shift |
| 4 | `regime_hmm_prob_2` | HMM posterior probability for regime 2 | Continuous (0-1) | Forecasting regime shift |
| 5 | `regime_hmm_prob_3` | HMM posterior probability for regime 3 | Continuous (0-1) | Forecasting regime shift |
| 6 | `regime_gmm` | GMM unsupervised clustering labels (sklearn GaussianMixture, BIC-selected) | Integer (0-3) | GMM secondary fallback for degenerate HMM |
| 7 | `regime_label` | Consensus: HMM primary -> GMM fallback -> PELT fallback -> vol quintile fallback. Values: bull/bear/high_vol/low_vol | Categorical | Monte Carlo regime distributions, forecasting, prediction aggregator, all temporal models |
| 8 | `structural_break` | 1 on days with PELT or BCP structural break, 0 otherwise | Binary (0/1) | PELT fallback labeling, profile |
| 9 | `breakpoint_method` | "pelt" / "bcp" / "pelt,bcp" | Categorical | Profile, report |
| 10 | `online_change_score` | ChangeFinder SDAR sequential score (no look-ahead) | Continuous | Extra vars |

**10 cache columns**

---

## 3.2 Regime Mixer

**File:** `operator1/models/regime_mixer.py` (389 lines)
**Pipeline step:** Step 6b

### Result Object (DualRegimeResult, not cache columns)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `market_regime_labels` | Per-day market regime from HMM: bull/bear/high_vol/low_vol | Prediction aggregator (regime blending) |
| 2 | `fund_regime_labels` | Per-day fundamental regime: healthy/stressed/distress | Prediction aggregator |
| 3 | `blended_weights` | Per-day soft transition weights between regimes (logistic sigmoid) | Prediction aggregator (step 2: regime blending) |

**0 cache columns, 3 result fields**

---

## 3.3 Granger Causality

**File:** `operator1/models/granger_causality.py` (519 lines)
**Pipeline step:** Step 6c

### Result Object (GrangerResult, not cache columns)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `causality_matrix` | NxN boolean matrix of significant Granger-causal relationships | Model synergies unified causal network |
| 2 | `significant_pairs` | List of (X, Y, lag, p_value) for significant pairs | Profile, report |
| 3 | `retained_variables` | Variables that have at least one causal link | `_extra_vars` list pruning |
| 4 | `pruned_variables` | Variables removed (no causal links) | Logging |
| 5 | `network_density` | Fraction of possible edges that are significant | Profile, model diagnostics |

### Time-Varying Granger (separate result)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 6 | `emerging_pairs` | Causal links that strengthened in recent windows | Profile |
| 7 | `disappearing_pairs` | Causal links that weakened/vanished | Profile |

**0 cache columns, 7 result fields**

---

## 3.4 Transfer Entropy

**File:** `operator1/models/causality.py` (417 lines)
**Pipeline step:** Step 6d

### Result Object (TransferEntropyResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `pairwise_scores` | Dict of (X, Y) -> TE value | Model synergies unified causal network |
| 2 | `top_pairs` | Top-K strongest information flow pairs | Profile |

**0 cache columns, 2 result fields**

---

## 3.5 Model Synergies

**File:** `operator1/models/model_synergies.py` (797 lines)
**Pipeline step:** Step 6g

### Cache Variables (injected synergy features)

| # | Variable | Formula | Consumers |
|---|----------|---------|-----------|
| 1 | `cycle_phase_*` | sin/cos of dominant cycle phases from cycle decomposition (per detected cycle) | Extra vars for forecasting |

### Result Outputs

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 2 | `_extra_vars` (updated) | Pruned feature list after causal analysis | Forecasting, forward pass |
| 3 | `_synergy_meta` | Metadata: which synergies were applied, pattern drift multiplier | Profile, OHLC predictor |
| 4 | `pattern_drift_multiplier` | Computed from candlestick pattern detection | OHLC predictor |

**~2-6 cache columns (cycle_phase_*), 3 result fields**

---

## 3.6 Forecasting

**File:** `operator1/models/forecasting.py` (4,588 lines)
**Pipeline step:** Step 6h

### Result Object (ForecastResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `forecasts` | `{var: {horizon: point_forecast}}` for each variable at 1d/5d/21d/252d | Prediction aggregator, OHLC predictor, conformal |
| 2 | `metrics` | `{var: ModelMetrics}` with RMSE, MAE, model name, test_residuals | Prediction aggregator weights, genetic optimizer |
| 3 | `model_used` | `{var: model_name}` -- which model won the cascade | Profile, model diagnostics |
| 4 | `residuals` | List of validation residuals for conformal calibration | Conformal calibrator |

**0 cache columns, 4 result fields**

---

## 3.7 Forward Pass

**File:** `operator1/models/forecasting.py` (within)
**Pipeline step:** Step 6i

### Result Object (ForwardPassResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `errors_by_tier` | Per-tier prediction error history | PID controller |
| 2 | `errors_by_regime` | Per-regime prediction errors | Burn-out, profile |
| 3 | `model_states` | Fitted model objects (tree, linear, etc.) | SHAP explainability (predict_fns) |
| 4 | `predictions_log` | Day-by-day prediction records | Walk-forward error aggregation, FixedShare |
| 5 | `total_days` | Number of days walked | Profile |
| 6 | `pid_summary` | PID controller Kp/Ki/Kd + error trajectory | Profile |
| 7 | `conformal_calibrator` | Fitted ConformalPIDCalibrator from forward pass residuals | Conformal prediction |

**0 cache columns, 7 result fields**

---

## 3.8 Walk-Forward Evaluation

**File:** `operator1/models/walk_forward.py` (700 lines)
**Pipeline step:** Step 6k

### Result Object (WalkForwardResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `day_errors` | Per-day per-model error records | Mode scores |
| 2 | `mode_scores` | Per-survival-mode per-model MAE | Mode-conditioned leaderboard |
| 3 | `best_model_by_mode` | Best model for each survival mode | Prediction aggregator |
| 4 | `retrain_dates` | Dates where models were retrained (switch points) | Profile |
| 5 | `overall_best_model` | Global best model across all modes | Profile |
| 6 | `overall_mae` | Global MAE | Profile, model diagnostics |
| 7 | `mode_confidence_sets` | MCS-filtered model subsets per mode (arch.bootstrap.MCS) | Prediction aggregator step 10 |

**0 cache columns, 7 result fields**

---

## 3.9 Burn-Out

**File:** `operator1/models/forecasting.py` (within)
**Pipeline step:** Step 6j

### Result Object (BurnoutResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `iterations_completed` | Number of burn-out iterations run | Profile |
| 2 | `converged` | Whether early stopping fired | Profile |
| 3 | `calibrated` | Whether regime weights were successfully calibrated | MC, prediction aggregator |
| 4 | `regime_weights` | Per-regime per-model weight dicts from exponential gradient | Prediction aggregator, MC |
| 5 | `regime_distributions` | Per-regime return distributions calibrated during burn-out | MC path generation |

**0 cache columns, 5 result fields**

---

## 3.10 Monte Carlo Simulation

**File:** `operator1/models/monte_carlo.py` (1,304 lines)
**Pipeline step:** Step 6l

### Result Object (MonteCarloResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `survival_probability` | `{horizon: {mean, p5, p95}}` at 90d, 252d | Profile, MF fusion, triage card, report |
| 2 | `regime_distributions` | Per-regime (mean, std, df) for Student-t returns | HF DCF growth assumptions, scenario engine |
| 3 | `transition_matrix` | Regime transition probability matrix from HMM | Regime shift predictor |
| 4 | `regime_order` | Order of regimes in transition matrix | Regime shift predictor |
| 5 | `terminal_values` | `{horizon: array of 10K terminal cumulative returns}` | Conformal intervals (MC P5/P95 override at 21d+) |
| 6 | `anticipated_survival` | Path-wise MC trigger checking at 63d and 252d | Profile |
| 7 | `segment_hhi` | Injected from cache for concentration risk | Profile |
| 8 | `concentration_risk_flag` | `segment_hhi > 0.5` | Profile |

**0 cache columns, 8 result fields**

---

## 3.11 PID Controller

**File:** `operator1/models/pid_controller.py` (323 lines)
**Pipeline step:** Called inside forward pass

### Internal State (not exposed)

| # | Variable | Formula | Consumers |
|---|----------|---------|-----------|
| 1 | `adjustment` | `Kp*error + Ki*integral + Kd*derivative` | Forward pass learning rate |
| 2 | `learning_rate_multiplier` | `clip(1.0 + adjustment, 0.1, 5.0)` | Forward pass |

**0 cache columns, 0 exposed result fields** (internal to forward pass)

---

## 3.12 Conformal Prediction

**File:** `operator1/models/conformal.py` (1,029 lines)
**Pipeline step:** Step 6p

### Result Object (ConformalResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `intervals` | `{var: {horizon: {lower, upper, forecast, coverage}}}` | Prediction aggregator (uncertainty bands) |
| 2 | `coverage_level` | Target coverage (e.g. 0.90) | Profile |
| 3 | `method` | "adaptive_conformal" or "split_conformal" | Profile |
| 4 | `calibration_scores_count` | Number of residuals used for calibration | Profile |

**0 cache columns, 4 result fields**

---

## 3.13 Copula Analysis

**File:** `operator1/models/copula.py` (417 lines)
**Pipeline step:** Step 6m

### Result Object (CopulaResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `copula_correlation` | Estimated correlation matrix | Profile |
| 2 | `tail_dependence` | Lower and upper tail dependence coefficients | Prediction aggregator (band widening), adaptive model params |
| 3 | `joint_crisis_probability` | P(2+ variables breach crisis thresholds simultaneously) | Prediction aggregator |
| 4 | `best_copula` | Gaussian / Student-t / Clayton (AIC selected) | Profile, model diagnostics |

**0 cache columns, 4 result fields**

---

## 3.14 Transformer Forecaster

**File:** `operator1/models/transformer_forecaster.py` (451 lines)
**Pipeline step:** Step 6n

### Result Object (TransformerResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `forecasts` | Per-variable 1-step-ahead predictions | Injected into ForecastResult |
| 2 | `feature_importance` | Attention-based feature importance | Profile |
| 3 | `train_loss_history` | Training loss per epoch | Model diagnostics |

**0 cache columns, 3 result fields**

---

## 3.15 Particle Filter

**File:** `operator1/models/particle_filter.py` (380 lines)
**Pipeline step:** Step 6o

### Result Object (ParticleFilterResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `filtered_states` | Per-variable posterior mean trajectories | Profile |
| 2 | `particles_final` | Particle positions at final time step | Profile |
| 3 | `weights_final` | Particle weights at final time step | Profile |
| 4 | `percentiles` | P5/P25/P50/P75/P95 bands per variable | Profile |

**0 cache columns, 4 result fields**

---

## 3.16 Cycle Decomposition

**File:** `operator1/models/cycle_decomposition.py` (336 lines)
**Pipeline step:** Step 6e

### Result Object (CycleResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `dominant_cycles` | List of (period_days, amplitude, phase_radians, power) sorted by amplitude | Model synergies (cycle phase injection), OHLC predictor, profile |
| 2 | `method` | "ceemdan" or "fft" | Profile |

**0 cache columns, 2 result fields**

---

## 3.17 Pattern Detector

**File:** `operator1/models/pattern_detector.py` (507 lines)
**Pipeline step:** Step 6f

### Result Object (PatternResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `patterns` | List of detected candlestick patterns with date, name, confidence | Model synergies (drift), profile |
| 2 | `motifs` | Recurring subsequences from Matrix Profile (stumpy) | Profile |
| 3 | `discords` | Anomalous subsequences from Matrix Profile | Profile |
| 4 | `predicted_patterns_week` | Patterns detected on predicted OHLC candles | Profile |

**0 cache columns, 4 result fields**

---

## 3.18 SHAP Explainability

**File:** `operator1/models/explainability.py` (509 lines)
**Pipeline step:** Step 6s

### Result Object (SHAPResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `feature_importance` | Per-variable top-5 SHAP values | Prediction aggregator (step 8: SHAP attachment) |
| 2 | `top_drivers` | Most important features per prediction | Profile, report |
| 3 | `narratives` | Natural language explanations per variable | Report section 19 |

**0 cache columns, 3 result fields**

---

## 3.19 Sobol Sensitivity

**File:** `operator1/models/sensitivity.py` (330 lines)
**Pipeline step:** Step 6t

### Result Object (SobolResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `first_order_indices` | S1 per feature -- variance from variable alone | Hierarchy feedback, profile |
| 2 | `total_order_indices` | ST per feature -- variance from variable + interactions | Profile |
| 3 | `adjusted_hierarchy` | Sobol-informed hierarchy weight adjustments | `adjust_hierarchy_from_sobol()` |

**0 cache columns, 3 result fields**

---

## 3.20 Prediction Aggregator

**File:** `operator1/models/prediction_aggregator.py` (2,253 lines)
**Pipeline step:** Step 6r

### Result Object (PredictionAggregatorResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `predictions` | `{var: {horizon: HorizonPrediction}}` where HorizonPrediction has: point_forecast, lower_ci, upper_ci, confidence, model_weights, shap_explanation | Profile, report, backtest validation |
| 2 | `technical_alpha` | Mask: which variables/horizons have predictions hidden (OHLC except Low) | Profile |
| 3 | `metadata` | Aggregation method used, n_models, regime weights applied | Profile |

**0 cache columns, 3 result fields**

---

## 3.21 Graph Risk

**File:** `operator1/models/graph_risk.py` (654 lines)
**Pipeline step:** Step 5e

### Result Object (GraphRiskResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `n_nodes` | Number of entities in the network | Profile |
| 2 | `target_degree_centrality` | Fraction of entities connected to target | Profile |
| 3 | `target_pagerank` | PageRank score | Profile |
| 4 | `covar_by_entity` | Conditional VaR per linked entity | Profile |
| 5 | `srisk` | Systemic risk capital shortfall | Profile |
| 6 | `contagion_target_infection_prob` | SIR infection probability | Profile |
| 7 | `supply_chain_hhi` | Supplier/customer concentration | Profile |

**0 cache columns, 7 result fields**

---

## 3.22 Game Theory

**File:** `operator1/models/game_theory.py` (460 lines)
**Pipeline step:** Step 5e

### Result Object (GameTheoryResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `market_structure` | monopoly / oligopoly / competitive (CR4) | Profile, report |
| 2 | `competitive_pressure` | 0-1 composite index | Profile, report |
| 3 | `cournot_equilibrium` | Nash equilibrium quantities | Profile |
| 4 | `stackelberg_result` | Leader/follower analysis | Profile |

**0 cache columns, 4 result fields**

---

## 3.23 Ownership Contagion

**File:** `operator1/models/ownership_contagion.py` (595 lines)
**Pipeline step:** Step 5f.2

### Cache Variables (injected via `inject_contagion_into_cache`)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `inst_mhhi_delta` | Modified HHI: anti-competitive common ownership effect | Extra vars |
| 2 | `inst_crowding_score` | Ownership crowding (same funds own same stocks) | Survival trigger (crowding + illiquidity) |
| 3 | `inst_liquidation_days` | Days to unwind position given Amihud illiquidity | Profile |

### Result Object (ContagionResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 4 | `mhhi_delta` | MHHI value | Profile |
| 5 | `crowding_score` | Crowding value | Cache injection |
| 6 | `liquidation_days` | Liquidation estimate | Cache injection |
| 7 | `n_shared_institutions` | Number of institutions holding both target and competitors | Profile |
| 8 | `bipartite_centrality` | Target centrality in company-institution network | Profile |

**3 cache columns, 5 result fields**

---

## 3.24 DTW Historical Analogs

**File:** `operator1/models/dtw_analogs.py` (453 lines)
**Pipeline step:** Step 6q

### Result Object (DTWAnalogResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `matches` | Top-K historical analog matches with DTW distance | Profile |
| 2 | `empirical_forecasts` | Median post-match trajectory as empirical forecast | Prediction aggregator (step 6: DTW overlay) |
| 3 | `cross_company_matches` | Peer analog matches (0.7x weighted) | Profile |

**0 cache columns, 3 result fields**

---

## 3.25 Genetic Optimizer

**File:** `operator1/models/genetic_optimizer.py` (504 lines)
**Pipeline step:** Step 6u

### Result Object (GAResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `best_weights` | Optimal ensemble weight vector | Profile |
| 2 | `tier_weights` | Per-tier optimized weights | Hierarchy adjustment |
| 3 | `per_regime_weights` | Separate weight vectors per survival regime | Profile |
| 4 | `fitness_history` | RMSE per generation | Profile |
| 5 | `converged` | Whether convergence criteria met | Profile |

**0 cache columns, 5 result fields**

---

## 3.26 OHLC Predictor

**File:** `operator1/models/ohlc_predictor.py` (379 lines)
**Pipeline step:** Step 6v

### Result Object (OHLCResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `next_day` | OHLCCandle: open, high, low, close for next trading day | Profile, pattern prediction |
| 2 | `next_week` | 5 OHLCCandle objects | Profile |
| 3 | `next_month` | 21 OHLCCandle objects | Profile |
| 4 | `next_year` | 252 OHLCCandle objects | Profile |

**0 cache columns, 4 result fields**

---

## 3.27 Regime Shift Predictor

**File:** `operator1/models/regime_shift_predictor.py` (351 lines)
**Pipeline step:** Step 6 (after MC)

### Result Object (RegimeShiftResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `prob_exit_21d` | P(leaving current regime within 21 days) | Profile, report |
| 2 | `prob_exit_252d` | P(leaving current regime within 252 days) | Profile |
| 3 | `expected_days_to_shift` | Expected days until regime change | Profile |
| 4 | `most_probable_next_regime` | Which regime is most likely next | Profile, report |
| 5 | `current_regime` | Current regime for reference | Profile |

**0 cache columns, 5 result fields**

---

## 3.28 Retroactive Calibration

**File:** `operator1/analysis/retroactive_calibration.py` (552 lines)
**Pipeline step:** Step 6.5

### Result Object (RetroCalibrationResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `group_weights` | Calibrated linked entity group weights | Subsequent runs |
| 2 | `retrain_triggers` | Calibrated walk-forward retrain frequency | Subsequent runs |
| 3 | `blend_ratios` | Calibrated forecast ensemble blend | Subsequent runs |
| 4 | `n_calibrated` | Number of parameter groups calibrated | Profile |

**0 cache columns, 4 result fields**

---

## 3.29 Feature Selector

**File:** `operator1/models/feature_selector.py` (~500 lines)
**Pipeline step:** Step 6c (after Granger)

### Result Object (FeatureSelectionResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `retained_features` | Features passing Boruta + PIMP + mRMR | `_extra_vars` list |
| 2 | `dropped_features` | Features rejected | Logging |
| 3 | `importance_scores` | Per-feature importance from ensemble | Profile |

**0 cache columns, 3 result fields**

---

## 3.30 Model Diagnostics

**File:** `operator1/monitoring/model_diagnostics.py` (875 lines)
**Pipeline step:** Step 6.6

### Result Object (ModelDiagnosticsResult)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 1 | `n_models_assessed` | How many models were evaluated | Profile |
| 2 | `n_models_on_track` | How many are behaving as expected | Profile, report |
| 3 | `overall_robustness` | Fraction of on_track models | Profile |
| 4 | `per_model_assessments` | Per-model: expected behavior, actual result, rating (on_track/degraded/failed) | Profile, report section 19.97 |

**0 cache columns, 4 result fields**

---

## Layer 3 Grand Total

| Module | Cache Columns | Result Fields | Total |
|--------|--------------|---------------|-------|
| 3.1 Regime Detector | **10** | 0 | 10 |
| 3.2 Regime Mixer | 0 | 3 | 0 |
| 3.3 Granger Causality | 0 | 7 | 0 |
| 3.4 Transfer Entropy | 0 | 2 | 0 |
| 3.5 Model Synergies | **~4** | 3 | ~4 |
| 3.6 Forecasting | 0 | 4 | 0 |
| 3.7 Forward Pass | 0 | 7 | 0 |
| 3.8 Walk-Forward | 0 | 7 | 0 |
| 3.9 Burn-Out | 0 | 5 | 0 |
| 3.10 Monte Carlo | 0 | 8 | 0 |
| 3.11 PID Controller | 0 | 0 (internal) | 0 |
| 3.12 Conformal Prediction | 0 | 4 | 0 |
| 3.13 Copula | 0 | 4 | 0 |
| 3.14 Transformer | 0 | 3 | 0 |
| 3.15 Particle Filter | 0 | 4 | 0 |
| 3.16 Cycle Decomposition | 0 | 2 | 0 |
| 3.17 Pattern Detector | 0 | 4 | 0 |
| 3.18 SHAP | 0 | 3 | 0 |
| 3.19 Sobol Sensitivity | 0 | 3 | 0 |
| 3.20 Prediction Aggregator | 0 | 3 | 0 |
| 3.21 Graph Risk | 0 | 7 | 0 |
| 3.22 Game Theory | 0 | 4 | 0 |
| 3.23 Ownership Contagion | **3** | 5 | 3 |
| 3.24 DTW Analogs | 0 | 3 | 0 |
| 3.25 Genetic Optimizer | 0 | 5 | 0 |
| 3.26 OHLC Predictor | 0 | 4 | 0 |
| 3.27 Regime Shift Predictor | 0 | 5 | 0 |
| 3.28 Retroactive Calibration | 0 | 4 | 0 |
| 3.29 Feature Selector | 0 | 3 | 0 |
| 3.30 Model Diagnostics | 0 | 4 | 0 |
| **Total** | **~17** | **~121** | **~17 cache + 121 result** |

The temporal models primarily produce result objects (~121 fields) stored in the profile, not cache columns (~17). This is by design -- they consume the enriched cache and produce predictions/analyses that flow to the profile builder and report generator.

---

## Running Total Across All 3 Layers

| Layer | Cache Columns | Result Fields | Purpose |
|-------|--------------|---------------|---------|
| Layer 1: Features | ~311 | ~9 | Raw data -> enriched features |
| Layer 2: Analysis | ~61 | ~71 | Survival flags, regime, calibration |
| Layer 3: Temporal | ~17 | ~121 | Forecasting, uncertainty, ensemble |
| **Total** | **~389** | **~201** | **~389 cache columns + ~201 result fields** |

All ~389 cache columns + ~201 result fields flow into the Profile Builder (Step 7), which assembles them into `company_profile.json` for report generation.

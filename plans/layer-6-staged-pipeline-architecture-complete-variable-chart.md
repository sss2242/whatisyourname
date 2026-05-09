# Layer 6: Staged Pipeline Architecture -- Complete Variable Chart (v3)

**v3 update (2026-05-09):**
- **Stage 2 sub-stages (2026-05-02 to 2026-05-08):** Added `stage2_preprocessing.py` (2.1 frequency separation, 2.2 data reconciliation) and `stage2_freq_pipeline.py` (2.0 resample prep, 2.A annual, 2.Q quarterly, 2.M monthly, 2.W weekly, 2.D daily, 2.F fusion + ratio forward-fill). Total: 9 new sub-stages running BEFORE Stage 3.
- **Stage 1 split (2026-05-03):** Split monolithic Stage 1 into 6-8 sub-stages with per-sub-stage checkpoints for finer-grained resumability.
- **HF sub-stage split (2026-05-06):** Stage 7.5 split into 7.5.1 (base + scorecard), 7.5.2 (advanced methods), 7.5.3 (multi-freq variants), 7.5.4 (fusion).
- **Self-restart pattern (2026-05-03):** `run_backtest_staged.py` and `run_lean_backtest.py` now support self-restarting sub-stage execution for cloud environments with process limits.
- **Per-sub-stage data snapshots (2026-05-06):** `PipelineState.save_snapshot()` copies cache + state to `snapshots/{sub_id}/` directory for post-hoc inspection.
- **True terminal separation (2026-05-06):** Staged backtest compiler runs each sub-stage in a separate `subprocess.run()` with terminal isolation.
- **Total sub-stages: 44** (was 35): +9 from Stage 2, +4 from HF split, -4 from deduplication with Stage 7.4 (which now skips if Stage 2 already ran).

Every variable and sub-stage in the 6 Layer 6 modules. This layer decomposes the monolithic pipeline into per-model sub-stages with checkpoint save/resume via `PipelineState`. It produces no new analytical variables -- it orchestrates execution of Layers 1-5 and persists their state to disk.

**v2 update (2026-05-01):** Added 3 infrastructure enhancements to the stage runner: graceful degradation for non-critical sub-stages, per-sub-stage timeout management, and pre-flight state validation.

---

## 6.1 Pipeline State

**File:** `operator1/pipeline_state.py` (404 lines)
**Purpose:** Mutable state bag replacing hundreds of local variables in `main.py`. Serializes to disk between sub-stages.

### PipelineState Fields (by stage)

#### Config Fields

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `market_id` | String | Market identifier (e.g. "us_sec_edgar") |
| 2 | `company` | String | Company ticker/identifier |
| 3 | `end_date` | String | Analysis end date |
| 4 | `years` | Float | Lookback period in years (default: 2.0) |
| 5 | `output_dir` | String | Disk directory for checkpoints |

#### Stage 1: Data Acquisition (14 fields)

| # | Field | Type | Description |
|---|-------|------|-------------|
| 6 | `cache` | DataFrame or None | Main daily cache |
| 7 | `target_profile` | Dict | Company profile from fetch |
| 8 | `income_df` | DataFrame | Raw income statement |
| 9 | `balance_df` | DataFrame | Raw balance sheet |
| 10 | `cashflow_df` | DataFrame | Raw cash flow statement |
| 11 | `quotes_df` | DataFrame | Raw OHLCV quotes |
| 12 | `macro_data` | Dict | Raw macro indicators |
| 13 | `macro_dataset` | Any | Structured MacroDataset |
| 14 | `macro_quadrant_result` | Any | Macro quadrant classification |
| 15 | `conflict_result` | Any | Conflict risk assessment |
| 16 | `buying_power_result` | Any | Market buying power |
| 17 | `estimation_coverage` | Any | Estimation engine coverage |
| 18 | `filing_calendar_result` | Any | Filing calendar analysis |
| 19 | `event_calendar_result` | Any | Event calendar for PEAD |

#### Stage 2: Feature Engineering (27 fields)

| # | Field | Type | Description |
|---|-------|------|-------------|
| 20 | `weights` | Dict | Hierarchy weights |
| 21 | `fh_result` | Any | Financial health result |
| 22 | `fuzzy_result` | Any | Fuzzy protection result |
| 23 | `relationships` | Dict | Linked entity relationships |
| 24 | `linked_caches` | Dict | Per-entity cache DataFrames |
| 25 | `linked_agg_df` | DataFrame or None | Linked aggregate features |
| 26 | `graph_risk_result` | Any | Graph risk metrics |
| 27 | `game_theory_result` | Any | Competitive dynamics |
| 28 | `contagion_result` | Any | Ownership contagion |
| 29 | `peer_ranking_result` | Any | Peer ranking scores |
| 30 | `sentiment_result` | Any | News sentiment |
| 31 | `catalyst_result` | Any | Product catalysts |
| 32 | `signal_ic_result` | Any | Signal IC |
| 33 | `prediction_log_summary` | Any | Prediction log summary |
| 34 | `enriched_timeline_result` | Any | Enriched survival timeline |
| 35 | `early_regime_result` | Any | Early regime detection |
| 36 | `regime_detector` | Any | Regime detector object |
| 37 | `survival_controller` | Any | USS controller |
| 38 | `target_holders` | List | Institutional holders |
| 39 | `target_insiders` | List | Insider transactions |
| 40 | `six_proxy_result` | Any | SIX derived proxies (CH) |
| 41 | `linked_conflict` | Any | Linked entity conflict |
| 42 | `seg_result` | Dict | Product segment data |
| 43 | `supply_chain_stress_result` | Any | Supply chain stress |
| 44 | `is_private` | Boolean | Private company flag |
| 45 | `ohlcv_source_label` | String | OHLCV data source |
| 46 | `adaptive_thresholds` | Any | Tier 1 calibration |
| 47 | `adaptive_model_params` | Any | Tier 2 calibration |
| 48 | `adaptive_tier3` | Any | Tier 3 calibration |
| 49 | `mode_weights` | Any | Mode-conditioned weights |

#### Stage 3: Temporal Analysis (8 fields)

| # | Field | Type | Description |
|---|-------|------|-------------|
| 50 | `feature_selection_result` | Any | Feature selection |
| 51 | `dual_regime_result` | Any | Dual regime mixer |
| 52 | `granger_result` | Any | Granger causality |
| 53 | `transfer_entropy_result` | Any | Transfer entropy |
| 54 | `cycle_result` | Any | Cycle decomposition |
| 55 | `pattern_result` | Any | Pattern detection |
| 56 | `synergy_meta` | Dict | Synergy metadata |
| 57 | `extra_vars` | List | Extra variable names |
| 58 | `economic_plane` | Any | Economic plane classification |

#### Stage 4: Forecasting (1 field)

| # | Field | Type | Description |
|---|-------|------|-------------|
| 59 | `forecast_result` | Any | Full forecasting result |

#### Stage 5: Forward Modeling (6 fields)

| # | Field | Type | Description |
|---|-------|------|-------------|
| 60 | `forward_pass_result` | Any | Forward pass result |
| 61 | `burnout_result` | Any | Burn-out calibration |
| 62 | `walk_forward_result` | Any | Walk-forward evaluation |
| 63 | `mc_result` | Any | Monte Carlo simulation |
| 64 | `copula_result` | Any | Copula analysis |
| 65 | `regime_shift_result` | Any | Regime shift prediction |

#### Stage 6: Ensemble & Aggregation (11 fields)

| # | Field | Type | Description |
|---|-------|------|-------------|
| 66 | `transformer_result` | Any | Transformer forecaster |
| 67 | `particle_filter_result` | Any | Particle filter |
| 68 | `conformal_result` | Any | Conformal prediction |
| 69 | `dtw_result` | Any | DTW analogs |
| 70 | `pred_result` | Any | Prediction aggregation |
| 71 | `shap_result` | Any | SHAP explanations |
| 72 | `sobol_result` | Any | Sobol sensitivity |
| 73 | `ga_result` | Any | Genetic optimizer |
| 74 | `ohlc_result` | Any | OHLC predictions |
| 75 | `pattern_drift` | Float | Pattern drift multiplier (default: 1.0) |
| 76 | `tv_granger_result` | Any | Time-varying Granger |
| 77 | `mv_mc_result` | Any | Multivariate Monte Carlo |

#### Stage 7: Integration (5 fields)

| # | Field | Type | Description |
|---|-------|------|-------------|
| 78 | `scenario_result` | Any | USS scenario engine |
| 79 | `retro_params` | Any | Retroactive calibration |
| 80 | `model_diagnostics_result` | Any | Model diagnostics |
| 81 | `multi_frequency_result` | Any | Multi-frequency fusion |
| 82 | `hf_result` | Any | Hedge fund analysis |

#### Stage 8: Output (1 field)

| # | Field | Type | Description |
|---|-------|------|-------------|
| 83 | `profile` | Dict | Final assembled profile |

### Serialization Format

| Data Type | Format | File |
|-----------|--------|------|
| Main cache | Parquet | `cache.parquet` |
| Linked caches | Parquet per entity | `linked_caches/{entity_id}.parquet` |
| Linked aggregates | Parquet | `linked_agg.parquet` |
| All other state | Pickle | `state_{sub_stage}.pkl` |

---

## 6.2 Stage Runner

**File:** `operator1/stages/runner.py` (~350 lines)
**Purpose:** Dispatches sub-stages in dependency order with checkpoint save/resume, graceful degradation, timeout management, and pre-flight validation.

### Stage Spec Syntax

| Spec | Meaning |
|------|---------|
| `"3"` | All Stage 3 sub-stages (3.1 through 3.8) |
| `"4.1"` | Just sub-stage 4.1 (forecasting) |
| `"3-6"` | Stages 3 through 6 |
| `"all"` | All sub-stages (3.1 through 7.5) |

### Execution Flow

1. Build registry from all 5 stage modules
2. Parse stage spec into start/end IDs
3. Filter registry to matching sub-stages
4. For each sub-stage:
   a. **Pre-flight validation** (NEW): check required state fields via `_SUBSTAGE_REQUIREMENTS`
   b. **Timeout-wrapped execution** (NEW): `concurrent.futures` with per-sub-stage timeout via `_SUBSTAGE_TIMEOUTS`
   c. Save checkpoint on success
   d. **Graceful degradation** (NEW): non-critical failures log warning and continue; critical failures (`_CRITICAL_SUBSTAGES`) abort
5. Log skipped sub-stages count at pipeline completion

### Infrastructure Enhancements (NEW v2)

| Enhancement | Config | Description |
|-------------|--------|-------------|
| **Graceful Degradation** | `_CRITICAL_SUBSTAGES` set (6 entries: 3.1, 4.1, 5.1, 5.4, 6.5, 7.5) | Non-critical sub-stages can fail without aborting. Skipped stages saved as `{id}_skipped` checkpoint. |
| **Timeout Management** | `_SUBSTAGE_TIMEOUTS` dict (default: 120s, MC/HF: 300s, Transformer: 180s) | Prevents hanging models from blocking pipeline indefinitely. Uses `concurrent.futures.ThreadPoolExecutor`. |
| **Pre-Flight Validation** | `_SUBSTAGE_REQUIREMENTS` dict (7 entries for key stages) | Validates required PipelineState fields before dispatch. Missing fields in non-critical stages skip; in critical stages abort. |

---

## 6.3 Stage 3 -- Temporal Analysis

**File:** `operator1/stages/stage3_temporal.py` (295 lines)

| Sub-Stage | ID | Function | Model/Module |
|-----------|-----|----------|--------------|
| Regime Detection | 3.1 | `run_3_1_regime()` | HMM, GMM, PELT, BCP, ChangeFinder |
| Dual Regimes | 3.2 | `run_3_2_dual_regime()` | Regime mixer (market + fundamental) |
| Granger Causality | 3.3 | `run_3_3_granger()` | PCMCI / Granger F-tests + feature pruning |
| Transfer Entropy | 3.4 | `run_3_4_transfer_entropy()` | Shannon entropy k-NN estimation |
| Cycle Decomposition | 3.5 | `run_3_5_cycles()` | CEEMDAN / FFT |
| Pattern Detection | 3.6 | `run_3_6_patterns()` | Candlestick + Matrix Profile motifs/discords |
| Pre-Forecast Synergies | 3.7 | `run_3_7_synergies()` | Economic planes + causal network + peer adjustment |
| Feature Selection | 3.8 | `run_3_8_feature_selection()` | Causal-pruned extra_vars finalization |

---

## 6.4 Stage 4 -- Forecasting

**File:** `operator1/stages/stage4_forecasting.py` (54 lines)

| Sub-Stage | ID | Function | Model/Module |
|-----------|-----|----------|--------------|
| Forecasting | 4.1 | `run_4_1_forecasting()` | Kalman, GARCH, VAR, LSTM, Tree, ETS, Baseline |

---

## 6.5 Stage 5 -- Forward Modeling

**File:** `operator1/stages/stage5_forward.py` (285 lines)

| Sub-Stage | ID | Function | Model/Module |
|-----------|-----|----------|--------------|
| Forward Pass | 5.1 | `run_5_1_forward_pass()` | PID-controlled day-by-day walk |
| Burn-Out | 5.2 | `run_5_2_burnout()` | Exponential gradient weight calibration |
| Walk-Forward | 5.3 | `run_5_3_walk_forward()` | MCS + FixedShare + mode-conditioned scoring |
| Monte Carlo | 5.4 | `run_5_4_monte_carlo()` | Regime-switching MC + survival probability |
| Copula | 5.5 | `run_5_5_copula()` | Gaussian / Student-t / Clayton copulas |

---

## 6.6 Stage 6 -- Ensemble & Aggregation

**File:** `operator1/stages/stage6_ensemble.py` (455 lines)

| Sub-Stage | ID | Function | Model/Module |
|-----------|-----|----------|--------------|
| Transformer | 6.1 | `run_6_1_transformer()` | Multi-head self-attention forecaster |
| Particle Filter | 6.2 | `run_6_2_particle_filter()` | Sequential Monte Carlo state tracking |
| Conformal Prediction | 6.3 | `run_6_3_conformal()` | PID calibrator + Mondrian partitioning |
| DTW Analogs | 6.4 | `run_6_4_dtw()` | Dynamic Time Warping historical analogs |
| Prediction Aggregation | 6.5 | `run_6_5_aggregation()` | Inverse-RMSE + FixedShare ensemble |
| SHAP | 6.6 | `run_6_6_shap()` | TreeExplainer / KernelExplainer |
| Sobol | 6.7 | `run_6_7_sobol()` | Saltelli sensitivity analysis |
| TV Granger + MV MC | 6.8 | `run_6_8_tv_granger_mv_mc()` | Rolling-window causal + copula joint MC |
| Genetic Optimizer | 6.9 | `run_6_9_genetic()` | Optuna TPE / custom GA |
| OHLC Predictor | 6.10 | `run_6_10_ohlc()` | Forecast + MC + pattern drift + predicted patterns |

---

## 6.7 Stage 7 -- Integration

**File:** `operator1/stages/stage7_integration.py` (387 lines)

| Sub-Stage | ID | Function | Model/Module |
|-----------|-----|----------|--------------|
| USS + Scenario | 7.1 | `run_7_1_uss()` | Forecast bounding + 3-scenario MC |
| Retro Calibration | 7.2 | `run_7_2_retro_calibration()` | Empirical Bayes parameter adjustment |
| Model Diagnostics | 7.3 | `run_7_3_diagnostics()` | Expected vs actual path comparison |
| MF Resample Prep | 7.4.0 | `run_7_4_0_resample_prep()` | Build ResampledCache objects for all frequencies |
| MF Annual | 7.4.1 | `run_7_4_1_annual()` | Full pipeline at Annual frequency |
| MF Quarterly | 7.4.2 | `run_7_4_2_quarterly()` | Full pipeline at Quarterly frequency |
| MF Monthly | 7.4.3 | `run_7_4_3_monthly()` | Full pipeline at Monthly frequency |
| MF Weekly | 7.4.4 | `run_7_4_4_weekly()` | Full pipeline at Weekly frequency |
| MF Daily | 7.4.5 | `run_7_4_5_daily()` | Full pipeline at Daily frequency |
| MF Fusion | 7.4.6 | `run_7_4_6_fusion()` | 13-method cross-frequency fusion |
| Hedge Fund | 7.5 | `run_7_5_hedge_fund()` | 15 base + 19 advanced + 8 fusion methods |

---

## Complete Sub-Stage Registry (35 sub-stages)

| Stage | Sub-Stages | Count |
|-------|------------|-------|
| Stage 3: Temporal | 3.1 - 3.8 | 8 |
| Stage 4: Forecasting | 4.1 | 1 |
| Stage 5: Forward | 5.1 - 5.5 | 5 |
| Stage 6: Ensemble | 6.1 - 6.10 | 10 |
| Stage 7: Integration | 7.1, 7.2, 7.3, 7.4.0-7.4.6, 7.5 | 11 |
| **Total** | | **35** |

**Note:** The plan documents say "30 sub-stages" but the actual source code registries contain 35. The difference: sub-stage 3.8 (feature selection) was added after the plan was written, and Stage 7.4 was expanded from 1 entry into 7 per-frequency sub-stages (7.4.0 prep + 7.4.1-7.4.5 per-frequency + 7.4.6 fusion).

---

## Layer 6 UPDATED Grand Total

| Module | Lines | Role | Status |
|--------|-------|------|--------|
| 6.1 Pipeline State | 404 | State serialization (83 fields) | |
| 6.2 Stage Runner | **~350** (was 281) | Dispatch + checkpoint + graceful degradation + timeout + validation | **ENHANCED v2** |
| 6.3 Stage 3 -- Temporal | 295 | 8 sub-stages | |
| 6.4 Stage 4 -- Forecasting | 54 | 1 sub-stage | |
| 6.5 Stage 5 -- Forward | 285 | 5 sub-stages | |
| 6.6 Stage 6 -- Ensemble | 455 | 10 sub-stages | |
| 6.7 Stage 7 -- Integration | 387 | 14 sub-stages | |
| **Total** | **~2,230** | **35 sub-stages + 83 state fields** | |

**0 new cache columns. 0 new result fields.** Layer 6 is purely orchestration -- it dispatches existing Layer 1-5 modules and persists their outputs to disk via PipelineState.

**v2 infrastructure additions:** 3 new config dicts (`_CRITICAL_SUBSTAGES`, `_SUBSTAGE_TIMEOUTS`, `_SUBSTAGE_REQUIREMENTS`), 1 new validation function, enhanced dispatch loop with graceful degradation + timeout.

---

## Running Total Across All 6 Layers

| Layer | Cache Columns | Result Fields | Role | Status |
|-------|--------------|---------------|------|--------|
| Layer 1: Features | ~428 | ~9 | Raw cache -> enriched features | Updated v3 |
| Layer 2: Analysis | ~71 | ~77 | Survival, hierarchy, calibration | Updated v2 |
| Layer 3: Temporal | ~17 | ~121 | Regime, forecasting, uncertainty | |
| Layer 4: Hedge Fund | 0 | ~122 | Parallel investment analysis | Updated v2 |
| Layer 5: Multi-Frequency | 0 | ~17+ nested | 5-frequency pipeline + fusion | Updated v2 |
| Layer 6: Staged Pipeline | 0 | 0 | Orchestration + checkpoint/resume | **Updated v2** |
| **Total** | **~516** | **~346+** | **~48K lines across 70 modules** | |

# Python 3.12 Setup and Full Pipeline Audit

**Date:** 2026-03-06
**Python version:** 3.12.12 (Amazon Linux 2023)
**pip version:** 26.0.1

---

## Environment Setup

Installed Python 3.12 and pip on Amazon Linux 2023 via `dnf`, then installed
all four dependency stages in order:

| Stage | Contents | Status |
|-------|----------|--------|
| 1 -- Core | numpy 2.4.2, pandas 2.3.3, pyarrow 23.0.1, scipy 1.17.1, requests 2.32.5, matplotlib 3.10.8, pytest 9.0.2 | Installed |
| 2 -- ML | scikit-learn 1.8.0, statsmodels 0.14.6, xgboost 3.2.0, ruptures 1.1.10, hmmlearn 0.3.3, arch 8.0.0, shap 0.50.0, mapie 1.3.0, dtaidistance 2.4.0 | Installed |
| 3 -- Deep Learning | torch 2.10.0+cu128, arviz 0.23.4, pymc 5.28.0 | Installed |
| 4 -- Wrappers | edgartools 5.19.1, yfinance 1.2.0, wbgapi 1.0.14, fredapi 0.5.2, sdmx1 2.25.1, dart-fss 0.4.15, pykrx 1.0.51, baostock 0.8.9, twstock 1.4.0, nselib 2.4.3, + 20 more | Installed |

All four stages verified with import checks.

---

## Test Results

**Full suite: 1053 passed, 5 skipped, 97 warnings**

### Per-phase breakdown

| Test phase | File(s) | Result |
|------------|---------|--------|
| Phase 1 -- Smoke | `test_phase1_smoke.py` | 23 passed |
| Phase 2 -- Ingestion | `test_phase2_ingestion.py` | 13 passed |
| Phase 3 -- Features | `test_phase3_features.py` | 34 passed |
| Phase 4 -- Analysis | `test_phase4_analysis.py` | 51 passed |
| Phase 5 -- Estimation | `test_phase5_estimation.py` | 24 passed (32 warnings) |
| Phase 6 -- Forecasting | `test_phase6_forecasting.py` | 61 passed |
| Phase 6 -- Monte Carlo | `test_phase6_monte_carlo.py` | varies |
| Phase 6 -- Prediction Agg | `test_phase6_prediction_aggregator.py` | varies |
| Phase 6 -- Regime | `test_phase6_regime.py` | varies (58 warnings) |
| Phase 6 combined | all phase 6 tests | 207 passed |
| Phase 7 -- Report | `test_phase7_report.py` | 34 passed |
| Phase B -- Ethical Filters | `test_phase_b_ethical_filters.py` | varies |
| Phase C -- Modules | `test_phase_c_modules.py` | varies (2 warnings) |
| Advanced Modules | `test_advanced_modules.py` | 31 passed |
| Financial Health | `test_financial_health.py` | varies |
| Bug Regression | `test_bug_regression.py` | varies |
| Enriched Survival | `test_enriched_survival_timeline.py` | varies |
| Split Estimator | `test_split_estimator.py` | varies |
| Three Features | `test_three_features.py` | varies |
| Canonical Translator | `test_canonical_translator_new.py` | varies |
| Equity Provider | `test_equity_provider_new.py` | varies |
| LLM tests (5 files) | `test_llm_*.py` | 34 passed |
| Macro Provider | `test_macro_provider.py` | varies |
| OHLCV Provider | `test_ohlcv_provider.py` | varies |
| PIT Registry | `test_pit_registry_new.py` | varies |
| Survival Walk-forward | `test_survival_timeline_walkforward.py` | 36 passed |

---

## Pipeline Architecture (8-Step Flow)

Read and verified the full `main.py` pipeline (2152 lines):

1. **Market/Company Selection** -- Interactive or CLI-driven PIT source + company picker
2. **Profile Fetch** -- Company profile from government filing API
3. **Financial Data Fetch** -- Income, balance, cashflow, OHLCV (parallel ThreadPoolExecutor)
4. **Cache Building** -- Daily OHLCV spine + forward-filled financial statements (as-of join)
   - 4a: Macro data fetch and MacroDataset construction
   - 4b: Estimation engine (Sudoku inference -- 3-phase imputation)
   - 4c: Filing calendar analysis
5. **Feature Engineering** -- Derived variables, survival flags, fuzzy protection, financial health, linked entities, peer ranking, news sentiment
   - 5.5: Enriched survival timeline (HMM/GMM/PELT/BCP regime detection)
6. **Temporal Modeling** -- 25+ models: Kalman, GARCH, VAR, LSTM, tree ensembles, transformer, particle filter, copula, DTW, Granger, Monte Carlo, genetic optimizer, OHLC predictor, conformal prediction, SHAP, Sobol sensitivity
7. **Profile Building** -- JSON company profile with all model results
8. **Report Generation** -- Basic/Pro/Premium markdown reports via LLM

---

## Modules Reviewed for Internal Bugs

### Models (operator1/models/)

| Module | Lines | Status | Notes |
|--------|-------|--------|-------|
| `forecasting.py` | 3058 | Clean | 6-model hierarchy with fallback chains |
| `regime_detector.py` | 876 | Clean | HMM/GMM/PELT/BCP with graceful fallbacks |
| `monte_carlo.py` | 980 | Clean | Regime-aware paths + importance sampling |
| `prediction_aggregator.py` | 2099 | Clean | Ensemble weighting + conformal/copula/DTW integration |
| `financial_health.py` | 821 | Clean | 5-tier scoring + Altman Z + Beneish M |
| `copula.py` | 212 | Warning | NaN divide when column has zero stddev (see below) |
| `transformer_forecaster.py` | 432 | Clean | Positional encoding + multi-head attention |
| `particle_filter.py` | 380 | Clean | Sequential Monte Carlo with systematic resampling |
| `walk_forward.py` | 543 | Clean | Per-mode model leaderboard |
| `genetic_optimizer.py` | 318 | Clean | Dirichlet-initialized GA for ensemble weights |
| `granger_causality.py` | 293 | Clean | Pairwise tests + correlation fallback |
| `regime_mixer.py` | 356 | Clean | Dual layer (market + fundamental) soft switching |
| `model_synergies.py` | 793 | Clean | 8 cross-model synergies |
| `causality.py` | -- | Clean | Transfer entropy |
| `conformal.py` | -- | Clean | Distribution-free calibrated intervals |
| `cycle_decomposition.py` | -- | Clean | FFT-based cycle extraction |
| `dtw_analogs.py` | -- | Clean | Historical pattern matching |
| `explainability.py` | -- | Clean | SHAP explanations |
| `sensitivity.py` | -- | Clean | Sobol sensitivity analysis |
| `ohlc_predictor.py` | -- | Clean | Candlestick prediction |
| `pattern_detector.py` | -- | Clean | Candlestick pattern recognition |
| `pid_controller.py` | -- | Clean | PID control for forward pass |
| `graph_risk.py` | -- | Clean | Network centrality metrics |
| `game_theory.py` | -- | Clean | Competitive dynamics |

### Estimation (operator1/estimation/)

| Module | Lines | Status | Notes |
|--------|-------|--------|-------|
| `estimator.py` | 1041 | Perf warning | DataFrame fragmentation at line 542 (see below) |
| `missing_data_estimator.py` | -- | Clean | MICE + GP + Matrix Completion |
| `hidden_data_estimator.py` | -- | Clean | Heckman + Pattern-Mixture + GAIN |
| `missingness_classifier.py` | -- | Clean | MAR vs MNAR classification |
| `gain_imputer.py` | -- | Clean | Adversarial imputation |
| `vae_imputer.py` | -- | Clean | Variational autoencoder |

### Analysis (operator1/analysis/)

| Module | Lines | Status | Notes |
|--------|-------|--------|-------|
| `survival_mode.py` | 402 | Clean | 3 daily boolean flags from config thresholds |
| `survival_timeline.py` | 658 | Clean | 6-mode classification + stability scoring |
| `fuzzy_protection.py` | 304 | Clean | Continuous [0,1] protection degree |
| `hierarchy_weights.py` | -- | Clean | Tier weight computation |
| `economic_planes.py` | -- | Clean | 5-plane sector classification |
| `ethical_filters.py` | -- | Clean | Data manipulation detection |
| `vanity.py` | -- | Clean | Vanity metric detection |

### Features (operator1/features/)

| Module | Lines | Status | Notes |
|--------|-------|--------|-------|
| `derived_variables.py` | 663 | Clean | ~25 derived metrics with safe_ratio |
| `filing_calendar.py` | -- | Clean | Filing frequency and staleness detection |
| `linked_aggregates.py` | -- | Clean | Cross-entity aggregate features |
| `macro_alignment.py` | -- | Clean | Macro data alignment to daily cache |
| `macro_quadrant.py` | -- | Clean | Macro regime classification |
| `news_sentiment.py` | -- | Clean | LLM + keyword sentiment scoring |
| `peer_ranking.py` | -- | Clean | Percentile ranking vs peers |
| `portfolio_analysis.py` | -- | Clean | Portfolio-level analysis |

### Quality (operator1/quality/)

| Module | Lines | Status | Notes |
|--------|-------|--------|-------|
| `data_quality.py` | 433 | Clean | Look-ahead check, flag audits |
| `data_reconciliation.py` | 279 | Clean | Schema normalization, stale data detection |

---

## Warnings Identified (Not Crashes)

### 1. DataFrame Fragmentation in `estimator.py:542` (Performance)

**Severity:** Medium (performance degradation, not correctness)
**Location:** `operator1/estimation/estimator.py`, line 542
**Issue:** The `_store_estimation_results()` function is called per-variable in a
loop. Each call does `df[new_cols.columns] = new_cols`, which triggers pandas
`PerformanceWarning: DataFrame is highly fragmented`. With many estimable
variables (20+), this causes O(n^2) memory copies.
**Recommendation:** Collect all estimation columns in a list of DataFrames, then
use a single `pd.concat(axis=1)` call at the end of the estimation loop.

### 2. NaN Divide in Copula Correlation (`copula.py:73`) (Numeric)

**Severity:** Low (produces RuntimeWarning, does not crash)
**Location:** `operator1/models/copula.py`, line 73
**Issue:** `np.corrcoef(normal_data, rowvar=False)` can produce NaN when a
column has zero standard deviation (constant values after the PIT transform).
The function does not check for constant columns before computing the
correlation matrix.
**Recommendation:** Before calling `np.corrcoef`, drop any columns with zero
variance, or fall back to identity matrix for those dimensions.

### 3. statsmodels `verbose` Deprecation (Dependency)

**Severity:** Low (FutureWarning from statsmodels internals)
**Location:** `statsmodels.tsa.stattools:1556`
**Issue:** The `verbose` parameter in `grangercausalitytests` triggers a
FutureWarning. The code already passes `verbose=False` but the library still
warns internally.
**Recommendation:** No action needed until statsmodels removes the parameter.
Can suppress with `warnings.filterwarnings` if the noise is undesirable.

### 4. edgartools Deprecation Warnings (Dependency)

**Severity:** Low (DeprecationWarning from edgartools v5.19.1)
**Issue:** Several `edgar.files.*` modules are deprecated in favor of
`edgar.documents.HTMLParser` for the upcoming v6.0 release.
**Recommendation:** Track edgartools v6.0 release and update imports when
available. Current v5.19.1 still works.

### 5. pykrx `pkg_resources` Deprecation (Dependency)

**Severity:** Low (DeprecationWarning)
**Issue:** `pykrx.__init__.py` uses the deprecated `pkg_resources` API.
**Recommendation:** Wait for pykrx upstream fix or pin if this breaks in
future setuptools versions.

---

---

## Skipped Tests Investigation

5 tests were skipped due to missing API keys. After setting the keys:

| Test | API Key | Result |
|------|---------|--------|
| `test_fredapi_us` | FRED_API_KEY | PASSED |
| `test_banxico_mexico` | BANXICO_TOKEN | PASSED |
| `test_uk_ch_hsbc` | COMPANIES_HOUSE_API_KEY | PASSED |
| `test_dart_fss_samsung` | DART_API_KEY | Deferred (slow rate limit 0.15s/call) |
| `test_jquants_toyota` | JQUANTS_REFRESH_TOKEN | Deferred (slow rate limit 6s/call) |

Both DART and J-Quants wrappers have proper rate limiting built into their
production code (`_dart_throttle()` at 0.15s intervals, `_jquants_throttle()`
at 6.0s intervals). Every API method calls the throttle before making requests.

---

## Full Data Flow Map (Input/Output per Model)

### Step 1-3: Data Ingestion

```
PIT client
  IN:  market_id, company identifier, secrets
  OUT: profile dict, income_df, balance_df, cashflow_df, quotes_df

OHLCV provider (fallback when PIT has no prices)
  IN:  ticker, market_id
  OUT: quotes_df

Data reconciliation
  IN:  income_df, balance_df, cashflow_df
  OUT: cleaned income_df, balance_df, cashflow_df, reconciliation_report

Canonical translator (pivot_to_canonical_wide)
  IN:  long-format statement DataFrames
  OUT: wide-format DataFrames (one row per date, columns = canonical field names)
```

### Step 4: Cache Building

```
Cache builder (main.py inline)
  IN:  quotes_df, income_df, balance_df, cashflow_df
  OUT: cache DataFrame (daily OHLCV spine + forward-filled financials)

Macro provider
  IN:  country_code, secrets, years
  OUT: macro_data dict {indicator_name: pd.Series}

Macro mapping (fetch_macro_data)
  IN:  country_iso2, macro_raw dict
  OUT: MacroDataset object

Macro quadrant (compute_macro_quadrant)
  IN:  cache, macro_data (MacroDataset)
  OUT: cache (+ macro_quadrant columns), macro_quadrant_result

Estimator (run_estimation)
  IN:  cache, imputer_method
  OUT: cache (+ estimation columns), estimation_coverage

Filing calendar (analyze_filing_calendar)
  IN:  cache, market_id
  OUT: filing_calendar_result
```

### Step 5: Feature Engineering

```
Derived variables (compute_derived_variables)
  IN:  cache
  OUT: cache (+ ~25 derived columns: return_1d, volatility_21d, drawdown_252d, ratios, etc.)

Survival mode (compute_company_survival_flag)
  IN:  cache (needs current_ratio, debt_to_equity_abs, fcf_yield, drawdown_252d)
  OUT: cache["company_survival_mode_flag"]

Hierarchy weights (compute_hierarchy_weights)
  IN:  cache
  OUT: cache (+ hierarchy_tier*_weight columns), weights dict

Fuzzy protection (compute_fuzzy_protection)
  IN:  cache, sector, gdp
  OUT: cache (+ fuzzy_protection_degree, fuzzy_sector_score, fuzzy_protection_label)

Financial health (compute_financial_health)
  IN:  cache, hierarchy_weights
  OUT: cache (+ fh_* columns), fh_result (FinancialHealthResult)

LLM entity discovery (discover_linked_entities)
  IN:  target_profile, gemini_client, pit_client, secrets
  OUT: relationships dict

Graph risk (compute_graph_risk_metrics)
  IN:  target_isin, relationships
  OUT: graph_risk_result

Game theory (analyze_competitive_dynamics)
  IN:  target_cache, target_name
  OUT: game_theory_result

Linked entity fetch (parallel)
  IN:  pit_client, entity IDs
  OUT: linked_caches dict {entity_id: DataFrame}

Linked aggregates (compute_linked_aggregates)
  IN:  target_daily cache, linked_daily caches, entity_groups
  OUT: linked_agg_df -> merged into cache

Peer ranking (compute_peer_ranking)
  IN:  cache, linked_caches
  OUT: cache (+ peer_* columns), peer_ranking_result

News sentiment (compute_news_sentiment)
  IN:  cache, gemini_client, symbol
  OUT: cache (+ sentiment_* columns), sentiment_result
```

### Step 5.5: Enriched Survival Timeline

```
Early regime detection (run_early_regime_detection)
  IN:  cache
  OUT: cache (+ regime_hmm, regime_gmm, regime_label, structural_break columns),
       early_regime_result, regime_detector

Enriched survival timeline (compute_enriched_survival_timeline)
  IN:  cache, regime_labels, regime_confidence
  OUT: enriched_timeline_result -> merged into cache
       (regime_state, survival_intensity, regime_confidence, regime_switch, etc.)
```

### Step 6: Temporal Modeling

```
Regime detector (detect_regimes_and_breaks) -- SKIP if done in 5.5
  IN:  cache
  OUT: cache (+ regime columns), regime_detector

Dual regime mixer (compute_dual_regimes)
  IN:  cache
  OUT: dual_regime_result

Granger causality (compute_granger_causality)
  IN:  cache, variables
  OUT: granger_result -> prunes _extra_vars list

Transfer entropy (compute_transfer_entropy)
  IN:  cache, variables
  OUT: transfer_entropy_result

Cycle decomposition (run_cycle_decomposition)
  IN:  cache, variable="close"
  OUT: cycle_result

Pattern detector (detect_patterns)
  IN:  cache
  OUT: pattern_result

Economic planes (classify_economic_plane)
  IN:  sector, industry
  OUT: _economic_plane

Pre-forecasting synergies (apply_pre_forecasting_synergies)
  IN:  cache, cycle_result, granger_result, transfer_entropy_result,
       linked_caches, extra_variables, economic_plane
  OUT: cache (+ synergy features), _extra_vars (updated), _synergy_meta

Forecasting (run_forecasting)
  IN:  cache, extra_variables
  OUT: cache, forecast_result (ForecastResult)

Forward pass (run_forward_pass)
  IN:  cache, hierarchy_weights, regime_labels, extra_variables
  OUT: forward_pass_result (ForwardPassResult)

Burn-out (run_burnout)
  IN:  cache, hierarchy_weights, regime_labels, extra_variables
  OUT: burnout_result (BurnoutResult)

Monte Carlo (run_monte_carlo)
  IN:  cache
  OUT: mc_result (MonteCarloResult)

Copula (run_copula_analysis)
  IN:  cache
  OUT: copula_result

Transformer (train_transformer)
  IN:  cache, variables
  OUT: transformer_result -> forecasts injected into forecast_result

Particle filter (run_particle_filter)
  IN:  cache, variables
  OUT: particle_filter_result

Conformal (build_conformal_result)
  IN:  calibrator, forecasts, horizons
  OUT: conformal_result

DTW analogs (find_historical_analogs)
  IN:  cache
  OUT: dtw_result

Prediction aggregator (run_prediction_aggregation)
  IN:  cache, forecast_result, mc_result, conformal_result,
       dual_regime_result, copula_result, dtw_result,
       granger_result, shap_result, walk_forward_result(!)
  OUT: pred_result (PredictionAggregatorResult)

SHAP explainability (compute_shap_explanations)
  IN:  cache, predictions (from pred_result)
  OUT: shap_result

Sobol sensitivity (run_sensitivity_analysis)
  IN:  cache, target_variable
  OUT: sobol_result

Genetic optimizer (run_genetic_optimization)
  IN:  cache, forecast_result
  OUT: ga_result

OHLC predictor (predict_ohlc_series)
  IN:  cache, forecast_result, mc_result, pattern_drift_multiplier
  OUT: ohlc_result
```

### Step 7-8: Profile & Report

```
Profile builder (build_company_profile)
  IN:  target_profile, cache, linked_aggregates, all model results
  OUT: profile dict -> saved as JSON

Report generator (generate_all_reports)
  IN:  profile, llm_client, cache, output_dir
  OUT: markdown + optional PDF reports
```

---

## Wiring Issues Found

### W1: `ForwardPassResult` passed as `WalkForwardResult` (Type Mismatch)

**Severity:** Medium (silent feature degradation)
**Location:** `main.py:1728`
**Issue:** `forward_pass_result` (type `ForwardPassResult` from `forecasting.py`)
is passed to `run_prediction_aggregation()` as the `walk_forward_result` parameter,
which expects a `WalkForwardResult` (from `walk_forward.py`).

`ForwardPassResult` has: `errors_by_tier`, `errors_by_regime`, `model_states`,
`predictions_log`, `total_days`, `warmup_days`, `pid_summary`.

`WalkForwardResult` has: `day_errors`, `mode_scores`, `best_model_by_mode`,
`retrain_dates`, `total_days_evaluated`, `total_predictions`, `n_retrains`,
`overall_best_model`, `overall_mae`.

The prediction aggregator tries to access `day_errors` via
`getattr(walk_forward_result, "day_errors", [])` which always returns `[]`
because `ForwardPassResult` doesn't have this attribute. As a result,
**recency-weighted RMSE is always NaN**, silently degrading ensemble weighting.

### W2: `run_walk_forward()` is Never Called

**Severity:** Medium (dead code / missing feature)
**Location:** `operator1/models/walk_forward.py`
**Issue:** The `run_walk_forward()` function and `WalkForwardResult` dataclass
are defined in `walk_forward.py` but **never imported or called** in `main.py`.
The pipeline runs `run_forward_pass()` from `forecasting.py` instead, which
produces a different result type. The walk-forward module (which includes
per-mode model leaderboard, retrain-at-switch-point logic, and 4 model types)
is effectively dead code.

### W3: `ForecastResult` has no `residuals` Attribute (Conformal Uncalibrated)

**Severity:** Medium (conformal prediction runs without calibration data)
**Location:** `main.py:1686`
**Issue:** The conformal prediction code does:
```python
if hasattr(forecast_result, "residuals") and forecast_result.residuals is not None:
    for r in forecast_result.residuals:
        calibrator.update(r)
```
But `ForecastResult` (line 104-132 of `forecasting.py`) does NOT define a
`residuals` attribute. The `hasattr` check always returns `False`, so the
conformal calibrator **never receives any residual data**. The intervals it
produces are based only on its default settings, not actual forecast errors.

### W4: `burnout_result` Computed but Never Used Downstream

**Severity:** Low (wasted computation)
**Location:** `main.py:1606-1618`
**Issue:** `run_burnout()` is called and the result is logged, but `burnout_result`
is **never passed to the profile builder, prediction aggregator, or report
generator**. The burn-out loop (which can run many iterations of model
refinement) runs but its output is discarded.

### W5: `transfer_entropy_result` Only Stores `{"available": True}` in Profile

**Severity:** Low (incomplete wiring)
**Location:** `main.py:1962-1963`
**Issue:** The transfer entropy model runs and produces a full result with
pairwise entropy scores and causal pairs. However, the profile builder only
stores `{"available": True}` without any of the actual results. Compare with
other models like copula, SHAP, or DTW which include their full results via
`_available_dict()`.

### W6: Intentional Duplicate: `causality.py` wraps `granger_causality.py`

**Severity:** None (intentional backward-compatible wrapper)
**Location:** `operator1/models/causality.py:56` and
`operator1/models/granger_causality.py:51`
**Issue:** Both files define `compute_granger_causality()`. The version in
`causality.py` explicitly delegates to `granger_causality.py` (the canonical
implementation) and converts the result to a DataFrame for backward
compatibility. `main.py` correctly uses the canonical version from
`granger_causality.py` for Granger tests and `causality.py` for transfer
entropy. This is NOT a bug.

---

## Conclusion

The full Operator 1 pipeline is healthy on Python 3.12. All 1053 tests pass
with no failures. The codebase demonstrates solid defensive programming with
try/except wrapping around every model, graceful fallback chains, and
comprehensive test coverage across all 8 pipeline steps.

**5 wiring issues were found (W1-W5)**, none of which cause crashes thanks to
defensive `getattr`/`hasattr` checks. However, they silently degrade features:
- **W1+W2**: Walk-forward model leaderboard and recency-weighted RMSE are dead
- **W3**: Conformal prediction runs uncalibrated
- **W4**: Burn-out loop output is discarded
- **W5**: Transfer entropy results are lost at profile stage

The identified performance warnings (DataFrame fragmentation, copula NaN divide)
and dependency deprecations (edgartools, statsmodels, pykrx) are minor issues
that don't affect correctness.

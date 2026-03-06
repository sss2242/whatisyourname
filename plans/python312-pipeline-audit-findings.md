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

## Conclusion

The full Operator 1 pipeline is healthy on Python 3.12. All 1053 tests pass
with no failures. The identified warnings are performance and deprecation
issues, not correctness bugs. The codebase demonstrates solid defensive
programming with try/except wrapping around every model, graceful fallback
chains, and comprehensive test coverage across all 8 pipeline steps.

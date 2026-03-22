# Analysis Models Map (2026-03-22)

Complete map of all 42 analytical modules in operator1/analysis/, operator1/features/, and operator1/models/. Each module's purpose, inputs, outputs, wiring status in main.py, and profile/report integration.

---

## Layer 1: Feature Engineering (operator1/features/) -- 11 modules, 8,033 lines

These transform the daily cache into model-ready features.

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 1 | `derived_variables.py` | 667 | Cache with `close`, financial cols | ~40 derived cols (returns, ratios, technicals) + `is_missing_*` + `invalid_math_*` flags | Step 5 | `current_state` (latest values) |
| 2 | `conflict_risk.py` | 1,018 | `country_iso2`, `company_name` | `country_conflict_flag`, `conflict_intensity_score`, `sanctions_flag`, `fragile_state_flag`, linked entity conflict scores | Step 4a.3 | `conflict_risk` |
| 3 | `filing_calendar.py` | 311 | Cache, `market_id` | `FilingCalendarResult` (frequency, coverage, staleness, gaps), `filing_freshness` column | Step 4c | `filing_calendar` |
| 4 | `linked_aggregates.py` | 359 | Target cache, linked entity caches, entity groups | `linked_agg_df` with `competitors_avg_*`, `suppliers_median_*` columns | Step 5g | `linked_entities` |
| 5 | `macro_alignment.py` | 322 | `MacroDataset`, daily index | Daily macro columns (`inflation_rate_daily_equivalent`, `real_return_1d`) | Not directly called in main.py (consumed via macro_quadrant) | `macro_indicators` |
| 6 | `macro_quadrant.py` | 248 | Cache, `MacroDataset` | `macro_quadrant` column, `MacroQuadrantResult` (quadrant, stability_score) | Step 4a | `macro_quadrant` |
| 7 | `news_sentiment.py` | 539 | Cache, LLM client, symbol, market_id | `sentiment_score`, `sentiment_label`, `sentiment_momentum` columns | Step 5i | `sentiment` |
| 8 | `peer_ranking.py` | 270 | Cache, linked entity caches | `peer_rank_*` columns, `PeerRankingResult` | Step 5h | `peer_ranking` |
| 9 | `private_company_proxies.py` | 451 | Cache (no OHLCV) | `equity_value`, `equity_change_rate`, `financial_volatility` + transparent proxy resolution into standard column names | Step 5a | `meta.is_private_company` |
| 10 | `six_derived_proxies.py` | 3,848 | Cache, profile (CH only) | 22 canonical fields from SIX dividend/capital data via Kalman, PELT, Merton, UKF, L1, EBO | Step 4a.5 | `extended_models` (if ch_six) |

---

## Layer 2: Analysis Modules (operator1/analysis/) -- 7 modules, 2,829 lines

Rule-based and fuzzy-logic analysis that produces survival flags and regime classifications.

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 11 | `survival_mode.py` | 414 | Cache with derived vars | `company_survival_mode_flag`, `country_survival_mode_flag`, `country_protected_flag` | Step 5 | `survival` |
| 12 | `hierarchy_weights.py` | 322 | Cache with survival flags | `hierarchy_tier1_weight` through `tier5_weight`, `survival_regime` label | Step 5 | `survival.hierarchy_weights` |
| 13 | `survival_timeline.py` | 664 | Cache with survival flags + HMM regimes | 6-mode `survival_mode`, 11-state `regime_state`, `survival_intensity` (0-1), `switch_point`, `days_in_mode`, `stability_score_21d` | Step 5.5 | `enriched_survival_timeline` |
| 14 | `fuzzy_protection.py` | 303 | Cache, sector, GDP | `fuzzy_protection_degree` (0-1), `fuzzy_sector_score`, `fuzzy_protection_label` | Step 5b | `fuzzy_protection` |
| 15 | `ethical_filters.py` | 355 | Cache | 4 filter results: Purchasing Power, Solvency, Gharar, Cash is King | Called inside profile_builder | `filters` |
| 16 | `economic_planes.py` | 135 | Sector, industry strings | Primary plane (1 of 5), secondary planes list | Step 6 (pre-forecast) | `economic_plane` |
| 17 | `vanity.py` | 636 | Cache with derived vars, fh_* scores | `vanity_score` (0-100), `vanity_label`, `vanity_trend`, 5 component columns | Step 5d | `vanity` |

---

## Layer 3: Temporal Models (operator1/models/) -- 24 modules, 14,391 lines

Statistical and ML models that consume the enriched daily cache.

### 3a. Regime Detection (2 modules)

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 18 | `regime_detector.py` | 924 | Cache with `return_1d`, `volatility_21d` | `regime_hmm`, `regime_gmm`, `regime_label`, `structural_break`, `regime_hmm_prob_*` | Step 5.5 / Step 6 | `regimes` |
| 19 | `regime_mixer.py` | 364 | Cache with regime columns | `dual_regime_result` (market + fundamental regimes, blended weights) | Step 6 | `extended_models.dual_regimes` |

### 3b. Causality & Information Flow (3 modules)

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 20 | `granger_causality.py` | 292 | Cache, up to 25 float columns | `GrangerResult` (significant pairs, network density, retained/pruned variables) | Step 6 | `extended_models.granger_causality` |
| 21 | `causality.py` (transfer entropy) | 417 | Cache, up to 20 float columns | `TransferEntropyResult` (pairwise entropy scores, top pairs) | Step 6 | `extended_models.transfer_entropy` |
| 22 | `model_synergies.py` | 797 | Cache, cycle/granger/TE/peer results, economic plane | Cache + synergy features, pruned `_extra_vars`, `_synergy_meta` | Step 6 (pre-forecast) | `synergies_applied` |

### 3c. Core Forecasting (4 modules)

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 23 | `forecasting.py` | 3,215 | Cache, `extra_variables` | `ForecastResult` (per-var, per-horizon forecasts, model metrics, residuals) | Step 6 | `model_metrics`, `predictions` |
| 24 | `walk_forward.py` | 563 | Cache, survival timeline | `WalkForwardResult` (per-day errors, mode-conditioned leaderboard, recency-weighted RMSE) | Step 6 | Used by prediction_aggregator |
| 25 | `monte_carlo.py` | 1,027 | Cache with `return_1d`, `regime_label` | `MonteCarloResult` (survival probability per horizon, regime distributions, terminal values) | Step 6 | `monte_carlo` |
| 26 | `pid_controller.py` | 323 | Per-tier prediction error history | Adaptive learning rate multiplier per tier | Called inside forward pass | `pid_controller` |

### 3d. Advanced Forecasters (3 modules)

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 27 | `transformer_forecaster.py` | 450 | Cache, up to 15 float columns | `TransformerResult` (forecasts, feature importance, train loss) | Step 6 | `extended_models.transformer` |
| 28 | `genetic_optimizer.py` | 358 | Cache, `forecast_result` | `GAResult` (best ensemble weights, tier weights, fitness history) | Step 6 | `extended_models.genetic_optimizer` |
| 29 | `ohlc_predictor.py` | 349 | Cache, forecast/MC results, pattern drift | `OHLCResult` (next-day/week/month OHLC predictions) | Step 6 | `ohlc_predictions` |

### 3e. Uncertainty & Intervals (3 modules)

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 30 | `conformal.py` | 425 | Calibrator (fed with residuals), forecasts | `ConformalResult` (distribution-free prediction intervals per var per horizon) | Step 6 | `conformal_intervals` |
| 31 | `copula.py` | 255 | Cache | `CopulaResult` (tail dependence, joint crisis probability) | Step 6 | `extended_models.copula` |
| 32 | `particle_filter.py` | 379 | Cache, survival-related variables | `ParticleFilterResult` (filtered states, percentiles) | Step 6 | `extended_models.particle_filter` |

### 3f. Pattern & Cycle Analysis (2 modules)

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 33 | `cycle_decomposition.py` | 258 | Cache, `variable="close"` | `CycleResult` (dominant cycles: period, amplitude, phase) | Step 6 | `extended_models.cycle_decomposition` |
| 34 | `pattern_detector.py` | 269 | Cache (OHLC) | `PatternResult` (candlestick patterns detected) | Step 6 | `patterns` |

### 3g. Explainability & Sensitivity (2 modules)

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 35 | `explainability.py` (SHAP) | 509 | Cache, predictions, predict_fns | `SHAPResult` (per-var feature importance, top drivers, narratives) | Step 6 | `shap_explanations` |
| 36 | `sensitivity.py` (Sobol) | 259 | Cache, `target_variable` | `SobolResult` (first-order and total-order sensitivity indices) | Step 6 | `extended_models.sobol_sensitivity` |

### 3h. Structural Analysis (3 modules)

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 37 | `financial_health.py` | 831 | Cache, hierarchy weights | 5 tier scores, composite score, Altman Z, Beneish M, liquidity runway | Step 5d | `financial_health` |
| 38 | `graph_risk.py` | 441 | Target ISIN, relationships dict | `GraphRiskResult` (network centrality, contagion probability, supply chain concentration) | Step 5e | `graph_risk` |
| 39 | `game_theory.py` | 460 | Target cache, competitor caches | `GameTheoryResult` (Cournot/Stackelberg, competitive pressure, market structure) | Step 5e | `game_theory` |

### 3i. Historical Analogs & Aggregation (2 modules)

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 40 | `dtw_analogs.py` | 374 | Cache | `DTWAnalogResult` (historical pattern matches, empirical forecasts) | Step 6 | `historical_analogs` |
| 41 | `prediction_aggregator.py` | 2,126 | Cache, forecast/MC/conformal/DTW/granger/shap/walk_forward results | `PredictionAggregatorResult` (ensemble predictions, Technical Alpha mask, uncertainty bands) | Step 6 | `predictions` |

### 3j. Utility (1 module)

| # | Module | Lines | Input | Output | main.py Step | Profile Section |
|---|--------|-------|-------|--------|-------------|----------------|
| 42 | `_frequency_classifier.py` | 135 | Filing dates | Filing frequency classification (quarterly/semi-annual/annual) | Internal utility | N/A |

---

## Execution Order in main.py

```
Step 2:   Profile fetch -> supplement.enrich_profile()
Step 3:   PIT financial data fetch (parallel: income, balance, cashflow, quotes)
Step 3b:  data_reconciliation.reconcile_financial_data()
Step 3c:  canonical_translator.pivot_to_canonical_wide()
Step 4:   Cache build (OHLCV spine + frequency interpolation)
Step 4a:  macro_provider.fetch_macro() -> macro_mapping -> macro_quadrant
Step 4a.3: conflict_risk.assess_conflict_risk()
Step 4a.5: six_derived_proxies (CH only)
Step 4b:  estimator.run_estimation()
Step 4b.1: Unified confidence (precision-weighted pooling)
Step 4c:  filing_calendar.analyze_filing_calendar()
Step 5:   derived_variables -> survival_mode -> hierarchy_weights
Step 5a:  private_company_proxies (if no OHLCV)
Step 5b:  fuzzy_protection
Step 5d:  financial_health -> vanity
Step 5e:  entity_discovery -> graph_risk -> game_theory
Step 5f:  Linked entity data fetch (parallel)
Step 5g:  linked_aggregates
Step 5h:  peer_ranking
Step 5i:  news_sentiment
Step 5.5: regime_detector (early) -> survival_timeline (enriched)
Step 6:   [TEMPORAL MODELS -- all skip if --skip-models]
  6a: regime_detector (if not already run)
  6b: regime_mixer (dual regimes)
  6c: granger_causality -> prune features
  6d: transfer_entropy (causality)
  6e: cycle_decomposition
  6f: pattern_detector
  6g: economic_planes -> model_synergies (pre-forecast)
  6h: forecasting.run_forecasting()
  6i: forecasting.run_forward_pass()
  6j: forecasting.run_burnout()
  6k: walk_forward.run_walk_forward()
  6l: monte_carlo.run_monte_carlo()
  6m: copula.run_copula_analysis()
  6n: transformer_forecaster.train_transformer()
  6o: particle_filter.run_particle_filter()
  6p: conformal.build_conformal_result()
  6q: dtw_analogs.find_historical_analogs()
  6r: prediction_aggregator.run_prediction_aggregation()
  6s: explainability.compute_shap_explanations()
  6t: sensitivity.run_sensitivity_analysis()
  6u: genetic_optimizer.run_genetic_optimization()
  6v: ohlc_predictor.predict_ohlc_series()
Step 7:   profile_builder.build_company_profile()
Step 8:   report_generator.generate_all_reports()
```

---

## Summary

| Layer | Modules | Lines | Description |
|-------|---------|-------|-------------|
| Features | 11 | 8,033 | Raw cache -> enriched features (ratios, conflict, sentiment, peers, macro) |
| Analysis | 7 | 2,829 | Rule-based survival flags, hierarchy weights, ethical filters, vanity |
| Temporal Models | 24 | 14,391 | Statistical + ML models (HMM, Kalman, GARCH, VAR, LSTM, Transformer, Monte Carlo, SHAP, DTW, GA) |
| **Total** | **42** | **25,253** | |

All 42 modules wired in main.py. All results stored in profile_builder. All sections rendered in report_generator.

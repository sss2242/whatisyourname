# Operator 1 -- Architecture Overview

## What is Operator 1?

Operator 1 is an **institutional-grade financial analysis pipeline** that generates equity research reports from free government filings. It covers **10 markets** across 4 continents totalling **$91T+ in market cap**, using 25+ mathematical models and strictly point-in-time (PIT) audited data to prevent look-ahead bias.

## Environment Setup

- **Python**: 3.12.3 (compiled from source via `install.sh`)
- **pip**: bundled with Python 3.12 via `ensurepip`
- **OS tested**: Amazon Linux 2023 (x86_64)

### Dependency Stages

Dependencies are split into four progressive stages in `requirements/`:

| Stage | File | Contents | Install time |
|-------|------|----------|-------------|
| 1 | `stage1-core.txt` | numpy, pandas, scipy, requests, pyyaml, matplotlib, pytest | ~30s |
| 2 | `stage2-ml.txt` | scikit-learn, statsmodels, xgboost, ruptures, hmmlearn, arch, shap, mapie, dtaidistance | ~1-2min |
| 3 | `stage3-deeplearning.txt` | torch, arviz, pymc | ~5-8min |
| 4 | `stage4-wrappers.txt` | edgartools, yfinance, wbgapi, fredapi, dart-fss, pykrx, ixbrl-parse, pycvm, baostock, nselib, twstock, pdfplumber, camelot-py, gnews, feedparser, and more | ~1-2min |

The pipeline degrades gracefully: if stage 3 (deep learning) is missing, it falls back to statistical models.

## Entry Points

- **`run.py`** -- Interactive terminal launcher. Guides the user through a 10-step flow: system checks, dependency verification, LLM provider selection, data source mode, API keys, region/market/company selection, LLM-assisted company validation, pipeline options, confirmation, and execution.
- **`install.sh`** -- Automated setup script. Compiles Python 3.12.3 from source and installs all four dependency stages.

## Core Package: `operator1/`

### Pipeline Steps (`operator1/steps/`)

These run in sequence as the main pipeline phases:

| Module | Phase | Purpose |
|--------|-------|---------|
| `verify_identifiers.py` | A | Verify company identifiers via PIT API |
| `entity_discovery.py` | A/B | LLM-driven discovery of linked entities (competitors, suppliers, customers, etc.) with fuzzy matching |
| `data_extraction.py` | C | Bulk fetch of profiles, financials, OHLCV for target + linked entities; caches as Parquet |
| `cache_builder.py` | D | Builds daily-frequency as-of DataFrame with strict no-look-ahead invariant |
| `macro_mapping.py` | D+ | Maps macro indicators to the target's country/region |
| `parallel_executor.py` | F | Thread-pool executor for running independent models concurrently |

### Clients (`operator1/clients/`)

#### PIT Data Clients (one per market)
Each implements the `PITClient` protocol from `pit_base.py`:

| Client | Market | Exchange | Data Source |
|--------|--------|----------|-------------|
| `us_edgar.py` | US | NYSE/NASDAQ | SEC EDGAR |
| `uk_ch_wrapper.py` | UK | LSE | Companies House |
| `eu_esef_wrapper.py` | EU | Euronext, Frankfurt | ESEF/XBRL |
| `jp_jquants_wrapper.py` | Japan | TSE | J-Quants (EDINET) |
| `kr_dart_wrapper.py` | South Korea | KOSPI/KOSDAQ | DART |
| `tw_mops_wrapper.py` | Taiwan | TWSE/TPEX | MOPS |
| `br_cvm_wrapper.py` | Brazil | B3 | CVM |
| `cl_cmf_wrapper.py` | Chile | CMF | CMF API |

#### OHLCV Providers
| Client | Market |
|--------|--------|
| `ohlcv_yfinance.py` | Global fallback |
| `ohlcv_baostock.py` | China |
| `ohlcv_pykrx.py` | South Korea |
| `ohlcv_twstock.py` | Taiwan |
| `ohlcv_nselib.py` | India |

#### Macro Data Providers
| Client | Source |
|--------|--------|
| `macro_fredapi.py` | US Federal Reserve (FRED) |
| `macro_wbgapi.py` | World Bank |
| `macro_sdmx.py` | SDMX (EU statistics) |
| `macro_bcb.py` | Brazil Central Bank |
| `macro_bcch.py` | Chile Central Bank |
| `macro_banxico.py` | Mexico Central Bank |
| `macro_ons.py` | UK ONS |
| `macro_estat.py` | Japan e-Stat |
| `macro_kosis.py` | South Korea KOSIS |
| `macro_dgbas.py` | Taiwan DGBAS |

#### LLM Clients
| Client | Provider |
|--------|----------|
| `gemini.py` | Google Gemini |
| `claude.py` | Anthropic Claude |
| `openrouter.py` | OpenRouter (200+ models) |
| `llm_factory.py` | Factory with multi-key rotation and cross-provider fallback |
| `llm_base.py` | Abstract base with retry logic and rate limiting |

#### Other Clients
- `canonical_translator.py` -- Normalizes financial data from different markets into a common schema
- `equity_provider.py` -- Provider abstraction and PIT client factory
- `pit_registry.py` -- Global registry of all 10+ supported markets with metadata
- `filing_discoverer.py` -- Discovers available filings for a company
- `llm_filing_extractor.py` -- LLM-powered extraction of structured data from PDF filings
- `supplement.py` -- Supplementary data enrichment

### Features (`operator1/features/`)

| Module | Task | Purpose |
|--------|------|---------|
| `derived_variables.py` | T3.3 | ~25 derived financial metrics (ratios, yields, margins) with safe division |
| `conflict_risk.py` | -- | Geopolitical conflict and sanctions risk scoring |
| `filing_calendar.py` | -- | Filing date tracking and calendar enforcement |
| `linked_aggregates.py` | -- | Aggregate statistics across linked entities |
| `macro_alignment.py` | -- | Align macro indicators to the daily cache frequency |
| `macro_quadrant.py` | -- | Classify macro environment into quadrants |
| `news_sentiment.py` | -- | News-based sentiment scoring |
| `peer_ranking.py` | -- | Peer comparison and ranking |
| `portfolio_analysis.py` | -- | Portfolio-level analysis |
| `private_company_proxies.py` | -- | Proxy metrics for private companies without OHLCV data |

### Analysis (`operator1/analysis/`)

| Module | Task | Purpose |
|--------|------|---------|
| `survival_mode.py` | T4.1 | Detects company/country survival flags from financial distress indicators |
| `survival_timeline.py` | Phase 2 | Classifies each day into one of 6 survival modes with stability scoring |
| `economic_planes.py` | -- | Multi-plane economic analysis framework |
| `ethical_filters.py` | -- | Ethical screening filters |
| `fuzzy_protection.py` | -- | Fuzzy logic for government protection assessment |
| `hierarchy_weights.py` | -- | Hierarchical weighting system |
| `vanity.py` | -- | Vanity percentage computation (market sentiment distortion metric) |

### Estimation (`operator1/estimation/`)

| Module | Task | Purpose |
|--------|------|---------|
| `estimator.py` | T5.1 | Three-phase estimation engine: deterministic identity fill, missingness classification (MAR vs MNAR), specialized imputation |
| `missing_data_estimator.py` | -- | MAR path: MICE, Gaussian Process, Matrix Completion (Soft-Impute) |
| `hidden_data_estimator.py` | -- | MNAR path: Heckman Selection, Pattern-Mixture, Sensitivity Bounds, GAIN |
| `missingness_classifier.py` | -- | Classifies NaN values as MAR/MCAR/MNAR |
| `frequency_interpolator.py` | -- | Frequency conversion (quarterly to daily) |
| `gain_imputer.py` | -- | Generative Adversarial Imputation Network |
| `vae_imputer.py` | -- | Variational Autoencoder imputation (legacy) |

### Models (`operator1/models/`)

25+ mathematical/statistical models organized by function:

| Category | Models |
|----------|--------|
| **Regime Detection** | `regime_detector.py` (HMM + structural breaks), `regime_mixer.py` (dual-regime blending) |
| **Forecasting** | `forecasting.py` (Kalman, GARCH, VAR, LSTM, RF/GBM/XGB with fallback chains) |
| **Monte Carlo** | `monte_carlo.py` (survival probability simulation) |
| **Prediction** | `prediction_aggregator.py` (ensemble weighting, multi-horizon, conformal intervals) |
| **Causality** | `granger_causality.py`, `causality.py` |
| **Risk** | `graph_risk.py` (network risk propagation), `sensitivity.py` (Sobol indices), `game_theory.py` (Nash equilibria) |
| **Pattern** | `pattern_detector.py`, `dtw_analogs.py` (Dynamic Time Warping) |
| **Advanced** | `transformer_forecaster.py`, `particle_filter.py`, `copula.py`, `cycle_decomposition.py` |
| **Optimization** | `genetic_optimizer.py` (genetic algorithm), `walk_forward.py` |
| **Interpretability** | `explainability.py` (SHAP), `conformal.py` (conformal prediction) |
| **Health** | `financial_health.py` (composite health scoring), `ohlc_predictor.py` |

### Quality (`operator1/quality/`)

| Module | Task | Purpose |
|--------|------|---------|
| `data_quality.py` | T4.4 | Look-ahead violation detection, ratio safety audit, missing-data audit |
| `data_reconciliation.py` | -- | Cross-source data reconciliation |

### Report Generation (`operator1/report/`)

| Module | Task | Purpose |
|--------|------|---------|
| `profile_builder.py` | T7.1 | Aggregates all pipeline outputs into `company_profile.json` |
| `profile_schema.py` | -- | JSON schema for the profile |
| `report_generator.py` | T7.2 | Generates branded Markdown/PDF reports with three tiers (Basic/Pro/Premium) and two modes (Learn/Results) |

## Configuration (`config/`)

| File | Purpose |
|------|---------|
| `global_config.yml` | HTTP settings, rate limits, cache TTL, estimation config, LLM provider settings |
| `canonical_fields.yml` | Mapping of market-specific field names to canonical schema |
| `country_protection_rules.yml` | Thresholds for government protection detection |
| `country_survival_rules.yml` | Macro thresholds for country survival flags |
| `economic_planes.yml` | Economic plane analysis configuration |
| `survival_hierarchy.yml` | Hierarchy weights for survival analysis |

## Data Flow Summary

```
User Input (company + market)
    |
    v
[A] Entity Discovery (LLM proposes linked entities, fuzzy-matched via PIT API)
    |
    v
[B] Identifier Verification (validate against PIT data source)
    |
    v
[C] Data Extraction (profiles, financials, OHLCV -> Parquet cache)
    |
    v
[D] Cache Builder (daily as-of DataFrame, no look-ahead)
    |
    v
[D+] Feature Engineering (derived ratios, macro alignment, sentiment)
    |
    v
[E] Estimation (fill missing values: MAR via MICE/GP/Matrix, MNAR via Heckman/GAIN)
    |
    v
[F] Temporal Models (regime detection, forecasting, Monte Carlo, parallel execution)
    |
    v
[F+] Prediction Aggregation (ensemble weighting, conformal intervals)
    |
    v
[G] Quality Audit (look-ahead check, ratio safety, coverage report)
    |
    v
[H] Profile Builder (aggregate all outputs -> company_profile.json)
    |
    v
[I] Report Generator (LLM-powered Markdown/PDF with charts)
```

## Key Design Principles

1. **Point-in-Time (PIT) data only**: All financial data is joined as-of filing date, never report date, to prevent look-ahead bias.
2. **Graceful degradation**: Every model and data source has fallback chains. Missing optional dependencies (e.g., PyTorch) trigger warnings, not failures.
3. **Multi-provider LLM**: Supports Gemini, Claude, and OpenRouter with automatic key rotation and cross-provider fallback.
4. **Observed values are sacred**: The estimation engine never overwrites observed data -- it only fills gaps.
5. **Free data sources**: All PIT APIs are free government sources. No paid data subscriptions required.

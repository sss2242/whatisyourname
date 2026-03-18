# Sandbox Environment Setup and Codebase Overview

## Environment Setup

### Python 3.12.3 Installation
- Built from source on Amazon Linux 2023 (x86_64)
- Installed to `/usr/local/bin/python3.12` via `make altinstall`
- pip 24.0 bundled via `--with-ensurepip=install`

### Dependency Installation (4 stages)
All stages installed successfully with `pip3.12 install --timeout 300 -r requirements/stageN.txt`:

| Stage | Contents | Key Packages | Status |
|-------|----------|-------------|--------|
| 1 - Core | NumPy, Pandas, SciPy, requests, PyYAML, matplotlib, pytest | numpy 2.4.2, pandas 2.3.3, scipy 1.17.1 | OK |
| 2 - ML | scikit-learn, statsmodels, XGBoost, ruptures, hmmlearn, arch, SHAP, MAPIE, DTW | scikit-learn 1.8.0, xgboost 3.2.0 | OK |
| 3 - Deep Learning | PyTorch, ArviZ, PyMC | torch 2.10.0, pymc 5.28.0 | OK |
| 4 - Wrappers | edgartools, dart-fss, pykrx, yfinance, wbgapi, fredapi, pdfplumber, camelot-py, gnews | edgartools 5.19.1, yfinance 1.2.0 | OK |

### Smoke Test
All stage imports verified OK via `python3.12 -c "import numpy, pandas, ... print('All stage imports OK')"`.

---

## Codebase Architecture Summary

### What Is Operator 1?
Operator 1 is an institutional-grade financial analysis pipeline that:
1. Fetches company data from free government Point-in-Time (PIT) filing APIs (SEC EDGAR, DART, ESEF, etc.)
2. Builds a 2-year daily cache with financial statements, OHLCV prices, and macro indicators
3. Runs 25+ statistical/ML/deep learning models for regime detection, forecasting, and survival analysis
4. Generates Bloomberg-style reports in 3 tiers (Basic/Pro/Premium) via LLM or template fallback

### Pipeline Flow (8 Steps in main.py)

```
Step 1: Market/Company Selection
  pit_registry -> create_pit_client -> search_company

Step 2: Company Profile
  pit_client.get_profile -> canonical_translator -> supplement.enrich_profile

Step 3: PIT Financial Data (parallel fetch)
  income_statement + balance_sheet + cashflow + quotes
  -> data_reconciliation -> pivot_to_canonical_wide
  -> OHLCV fallback (yfinance/pykrx/baostock/twstock/nselib)

Step 4: Daily Cache Build
  OHLCV spine + forward-fill statements (as-of join, no look-ahead)
  -> macro_provider.fetch_macro -> MacroDataset -> macro_quadrant
  -> estimation.run_estimation (Split: MAR/MNAR classification)
  -> filing_calendar analysis

Step 5: Feature Engineering
  derived_variables (25+ ratios) -> survival_mode -> hierarchy_weights
  -> fuzzy_protection -> financial_health (Altman Z, Beneish M)
  -> vanity_score -> entity_discovery (LLM) -> linked_aggregates
  -> peer_ranking -> news_sentiment -> conflict_risk

Step 5.5: Enriched Survival Timeline
  early_regime_detection (HMM/GMM/PELT/BCP)
  -> compute_enriched_survival_timeline

Step 6: Temporal Modeling (25+ models, all optional)
  regime_detector -> dual_regimes -> granger_causality
  -> transfer_entropy -> cycle_decomposition -> pattern_detector
  -> pre_forecasting_synergies (plane-aware weighting)
  -> forecasting (Kalman/GARCH/VAR/LSTM/Tree/Baseline)
  -> forward_pass (day-by-day with PID controller)
  -> burn_out -> walk_forward -> monte_carlo -> copula
  -> transformer -> particle_filter -> conformal_prediction
  -> dtw_analogs -> prediction_aggregator -> SHAP
  -> sobol_sensitivity -> genetic_optimizer -> ohlc_predictor

Step 7: Profile Build
  build_company_profile -> inject all model results + metadata

Step 8: Report Generation
  generate_all_reports (Basic + Pro + Premium markdown + optional PDF)
```

### Module Organization

| Package | Purpose | Key Modules |
|---------|---------|-------------|
| `operator1/clients/` | Data source wrappers (10 PIT markets + OHLCV + macro + LLM) | `equity_provider`, `canonical_translator`, `llm_factory`, `ohlcv_provider`, `macro_provider` |
| `operator1/features/` | Feature engineering | `derived_variables`, `filing_calendar`, `linked_aggregates`, `peer_ranking`, `news_sentiment`, `conflict_risk`, `macro_quadrant` |
| `operator1/analysis/` | Survival/protection/vanity | `survival_mode`, `hierarchy_weights`, `fuzzy_protection`, `economic_planes`, `vanity`, `ethical_filters` |
| `operator1/estimation/` | Missing data imputation | `estimator` (Split: MAR via MICE+GP+MatrixCompletion, MNAR via Heckman+PatternMixture+GAIN) |
| `operator1/models/` | 25+ temporal models | `regime_detector`, `forecasting`, `monte_carlo`, `prediction_aggregator`, `copula`, `transformer_forecaster`, `particle_filter`, etc. |
| `operator1/quality/` | Data quality enforcement | `data_quality`, `data_reconciliation` |
| `operator1/report/` | Report generation | `profile_builder`, `report_generator` (3 tiers, 22 sections) |
| `operator1/steps/` | Pipeline orchestration | `entity_discovery`, `data_extraction`, `macro_mapping`, `parallel_executor` |

### Supported Markets (10 PIT sources)

| Market | API | Country | Wrapper |
|--------|-----|---------|---------|
| US | SEC EDGAR | United States | `edgartools` + `sec-edgar-api` |
| EU | ESEF/XBRL | Europe | `eu_esef_wrapper` |
| UK | Companies House | United Kingdom | `ixbrl-parse` |
| JP | J-Quants | Japan | `jquants-api-client` |
| KR | DART | South Korea | `dart-fss` |
| TW | MOPS | Taiwan | `twstock` |
| BR | CVM | Brazil | `pycvm` |
| CL | CMF | Chile | `cl_cmf_wrapper` |
| CN | SSE | China | `baostock` |
| IN | BSE | India | `nselib` |

### Configuration Files

| File | Purpose |
|------|---------|
| `config/global_config.yml` | HTTP timeouts, rate limits, estimation method, LLM provider, filing extraction pacing |
| `config/survival_hierarchy.yml` | 5-tier hierarchy weight presets (normal, survival, modified, extreme, vanity-adjusted) |
| `config/country_protection_rules.yml` | Strategic sector definitions, GDP threshold for government protection |
| `config/country_survival_rules.yml` | Country-level survival triggers (credit spread, unemployment surge, yield curve) |
| `config/canonical_fields.yml` | Canonical field name mappings across all PIT sources |
| `config/economic_planes.yml` | 5-plane economic classification (Real Economy, Financial, Technology, Resources, Services) |

### Test Suite
~80 test files in `tests/`, covering:
- Phase smoke tests (1-7)
- Per-wrapper tests (US EDGAR, JP J-Quants, KR DART, etc.)
- Per-model tests (forecasting, Monte Carlo, regime, prediction aggregator)
- Live integration tests (per-country pipeline runs)
- Feature tests (canonical translator, equity provider, LLM clients)

### Key Design Decisions
1. **PIT-only data**: All financial data comes from government filing APIs (no commercial data providers)
2. **Graceful degradation**: Every model wrapped in try/except; pipeline never crashes from a single model failure
3. **Canonical translation**: All 10 PIT sources produce uniform field names via `canonical_translator.py`
4. **As-of join**: Financial statements forward-filled to daily using report_date (never filing_date) to prevent look-ahead bias
5. **3-tier reports**: Basic (5 sections, quick screening), Pro (13 sections, peers + macro), Premium (22 sections, full institutional)
6. **LLM-agnostic**: Supports Gemini, Claude, and OpenRouter with auto-detection based on available API keys

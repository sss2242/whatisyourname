# Operator 1 -- Point-in-Time Financial Analysis Pipeline

Institutional-grade equity research powered by free government filing APIs. Covers 25 markets across 7 regions, $100+ trillion in market cap, with 25+ mathematical models and zero data subscription costs.

---

## What Is Operator 1?

Operator 1 is an open-source equity research pipeline that performs the same type of fundamental and quantitative analysis that institutional investors pay $20,000+/year per Bloomberg Terminal to access -- except it pulls data directly from free government filing APIs and runs everything locally on your machine.

The core idea: every public company in the world is required by law to file financial statements with a government regulator. Those filings are public, free, and immutable (they have a filing date that never changes). Operator 1 taps into those regulatory APIs across 25 global markets, normalizes the data into a single format, runs 25+ quantitative models, and produces a professional investment report.

No subscriptions. No paid data feeds. Just government data and math.

---

## How It Works

You pick a market, search for a company, and Operator 1 runs an 8-step pipeline:

**Step 1 -- Data Source Selection**
Choose a region and market. The app connects to the corresponding government filing API (SEC EDGAR for the US, J-Quants for Japan, DART for Korea, etc.).

**Step 2 -- Company Profile**
Fetches the company's identity, sector, industry, and exchange information from the regulatory database.

**Step 3 -- Point-in-Time Financial Data**
Downloads income statements, balance sheets, and cash flow statements with their original filing dates. Also fetches OHLCV price data via yfinance (and regional providers like pykrx, akshare, jugaad-data, twstock). All data is merged using filing dates (not report dates) to prevent look-ahead bias -- the same discipline used by quantitative hedge funds.

**Step 4 -- Daily Cache Construction**
Builds a unified time-indexed table where financial statement data is forward-filled onto daily price data using as-of joins. Macroeconomic indicators (GDP, inflation, interest rates, unemployment, FX) from central bank APIs are merged in. Missing values are filled using accounting identities first, then Bayesian Ridge regression or a Variational Autoencoder.

**Step 5 -- Feature Engineering & Analysis**
Computes 25+ derived financial ratios (margins, coverage ratios, yields, returns). Runs survival mode detection to flag companies in financial distress. Applies fuzzy logic government protection scoring. Calculates financial health scores (0-100) across five tiers: liquidity, solvency, stability, profitability, and growth. Runs ethical filters for purchasing power, leverage, speculation risk, and cash flow quality. Optionally discovers linked entities (competitors, suppliers, customers) via Gemini or Claude AI.

**Step 6 -- Temporal Modeling**
Runs 25+ models including:
- Regime detection (Hidden Markov Model, Gaussian Mixture, PELT structural breaks, Bayesian change point)
- Forecasting (Kalman Filter, GARCH, VAR, LSTM, Random Forest, XGBoost, Transformer)
- Monte Carlo simulation (10,000 regime-aware paths with importance sampling)
- Causality analysis (Granger, Transfer Entropy, Copula tail dependency)
- Uncertainty quantification (Conformal Prediction, distribution-free intervals)
- Pattern recognition (candlestick detector, Fourier/wavelet cycles, DTW historical analogs)
- Optimization (Genetic Algorithm for ensemble weights, PID adaptive learning)
- Explainability (SHAP feature attribution, Sobol sensitivity)

Each model is wrapped in a fallback chain -- if a dependency is missing or data is insufficient, the next model in the chain picks up automatically.

**Step 7 -- Profile Assembly**
All results are collected into a single `company_profile.json` containing every metric, score, forecast, and model output.

**Step 8 -- Report Generation**
Produces three tiers of investment reports:
- **Basic** (5 sections) -- quick screening
- **Pro** (13 sections) -- with peers and macro context
- **Premium** (22 sections) -- full institutional-grade analysis

Reports are generated via Google Gemini or Anthropic Claude AI for natural language narrative, with a structured template fallback when neither is available. Optional PDF output via pandoc.

---

## Supported Markets ($100T+ Coverage)

### Tier 1 -- Core Markets (10)

| Region | Country | Exchange | Market Cap | PIT Data Source | Macro API | Key Required? |
|--------|---------|----------|-----------|-----------------|-----------|---------------|
| North America | **United States** | NYSE / NASDAQ | $50T | SEC EDGAR | FRED | Email only |
| Europe | **United Kingdom** | LSE | $3.18T | Companies House + ixbrl-parse | FRED (UK series) | Yes (free) |
| Europe | **European Union** | ESEF (pan-EU) | $8-9T | ESEF / XBRL Europe | ECB SDW | No |
| Europe | **France** | Paris / Euronext | $3.13T | ESEF | ECB SDW | No |
| Europe | **Germany** | Frankfurt / XETRA | $2.04T | ESEF | ECB SDW | No |
| Asia | **Japan** | Tokyo (JPX) | $6.5T | J-Quants | FRED (JP series) | Yes (free) |
| Asia | **South Korea** | KOSPI / KOSDAQ | $2.5T | DART | FRED (KR series) | Yes (free) |
| Asia | **Taiwan** | TWSE / TPEX | ~$1.2T | MOPS | FRED (TW series) | No |
| South America | **Brazil** | B3 | $2.2T | CVM | BCB | No |
| South America | **Chile** | Santiago | $0.4T | CMF | FRED (CL series) | No |

### Tier 2 -- Phase 2 Markets (15)

| Region | Country | Exchange | Market Cap | PIT Data Source | Key Required? |
|--------|---------|----------|-----------|-----------------|---------------|
| North America | **Canada** | TSX / TSXV | ~$3T | SEDAR+ | No |
| Oceania | **Australia** | ASX | ~$1.8T | ASX API | No |
| Asia | **India** | BSE / NSE | ~$4T | BSE India | No |
| Asia | **China** | SSE / SZSE | ~$10T | SSE / CSRC | No |
| Asia | **Hong Kong** | HKEX | ~$4.5T | HKEX | No |
| Asia | **Singapore** | SGX | ~$0.6T | SGX | No |
| Latin America | **Mexico** | BMV | ~$0.5T | BMV | No |
| Africa | **South Africa** | JSE | ~$1T | JSE | No |
| Europe | **Switzerland** | SIX | ~$1.8T | SIX | No |
| Europe | **Netherlands** | Euronext Amsterdam | ~$1.2T | ESEF (NL) | No |
| Europe | **Spain** | BME (Madrid) | ~$0.7T | ESEF (ES) | No |
| Europe | **Italy** | Borsa Italiana | ~$0.8T | ESEF (IT) | No |
| Europe | **Sweden** | Nasdaq Stockholm | ~$0.9T | ESEF (SE) | No |
| Middle East | **Saudi Arabia** | Tadawul | ~$2.7T | Tadawul | No |
| Middle East | **UAE** | DFM / ADX | ~$0.8T | DFM / ADX | No |

All data sources are free government APIs. No Bloomberg Terminal or paid data subscriptions required.

---

## OHLCV Price Data Providers

| Provider | Markets | Key? | Package |
|----------|---------|------|---------|
| yfinance | All 25 markets (fallback) | No | yfinance |
| pykrx | South Korea (primary) | No | pykrx |
| akshare | China (primary) | No | akshare |
| jugaad-data | India (primary) | No | jugaad-data |
| twstock | Taiwan (primary) | No | twstock |

---

## Macro Economic Data Providers

| Provider | Countries | Key? | Package |
|----------|-----------|------|---------|
| FRED | US, UK, JP, KR, TW, CL | Yes (free) | fredapi |
| World Bank (wbgapi) | All (fallback) | No | wbgapi |
| BCB | Brazil | No | python-bcb |
| ECB (SDMX) | EU, FR, DE, NL, ES, IT, SE | No | sdmx1 |
| Banxico | Mexico | Yes (free) | banxicoapi |

---

## Test Suite

The project has 1000+ tests covering every wrapper and module:

```bash
# Run all tests (mock-based, no network/API keys needed)
python3 -m pytest tests/ -v

# Run wrapper tests only
python3 -m pytest tests/test_ohlcv_*.py tests/test_macro_*.py tests/test_pit_*.py -v

# Run live integration tests (requires API keys in .env)
python3 -m pytest tests/test_live_wrappers.py -v
```

| Test Category | Files | Tests |
|--------------|-------|-------|
| OHLCV wrappers | 6 files | ~20 |
| Macro wrappers | 11 files | ~40 |
| PIT clients (Tier 1) | 8 files | ~50 |
| PIT clients (Tier 2) | 1 file (batch) | ~30 |
| Equity provider / PIT registry | 2 files | ~30 |
| Canonical translator | 1 file | ~10 |
| LLM wrappers | 4 files | ~25 |
| UK iXBRL integration | 1 file | 9 |
| Live integration | 1 file | ~20 (skipped without keys) |
| Existing tests | 15 files | ~850 |

---

## What's in the Report?

The premium report covers 22 sections:

| # | Section | What it tells you |
|---|---------|------------------|
| 1 | Executive Summary | Key findings and recommendation at a glance |
| 2 | Company Overview | Identity, sector, exchange, currency |
| 3 | Historical Performance | 2-year return, Sharpe ratio, max drawdown, up/down day statistics |
| 4 | Current Financial Snapshot | Actual ratio values organized by tier (cash ratio, debt/equity, margins, P/E) |
| 5 | Financial Health Scoring | Composite 0-100 score with tier breakdown, Altman Z-Score, Beneish M-Score |
| 6 | Survival Mode Analysis | Distress detection, episode history, macro context from central bank data |
| 7 | Linked Variables | Sector performance, competitor health, supply chain risk |
| 8 | Temporal Analysis | Market regime classification, structural breaks, model insights |
| 9 | Predictions & Forecasts | Day/week/month/year forecasts with uncertainty bands |
| 10 | Technical Patterns | Candlestick patterns detected + predicted OHLC chart series |
| 11 | Ethical Filters | Purchasing power, solvency, speculation risk, cash flow quality |
| 12 | Supply Chain Risk | Network topology, contagion probability, concentration risk |
| 13 | Competitive Landscape | Market structure, leadership position, equilibrium shares |
| 14 | Government Protection | Regulatory protection degree, sector strategicness |
| 15 | Model Calibration | Adaptive learning feedback, recalibration history |
| 16 | Market Sentiment | News flow scoring, sentiment trend |
| 17 | Peer Comparison | Percentile ranking against peers |
| 18 | Macroeconomic Environment | GDP/inflation quadrant classification |
| 19 | Advanced Insights | Cycle analysis, tail risk, causality network, sensitivity, deep learning |
| 20 | Risk Factors & Limitations | Data caveats, model assumptions, failed modules |
| 21 | Investment Recommendation | Multi-factor BUY/HOLD/SELL with detailed rationale |
| 22 | Appendix & Methodology | Technical details, tier definitions, data sources |

---

## The Survival Hierarchy

The app uses a 5-tier survival hierarchy that dynamically reweights what matters based on whether a company (or its country) is in financial distress:

| Tier | Category | Normal Weight | Extreme Survival Weight |
|------|----------|--------------|------------------------|
| 1 | Liquidity & Cash | 20% | 60% |
| 2 | Debt & Solvency | 20% | 30% |
| 3 | Market Stability | 20% | 10% |
| 4 | Profitability | 20% | 0% |
| 5 | Growth & Valuation | 20% | 0% |

When a company is in crisis, growth metrics like P/E ratio become irrelevant -- what matters is whether it can make payroll. The system mirrors how credit analysts actually think during a downturn.

Four regimes are supported:
- **Normal** -- equal weights across all tiers
- **Company Survival** -- company in distress, environment stable
- **Modified Survival** -- healthy company in a toxic macro environment
- **Extreme Survival** -- both company and country in crisis

---

## Ethical Filters

Four investment quality filters inspired by Islamic finance principles and universal risk management:

1. **Purchasing Power** -- Does the investment beat inflation? A stock that rises 5% while inflation is 7% actually lost you money.
2. **Solvency** -- Is the company dangerously leveraged? Debt-to-equity above 3.0 means vulnerability to bankruptcy in a recession.
3. **Speculation (Gharar)** -- Is this investing or gambling? Extreme volatility means predictions are unreliable regardless of the model.
4. **Cash is King** -- Does the company generate real cash? "Profit is an opinion, but cash is a fact."

---

## Mathematical Models (25+)

| Category | Models |
|----------|--------|
| Regime Detection | Hidden Markov Model, Gaussian Mixture, PELT structural breaks, Bayesian change point |
| Forecasting | Adaptive Kalman Filter, GARCH volatility, Vector Autoregression, LSTM neural network |
| Deep Learning | Temporal Fusion Transformer (attention-based, multi-horizon) |
| Tree Ensembles | Random Forest, XGBoost, Gradient Boosting |
| Causality | Granger Causality, Transfer Entropy, Copula tail dependency |
| Uncertainty | Conformal Prediction (distribution-free), MC Dropout, Regime-Aware Monte Carlo |
| Explainability | Feature attribution (per-prediction), Sobol global sensitivity |
| Pattern Recognition | Candlestick detector, Fourier/Wavelet cycle decomposition, DTW historical analogs |
| Optimization | Genetic Algorithm (ensemble weight evolution), PID adaptive learning |
| State Estimation | Particle Filter (Sequential Monte Carlo) for non-linear tracking |

Every model has a fallback -- if a dependency is missing or data is insufficient, the pipeline degrades gracefully and continues with the next available method.

---

## Architecture

```
main.py                     # CLI entry point (8-step pipeline)
run.py                      # Interactive terminal launcher
operator1/
  clients/
    pit_registry.py         # Market registry (25 markets)
    pit_base.py             # PITClient protocol (filing_date + report_date)
    equity_provider.py      # Factory: create_pit_client() for any market
    us_edgar.py             # US SEC EDGAR (edgartools + direct requests fallback)
    uk_ch_wrapper.py        # UK Companies House + ixbrl-parse for financial data
    eu_esef_wrapper.py      # EU ESEF/XBRL Europe (filings.xbrl.org)
    jp_jquants_wrapper.py   # Japan J-Quants (replaces EDINET)
    kr_dart_wrapper.py      # South Korea DART (via dart-fss)
    tw_mops_wrapper.py      # Taiwan MOPS scraper
    br_cvm_wrapper.py       # Brazil CVM
    cl_cmf_wrapper.py       # Chile CMF
    ca_sedar.py             # Canada SEDAR+
    au_asx.py               # Australia ASX
    in_bse.py               # India BSE
    cn_sse.py               # China SSE
    hk_hkex.py              # Hong Kong HKEX
    sg_sgx.py               # Singapore SGX
    mx_bmv.py               # Mexico BMV
    za_jse.py               # South Africa JSE
    ch_six.py               # Switzerland SIX
    sa_tadawul.py           # Saudi Arabia Tadawul
    ae_dfm.py               # UAE DFM
    ohlcv_yfinance.py       # OHLCV: yfinance (global fallback)
    ohlcv_pykrx.py          # OHLCV: Korea via pykrx
    ohlcv_akshare.py        # OHLCV: China via akshare
    ohlcv_jugaad.py         # OHLCV: India via jugaad-data
    ohlcv_twstock.py        # OHLCV: Taiwan via twstock
    ohlcv_provider.py       # OHLCV dispatcher (primary + yfinance fallback)
    macro_fredapi.py        # Macro: US FRED
    macro_bcb.py            # Macro: Brazil BCB
    macro_wbgapi.py         # Macro: World Bank (global fallback)
    macro_banxico.py        # Macro: Mexico Banxico
    macro_sdmx.py           # Macro: ECB/EU via SDMX
    macro_ons.py            # Macro: UK via FRED
    macro_estat.py          # Macro: Japan via FRED
    macro_kosis.py          # Macro: Korea via FRED
    macro_dgbas.py          # Macro: Taiwan via FRED
    macro_bcch.py           # Macro: Chile via FRED
    macro_provider.py       # Macro dispatcher (primary + wbgapi fallback)
    canonical_translator.py # Normalizes all filing formats (US GAAP, IFRS, UK GAAP, FRS 102, etc.)
    supplement.py           # OpenFIGI, Euronext, JPX, TWSE enrichment
    gemini.py               # Google Gemini LLM client
    claude.py               # Anthropic Claude LLM client
    llm_base.py             # Abstract LLM base class with retry/rate limiting
    llm_factory.py          # LLM provider factory (auto-detect from keys)
  estimation/
    estimator.py            # Two-pass missing data inference (accounting + ML)
    vae_imputer.py          # Variational Autoencoder imputation option
  features/
    derived_variables.py    # 25+ financial ratios with rolling 4Q TTM
    linked_aggregates.py    # Peer group aggregation with temporal masking
    macro_alignment.py      # Macro indicator alignment to daily cache
    macro_quadrant.py       # GDP/inflation quadrant classification
  analysis/
    survival_mode.py        # Country + company distress detection
    hierarchy_weights.py    # Dynamic tier weight computation
    fuzzy_protection.py     # Government protection scoring
    ethical_filters.py      # 4 investment quality filters
    economic_planes.py      # 5-plane economic sector classification
  models/
    regime_detector.py      # HMM + GMM + PELT + BCP
    forecasting.py          # Kalman + GARCH + VAR + LSTM + trees + baseline
    monte_carlo.py          # Regime-aware simulations with importance sampling
    prediction_aggregator.py# Inverse-RMSE ensemble weighting
    financial_health.py     # 5-tier scoring + Altman Z + Beneish M
    transformer_forecaster.py # Temporal Fusion Transformer
    conformal.py            # Distribution-free prediction intervals
    copula.py               # Tail dependency analysis
    causality.py            # Transfer entropy
    granger_causality.py    # Granger causality + feature pruning
    cycle_decomposition.py  # Fourier + wavelet cycle analysis
    dtw_analogs.py          # Dynamic Time Warping historical analogs
    explainability.py       # SHAP feature attribution
    sensitivity.py          # Sobol global sensitivity
    genetic_optimizer.py    # GA ensemble weight evolution
    particle_filter.py      # Sequential Monte Carlo state estimation
    pattern_detector.py     # Candlestick pattern recognition
    ohlc_predictor.py       # Predicted OHLC candlestick series
    graph_risk.py           # Supply chain network risk
    game_theory.py          # Competitive dynamics analysis
    model_synergies.py      # Cross-model synergies + plane-aware weighting
    regime_mixer.py         # Dual regime classification
    pid_controller.py       # PID adaptive learning
  quality/
    data_quality.py         # Look-ahead bias detection (hard fail on violation)
    data_reconciliation.py  # Field normalization, filing date validation
  report/
    profile_builder.py      # Company profile JSON assembly
    report_generator.py     # 3-tier Markdown/PDF report generation
config/
  global_config.yml         # HTTP settings, rate limits, estimation method
  survival_hierarchy.yml    # 5-tier weights per regime
  economic_planes.yml       # Sector-to-plane mapping
  canonical_fields.yml      # Cross-market field normalization rules
  country_protection_rules.yml  # Government protection thresholds
  country_survival_rules.yml    # Country distress thresholds
```

---

## API Keys

| Key | Purpose | Required? | Cost | Where to get it |
|-----|---------|-----------|------|-----------------|
| `EDGAR_IDENTITY` | US SEC EDGAR User-Agent | Recommended | Free | Just your email address |
| `COMPANIES_HOUSE_API_KEY` | UK Companies House + iXBRL | For UK market | Free | [developer.company-information.service.gov.uk](https://developer.company-information.service.gov.uk/) |
| `JQUANTS_API_KEY` | Japan J-Quants (JPX) | For JP market | Free | [jpx-jquants.com](https://jpx-jquants.com/login) |
| `DART_API_KEY` | South Korea DART | For KR market | Free | [opendart.fss.or.kr](https://opendart.fss.or.kr/) |
| `FRED_API_KEY` | FRED macro data (US, UK, JP, KR, TW, CL) | Recommended | Free | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `GEMINI_API_KEY` | AI report generation (Gemini) | Optional | Free | [ai.google.dev](https://ai.google.dev/) |
| `ANTHROPIC_API_KEY` | AI report generation (Claude) | Optional | Paid | [console.anthropic.com](https://console.anthropic.com/) |
| `BANXICO_API_TOKEN` | Mexico macro data | For MX market | Free | [banxico.org.mx](https://www.banxico.org.mx/SieAPIRest/service/v1/) |

Without any keys, the pipeline works for US (SEC EDGAR needs only an email), EU, Taiwan, Brazil, Chile, and all Phase 2 markets using free APIs. World Bank (wbgapi) provides macro data as a zero-key fallback for all countries.

---

## Running the App

For detailed setup and usage instructions, see [HOW-TO-RUN.md](HOW-TO-RUN.md).

Quick start:

```bash
# Clone and setup
git clone https://github.com/sos0240/github-is-mean.git
cd github-is-mean
python3 -m venv venv && source venv/bin/activate
pip install --timeout 300 -r requirements.txt

# Copy and edit API keys
cp .env.example .env
nano .env  # Add your keys

# Run interactive mode
python3 run.py

# Or direct command
python3 main.py --market us_sec_edgar --company AAPL
```

---

## License

This project is provided as-is for educational and research purposes.

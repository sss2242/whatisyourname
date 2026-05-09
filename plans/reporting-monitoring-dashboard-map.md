# Reporting, Monitoring & Dashboard Map (2026-05-09)

**Changes since 2026-04-17:**
- Report generator now renders Beyond Bands scenario decomposition, ensemble distress, CVaR composite, DuPont decomposition, Kelly sizing, cross-frequency momentum sections.
- HF sub-stage 7.5 split into 7.5.1-7.5.4 for finer timeout control.
- Per-sub-stage data snapshots added to Stage Runner for inspection.
- Signal IC computation vectorized (270s -> 1s, 287x speedup).

Complete map of all modules in the output/presentation layer: report generation (4 modules, ~6,850 lines), monitoring (3 modules, ~3,100 lines), and dashboard (1 module, ~1,850 lines). Covers inputs, outputs, wiring status, and current state.

---

## Section 1: Report Generation (operator1/report/)

The report layer consumes `company_profile.json` from the profile builder and produces Markdown, PDF, HTML, and chart outputs.

### 1.1 Profile Builder -- `profile_builder.py`

**Lines:** 1,276 | **Location:** `operator1/report/profile_builder.py` | **Status:** WORKING

| | Detail |
|---|--------|
| **Entry point** | `build_company_profile()` |
| **Input** | `verified_target` (profile dict), `cache` (daily DataFrame), all model results (regime, forecast, MC, prediction, estimation, graph_risk, game_theory, fuzzy, fh, sentiment, peer_ranking, macro_quadrant) |
| **Output** | `profile` dict (JSON-serializable), persisted to `cache/company_profile.json` |
| **Wired in** | `main.py` Step 7 |

**Profile sections built (26 required + 12 optional):**

| # | Key | Builder Function | Source |
|---|-----|-----------------|--------|
| 1 | `meta` | inline | Pipeline metadata (market_id, pit_source, ohlcv_source, generated_at) |
| 2 | `identity` | `_build_identity_section()` | verified_target profile dict |
| 3 | `current_state` | `_build_current_state_section()` | Latest cache row, 5-tier breakdown |
| 4 | `historical` | `_build_historical_section()` | Time-series summary stats for key variables |
| 5 | `survival` | `_build_survival_section()` | Survival flags, probability, regime, hierarchy weights |
| 6 | `survival_episodes` | `_build_survival_episodes()` | Switch points, durations, mode transitions |
| 7 | `vanity` | `_build_vanity_section()` | Vanity score, label, trend, 5 components |
| 8 | `linked_entities` | `_build_linked_section()` | Linked aggregate DataFrame summary |
| 9 | `regimes` | `_build_regime_section()` | HMM/GMM/PELT regime labels, transitions |
| 10 | `predictions` | `_build_predictions_section()` | Per-variable per-horizon forecasts with uncertainty |
| 11 | `monte_carlo` | `_build_monte_carlo_section()` | Survival probability, terminal values, regime distributions |
| 12 | `model_metrics` | `_build_model_metrics_section()` | Per-model RMSE/MAE, best model per variable |
| 13 | `filters` | `_build_ethical_filters_section()` | 4 ethical filters (purchasing power, solvency, gharar, cash) |
| 14 | `graph_risk` | `_available_dict(graph_risk_result)` | Network centrality, contagion, CoVaR, SRISK |
| 15 | `game_theory` | `_available_dict(game_theory_result)` | Cournot/Stackelberg, competitive pressure |
| 16 | `fuzzy_protection` | `_available_dict(fuzzy_result)` | Fuzzy protection degree, sector score |
| 17 | `pid_controller` | forward_pass_result.pid_summary | PID gains, error history |
| 18 | `financial_health` | `fh_dict` | Composite score, Altman Z, Beneish M, tier scores |
| 19 | `sentiment` | `_build_sentiment_section()` | Sentiment score, articles scored, method |
| 20 | `peer_ranking` | `_build_peer_ranking_section()` | Percentile rank, peer count, composite |
| 21 | `macro_quadrant` | `_build_macro_quadrant_section()` | Quadrant label, stability score |
| 22 | `conflict_risk` | `_build_conflict_risk_profile_section()` | Conflict flags, intensity, sanctions, linked exposure |
| 23 | `data_quality` | `_build_data_quality_section()` | Coverage percentages, look-ahead audit |
| 24 | `estimation` | `_build_estimation_section()` | Per-variable observed vs estimated stats |
| 25 | `failed_modules` | `_build_failed_modules_section()` | List of modules that raised exceptions |
| 26 | `enriched_survival_timeline` | inline (main.py) | Mean intensity, combined state distribution |

**Optional profile keys (injected in main.py Step 7):**

| Key | Source | When present |
|-----|--------|-------------|
| `extended_models` | All Step 6 model results | Always (may be empty dict) |
| `ohlc_predictions` | `format_ohlc_for_profile()` | When OHLC predictor ran |
| `filing_calendar` | inline | When filing calendar analysis succeeded |
| `economic_plane` | `classify_economic_plane()` | Always |
| `corporate_structure` | GLEIF API results | When GLEIF data available |
| `institutional_holders` | `target_holders` list | When holders fetched |
| `institutional_ownership_analysis` | Contagion + flow results | When contagion or flow computed |
| `macro_indicators` | Macro data summary | When macro data fetched |
| `synergies_applied` | `_synergy_meta` dict | When synergies ran |
| `predicted_regime_shifts` | `regime_shift_result.to_dict()` | When regime shift predictor ran |
| `model_diagnostics` | `model_diagnostics_result.to_dict()` | When model diagnostics ran |
| `market_buying_power` | `BuyingPowerResult` | When buying power computed |
| `supply_chain_stress` | `supply_chain_stress_result` | When supply chain stress computed |
| `product_catalysts` | `CatalystResult` | When catalyst detection ran |
| `scenario_analysis` | `scenario_result.to_dict()` | When USS scenario engine ran |
| `product_segments` | Segment data dict from `extract_segment_data()` | When segment extraction succeeded (15 markets, NEW 2026-04-11) |
| `hedge_fund.advanced_methods` | Advanced HF forensic results | When advanced_methods.py ran (NEW 2026-04-03) |
| `hedge_fund.fusion` | Cross-pipeline insight fusion | When fusion.py ran (NEW 2026-04-03) |
| `signal_ic` | `signal_ic_result.to_profile_dict()` | When Signal IC measurement ran |
| `prediction_log` | Prediction log summary | When prediction log fill ran |
| `position_signal` | Position signal (-1 to +1) | When position signal computed |
| `multi_frequency` | `multi_frequency_result.to_profile_dict()` | When multi-frequency pipeline ran |
| `unified_survival_system` | `survival_controller.to_profile_dict()` | When USS controller active |

### 1.2 Profile Schema -- `profile_schema.py`

**Lines:** 121 | **Location:** `operator1/report/profile_schema.py` | **Status:** WORKING

| | Detail |
|---|--------|
| **Entry point** | `validate_profile(profile)` |
| **Input** | Profile dict from `build_company_profile()` |
| **Output** | List of issue strings (empty = valid) |
| **Purpose** | Catch missing keys at runtime before report generation, preventing silent "No data" sections |

**Validates:** 26 required keys + `available` flag on 5 optional sections (graph_risk, game_theory, fuzzy_protection, pid_controller, conflict_risk).

### 1.3 Report Generator -- `report_generator.py`

**Lines:** 4,628 | **Location:** `operator1/report/report_generator.py` | **Status:** WORKING

| | Detail |
|---|--------|
| **Entry points** | `generate_all_reports()` (3-tier), `generate_report()` (single tier) |
| **Input** | `profile` dict, `llm_client` (optional), `cache` DataFrame, `output_dir` |
| **Output** | Dict per tier: `{markdown_path, pdf_path, charts}` |
| **Wired in** | `main.py` Step 8 |

**Report tiers:**

| Tier | Filename | Sections | Description |
|------|----------|----------|-------------|
| Basic | `basic_report.md` | 1, 2, 4, 6, 20 | Quick screening -- 5 sections |
| Pro | `pro_report.md` | 1-7, 11, 14, 16-18, 20 + 75, 195-198 | Peers + macro context -- 18 sections |
| Premium | `premium_report.md` | 1-22 + 75, 195-198 | Full institutional-grade -- 27 sections |

**Report modes:** LEARN (plain-English explanations) vs RESULTS (data-forward, minimal prose).

**22 base sections + 5 extended sections:**

| # | Section | Builder | Profile Key | Tier |
|---|---------|---------|-------------|------|
| 1 | Executive Summary | `_build_executive_summary()` | identity, survival, monte_carlo | All |
| 2 | Company Overview | `_build_company_overview()` | identity | All |
| 3 | Historical Performance | `_build_historical_performance()` | historical | Pro+ |
| 4 | Current Snapshot (Tier-by-Tier) | `_build_current_state_snapshot()` | current_state | All |
| 5 | Financial Health Scoring | `_build_financial_health()` | financial_health | Pro+ |
| 6 | Survival Mode Analysis | `_build_survival_analysis()` | survival, survival_episodes | All |
| 7 | Linked Variables & Market Context | `_build_linked_entities()` | linked_entities | Pro+ |
| 7.5 | Economic Position | `_build_economic_position()` | economic_plane | Pro+ |
| 8 | Temporal Analysis & Model Insights | `_build_regime_analysis()` | regimes | Premium |
| 9 | Predictions & Forecasts | `_build_predictions_forecasts()` | predictions, extended_models | Premium |
| 10 | Technical Patterns & Charts | `_build_technical_patterns()` | patterns | Premium |
| 11 | Ethical Filter Assessment | `_build_ethical_filters()` | filters | Pro+ |
| 12 | Supply Chain & Contagion Risk | `_build_graph_risk()` | graph_risk | Premium |
| 13 | Competitive Landscape | `_build_game_theory()` | game_theory | Premium |
| 14 | Regulatory & Government Protection | `_build_fuzzy_protection()` | fuzzy_protection | Pro+ |
| 15 | Model Calibration | `_build_pid_controller()` | pid_controller | Premium |
| 16 | Market Sentiment & News | `_build_sentiment_analysis()` | sentiment | Pro+ |
| 17 | Peer Comparison | `_build_peer_ranking()` | peer_ranking | Pro+ |
| 18 | Macroeconomic Environment | `_build_macro_quadrant()` | macro_quadrant | Pro+ |
| 19 | Advanced Quantitative Insights | `_build_advanced_insights()` | extended_models | Premium |
| 19.5 | Geopolitical & Conflict Risk | `_build_geopolitical_risk_section()` | conflict_risk | Pro+ |
| 19.6 | SIX Proxy Analysis | `_build_six_proxy_section()` | extended_models (if ch_six) | Pro+ |
| 19.7 | Institutional Holders | `_build_institutional_holders_section()` | institutional_holders | Pro+ |
| 19.8 | Ownership Deep Analysis | `_build_ownership_deep_section()` | institutional_ownership_analysis | Pro+ |
| 19.95 | Unified Survival System | `_build_uss_section()` | unified_survival_system | Premium |
| 19.96 | Scenario Analysis | `_build_scenario_section()` | scenario_analysis | Premium |
| 19.97 | Model Diagnostics | `_build_model_diagnostics_section()` | model_diagnostics | Premium |
| 19.98 | Predicted Regime Shifts | `_build_regime_shifts_section()` | predicted_regime_shifts | Premium |
| 20 | Risk Factors & Limitations | `_build_risk_assessment()` | data_quality, estimation | All |
| 21 | Investment Recommendation | `_build_investment_recommendation()` | All sections | Premium |
| 22 | Appendix & Methodology | `_build_appendix()` | model_metrics, meta | Premium |

**Chart generation (9 charts, Pro+ tiers):**

| # | Chart | Filename | Content |
|---|-------|----------|---------|
| 1 | Price & Volume | `price_volume.png` | OHLCV candlestick with volume bars |
| 2 | Financial Health Timeline | `financial_health.png` | 5-tier score + composite over time |
| 3 | Survival Timeline | `survival_timeline.png` | Mode bands + probability + switches |
| 4 | Regime Detection | `regimes.png` | HMM states + structural breaks |
| 5 | Monte Carlo Fan | `monte_carlo.png` | Simulated paths with p5/p95 |
| 6 | Prediction Fan | `predictions.png` | Multi-horizon forecasts with bands |
| 7 | Hierarchy Weights | `hierarchy_weights.png` | Tier weights over time |
| 8 | Linked Entity Heatmap | `linked_heatmap.png` | Cross-entity correlation |
| 9 | Conflict Risk Dashboard | `conflict_risk.png` | Intensity gauge + status indicators |

**LLM integration:**
- Primary: LLM generates narrative for each section from structured profile data
- Fallback: Template builder functions populate sections from profile data without LLM
- Auto-patching: Missing LLM sections are appended from fallback builders

### 1.4 Enhanced Outputs -- `enhanced_outputs.py`

**Lines:** 762 | **Location:** `operator1/report/enhanced_outputs.py` | **Status:** WORKING

| # | Output | Function | Package | Description |
|---|--------|----------|---------|-------------|
| 1 | Branded PDF | `generate_fpdf2_pdf()` | `fpdf2` | Pure-Python PDF with cover page, TOC, embedded charts. No pandoc needed. |
| 2 | Candlestick Charts | `generate_mplfinance_charts()` | `mplfinance` | Professional OHLCV charts with volume, moving averages, regime color bands |
| 3 | Performance Tearsheet | `generate_quantstats_tearsheet()` | `quantstats` | 40+ metrics: Sharpe, Sortino, max drawdown, rolling returns, underwater plot |
| 4 | Interactive Dashboard | `generate_plotly_dashboard()` | `plotly` | Zoomable HTML with candlestick, financial health, survival timeline, regime overlay |

**Brand palette:** Background `#1c1b22`, Foreground `#eae7e1`, Accent `#6d4aff`, Red `#dc3545`, Green `#1ea885`, Gold `#e8950a`.

---

## Section 2: Monitoring (operator1/monitoring/)

The monitoring layer checks all 25 PIT market wrappers and OHLCV providers, detects breakage, and routes traffic to fallback paths.

### 2.1 Health Check -- `health_check.py`

**Lines:** 1,073 | **Location:** `operator1/monitoring/health_check.py` | **Status:** WORKING

| | Detail |
|---|--------|
| **Entry points** | `run_health_check()`, `check_market_health()`, `preflight_check()` |
| **CLI** | `python -m operator1.monitoring.health_check [--market X] [--quick] [--level L0-L5] [--json]` |
| **Output** | `HealthReport` -> persisted to `cache/wrapper_health.json` + `cache/wrapper_health_history.jsonl` |

**Probe levels (6 levels):**

| Level | Name | API Calls | Tests |
|-------|------|-----------|-------|
| L0 | DNS/TLS Connectivity | 0 | TCP socket connect to API host |
| L1 | Company Search | 1 | `search_company()` for canary company |
| L2 | Profile Fetch | 1 | `get_profile()` for canary company |
| L3 | Financial Data | 1 | `get_balance_sheet()` or `get_income_statement()` |
| L5 | Holder Data | 1 | `get_holders()` for canary company (newest, most fragile) |

**Canary companies (1 per market):**

| Market | Canary | Identifier |
|--------|--------|-----------|
| us_sec_edgar | Apple | AAPL |
| uk_companies_house | Unilever | Unilever |
| jp_jquants | Toyota | 7203 |
| kr_dart | Samsung | 005930 |
| tw_mops | TSMC | 2330 |
| br_cvm | Petrobras | PETR4 |
| in_bse | Reliance | 500325 |
| cn_sse | Moutai | 600519 |
| hk_hkex | Tencent | 00700 |
| sg_sgx | DBS | D05 |
| sa_tadawul | Aramco | 2222 |
| ch_six | Nestle | NESN |
| ... | ... | ... |

**Status classification:**

| Status | Meaning | Routing |
|--------|---------|---------|
| healthy | Primary path works at L3+ | Use primary |
| degraded | Primary broken, fallback works | Use fallback |
| critical | All paths failing | Warn user |
| unknown | Not yet checked | Try primary |

**OHLCV provider probes (4 providers, parallel):**

| Provider | Canary | Package |
|----------|--------|---------|
| yfinance | AAPL | `yfinance` |
| baostock | sh.600519 | `baostock` |
| pykrx | 005930 | `pykrx` |
| twstock | 2330 | `twstock` |

**Latency degradation detection:**
- Reads JSONL history file for per-market latency_ms
- Alerts when recent p95 latency exceeds 2x historical median
- Requires 5+ history entries before alerting

**Auto-healing functions:**

| Function | Purpose |
|----------|---------|
| `get_active_path(market_id)` | Returns "primary", "fallback", or "none" based on health data |
| `is_market_healthy(market_id)` | Quick boolean check |
| `get_degraded_markets()` | List of markets in degraded/critical state |
| `preflight_check(market_id)` | Pre-pipeline validation with L1 fallback |

**Alerting:**
- Detects status transitions between health checks
- Logs warnings (degraded) or errors (critical) with reasons
- Optional webhook via `HEALTH_WEBHOOK_URL` environment variable
- Recovery detection (critical -> healthy logged as INFO)

**Execution:** All 25 markets probed in parallel via `ThreadPoolExecutor` (8 workers max).

### 2.2 Wrapper Probes -- `wrapper_probes.py`

**Lines:** 1,147 | **Location:** `operator1/monitoring/wrapper_probes.py` | **Status:** WORKING

| | Detail |
|---|--------|
| **Entry points** | `run_deep_probes()`, `run_deep_probe(market_id)` |
| **Output** | `WrapperProbeResult` per market (status, pattern, steps, schema_drift) |
| **Execution** | Parallel via `ThreadPoolExecutor` (8 workers max) |

**9 probe types:**

| # | Probe | Function | When Used |
|---|-------|----------|-----------|
| 1 | DNS Resolution | `_probe_dns()` | All markets (detect domain migrations) |
| 2 | TLS Certificate | `_probe_tls()` | All HTTPS markets (detect cert expiry/CA changes) |
| 3 | HTTP GET | `_probe_http()` | All markets (basic connectivity) |
| 4 | WAF Detection | `_probe_waf()` | WAF markets: tw_mops, sa_tadawul (plain vs curl_cffi) |
| 5 | Session Lifecycle | `_probe_session()` | Session markets: hk_hkex, mx_bmv (cookie + API flow) |
| 6 | Schema Drift | `_probe_schema()` | Markets with JSON APIs (detect structural changes vs baseline) |
| 7 | Referer Gate | `_probe_referer()` | BSE India (Referer header required) |
| 8 | Token Lifecycle | `_probe_token()` | BMV Mexico (WSO2 token acquisition + authenticated call) |
| 9 | Content Validation | `_probe_content_validation()` | All markets (detect WAF pages, CAPTCHAs, empty bodies in 200 OK) |
| 10 | Date Window | `_probe_date_window()` | HKEX (detect narrowed date range restrictions) |
| 11 | HTTP POST | `_probe_post()` | SEDAR Canada (Catalyst form submission) |

**Content rejection patterns (8 regex):**
- WAF challenge pages (Akamai, Cloudflare)
- CAPTCHA (hCaptcha, reCAPTCHA)
- JavaScript-required pages
- API deprecation notices
- Rate limit/forbidden error bodies
- Service unavailable/maintenance

**Per-market probe configurations (25 markets):**

| Market | Pattern | Probes | Status |
|--------|---------|--------|--------|
| us_sec_edgar | free_api | DNS, TLS, HTTP, Schema | Rich |
| uk_companies_house | api_key | DNS, TLS | Minimal (needs key) |
| eu_esef | free_api | DNS, HTTP, Schema | Rich |
| fr_esef | free_api | HTTP | Light |
| de_esef | free_api | HTTP | Light |
| jp_jquants | api_key | DNS, TLS | Minimal (needs key) |
| kr_dart | api_key | DNS, HTTP | Light |
| tw_mops | waf | DNS, WAF | Pattern-specific |
| br_cvm | free_api | DNS, HTTP | Light |
| cl_cmf | free_api | DNS, HTTP | Light |
| hk_hkex | session | DNS, TLS, Session, DateWindow | Rich |
| sg_sgx | free_api | DNS, HTTP, Schema | Rich |
| sa_tadawul | waf | DNS, TLS, WAF | Pattern-specific |
| in_bse | referer | DNS, Referer | Pattern-specific |
| cn_sse | free_api | DNS, HTTP, ContentValidation | Upgraded |
| ca_sedar | session | DNS, HTTP, POST | Upgraded |
| au_asx | free_api | DNS, HTTP | Light |
| za_jse | free_api | DNS, HTTP, ContentValidation | Upgraded |
| mx_bmv | session | DNS, HTTP, Token, Schema | Rich (upgraded) |
| ae_dfm | free_api | DNS, HTTP, ContentValidation x2 | Upgraded |
| ch_six | free_api | DNS, HTTP, Schema, ContentValidation | Rich (upgraded) |
| nl/es/it/se_esef | free_api | HTTP | Light (ESEF shared) |

**Schema baselines:** Stored in `config/probe_baselines/{market_id}.json`. First probe creates baseline; subsequent probes diff against it for drift detection.

**Deep probe status values:** working, restructured, down, geo_blocked, waf_blocked, rate_limited, auth_changed, partial.

### 2.3 Model Diagnostics -- `model_diagnostics.py`

**Lines:** 875 | **Location:** `operator1/monitoring/model_diagnostics.py` | **Status:** WORKING

| | Detail |
|---|--------|
| **Entry point** | `compute_model_diagnostics()` |
| **Input** | `cache`, all temporal model results (forecast, MC, copula, granger, cycle, DTW, conformal) |
| **Output** | `ModelDiagnosticsResult` -> stored in `profile["model_diagnostics"]` |
| **Wired in** | `main.py` Step 6.6 |

**Purpose:** For each of 10 models, pre-computes what it SHOULD produce based on data characteristics (series length, stationarity, variable count), then compares against actual output. Produces per-model robustness ratings.

**Models assessed (10):** Kalman, GARCH, VAR, LSTM, Tree, Monte Carlo, Copula, Granger, Cycle, DTW, Conformal.

**Robustness ratings:**

| Rating | Meaning |
|--------|---------|
| on_track | Model output matches expected behavior for the data |
| degraded | Model ran but results are questionable (e.g., poor convergence) |
| failed | Model did not produce usable output |

**Overall robustness:** Fraction of models rated `on_track`. Used by report generator for confidence disclaimers.

**Report integration:** Rendered in Premium report section 19.97 (Model Diagnostics).

---

## Section 3: Dashboard (dashboard.py)

The NiceGUI desktop dashboard provides a visual interface for running analyses, viewing reports, monitoring health, and configuring the pipeline.

### 3.1 Dashboard -- `dashboard.py`

**Lines:** 1,853 | **Location:** `dashboard.py` | **Status:** WORKING

| | Detail |
|---|--------|
| **Framework** | NiceGUI (Quasar/Vue3 + Python) |
| **Port** | 8080 |
| **Theme** | Proton.me-inspired Monokai dark (deep navy-purple + vibrant accents) |
| **Entry point** | `python dashboard.py` |

**Theme palette (Proton.me Monokai):**

| Role | Color | Hex |
|------|-------|-----|
| Background | Deep navy-purple | `#1a1a2e` |
| Surface | Dark slate | `#16213e` |
| Card | Muted indigo | `#0f3460` |
| Primary | Proton purple | `#6c5ce7` |
| Secondary | Light purple | `#a29bfe` |
| Success | Proton green | `#00b894` |
| Warning | Warm amber | `#fdcb6e` |
| Error | Soft coral | `#e17055` |
| Text primary | Off-white | `#dfe6e9` |
| Text secondary | Muted gray | `#b2bec3` |
| Accent | Monokai pink | `#e84393` |

**5 pages:**

| Page | Route | Content |
|------|-------|---------|
| Home | default | Last analysis summary or welcome screen + history table + **all 20 profile outputs displayed** (NEW 2026-04-03) |
| New Analysis | "analyze" | LLM provider, region/market/company selection, pipeline options, run button with live log + **all 7 advanced pipeline options** (NEW 2026-04-03) |
| Report | "report" | Tabbed view: Summary (**key results + advanced analytics**, NEW 2026-04-03), Interactive (plotly), Charts (ECharts), Full Report (markdown), Tearsheet |
| Health | "health" | System health: Tier 1/2 market cards, dependency status, run full check button |
| Config | "config" | API keys status, cache management, about + **Scoring Weights panel** (Panel A: tweakable inputs, Panel B: computed outputs, NEW 2026-04-03) |

**Dashboard components:**

| Component | Source Concept | Implementation |
|-----------|---------------|----------------|
| Splash screen | Loading bar | Dependency check with per-stage progress (Stage 1-4) |
| Candlestick chart | DearPyGui GPU chart | ECharts candlestick + volume bars (120 trading days) |
| Radar chart | DearPyGui polar plot | ECharts 5-tier financial health radar |
| Meter gauges | ttkbootstrap Meter | ECharts gauge series (health score, survival prob, Z-score) |
| Status bar | Textual Footer | Always-visible: Python version, market status, LLM count |
| Command palette | Textual Ctrl+P | Ctrl+K search: actions, markets, companies |
| Toast notifications | ttkbootstrap Toast | NiceGUI notify with auto-dismiss (3-8s by type) |
| Health cards | Custom | Per-market expansion cards with deep probe details, pattern badges, schema drift indicators |

**State management:**
- `DashboardState` persisted to `cache/dashboard_state.json` (LLM provider, region, market, company, options)
- Analysis history persisted to `cache/analysis_history.json` (last 50 runs)
- API keys loaded from `.env` + environment variables

**Pipeline execution:**
- Runs `main.py` as subprocess with `asyncio.create_subprocess_exec()`
- Streams stdout to live log area
- Tracks step progress (0-100%) from "Step" keywords in output
- Records history entry on completion

---

## Summary

| Layer | Modules | Lines | Key Outputs |
|-------|---------|-------|-------------|
| Report Generation | 4 | ~6,850 | Markdown (3 tiers), PDF (pandoc + fpdf2), charts (matplotlib + mplfinance), interactive HTML (plotly), tearsheet (quantstats) |
| Monitoring | 3 | ~3,100 | Health JSON, history JSONL, 6 probe levels, 11 probe types, OHLCV probes, latency trending, model diagnostics, parallel execution |
| Dashboard | 1 | ~1,853 | NiceGUI desktop app, 5 pages, ECharts, Proton.me Monokai theme, command palette, live pipeline execution |
| **Total** | **8** | **~11,800** | |

# Analysis Models Map (2026-05-09)

Comprehensive reference for all 78+ analytical modules in Operator 1. Each module documented with purpose, mathematical basis, inputs, outputs, pipeline wiring, profile/report integration, dependencies, and current status.

Total: ~118,000 lines of Python code across 6 layers + orchestration.

**Changes since 2026-05-02:**
- **Frequency-first pipeline (Stage 2):** Multi-frequency pipeline moved BEFORE temporal models. Each frequency (A/Q/S/M/W/D) runs its own full pipeline. Fusion forward-fills correct Q/A ratios into daily cache. Fixes distorted PE/EV/fcf_yield from flow-variable interpolation.
- **Frequency-aware formulas:** `derived_variables.py`, `survival_mode.py`, `financial_health.py` accept `freq` parameter. `freq_constants.py` provides `steps_per_year(freq)` replacing hardcoded 252.
- **Beyond Bands distributional forecasting:** 6 methods (DRF, QRF, CQR, NGBoost, GAMLSS, BNN) added to forecasting cascade via `fit_distributional()`.
- **HF 30-method upgrade:** Expanded from 15 to 30+ methods in 9 phases. Split sub-stage 7.5 into 7.5.1-7.5.4. Multi-frequency HF metrics. Data sufficiency validation.
- **Confidence band inversion fix:** 6 expert uncertainty methods (PID calibrator, IV blend, width caps, skew-aware widening, regime scaling, event proximity).
- **Signal IC 287x speedup:** Vectorized Spearman IC.
- **shares_outstanding/market_cap injection:** From profile dict into cache.
- **Frequency separator upgrade:** Bayesian detection + Chow-Lin + Denton disaggregation.
- **4 cascading backtest bug fixes (2026-05-09):** MC burn-out contamination filter, sector-aware MC survival thresholds, OHLC sqrt noise growth, sector-aware FH baseline floors.
- **30+ deep scan bug fixes:** Forward pass KeyError, pickle serialization, CopulaResult mismatches, profile defaults, balance field gaps, NameError bugs.

---

## Layer 1: Feature Engineering

21 modules in `operator1/features/` that transform the raw daily cache into model-ready features. These run in Steps 4-5 of main.py before any temporal modeling. **Updated 2026-05-01:** expanded from 14 to 21 modules with 7 new feature modules (options signals, cross-asset signals, event calendar, behavioral signals, complexity signals, feature normalization, operational efficiency).

---

### 1.1 Derived Variables

**File:** `operator1/features/derived_variables.py` (~1,468 lines)
**Pipeline step:** Step 5
**Profile key:** `current_state` (latest values of all derived columns)

**Purpose:** Computes ~96 derived financial and technical columns across 24 computation stages from the raw cache. This is the primary feature engineering module -- every downstream model consumes its output. **Expanded 2026-05-01** from ~50 to ~96 variables with 5 new computation stages: microstructure signals (Corwin-Schultz spread, Kyle lambda, Parkinson/Yang-Zhang vol), stationarity features (fractional differentiation, Hurst exponent, autocorrelation, momentum 12-1), credit risk (cash burn rate, debt maturity pressure, CCC, covenant proximity), tail risk (skewness, kurtosis, tail ratio, vol-of-vol), forensic accounting (revenue-receivables divergence, capex-depreciation ratio, soft asset ratio, OCF ratio).

**Mathematical operations:**

- **Returns:** `return_1d = close.pct_change()`, `log_return_1d = ln(close/close.shift(1))`, `return_5d`, `return_21d`
- **Volatility:** `volatility_21d = return_1d.rolling(21).std()`, `volatility_63d`
- **Drawdown:** `drawdown_252d = (close - close.rolling(252).max()) / close.rolling(252).max()`
- **Solvency ratios:** `current_ratio = current_assets / current_liabilities`, `debt_to_equity_abs = total_debt / |total_equity|`, `net_debt = total_debt - cash_and_equivalents`, `interest_coverage = ebit / interest_expense`
- **Liquidity:** `cash_ratio = cash_and_equivalents / current_liabilities`
- **Profitability:** `gross_margin`, `operating_margin`, `net_margin`, `ebitda_margin`
- **Cash flow:** `free_cash_flow = operating_cash_flow - |capex|`, `fcf_yield = fcf / market_cap`
- **Valuation:** `pe_ratio_calc = close / (eps_diluted or eps)`, `ev_to_ebitda`
- **TTM computations:** Rolling 4-quarter sums for `revenue_ttm`, `net_income_ttm`, `operating_cash_flow_ttm`
- **Technical indicators (via `ta` library):** ADX (trend strength), OBV (on-balance volume), Bollinger Band width, MACD histogram
- **Beta:** `beta_252d` = 252-day rolling beta vs market benchmark index (per-market index from `config/market_benchmarks.yml`; benchmark returns fetched via `ohlcv_provider.fetch_benchmark_returns()`)
- **Debt serviceability:** `debt_service_coverage = operating_cash_flow / (interest_expense + current_portion_ltd)`

All division operations use `safe_ratio()` which returns NaN for zero/tiny denominators and sets `is_missing_{var}` and `invalid_math_{var}` companion flags.

**Input:** Cache DataFrame with `close`, financial statement columns (revenue, total_assets, etc.)
**Output:** Cache + ~50 new columns + `is_missing_*` + `invalid_math_*` flags
**Dependencies:** `ta` (technical indicators), `pandas`, `numpy`

---

### 1.2 Conflict Risk Assessment

**File:** `operator1/features/conflict_risk.py` (1,111 lines)
**Pipeline step:** Step 4a.3
**Profile key:** `conflict_risk`

**Purpose:** Assesses geopolitical and conflict risk for the target company's country and linked entities. Produces binary flags and continuous intensity scores used by survival mode detection.

**Data sources (3 layers):**
1. **Static lists** (always available): World Bank Fragile and Conflict-affected States (31 countries), OFAC/EU sanctions (13 countries), active wars (9 countries)
2. **UCDP GED API** (free, auth required since 2025): Uppsala Conflict Data Program armed conflict events with geolocation and fatality counts
3. **GDELT** (free, rate-limited to 1 request per 5 seconds): Global Database of Events, Language, and Tone for real-time news conflict mentions and sentiment tone

**Scoring formula:**
```
conflict_intensity = 0.40 * event_score + 0.20 * fatality_score + 0.25 * flag_score + 0.15 * news_score
```

**Linked entity conflict propagation** (`assess_linked_entity_conflict`): For each linked entity group (suppliers, customers, competitors, financial institutions), checks if any entity's country appears in conflict lists. Produces:
- `supply_chain_risk_score` (0-1): suppliers/logistics in war zone
- `revenue_exposure_score` (0-1): customers in conflict regions
- `competitive_advantage_score` (0-1): competitors in conflict (positive for target)

**Input:** `country_iso2` (ISO-2 code), optional `company_name`
**Output:** `ConflictRiskResult` with 12 fields, injected into cache as daily columns
**Downstream consumers:** `survival_mode.py` (conflict_flag + sanctions_flag are survival triggers), `profile_builder.py`, `report_generator.py` (Section 19.5)

---

### 1.3 Filing Calendar

**File:** `operator1/features/filing_calendar.py` (385 lines)
**Pipeline step:** Step 4c
**Profile key:** `filing_calendar`

**Purpose:** Analyzes the company's filing pattern to detect frequency, coverage gaps, and data staleness. Critical for the adaptive windows module which anchors all rolling windows to filing frequency.

**Detection logic:**
- Extracts unique filing dates from cache columns that have `report_date` or `filing_date`
- Computes median gap between consecutive filings
- Classifies: <120 days = quarterly, 120-250 = semi-annual, 250-500 = annual, >500 = unknown
- Counts expected vs actual filings in 2-year window
- Computes coverage ratio and identifies gaps (missing expected filings)

**Staleness detection:** `is_stale = latest_filing_age_days > stale_threshold_days`, where the threshold is market-specific (90 days for quarterly filers, 200 for semi-annual, 400 for annual).

**Filing freshness injection** (`inject_filing_freshness`): Adds a `filing_freshness` column to cache -- a 0-1 score that decays with distance from the nearest filing date.

**Predicted next filing date** (`predict_next_filing_date`): Extrapolates next expected filing date from detected filing frequency and last filing date. Uses median inter-filing gap + market-specific calendar adjustments. Stored in `profile["filing_calendar"]["next_expected_filing"]`.

**Input:** Cache DataFrame, `market_id` string
**Output:** `FilingCalendarResult` dataclass with frequency, coverage, staleness, gaps, predicted next filing
**Downstream consumers:** `adaptive_windows.py` (Nyquist-anchored window sizes), `triage_card.py` (staleness warnings), `report_generator.py` (next filing date in report)

---

### 1.4 Linked Aggregates

**File:** `operator1/features/linked_aggregates.py` (359 lines)
**Pipeline step:** Step 5g
**Profile key:** `linked_entities`

**Purpose:** Computes cross-entity aggregate statistics from linked entity caches (competitors, suppliers, customers, financial institutions). These become features for temporal models to learn cross-entity dynamics.

**Variables aggregated:** `close`, `return_1d`, `volatility_21d`, `revenue`, `net_income`, `total_debt`, `free_cash_flow`, `fcf_yield`, `gross_margin`, `pe_ratio_calc`

**Aggregation methods:** Mean and median per entity group, producing columns like `competitors_avg_return_1d`, `suppliers_median_volatility_21d`, `customers_avg_gross_margin`.

**Relative metrics** (`compute_relative_metrics`): Computes target vs sector:
- `rel_strength_vs_sector`: target return / sector avg return
- `valuation_premium`: target PE / sector avg PE
- `rel_volatility`: target vol / sector avg vol

**Input:** Target cache, dict of linked entity caches, entity groups dict
**Output:** `linked_agg_df` DataFrame merged into main cache
**Downstream consumers:** Temporal models (via `_extra_vars`), `prediction_aggregator.py`

---

### 1.5 Macro Alignment

**File:** `operator1/features/macro_alignment.py` (322 lines)
**Pipeline step:** Consumed via `macro_quadrant.py`
**Profile key:** `macro_indicators`

**Purpose:** Aligns yearly/quarterly macro indicators (GDP, inflation, interest rates, unemployment, currency) to daily frequency using as-of logic. Handles the frequency mismatch between macro data (annual/quarterly) and the daily cache.

**Alignment method:** For each day, finds the latest available macro observation where `observation_year <= day.year`. Forward-fills between observations.

**Derived computations:**
- `inflation_rate_daily_equivalent = inflation_rate_yoy / 365`
- `real_return_1d = return_1d - inflation_rate_daily_equivalent`
- All variables get `is_missing_*` companion columns

**Input:** `MacroDataset` from macro_mapping, daily DatetimeIndex
**Output:** Daily-frequency macro columns added to cache

---

### 1.6 Macro Quadrant

**File:** `operator1/features/macro_quadrant.py` (248 lines)
**Pipeline step:** Step 4a
**Profile key:** `macro_quadrant`

**Purpose:** Classifies the macroeconomic environment into one of four quadrants based on GDP growth trend and inflation trend. Used for regime-aware model weighting.

**Quadrant classification:**
| GDP Growth | Inflation | Quadrant |
|---|---|---|
| Rising | Rising | Overheating |
| Rising | Falling | Goldilocks |
| Falling | Rising | Stagflation |
| Falling | Falling | Recession |

**Stability score:** Rolling 63-day fraction of days in the same quadrant (1.0 = fully stable macro environment).

**Input:** Cache DataFrame, `MacroDataset`
**Output:** Cache + `macro_quadrant` column, `MacroQuadrantResult` with `latest_quadrant` and `stability_score`

---

### 1.7 News Sentiment

**File:** `operator1/features/news_sentiment.py` (563 lines)
**Pipeline step:** Step 5i
**Profile key:** `sentiment`

**Purpose:** Fetches recent news articles for the target company and scores sentiment. Provides a short-term signal that captures market perception and news-driven price movements.

**Three-tier scoring hierarchy:**
1. **LLM scoring** (if API key available): Sends article titles + snippets to Gemini/Claude/OpenRouter with structured prompt asking for -1 to +1 sentiment score + reasoning
2. **VADER scoring** (fallback): `vaderSentiment` library -- rule-based sentiment analysis that handles negation ("not good" = negative), intensifiers ("very good" > "good"), and contextual valence shifts. Compound score normalized to -1 to +1
3. **Keyword scoring** (final fallback): Simple positive/negative word count ratio

**News sources:** `gnews` library (Google News scraper, no API key) + `feedparser` (RSS feeds from financial news sources)

**Output columns:** `sentiment_score` (daily, -1 to +1), `sentiment_label` (Bullish/Neutral/Bearish), `sentiment_momentum` (5-day rolling change)

**Input:** Cache, LLM client (optional), symbol, market_id, company_name
**Output:** Cache + sentiment columns, `SentimentResult` with n_articles, scoring method, mean/latest sentiment

---

### 1.8 Peer Ranking

**File:** `operator1/features/peer_ranking.py` (270 lines)
**Pipeline step:** Step 5h
**Profile key:** `peer_ranking`

**Purpose:** Computes cross-sectional percentile rank of the target company vs its linked entity peers for key financial variables. Transforms absolute values into relative positioning.

**Variables ranked:** `close`, `return_1d`, `volatility_21d`, `revenue`, `net_income`, `total_debt`, `free_cash_flow`, `fcf_yield`, `gross_margin`, `pe_ratio_calc`

**Method:** For each variable, collects the latest value from target + all peer caches, computes the percentile rank of the target within the peer group. Output: `peer_rank_{var}` columns (0-100 percentile) + composite rank.

**Composite rank:** Weighted average of individual variable ranks (equal weights).

**Labels:** Top Quartile (>75th), Above Median (50-75th), Below Median (25-50th), Bottom Quartile (<25th).

**Input:** Cache, linked entity caches dict
**Output:** Cache + `peer_*` columns, `PeerRankingResult` with n_peers, variable_ranks, composite_rank, label

---

### 1.9 Private Company Proxies

**File:** `operator1/features/private_company_proxies.py` (451 lines)
**Pipeline step:** Step 5a
**Profile key:** `meta.is_private_company`

**Purpose:** When no OHLCV data is available (private companies, some Tier 2 markets), generates proxy variables that stand in for price-based features. Enables the full pipeline to run without market data.

**Proxy mappings:**
| Standard Column | Proxy Column | Derivation |
|---|---|---|
| `close` | `equity_value` | `total_equity` (latest) |
| `return_1d` | `equity_change_rate` | `total_equity.pct_change()` |
| `volatility_21d` | `financial_volatility` | `net_income.rolling(21).std() / revenue` |
| `return_5d` | `revenue_momentum_5d` | `revenue.pct_change(5)` |
| `drawdown_252d` | `equity_drawdown` | Max drawdown of equity_value |

**Transparent resolution** (`resolve_proxies`): After computing proxies, writes their values INTO the standard column names (close, return_1d, etc.) so all downstream models work without code changes. The proxy origin is tracked via `_proxy_source_{var}` flags.

**Input:** Cache without OHLCV data
**Output:** Cache with proxy columns + standard columns populated

---

### 1.10 Market Buying Power

**File:** `operator1/features/market_buying_power.py` (366 lines)
**Pipeline step:** Step 4a.4
**Profile key:** `market_buying_power`

**Purpose:** Estimates demand-side economic signals for the company's sector and country. Captures whether consumers/businesses can afford to buy the company's products.

**Components:**
- `buying_power_index`: Composite of real GDP per capita, inflation-adjusted revenue growth, sector demand momentum
- `sector_demand_momentum`: PCE (Personal Consumption Expenditure) growth rate for the relevant sector
- `real_revenue_growth_ppp`: Revenue growth adjusted for purchasing power parity
- `demand_risk_flag`: True when buying power is declining and sector momentum is negative
- `consumer_confidence_trend`: Direction of consumer sentiment from macro data
- `inflation_drag`: Revenue growth erosion from inflation

**Input:** Cache, sector string, country_iso2, macro_data dict
**Output:** Cache + buying power columns, `BuyingPowerResult`

---

### 1.11 SIX Derived Proxies (Switzerland Only)

**File:** `operator1/features/six_derived_proxies.py` (3,848 lines)
**Pipeline step:** Step 4a.5 (only for `ch_six` market)
**Profile key:** `extended_models` (if ch_six)

**Purpose:** The most complex feature module. Derives 22 canonical financial fields from SIX Swiss Exchange's dividend history, capital structure data, and corporate notices -- without any financial statement data. This enables full pipeline execution for Swiss companies where no structured XBRL filings are publicly available.

**Mathematical methods (7 algorithms):**
1. **Kalman Filter** (Harvey 1989): Estimates latent earnings trajectory from noisy dividend observations using a local-level state-space model
2. **PELT Regime Detection** (Killick 2012): Identifies structural breaks in dividend/capital trajectories to detect policy changes
3. **UKF Two-Factor Earnings**: Unscented Kalman Filter with two latent factors (core earnings + cyclical component) for non-linear earnings dynamics
4. **Merton Structural Debt Model** (1974): Estimates total debt and default probability from equity value + equity volatility using the Black-Scholes-Merton framework
5. **L1 Balance Sheet Reconstruction** (Candes & Tao 2005): Compressed sensing to reconstruct sparse balance sheet from limited observations (dividends, buybacks, capital changes)
6. **EBO Reverse Earnings** (Edwards-Bell-Ohlson): Backs out implied earnings from market price using residual income valuation
7. **Ohlson Residual Income**: Estimates equity from earnings + book value persistence model

**TDA (Topological Data Analysis):** Persistent homology via `ripser` (Bauer 2021) on dividend trajectories to detect monotonicity (stable payer), cyclicality (boom-bust), and structural breaks.

**Input:** Cache, target profile (for `ch_six` market only)
**Output:** 22 canonical fields (revenue, total_assets, eps, operating_cash_flow, etc.) seeded into cache
**Dependencies:** `ripser` (persistent homology), `scipy` (optimization), `numpy`

---

### 1.12 Product Catalysts

**File:** `operator1/features/product_catalysts.py` (338 lines)
**Pipeline step:** Step 5i.5
**Profile key:** `product_catalysts`

**Purpose:** Detects forward-looking catalyst signals from R&D spending patterns, earnings acceleration, revenue diversification changes, and news article keywords.

**Components:**
- `rnd_acceleration`: Rate of change of R&D spending relative to revenue (accelerating R&D = potential product pipeline)
- `earnings_momentum`: Sequential EPS acceleration (2nd derivative of earnings)
- `revenue_diversification_delta`: Change in revenue concentration (Herfindahl-like measure)
- `news_catalyst_score`: Frequency of catalyst keywords (launch, FDA, approval, patent, partnership) in recent news
- `catalyst_type`: Classification (product_launch, regulatory, partnership, organic_growth, none)

**Composite score:** `catalyst_score = weighted_sum(rnd, earnings, diversification, news)`

**Input:** Cache, profile dict, optional news articles from sentiment module
**Output:** Cache + catalyst columns, `CatalystResult`

---

### 1.13 Product Metrics (NEW -- 2026-04-11)

**File:** `operator1/features/product_metrics.py` (289 lines)
**Pipeline step:** Step 5i.6
**Profile key:** `product_segments`

**Purpose:** Computes quantitative product-level metrics from segment revenue data extracted by per-market `extract_segment_data()` methods. These metrics feed into temporal models as extra variables and into Monte Carlo for concentration risk assessment.

**Metrics computed:**
- **segment_hhi**: Herfindahl-Hirschman Index of revenue concentration across segments (0-1, higher = more concentrated)
- **cannibalization_rate**: Revenue shift between segments (negative = segments cannibalizing each other)
- **network_effect_score**: Revenue acceleration from user/customer network effects
- **input_cost_pressure**: Cost growth rate relative to revenue growth
- **growth_runway_quarters**: Estimated quarters before dominant segment matures
- **maturity_concentration**: Fraction of revenue from mature (low-growth) segments
- **estimated_market_share**: Inferred from segment revenue vs industry benchmarks
- **dominant_segment_growth**: YoY growth rate of the largest revenue segment
- **net_new_revenue_pct**: Revenue from segments that didn't exist N quarters ago

**Segment extraction coverage:** 15 markets have `extract_segment_data()` implementations: US (XBRL + 10-K text), EU/NL/ES/IT/SE (ESEF XBRL IFRS 8), UK (ESEF crossover + PDF + docTR OCR), KR (DART XBRL), JP (IRBank + SEC EDGAR ADR), CN (akshare/EastMoney), TW (doc.twse.com.tw + SEC EDGAR ADR), CL (SEC EDGAR ADR), BR (DFP PDF + fuzzy parser), SA (Tadawul XBRL + PDF), CH (3 paths), ZA (SENS PDF), MX (BMV XBRL IFRS 8).

**Input:** Cache DataFrame, segment data dict from `extract_segment_data()`
**Output:** Cache + 9 product metric columns
**Downstream consumers:** Temporal models (via `_extra_vars`), Monte Carlo (`segment_hhi` for concentration risk flag), OHLC predictor, profile builder

---

### 1.14 OCR Pipeline for Image-Based PDFs (NEW -- 2026-04-13)

**File:** `operator1/clients/fuzzy_pdf_parser.py` (OCR integration within existing module)
**Pipeline step:** Called during filing extraction for image-based PDFs
**Profile key:** N/A (transparent fallback)

**Purpose:** When camelot-py and pdfplumber find no extractable text in a PDF (scanned/image-based documents, common in UK Companies House filings), falls back to deep learning OCR via `python-doctr` (Mindee).

**Batched processing architecture:**
- Pages processed in batches of 50 (configurable via `max_batches_per_run`)
- Each batch's OCR output cached to disk (`cache/ocr/{hash}/page_{n}.txt`)
- On timeout or interruption, next run resumes from last cached page
- `scripts/run_ocr_batch.py` for staged processing of large PDFs
- `scripts/check_ocr_cache.py` for cache inspection

**Dependencies:** `python-doctr>=1.0` (optional, graceful fallback to empty extraction)

---

### 1.15 Institutional Flow (previously unlisted)

**File:** `operator1/features/institutional_flow.py` (372 lines)
**Pipeline step:** Step 5.inst
**Profile key:** `institutional_ownership_analysis.flow`

**Purpose:** Computes institutional ownership flow signals from holder data. Detects accumulation/distribution patterns, crowding risk, smart money divergence, and insider trading signals.

**9 output variables:** `inst_flow_momentum` (EMA-smoothed QoQ change, survival trigger at <-0.15), `inst_flow_momentum_label`, `inst_crowding_risk` (concentration x ownership level), `inst_crowding_risk_label`, `inst_smart_money_signal` (top-5 divergence), `inst_smart_money_label`, `inst_insider_signal` (net insider buying/selling), `inst_insider_label`, `inst_amihud_illiquidity` (Amihud 2002 ratio).

**Input:** Cache, insider_transactions list
**Output:** Cache + 9 inst_* columns

---

### 1.16 Options Signals (NEW -- 2026-04-18)

**File:** `operator1/features/options_signals.py` (~530 lines)
**Pipeline step:** Step 4a.7
**Profile key:** `options_signals`

**Purpose:** Fetches full options surface and computes 6 forward-looking features that move before price (institutional positioning visible in options first).

**6 output variables:** `put_call_ratio` (total put OI / call OI), `risk_reversal_25d` (25-delta put IV - call IV), `iv_skew` (OTM put IV / ATM IV), `vix_term_structure` (VIX / VIX3M ratio), `skew_index` (CBOE SKEW level), `variance_risk_premium` (IV30 - RV21).

**Input:** Cache, ticker, market_id
**Output:** Cache + 6 options columns, `OptionsSignalResult`
**Dependencies:** `yfinance` (for options chain data)

---

### 1.17 Cross-Asset Signals (NEW -- 2026-04-18)

**File:** `operator1/features/cross_asset_signals.py` (~420 lines)
**Pipeline step:** Step 4a.8
**Profile key:** `cross_asset_signals`

**Purpose:** Tracks 11 sector ETFs + Treasury yields + USD + gold to detect institutional capital rotation before it hits individual stocks.

**6 output variables:** `sector_relative_strength` (target sector ETF / SPY, 12-month), `sector_rank_12m` (rank among 11 sectors, 1=best), `sector_dispersion` (cross-sector return std), `yield_curve_10y2y` (10Y-2Y Treasury spread), `usd_momentum_21d` (DXY 21-day momentum), `cross_asset_stress` (composite stress index from treasury/gold/VIX/USD).

**Input:** Cache, sector string
**Output:** Cache + 6 cross-asset columns, `CrossAssetResult`

---

### 1.18 Event Calendar (NEW -- 2026-04-18)

**File:** `operator1/features/event_calendar.py` (~340 lines)
**Pipeline step:** Step 4c.1
**Profile key:** `event_calendar_signals`

**Purpose:** Tracks known upcoming events (FOMC meetings, estimated earnings, political dates from `config/event_calendar.json`) and computes proximity features that adjust prediction confidence.

**5 output variables:** `days_to_next_event`, `event_uncertainty_premium` (0-1 multiplier), `fomc_proximity` (days to nearest FOMC), `earnings_proximity` (days to estimated next earnings), `event_density_30d` (count of events in next 30 days).

**Downstream consumers:** Conformal prediction (interval widening near events), PEAD (drift decay), prediction aggregator (confidence adjustment).

**Input:** Cache, ticker, filing_calendar_result, reference_date
**Output:** Cache + 5 event columns, `EventCalendarResult`

---

### 1.19 Behavioral Signals (NEW -- 2026-05-01)

**File:** `operator1/features/behavioral_signals.py` (121 lines)
**Pipeline step:** Step 5i.7

**Purpose:** 5 behavioral finance signals that capture investor psychology biases exploitable for prediction.

**5 output variables:** `anchoring_52w_high` (close / 52-week high, George & Hwang 2004), `anchoring_52w_low` (close / 52-week low), `disposition_effect_proxy` (return-flow correlation, Shefrin & Statman 1985), `attention_spike` (volume z-score > 2.0, Barber & Odean 2008), `lottery_characteristics` (idio vol + skew + low price composite, Bali, Cakici & Whitelaw 2011).

**Critical dependency:** Must run AFTER institutional flow (requires `inst_flow_momentum`).

**Input:** Cache with inst_flow_momentum, beta_252d, benchmark_return_1d
**Output:** Cache + 5 behavioral columns

---

### 1.20 Complexity Signals (NEW -- 2026-05-01)

**File:** `operator1/features/complexity_signals.py` (254 lines)
**Pipeline step:** Step 5i.8

**Purpose:** 4 information-theoretic complexity measures. High complexity = low predictability = wider conformal bands.

**4 output variables:** `sample_entropy_21d` (Richman & Moorman 2000, template matching within tolerance), `perm_entropy_21d` (Bandt & Pompe 2002, ordinal pattern frequency), `lz_complexity` (Lempel-Ziv 1976, binarized return compression ratio), `approx_entropy_price` (Pincus 1991, price-level predictability).

**Implementation:** Inlined from `antropy` and `tsfresh` -- zero added dependencies.

**Input:** Cache with return_1d, close
**Output:** Cache + 4 complexity columns

---

### 1.21 Feature Normalization (NEW -- 2026-05-01)

**File:** `operator1/features/feature_normalization.py` (130 lines)
**Pipeline step:** Step 5i.9 (MUST run LAST before temporal models)

**Purpose:** Produces normalized versions of key features for ML model consumption and regime-conditional z-scores for USS controller.

**~35 output variables:** For 10 key variables: `{var}_zscore_63d`, `{var}_percentile_252d`, `{var}_change_21d`. For 5 survival-critical variables: `{var}_regime_zscore` (per-regime expanding z-score for "is this unusual FOR THIS REGIME?" detection).

**Critical dependency:** Requires `survival_regime` column from hierarchy_weights (Step 5).

**Input:** Cache with survival_regime column
**Output:** Cache + ~35 normalized columns

---

## Layer 2: Analysis Modules

10 modules in `operator1/analysis/` that produce survival flags, regime classifications, protection assessments, and adaptive parameter calibration. These provide the interpretive framework that temporal models operate within. **Updated 2026-05-01:** survival mode expanded with gradient velocity early warning (Duffie et al. 2007), bootstrap uncertainty bands, semi-Markov duration modeling; financial health expanded with ensemble distress prediction (Altman Z + Ohlson O + Zmijewski + Merton PD) and CVaR-weighted composite; hierarchy weights enhanced with entropy-based blending; USS controller enhanced with soft regime transitions; scenario engine enhanced with reverse stress testing (Basel III).

---

### 2.1 Survival Mode Detection

**File:** `operator1/analysis/survival_mode.py` (~822 lines)
**Pipeline step:** Step 5
**Profile key:** `survival`

**Purpose:** The core binary classification that determines whether a company is in financial distress. Produces three daily flags that control the entire pipeline's behavior via hierarchy weights. **Enhanced 2026-05-01:** added gradient velocity early warning (`compute_survival_velocity`, Duffie, Saita & Wang 2007), bootstrap uncertainty bands (`compute_survival_uncertainty`, P10/P90), and cash adequacy floor (suppresses liquidity triggers when cash/market_cap > 5%).

**Company survival flag** -- triggered when ANY condition is true:
```
current_ratio < 1.0                    (liquidity crisis)
debt_to_equity_abs > 3.0               (leverage crisis)
fcf_yield < 0                          (cash burn)
drawdown_252d < -0.40                  (market crash)
conflict_intensity_score > 0.7         (geopolitical crisis)
sanctions_flag == 1                    (sanctioned entity)
inst_flow_momentum < -0.15            (institutional exodus)
```

**Country survival flag** -- from macro thresholds in `config/country_protection_rules.yml`:
```
credit_spread > crisis_threshold
fx_volatility > crisis_threshold
unemployment_rate > crisis_threshold
yield_curve_slope < inversion_threshold
```

**Country protection flag** -- triggered when ANY is true:
```
sector in strategic_sectors
market_cap > 0.001 * country_gdp
emergency_rate_cut_detected
```

**Survival probability** (continuous 0-1): Sigmoid function of distance to each trigger threshold, with 6 components weighted by severity.

**Cox PH Survival Score** (data-driven, via `lifelines`):
```python
CoxPHFitter.fit(df[covariates], duration, event_observed)
```
Covariates: current_ratio, debt_to_equity_abs, fcf_yield, drawdown_252d. Learns hazard ratios from the company's own distress episodes. Blended with sigmoid: `survival_probability = 0.4 * sigmoid + 0.6 * cox`.

**Input:** Cache with derived variables
**Output:** Three integer flag Series + continuous survival_probability + cox_survival_score
**Dependencies:** `lifelines` (Cox PH)

---

### 2.2 Hierarchy Weights

**File:** `operator1/analysis/hierarchy_weights.py` (322 lines)
**Pipeline step:** Step 5
**Profile key:** `survival.hierarchy_weights`

**Purpose:** Assigns per-day per-tier weights based on the survival regime. These weights control how much computational and analytical attention each variable tier receives.

**Regime selection logic:**
| Company Flag | Country Flag | Protected | Regime | Weights (T1/T2/T3/T4/T5) |
|---|---|---|---|---|
| 0 | 0 | - | `normal` | 20/20/20/20/20 |
| 1 | 0 | - | `company_survival` | 50/30/15/4/1 |
| 0 | 1 | No | `modified_survival` | 40/35/20/4/1 |
| 0 | 1 | Yes | `normal` | 20/20/20/20/20 |
| 1 | 1 | - | `extreme_survival` | 60/30/10/0/0 |

**Vanity adjustment:** When `vanity_percentage > threshold` AND in survival regime, shifts 5% weight from Tier 4/5 to Tier 1 (company wastes money on non-essentials during distress).

**Sobol feedback loop:** After Sobol sensitivity analysis runs in Step 6t, `adjust_hierarchy_from_sobol()` nudges weights toward data-driven variable importance.

**Input:** Cache with survival flags
**Output:** Cache + `hierarchy_tier1_weight` through `tier5_weight`, `survival_regime` label

---

### 2.3 Survival Timeline

**File:** `operator1/analysis/survival_timeline.py` (664 lines)
**Pipeline step:** Step 5.5
**Profile key:** `enriched_survival_timeline`

**Purpose:** Bridges rule-based survival flags with HMM market regime labels into a unified state vector. Produces the most granular daily classification available -- 11 combined states that capture both company health and market environment.

**Base survival timeline** -- 6 modes from 3 binary flags:
| Mode | company | country | protected | Intensity Range |
|---|---|---|---|---|
| `normal` | 0 | 0 | * | 0.00-0.10 |
| `company_only` | 1 | 0 | * | 0.45-0.70 |
| `country_protected` | 0 | 1 | 1 | 0.15-0.35 |
| `country_exposed` | 0 | 1 | 0 | 0.40-0.70 |
| `both_protected` | 1 | 1 | 1 | 0.45-0.65 |
| `both_unprotected` | 1 | 1 | 0 | 0.70-1.00 |

**Enriched states** -- (survival_mode x market_regime) mapped to 11 combined states:
`stable_growth`, `elevated_risk`, `market_stress`, `company_distress_mild`, `company_distress_severe`, `country_crisis_mild`, `country_crisis_severe`, `protected_stress`, `protected_crisis`, `crisis`, `extreme_crisis`

**Output columns:** `regime_state`, `survival_intensity` (0-1 continuous), `regime_confidence`, `regime_switch`, `regime_transition_prob`, `survival_mode`, `switch_point`, `days_in_mode`, `stability_score_21d`

**Input:** Cache with survival flags + HMM regime labels
**Output:** Enriched timeline DataFrame merged into cache via reindex

---

### 2.4 Fuzzy Protection

**File:** `operator1/analysis/fuzzy_protection.py` (394 lines)
**Pipeline step:** Step 5b
**Profile key:** `fuzzy_protection`

**Purpose:** Assesses the degree to which a company is protected by government intervention using fuzzy logic. Produces a continuous 0-1 protection degree instead of the binary `country_protected_flag`.

**Three fuzzy input variables:**
1. **Sector strategicness** (0-1): Membership function over sector classification (defense, energy, banking, telecommunications = high; retail, entertainment = low)
2. **Economic significance** (0-1): `market_cap / GDP` ratio mapped through trapezoidal membership function
3. **Policy responsiveness** (0-1): Emergency rate cut magnitude within lookback window

**Mamdani fuzzy inference engine** (via `scikit-fuzzy`):
- 11 interaction rules mapping input combinations to output protection degree
- Defuzzification via centroid method
- Fallback: fuzzy OR (maximum) if scikit-fuzzy not available

**Parent sector inheritance:** If GLEIF corporate structure identifies a parent company, inherits parent's sector protection score.

**Input:** Cache, sector string, GDP float (optional), parent_sector (optional)
**Output:** Cache + `fuzzy_protection_degree` (0-1), `fuzzy_sector_score`, `fuzzy_protection_label`
**Dependencies:** `scikit-fuzzy` (optional, graceful fallback)

---

### 2.5 Ethical Filters

**File:** `operator1/analysis/ethical_filters.py` (355 lines)
**Pipeline step:** Called inside profile_builder
**Profile key:** `filters`

**Purpose:** Four independent financial quality screens that flag potential issues from different philosophical frameworks. Each filter produces a PASS/WARNING/FAIL verdict.

**Filter 1 -- Purchasing Power:** Is the company maintaining or growing real value for shareholders? Checks inflation-adjusted return and real dividend yield.

**Filter 2 -- Solvency:** Can the company meet all obligations? Checks current ratio, debt-to-equity, interest coverage against conservative thresholds.

**Filter 3 -- Gharar (Excessive Uncertainty):** Is the company's financial reporting transparent? Flags high earnings volatility, frequent restatements, and opaque accounting.

**Filter 4 -- Cash is King:** Is the company generating real cash? Checks FCF yield, OCF/NI ratio (cash conversion), and cash trend direction.

**Input:** Cache DataFrame
**Output:** Dict of 4 filter results, each with verdict + supporting metrics

---

### 2.6 Economic Planes

**File:** `operator1/analysis/economic_planes.py` (135 lines)
**Pipeline step:** Step 6 (pre-forecast)
**Profile key:** `economic_plane`

**Purpose:** Classifies the company into one of 5 economic planes from the Sudoku framework. Used for plane-aware model weighting in pre-forecasting synergies.

**Five planes:**
1. **Real Economy**: Manufacturing, industrials, consumer goods
2. **Financial**: Banks, insurance, asset management
3. **Technology**: Software, hardware, semiconductors, internet
4. **Resources**: Energy, mining, agriculture, utilities
5. **Services**: Healthcare, education, consulting, hospitality

**Input:** Sector string, industry string
**Output:** Dict with `primary_plane` and `secondary_planes` list

---

### 2.7 Vanity (Capital Allocation Quality)

**File:** `operator1/analysis/vanity.py` (636 lines)
**Pipeline step:** Step 5d
**Profile key:** `vanity`

**Purpose:** Scores how well management allocates capital. High vanity = management spending on non-essentials while core metrics deteriorate. Used to adjust hierarchy weights during survival mode.

**5-component composite (0-100 each):**
1. **R&D Mismatch** (15% weight): R&D spending vs revenue growth. High R&D with declining revenue = mismatch
2. **SGA Bloat** (25%): SGA expenses vs industry benchmarks. Excessive overhead
3. **Capital Misallocation** (30%): Debt-funded dividends/buybacks while FCF negative. Borrowing to return capital
4. **Competitive Decay** (15%): Losing market share vs peers (from peer_ranking)
5. **Sentiment Gap** (15%): Market sentiment diverging from fundamental reality

**Labels:** Disciplined (0-20), Moderate (20-40), Wasteful (40-70), Reckless (70-100)

**Trend:** `vanity_trend` = slope of vanity_score over short (21d) vs long (63d) MA. Rising = deteriorating management.

**Input:** Cache with derived vars, fh_* scores, optional sentiment and peer data
**Output:** Cache + `vanity_score`, `vanity_label`, `vanity_trend`, 5 component columns

---

### 2.8a Adaptive Thresholds (Tier 1 Calibration)

**File:** `operator1/analysis/adaptive_thresholds.py` (836 lines)
**Pipeline step:** Step 5j
**Profile key:** Consumed by survival_mode, monte_carlo, regime_mixer

**Purpose:** Replaces fixed textbook survival thresholds with peer-calibrated, data-derived values. The most important calibration module -- determines when survival mode triggers.

**5 calibration methods:**

**(A) Peer Percentile** (Huber 1981): Survival triggers set at P10/P90 of sector peer distribution. E.g., if peers' current_ratio P10 = 0.85, threshold becomes 0.85 instead of fixed 1.0. MAD-based fallback for small peer groups (<5).

**(D) BOCPD Deterioration Tightening** (Adams & MacKay 2007): When the company's own history shows a structural downward shift in a metric (detected via Bayesian Online Change Point Detection), thresholds tighten by 20%. Prevents ignoring gradual deterioration.

**(E) Sector Z-Score** (Iglewicz & Hoaglin 1993): Vanity metrics flagged at 2 modified-Z-scores from sector median using MAD (Median Absolute Deviation) instead of standard deviation.

**(H) Jenks Natural Breaks** (Fisher 1958): Financial health composite label breakpoints computed from optimal class boundaries via dynamic programming, replacing fixed percentile cuts.

**(J) HMM Emission Crossover** (Rabiner 1989): Regime mixer thresholds set at the Gaussian crossover point between HMM emission distributions of adjacent regimes.

**Safety:** All thresholds have absolute floors to prevent over-adaptation (e.g., current_ratio floor = 0.3 regardless of peers).

**Input:** Cache, linked_caches, regime_detector, fh_composite_scores
**Output:** `ThresholdSet` dataclass consumed by survival_mode, monte_carlo, regime_mixer

---

### 2.8b Adaptive Model Parameters (Tier 2 Calibration)

**File:** `operator1/analysis/adaptive_model_params.py` (1,185 lines)
**Pipeline step:** Step 5k
**Profile key:** Consumed by survival_probability, monte_carlo, forward_pass, graph_risk

**Purpose:** Computes data-driven model parameters that replace fixed constants across the pipeline. 10+ calibrated values.

**Key methods:**

**(5) Kish Effective Sample Size** (Kish 1965): True information content from interpolation weights. An annual filer with 252 daily forward-filled values has n_eff << 252.

**(6) Inverse-Variance Blending** (Cochrane 1954): Cox/sigmoid survival blend weights from prediction variance. Higher-precision model gets more weight.

**(7) Lambda PID Tuning** (Dahlin 1968): PID controller gains derived from error ACF half-life.

**(8a) Copula Tail Contagion** (Joe 2014): Edge-specific contagion probabilities from lower tail dependence coefficients.

**(8b) Amihud Participation Rate** (Amihud 2002): Liquidation rate from illiquidity ratio.

**(9a) Regime Risk Multiplier**: HMM volatility ratio between current and base regime.

**(9b) Garman-Klass Factor** (1980): Intraday volatility factor from OHLC data.

**(9c) Transition Half-Life**: From enriched timeline switch durations.

**(10) Precision-Targeted MC** (Glasserman 2003): Monte Carlo path count for target standard error.

Also includes Hurst exponent (trend/mean-reversion classification), bootstrap spread, conflict scoring weights, and train/test split ratios based on n_eff.

**Input:** Cache, regime_detector, enriched_timeline_result, Cox/sigmoid series
**Output:** `AdaptiveModelParams` dataclass with all calibrated values

---

### 2.8c Adaptive Windows & Hyperparameters (Tier 3 Calibration)

**File:** `operator1/analysis/adaptive_windows.py` (552 lines)
**Pipeline step:** Step 5k.2
**Profile key:** Consumed by derived_variables, forecasting, transformer, pattern_detector

**Purpose:** Anchors all rolling window sizes to the company's filing frequency using Nyquist sampling theory. Also calibrates neural network architecture and pattern detection thresholds.

**Key methods:**

**(11) Filing-Frequency-Anchored Windows** (Nyquist-Shannon): Base period = filing frequency in business days (63 for quarterly, 126 for semi-annual, 252 for annual). Windows: short = base/3, medium = base, long = 2*base, trend = 4*base.

**(12) Scaling-Law NN Architecture** (Kaplan 2020): `d_model` and `hidden_dim` proportional to n_eff^0.5. Larger effective samples = larger model capacity.

**(13) Innovation-Based Particle Noise** (Mehra 1970): State and observation noise from difference series standard deviation.

**(14) Distribution-Based Pattern Thresholds** (Bulkowski 2008): Doji threshold = P10 of body/range ratio distribution. Body threshold = P50.

**(16) Confidence Decay Half-Life**: Proportional to filing period.

**(18) Filing-Frequency Staleness**: 2x filing period.

**(19) Market-Specific Entity Scoring**: CJK markets get higher ticker weight, lower name weight in entity matching.

**Input:** Filing frequency, Kish n_eff, cache OHLC data
**Output:** `AdaptiveTier3Params` with windows, NN params, noise params, thresholds

---

### 2.9 Survival Regime Controller (USS)

**File:** `operator1/analysis/survival_regime_controller.py` (653 lines)
**Pipeline step:** Step 5-USS
**Profile key:** `unified_survival_system`

**Purpose:** Central controller for the Unified Survival System. Reconfigures the pipeline across 5 dimensions when survival mode activates.

**5 reconfiguration dimensions:**

**Dimension 1 -- Variable Triage:** Active/frozen/priority classification per (regime, tier). Tier 4-5 frozen in survival. Tier 1-2 get priority resources.

**Dimension 2 -- Model Switching:** Regime-specific parameters: Kalman noise 3-5x in crisis, LSTM lookback capped at 5-10 days, MC paths 20-30K, tree depth caps.

**Dimension 3 -- Horizon Compression:** extreme_survival = 1d/5d only. company_survival = 1d/5d/21d. Normal = all 4 horizons.

**Dimension 4 -- Correlation Switching:** 0.85-0.90 crisis correlation override + Clayton copula for lower tail dependence.

**Dimension 5 -- Forecast Bounding:** Revenue capped at last actual, debt floored, margins capped, cash bounded by max burn rate.

**Early Warning System:** Continuous 0-1 score based on proximity to survival triggers. Fires at 0.7 threshold (informational, does not change regime).

**Input:** Cache with survival flags
**Output:** `SurvivalRegimeController` with regime timeline, triage, model config, horizons, correlation/copula overrides

---

### 2.10 Scenario Engine (USS)

**File:** `operator1/analysis/scenario_engine.py` (387 lines)
**Pipeline step:** Step 6-USS
**Profile key:** `scenario_analysis`

**Purpose:** 3-scenario Monte Carlo simulation for when survival mode makes point forecasts unreliable (Knightian uncertainty).

**Scenario 1 -- Orderly Resolution:** Management restructures. Cash burn -30%, debt renegotiated, revenue -15%. MC uses 25th-75th percentile returns with -0.1%/day shift.

**Scenario 2 -- Muddle Through:** Status quo. Full historical return distribution, no shift.

**Scenario 3 -- Catastrophic:** Fire sale. Assets -40%, all debt called, revenue -40%. MC uses bottom 10% tail returns with -0.3%/day shift.

Each scenario produces: cash runway (days), survival probability at 90d and 252d, terminal equity value, terminal revenue, terminal cash, median return, P5/P95 returns, median max drawdown.

**Input:** Cache, regime, n_paths, horizon_days
**Output:** `ScenarioEngineResult` with 3 `ScenarioResult` objects

---

## Layer 3: Temporal Models

26 modules in `operator1/models/` that consume the enriched daily cache. Statistical and ML models for regime detection, forecasting, uncertainty quantification, and prediction aggregation. **Updated 2026-05-01:** added Boruta + PIMP + mRMR feature selector (replacing Granger pruning), recursive day-by-day prediction aggregation, jump-diffusion Monte Carlo with antithetic variates, temporal model feature integration (parallel tree, GARCH-X, residual regression).

---

### 3.1 Regime Detector

**File:** `operator1/models/regime_detector.py` (1,058 lines)
**Pipeline step:** Step 5.5 / Step 6
**Profile key:** `regimes`

**Purpose:** Identifies latent market regimes (bull, bear, high-volatility, low-volatility) and structural breaks using 5 complementary methods.

**Method 1 -- HMM** (Hidden Markov Model): 4-regime Gaussian HMM fitted on (return_1d, volatility_21d). Uses `hmmlearn.GaussianHMM`. Outputs `regime_hmm` labels + `regime_hmm_prob_*` state probabilities per day.

**Method 2 -- GMM** (Gaussian Mixture Model): Unsupervised clustering via `sklearn.mixture.GaussianMixture`. Cross-validates with BIC to select optimal n_components.

**Method 3 -- PELT** (Pruned Exact Linear Time): Structural break detection via `ruptures.Pelt`. Identifies dates where the return distribution changed permanently.

**Method 4 -- BCP** (Bayesian Change Point): Probabilistic change point detection using Bayesian posterior probability.

**Method 5 -- ChangeFinder** (online): Real-time sequential change point detection via SDAR (Sequentially Discounting AR model). No look-ahead -- each score is computed from past data only. Outputs `online_change_score` column.

**Consensus label:** `regime_label` = majority vote across methods that successfully fitted.

**Input:** Cache with `return_1d`, `volatility_21d`
**Output:** Cache + `regime_hmm`, `regime_gmm`, `regime_label`, `structural_break`, `online_change_score`, detector object
**Dependencies:** `hmmlearn`, `ruptures`, `changefinder`

---

### 3.2 Regime Mixer

**File:** `operator1/models/regime_mixer.py` (389 lines)
**Pipeline step:** Step 6b
**Profile key:** `extended_models.dual_regimes`

**Purpose:** Dual regime classification that separates market regimes (from price data) from fundamental regimes (from financial ratios). Produces blended weights for soft regime switching.

**Market regime:** Extracted from HMM columns (bull/bear/high_vol/low_vol).

**Fundamental regime:** Classified from financial ratios:
- Healthy: current_ratio > 1.5, debt_to_equity < 2, fcf_yield > 0
- Stressed: 1-2 triggers near thresholds
- Distress: 2+ triggers breached

**Blended weights:** Soft transition between regimes using logistic sigmoid of distance to regime boundary.

**Input:** Cache with regime columns
**Output:** `DualRegimeResult` with market_regime_labels, fund_regime_labels, blended_weights

---

### 3.3 Granger Causality

**File:** `operator1/models/granger_causality.py` (519 lines)
**Pipeline step:** Step 6c
**Profile key:** `extended_models.granger_causality`, `extended_models.time_varying_granger`

**Purpose:** Tests which variables Granger-cause which others. Used to prune irrelevant features before forecasting.

**Method 1 -- PCMCI** (preferred, via `tigramite`): Peter and Clark Momentary Conditional Independence test. Handles autocorrelation and confounders that invalidate standard Granger tests. Produces a causal graph with directed edges.

**Method 2 -- Pairwise Granger F-tests** (fallback): Standard `statsmodels.tsa.stattools.grangercausalitytests` at multiple lags (1, 2, 5, 10).

**Feature pruning:** `prune_features_by_causality()` **REMOVED (2026-04-19)** -- replaced by Boruta + PIMP + mRMR 3-layer feature selection (see 3.5b below). Granger result kept for informational purposes (profile, report, model synergies unified causal network).

**Time-varying Granger** (`compute_time_varying_granger`): Rolling-window causal analysis that detects emerging and disappearing causal relationships over time.

**Input:** Cache, up to 25 float columns with >50 non-NaN observations
**Output:** `GrangerResult` with causality_matrix, significant_pairs, retained/pruned variables, network_density
**Dependencies:** `tigramite` (optional, falls back to statsmodels)

---

### 3.4 Transfer Entropy

**File:** `operator1/models/causality.py` (417 lines)
**Pipeline step:** Step 6d
**Profile key:** `extended_models.transfer_entropy`

**Purpose:** Measures directed information flow between variable pairs using Shannon entropy. Complementary to Granger causality -- captures non-linear relationships that linear Granger tests miss.

**Method:** For each pair (X, Y), compute:
```
TE(X -> Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
```
Where H is conditional entropy estimated via k-nearest-neighbor density estimation.

**Input:** Cache, up to 20 float columns with >30 non-NaN observations
**Output:** `TransferEntropyResult` with pairwise entropy scores and top pairs

---

### 3.5 Model Synergies

**File:** `operator1/models/model_synergies.py` (797 lines)
**Pipeline step:** Step 6g (pre-forecast)
**Profile key:** `synergies_applied`

**Purpose:** Pre-forecasting integration that combines outputs from multiple upstream models into unified features and adjusts parameters for downstream models.

**Synergy A -- Cycle phase features:** Injects `cycle_phase_*` columns from cycle decomposition results (sine/cosine of dominant cycle phases).

**Synergy B -- Unified causal network:** Merges Granger + Transfer Entropy results into a single directed graph. Variables with both Granger and TE significance get higher causal confidence.

**Synergy C -- Pattern drift adjustment:** Computes multiplier from candlestick pattern detection for OHLC predictor.

**Synergy D -- Peer-adjusted survival thresholds:** Adjusts survival trigger levels using peer distribution data.

**Synergy E -- Plane-aware model weighting:** Economic plane classification adjusts which model types get priority (financial companies get different model mix than technology).

**Input:** Cache, cycle/granger/TE/peer results, economic plane
**Output:** Cache + synergy features, pruned `_extra_vars`, `_synergy_meta` dict

---

### 3.5b Feature Selector (NEW -- 2026-04-19)

**File:** `operator1/models/feature_selector.py` (~500 lines)
**Pipeline step:** Step 6c (sub-stage 3.8, after Granger)
**Profile key:** `feature_selection`

**Purpose:** 3-layer feature selection that replaced Granger-based pruning. Selects the optimal feature subset for temporal models using three complementary methods.

**Method 1 -- Boruta** (Kursa & Rudnicki 2010): Creates shadow features (random permutations of each real feature), trains random forest, compares real vs shadow feature importance. Features consistently more important than shadows are "confirmed"; borderline features are "tentative".

**Method 2 -- PIMP** (Altmann et al. 2010): Permutation Importance with P-values. Per-regime feature ranking -- computes permutation importance within each survival regime separately, so features important during distress may differ from features important during normal operations.

**Method 3 -- mRMR** (Peng, Long & Ding 2005): Minimum Redundancy Maximum Relevance. Selects features that are maximally relevant to the target while minimally redundant with each other. Uses mutual information estimation.

**Final selection:** Union of all three methods retained. Typically reduces ~300 extra_vars to ~50-80 informative features.

**Input:** Cache, extra_variables list, regime_labels
**Output:** `FeatureSelectionResult` with boruta_confirmed, boruta_tentative, regime_selected, mrmr_selected, method_contributions

---

### 3.6 Forecasting

**File:** `operator1/models/forecasting.py` (~4,588 lines)
**Pipeline step:** Step 6h
**Profile key:** `model_metrics`, `predictions`

**Purpose:** The primary forecasting engine. For each variable in the cache, tries 6 model types in sequence. First model that fits successfully wins.

**Model cascade (in priority order):**

1. **Kalman Filter** (local-level state-space): `filterpy` or custom implementation. Best for smooth time series with trend. Produces one-step-ahead prediction + Kalman gain.

2. **GARCH** (Generalized Autoregressive Conditional Heteroskedasticity): `arch` library. Models conditional volatility -- volatility clusters. Used primarily for `volatility_21d` and risk-related variables.

3. **VAR** (Vector Autoregression): `statsmodels.tsa.api.VAR`. Multivariate model that captures cross-variable dependencies. Falls back to AR(1) if VAR fails to converge.

4. **LSTM** (Long Short-Term Memory): PyTorch neural network. Falls back to GBM or linear regression if torch unavailable or insufficient data.

5. **Tree Ensemble** (RF/GBM/XGBoost): `sklearn.ensemble.RandomForestRegressor`, `GradientBoostingRegressor`, or `xgboost.XGBRegressor`. Feature-based prediction using all available cache columns.

6. **Baseline** (last-value or EMA): Simple carry-forward or exponential moving average. Always available as fallback.

**Standalone functions:** `fit_ets()` (via `statsforecast` ETS, replaced AutoARIMA on 2026-04-16, 10-50x faster) and `fit_dynamic_factor()` (multi-variable DFM via `statsmodels.tsa.DynamicFactor`). `fit_autoarima()` was removed -- ETS provides equivalent accuracy with dramatically lower latency and no hanging risk.

**Output:** `ForecastResult` with `.forecasts` (per-var, per-horizon point forecasts), `.metrics` (per-model RMSE/MAE), `.model_used` (which model won per variable), `.residuals` (for conformal calibration).

**Dependencies:** `arch`, `torch`, `xgboost`, `sklearn`, `statsforecast`, `filterpy`

---

### 3.7 Forward Pass

**File:** `operator1/models/forecasting.py` (within the 4,074 lines)
**Pipeline step:** Step 6i
**Profile key:** `pid_controller`

**Purpose:** Day-by-day temporal walk through history. For each day t, predicts t+1 using warmup models, compares to actual, updates PID controller for adaptive learning rate adjustment.

**Walk logic:**
```
For t in [warmup_end ... cache_end]:
    current_regime = cache['survival_regime'][t]
    for each variable in active_variables:
        prediction = model.predict(cache[:t])
        actual = cache[variable][t+1]
        error = actual - prediction
        pid_adjustment = pid_controller.update(error)
        model.update_learning_rate(pid_adjustment)
```

**Output:** `ForwardPassResult` with `errors_by_tier`, `errors_by_regime`, `model_states` (fitted models), `predictions_log`, `pid_summary`, `conformal_calibrator`

---

### 3.8 Walk-Forward Evaluation

**File:** `operator1/models/walk_forward.py` (700 lines)
**Pipeline step:** Step 6k
**Profile key:** Used by prediction_aggregator

**Purpose:** Independent model evaluation via expanding-window walk-forward testing. Produces per-model per-survival-mode error scores and a mode-conditioned leaderboard.

**4 model types evaluated:**
1. Baseline (last-value carry-forward)
2. EMA (exponential moving average)
3. Linear trend (OLS on recent window)
4. Mean reversion (predict toward expanding mean)

**Mode-conditioned scoring:** Errors tracked separately per survival mode. Models are ranked within each mode -- the best model for `company_survival` may differ from best for `normal`.

**Model Confidence Sets** (`compute_mode_confidence_sets` via `arch.bootstrap.MCS`): Identifies the subset of models that are statistically equivalent at the 5% significance level.

**Forward pass error aggregation** (`aggregate_forward_pass_errors`): Extracts per-model per-mode errors from the forward pass predictions log for use in the prediction aggregator.

**Input:** Cache, survival timeline modes/switches
**Output:** `WalkForwardResult` with day_errors, mode_scores, best_model_by_mode, retrain_dates, overall_mae
**Dependencies:** `arch` (for MCS bootstrap)

---

### 3.9 Burn-Out (Weight Calibration)

**File:** `operator1/models/forecasting.py` (within the 4,074 lines)
**Pipeline step:** Step 6j
**Profile key:** `extended_models.burnout`

**Purpose:** Intensive retraining on recent data with convergence detection. Uses exponential gradient learning to calibrate model weights per regime.

**Algorithm:**
1. Reset to burnout window (recent N days)
2. Run forward pass with higher learning rates
3. Track per-regime per-model RMSE
4. Update weights via exponential gradient: `w_new = w_old * exp(-eta * loss)`
5. Early-stop if no improvement for 3 iterations

**Output:** `BurnoutResult` with iterations_completed, converged, calibrated, regime_weights (per-regime per-model weight dicts), regime_distributions

---

### 3.10 Monte Carlo Simulation

**File:** `operator1/models/monte_carlo.py` (1,304 lines)
**Pipeline step:** Step 6l
**Profile key:** `monte_carlo`, `extended_models.multivariate_monte_carlo`

**Purpose:** Regime-switching Monte Carlo simulation for survival probability estimation. The primary uncertainty quantification tool.

**Standard MC (`run_monte_carlo`):**
1. Estimate per-regime return distributions from historical data
2. Build regime transition matrix from HMM
3. Simulate 10,000 paths with regime switching (sample regime at each step via transition matrix)
4. Apply importance sampling for tail events (tilt distribution toward crisis scenarios)
5. Compute survival probability = fraction of paths not triggering any survival threshold
6. **NEW (2026-05-01):** Jump-diffusion model (Merton 1976): `dS/S = (mu - lambda*k)dt + sigma*dW + J*dN` where N is Poisson process with intensity lambda and J is log-normal jump size. Jump parameters (lambda, mu_j, sigma_j) estimated from return distribution tail analysis. Antithetic variates for variance reduction (generates mirrored paths to halve MC standard error). Jump parameters passed to importance sampling paths (jumps are real events, not sampling artifacts).

**Frequency-aware evolution:** Quarterly/annual variables (current_ratio, fcf_yield) update only at estimated filing intervals (~63 days for quarterly), holding constant between filings.

**Multivariate MC (`run_multivariate_monte_carlo`):** Jointly simulates (return, delta_current_ratio, delta_fcf_yield, delta_debt_to_equity) using copula correlation structure. Checks survival triggers on simulated ratios directly, not via proxy mapping.

**Input:** Cache with return_1d, regime_label
**Output:** `MonteCarloResult` with survival_probability per horizon (90d, 252d), regime_distributions, transition_matrix, terminal_values
**Dependencies:** `numpy`

---

### 3.11 PID Controller

**File:** `operator1/models/pid_controller.py` (323 lines)
**Pipeline step:** Called inside forward pass
**Profile key:** `pid_controller`

**Purpose:** Adaptive learning rate control via PID (Proportional-Integral-Derivative) loop. Prevents overshoot and oscillation during the forward pass.

**PID formula:**
```
adjustment = Kp * error + Ki * integral(errors) + Kd * derivative(error)
learning_rate_multiplier = clip(1.0 + adjustment, 0.1, 5.0)
```

Gains (Kp, Ki, Kd) are derived from error ACF half-life via Dahlin tuning (adaptive_model_params step 5k).

**Input:** Per-tier prediction error history
**Output:** Adaptive learning rate multiplier per tier

---

### 3.12 Conformal Prediction

**File:** `operator1/models/conformal.py` (829 lines)
**Pipeline step:** Step 6p
**Profile key:** `conformal_intervals`

**Purpose:** Distribution-free prediction intervals that provide valid coverage guarantees without distributional assumptions.

**ConformalPIDCalibrator** (preferred, Angelopoulos 2023):
- PID-controlled coverage: dynamically adjusts quantile to maintain target coverage (e.g., 90%)
- If recent intervals are too wide (over-covering), PID shrinks them; if too narrow (under-covering), PID widens them
- Hierarchical Mondrian partitioning: separate calibration per survival mode, so crisis intervals are wider than normal intervals

**ConformalCalibrator** (fallback):
- Standard split conformal with exponentially weighted quantile
- Adaptive: recent residuals weighted more than old ones

**Regime-aware widening:** When regime transition probability is high or vol ratio between regimes is large, intervals are widened proportionally.

**Input:** Calibrator fed with residuals, forecasts dict, horizons dict
**Output:** `ConformalResult` with per-variable per-horizon prediction intervals (lower, upper, coverage)
**Dependencies:** None beyond numpy/pandas

---

### 3.13 Copula Analysis

**File:** `operator1/models/copula.py` (417 lines)
**Pipeline step:** Step 6m
**Profile key:** `extended_models.copula`

**Purpose:** Models the joint tail dependence structure between variables. Critical for understanding how risks compound during crisis -- "everything crashes together" is a copula phenomenon.

**Three copula types fitted:**
1. **Gaussian copula**: Symmetric, no tail dependence. Good for normal markets.
2. **Student-t copula**: Symmetric with tail dependence. Captures "fat tail" co-crashes.
3. **Clayton copula**: Asymmetric lower tail dependence. Best for modeling crisis contagion.

**Model selection:** AIC (Akaike Information Criterion) across all three. Best-fitting copula selected.

**Outputs:**
- `copula_correlation`: Estimated correlation matrix
- `tail_dependence`: Lower and upper tail dependence coefficients
- `joint_crisis_probability`: Probability that 2+ variables breach crisis thresholds simultaneously
- `best_copula`: Which copula type won AIC selection

**Input:** Cache with auto-selected float variables
**Output:** `CopulaResult` with correlation, tail dependence, joint crisis probability
**Dependencies:** `copulae` (optional, falls back to Gaussian-only)

---

### 3.14 Transformer Forecaster

**File:** `operator1/models/transformer_forecaster.py` (451 lines)
**Pipeline step:** Step 6n
**Profile key:** `extended_models.transformer`

**Purpose:** Multi-head self-attention neural network for time series forecasting. Captures long-range dependencies that LSTM may miss.

**Architecture:**
- Input: sliding windows of multivariate time series (up to 15 variables x lookback days)
- Positional encoding: sinusoidal (standard Transformer)
- Multi-head self-attention: 4 heads, d_model scaled by n_eff (from adaptive_windows)
- Feed-forward: 2 layers with ReLU + dropout
- Output: 1-step-ahead forecast per variable

**Training:** Adam optimizer, MSE loss, early stopping on validation loss. Lookback and architecture scaled by `AdaptiveTier3Params.nn_params`.

**Integration:** Forecasts injected into `forecast_result.forecasts` alongside other models. Gets `ModelMetrics` entry with final training loss as RMSE proxy.

**Input:** Cache, up to 15 float columns with >100 non-NaN observations
**Output:** `TransformerResult` with forecasts, feature_importance, train_loss_history
**Dependencies:** `torch`

---

### 3.15 Particle Filter

**File:** `operator1/models/particle_filter.py` (380 lines)
**Pipeline step:** Step 6o
**Profile key:** `extended_models.particle_filter`

**Purpose:** Sequential Monte Carlo for tracking latent survival-related state variables. Maintains a particle swarm that represents the distribution of possible current states.

**Algorithm:**
1. Initialize N particles from the prior (uniform around latest observed value)
2. For each time step:
   - Propagate: `x_t = x_{t-1} + noise` (random walk state transition)
   - Update: weight each particle by `likelihood(observation | x_t)` using Gaussian kernel
   - Resample: systematic resampling when ESS (Effective Sample Size) drops below N/2

**Variables tracked:** cash_ratio, free_cash_flow_ttm, current_ratio, debt_to_equity -- the survival trigger variables.

**Output:** Filtered state estimates (posterior mean), particle distributions at final time, weight distributions, percentile bands (P5, P25, P50, P75, P95).

**Input:** Cache, survival-related variables list
**Output:** `ParticleFilterResult` with filtered_states, particles_final, weights_final, percentiles

---

### 3.16 Cycle Decomposition

**File:** `operator1/models/cycle_decomposition.py` (336 lines)
**Pipeline step:** Step 6e
**Profile key:** `extended_models.cycle_decomposition`

**Purpose:** Identifies dominant periodicities in the price/variable time series. Used by model_synergies to inject cycle phase features.

**Method 1 -- CEEMDAN** (preferred, via `EMD-signal`): Complete Ensemble Empirical Mode Decomposition with Adaptive Noise. Decomposes the signal into Intrinsic Mode Functions (IMFs) -- each representing a different timescale oscillation. Adaptive to non-stationary signals (unlike FFT).

**Method 2 -- FFT** (fallback): Fast Fourier Transform spectral analysis. Identifies dominant frequencies from the power spectrum. Works well for stationary signals.

**Output per cycle:** period (days), amplitude, phase (radians), power (spectral density).

**Input:** Cache, variable name (default: "close")
**Output:** `CycleResult` with `dominant_cycles` list (sorted by amplitude)
**Dependencies:** `EMD-signal` (optional, falls back to FFT)

---

### 3.17 Pattern Detector

**File:** `operator1/models/pattern_detector.py` (507 lines)
**Pipeline step:** Step 6f
**Profile key:** `patterns`

**Purpose:** Detects candlestick patterns and recurring motifs/anomalies in OHLC data.

**Candlestick patterns detected:** doji, hammer, inverted_hammer, engulfing_bullish, engulfing_bearish, morning_star, evening_star, three_white_soldiers, three_black_crows. Uses body/shadow ratios with adaptive thresholds from `AdaptiveTier3Params.pattern_body_threshold`.

**Matrix Profile motifs** (via `stumpy`): Identifies the most frequently recurring subsequences in the price series. These are patterns that repeat historically -- potential predictive templates.

**Matrix Profile discords** (via `stumpy`): Identifies the most unusual/anomalous subsequences. These are patterns that have never occurred before -- potential regime change indicators.

**Predicted OHLC patterns** (`detect_patterns_on_predicted_ohlc`): Runs candlestick pattern detection on the predicted OHLC series from the OHLC predictor (Step 6v). Detects doji, hammer, engulfing, etc. on forward-looking candles. Results stored in `pattern_result.predicted_patterns_week`.

**Input:** Cache with open, high, low, close columns
**Output:** `PatternResult` with detected patterns, motifs, discords, predicted_patterns_week
**Dependencies:** `stumpy` (optional, pattern detection works without it)

---

### 3.18 SHAP Explainability

**File:** `operator1/models/explainability.py` (509 lines)
**Pipeline step:** Step 6s
**Profile key:** `extended_models.shap_explanations`

**Purpose:** Explains WHY each prediction was made by attributing contribution to individual features. Uses SHAP (SHapley Additive exPlanations) values.

**Method 1 -- TreeExplainer** (fast, for tree models): If fitted tree models (RF, GBM, XGB) are available from the forward pass, uses SHAP's exact TreeExplainer. O(TLD) complexity.

**Method 2 -- KernelExplainer** (general, for any model): If no tree models but predict functions are available, uses model-agnostic KernelExplainer. Slower but works with any differentiable model.

**Output per variable:** Top 5 contributing features with SHAP values, direction (positive/negative), and natural language narrative ("Revenue growth (+0.023) was the strongest driver of the close prediction").

**Input:** Cache, predictions dict, predict_fns dict (from forward_pass model_states)
**Output:** `SHAPResult` with per-variable feature importance, top drivers, narratives
**Dependencies:** `shap`

---

### 3.19 Sobol Sensitivity

**File:** `operator1/models/sensitivity.py` (330 lines)
**Pipeline step:** Step 6t
**Profile key:** `extended_models.sobol_sensitivity`

**Purpose:** Global sensitivity analysis that measures how much each input variable contributes to the variance of the output. Unlike SHAP (local, per-prediction), Sobol is global across the entire input space.

**Method:** Saltelli sampling scheme to estimate first-order (S1) and total-order (ST) Sobol indices:
- S1: variance due to variable alone
- ST: variance due to variable + all its interactions

**Hierarchy feedback:** `adjust_hierarchy_from_sobol()` nudges hierarchy weights toward Sobol-measured importance. If Sobol shows Tier 3 variables explain more variance than Tier 1, the weights shift slightly.

**Input:** Cache, target_variable (default: "return_1d")
**Output:** `SobolResult` with first-order and total-order sensitivity indices per feature

---

### 3.20 Financial Health Scoring

**File:** `operator1/models/financial_health.py` (1,081 lines)
**Pipeline step:** Step 5d
**Profile key:** `financial_health`

**Purpose:** Daily composite health score across 5 tiers, plus Altman Z-Score, Beneish M-Score, and liquidity runway.

**5-tier scoring (expanding percentile rank normalization):**
- Tier 1 (Liquidity): cash_ratio, current_ratio, free_cash_flow
- Tier 2 (Solvency): debt_to_equity, interest_coverage, net_debt_to_ebitda
- Tier 3 (Stability): volatility, drawdown, volume trend
- Tier 4 (Profitability): gross_margin, operating_margin, net_margin
- Tier 5 (Growth): revenue_growth, pe_ratio, ev_to_ebitda

**Composite:** Weighted average using hierarchy weights (20/20/20/20/20 in normal; 50/30/15/4/1 in survival).

**Altman Z-Score** (1968): `Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MVE/TL + 0.999*S/TA`
- Z > 2.99: Safe zone, Z < 1.81: Distress zone, Between: Grey zone

**Beneish M-Score** (1999): 8-factor manipulation detector:
DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA. M > -2.22 = likely manipulator.

**Liquidity Runway:** `months_of_runway = cash / |monthly_burn_rate|`

**Adaptive PE/EV caps:** Growth tier uses adaptive valuation caps via 3-method consensus: Log-Normal P99.5, Tukey Extreme Fence, MAD-Based Cap. Replaces fixed PE=200/EV=100.

**Input:** Cache, hierarchy_weights dict
**Output:** Cache + `fh_*` columns (12 total), `FinancialHealthResult` dataclass

---

### 3.21 Graph Risk

**File:** `operator1/models/graph_risk.py` (654 lines)
**Pipeline step:** Step 5e
**Profile key:** `graph_risk`

**Purpose:** Network analysis of the target company's position within its linked entity graph. Quantifies systemic risk, contagion exposure, and supply chain concentration.

**Network construction:** Nodes = target + all linked entities. Edges = relationship type (competitor, supplier, customer, financial_institution, parent, subsidiary). Edge weights from ownership overlap or revenue exposure.

**Metrics computed:**
- **Degree centrality:** Fraction of entities connected to target
- **PageRank:** Recursive importance (entities connected to important entities are themselves important)
- **CoVaR** (Conditional Value-at-Risk): VaR of target conditional on a linked entity being in distress
- **SRISK** (Systemic Risk): Capital shortfall of target under a market-wide stress scenario (40% market decline)
- **Contagion probability:** SIR-model infection probability (from adaptive_model_params copula tail dependence)
- **Supply chain concentration:** HHI of supplier/customer revenue shares

**Input:** Target ISIN/ticker, relationships dict, target_cache, linked_caches
**Output:** `GraphRiskResult` with n_nodes, centrality, CoVaR, SRISK, contagion metrics

---

### 3.22 Game Theory

**File:** `operator1/models/game_theory.py` (460 lines)
**Pipeline step:** Step 5e
**Profile key:** `game_theory`

**Purpose:** Analyzes competitive dynamics between the target and its competitors using game-theoretic frameworks.

**Models:**
- **Cournot quantity game:** Each firm chooses output quantity simultaneously. Nash equilibrium = where no firm can improve profit by changing output alone.
- **Bertrand price game:** Each firm chooses price simultaneously. Equilibrium depends on product differentiation.
- **Stackelberg leadership:** One firm (market leader) moves first. Tests whether target is leader or follower.
- **CR4 market structure:** 4-firm concentration ratio. Classifies as monopoly, oligopoly, or competitive.

**Competitive pressure index:** 0-1 composite measuring how much competitive forces threaten the target's profitability.

**Input:** Target cache, competitor caches dict, target_name
**Output:** `GameTheoryResult` with market_structure, competitive_pressure, Cournot/Stackelberg results
**Dependencies:** `nashpy` (optional, for exact Nash equilibria)

---

### 3.23 Ownership Contagion

**File:** `operator1/models/ownership_contagion.py` (595 lines)
**Pipeline step:** Step 5f.2
**Profile key:** `institutional_ownership_analysis`

**Purpose:** Analyzes shared institutional ownership between the target and competitors to assess crowding risk, liquidation risk, and portfolio-level contagion.

**Metrics:**
- **MHHI Delta** (Modified Herfindahl-Hirschman Index): Measures anti-competitive effects of common ownership. High MHHI = institutions that own both target and competitors have reduced incentive to compete.
- **Crowding score:** Fraction of target's ownership held by institutions also holding competitors. High crowding = all the same funds own the same stocks.
- **Liquidation days:** Estimated days to unwind the position given Amihud illiquidity. `days = position_value / (daily_volume * participation_rate * price)`.
- **Bipartite centrality:** Target's centrality in the company-institution bipartite network.

**Input:** Target holders list, competitor holders dict, cache
**Output:** `ContagionResult` with MHHI, crowding, liquidation, shared institutions

---

### 3.24 DTW Historical Analogs

**File:** `operator1/models/dtw_analogs.py` (453 lines)
**Pipeline step:** Step 6q
**Profile key:** `extended_models.dtw_analogs`

**Purpose:** Finds historical periods that look most similar to the current price/variable pattern using Dynamic Time Warping distance. Derives empirical forecasts from what happened after each analog.

**Algorithm:**
1. Extract recent window (default: 63 days) of close prices
2. Slide over all historical windows of the same length
3. Compute DTW distance (allows time-axis stretching, unlike Euclidean)
4. Rank matches by distance
5. For top-K matches: extract what happened in the N days AFTER the match
6. Derive empirical forecast as median of post-match trajectories

**Cross-company search:** Also searches linked entity caches for analogs, with 0.7x weighting (peer analogs are informative but less directly relevant).

**Catalyst adjustment:** If catalyst_score is high (product launch detected), analog weight shifts toward recovery patterns.

**Input:** Cache, linked_caches (optional), catalyst_score (optional)
**Output:** `DTWAnalogResult` with matches and empirical forecasts
**Dependencies:** `dtaidistance`

---

### 3.25 Prediction Aggregator

**File:** `operator1/models/prediction_aggregator.py` (2,253 lines)
**Pipeline step:** Step 6r
**Profile key:** `predictions`

**Purpose:** The final ensemble that combines all model outputs into unified predictions with uncertainty bands. The most complex temporal module.

**Aggregation pipeline (10 steps):**

1. **Ensemble weights:** Inverse-RMSE weights OR FixedShareForecaster weights (Herbster & Warmuth 1998) filtered by Model Confidence Sets
2. **Regime blending:** Soft switching using dual_regime_result blended weights
3. **Point forecast aggregation:** Weighted average of all model forecasts
4. **Uncertainty bands:** ConformalPIDCalibrator intervals (preferred) or RMSE-based fallback
5. **Copula widening:** Widen bands using copula tail dependence (joint crisis probability)
6. **DTW analog overlay:** Add empirical forecast from historical analogs
7. **Granger/PCMCI adjustment:** Propagate causal chain effects
8. **SHAP attachment:** Attach per-prediction explanation narratives
9. **Technical Alpha mask:** Hide OHLC predictions except Low (proprietary methodology)
10. **Walk-forward MCS weighting:** Mode-conditioned weights from walk-forward + MCS

**FixedShareForecaster** (Herbster & Warmuth 1998): Online learning algorithm that maintains model weights, allowing weights to shift over time as model performance changes. The "share" parameter controls adaptation speed.

**Output:** `PredictionAggregatorResult` with per-variable per-horizon `HorizonPrediction` (point_forecast, lower_ci, upper_ci, confidence, model_weights), technical_alpha mask, metadata

---

### 3.26 Genetic Optimizer

**File:** `operator1/models/genetic_optimizer.py` (504 lines)
**Pipeline step:** Step 6u
**Profile key:** `extended_models.genetic_optimizer`

**Purpose:** Optimizes ensemble model weights using evolutionary algorithm. Finds the weight combination that minimizes prediction error.

**Method 1 -- Optuna TPE** (preferred, faster convergence): Tree-structured Parzen Estimator via `optuna.create_study`. Each trial tests a different weight vector sampled from a Dirichlet distribution. Prunes unpromising trials early.

**Method 2 -- Genetic Algorithm** (fallback):
1. Initialize population via Dirichlet distribution (ensures weights sum to 1)
2. Evaluate fitness = -RMSE of weighted ensemble
3. Tournament selection (k=3)
4. Crossover: weighted average of parents + random perturbation
5. Mutation: Dirichlet noise
6. Elite carryover: top 10% preserved unchanged
7. Converge after N generations or no improvement for 5 generations

**Per-regime optimization:** Separate weight vectors for each survival regime.

**Input:** Cache, forecast_result
**Output:** `GAResult` with best_weights, tier_weights, per_regime_weights, fitness_history, converged
**Dependencies:** `optuna` (optional, falls back to custom GA)

---

### 3.27 OHLC Predictor

**File:** `operator1/models/ohlc_predictor.py` (379 lines)
**Pipeline step:** Step 6v
**Profile key:** `ohlc_predictions`

**Purpose:** Predicts next-day/week/month Open-High-Low-Close candlesticks by combining forecast results, Monte Carlo simulations, pattern drift, and cycle phase.

**Prediction logic:**
- **Close:** From forecast_result (best model for "close")
- **Open:** Previous close + overnight gap estimate (from return_1d distribution)
- **High:** Close + upper tail from MC paths (scaled by pattern drift multiplier)
- **Low:** Close - lower tail from MC paths (scaled by Garman-Klass factor)
- **Cycle adjustment:** If cycle decomposition detected a dominant cycle, adjust predictions by cycle phase (positive phase = bias toward higher high, negative phase = bias toward lower low)

**Output horizons:** next_day (1 candle), next_week (5 candles), next_month (21 candles), next_year (252 candles).

**Input:** Cache, forecast_result, mc_result, pattern_drift_multiplier, cycle_result
**Output:** `OHLCResult` with OHLCCandle objects per horizon

---

### 3.28 Retroactive Calibration

**File:** `operator1/analysis/retroactive_calibration.py` (552 lines)
**Pipeline step:** Step 6.5
**Profile key:** Not stored directly (consumed by subsequent runs)

**Purpose:** After all temporal models have run, uses their outputs to calibrate model weight matrices that were initially set to fixed defaults. Empirical Bayes: use first-pass data to set second-pass priors.

**Calibration groups:**
- Linked entity group weights (how much each entity group contributes to aggregate features)
- Walk-forward retrain triggers (how often to retrain vs carry forward)
- Forecast-based ensemble blend ratios
- Sobol-informed variable importance rankings

**Input:** Cache, linked_caches, entity_groups, walk_forward_result, forecast_result, sobol_result, target_profile
**Output:** `RetroCalibrationResult` with calibrated group weights and parameter adjustments

---

### 3.28b Recursive Day-by-Day Predictions (NEW -- 2026-04-24)

**File:** `operator1/models/recursive_aggregator.py` (~350 lines)
**Pipeline step:** Sub-stage 6.11
**Profile key:** `extended_models.recursive_predictions`

**Purpose:** Autoregressive chaining of day-by-day predictions. Instead of predicting each horizon independently, predicts day t+1, injects predicted values as inputs for day t+2 prediction, and repeats for 5-21 days forward. Handles uncertainty propagation -- confidence bands widen at each step.

**Algorithm:**
1. Initialize from latest actual cache values
2. For each forward day d in [1..N]:
   - Build feature vector from actuals (d < today) + predicted values (d >= today)
   - Run prediction aggregation for day d
   - Store prediction + confidence
   - Inject predicted values into the feature vector for day d+1
3. Track per-step model weights, confidence decay, regime transitions

**Output:** `RecursiveResult` with per-day predictions (point + bands), cumulative confidence decay, regime tracking, total days predicted, method used.

**Input:** Cache, forecast_result, mc_result, pred_result, regime_labels, conformal_result
**Output:** `RecursiveResult` stored in profile via `result.to_dict()`

---

### 3.29 Regime Shift Predictor

**File:** `operator1/models/regime_shift_predictor.py` (351 lines)
**Pipeline step:** Step 6 (after Monte Carlo)
**Profile key:** `predicted_regime_shifts`

**Purpose:** Predicts when the current market regime is likely to change and to which regime it will transition. Uses the HMM transition matrix from Monte Carlo simulation.

**Method:** Geometric CDF of regime exit probability. For each future horizon (21d, 63d, 252d), computes the probability that the current regime will have ended, using the Markov chain transition matrix. Adjusts by:
- `stability_score_21d` from the enriched survival timeline (high stability = lower exit probability)
- `transition_halflife` from adaptive model params (empirical decay rate)

**Output fields:**
- `prob_exit_21d`, `prob_exit_252d`: Probability of leaving current regime within N days
- `expected_days_to_shift`: Expected number of days until regime change
- `most_probable_next_regime`: Which regime is most likely next (from transition matrix row)
- `current_regime`: Current regime label for reference

**Input:** Cache, MC transition_matrix, regime_order, stability_score, transition_halflife, reference_date
**Output:** `RegimeShiftResult` stored in profile via `result.to_dict()`

---

### 3.30 Model Diagnostics

**File:** `operator1/monitoring/model_diagnostics.py` (875 lines)
**Pipeline step:** Step 6.6 (after all temporal models)
**Profile key:** `model_diagnostics`

**Purpose:** For each model, pre-computes what it SHOULD produce based on data characteristics, then compares against what it actually produced. Produces per-model robustness ratings for the profile and report.

**10 models assessed:**
1. **Kalman**: Expected to fit with state_dim=1, produces MSE. Check: fitted + low MSE.
2. **GARCH**: Expected for volatile series. Check: fitted + reasonable persistence params.
3. **VAR**: Expected for multivariate series. Check: stable roots + reasonable lag order.
4. **LSTM**: Expected with sufficient data (>200 points). Check: training converged.
5. **Tree**: Expected always (no convergence issues). Check: reasonable feature importance.
6. **Monte Carlo**: Expected survival_probability in [0, 1]. Check: path count matches config.
7. **Copula**: Expected correlation structure. Check: AIC selection meaningful.
8. **Granger**: Expected causal pairs. Check: network density reasonable.
9. **Cycle**: Expected dominant cycles. Check: dominant period within plausible range.
10. **DTW**: Expected analog matches. Check: distances within plausible range.
11. **Conformal**: Expected coverage near target (90%). Check: empirical coverage.

**Robustness ratings:** `on_track` (model behaves as expected), `degraded` (model ran but results questionable), `failed` (model did not produce usable output).

**Input:** Cache, all temporal model results
**Output:** `ModelDiagnosticsResult` with per-model assessments, overall_robustness score
**Report:** Rendered in report section 19.97 (Premium tier)

---

## Layer 4: Hedge Fund Analysis (NEW)

18 modules in `operator1/hedge_fund/` that run as a parallel analytical track. While the existing pipeline answers "What state is this company in?", the HF pipeline answers "Can I make money on this, when, and how much?" Primary data source: raw quarterly statement DataFrames (8-24 rows), NOT the 504-row daily cache.

---

### 4.1 FCF Quality Scoring

**File:** `operator1/hedge_fund/fcf_quality.py` (~200 lines)
**Pipeline step:** Step 6-HF-a
**Profile key:** `hedge_fund.fcf_quality`

**Purpose:** Measures whether reported earnings are backed by real cash generation. Degrades 2-4 quarters before blowups.

**Formula:** `FCF_Quality = 0.40 * OCF_NI_Ratio_8Q + 0.30 * (1 - |Accruals|) + 0.20 * FCF_Trend + 0.10 * (1 - CapEx_Vol)`

**Negative NI handling:** When NI is negative, OCF/NI ratio is meaningless. Scores based on whether OCF is at least positive (35/100) or both negative (5/100).

**Input:** Raw `income_df`, `cashflow_df`, `balance_df` (8 quarterly filings each)
**Output:** `FCFQualityResult` with score 0-100, degradation_flag, narrative

---

### 4.2 Accruals Forensics

**File:** `operator1/hedge_fund/accruals_forensics.py` (~350 lines)
**Pipeline step:** Step 6-HF-b
**Profile key:** `hedge_fund.accruals_forensic`

**Purpose:** Detects earnings manipulation via 5-component composite.

**Components:**
1. **Sloan Accruals** (30%): `(NI - OCF) / Total_Assets` -- high absolute value = earnings not backed by cash
2. **Modified Jones Model** (25%): OLS regression: `Total_Accruals/TA_lag = a*(1/TA_lag) + b*(delta_REV - delta_REC)/TA_lag`. Residuals = discretionary accruals = manipulation signal
3. **NI-OCF Divergence Trend** (20%): Slope of (NI - OCF) over 8 quarters. Positive slope = NI growing faster than cash
4. **Working Capital Anomalies** (15%): Coefficient of variation of WC changes. High CV = unexplained swings
5. **Cash Conversion Efficiency** (10%): `OCF / (Revenue - COGS)` trend. Declining = stop converting sales to cash

**Input:** Raw `income_df`, `balance_df`, `cashflow_df` (8Q + 5A for Jones regression)
**Output:** `AccrualsForensicResult` with red_flag_score 0-100, discretionary_accruals, cce_trend

---

### 4.3 Earnings Smoothing Detector

**File:** `operator1/hedge_fund/earnings_smoothing.py` (~300 lines)
**Pipeline step:** Step 6-HF-c
**Profile key:** `hedge_fund.smoothing`

**Purpose:** Extends Beneish M-Score with additional smoothing detection.

**Components:**
1. **Beneish M-Score Probability** (25%): From existing `financial_health.py` (M > -2.22 = likely manipulator)
2. **Earnings Vol / Cash Flow Vol Ratio** (25%): NI should be AT LEAST as volatile as OCF. Ratio < 0.5 = suspicious smoothing
3. **Benford's Law Digit Analysis** (20%): Chi-squared test of first-digit distribution of reported revenue against Benford expected distribution. Needs 30+ data points
4. **Sequential Surprise Pattern** (15%): Count consecutive EPS beats. 6+ consecutive = statistically unlikely without smoothing
5. **Restatement Risk Proxy** (15%): Rate of accruals growth as proxy for restatement likelihood

**Input:** Raw `income_df`, `cashflow_df`, `fh_result` (for Beneish)
**Output:** `SmoothingResult` with smoothing_index 0-100, benford_deviation, earnings_vol_ratio

---

### 4.4 Dividend Burn Risk

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_dividend_burn`)
**Pipeline step:** Step 6-HF-d
**Profile key:** `hedge_fund.dividend_burn`

**Purpose:** Predicts dividend sustainability. Fires 2-4 quarters before cuts.

**Formula:** `Risk = 0.40 * (Divs/FCF) + 0.30 * (Debt_Service/FCF) + 0.20 * WC_Drain + 0.10 * Earnings_Vol`

**Negative FCF handling:** When FCF < 0 and dividends > 0, automatic 95/100 risk score (paying from debt/asset liquidation).

**Input:** Raw `cashflow_df`, `income_df`, `balance_df` (8Q)
**Output:** `DividendBurnResult` with risk_score 0-100, coverage_ratio, months_to_cut

---

### 4.5 CROA vs ROIC Spread

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_return_spread`)
**Pipeline step:** Step 6-HF-e
**Profile key:** `hedge_fund.return_spread`

**Purpose:** Compares cash returns (CROA = OCF/Total_Assets) vs accounting returns (ROIC = NOPAT/Invested_Capital). Divergence signals accrual inflation.

**Input:** Raw `income_df`, `balance_df`, `cashflow_df` (latest filing)
**Output:** `ReturnSpreadResult` with croa, roic, spread_bps, quality_label

---

### 4.6 Operating Leverage

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_operating_leverage`)
**Pipeline step:** Step 6-HF-f
**Profile key:** `hedge_fund.operating_leverage`

**Purpose:** Measures earnings sensitivity to revenue changes.

**Formulas:** `DOL = %delta_EBIT / %delta_Revenue`, `DFL = %delta_EPS / %delta_EBIT`, `DTL = DOL * DFL`

**Classification:** |DOL| > 5 = extreme, > 3 = high, > 1.5 = moderate, else low

**Input:** Raw `income_df` (8Q, Q/Q percentage changes)
**Output:** `OperatingLeverageResult` with dol, dfl, dtl, earnings_sensitivity

---

### 4.7 Off-Balance-Sheet Risk

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_obs_risk`)
**Pipeline step:** Step 6-HF-g
**Profile key:** `hedge_fund.obs_risk`

**Purpose:** Scores hidden liabilities from goodwill/intangibles ratio + SGA anomaly proxy.

**Formula:** `OBS_Risk = 0.35 * (Goodwill/TA) + 0.25 * (Intangibles/TA) + 0.20 * SGA_Anomaly + 0.10 * Keyword_Score + 0.10 * Policy_Change`

**Input:** Raw `balance_df`, `income_df` (latest 4-8 filings)
**Output:** `OBSRiskResult` with risk_score 0-100, goodwill_to_assets, sga_anomaly

---

### 4.8 Asset Quality Deterioration

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_asset_quality`)
**Pipeline step:** Step 6-HF-h
**Profile key:** `hedge_fund.asset_quality`

**Purpose:** Detects when balance sheet asset quality is degrading.

**Key metrics:** DSO (Days Sales Outstanding) = `receivables / (revenue/90)`, Inventory Days = `inventory / (COGS/90)`. Rising DSO = revenue quality issue. Rising inventory = obsolescence risk.

**Input:** Raw `income_df`, `balance_df` (8Q, Q/Q deltas)
**Output:** `AssetQualityResult` with deterioration_score 0-100, dso, dso_change_pct, inventory_days

---

### 4.9 Leverage Stress Scenario Engine

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_leverage_stress`)
**Pipeline step:** Step 6-HF-i
**Profile key:** `hedge_fund.leverage_stress`

**Purpose:** Tests whether current leverage is survivable under 3 scenarios.

**Scenarios:**
1. **Base case:** Current trajectory
2. **Revenue miss:** -15% revenue, -200bps margin compression
3. **Systemic crisis:** -25% revenue, -400bps margin, +200bps interest rate

**Negative EBITDA handling:** Automatic covenant breach + distress classification.

**Covenant check:** Debt/EBITDA > 5.5x = covenant breach flag

**Input:** Raw `income_df`, `balance_df`, `mc_result` (for probability weighting)
**Output:** `LeverageStressResult` with 3 scenario objects, refinancing_risk flag

---

### 4.10 Momentum Composite

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_momentum`)
**Pipeline step:** Step 6-HF-j
**Profile key:** `hedge_fund.momentum`

**Purpose:** Multi-dimensional fundamental momentum.

**Formula:** `Momentum = 0.40 * Revenue_Accel + 0.30 * Margin_Slope + 0.20 * FCF_Conv + 0.10 * ROIC_Traj`

Where revenue acceleration = 2nd derivative (change in growth rate).

**Input:** Raw `income_df`, `cashflow_df` (8Q), daily cache (63d for price momentum divergence)
**Output:** `MomentumCompositeResult` with score 0-100, inflection_detected flag

---

### 4.11 Growth Quality

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_growth_quality`)
**Pipeline step:** Step 6-HF-k
**Profile key:** `hedge_fund.growth_quality`

**Purpose:** Decomposes growth into organic vs inorganic (M&A).

**M&A detection:** Goodwill/intangible jumps in balance sheet = acquisition signal.

**Formula:** `Quality = 0.50 * Organic_Fraction + 0.25 * Margin_Adjusted + 0.15 * Incremental_ROIC + 0.10 * Concentration`

**Input:** Raw `income_df`, `balance_df` (8Q)
**Output:** `GrowthQualityResult` with score 0-100, organic_fraction, incremental_roic

---

### 4.12 Earnings Surprise Probability

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_earnings_surprise`)
**Pipeline step:** Step 6-HF-l
**Profile key:** `hedge_fund.earnings_surprise`

**Purpose:** Estimates P(beat), P(miss), P(inline) for next earnings.

**Method:** Historical SUE (Standardized Unexpected Earnings) distribution over 8Q. P(miss) = CDF at -1.5 sigma. Adjusted by recent trend.

**Input:** Raw `income_df` (8Q EPS), `filing_calendar_result` (days to next filing)
**Output:** `SurpriseResult` with p_beat, p_miss, p_inline, days_to_next_filing

---

### 4.13 DCF Monte Carlo Valuation

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_dcf`)
**Pipeline step:** Step 6-HF-m
**Profile key:** `hedge_fund.dcf`

**Purpose:** Probabilistic DCF with 10K simulations and regime-specific growth.

**Method:** For each simulation: sample growth from regime distribution, sample WACC from N(0.09, 0.01), compute 5-year explicit FCF + terminal value. Discount and subtract net debt.

**Integration:** Uses `mc_result.regime_distributions` for growth assumptions. Annual frequency forecast bounds from `multi_frequency_result` constrain terminal growth.

**Input:** Raw `cashflow_df`, `balance_df` (latest), `mc_result`, `target_profile`, cache (close price)
**Output:** `DCFResult` with intrinsic_p10/p25/p50/p75/p90, upside_pct, risk_reward_ratio

---

### 4.14 Valuation-Quality Matrix

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_valuation_quality`)
**Pipeline step:** Step 6-HF-n
**Profile key:** `hedge_fund.valuation_quality`

**Purpose:** Maps company on 2D plane: X = Quality (composite of HF Tiers 1-3), Y = Valuation (PE rank vs peers). Identifies mispricings.

**Quadrants:** undervalued (cheap + quality), fair (quality + fair price), overvalued (expensive + quality), value_trap (cheap + low quality)

**Input:** All HF Tier 1-3 results, `peer_ranking_result`, cache (PE, EV/EBITDA)
**Output:** `ValuationQualityResult` with quality_score, valuation_percentile, quadrant

---

### 4.15 PEG Composite

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_peg`)
**Pipeline step:** Step 6-HF-o
**Profile key:** `hedge_fund.peg_composite`

**Purpose:** Quality-adjusted PEG + FCF yield vs debt cost spread.

**Formulas:** `PEG_Adjusted = (PE / Growth%) / Quality_Multiplier`, `FCF_Spread = FCF_Yield - Debt_Cost` (bps)

**Input:** Raw `income_df` (4Q for growth), cache (PE, FCF yield), HF-5.2 quality_score
**Output:** `PEGCompositeResult` with peg_adjusted, fcf_spread_bps, cheap_flag

---

### 4.16 Investment Thesis Scorecard

**File:** `operator1/hedge_fund/engine.py` (inline, `_build_scorecard`)
**Pipeline step:** Step 6-HF-q
**Profile key:** `hedge_fund.scorecard`

**Purpose:** Synthesizes all 15 HF metrics into a 5-tier scorecard with investment grade.

**Grading:** Weighted composite: Earnings Quality 25% + Cash Flow 20% + Balance Sheet 20% + Inflection 20% + Valuation 15%. A+ (>85) through F (<25).

**Distress override:** When covenant breach or FCF quality < 30, grade capped at D regardless of other tier scores.

**Output:** `ThesisScorecard` with 5 TierScore objects, investment_grade (A+ to F), conviction (0-10)

---

### 4.17 Position Signal Engine

**File:** `operator1/hedge_fund/engine.py` (inline, `_compute_position_signal`)
**Pipeline step:** Step 6-HF-r
**Profile key:** `hedge_fund.position`

**Purpose:** Converts thesis scorecard into actionable signal with sizing guidance.

**Formula:** `signal = alpha * quality_mult * survival_mult * decay_mult * conviction`

Where alpha comes from `forecast_result.return_5d` (primary) or scorecard grade (fallback: A+=0.08, D=-0.04, F=-0.08).

**Survival-aware sizing:** Normal=1.0x, Modified=0.5x, Company_Survival=0.0x, Extreme=-0.5x, Recovery=1.5x.

**Entry/stop/target:** Derived from 63-day price support/resistance levels in the daily cache.

**Output:** `PositionSignalResult` with signal (-1 to +1), label, conviction, entry/stop/target prices

---

### 4.18 Orchestrator

**File:** `operator1/hedge_fund/engine.py` (`run_hedge_fund_analysis`)
**Pipeline step:** Step 6-HF
**Profile key:** `hedge_fund`

**Purpose:** Runs all 15 HF metrics in dependency order, builds scorecard, computes position signal. Single entry point called from main.py.

**Input:** Raw statement DFs + cache + all upstream model results
**Output:** `HedgeFundResult` with all 15 metric results + scorecard + position signal

---

## Layer 5: Multi-Frequency Pipeline

3 modules that run the analytical pipeline at 5 frequencies (Annual -> Daily) with cascading context, then fuse the results.

---

### 5.1 Frequency Resampler

**File:** `operator1/features/frequency_resampler.py` (688 lines)
**Pipeline step:** Step 6.7 (pre-processing)
**Profile key:** Consumed by multi_frequency_runner

**Purpose:** Resamples the daily cache to lower frequencies (Weekly, Monthly, Quarterly, Semi-Annual, Annual) while preserving PIT constraints.

**Frequency configuration:**

| Freq | Label | Resample Rule | Lookback |
|------|-------|---------------|----------|
| D | Daily | None (native) | 2 years |
| W | Weekly | W-FRI | 3 years |
| M | Monthly | ME | 5 years |
| Q | Quarterly | QE | 6 years |
| S | Semi-Annual | 2QE | 7 years |
| A | Annual | YE | 8 years |

**OHLCV resampling:** Open=first, High=max, Low=min, Close=last, Volume=sum.
**Statement resampling:** Stock variables (balance sheet) = last, Flow variables (income/cashflow) = sum.
**No-look-ahead:** Last period truncated to current date. Incomplete periods flagged.

**Direct construction from raw filings:** `build_cache_from_raw_filings()` constructs Q/A caches directly from raw statement DataFrames without going through daily forward-fill, avoiding interpolation artifacts.

**Input:** Daily cache or raw statement DFs, target frequency
**Output:** `ResampledCache` with resampled DataFrame + metadata

---

### 5.2 Multi-Frequency Runner

**File:** `operator1/steps/multi_frequency_runner.py` (606 lines)
**Pipeline step:** Step 6.7
**Profile key:** `multi_frequency` (via fusion)

**Purpose:** Runs the full analytical pipeline (derived variables, survival mode, regime detection, forecasting, Monte Carlo) at each frequency in slow-to-fast order. Each frequency passes context to the next.

**Execution order:** Annual -> Quarterly -> Monthly -> Weekly -> Daily

**Cascading context** (`FrequencyContext`): Each frequency passes to the next:
- `trend_direction`: "up", "down", "flat"
- `secular_regime`: "bull", "bear", "sideways", "recovery"
- `survival_probability_latest`: 0-1
- `forecast_bounds`: Per-variable (min, max) from historical range at this frequency

**Per-frequency pipeline:** For each `ResampledCache`:
1. `compute_derived_variables(cache)`
2. `compute_company_survival_flag(cache)` + `compute_survival_probability(cache)`
3. `detect_regimes_and_breaks(cache)`
4. `run_forecasting(cache)` (if not skip_models)
5. `run_monte_carlo(cache)` (if not skip_models)
6. Extract summary context for next frequency

**Input:** Daily cache, secrets, market_id, ticker, raw statement DFs
**Output:** `MultiFrequencyResult` with per-frequency `FrequencyResult` objects

---

### 5.3 Frequency Fusion

**File:** `operator1/models/frequency_fusion.py` (426 lines)
**Pipeline step:** Step 6.7 (post-processing)
**Profile key:** `multi_frequency`

**Purpose:** Reconciles predictions, regimes, and survival probabilities from 5 frequency pipelines into a single coherent output.

**Horizon-to-frequency weights:** Each prediction horizon is dominated by the frequency most informative at that timescale:

| Horizon | D | W | M | Q | A |
|---------|---|---|---|---|---|
| 1d | 1.0 | | | | |
| 5d | 0.7 | 0.3 | | | |
| 21d | 0.3 | 0.4 | 0.3 | | |
| 3m | | 0.15 | 0.35 | 0.50 | |
| 1y | | | 0.15 | 0.35 | 0.50 |
| 2y | | | | 0.30 | 0.70 |

**Fusion methods:**
1. **Regime consensus:** Majority vote across frequencies + disagreement detection. Agreement ratio = fraction of frequencies that agree on regime label.
2. **Survival probability:** Harmonic mean (weakest-link principle). One frequency showing distress dominates the fused probability.
3. **Prediction reconciliation:** Inverse-variance weighted average of per-frequency forecasts at each horizon.

**Output:** `FusedMultiFreqResult` with:
- `regime_consensus`: Consensus regime + agreement ratio + per-frequency regimes
- `survival`: Fused probability + per-frequency probabilities + weakest frequency
- `predictions`: Per-horizon fused forecasts with contributing frequencies
- `n_frequencies_used`, `available` flag

**Input:** `MultiFrequencyResult` from runner
**Output:** `FusedMultiFreqResult` stored in `profile["multi_frequency"]`

---

## Layer 6: Staged Pipeline Architecture (NEW -- 2026-04-16)

5 modules in `operator1/stages/` that decompose the monolithic pipeline into per-model sub-stages with checkpoint save/resume via `PipelineState`.

---

### 6.1 Pipeline State

**File:** `operator1/pipeline_state.py` (401 lines)
**Purpose:** Mutable state bag replacing hundreds of local variables in `main.py`. Serializes to disk between sub-stages: DataFrames as Parquet, model results as pickle.

### 6.2 Stage Runner

**File:** `operator1/stages/runner.py` (279 lines)
**Purpose:** Dispatches sub-stages in dependency order with checkpoint save/resume. Supports stage specs: `"3"` (all stage 3), `"4.1"` (just forecasting), `"3-6"` (range), `"all"`.

### 6.3 Stage 3 -- Temporal Analysis

**File:** `operator1/stages/stage3_temporal.py` (247 lines)
**Sub-stages:** 3.1 (regime detection), 3.2 (dual regimes), 3.3 (Granger), 3.4 (transfer entropy), 3.5 (cycle decomposition), 3.6 (pattern detection), 3.7 (pre-forecast synergies)

### 6.4 Stage 4 -- Forecasting

**File:** `operator1/stages/stage4_forecasting.py` (54 lines)
**Sub-stages:** 4.1 (forecasting -- Kalman, GARCH, VAR, LSTM, Tree, ETS, baseline)

### 6.5 Stage 5 -- Forward Modeling

**File:** `operator1/stages/stage5_forward.py` (274 lines)
**Sub-stages:** 5.1 (forward pass), 5.2 (burn-out), 5.3 (walk-forward + MCS + FixedShare), 5.4 (Monte Carlo), 5.5 (copula), 5.6 (regime shift prediction)

### 6.6 Stage 6 -- Ensemble & Aggregation

**File:** `operator1/stages/stage6_ensemble.py` (414 lines)
**Sub-stages:** 6.1 (transformer), 6.2 (particle filter), 6.3 (conformal), 6.4 (DTW), 6.5 (prediction aggregation), 6.6 (SHAP), 6.7 (Sobol), 6.8 (TV Granger), 6.9 (MV Monte Carlo), 6.10 (genetic optimizer), 6.11 (OHLC predictor + predicted patterns)

### 6.7 Stage 7 -- Integration

**File:** `operator1/stages/stage7_integration.py` (371 lines)
**Sub-stages:** 7.1 (USS + scenario engine), 7.2 (retroactive calibration), 7.3 (model diagnostics), 7.4 (multi-frequency pipeline + fusion), 7.5 (hedge fund analysis)

---

### Layer 4 Addendum: Advanced HF Methods + Fusion (NEW -- 2026-04-03)

### 4.19 Advanced HF Methods

**File:** `operator1/hedge_fund/advanced_methods.py` (911 lines)
**Pipeline step:** Step 6-HF (called from engine.py)
**Profile key:** `hedge_fund.advanced_methods`

**Purpose:** 15 additional investment-grade forensic and valuation metrics that extend the base HF pipeline:

- Piotroski F-Score (9-factor binary, Piotroski 2000)
- Ohlson-Udell Bankruptcy (O-Score, Ohlson 1980)
- Altman Z''' (emerging market variant, Altman 2014)
- Forensic Cash Flow (OCF decomposition + quality scoring)
- Merton Default Probability (structural model, Black-Scholes framework)
- Springate S-Score (4-factor distress, Springate 1978)
- Zmijewski Score (probit model, Zmijewski 1984)
- Laitinen Failure Process (3-phase deterioration)
- Beneish Extended (8-factor + sector calibration)
- Revenue Quality Decomposition (organic vs inorganic)
- Working Capital Efficiency Score
- Capital Allocation Efficiency (ROIC vs WACC spread)
- Earnings Persistence (AR coefficient of NI)
- Free Cash Flow Sustainability Index
- Margin of Safety (Graham-Dodd, intrinsic vs market)

### 4.20 Cross-Pipeline Insight Fusion

**File:** `operator1/hedge_fund/fusion.py` (652 lines)
**Pipeline step:** Step 6-HF (post-scorecard)
**Profile key:** `hedge_fund.fusion`

**Purpose:** 8-method fusion layer combining HF metrics with pipeline temporal/multi-frequency results:

1. **Signal quality fusion:** IC-weighted HF metric reliability
2. **Filing frequency alignment:** HF metrics anchored to filing calendar
3. **Survival regime context:** Regime-conditional HF weighting
4. **Multi-frequency trend consistency:** Cross-frequency directional agreement
5. **Cross-method convergence scoring:** Agreement across independent methods
6. **Regime-conditional HF weighting:** Distress metrics amplified in survival
7. **Temporal decay adjustment:** Recent HF data weighted more
8. **Confidence calibration:** Inter-method agreement drives confidence

---

## Execution Order

```
Step 2:    Profile fetch -> supplement.enrich_profile()
Step 3:    PIT data fetch (parallel: income, balance, cashflow, quotes)
Step 3b:   data_reconciliation
Step 3c:   canonical_translator.pivot_to_canonical_wide()
Step 4:    Cache build (OHLCV spine + frequency interpolation)
Step 4a:   macro_provider -> macro_mapping -> macro_quadrant
Step 4a.3: conflict_risk
Step 4a.4: market_buying_power
Step 4a.5: six_derived_proxies (CH only)
Step 4b:   estimator (3-phase imputation)
Step 4c:   filing_calendar
Step 5:    derived_variables -> survival_mode -> hierarchy_weights
Step 5-USS: survival_regime_controller
Step 5a:   private_company_proxies (if no OHLCV)
Step 5b:   fuzzy_protection
Step 5d:   financial_health -> vanity
Step 5e:   entity_discovery -> graph_risk -> game_theory
Step 5f:   Linked entity data fetch -> ownership_contagion
Step 5g:   linked_aggregates
Step 5h:   peer_ranking
Step 5i:   news_sentiment
Step 5i.5: product_catalysts
Step 5i.6: product_metrics (segment_hhi, cannibalization, etc.) + geographic_metrics (geo_hhi, china_pct)
Step 5i.6b: operational_efficiency (inventory/receivables/payables turnover, SGA efficiency, capex intensity)
Step 5i.7: behavioral_signals (anchoring, disposition, attention, lottery)
Step 5i.8: complexity_signals (sample entropy, permutation entropy, LZ complexity, approx entropy)
Step 5i.9: feature_normalization (z-scores, percentiles, changes, regime z-scores) -- MUST RUN LAST
Step 5j:   adaptive_thresholds -> recalibrate survival
Step 5.5:  regime_detector (early) -> survival_timeline (enriched)
Step 5k:   adaptive_model_params
Step 5k.2: adaptive_windows
Step 6:    [TEMPORAL MODELS -- skip if --skip-models]
  6a: regime_detector (if not already run)
  6b: regime_mixer
  6c: granger_causality (informational, no pruning)
  6c.1: feature_selector (Boruta + PIMP + mRMR -> prune extra_vars)
  6d: transfer_entropy
  6e: cycle_decomposition
  6f: pattern_detector
  6g: economic_planes -> model_synergies
  6h: forecasting
  6i: forward_pass
  6j: burnout
  6k: walk_forward + MCS + FixedShare
  6l: monte_carlo
  6m: copula
  6n: transformer_forecaster
  6o: particle_filter
  6p: conformal prediction
  6q: dtw_analogs
  6r: prediction_aggregator
  6s: shap explainability
  6t: sobol sensitivity -> hierarchy feedback
  6t.1: time_varying_granger
  6t.2: multivariate_monte_carlo
  6u: genetic_optimizer
  6v: ohlc_predictor
  6v.1: detect_patterns_on_predicted_ohlc (predicted OHLC patterns)
  6w: recursive_aggregator (day-by-day autoregressive chaining, sub-stage 6.11)
Step 6-USS: forecast bounding + scenario engine + reverse stress test (Basel III)
Step 6.5:  retroactive_calibration
Step 6.6:  model_diagnostics (expected path vs actual path)
Step 6.7:  multi_frequency_runner -> frequency_fusion
Step 6-HF: [HEDGE FUND ANALYSIS -- parallel to Step 6]
  6-HF-a: fcf_quality (Tier 1)
  6-HF-b: accruals_forensics (Tier 1)
  6-HF-c: earnings_smoothing (Tier 1)
  6-HF-d: dividend_burn (Tier 2)
  6-HF-e: return_spread (Tier 2)
  6-HF-f: operating_leverage (Tier 2)
  6-HF-g: obs_risk (Tier 3)
  6-HF-h: asset_quality (Tier 3)
  6-HF-i: leverage_stress (Tier 3, uses mc_result)
  6-HF-j: momentum_composite (Tier 4)
  6-HF-k: growth_quality (Tier 4)
  6-HF-l: earnings_surprise (Tier 4, uses filing_calendar)
  6-HF-m: dcf_valuation (Tier 5, uses mc_result + multi_freq)
  6-HF-n: valuation_quality (Tier 5, uses peer_ranking)
  6-HF-o: peg_composite (Tier 5)
  6-HF-q: thesis_scorecard (synthesize all 15)
  6-HF-r: position_signal (final actionable output)
Step 7:    profile_builder (+ hedge_fund section injection)
Step 8:    report_generator + triage_card (USS)
```

---

## Summary

| Layer | Modules | Lines | Description |
|-------|---------|-------|-------------|
| Features | **21** | **~14,500** | Raw cache -> enriched features (96 derived vars + options + cross-asset + events + behavioral + complexity + normalization + product_metrics + OCR) |
| Analysis | 10 | **~7,200** | Survival flags (+ velocity + uncertainty + ensemble distress), hierarchy (+ entropy blend), protection, adaptive calibration, USS (+ soft transition + reverse stress) |
| Temporal | **27** | **~20,500** | Regime, feature selection (Boruta/PIMP/mRMR), forecasting (ETS, GARCH-X, parallel tree), MC (+ jump-diffusion + antithetic), recursive aggregator, uncertainty, aggregation |
| USS | 2 | **~1,300** | Unified Survival System (controller + scenario engine + reverse stress test) |
| Monitoring/Diagnostics | 1 | ~875 | Model expected path vs actual path diagnostics |
| **Hedge Fund** | **9** | **~5,500** | **15 base metrics + 19 advanced methods (Piotroski, DuPont, Kelly, Altman Z''', etc.) + 8-method fusion + scorecard + position signal** |
| **Multi-Frequency** | **3** | **~2,600** | **5-frequency pipeline (A/Q/M/W/D) with 13-method fusion + cross-frequency momentum (Moskowitz et al.)** |
| **Staged Pipeline** | **6** | **~2,230** | **PipelineState + runner (graceful degradation + timeout + validation) + 5 stage modules (35 sub-stages with checkpoint save/resume)** |
| **Total** | **~78** | **~54,700** | |

All ~78 modules wired in main.py (or via staged runner). All results stored in profile_builder. HF results in `profile["hedge_fund"]`. Multi-frequency results in `profile["multi_frequency"]`. Staged pipeline accessible via `--stage 3-6 --run-dir cache/AAPL`. Staged backtest compiler available via `python run_backtest_staged.py`.

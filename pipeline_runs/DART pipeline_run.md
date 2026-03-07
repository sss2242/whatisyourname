2026-03-07 16:39:33 | INFO    | operator1.main | ============================================================
2026-03-07 16:39:33 | INFO    | operator1.main | OPERATOR 1 -- Point-in-Time Financial Analysis
2026-03-07 16:39:33 | INFO    | operator1.main | ============================================================
2026-03-07 16:39:33 | INFO    | operator1.secrets_loader | Loaded environment from /home/manem/githubu-isu-meanu/.env
2026-03-07 16:39:33 | INFO    | operator1.secrets_loader | All 9 required API keys loaded successfully.
2026-03-07 16:39:33 | INFO    | operator1.main | 
2026-03-07 16:39:33 | INFO    | operator1.main | Step 1: Selecting data source and company...
2026-03-07 16:39:33 | INFO    | operator1.main | Market: South Korea (KOSPI / KOSDAQ) -- DART
2026-03-07 16:39:38 | INFO    | operator1.clients.kr_dart_wrapper | dart-fss initialized with API key
2026-03-07 16:46:47 | INFO    | operator1.main | Company found: LF (093050)
2026-03-07 16:46:47 | INFO    | operator1.main | Target: LF (093050) via DART
2026-03-07 16:46:47 | INFO    | operator1.main | Macro source: KOSIS (Korean Statistical Information) (South Korea)
2026-03-07 16:46:47 | INFO    | operator1.main | 
2026-03-07 16:46:47 | INFO    | operator1.main | Step 2: Fetching company profile from DART...
2026-03-07 16:46:49 | INFO    | operator1.main | Profile loaded: (주)LF (093050), sector=14111
2026-03-07 16:46:49 | INFO    | operator1.main | 
2026-03-07 16:46:49 | INFO    | operator1.main | Step 3: Fetching point-in-time financial data...
2026-03-07 16:46:49 | INFO    | operator1.main | Quotes: 0 rows
2026-03-07 16:48:34 | INFO    | operator1.main | Income: 186 rows
2026-03-07 16:48:35 | INFO    | operator1.main | Cashflow: 186 rows
2026-03-07 16:48:35 | INFO    | operator1.main | Balance: 186 rows
2026-03-07 16:48:35 | INFO    | operator1.main | PIT source DART does not provide OHLCV -- fetching from OHLCV provider.
2026-03-07 16:48:36 | INFO    | operator1.clients.ohlcv_provider | Per-region OHLCV (pykrx) returned empty for 093050; falling back to yfinance
2026-03-07 16:48:38 | INFO    | operator1.clients.ohlcv_yfinance | yfinance fetched 1222 rows for 093050.KS
2026-03-07 16:48:38 | INFO    | operator1.main | OHLCV fetched from yfinance provider: 1222 rows
2026-03-07 16:48:38 | WARNING | operator1.quality.data_reconciliation | Found 80 filings where filing_date < report_date (time travel) -- correcting to report_date + 1 day
2026-03-07 16:48:38 | INFO    | operator1.quality.data_reconciliation | Removed 179 duplicate filings (same report_date)
2026-03-07 16:48:38 | WARNING | operator1.quality.data_reconciliation | Found 80 filings where filing_date < report_date (time travel) -- correcting to report_date + 1 day
2026-03-07 16:48:38 | INFO    | operator1.quality.data_reconciliation | Removed 179 duplicate filings (same report_date)
2026-03-07 16:48:38 | WARNING | operator1.quality.data_reconciliation | Found 80 filings where filing_date < report_date (time travel) -- correcting to report_date + 1 day
2026-03-07 16:48:38 | INFO    | operator1.quality.data_reconciliation | Removed 179 duplicate filings (same report_date)
2026-03-07 16:48:38 | INFO    | operator1.quality.data_reconciliation | Reconciliation complete: 3 statements, no issues
2026-03-07 16:48:38 | INFO    | operator1.main | Data reconciliation: clean
2026-03-07 16:48:38 | INFO    | operator1.main | Pivoted income to wide: 7 periods x 6 columns
2026-03-07 16:48:38 | INFO    | operator1.main | Pivoted balance to wide: 7 periods x 6 columns
2026-03-07 16:48:38 | INFO    | operator1.main | Pivoted cashflow to wide: 7 periods x 6 columns
2026-03-07 16:48:38 | INFO    | operator1.main | 
2026-03-07 16:48:38 | INFO    | operator1.main | Step 4: Building daily cache from PIT data...
2026-03-07 16:48:38 | INFO    | operator1.main | Merged income data: 4 columns (4 new)
2026-03-07 16:48:38 | INFO    | operator1.main | Merged balance data: 4 columns (0 new)
2026-03-07 16:48:38 | INFO    | operator1.main | Merged cashflow data: 4 columns (0 new)
2026-03-07 16:48:38 | INFO    | operator1.main | Cache built: 1222 rows x 9 columns
2026-03-07 16:48:38 | INFO    | operator1.main | 
2026-03-07 16:48:38 | INFO    | operator1.main | Step 4a: Fetching macro data from KOSIS (Korean Statistical Information)...
2026-03-07 16:48:43 | INFO    | operator1.clients.macro_kosis | Korea macro via FRED: 3/5 indicators
2026-03-07 16:48:43 | INFO    | operator1.main | Macro data: 3 indicators fetched
2026-03-07 16:48:43 | INFO    | operator1.main |     [OK] inflation_rate_yoy: 1 observations
2026-03-07 16:48:43 | INFO    | operator1.main |     [OK] interest_rate: 23 observations
2026-03-07 16:48:43 | INFO    | operator1.main |     [OK] exchange_rate: 517 observations
2026-03-07 16:48:43 | INFO    | operator1.steps.macro_mapping |   inflation -> inflation_rate_yoy: 1 yearly observations
2026-03-07 16:48:43 | INFO    | operator1.steps.macro_mapping |   interest_rate -> real_interest_rate: 3 yearly observations
2026-03-07 16:48:43 | INFO    | operator1.steps.macro_mapping | MacroDataset built: 2 indicators available, 3 missing (country=KR)
2026-03-07 16:48:43 | INFO    | operator1.main | MacroDataset built: 2 indicators, 3 missing
2026-03-07 16:48:43 | INFO    | operator1.features.macro_quadrant | Computing macro quadrant classification...
2026-03-07 16:48:43 | INFO    | operator1.features.macro_quadrant | Macro quadrant: unknown, 0 days classified, 0 transitions, distribution={}
2026-03-07 16:48:43 | INFO    | operator1.main | Macro quadrant: N/A, stability=0.000
2026-03-07 16:48:43 | INFO    | operator1.main | 
2026-03-07 16:48:43 | INFO    | operator1.main | --- API Data Validation Summary ---
2026-03-07 16:48:43 | INFO    | operator1.main |   PIT source : DART (South Korea, KOSPI / KOSDAQ)
2026-03-07 16:48:43 | INFO    | operator1.main |     [OK] profile
2026-03-07 16:48:43 | INFO    | operator1.main |     [OK] income_statement
2026-03-07 16:48:43 | INFO    | operator1.main |     [OK] balance_sheet
2026-03-07 16:48:43 | INFO    | operator1.main |     [OK] cashflow_statement
2026-03-07 16:48:43 | INFO    | operator1.main |     [OK] quotes_ohlcv
2026-03-07 16:48:43 | INFO    | operator1.main |   Macro source: KOSIS (Korean Statistical Information) (South Korea)
2026-03-07 16:48:43 | INFO    | operator1.main |     [MISSING] macro.gdp
2026-03-07 16:48:43 | INFO    | operator1.main |     [MISSING] macro.inflation
2026-03-07 16:48:43 | INFO    | operator1.main |     [OK]      macro.interest_rate -- 23 observations (2024-03-01 to 2026-01-01)
2026-03-07 16:48:43 | INFO    | operator1.main |     [MISSING] macro.unemployment
2026-03-07 16:48:43 | INFO    | operator1.main |     [MISSING] macro.currency
2026-03-07 16:48:43 | INFO    | operator1.main |   MacroDataset: 2 indicators ready, 3 missing ['gdp_growth', 'unemployment_rate', 'official_exchange_rate_lcu_per_usd']
2026-03-07 16:48:43 | INFO    | operator1.main | --- End Validation Summary ---
2026-03-07 16:48:43 | INFO    | operator1.main | 
2026-03-07 16:48:43 | INFO    | operator1.main | 
2026-03-07 16:48:43 | INFO    | operator1.main | Step 4b: Running estimation (Sudoku inference)...
2026-03-07 16:48:43 | INFO    | operator1.estimation.estimator | Running Phase 1: Deterministic identity fill ...
2026-03-07 16:48:43 | INFO    | operator1.estimation.estimator | Pass 1 complete: 0 total cells filled across 0 variables
2026-03-07 16:48:43 | INFO    | operator1.estimation.estimator | Running Phase 2: Missingness classification for 3 variables ...
2026-03-07 16:48:43 | INFO    | operator1.estimation.missingness_classifier | Missingness classification: 2693 MAR/MCAR cells, 0 MNAR cells across 3 variables
2026-03-07 16:48:43 | INFO    | operator1.estimation.estimator | Classification result: 3 vars with MAR, 0 vars with MNAR
2026-03-07 16:48:43 | INFO    | operator1.estimation.estimator | Running Phase 3a: MAR estimator (MICE+GP+MatrixCompletion) for 3 variables ...
2026-03-07 16:48:43 | INFO    | operator1.estimation.missing_data_estimator | Running MAR estimator for 3 variables using 5 features
2026-03-07 16:48:51 | INFO    | operator1.estimation.missing_data_estimator | MAR estimation complete: 2693 values estimated across 3 variables
2026-03-07 16:48:51 | INFO    | operator1.estimation.estimator | Estimation complete: Phase1 filled 0 cells, Phase2+3 estimated 3 variables
2026-03-07 16:48:51 | INFO    | operator1.main | Estimation complete: method=split, variables=3
2026-03-07 16:48:51 | INFO    | operator1.main | Estimation coverage saved: cache/estimation_coverage.json
2026-03-07 16:48:51 | INFO    | operator1.main | Filing calendar: expected=8, actual=3 (38%), freq=semiannual, stale=True (age=158d)
2026-03-07 16:48:51 | WARNING | operator1.main | STALE DATA: Latest filing is 158 days old (threshold: 135 days for kr_dart)
2026-03-07 16:48:51 | WARNING | operator1.main | Filing gaps detected: 1 gaps in 2-year window
2026-03-07 16:48:51 | INFO    | operator1.main | 
2026-03-07 16:48:51 | INFO    | operator1.main | Step 5: Computing derived features...
2026-03-07 16:48:51 | INFO    | operator1.features.derived_variables | Derived variables computed: 111 new columns
2026-03-07 16:48:51 | INFO    | operator1.main | Features computed: 147 columns
2026-03-07 16:48:51 | INFO    | operator1.analysis.survival_mode | Company survival flag: 0 / 1222 days triggered (0.0%)
2026-03-07 16:48:51 | INFO    | operator1.analysis.hierarchy_weights | Regime distribution:
survival_regime
normal    1222
2026-03-07 16:48:51 | INFO    | operator1.analysis.hierarchy_weights | Hierarchy weights computed: 1222 days, 1 unique regimes
2026-03-07 16:48:51 | INFO    | operator1.main | Survival mode: 0 days flagged
2026-03-07 16:48:51 | INFO    | operator1.analysis.fuzzy_protection | Fuzzy protection: sector=0.10, mean_economic=0.00, mean_policy=0.00, mean_degree=0.10
2026-03-07 16:48:51 | INFO    | operator1.main | Fuzzy protection: degree=0.100 (unprotected)
2026-03-07 16:48:51 | INFO    | operator1.models.financial_health | Computing financial health scores...
2026-03-07 16:48:51 | INFO    | operator1.models.financial_health | Altman Z-Score: 0.00 (unknown)
2026-03-07 16:48:51 | INFO    | operator1.models.financial_health | Financial health: 1222 days scored, composite=44.4 (Fair), cols=9
2026-03-07 16:48:51 | INFO    | operator1.main | Financial health: composite=44.4 (Fair), 9 columns added
2026-03-07 16:48:51 | INFO    | operator1.analysis.vanity | Vanity percentage (legacy): 0 / 1222 days with non-zero vanity (max=0.0%)
2026-03-07 16:48:51 | INFO    | operator1.analysis.vanity | Vanity score v2: 0 / 1222 days scored (mean=0.0, label=Unknown)
2026-03-07 16:48:51 | INFO    | operator1.main | Vanity scores computed
2026-03-07 16:48:51 | INFO    | operator1.clients.llm_base | Gemini model 'gemini-1.5-flash': max_output=8192, context=1048576, report_capable=True
2026-03-07 16:48:51 | INFO    | operator1.clients.llm_base | Gemini model 'gemini-1.5-flash': max_output=8192, context=1048576, report_capable=True
2026-03-07 16:48:51 | INFO    | operator1.clients.llm_base | Gemini model 'gemini-1.5-flash': max_output=8192, context=1048576, report_capable=True
2026-03-07 16:48:51 | INFO    | operator1.clients.llm_base | Gemini model 'gemini-1.5-flash': max_output=8192, context=1048576, report_capable=True
2026-03-07 16:48:51 | INFO    | operator1.clients.llm_base | Gemini model 'gemini-1.5-flash': max_output=8192, context=1048576, report_capable=True
2026-03-07 16:48:51 | WARNING | operator1.clients.llm_base | Claude model 'gemini-1.5-flash' not in known registry. Using 'claude-sonnet-4-20250514' instead. Known models: claude-opus-4-20250514, claude-sonnet-4-20250514, claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
2026-03-07 16:48:51 | INFO    | operator1.clients.llm_factory | LLM key pool: 5 primary (gemini) + 1 fallback (claude) keys. Active: Gemini / gemini-1.5-flash
2026-03-07 16:48:51 | INFO    | operator1.main | 
2026-03-07 16:48:51 | INFO    | operator1.main | Step 5e: Discovering linked entities via Gemini...
2026-03-07 16:48:51 | INFO    | operator1.clients.eu_esef_wrapper | pyesef not available; using filings.xbrl.org API only
2026-03-07 16:48:51 | INFO    | operator1.clients.eu_esef_wrapper | pyesef not available; using filings.xbrl.org API only
2026-03-07 16:48:51 | INFO    | operator1.clients.eu_esef_wrapper | pyesef not available; using filings.xbrl.org API only
2026-03-07 16:48:52 | INFO    | operator1.clients.jp_jquants_wrapper | J-Quants ClientV2 initialized successfully
2026-03-07 16:48:54 | INFO    | operator1.clients.kr_dart_wrapper | dart-fss initialized with API key
2026-03-07 16:48:54 | INFO    | operator1.clients.br_cvm_wrapper | pycvm not available; using direct CVM API
2026-03-07 16:48:54 | INFO    | operator1.clients.cn_sse | baostock available for Chinese market data
2026-03-07 16:48:54 | INFO    | operator1.clients.eu_esef_wrapper | pyesef not available; using filings.xbrl.org API only
2026-03-07 16:48:54 | INFO    | operator1.clients.eu_esef_wrapper | pyesef not available; using filings.xbrl.org API only
2026-03-07 16:48:54 | INFO    | operator1.clients.eu_esef_wrapper | pyesef not available; using filings.xbrl.org API only
2026-03-07 16:48:54 | INFO    | operator1.clients.eu_esef_wrapper | pyesef not available; using filings.xbrl.org API only
2026-03-07 16:48:54 | INFO    | operator1.steps.entity_discovery | Cross-region discovery: 25 PIT clients available
2026-03-07 16:48:54 | ERROR   | operator1.clients.llm_base | Gemini API error 404 (non-retryable): {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:48:54 | WARNING | operator1.clients.llm_base | Gemini request error on attempt 1/5 (0.4s): Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}
 -- retrying in 2.0s
2026-03-07 16:48:58 | ERROR   | operator1.clients.llm_base | Gemini API error 404 (non-retryable): {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:48:58 | WARNING | operator1.clients.llm_base | Gemini request error on attempt 2/5 (0.4s): Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}
 -- retrying in 4.0s
2026-03-07 16:49:03 | ERROR   | operator1.clients.llm_base | Gemini API error 404 (non-retryable): {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:49:03 | WARNING | operator1.clients.llm_base | Gemini request error on attempt 3/5 (0.4s): Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}
 -- retrying in 8.0s
2026-03-07 16:49:11 | ERROR   | operator1.clients.llm_base | Gemini API error 404 (non-retryable): {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:49:11 | WARNING | operator1.clients.llm_base | Gemini request error on attempt 4/5 (0.4s): Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}
 -- retrying in 16.0s
2026-03-07 16:49:28 | ERROR   | operator1.clients.llm_base | Gemini API error 404 (non-retryable): {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:49:28 | WARNING | operator1.clients.llm_base | Gemini request error on attempt 5/5 (0.5s): Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}
 -- retrying in 32.0s
2026-03-07 16:50:00 | WARNING | operator1.clients.llm_base | Gemini linked-entity proposal failed: Gemini: all 5 retries exhausted. Last error: Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:50:00 | INFO    | operator1.steps.entity_discovery | Gemini proposed entities for 0 groups: {}
2026-03-07 16:50:00 | INFO    | operator1.steps.entity_discovery | Discovery complete: 4 entities across 1 groups, 0 search calls, 0 dropped
2026-03-07 16:50:00 | INFO    | operator1.main | Linked entities discovered: 4
2026-03-07 16:50:00 | WARNING | operator1.models.graph_risk | Graph risk computation failed: 'LinkedEntity' object has no attribute 'get'
2026-03-07 16:50:00 | INFO    | operator1.main | Graph risk: 0 nodes, centrality=0.000
2026-03-07 16:50:00 | INFO    | operator1.main | Game theory: monopoly, pressure=0.000
2026-03-07 16:50:00 | INFO    | operator1.main | 
2026-03-07 16:50:00 | INFO    | operator1.main | Step 5f: Fetching linked entity data...
2026-03-07 16:50:04 | INFO    | operator1.main | Linked entity data: 0/4 entities fetched
2026-03-07 16:50:04 | INFO    | operator1.features.news_sentiment | Computing news sentiment for 093050...
2026-03-07 16:50:05 | INFO    | operator1.features.news_sentiment | No news articles available for sentiment scoring
2026-03-07 16:50:05 | INFO    | operator1.main | Sentiment: no articles available for scoring
2026-03-07 16:50:05 | INFO    | operator1.main | 
2026-03-07 16:50:05 | INFO    | operator1.main | Step 5.5: Enriched survival timeline...
2026-03-07 16:50:05 | INFO    | operator1.models.regime_detector | Starting regime detection and structural break analysis...
2026-03-07 16:50:06 | INFO    | operator1.models.regime_detector | HMM full covariance failed ('covars' must be symmetric, positive-definite), retrying with diagonal covariance
2026-03-07 16:50:06 | INFO    | operator1.models.regime_detector | HMM fit: 4 regimes, 1217 observations, cov=diag, converged=True
2026-03-07 16:50:06 | INFO    | operator1.models.regime_detector | GMM fit: 4 components, 1221 observations, converged=True
2026-03-07 16:50:12 | INFO    | operator1.models.regime_detector | PELT detected 12 structural breaks (penalty=10.0)
2026-03-07 16:50:13 | INFO    | operator1.models.regime_detector | BCP detected 0 change points (lambda=200, threshold=0.50)
2026-03-07 16:50:13 | INFO    | operator1.models.regime_detector | Regime label mapping: {2: 'bear', 3: 'low_vol', 1: 'high_vol', 0: 'bull'}
2026-03-07 16:50:13 | INFO    | operator1.models.regime_detector | Regime detection complete: 12 structural breaks, regime distribution: {'bear': 898, 'high_vol': 307, 'bull': 12, nan: 5}
2026-03-07 16:50:13 | INFO    | operator1.models.regime_detector | Early regime detection complete for enriched survival timeline
2026-03-07 16:50:13 | INFO    | operator1.main | Early regime detection complete
2026-03-07 16:50:13 | INFO    | operator1.analysis.survival_timeline | Survival timeline computed: 1222 days, 0 switches, mean_stability=1.000, distribution={'normal': '100.00%'}
2026-03-07 16:50:13 | INFO    | operator1.analysis.survival_timeline | Enriched survival timeline: 1222 days, mean_intensity=0.283, regime_available=True, confidence=[min=0.476, mean=0.975, max=1.000], states={'market_stress': '73.5%', 'elevated_risk': '25.1%', 'stable_growth': '1.4%'}
2026-03-07 16:50:13 | INFO    | operator1.main | Enriched survival timeline: mean_intensity=0.283, regime_available=True, states={'market_stress': '73.5%', 'elevated_risk': '25.1%', 'stable_growth': '1.4%'}
2026-03-07 16:50:13 | INFO    | operator1.main | 
2026-03-07 16:50:13 | INFO    | operator1.main | Step 6: Running temporal models...
2026-03-07 16:50:13 | INFO    | operator1.main | Extra variables for temporal models (21): ['macro_growth_vs_trend', 'macro_inflation_vs_target', 'macro_quadrant_numeric', 'macro_quadrant_stability_21d', 'fh_liquidity_score', 'fh_solvency_score', 'fh_stability_score', 'fh_profitability_score', 'fh_growth_score', 'fh_composite_score']
2026-03-07 16:50:13 | INFO    | operator1.main | Regime detection: using results from Step 5.5 (early detection)
2026-03-07 16:50:13 | INFO    | operator1.models.regime_mixer | Dual regime classification complete: market={'bear': 898, 'high_vol': 307, 'bull': 12}, fundamental={'healthy': 1222}
2026-03-07 16:50:13 | INFO    | operator1.main | Dual regime classification complete
2026-03-07 16:50:14 | WARNING | operator1.models.causality | Insufficient data for transfer entropy (0 rows, 20 vars)
2026-03-07 16:50:14 | INFO    | operator1.main | Transfer entropy computed: 380 variable pairs
2026-03-07 16:50:14 | INFO    | operator1.models.cycle_decomposition | pywt not installed, skipping wavelet decomposition
2026-03-07 16:50:14 | INFO    | operator1.models.cycle_decomposition | Cycle decomposition: 5 dominant cycles, trend=0.04, noise=0.01
2026-03-07 16:50:14 | INFO    | operator1.main | Cycle decomposition complete
2026-03-07 16:50:14 | INFO    | operator1.models.pattern_detector | Pattern detection: 64 patterns found in 130 days (7 bullish, 1 bearish)
2026-03-07 16:50:14 | INFO    | operator1.main | Candlestick patterns detected
2026-03-07 16:50:14 | INFO    | operator1.analysis.economic_planes | Economic plane: 14111 -> manufacturing (Manufacturing & Production)
2026-03-07 16:50:14 | INFO    | operator1.models.model_synergies | Injected cycle phase feature: cycle_phase_244d (period=244d)
2026-03-07 16:50:14 | INFO    | operator1.models.model_synergies | Injected cycle phase feature: cycle_phase_203d (period=203d)
2026-03-07 16:50:14 | INFO    | operator1.models.model_synergies | Injected cycle phase feature: cycle_phase_101d (period=101d)
2026-03-07 16:50:14 | INFO    | operator1.models.model_synergies | Unified causal network: 0 pairs (0 Granger, 0 TE, 0 both), 0 retained, 0 pruned, density=0.000
2026-03-07 16:50:14 | INFO    | operator1.models.model_synergies | Plane-aware weights: plane=manufacturing, top_boosted=['granger_causality', 'forecasting', 'transfer_entropy'], top_dampened=['copula', 'monte_carlo', 'transformer']
2026-03-07 16:50:14 | INFO    | operator1.main | Pre-forecasting synergies applied
2026-03-07 16:50:14 | INFO    | operator1.models.forecasting | Starting forecasting pipeline...
2026-03-07 16:50:14 | INFO    | operator1.models.forecasting | Forecasting: 6 variables not in cache, skipping: ['cash_and_equivalents_asof', 'debt_to_equity', 'free_cash_flow_ttm', 'operating_cash_flow_asof', 'pe_ratio']
2026-03-07 16:50:15 | INFO    | operator1.models.forecasting | GARCH(1,1) fit: 1037 train, 184 test, MAE=0.011203, RMSE=0.014998
2026-03-07 16:50:16 | WARNING | operator1.models.forecasting | Insufficient observations for Kalman (0 < 30)
2026-03-07 16:50:16 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:22 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:22 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:23 | INFO    | operator1.models.forecasting | Kalman fit: 1038 train, 184 test, MAE=0.000000, RMSE=0.000000
2026-03-07 16:50:24 | INFO    | operator1.models.forecasting | Kalman fit: 107 train, 19 test, MAE=0.000000, RMSE=0.000000
2026-03-07 16:50:24 | INFO    | operator1.models.forecasting | Kalman fit: 79 train, 15 test, MAE=0.000000, RMSE=0.000000
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | Kalman fit: 53 train, 10 test, MAE=0.000000, RMSE=0.000000
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for Kalman (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for Kalman (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for Kalman (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 1-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=0.005367, RMSE=0.006587
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 1-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=0.043259, RMSE=0.056766
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 1-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=23744.701292, RMSE=36516.836830
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 1-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=0.000000, RMSE=0.000000
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 1-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=14.950700, RMSE=18.092440
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 1-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=14.950700, RMSE=18.092440
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 1-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=11.256339, RMSE=13.973658
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 1-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=18.392781, RMSE=23.235761
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 1-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=0.000000, RMSE=0.000000
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for VAR (0 < 50)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for LSTM (0 < 100) -- falling back to tree/linear
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | Insufficient observations for tree ensemble (0 < 30)
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 2-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=0.029895, RMSE=0.052438
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 2-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=0.045976, RMSE=0.095557
2026-03-07 16:50:25 | WARNING | operator1.models.forecasting | VAR fitting failed: 2-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:25 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=0.035143, RMSE=0.044853
2026-03-07 16:50:26 | WARNING | operator1.models.forecasting | VAR fitting failed: 2-th leading minor of the array is not positive definite -- falling back to AR(1)
2026-03-07 16:50:26 | INFO    | operator1.models.forecasting | AR(1) fallback fit: MAE=0.000000, RMSE=0.000000
2026-03-07 16:50:26 | INFO    | operator1.models.forecasting | Forecasting complete: 34 variables forecasted, 4 model types failed, model distribution: {'garch': 1, 'baseline_zero': 19, 'kalman': 1, 'ar1': 13}
2026-03-07 16:50:26 | WARNING | operator1.models.forecasting | Forecasting: 19 variables had 0 non-NaN observations (all-NaN columns, likely missing from data source): ['cash_ratio', 'net_debt_to_ebitda', 'interest_coverage', 'current_ratio', 'gross_margin', 'operating_margin', 'net_margin', 'ev_to_ebitda', 'macro_growth_vs_trend', 'macro_quadrant_numeric', 'macro_quadrant_stability_21d', 'fh_liquidity_score', 'fh_solvency_score', 'fh_profitability_score', 'fh_growth_score', 'sentiment_score', 'sentiment_momentum_5d', 'sentiment_momentum_21d', 'sentiment_volatility_21d']
2026-03-07 16:50:26 | INFO    | operator1.models.forecasting | Collected 210 real residual samples for conformal calibration
2026-03-07 16:50:26 | INFO    | operator1.main | Forecasting complete
2026-03-07 16:50:26 | INFO    | operator1.models.forecasting | Starting forward pass (warmup=60 days)...
2026-03-07 16:50:26 | INFO    | operator1.models.forecasting | Conformal calibrator initialised (target coverage=90%%)
2026-03-07 16:50:26 | INFO    | operator1.models.forecasting | Candlestick pattern features injected for forward pass
2026-03-07 16:50:26 | INFO    | operator1.models.cycle_decomposition | pywt not installed, skipping wavelet decomposition
2026-03-07 16:50:26 | INFO    | operator1.models.cycle_decomposition | Cycle decomposition: 5 dominant cycles, trend=0.04, noise=0.01
2026-03-07 16:50:26 | INFO    | operator1.models.pid_controller | Created PID bank with 33 controllers
2026-03-07 16:50:26 | INFO    | operator1.models.forecasting | PID bank initialised for 33 variables
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for total_debt_asof: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for volatility_21d: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for drawdown_252d: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for volume: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for fh_stability_score: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for fh_composite_score: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for fh_composite_delta_5d: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for sentiment_count: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for survival_intensity: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for regime_confidence: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for regime_transition_prob: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.models.forecasting | TFT fitting failed for stability_score_21d: module 'torch' has no attribute 'elu'
2026-03-07 16:50:32 | WARNING | operator1.main | Forward pass failed: 0
2026-03-07 16:50:32 | INFO    | operator1.models.forecasting | Starting burn-out (window=130, max_iter=10, patience=3)...
2026-03-07 16:50:32 | INFO    | operator1.models.forecasting | Burn-out iteration 1/10
2026-03-07 16:50:32 | INFO    | operator1.models.forecasting | Starting forward pass (warmup=60 days)...
2026-03-07 16:50:32 | INFO    | operator1.models.forecasting | Conformal calibrator initialised (target coverage=90%%)
2026-03-07 16:50:32 | INFO    | operator1.models.forecasting | Candlestick pattern features injected for forward pass
2026-03-07 16:50:32 | INFO    | operator1.models.cycle_decomposition | pywt not installed, skipping wavelet decomposition
2026-03-07 16:50:32 | INFO    | operator1.models.cycle_decomposition | Cycle decomposition: 5 dominant cycles, trend=0.23, noise=0.02
2026-03-07 16:50:32 | INFO    | operator1.models.pid_controller | Created PID bank with 33 controllers
2026-03-07 16:50:32 | INFO    | operator1.models.forecasting | PID bank initialised for 33 variables
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for total_debt_asof: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for volatility_21d: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for drawdown_252d: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for volume: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for macro_inflation_vs_target: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for fh_stability_score: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for fh_composite_score: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for fh_composite_delta_5d: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for fh_composite_delta_21d: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for sentiment_count: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for survival_intensity: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for regime_confidence: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for regime_transition_prob: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.models.forecasting | TFT fitting failed for stability_score_21d: module 'torch' has no attribute 'elu'
2026-03-07 16:50:33 | WARNING | operator1.main | Burn-out failed: 0
2026-03-07 16:50:33 | INFO    | operator1.analysis.survival_timeline | Survival timeline computed: 1222 days, 0 switches, mean_stability=1.000, distribution={'normal': '100.00%'}
2026-03-07 16:50:33 | INFO    | operator1.models.walk_forward | Starting walk-forward: 1222 days, 5 variables, 4 models
2026-03-07 16:50:33 | ERROR   | operator1.models.walk_forward | Walk-forward failed: object of type 'SurvivalTimelineResult' has no len()
2026-03-07 16:50:33 | INFO    | operator1.models.monte_carlo | Starting Monte Carlo simulation: 10000 paths, 4 horizons...
2026-03-07 16:50:33 | INFO    | operator1.models.monte_carlo | Simulating horizon '1d' (1 steps)...
2026-03-07 16:50:34 | INFO    | operator1.models.monte_carlo | Horizon '1d': survival_prob=1.0000 (p5=1.0000, p95=1.0000), ESS=4325.6
2026-03-07 16:50:34 | INFO    | operator1.models.monte_carlo | Simulating horizon '5d' (5 steps)...
2026-03-07 16:50:37 | INFO    | operator1.models.monte_carlo | Horizon '5d': survival_prob=1.0000 (p5=1.0000, p95=1.0000), ESS=73.8
2026-03-07 16:50:37 | INFO    | operator1.models.monte_carlo | Simulating horizon '21d' (21 steps)...
2026-03-07 16:50:47 | INFO    | operator1.models.monte_carlo | Horizon '21d': survival_prob=1.0000 (p5=1.0000, p95=1.0000), ESS=6962.3
2026-03-07 16:50:47 | INFO    | operator1.models.monte_carlo | Simulating horizon '252d' (252 steps)...
2026-03-07 16:52:38 | INFO    | operator1.models.monte_carlo | Horizon '252d': survival_prob=0.9691 (p5=0.9659, p95=0.9725), ESS=7000.0
2026-03-07 16:52:38 | INFO    | operator1.models.monte_carlo | Monte Carlo complete: 10000 paths x 4 horizons, overall survival_mean=0.9923, p5=0.9738, p95=1.0000, current_regime='high_vol', IS_used=True
2026-03-07 16:52:38 | INFO    | operator1.main | Monte Carlo simulation complete
2026-03-07 16:52:38 | INFO    | operator1.main | Copula analysis complete
2026-03-07 16:52:39 | INFO    | operator1.models.particle_filter | Particle filter: 1222 steps, 2 states, 0 resamples, final ESS=1000
2026-03-07 16:52:39 | INFO    | operator1.main | Particle filter complete
2026-03-07 16:52:39 | WARNING | operator1.main | Conformal prediction failed: 'ConformalCalibrator' object has no attribute 'update'
2026-03-07 16:52:40 | INFO    | operator1.models.dtw_analogs | DTW analogs: found 5 matches (method=dtw_dtaidistance), empirical return: mean=-6.84%, median=-7.11%, range=[-8.76%, -4.76%]
2026-03-07 16:52:40 | INFO    | operator1.main | DTW analogs complete
2026-03-07 16:52:40 | INFO    | operator1.models.prediction_aggregator | Starting prediction aggregation pipeline...
2026-03-07 16:52:40 | INFO    | operator1.models.prediction_aggregator | Regime probability blending applied to ensemble weights
2026-03-07 16:52:40 | INFO    | operator1.models.prediction_aggregator | Technical Alpha mask: last_close=20750.0000, vol=0.029120, estimated_low=19843.6373
2026-03-07 16:52:40 | INFO    | operator1.models.prediction_aggregator | Saved predictions to cache/predictions.parquet (136 rows)
2026-03-07 16:52:40 | INFO    | operator1.models.prediction_aggregator | Saved prediction summary to cache/prediction_summary.json
2026-03-07 16:52:40 | INFO    | operator1.models.prediction_aggregator | Prediction aggregation complete: 34 variables, 4 horizons, 1 models available (4 failed), survival_mean=0.9923, regime='high_vol', blend='soft_probability', extras=[dtw_analogs=5]
2026-03-07 16:52:40 | INFO    | operator1.main | Predictions aggregated (with 4 sibling module results)
2026-03-07 16:52:41 | WARNING | operator1.models.explainability | Insufficient data for SHAP (0 < 30)
2026-03-07 16:52:41 | INFO    | operator1.main | SHAP explanations computed
2026-03-07 16:52:41 | INFO    | operator1.main | Sobol sensitivity analysis complete
2026-03-07 16:52:41 | INFO    | operator1.models.genetic_optimizer | GA optimization complete: 17 generations, best_fitness=-0.020027, converged=True, weights={'kalman': 1.0, 'var': 0.0, 'baseline': 0.0, 'garch': 0.0, 'lstm': 0.0, 'tree': 0.0, 'transformer': 0.0}
2026-03-07 16:52:41 | INFO    | operator1.main | GA optimization complete
2026-03-07 16:52:41 | INFO    | operator1.models.ohlc_predictor | OHLC prediction complete: 1+5+21+252 = 279 candles generated
2026-03-07 16:52:41 | INFO    | operator1.main | OHLC prediction complete
2026-03-07 16:52:41 | INFO    | operator1.main | 
2026-03-07 16:52:41 | INFO    | operator1.main | Step 7: Building company profile...
2026-03-07 16:52:41 | INFO    | operator1.report.profile_builder | Building company profile JSON...
2026-03-07 16:52:41 | INFO    | operator1.analysis.ethical_filters | Computing ethical filters...
2026-03-07 16:52:41 | INFO    | operator1.analysis.ethical_filters |   purchasing_power: PASS
2026-03-07 16:52:41 | INFO    | operator1.analysis.ethical_filters |   solvency: UNAVAILABLE
2026-03-07 16:52:41 | INFO    | operator1.analysis.ethical_filters |   gharar: LOW - Stable
2026-03-07 16:52:41 | INFO    | operator1.analysis.ethical_filters |   cash_is_king: UNAVAILABLE
2026-03-07 16:52:41 | INFO    | operator1.report.profile_builder | Company profile written to cache/company_profile.json
2026-03-07 16:52:41 | INFO    | operator1.analysis.economic_planes | Economic plane: 14111 -> manufacturing (Manufacturing & Production)
2026-03-07 16:52:41 | INFO    | operator1.main | Profile saved: cache/company_profile.json
2026-03-07 16:52:41 | INFO    | operator1.main | 
2026-03-07 16:52:41 | INFO    | operator1.main | Step 8: Generating reports (Basic + Pro + Premium)...
2026-03-07 16:52:41 | INFO    | operator1.report.report_generator | Generating Basic Report...
2026-03-07 16:52:41 | INFO    | operator1.report.report_generator | Basic Report (Results mode) generated using local template.
2026-03-07 16:52:41 | INFO    | operator1.report.report_generator | Markdown report saved to cache/report/basic_report.md
2026-03-07 16:52:41 | INFO    | operator1.report.report_generator | pandoc not found; skipping PDF generation.
2026-03-07 16:52:41 | INFO    | operator1.report.report_generator | Generating Pro Report...
2026-03-07 16:52:41 | INFO    | operator1.report.report_generator | Pro Report (Results mode) generated using local template.
2026-03-07 16:52:41 | INFO    | operator1.report.report_generator | Markdown report saved to cache/report/pro_report.md
2026-03-07 16:52:42 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/price_history.png
2026-03-07 16:52:43 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/survival_timeline.png
2026-03-07 16:52:44 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/hierarchy_weights.png
2026-03-07 16:52:44 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/volatility.png
2026-03-07 16:52:45 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/financial_health.png
2026-03-07 16:52:45 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/predicted_ohlc_month.png
2026-03-07 16:52:45 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/predicted_ohlc_week.png
2026-03-07 16:52:45 | INFO    | operator1.report.report_generator | Embedded 7 charts into report.
2026-03-07 16:52:45 | INFO    | operator1.report.report_generator | pandoc not found; skipping PDF generation.
2026-03-07 16:52:45 | INFO    | operator1.report.report_generator | Generating Premium Report...
2026-03-07 16:52:45 | INFO    | operator1.clients.llm_base | Gemini model 'gemini-1.5-flash' max output is 8192 tokens (requested 16000); capping to model limit.
2026-03-07 16:52:47 | ERROR   | operator1.clients.llm_base | Gemini API error 404 (non-retryable): {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:52:47 | WARNING | operator1.clients.llm_base | Gemini request error on attempt 1/5 (1.3s): Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}
 -- retrying in 2.0s
2026-03-07 16:52:51 | ERROR   | operator1.clients.llm_base | Gemini API error 404 (non-retryable): {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:52:51 | WARNING | operator1.clients.llm_base | Gemini request error on attempt 2/5 (1.2s): Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}
 -- retrying in 4.0s
2026-03-07 16:52:56 | ERROR   | operator1.clients.llm_base | Gemini API error 404 (non-retryable): {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:52:56 | WARNING | operator1.clients.llm_base | Gemini request error on attempt 3/5 (1.1s): Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}
 -- retrying in 8.0s
2026-03-07 16:53:05 | ERROR   | operator1.clients.llm_base | Gemini API error 404 (non-retryable): {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:53:05 | WARNING | operator1.clients.llm_base | Gemini request error on attempt 4/5 (1.0s): Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}
 -- retrying in 16.0s
2026-03-07 16:53:22 | ERROR   | operator1.clients.llm_base | Gemini API error 404 (non-retryable): {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:53:22 | WARNING | operator1.clients.llm_base | Gemini request error on attempt 5/5 (1.2s): Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}
 -- retrying in 32.0s
2026-03-07 16:53:54 | ERROR   | operator1.clients.llm_base | Gemini report generation failed: Gemini: all 5 retries exhausted. Last error: Gemini API error 404: {
  "error": {
    "code": 404,
    "message": "models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.",
    "status": "NOT_FOUND"
  }
}

2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Premium Report narrative generated via Gemini.
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Appended LIMITATIONS section to report.
2026-03-07 16:53:54 | WARNING | operator1.report.report_generator | Gemini report validation: 18 issue(s) found: Missing section: 'executive summary'; Missing section: 'company overview'; Missing section: 'historical performance'; Missing section: 'financial health'; Missing section: 'survival'
2026-03-07 16:53:54 | WARNING | operator1.report.report_generator | Gemini report validation found 18 issues; appending missing sections from fallback template.
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: executive summary
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: company overview
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: historical performance
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: financial health
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: survival
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: linked variables
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: temporal analysis
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: predictions
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: technical patterns
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: ethical filter
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: supply chain
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: competitive
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: risk factors
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: investment recommendation
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Attempting to append missing section: appendix
2026-03-07 16:53:54 | INFO    | operator1.report.report_generator | Markdown report saved to cache/report/premium_report.md
2026-03-07 16:53:55 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/price_history.png
2026-03-07 16:53:56 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/survival_timeline.png
2026-03-07 16:53:57 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/hierarchy_weights.png
2026-03-07 16:53:58 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/volatility.png
2026-03-07 16:53:59 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/financial_health.png
2026-03-07 16:53:59 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/predicted_ohlc_month.png
2026-03-07 16:54:00 | INFO    | operator1.report.report_generator | Generated chart: cache/report/charts/predicted_ohlc_week.png
2026-03-07 16:54:00 | INFO    | operator1.report.report_generator | Embedded 7 charts into report.
2026-03-07 16:54:00 | INFO    | operator1.report.report_generator | pandoc not found; skipping PDF generation.
2026-03-07 16:54:00 | INFO    | operator1.report.report_generator | All three reports generated: cache/report/basic_report.md, cache/report/pro_report.md, cache/report/premium_report.md
2026-03-07 16:54:00 | INFO    | operator1.main |   Basic report: cache/report/basic_report.md
2026-03-07 16:54:00 | INFO    | operator1.main |   Pro report: cache/report/pro_report.md
2026-03-07 16:54:00 | INFO    | operator1.main |   Premium report: cache/report/premium_report.md
2026-03-07 16:54:00 | INFO    | operator1.main | 
2026-03-07 16:54:00 | INFO    | operator1.main | ============================================================
2026-03-07 16:54:00 | INFO    | operator1.main | PIPELINE COMPLETE
2026-03-07 16:54:00 | INFO    | operator1.main | ============================================================
2026-03-07 16:54:00 | INFO    | operator1.main | Market: South Korea (DART)
2026-03-07 16:54:00 | INFO    | operator1.main | Company: LF (093050)
2026-03-07 16:54:00 | INFO    | operator1.main | Output directory: cache
```

**Log saved to:** `cache/pipeline_run.md`

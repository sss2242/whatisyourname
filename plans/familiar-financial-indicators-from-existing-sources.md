# Familiar Financial Indicators from Existing Sources

## Summary

This plan maps what the codebase **already computes** to the well-known indicators that retail and institutional investors recognize from Bloomberg Terminal, Morningstar, Yahoo Finance, Seeking Alpha, and similar paid/free platforms. The goal is to surface these familiar names in reports so users immediately understand the output without learning new terminology.

---

## 1. What We Already Have (and What People Pay For)

The table below maps each indicator that people recognize and pay for to the **exact existing source** in our codebase that already produces it.

### 1A. Valuation Ratios (Bloomberg, Morningstar, Yahoo Finance)

| Familiar Name | Bloomberg Field | Our Source | Our Column Name |
|---|---|---|---|
| P/E Ratio | `PE_RATIO` | [`derived_variables.py`](operator1/features/derived_variables.py:345) `_compute_valuation()` | `pe_ratio_calc` |
| Earnings Yield | `EARN_YLD` | [`derived_variables.py`](operator1/features/derived_variables.py:345) `_compute_valuation()` | `earnings_yield_calc` |
| P/S Ratio | `PX_TO_SALES_RATIO` | [`derived_variables.py`](operator1/features/derived_variables.py:345) `_compute_valuation()` | `ps_ratio_calc` |
| P/B Ratio | `PX_TO_BOOK_RATIO` | [`derived_variables.py`](operator1/features/derived_variables.py:345) `_compute_valuation()` | `pb_ratio` |
| EV/EBITDA | `BEST_CUR_EV_TO_EBITDA` | [`derived_variables.py`](operator1/features/derived_variables.py:345) `_compute_valuation()` | `ev_to_ebitda` |
| Enterprise Value | `CURR_ENTP_VAL` | [`derived_variables.py`](operator1/features/derived_variables.py:345) `_compute_valuation()` | `enterprise_value` |
| FCF Yield | `FCF_YIELD` | [`derived_variables.py`](operator1/features/derived_variables.py:263) `_compute_cash_reality()` | `fcf_yield` |

### 1B. Profitability Metrics (Morningstar, S&P Capital IQ)

| Familiar Name | Our Source | Our Column Name |
|---|---|---|
| Gross Margin | [`derived_variables.py`](operator1/features/derived_variables.py:298) `_compute_profitability()` | `gross_margin` |
| Operating Margin | [`derived_variables.py`](operator1/features/derived_variables.py:298) `_compute_profitability()` | `operating_margin` |
| Net Profit Margin | [`derived_variables.py`](operator1/features/derived_variables.py:298) `_compute_profitability()` | `net_margin` |
| Return on Equity (ROE) | [`derived_variables.py`](operator1/features/derived_variables.py:298) `_compute_profitability()` | `roe` |
| Return on Assets (ROA) | [`derived_variables.py`](operator1/features/derived_variables.py:419) `_compute_roa()` | `roa` |

### 1C. Liquidity & Solvency (S&P, Moody's, CreditSights)

| Familiar Name | Our Source | Our Column Name |
|---|---|---|
| Current Ratio | [`derived_variables.py`](operator1/features/derived_variables.py:229) `_compute_liquidity()` | `current_ratio` |
| Quick Ratio | [`derived_variables.py`](operator1/features/derived_variables.py:229) `_compute_liquidity()` | `quick_ratio` |
| Cash Ratio | [`derived_variables.py`](operator1/features/derived_variables.py:229) `_compute_liquidity()` | `cash_ratio` |
| Debt-to-Equity | [`derived_variables.py`](operator1/features/derived_variables.py:186) `_compute_solvency()` | `debt_to_equity_abs` |
| Net Debt/EBITDA | [`derived_variables.py`](operator1/features/derived_variables.py:186) `_compute_solvency()` | `net_debt_to_ebitda` |
| Interest Coverage | [`derived_variables.py`](operator1/features/derived_variables.py:398) `_compute_interest_coverage()` | `interest_coverage` |

### 1D. Risk & Returns (Bloomberg PORT, MSCI, FactSet)

| Familiar Name | Our Source | Our Column Name |
|---|---|---|
| Daily Return | [`derived_variables.py`](operator1/features/derived_variables.py:150) `_compute_returns_and_risk()` | `return_1d` |
| 21-Day Volatility | [`derived_variables.py`](operator1/features/derived_variables.py:150) `_compute_returns_and_risk()` | `volatility_21d` |
| Max Drawdown (1Y) | [`derived_variables.py`](operator1/features/derived_variables.py:150) `_compute_returns_and_risk()` | `drawdown_252d` |
| Revenue Growth YoY | [`derived_variables.py`](operator1/features/derived_variables.py:438) `_compute_ttm_and_growth()` | `revenue_growth_yoy` |
| Earnings Growth YoY | [`derived_variables.py`](operator1/features/derived_variables.py:438) `_compute_ttm_and_growth()` | `earnings_growth_yoy` |

### 1E. Proprietary Scores (Altman, Beneish, Piotroski)

| Familiar Name | Who Charges For It | Our Source | Our Column Name |
|---|---|---|---|
| Altman Z-Score | S&P Capital IQ, Bloomberg | [`financial_health.py`](operator1/models/financial_health.py:76) | `fh_altman_z_score`, `fh_altman_z_zone` |
| Beneish M-Score | Forensic accounting tools | [`financial_health.py`](operator1/models/financial_health.py:87) | `fh_beneish_m_score`, `fh_beneish_flag` |
| Financial Health Composite | Morningstar "Financial Health" | [`financial_health.py`](operator1/models/financial_health.py:1) | `fh_composite_score`, `fh_composite_label` |
| Cash Runway | Startup analytics (PitchBook) | [`financial_health.py`](operator1/models/financial_health.py:1) | `fh_runway_months` |

### 1F. Technical & Pattern Analysis (TradingView, TC2000)

| Familiar Name | Our Source | Our Column Name |
|---|---|---|
| Candlestick Patterns | [`pattern_detector.py`](operator1/models/pattern_detector.py:1) | Doji, Hammer, Engulfing, Morning/Evening Star, etc. |
| Regime Detection (Bull/Bear) | [`regime_detector.py`](operator1/models/regime_detector.py:1) | `regime_hmm`, `regime_label` |
| Structural Breaks | [`regime_detector.py`](operator1/models/regime_detector.py:1) | `structural_break` |

### 1G. Macro Context (Bloomberg Economics, Trading Economics)

| Familiar Name | Our Source | Our Column |
|---|---|---|
| Macro Quadrant (Goldilocks/Stagflation/etc.) | [`macro_quadrant.py`](operator1/features/macro_quadrant.py:1) | `macro_quadrant`, `macro_quadrant_numeric` |
| GDP + Inflation alignment | [`macro_alignment.py`](operator1/features/macro_alignment.py) | macro alignment columns |

### 1H. Peer & Sector Analysis (FactSet, S&P Capital IQ)

| Familiar Name | Our Source | Our Column |
|---|---|---|
| Peer Percentile Rank | [`peer_ranking.py`](operator1/features/peer_ranking.py:1) | `peer_rank_<variable>`, `peer_composite_rank` |
| Sector Relative Strength | [`linked_aggregates.py`](operator1/features/linked_aggregates.py:1) | `<group>_mean_<var>`, `<group>_median_<var>` |

### 1I. Sentiment & News (Refinitiv, Bloomberg News)

| Familiar Name | Our Source | Our Column |
|---|---|---|
| News Sentiment Score | [`news_sentiment.py`](operator1/features/news_sentiment.py:1) | `news_sentiment_score`, `news_sentiment_label` |
| Sentiment Momentum | [`news_sentiment.py`](operator1/features/news_sentiment.py:1) | rolling sentiment windows |

### 1J. Portfolio Risk (Bloomberg PORT, Aladdin)

| Familiar Name | Our Source | Our Column |
|---|---|---|
| Marginal VaR Contribution | [`portfolio_analysis.py`](operator1/features/portfolio_analysis.py:1) | `marginal_var_contribution` |
| Institutional Overlap / HHI | [`portfolio_analysis.py`](operator1/features/portfolio_analysis.py:1) | `portfolio_concentration_hhi` |
| Correlation with Portfolio | [`portfolio_analysis.py`](operator1/features/portfolio_analysis.py:1) | `correlation_with_portfolio` |

### 1K. Forecasting & Probability (Bloomberg FCAST, FactSet Estimates)

| Familiar Name | Our Source | Details |
|---|---|---|
| Multi-horizon Point Forecasts | [`forecasting.py`](operator1/models/forecasting.py:1) | 1d, 5d, 21d, 252d horizons |
| Survival Probability | [`monte_carlo.py`](operator1/models/monte_carlo.py:1) | MC simulation with importance sampling |
| Prediction Confidence Intervals | [`prediction_aggregator.py`](operator1/models/prediction_aggregator.py:1) | Conformal-calibrated uncertainty bands |
| SHAP Feature Attribution | [`explainability.py`](operator1/models/explainability.py:1) | Per-prediction "why" explanations |

---

## 2. What We Could Add with Minimal Effort (Low-Hanging Fruit)

These are indicators people recognize that we could derive from data we **already fetch** but don't currently compute:

| Indicator | Source Data We Have | Implementation Effort |
|---|---|---|
| **Sharpe Ratio** | `return_1d` + risk-free rate from macro APIs | ~15 lines in `derived_variables.py` |
| **Beta (vs market)** | `return_1d` + market index OHLCV (yfinance) | ~25 lines; covariance of returns vs index |
| **Dividend Yield** | PIT financials (some regions report dividends) | ~20 lines if dividend data present |
| **Piotroski F-Score** | All 9 inputs already in cache (ROA, CFO, margins, leverage, liquidity, shares) | ~60 lines in `financial_health.py` |
| **Revenue per Share** | `revenue` + `shares_outstanding` | 5 lines |
| **Book Value per Share** | `total_equity` + `shares_outstanding` | 5 lines |
| **EPS (calc)** | Already computed internally in `_compute_valuation` but not exposed | 1 line to expose |
| **Moving Averages (SMA 50/200)** | `close` price | ~10 lines; SMA crossover signals |
| **RSI (14-day)** | `return_1d` | ~15 lines |
| **MACD** | `close` price | ~20 lines |
| **Bollinger Bands** | `close` + `volatility_21d` | ~10 lines |

---

## 3. Should We Do This?

**Yes, selectively.** Here is the reasoning:

### Arguments For

1. **Instant credibility**: When a user sees "P/E Ratio: 18.2" or "Altman Z-Score: 2.4 (Grey Zone)", they immediately know what they are looking at. No onboarding friction.

2. **Competitive positioning**: We already compute these -- we just need to surface them with recognizable labels in reports. Bloomberg charges $24k/year for a terminal that shows many of the same ratios.

3. **Trust anchor**: Familiar indicators serve as "sanity check" reference points. If our P/E matches what the user sees on Yahoo Finance, they trust the rest of our analysis (regime detection, survival probability, SHAP explanations) which they cannot get elsewhere.

4. **Low cost**: Most of these indicators are already computed. The work is primarily presentation/labeling, not computation.

### Arguments Against (and Mitigations)

1. **"Why would I pay for what Yahoo gives me free?"** -- Mitigation: The familiar indicators are the anchor, not the product. The product is the survival analysis, regime detection, Monte Carlo simulations, SHAP explanations, and multi-horizon forecasts that Yahoo/Google Finance do not offer.

2. **Accuracy risk if our numbers differ slightly from Bloomberg** -- Mitigation: Always show the data source and methodology. Document that P/E uses trailing EPS from PIT filings, not consensus estimates.

3. **Feature creep** -- Mitigation: Add only the indicators listed in Section 2 that require minimal code. Don't build a Bloomberg clone.

### Recommended Approach: "Familiar Anchors + Unique Value"

Structure reports in two tiers:

1. **Familiar Anchors Section** (top of report) -- Show the 10-15 indicators everyone knows: P/E, P/B, ROE, Debt/Equity, Current Ratio, Gross Margin, Altman Z, Revenue Growth YoY, Volatility, Max Drawdown. Use exact names from Bloomberg/Yahoo. This is the "I recognize this" moment.

2. **Unique Analysis Section** (body of report) -- This is where we differentiate: regime detection, survival probability, Monte Carlo paths, peer percentile ranking, macro quadrant, SHAP explanations, candlestick predictions. These are the things people cannot get from Yahoo Finance.

---

## 4. Implementation Roadmap

### Phase 1: Relabeling (0 new computation, report changes only)
- Add a "Key Financial Indicators" summary table to the report template in [`report_generator.py`](operator1/report/report_generator.py:1)
- Map internal column names to standard financial terminology
- Show latest values for the ~30 indicators already computed

### Phase 2: Quick Adds (~100 lines of new code)
- Sharpe Ratio, Beta, Piotroski F-Score
- SMA 50/200, RSI, MACD, Bollinger Bands
- Expose EPS, Book Value/Share, Revenue/Share

### Phase 3: Presentation Polish
- Add sparkline-style trend indicators (up/down arrows, color coding)
- Add peer comparison context ("P/E of 18.2 vs sector median of 22.1")
- Add historical percentile ("Current ratio in 75th percentile of its 5-year range")

---

## 5. Data Source Coverage Matrix

All indicators above are derived from these already-implemented data pipelines:

| Data Pipeline | Source | Key | Coverage |
|---|---|---|---|
| **OHLCV prices** | yfinance (global fallback) + pykrx/twstock/baostock/nselib | Free, no key | 25+ markets |
| **Financial statements** | SEC EDGAR, Companies House, ESEF, EDINET, DART, MOPS, CVM, CMF | Free PIT APIs | $91T+ market cap |
| **Macro data** | FRED, World Bank, SDMX, KOSIS, DGBAS, Banxico, BCB, BCCh, ONS, Eurostat | Free | 15+ countries |
| **Classification** | OpenFIGI | Free | Global |
| **News** | FMP stock news | Free tier | US-centric, global tickers |
| **LLM analysis** | Gemini / Claude | API key | Narrative generation |

No new data sources are needed. Every familiar indicator listed above can be produced from data we already fetch.

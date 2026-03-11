# Familiar Financial Indicators from Existing Sources

## The question

Can we use what we already have to show people indicators they recognize from Bloomberg, Morningstar, Yahoo Finance, and TradingView? Should we? And should we change our report style to do it?

## The short answer

We already compute ~30 of the indicators people pay Bloomberg $24k/year to see. We just don't label them that way in the report. The fix is mostly presentation, not computation. And the Bloomberg-style dark theme we already use is exactly right -- don't change it.

---

## Part 1: What we already compute vs what people pay for

The pipeline in `derived_variables.py` (lines 516-538) lists every derived variable. Here's how they map to what people actually look up on paid platforms.

### Valuation (Bloomberg, Yahoo Finance, Morningstar)

| What people call it | Bloomberg field | Our column | Computed in |
|---|---|---|---|
| P/E Ratio | `PE_RATIO` | `pe_ratio_calc` | `_compute_valuation()` line 345 |
| Earnings Yield | `EARN_YLD` | `earnings_yield_calc` | `_compute_valuation()` line 345 |
| P/S Ratio | `PX_TO_SALES_RATIO` | `ps_ratio_calc` | `_compute_valuation()` line 345 |
| P/B Ratio | `PX_TO_BOOK_RATIO` | `pb_ratio` | `_compute_valuation()` line 345 |
| EV/EBITDA | `BEST_CUR_EV_TO_EBITDA` | `ev_to_ebitda` | `_compute_valuation()` line 345 |
| Enterprise Value | `CURR_ENTP_VAL` | `enterprise_value` | `_compute_valuation()` line 345 |
| FCF Yield | `FCF_YIELD` | `fcf_yield` | `_compute_cash_reality()` line 263 |

### Profitability (Morningstar, S&P Capital IQ)

| What people call it | Our column | Computed in |
|---|---|---|
| Gross Margin | `gross_margin` | `_compute_profitability()` line 298 |
| Operating Margin | `operating_margin` | `_compute_profitability()` line 298 |
| Net Profit Margin | `net_margin` | `_compute_profitability()` line 298 |
| Return on Equity | `roe` | `_compute_profitability()` line 298 |
| Return on Assets | `roa` | `_compute_roa()` line 419 |

### Solvency & Liquidity (S&P, Moody's, CreditSights)

| What people call it | Our column | Computed in |
|---|---|---|
| Current Ratio | `current_ratio` | `_compute_liquidity()` line 229 |
| Quick Ratio | `quick_ratio` | `_compute_liquidity()` line 229 |
| Cash Ratio | `cash_ratio` | `_compute_liquidity()` line 229 |
| Debt-to-Equity | `debt_to_equity_abs` | `_compute_solvency()` line 186 |
| Net Debt/EBITDA | `net_debt_to_ebitda` | `_compute_solvency()` line 186 |
| Interest Coverage | `interest_coverage` | `_compute_interest_coverage()` line 398 |

### Risk & Returns (Bloomberg PORT, MSCI, FactSet)

| What people call it | Our column | Computed in |
|---|---|---|
| Daily Return | `return_1d` | `_compute_returns_and_risk()` line 150 |
| 21-Day Volatility | `volatility_21d` | `_compute_returns_and_risk()` line 150 |
| Max Drawdown (1Y) | `drawdown_252d` | `_compute_returns_and_risk()` line 150 |
| Revenue Growth YoY | `revenue_growth_yoy` | `_compute_ttm_and_growth()` line 438 |
| Earnings Growth YoY | `earnings_growth_yoy` | `_compute_ttm_and_growth()` line 438 |

### Proprietary Scores (these are the ones people actually pay for)

| What people call it | Who charges for it | Our column | Computed in |
|---|---|---|---|
| Altman Z-Score | S&P Capital IQ, Bloomberg | `fh_altman_z_score` | `financial_health.py` line 76 |
| Beneish M-Score | Forensic accounting tools | `fh_beneish_m_score` | `financial_health.py` line 87 |
| Financial Health Composite | Morningstar "Financial Health" | `fh_composite_score` | `financial_health.py` line 1 |
| Cash Runway (months) | PitchBook, startup analytics | `fh_runway_months` | `financial_health.py` line 1 |

### Technical & Pattern Analysis (TradingView, TC2000)

| What people call it | Our source | Details |
|---|---|---|
| Candlestick Patterns | `pattern_detector.py` | Doji, Hammer, Engulfing, Morning/Evening Star, Three White Soldiers/Black Crows |
| Bull/Bear Regime | `regime_detector.py` | HMM + GMM regime classification |
| Structural Breaks | `regime_detector.py` | PELT + Bayesian change point detection |

### Macro, Peer, Sentiment, Portfolio (Bloomberg Economics, FactSet, Refinitiv)

| What people call it | Our source | Our column(s) |
|---|---|---|
| Macro Quadrant | `macro_quadrant.py` | `macro_quadrant`, `macro_quadrant_numeric` |
| Peer Percentile Rank | `peer_ranking.py` | `peer_rank_<variable>`, `peer_composite_rank` |
| News Sentiment Score | `news_sentiment.py` | `news_sentiment_score`, `news_sentiment_label` |
| Marginal VaR Contribution | `portfolio_analysis.py` | `marginal_var_contribution` |
| Correlation with Portfolio | `portfolio_analysis.py` | `correlation_with_portfolio` |

### Forecasting & Probability (Bloomberg FCAST, FactSet Estimates)

| What people call it | Our source | Details |
|---|---|---|
| Multi-horizon Forecasts | `forecasting.py` | 1d, 5d, 21d, 252d horizons via Kalman/GARCH/VAR/LSTM/XGB/Baseline |
| Survival Probability | `monte_carlo.py` | 10k-path MC with importance sampling |
| Prediction Intervals | `prediction_aggregator.py` | Conformal-calibrated confidence bands |
| Feature Attribution | `explainability.py` | SHAP values per prediction |

---

## Part 2: What we can add with minimal effort

These are indicators people recognize that we can derive from data we already fetch but don't currently compute. All live in existing files, no new data sources needed.

| Indicator | What we need | Where it goes | Lines of code |
|---|---|---|---|
| Sharpe Ratio | `return_1d` + risk-free rate from macro APIs (FRED) | `derived_variables.py` | ~15 |
| Beta (vs market index) | `return_1d` + index OHLCV via yfinance | `derived_variables.py` | ~25 |
| Piotroski F-Score | All 9 inputs already in cache (ROA, CFO, margins, leverage, liquidity, shares) | `financial_health.py` | ~60 |
| SMA 50/200 + Golden/Death Cross | `close` price | `derived_variables.py` | ~10 |
| RSI (14-day) | `return_1d` | `derived_variables.py` | ~15 |
| MACD | `close` price | `derived_variables.py` | ~20 |
| Bollinger Bands | `close` + `volatility_21d` | `derived_variables.py` | ~10 |
| EPS (exposed) | Already computed internally in `_compute_valuation` but not saved as a column | `derived_variables.py` | 1 |
| Book Value per Share | `total_equity` / `shares_outstanding` | `derived_variables.py` | 5 |
| Revenue per Share | `revenue` / `shares_outstanding` | `derived_variables.py` | 5 |
| Dividend Yield | PIT financials (some regions report dividends) | `derived_variables.py` | ~20 |

Total: ~186 lines. All from data we already fetch.

---

## Part 3: Should we do this?

Yes, selectively. Here's why.

### The trust anchor problem

When someone sees "P/E Ratio: 18.2" they instantly know whether that's cheap or expensive for the sector. When they see "fh_composite_score: 72.4" they have no reference point. The familiar indicator is the trust anchor -- if our P/E matches what Yahoo shows, the user trusts the rest of our analysis (regime detection, survival probability, SHAP explanations) which they can't get anywhere else.

### The "why would I pay for this?" problem

If we only show P/E, ROE, and margins, users will say "I can get this on Yahoo for free." The familiar indicators aren't the product. They're the *anchor*. The product is the survival analysis, Monte Carlo simulations, regime detection, Granger causality, SHAP feature attribution, and multi-horizon forecasts. Those don't exist on Yahoo Finance.

### What to actually do

Structure reports in two layers:

1. **Familiar Anchors** (top of report) -- 10-15 indicators everyone knows: P/E, P/B, ROE, Debt/Equity, Current Ratio, Gross Margin, Altman Z, Revenue Growth YoY, Volatility, Drawdown. Standard labels from Bloomberg/Yahoo. This is the "I recognize this" moment.

2. **Unique Analysis** (body of report) -- Regime detection, survival probability, Monte Carlo paths, peer percentile ranking, macro quadrant, SHAP explanations, candlestick predictions. The stuff you can't get from Yahoo.

---

## Part 4: Report style, accessibility tiers, and brand coloring

### The problem with what we have now

The current system has three tiers (Basic/Pro/Premium) that control *how many sections* you see:
- `ReportTier.BASIC`: 5 sections (1, 2, 4, 6, 20)
- `ReportTier.PRO`: 13 sections (adds peers, macro, sentiment, etc.)
- `ReportTier.PREMIUM`: all 22 sections

But the content within each section is the same regardless of tier. A Basic user and a Premium user both see "debt_to_equity_abs: 1.47" with no explanation. A finance professional knows what that means. A small business owner or a first-time investor doesn't.

The tiers control *depth* but not *understanding*. We need both.

### Two-axis tier system: Understanding x Detail

The right structure is a 2x3 matrix. Each tier has two report modes:

**Mode 1: "Learn" reports** -- help people understand what the numbers mean

| Tier | Sections | What's different in Learn mode |
|---|---|---|
| Basic Learn | 5 sections | Each indicator gets a 1-sentence plain-English explanation. "Debt-to-Equity of 1.47 means the company owes $1.47 for every $1 of shareholder money. Above 2.0 is usually concerning." Color-coded green/yellow/red with thresholds explained. No jargon. |
| Pro Learn | 13 sections | Same explanations, plus "what this means for you" context. "The macro environment is Stagflation (slow growth + high inflation). Historically, companies with high debt struggle in this environment because borrowing costs rise." Peer comparisons with plain language. |
| Premium Learn | 22 sections | Full institutional depth, but with an expandable "Explain this" block after each technical section. Monte Carlo gets "We simulated 10,000 possible futures. In 87% of them, the company survived the next year." SHAP gets "The model thinks rising debt is the #1 risk factor, contributing 32% to the negative outlook." |

**Mode 2: "Results" reports** -- show the numbers, skip the explanations

| Tier | Sections | What's different in Results mode |
|---|---|---|
| Basic Results | 5 sections | Clean indicator grid, charts, verdict. No prose. Like a Bloomberg COMP screen printout. |
| Pro Results | 13 sections | Full data tables, all charts, peer rankings as numbers. Designed for scanning. |
| Premium Results | 22 sections | Everything. Dense. Every model output, every confidence interval, every SHAP waterfall. The institutional analyst's working document. |

### Implementation

In `report_generator.py`, the existing `ReportTier` enum gets a companion:

```python
class ReportMode(str, Enum):
    LEARN = "learn"       # Explanations + context for understanding
    RESULTS = "results"   # Data-forward, minimal prose
```

Each section builder (like `_build_executive_summary()`, `_build_survival_analysis()`, etc.) takes a `mode` parameter and adjusts its output. The Learn mode adds explanation blocks; the Results mode strips them. The underlying data is identical -- only the presentation layer changes.

Estimated code: ~200 lines total (mostly string templates for the explanation blocks). The actual computation doesn't change at all.

### Brand coloring: Proton-inspired as our signature

The coloring shouldn't be borrowed from Bloomberg or Discord. It should be *ours*. ProtonMail's approach is the right model: clean, warm, trustworthy, distinctive. Nobody in finance uses this palette, which means it becomes immediately recognizable as our product.

#### The Operator 1 brand palette

```python
# --- Operator 1 Brand Palette ---
# Inspired by Proton's "trust through clarity" aesthetic
# Warm, clean, readable -- deliberately NOT Bloomberg

# Core
_BRAND_BG = "#1c1b22"          # deep warm charcoal (not cold navy)
_BRAND_BG_LIGHT = "#f5f0ec"    # warm parchment (for light mode / PDF)
_BRAND_FG = "#eae7e1"          # warm off-white text
_BRAND_FG_LIGHT = "#1b1340"    # deep purple-black (for light mode)

# Signature accent -- our purple
_BRAND_ACCENT = "#6d4aff"      # Proton-inspired purple (our signature)
_BRAND_ACCENT_MUTED = "#6d4aff33"  # 20% opacity for bands/fills

# Semantic colors (financial meaning)
_BRAND_POSITIVE = "#1ea885"    # teal-green (calmer than neon green)
_BRAND_NEGATIVE = "#dc3545"    # clear red (universally understood)
_BRAND_WARNING = "#e8950a"     # warm amber (not screaming yellow)
_BRAND_NEUTRAL = "#8b8694"     # muted purple-grey (secondary info)

# Chart-specific
_BRAND_GRID = "#2d2b33"        # subtle warm grid (dark mode)
_BRAND_GRID_LIGHT = "#e0d8d0"  # subtle warm grid (light mode)
```

#### Why this works as a brand

1. **The purple is the signature.** Nobody in finance uses purple as their primary accent. Bloomberg is cyan. Morningstar is navy/orange. Yahoo is purple but their finance product uses blue. Robinhood is green. Our purple is distinctive and ownable.

2. **Warm tones, not cold ones.** Bloomberg's `#1a1a2e` is cold navy. Ours is `#1c1b22` -- warm charcoal with a hint of brown/purple. The difference is subtle but it makes the reports feel less clinical and more approachable. Proton does this exact thing.

3. **The teal-green instead of neon green.** Our current `#2ed573` is fun but screams "gaming." The `#1ea885` teal-green is calmer, more confident. It says "this is good" without shouting. Proton uses a similar shade.

4. **Light mode option for PDFs.** Some people will print reports or read them on tablets in daylight. The `_BRAND_BG_LIGHT = "#f5f0ec"` warm parchment background with `_BRAND_FG_LIGHT = "#1b1340"` deep purple-black text gives a distinctive Proton-like reading experience on paper.

#### Migration plan

Rename `_apply_bloomberg_style()` to `_apply_brand_style()` in `report_generator.py`. Replace the 7 color constants with the brand palette above. Add a `dark_mode: bool` parameter that switches between dark and light variants (for screen vs PDF). Every chart that calls the function gets the new palette automatically.

Total code change: ~30 lines in `report_generator.py`. Zero changes to any computation, feature, or model code.

### What to improve regardless of palette choice

These improvements work with any color scheme:

- Add a **Key Indicators Summary Table** at the top of Section 4 (Current Financial Snapshot) with ~15 familiar metrics in a clean grid
- Use semantic coloring (`_BRAND_POSITIVE` for strong, `_BRAND_NEGATIVE` for weak, `_BRAND_WARNING` for neutral) on indicator values
- Add **peer context** inline: "P/E: 18.2 (sector median: 22.1)"
- Add **historical percentile** inline: "Current Ratio: 1.8 (75th percentile of 5-year range)"
- In Learn mode, add explanation tooltips: "P/E of 18.2 means investors pay $18.20 for every $1 of earnings. Lower than sector average of 22.1 suggests this stock may be undervalued."

---

## Part 5: Multi-horizon zoomed prediction charts

Since we already produce forecasts at 1d, 5d, 21d, and 252d horizons (via `forecasting.py` line 55 and `prediction_aggregator.py`), and we already render charts (via `generate_charts()` at line 2593), we should show predicted familiar indicators as four zoomed chart panels so users see both the trajectory and the uncertainty at each time scale.

### Chart layout: the "Zoom Ladder"

For each key indicator, generate a 2x2 figure where each panel zooms into a different horizon:

```
+-------------------------------+-------------------------------+
|  TOMORROW (1d)                |  NEXT WEEK (5d)               |
|  [last 10 days] + [1d pred]  |  [last 30 days] + [5d pred]   |
|  Y-axis: tight zoom           |  Y-axis: medium zoom          |
+-------------------------------+-------------------------------+
|  NEXT MONTH (21d)             |  NEXT YEAR (252d)             |
|  [last 63 days] + [21d pred] |  [last 252 days] + [252d pred]|
|  Y-axis: wider zoom           |  Y-axis: full range           |
+-------------------------------+-------------------------------+
```

### Which indicators get prediction charts

Not every indicator is forecastable -- only time-series variables that the forecasting models actually predict. The recommended set:

| Indicator | Why it works as a chart |
|---|---|
| Close Price | What everyone looks at first; already charted |
| P/E Ratio | Valuation trajectory; derived from predicted EPS + price |
| EV/EBITDA | Enterprise valuation trend; derived from predicted EBITDA + debt + price |
| Gross Margin | Profitability direction; directly forecast by the model suite |
| Debt-to-Equity | Solvency trajectory; directly forecast |
| Current Ratio | Liquidity trend; directly forecast (Tier 1 survival variable) |
| FCF Yield | Cash reality direction; directly forecast |
| Volatility (21d) | Risk outlook; GARCH specializes in this |
| Revenue Growth YoY | Growth trajectory; derived from predicted revenue TTM |
| Altman Z-Score | Bankruptcy risk path; composed from 5 predicted sub-variables |

### Zooming strategy

| Panel | Horizon | Historical tail | Y-axis padding |
|---|---|---|---|
| Tomorrow | 1d | Last 10 trading days | 5% |
| Next Week | 5d | Last 30 trading days | 10% |
| Next Month | 21d | Last 63 trading days | 15% |
| Next Year | 252d | Last 252 trading days | 20% |

The tight zoom on tomorrow's panel makes even a small predicted move visually clear. The wide zoom on the yearly panel shows the full trajectory.

### Confidence band rendering

- 1d panel: narrow band (low uncertainty) -- gives confidence
- 5d panel: slightly wider -- still precise
- 21d panel: visible cone -- honest about uncertainty
- 252d panel: wide fan -- overlay Monte Carlo percentiles (p5, p25, p50, p75, p95)

For the 252d panel, color the confidence band by survival probability (green = high survival, red = low survival). This connects the familiar indicator chart to our unique survival analysis without the user needing to understand what Monte Carlo simulation is.

### Implementation

Builds on existing `generate_charts()` infrastructure. Uses the same `_CHART_BG`, `_CHART_ACCENT`, `_CHART_RED`, `_CHART_GREEN` constants and `_apply_bloomberg_style()`. Estimated: ~150 lines of matplotlib code.

```python
def generate_prediction_zoom_charts(
    cache: pd.DataFrame,
    predictions: dict[str, dict[str, float]],
    confidence_bands: dict[str, dict[str, tuple[float, float]]],
    mc_result: MonteCarloResult | None = None,
    output_dir: str = CACHE_DIR,
    indicators: list[str] | None = None,
) -> list[str]:
    """Generate 4-panel zoom-ladder charts for each predicted indicator."""
```

---

## Part 6: Data source coverage (current state)

Everything above runs on data we already fetch from these pipelines. No new sources needed.

### PIT Financial Statements (Tier 1, $91T+ coverage)

| Market | Client | Coverage | Key needed |
|---|---|---|---|
| US | `us_edgar.py` (edgartools + sec-edgar-api) | $50T, NYSE/NASDAQ | No |
| UK | `uk_ch_wrapper.py` (Companies House) | $3.18T, LSE | No |
| EU | `eu_esef_wrapper.py` (ESEF/XBRL) | $8-9T, pan-EU | No |
| Japan | `jp_jquants_wrapper.py` (J-Quants) | $6.5T, TSE | Free key |
| South Korea | `kr_dart_wrapper.py` (dart-fss) | $2.5T, KOSPI/KOSDAQ | Free key |
| Taiwan | `tw_mops_wrapper.py` (MOPS) | $1.2T, TWSE | No |
| Brazil | `br_cvm_wrapper.py` (CVM) | $2.2T, B3 | No |
| Chile | `cl_cmf_wrapper.py` (CMF) | $0.4T | No |

### OHLCV Price Data

| Market | Primary | Fallback | Key needed |
|---|---|---|---|
| China | `ohlcv_akshare.py` (akshare) | yfinance (.SS/.SZ) | No |
| India | `ohlcv_jugaad.py` (jugaad-data) | yfinance (.NS/.BO) | No |
| South Korea | `ohlcv_pykrx.py` (pykrx) | yfinance (.KS) | No |
| Taiwan | `ohlcv_twstock.py` (twstock) | yfinance (.TW) | No |
| All others | yfinance directly | -- | No |

Dispatcher: `ohlcv_provider.py` routes to per-region primary, falls back to yfinance.

### Macro Data (15+ countries)

| Source | Client | Coverage |
|---|---|---|
| FRED | `macro_fredapi.py` | US macro (GDP, CPI, unemployment, rates) |
| World Bank | `macro_wbgapi.py` | 200+ countries (annual) |
| SDMX (Eurostat/OECD) | `macro_sdmx.py` | EU, OECD countries |
| KOSIS | `macro_kosis.py` | South Korea |
| DGBAS | `macro_dgbas.py` | Taiwan |
| Banxico | `macro_banxico.py` | Mexico |
| BCB | `macro_bcb.py` | Brazil |
| BCCh | `macro_bcch.py` | Chile |
| ONS | `macro_ons.py` | UK |
| Eurostat | `macro_estat.py` | EU aggregate |

### Supplementary Data

| Source | Client | Fills |
|---|---|---|
| OpenFIGI | `supplement.py` | FIGI, sector classification (global) |
| Euronext, JPX, TWSE, B3, Bolsa | `supplement.py` | Sector, industry, profile for non-US |

### Estimation (missing data recovery)

| Method | Client | Details |
|---|---|---|
| Accounting identity fill | `estimator.py` Pass 1 | A = L + E, FCF = OCF - capex, etc. |
| BayesianRidge imputer | `estimator.py` Pass 2 (default) | Per-variable rolling imputer |
| VAE imputer | `vae_imputer.py` Pass 2 (optional) | Nonlinear cross-variable imputation |

### LLM (narrative + entity discovery)

| Provider | Client | Used for |
|---|---|---|
| Gemini | `gemini.py` via `llm_base.py` / `llm_factory.py` | Report narrative, linked entity discovery, sentiment scoring |
| Claude | `claude.py` via `llm_base.py` / `llm_factory.py` | Alternative to Gemini (same interface) |

---

## Part 7: Implementation roadmap

### Phase 1: Relabeling (zero new computation)

Add a "Key Financial Indicators" summary table to the report template. Map internal column names to standard financial terminology. Show latest values for the ~30 indicators already computed. This is a `report_generator.py` change only.

### Phase 2: Quick adds (~186 lines of new code)

Add Sharpe Ratio, Beta, Piotroski F-Score, SMA 50/200, RSI, MACD, Bollinger Bands. Expose EPS, Book Value/Share, Revenue/Share as named columns. All in `derived_variables.py` and `financial_health.py`.

### Phase 3: Zoom Ladder charts (~150 lines)

Add `generate_prediction_zoom_charts()` to `report_generator.py`. 2x2 panels per indicator, 10 indicators = 10 chart images. Uses existing Bloomberg theme infrastructure.

### Phase 4: Inline context (~50 lines)

Add peer median comparison and historical percentile to the fallback template sections. "P/E: 18.2 (sector: 22.1, 5yr percentile: 35th)". Uses data already in the linked aggregates and cache.

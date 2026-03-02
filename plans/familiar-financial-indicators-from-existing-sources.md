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

## Part 4: Which visual style should we use?

### The three candidates

We have three real options for the chart and report coloring. Here's a concrete comparison.

#### Option A: Current Bloomberg-style (what we have now)

```python
_CHART_BG = "#1a1a2e"      # dark navy background
_CHART_FG = "#e0e0e0"      # light grey text
_CHART_GRID = "#2d2d44"    # subtle grid
_CHART_ACCENT = "#00d4ff"  # Bloomberg cyan accent
_CHART_RED = "#ff4757"     # danger/bearish
_CHART_GREEN = "#2ed573"   # positive/bullish
_CHART_GOLD = "#ffa502"    # warning/neutral
```

**Who this is for:** Finance professionals, Bloomberg Terminal users, institutional analysts. People who stare at dark terminals all day.

**Pros:** Looks like what traders and analysts already use. High information density works on dark backgrounds. The cyan-on-navy color scheme says "this is a professional financial tool."

**Cons:** Can feel intimidating or cold to retail investors. Dense information on dark backgrounds causes eye strain for long reading sessions. Non-finance people (small business owners, casual investors) may feel this "isn't for them."

#### Option B: Proton Mail / Clean Modern style

```python
_CHART_BG = "#f5f0ec"      # warm off-white (Proton's parchment)
_CHART_FG = "#1b1340"      # deep purple-black text
_CHART_GRID = "#e0d8d0"    # subtle warm grid
_CHART_ACCENT = "#6d4aff"  # Proton purple
_CHART_RED = "#dc3545"     # standard danger red
_CHART_GREEN = "#1ea885"   # muted teal-green
_CHART_GOLD = "#ff8c00"    # warm amber
```

**Who this is for:** Privacy-conscious tech users, modern SaaS consumers, people who trust "clean design." Proton's brand is "we respect you and your data."

**Pros:** Easier on the eyes for reading. The warm off-white background feels less aggressive than pitch-black. The purple accent is distinctive -- nobody else in finance uses it. Strong "trust through clarity" signal. Better for long reports that people actually read cover-to-cover.

**Cons:** Doesn't scream "finance." A P/E ratio on a lavender background might feel less authoritative to someone who's used to Bloomberg. Charts with many overlapping series are harder to read on light backgrounds.

#### Option C: Discord / App-native dark style

```python
_CHART_BG = "#313338"      # Discord dark (softer than Bloomberg navy)
_CHART_FG = "#dbdee1"      # Discord light text
_CHART_GRID = "#3f4147"    # subtle grey grid
_CHART_ACCENT = "#5865f2"  # Discord blurple
_CHART_RED = "#ed4245"     # Discord red
_CHART_GREEN = "#57f287"   # Discord green (bright, fun)
_CHART_GOLD = "#fee75c"    # Discord yellow
```

**Who this is for:** Younger investors (25-40), tech workers, the Robinhood/WeBull generation. People who spend 4+ hours a day in Discord, Slack, or similar apps.

**Pros:** Feels native to how this generation consumes information. The softer dark grey (#313338 vs our #1a1a2e) is less aggressive than Bloomberg but still dark-mode. The blurple accent is distinctive and modern. The brighter green/red are more visible than our current muted versions. This crowd already trusts this color language.

**Cons:** Might feel "casual" to institutional users. The Discord association could undermine seriousness for fund managers or compliance officers.

### The actual recommendation: Discord dark, but keep the Bloomberg structure

Here's the reasoning:

1. **Our audience is not Bloomberg Terminal users.** Bloomberg users already have a Bloomberg Terminal. They're not looking for another one. Our audience is people who want institutional-quality analysis without the $24k/year price tag. Those people are more likely to be in Discord than on a Bloomberg chat channel.

2. **The Discord dark palette is objectively better for readability.** Our current `#1a1a2e` (very dark navy) has low contrast with `#2d2d44` (grid lines) -- the difference is only ~10% luminance. Discord's `#313338` background vs `#3f4147` grid has better contrast. The brighter accent colors (`#5865f2` blurple, `#57f287` green) pop more on the darker grey.

3. **The structure should stay exactly the same.** The 22-section institutional format, the three-tier system (Basic/Pro/Premium), the chart types -- all of this is correct. Only the *colors* change, not the layout or content hierarchy.

4. **The migration is contained.** We change 7 color constants in `report_generator.py` and the `_apply_bloomberg_style()` function (rename it to `_apply_chart_style()`). That's it. Every chart that calls the function gets the new palette automatically.

### Proposed new palette (Discord-inspired financial)

```python
# Modern dark theme -- inspired by Discord/app-native coloring
# but tuned for financial data readability
_CHART_BG = "#2b2d31"      # Discord dark-mode background
_CHART_FG = "#e0e2e6"      # slightly warmer than pure white
_CHART_GRID = "#3a3c42"    # visible but not distracting
_CHART_ACCENT = "#5865f2"  # blurple -- our brand color
_CHART_RED = "#ed4245"     # clear danger/bearish
_CHART_GREEN = "#57f287"   # clear positive/bullish
_CHART_GOLD = "#fee75c"    # attention/warning (Discord yellow)

# Financial-specific additions
_CHART_MUTED = "#949ba4"   # for secondary data series
_CHART_BAND = "#5865f233"  # blurple at 20% opacity for confidence bands
```

This keeps the dark-mode feel that works for charts (data visualization is genuinely better on dark backgrounds) while making it feel modern and approachable instead of intimidating.

### Alternative: offer both as a user preference

The cleanest solution might be to make the palette a config option. Define two presets:

- `chart_theme: "terminal"` -- current Bloomberg palette (for institutional users)
- `chart_theme: "modern"` -- Discord-inspired palette (default for new users)

The implementation is trivial: load the 7 color values from `global_config.yml` instead of hardcoding them. Cost: ~20 lines in `report_generator.py`.

### What to improve regardless of palette choice

These improvements work with any color scheme:

- Add a **Key Indicators Summary Table** at the top of Section 4 (Current Financial Snapshot) with ~15 familiar metrics in a clean grid
- Use semantic coloring (`_CHART_GREEN` for strong, `_CHART_RED` for weak, `_CHART_GOLD` for neutral) on indicator values
- Add **peer context** inline: "P/E: 18.2 (sector median: 22.1)"
- Add **historical percentile** inline: "Current Ratio: 1.8 (75th percentile of 5-year range)"

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

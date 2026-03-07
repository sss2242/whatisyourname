# Plan: Data Quality Solutions and Missing Charts -- Revised

Based on user feedback:
- No yfinance supplement (not PIT-compliant)
- Per-region news RSS sources instead of English-only GNews
- Modified Kalman for forward-filled financial data
- Proper FRED fallback chain

---

## Part 1: Missing Charts

### P1: Add Next-Year OHLCV Chart

The OHLC predictor generates 252 candles for `next_year` but no chart exists. Add Chart 10 to [`report_generator.py`](operator1/report/report_generator.py:3280):
- Aggregate 252 daily candles into ~50 weekly bars (Open=first, High=max, Low=min, Close=last)
- Same candlestick drawing logic as month/week charts
- Add predicted annual return annotation
- Save as `predicted_ohlc_year.png`
- Embed after the month chart in the report

### P2: Add Next-Day Low Annotation to Price Chart

The Technical Alpha mask exposes only the next-day Low estimate. Add to Chart 1 (Price History):
- Dashed horizontal line at `estimated_low`
- Label: "Next-Day Low Estimate"
- Gold color to distinguish from historical data

---

## Part 2: Modified Kalman for Financial Data

### D2-Revised: Financial-Aware Kalman Filter

**Problem**: Standard Kalman treats forward-filled financial data as perfect observations (MAE=0, RMSE=0). This gives Kalman infinite ensemble weight and drowns out models that actually forecast.

**Root cause**: Financial statement data is quarterly (4 changes per year) but forward-filled to daily (252 rows). Between filings, the "observation" is just the previous filing's value repeated -- it's not new information.

**Fix -- Filing-Aware Kalman**:

Modify the Kalman wrapper in [`forecasting.py`](operator1/models/forecasting.py) to:

1. **Detect forward-filled structure**: Check if the series has < 20 unique values in 252+ rows (indicating quarterly/annual forward-fill)
2. **Mark true observation days**: Days where the value changes are "real observations"; other days are "stale repeats"
3. **Variable observation noise**: On real observation days, use small observation noise (R_obs). On stale days, use very large observation noise (R_stale >> R_obs), effectively telling the Kalman to ignore the repeated value
4. **Growing uncertainty**: Between filings, the state uncertainty grows naturally via process noise, reflecting genuine uncertainty about the company's financials between reports

This makes the Kalman:
- Produce meaningful (non-zero) RMSE on validation data
- Generate forecasts that decay toward a mean between filings
- Widen confidence bands as time since last filing increases
- Tighten confidence bands right after a new filing

**Implementation**:
```python
# In KalmanWrapper.fit()
unique_ratio = series.nunique() / len(series)
if unique_ratio < 0.05:
    # Forward-filled quarterly data -- use filing-aware mode
    changes = series.diff().abs() > 0
    R = np.where(changes, R_obs, R_stale)  # variable observation noise
```

**Files to modify**:
- [`operator1/models/forecasting.py`](operator1/models/forecasting.py) -- Modify Kalman wrapper with filing-aware observation noise

---

## Part 3: Per-Region News Sources

### D3-Revised: Regional News RSS Feeds

**Problem**: GNews searches English articles only. Non-US companies get zero news sentiment.

**Fix**: Add per-region RSS feed URLs to [`news_sentiment.py`](operator1/features/news_sentiment.py):

| Region | Source | RSS/API | Language |
|--------|--------|---------|----------|
| Korea | Naver Finance | `https://finance.naver.com/item/news_news.naver?code={ticker}` | Korean |
| Japan | Yahoo Japan Finance | `https://finance.yahoo.co.jp/news/list/{ticker}` | Japanese |
| Brazil | InfoMoney | `https://www.infomoney.com.br/cotacoes/{ticker}` | Portuguese |
| China | Sina Finance | `https://finance.sina.com.cn/stock/{ticker}` | Chinese |
| Taiwan | TWSE MOPS news | `https://mops.twse.com.tw/mops/web/t05sr01_1` | Chinese |
| UK/EU | Google News RSS | `https://news.google.com/rss/search?q={company}` | English |
| US | Google News RSS (existing) | Already works via gnews | English |

**Strategy in `compute_news_sentiment()`**:
1. Try gnews/Google News with company English name (existing)
2. If no results + non-US market: try per-region RSS with local ticker
3. Parse RSS feeds with `feedparser` (already installed)
4. For non-English articles: use LLM (OpenRouter) to translate headlines before sentiment scoring

**Files to modify**:
- [`operator1/features/news_sentiment.py`](operator1/features/news_sentiment.py) -- Add regional RSS feed sources and multi-language search

---

## Part 4: FRED Macro Fallback Chain

### D4-Revised: FRED -> World Bank -> Static Defaults

**Problem**: FRED fetch fails for 4/5 US indicators. The missing data cascades to ~10 downstream NaN columns (macro_quadrant, fh_growth_score, fh_liquidity_score, etc.).

**Cascade impact**:
```
FRED fails -> macro_quadrant = unknown -> fh_growth_score = NaN
           -> macro_alignment = empty  -> country_survival_flag = always 0
           -> macro columns in cache = missing
```

**Fix -- Three-tier fallback**:

1. **Tier 1: FRED** (preferred, monthly/quarterly, already implemented)
2. **Tier 2: World Bank API** (`wbgapi`, annual, already installed):
   ```
   gdp_growth: NY.GDP.MKTP.KD.ZG
   inflation: FP.CPI.TOTL.ZG
   unemployment: SL.UEM.TOTL.ZS
   exchange_rate: PA.NUS.FCRF
   ```
3. **Tier 3: Static recent values** (hardcoded from latest available):
   - For US: GDP ~2.8%, inflation ~3.2%, unemployment ~4.2%, fed_funds ~5.25%
   - Updated annually. Stored in `config/macro_fallback_defaults.yml`
   - Logged with `[STATIC FALLBACK]` warning

**Files to modify**:
- [`operator1/clients/macro_provider.py`](operator1/clients/macro_provider.py) -- Add World Bank + static fallback after FRED failure
- New: `config/macro_fallback_defaults.yml` -- Static defaults per country

---

## Part 5: DART Sparse Data -- LLM Filing Extraction

### D1-Revised: Use LLM Filing Extraction (PIT-Compliant)

**Problem**: DART provides only 6 canonical fields. 19 derived ratios are all-NaN.

**Fix**: The filing extraction pipeline is already built:
- [`filing_discoverer.py`](operator1/clients/filing_discoverer.py) has `try_filing_extraction()`
- [`llm_filing_extractor.py`](operator1/clients/llm_filing_extractor.py) has `extract_from_pdf()`

Wire it into `kr_dart_wrapper.py`:
1. After the DART API returns sparse data, check if key fields are missing
2. If `current_liabilities` or `interest_expense` are missing, try filing extraction
3. Use OpenRouter LLM to parse the Korean PDF filing
4. Merge extracted fields into the canonical DataFrame
5. This is PIT-compliant because it extracts from the actual filed document

**Files to modify**:
- [`operator1/clients/kr_dart_wrapper.py`](operator1/clients/kr_dart_wrapper.py) -- Add LLM filing extraction fallback for sparse data

---

## Part 6: Regime NaN Labels

### D5: Forward-Fill NaN Regime Labels

Forward-fill the 5 NaN regime days from HMM warmup in [`regime_detector.py`](operator1/models/regime_detector.py).

---

## Execution Order

```
P1: Next-year OHLCV chart (quick)
P2: Next-day Low annotation (quick)
D2: Filing-aware Kalman (medium)
D5: Forward-fill regime NaN (quick)
D4: FRED -> World Bank -> static fallback (medium)
D3: Per-region news RSS (medium)
D1: LLM filing extraction for DART (larger)
```

## Todo List

```
[ ] P1: Add Chart 10 -- next-year OHLCV candlestick
[ ] P2: Add next-day Low annotation to price chart
[ ] D2: Modify Kalman for filing-aware observation noise
[ ] D5: Forward-fill NaN regime labels
[ ] D4: Add FRED -> World Bank -> static fallback chain
[ ] D3: Add per-region news RSS feeds
[ ] D1: Wire LLM filing extraction into DART wrapper
[ ] Run tests
[ ] Push to PR
```

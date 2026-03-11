# Plan: Data Quality Solutions and Missing Charts

## Part 1: Missing Charts

### P1: Add Next-Year OHLCV Chart (Chart 10)

**Problem**: The OHLC predictor generates 252 candles for the next year (`ohlc_predictions.next_year.series` in the profile), but [`report_generator.py`](operator1/report/report_generator.py) only renders Charts 7 (next-month) and 8 (next-week). The next-year prediction is available in the profile JSON but has no chart.

**Fix**: Add a Chart 10 for next-year OHLC candles after Chart 8 (next-week) in `generate_charts()`:
- Use the same candlestick drawing logic as Charts 7/8
- For 252 candles, group into weekly candles (OHLC aggregation: Open=first open, High=max high, Low=min low, Close=last close) to avoid visual clutter
- This gives ~50 weekly bars instead of 252 daily bars
- Add the predicted annual return annotation
- Save as `predicted_ohlc_year.png`
- Add to `_embed_charts_in_markdown()` section mapping

**Files to modify**:
- [`operator1/report/report_generator.py`](operator1/report/report_generator.py:3280) -- Add Chart 10 after Chart 8

### P2: Add Next-Day Low Estimate Annotation to Price Chart

**Problem**: The Technical Alpha mask exposes only the next-day Low estimate (`estimated_low=19843.6373` in the DART run). This appears in the report text but is not visualized on any chart.

**Fix**: Add a horizontal line or marker on Chart 1 (Price History) showing the next-day estimated Low:
- Read `profile.get("ohlc_predictions", {}).get("next_day", {}).get("low")` or `profile.get("predictions", {}).get("technical_alpha", {}).get("estimated_low")`
- Draw as a dashed horizontal line at the estimated low price
- Annotate with "Next-Day Low Estimate: $XX,XXX"
- Color: warning gold to distinguish from historical data

**Files to modify**:
- [`operator1/report/report_generator.py`](operator1/report/report_generator.py:2986) -- Add annotation to Chart 1

---

## Part 2: Data Quality Solutions

### D1: DART Sparse Financial Fields (19 All-NaN Variables)

**Problem**: DART provides only 6 canonical fields per statement (revenue, operating_income, net_income, total_assets, total_equity, operating_cashflow). This means 19 derived variables and financial ratios are all-NaN.

**Solutions (in priority order)**:

1. **LLM Filing Extraction**: The `filing_discoverer.py` + `llm_filing_extractor.py` modules already exist and can extract financial data from DART filing PDFs. Wire `try_filing_extraction()` into the DART wrapper's `_fetch_financials()` method as a fallback when the API returns sparse data. This would use the OpenRouter LLM to parse Korean financial statements.

2. **yfinance Financial Supplement**: yfinance provides quarterly financials for Korean stocks via Yahoo Finance Japan. After the DART data is fetched, supplement missing fields with yfinance data:
   ```python
   import yfinance as yf
   ticker = yf.Ticker("093050.KS")
   income = ticker.quarterly_income_stmt
   balance = ticker.quarterly_balance_sheet
   ```
   This is not PIT-compliant but better than all-NaN.

3. **Derive from Available Fields**: For some ratios, compute approximations from available data:
   - `gross_margin` = cannot compute (needs `cost_of_revenue`)
   - `current_ratio` = cannot compute (needs `current_assets`, `current_liabilities`)
   - `interest_coverage` = cannot compute (needs `interest_expense`)
   - `net_margin` = `net_income / revenue` (both available)
   - `operating_margin` = `operating_income / revenue` (both available)

**Files to modify**:
- [`operator1/clients/kr_dart_wrapper.py`](operator1/clients/kr_dart_wrapper.py) -- Add yfinance supplement fallback
- [`operator1/features/derived_variables.py`](operator1/features/derived_variables.py) -- Ensure margins use available fields

### D2: Kalman MAE=0 on Constant Series

**Problem**: Kalman filter reports MAE=0.000, RMSE=0.000 for forward-filled financial variables. This gives it infinite ensemble weight (1/RMSE = infinity), drowning out other models.

**Fix**: In the Kalman wrapper's fit method, detect constant-value or near-constant series and skip:
```python
if series.nunique() < 3 or series.std() < 1e-10:
    logger.info("Skipping Kalman for %s (constant/near-constant series)", var)
    return None  # fall through to next model
```

**Files to modify**:
- [`operator1/models/forecasting.py`](operator1/models/forecasting.py) -- Add constant-series guard in Kalman wrapper

### D3: No News Sentiment for Non-English Companies

**Problem**: GNews searches for English articles only. Korean company "LF" yields no results.

**Fix**:
1. Try the company's English name first (from profile)
2. If no results, try the local-language name
3. If still no results, try the ticker + exchange name ("093050 KOSPI")
4. Add regional news sources: for Korea, try Naver Finance RSS feed

**Files to modify**:
- [`operator1/features/news_sentiment.py`](operator1/features/news_sentiment.py) -- Multi-language search strategy

### D4: Macro Data Missing (FRED 4/5 Indicators)

**Problem**: FRED fetch failures are now logged as WARNING (B16 fix), but 4/5 US indicators still fail.

**Fix**: Add fallback to World Bank API (`wbgapi`) when FRED fails:
```python
# In macro_provider.py
results = fetch_macro_fred(api_key, years)
if len(results) < 3:
    logger.info("FRED returned %d/%d; supplementing from World Bank", len(results), 5)
    wb_results = fetch_macro_wbgapi(country_code, years)
    for key, series in wb_results.items():
        if key not in results:
            results[key] = series
```

**Files to modify**:
- [`operator1/clients/macro_provider.py`](operator1/clients/macro_provider.py) -- Add FRED->wbgapi fallback chain

### D5: Regime NaN Labels (5 Days)

**Problem**: 5 days have NaN regime labels from HMM warmup period.

**Fix**: Forward-fill NaN regime labels after HMM prediction:
```python
regimes = model.predict(X_clean)
# Fill warmup NaNs with first detected regime
if np.isnan(regimes[0]):
    first_valid = np.argmax(~np.isnan(regimes))
    regimes[:first_valid] = regimes[first_valid]
```

**Files to modify**:
- [`operator1/models/regime_detector.py`](operator1/models/regime_detector.py) -- Forward-fill NaN regime labels

---

## Part 3: Where is the Next-Day Low?

The next-day Low estimate is produced by `apply_technical_alpha_mask()` in [`prediction_aggregator.py:553`](operator1/models/prediction_aggregator.py:553). It's stored in:
- `profile["predictions"]["technical_alpha"]["estimated_low"]`
- `profile["ohlc_predictions"]["next_day"]["low"]`

In the DART run: `Technical Alpha mask: last_close=20750.0000, vol=0.029120, estimated_low=19843.6373`

This appears in the report text (Section 9: Predictions) but is NOT annotated on any chart. Fix P2 above addresses this.

---

## Execution Order

```
P1: Add next-year OHLC chart (252 candles -> weekly aggregation)
P2: Add next-day Low annotation to price chart
D2: Kalman constant-series guard
D5: Forward-fill NaN regime labels
D1: Supplement DART with yfinance financial data + compute margins from available fields
D3: Multi-language news search strategy
D4: FRED -> World Bank fallback chain
```

## Todo List

```
[ ] P1: Add Chart 10 -- next-year OHLCV candlestick chart
[ ] P2: Add next-day Low estimate annotation to price history chart
[ ] D2: Skip Kalman on constant/near-constant series
[ ] D5: Forward-fill NaN regime labels after HMM fit
[ ] D1: Supplement DART sparse data from yfinance + compute missing margins
[ ] D3: Multi-language news sentiment search
[ ] D4: FRED -> World Bank macro fallback
```

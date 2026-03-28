# AAPL Backtest Failure Diagnosis and Fix Plan

9 concrete problems found in the 2023-2024 backtest predicting 2025. Each diagnosed to root cause with a specific fix.

---

## Problem 1: No close price forecast

**Evidence:** `predictions_summary.json` contains 14 forecast variables. `close` is not among them. The validation has no price comparison data.

**Root cause:** `run_forecasting()` at line 1637 loads target variables from `config/survival_hierarchy.yml`. The 5 tiers contain 17 financial health variables (cash_ratio, debt_to_equity, volatility_21d, drawdown_252d, etc.). `close`, `return_1d`, and `log_return_1d` are intentionally excluded because the pipeline was designed as a survival analyzer, not a stock price predictor.

**Fix:** Add `close` and `return_1d` to the survival hierarchy as a new **Tier 3b: Price** sub-tier, OR add them in `run_forecasting()` as hardcoded mandatory variables that always get forecasted regardless of the tier config. The second approach is cleaner because price forecasting is fundamentally different from ratio forecasting -- it needs ARIMA/EMA anchoring, not Kalman on stale filing data.

**Expert method:** Christoffersen (2012) "Elements of Financial Risk Management" -- price forecasting should use the most recent observed close as the anchor (random walk hypothesis), not a model-derived state. The correct baseline for close is `last_close * (1 + predicted_return)`, where `predicted_return` comes from the ensemble of GARCH drift + regime transition probability + DTW analog median.

**Implementation:**
1. In `run_forecasting()`, after the tier variable loop, add a dedicated close price forecasting block
2. Use the last observed close from the cache as the anchor
3. Compute predicted close = `last_close * exp(drift + vol_adjustment)` where drift comes from GARCH mean and vol_adjustment comes from regime transition probabilities
4. Store as `result.forecasts["close"]` with all 4 horizons

---

## Problem 2: 8 of 14 variables return baseline_zero

**Evidence:** `model_used` shows baseline_zero for: cash_ratio, net_debt_to_ebitda, interest_coverage, current_ratio, gross_margin, operating_margin, net_margin, ev_to_ebitda.

**Root cause:** These variables are forward-filled quarterly values. In the daily cache, cash_ratio has exactly 4-8 unique values across 502 days (one per quarterly filing). When `_extract_series(var)` returns these flat-then-step values, the Kalman filter fails (degenerate covariance), VAR fails (near-singular), LSTM fails (no learnable pattern), tree ensemble fails (insufficient splits). Then `fit_baseline()` is called, and since the series has data, it should return `baseline_ema` or `baseline_last`. But the actual result is `baseline_zero`.

Wait -- re-reading the code at line 1490-1495: `baseline_zero` is only returned when `len(clean) == 0` (all NaN). So these 8 variables are entirely NaN in the cache, meaning the estimation step failed to populate them.

**Deeper root cause:** The estimation step (Step 4b) did not fill these variables. Looking at the `_asof` suffix variables: `cash_and_equivalents_asof`, `operating_cash_flow_asof`, `total_debt_asof`, `revenue_asof` -- these are the tier config names but the actual cache columns are `cash_and_equivalents`, `operating_cash_flow`, `total_debt`, `revenue` (without `_asof`). The tier config has stale column names that don't match what the canonical translator produces.

**Fix:** Audit `config/survival_hierarchy.yml` variable names against actual cache column names from the AAPL run. Replace `cash_and_equivalents_asof` with `cash_and_equivalents`, `total_debt_asof` with `total_debt`, `operating_cash_flow_asof` with `operating_cash_flow`, `revenue_asof` with `revenue`, and `free_cash_flow_ttm` with `free_cash_flow` or `free_cash_flow_ttm_asof` (whichever the cache actually has). Also add column name resolution: if `var_name` is not in the cache, try stripping `_asof` suffix and try the base name.

**Expert method:** This is a classic schema drift bug. The fix is to add a name resolution layer in `_load_tier_variables()` that maps config names to actual cache column names using the field registry (`operator1/types.py:get_alias_map()`).

---

## Problem 3: All confidence bounds are null

**Evidence:** Every entry in `aggregated_predictions` has `lower_bound: null, upper_bound: null`.

**Root cause:** The conformal calibrator at line 2609-2662 of `forecasting.py` builds intervals from `forecast_result.residuals`. Looking at the backtest runner Stage 2, the conformal result was built (`"method=adaptive_conformal"` in logs), but the prediction aggregator is not propagating the bounds into the `HorizonPrediction` objects.

Tracing the aggregator: in `prediction_aggregator.py`, the conformal intervals are only applied when the aggregator has both the conformal result AND the variable appears in the conformal result's interval dict. The conformal result was built from `forecast_result.forecasts` which has variables like `volatility_garch`, `cash_ratio`, etc. -- but the aggregator may be looking for different variable names, or the interval width computation returns 0 because calibrator had insufficient scores.

**Fix:** 
1. Verify the conformal calibrator received enough residuals (log says "172 total calibration scores" which should be sufficient)
2. Check if `build_conformal_result()` is correctly mapping variable names between the forecast result and the conformal output
3. Ensure the prediction aggregator's `_apply_conformal_intervals()` reads from the correct conformal result path

---

## Problem 4: OHLC next-day predictions all null in backtest_runner output

**Evidence:** `ohlc_next_day: {open: null, high: null, low: null, close: null}` in predictions_summary.json. But the Stage 2 log shows "OHLC prediction complete: 279 candles generated" and "Next-day Low estimate: 247.0009".

**Root cause:** This is a bug in `backtest_runner.py`'s `extract_predictions()` function. It reads `getattr(state.ohlc_result, "next_open", None)` but the `OHLCPredictionResult` dataclass stores the next-day prediction as `result.next_day` (an `OHLCCandle` object with `.open`, `.high`, `.low`, `.close` attributes), not as `result.next_open`.

**Fix:** Change the extraction in `extract_predictions()` to:
```python
if state.ohlc_result is not None and state.ohlc_result.fitted and state.ohlc_result.next_day:
    nd = state.ohlc_result.next_day
    summary["ohlc_next_day"] = {
        "open": nd.open, "high": nd.high, "low": nd.low, "close": nd.close,
    }
```

---

## Problem 5: Volatility underestimated by 35%

**Evidence:** GARCH predicted 0.0128, VAR predicted 0.0103 (21d), actual was 0.0200.

**Root cause:** Single-regime GARCH(1,1) mean-reverts to the unconditional variance of the training sample. The 2023-2024 period was relatively low-volatility for Apple. GARCH cannot anticipate a regime change to higher volatility in 2025. The HMM classified 495 of 502 days as "high_vol" and only 2 as "bull" -- effectively a single regime, so regime-switching added nothing.

**Fix (expert methods):**

1. **Markov-Switching GARCH (Hamilton 1989):** When the HMM identifies only 1 effective regime (>95% of observations in one state), force a 2-regime split at the median volatility point. This ensures the forecast includes non-zero transition probability to a higher-vol state.

2. **HAR-RV (Corsi 2009):** The HAR model is already implemented in the code (line 1773-1796) but stores its output as `volatility_har` which is separate from the `volatility_21d` forecast. The HAR forecast should be blended with GARCH, not stored as a parallel output. HAR captures long-memory (monthly RV drives weekly, weekly drives daily) which GARCH misses.

3. **VIX-implied forward volatility:** For US stocks, yfinance provides option chain data. Extract the at-the-money implied volatility from the nearest-expiry options. This is the market's forward-looking vol estimate and would have shown ~0.020 when GARCH showed 0.013.

**Recommended approach:** Blend GARCH + HAR-RV + regime transition widening:
```
vol_forecast = 0.4 * garch_vol + 0.3 * har_rv_vol + 0.3 * regime_transition_vol
```
Where `regime_transition_vol` = `current_vol + transition_prob * (high_regime_vol - current_vol)`.

---

## Problem 6: Max drawdown underestimated 6x

**Evidence:** Forecast drawdown_252d = -5.0%, actual = -30.2%.

**Root cause:** The drawdown is forecasted using VAR(lag=1) on the `drawdown_252d` time series itself. VAR on a drawdown series extrapolates the current trend -- Apple's drawdown was recovering at end-2024, so VAR predicted continued recovery (-5%). It has no mechanism to predict a NEW drawdown that hasn't started yet.

**Fix (expert methods):**

1. **Monte Carlo max-drawdown distribution (standard practice):** The MC simulation already runs 10,000 paths. For each path, compute the maximum drawdown over the forecast horizon. Report the median, P10, and P90 of the max-drawdown distribution. This replaces the meaningless VAR point forecast. Implementation: in `monte_carlo.py` after path generation, add `max_dd = min((path - np.maximum.accumulate(path)) / np.maximum.accumulate(path))` per path.

2. **Optimal stopping / Cusum-Shiryaev (Shiryaev 2007):** Model drawdown as a sequential detection problem. Compute the posterior probability of being in a deepening drawdown using the change-point detection that is already implemented (PELT, BCP). When the detection statistic is rising, widen the drawdown forecast.

3. **Copula joint drawdown:** When peer entities (MSFT, GOOGL, etc.) are simultaneously in drawdown, the probability of a deep AAPL drawdown rises sharply. Use the fitted copula's lower tail dependence coefficient to scale the MC drawdown paths. This requires linked entity data (Problem 7).

**Recommended approach:** Replace VAR drawdown forecast with MC P10 drawdown distribution. This is a 15-line change in the profile builder to read from `mc_result.max_drawdown_distribution` instead of `forecast_result.forecasts["drawdown_252d"]`.

---

## Problem 7: Entity discovery found 0 entities

**Evidence:** Log shows "Discovery complete: 0 entities across 0 groups, 28 search calls, 28 dropped". GLEIF found 8 subsidiaries but 0 competitors/suppliers/customers.

**Root cause:** Two cascading failures:
1. The LLM (OpenRouter) rate-limited after entity proposal -- the discovery module makes 3+ LLM calls (propose entities, resolve tickers, validate matches) and all hit rate limits
2. The SIC code fallback searched for SIC 3571 "electronic computers" peers in EDGAR but returned 0 (Apple is the only major company still classified under this obsolete SIC code; Microsoft is 7372, Google is 7372, Samsung is Korean)

**Fix (expert methods):**

1. **Hardcoded whale competitor registry (immediate fix):** For mega-cap companies (top 20 per exchange), maintain a static YAML mapping. Apple's competitors are MSFT, GOOGL, SAMSUNG, AMZN -- these don't change. This costs zero LLM calls and zero API calls. Already proposed in `plans/whale-company-detection-and-adaptive-entity-discovery.md`.

2. **10-K supply chain extraction (no LLM):** SEC 10-K filings contain "Risk Factors" and "Business" sections that name specific suppliers and customers. Use regex patterns: `"principal suppliers include"`, `"significant customers include"`, `"we compete with"`. The edgartools library provides `Filing.text()` for text extraction.

3. **Single-call LLM strategy:** Instead of 3 separate LLM calls, combine into ONE call: "For {company}, list the top 3 competitors, top 3 suppliers, and top 3 customers. Return as JSON." This stays within rate limits for even the most restrictive free tier.

---

## Problem 8: Financial health score too conservative (48.7 "Fair" for Apple)

**Evidence:** Apple -- the most profitable and cash-rich company on Earth -- scored 48.7/100 ("Fair"). This is clearly wrong.

**Root cause:** The FH scoring uses percentile-rank normalization against the company's own history. Apple's high debt ($100B+) drags the solvency tier severely. But Apple's debt is investment-grade with 29x interest coverage and $162B cash reserve. The scoring treats debt-to-equity the same regardless of whether the company can service it easily or is struggling.

**Fix (expert methods):**

1. **Debt serviceability override (Moody's approach):** When interest_coverage > 10x AND cash_and_equivalents > total_debt, the solvency tier score cannot go below 40/100. This is how credit rating agencies think -- high debt with high coverage is NOT a risk signal.

2. **Merton Distance-to-Default (Merton 1974):** Model equity as a call option on firm assets. DD = (ln(V/D) + (r - 0.5*sigma^2)*T) / (sigma * sqrt(T)). Apple's DD would be ~8-10, placing it in the safest 0.1% of all companies. Add DD as a solvency tier modifier.

3. **Sector-relative scoring (Piotroski 2000):** Instead of absolute thresholds, score each ratio against the sector median. Apple's D/E of 1.7 is average for tech (which routinely uses debt for buybacks) but would be terrible for utilities. The adaptive thresholds module partially does this but it had no peer data because entity discovery failed (Problem 7). Fixing Problem 7 fixes this.

---

## Problem 9: News sentiment scores all NaN

**Evidence:** Log shows "Scored 50 headlines via Gemini" but then "expected 50 scores, got NoneType" and "mean=nan, latest=nan (Unknown)".

**Root cause:** The OpenRouter LLM returned a response that failed JSON parsing. The sentiment module sends 50 headlines to the LLM asking for a JSON array of scores. The LLM returned text that wasn't valid JSON (perhaps markdown-wrapped or with preamble text). The parser returned None, which becomes NaN for all scores.

**Fix:**

1. **Robust JSON extraction:** Instead of `json.loads(response)`, use a regex to extract the JSON array from anywhere in the response: `re.search(r'\[[\s\S]*\]', response)`. LLMs frequently wrap JSON in markdown code blocks or add preamble.

2. **VADER fallback (already exists but not triggered):** The code has a VADER sentiment fallback path but it's only used when no LLM client is available. Change the logic: if LLM scoring fails (returns None/NaN), fall back to VADER immediately rather than returning NaN.

3. **Batch size reduction:** Instead of sending all 50 headlines in one call, batch into groups of 10. This reduces the chance of rate limiting and makes the JSON response smaller and easier to parse.

---

## Implementation Priority

| # | Problem | Severity | Complexity | Fix |
|---|---------|----------|------------|-----|
| 1 | No close forecast | Critical | Medium | Add close/return_1d to forecast targets with ARIMA anchor |
| 2 | 8 vars baseline_zero | Critical | Low | Fix column name mismatch in survival_hierarchy.yml |
| 3 | Null confidence bounds | High | Low | Debug conformal-to-aggregator propagation path |
| 4 | OHLC nulls in runner | Medium | Trivial | Fix attribute name in extract_predictions() |
| 5 | Vol underestimated | High | Medium | Blend GARCH + HAR-RV + regime transition widening |
| 6 | Drawdown underestimated | High | Low | Replace VAR drawdown with MC P10 distribution |
| 7 | 0 entities | Critical | Medium | Hardcoded whale competitors + single-call LLM + 10-K regex |
| 8 | FH too conservative | Medium | Medium | Debt serviceability override + Merton DD |
| 9 | Sentiment NaN | Medium | Low | Robust JSON extraction + VADER fallback |

**Execution order (sequential, not parallel):**
1. Problem 4 first (trivial backtest runner bug)
2. Problem 2 (column name mismatch -- unlocks 8 variables)
3. Problem 1 (add close price forecasting)
4. Problem 9 (sentiment NaN -- low complexity, high visibility)
5. Problem 3 (conformal bounds propagation)
6. Problem 6 (MC drawdown distribution)
7. Problem 5 (volatility blending)
8. Problem 7 (entity discovery -- medium complexity, unlocks peer-relative scoring)
9. Problem 8 (FH calibration -- depends on Problem 7 for sector peers)

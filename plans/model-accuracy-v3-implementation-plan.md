# Model Accuracy v3 -- Complete Implementation Plan

Based on AAPL backtest post-mortem (3.13% 1d error, 9.69% 21d error) and detailed review of all 56 client files (37,170 lines) plus the full pipeline architecture.

---

## Gap 1: Options-Derived Forward-Looking Signals

### New File: `operator1/features/options_signals.py`

**Purpose:** Fetch full options surface data and compute 6 forward-looking features that predict directional moves before they happen in price.

**Data source:** `yfinance.Ticker.option_chain()` -- already installed, no new packages.

**Functions:**

```
fetch_options_signals(ticker: str, cache: pd.DataFrame) -> tuple[pd.DataFrame, OptionsSignalResult]
```

1. `_fetch_options_chain(ticker)` -- calls `yf.Ticker(ticker).option_chain(exp_date)` for nearest 2 expirations. Returns calls_df, puts_df with strike, IV, volume, openInterest.

2. `_compute_put_call_ratio(calls_df, puts_df)` -- `sum(puts.volume) / sum(calls.volume)`. Above 1.0 = bearish. Inject as `put_call_ratio` column.

3. `_compute_risk_reversal(calls_df, puts_df, spot_price)` -- Interpolate to find 25-delta call IV and 25-delta put IV using Black-Scholes delta. `risk_reversal_25d = call_iv_25d - put_iv_25d`. Negative = puts more expensive = bearish.

4. `_compute_iv_skew(calls_df, puts_df, spot_price)` -- `iv_skew = OTM_put_IV(90% strike) / ATM_IV(100% strike)`. Above 1.2 = elevated tail fear.

5. `_compute_vix_term_structure()` -- Fetch `^VIX` and `^VIX3M` via yfinance.download. `vix_term_structure = VIX / VIX3M`. Above 1.0 = backwardation = stress.

6. `_compute_skew_index()` -- Fetch `^SKEW` via yfinance. Raw value (typically 100-170, higher = more tail risk).

7. `_compute_variance_risk_premium(iv30, cache)` -- `variance_risk_premium = iv30^2 - realized_vol_21d^2` (annualized). Positive = market expects more vol than realized.

**Cache columns added (6):**
- `put_call_ratio` (float, 0-5+)
- `risk_reversal_25d` (float, -0.3 to +0.3)
- `iv_skew` (float, 0.5-2.0)
- `vix_term_structure` (float, 0.7-1.3)
- `skew_index` (float, 100-170)
- `variance_risk_premium` (float, -0.1 to +0.1)

**Wiring in main.py:**
- Step 4a.7 (after `fetch_implied_volatility` at line ~996, before estimation at line ~1246)
- Add all 6 features to `_extra_vars` list (line ~2559)
- Add to `_hmm_lookahead_cols` exclusion: none needed (all are forward-looking market data, not model outputs)

**Wiring in backtest_runner.py:**
- Add after IV fetch (line ~562) in `run_stage1()`
- Same pattern as IV: try/except with `logger.debug` on failure

**Wiring in staged pipeline:**
- Not needed (these are Step 4 features, computed before temporal stages 3-7)

**Error handling:**
- Options chain unavailable for non-US markets: return empty (all 6 columns NaN). Only US-listed stocks with liquid options will have data.
- Weekend/holiday: use last available chain data (yfinance caches recent chains)
- Illiquid options (volume < 100): set `put_call_ratio` to NaN to avoid noise

**Profile integration:**
- Add to `profile["options_signals"]` with latest values for all 6 features
- Report section: extend Section 19 (Advanced Quantitative Insights) in Premium tier

---

## Gap 2: Geographic Supply Chain Concentration Risk

### Modified Files:
1. `operator1/features/product_metrics.py` -- extend with geographic HHI
2. `operator1/clients/us_edgar.py` -- parse `GeographicAreasMember` alongside `OperatingSegmentsMember`
3. `operator1/clients/eu_esef_wrapper.py` -- parse geographic dimension in XBRL JSON
4. `operator1/features/conflict_risk.py` -- add `compute_supply_chain_geography()`

### New File: None (extend existing modules)

**Changes to `product_metrics.py`:**

Add function:
```
compute_geographic_metrics(cache: pd.DataFrame, geo_segments: dict, gleif_subsidiaries: list) -> pd.DataFrame
```

1. `geo_hhi` -- HHI of revenue across countries (from XBRL geographic segments). Same formula as `segment_hhi` but on geographic data.

2. `china_revenue_pct` -- Revenue from China/Greater China / total revenue. Extracted from geographic segment names containing "China", "PRC", "Greater China", "Asia Pacific" (weighted 0.5 for Asia Pacific).

3. `supply_chain_geo_hhi` -- HHI of GLEIF subsidiary country distribution. Uses `relationships["subsidiaries"]` from Step 5e.1 (already fetched). Each subsidiary has a `country` field.

**Changes to `us_edgar.py` `_extract_segments_from_companyfacts()`:**

The existing method searches for `StatementBusinessSegmentsAxis` and `ProductOrServiceAxis`. Add:
- `srt:StatementGeographicalAxis` to `segment_axes` list (line ~629)
- Return both `product_segments` and `geo_segments` in the result dict
- New key: `"geo_segments": {"Americas": 50B, "Europe": 30B, "Greater China": 20B}`

**Changes to `eu_esef_wrapper.py` `_extract_segments_from_xbrl_dimensions()`:**

Add geographic axis detection:
- `ifrs-full:GeographicAreasAxis` to `segment_axes` set (line ~1039)
- Store geographic facts separately from operating segment facts
- Return `geo_segments` alongside `segments` in result dict

**Trade Policy Uncertainty Index:**

Add to `macro_provider.py`:
```
fetch_trade_policy_uncertainty() -> pd.Series
```
- Download CSV from `policyuncertainty.com/data/Trade_Policy_Uncertainty_Index.csv`
- Parse date column, resample to daily via ffill
- Cache to disk (monthly updates)

**New cache columns (5):**
- `geo_hhi` (float, 0-1)
- `china_revenue_pct` (float, 0-1)
- `supply_chain_geo_hhi` (float, 0-1)
- `trade_policy_uncertainty` (float, 0-1000)
- `tariff_exposure_score` (float, composite: `geo_hhi * tpu * china_pct`)

**Wiring in main.py:**
- Step 5i.6 (after segment extraction, line ~2060): call `compute_geographic_metrics()` with geo_segments from `extract_segment_data()` + GLEIF subsidiaries from `relationships["subsidiaries"]`
- Add all 5 features to `_extra_vars` list

---

## Gap 3: Cross-Asset Sector Rotation Signals

### New File: `operator1/features/cross_asset_signals.py`

**Purpose:** Track 11 sector ETFs + cross-asset indicators to detect institutional rotation before it hits individual stocks.

**Data source:** `yfinance.download()` batch fetch -- single API call for all tickers.

**Functions:**

```
compute_cross_asset_signals(cache: pd.DataFrame, sector: str) -> tuple[pd.DataFrame, CrossAssetResult]
```

1. `_fetch_sector_etf_data(years=2)` -- Batch fetch via `yf.download(["XLK","XLE","XLF","XLV","XLI","XLP","XLY","XLB","XLU","XLRE","XLC","SPY"], period="2y")`. Returns DataFrame with daily close per ETF.

2. `_compute_sector_relative_strength(etf_data, target_sector)` -- Map target company's sector (from `target_profile["sector"]`) to the matching ETF (Technology -> XLK, Energy -> XLE, etc.). Compute 21d rolling `sector_etf_return / SPY_return`.

3. `_compute_sector_rank(etf_data)` -- Rank all 11 sectors by 252d return. Return target sector's rank (1=best, 11=worst).

4. `_compute_sector_dispersion(etf_data)` -- 21d rolling cross-sectional std of all 11 sector ETF daily returns. High dispersion = differentiation = rotation in progress.

5. `_fetch_cross_asset_indicators()` -- Fetch `^TNX` (10Y yield), `^IRX` (3M yield), `DX-Y.NYB` (USD index), `GC=F` (gold) via yfinance.download. Compute:
   - `yield_curve_10y2y = TNX - IRX` (or use FRED T10Y2Y if FRED_API_KEY available)
   - `usd_momentum_21d = DXY.pct_change(21)`

6. `_compute_cross_asset_stress(sector_dispersion, yield_change, usd_change, gold_change)` -- Composite: normalize each to z-score, average. High = stress environment.

**Sector mapping dict:**
```python
_SECTOR_TO_ETF = {
    "Technology": "XLK", "Energy": "XLE", "Financial Services": "XLF",
    "Healthcare": "XLV", "Industrials": "XLI", "Consumer Defensive": "XLP",
    "Consumer Cyclical": "XLY", "Basic Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}
```

**Cache columns added (6):**
- `sector_relative_strength` (float, 0.5-2.0)
- `sector_rank_12m` (int, 1-11)
- `sector_dispersion` (float, 0-0.05)
- `yield_curve_10y2y` (float, -3.0 to +4.0 pct)
- `usd_momentum_21d` (float, -0.1 to +0.1)
- `cross_asset_stress` (float, z-score composite)

**Wiring in main.py:**
- Step 4a.8 (after `fetch_sector_leading_indicators` at line ~1013, before estimation)
- Add all 6 to `_extra_vars`

---

## Gap 4: Event Calendar and Event-Driven Uncertainty

### New File: `operator1/features/event_calendar.py`

**Purpose:** Track known upcoming events and adjust prediction confidence/interval width based on event proximity.

**Data sources:**
- FOMC dates: static list + FRED release calendar
- Earnings dates: `yfinance.Ticker.calendar` (already partially used via `filing_calendar.py`)
- CPI/NFP dates: FRED release schedule
- Political events: static JSON file `config/political_events.json`

### New File: `config/political_events.json`

Static JSON with known political events:
```json
{
  "events": [
    {"date": "2025-01-20", "type": "inauguration", "impact": "high", "description": "US Presidential Inauguration"},
    {"date": "2025-04-15", "type": "budget", "impact": "medium", "description": "US Tax Day"},
    ...
  ],
  "fomc_dates_2025": ["2025-01-29", "2025-03-19", "2025-05-07", ...],
  "fomc_dates_2026": ["2026-01-28", "2026-03-18", ...]
}
```

**Functions:**

```
compute_event_calendar_features(cache: pd.DataFrame, ticker: str, filing_calendar_result) -> tuple[pd.DataFrame, EventCalendarResult]
```

1. `_load_fomc_dates()` -- Load from `config/political_events.json` FOMC section. These are published 12 months ahead by the Fed.

2. `_fetch_earnings_date(ticker)` -- `yf.Ticker(ticker).calendar` returns next earnings date. Also use `filing_calendar_result.next_expected_filing` (already computed).

3. `_load_political_events()` -- Load from `config/political_events.json`. Filter to next 90 days.

4. `_compute_days_to_next_event(date, fomc_dates, earnings_date, political_events)` -- Minimum days until any event.

5. `_compute_event_uncertainty_premium(days_to_event, event_type)` -- Multiplier for conformal interval width: 1.0 (no event near), 1.5 (event in 3-5 days), 2.0 (event in 1-2 days), 2.5 (event day).

6. `_compute_event_density(date, all_events, window=30)` -- Count of events in next 30 days. High density = information overload discount.

**Cache columns added (5):**
- `days_to_next_event` (int, 0-90)
- `event_uncertainty_premium` (float, 1.0-2.5)
- `fomc_proximity` (int, 0-42)
- `earnings_proximity` (int, 0-90, from filing_calendar)
- `event_density_30d` (int, 0-10)

**Wiring in main.py:**
- Step 4a.9 (new step, after cross-asset signals)
- `event_uncertainty_premium` consumed by conformal prediction in `build_conformal_result()` (line ~3078): multiply interval width by this factor
- `days_to_next_event` consumed by prediction aggregator: reduce confidence when < 5

**Wiring in conformal.py:**
- `build_conformal_result()` accepts new `event_uncertainty_premium` parameter
- Multiply final interval half-width by this factor

---

## Gap 5: DCF Model Calibration

### Modified File: `operator1/hedge_fund/engine.py` (`_compute_dcf()`)

**5 fixes to existing DCF:**

1. **Growth prior fix:** Replace `mc_result.regime_distributions` sampling (which produces near-zero growth in high_vol regimes) with company's own 3-year revenue CAGR:
```python
# Before: growth = sample from regime distribution
# After:
revenue_series = income_df["revenue"].dropna()
if len(revenue_series) >= 8:  # 2+ years quarterly
    cagr_3y = (revenue_series.iloc[-1] / revenue_series.iloc[-12]) ** (1/3) - 1
    growth = max(0.02, min(cagr_3y, 0.30))  # clamp to 2-30%
```

2. **WACC cap:** Use sector median WACC as Bayesian prior:
```python
# Compute sector median WACC from peer data if available
sector_wacc = 0.09  # default
if peer_ranking_result and peer_ranking_result.get("n_peers", 0) > 3:
    # Use peer FCF yields as WACC proxy
    ...
wacc = min(computed_wacc, sector_wacc + 0.02)  # cap at sector + 2%
```

3. **3-stage model for high-growth:** When revenue CAGR > 10%, use:
   - Years 1-5: company CAGR (high growth phase)
   - Years 6-10: fade linearly to terminal growth
   - Years 11+: terminal growth (3%)

4. **Reverse DCF:** Add `_compute_reverse_dcf()`:
```python
def _compute_reverse_dcf(current_price, fcf, shares, wacc, terminal_growth=0.03):
    # Solve: price = sum(FCF * (1+g)^t / (1+wacc)^t) + terminal
    # Binary search for g that makes NPV = current_price * shares
    ...
    return implied_growth_rate
```

5. **Sanity gate:** After DCF computation:
```python
ratio = intrinsic_p50 / current_price
if ratio > 5.0 or ratio < 0.2:
    dcf_result.reliable = False
    dcf_result.warning = f"DCF/price ratio {ratio:.1f}x is extreme"
```

**Profile integration:**
- Add `implied_growth_rate` and `reliable` flag to `profile["hedge_fund"]["dcf"]`
- Report narrative includes reverse DCF comparison

---

## Gap 6: Ensemble Model Diversity

### Modified Files:
1. `operator1/models/prediction_aggregator.py` -- feature-driven model routing + reject option
2. `operator1/models/prediction_aggregator.py` (`FixedShareForecaster`) -- adaptive share parameter
3. `operator1/models/genetic_optimizer.py` -- equal-weight fallback

**Change 1: Feature-driven model router**

Add to `prediction_aggregator.py`:
```python
def _compute_model_routing_weights(cache: pd.DataFrame) -> dict[str, float]:
    """Route to dominant model based on current market characteristics."""
    latest = cache.iloc[-1]
    weights = {"kalman": 1.0, "garch": 1.0, "var": 1.0, "lstm": 1.0, "tree": 1.0, "baseline": 1.0}
    
    # ADX > 25 = strong trend -> Kalman dominates
    adx = latest.get("adx_14", 20)
    if adx > 25:
        weights["kalman"] *= 2.0
    
    # IV-RV spread > 0 = vol expansion expected -> GARCH dominates
    iv_rv = latest.get("iv_rv_spread", 0)
    if iv_rv > 0.05:
        weights["garch"] *= 2.0
    
    # Low autocorrelation = mean reverting -> baseline/mean-reversion
    # (autocorrelation computed from return_1d)
    ...
    
    # VIX > 25 = stressed -> tree ensemble (handles non-linearity)
    vix_ts = latest.get("vix_term_structure", 1.0)
    if vix_ts > 1.0:  # backwardation
        weights["tree"] *= 1.5
    
    # Normalize to sum to 1
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}
```

Wire into `run_prediction_aggregation()` as an additional weight source that's multiplied with existing inverse-RMSE/FixedShare weights.

**Change 2: Adaptive FixedShare**

Modify `FixedShareForecaster.__init__()`:
```python
def __init__(self, model_names, share=0.05, adaptive=True):
    self._base_share = share
    self._adaptive = adaptive
    ...

def update(self, losses, online_change_score=0.0):
    if self._adaptive and online_change_score > 0.5:
        # High change score = regime change detected
        # Increase share parameter to adapt faster
        effective_share = min(self._base_share * (1 + online_change_score * 2), 0.3)
    else:
        effective_share = self._base_share
    ...
```

**Change 3: Equal-weight fallback in GA**

In `genetic_optimizer.py`, after GA converges:
```python
if ga_result.fitted:
    max_weight = max(ga_result.best_weights.values())
    if max_weight > 0.85:
        # Single-model dominance detected
        # Force equal weights as diversity mechanism
        n_models = len(ga_result.best_weights)
        equal = {k: 1.0/n_models for k in ga_result.best_weights}
        # Blend: 70% GA + 30% equal
        ga_result.best_weights = {
            k: 0.7 * ga_result.best_weights[k] + 0.3 * equal[k]
            for k in ga_result.best_weights
        }
        ga_result.diversity_forced = True
```

**Change 4: Reject option**

In `run_prediction_aggregation()`, after computing intervals:
```python
for var, horizons in pred_result.predictions.items():
    for h, hp in horizons.items():
        if hp.upper_ci and hp.lower_ci and hp.point_forecast:
            interval_width = hp.upper_ci - hp.lower_ci
            if var == "close" and interval_width > 2.0 * abs(hp.point_forecast):
                # Interval wider than 2x the price = "no opinion"
                hp.point_forecast = cache[var].iloc[-1]  # last close
                hp.confidence *= 0.3  # reduce confidence
                hp.reject_flag = True
```

---

## Pipeline Wiring Summary

### main.py Step Order (new steps in bold):

```
Step 4a.5: SIX proxies (CH only)
Step 4a.6: Pre-estimation ratios
**Step 4a.7: Options signals (Gap 1)**
**Step 4a.8: Cross-asset rotation signals (Gap 3)**
**Step 4a.9: Event calendar features (Gap 4)**
Step 4b:   Estimation
Step 4c:   Filing calendar
Step 5:    Derived variables
Step 5i.6: Segment extraction + **geographic metrics (Gap 2)**
...
Step 6:    Temporal models (with new _extra_vars from Gaps 1-4)
Step 6-HF: Hedge fund analysis (with **DCF fixes, Gap 5**)
Step 6r:   Prediction aggregation (with **ensemble fixes, Gap 6**)
```

### backtest_runner.py Parity:

All 6 gaps must also be added to `run_stage1()` in `backtest_runner.py`:
- Gap 1: After IV fetch (line ~562)
- Gap 2: After segment extraction (line ~1131)
- Gap 3: After benchmark returns (line ~555)
- Gap 4: After filing calendar (line ~720)
- Gap 5: In HF engine (automatic via engine.py changes)
- Gap 6: In prediction aggregator (automatic via code changes)

### Staged Pipeline:

Gaps 1-4 are Step 4 features (before Stage 3). No stage module changes needed.
Gap 5 runs in Stage 7.5 (HF analysis). No stage module changes needed.
Gap 6 runs in Stage 6.5 (prediction aggregation). No stage module changes needed.

---

## New Files Summary

| File | Lines (est.) | Gap |
|------|-------------|-----|
| `operator1/features/options_signals.py` | ~300 | Gap 1 |
| `operator1/features/cross_asset_signals.py` | ~250 | Gap 3 |
| `operator1/features/event_calendar.py` | ~200 | Gap 4 |
| `config/political_events.json` | ~100 | Gap 4 |

## Modified Files Summary

| File | Changes | Gap |
|------|---------|-----|
| `main.py` | +3 new steps (4a.7, 4a.8, 4a.9), extend _extra_vars | Gaps 1,2,3,4 |
| `backtest_runner.py` | +3 new steps in run_stage1() | Gaps 1,2,3,4 |
| `operator1/features/product_metrics.py` | +compute_geographic_metrics() | Gap 2 |
| `operator1/clients/us_edgar.py` | +GeographicAreasMember parsing | Gap 2 |
| `operator1/clients/eu_esef_wrapper.py` | +GeographicAreasAxis parsing | Gap 2 |
| `operator1/clients/macro_provider.py` | +fetch_trade_policy_uncertainty() | Gap 2 |
| `operator1/hedge_fund/engine.py` | Fix _compute_dcf() (5 changes) | Gap 5 |
| `operator1/models/prediction_aggregator.py` | +model router, +reject option | Gap 6 |
| `operator1/models/prediction_aggregator.py` | Adaptive FixedShare | Gap 6 |
| `operator1/models/genetic_optimizer.py` | +equal-weight fallback | Gap 6 |
| `operator1/models/conformal.py` | +event_uncertainty_premium param | Gap 4 |
| `operator1/report/report_generator.py` | +options signals section | Gap 1 |
| `operator1/report/profile_builder.py` | +options/geo/event profile keys | Gaps 1,2,4 |

## Execution Order

1. **Gap 1: Options signals** -- highest impact, directly predicts moves
2. **Gap 3: Cross-asset rotation** -- captures sector flows, low effort
3. **Gap 4: Event calendar** -- widens intervals, low effort
4. **Gap 2: Geographic supply chain** -- extends existing segment infra
5. **Gap 5: DCF calibration** -- code fixes only, no new data
6. **Gap 6: Ensemble diversity** -- algorithm improvements

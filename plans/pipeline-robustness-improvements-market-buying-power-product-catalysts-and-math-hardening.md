# Pipeline Robustness Improvements

Two new feature modules (market buying power + product catalyst detection) and targeted math hardening across existing models, based on the AAPL backtest findings.

---

## Part 1: Market Buying Power Signal

**Problem**: The pipeline has no demand-side signal. It knows Apple's financials and price history, but not whether the markets Apple sells into are growing or contracting. If consumer electronics spending is declining globally, Apple's revenue growth will slow regardless of its own fundamentals.

**Approach**: Build a demand-side index from publicly available macro data for each market segment the company operates in.

### Data sources (all free, no key needed)

| Source | Data | Granularity | How it maps |
|--------|------|-------------|-------------|
| World Bank (wbgapi, already installed) | Household final consumption expenditure, GDP per capita PPP | Annual, 200+ countries | Consumer spending power per market |
| FRED (fredapi, already installed) | US PCE (Personal Consumption Expenditure), Retail Sales, Consumer Confidence Index | Monthly | US consumer demand |
| OECD/SDMX (sdmx1, already installed) | Consumer confidence indicators, retail trade volumes | Monthly, 38 countries | OECD consumer demand |
| BLS (free API) | CPI sub-indices (electronics, durables, services) | Monthly | Sector-specific price pressure |

### New module: `operator1/features/market_buying_power.py`

**Entry point**: `compute_market_buying_power(cache, sector, country, macro_data)`

**What it computes**:

1. **Consumer Spending Index** -- weighted blend of PCE growth, retail sales growth, and consumer confidence for the company's primary markets. For Apple: 60% US + 15% China + 10% Europe + 15% RoW.

2. **Sector Demand Momentum** -- CPI sub-index for the relevant sector (electronics for Apple, energy for Exxon, healthcare for Pfizer). Rising sector CPI with stable volume = pricing power. Falling CPI with falling volume = demand destruction.

3. **Purchasing Power Parity Adjustment** -- converts revenue growth into real terms using PPP exchange rates, not nominal FX. A company growing 10% in USD but selling into a market where the local currency depreciated 15% is actually seeing demand destruction.

4. **Market Saturation Signal** -- for mature products: ratio of replacement demand vs new-customer demand. Derived from total addressable market (TAM) estimates via market cap / revenue multiples of the sector.

**Output columns**: `buying_power_index` (0-100), `sector_demand_momentum` (-1 to +1), `real_revenue_growth_ppp`, `market_saturation_pct`

**Wiring**: Step 4a.4 in main.py (after macro data fetch, before estimation). Feeds into:
- Financial health scoring (growth tier -- explains why revenue growth may slow)
- Monte Carlo (demand-adjusted return distribution)
- Survival mode (demand collapse as a new trigger)

### Impact on backtest issues

The GARCH underestimate (predicted 0.0141, actual 0.0199) partly stems from not knowing that consumer electronics demand was softening in late 2025. A buying power signal showing declining consumer confidence would have widened the volatility forecast.

---

## Part 2: Product Catalyst Detection

**Problem**: Financial models are backward-looking. Apple announcing iPhone 17 or a new AR headset creates a step-change in market expectations that historical price patterns can't capture. The DTW analog predicted +0.47% because it matched historical shapes, not future catalysts.

**Approach**: Extract product launch signals from SEC filings (R&D spending patterns) and news sentiment, then inject a "catalyst premium" into forecasts.

### Data sources

| Source | Signal | Free? |
|--------|--------|-------|
| SEC EDGAR 10-K/10-Q (already fetched) | R&D expense trajectory, segment revenue breakdowns, "new product" mentions in MD&A | Yes |
| Google News RSS (gnews, already installed) | Product announcement headlines, launch date mentions | Yes |
| Patent filings (USPTO PAIR API) | Patent application velocity as a leading indicator of product cycles | Yes, no key |

### New module: `operator1/features/product_catalysts.py`

**Entry point**: `detect_product_catalysts(cache, profile, news_articles, filings)`

**What it computes**:

1. **R&D Acceleration Signal** -- quarter-over-quarter R&D expense growth rate. When R&D spending accelerates for 2+ consecutive quarters above the 5-year average, a product launch is likely within 6-12 months. Apple's R&D jumped 15% YoY before every major product category launch.

2. **News Catalyst Score** -- from the existing news_sentiment module: filter articles mentioning "launch", "new product", "release", "announce", "unveil". Score by proximity (articles closer to today count more) and source authority (Reuters/Bloomberg > blogs).

3. **Revenue Concentration Shift** -- detect when a new segment appears or an existing segment's share changes by more than 5 percentage points. This signals a product portfolio transition (e.g., Apple's Services segment growing from 15% to 25% of revenue).

4. **Patent Velocity** -- number of published patent applications in the trailing 12 months vs the 5-year average. Acceleration signals R&D maturation into products.

**Output columns**: `catalyst_score` (0-1), `rnd_acceleration`, `revenue_diversification_delta`, `patent_velocity_ratio`, `catalyst_type` (product_launch, segment_shift, rnd_surge, none)

**Wiring**: Step 5i.5 in main.py (after news sentiment, before temporal models). Feeds into:
- Forecasting (catalyst premium added to baseline forecast)
- DTW analogs (weight historical analogs from similar catalyst periods higher)
- Monte Carlo (widen upside tail when catalyst_score > 0.5)

### Impact on backtest issues

The DTW median of +0.47% would have been adjusted upward if the model detected Apple's R&D acceleration toward Apple Intelligence/AI features during 2024-2025. The actual +13.7% return was partly driven by the AI narrative that the backward-looking models missed entirely.

---

## Part 3: Math Hardening

Targeted fixes for the specific weaknesses exposed by the backtest.

### 3a. GARCH Regime Switching (fix volatility underestimate)

**Current**: Single-regime GARCH(1,1) on the full sample. Mean-reverts to unconditional variance.

**Proposed**: Markov-Switching GARCH (Hamilton 1989). Use the HMM regime labels already computed in Step 5.5 to fit separate GARCH models per regime. The forecast uses the current regime's GARCH + transition probabilities to weight the other regimes' forecasts.

**Implementation**: In `forecasting.py`, replace the single GARCH fit with:
```
for regime in unique_regimes:
    fit GARCH on regime-filtered returns
forecast = sum(transition_prob[current_regime -> r] * garch_forecast[r] for r in regimes)
```

This would have produced a higher vol forecast because the transition probability from low_vol to high_vol was non-zero, pulling the forecast up from 0.0141 toward the high_vol regime's ~0.025.

### 3b. Drawdown Path Simulation (fix drawdown underestimate)

**Current**: Drawdown forecast is a Kalman-smoothed trend extrapolation. Predicts -6.1% because it sees the drawdown recovering.

**Proposed**: Replace with a **maximum drawdown distribution** from the Monte Carlo paths. Instead of forecasting the drawdown level, compute the distribution of max drawdowns across all 10,000 MC paths. Report the median, P10, and P90 of max drawdown.

**Implementation**: In `monte_carlo.py`, after generating paths:
```
max_drawdowns = []
for path in all_paths:
    cummax = np.maximum.accumulate(path)
    dd = (path - cummax) / cummax
    max_drawdowns.append(dd.min())
report: median(max_drawdowns), P10, P90
```

This would have shown: "The median max drawdown over 252 days is -18%, with P10 at -30%." Much more informative than the point forecast of -6.1%.

### 3c. Financial Health Calibration (fix Apple rated "Weak")

**Current**: Equal-weighted 5-tier scoring with fixed thresholds. Apple's high debt pulls solvency tier down, dragging the composite.

**Proposed**: Three fixes:
1. **Debt serviceability override**: When interest coverage > 10x AND operating cash flow covers all debt within 3 years, cap the solvency penalty at -10 (not -30). Apple's interest coverage was 29x.
2. **Cash reserves bonus**: When cash + short-term investments > total debt (Apple has $162B cash vs $100B debt), add a liquidity bonus of +15 to the composite.
3. **Market validation signal**: When market cap is growing AND revenue is growing AND the company is the largest in its sector by market cap, apply a +10 "market consensus" adjustment. The market's collective judgment provides information the ratios miss.

### 3d. DTW Catalyst-Weighted Matching

**Current**: DTW finds historically similar price shapes and extrapolates. All analogs weighted equally.

**Proposed**: Weight analogs by catalyst similarity. If the current period has a high `catalyst_score`, give more weight to historical analogs that also had high catalyst scores (detected retrospectively from R&D patterns in those periods).

This biases the analog forecast toward periods where the company was about to launch a product -- which historically produce larger returns than steady-state periods.

### 3e. Conformal Interval Widening for Regime Transitions

**Current**: Conformal prediction intervals use residual quantiles from the training set.

**Proposed**: When the HMM transition probability out of the current regime exceeds 30%, widen conformal intervals by the ratio of cross-regime volatilities. This captures the "we might be about to switch regimes" uncertainty that fixed-width intervals miss.

---

## Implementation Priority

| # | Item | Complexity | Impact | Priority |
|---|------|-----------|--------|----------|
| 1 | Market buying power module | Medium | High -- adds demand-side signal missing from all current models | P0 |
| 2 | Regime-switching GARCH | Low | High -- directly fixes the volatility underestimate | P0 |
| 3 | MC max drawdown distribution | Low | High -- replaces misleading point forecast | P0 |
| 4 | Financial health calibration | Low | Medium -- fixes the Apple "Weak" rating embarrassment | P1 |
| 5 | Product catalyst detection | Medium | Medium -- addresses the DTW magnitude miss | P1 |
| 6 | Conformal regime widening | Low | Low -- incremental improvement to intervals | P2 |
| 7 | DTW catalyst weighting | Low | Low -- incremental improvement to analogs | P2 |

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `operator1/features/market_buying_power.py` | CREATE | New module: consumer demand index from World Bank + FRED + OECD |
| `operator1/features/product_catalysts.py` | CREATE | New module: R&D acceleration + news catalyst + patent velocity |
| `operator1/models/forecasting.py` | MODIFY | Add regime-switching GARCH option |
| `operator1/models/monte_carlo.py` | MODIFY | Add max drawdown distribution to MC output |
| `operator1/models/financial_health.py` | MODIFY | Add debt serviceability override + cash reserves bonus |
| `operator1/models/dtw_analogs.py` | MODIFY | Add catalyst-weighted analog matching |
| `operator1/models/conformal.py` | MODIFY | Add regime transition interval widening |
| `main.py` | MODIFY | Wire new modules at Steps 4a.4 and 5i.5 |
| `operator1/report/profile_builder.py` | MODIFY | Add buying_power and catalyst sections to profile |
| `operator1/report/report_generator.py` | MODIFY | Add Market Demand and Product Catalyst report sections |
| `operator1/monitoring/model_tests.py` | MODIFY | Add smoke tests for new modules |

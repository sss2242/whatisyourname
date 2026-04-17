# Model Accuracy Improvement Plan v3: Filling the Gaps

Based on the AAPL backtest post-mortem (3.13% 1d error, 9.69% 21d error), this plan addresses the specific data and modeling gaps that caused prediction errors. Each gap is paired with research into methods used by quantitative hedge funds and academic practitioners.

---

## Gap 1: Options-Derived Forward-Looking Signals

**The problem:** We fetch IV30 (30-day ATM implied volatility) but don't use the full options surface. The January 2025 AAPL selloff was visible in the put skew weeks before it happened -- smart money was buying downside protection.

### What experts do

**Popular methods:**
- **Put/Call Ratio** (CBOE): Simple ratio of put volume to call volume. Readings above 1.0 signal bearish positioning. Used by nearly every institutional desk.
- **IV-RV Spread** (already implemented): Implied minus realized volatility. Positive spread = market expects more vol than recent history shows. We compute this but don't use it as a forecasting feature.
- **VIX Term Structure** (contango/backwardation): When short-term VIX > long-term VIX (backwardation), it signals imminent stress. We don't fetch VIX term structure.

**Less common but effective:**
- **25-Delta Risk Reversal** (Bollen & Whaley 2004): Price difference between OTM calls and OTM puts at the same delta. Measures directional skew. Negative = puts more expensive = bearish expectations. This is the single best predictor of large moves that we're missing.
- **Variance Risk Premium** (Bollerslev, Tauchen & Zhou 2009): Difference between risk-neutral and physical variance. Predicts equity returns 1-6 months ahead. Computed from options prices vs realized vol.
- **SKEW Index** (CBOE): Tail risk pricing from S&P 500 options. High SKEW = market pricing in tail events. Available via yfinance (^SKEW).
- **Gamma Exposure (GEX)** (Squeezemetrics): Net gamma of market makers. When GEX is negative, market makers amplify moves. When positive, they dampen moves. Not directly available from free APIs but can be approximated from options chain data.

### Implementation plan

**Data sources (all free):**
- yfinance `Ticker.option_chain()` -- provides calls/puts with strikes, IVs, volumes, open interest for AAPL specifically
- yfinance `^VIX`, `^SKEW` -- market-wide risk gauges
- CBOE data (free delayed) -- put/call ratio

**New features to add to cache:**
- `put_call_ratio`: put volume / call volume from options chain
- `risk_reversal_25d`: 25-delta call IV minus 25-delta put IV
- `iv_skew`: OTM put IV / ATM IV (measures tail fear)
- `vix_term_structure`: VIX 1M / VIX 3M ratio (>1 = backwardation = stress)
- `skew_index`: CBOE SKEW index value
- `variance_risk_premium`: IV^2 - RV^2 (annualized)

**Where to wire:**
- New module: `operator1/features/options_signals.py`
- Called in main.py Step 4a.7 (after benchmark returns, before estimation)
- Features added to `_extra_vars` for temporal models
- Forecasting models can use these as leading indicators

---

## Gap 2: Geographic Supply Chain Concentration Risk

**The problem:** We compute product segment HHI (revenue concentration) but not geographic manufacturing concentration. AAPL's 20% China exposure was the key risk factor for the January 2025 tariff-driven selloff.

### What experts do

**Popular methods:**
- **Country Revenue Exposure** (Bloomberg/FactSet): Revenue breakdown by geography. Available in 10-K filings under "Geographic Information" XBRL tag.
- **Supply Chain Mapping** (Facteus, Panjiva): Map supplier locations from shipping/customs data. Not available from free APIs.

**Less common but effective:**
- **XBRL Geographic Segment Extraction** (our existing segment extraction can do this): Most companies report geographic revenue in the same XBRL Operating Segments we already extract. We just need to detect geographic vs product segments and store both.
- **Trade Policy Risk Index** (Baker, Bloom & Davis 2016): News-based policy uncertainty index. Available from policyuncertainty.com (free CSV downloads). Specifically: Trade Policy Uncertainty component.
- **Tariff Exposure Score** (Amiti, Redding & Weinstein 2019): Estimate tariff impact from product categories (HS codes) x country exposure. Can be approximated from segment data + tariff schedule lookups.
- **GLEIF Subsidiary Geography** (already fetched): We fetch GLEIF corporate structure (parents/subsidiaries). Each subsidiary has a country code. We can compute geographic concentration from subsidiary locations as a proxy for supply chain exposure.

### Implementation plan

**Data sources:**
- XBRL geographic segments: already in `extract_segment_data()` -- just need to parse `GeographicAreasMember` alongside `OperatingSegmentsMember`
- GLEIF subsidiaries: already fetched in main.py Step 5e.1 -- just need to aggregate country distribution
- Economic Policy Uncertainty Index: free CSV from policyuncertainty.com

**New features:**
- `geo_hhi`: Geographic revenue concentration (HHI across countries)
- `china_revenue_pct`: Revenue from China/Greater China (critical for tariff risk)
- `supply_chain_geo_hhi`: Manufacturing concentration from GLEIF subsidiary countries
- `trade_policy_uncertainty`: Baker-Bloom-Davis TPU index (daily)
- `tariff_exposure_score`: Composite of geo_hhi * trade_policy_uncertainty * china_pct

**Where to wire:**
- Extend `operator1/features/product_metrics.py` to compute geographic HHI alongside product HHI
- New function `compute_supply_chain_geography()` using GLEIF subsidiary data
- Fetch TPU index in `macro_provider.py` as a new macro indicator

---

## Gap 3: Cross-Asset Sector Rotation Signals

**The problem:** We fetch SPY benchmark returns for beta computation but don't track sector ETF flows. The January 2025 tech-to-value rotation was visible from XLK vs XLE relative strength weeks before AAPL dropped.

### What experts do

**Popular methods:**
- **Sector Relative Strength** (Dorsey 1995): Compare sector ETF returns to SPY. When a sector underperforms for 2+ weeks, it signals rotation.
- **13F Institutional Flows** (SEC): Quarterly institutional holdings. 45-day lag makes it backward-looking for current quarter but useful for detecting accumulation/distribution trends.

**Less common but effective:**
- **Cross-Sector Momentum** (Moskowitz & Grinblatt 1999): Industry momentum predicts returns 6-12 months ahead. The top-performing sector from the prior 12 months tends to continue outperforming.
- **Intermarket Analysis** (Murphy 1991): Bond yields, commodities, and currencies lead equities. Rising 10-year yields + strong USD historically precede tech sector weakness. We can fetch UST 10Y yield from FRED (already have FRED API key).
- **ETF Fund Flows** (ICI/EPFR): Weekly net inflows/outflows by sector. Not available from free APIs, but we can approximate using ETF volume anomalies (unusually high volume in XLE relative to its 20-day average = institutional accumulation).
- **Sector Dispersion** (Pollet & Wilson 2010): When cross-sector return dispersion increases, it predicts higher subsequent market volatility. Computed as the standard deviation of sector ETF returns.

### Implementation plan

**Data sources (all via yfinance, free):**
- Sector ETFs: XLK (tech), XLE (energy), XLF (financials), XLV (healthcare), XLI (industrials), XLP (consumer staples), XLY (consumer discretionary), XLB (materials), XLU (utilities), XLRE (real estate), XLC (communications)
- 10Y Treasury: ^TNX (via yfinance)
- USD Index: DX-Y.NYB (via yfinance)
- Gold: GC=F (via yfinance)

**New features:**
- `sector_relative_strength`: Target sector ETF return / SPY return (21d rolling)
- `sector_rank_12m`: Rank of target's sector among 11 sectors by 252d return
- `sector_dispersion`: Cross-sector return standard deviation (21d rolling)
- `yield_curve_10y2y`: 10Y - 2Y Treasury spread (recession predictor)
- `usd_momentum_21d`: USD index 21d return (strong USD = tech headwind)
- `cross_asset_stress`: Composite of yield rise + USD strength + gold rally + sector dispersion

**Where to wire:**
- New module: `operator1/features/cross_asset_signals.py`
- Called in main.py Step 4a.8 (after sector leading indicators)
- All features added to `_extra_vars`

---

## Gap 4: Event Calendar and Event-Driven Uncertainty

**The problem:** The January 2025 AAPL drop was driven by known events (inauguration, tariff announcements). A political event calendar could have flagged elevated uncertainty in advance.

### What experts do

**Popular methods:**
- **Earnings Calendar** (already partially implemented via filing_calendar): We predict the next filing date. But we don't use it to adjust prediction confidence.
- **FOMC Calendar** (Federal Reserve): Known meeting dates 12 months in advance. Markets are volatile around FOMC meetings.

**Less common but effective:**
- **Event Study Methodology** (MacKinlay 1997): Measure abnormal returns around known events. Build a database of event types and their typical impact on the target stock. Use this to create event-adjusted predictions.
- **Anticipation Effect** (Rigobon & Sack 2004): Markets often react BEFORE the event (pricing in expectations). The anticipation premium can be estimated from options prices around the event date.
- **Uncertainty Channels** (Bloom 2009): Different events create different types of uncertainty: policy uncertainty (elections, tariffs), monetary uncertainty (FOMC), and idiosyncratic uncertainty (earnings, product launches). Each channel affects stocks differently based on sector and factor exposures.
- **Information Overload Discount** (Da, Engelberg & Gao 2011): When many events coincide, attention-limited investors react more slowly. Stocks with event clustering show delayed price adjustment.

### Implementation plan

**Data sources:**
- FOMC meeting dates: FRED API (`FOMC` calendar) or static list from federalreserve.gov
- Earnings dates: yfinance `Ticker.calendar` (next earnings date)
- Political events: static JSON maintained manually (inaugurations, budget deadlines, known policy announcements)
- Economic releases: FRED release calendar (`fred/releases/dates`)

**New features:**
- `days_to_next_event`: Minimum days until any known event (FOMC, earnings, political)
- `event_uncertainty_premium`: Widening factor for prediction intervals around events
- `fomc_proximity`: Days until next FOMC meeting (0-42)
- `earnings_proximity`: Days until next earnings (from filing_calendar)
- `event_density_30d`: Number of known events in the next 30 days

**Where to wire:**
- New module: `operator1/features/event_calendar.py`
- Called in main.py Step 4a.9
- Conformal prediction uses `event_uncertainty_premium` to widen intervals around events
- Prediction aggregator reduces point forecast confidence when `days_to_next_event < 5`

---

## Gap 5: DCF Model Calibration

**The problem:** The HF DCF produced an intrinsic value of $33 for AAPL (trading at $250). This is clearly broken -- the growth rate or WACC assumptions are wrong.

### What experts do

**Popular methods:**
- **Consensus Estimates** (Bloomberg, FactSet): Use analyst consensus for growth rates. Not available from free APIs.
- **Historical Growth Calibration**: Use the company's own 5-year revenue/earnings CAGR as the base growth rate instead of sampling from regime distributions.

**Less common but effective:**
- **Reverse DCF** (Mauboussin 2006): Instead of estimating intrinsic value, solve for the growth rate implied by the current market price. Then compare implied growth to historical growth. If implied growth < historical growth, the stock is undervalued.
- **Probabilistic DCF with Informed Priors** (Damodaran 2012): Use sector median WACC and growth rates as Bayesian priors, then update with company-specific data. Prevents extreme values from thin data.
- **Multi-Stage Growth Modeling**: High-growth companies need 3 stages: rapid growth (5yr), transition (5yr), terminal (stable). Single-stage terminal value dominates when growth is high.

### Implementation plan

**Fixes to existing DCF:**
- Use company's own 3-year revenue CAGR as the growth prior (not regime distribution means which can be near-zero in high_vol regimes)
- Cap WACC at sector median + 2% (prevent extreme discount rates)
- Use 3-stage model for companies with revenue growth > 10%
- Add reverse DCF to produce implied growth rate for sanity check
- Compare DCF output to current price -- if ratio > 5x or < 0.2x, flag as unreliable

---

## Gap 6: Ensemble Model Diversity

**The problem:** The genetic optimizer found Kalman=1.0, everything else=0.0. The other models (GARCH, VAR, tree, LSTM) aren't contributing useful signal for AAPL close price.

### What experts do

**Popular methods:**
- **Stacking** (Wolpert 1992): Train a meta-learner on the outputs of base models. The meta-learner learns when each model is reliable.
- **Forecast Combination** (Bates & Granger 1969): Equal-weight averaging often outperforms optimized weighting because it avoids overfitting to training-set model performance.

**Less common but effective:**
- **Regime-Conditional Ensembles** (already partially implemented via dual regime): Different models dominate in different regimes. Kalman dominates in trending markets, GARCH in volatile, mean reversion in range-bound.
- **Feature-Driven Model Selection** (Tsay 2017): Instead of fixed weights, select the model based on current data characteristics (trend strength, volatility level, autocorrelation). ADX > 25 -> Kalman. IV-RV spread > 0 -> GARCH. etc.
- **Online Convex Optimization** (Hazan 2016): The FixedShare algorithm we already use is from this family. The improvement is to use Adaptive FixedShare where the share parameter itself adapts to the rate of regime changes (higher share in volatile regimes).
- **Reject Option** (Chow 1970): Allow the ensemble to say "I don't know" when no model is confident. Replace the point forecast with a wider interval and reduce position sizing. Our conformal calibrator partially does this but doesn't feed back into the point forecast.

### Implementation plan

- Feature-driven model selection: add a model router that checks ADX, IV-RV spread, autocorrelation, and VIX to pick the dominant model per prediction step
- Adaptive FixedShare: modify `FixedShareForecaster` to increase the share parameter when `online_change_score` is high
- Equal-weight fallback: when GA converges to single-model dominance, force equal weights as a diversity mechanism
- Reject option: when conformal interval width > 2x the stock price, flag as "no opinion" and set forecast = last close (no directional bet)

---

## Execution Priority

| Priority | Gap | Impact | Effort | Free Data Available |
|----------|-----|--------|--------|---------------------|
| 1 | Options signals (Gap 1) | High -- directly predicts directional moves | Medium | Yes (yfinance options chain) |
| 2 | Cross-asset rotation (Gap 3) | High -- captures institutional flows | Low | Yes (yfinance ETFs) |
| 3 | Event calendar (Gap 4) | Medium -- widens intervals around known events | Low | Yes (FRED + static lists) |
| 4 | Geographic supply chain (Gap 2) | Medium -- tariff/geopolitical risk | Medium | Partial (XBRL + GLEIF + TPU CSV) |
| 5 | DCF calibration (Gap 5) | Medium -- fixes HF thesis accuracy | Low | N/A (code fix) |
| 6 | Ensemble diversity (Gap 6) | Low -- marginal improvement on well-calibrated ensemble | Medium | N/A (code changes) |

Gaps 1-3 can be implemented using only yfinance (already installed). Gap 4 uses FRED (already have API key) + static JSON. Gap 5 is a code fix to existing HF DCF. Gap 6 is algorithmic improvements to existing ensemble code.

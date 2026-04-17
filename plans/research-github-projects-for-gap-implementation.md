# GitHub Projects Research for Gap Implementation

Researched 2026-04-17. Projects organized by gap area with implementation relevance assessed.

---

## Gap 1: Options-Derived Signals

### Key Projects

**[Matteo-Ferrara/gex-tracker](https://github.com/Matteo-Ferrara/gex-tracker)** (188 stars, Python)
- Dealers' gamma exposure (GEX) tracker
- Computes net gamma from options chain data
- Shows how to fetch options chains, calculate dealer positioning, and identify gamma flip points
- **Relevance: HIGH** -- directly implements the gamma exposure computation we need. Can adapt their GEX calculation for our `options_signals.py` module.

**[aaguiar10/gflows](https://github.com/aaguiar10/gflows)** (99 stars, Python)
- Delta, gamma, vanna, and charm exposure visualization
- Uses plotly for interactive charts
- Supports both stocks and indexes
- **Relevance: HIGH** -- implements all four Greek exposures from options chains. Their vanna exposure calculation (sensitivity to vol changes) is particularly useful for predicting vol regime shifts.

**[vollib/py_vollib](https://github.com/vollib/py_vollib)** (392 stars, Python)
- Black-Scholes implied volatility calculation
- Supports American and European options
- Fast vectorized computation
- **Relevance: MEDIUM** -- we already compute IV via yfinance, but py_vollib could provide more accurate IV surface computation from raw options chain data (strike-by-strike IV rather than single ATM IV30).

**[goldmansachs/gs-quant](https://github.com/goldmansachs/gs-quant)** (10,104 stars, Jupyter Notebook)
- Goldman Sachs quantitative finance toolkit
- Includes derivatives pricing, risk management, portfolio construction
- Requires GS Marquee API access for full functionality, but the open-source code shows their methodology
- **Relevance: MEDIUM** -- their volatility surface fitting and risk premium decomposition code provides reference implementations even if we can't use their API.

### Implementation approach
- Use gex-tracker's approach: fetch full options chain via `yfinance.Ticker.option_chain()`, compute net gamma per strike, identify gamma flip point
- Use gflows' Greek exposure calculation: aggregate delta, gamma, vanna across all strikes and expirations
- Compute put/call ratio from options chain volume data (trivial once chain is fetched)
- Risk reversal = 25-delta call IV - 25-delta put IV (interpolate from chain)
- SKEW from `yfinance.Ticker("^SKEW").history()`

---

## Gap 2: Supply Chain Geographic Risk

### Key Projects

No high-quality open-source projects found for geographic supply chain risk scoring. This is typically done with proprietary data (Panjiva, Facteus, Bloomberg SPLC).

**Our approach:**
- Leverage existing XBRL geographic segment extraction (already in `extract_segment_data()` for 15 markets -- just need to parse `GeographicAreasMember` alongside `OperatingSegmentsMember`)
- Use GLEIF subsidiary country distribution (already fetched)
- Fetch Trade Policy Uncertainty Index from policyuncertainty.com (free CSV)

---

## Gap 3: Cross-Asset Rotation Signals

### Key Projects

**[stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading)** (17,079 stars, Jupyter Notebook)
- Code for "Machine Learning for Algorithmic Trading" (2nd edition)
- Chapter 4: sector rotation strategies using ETF data
- Chapter 7: cross-sectional momentum and factor models
- Chapter 24: intermarket relationships
- **Relevance: HIGH** -- reference implementation for sector rotation, cross-asset momentum, and factor model construction. Their ETF data pipeline is directly applicable.

**[Pratikbhanuse/relative-rotation-graph](https://github.com/Pratikbhanuse/relative-rotation-graph)** (1 star, Python)
- Implements Relative Rotation Graphs (RRG) for US sector ETFs vs S&P 500
- Computes relative strength (RS) and relative momentum (RS-Momentum)
- Classifies sectors into 4 quadrants: Leading, Weakening, Lagging, Improving
- **Relevance: HIGH** -- small but directly implements the exact sector rotation visualization and classification we need. Their RS/RS-Momentum computation is the standard Dorsey method.

**[rsheftel/pandas_market_calendars](https://github.com/rsheftel/pandas_market_calendars)** (962 stars, Python)
- Exchange calendars with holiday/early-close handling for 30+ exchanges
- Already installed (dependency of `exchange-calendars` which is a dependency of `pandas-market-calendars`)
- **Relevance: MEDIUM** -- useful for event calendar (Gap 4) to know trading days around events.

### Implementation approach
- Fetch 11 sector ETFs + ^TNX + DX-Y.NYB via `yfinance.download()` (batch fetch, single API call)
- Compute 21d/63d relative strength vs SPY
- Compute sector dispersion (cross-sectional std of sector returns)
- Yield curve from ^TNX - ^IRX (10Y - 3M) or FRED T10Y2Y series
- USD momentum from DX-Y.NYB

---

## Gap 4: Event Calendar

### Key Projects

**[pavelkrusek/market-calendar-tool](https://github.com/pavelkrusek/market-calendar-tool)** (12 stars, Python)
- Scrapes economic calendar from multiple financial websites
- Returns data as pandas DataFrames
- Covers FOMC, NFP, CPI, GDP releases
- **Relevance: HIGH** -- directly implements economic event scraping. We can use their approach or integrate their package.

**[rsheftel/pandas_market_calendars](https://github.com/rsheftel/pandas_market_calendars)** (962 stars, Python)
- Already available in our environment (installed as dependency)
- Provides market open/close times, holidays, early closes
- **Relevance: MEDIUM** -- useful for determining trading days around events.

### Implementation approach
- FOMC dates: static list from federalreserve.gov (published 12 months ahead) + FRED API
- Earnings dates: `yfinance.Ticker.calendar` (provides next earnings date)
- Political events: manually maintained JSON (inaugurations, budget deadlines, known trade policy dates)
- CPI/NFP/GDP: from market-calendar-tool or FRED release schedule

---

## Gap 5: DCF Calibration

### Key Projects

**[virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)** (55,901 stars, Python)
- The most popular AI hedge fund project on GitHub
- Implements multi-agent analysis with different "analyst personas"
- Their DCF agent uses LLM to estimate growth rates and WACC
- Their valuation module compares multiple approaches (DCF, relative, asset-based)
- **Relevance: HIGH** -- their multi-method valuation approach (not just DCF) and LLM-assisted growth estimation are directly applicable to fixing our DCF. Their code shows how to sanity-check DCF output against relative valuation.

**[AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)** (14,788 stars, Jupyter Notebook)
- Deep reinforcement learning for trading
- Includes environment with realistic market dynamics
- Their reward function design is relevant for our genetic optimizer
- **Relevance: LOW** for DCF specifically, but useful for ensemble optimization (Gap 6).

### Implementation approach
- Use company's own 3-year revenue CAGR as growth prior (not regime distributions)
- Add reverse DCF: solve for implied growth rate from current price
- Compare DCF output to PE-based and EV/EBITDA-based relative valuations
- Flag when DCF differs from relative valuation by >3x

---

## Gap 6: Ensemble Diversity

### Key Projects

**[unit8co/darts](https://github.com/unit8co/darts)** (9,332 stars, Python)
- Comprehensive time series library with 30+ models
- Built-in ensemble methods: NaiveEnsembleModel, RegressionEnsembleModel
- Their `RegressionEnsembleModel` trains a meta-learner on base model outputs -- exactly the stacking approach we need
- **Relevance: HIGH** -- their ensemble implementation is production-grade and shows best practices for model combination.

**[salesforce/Merlion](https://github.com/salesforce/Merlion)** (4,473 stars, Python)
- Salesforce's time series intelligence framework
- Built-in model selection via AutoML
- Ensemble with automatic model weighting
- Their `SequentialEnsemble` and `MoE_ForecasterEnsemble` (Mixture of Experts) are relevant
- **Relevance: HIGH** -- their MoE approach (different models for different data regimes) is exactly the feature-driven model selection we need.

**[nixtla/statsforecast](https://github.com/nixtla/statsforecast)** (4,760 stars, Python)
- Already installed and used (our ETS provider)
- Their `StatsForecast` class supports ensemble via `level` parameter for prediction intervals
- **Relevance: MEDIUM** -- we already use it, but could leverage their cross-validation utilities.

**[nixtla/mlforecast](https://github.com/nixtla/mlforecast)** (1,213 stars, Python)
- ML-based forecasting (LightGBM, XGBoost, sklearn)
- Supports conformal prediction intervals
- Their feature engineering (lags, rolling means, date features) is well-optimized
- **Relevance: MEDIUM** -- their conformal prediction approach could improve our PID calibrator.

---

## Summary: Projects to Study for Each Gap

| Gap | Primary Reference Projects | Key Takeaway |
|-----|---------------------------|--------------|
| 1. Options | gex-tracker (188), gflows (99), py_vollib (392) | Fetch full options chain, compute GEX + risk reversal + put/call ratio |
| 2. Supply Chain | (none found) | Use existing XBRL/GLEIF + TPU CSV |
| 3. Cross-Asset | ML4Trading (17K), relative-rotation-graph | Sector ETF relative strength + RRG quadrant classification |
| 4. Event Calendar | market-calendar-tool (12), pandas_market_calendars (962) | Scrape economic calendar + FOMC/earnings dates from yfinance |
| 5. DCF | ai-hedge-fund (55K) | Multi-method valuation, reverse DCF, LLM growth estimation |
| 6. Ensemble | darts (9.3K), Merlion (4.5K) | RegressionEnsemble (stacking) + MoE (regime-conditional selection) |

### No new pip packages needed
All implementations can use libraries we already have installed: yfinance, pandas, numpy, scipy, sklearn. The options chain data comes from `yfinance.Ticker.option_chain()` which is already available.

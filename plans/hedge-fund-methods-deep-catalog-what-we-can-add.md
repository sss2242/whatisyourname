# Hedge Fund Methods Deep Catalog

*Every technique worth considering for the HF pipeline, from textbook to obscure.*
*Organized by: What we already have, what we should add, and what is not feasible with free PIT data.*

---

## What We Already Have (50 existing + 18 HF modules)

Before cataloging new methods, here is what the combined system already covers:

**Factor models:** Accruals (Sloan 1996), earnings quality (Beneish M-Score), momentum (revenue acceleration), value (PE, PEG, FCF yield), quality (CROA/ROIC spread), size-adjusted (peer percentile ranking)

**Risk models:** Regime detection (HMM/GMM/PELT/BCP/ChangeFinder), copula tail dependence (Gaussian/Student-t/Clayton), graph risk (CoVaR, SRISK), contagion (MHHI, crowding), Monte Carlo (regime-switching 10K paths), conformal prediction (PID-calibrated, Mondrian-partitioned)

**Forecasting:** Kalman, GARCH, VAR, LSTM, Transformer, XGBoost/RF, AutoARIMA, Dynamic Factor Model, DTW analogs, cycle decomposition (CEEMDAN)

**Stress testing:** 3-scenario USS (orderly/muddle/catastrophic), leverage stress (3 scenarios), dividend burn

**Forensics:** Beneish M-Score, Sloan accruals, Modified Jones Model, Benford's Law, earnings/cashflow vol ratio, cash conversion efficiency

**Signal infrastructure:** IC measurement, signal decay classification, prediction log with realized IC, PEAD signal, SUE score, adaptive thresholds (peer-calibrated), position signal with Kelly sizing

---

## Category A: Well-Known Methods We Should Add

These are well-documented in academic literature, widely used by quant funds, and feasible with our data.

### A1. Fama-French Factor Decomposition

**What it is:** Decompose returns into market (beta), size (SMB), value (HML), profitability (RMW), investment (CMA), and momentum (MOM) factor exposures. The residual is "alpha" -- return not explained by known factors.

**Why it matters:** This is how every institutional investor attributes returns. Without it, you cannot tell whether a company's performance is genuine alpha or just factor exposure. A company returning 15% in a year where the value factor returned 12% has only 3% alpha.

**Formula:**
```
R - Rf = alpha + beta*(Rm-Rf) + s*SMB + h*HML + r*RMW + c*CMA + m*MOM + epsilon
```

**What we need:** Factor return series (available free from Ken French's data library). Rolling regression (60-month) against company returns.

**Output:** Per-factor beta exposure, alpha, R-squared (how much of returns factors explain), factor tilt labels (value/growth/momentum/quality)

**Feasibility:** HIGH -- we have daily returns, beta_252d already computed. Just need factor return data (free CSV from mba.tufts.edu/pages/faculty/ken.french/).

**Dependencies:** None beyond numpy/scipy (OLS regression)

---

### A2. Ornstein-Uhlenbeck Mean Reversion Speed

**What it is:** Fits the OU process `dX = theta*(mu - X)*dt + sigma*dW` to identify mean-reverting vs trending securities. The parameter theta is the speed of reversion.

**Why it matters:** Tells you whether the current price deviation from fair value will correct (mean-revert) and how fast. Half-life = `ln(2)/theta`. Short half-life (<21 days) = stat arb opportunity. Long half-life (>252 days) = structural shift, not mean-reverting.

**What we need:** Daily close prices (already in cache). Can also apply to fundamental ratios (PE mean-reversion, margin mean-reversion).

**Output:** theta (reversion speed), mu (long-run mean), sigma (volatility), half_life_days, hurst_exponent (already have this in adaptive_model_params)

**Feasibility:** HIGH -- pure math on existing data.

**How it integrates:** Feeds into position signal -- high theta + wide deviation = trade, low theta = avoid.

---

### A3. Volatility Risk Premium (VRP)

**What it is:** The systematic difference between implied volatility (what options cost) and realized volatility (what actually happened). On average, IV exceeds RV by 3-5 vol points in equities.

**Why it matters:** Positive VRP = market is overpaying for insurance = systematic selling opportunity. Negative VRP (realized > implied) = tail event occurring = defensive.

**What we need:** We already have realized volatility (`volatility_21d`). For implied volatility, we need options data -- which is not available from free PIT APIs.

**Feasibility:** LOW without options data. PARTIAL: can approximate using the VIX-RV spread for US companies (VIX is free), or use GARCH forecast vs realized as a proxy.

**Proxy approach:** `VRP_proxy = GARCH_forecast_vol - realized_vol_21d`. When GARCH predicts higher vol than what happened, the market is in a risk-premium regime.

---

### A4. Short Interest / Days to Cover

**What it is:** Fraction of shares outstanding that are sold short. Days to cover = short interest / average daily volume. High short interest signals either danger (shorts know something) or squeeze potential.

**Why it matters:** >20% short interest is a major warning. Rapid increase in short interest precedes downgrades/blowups. But extreme short interest (>40%) can also signal squeeze setups.

**What we need:** Short interest data -- NOT available from free PIT APIs. Some markets (US) publish bi-monthly. Most Tier 2 markets don't report it at all.

**Feasibility:** LOW for non-US markets. For US, SEC publishes semi-monthly short interest data (free). Could add as a US-only enrichment.

---

### A5. Earnings Revision Momentum

**What it is:** Rate of change in analyst consensus estimates over 30/60/90 days. Downward revisions precede stock price declines by 4-8 weeks.

**What we need:** Analyst consensus estimates -- NOT available from free PIT APIs. Requires FactSet, Refinitiv, or Bloomberg.

**Feasibility:** LOW directly. PARTIAL: we can compute a **self-consensus** from the company's own guidance trajectory (sequential EPS changes in filings serve as a proxy).

**Proxy approach:** `Revision_Proxy = (Latest_EPS_TTM - 90_days_ago_EPS_TTM) / abs(90_days_ago_EPS_TTM)`. This is what our `sue_score` and `pead_signal` partially capture.

---

### A6. Insider Transaction Signal

**What it is:** Net insider buying/selling aggregated over 3/6 months. Insiders buying = bullish signal (they know more about the company). Insiders selling = ambiguous (could be diversification, tax, or bearish).

**What we need:** Insider transaction data -- we ALREADY HAVE this from `get_insider_transactions()` across 16 markets. The `inst_insider_signal` column exists in the cache.

**Feasibility:** HIGH -- data already collected. Need to build a proper signal module around it.

**What to compute:**
- Net insider buy/sell ratio (weighted by transaction size)
- Insider buying cluster detection (multiple insiders buying in the same week = strong signal)
- Insider-to-price divergence (insiders buying while stock falling = contrarian bullish)

**Where it lives now:** Partially in `institutional_flow.py:compute_institutional_flow()`. Could be enhanced in the HF pipeline.

---

### A7. Piotroski F-Score

**What it is:** 9-binary-signal composite (0-9) from financial statements. Each signal is 1 (good) or 0 (bad). Score >= 7 = strong fundamentals. Score <= 2 = weak.

**9 signals:**
1. Positive net income
2. Positive OCF
3. Rising ROA (Y/Y)
4. OCF > net income (quality)
5. Decreasing leverage (Y/Y)
6. Increasing current ratio (Y/Y)
7. No new shares issued
8. Rising gross margin (Y/Y)
9. Rising asset turnover (Y/Y)

**Why it matters:** Simple, transparent, and one of the best-performing fundamental signals in academic backtests. Works especially well in emerging markets (where accounting quality varies widely -- exactly our Tier 2 coverage).

**What we need:** All 9 inputs are already in the cache/raw DFs.

**Feasibility:** VERY HIGH -- ~30 lines of code. Pure logic on existing data.

---

### A8. Altman Z-Score Variants (Z'', Z-Score for Emerging Markets)

**What it is:** The original Altman Z is for US manufacturing. Z'' (Z-double-prime) works for non-manufacturing and private companies. The Emerging Markets Score (EMS) adjusts coefficients for developing-market accounting standards.

**Why it matters:** Our system covers 25 markets. Using the US manufacturing Z-Score for a Japanese service company or Brazilian bank is wrong. The model coefficient should match the company type.

**What we need:** Already have Z-Score in `financial_health.py`. Need to add Z'' and EMS variants.

**Z'' formula:** `Z'' = 6.56*(WC/TA) + 3.26*(RE/TA) + 6.72*(EBIT/TA) + 1.05*(BV_Equity/TL)`
**EMS formula:** `EMS = Z'' + 3.25` (constant adjustment for emerging market baseline)

**Feasibility:** HIGH -- 20 lines of code, same inputs as existing Z-Score.

---

## Category B: Less Common but High-Value Methods

These are used by sophisticated quant funds but rarely appear in mainstream literature.

### B1. Earnings Torpedo Detection

**What it is:** Detects companies where the stock price embeds optimistic expectations that cannot be met. When expectations are sky-high and any miss triggers a disproportionate crash (30-50% in a day).

**Detection signals:**
- PE > 2x sector median AND revenue deceleration detected
- Sequential beats narrowing (beating by smaller amounts each quarter)
- Buyback-inflated EPS (EPS growing but revenue flat = financial engineering)
- Management lowering guidance while analysts raising estimates (divergence)

**Why it matters:** One of the most profitable short signals. Companies trading at 40x earnings with decelerating revenue are accidents waiting to happen.

**Feasibility:** MEDIUM -- most inputs available. Missing analyst estimates but can proxy with sequential EPS trajectory.

---

### B2. Accrual Anomaly Factor (Long-Short Signal)

**What it is:** Not just detecting manipulation (Beneish) -- but using accruals as a return predictor. Long low-accruals, short high-accruals. The Sloan (1996) signal has been persistent for 30 years.

**Enhancement beyond what we have:** Our `accruals_signal` is computed but we don't generate a cross-sectional rank or a long/short classification from it. The signal needs to be ranked against peers.

**What to add:** `accruals_percentile_rank` from peer_ranking integration. If target is in bottom decile of accruals (low = quality), that is a buy signal. Top decile = sell.

**Feasibility:** HIGH -- 10 lines using existing `accruals_signal` + `peer_ranking`.

---

### B3. Capital Cycle Analysis (Industry Supply/Demand)

**What it is:** Industries cycle between under-investment (shortage, high margins, attractive returns) and over-investment (glut, margin compression, poor returns). The cycle takes 3-7 years. Companies entering a cycle at the wrong time (heavy capex at the peak) get destroyed.

**Detection signals:**
- Sector aggregate capex/revenue ratio trend (rising = late cycle)
- Competitor count increasing (from entity discovery)
- Margin compression across sector peers (from linked_aggregates)
- Rising inventory across sector (supply building up)

**Why it matters:** This is how Marathon Asset Management (one of the best-performing UK hedge funds) has generated returns for 30 years. Pure fundamental, no technical signals.

**Feasibility:** MEDIUM -- we have peer capex data from linked_caches, sector margin from linked_aggregates. Need to aggregate at sector level.

---

### B4. Management Quality Proxy (Insider Alignment Score)

**What it is:** Measures whether management incentives are aligned with shareholders. High insider ownership + no recent sells + decreasing SGA + increasing R&D = aligned management. Low ownership + aggressive selling + rising perks = agency problem.

**Components:**
- Insider ownership % (from holder data)
- Net insider buying/selling (from insider transactions)
- SGA trend vs revenue trend (empire-building signal)
- Share dilution (new shares issued)
- Debt-funded buybacks while FCF negative (our vanity.capital_misallocation)

**Feasibility:** HIGH -- most data already collected. Insider ownership from `get_holders()`, SGA from income_df, dilution from balance_df.

---

### B5. Implied Cost of Capital (ICC)

**What it is:** Instead of using CAPM (which requires market risk premium assumptions), reverse-engineer the discount rate that makes the current stock price equal to a simple earnings model. This is the "market-implied" required return.

**Formula (Ohlson-Juettner 2005):**
```
ICC = A + sqrt(A^2 + EPS_1/P0 * (g2 - (gamma-1)))
where A = 0.5 * ((gamma-1) + EPS_1/P0)
```
Where EPS_1 = next-year EPS estimate, g2 = short-term growth, gamma = long-term growth.

**Why it matters:** ICC > 12% = market perceives high risk = potential deep value OR genuine distress. ICC < 6% = market prices very low risk = potential complacency.

**Feasibility:** MEDIUM -- need forward EPS estimate (can proxy with current EPS * growth rate from our momentum module).

---

### B6. Forensic Cash Flow Analysis (Mulford & Comiskey)

**What it is:** Goes beyond our current OCF/NI ratio to detect specific cash flow manipulation techniques:

1. **Supplier financing games:** Extending payables to boost OCF (payables days increasing while revenue flat)
2. **Receivables factoring:** Selling receivables to third parties to convert to cash (sudden drop in receivables without revenue decline)
3. **CapEx reclassification:** Moving maintenance capex to "investing" to inflate OCF
4. **Working capital smoothing:** Building/releasing WC reserves to smooth OCF

**Detection:** All four are detectable from the raw statement DFs we already have.

**Feasibility:** HIGH -- pure ratio analysis on existing data. ~100 lines of code.

---

## Category C: Obscure/Practitioner Methods (The "Secret Sauce")

These are methods used by elite quants that rarely appear in any publication.

### C1. Cross-Asset Macro Regime Indicator

**What it is:** Instead of running HMM on just equity returns (which is noisy), use cross-asset signals: yield curve slope, credit spreads, commodity prices, FX volatility. When ALL of these flash danger simultaneously, the probability of equity crash is 5x higher than any single indicator.

**Why it matters:** This is reportedly how Bridgewater's "All Weather" strategy works -- they don't predict individual stocks, they predict macro regimes and position accordingly.

**What we need:** We already have macro_data with gdp, inflation, interest_rate, unemployment, currency. We need credit spreads (partially available from FRED) and commodity prices (not currently fetched).

**Feasibility:** MEDIUM -- could add credit spread from FRED (free), use our existing macro quadrant as a starting point, and build a multi-signal regime indicator.

---

### C2. Order Flow Imbalance (OFI) -- Microstructure Signal

**What it is:** Measures the imbalance between buy-initiated and sell-initiated volume at the best bid/ask. Persistent buy imbalance predicts short-term price increases.

**Why it matters:** This is the primary signal for most HFT firms and short-horizon systematic funds. The academic evidence (Cont, Kukanov, Stoikov 2014) shows OFI predicts 1-5 minute returns with IC > 0.10.

**What we need:** Level-2 order book data or tick-by-tick trade data. NOT available from free APIs.

**Feasibility:** VERY LOW -- data not accessible. Skip.

---

### C3. Earnings Call Tone Analysis (NLP on Transcripts)

**What it is:** Analyze the language in earnings call transcripts for: management confidence (hedging words = negative), forward guidance tone, Q&A evasiveness, Loughran-McDonald financial sentiment dictionary scores.

**Why it matters:** Academic research (Mayew & Venkatachalam 2012) shows that the vocal tone of CEO during earnings calls predicts future returns. Written transcript analysis is nearly as effective.

**What we need:** Earnings call transcripts -- NOT available from free PIT APIs. However, some companies post transcripts on their investor relations pages, and we could potentially extract them via our LLM filing extractor.

**Feasibility:** LOW for automation (transcript sourcing). MEDIUM if we limit to companies where transcripts are available in annual reports (which our filing discoverers can fetch).

---

### C4. Persistence of Returns (Markov Chains on Quarterly EPS)

**What it is:** Model the transition probabilities between earnings states: {beat, miss, inline} -> {beat, miss, inline}. Companies that beat 4 consecutive quarters have a 72% probability of beating the 5th (momentum persistence). Companies that miss after 4 beats have only 38% of beating next (pattern break).

**Why it matters:** This is a refined version of our earnings surprise module. Instead of just P(beat/miss), model the transition dynamics explicitly.

**What we need:** Historical quarterly EPS sequence -- already in raw `income_df`.

**Feasibility:** HIGH -- ~50 lines of code. Build a 3x3 transition matrix from EPS surprise history.

---

### C5. Implied Volatility Term Structure (via GARCH Forward Curve)

**What it is:** Without options data, construct a synthetic "implied vol term structure" by fitting GARCH at multiple horizons (5d, 21d, 63d, 252d). When short-term GARCH > long-term GARCH, the term structure is inverted -- historically precedes large moves.

**Why it matters:** Vol term structure inversion is one of the best-known options trading signals. We can approximate it without options data using our existing GARCH model.

**Feasibility:** HIGH -- we already have GARCH in `forecasting.py`. Just need to fit at multiple horizons and compare.

---

### C6. Benford's Law on Revenue Segments (Not Just Total)

**What it is:** Our current Benford test runs on total revenue. A more powerful test runs Benford on revenue by segment/geography (if reported). When one segment's digits deviate while others conform, that specific segment is likely manipulated.

**Feasibility:** LOW -- segment data not available from standard PIT filings in structured form. Would need LLM extraction from 10-K footnotes.

---

### C7. Balance Sheet Velocity (Delta Analysis)

**What it is:** Instead of looking at point-in-time ratios, look at the RATE OF CHANGE of balance sheet items across the most recent 4 filings. Rapidly accelerating receivables + flat revenue = revenue quality issue. Rapidly growing goodwill = M&A binge. Rapidly declining cash + rising debt = refinancing risk.

**Why it matters:** This is what distressed debt funds focus on -- they don't care about the current ratio, they care about whether it's getting worse FAST.

**Feasibility:** VERY HIGH -- pure delta/acceleration analysis on existing raw DFs. ~80 lines.

---

## Category D: Methods That Need External Data (Not Currently Feasible)

These would require data sources we don't have access to:

| Method | Data Needed | Source | Cost |
|--------|------------|--------|------|
| Full factor model (Fama-French 6-factor) | Factor return series | Ken French library | Free (CSV) |
| Short interest / days to cover | Bi-monthly short interest | SEC (US only) | Free |
| Analyst consensus estimates | Sell-side EPS estimates | FactSet/Refinitiv | Paid |
| Options implied volatility | Options chain data | CBOE/IEX | Paid |
| Credit default swap spreads | CDS market data | Bloomberg | Paid |
| Order flow / Level-2 data | Tick-by-tick trades | Exchange feeds | Paid |
| Satellite imagery | Foot traffic, parking lots | Orbital Insight | Paid |
| Credit card transaction data | Consumer spending | Second Measure | Paid |
| Web traffic / app downloads | Digital engagement | SimilarWeb | Paid |
| Job posting data | Hiring momentum | Indeed/LinkedIn | Paid |

Of these, only **Fama-French factor returns** and **SEC short interest** are free and worth adding.

---

## Recommended Additions: Priority-Ranked

Based on impact, feasibility, and complementarity with what already exists:

### Priority 1 (High impact, high feasibility, no new data needed)

| # | Method | Lines | Input | Integrates With |
|---|--------|-------|-------|----------------|
| 1 | **Piotroski F-Score** | ~30 | Raw DFs | HF Tier 1 (earnings quality) |
| 2 | **Balance Sheet Velocity** | ~80 | Raw DFs | HF Tier 3 (balance sheet risk) |
| 3 | **Earnings Persistence Transitions** | ~50 | Raw DFs | HF Tier 4 (surprise probability) |
| 4 | **Forensic Cash Flow (Mulford)** | ~100 | Raw DFs | HF Tier 1 (FCF quality) |
| 5 | **OU Mean Reversion Speed** | ~60 | Cache (close) | Position signal (entry timing) |
| 6 | **Accruals Percentile Rank** | ~10 | Existing signals + peers | HF Tier 1 (accruals forensics) |
| 7 | **Altman Z'' for non-US** | ~20 | Existing Z-Score inputs | Financial health (all 25 markets) |

### Priority 2 (Medium impact, requires modest new computation)

| # | Method | Lines | Input | Integrates With |
|---|--------|-------|-------|----------------|
| 8 | **GARCH Vol Term Structure** | ~60 | Cache (return_1d) | Risk assessment, position sizing |
| 9 | **Insider Alignment Score** | ~80 | Holder + insider data | HF Tier 4 (management quality) |
| 10 | **Capital Cycle Position** | ~100 | Linked caches (sector) | HF Tier 4 (inflection) |
| 11 | **Earnings Torpedo Detection** | ~80 | Cache + raw DFs | HF Tier 4 (risk flag) |
| 12 | **VRP Proxy via GARCH** | ~40 | GARCH forecast + realized | Risk premium indicator |

### Priority 3 (High impact but requires new free data source)

| # | Method | Lines | Data Needed | Source |
|---|--------|-------|-------------|--------|
| 13 | **Fama-French 5+1 Factor Decomposition** | ~150 | Factor return CSVs | Ken French library (free) |
| 14 | **Implied Cost of Capital** | ~60 | Forward EPS proxy | Computed from existing |
| 15 | **Cross-Asset Regime Indicator** | ~100 | Credit spread | FRED (free) |

### Not Recommended (insufficient data or marginal value)

| Method | Why Not |
|--------|---------|
| Order flow / microstructure | No data (paid feeds only) |
| Full options analysis | No data |
| Satellite / alt data | No data |
| Earnings call NLP | Transcript sourcing unreliable |
| Segment-level Benford | Segment data not in structured filings |

---

## How These Integrate with Multi-Frequency

Each recommended addition has a natural frequency:

| Method | Primary Freq | Why |
|--------|-------------|-----|
| Piotroski F-Score | Q | 9 binary signals update at filing frequency |
| Balance Sheet Velocity | Q | Delta analysis on quarterly snapshots |
| Earnings Persistence | Q | Transition matrix from quarterly EPS |
| Forensic Cash Flow | Q | Payables/receivables from quarterly BS |
| OU Mean Reversion | D/W | Fitted on daily closes, used for weekly timing |
| GARCH Vol Term Structure | D | Multi-horizon GARCH on daily returns |
| Capital Cycle | A/Q | Sector capex cycles are multi-year |
| Fama-French Factors | M | Monthly factor returns from Ken French |
| Cross-Asset Regime | M | Monthly macro + credit data from FRED |

The HF multi-frequency fusion module routes each method to its natural frequency for computation, then fuses with the daily-frequency existing signals for the final scorecard.

---

## Implementation Estimate

| Priority | Methods | Total Lines | Dependencies |
|----------|---------|-------------|-------------|
| P1 (7 methods) | Piotroski, BS Velocity, Earnings Persistence, Forensic CF, OU, Accruals Rank, Altman Z'' | ~350 | None (pure math on existing data) |
| P2 (5 methods) | GARCH Vol TS, Insider Alignment, Capital Cycle, Torpedo, VRP Proxy | ~360 | None new |
| P3 (3 methods) | Fama-French, ICC, Cross-Asset | ~310 | Ken French CSV download, FRED credit spread |
| **Total** | **15 methods** | **~1,020** | |

All P1 methods can be implemented as additional computations inside the existing HF engine.py or as new functions in the HF tier modules. No new packages needed.

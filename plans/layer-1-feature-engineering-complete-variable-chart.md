# Layer 1: Feature Engineering -- Complete Variable Chart (v2)

Every variable produced by the 18 Layer 1 modules (14 from analysis-models-map + 4 additional feature modules found in source code), with formula, input dependencies, tier classification, and downstream consumers.

**Note:** The analysis-models-map listed 14 modules but there are actually 18 modules in `operator1/features/` that inject cache columns. The 4 unlisted modules are: Institutional Flow, Cross-Asset Signals, Options Signals, and Event Calendar.

---

## 1.1 Derived Variables

**File:** `operator1/features/derived_variables.py` (1,042 lines)
**Pipeline step:** Step 5
**Entry point:** `compute_derived_variables(df)` -- runs 17 computation stages sequentially

### Stage 1: Returns and Risk (`_compute_returns_and_risk`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 1 | `return_1d` | `close.pct_change()` | `close` | Daily return | T3 Stability | Survival mode, regime detector, forecasting, Monte Carlo, all temporal models |
| 2 | `log_return_1d` | `ln(close / close.shift(1))` | `close` | Log return | T3 | Forecasting (GARCH), copula |
| 3 | `volatility_21d` | `return_1d.rolling(21).std()` | `return_1d` | Rolling std | T3 | Regime detector (HMM input), survival timeline, beta, Merton DD |
| 4 | `volatility_ewma_21d` | `return_1d.ewm(span=21).std()` | `return_1d` | EWMA std | T3 | Regime-aware model switching (faster adaptation) |
| 5 | `drawdown_252d` | `(close - close.rolling(252).max()) / close.rolling(252).max()` | `close` | Max drawdown | T3 | Survival mode trigger (threshold: -0.40), Monte Carlo |

### Stage 2: Solvency / Leverage (`_compute_solvency`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 6 | `total_debt_asof` | `short_term_debt.fillna(0) + long_term_debt.fillna(0)` | `short_term_debt`, `long_term_debt` | Balance sheet | T2 Solvency | Net debt, debt-to-equity, EV, MC |
| 7 | `debt_to_equity_signed` | `total_debt_asof / total_equity` | `total_debt_asof`, `total_equity` | Ratio | T2 | Informational |
| 8 | `debt_to_equity_abs` | `total_debt_asof / abs(total_equity)` | `total_debt_asof`, `total_equity` | Ratio | T2 | Survival trigger (>3.0), hierarchy, MC |
| 9 | `net_debt` | `total_debt_asof - cash_and_equivalents.fillna(0)` | `total_debt_asof`, `cash_and_equivalents` | Balance sheet | T2 | Financial health, EV |
| 10 | `net_debt_to_ebitda` | `net_debt / ebitda` | `net_debt`, `ebitda` | Ratio | T2 | Financial health T2 |

### Stage 3: Liquidity (`_compute_liquidity`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 11 | `current_ratio` | `current_assets / current_liabilities` | `current_assets`, `current_liabilities` | Ratio | T1 Liquidity | Survival trigger (<1.0), FH T1, MC, particle filter |
| 12 | `quick_ratio` | `(cash + receivables) / current_liabilities` | `cash_and_equivalents`, `receivables`, `current_liabilities` | Ratio | T1 | Financial health T1 |
| 13 | `cash_ratio` | `cash_and_equivalents / current_liabilities` | `cash_and_equivalents`, `current_liabilities` | Ratio | T1 | Financial health T1, particle filter |

### Stage 4: Interest Coverage (`_compute_interest_coverage`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 14 | `interest_coverage` | `ebit / interest_expense` | `ebit`/`operating_income`, `interest_expense` | Ratio | T2 | Financial health T2, HF advanced methods |

### Stage 5: Cash Reality (`_compute_cash_reality`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 15 | `free_cash_flow` | `operating_cash_flow - abs(capex)` | `operating_cash_flow`, `capex` | Cash flow | T1 | FCF yield, FH T1, HF FCF quality, MC |
| 16 | `free_cash_flow_ttm_asof` | `_rolling_4q_ttm(free_cash_flow)` | `free_cash_flow` | TTM aggregate | T1 | FH runway, particle filter, HF dividend burn |
| 17 | `fcf_yield` | `free_cash_flow / market_cap` | `free_cash_flow`, `market_cap` | Ratio | T1 | Survival trigger (<0.0), FH T1, MC |

### Stage 6: Profitability (`_compute_profitability`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 18 | `gross_margin` | `gross_profit / revenue` | `gross_profit`, `revenue` | Ratio | T4 Profitability | FH T4, peer ranking, linked agg, HF |
| 19 | `operating_margin` | `ebit / revenue` | `ebit`/`operating_income`, `revenue` | Ratio | T4 | FH T4 |
| 20 | `net_margin` | `net_income / revenue` | `net_income`, `revenue` | Ratio | T4 | FH T4, HF momentum |
| 21 | `roe` | `net_income / total_equity` | `net_income`, `total_equity` | Ratio | T4 | FH T4 |
| 22 | `ebitda` | `ebit` or `operating_income` (proxy) | `ebit`, `operating_income` | Income | T4 | Net debt/EBITDA, EV/EBITDA, TTM, Altman Z |

### Stage 7: Return on Assets (`_compute_roa`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 23 | `roa` | `net_income / total_assets` | `net_income`, `total_assets` | Ratio | T4 | FH T4, HF Piotroski |

### Stage 8: Valuation (`_compute_valuation`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 24 | `pe_ratio_calc` | `close / (net_income / shares)` -- with synthetic PE fallback from `operating_income` | `close`, `net_income`/`operating_income`, `shares_outstanding` | Ratio | T5 Growth | FH T5, prediction aggregator PE anchor, peer ranking, HF valuation |
| 25 | `earnings_yield_calc` | `(net_income / shares) / close` | `net_income`, `shares_outstanding`, `close` | Ratio | T5 | FH T5, HF PEG |
| 26 | `ps_ratio_calc` | `market_cap / revenue` | `market_cap`, `revenue` | Ratio | T5 | FH T5 |
| 27 | `enterprise_value` | `market_cap + total_debt.fillna(0) - cash.fillna(0)` | `market_cap`, `total_debt_asof`, `cash_and_equivalents` | Valuation | T5 | EV/EBITDA |
| 28 | `ev_to_ebitda` | `enterprise_value / ebitda` | `enterprise_value`, `ebitda` | Ratio | T5 | FH T5, peer ranking, HF valuation |
| 29 | `pb_ratio` | `market_cap / total_equity` | `market_cap`, `total_equity` | Ratio | T5 | FH T5 |

### Stage 9: TTM and Growth (`_compute_ttm_and_growth`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 30 | `revenue_ttm_asof` | `_rolling_4q_ttm(revenue)` | `revenue` | TTM | T5 | Revenue growth YoY, HF |
| 31 | `net_income_ttm_asof` | `_rolling_4q_ttm(net_income)` | `net_income` | TTM | T5 | Earnings growth YoY, SUE |
| 32 | `ebitda_ttm_asof` | `_rolling_4q_ttm(ebitda)` | `ebitda` | TTM | T5 | Net debt/EBITDA, HF leverage |
| 33 | `revenue_growth_yoy` | `(rev_ttm - rev_ttm.shift(252)) / abs(rev_ttm.shift(252))` | `revenue_ttm_asof` | Growth | T5 | FH T5, HF growth quality |
| 34 | `earnings_growth_yoy` | `(ni_ttm - ni_ttm.shift(252)) / abs(ni_ttm.shift(252))` | `net_income_ttm_asof` | Growth | T5 | FH T5 |

### Stage 10: Volume Average (`_compute_volume_avg`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 35 | `volume_avg_21d` | `volume.rolling(21).mean()` | `volume` | Technical | T3 | FH T3, Amihud illiquidity |

### Stage 11: Per-Share Metrics (`_compute_per_share`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 36 | `eps_calc` | `net_income / shares_outstanding` | `net_income`, `shares_outstanding` | Per-share | T5 | PE ratio, SUE, HF earnings surprise |
| 37 | `book_value_per_share` | `total_equity / shares_outstanding` | `total_equity`, `shares_outstanding` | Per-share | T5 | PB ratio, HF Altman Z |
| 38 | `revenue_per_share` | `revenue / shares_outstanding` | `revenue`, `shares_outstanding` | Per-share | T5 | PS ratio |

### Stage 12: Technical Indicators (`_compute_technical_indicators`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 39 | `sma_50` | `close.rolling(50).mean()` | `close` | Technical | T3 | Report charts, signal IC |
| 40 | `sma_200` | `close.rolling(200).mean()` | `close` | Technical | T3 | Golden/death cross, signal IC |
| 41 | `rsi_14` | `100 - 100/(1 + avg_gain_14/avg_loss_14)` | `close` | Technical (0-100) | T3 | Signal IC |
| 42 | `macd` | `EMA(close, 12) - EMA(close, 26)` | `close` | Technical | T3 | MACD histogram, signal IC |
| 43 | `macd_signal` | `EMA(macd, 9)` | `macd` | Technical | T3 | MACD histogram |
| 44 | `macd_histogram` | `macd - macd_signal` | `macd`, `macd_signal` | Technical | T3 | Extra vars |
| 45 | `bollinger_upper` | `SMA(close, 20) + 2 * STD(close, 20)` | `close` | Technical | T3 | BB width |
| 46 | `bollinger_lower` | `SMA(close, 20) - 2 * STD(close, 20)` | `close` | Technical | T3 | BB width |
| 47 | `bb_width` | `(upper - lower) / SMA(close, 20)` | `bollinger_upper`, `bollinger_lower` | Normalized | T3 | Extra vars |
| 48 | `adx` | ADX(high, low, close, 14) via `ta` | `high`, `low`, `close` | Technical (0-100) | T3 | Extra vars (trend strength) |
| 49 | `adx_pos` | +DI directional indicator via `ta` | `high`, `low`, `close` | Technical | T3 | ADX interpretation |
| 50 | `adx_neg` | -DI directional indicator via `ta` | `high`, `low`, `close` | Technical | T3 | ADX interpretation |
| 51 | `obv` | On-Balance Volume cumulative via `ta` | `close`, `volume` | Technical | T3 | Volume-price divergence |

### Stage 13: Recovery Time (`_compute_recovery_time`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 52 | `recovery_time_avg` | Mean days from drawdown trough to prior peak | `close` | Scalar | T3 | Profile, report |
| 53 | `recovery_time_max` | Max recovery time across all episodes | `close` | Scalar | T3 | Profile, report |
| 54 | `n_recovery_episodes` | Count of completed drawdown-recovery cycles | `close` | Scalar | T3 | Profile |

### Stage 14: Beta (`_compute_beta`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 55 | `beta_252d` | `Cov(return_1d, benchmark_return_1d, 252) / Var(benchmark_return_1d, 252)` | `return_1d`, `benchmark_return_1d` | Risk | T3 | FH T3, report |

### Stage 15: Earnings Quality Signals (`_compute_earnings_quality_signals`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 56 | `accruals` | `(net_income - operating_cash_flow) / total_assets` (Sloan 1996) | `net_income`, `operating_cash_flow`, `total_assets` | Quality | T4 | HF accruals forensics |
| 57 | `accruals_signal` | `-accruals` (lower accruals = buy) | `accruals` | Alpha signal | T4 | Signal IC, extra vars |
| 58 | `eps_surprise_proxy` | `eps_calc - eps_calc.shift(252)` | `eps_calc`/`net_income_ttm_asof` | Signal | T4 | SUE score |
| 59 | `sue_score` | `eps_surprise / rolling_std(eps_surprise, 504)` (Foster/Olsen/Shevlin 1984) | `eps_surprise_proxy` | Standardized | T4 | PEAD signal, HF |
| 60 | `pead_signal` | `sue_score * exp(-0.693 * days_since_filing / 30)` (Bernard & Thomas 1989) | `sue_score`, `revenue` | Decay-weighted | T4 | Prediction aggregator PEAD drift |

### Stage 16: Realized Vol Decomposition (`_compute_realized_vol_decomposition`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 61 | `realized_vol_21d` | `sqrt(sum(return_1d^2, 21))` | `return_1d` | Volatility | T3 | IV-RV spread |
| 62 | `continuous_vol_21d` | `sqrt(bipower_variation)` (Barndorff-Nielsen & Shephard 2004) | `return_1d` | Continuous vol | T3 | Diffusion vs jump |
| 63 | `jump_vol_21d` | `sqrt(max(RV - BV, 0))` | `realized_vol_21d`, `continuous_vol_21d` | Jump vol | T3 | Shock detection |
| 64 | `jump_ratio_21d` | `jump_vol^2 / realized_vol^2` | `jump_vol_21d`, `realized_vol_21d` | Ratio (0-1) | T3 | Jump spike detection |
| 65 | `jump_spike_flag` | `1 if jump_ratio > P90(own history)` | `jump_ratio_21d` | Binary | T3 | Extra vars |

### Stage 17: Merton Distance-to-Default (`_compute_merton_distance_to_default`)

| # | Variable | Formula | Inputs | Type | Tier | Consumers |
|---|----------|---------|--------|------|------|-----------|
| 66 | `merton_dd` | `(log(V/D) + (-0.5*sigma_V^2)*T) / (sigma_V*sqrt(T))` (Merton 1974) | `close`, `volatility_21d`, `total_debt`/`total_debt_asof`, `market_cap`/`shares_outstanding` | Credit | T2 | HF Merton default, MC survival anchor |
| 67 | `merton_dd_change_21d` | `merton_dd - merton_dd.shift(21)` | `merton_dd` | Change | T2 | Extra vars |

**Companion flags for all 67 variables:** `is_missing_{var}` (67 flags) + `invalid_math_{var}` (~20 for ratio variables) = **~154 total columns**

---

## 1.2 Conflict Risk Assessment

**File:** `operator1/features/conflict_risk.py` (1,111 lines)
**Pipeline step:** Step 4a.3

| # | Variable | Source / Formula | Type | Consumers |
|---|----------|-----------------|------|-----------|
| 1 | `country_conflict_flag` | WB FCS OR UCDP events OR active war | Binary (0/1) | Survival trigger, hierarchy |
| 2 | `company_conflict_flag` | OFAC/EU sanctions entity match | Binary (0/1) | Profile |
| 3 | `conflict_intensity_score` | `0.40*event + 0.20*fatality + 0.25*flag + 0.15*news` | Continuous (0-1) | Survival trigger (>0.7), extra vars |
| 4 | `sanctions_flag` | OFAC/EU sanctions list (13 countries) | Binary (0/1) | Survival trigger |
| 5 | `fragile_state_flag` | World Bank FCS list (31 countries) | Binary (0/1) | Profile |
| 6 | `conflict_type` | armed_conflict / sanctions / political_instability / none | Categorical | Report |
| 7 | `supply_chain_risk_score` | Linked suppliers/logistics in conflict zones | Continuous (0-1) | Extra vars, supply chain stress |
| 8 | `revenue_exposure_score` | Linked customers in conflict regions | Continuous (0-1) | Extra vars |
| 9 | `competitive_advantage_score` | Linked competitors in conflict | Continuous (0-1) | Extra vars |

**9 columns**

---

## 1.3 Filing Calendar

**File:** `operator1/features/filing_calendar.py` (385 lines)
**Pipeline step:** Step 4c

### Result Object (not cache columns, passed as parameter)

| # | Field | Description | Consumers |
|---|-------|-------------|-----------|
| 1 | `detected_frequency` | quarterly / semiannual / annual / unknown | Adaptive windows, resampler |
| 2 | `expected_filings_2yr` | 8 / 4 / 2 | Coverage ratio |
| 3 | `actual_filings_2yr` | Count of distinct filing dates | Coverage ratio |
| 4 | `coverage_ratio` | actual / expected | Report, triage card |
| 5 | `latest_filing_age_days` | Days since last filing | Staleness |
| 6 | `is_stale` | age > stale_threshold | Triage card |
| 7 | `stale_threshold_days` | 90 / 200 / 400 by frequency | Staleness check |
| 8 | `gaps` | List of missing filing periods | Report |
| 9 | `next_expected_filing` | Extrapolated date + confidence | Report, HF earnings surprise |

### Cache Column

| # | Variable | Formula | Consumers |
|---|----------|---------|-----------|
| 10 | `filing_freshness` | 0-1 decay score from nearest filing | Extra vars |

**1 cache column + 9 result fields**

---

## 1.4 Linked Aggregates

**File:** `operator1/features/linked_aggregates.py` (359 lines)
**Pipeline step:** Step 5g

For each group G in {competitors, suppliers, customers, financial_institutions, sector_peers}:

| # | Pattern | Variables Aggregated | Consumers |
|---|---------|---------------------|-----------|
| 1 | `{G}_avg_{var}` | close, return_1d, revenue, net_income, total_debt, free_cash_flow, fcf_yield, gross_margin, pe_ratio_calc | Extra vars |
| 2 | `{G}_median_{var}` | volatility_21d | Extra vars |
| 3 | `rel_strength_vs_sector` | target return / sector avg return | Extra vars |
| 4 | `valuation_premium` | target PE / sector avg PE | Extra vars |
| 5 | `rel_volatility` | target vol / sector avg vol | Extra vars |

**~33 columns** (5 groups x 6 avg/median vars + 3 relative metrics)

---

## 1.5 Macro Alignment

**File:** `operator1/features/macro_alignment.py` (322 lines)
**Pipeline step:** Consumed via `macro_quadrant.py`

| # | Variable | Source | Consumers |
|---|----------|--------|-----------|
| 1 | `gdp_growth` | World Bank / FRED / regional | Macro quadrant, extra vars |
| 2 | `inflation_rate_yoy` | CPI from central banks | Macro quadrant, buying power |
| 3 | `inflation_rate_daily_equivalent` | `inflation_rate_yoy / 365` | Real return |
| 4 | `real_interest_rate` | Central bank rate - inflation | Macro quadrant, extra vars |
| 5 | `unemployment_rate` | Labor statistics | Macro quadrant, extra vars |
| 6 | `official_exchange_rate_lcu_per_usd` | Currency vs USD | Extra vars |
| 7 | `real_return_1d` | `return_1d - inflation_daily` | Ethical filter, extra vars |

**7 columns + 7 `is_missing_*` flags = 14 columns**

---

## 1.6 Macro Quadrant

**File:** `operator1/features/macro_quadrant.py` (248 lines)
**Pipeline step:** Step 4a

| # | Variable | Formula | Consumers |
|---|----------|---------|-----------|
| 1 | `macro_quadrant` | GDP trend x Inflation trend -> Overheating/Goldilocks/Stagflation/Recession | Regime weighting, profile |
| 2 | `macro_quadrant_stability` | Rolling 63-day same-quadrant fraction | Profile |

**2 columns**

---

## 1.7 News Sentiment

**File:** `operator1/features/news_sentiment.py` (563 lines)
**Pipeline step:** Step 5i

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `sentiment_score` | LLM / VADER / keyword scoring (-1 to +1) | Continuous | Extra vars, vanity, catalysts |
| 2 | `sentiment_count` | Number of articles scored per day | Integer | Profile |
| 3 | `sentiment_momentum_5d` | `sentiment_score.rolling(5).mean().diff()` | Continuous | Extra vars |
| 4 | `sentiment_volatility_21d` | `sentiment_score.rolling(21).std()` | Continuous | Extra vars |
| 5 | `is_missing_sentiment` | `sentiment_score.isna()` | Binary | Companion flag |
| 6 | `policy_risk_score` | Mean policy-risk label from LLM scoring | Continuous | Extra vars |
| 7 | `policy_risk_max` | Max policy-risk label from LLM scoring | Continuous | Extra vars |

**7 columns** (corrected from 3)

---

## 1.8 Peer Ranking

**File:** `operator1/features/peer_ranking.py` (270 lines)
**Pipeline step:** Step 5h

| # | Variable | Formula | Consumers |
|---|----------|---------|-----------|
| 1-10 | `peer_rank_{var}` for 10 vars | Percentile rank of target vs peers (0-100) | Extra vars, HF valuation |
| 11 | `peer_composite_rank` | Weighted average of all ranks | Profile, report |
| 12 | `peer_rank_label` | Top Quartile / Above Median / Below Median / Bottom Quartile | Report |

Variables ranked: close, return_1d, volatility_21d, revenue, net_income, total_debt, free_cash_flow, fcf_yield, gross_margin, pe_ratio_calc

**12 columns**

---

## 1.9 Private Company Proxies

**File:** `operator1/features/private_company_proxies.py` (451 lines)
**Pipeline step:** Step 5a (only when no OHLCV)

| # | Proxy Variable | Formula | Replaces | Consumers |
|---|----------------|---------|----------|-----------|
| 1 | `equity_value` | `total_equity` latest | `close` | All models |
| 2 | `equity_change_rate` | `total_equity.pct_change()` | `return_1d` | Regime, forecasting, MC |
| 3 | `financial_volatility` | `net_income.rolling(21).std() / revenue` | `volatility_21d` | Regime, survival |
| 4 | `revenue_momentum_5d` | `revenue.pct_change(5)` | `return_5d` | Forecasting |
| 5 | `equity_drawdown` | Max drawdown of equity_value | `drawdown_252d` | Survival |

**5 proxy columns + 5 `_proxy_source_*` flags = 10 columns**

---

## 1.10 Market Buying Power

**File:** `operator1/features/market_buying_power.py` (366 lines)
**Pipeline step:** Step 4a.4

| # | Variable | Formula | Consumers |
|---|----------|---------|-----------|
| 1 | `buying_power_index` | Composite of GDP/capita + inflation-adj revenue + sector demand | Extra vars, profile |
| 2 | `sector_demand_momentum` | PCE growth rate for sector | Extra vars, profile |
| 3 | `real_revenue_growth_ppp` | Revenue growth adjusted for PPP | Extra vars |
| 4 | `demand_risk_flag` | buying power declining AND sector momentum negative | Profile, report |

**4 columns** (corrected: added `real_revenue_growth_ppp`)

---

## 1.11 SIX Derived Proxies (Switzerland Only)

**File:** `operator1/features/six_derived_proxies.py` (3,848 lines)
**Pipeline step:** Step 4a.5 (only for `ch_six`)

22 canonical fields seeded + 4 diagnostic variables = **26 columns**

---

## 1.12 Product Catalysts

**File:** `operator1/features/product_catalysts.py` (338 lines)
**Pipeline step:** Step 5i.5

| # | Variable | Formula | Consumers |
|---|----------|---------|-----------|
| 1 | `catalyst_score` | `weighted_sum(rnd, earnings, diversification, news)` | Extra vars, DTW |
| 2 | `catalyst_type` | product_launch / regulatory / partnership / organic_growth / none | Profile |
| 3 | `rnd_acceleration` | R&D/revenue rate of change | Catalyst score |
| 4 | `earnings_momentum` | 2nd derivative of EPS | Catalyst score |

**4 columns**

---

## 1.13 Product Metrics

**File:** `operator1/features/product_metrics.py` (289 lines)
**Pipeline step:** Step 5i.6

| # | Variable | Formula | Consumers |
|---|----------|---------|-----------|
| 1 | `segment_hhi` | `sum(share_i^2)` across segments | MC concentration risk, extra vars |
| 2 | `cannibalization_rate` | Revenue shift between segments | Extra vars |
| 3 | `network_effect_score` | Revenue acceleration from network effects | Extra vars |
| 4 | `input_cost_pressure` | Cost growth vs revenue growth | Extra vars |
| 5 | `growth_runway_quarters` | Quarters before dominant segment matures | Extra vars |
| 6 | `maturity_concentration` | Revenue from low-growth segments | Extra vars |
| 7 | `estimated_market_share` | Inferred from revenue vs benchmarks | Extra vars |
| 8 | `dominant_segment_growth` | YoY growth of largest segment | Extra vars |
| 9 | `net_new_revenue_pct` | Revenue from new segments / total | Extra vars |

**9 columns**

---

## 1.14 OCR Pipeline

No cache variables (transparent PDF fallback). **0 columns**

---

## 1.15 Institutional Flow (PREVIOUSLY UNLISTED)

**File:** `operator1/features/institutional_flow.py` (372 lines)
**Pipeline step:** Step 5.inst
**Entry point:** `compute_institutional_flow(cache, insider_transactions)`

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `inst_flow_momentum` | EMA-smoothed QoQ change in `inst_ownership_pct` (Brunnermeier & Nagel 2004) | Continuous | Survival trigger (<-0.15), extra vars |
| 2 | `inst_flow_momentum_label` | distributing / reducing / stable / accumulating / strong_accumulating | Categorical | Profile |
| 3 | `inst_crowding_risk` | `concentration * ownership_level`, amplified by negative flow momentum | Continuous (0-1) | Survival trigger (>0.8 + illiquidity), extra vars |
| 4 | `inst_crowding_risk_label` | low / moderate / high / extreme | Categorical | Profile |
| 5 | `inst_smart_money_signal` | Divergence: top-5 concentration change vs total ownership change | Continuous (-1 to +1) | Extra vars |
| 6 | `inst_smart_money_label` | distribution / reducing / neutral / accumulating / strong | Categorical | Profile |
| 7 | `inst_insider_signal` | Net insider buying/selling from transaction data | Continuous (-1 to +1) | Extra vars |
| 8 | `inst_insider_label` | heavy_selling / selling / neutral / buying / heavy_buying | Categorical | Profile |
| 9 | `inst_amihud_illiquidity` | `rolling_mean(abs(return) / dollar_volume, 21)` (Amihud 2002) | Continuous | Survival trigger (crowding + illiquidity), ownership contagion |

**9 columns**

---

## 1.16 Cross-Asset Signals (PREVIOUSLY UNLISTED)

**File:** `operator1/features/cross_asset_signals.py` (~420 lines)
**Pipeline step:** Step 4a.8
**Entry point:** `compute_cross_asset_signals(cache, sector)`

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `sector_relative_strength` | Target sector ETF return / S&P 500 return (12-month) | Continuous | Extra vars |
| 2 | `sector_rank_12m` | Rank of target sector among 11 sector ETFs (1=best) | Integer (1-11) | Extra vars, profile |
| 3 | `sector_dispersion` | Cross-sector return dispersion (std of 11 ETF returns) | Continuous | Extra vars |
| 4 | `yield_curve_10y2y` | 10Y Treasury yield - 2Y Treasury yield | Continuous (bps) | Extra vars |
| 5 | `usd_momentum_21d` | 21-day momentum of US Dollar Index (DXY) | Continuous | Extra vars |
| 6 | `cross_asset_stress` | Composite stress index from treasury/gold/VIX/USD signals | Continuous (0-1) | Extra vars |

**6 columns**

---

## 1.17 Options Signals (PREVIOUSLY UNLISTED)

**File:** `operator1/features/options_signals.py` (~530 lines)
**Pipeline step:** Step 4a.7
**Entry point:** `compute_options_signals(cache, ticker, market_id)`

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `put_call_ratio` | Total put OI / Total call OI from options chain | Continuous | Extra vars, profile |
| 2 | `risk_reversal_25d` | 25-delta put IV - 25-delta call IV | Continuous (spread) | Extra vars |
| 3 | `iv_skew` | OTM put IV / ATM IV | Continuous | Extra vars |
| 4 | `vix_term_structure` | VIX / VIX3M ratio (contango/backwardation) | Continuous | Extra vars |
| 5 | `skew_index` | CBOE SKEW index level | Continuous | Extra vars |
| 6 | `variance_risk_premium` | IV30 - RV21 (implied minus realized vol) | Continuous | Extra vars |

**6 columns**

---

## 1.18 Event Calendar (PREVIOUSLY UNLISTED)

**File:** `operator1/features/event_calendar.py` (~340 lines)
**Pipeline step:** Step 4c.1
**Entry point:** `compute_event_calendar_features(cache, ticker, filing_calendar_result, reference_date)`

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `days_to_next_event` | Days until next known event (FOMC, earnings, political) | Integer | Extra vars, prediction aggregator |
| 2 | `event_uncertainty_premium` | Uncertainty multiplier based on event proximity | Continuous (0-1) | Conformal widening, PEAD drift |
| 3 | `fomc_proximity` | Days to nearest FOMC meeting (0-42) | Integer | Extra vars |
| 4 | `earnings_proximity` | Days to estimated next earnings filing | Integer | Extra vars |
| 5 | `event_density_30d` | Count of known events in next 30 days | Integer | Extra vars |

**5 columns**

---

## Layer 1 CORRECTED Grand Total

| Module | Primary Variables | Companion Flags | Total Columns |
|--------|-------------------|-----------------|---------------|
| 1.1 Derived Variables | 67 | ~87 | ~154 |
| 1.2 Conflict Risk | 9 | 0 | 9 |
| 1.3 Filing Calendar | 1 cache + 9 result | 0 | 1 |
| 1.4 Linked Aggregates | ~33 | 0 | ~33 |
| 1.5 Macro Alignment | 7 | 7 | 14 |
| 1.6 Macro Quadrant | 2 | 0 | 2 |
| 1.7 News Sentiment | **7** | 0 | **7** |
| 1.8 Peer Ranking | 12 | 0 | 12 |
| 1.9 Private Company Proxies | 5 | 5 | 10 |
| 1.10 Market Buying Power | **4** | 0 | **4** |
| 1.11 SIX Derived Proxies | 26 | 0 | 26 |
| 1.12 Product Catalysts | 4 | 0 | 4 |
| 1.13 Product Metrics | 9 | 0 | 9 |
| 1.14 OCR Pipeline | 0 | 0 | 0 |
| **1.15 Institutional Flow** | **9** | **0** | **9** |
| **1.16 Cross-Asset Signals** | **6** | **0** | **6** |
| **1.17 Options Signals** | **6** | **0** | **6** |
| **1.18 Event Calendar** | **5** | **0** | **5** |
| **Total** | **~212** | **~99** | **~311** |

All ~311 columns flow into the daily cache DataFrame consumed by Layer 2 (Analysis) and Layer 3 (Temporal Models).

# Layer 4: Hedge Fund Analysis -- Complete Variable Chart (v2)

Every variable produced by the 20 Layer 4 modules in `operator1/hedge_fund/`. This is a parallel analytical track that answers "Can I make money on this, when, and how much?" Primary data source: raw quarterly statement DataFrames (8-24 rows), NOT the 504-row daily cache. All outputs are result objects stored in `profile["hedge_fund"]` -- no cache columns.

**v2 update (2026-05-01):** Added expert-method enhancements from PR: 5-factor DuPont decomposition (Palepu, Healy & Peek 2019), regime-conditional momentum, market-implied growth + RIV valuation fields, Kelly criterion position sizing (Kelly 1956).

---

## 4.1 FCF Quality Scoring

**File:** `operator1/hedge_fund/fcf_quality.py`
**Pipeline step:** Step 6-HF-a (Tier 1: Earnings Quality)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `score` | `0.40*OCF_NI_Ratio_8Q + 0.30*(1-abs(Accruals)) + 0.20*FCF_Trend + 0.10*(1-CapEx_Vol)` | 0-100 | `hedge_fund.fcf_quality.score` |
| 2 | `degradation_flag` | True when score declining over 4Q | Boolean | `hedge_fund.fcf_quality.degradation_flag` |
| 3 | `ocf_ni_ratio` | `mean(OCF/NI)` over 8Q (NaN when NI < 0) | Ratio | `hedge_fund.fcf_quality.ocf_ni_ratio` |
| 4 | `narrative` | Text explanation of quality assessment | String | `hedge_fund.fcf_quality.narrative` |

---

## 4.2 Accruals Forensics

**File:** `operator1/hedge_fund/accruals_forensics.py`
**Pipeline step:** Step 6-HF-b (Tier 1)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `red_flag_score` | `0.30*Sloan + 0.25*Jones + 0.20*Divergence + 0.15*WC_Anomaly + 0.10*CCE` | 0-100 | `hedge_fund.accruals_forensic.red_flag_score` |
| 2 | `discretionary_accruals` | Jones Model residuals | Float | `hedge_fund.accruals_forensic.discretionary_accruals` |
| 3 | `cce_trend` | Cash Conversion Efficiency slope over 8Q | Float | `hedge_fund.accruals_forensic.cce_trend` |
| 4 | `sloan_accruals` | `(NI - OCF) / Total_Assets` | Float | Component |

---

## 4.3 Earnings Smoothing Detector

**File:** `operator1/hedge_fund/earnings_smoothing.py`
**Pipeline step:** Step 6-HF-c (Tier 1)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `smoothing_index` | `0.25*Beneish + 0.25*Vol_Ratio + 0.20*Benford + 0.15*Sequential + 0.15*Restatement` | 0-100 | `hedge_fund.smoothing.smoothing_index` |
| 2 | `benford_deviation` | Chi-squared test vs Benford's Law | Float | `hedge_fund.smoothing.benford_deviation` |
| 3 | `earnings_vol_ratio` | `std(NI) / std(OCF)` -- ratio < 0.5 = suspicious | Float | `hedge_fund.smoothing.earnings_vol_ratio` |
| 4 | `consecutive_beats` | Count of sequential EPS beats | Integer | Component |

---

## 4.4 Dividend Burn Risk

**Pipeline step:** Step 6-HF-d (Tier 2: Cash Flow)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `risk_score` | `0.40*(Divs/FCF) + 0.30*(Debt_Service/FCF) + 0.20*WC_Drain + 0.10*Earnings_Vol` | 0-100 | `hedge_fund.dividend_burn.risk_score` |
| 2 | `coverage_ratio` | `FCF / Dividends_Paid` | Float | `hedge_fund.dividend_burn.coverage_ratio` |
| 3 | `months_to_cut` | Estimated months until dividend unsustainable | Float | `hedge_fund.dividend_burn.months_to_cut` |

---

## 4.5 CROA vs ROIC Spread

**Pipeline step:** Step 6-HF-e (Tier 2)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `croa` | `OCF / Total_Assets` (cash return) | Float | `hedge_fund.return_spread.croa` |
| 2 | `roic` | `NOPAT / Invested_Capital` (accounting return) | Float | `hedge_fund.return_spread.roic` |
| 3 | `spread_bps` | `(CROA - ROIC) * 10000` | Integer (bps) | `hedge_fund.return_spread.spread_bps` |
| 4 | `quality_label` | high_quality / neutral / accrual_inflation | Categorical | `hedge_fund.return_spread.quality_label` |
| 5 | `dupont_tax_burden` | `NI / EBT` -- tax efficiency (Palepu, Healy & Peek 2019) | Float | `hedge_fund.return_spread.dupont_tax_burden` | **NEW** |
| 6 | `dupont_interest_burden` | `EBT / EBIT` -- debt cost efficiency | Float | `hedge_fund.return_spread.dupont_interest_burden` | **NEW** |
| 7 | `dupont_asset_turnover` | `Revenue / Total_Assets` -- asset utilization | Float | `hedge_fund.return_spread.dupont_asset_turnover` | **NEW** |
| 8 | `dupont_equity_multiplier` | `Total_Assets / Equity` -- leverage | Float | `hedge_fund.return_spread.dupont_equity_multiplier` | **NEW** |
| 9 | `dupont_quality_driver` | Which factor dominates ROE: margin / leverage / turnover | Categorical | `hedge_fund.return_spread.dupont_quality_driver` | **NEW** |

---

## 4.6 Operating Leverage

**Pipeline step:** Step 6-HF-f (Tier 2)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `dol` | `%delta_EBIT / %delta_Revenue` (Degree of Operating Leverage) | Float | `hedge_fund.operating_leverage.dol` |
| 2 | `dfl` | `%delta_EPS / %delta_EBIT` (Degree of Financial Leverage) | Float | `hedge_fund.operating_leverage.dfl` |
| 3 | `dtl` | `DOL * DFL` (Degree of Total Leverage) | Float | `hedge_fund.operating_leverage.dtl` |
| 4 | `earnings_sensitivity` | Classification: extreme/high/moderate/low from abs(DOL) | Categorical | `hedge_fund.operating_leverage.earnings_sensitivity` |

---

## 4.7 Off-Balance-Sheet Risk

**Pipeline step:** Step 6-HF-g (Tier 3: Balance Sheet)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `risk_score` | `0.35*(Goodwill/TA) + 0.25*(Intangibles/TA) + 0.20*SGA_Anomaly + 0.10*Keyword + 0.10*Policy_Change` | 0-100 | `hedge_fund.obs_risk.risk_score` |
| 2 | `goodwill_to_assets` | `Goodwill / Total_Assets` | Float | `hedge_fund.obs_risk.goodwill_to_assets` |
| 3 | `sga_anomaly` | SGA deviation from sector norm | Float | `hedge_fund.obs_risk.sga_anomaly` |

---

## 4.8 Asset Quality Deterioration

**Pipeline step:** Step 6-HF-h (Tier 3)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `deterioration_score` | Composite from DSO change + inventory days change | 0-100 | `hedge_fund.asset_quality.deterioration_score` |
| 2 | `dso` | `Receivables / (Revenue/90)` (Days Sales Outstanding) | Float (days) | `hedge_fund.asset_quality.dso` |
| 3 | `dso_change_pct` | QoQ change in DSO | Float (%) | `hedge_fund.asset_quality.dso_change_pct` |
| 4 | `inventory_days` | `Inventory / (COGS/90)` | Float (days) | `hedge_fund.asset_quality.inventory_days` |

---

## 4.9 Leverage Stress Scenario Engine

**Pipeline step:** Step 6-HF-i (Tier 3)

| # | Variable | Description | Profile Key |
|---|----------|-------------|-------------|
| 1 | `base_case` | Current trajectory: Debt/EBITDA, interest coverage, cash runway | `hedge_fund.leverage_stress.base_case` |
| 2 | `revenue_miss` | -15% revenue, -200bps margin: Debt/EBITDA, covenant status | `hedge_fund.leverage_stress.revenue_miss` |
| 3 | `systemic_crisis` | -25% revenue, -400bps margin, +200bps rate: full distress analysis | `hedge_fund.leverage_stress.systemic_crisis` |
| 4 | `refinancing_risk` | Boolean: any scenario triggers covenant breach (Debt/EBITDA > 5.5x) | `hedge_fund.leverage_stress.refinancing_risk` |

---

## 4.10 Momentum Composite

**Pipeline step:** Step 6-HF-j (Tier 4: Inflection)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `score` | `0.40*Revenue_Accel + 0.30*Margin_Slope + 0.20*FCF_Conv + 0.10*ROIC_Traj` | 0-100 | `hedge_fund.momentum.score` |
| 2 | `inflection_detected` | True when 2nd derivative of revenue/margins turns positive | Boolean | `hedge_fund.momentum.inflection_detected` |
| 3 | `revenue_acceleration` | 2nd derivative (change in growth rate) | Float | Component |
| 4 | `price_momentum_divergence` | Fundamental momentum vs 63-day price momentum | Float | Component |
| 5 | `regime_adjusted_momentum` | Momentum score with regime-conditional weights: revenue_accel 70% in survival, margin_slope 40% in normal | 0-100 | `hedge_fund.momentum.regime_adjusted_momentum` | **NEW** |
| 6 | `momentum_regime_bias` | Which component drives the score in the current regime | Categorical | `hedge_fund.momentum.momentum_regime_bias` | **NEW** |

---

## 4.11 Growth Quality

**Pipeline step:** Step 6-HF-k (Tier 4)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `score` | `0.50*Organic_Fraction + 0.25*Margin_Adj + 0.15*Incremental_ROIC + 0.10*Concentration` | 0-100 | `hedge_fund.growth_quality.score` |
| 2 | `organic_fraction` | Revenue growth excluding M&A (goodwill jump detection) | Float (0-1) | `hedge_fund.growth_quality.organic_fraction` |
| 3 | `incremental_roic` | Return on each dollar of incremental invested capital | Float | `hedge_fund.growth_quality.incremental_roic` |

---

## 4.12 Earnings Surprise Probability

**Pipeline step:** Step 6-HF-l (Tier 4)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `p_beat` | CDF of SUE distribution at +1.5 sigma | Float (0-1) | `hedge_fund.earnings_surprise.p_beat` |
| 2 | `p_miss` | CDF of SUE at -1.5 sigma | Float (0-1) | `hedge_fund.earnings_surprise.p_miss` |
| 3 | `p_inline` | `1 - p_beat - p_miss` | Float (0-1) | `hedge_fund.earnings_surprise.p_inline` |
| 4 | `days_to_next_filing` | From filing_calendar_result | Integer | `hedge_fund.earnings_surprise.days_to_next_filing` |

---

## 4.13 DCF Monte Carlo Valuation

**Pipeline step:** Step 6-HF-m (Tier 5: Valuation)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `intrinsic_p10` | 10th percentile intrinsic value from 10K MC paths | Float ($) | `hedge_fund.dcf.intrinsic_p10` |
| 2 | `intrinsic_p25` | 25th percentile | Float ($) | `hedge_fund.dcf.intrinsic_p25` |
| 3 | `intrinsic_p50` | Median intrinsic value | Float ($) | `hedge_fund.dcf.intrinsic_p50` |
| 4 | `intrinsic_p75` | 75th percentile | Float ($) | `hedge_fund.dcf.intrinsic_p75` |
| 5 | `intrinsic_p90` | 90th percentile | Float ($) | `hedge_fund.dcf.intrinsic_p90` |
| 6 | `upside_pct` | `(p50 - current_price) / current_price` | Float (%) | `hedge_fund.dcf.upside_pct` |
| 7 | `risk_reward_ratio` | `upside / downside` | Float | `hedge_fund.dcf.risk_reward_ratio` |
| 8 | `current_price` | Latest close from cache | Float ($) | `hedge_fund.dcf.current_price` |
| 9 | `growth_gap` | `implied_growth_rate - actual_growth_rate` -- positive = priced for more growth than delivered | Float (%) | `hedge_fund.dcf.growth_gap` | **NEW** |
| 10 | `priced_for_perfection_flag` | True when growth_gap > 2x actual growth | Boolean | `hedge_fund.dcf.priced_for_perfection_flag` | **NEW** |
| 11 | `riv_intrinsic` | Residual Income Valuation (Ohlson 1995): `BV + sum(RI_t/(1+r)^t)` | Float ($) | `hedge_fund.dcf.riv_intrinsic` | **NEW** |
| 12 | `riv_excess_return` | `(NI - r*BV) / BV` -- is the company creating value above cost of equity? | Float | `hedge_fund.dcf.riv_excess_return` | **NEW** |
| 13 | `riv_vs_dcf_divergence` | `riv_intrinsic - intrinsic_p50` -- large divergence = model uncertainty | Float ($) | `hedge_fund.dcf.riv_vs_dcf_divergence` | **NEW** |

---

## 4.14 Valuation-Quality Matrix

**Pipeline step:** Step 6-HF-n (Tier 5)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `quality_score` | Composite of HF Tiers 1-3 | 0-100 | `hedge_fund.valuation_quality.quality_score` |
| 2 | `valuation_percentile` | PE rank vs peer group | 0-100 | `hedge_fund.valuation_quality.valuation_percentile` |
| 3 | `quadrant` | undervalued / fair / overvalued / value_trap | Categorical | `hedge_fund.valuation_quality.quadrant` |

---

## 4.15 PEG Composite

**Pipeline step:** Step 6-HF-o (Tier 5)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `peg_adjusted` | `(PE / Growth%) / Quality_Multiplier` | Float | `hedge_fund.peg_composite.peg_adjusted` |
| 2 | `fcf_spread_bps` | `(FCF_Yield - Debt_Cost) * 10000` | Integer (bps) | `hedge_fund.peg_composite.fcf_spread_bps` |
| 3 | `cheap_flag` | True when PEG < 1.0 and FCF spread positive | Boolean | `hedge_fund.peg_composite.cheap_flag` |

---

## 4.16 Investment Thesis Scorecard

**Pipeline step:** Step 6-HF-q (Synthesis)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `investment_grade` | A+ / A / B+ / B / C / D / F from weighted composite | Categorical | `hedge_fund.scorecard.investment_grade` |
| 2 | `conviction` | 0-10 conviction score | Integer | `hedge_fund.scorecard.conviction` |
| 3 | `tier1_score` | Earnings Quality (25% weight): FCF quality + accruals + smoothing | 0-100 | `hedge_fund.scorecard.tier1_score` |
| 4 | `tier2_score` | Cash Flow (20%): dividend burn + return spread + leverage | 0-100 | `hedge_fund.scorecard.tier2_score` |
| 5 | `tier3_score` | Balance Sheet (20%): OBS risk + asset quality + leverage stress | 0-100 | `hedge_fund.scorecard.tier3_score` |
| 6 | `tier4_score` | Inflection (20%): momentum + growth quality + earnings surprise | 0-100 | `hedge_fund.scorecard.tier4_score` |
| 7 | `tier5_score` | Valuation (15%): DCF + val-quality + PEG | 0-100 | `hedge_fund.scorecard.tier5_score` |

---

## 4.17 Position Signal Engine

**Pipeline step:** Step 6-HF-r (Actionable Output)

| # | Variable | Formula | Type | Profile Key |
|---|----------|---------|------|-------------|
| 1 | `signal` | `alpha * quality_mult * survival_mult * decay_mult * conviction` | Float (-1 to +1) | `hedge_fund.position.signal` or `position_signal.signal` |
| 2 | `label` | strong_buy / buy / hold / sell / strong_sell | Categorical | `hedge_fund.position.label` |
| 3 | `conviction` | 0-10 from scorecard | Integer | `hedge_fund.position.conviction` |
| 4 | `entry_price` | 63-day support level | Float ($) | `hedge_fund.position.entry_price` |
| 5 | `stop_price` | Stop-loss level | Float ($) | `hedge_fund.position.stop_price` |
| 6 | `target_price` | 63-day resistance level | Float ($) | `hedge_fund.position.target_price` |
| 7 | `survival_sizing` | Normal=1.0x, Modified=0.5x, Company_Survival=0.0x, Extreme=-0.5x, Recovery=1.5x | Float | Component |
| 8 | `kelly_fraction` | Full Kelly optimal bet size: `(p*b - q) / b` where p=P(win from MC), b=avg_win/avg_loss (Kelly 1956) | Float (-1 to +1) | `hedge_fund.position.kelly_fraction` | **NEW** |
| 9 | `half_kelly_size` | Conservative half-Kelly: `kelly_fraction / 2`, capped at [-0.5, 0.5] | Float | `hedge_fund.position.half_kelly_size` | **NEW** |
| 10 | `kelly_edge` | `p*b - q` -- positive = there is a mathematical edge, negative = no edge | Float | `hedge_fund.position.kelly_edge` | **NEW** |

---

## 4.18 Orchestrator

**Pipeline step:** Step 6-HF (Entry Point)

Runs all 15 base metrics + scorecard + position signal in dependency order. Single entry point: `run_hedge_fund_analysis()`.

---

## 4.19 Advanced HF Methods

**File:** `operator1/hedge_fund/advanced_methods.py` (965 lines)
**Pipeline step:** Step 6-HF (called from engine.py via `run_advanced_methods()`)

**Note:** The plan document lists 15 aspirational methods; the actual `AdvancedMethodsResult` dataclass implements 19 fields across 3 phases (P1/P2/P3). Several plan-listed methods (Ohlson, Springate, Zmijewski, Laitinen, Beneish Extended, Revenue Quality, WC Efficiency, FCF Sustainability, Margin of Safety) are NOT implemented. Instead, the code contains market-data-driven methods like OU mean reversion, GARCH vol term structure, insider alignment, capital cycle, VRP proxy, and implied cost of capital.

### P1 Methods (Fundamental Quality)

| # | Variable | Method | Type | Profile Key |
|---|----------|--------|------|-------------|
| 1 | `piotroski_f_score` | 9-factor binary (Piotroski 2000) | Integer (0-9) | `hedge_fund.advanced_methods.piotroski_f_score` |
| 2 | `piotroski_n_available` | How many of 9 signals had data | Integer (0-9) | `hedge_fund.advanced_methods.piotroski_n_available` |
| 3 | `piotroski_label` | strong / moderate / weak | Categorical | `hedge_fund.advanced_methods.piotroski_label` |
| 4 | `balance_sheet_velocity` | Rate of change in key B/S items (assets, debt, equity) | Dict | `hedge_fund.advanced_methods.balance_sheet_velocity` |
| 5 | `earnings_persistence` | 3x3 Markov transition matrix from quarterly EPS surprises | Dict | `hedge_fund.advanced_methods.earnings_persistence` |
| 6 | `forensic_cashflow` | OCF decomposition quality with `risk_score` | Dict | `hedge_fund.advanced_methods.forensic_cashflow` |
| 7 | `ou_mean_reversion` | Ornstein-Uhlenbeck fit on close prices: `half_life_days`, `theta`, `mu` | Dict | `hedge_fund.advanced_methods.ou_mean_reversion` |
| 8 | `accruals_rank` | Percentile rank from peer_ranking variable_ranks | Float (0-100) | `hedge_fund.advanced_methods.accruals_rank` |
| 9 | `altman_z_double_prime` | Altman Z''' (emerging market variant, Altman 2014) | Float | `hedge_fund.advanced_methods.altman_z_double_prime` |
| 10 | `altman_z_dp_zone` | safe / grey / distress | Categorical | `hedge_fund.advanced_methods.altman_z_dp_zone` |

### P2 Methods (Market-Driven Signals)

| # | Variable | Method | Type | Profile Key |
|---|----------|--------|------|-------------|
| 11 | `garch_vol_term_structure` | Multi-horizon realized vol curve with `inverted` flag | Dict | `hedge_fund.advanced_methods.garch_vol_term_structure` |
| 12 | `insider_alignment` | Net insider buying/selling vs earnings trajectory alignment | Dict | `hedge_fund.advanced_methods.insider_alignment` |
| 13 | `capital_cycle` | CapEx cycle phase detection from I/S + C/F + linked peers | Dict | `hedge_fund.advanced_methods.capital_cycle` |
| 14 | `earnings_torpedo` | High-expectation miss risk: `torpedo_risk` (0-100), `flags` | Dict | `hedge_fund.advanced_methods.earnings_torpedo` |
| 15 | `vrp_proxy` | Volatility risk premium: GARCH forecast vs realized vol spread | Dict | `hedge_fund.advanced_methods.vrp_proxy` |

### P3 Methods (Cross-Asset / Macro)

| # | Variable | Method | Type | Profile Key |
|---|----------|--------|------|-------------|
| 16 | `fama_french_alpha` | Fama-French factor alpha (stub -- needs external factor data, always None) | Float or None | `hedge_fund.advanced_methods.fama_french_alpha` |
| 17 | `implied_cost_of_capital` | Reverse-engineered discount rate from earnings + price | Float | `hedge_fund.advanced_methods.implied_cost_of_capital` |
| 18 | `merton_default_probability` | Structural default model: `pd_1yr`, `distance_to_default` | Dict | `hedge_fund.advanced_methods.merton_default_probability` |
| 19 | `cross_asset_regime` | Multi-signal macro regime from GDP, inflation, rates, FX | Dict | `hedge_fund.advanced_methods.cross_asset_regime` |

---

## 4.20 Cross-Pipeline Insight Fusion

**File:** `operator1/hedge_fund/fusion.py` (653 lines)
**Pipeline step:** Step 6-HF (post-scorecard)

**8 fusion methods:**
1. Anomaly-Driven Priority Routing (override on critical signals like covenant breach, torpedo risk)
2. Cross-Frequency Disagreement Signals (meta-signal from MF agreement)
3. Regime-Conditioned Signal Weighting (per-regime IC table)
4. Meta-Ensemble Stacking (combine both HF and pipeline position signals)
5. Catalyst-Driven Temporal Weighting (event-proximity modulation)
6. Bottom-Up Early Warning Propagation (fast -> slow override)
7. Multi-Frequency IC Attribution (data-driven freq assignment)
8. Bayesian Belief Network (simplified probabilistic inference)

| # | Variable | Method | Type | Profile Key |
|---|----------|--------|------|-------------|
| 1 | `fused_signal` | Final combined signal from all 8 methods | Float (-1 to +1) | `hedge_fund.fusion.fused_signal` |
| 2 | `fused_label` | strong_buy / buy / hold / sell / strong_sell | Categorical | `hedge_fund.fusion.fused_label` |
| 3 | `fused_conviction` | Inter-method agreement-driven confidence | Float (0-1) | `hedge_fund.fusion.fused_conviction` |
| 4 | `anomaly_overrides` | List of triggered anomaly rules from _ANOMALY_RULES | List of Dict | `hedge_fund.fusion.anomaly_overrides` |
| 5 | `anomaly_frozen` | True if anomaly override freezes the signal | Boolean | `hedge_fund.fusion.anomaly_frozen` |
| 6 | `freq_disagreement_score` | Cross-frequency disagreement (0=full agreement, 1=max) | Float (0-1) | `hedge_fund.fusion.freq_disagreement_score` |
| 7 | `regime_adjusted_weights` | Per-regime IC-driven weight adjustments | Dict | `hedge_fund.fusion.regime_adjusted_weights` |
| 8 | `meta_ensemble_agreement` | HF vs pipeline signal agreement (-1 to +1) | Float | `hedge_fund.fusion.meta_ensemble_agreement` |
| 9 | `catalyst_proximity_days` | Days to next catalyst event | Integer or None | `hedge_fund.fusion.catalyst_proximity_days` |
| 10 | `catalyst_weight_regime` | pre_catalyst / catalyst_zone / post_catalyst / normal | Categorical | `hedge_fund.fusion.catalyst_weight_regime` |
| 11 | `early_warnings` | List of fast-frequency warning signals propagated upward | List of String | `hedge_fund.fusion.early_warnings` |
| 12 | `freq_ic_attribution` | Per-frequency information coefficient attribution | Dict | `hedge_fund.fusion.freq_ic_attribution` |
| 13 | `belief_network_posterior` | Bayesian posterior P(positive return) | Float (0-1) | `hedge_fund.fusion.belief_network_posterior` |
| 14 | `action` | Human-readable action sentence | String | `hedge_fund.fusion.action` |
| 15 | `primary_risk` | Top risk factor identified | String | `hedge_fund.fusion.primary_risk` |
| 16 | `next_catalyst` | Next expected event description | String | `hedge_fund.fusion.next_catalyst` |

---

## Layer 4 UPDATED Grand Total

| Module | Variables | Type | Status |
|--------|----------|------|--------|
| 4.1 FCF Quality | 4 | Result | |
| 4.2 Accruals Forensics | 4 | Result | |
| 4.3 Earnings Smoothing | 4 | Result | |
| 4.4 Dividend Burn | 3 | Result | |
| 4.5 CROA vs ROIC + DuPont | **9** (was 4) | Result | **ENHANCED** (+5 DuPont decomposition) |
| 4.6 Operating Leverage | 4 | Result | |
| 4.7 OBS Risk | 3 | Result | |
| 4.8 Asset Quality | 4 | Result | |
| 4.9 Leverage Stress | 4 | Result | |
| 4.10 Momentum | **6** (was 4) | Result | **ENHANCED** (+2 regime-conditional) |
| 4.11 Growth Quality | 3 | Result | |
| 4.12 Earnings Surprise | 4 | Result | |
| 4.13 DCF Valuation | **13** (was 8) | Result | **ENHANCED** (+5 growth gap, RIV) |
| 4.14 Valuation-Quality | 3 | Result | |
| 4.15 PEG Composite | 3 | Result | |
| 4.16 Scorecard | 7 | Result | |
| 4.17 Position Signal | **10** (was 7) | Result | **ENHANCED** (+3 Kelly criterion) |
| 4.18 Orchestrator | 0 (wrapper) | -- | |
| 4.19 Advanced Methods | 19 (P1: 10, P2: 5, P3: 4) | Result | |
| 4.20 Fusion | 16 | Result | |
| **Total** | **~122** | **All result objects** | |

**0 cache columns, ~122 result fields** (was ~107) stored in `profile["hedge_fund"]`.

**PR delta:** +15 new result fields (+5 DuPont + 2 regime momentum + 5 growth/RIV + 3 Kelly) vs v1.

---

## Running Total Across All 4 Layers

| Layer | Cache Columns | Result Fields | Status |
|-------|--------------|---------------|--------|
| Layer 1: Features | ~428 | ~9 | Updated v3 |
| Layer 2: Analysis | ~71 | ~77 | Updated v2 |
| Layer 3: Temporal | ~17 | ~121 | |
| Layer 4: Hedge Fund | 0 | **~122** (was ~107) | **Updated v2** |
| **Total** | **~516** | **~329** | |

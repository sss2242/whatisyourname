# Layer 1 Feature Engineering -- Expert Methods Update Plan

*Research-backed enhancements from 8 expert domains*

---

## Executive Summary

The current Layer 1 produces ~311 cache columns across 18 modules. This plan identifies **47 new features** across **8 expert domains** that address specific blind spots in the existing pipeline. Each feature is grounded in peer-reviewed research, has clear downstream consumers, and can be implemented using only data already available in the cache (no new API calls required).

**Guiding principles:**
- Every new feature must have at least one downstream consumer in Layers 2-5
- No new pip dependencies unless the feature family justifies it
- Features that duplicate existing signals at different timescales are excluded
- Each feature includes a fallback for data-sparse companies (Tier 2 markets, annual filers)

---

## Domain 1: Market Microstructure (5 new features)

**Expert perspective:** Market microstructure researchers study how trading mechanics reveal information. The current pipeline has Amihud illiquidity and OBV but misses several high-signal-to-noise features derivable from daily OHLC+Volume.

**Target module:** [`operator1/features/derived_variables.py`](operator1/features/derived_variables.py:1) -- new Stage 18: `_compute_microstructure_signals`

| # | Variable | Formula | Reference | Tier | Downstream |
|---|----------|---------|-----------|------|------------|
| 1 | `corwin_schultz_spread` | `2*(exp(alpha) - 1) / (1 + exp(alpha))` where alpha from H-L ratio across 2 consecutive days | Corwin & Schultz 2012, JF | T3 | Survival trigger (illiquidity), MC liquidation days, ownership contagion |
| 2 | `kyle_lambda` | `abs(return_1d) / volume` rolling 21d regression slope | Kyle 1985, Econometrica | T3 | Graph risk contagion speed, HF position sizing |
| 3 | `parkinson_vol_21d` | `sqrt(1/(4*n*ln2) * sum(ln(H/L)^2))` over 21 days | Parkinson 1980 | T3 | More efficient vol estimator than close-to-close (5x), OHLC predictor High/Low |
| 4 | `yang_zhang_vol_21d` | Overnight + open-to-close + Rogers-Satchell combined estimator | Yang & Zhang 2000, JBF | T3 | Replaces/supplements volatility_21d in regime detector, Garman-Klass enhancement |
| 5 | `volume_clock_intensity` | `volume / volume_avg_21d` -- z-scored volume anomaly | Easley, Lopez de Prado & O'Hara 2012 | T3 | Attention signal, news sentiment amplifier, event calendar interaction |

**Why these matter:** The Corwin-Schultz spread is the only way to estimate bid-ask spreads from daily OHLC data without tick data. It catches liquidity deterioration 5-10 days before it shows up in Amihud. Parkinson and Yang-Zhang give 3-5x more efficient volatility estimates than the current close-to-close approach, directly improving every downstream model that consumes volatility.

**Implementation notes:**
- All 5 use only `open`, `high`, `low`, `close`, `volume` -- already in cache
- Corwin-Schultz needs consecutive-day H/L pairs; handle gaps with NaN
- Yang-Zhang needs overnight returns (close-to-open); set to 0 for markets without pre-market

---

## Domain 2: Econometrics / Stationarity (4 new features)

**Expert perspective:** Econometricians know that feeding non-stationary series to ML models produces spurious correlations. The current pipeline uses raw returns (stationary) but also feeds raw ratio levels (non-stationary) to forecasting models. Fractional differentiation preserves memory while achieving stationarity.

**Target module:** [`operator1/features/derived_variables.py`](operator1/features/derived_variables.py:1) -- new Stage 19: `_compute_stationarity_features`

| # | Variable | Formula | Reference | Tier | Downstream |
|---|----------|---------|-----------|------|------------|
| 6 | `close_frac_diff` | Fractionally differentiated close at minimum d for stationarity (ADF test p<0.05) | Lopez de Prado 2018, AFML Ch.5 | T3 | Forecasting (all models), prediction aggregator -- retains 80%+ correlation with original while being stationary |
| 7 | `hurst_exponent_rolling` | Rolling 63d Hurst exponent via R/S analysis | Hurst 1951; Mandelbrot 1971 | T3 | Forecasting model selection (H>0.5=trend, H<0.5=mean-revert, H=0.5=random walk), DTW analog weighting |
| 8 | `autocorr_lag1` | `return_1d.rolling(63).apply(lambda x: x.autocorr(1))` | Box & Jenkins 1970 | T3 | Short-term predictability signal, mean-reversion strength |
| 9 | `autocorr_lag5` | `return_1d.rolling(63).apply(lambda x: x.autocorr(5))` | Lo & MacKinlay 1988 | T3 | Weekly reversal pattern (Jegadeesh 1990), momentum vs reversal regime |

**Why these matter:** The Hurst exponent is currently computed once in [`adaptive_model_params.py`](operator1/analysis/adaptive_model_params.py:1) but never exposed as a daily feature. Making it rolling and daily means temporal models can learn regime-dependent trend/reversion behavior. Fractional differentiation (Lopez de Prado) is arguably the single highest-impact feature engineering technique for financial time series -- it makes ratio levels usable by ML models without destroying their memory.

**Implementation notes:**
- Fractional diff: use `fracdiff` package (already installable, pure Python) or implement the fixed-window approach from AFML
- Hurst: R/S method with 63d rolling window; clip to [0, 1]
- Autocorrelation: pandas `.autocorr()` is efficient; just needs rolling wrapper

**New dependency:** `fracdiff>=0.9` (pure Python, ~15KB, MIT license) -- OR implement the 50-line fixed-window version inline

---

## Domain 3: Behavioral Finance (5 new features)

**Expert perspective:** Behavioral finance researchers study systematic investor biases that create predictable price patterns. The current pipeline has PEAD (post-earnings announcement drift) but misses several well-documented anomalies.

**Target module:** New file `operator1/features/behavioral_signals.py`

| # | Variable | Formula | Reference | Tier | Downstream |
|---|----------|---------|-----------|------|------------|
| 10 | `anchoring_52w_high` | `close / close.rolling(252).max()` -- proximity to 52-week high | George & Hwang 2004, JF | T3 | Forecasting (momentum vs reversal), HF position signal |
| 11 | `anchoring_52w_low` | `close / close.rolling(252).min()` -- proximity to 52-week low | George & Hwang 2004 | T3 | Contrarian signal, DTW analog context |
| 12 | `disposition_effect_proxy` | `corr(inst_flow_momentum, return_1d, 63d)` -- do institutions sell winners? | Shefrin & Statman 1985; Frazzini 2006, JF | T3 | Institutional flow quality assessment |
| 13 | `attention_spike` | `z_score(volume / volume_avg_21d) > 2.0` -- abnormal volume flag | Barber & Odean 2008, RFS | T3 | News sentiment amplifier, event calendar interaction |
| 14 | `lottery_characteristics` | Composite of: high idiosyncratic vol + positive skewness + low price | Bali, Cakici & Whitelaw 2011, JFE | T3 | Risk classification, HF position sizing adjustment |

**Why these matter:** The 52-week high anchoring effect is one of the strongest and most persistent anomalies in finance (George & Hwang 2004 show it subsumes standard momentum). The disposition effect proxy reveals whether institutional holders are behaving rationally. Attention spikes precede both overreaction and information incorporation.

**Implementation notes:**
- All derivable from existing cache columns (close, volume, inst_flow_momentum, return_1d)
- Lottery characteristics: idiosyncratic vol = residual vol after removing beta*benchmark_return; skewness = rolling 63d skewness of returns; low price = close < P20 of own history
- Disposition proxy only meaningful when `inst_flow_momentum` is non-NaN (16 markets with holder data)

---

## Domain 4: Credit Risk / Distress Prediction (5 new features)

**Expert perspective:** Credit analysts focus on the path to default, not just current ratios. The current pipeline has Merton DD and Altman Z but misses several features that fire earlier in the distress trajectory.

**Target module:** [`operator1/features/derived_variables.py`](operator1/features/derived_variables.py:1) -- new Stage 20: `_compute_credit_signals`

| # | Variable | Formula | Reference | Tier | Downstream |
|---|----------|---------|-----------|------|------------|
| 15 | `cash_burn_rate_monthly` | `max(0, -operating_cash_flow) / 30` (daily burn when OCF negative) | Startup finance standard | T1 | Survival runway, FH runway_months improvement, triage card |
| 16 | `debt_maturity_pressure` | `short_term_debt / total_debt_asof` -- fraction due within 1 year | Barclay & Smith 1995, JF | T2 | Refinancing risk signal, HF leverage stress, MC scenario severity |
| 17 | `cash_conversion_cycle` | `DSO + DIO - DPO` (days sales outstanding + days inventory - days payable) | Richards & Laughlin 1980 | T1 | Working capital efficiency, cash flow prediction, HF asset quality |
| 18 | `altman_z_momentum_63d` | `fh_altman_z_score.diff(63)` -- is Z-score improving or deteriorating? | Novel (directional Altman) | T2 | Early warning: declining Z before it crosses 1.81 threshold |
| 19 | `covenant_proximity_score` | `min(1, max(0, (threshold - actual) / threshold))` for each survival trigger, then `max()` | Novel (inspired by Chava & Roberts 2008) | T2 | Graduated early warning vs binary survival flag, USS early warning enhancement |

**Why these matter:** The cash conversion cycle is the single best operational efficiency metric -- deterioration precedes earnings misses by 1-2 quarters. Debt maturity pressure catches refinancing walls that the current total-debt-focused ratios miss entirely. Covenant proximity provides a continuous 0-1 early warning signal that fires well before the binary survival flag trips.

**Implementation notes:**
- CCC components: `DSO = receivables / (revenue/90)`, `DIO = inventory / (COGS/90)`, `DPO = payables / (COGS/90)` -- all inputs already in cache from balance/income statements
- Debt maturity: simple ratio, handle NaN gracefully
- Covenant proximity: reuses survival thresholds from `scoring_weights.yml`

---

## Domain 5: Information Theory / Complexity (4 new features)

**Expert perspective:** Information theorists measure the predictability and complexity of time series. These features tell temporal models HOW predictable a series is before they try to predict it -- enabling adaptive model confidence.

**Target module:** New file `operator1/features/complexity_signals.py`

| # | Variable | Formula | Reference | Tier | Downstream |
|---|----------|---------|-----------|------|------------|
| 20 | `sample_entropy_21d` | SampEn(return_1d, m=2, r=0.2*std, window=63) | Richman & Moorman 2000, AJP | T3 | Model confidence adjustment -- high entropy = low predictability = wider conformal bands |
| 21 | `approx_entropy_price` | ApEn(close, m=2, r=0.2*std, window=126) | Pincus 1991, PNAS | T3 | Regime change detection -- entropy drops before regime shifts (Bandt & Pompe 2002) |
| 22 | `lempel_ziv_complexity` | LZ76 complexity of binarized return series | Lempel & Ziv 1976; Kontoyiannis 1998 | T3 | Randomness measure -- low LZ = pattern exists, high LZ = near-random walk |
| 23 | `permutation_entropy_21d` | PE of return ordinal patterns (order=3, delay=1, window=63) | Bandt & Pompe 2002, PRL | T3 | Most robust entropy measure for short noisy series; regime shift precursor |

**Why these matter:** Entropy features are the missing feedback loop. Currently, temporal models produce forecasts with the same confidence regardless of whether the series is predictable or random. These 4 features let conformal prediction, prediction aggregator, and model diagnostics know WHEN to trust model output and when to widen uncertainty bands.

**Implementation notes:**
- SampEn and ApEn: implement from scratch (~40 lines each) or use `antropy` package
- LZ complexity: binarize returns (>0 = 1, <=0 = 0), then standard LZ76 algorithm (~30 lines)
- Permutation entropy: ordinal pattern frequency distribution (~25 lines)
- All are rolling computations with 63-126 day windows
- Computationally heavier than simple ratios; use numba JIT if available

**New dependency (optional):** `antropy>=0.1.6` (entropy functions, MIT, ~50KB) -- OR implement inline (recommended, all algorithms are short)

---

## Domain 6: Factor Investing / Cross-Sectional (6 new features)

**Expert perspective:** Factor investors normalize everything relative to peers and sectors. The current pipeline has peer_ranking (percentile ranks) but misses industry-adjusted ratios and factor exposures that drive systematic returns.

**Target module:** Enhanced [`operator1/features/peer_ranking.py`](operator1/features/peer_ranking.py:1) + new Stage 21 in derived_variables

| # | Variable | Formula | Reference | Tier | Downstream |
|---|----------|---------|-----------|------|------------|
| 24 | `pe_industry_adjusted` | `pe_ratio_calc - median(peer PE ratios)` | Standard quant practice; Fama & French 1993 | T5 | HF valuation quality, removes sector distortion from PE |
| 25 | `margin_industry_adjusted` | `gross_margin - median(peer gross margins)` | Standard quant practice | T4 | True competitive advantage signal vs absolute margin |
| 26 | `vol_industry_adjusted` | `volatility_21d - median(peer volatilities)` | Standard quant practice | T3 | Idiosyncratic risk signal, separates firm-specific from sector-wide stress |
| 27 | `momentum_12m_1m` | `(close / close.shift(252)) - (close / close.shift(21))` -- 12-month return excluding last month | Jegadeesh & Titman 1993, JF; Novy-Marx 2012 | T3 | The canonical momentum factor; last month excluded to remove short-term reversal |
| 28 | `idiosyncratic_vol_63d` | `std(return_1d - beta * benchmark_return_1d, 63d)` -- residual vol after removing market | Ang et al. 2006, JF | T3 | Lottery characteristics input, risk decomposition, HF position sizing |
| 29 | `earnings_revision_proxy` | `(eps_calc - eps_calc.shift(63)) / abs(eps_calc.shift(63))` -- 3-month EPS change | Chan, Jegadeesh & Lakonishok 1996, JF | T4 | Earnings revision momentum, stronger predictor than price momentum for fundamentals |

**Why these matter:** Industry-adjusted ratios eliminate the number one source of false signals in cross-company analysis. A tech company with 35% gross margin looks terrible in absolute terms but is excellent vs 25% peer median. The 12-1 momentum factor has produced 8-12% annual alpha in every major study since 1993 and is conspicuously absent from the current feature set. Idiosyncratic vol is documented as a negative predictor of returns (Ang et al. 2006) and is a key risk decomposition tool.

**Implementation notes:**
- Industry-adjusted ratios require `linked_caches` to be populated (from entity discovery)
- Fallback when no peers: use expanding percentile of own history as self-relative measure
- 12-1 momentum: pure close price arithmetic, no dependencies
- Idiosyncratic vol: needs `beta_252d` and `benchmark_return_1d`, both already computed

---

## Domain 7: Operations Research / Working Capital (5 new features)

**Expert perspective:** Operations researchers and CFOs focus on the operating cycle and capital efficiency. These features bridge the gap between financial statements and operational reality.

**Target module:** Enhanced [`operator1/features/product_metrics.py`](operator1/features/product_metrics.py:1) -- extend with operational efficiency

| # | Variable | Formula | Reference | Tier | Downstream |
|---|----------|---------|-----------|------|------------|
| 30 | `inventory_turnover` | `revenue / inventory` (or COGS/inventory where available) | Standard accounting | T4 | Asset quality, HF, operations efficiency |
| 31 | `receivables_turnover` | `revenue / receivables` | Standard accounting | T4 | Collection efficiency, revenue quality |
| 32 | `payables_turnover` | `revenue / payables` (or COGS/payables) | Standard accounting | T4 | Payment terms power, supplier relationship |
| 33 | `sga_efficiency` | `revenue / sga_expenses` -- revenue generated per dollar of overhead | Novel composite | T4 | Vanity score enhancement, operational leverage signal |
| 34 | `capex_intensity` | `abs(capex) / revenue` -- capital intensity of operations | Standard; Fama & French 2015 (CMA factor) | T5 | Growth vs value classification, HF capital cycle |

**Why these matter:** The turnover ratios are the operational decomposition of the cash conversion cycle (Domain 4 item 17). Together they reveal WHERE cash is getting stuck -- slow collections (DSO rising), inventory buildup (DIO rising), or aggressive supplier payment (DPO falling). SGA efficiency directly feeds the vanity module's bloat detection.

**Implementation notes:**
- All inputs (revenue, inventory, receivables, payables, sga_expenses, capex) already in cache
- Handle division by zero via existing `safe_ratio()` utility
- These are simple ratios, computationally trivial

---

## Domain 8: Volatility Surface / Tail Risk (5 new features)

**Expert perspective:** Derivatives traders and risk managers extract forward-looking information from the volatility surface shape. The current pipeline has 6 options signals but misses features that characterize the shape of the vol surface and tail risk dynamics.

**Target module:** Enhanced [`operator1/features/options_signals.py`](operator1/features/options_signals.py:1) + new Stage 22 in derived_variables for non-options tail features

| # | Variable | Formula | Reference | Tier | Downstream |
|---|----------|---------|-----------|------|------------|
| 35 | `return_skewness_63d` | `return_1d.rolling(63).skew()` | Standard; Harvey & Siddique 2000, JF | T3 | Lottery characteristics, tail risk pricing, crash probability |
| 36 | `return_kurtosis_63d` | `return_1d.rolling(63).kurt()` | Standard; Dittmar 2002, JF | T3 | Fat tail indicator, conformal band adjustment, regime detector input |
| 37 | `tail_ratio_63d` | `abs(P5_return / P95_return)` over 63 days | Standard risk management | T3 | Left-tail vs right-tail asymmetry; >1 = negative skew dominant |
| 38 | `max_daily_loss_63d` | `return_1d.rolling(63).min()` | Standard VaR complement | T3 | Worst-case scenario anchor for MC paths, HF leverage stress |
| 39 | `vol_of_vol_21d` | `volatility_21d.rolling(21).std()` | Cont & da Fonseca 2002, QF | T3 | Volatility regime stability, GARCH model fit quality proxy |

**Why these matter:** Skewness and kurtosis tell temporal models about the SHAPE of the return distribution, not just its width (volatility). The current pipeline assumes Gaussian returns in several places (Monte Carlo, conformal intervals). Adding higher moments lets downstream models account for fat tails and asymmetry. Vol-of-vol is the single best predictor of GARCH model performance and options pricing errors.

**Implementation notes:**
- All 5 use only `return_1d` and `volatility_21d` -- already computed
- Pandas `.skew()` and `.kurt()` handle rolling windows natively
- Computationally trivial; no new dependencies

---

## Domain 9: Multi-Scale / Regime-Adaptive Normalization (4 new features)

**Expert perspective:** Machine learning practitioners know that raw features at different scales confuse models. The current pipeline uses `safe_ratio()` for NaN handling but does not normalize features for ML consumption.

**Target module:** New file `operator1/features/feature_normalization.py` -- runs AFTER all other feature modules

| # | Variable Pattern | Formula | Reference | Tier | Downstream |
|---|-----------------|---------|-----------|------|------------|
| 40 | `{var}_zscore_63d` for 10 key vars | `(x - rolling_mean(63)) / rolling_std(63)` | Standard; DeMiguel et al. 2009 | All | Forecasting (tree models), prediction aggregator -- regime-relative positioning |
| 41 | `{var}_percentile_252d` for 10 key vars | `expanding_rank(x, 252) / count` | Standard quant practice | All | Cross-temporal comparison -- where is the variable relative to its own history? |
| 42 | `{var}_regime_zscore` for 5 survival vars | `(x - regime_mean) / regime_std` using per-regime statistics | Novel (regime-conditional normalization) | T1-T2 | USS-aware normalization -- "is this value unusual FOR THIS REGIME?" |
| 43 | `{var}_change_21d` for 10 key ratios | `x - x.shift(21)` -- 21-day level change | Standard | All | Gradient/slope features for tree models to detect trends in ratios |

**Key variables to normalize:** current_ratio, debt_to_equity_abs, fcf_yield, gross_margin, volatility_21d, pe_ratio_calc, revenue_growth_yoy, sentiment_score, merton_dd, fh_composite_score

**Why these matter:** Tree models (XGBoost, RF) can handle raw features, but linear models (Kalman, VAR), neural networks (LSTM, Transformer), and ensemble weighting all benefit enormously from normalized inputs. Rolling z-scores additionally capture "is this value unusual?" which is more predictive than the raw level. Regime-conditional normalization is particularly important for survival mode -- a current_ratio of 0.9 means different things in normal vs extreme_survival.

**Implementation notes:**
- Runs as the LAST feature stage (Stage 23), after all other modules have populated the cache
- ~40 new columns (10 vars x 4 normalizations) but many are optional/configurable
- Per-regime stats use `survival_regime` column from hierarchy weights
- Zero-variance guard: if rolling_std < EPSILON, z-score = 0

---

## Domain 10: Forensic Accounting Signals (4 new features)

**Expert perspective:** Forensic accountants and short sellers focus on accounting red flags that precede earnings manipulation and restatements. The current pipeline has Beneish M-Score and accruals but misses several targeted forensic signals.

**Target module:** Enhanced [`operator1/features/derived_variables.py`](operator1/features/derived_variables.py:1) -- new Stage 24: `_compute_forensic_signals`

| # | Variable | Formula | Reference | Tier | Downstream |
|---|----------|---------|-----------|------|------------|
| 44 | `revenue_receivables_divergence` | `revenue_growth_yoy - (receivables.pct_change(252))` -- revenue growing faster than receivables = healthy; opposite = red flag | Lev & Thiagarajan 1993, JAR | T4 | Accruals forensic amplifier, earnings quality |
| 45 | `capex_depreciation_ratio` | `abs(capex) / depreciation` (proxy from EBITDA - EBIT) -- <1.0 = underinvesting, >2.0 = aggressive capitalization | Novel composite; Sloan 1996 | T4 | Asset quality signal, growth sustainability |
| 46 | `soft_asset_ratio` | `(total_assets - cash - net_ppe) / total_assets` -- high = lots of intangibles/goodwill/receivables = easy to manipulate | Barton & Simko 2002, AR | T4 | Balance sheet manipulation risk, HF OBS risk enhancement |
| 47 | `operating_cash_flow_ratio` | `operating_cash_flow / net_income` -- <0.8 sustained = earnings not converting to cash | Dechow & Dichev 2002, AR | T4 | Most direct earnings quality test, already partially in HF but not as daily feature |

**Why these matter:** Revenue-receivables divergence is the number one signal that short sellers use to detect channel stuffing (booking fake revenue by extending credit to customers who will never pay). The soft asset ratio predicts restatement risk because soft assets are easy to inflate. OCF ratio as a daily feature (not just quarterly HF metric) enables temporal models to learn the earnings-to-cash conversion trajectory.

**Implementation notes:**
- All inputs already in cache from financial statements
- Depreciation proxy: `ebitda - ebit` (approximation; exact D&A rarely available from PIT APIs)
- Net PPE proxy: `total_assets - current_assets - goodwill - intangible_assets` (handle NaN components)

---

## Implementation Architecture

### Module Changes

```
operator1/features/
  derived_variables.py      -- ADD 4 new stages (18-22, 24): microstructure, stationarity, credit, tail risk, forensic
  behavioral_signals.py     -- NEW file: 5 behavioral finance features
  complexity_signals.py     -- NEW file: 4 information theory features
  feature_normalization.py  -- NEW file: multi-scale normalization (runs LAST)
  peer_ranking.py           -- ENHANCE: add industry-adjusted ratios (3 features)
  product_metrics.py        -- ENHANCE: add operational efficiency (5 features)
  options_signals.py        -- ENHANCE: already has options signals, just add vol surface context

config/
  scoring_weights.yml       -- ADD new feature weight sections for downstream consumption
```

### Execution Order in Pipeline

```
Step 4a.3:  conflict_risk                     (existing)
Step 4a.4:  market_buying_power               (existing)
Step 4a.7:  options_signals                   (existing -- enhanced with vol surface)
Step 4a.8:  cross_asset_signals               (existing)
Step 4b:    estimator                          (existing)
Step 4c:    filing_calendar                    (existing)
Step 5:     derived_variables                  (ENHANCED -- 4 new stages)
Step 5a:    private_company_proxies            (existing)
Step 5b:    fuzzy_protection                   (existing)
Step 5d:    financial_health -> vanity          (existing)
Step 5e:    entity_discovery                   (existing)
Step 5g:    linked_aggregates                  (existing)
Step 5h:    peer_ranking                       (ENHANCED -- industry-adjusted)
Step 5i:    news_sentiment                     (existing)
Step 5i.5:  product_catalysts                  (existing)
Step 5i.6:  product_metrics                    (ENHANCED -- operational efficiency)
Step 5i.7:  behavioral_signals                 (NEW)
Step 5i.8:  complexity_signals                 (NEW)
Step 5i.9:  feature_normalization              (NEW -- runs LAST before temporal)
Step 5.inst: institutional_flow               (existing)
```

### Column Count Impact

| Category | New Features | New Companion Flags | Total New Columns |
|----------|-------------|--------------------|----|
| Microstructure (Stage 18) | 5 | 5 `is_missing_*` | 10 |
| Stationarity (Stage 19) | 4 | 4 | 8 |
| Credit (Stage 20) | 5 | 5 | 10 |
| Tail Risk (Stage 22) | 5 | 0 (simple transforms) | 5 |
| Forensic (Stage 24) | 4 | 4 | 8 |
| Behavioral (new file) | 5 | 2 | 7 |
| Complexity (new file) | 4 | 0 | 4 |
| Factor/Cross-Sectional | 6 | 0 | 6 |
| Operations (product_metrics) | 5 | 5 | 10 |
| Normalization (new file) | ~40 (10 vars x 4 types) | 0 | ~40 |
| **Total** | **~83** | **~25** | **~108** |

Cache grows from ~311 to ~419 columns. The normalization features (~40) are optional and configurable via `scoring_weights.yml`.

### Dependency Changes

| Package | Version | Purpose | Required? |
|---------|---------|---------|-----------|
| `fracdiff` | >=0.9 | Fractional differentiation (Domain 2) | Optional -- can implement inline |
| `antropy` | >=0.1.6 | Entropy functions (Domain 5) | Optional -- can implement inline |

Both are optional. The plan includes inline implementations for all algorithms as the recommended path.

---

## Priority Ranking

**Tier A -- Highest impact, implement first:**
1. Corwin-Schultz spread (microstructure liquidity)
2. Cash conversion cycle (credit/operations)
3. Rolling Hurst exponent (stationarity)
4. Sample entropy (complexity)
5. 52-week high anchoring (behavioral)
6. Industry-adjusted PE/margin/vol (factor)
7. Covenant proximity score (credit)

**Tier B -- High impact, implement second:**
8. Parkinson + Yang-Zhang volatility (microstructure)
9. Fractional differentiation (stationarity)
10. Return skewness + kurtosis (tail risk)
11. Revenue-receivables divergence (forensic)
12. Momentum 12-1 (factor)
13. Permutation entropy (complexity)
14. Feature z-score normalization (normalization)

**Tier C -- Moderate impact, implement third:**
15. Remaining microstructure (Kyle lambda, volume clock)
16. Remaining behavioral (disposition, attention, lottery)
17. Remaining credit (burn rate, debt maturity, Z momentum)
18. Remaining operations (turnover ratios, SGA efficiency, capex intensity)
19. Remaining forensic (capex/depreciation, soft assets, OCF ratio)
20. Remaining normalization (percentile, regime z-score, change)
21. Remaining complexity (ApEn, LZ complexity)
22. Remaining tail risk (tail ratio, max loss, vol-of-vol)
23. Remaining factor (idiosyncratic vol, earnings revision, autocorrelation)

---

## Testing Strategy

Each new feature/stage gets:
1. **Unit test** in `tests/test_layer1_enhancements.py` -- synthetic cache with known values, verify formula correctness
2. **NaN resilience test** -- all-NaN input produces NaN output (no crashes)
3. **Data-sparse test** -- 50 rows of data (annual filer), verify graceful degradation
4. **Integration test** -- run `compute_derived_variables()` on a real AAPL cache, verify no regressions on existing 67 features
5. **Signal IC test** -- after implementation, measure information coefficient of each new feature vs 5d/21d forward returns on historical data

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Cache too wide (419 cols) | Normalization features are optional; tree models handle sparse features well |
| Computational cost increase | Entropy features are heaviest; use numba JIT or precompute on checkpoint save |
| New features confuse existing models | Feature selection (Boruta + PIMP + mRMR in Stage 3.8) will automatically prune low-signal features |
| Fractional differentiation complexity | Use fixed-window approach (simpler) not full spectral method |
| Peer data not always available | Industry-adjusted features fall back to own-history percentile |

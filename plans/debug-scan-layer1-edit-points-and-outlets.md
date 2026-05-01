# Debug Scan: Layer 1 Edit Points, Inputs, Outputs & Inter-Model Outlets

*Systematic scan of every file affected by the Layer 1 feature engineering update*

---

## 1. Files That Will Be DIRECTLY EDITED

### 1a. Primary edit target: `operator1/features/derived_variables.py` (1,042 lines)

**What changes:** Add 4 new computation stages (18-22, 24) to the `_COMPUTE_STAGES` tuple at line 942:
- Stage 18: `_compute_microstructure_signals` (5 features)
- Stage 19: `_compute_stationarity_features` (4 features)
- Stage 20: `_compute_credit_signals` (5 features)
- Stage 22: `_compute_tail_risk_features` (5 features)
- Stage 24: `_compute_forensic_signals` (4 features)

**Critical outlet:** The `DERIVED_VARIABLES` tuple at line 963 MUST be updated with all new variable names. This tuple is consumed by:
- [`operator1/quality/data_quality.py:343`](operator1/quality/data_quality.py:343) -- computes coverage percentages for each variable
- [`operator1/steps/cache_builder.py:147`](operator1/steps/cache_builder.py:147) -- column namespace registry

**Critical outlet:** The `_COMPUTE_STAGES` tuple at line 942 is iterated by [`compute_derived_variables()`](operator1/features/derived_variables.py:999) which is called from:
- [`main.py:1424`](main.py:1424) -- Step 5 feature engineering
- [`backtest_runner.py:604`](backtest_runner.py:604) -- Stage 1 data fetch
- [`operator1/steps/multi_frequency_runner.py:202`](operator1/steps/multi_frequency_runner.py:202) -- per-frequency pipeline
- [`main.py:1837`](main.py:1837) -- linked entity cache enrichment
- [`operator1/monitoring/model_tests.py:99`](operator1/monitoring/model_tests.py:99) -- smoke tests

### 1b. New file: `operator1/features/behavioral_signals.py`

**What it creates:** 5 behavioral finance features (52w high/low anchoring, disposition effect, attention spike, lottery characteristics)

**Integration point:** Must be called from:
- [`main.py`](main.py) -- add Step 5i.7 between product_catalysts (5i.5) and institutional_flow (5.inst)
- [`backtest_runner.py`](backtest_runner.py) -- add matching call after product_catalysts

### 1c. New file: `operator1/features/complexity_signals.py`

**What it creates:** 4 information theory features (sample entropy, approximate entropy, LZ complexity, permutation entropy)

**Integration point:** Must be called from:
- [`main.py`](main.py) -- add Step 5i.8 after behavioral_signals
- [`backtest_runner.py`](backtest_runner.py) -- add matching call

### 1d. New file: `operator1/features/feature_normalization.py`

**What it creates:** ~40 normalization columns (z-scores, percentiles, regime z-scores, level changes for 10 key variables)

**Integration point:** Must be called LAST in the feature pipeline:
- [`main.py`](main.py) -- add Step 5i.9 after all other feature modules and BEFORE temporal models
- [`backtest_runner.py`](backtest_runner.py) -- add matching call

**Critical dependency:** Requires `survival_regime` column from [`operator1/analysis/hierarchy_weights.py`](operator1/analysis/hierarchy_weights.py) for regime-conditional z-scores. This means it must run AFTER Step 5 (survival + hierarchy) but BEFORE Step 6 (temporal).

### 1e. Enhanced: `operator1/features/peer_ranking.py` (270 lines)

**What changes:** Add 3 industry-adjusted ratio computations (pe_industry_adjusted, margin_industry_adjusted, vol_industry_adjusted)

**Current variable list (line 44):** `close, return_1d, volatility_21d, revenue, net_income, total_debt, free_cash_flow, fcf_yield, gross_margin, pe_ratio_calc`

**Add to ranked variables:** The new industry-adjusted features should NOT be added to the ranked list (they ARE the adjustment). They're computed as: `target_value - median(peer_values)`.

### 1f. Enhanced: `operator1/features/product_metrics.py` (289 lines)

**What changes:** Add 5 operational efficiency features (inventory_turnover, receivables_turnover, payables_turnover, sga_efficiency, capex_intensity)

**Current 9 variables:** segment_hhi, cannibalization_rate, network_effect_score, input_cost_pressure, growth_runway_quarters, maturity_concentration, estimated_market_share, dominant_segment_growth, net_new_revenue_pct

**Add 5 more:** The new features use statement-level inputs (revenue, inventory, receivables, payables, sga_expenses, capex, cost_of_revenue) already in cache.

---

## 2. Files That Will Be INDIRECTLY AFFECTED (downstream consumers)

### 2a. `operator1/stages/stage3_temporal.py` -- _init_extra_vars() at line 24

**Impact:** The `_init_extra_vars()` function at line 24-60 builds the `extra_vars` list from cache column name prefixes. Currently matches:
```python
c.startswith("fh_") or c.startswith("sentiment_")
or c.startswith("peer_") or c.startswith("macro_")
or c.startswith("inst_") or c.startswith("buying_power_")
or c.startswith("catalyst_") or c.startswith("conflict_")
or c.startswith("demand_") or c.startswith("merton_")
or c.startswith("rv_") or c.startswith("policy_risk_")
or c.startswith("sector_leader_") or c.startswith("segment_")
or c.startswith("product_") or c.startswith("pricing_")
or c.startswith("margin_") or c.startswith("som_")
or c.startswith("customer_")
or c in ("stability_score_21d", "buying_power_index", ...)
```

**MUST ADD prefixes for new features:**
- `"corwin_schultz"` or `"cs_spread"` (microstructure)
- `"kyle_"` (microstructure)
- `"parkinson_"`, `"yang_zhang_"`, `"rogers_satchell_"` (vol estimators)
- `"volume_clock"` (microstructure)
- `"hurst_"` (stationarity)
- `"autocorr_"` (stationarity)
- `"frac_diff"` or `"close_frac_diff"` (stationarity)
- `"anchoring_"` (behavioral)
- `"disposition_"` (behavioral)
- `"attention_"` (behavioral)
- `"lottery_"` (behavioral)
- `"cash_burn_"`, `"debt_maturity_"`, `"ccc_"`, `"covenant_"` (credit)
- `"sample_entropy"`, `"approx_entropy"`, `"lz_complexity"`, `"perm_entropy"` (complexity)
- `"skewness_"`, `"kurtosis_"`, `"tail_ratio"`, `"vol_of_vol"` (tail risk)
- `"revenue_rec_div"`, `"capex_depr"`, `"soft_asset"`, `"ocf_ratio"` (forensic)
- `"_zscore_"`, `"_percentile_"`, `"_regime_zscore"`, `"_change_"` (normalization)

**OR (simpler approach):** Add a catch-all for new prefixes, or use a config-driven allowlist.

### 2b. `operator1/analysis/signal_ic.py` -- Signal IC measurement

**Impact:** The `_SIGNAL_SPEED` dict at lines 37-78 controls which columns are measured for information coefficient. Currently has 39 signals in 3 speed categories (slow/medium/fast).

**SHOULD ADD new features to IC measurement:**
- Slow signals: `cash_conversion_cycle`, `debt_maturity_pressure`, `soft_asset_ratio`, `ocf_ratio`, `capex_intensity`, `sga_efficiency`, `revenue_receivables_divergence`
- Medium signals: `covenant_proximity_score`, `altman_z_momentum_63d`, `earnings_revision_proxy`, `disposition_effect_proxy`, `pe_industry_adjusted`
- Fast signals: `corwin_schultz_spread`, `parkinson_vol_21d`, `yang_zhang_vol_21d`, `hurst_exponent_rolling`, `sample_entropy_21d`, `skewness_63d`, `kurtosis_63d`, `anchoring_52w_high`, `momentum_12_1`, `attention_spike`

### 2c. `config/survival_hierarchy.yml` -- Tier variable membership

**Impact:** Defines which variables belong to each survival tier. New features should be classified:

- **Tier 1 (Liquidity):** `cash_burn_rate_monthly`, `cash_conversion_cycle`
- **Tier 2 (Solvency):** `debt_maturity_pressure`, `covenant_proximity_score`, `corwin_schultz_spread` (liquidity proxy), `altman_z_momentum_63d`
- **Tier 3 (Stability):** `parkinson_vol_21d`, `yang_zhang_vol_21d`, `hurst_exponent_rolling`, `skewness_63d`, `kurtosis_63d`, `vol_of_vol_21d`, `anchoring_52w_high`, `momentum_12_1`, `sample_entropy_21d`
- **Tier 4 (Profitability):** `sga_efficiency`, `inventory_turnover`, `receivables_turnover`, `revenue_receivables_divergence`, `capex_depreciation_ratio`, `soft_asset_ratio`, `ocf_ratio`, `earnings_revision_proxy`
- **Tier 5 (Growth):** `pe_industry_adjusted`, `capex_intensity`, `idiosyncratic_vol_63d`

### 2d. `config/scoring_weights.yml` -- Survival thresholds

**Impact:** May need new thresholds for features that become survival triggers:
- `corwin_schultz_spread > X` (illiquidity threshold)
- `covenant_proximity_score > 0.8` (near-breach early warning)
- `cash_burn_rate_monthly > Y` (rapid cash depletion)

### 2e. `operator1/monitoring/model_tests.py` -- Smoke test registry

**Impact:** The `MODEL_TESTS` dict at line 320 needs new entries:
- `"behavioral_signals"` -> smoke test for new module
- `"complexity_signals"` -> smoke test for new module
- `"feature_normalization"` -> smoke test for new module
- Update `"derived_variables"` test to verify new stages produce output

---

## 3. COMPLETE INPUT/OUTPUT/OUTLET MAP

### 3.1 New Features -> Downstream Consumers (Output Outlets)

| New Feature | Direct Consumer | How Consumed |
|---|---|---|
| `corwin_schultz_spread` | survival_mode.py | Potential new survival trigger (illiquidity) |
| `corwin_schultz_spread` | monte_carlo.py | Liquidation days estimation |
| `corwin_schultz_spread` | ownership_contagion.py | Crowding + illiquidity interaction |
| `corwin_schultz_spread` | signal_ic.py | Fast signal IC measurement |
| `kyle_lambda` | graph_risk.py | Contagion speed through network |
| `kyle_lambda` | hedge_fund/engine.py | Position sizing (higher lambda = less liquid) |
| `parkinson_vol_21d` | regime_detector.py | Alternative HMM input (more efficient than close-close vol) |
| `parkinson_vol_21d` | ohlc_predictor.py | High/Low prediction improvement |
| `yang_zhang_vol_21d` | adaptive_model_params.py | Replaces/supplements Garman-Klass factor |
| `yang_zhang_vol_21d` | forecasting.py (GARCH) | Better vol initialization |
| `volume_clock_intensity` | news_sentiment.py | Attention signal amplifier |
| `volume_clock_intensity` | event_calendar.py | Event impact confirmation |
| `close_frac_diff` | forecasting.py (all models) | Stationary close proxy preserving memory |
| `close_frac_diff` | transformer_forecaster.py | Better input than raw close |
| `hurst_exponent_rolling` | forecasting.py | Model selection (H>0.5=trend models, H<0.5=mean-reversion) |
| `hurst_exponent_rolling` | dtw_analogs.py | Analog weighting (similar Hurst = better analog) |
| `hurst_exponent_rolling` | prediction_aggregator.py | Confidence adjustment based on predictability |
| `autocorr_lag1` | forecasting.py | Short-term predictability signal for VAR/AR models |
| `autocorr_lag5` | forecasting.py | Weekly reversal pattern detection |
| `anchoring_52w_high` | prediction_aggregator.py | Momentum/reversal regime signal |
| `anchoring_52w_high` | hedge_fund/engine.py | Position signal context |
| `disposition_effect_proxy` | institutional_flow.py | Flow quality assessment |
| `attention_spike` | news_sentiment.py | Sentiment signal amplifier |
| `lottery_characteristics` | hedge_fund/engine.py | Risk classification for position sizing |
| `cash_burn_rate_monthly` | financial_health.py | Better runway_months computation |
| `cash_burn_rate_monthly` | scenario_engine.py | Cash runway under stress scenarios |
| `debt_maturity_pressure` | hedge_fund/engine.py | Leverage stress scenario severity |
| `debt_maturity_pressure` | monte_carlo.py | Refinancing risk in survival simulation |
| `cash_conversion_cycle` | hedge_fund/engine.py | Asset quality assessment input |
| `cash_conversion_cycle` | prediction_aggregator.py | Operations efficiency signal |
| `altman_z_momentum_63d` | survival_mode.py | Early warning (declining Z before threshold) |
| `altman_z_momentum_63d` | survival_regime_controller.py | Early warning score enhancement |
| `covenant_proximity_score` | survival_regime_controller.py | Graduated early warning (replaces binary) |
| `covenant_proximity_score` | monte_carlo.py | Survival probability calibration |
| `sample_entropy_21d` | conformal.py | Wider bands when entropy high (low predictability) |
| `sample_entropy_21d` | model_diagnostics.py | Expected model performance assessment |
| `approx_entropy_price` | regime_detector.py | Entropy drops precede regime shifts |
| `lz_complexity` | prediction_aggregator.py | Confidence: near 1.0 = random walk = widen bands |
| `perm_entropy_21d` | forecasting.py | Model trust signal |
| `pe_industry_adjusted` | hedge_fund/engine.py | Valuation quality (removes sector distortion) |
| `margin_industry_adjusted` | hedge_fund/engine.py | True competitive advantage signal |
| `vol_industry_adjusted` | prediction_aggregator.py | Idiosyncratic vs systematic risk decomposition |
| `momentum_12_1` | prediction_aggregator.py | Canonical momentum factor |
| `idiosyncratic_vol_63d` | hedge_fund/engine.py | Position sizing, lottery characteristics input |
| `earnings_revision_proxy` | hedge_fund/engine.py | Earnings momentum (stronger than price momentum) |
| `inventory_turnover` | hedge_fund/engine.py | Asset quality decomposition |
| `receivables_turnover` | hedge_fund/engine.py | Revenue quality signal |
| `payables_turnover` | hedge_fund/engine.py | Supplier power signal |
| `sga_efficiency` | vanity.py | SGA bloat detection improvement |
| `capex_intensity` | hedge_fund/engine.py | Capital cycle classification |
| `skewness_63d` | monte_carlo.py | Non-Gaussian path generation |
| `skewness_63d` | conformal.py | Asymmetric band widening |
| `kurtosis_63d` | monte_carlo.py | Fat-tail path generation |
| `kurtosis_63d` | conformal.py | Heavy-tail band widening |
| `tail_ratio_63d` | prediction_aggregator.py | Downside vs upside asymmetry |
| `max_daily_loss_63d` | monte_carlo.py | Worst-case scenario anchor |
| `vol_of_vol_21d` | forecasting.py (GARCH) | GARCH fit quality proxy |
| `{var}_zscore_63d` (x10) | forecasting.py (tree models) | Normalized features for ML models |
| `{var}_percentile_252d` (x10) | prediction_aggregator.py | Historical context signal |
| `{var}_regime_zscore` (x5) | survival_regime_controller.py | "Is this unusual FOR THIS REGIME?" |
| `{var}_change_21d` (x10) | forecasting.py (tree models) | Gradient features for trend detection |
| `revenue_receivables_divergence` | hedge_fund/accruals_forensics.py | Channel stuffing detection |
| `capex_depreciation_ratio` | hedge_fund/engine.py | Investment quality signal |
| `soft_asset_ratio` | hedge_fund/engine.py | Manipulation risk proxy |
| `ocf_ratio` | hedge_fund/fcf_quality.py | Daily cash conversion tracking |

### 3.2 Input Dependencies (What Each New Feature Needs From Cache)

| New Feature | Required Cache Columns | Available? |
|---|---|---|
| `corwin_schultz_spread` | `high`, `low` | Yes (OHLCV) |
| `kyle_lambda` | `return_1d`, `volume`, `close` | Yes |
| `parkinson_vol_21d` | `high`, `low` | Yes (OHLCV) |
| `yang_zhang_vol_21d` | `open`, `high`, `low`, `close` | Yes (OHLCV) |
| `volume_clock_intensity` | `volume`, `volume_avg_21d` | Yes (vol_avg computed in Stage 10) |
| `close_frac_diff` | `close` | Yes |
| `hurst_exponent_rolling` | `return_1d` | Yes (computed in Stage 1) |
| `autocorr_lag1/5` | `return_1d` | Yes |
| `anchoring_52w_high/low` | `close` | Yes |
| `disposition_effect_proxy` | `return_1d`, `inst_flow_momentum` | Yes but `inst_flow_momentum` computed AFTER derived_vars -> must run in behavioral_signals.py not derived_variables.py |
| `attention_spike` | `volume`, `volume_avg_21d` | Yes |
| `lottery_characteristics` | `return_1d`, `close`, `beta_252d`, `benchmark_return_1d` | Yes (beta computed in Stage 14) |
| `cash_burn_rate_monthly` | `operating_cash_flow` | Yes |
| `debt_maturity_pressure` | `short_term_debt`, `total_debt_asof` | Yes (total_debt_asof computed in Stage 2) |
| `cash_conversion_cycle` | `receivables`, `inventory`, `revenue`, `cost_of_revenue`, `payables` | Partial -- `cost_of_revenue` may be NaN; fallback to `revenue - gross_profit` |
| `altman_z_momentum_63d` | `fh_altman_z_score` | Yes but computed in financial_health.py (Step 5d) -> must run AFTER FH |
| `covenant_proximity_score` | `current_ratio`, `debt_to_equity_abs`, `fcf_yield`, `drawdown_252d` | Yes (all computed in derived_variables Stages 2-5) |
| `sample_entropy_21d` | `return_1d` | Yes |
| `approx_entropy_price` | `close` | Yes |
| `lz_complexity` | `return_1d` | Yes |
| `perm_entropy_21d` | `return_1d` | Yes |
| `pe_industry_adjusted` | `pe_ratio_calc`, linked_caches PE values | Needs peer data -> must run in peer_ranking.py or behavioral_signals.py AFTER linked entity fetch |
| `margin_industry_adjusted` | `gross_margin`, linked_caches margins | Same as above |
| `vol_industry_adjusted` | `volatility_21d`, linked_caches vol | Same as above |
| `momentum_12_1` | `close` | Yes |
| `idiosyncratic_vol_63d` | `return_1d`, `beta_252d`, `benchmark_return_1d` | Yes |
| `earnings_revision_proxy` | `eps_calc` or `net_income_ttm_asof` | Yes (computed in Stages 9, 11) |
| `inventory_turnover` | `revenue`, `inventory` | Yes |
| `receivables_turnover` | `revenue`, `receivables` | Yes |
| `payables_turnover` | `revenue`/`cost_of_revenue`, `payables` | Partial |
| `sga_efficiency` | `revenue`, `sga_expenses` | Yes |
| `capex_intensity` | `capex`, `revenue` | Yes |
| `skewness_63d` | `return_1d` | Yes |
| `kurtosis_63d` | `return_1d` | Yes |
| `tail_ratio_63d` | `return_1d` | Yes |
| `max_daily_loss_63d` | `return_1d` | Yes |
| `vol_of_vol_21d` | `volatility_21d` | Yes |
| `{var}_zscore_63d` | 10 key variables | Yes (all computed by prior stages) |
| `{var}_percentile_252d` | 10 key variables | Yes |
| `{var}_regime_zscore` | 5 survival variables + `survival_regime` | Yes but `survival_regime` from hierarchy_weights -> normalization must run AFTER Step 5 |
| `{var}_change_21d` | 10 key ratio variables | Yes |
| `revenue_receivables_divergence` | `revenue`, `receivables` | Yes |
| `capex_depreciation_ratio` | `capex`, `ebitda`, `ebit`/`operating_income` | Yes (ebitda computed in Stage 6) |
| `soft_asset_ratio` | `total_assets`, `cash_and_equivalents` | Yes |
| `ocf_ratio` | `operating_cash_flow`, `net_income` | Yes |

---

## 4. PIPELINE EXECUTION ORDER CONSTRAINTS

```
SAFE TO ADD TO derived_variables.py (no external deps):
  Stage 18: microstructure (needs OHLCV -- always available at this point)
  Stage 19: stationarity (needs return_1d from Stage 1)
  Stage 20: credit signals (needs Stage 2/3/5 outputs + statement columns)
  Stage 22: tail risk (needs return_1d from Stage 1, volatility_21d from Stage 1)
  Stage 24: forensic (needs statement columns + Stage 6 ebitda)

MUST BE SEPARATE MODULE (depends on other feature modules):
  behavioral_signals.py: disposition_effect needs inst_flow_momentum (Step 5.inst)
                         lottery_characteristics needs beta_252d (Stage 14 of derived_vars -- OK)
                         pe_industry_adjusted needs linked_caches (Step 5g)
  
  complexity_signals.py: no external deps, pure return_1d/close computation
                         CAN go in derived_variables.py but better separate for clarity

  feature_normalization.py: MUST run LAST
                            needs survival_regime (Step 5 hierarchy weights)
                            needs fh_altman_z_score (Step 5d financial_health)
                            needs all prior feature columns

EXECUTION ORDER (amended pipeline):
  Step 4a.3:  conflict_risk           (existing)
  Step 4a.4:  market_buying_power     (existing)
  Step 4a.7:  options_signals         (existing)
  Step 4a.8:  cross_asset_signals     (existing)
  Step 4b:    estimator               (existing)
  Step 4c:    filing_calendar         (existing)
  Step 5:     derived_variables       (ENHANCED: +23 features in 5 new stages)
  Step 5a:    private_company_proxies (existing)
  Step 5.inst: institutional_flow     (existing)
  Step 5b:    fuzzy_protection        (existing)
  Step 5-USS: survival_regime_ctrl    (existing)
  Step 5d:    financial_health        (existing)
  Step 5d.v:  vanity                  (existing)
  Step 5e:    entity_discovery        (existing)
  Step 5g:    linked_aggregates       (existing)
  Step 5h:    peer_ranking            (ENHANCED: +3 industry-adjusted features)
  Step 5i:    news_sentiment          (existing)
  Step 5i.5:  product_catalysts       (existing)
  Step 5i.6:  product_metrics         (ENHANCED: +5 operational features)
  Step 5i.7:  behavioral_signals      (NEW: 5 features)
  Step 5i.8:  complexity_signals      (NEW: 4 features)
  Step 5i.9:  feature_normalization   (NEW: ~40 features, MUST BE LAST)
  Step 5j:    adaptive_thresholds     (existing)
  Step 5.5:   regime_detector         (existing)
  Step 5k:    adaptive_model_params   (existing)
  Step 5k.2:  adaptive_windows        (existing)
  [TEMPORAL MODELS START]
```

---

## 5. RISK: Potential Breaking Points

| Risk | Location | Mitigation |
|---|---|---|
| `DERIVED_VARIABLES` tuple out of sync | [`derived_variables.py:963`](operator1/features/derived_variables.py:963) | Must add ALL new variable names to tuple; `data_quality.py` uses this for coverage |
| `_init_extra_vars` misses new prefixes | [`stage3_temporal.py:24`](operator1/stages/stage3_temporal.py:24) | Must add new prefix patterns to the filter logic |
| Column count regression in model_tests | [`model_tests.py:99`](operator1/monitoring/model_tests.py:99) | `_test_derived_variables` checks output column count -- update expected count |
| Normalization runs before hierarchy | [`feature_normalization.py`](operator1/features/feature_normalization.py) (new) | Regime z-scores need `survival_regime`; enforce execution order |
| CCC uses `cost_of_revenue` (often NaN) | [`derived_variables.py`](operator1/features/derived_variables.py) new Stage 20 | Fallback: `cost_of_revenue = revenue - gross_profit` |
| Entropy features slow on large cache | [`complexity_signals.py`](operator1/features/complexity_signals.py) (new) | Use numba JIT for inner loops; 504 windows of 63 points ~0.5s with numba |
| Backtest runner missing new modules | [`backtest_runner.py:600-625`](backtest_runner.py:600) | Must add calls to behavioral_signals, complexity_signals, feature_normalization |
| MF runner missing new modules | [`multi_frequency_runner.py:201`](operator1/steps/multi_frequency_runner.py:201) | Only calls `compute_derived_variables` -- new stages inside it are auto-included; new MODULES need separate calls |
| Signal IC missing new signals | [`signal_ic.py:37-78`](operator1/analysis/signal_ic.py:37) | Must add new features to `_SIGNAL_SPEED` dict |
| `survival_hierarchy.yml` tier membership | [`config/survival_hierarchy.yml`](config/survival_hierarchy.yml) | Must add new variables to appropriate tiers |

---

## 6. COMPLETE FILE EDIT LIST (ordered by priority)

| # | File | Lines | Edit Type | What Changes |
|---|------|-------|-----------|-------------|
| 1 | `operator1/features/derived_variables.py` | 1042 | MAJOR | Add 5 new `_compute_*` stages (~200 lines), update `_COMPUTE_STAGES` tuple, update `DERIVED_VARIABLES` tuple |
| 2 | `operator1/features/behavioral_signals.py` | NEW | CREATE | 5 behavioral features (~150 lines) |
| 3 | `operator1/features/complexity_signals.py` | NEW | CREATE | 4 entropy features (~200 lines with numba hot paths) |
| 4 | `operator1/features/feature_normalization.py` | NEW | CREATE | z-score/percentile/regime normalization (~150 lines) |
| 5 | `operator1/features/peer_ranking.py` | 270 | MINOR | Add 3 industry-adjusted ratio computations (~40 lines) |
| 6 | `operator1/features/product_metrics.py` | 289 | MINOR | Add 5 operational efficiency features (~60 lines) |
| 7 | `main.py` | 3453 | MINOR | Add Step 5i.7/5i.8/5i.9 calls (~15 lines) |
| 8 | `backtest_runner.py` | 2000 | MINOR | Add matching feature module calls (~15 lines) |
| 9 | `operator1/stages/stage3_temporal.py` | 295 | MINOR | Update `_init_extra_vars()` prefix list (~10 lines) |
| 10 | `operator1/analysis/signal_ic.py` | ~400 | MINOR | Add ~20 new signals to `_SIGNAL_SPEED` dict (~20 lines) |
| 11 | `config/survival_hierarchy.yml` | ~50 | MINOR | Add new variables to tier lists (~15 lines) |
| 12 | `config/scoring_weights.yml` | ~400 | MINOR | Add new threshold sections if any become survival triggers (~10 lines) |
| 13 | `operator1/monitoring/model_tests.py` | ~430 | MINOR | Add 3 new smoke test entries + update derived_variables test (~30 lines) |
| 14 | `operator1/quality/data_quality.py` | 433 | NO CHANGE | Auto-picks up new vars from `DERIVED_VARIABLES` tuple |
| 15 | `operator1/steps/multi_frequency_runner.py` | 606 | MINOR | Add calls to new feature modules for per-frequency pipeline (~6 lines) |
| 16 | `tests/test_phase3_features.py` | ~200 | MINOR | Add test cases for new features (~50 lines) |

**Total estimated code changes:** ~800 lines of new code across 13 files (3 new, 10 modified).

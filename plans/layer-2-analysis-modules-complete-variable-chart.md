# Layer 2: Analysis Modules -- Complete Variable Chart

Every variable produced by the 10 Layer 2 modules in `operator1/analysis/` + the Financial Health module in `operator1/models/`. These modules produce survival flags, regime classifications, protection assessments, adaptive calibration, and health scores that control how all downstream temporal models behave.

---

## 2.1 Survival Mode Detection

**File:** `operator1/analysis/survival_mode.py` (618 lines)
**Pipeline step:** Step 5
**Entry points:** `compute_company_survival_flag()`, `compute_country_survival_flag()`, `compute_country_protected_flag()`, `compute_survival_probability()`, `compute_cox_survival_score()`

### Cache Variables

| # | Variable | Formula / Logic | Type | Consumers |
|---|----------|----------------|------|-----------|
| 1 | `company_survival_mode_flag` | OR of 7+ conditions: `current_ratio < 1.0`, `debt_to_equity_abs > 3.0`, `fcf_yield < 0`, `drawdown_252d < -0.40`, `conflict_intensity_score > 0.7`, `sanctions_flag == 1`, `inst_flow_momentum < -0.15`, `inst_crowding_score > 0.8 AND illiquidity > P90`. **NEW (P6):** suppressed when `cash/market_cap > 5%` | Integer (0/1) | Hierarchy weights, survival timeline, USS controller, Monte Carlo, all temporal models |
| 2 | `country_survival_mode_flag` | Macro thresholds from `config/country_protection_rules.yml`: `credit_spread`, `fx_volatility`, `unemployment_rate`, `yield_curve_slope` | Integer (0/1) | Hierarchy weights, survival timeline |
| 3 | `country_protected_flag` | Any: sector in `strategic_sectors`, `market_cap > 0.001 * GDP`, emergency rate cut detected. Also set by fuzzy protection (`fuzzy_protection_degree >= 0.5`) | Integer (0/1) | Hierarchy weights, survival timeline |
| 4 | `survival_probability` | Sigmoid of distance to each trigger threshold (6 components weighted by severity). Blended: `0.4 * sigmoid + 0.6 * cox` | Continuous (0-1) | Profile, report, Monte Carlo, MF fusion |
| 5 | `cox_survival_score` | `CoxPHFitter.fit(df, duration, event_observed)` with covariates: current_ratio, debt_to_equity_abs, fcf_yield, drawdown_252d. Learns hazard ratios from own distress episodes (via `lifelines`) | Continuous (0-1) | Survival probability blending |

**5 cache columns**

---

## 2.2 Hierarchy Weights

**File:** `operator1/analysis/hierarchy_weights.py` (322 lines)
**Pipeline step:** Step 5
**Entry point:** `compute_hierarchy_weights(cache)`

### Cache Variables

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `survival_regime` | Regime classification from flags: normal (0/0/-), company_survival (1/0/-), modified_survival (0/1/No), extreme_survival (1/1/-) | Categorical | All temporal models, Monte Carlo, HF position signal |
| 2 | `hierarchy_tier1_weight` | Weight for Tier 1 (Liquidity) -- ranges from 20 (normal) to 60 (extreme_survival) | Float (0-100) | Financial health composite, forecasting priority |
| 3 | `hierarchy_tier2_weight` | Weight for Tier 2 (Solvency) | Float (0-100) | Financial health composite |
| 4 | `hierarchy_tier3_weight` | Weight for Tier 3 (Stability) | Float (0-100) | Financial health composite |
| 5 | `hierarchy_tier4_weight` | Weight for Tier 4 (Profitability) | Float (0-100) | Financial health composite |
| 6 | `hierarchy_tier5_weight` | Weight for Tier 5 (Growth) | Float (0-100) | Financial health composite |

**Regime-to-weight mapping:**

| Regime | T1 | T2 | T3 | T4 | T5 |
|--------|----|----|----|----|-----|
| normal | 20 | 20 | 20 | 20 | 20 |
| company_survival | 50 | 30 | 15 | 4 | 1 |
| modified_survival | 40 | 35 | 20 | 4 | 1 |
| extreme_survival | 60 | 30 | 10 | 0 | 0 |

**Vanity adjustment:** When `vanity_percentage > threshold` AND in survival regime, shifts 5% from T4/T5 to T1.

**6 cache columns**

---

## 2.3 Survival Timeline

**File:** `operator1/analysis/survival_timeline.py` (664 lines)
**Pipeline step:** Step 5.5
**Entry points:** `compute_survival_timeline()`, `compute_enriched_survival_timeline()`

### Base Timeline Variables (from `compute_survival_timeline`)

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `survival_mode` | 6-mode classification from 3 flags: normal, company_only, country_protected, country_exposed, both_protected, both_unprotected | Categorical | Walk-forward (mode-conditioned scoring), USS |
| 2 | `survival_mode_code` | Integer code for survival_mode (0-5) | Integer | Walk-forward |
| 3 | `switch_point` | 1 on days where mode changes, 0 otherwise | Binary (0/1) | Walk-forward (retrain trigger), profile |
| 4 | `days_in_mode` | Running counter since last switch (resets at each switch_point) | Integer | Profile, stability score |
| 5 | `stability_score_21d` | Rolling 21-day fraction of same mode (1.0 = fully stable) | Continuous (0-1) | Extra vars, regime shift predictor, adaptive model params |

### Enriched Timeline Variables (from `compute_enriched_survival_timeline`)

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 6 | `regime_state` | 11 combined states from (survival_mode x market_regime): stable_growth, elevated_risk, market_stress, company_distress_mild/severe, country_crisis_mild/severe, protected_stress/crisis, crisis, extreme_crisis | Categorical | Profile, report |
| 7 | `survival_intensity` | Continuous 0-1 metric blending rule-based intensity + HMM regime confidence | Continuous (0-1) | Extra vars, prediction aggregator |
| 8 | `regime_confidence` | HMM posterior probability of the assigned regime state | Continuous (0-1) | Extra vars |
| 9 | `regime_switch` | 1 on enriched regime transition days | Binary (0/1) | Profile |
| 10 | `regime_transition_prob` | Probability of transitioning to a different regime on next day | Continuous (0-1) | Conformal widening, prediction aggregator |
| 11 | `market_regime` | HMM-derived market regime label (bull/bear/high_vol/low_vol/unknown) | Categorical | Combined state mapping |

**11 cache columns**

---

## 2.4 Fuzzy Protection

**File:** `operator1/analysis/fuzzy_protection.py` (394 lines)
**Pipeline step:** Step 5b
**Entry point:** `compute_fuzzy_protection(cache, sector, gdp, parent_sector)`

### Cache Variables

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `fuzzy_sector_score` | Membership function: defense/energy/banking/telecom = high (0.8-1.0), retail/entertainment = low (0.1-0.3) | Continuous (0-1) | Fuzzy inference, profile |
| 2 | `fuzzy_economic_score` | Trapezoidal membership of `market_cap / GDP` ratio | Continuous (0-1) | Fuzzy inference, profile |
| 3 | `fuzzy_policy_score` | Emergency rate cut magnitude within lookback window | Continuous (0-1) | Fuzzy inference, profile |
| 4 | `fuzzy_protection_degree` | Mamdani fuzzy inference engine (11 rules, centroid defuzzification via scikit-fuzzy) OR fuzzy OR max (fallback) | Continuous (0-1) | Country protected flag, hierarchy weights, profile |
| 5 | `fuzzy_protection_label` | "Low" (<0.3) / "Moderate" (0.3-0.6) / "High" (0.6-0.8) / "Very High" (>0.8) | Categorical | Profile, report |
| 6 | `fuzzy_parent_protection_floor` | Parent sector protection score from GLEIF corporate structure (optional) | Continuous (0-1) | Floor for fuzzy_protection_degree |
| 7 | `country_protected_flag` | `fuzzy_protection_degree >= 0.5` (overrides or supplements the rule-based flag from survival_mode) | Binary (0/1) | Hierarchy weights |

**7 cache columns**

---

## 2.5 Ethical Filters

**File:** `operator1/analysis/ethical_filters.py` (355 lines)
**Pipeline step:** Called inside profile_builder
**Entry point:** `compute_all_ethical_filters(cache)`

### Output (result dict, not cache columns)

| # | Variable | Formula | Consumers |
|---|----------|---------|-----------|
| 1 | `filter_purchasing_power.verdict` | PASS/WARNING/FAIL from inflation-adjusted return + real dividend yield | Profile `filters` section |
| 2 | `filter_solvency.verdict` | PASS/WARNING/FAIL from current_ratio, debt_to_equity, interest_coverage vs conservative thresholds | Profile `filters` section |
| 3 | `filter_gharar.verdict` | PASS/WARNING/FAIL from earnings volatility, restatement frequency, accounting opacity | Profile `filters` section |
| 4 | `filter_cash_is_king.verdict` | PASS/WARNING/FAIL from FCF yield, OCF/NI ratio (cash conversion), cash trend | Profile `filters` section |

**0 cache columns** (result dict only, consumed by profile builder)

---

## 2.6 Economic Planes

**File:** `operator1/analysis/economic_planes.py` (135 lines)
**Pipeline step:** Step 6 (pre-forecast)
**Entry point:** `classify_economic_plane(sector, industry)`

### Output (result dict, not cache columns)

| # | Variable | Value | Consumers |
|---|----------|-------|-----------|
| 1 | `primary_plane` | One of: Real Economy, Financial, Technology, Resources, Services | Model synergies (plane-aware weighting), profile |
| 2 | `secondary_planes` | List of secondary plane classifications | Profile |

**0 cache columns** (result dict only)

---

## 2.7 Vanity (Capital Allocation Quality)

**File:** `operator1/analysis/vanity.py` (636 lines)
**Pipeline step:** Step 5d
**Entry point:** `compute_vanity_score(cache)`

### Cache Variables

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `vanity_rnd_mismatch` | R&D spending vs revenue growth rate. High R&D + declining revenue = mismatch. (15% weight) | Continuous (0-100) | Vanity score component |
| 2 | `vanity_sga_bloat_v2` | SGA expenses vs industry benchmarks. (25% weight) | Continuous (0-100) | Vanity score component |
| 3 | `vanity_capital_misallocation` | Debt-funded dividends/buybacks while FCF negative. (30% weight) | Continuous (0-100) | Vanity score component |
| 4 | `vanity_competitive_decay` | Losing market share vs peers from peer_ranking. (15% weight) | Continuous (0-100) | Vanity score component |
| 5 | `vanity_sentiment_gap` | Market sentiment diverging from fundamental reality. (15% weight) | Continuous (0-100) | Vanity score component |
| 6 | `vanity_score` | Weighted composite of 5 components, normalized to 0-100 | Continuous (0-100) | Hierarchy weight adjustment, profile |
| 7 | `vanity_score_21d` | Rolling 21-day average of vanity_score | Continuous (0-100) | Trend detection |
| 8 | `vanity_trend` | Slope: short MA (21d) vs long MA (63d) of vanity_score. Rising = deteriorating. | Continuous | Profile |
| 9 | `vanity_label` | Disciplined (0-20) / Moderate (20-40) / Wasteful (40-70) / Reckless (70-100) | Categorical | Profile, report |
| 10 | `vanity_exec_comp_excess` | Legacy: executive compensation excess | Continuous | Vanity total spend |
| 11 | `vanity_buyback_waste` | Legacy: buyback waste score | Continuous | Vanity total spend |
| 12 | `vanity_marketing_excess` | Legacy: marketing excess score | Continuous | Vanity total spend |
| 13 | `vanity_total_spend` | Sum of legacy vanity components | Continuous | Vanity percentage |
| 14 | `vanity_percentage` | `vanity_total_spend / revenue * 100` | Continuous (%) | Hierarchy weight vanity adjustment |
| 15 | `is_missing_vanity_percentage` | Companion flag | Binary (0/1) | |

**15 cache columns**

---

## 2.8 Financial Health Scoring

**File:** `operator1/models/financial_health.py` (1,081 lines)
**Pipeline step:** Step 5d
**Entry point:** `compute_financial_health(cache, hierarchy_weights)`

### Cache Variables

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `fh_liquidity_score` | Expanding percentile rank: cash_ratio, current_ratio, free_cash_flow | Continuous (0-100) | Composite score (T1 weight) |
| 2 | `fh_solvency_score` | Expanding percentile rank: debt_to_equity, interest_coverage, net_debt_to_ebitda | Continuous (0-100) | Composite score (T2 weight) |
| 3 | `fh_stability_score` | Expanding percentile rank: volatility_21d, drawdown_252d, volume trend | Continuous (0-100) | Composite score (T3 weight) |
| 4 | `fh_profitability_score` | Expanding percentile rank: gross_margin, operating_margin, net_margin | Continuous (0-100) | Composite score (T4 weight) |
| 5 | `fh_growth_score` | Expanding percentile rank: revenue_growth_yoy, pe_ratio_calc, ev_to_ebitda (adaptive caps) | Continuous (0-100) | Composite score (T5 weight) |
| 6 | `fh_composite_score` | Weighted average of 5 tier scores using hierarchy weights | Continuous (0-100) | Profile, report, adaptive thresholds, HF scorecard |
| 7 | `fh_composite_label` | Excellent (>80) / Good (60-80) / Fair (40-60) / Poor (20-40) / Critical (<20) | Categorical | Profile, report |
| 8 | `fh_composite_delta_5d` | `fh_composite_score.diff(5)` -- 5-day change in composite | Continuous | Profile, trend detection |
| 9 | `fh_composite_delta_21d` | `fh_composite_score.diff(21)` -- 21-day change | Continuous | Profile |
| 10 | `fh_altman_z_score` | `Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MVE/TL + 0.999*S/TA` (Altman 1968) | Continuous | Profile, HF advanced methods, report |
| 11 | `fh_altman_z_zone` | safe (>2.99) / grey (1.81-2.99) / distress (<1.81) | Categorical | Profile, report |
| 12 | `fh_beneish_m_score` | 8-factor M-Score: DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA (Beneish 1999) | Continuous | Profile, HF earnings smoothing |
| 13 | `fh_beneish_flag` | 1.0 if M > -2.22 (likely manipulator), else 0.0 | Binary (0/1) | Profile, report |
| 14 | `fh_merton_dd` | Merton distance-to-default (from financial_health module, separate from derived_variables) | Continuous | Profile, HF advanced methods |
| 15 | `fh_merton_pd` | Merton probability of default | Continuous (0-1) | MC survival anchor, HF |
| 16 | `fh_runway_months` | `cash / abs(monthly_burn_rate)` -- months until cash exhaustion | Continuous (months) | Profile, report, triage card |

**16 cache columns**

---

## 2.9 Adaptive Thresholds (Tier 1 Calibration)

**File:** `operator1/analysis/adaptive_thresholds.py` (836 lines)
**Pipeline step:** Step 5j
**Entry point:** `compute_adaptive_thresholds(cache, linked_caches, regime_detector, fh_composite_scores)`

### Output (ThresholdSet dataclass, not cache columns)

| # | Variable | Method | Consumers |
|---|----------|--------|-----------|
| 1 | `survival_thresholds` | (A) Peer Percentile P10/P90 + (D) BOCPD tightening | `compute_company_survival_flag()` (recalibrated) |
| 2 | `regime_mixer_thresholds` | (J) HMM Emission Gaussian crossover | `compute_dual_regimes()` |
| 3 | `vanity_thresholds` | (E) Sector Z-Score (MAD-based) | Vanity flagging |
| 4 | `fh_label_breaks` | (H) Jenks Natural Breaks (Fisher DP) | Financial health label assignment |
| 5 | `mc_thresholds` | `threshold_set_to_mc_dict()` conversion | `run_monte_carlo()` |
| 6 | `adapted` | Boolean: True if any threshold was calibrated | All consumers |

**0 cache columns** (consumed via parameter passing)

---

## 2.10 Adaptive Model Parameters (Tier 2 Calibration)

**File:** `operator1/analysis/adaptive_model_params.py` (1,185 lines)
**Pipeline step:** Step 5k
**Entry point:** Multiple `compute_*` functions, collected into `AdaptiveModelParams`

### Output (AdaptiveModelParams dataclass, not cache columns)

| # | Variable | Method | Consumers |
|---|----------|--------|-----------|
| 1 | `blend_w_sig` | (6) Inverse-Variance Blending (Cochrane 1954) -- sigmoid blend weight | `survival_probability` re-blending |
| 2 | `blend_w_cox` | (6) Cox PH blend weight = 1 - sig_weight | `survival_probability` re-blending |
| 3 | `survival_risk_multiplier` | (9a) Regime Risk Multiplier -- HMM vol ratio between current/base regime | `run_monte_carlo()` path scaling |
| 4 | `intraday_low_factor` | (9b) Garman-Klass Factor (1980) -- intraday vol from OHLC | OHLC predictor low prediction |
| 5 | `transition_halflife` | (9c) Transition Half-Life from enriched timeline switch durations | Regime shift predictor, conformal widening |
| 6 | `mc_n_paths` | (10) Precision-Targeted MC (Glasserman 2003) -- path count for target SE | `run_monte_carlo()` n_paths |
| 7 | `mc_is_tilt` | (10) Importance sampling tilt factor | `run_monte_carlo()` importance_tilt |
| 8 | `participation_rate` | (8b) Amihud Participation Rate (Amihud 2002) | Ownership contagion liquidation |
| 9 | `pid_kp`, `pid_ki`, `pid_kd` | (7) Lambda PID Tuning (Dahlin 1968) from error ACF half-life | Forward pass PID controller |
| 10 | `contagion_probs` | (8a) Copula Tail Contagion (Joe 2014) edge-specific probs | Graph risk contagion |
| 11 | `hurst_exponent` | Hurst exponent from R/S analysis -- trend (>0.5) vs mean-reversion (<0.5) | Forecasting model selection |
| 12 | `adapted` | Boolean: True if any param was calibrated | All consumers |

**0 cache columns** (consumed via parameter passing)

---

## 2.11 Adaptive Windows (Tier 3 Calibration)

**File:** `operator1/analysis/adaptive_windows.py` (552 lines)
**Pipeline step:** Step 5k.2
**Entry point:** `compute_adaptive_windows()`, `compute_nn_hyperparams()`, etc.

### Output (AdaptiveTier3Params dataclass, not cache columns)

| # | Variable | Method | Consumers |
|---|----------|--------|-----------|
| 1 | `windows.short` | (11) base/3 (Nyquist) -- 21 for quarterly, 42 for semi-annual, 84 for annual | Derived variables rolling windows |
| 2 | `windows.medium` | (11) base -- 63/126/252 | Forecasting, forward pass |
| 3 | `windows.long` | (11) 2*base | Forecasting, burnout |
| 4 | `windows.trend` | (11) 4*base | Trend detection |
| 5 | `nn_params.d_model` | (12) Scaling-Law (Kaplan 2020): proportional to n_eff^0.5 | Transformer forecaster |
| 6 | `nn_params.hidden_dim` | (12) proportional to n_eff^0.5 | Transformer, LSTM |
| 7 | `nn_params.lookback` | (12) proportional to n_eff | Transformer, LSTM |
| 8 | `pattern_body_threshold` | (14) P50 of body/range ratio distribution (Bulkowski 2008) | Pattern detector |
| 9 | `pattern_doji_threshold` | (14) P10 of body/range ratio | Pattern detector |
| 10 | `stale_threshold_days` | (18) 2x filing period | Filing calendar staleness |
| 11 | `adapted` | Boolean | All consumers |

**0 cache columns** (consumed via parameter passing)

---

## 2.12 Survival Regime Controller (USS)

**File:** `operator1/analysis/survival_regime_controller.py` (653 lines)
**Pipeline step:** Step 5-USS
**Entry point:** `SurvivalRegimeController.from_cache(cache)`

### Cache Variables

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `early_warning_score` | Continuous 0-1 proximity score to survival triggers. Fires at 0.7 (informational) | Continuous (0-1) | Profile, report |

### Controller Outputs (not cache, passed as parameters)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 2 | `current_regime` | Active USS regime: normal, company_survival, extreme_survival | Forecast bounding, scenario engine, model switching |
| 3 | `is_survival` | Boolean: True when in any survival regime | Scenario engine trigger |
| 4 | `frozen_variables` | List of variables frozen (Tier 4-5 in survival) | Variable triage |
| 5 | `horizons` | Available forecast horizons per regime: extreme=1d/5d, survival=1d/5d/21d, normal=all | Horizon compression |
| 6 | `model_config` | Regime-specific params: Kalman noise 3-5x, LSTM lookback 5-10, MC paths 20-30K | Model switching |
| 7 | `correlation_override` | 0.85-0.90 crisis correlation + Clayton copula for lower tail | Correlation switching |

**1 cache column + 6 controller parameters**

---

## 2.13 Scenario Engine (USS)

**File:** `operator1/analysis/scenario_engine.py` (387 lines)
**Pipeline step:** Step 6-USS
**Entry point:** `run_scenario_engine(cache, regime, n_paths, horizon_days)`

### Output (ScenarioEngineResult, not cache columns)

| # | Output | Description | Per-Scenario Variables |
|---|--------|-------------|----------------------|
| 1 | `orderly` | Management restructures: cash burn -30%, debt renegotiated, revenue -15% | `cash_runway_days`, `survival_prob_90d`, `survival_prob_252d`, `terminal_equity`, `terminal_revenue`, `terminal_cash`, `median_return`, `p5_return`, `p95_return`, `median_max_drawdown` |
| 2 | `muddle_through` | Status quo: full historical return distribution | Same 10 variables |
| 3 | `catastrophic` | Fire sale: assets -40%, all debt called, revenue -40% | Same 10 variables |

**0 cache columns** (result object stored in profile)

---

## Layer 2 Grand Total

| Module | Cache Columns | Result-Only Fields | Total |
|--------|--------------|-------------------|-------|
| 2.1 Survival Mode | 5 | 0 | 5 |
| 2.2 Hierarchy Weights | 6 | 0 | 6 |
| 2.3 Survival Timeline | 11 | 0 | 11 |
| 2.4 Fuzzy Protection | 7 | 0 | 7 |
| 2.5 Ethical Filters | 0 | 4 verdicts | 0 |
| 2.6 Economic Planes | 0 | 2 fields | 0 |
| 2.7 Vanity | 15 | 0 | 15 |
| 2.8 Financial Health | 16 | 0 | 16 |
| 2.9 Adaptive Thresholds | 0 | 6 fields | 0 |
| 2.10 Adaptive Model Params | 0 | 12 fields | 0 |
| 2.11 Adaptive Windows | 0 | 11 fields | 0 |
| 2.12 USS Controller | 1 | 6 params | 1 |
| 2.13 Scenario Engine | 0 | 30 (3x10) | 0 |
| **Total** | **61** | **71** | **61 cache + 71 result** |

These 61 cache columns are added on top of the ~311 from Layer 1, bringing the total cache to **~372 columns** before temporal models run.

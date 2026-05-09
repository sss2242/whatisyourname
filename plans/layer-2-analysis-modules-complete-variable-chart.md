# Layer 2: Analysis Modules -- Complete Variable Chart (v3)

**v3 update (2026-05-09):**
- **Frequency-aware survival mode:** `compute_company_survival_flag(cache, freq="D")` accepts frequency parameter. Survival thresholds scale by frequency (e.g., fcf_yield at Q frequency uses quarterly FCF, not daily-interpolated).
- **Frequency-aware financial health:** `compute_financial_health(cache, freq="D", sector="")` accepts both frequency and sector parameters. Altman Z components x3 (EBIT/TA) and x5 (Revenue/TA) use annualized flow values at Q/A/S. Liquidity runway uses frequency-appropriate monthly burn rate.
- **Sector-aware FH baseline floors (2026-05-09):** Technology companies with >30% gross margin + positive FCF get profitability floor=40, solvency floor=35. Financial services get solvency floor=40, liquidity floor=35. Communication services get profitability floor=35, solvency floor=30. Prevents Apple from scoring 28/100 "Weak".
- **Sector-aware MC survival thresholds (2026-05-09):** `SECTOR_SURVIVAL_OVERRIDES` in `monte_carlo.py` adjusts survival triggers per sector (technology: current_ratio<0.7 instead of <1.0, D/E<5.0 instead of <3.0). Wired through `stage5_forward.py` via `get_sector_aware_thresholds()`.
- **MC burn-out scale guard (2026-05-09):** `get_regime_distributions()` filters to return-scale values only (`|value|<1.0`). `run_monte_carlo()` rejects burn-out distributions with `|mean|>1.0` or `std>1.0`. Prevents billion-scale financial values from contaminating MC path generation.

Every variable produced by the 10 Layer 2 modules in `operator1/analysis/` + the Financial Health module in `operator1/models/`. These modules produce survival flags, regime classifications, protection assessments, adaptive calibration, and health scores that control how all downstream temporal models behave.

**v2 update (2026-05-01):** Added 10 expert-method enhancements from PR #2: gradient velocity early warning, bootstrap uncertainty bands, semi-Markov duration modeling, EWM percentile rank, entropy-based hierarchy weights, ensemble distress prediction (Altman Z + Ohlson O + Zmijewski + Merton PD), CVaR-weighted composite, soft regime transition, reverse stress testing.

---

## 2.1 Survival Mode Detection

**File:** `operator1/analysis/survival_mode.py` (~822 lines)
**Pipeline step:** Step 5
**Entry points:** `compute_company_survival_flag()`, `compute_country_survival_flag()`, `compute_country_protected_flag()`, `compute_survival_probability()`, `compute_cox_survival_score()`, `compute_survival_velocity()`, `compute_survival_uncertainty()`

### Cache Variables

| # | Variable | Formula / Logic | Type | Consumers |
|---|----------|----------------|------|-----------|
| 1 | `company_survival_mode_flag` | OR of 7+ conditions: `current_ratio < 1.0`, `debt_to_equity_abs > 3.0`, `fcf_yield < 0`, `drawdown_252d < -0.40`, `conflict_intensity_score > 0.7`, `sanctions_flag == 1`, `inst_flow_momentum < -0.15`, `inst_crowding_score > 0.8 AND illiquidity > P90`. **NEW (P6):** suppressed when `cash/market_cap > 5%` | Integer (0/1) | Hierarchy weights, survival timeline, USS controller, Monte Carlo, all temporal models |
| 2 | `country_survival_mode_flag` | Macro thresholds from `config/country_protection_rules.yml`: `credit_spread`, `fx_volatility`, `unemployment_rate`, `yield_curve_slope` | Integer (0/1) | Hierarchy weights, survival timeline |
| 3 | `country_protected_flag` | Any: sector in `strategic_sectors`, `market_cap > 0.001 * GDP`, emergency rate cut detected. Also set by fuzzy protection (`fuzzy_protection_degree >= 0.5`) | Integer (0/1) | Hierarchy weights, survival timeline |
| 4 | `survival_probability` | Sigmoid of distance to each trigger threshold (6 components weighted by severity). Blended: `0.4 * sigmoid + 0.6 * cox` | Continuous (0-1) | Profile, report, Monte Carlo, MF fusion |
| 5 | `cox_survival_score` | `CoxPHFitter.fit(df, duration, event_observed)` with covariates: current_ratio, debt_to_equity_abs, fcf_yield, drawdown_252d. Learns hazard ratios from own distress episodes (via `lifelines`) | Continuous (0-1) | Survival probability blending |

### Gradient Early Warning Variables (from `compute_survival_velocity`) -- NEW PR #2

| # | Variable | Formula / Logic | Type | Consumers |
|---|----------|----------------|------|-----------|
| 6 | `survival_velocity_flag` | `1` when any trigger variable is deteriorating toward its threshold faster than 30% of threshold distance per 21 days (Duffie, Saita & Wang 2007) | Integer (0/1) | USS early warning, extra vars |
| 7 | `survival_deterioration_rate` | Worst (most negative) normalized deterioration rate across all trigger variables: `delta / abs(threshold)` over 21-day window | Continuous | Extra vars, conformal band widening |

### Uncertainty Band Variables (from `compute_survival_uncertainty`) -- NEW PR #2

| # | Variable | Formula / Logic | Type | Consumers |
|---|----------|----------------|------|-----------|
| 8 | `survival_probability_p10` | 10th percentile from 100 bootstrap resamples of trigger variable noise | Continuous (0-1) | Profile, report |
| 9 | `survival_probability_p90` | 90th percentile from bootstrap | Continuous (0-1) | Profile, report |
| 10 | `survival_uncertainty` | `p90 - p10` spread -- wider = more uncertain about survival status | Continuous | Conformal band scaling, extra vars |

**10 cache columns** (was 5)

---

## 2.2 Hierarchy Weights

**File:** `operator1/analysis/hierarchy_weights.py` (~400 lines)
**Pipeline step:** Step 5
**Entry point:** `compute_hierarchy_weights(cache, config=None, forward_pass_errors=None)`

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

**Entropy-based blending (NEW PR #2):** When `forward_pass_errors` dict is provided (from prior pipeline run), applies `_entropy_blend_weights()` (Jaynes 1957 maximum entropy) to shift weight toward tiers with higher prediction uncertainty. Shannon entropy of per-tier error distributions drives allocation. Capped at +/-10% shift from regime defaults. Falls back to pure regime defaults when no errors available.

**6 cache columns**

---

## 2.3 Survival Timeline

**File:** `operator1/analysis/survival_timeline.py` (~780 lines)
**Pipeline step:** Step 5.5
**Entry points:** `compute_survival_timeline()`, `compute_enriched_survival_timeline()`

### Base Timeline Variables (from `compute_survival_timeline`)

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `survival_mode` | 6-mode classification from 3 flags: normal, company_only, country_protected, country_exposed, both_protected, both_unprotected | Categorical | Walk-forward (mode-conditioned scoring), USS |
| 2 | `survival_mode_code` | Integer code for survival_mode (0-5) | Integer | Walk-forward |
| 3 | `switch_point` | 1 on days where mode changes, 0 otherwise | Binary (0/1) | Walk-forward (retrain trigger), profile |
| 4 | `days_in_mode` | Running counter since last switch (resets at each switch_point) | Integer | Profile, stability score, semi-Markov |
| 5 | `stability_score_21d` | Rolling 21-day fraction of same mode (1.0 = fully stable) | Continuous (0-1) | Extra vars, regime shift predictor, adaptive model params |

### Semi-Markov Duration Variables (from `_compute_semi_markov_exit`) -- NEW PR #2

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 6 | `expected_remaining_days_in_mode` | Weibull residual life: `scale * Gamma(1 + 1/shape) - days_in_mode` (Barbu & Limnios 2008). Shape <1 = distress trap, >1 = recovery likely. Falls back to geometric (Markov) for <3 dwell episodes. | Integer | Profile, report |
| 7 | `mode_exit_probability_21d` | Duration-aware exit probability: `1 - S(d+21)/S(d)` from Weibull survival function fitted on per-mode dwell times via `scipy.stats.weibull_min` | Continuous (0-1) | Extra vars, prediction aggregator |

### Enriched Timeline Variables (from `compute_enriched_survival_timeline`)

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 8 | `regime_state` | 11 combined states from (survival_mode x market_regime): stable_growth, elevated_risk, market_stress, company_distress_mild/severe, country_crisis_mild/severe, protected_stress/crisis, crisis, extreme_crisis | Categorical | Profile, report |
| 9 | `survival_intensity` | Continuous 0-1 metric blending rule-based intensity + HMM regime confidence | Continuous (0-1) | Extra vars, prediction aggregator |
| 10 | `regime_confidence` | HMM posterior probability of the assigned regime state | Continuous (0-1) | Extra vars |
| 11 | `regime_switch` | 1 on enriched regime transition days | Binary (0/1) | Profile |
| 12 | `regime_transition_prob` | Probability of transitioning to a different regime on next day | Continuous (0-1) | Conformal widening, prediction aggregator |
| 13 | `market_regime` | HMM-derived market regime label (bull/bear/high_vol/low_vol/unknown) | Categorical | Combined state mapping |

**13 cache columns** (was 11)

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

**File:** `operator1/models/financial_health.py` (~1,260 lines)
**Pipeline step:** Step 5d
**Entry point:** `compute_financial_health(cache, hierarchy_weights)`

**EWM percentile rank (NEW PR #2):** `_normalize_series()` now accepts optional `halflife` parameter for exponentially weighted percentile ranking. When set, recent observations are weighted more heavily, preventing long healthy periods from diluting recent deterioration. Typical values: 63 (quarterly filer), 126 (semi-annual), 252 (annual).

### Cache Variables

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 1 | `fh_liquidity_score` | Expanding (or EWM) percentile rank: cash_ratio, current_ratio, free_cash_flow | Continuous (0-100) | Composite score (T1 weight) |
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

### Ensemble Distress Prediction (NEW PR #2)

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 17 | `fh_ensemble_distress_prob` | Stacked ensemble of 4 models: Altman Z (logistic transform), Ohlson O-Score (Ohlson 1980, 9 coefficients), Zmijewski (probit, 3 covariates), Merton PD (structural). Equal-weighted average, clipped (0-1). | Continuous (0-1) | Profile, report, USS early warning, HF risk decomposition |
| 18 | `fh_ensemble_distress_label` | safe (<0.15) / watch (0.15-0.30) / warning (0.30-0.50) / critical (>0.50) | Categorical | Profile, report |

### CVaR-Weighted Composite (NEW PR #2)

| # | Variable | Formula | Type | Consumers |
|---|----------|---------|------|-----------|
| 19 | `fh_cvar_composite` | `E[tier_scores | tier_scores < VaR_30%]` -- average of worst 30% of tier scores (Rockafellar & Uryasev 2000). More sensitive to weakest tier than weighted average. | Continuous (0-100) | Profile |

**19 cache columns** (was 16)

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

**File:** `operator1/analysis/survival_regime_controller.py` (~780 lines)
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

### Soft Regime Transition (NEW PR #2)

| # | Variable | Description | Consumers |
|---|----------|-------------|-----------|
| 8 | `get_soft_transition_config()` | Exponentially interpolates between old and new `ModelConfig` over a transition halflife (default 5 days). `lam = 1 - exp(-days_since_switch * ln(2) / halflife)`. Numeric fields interpolated, categorical switch immediately. Prevents discontinuities when regimes change. | Forward pass, forecasting, MC (available but not auto-wired yet -- needs controller state tracking for `previous_regime`) |

**1 cache column + 7 controller parameters** (was 6)

---

## 2.13 Scenario Engine (USS)

**File:** `operator1/analysis/scenario_engine.py` (~532 lines)
**Pipeline step:** Step 6-USS (Stage 7.1)
**Entry points:** `run_scenario_engine(cache, regime, n_paths, horizon_days)`, `compute_reverse_stress_test(cache, thresholds)`

### Forward Scenarios (from `run_scenario_engine`)

| # | Output | Description | Per-Scenario Variables |
|---|--------|-------------|----------------------|
| 1 | `orderly` | Management restructures: cash burn -30%, debt renegotiated, revenue -15% | `cash_runway_days`, `survival_prob_90d`, `survival_prob_252d`, `terminal_equity`, `terminal_revenue`, `terminal_cash`, `median_return`, `p5_return`, `p95_return`, `median_max_drawdown` |
| 2 | `muddle_through` | Status quo: full historical return distribution | Same 10 variables |
| 3 | `catastrophic` | Fire sale: assets -40%, all debt called, revenue -40% | Same 10 variables |

### Reverse Stress Test (from `compute_reverse_stress_test`) -- NEW PR #2

| # | Output | Description | Type | Profile Key |
|---|--------|-------------|------|-------------|
| 4 | `revenue_shock_pct` | Minimum revenue drop (%) that triggers survival mode | Float (%) | `scenario_analysis.reverse_stress.revenue_shock_pct` |
| 5 | `margin_shock_pp` | Minimum margin compression (pp) | Float (pp) | `scenario_analysis.reverse_stress.margin_shock_pp` |
| 6 | `rate_shock_bps` | Minimum interest rate increase (bps) | Float (bps) | `scenario_analysis.reverse_stress.rate_shock_bps` |
| 7 | `triggered_variable` | Which survival trigger fires first under the minimum shock | Categorical | `scenario_analysis.reverse_stress.triggered_variable` |
| 8 | `shock_magnitude` | L2 norm of the shock vector (lower = more fragile) | Float | Internal |

Uses `scipy.optimize.minimize` with SLSQP to find the minimum-norm shock that triggers any survival threshold. Basel III reverse stress testing concept.

**0 cache columns** (result objects stored in profile `scenario_analysis`)

---

## Layer 2 UPDATED Grand Total

| Module | Cache Columns | Result-Only Fields | Total | Status |
|--------|--------------|-------------------|-------|--------|
| 2.1 Survival Mode | **10** (was 5) | 0 | **10** | **ENHANCED PR #2** (+5: velocity flag/rate, P10/P90, uncertainty) |
| 2.2 Hierarchy Weights | 6 | 0 | 6 | **ENHANCED PR #2** (entropy blending via `forward_pass_errors` param) |
| 2.3 Survival Timeline | **13** (was 11) | 0 | **13** | **ENHANCED PR #2** (+2: semi-Markov expected days, exit probability) |
| 2.4 Fuzzy Protection | 7 | 0 | 7 | |
| 2.5 Ethical Filters | 0 | 4 verdicts | 0 | |
| 2.6 Economic Planes | 0 | 2 fields | 0 | |
| 2.7 Vanity | 15 | 0 | 15 | |
| 2.8 Financial Health | **19** (was 16) | 0 | **19** | **ENHANCED PR #2** (+3: ensemble distress prob/label, CVaR composite) |
| 2.9 Adaptive Thresholds | 0 | 6 fields | 0 | |
| 2.10 Adaptive Model Params | 0 | 12 fields | 0 | |
| 2.11 Adaptive Windows | 0 | 11 fields | 0 | |
| 2.12 USS Controller | 1 | **7** (was 6) | 1 | **ENHANCED PR #2** (+1: soft transition config) |
| 2.13 Scenario Engine | 0 | **35** (was 30) | 0 | **ENHANCED PR #2** (+5: reverse stress test results) |
| **Total** | **71** | **77** | **71 cache + 77 result** | |

These 71 cache columns are added on top of the ~428 from Layer 1 (v3), bringing the total cache to **~499 columns** before temporal models run.

**PR #2 delta:** +10 new cache columns (5 survival mode + 2 survival timeline + 3 financial health), +6 new result fields (1 USS controller + 5 reverse stress), **+16 total new variables** vs v1.

# HF Analysis -- Full Implementation Plan

*Created: 2026-05-06. Combines upgrade plan + GitHub research + debug scan of all connection points.*

---

## Part 1: Complete File Map (What We Edit, Where, Why)

### Files Modified (8 files)

| File | Lines | Changes | New Methods |
|------|-------|---------|-------------|
| [`operator1/hedge_fund/types.py`](operator1/hedge_fund/types.py) | 479 | Add ~15 new fields to existing dataclasses + 3 new dataclasses | EVAResult, SOTPResult, CreditTermStructure, MoatResult, GovernanceResult |
| [`operator1/hedge_fund/engine.py`](operator1/hedge_fund/engine.py) | 1480 | Add 7 new inline compute functions + wire into orchestrator | EVA, SOTP, reverse DCF upgrade, CDaR, risk budget, SGR, adaptive scorecard |
| [`operator1/hedge_fund/advanced_methods.py`](operator1/hedge_fund/advanced_methods.py) | 965 | Add 10 new methods in P1/P2/P3 sections | Merton term structure, credit migration, options PD, capital cycle upgrade, moat, factor exposure, equity duration, commodity beta, FX risk, governance |
| [`operator1/hedge_fund/accruals_forensics.py`](operator1/hedge_fund/accruals_forensics.py) | 202 | Add 2 new components + 1 new function | Dechow F-Score, real activities manipulation, accounting conservatism |
| [`operator1/hedge_fund/earnings_smoothing.py`](operator1/hedge_fund/earnings_smoothing.py) | 228 | Add M5-Score variant + probability conversion | Beneish M5, probit probability |
| [`operator1/hedge_fund/multi_freq_metrics.py`](operator1/hedge_fund/multi_freq_metrics.py) | 297 | Add 3 new MF functions + coherence test utility | FCF quality MF, accruals MF, growth quality MF, coherence test |
| [`operator1/hedge_fund/fusion.py`](operator1/hedge_fund/fusion.py) | 653 | Add 3 new fusion methods (9-11) | HRP signal combination, Brier calibration, adaptive weights |
| [`config/hedge_fund_weights.yml`](config/hedge_fund_weights.yml) | 162 | Add weight config for new methods | Weights for all 30 new methods |

### Files NOT Modified (outlet points -- backward compatible)

| File | Why No Changes Needed |
|------|----------------------|
| [`operator1/stages/stage7_integration.py`](operator1/stages/stage7_integration.py) | `run_7_5_hedge_fund()` passes all inputs via kwargs to `run_hedge_fund_analysis()`. New methods consume the same inputs. |
| [`operator1/pipeline_state.py`](operator1/pipeline_state.py) | `hf_result: Any` stores the entire HedgeFundResult. New fields are within the same object. |
| [`operator1/report/report_generator.py`](operator1/report/report_generator.py) | Report sections 23-29 read from `profile["hedge_fund"]` dict. New fields automatically appear via `to_profile_dict()`. Report enhancement is a separate task. |
| [`dashboard.py`](dashboard.py) | Dashboard reads `profile["hedge_fund"]` dict generically. New cards can be added later. |
| [`operator1/report/profile_builder.py`](operator1/report/profile_builder.py) | Profile builder stores `hf_result.to_profile_dict()` which auto-includes new fields. |
| [`operator1/report/profile_schema.py`](operator1/report/profile_schema.py) | Schema validates `hedge_fund` key exists with `available` flag. New sub-keys are optional. |

---

## Part 2: Complete Input/Output Connection Map

### What `run_hedge_fund_analysis()` Receives (18 parameters)

```
INPUTS FROM PIPELINE STATE:
  income_df          <- state.income_df       [raw quarterly income statement, 8-24 rows]
  balance_df         <- state.balance_df      [raw quarterly balance sheet]
  cashflow_df        <- state.cashflow_df     [raw quarterly cash flow]
  cache              <- state.cache           [daily cache, ~500 columns x ~504 rows]
  target_profile     <- state.target_profile  [company identity dict]
  forecast_result    <- state.forecast_result [ForecastResult from Stage 4]
  mc_result          <- state.mc_result       [MonteCarloResult from Stage 5.4]
  scenario_result    <- state.scenario_result [ScenarioEngineResult from Stage 7.1]
  multi_frequency_result <- state.multi_frequency_result [FusedMultiFreqResult from Stage 7.4.6]
  signal_ic_result   <- state.signal_ic_result [SignalICResult from Step 5j.5]
  filing_calendar_result <- state.filing_calendar_result [FilingCalendarResult from Step 4c]
  fh_result          <- state.fh_result       [FinancialHealthResult from Step 5d]
  peer_ranking_result <- state.peer_ranking_result [dict from Step 5h]
  sentiment_result   <- state.sentiment_result [dict from Step 5i]
  survival_controller <- state.survival_controller [SurvivalRegimeController from Step 5-USS]
  linked_caches      <- state.linked_caches   [dict of entity_id -> DataFrame]
  macro_data         <- state.macro_data      [dict of indicator -> Series]
  income/balance/cashflow_freq_groups <- state.*_freq_groups [per-frequency raw statement groups]
```

### What Each New Method Needs from These Inputs

| New Method | Primary Data | Pipeline Results Used | Cache Columns Used |
|------------|-------------|----------------------|-------------------|
| **1.1 Merton Term Structure** | cache | - | close, volatility_21d, total_debt/total_debt_asof, market_cap, shares_outstanding |
| **1.2 Credit Migration** | cache | fh_result | fh_altman_z_zone, survival_regime |
| **1.3 Options-Implied PD** | cache | - | options chain data via yfinance (live fetch) |
| **2.1 Dechow F-Score** | income_df, balance_df, cashflow_df | - | accruals, receivables, inventory, soft_asset_ratio |
| **2.2 Real Activities** | income_df, balance_df, cashflow_df | - | revenue, COGS, inventory, R&D, SGA, OCF |
| **2.3 Beneish M5** | - | fh_result | fh_beneish_m_score (already computed) |
| **3.1 Stochastic DCF** | cashflow_df, balance_df | mc_result | close, revenue_growth_yoy |
| **3.2 EVA** | income_df, balance_df | - | close (for market_cap) |
| **3.3 SOTP** | - | - | seg_result (from state), close |
| **3.4 Reverse DCF** | cashflow_df | - | close, revenue_growth_yoy |
| **4.1 Component CVaR** | cache | - | return_1d + key risk factor columns |
| **4.2 CDaR** | - | mc_result | - (uses MC terminal_values) |
| **4.3 Risk Budget** | - | mc_result, survival_controller | - |
| **5.1 Capital Cycle** | income_df, cashflow_df | - | - |
| **5.2 Moat** | income_df | peer_ranking_result | gross_margin, operating_margin, revenue |
| **5.3 SGR** | income_df, cashflow_df | - | - |
| **6.1-6.3 MF variants** | *_freq_groups | - | - |
| **6.4 Coherence** | - | - | - (utility function) |
| **7.1 Factor Exposure** | cache | - | return_1d, yield_curve_10y2y, usd_momentum_21d, sector_relative_strength |
| **7.2 Equity Duration** | cache | - | return_1d, yield_curve_10y2y |
| **7.3 Commodity Beta** | cache | - | return_1d (+ live commodity ETF fetch) |
| **7.4 FX Risk** | - | - | seg_result.geo_segments, macro_data.currency |
| **8.1 Governance** | - | - | inst_insider_signal, inst_crowding_risk, vanity_score |
| **8.2 Capital Allocation** | cashflow_df | - | close, pe_ratio_calc |
| **8.3 Conservatism** | income_df | - | return_1d |
| **9.1 HRP** | - | all HF metric scores | - |
| **9.2 Brier** | - | prediction_log_summary | - |
| **9.3 Adaptive Weights** | cache | - | return_1d (forward returns for IC) |

### What Goes OUT from HF -> Profile -> Report -> Dashboard

```
HedgeFundResult.to_profile_dict()
  |
  +-> profile["hedge_fund"] = {
  |     "available": True,
  |     "fcf_quality": {...},           <-- Report Section 23
  |     "accruals_forensic": {...},     <-- Report Section 23
  |     "smoothing": {...},             <-- Report Section 23
  |     "dividend_burn": {...},         <-- Report Section 24
  |     "return_spread": {...},         <-- Report Section 24
  |     "operating_leverage": {...},    <-- Report Section 24
  |     "obs_risk": {...},              <-- Report Section 25
  |     "asset_quality": {...},         <-- Report Section 25
  |     "leverage_stress": {...},       <-- Report Section 25
  |     "momentum": {...},              <-- Report Section 26
  |     "growth_quality": {...},        <-- Report Section 26
  |     "earnings_surprise": {...},     <-- Report Section 26
  |     "dcf": {...},                   <-- Report Section 27
  |     "valuation_quality": {...},     <-- Report Section 27
  |     "peg_composite": {...},         <-- Report Section 27
  |     "scorecard": {...},             <-- Report Section 28 + Dashboard scorecard card
  |     "position": {...},              <-- Report Section 29 + Dashboard signal card
  |     "advanced": {...},              <-- Report Section 19 (Premium)
  |     "fusion": {...},                <-- Dashboard fusion card
  |     "data_readiness": {...},
  |     === NEW FIELDS (auto-included via to_profile_dict) ===
  |     "eva": {...},                   <-- NEW: EVA decomposition
  |     "sotp": {...},                  <-- NEW: Sum-of-parts valuation
  |     "credit_term_structure": {...}, <-- NEW: Merton PD term structure
  |     "moat": {...},                  <-- NEW: Competitive moat score
  |     "governance": {...},            <-- NEW: Governance quality
  |     "risk_attribution": {...},      <-- NEW: Component CVaR + CDaR
  |   }
  |
  +-> Dashboard reads: profile["hedge_fund"]["scorecard"], ["position"], ["fusion"]
  |   Dashboard cards for DuPont + Kelly already exist at dashboard.py:504-518
  |
  +-> Report sections 23-29 read from profile["hedge_fund"] sub-keys
  |   New fields in existing sub-keys are auto-rendered
  |
  +-> LLM prompt in llm_base.py references hedge_fund.position.signal for recommendations
```

---

## Part 3: Implementation Checklist (Exact Edit Locations)

### Phase 1: Forensics Upgrade (3 methods)

- [ ] **2.1 Dechow F-Score** -- [`accruals_forensics.py`](operator1/hedge_fund/accruals_forensics.py)
  - Add `compute_dechow_f_score()` function after line 202
  - 8 probit coefficients from Dechow et al. 2011 Table 7
  - Inputs: income_df, balance_df, cashflow_df (already passed to `compute_accruals_forensics`)
  - Output: add `dechow_f_score: float`, `dechow_f_probability: float`, `dechow_f_label: str` to `AccrualsForensicResult`
  - Wire: call from `compute_accruals_forensics()` after existing 5 components, add as 6th component
  - Edit `types.py`: add 3 fields to `AccrualsForensicResult` dataclass at line ~74

- [ ] **2.2 Real Activities Manipulation** -- [`accruals_forensics.py`](operator1/hedge_fund/accruals_forensics.py)
  - Add `compute_real_activities_manipulation()` function
  - 3 OLS regressions via sklearn LinearRegression
  - Inputs: income_df, balance_df, cashflow_df
  - Output: add `real_manipulation_score: float`, `abnormal_cfo: float`, `abnormal_production: float`, `abnormal_discretionary: float`, `manipulation_type: str` to `AccrualsForensicResult`
  - Wire: call from `compute_accruals_forensics()`, feed into red_flag_score composite
  - Edit `types.py`: add 5 fields to `AccrualsForensicResult`

- [ ] **2.3 Beneish M5** -- [`earnings_smoothing.py`](operator1/hedge_fund/earnings_smoothing.py)
  - Add M5 computation within existing `compute_earnings_smoothing()` after Beneish component
  - Simple: 5 coefficients instead of 8, `P = 1/(1+exp(-M))`
  - Input: already available from fh_result
  - Output: add `beneish_m5_score: float`, `beneish_probability: float`, `beneish_m5_m8_agreement: bool` to `SmoothingResult`
  - Edit `types.py`: add 3 fields to `SmoothingResult` at line ~92

### Phase 2: Valuation Engine Upgrade (4 methods)

- [ ] **3.1 Stochastic DCF** -- [`engine.py`](operator1/hedge_fund/engine.py) `_compute_dcf()`
  - Modify existing function at line 659
  - Replace `np.random.normal(wacc_mean, wacc_std, n)` with `np.random.multivariate_normal` for correlated growth/WACC
  - Add `stochastic_wacc_mean`, `stochastic_wacc_vol`, `growth_wacc_correlation` to `DCFResult`
  - Edit `types.py`: add 3 fields to `DCFResult` at line ~296

- [ ] **3.2 EVA Decomposition** -- [`engine.py`](operator1/hedge_fund/engine.py)
  - Add `_compute_eva()` function after `_compute_peg()` (~line 965)
  - Formula: `NOPAT - WACC * IC`, 15-20 lines
  - Inputs: income_df, balance_df (already available in orchestrator scope)
  - Output: new `EVAResult` dataclass in types.py
  - Wire: call in orchestrator between Tier 5 and Scorecard, add `hf.eva = _compute_eva(...)`
  - Edit `types.py`: add `EVAResult` dataclass, add `eva: EVAResult` to `HedgeFundResult`

- [ ] **3.3 SOTP Valuation** -- [`engine.py`](operator1/hedge_fund/engine.py)
  - Add `_compute_sotp()` function
  - Inputs: seg_result (pass through from `run_hedge_fund_analysis` -- need to add `seg_result` param), cache
  - Guard: only runs when `seg_result` has 2+ segments
  - Output: new `SOTPResult` dataclass
  - Wire: add `seg_result` parameter to `run_hedge_fund_analysis()` signature
  - Edit [`stage7_integration.py`](operator1/stages/stage7_integration.py) line 457: add `seg_result=state.seg_result`
  - Edit `types.py`: add `SOTPResult`, add `sotp: SOTPResult` to `HedgeFundResult`

- [ ] **3.4 Reverse DCF Upgrade** -- [`engine.py`](operator1/hedge_fund/engine.py) `_compute_dcf()`
  - Replace heuristic `growth_gap` with `scipy.optimize.brentq` root-finding
  - Add `expectations_gap_pct`, `expectations_label` to `DCFResult`
  - Edit `types.py`: add 2 fields to `DCFResult`

### Phase 3: Credit Analysis (3 methods)

- [ ] **1.1 Merton Term Structure** -- [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py)
  - Add `compute_merton_term_structure()` after existing `compute_altman_z_double_prime()` (~line 530)
  - Loop over T = [1, 2, 3, 5] with parameterized d1/d2 (pattern from k-bargaoui)
  - Input: cache (close, volatility_21d, total_debt, market_cap)
  - Output: add `merton_term_structure: dict` to `AdvancedMethodsResult`

- [ ] **1.2 Credit Migration Matrix** -- [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py)
  - Add `compute_credit_migration()` function
  - Build 4x4 transition matrix from `fh_altman_z_zone` history (8Q transitions)
  - Input: cache (fh_altman_z_zone, survival_regime history)
  - Output: add `credit_migration: dict` to `AdvancedMethodsResult`

- [ ] **1.3 Options-Implied PD** -- [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py)
  - Add `compute_options_implied_pd()` function
  - Extract deep OTM put prices from existing options chain data
  - Input: cache (needs close), + live yfinance options fetch
  - Output: add `options_implied_pd: dict` to `AdvancedMethodsResult`

### Phase 4: Risk Management (3 methods)

- [ ] **4.1 Component CVaR** -- [`engine.py`](operator1/hedge_fund/engine.py)
  - Add `_compute_component_cvar()` function
  - Historical simulation: identify worst 5% days, compute per-factor contribution
  - Input: cache (return_1d + risk factor columns)
  - Output: add to `HedgeFundResult` as `risk_attribution: dict`

- [ ] **4.2 CDaR** -- [`engine.py`](operator1/hedge_fund/engine.py) within `_compute_position_signal()`
  - Add CDaR computation using MC paths (pattern from skfolio: `cvar(drawdowns, 0.95)`)
  - Input: mc_result.terminal_values
  - Output: add `cdar_95: float`, `cdar_adjusted_kelly: float` to `PositionSignalResult`
  - Edit `types.py`: add 2 fields to `PositionSignalResult` at line ~390

- [ ] **4.3 Risk Budget** -- [`engine.py`](operator1/hedge_fund/engine.py)
  - Add `_compute_risk_budget()` function
  - Per-regime risk allocation from Component CVaR output
  - Input: component_cvar result, survival_controller
  - Output: add `risk_budget: dict` to `HedgeFundResult`

### Phase 5: Capital Cycle & Competitive (3 methods)

- [ ] **5.1 Capital Cycle Upgrade** -- [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py) `compute_capital_cycle()`
  - Replace existing stub at line 631 with full 4-phase classification
  - CapEx/Revenue slope + Gross Margin slope -> phase classification
  - Input: income_df, cashflow_df (already passed)
  - Output: upgrade existing `capital_cycle: dict` with `phase`, `age_quarters`, `confidence`, `next_phase_prob`

- [ ] **5.2 Moat Quantification** -- [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py)
  - Add `compute_moat_score()` function
  - 4 components: pricing power, switching costs, cost advantage, network effects
  - Input: income_df, cache, peer_ranking_result
  - Output: add `moat: dict` to `AdvancedMethodsResult`

- [ ] **5.3 SGR** -- [`engine.py`](operator1/hedge_fund/engine.py)
  - Add `_compute_sgr()` function (~10 lines)
  - `SGR = ROE * (1 - payout_ratio)`
  - Input: income_df, cashflow_df
  - Output: add `sgr: float`, `growth_gap_sgr: float` to `GrowthQualityResult`
  - Edit `types.py`: add 2 fields to `GrowthQualityResult`

### Phase 6: Multi-Freq Expansion (4 methods)

- [ ] **6.1 FCF Quality MF** -- [`multi_freq_metrics.py`](operator1/hedge_fund/multi_freq_metrics.py)
  - Add `compute_fcf_quality_multi_freq()` function (pattern from existing momentum MF)
  - Merge: weighted average (Q=0.55, A=0.45) with divergence penalty
  - Wire: in `engine.py` orchestrator, before existing FCF quality call

- [ ] **6.2 Accruals MF** -- [`multi_freq_metrics.py`](operator1/hedge_fund/multi_freq_metrics.py)
  - Add `compute_accruals_multi_freq()` function
  - Jones model from annual (more power), Sloan/CCE from quarterly
  - Wire: in `engine.py` orchestrator

- [ ] **6.3 Growth Quality MF** -- [`multi_freq_metrics.py`](operator1/hedge_fund/multi_freq_metrics.py)
  - Add `compute_growth_quality_multi_freq()` function
  - Organic fraction from annual, incremental ROIC from quarterly

- [ ] **6.4 Coherence Test** -- [`multi_freq_metrics.py`](operator1/hedge_fund/multi_freq_metrics.py)
  - Add `test_frequency_coherence()` utility function
  - Spearman rank correlation of component scores across frequencies
  - Called before every MF merge to decide merge vs single-freq

### Phase 7: Macro Sensitivity (4 methods)

- [ ] **7.1 Factor Exposure** -- [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py)
  - Add `compute_factor_exposure()` in P3 section
  - Rolling 63d OLS: return_1d ~ yield_curve + usd_momentum + sector_return + vix
  - Input: cache (all factor columns from cross_asset_signals)
  - Output: add `factor_exposure: dict` to `AdvancedMethodsResult`

- [ ] **7.2 Equity Duration** -- [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py)
  - Add `compute_equity_duration()` in P3 section
  - `duration_beta = cov(return, d_yield) / var(d_yield)` over 252d
  - Input: cache (return_1d, yield_curve_10y2y)

- [ ] **7.3 Commodity Beta** -- [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py)
  - Add `compute_commodity_exposure()` in P3 section
  - Rolling OLS on commodity ETF returns (USO, GLD via yfinance)
  - Input: cache (return_1d) + live fetch

- [ ] **7.4 FX Risk** -- [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py)
  - Add `compute_fx_translation_risk()` in P3 section
  - From geo_segments in seg_result + currency data
  - Input: seg_result.geo_segments, macro_data.currency

### Phase 8: Governance (3 methods)

- [ ] **8.1 Governance Score** -- [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py)
  - Add `compute_governance_score()` function
  - Weighted: insider alignment 40% + ownership concentration 30% + capital discipline 30%
  - Input: cache (inst_insider_signal, inst_crowding_risk, vanity_score)
  - Output: add `governance: dict` to `AdvancedMethodsResult`

- [ ] **8.2 Capital Allocation Quality** -- [`advanced_methods.py`](operator1/hedge_fund/advanced_methods.py)
  - Add `compute_capital_allocation_quality()` function
  - Buyback timing + dividend consistency + reinvestment spread
  - Input: cashflow_df, cache (close, pe_ratio_calc)

- [ ] **8.3 Conservatism Index** -- [`accruals_forensics.py`](operator1/hedge_fund/accruals_forensics.py)
  - Add `compute_accounting_conservatism()` function
  - Basu asymmetric timeliness from returns vs earnings changes
  - Input: cache (return_1d), income_df

### Phase 9: Fusion Upgrade (3 methods)

- [ ] **9.1 HRP Signal Combination** -- [`fusion.py`](operator1/hedge_fund/fusion.py)
  - Add `compute_hrp_weights()` function after existing 8 methods
  - 3 functions: cluster, quasi-diag, recursive bisect (40 lines from Lopez de Prado)
  - Input: correlation matrix of all HF metric scores
  - Output: `hrp_weights: dict` in `FusionResult`
  - Edit `types.py` FusionResult: add `hrp_weights: dict`

- [ ] **9.2 Brier Calibration** -- [`fusion.py`](operator1/hedge_fund/fusion.py)
  - Add `compute_brier_calibration()` function
  - Track conviction vs actual outcome from prediction_log_summary
  - Input: prediction_log_summary (passed to run_fusion)
  - Output: `calibrated_conviction: float`, `overconfidence_flag: bool` in `FusionResult`

- [ ] **9.3 Adaptive Scorecard Weights** -- [`engine.py`](operator1/hedge_fund/engine.py) `_build_scorecard()`
  - Add rolling IC computation before fixed weight application at line 970
  - For each tier score, compute Spearman IC vs forward 21d return
  - Input: cache (return_1d shifted for forward returns)
  - Output: `adaptive_tier_weights: dict` in `ThesisScorecard`

---

## Part 4: Wiring Changes to Orchestrator

### Changes to `run_hedge_fund_analysis()` Signature

```python
# ADD these parameters:
def run_hedge_fund_analysis(
    ...existing 18 params...
    seg_result: dict | None = None,           # NEW: for SOTP valuation
    prediction_log_summary: dict | None = None,  # NEW: for Brier calibration
    options_signal_result: Any = None,        # NEW: for options-implied PD
) -> HedgeFundResult:
```

### Changes to `run_7_5_hedge_fund()` in stage7_integration.py

```python
# ADD to the kwargs at line 457:
state.hf_result = run_hedge_fund_analysis(
    ...existing kwargs...
    seg_result=state.seg_result,                    # NEW
    prediction_log_summary=state.prediction_log_summary,  # NEW
    options_signal_result=state.options_signal_result,     # NEW
)
```

### Orchestrator Execution Order (engine.py)

```
EXISTING:
  Tier 1: fcf_quality, accruals_forensic, smoothing
  Tier 2: dividend_burn, return_spread, operating_leverage
  Tier 3: obs_risk, asset_quality, leverage_stress
  Tier 4: momentum, growth_quality, earnings_surprise
  Tier 5: dcf, valuation_quality, peg_composite
  Scorecard + Position Signal
  Advanced Methods (15)
  Fusion (8)

NEW (interleaved):
  Tier 1: fcf_quality, accruals_forensic (+dechow, +real_activities, +conservatism), smoothing (+M5)
  Tier 2: dividend_burn, return_spread, operating_leverage
  Tier 3: obs_risk, asset_quality, leverage_stress
  Tier 4: momentum, growth_quality (+SGR), earnings_surprise
  Tier 5: dcf (+stochastic, +reverse_dcf_upgrade), valuation_quality, peg_composite
  NEW: eva, sotp (if segments exist)
  NEW: component_cvar, risk_budget
  Scorecard (+adaptive_weights)
  Position Signal (+CDaR)
  Advanced Methods (+merton_term, +credit_migration, +options_pd, +capital_cycle_upgrade, +moat, +factor_exposure, +equity_duration, +commodity, +fx, +governance, +capital_alloc)
  Fusion (+HRP, +Brier, existing 8)
```

---

## Part 5: Config Additions (hedge_fund_weights.yml)

```yaml
# NEW SECTIONS TO ADD:

# Domain 1: Credit
credit:
  merton_horizons: [1, 2, 3, 5]  # years
  migration_smoothing: 0.01       # Laplace smoothing for transition matrix
  options_pd_strike_floor: 0.70   # OTM put strike as fraction of current price

# Domain 2: Forensics
forensics:
  dechow_f_threshold: 0.50        # P(misstatement) above this = flag
  real_activities_weight: 0.15    # added to accruals composite
  conservatism_window: 252        # rolling days for Basu test

# Domain 3: Valuation
valuation_upgrade:
  growth_wacc_correlation: -0.30  # negative correlation for stochastic DCF
  sotp_corporate_overhead_pct: 0.05  # unallocated overhead as fraction of revenue
  reverse_dcf_growth_range: [-0.20, 0.50]  # search bounds

# Domain 4: Risk
risk:
  cvar_alpha: 0.05               # 5% tail for component CVaR
  cdar_alpha: 0.05               # 5% for CDaR
  cdar_max_drawdown_target: 0.20 # max acceptable drawdown for CDaR-adjusted Kelly

# Domain 5: Capital & Moat
capital_cycle:
  invest_capex_slope_threshold: 0.02
  harvest_margin_slope_threshold: 0.01
moat:
  pricing_power_weight: 0.30
  switching_costs_weight: 0.25
  cost_advantage_weight: 0.25
  network_effects_weight: 0.20

# Domain 8: Governance
governance:
  insider_alignment_weight: 0.40
  ownership_concentration_weight: 0.30
  capital_discipline_weight: 0.30

# Domain 9: Fusion
fusion_upgrade:
  hrp_method: 'single'           # scipy linkage method
  brier_min_predictions: 10      # minimum historical predictions for calibration
  adaptive_weights_window: 252   # rolling IC window for scorecard weights
```

---

## Part 6: Execution Summary

| Phase | Methods | Files Modified | New Lines (est.) | New Fields |
|-------|---------|---------------|-----------------|------------|
| 1. Forensics | 2.1 + 2.2 + 2.3 | accruals_forensics.py, earnings_smoothing.py, types.py | ~250 | +11 |
| 2. Valuation | 3.1 + 3.2 + 3.3 + 3.4 | engine.py, types.py, stage7_integration.py | ~350 | +15 |
| 3. Credit | 1.1 + 1.2 + 1.3 | advanced_methods.py | ~200 | +3 dicts |
| 4. Risk | 4.1 + 4.2 + 4.3 | engine.py, types.py | ~150 | +6 |
| 5. Capital | 5.1 + 5.2 + 5.3 | advanced_methods.py, engine.py, types.py | ~200 | +6 |
| 6. Multi-Freq | 6.1 + 6.2 + 6.3 + 6.4 | multi_freq_metrics.py, engine.py | ~250 | +4 |
| 7. Macro | 7.1 + 7.2 + 7.3 + 7.4 | advanced_methods.py | ~200 | +4 dicts |
| 8. Governance | 8.1 + 8.2 + 8.3 | advanced_methods.py, accruals_forensics.py | ~150 | +3 dicts |
| 9. Fusion | 9.1 + 9.2 + 9.3 | fusion.py, engine.py, types.py | ~180 | +5 |
| **Total** | **30 methods** | **8 files** | **~1,930 lines** | **~74 fields** |

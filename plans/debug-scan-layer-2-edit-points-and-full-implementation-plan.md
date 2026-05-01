# Debug Scan: Layer 2 Edit Points, Inputs, Outputs & Full Implementation Plan

*Systematic scan of every file affected by the Layer 2 analysis module enhancements*

---

## 1. Files That Will Be DIRECTLY EDITED

### 1a. `operator1/analysis/survival_mode.py` (664 lines)

**Enhancements:** 1A (gradient early warning), 1B (competing risks), 1C (Bayesian uncertainty)

**Edit point 1A -- after `compute_company_survival_flag()` at line 201:**
Add new function `compute_survival_velocity()` (~30 lines). Inputs: cache columns `current_ratio`, `debt_to_equity_abs`, `fcf_yield`, `drawdown_252d` + thresholds from `_COMPANY_THRESHOLDS`. Output: 2 new cache columns (`survival_velocity_flag`, `survival_deterioration_rate`).

**Edit point 1B -- after `compute_cox_survival_score()` at line ~380:**
Add new function `compute_competing_risks_survival()` (~60 lines). Inputs: cache with `days_in_mode`, `switch_point`, `survival_mode` + trigger variables. Output: 4 new columns (`survival_prob_liquidity`, `survival_prob_solvency`, `survival_prob_market`, `dominant_risk_channel`). Dependency: `lifelines.CoxPHFitter` (already imported).

**Edit point 1C -- extend `compute_survival_probability()` at line 204:**
Add uncertainty propagation (~25 lines) using CoxPH standard errors already computed by lifelines. Output: 3 new columns (`survival_probability_p10`, `survival_probability_p90`, `survival_uncertainty`).

**Current function signatures (must not break):**
- [`compute_company_survival_flag(df, thresholds=None)`](operator1/analysis/survival_mode.py:71) -> `pd.Series`
- [`compute_survival_probability(df, thresholds=None)`](operator1/analysis/survival_mode.py:204) -> `pd.Series`
- [`compute_cox_survival_score(df)`](operator1/analysis/survival_mode.py:287) -> `pd.Series`

**Called from:**
- [`main.py:1464`](main.py:1464) -- Step 5 initial computation
- [`main.py:2226`](main.py:2226) -- Step 5j recalibration with adaptive thresholds
- [`backtest_runner.py:631`](backtest_runner.py:631) -- Stage 1
- [`backtest_runner.py:807`](backtest_runner.py:807) -- adaptive recalibration

**Outlet to downstream:**
- `company_survival_mode_flag` -> [`hierarchy_weights.py:245`](operator1/analysis/hierarchy_weights.py:245) (regime selection)
- `survival_probability` -> profile builder, report, MC fusion
- `cox_survival_score` -> survival probability blending
- NEW `survival_velocity_flag` -> USS early warning enhancement, extra vars
- NEW `survival_deterioration_rate` -> extra vars, conformal band widening
- NEW `survival_prob_{liquidity,solvency,market}` -> profile, report, HF risk decomposition
- NEW `survival_uncertainty` -> conformal band scaling

---

### 1b. `operator1/analysis/hierarchy_weights.py` (322 lines)

**Enhancement:** 2A (entropy-based weight allocation)

**Edit point -- inside `compute_hierarchy_weights()` at line 268-300:**
After the regime weight lookup (line 282) and before vanity adjustment (line 284), add entropy-based blending (~25 lines). Inputs: forward pass `errors_by_tier` from `PipelineState` (if available from prior run -- loaded from disk). Output: modified `base_weights` list.

**Critical constraint:** Must not break the existing function signature: [`compute_hierarchy_weights(df, config=None)`](operator1/analysis/hierarchy_weights.py:214) -> `pd.DataFrame`. The entropy blending is additive -- it modifies the `base_weights` between regime lookup and vanity adjustment.

**New parameter:** Add optional `forward_pass_errors: dict | None = None` parameter. When None (first run), falls back to pure regime defaults.

**Called from:**
- [`main.py:1487`](main.py:1487) -- Step 5
- [`main.py:2233`](main.py:2233) -- Step 5j recalibration
- [`backtest_runner.py:642`](backtest_runner.py:642) -- Stage 1
- [`backtest_runner.py:809`](backtest_runner.py:809) -- adaptive recalibration

**Outlet to downstream:**
- `hierarchy_tier{1-5}_weight` -> [`financial_health.py`](operator1/models/financial_health.py) composite weighting
- `survival_regime` -> [`feature_normalization.py`](operator1/features/feature_normalization.py) regime z-scores
- NEW `hierarchy_tier{1-5}_entropy` -> profile (informational), extra vars

---

### 1c. `operator1/analysis/survival_timeline.py` (665 lines)

**Enhancement:** 3A (semi-Markov duration modeling)

**Edit point -- after `_compute_stability_score()` at line ~270:**
Add new function `_compute_semi_markov_exit()` (~25 lines). Uses `scipy.stats.weibull_min.fit()` on collected `days_in_mode` dwell times from history.

**Edit point -- inside `compute_survival_timeline()` at line 275:**
After computing `days_counter` (line 339) and before writing to cache (line 353), add call to `_compute_semi_markov_exit()`. Output: 2 new columns (`expected_remaining_days_in_mode`, `mode_exit_probability_21d`).

**Called from:**
- [`main.py:2316-2358`](main.py:2316) -- Step 5.5 enriched timeline
- [`backtest_runner.py:896-904`](backtest_runner.py:896) -- Stage 1 enriched timeline
- [`operator1/stages/stage5_forward.py:87-99`](operator1/stages/stage5_forward.py:87) -- walk-forward mode extraction

**Outlet to downstream:**
- `survival_mode` -> walk-forward mode-conditioned scoring, USS controller
- `switch_point` -> walk-forward retrain trigger
- `days_in_mode` -> profile, stability score
- `stability_score_21d` -> extra vars, regime shift predictor
- NEW `expected_remaining_days_in_mode` -> profile, report
- NEW `mode_exit_probability_21d` -> extra vars, prediction aggregator

---

### 1d. `operator1/models/financial_health.py` (1,100 lines)

**Enhancements:** 5A (ensemble distress), 5B (EWM percentile rank), 5C (CVaR composite)

**Edit point 5A -- after Altman Z + Beneish M section:**
Add `_compute_ohlson_o_score()` (~20 lines), `_compute_zmijewski_score()` (~8 lines), `_compute_chs_score()` (~15 lines), and `_ensemble_distress()` (~20 lines with isotonic calibration). Output: 2 new columns (`fh_ensemble_distress_prob`, `fh_ensemble_distress_label`).

**Edit point 5B -- modify `_expanding_percentile_rank()` (used in tier scoring):**
Add halflife parameter for exponential weighting. Must check where this function is defined and how it's called by each tier scorer.

**Edit point 5C -- after composite score computation:**
Add `_compute_cvar_composite()` (~10 lines). Output: 1 new column (`fh_cvar_composite`).

**Current composite computation flow:**
1. Per-tier scores computed via expanding percentile rank
2. Weighted average using hierarchy weights -> `fh_composite_score`
3. Label assigned via thresholds -> `fh_composite_label`

**Called from:**
- [`main.py:1602`](main.py:1602) -- Step 5d
- [`backtest_runner.py:670`](backtest_runner.py:670) -- Stage 1

**Outlet to downstream:**
- `fh_composite_score` -> adaptive thresholds, HF scorecard, profile
- `fh_altman_z_score` -> `altman_z_momentum_63d` (Layer 1 Stage 20), HF advanced methods
- `fh_beneish_m_score` -> HF earnings smoothing
- `fh_runway_months` -> profile, triage card
- NEW `fh_ensemble_distress_prob` -> profile, USS early warning, HF risk
- NEW `fh_cvar_composite` -> profile (more sensitive to weak-link tiers)

---

### 1e. `operator1/analysis/scenario_engine.py` (388 lines)

**Enhancement:** 8B (reverse stress testing)

**Edit point -- after `run_scenario_engine()` at line ~388:**
Add new function `compute_reverse_stress_test()` (~50 lines). Inputs: cache, thresholds. Uses `scipy.optimize.minimize` with SLSQP. Output: result dict with `reverse_stress_revenue_shock`, `reverse_stress_margin_shock`, `reverse_stress_rate_shock`.

**Called from:**
- [`operator1/stages/stage7_integration.py:53`](operator1/stages/stage7_integration.py:53) -- Stage 7.1

**Outlet to downstream:**
- `ScenarioEngineResult` -> profile `scenario_analysis` section, report
- NEW `reverse_stress` fields -> profile `scenario_analysis` section, report section 19.96

---

### 1f. `operator1/analysis/survival_regime_controller.py` (724 lines)

**Enhancement:** 7B (gradual regime transition)

**Edit point -- inside `SurvivalRegimeController` class (~line 435):**
Add `_soft_transition()` method (~15 lines). Modify `get_model_config()` to interpolate between old and new config when `days_since_switch < transition_window`.

**Edit point -- enhance `compute_early_warning_score()` at line 382:**
Integrate the new `survival_velocity_flag` from enhancement 1A as an additional early warning component.

**Called from:**
- [`main.py` Step 5-USS](main.py) -- USS controller instantiation
- [`operator1/stages/stage7_integration.py:32`](operator1/stages/stage7_integration.py:32) -- early warning check

**Outlet to downstream:**
- `early_warning_score` -> profile, report
- `model_config` -> forecasting, forward pass, MC, transformer, LSTM
- NEW soft-transitioned configs -> all model consumers (transparent change)

---

## 2. Files That Will Be INDIRECTLY AFFECTED (downstream wiring)

### 2a. `main.py` (~3,500 lines)

**Edit points for new function calls:**

| Line | Current | Add After |
|------|---------|-----------|
| ~1465 | `compute_company_survival_flag(cache)` | `compute_survival_velocity(cache)` (1A) |
| ~1469 | `compute_cox_survival_score(cache)` | `compute_competing_risks_survival(cache)` (1B) |
| ~1465 | `compute_survival_probability(cache)` | Extend to return uncertainty bands (1C) |
| ~1487 | `compute_hierarchy_weights(cache)` | Pass `forward_pass_errors` if available (2A) |
| ~1604 | `compute_financial_health(cache)` | Results now include ensemble + CVaR (5A/5C) |
| ~USS block | `run_scenario_engine(cache)` | `compute_reverse_stress_test(cache)` (8B) |

### 2b. `backtest_runner.py` (~2,028 lines)

**Mirror all main.py changes** at corresponding locations:
- Line ~631: Add velocity + competing risks calls
- Line ~642: Pass forward_pass_errors to hierarchy weights
- Line ~670: Financial health results now include new fields
- Line ~807: Adaptive recalibration gets velocity + competing risks

### 2c. `operator1/stages/stage3_temporal.py` -- `_init_extra_vars()` at line 24

**Add prefix patterns for new Layer 2 columns:**
```
or c.startswith("survival_velocity") or c.startswith("survival_deterioration")
or c.startswith("survival_prob_") or c.startswith("dominant_risk_")
or c.startswith("survival_uncertainty") or c.startswith("survival_probability_p")
or c.startswith("expected_remaining") or c.startswith("mode_exit_")
or c.startswith("changepoint_") or c.startswith("fh_ensemble_")
or c.startswith("fh_cvar_") or c.startswith("hierarchy_tier") and c.endswith("_entropy")
```

### 2d. `operator1/analysis/signal_ic.py` -- `_SIGNAL_SPEED` dict at line 35

**Add new signals:**
```python
# Slow (fundamental, quarterly frequency)
"survival_prob_liquidity": "slow",
"survival_prob_solvency": "slow",
"fh_ensemble_distress_prob": "slow",
"fh_cvar_composite": "slow",

# Medium (derived, changes with regimes)
"survival_deterioration_rate": "medium",
"covenant_proximity_score": "medium",  # from Layer 1 but not yet in IC
"dominant_risk_channel": "medium",
"mode_exit_probability_21d": "medium",

# Fast (daily updates)
"survival_velocity_flag": "fast",
"early_warning_score": "fast",
"survival_uncertainty": "fast",
```

### 2e. `config/survival_hierarchy.yml` -- tier variable membership

**Add new variables to tiers:**
```yaml
tier1:
  variables:
    # ... existing ...
    - survival_prob_liquidity      # NEW: competing risks liquidity channel
tier2:
  variables:
    # ... existing ...
    - survival_prob_solvency       # NEW: competing risks solvency channel
    - fh_ensemble_distress_prob    # NEW: ensemble distress probability
tier3:
  variables:
    # ... existing ...
    - survival_prob_market         # NEW: competing risks market channel
    - mode_exit_probability_21d    # NEW: semi-Markov exit probability
```

### 2f. `operator1/report/profile_builder.py`

**Sections to update:**
- `_build_survival_section()` -- add velocity, competing risks, uncertainty
- `_build_financial_health_section()` -- add ensemble distress, CVaR composite
- Scenario analysis section -- add reverse stress test results

### 2g. `operator1/report/report_generator.py`

**Sections to update:**
- Section 6 (Survival Analysis) -- render competing risks decomposition
- Section 5 (Financial Health) -- render ensemble distress probability
- Section 19.96 (Scenario Analysis) -- render reverse stress test

---

## 3. COMPLETE INPUT/OUTPUT/OUTLET MAP

### 3.1 New Variables -> Downstream Consumers

| New Variable | Producer | Direct Consumers | How Consumed |
|---|---|---|---|
| `survival_velocity_flag` | survival_mode.py | USS controller, extra vars | Early warning enhancement, temporal model feature |
| `survival_deterioration_rate` | survival_mode.py | extra vars, conformal.py | Feature for temporal models, band widening |
| `survival_prob_liquidity` | survival_mode.py | profile, report, HF | Risk decomposition per failure channel |
| `survival_prob_solvency` | survival_mode.py | profile, report, HF | Risk decomposition |
| `survival_prob_market` | survival_mode.py | profile, report, HF | Risk decomposition |
| `dominant_risk_channel` | survival_mode.py | profile, report | Text label for dominant risk |
| `survival_probability_p10` | survival_mode.py | profile, report | Lower credible interval |
| `survival_probability_p90` | survival_mode.py | profile, report | Upper credible interval |
| `survival_uncertainty` | survival_mode.py | conformal.py, prediction_aggregator | Band scaling by survival uncertainty |
| `hierarchy_tier{1-5}_entropy` | hierarchy_weights.py | profile | Informational, attention diagnostics |
| `expected_remaining_days_in_mode` | survival_timeline.py | profile, report | Duration-aware mode prediction |
| `mode_exit_probability_21d` | survival_timeline.py | extra vars, prediction_aggregator | Regime persistence signal |
| `fh_ensemble_distress_prob` | financial_health.py | profile, report, HF, USS | Calibrated multi-model distress |
| `fh_ensemble_distress_label` | financial_health.py | profile, report | Human-readable distress label |
| `fh_cvar_composite` | financial_health.py | profile | Tail-sensitive health composite |
| `reverse_stress_revenue_shock` | scenario_engine.py | profile, report | Minimum shock to trigger survival |
| `reverse_stress_margin_shock` | scenario_engine.py | profile, report | Minimum margin shock |
| `reverse_stress_rate_shock` | scenario_engine.py | profile, report | Minimum rate shock |

### 3.2 Input Dependencies (What Each Enhancement Needs)

| Enhancement | Required Inputs | Available? | Notes |
|---|---|---|---|
| 1A Gradient velocity | `current_ratio`, `debt_to_equity_abs`, `fcf_yield`, `drawdown_252d` + thresholds | Yes (derived vars Stage 2-5) | Pure computation on existing columns |
| 1B Competing risks | `days_in_mode`, `switch_point`, `survival_mode` + trigger vars | Yes (computed earlier in Step 5) | Needs lifelines (already imported) |
| 1C Bayesian uncertainty | Cox PH fitted model + standard errors | Yes (from `compute_cox_survival_score`) | Propagate existing SE to probability |
| 2A Entropy weights | `forward_pass_errors_by_tier` from prior run | Partial (only available after first Stage 2 run) | Falls back to regime defaults on first run |
| 3A Semi-Markov | `days_in_mode` history from current cache | Yes | scipy.stats.weibull_min (already available) |
| 5A Ensemble distress | `total_assets`, `total_liabilities`, `net_income`, `working_capital`, `market_cap`, `close`, `volatility_21d` | Yes | Inline O-Score/Zmijewski/CHS + sklearn isotonic |
| 5B EWM percentile | Per-tier score series | Yes (internal to financial_health) | Modify existing `_expanding_percentile_rank` |
| 5C CVaR composite | 5 tier scores | Yes (computed in same function) | 10-line addition after weighted average |
| 7B Soft transition | `prev_regime`, `days_since_switch` | Need to track | Add to PipelineState or USS controller state |
| 8B Reverse stress | cache financial columns + survival thresholds | Yes | scipy.optimize.minimize (already available) |

---

## 4. PIPELINE EXECUTION ORDER

```
EXISTING FLOW (no change to order):
  Step 5:     derived_variables (Layer 1 -- provides trigger variables)
  Step 5:     compute_company_survival_flag()          -- ENHANCED (1A: add velocity)
  Step 5:     compute_survival_probability()           -- ENHANCED (1C: add uncertainty bands)
  Step 5:     compute_cox_survival_score()             -- existing
  Step 5:     compute_competing_risks_survival()       -- NEW (1B: after Cox)
  Step 5:     compute_hierarchy_weights()              -- ENHANCED (2A: entropy blending)
  Step 5-USS: SurvivalRegimeController                 -- ENHANCED (7B: soft transition)
  Step 5b:    fuzzy_protection                         -- existing (no change)
  Step 5d:    compute_financial_health()               -- ENHANCED (5A/5B/5C)
  Step 5d:    compute_vanity_score()                   -- existing (no change)
  Step 5.5:   compute_survival_timeline()              -- ENHANCED (3A: semi-Markov)
  Step 5.5:   compute_enriched_survival_timeline()     -- existing
  Step 5j:    adaptive_thresholds + RECALIBRATE        -- existing
  ...
  Step 6-USS: run_scenario_engine()                    -- existing
  Step 6-USS: compute_reverse_stress_test()            -- NEW (8B: after scenarios)
```

**No execution order changes needed.** All enhancements either extend existing functions in-place or add new functions called immediately after their existing counterpart.

---

## 5. RISK ANALYSIS

| Risk | File | Mitigation |
|---|---|---|
| Breaking `compute_company_survival_flag` signature | survival_mode.py:71 | New velocity function is SEPARATE, not modifying existing function |
| Breaking `compute_hierarchy_weights` signature | hierarchy_weights.py:214 | New `forward_pass_errors` param has default `None` (backward compatible) |
| Competing risks needs 5+ events per cause | survival_mode.py | Fallback to unconditional Aalen-Johansen, then geometric (Markov) |
| Semi-Markov Weibull fit fails on <3 dwell times | survival_timeline.py | Explicit fallback to geometric distribution |
| Ensemble distress needs labeled default data | financial_health.py | Use `company_survival_mode_flag` as proxy label |
| EWM percentile rank O(n^2) performance | financial_health.py | Cap window at 252 days; consider online approximation |
| Reverse stress optimizer gets stuck | scenario_engine.py | Use `differential_evolution` as global fallback |
| Soft transition halflife wrong | survival_regime_controller.py | Configurable in `scoring_weights.yml`, default 5 days |
| stage3_temporal misses new prefixes | stage3_temporal.py:45 | Add all new prefixes to `_init_extra_vars` filter |

---

## 6. COMPLETE FILE EDIT LIST (ordered by execution sequence)

| # | File | Lines | Edit Type | Enhancement | New Lines |
|---|------|-------|-----------|-------------|-----------|
| 1 | `operator1/analysis/survival_mode.py` | 664 | MAJOR | 1A + 1B + 1C | ~115 |
| 2 | `operator1/analysis/hierarchy_weights.py` | 322 | MODERATE | 2A | ~25 |
| 3 | `operator1/analysis/survival_regime_controller.py` | 724 | MODERATE | 7B + early warning enhance | ~30 |
| 4 | `operator1/models/financial_health.py` | 1,100 | MAJOR | 5A + 5B + 5C | ~75 |
| 5 | `operator1/analysis/survival_timeline.py` | 665 | MODERATE | 3A | ~30 |
| 6 | `operator1/analysis/scenario_engine.py` | 388 | MODERATE | 8B | ~50 |
| 7 | `main.py` | ~3,500 | MINOR | Wire new calls | ~25 |
| 8 | `backtest_runner.py` | ~2,028 | MINOR | Mirror main.py wiring | ~25 |
| 9 | `operator1/stages/stage3_temporal.py` | 295 | MINOR | `_init_extra_vars` prefixes | ~8 |
| 10 | `operator1/analysis/signal_ic.py` | ~400 | MINOR | `_SIGNAL_SPEED` entries | ~12 |
| 11 | `config/survival_hierarchy.yml` | 162 | MINOR | New tier variables | ~8 |
| 12 | `operator1/report/profile_builder.py` | ~1,276 | MINOR | New profile sections | ~30 |
| 13 | `operator1/report/report_generator.py` | ~4,628 | MINOR | Render new data | ~20 |
| **Total** | | | | | **~453 lines** |

---

## 7. IMPLEMENTATION PHASES

### Phase 1: P1 Enhancements (immediate, ~170 lines)

**Step 1.1:** `survival_mode.py` -- add `compute_survival_velocity()` (1A, ~30 lines)
**Step 1.2:** `survival_mode.py` -- extend `compute_survival_probability()` with uncertainty (1C, ~25 lines)
**Step 1.3:** `financial_health.py` -- add EWM percentile rank option (5B, ~15 lines)
**Step 1.4:** `survival_timeline.py` -- add semi-Markov `_compute_semi_markov_exit()` (3A, ~30 lines)
**Step 1.5:** Wire Phase 1 in `main.py` + `backtest_runner.py` (~20 lines)
**Step 1.6:** Update `stage3_temporal.py` `_init_extra_vars` for new columns (~5 lines)
**Step 1.7:** Update `signal_ic.py` `_SIGNAL_SPEED` (~5 lines)
**Step 1.8:** Tests for Phase 1 features

### Phase 2: P2 Enhancements (~205 lines)

**Step 2.1:** `hierarchy_weights.py` -- add entropy-based blending (2A, ~25 lines)
**Step 2.2:** `financial_health.py` -- add ensemble distress (5A, ~65 lines: O-Score + Zmijewski + CHS + isotonic)
**Step 2.3:** `financial_health.py` -- add CVaR composite (5C, ~10 lines)
**Step 2.4:** `survival_regime_controller.py` -- add soft transition (7B, ~15 lines)
**Step 2.5:** `scenario_engine.py` -- add reverse stress test (8B, ~50 lines)
**Step 2.6:** Wire Phase 2 in `main.py` + `backtest_runner.py` (~25 lines)
**Step 2.7:** Update remaining downstream files (~15 lines)
**Step 2.8:** Tests for Phase 2 features

### Phase 3: P3 Enhancements (~60 lines)

**Step 3.1:** `survival_mode.py` -- add competing risks (1B, ~60 lines)
**Step 3.2:** Wire in `main.py` + `backtest_runner.py`
**Step 3.3:** Update profile builder + report generator for risk decomposition

### Phase 4: P4 Enhancements (deferred)

4A (Choquet integral) and 8A (conditional scenarios) deferred -- high complexity, low immediate impact.

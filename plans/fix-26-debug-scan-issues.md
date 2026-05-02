# Fix Plan: 26 Debug Scan Issues

*Created: 2026-05-02 | Branch: feature/fix-debug-scan-26-issues*

## Overview

Debug scan of the full codebase found 26 actionable issues across 6 categories. This plan organizes them into 5 implementation phases ordered by impact and dependency.

## Phase 1: Critical Runtime Bugs (3 fixes)

These cause actual failures. Must fix first.

### P1.1: Monte Carlo jump-diffusion NameError
**File:** `operator1/models/monte_carlo.py`
**Bug:** `run_simulation()` at line 758 uses `jump_lambda`, `jump_mean`, `jump_std` in its body (lines 828-830, 858-860) but these are NOT in its function signature. Causes `NameError` at runtime.
**Fix:**
- Add `jump_lambda: float = 0.0`, `jump_mean: float = 0.0`, `jump_std: float = 0.0` to `run_simulation()` signature (after `variable_sensitivities`)
- Find all call sites of `run_simulation()` (2 found at lines ~1489 and ~1260) and pass the jump params from the caller scope
- The caller `run_monte_carlo()` already has `jump_lambda`, `jump_mean`, `jump_std` as local variables (lines 1220-1238)
**Impact:** Restores MC simulation, survival probability, regime shift prediction, conformal MC override, and scenario engine

### P1.2: Backtest runner Stage 3 profile build loads empty state
**File:** `backtest_runner.py`
**Bug:** `_STAGE_DEPS["3"] = "2"` but `state_2.pkl` doesn't exist when sub-stages run via staged compiler. Profile build gets empty PipelineState.
**Fix:**
- Change Stage 3 dependency loading to use `_find_latest_checkpoint()` (same function added in commit 853e5aa for sub-stages)
- In the `for stage_key in stages:` loop, when `dep_key` is "2", try `_find_latest_checkpoint()` first before falling back to `state_1.pkl`
**Impact:** Profile build populates all 17 currently-empty sections

### P1.3: ForwardPassResult not serializable
**File:** `operator1/pipeline_state.py` + `operator1/models/forecasting.py`
**Bug:** `ForwardPassResult.model_states` contains fitted sklearn/torch model objects that may fail `pickle.dumps()` test in `PipelineState.save()`. The entire result is silently dropped.
**Fix:**
- In `PipelineState.save()`, the skip logic at line 193 catches `pickle.PicklingError` -- add logging when a key is skipped so it's not silent
- In `ForwardPassResult`, add a `__getstate__`/`__setstate__` that excludes non-picklable `model_states` but preserves all other fields (`errors_by_tier`, `predictions_log`, `pid_summary`, `conformal_calibrator`, `total_days`)
- Alternative: strip `model_states` before save and reconstruct on load (model_states only needed for SHAP, which runs before save)
**Impact:** Restores forward_pass_result, conformal prediction (6.3), PID summary in profile

## Phase 2: Backtest Profile Completeness (1 fix)

### P2.1: Verify all temporal results flow to profile
**Verification step:** After P1.1-P1.3 fixes, re-run the AAPL backtest and verify the 17 empty sections are now populated. The backtest_runner's `run_stage3()` already has the injection code for all 28 profile keys (confirmed by scan). The issue was purely that the state it loaded had None values.

## Phase 3: Report and Dashboard Wiring (17 fixes)

### P3.1: Add 5 new profile keys to dashboard Home page
**File:** `dashboard.py` - `render_home()` function
**Add cards/displays for:**
- `fh_ensemble_distress_prob` / `fh_ensemble_distress_label` (from `financial_health`)
- `fh_cvar_composite` (from `financial_health`)
- `cross_freq_momentum_score` / `cross_freq_direction_agreement` (from `multi_frequency`)
- `kelly_fraction` / `half_kelly_size` (from `hedge_fund.position`)
- `dupont_quality_driver` (from `hedge_fund.return_spread`)

### P3.2: Add 5 new sections to report_generator
**File:** `operator1/report/report_generator.py`
**Add builder functions and wire into TIER_SECTIONS:**
- `_build_behavioral_signals_section()` -> Pro+ tier
- `_build_complexity_signals_section()` -> Premium tier
- `_build_feature_selection_section()` -> Premium tier
- `_build_recursive_predictions_section()` -> Premium tier
- Feature normalization is internal (no user-facing section needed, remove from list)

### P3.3: Add 4 profile builder sections
**File:** `operator1/report/profile_builder.py`
**Add summary sections for:**
- `behavioral_signals`: latest values of 5 behavioral columns
- `complexity_signals`: latest values of 4 complexity columns
- `recursive_predictions`: summary from `recursive_result.to_dict()`
- Feature normalization produces ~35 columns that flow into `current_state` automatically (no separate section needed)

## Phase 4: Model Diagnostics Expansion (3 fixes)

### P4.1: Add transformer assessment to model_diagnostics
**File:** `operator1/monitoring/model_diagnostics.py`
**Expected behavior:** Transformer should fit when >100 data points and >2 float variables. Check: fitted + reasonable training loss + forecasts non-empty.

### P4.2: Add particle_filter assessment
**Expected behavior:** Particle filter should produce filtered states for survival variables. Check: filtered_states non-empty + percentiles reasonable.

### P4.3: Add recursive_aggregator assessment
**Expected behavior:** Recursive should produce N-day predictions with widening confidence bands. Check: day count matches request + confidence decays monotonically.

## Phase 5: Configuration Externalization (6 fixes)

### P5.1-P5.6: Move hardcoded constants to scoring_weights.yml
For each of 6 modules, extract tweakable constants into `config/scoring_weights.yml`:

| Module | Constants to extract |
|--------|---------------------|
| `behavioral_signals.py` | attention_spike_threshold (2.0), disposition_window (63) |
| `complexity_signals.py` | entropy_window (21), price_window (126), lz_window (252) |
| `feature_normalization.py` | zscore_window (63), percentile_window (252), change_window (21) |
| `options_signals.py` | delta_threshold (0.25), min_oi (100) |
| `cross_asset_signals.py` | sector_etfs list, momentum_window (21), stress_weights |
| `event_calendar.py` | fomc_dates list, uncertainty_decay_rate |

Each module change: replace inline constant with `get_weight("module_name.param", default)` call.

## Execution Order

```
Phase 1 (Critical):     P1.1 -> P1.2 -> P1.3
Phase 2 (Verification): P2.1 (re-run backtest)
Phase 3 (Wiring):       P3.1 -> P3.2 -> P3.3 (parallel)
Phase 4 (Diagnostics):  P4.1 -> P4.2 -> P4.3
Phase 5 (Config):       P5.1-P5.6 (parallel, independent)
```

## Verification

After all fixes:
1. Re-run AAPL backtest: `python3.12 run_backtest_staged.py --market us_sec_edgar --company AAPL --end-date 2024-12-31 --years 2 --validate`
2. Verify all 38 stages complete
3. Verify profile has 0 empty sections (was 17)
4. Verify MC survival_probability is non-None
5. Verify dashboard renders new cards
6. Verify report contains new sections

# Debug Scan: Layer 4 Edit Points, Inputs, Outputs & Full Implementation Plan

*Systematic scan of every file affected by the Layer 4 hedge fund enhancements*

---

## 1. Architecture Notes

Layer 4 is **self-contained** in `operator1/hedge_fund/`:
- `engine.py` (~1,250 lines) -- orchestrator + inline metrics (4.4-4.17)
- `types.py` (~420 lines) -- all dataclasses
- `fcf_quality.py` (~200 lines) -- 4.1
- `accruals_forensics.py` (~350 lines) -- 4.2
- `earnings_smoothing.py` (~300 lines) -- 4.3
- `advanced_methods.py` (~965 lines) -- 4.19
- `fusion.py` (~653 lines) -- 4.20
- `helpers.py` (~210 lines) -- shared utilities

**Key difference from Layers 1-3:** All outputs are result objects stored in `profile["hedge_fund"]` -- zero cache columns. Changes are confined to the HF modules with no impact on the main cache DataFrame.

**Pipeline entry point:** [`run_hedge_fund_analysis()`](operator1/hedge_fund/engine.py:1122) called from [`stage7_integration.py:run_7_5_hedge_fund()`](operator1/stages/stage7_integration.py) or [`main.py`](main.py).

---

## 2. Files That Will Be DIRECTLY EDITED

### 2a. `operator1/hedge_fund/engine.py` (~1,250 lines)

Most P1 enhancements go here since the base metrics are inline functions.

| Enhancement | Edit Point | New Lines |
|---|---|---|
| 4.13B: Market-implied growth | After `_compute_dcf()` at line 610 | ~15 |
| 4.17A: Kelly sizing | Inside `_compute_position_signal()` at line 1024 | ~10 |
| 4.10A: Regime-conditional momentum | Inside `_compute_momentum()` at line 428 | ~15 |
| 4.5A: DuPont decomposition | After `_compute_return_spread()` at line 144 | ~25 |
| 4.15A: Relative-absolute reconciliation | After `_compute_peg()` at line 865 | ~15 |
| 4.9A: Distress distance matrix | Inside `_compute_leverage_stress()` at line 330 | ~40 |
| 4.7A: Hidden leverage | Inside `_compute_obs_risk()` at line 232 | ~15 |

### 2b. `operator1/hedge_fund/fcf_quality.py` (~200 lines)

| Enhancement | Edit Point | New Lines |
|---|---|---|
| 4.1A: Quality degradation tracker | After `compute_fcf_quality()` at line 34 | ~20 |

### 2c. `operator1/hedge_fund/accruals_forensics.py` (~350 lines)

| Enhancement | Edit Point | New Lines |
|---|---|---|
| 4.2A: Peer-relative forensics | Inside `compute_accruals_forensics()` at line 28 | ~15 |

### 2d. `operator1/hedge_fund/earnings_smoothing.py` (~300 lines)

| Enhancement | Edit Point | New Lines |
|---|---|---|
| 4.3A: Revenue-expense Benford divergence | Inside existing `_benford_chi_squared()` at line 28 | ~15 |

### 2e. `operator1/hedge_fund/types.py` (~420 lines)

**Add new fields to existing dataclasses** (all optional with defaults for backward compat):

| Dataclass | New Fields |
|---|---|
| `FCFQualityResult` | `quality_slope`, `quality_runway_quarters`, `quality_acceleration` |
| `AccrualsForensicResult` | `peer_relative_red_flag`, `sector_accruals_percentile` |
| `SmoothingResult` | `benford_divergence_score` |
| `ReturnSpreadResult` | `dupont_tax_burden`, `dupont_interest_burden`, `dupont_asset_turnover`, `dupont_equity_multiplier`, `dupont_quality_driver` |
| `OBSRiskResult` | `adjusted_debt_to_equity`, `hidden_leverage_ratio` |
| `LeverageStressResult` | `distress_distance_matrix`, `quarters_of_buffer` |
| `MomentumCompositeResult` | `regime_adjusted_momentum`, `momentum_regime_bias` |
| `DCFResult` | `implied_growth_rate`, `growth_gap`, `priced_for_perfection_flag`, `riv_intrinsic`, `riv_excess_return`, `riv_vs_dcf_divergence` |
| `PEGCompositeResult` | `relative_absolute_agreement`, `value_trap_flag`, `conviction_boost` |
| `PositionSignalResult` | `kelly_fraction`, `half_kelly_size`, `kelly_edge` |

### 2f. `operator1/hedge_fund/fusion.py` (~653 lines)

| Enhancement | Edit Point | New Lines |
|---|---|---|
| 4.20A: Bayesian conviction updating | Inside `compute_belief_posterior()` at line 421 | ~25 |

---

## 3. PIPELINE WIRING

### 3.1 How enhancements flow through the pipeline

All Layer 4 enhancements are **internal to the HF module**. The pipeline entry point is:

```
main.py / stage7_integration.py
    -> run_hedge_fund_analysis()  [engine.py:1122]
        -> _compute_* functions (each metric)
        -> _build_scorecard()  [engine.py:922]
        -> _compute_position_signal()  [engine.py:1024]
        -> run_advanced_methods()  [advanced_methods.py:892]
        -> run_fusion()  [fusion.py:488]
    -> HedgeFundResult -> profile["hedge_fund"]
```

### 3.2 Input dependencies from other layers

| Enhancement | Inputs from Other Layers | Available? |
|---|---|---|
| 4.13B Implied growth | `close` (cache), `dcf_result` (self) | Yes |
| 4.17A Kelly | `mc_result.survival_probability`, `mc_result.terminal_values` | Yes (from Layer 3 MC) |
| 4.10A Regime momentum | `survival_regime` (cache, from Layer 2) | Yes |
| 4.5A DuPont | Raw `income_df`, `balance_df` | Yes (passed to engine) |
| 4.2A Peer forensics | `linked_caches` accruals | Partial (need to pass linked data) |
| 4.13A RIV | `book_value_per_share`, `beta_252d` (cache) | Yes |
| 4.7A Hidden leverage | `sga_expenses` (cache/statements) | Yes |
| 4.9A Distress matrix | Quarterly statements | Yes |
| 4.15A Reconciliation | `dcf_result`, `peer_ranking_result` | Yes (both available) |

### 3.3 No downstream wiring needed

Since all outputs are result objects added to existing dataclasses (with default values), nothing outside `operator1/hedge_fund/` needs modification:
- `profile_builder.py` already serializes the full HF result via `_available_dict()` -- new fields auto-appear
- `report_generator.py` renders HF sections from profile dict -- new fields auto-render
- `stage7_integration.py` calls `run_hedge_fund_analysis()` unchanged
- `run_backtest_staged.py` runs Stage 7.5 unchanged
- Dashboard reads profile JSON -- new fields auto-display

---

## 4. RISK ANALYSIS

| Risk | Mitigation |
|---|---|
| New dataclass fields break pickle deserialization of old checkpoints | All new fields have `default` or `default_factory` -- old pickles load fine |
| Implied growth model fails for negative FCF | Guard: return NaN when FCF <= 0 |
| Kelly fraction suggests huge position | Cap at [-0.5, 0.5] (half-Kelly) |
| DuPont denominator is zero | Use `safe_divide()` from helpers.py |
| Peer-relative forensics needs linked_caches | Fall back to absolute scoring when no peers |
| RIV needs cost of equity from CAPM | Fall back to fixed 10% when beta unavailable |

---

## 5. COMPLETE FILE EDIT LIST

| # | File | Lines | Edit Type | Enhancements | New Lines |
|---|------|-------|-----------|-------------|-----------|
| 1 | `operator1/hedge_fund/types.py` | ~420 | MODERATE | Add ~25 new dataclass fields | ~30 |
| 2 | `operator1/hedge_fund/engine.py` | ~1,250 | MAJOR | 4.5A, 4.7A, 4.9A, 4.10A, 4.13B, 4.15A, 4.17A | ~135 |
| 3 | `operator1/hedge_fund/fcf_quality.py` | ~200 | MINOR | 4.1A | ~20 |
| 4 | `operator1/hedge_fund/accruals_forensics.py` | ~350 | MINOR | 4.2A | ~15 |
| 5 | `operator1/hedge_fund/earnings_smoothing.py` | ~300 | MINOR | 4.3A | ~15 |
| 6 | `operator1/hedge_fund/fusion.py` | ~653 | MINOR | 4.20A | ~25 |
| **Total** | | | | **14 enhancements** | **~240 lines** |

No changes needed to: main.py, backtest_runner.py, stage7_integration.py, profile_builder.py, report_generator.py, dashboard.py, run.py, any config files.

---

## 6. IMPLEMENTATION PHASES

### Phase 1: P1 Enhancements (~80 lines)

**Step 1.1:** `types.py` -- add new fields to 4 dataclasses (DCFResult, PositionSignalResult, MomentumCompositeResult, ReturnSpreadResult)
**Step 1.2:** `engine.py` -- add `_compute_implied_growth()` after DCF block
**Step 1.3:** `engine.py` -- add Kelly sizing inside `_compute_position_signal()`
**Step 1.4:** `engine.py` -- add regime-conditional weights inside `_compute_momentum()`
**Step 1.5:** `engine.py` -- add DuPont decomposition after `_compute_return_spread()`

### Phase 2: P2 Enhancements (~100 lines)

**Step 2.1:** `types.py` -- add fields to FCFQualityResult, AccrualsForensicResult, PEGCompositeResult, DCFResult
**Step 2.2:** `fcf_quality.py` -- add quality degradation tracker
**Step 2.3:** `accruals_forensics.py` -- add peer-relative scoring
**Step 2.4:** `engine.py` -- add relative-absolute reconciliation after PEG
**Step 2.5:** `engine.py` -- add RIV inside DCF block

### Phase 3: P3 Enhancements (~45 lines)

**Step 3.1:** `types.py` -- add fields to OBSRiskResult, LeverageStressResult, SmoothingResult
**Step 3.2:** `engine.py` -- add hidden leverage inside OBS risk
**Step 3.3:** `engine.py` -- add distress distance matrix inside leverage stress
**Step 3.4:** `earnings_smoothing.py` -- add Benford divergence

### Phase 4: P4 Enhancements (~25 lines, deferred)

**Step 4.1:** `fusion.py` -- enhance `compute_belief_posterior()` with proper Bayesian updating

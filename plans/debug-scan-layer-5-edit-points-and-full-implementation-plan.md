# Debug Scan: Layer 5 Edit Points, Inputs, Outputs & Full Implementation Plan

---

## 1. Files That Will Be DIRECTLY EDITED

### 1a. `operator1/models/frequency_fusion.py` (~1,283 lines)

**Enhancements:** 5.3C (cross-freq momentum), 5.3B (weighted regime voting)

**Edit point 5.3C -- after `_m8_disagreement_signal()` at line ~635:**
Add new function `_compute_cross_frequency_momentum()` (~25 lines). Inputs: `results` dict with per-frequency `trend_direction`. Outputs: 3 new fields on `FrequencyFusionResult`.

**Edit point 5.3B -- inside `compute_regime_consensus()` at line 927:**
Replace the simple majority vote with weighted voting using `walk_forward_mae` and `n_periods`. ~20 lines modification.

**Edit point for result dataclass -- `FrequencyFusionResult` at line 163:**
Add 5 new fields: `cross_freq_momentum_score`, `cross_freq_direction_agreement`, `potential_reversal_flag`, `weighted_regime_consensus`, `regime_vote_weights`.

**Edit point for `to_profile_dict()` at line 184:**
Add serialization of new fields.

**Edit point in `fuse_multi_frequency_results()` at line 1060:**
Call `_compute_cross_frequency_momentum()` after M8 disagreement and add results to the final `FrequencyFusionResult`.

**Called from:**
- [`operator1/stages/stage7_integration.py`](operator1/stages/stage7_integration.py) -- `run_7_4_6_fusion()`

**Outlets:**
- `FrequencyFusionResult` -> `profile["multi_frequency"]` via `to_profile_dict()`
- New fields auto-serialize -- no profile_builder changes needed

### 1b. `operator1/steps/multi_frequency_runner.py` (~620 lines)

**Enhancement:** 5.2A (uncertainty propagation)

**Edit point -- `FrequencyContext` dataclass at line 48:**
Add 2 new fields: `forecast_uncertainty: dict[str, float]` and `regime_posterior: dict[str, float]`.

**Edit point -- inside `run_single_frequency_pipeline()` at line ~332:**
Populate `context_for_next.forecast_uncertainty` from MC result or conformal result.

**Called from:**
- [`operator1/stages/stage7_integration.py`](operator1/stages/stage7_integration.py) -- `run_7_4_1` through `run_7_4_5`

---

## 2. PIPELINE WIRING

Layer 5 is **self-contained** in the staged pipeline:
- Stage 7.4.0: Resample prep
- Stage 7.4.1-7.4.5: Per-frequency runs
- Stage 7.4.6: Fusion

All changes are internal to these sub-stages. No changes needed to main.py, backtest_runner.py, profile_builder.py, dashboard.py.

---

## 3. INPUT/OUTPUT MAP

| New Variable | Producer | Consumers |
|---|---|---|
| `cross_freq_momentum_score` | frequency_fusion.py | profile["multi_frequency"], report |
| `cross_freq_direction_agreement` | frequency_fusion.py | profile, HF fusion |
| `potential_reversal_flag` | frequency_fusion.py | profile, report |
| `weighted_regime_consensus` | frequency_fusion.py | profile (supplements existing consensus) |
| `regime_vote_weights` | frequency_fusion.py | profile (diagnostic) |
| `forecast_uncertainty` | multi_frequency_runner.py | Next faster frequency (prior) |
| `regime_posterior` | multi_frequency_runner.py | Next faster frequency (soft prior) |

---

## 4. BACKWARD COMPATIBILITY

All new dataclass fields have defaults (empty dicts, 0.0, False). Old checkpoints load fine. The `to_profile_dict()` method auto-serializes new fields.

---

## 5. COMPLETE FILE EDIT LIST

| # | File | Lines | Edit Type | Enhancement | New Lines |
|---|------|-------|-----------|-------------|-----------|
| 1 | `operator1/models/frequency_fusion.py` | 1,283 | MODERATE | 5.3C + 5.3B | ~55 |
| 2 | `operator1/steps/multi_frequency_runner.py` | 620 | MINOR | 5.2A | ~20 |
| **Total** | | | | | **~75 lines** |

---

## 6. IMPLEMENTATION PHASES

### Phase 1: P1 (~45 lines)

**Step 1.1:** Add `FrequencyFusionResult` new fields (5 fields with defaults)
**Step 1.2:** Add `_compute_cross_frequency_momentum()` function
**Step 1.3:** Call it from `fuse_multi_frequency_results()` and wire to result
**Step 1.4:** Enhance `compute_regime_consensus()` with weighted voting
**Step 1.5:** Update `to_profile_dict()` to serialize new fields

### Phase 2: P2 (~30 lines)

**Step 2.1:** Add `FrequencyContext` uncertainty fields
**Step 2.2:** Populate from MC/conformal in `run_single_frequency_pipeline()`

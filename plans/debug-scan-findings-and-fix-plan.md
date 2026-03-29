# Debug Scan Findings and Fix Plan

Results of a comprehensive debug scan across 146 Python files in the Operator 1 pipeline. 5 issues found: 2 bugs and 3 data flow gaps.

---

## Scan Summary

| Area | Result |
|------|--------|
| Syntax check (146 files) | CLEAN -- 0 errors |
| main.py wiring (59 model functions) | CLEAN -- all found |
| run.py delegation | CLEAN -- subprocess to main.py |
| Profile storage | CLEAN -- all model results stored |
| Duplication | CLEAN -- only deprecated wrapper + protocol implementations |
| None-access safety | CLEAN -- all guarded by try/except or None checks |
| **Bugs found** | **2** |
| **Data flow gaps** | **3** |

---

## Bug 1: persist_linked_aggregates unreachable dead code

**File:** `operator1/features/linked_aggregates.py` lines 332-413
**Severity:** Medium (data not persisted to disk, but not consumed downstream either)

**Problem:** The `compute_relative_metrics` function definition was inserted inside the body of `persist_linked_aggregates`. Python treats `compute_relative_metrics` as a nested function definition, and after `compute_relative_metrics` returns at line 405, the remaining lines 407-413 of `persist_linked_aggregates` (which do `os.makedirs` + `aggregates.to_parquet`) become unreachable. The function returns `None` instead of writing the parquet file.

**Current structure:**
```
def persist_linked_aggregates(...):
    if output_path is None:
        output_path = ...

    def compute_relative_metrics(...):   # <-- nested inside persist!
        ...
        return result                    # line 405

    os.makedirs(...)                     # line 407 -- UNREACHABLE
    aggregates.to_parquet(...)           # line 408 -- UNREACHABLE
    return output_path                   # line 413 -- UNREACHABLE
```

**Fix:** Move `compute_relative_metrics` outside `persist_linked_aggregates` so both are top-level functions.

**Edit location:** `operator1/features/linked_aggregates.py:350-413`
**What to change:** Add proper return + dedent so `compute_relative_metrics` is a sibling function, not a nested one. Restructure:
```python
def persist_linked_aggregates(...):
    if output_path is None:
        output_path = ...
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aggregates.to_parquet(output_path)
    logger.info(...)
    return output_path


def compute_relative_metrics(...):
    ...
    return result
```

---

## Bug 2: enriched_survival_timeline not rendered in reports

**File:** `operator1/report/report_generator.py`
**Severity:** Low (data is in profile but invisible to end users)

**Problem:** The profile stores `enriched_survival_timeline` data including `regime_available`, `mean_intensity`, `combined_state_distribution`, `base_n_switches`, and `base_mean_stability`. The report generator has no section that reads or renders this data. Users cannot see the enriched regime state analysis in any report tier.

**Fix:** Add a report section (or sub-section within the existing Survival section) that renders the enriched timeline data: current regime state, mean survival intensity, state distribution breakdown, number of regime switches, and mean stability score.

**Edit location:** `operator1/report/report_generator.py` -- add builder function `_build_enriched_timeline_section` and wire it into `_section_builders` for Pro and Premium tiers.

---

## Data Flow Gap 3: Adaptive windows not passed to run_forecasting

**File:** `main.py:2388`
**Severity:** Low (forecasting uses reasonable hardcoded defaults)

**Problem:** `_adaptive_tier3.windows` (Nyquist-anchored window sizes: short/medium/long/trend) are computed in Step 5k.2 but never passed to `run_forecasting()`. The forecasting module continues using hardcoded constants (21/63/126/252 days) regardless of filing frequency.

**Fix:** Add a `windows` parameter to `run_forecasting()` and pass `_adaptive_tier3.windows` from main.py. Inside forecasting.py, use the provided windows for rolling computations instead of the hardcoded constants.

**Edit locations:**
1. `operator1/models/forecasting.py` -- add `windows` parameter to `run_forecasting()` signature
2. `main.py:~2388` -- pass `_adaptive_tier3.windows` in the call

---

## Data Flow Gap 4: rel_ columns not in _extra_vars

**File:** `main.py:~2236`
**Severity:** Low (relative metrics computed but not used by temporal models)

**Problem:** The `_extra_vars` filter uses `_linked_prefixes` to auto-include cross-entity columns. The prefixes are `competitors_`, `suppliers_`, `customers_`, etc. but not `rel_`. So `rel_strength_vs_sector`, `valuation_premium_vs_industry`, and `rel_volatility_vs_sector` (added in our PR) are computed and merged into the cache but never passed as features to temporal models.

**Fix:** Add `"rel_"` to the `_linked_prefixes` tuple, or add explicit column name checks for the relative metrics in the `_extra_vars` filter.

**Edit location:** `main.py:~2237` -- modify `_linked_prefixes` tuple.

---

## Data Flow Gap 5: Mondrian partitioning not wired

**File:** `main.py` Step 6p (conformal prediction)
**Severity:** Low (conformal intervals work but are not survival-mode-aware)

**Problem:** `ConformalPIDCalibrator` supports Mondrian partitioning (separate calibrators per survival mode) but `survival_mode` from the enriched timeline is not passed as the partition variable. All intervals have the same width regardless of survival mode, even though crisis modes should have wider intervals.

**Fix:** Pass `enriched_timeline_result.timeline["survival_mode"]` to the calibrator setup so it creates per-mode score pools.

**Edit location:** `main.py` -- in the conformal prediction block (~Step 6p), pass the survival_mode Series when constructing or feeding the calibrator.

---

## Implementation Priority

| # | Issue | Severity | Effort | Order |
|---|-------|----------|--------|-------|
| 1 | persist_linked_aggregates dead code | Medium | Small | First |
| 4 | rel_ columns not in _extra_vars | Low | Trivial | Second |
| 2 | enriched_survival_timeline not in reports | Low | Medium | Third |
| 3 | Adaptive windows not passed to forecasting | Low | Medium | Fourth |
| 5 | Mondrian partitioning not wired | Low | Medium | Fifth |

Bugs 1 and 4 are quick fixes (move a function, add a prefix string). Bugs 2, 3, and 5 require adding new parameters and logic to existing functions.

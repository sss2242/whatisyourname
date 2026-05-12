# Sequential Frequency Pipeline -- Full Implementation Plan

*Implementation plan with exact code references from debug scan. Continues from the architecture plan and GitHub research documents.*

---

## Debug Scan Findings

### Files to Edit (4 files, ~80 lines new code)

| # | File | Lines Changed | What Changes |
|---|------|--------------|-------------|
| 1 | `operator1/stages/stage2_freq_pipeline.py` | +60 | Add 6 sequential sub-stage functions + `_run_freq_sequential()` helper + `build_freq_substages()` dynamic registry |
| 2 | `operator1/stages/runner.py` | ~3 | Change import from static constant to dynamic `build_freq_substages()` call |
| 3 | `config/global_config.yml` | +12 | Add `frequency_pipeline:` config section |
| 4 | `run_backtest_staged.py` | +8 | Add sequential sub-stage IDs to `ALL_STAGES` |

### Files NOT Changed (verified safe)

| File | Why No Change Needed |
|------|---------------------|
| `operator1/pipeline_state.py` | Existing `save_mf_result`/`load_mf_result`/`save_mf_context`/`load_mf_context` already work for both modes |
| `operator1/stages/stage7_integration.py` | Skip logic at line 438 (`multi_frequency_result is not None`) handles both modes |
| `run_lean_backtest.py` | Does not reference Stage 2 freq sub-stage IDs |
| `operator1/stages/runner.py` `_CRITICAL_SUBSTAGES` | Sequential stages are non-critical by design |
| `operator1/stages/runner.py` `_SUBSTAGE_REQUIREMENTS` | Sequential stages have same requirements as parallel (cache exists) |

### Risk Analysis

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | Import-time evaluation of `STAGE_2_FREQ_SUBSTAGES` | Change to lazy evaluation via function call in `_build_registry()` |
| 2 | Config not loaded when builder runs | `_build_registry()` is called inside `run_stages()`, not at import time -- safe |
| 3 | Backtest compiler dispatches unknown IDs | Runner's `_filter_substages()` returns empty list for unrecognized IDs -- safe (logs warning) |
| 4 | `_matches_stage("2.S.A", "2")` must match | Prefix matching `"2.S.A".startswith("2.")` = True -- correct |
| 5 | `_matches_stage("2.W1", "2.S")` must NOT match | `"2.W1".startswith("2.S.")` = False -- correct |

---

## Change 1: `operator1/stages/stage2_freq_pipeline.py`

### 1a. Add `_run_freq_sequential()` helper after `_run_freq_isolated()` (after line 132)

```python
def _run_freq_sequential(state: "PipelineState", freq: str, prior_context_freq: str | None) -> None:
    """Run one frequency pipeline sequentially with checkpoint resume + memory cleanup.

    Reuses _run_freq_isolated() but adds:
    1. Checkpoint check: skip if result already exists on disk (crash resume)
    2. Explicit gc.collect() after completion to free memory for next frequency
    """
    import gc

    freqs = state.load_mf_frequencies()
    if freq not in freqs:
        logger.info("[%s] Not in available frequencies, skipping", freq)
        return

    # Checkpoint resume: skip if this frequency already completed
    existing = state.load_mf_result(freq)
    if existing is not None:
        logger.info("[%s] Already completed (found checkpoint), skipping", freq)
        return

    _run_freq_isolated(state, freq, prior_context_freq)

    # Explicit memory cleanup for low-resource devices
    gc.collect()
    logger.info("[%s] Sequential pipeline complete, memory freed", freq)
```

**Input:** PipelineState with `load_mf_frequencies()`, `load_mf_result()`, `load_mf_cache()` populated from 2.0 resample prep.
**Output:** Per-frequency result saved via `state.save_mf_result(freq, result)` and context via `state.save_mf_context(freq, context)`.
**Calls:** `_run_freq_isolated()` which calls `run_single_frequency_pipeline()` from `multi_frequency_runner.py`.

### 1b. Add 6 sequential sub-stage functions (after the helper)

```python
def run_2_seq_annual(state: "PipelineState") -> None:
    """2.S.A: Run Annual frequency pipeline (sequential mode)."""
    logger.info("Stage 2.S.A: Annual pipeline (sequential)")
    _run_freq_sequential(state, "A", prior_context_freq=None)

def run_2_seq_quarterly(state: "PipelineState") -> None:
    """2.S.Q: Run Quarterly frequency pipeline (sequential mode)."""
    logger.info("Stage 2.S.Q: Quarterly pipeline (sequential)")
    _run_freq_sequential(state, "Q", prior_context_freq="A")

def run_2_seq_semiannual(state: "PipelineState") -> None:
    """2.S.S: Run Semi-annual frequency pipeline (sequential mode)."""
    logger.info("Stage 2.S.S: Semi-annual pipeline (sequential)")
    _run_freq_sequential(state, "S", prior_context_freq="Q")

def run_2_seq_monthly(state: "PipelineState") -> None:
    """2.S.M: Run Monthly frequency pipeline (sequential mode)."""
    logger.info("Stage 2.S.M: Monthly pipeline (sequential)")
    _run_freq_sequential(state, "M", prior_context_freq="Q")

def run_2_seq_weekly(state: "PipelineState") -> None:
    """2.S.W: Run Weekly frequency pipeline (sequential mode)."""
    logger.info("Stage 2.S.W: Weekly pipeline (sequential)")
    _run_freq_sequential(state, "W", prior_context_freq="Q")

def run_2_seq_daily(state: "PipelineState") -> None:
    """2.S.D: Run Daily frequency pipeline (sequential mode)."""
    logger.info("Stage 2.S.D: Daily pipeline (sequential)")
    _run_freq_sequential(state, "D", prior_context_freq="Q")
```

**Context chain:** A -> Q -> S (gets Q), M (gets Q), W (gets Q), D (gets Q)
- Annual runs first with no prior context
- Quarterly loads Annual context (trend, regime, survival, forecast bounds)
- All remaining frequencies load Quarterly context (the most relevant native filing frequency)
- This is BETTER information flow than parallel mode where A/Q/S have NO context from each other

### 1c. Add `build_freq_substages()` dynamic registry builder (replace static constant)

```python
def build_freq_substages() -> list[tuple[str, callable]]:
    """Build frequency pipeline sub-stages based on config mode.

    Reads frequency_pipeline.mode from global_config.yml:
    - "parallel" (default): 2-wave parallel (A+Q+S then M+W+D)
    - "sequential": one frequency at a time (A->Q->S->M->W->D)

    Both modes start with 2.0 (resample prep) and end with 2.F (fusion).
    """
    try:
        from operator1.config_loader import get_global_config
        cfg = get_global_config().get("frequency_pipeline", {})
    except Exception:
        cfg = {}

    mode = cfg.get("mode", "parallel")

    if mode == "sequential":
        return [
            ("2.0", run_2_0_resample_prep),
            ("2.S.A", run_2_seq_annual),
            ("2.S.Q", run_2_seq_quarterly),
            ("2.S.S", run_2_seq_semiannual),
            ("2.S.M", run_2_seq_monthly),
            ("2.S.W", run_2_seq_weekly),
            ("2.S.D", run_2_seq_daily),
            ("2.F", run_2_F_fusion),
        ]
    else:  # parallel (default)
        return [
            ("2.0", run_2_0_resample_prep),
            ("2.W1", run_2_wave1_native),
            ("2.W2", run_2_wave2_interpolated),
            ("2.F", run_2_F_fusion),
        ]


# Backward compat: static constant for direct imports.
# The runner uses build_freq_substages() dynamically instead.
STAGE_2_FREQ_SUBSTAGES = build_freq_substages()
```

**Key design decision:** `STAGE_2_FREQ_SUBSTAGES` is kept as a module-level constant for backward compatibility (any code that does `from stage2_freq_pipeline import STAGE_2_FREQ_SUBSTAGES`). But the runner calls `build_freq_substages()` directly for fresh evaluation.

---

## Change 2: `operator1/stages/runner.py`

### Lines 44-55: Change from importing constant to calling builder

**Before:**
```python
from operator1.stages.stage2_freq_pipeline import STAGE_2_FREQ_SUBSTAGES
```

**After:**
```python
from operator1.stages.stage2_freq_pipeline import build_freq_substages
```

And in the return statement:
```python
    return (
        STAGE_2_SUBSTAGES
        + build_freq_substages()  # dynamic: parallel or sequential based on config
        + STAGE_3_SUBSTAGES
        ...
    )
```

**Why:** This ensures the registry is built fresh each time `run_stages()` is called, picking up the latest config value. Import-time evaluation would lock in the mode before the user can change config.

---

## Change 3: `config/global_config.yml`

### Add after the `forecasting:` section (after line 115)

```yaml
# ---------------------------------------------------------------------------
# Frequency Pipeline Configuration
# ---------------------------------------------------------------------------
# Controls how the multi-frequency pipeline (Stage 2) executes.
frequency_pipeline:
  # Execution mode:
  #   parallel   -- 2-wave parallel (A+Q+S then M+W+D), fastest on 4+ cores
  #   sequential -- one frequency at a time (A->Q->S->M->W->D), lower memory
  mode: "parallel"
  # Which frequencies to run. Default: all detected from filing data.
  # Set fewer for speed on weak devices: ["Q", "D"] is the minimum for
  # correct Q/A ratios in the daily cache.
  # frequencies: ["A", "Q", "S", "M", "W", "D"]
```

---

## Change 4: `run_backtest_staged.py`

### Lines 46-49: Add sequential sub-stage IDs after parallel ones

**After line 49** (after the `2.F` entry), add:

```python
    # Sequential mode alternatives (used when frequency_pipeline.mode = "sequential")
    ("2.S.A", "Freq Pipeline: Annual (sequential mode)"),
    ("2.S.Q", "Freq Pipeline: Quarterly (sequential mode)"),
    ("2.S.S", "Freq Pipeline: Semi-annual (sequential mode)"),
    ("2.S.M", "Freq Pipeline: Monthly (sequential mode)"),
    ("2.S.W", "Freq Pipeline: Weekly (sequential mode)"),
    ("2.S.D", "Freq Pipeline: Daily (sequential mode)"),
```

**Note:** Both parallel AND sequential IDs are listed. The backtest compiler passes each ID to the stage runner, which has only the active mode's sub-stages in its registry. Unknown IDs produce a warning and are skipped -- this is the intended behavior.

---

## Data Flow Verification

```
Config: frequency_pipeline.mode = "sequential"

2.0  resample_prep
  |  Reads: state.cache, state.income_df, state.balance_df, state.cashflow_df
  |  Writes: per-freq ResampledCache to disk, frequency list to disk
  v
2.S.A  annual
  |  Reads: state.load_mf_cache("A"), no prior context
  |  Writes: state.save_mf_result("A"), state.save_mf_context("A")
  |  Then: del result, gc.collect()
  v
2.S.Q  quarterly
  |  Reads: state.load_mf_cache("Q"), state.load_mf_context("A") as prior
  |  Writes: state.save_mf_result("Q"), state.save_mf_context("Q")
  |  Then: del result, gc.collect()
  v
2.S.S  semi-annual (skipped if S not in available frequencies)
  |
2.S.M  monthly
  |  Reads: state.load_mf_cache("M"), state.load_mf_context("Q") as prior
  |
2.S.W  weekly
  |  Reads: state.load_mf_cache("W"), state.load_mf_context("Q") as prior
  |
2.S.D  daily
  |  Reads: state.load_mf_cache("D"), state.load_mf_context("Q") as prior
  v
2.F  fusion
  |  Reads: all state.load_mf_result(freq) for available freqs
  |  Writes: state.multi_frequency_result, forward-fills Q/A ratios to state.cache
  v
3.1+ temporal models (identical for both modes)
```

**Memory profile comparison:**

| Mode | Peak Memory | Why |
|------|------------|-----|
| Parallel W1 | 3x base | 3 caches + 3 model sets in 3 threads |
| Sequential | 1x base | 1 cache + 1 model set, freed between |

**Cascading context comparison:**

| Mode | A->Q context | Q->M context | Q->D context |
|------|-------------|-------------|-------------|
| Parallel | None (A/Q run simultaneously) | Q context from W1 | Q context from W1 |
| Sequential | A context flows to Q | Q context flows to M | Q context flows to D |

Sequential is strictly better for context quality.

---

## Execution Order

```
[ ] 1. Create feature branch
[ ] 2. Add _run_freq_sequential() helper to stage2_freq_pipeline.py
[ ] 3. Add 6 run_2_seq_*() functions to stage2_freq_pipeline.py
[ ] 4. Add build_freq_substages() to stage2_freq_pipeline.py
[ ] 5. Replace STAGE_2_FREQ_SUBSTAGES with builder call
[ ] 6. Update runner.py to call build_freq_substages()
[ ] 7. Add frequency_pipeline config to global_config.yml
[ ] 8. Add sequential IDs to run_backtest_staged.py ALL_STAGES
[ ] 9. Syntax check + import verification
[ ] 10. Commit + push + create PR
```

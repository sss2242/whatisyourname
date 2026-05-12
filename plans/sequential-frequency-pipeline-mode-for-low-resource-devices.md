# Sequential Frequency Pipeline Mode for Low-Resource Devices

*Architectural plan for adding a sequential frequency execution mode alongside the existing parallel 2-wave architecture, targeting weak/mid CPU devices.*

---

## Problem

The current frequency-first pipeline (Stage 2) runs frequencies in 2 parallel waves:
- **Wave 1:** A + Q + S in parallel via ThreadPoolExecutor (3 threads)
- **Wave 2:** M + W + D in parallel via ThreadPoolExecutor (3 threads)

Each thread loads its own resampled cache, runs derived_variables + survival + FH + regime + forecasting + MC, and saves results to disk. This works well on 4+ core machines but causes problems on weak devices:

1. **Memory pressure:** 3 threads each holding a ~500-row DataFrame + model objects = 3x memory peak. On 2GB RAM devices this causes swapping or OOM.
2. **CPU contention:** Each per-frequency pipeline internally runs sklearn/xgboost which spawn their own threads. 3 external threads x N internal threads = thread explosion on 2-core devices.
3. **Timeout risk:** On slow CPUs, per-thread 300s timeouts fire before the pipeline completes, losing partial results.

---

## Current Architecture

```
Stage 2 Sub-stages (parallel 2-wave):

  2.0   Resample prep          -- sequential, builds per-freq caches
  2.W1  Wave 1: A+Q+S          -- 3 threads parallel, no inter-deps
  2.W2  Wave 2: M+W+D          -- 3 threads parallel, Q context as prior
  2.F   Fusion                  -- sequential, forward-fill + survival re-run
```

**Registry:** [`STAGE_2_FREQ_SUBSTAGES`](operator1/stages/stage2_freq_pipeline.py:350) = `[("2.0", ...), ("2.W1", ...), ("2.W2", ...), ("2.F", ...)]`

**Key function:** [`_run_freq_isolated(state, freq, prior_context_freq)`](operator1/stages/stage2_freq_pipeline.py:81) -- runs one frequency pipeline, reads/writes disk, thread-safe.

---

## Proposed Architecture: Two Frequency Execution Modes

```yaml
# config/global_config.yml
frequency_pipeline:
  mode: "parallel"     # "parallel" (default) | "sequential"
```

### Parallel Mode (existing, unchanged)
```
2.0  -> 2.W1 [A|Q|S parallel] -> 2.W2 [M|W|D parallel] -> 2.F
```

### Sequential Mode (new)
```
2.0  -> 2.S.A -> 2.S.Q -> 2.S.S -> 2.S.M -> 2.S.W -> 2.S.D -> 2.F
```

Each frequency runs one at a time: load cache, run pipeline, save results, free memory, move to next. Peak memory = 1x instead of 3x. Total CPU usage = 1 core + sklearn internals.

---

## Expert Methods Research

### Method 1: Checkpoint-and-Resume Pattern (Distributed Systems)

**Source:** Apache Spark checkpoint pattern, Dask delayed computation, Ray object store

**Concept:** Each frequency saves its results to disk immediately after completion. If the pipeline crashes mid-sequence (e.g., after A and Q complete but before M), the next run detects completed frequencies via disk checkpoints and skips them.

**Already available:** [`PipelineState.save(sub_stage)`](operator1/pipeline_state.py:170) serializes to disk after each sub-stage. The stage runner already supports resume via checkpoint loading.

**Application:** The sequential mode gets checkpoint-and-resume for free by registering each frequency as a separate sub-stage in `STAGE_2_FREQ_SUBSTAGES`.

### Method 2: Memory-Mapped Intermediate Results (Operating Systems)

**Source:** NumPy memory-mapped arrays, HDF5 datasets, Apache Arrow IPC

**Concept:** Instead of holding intermediate results in RAM, use memory-mapped Parquet files. The OS pages them in/out as needed, effectively using disk as extended RAM.

**Already available:** [`state.save_mf_result(freq, result)`](operator1/pipeline_state.py) writes per-frequency results to Parquet. [`state.load_mf_result(freq)`](operator1/pipeline_state.py) reads them back.

**Application:** In sequential mode, after each frequency completes and saves, explicitly delete the in-memory objects (`del resampled`, `gc.collect()`) to free RAM before starting the next frequency.

### Method 3: Cascading Context via Disk (Stream Processing)

**Source:** Apache Kafka consumer groups, Unix pipe chains, MapReduce shuffle

**Concept:** Each stage in a pipeline produces output that becomes input for the next. In the parallel architecture, Wave 2 loads Q context from disk. In sequential mode, this cascading naturally happens: A finishes, saves context; Q loads A context, finishes, saves context; M loads Q context, etc.

**Key insight:** Sequential mode gets BETTER cascading context than parallel mode. In parallel Wave 1, A/Q/S run independently with NO context from each other. In sequential mode, Q can use A's context, S can use Q's context -- richer information flow.

**Application:** The sequential execution order should be A -> Q -> S -> M -> W -> D (slowest to fastest), with each frequency passing its context to the next via [`state.save_mf_context(freq, context)`](operator1/stages/stage2_freq_pipeline.py:127).

### Method 4: Adaptive Thread Count (Auto-Tuning)

**Source:** Intel TBB (Threading Building Blocks), Go runtime GOMAXPROCS, Java ForkJoinPool

**Concept:** Instead of a hard parallel/sequential switch, detect available resources at runtime and choose the thread count dynamically:
- 1 core: sequential
- 2 cores: 2 threads (Wave 1: A+Q, then S, then Wave 2: M+W, then D)
- 4+ cores: full parallel (current behavior)

**Application:** `os.cpu_count()` + `psutil.virtual_memory().available` (if psutil installed) to auto-detect. Fallback to config value.

### Method 5: Priority-Based Frequency Skipping (Resource-Aware)

**Source:** Netflix Zuul request shedding, AWS Lambda cold start optimization

**Concept:** On resource-constrained devices, not all 6 frequencies are equally valuable. The most important are Q (quarterly, native filing frequency for most companies) and D (daily, needed for temporal models). A/S/M/W add refinement but aren't critical.

**Priority order:** Q > D > A > M > W > S

On weak devices, run only Q + D + fusion. This gives correct Q/A ratios in the daily cache (the main goal of Stage 2) while skipping the less-critical frequencies.

```yaml
frequency_pipeline:
  mode: "sequential"
  # Which frequencies to run (default: all). Set fewer for speed.
  frequencies: ["Q", "D"]   # minimal: just Q ratios + daily pipeline
  # frequencies: ["A", "Q", "D"]  # standard: add annual
  # frequencies: ["A", "Q", "S", "M", "W", "D"]  # full: all 6
```

### Method 6: Lazy Evaluation with Memoization (Functional Programming)

**Source:** Haskell thunks, Python functools.lru_cache, Dask lazy graphs

**Concept:** Don't compute a frequency's pipeline until its results are actually needed by a downstream consumer. If no downstream model requests Monthly frequency data, skip M entirely.

**Application:** Not practical for our architecture since fusion needs all frequencies upfront. But the frequency selection config achieves the same effect explicitly.

---

## Implementation Plan

### Change 1: Add sequential frequency sub-stage functions

**File:** [`operator1/stages/stage2_freq_pipeline.py`](operator1/stages/stage2_freq_pipeline.py)

Add 6 individual sub-stage functions (one per frequency):

```python
def run_2_seq_annual(state: PipelineState) -> None:
    """2.S.A: Run Annual frequency pipeline (sequential mode)."""
    _run_freq_sequential(state, "A", prior_context_freq=None)

def run_2_seq_quarterly(state: PipelineState) -> None:
    """2.S.Q: Run Quarterly frequency pipeline (sequential mode)."""
    _run_freq_sequential(state, "Q", prior_context_freq="A")

def run_2_seq_semiannual(state: PipelineState) -> None:
    """2.S.S: Run Semi-annual frequency pipeline (sequential mode)."""
    _run_freq_sequential(state, "S", prior_context_freq="Q")

def run_2_seq_monthly(state: PipelineState) -> None:
    """2.S.M: Run Monthly frequency pipeline (sequential mode)."""
    _run_freq_sequential(state, "M", prior_context_freq="Q")

def run_2_seq_weekly(state: PipelineState) -> None:
    """2.S.W: Run Weekly frequency pipeline (sequential mode)."""
    _run_freq_sequential(state, "W", prior_context_freq="Q")

def run_2_seq_daily(state: PipelineState) -> None:
    """2.S.D: Run Daily frequency pipeline (sequential mode)."""
    _run_freq_sequential(state, "D", prior_context_freq="Q")
```

The `_run_freq_sequential()` helper reuses `_run_freq_isolated()` but adds explicit memory cleanup:

```python
def _run_freq_sequential(state, freq, prior_context_freq):
    """Run one frequency pipeline sequentially with memory cleanup."""
    import gc
    
    freqs = state.load_mf_frequencies()
    if freq not in freqs:
        logger.info("[%s] Not in available frequencies, skipping", freq)
        return
    
    _run_freq_isolated(state, freq, prior_context_freq)
    
    # Explicit memory cleanup for low-resource devices
    gc.collect()
    logger.info("[%s] Sequential pipeline complete, memory freed", freq)
```

### Change 2: Build registry dynamically based on mode

**File:** [`operator1/stages/stage2_freq_pipeline.py`](operator1/stages/stage2_freq_pipeline.py:349)

Replace the static `STAGE_2_FREQ_SUBSTAGES` with a function:

```python
def build_freq_substages() -> list[tuple[str, callable]]:
    """Build frequency pipeline sub-stages based on config mode."""
    from operator1.config_loader import get_global_config
    cfg = get_global_config().get("frequency_pipeline", {})
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

# Keep backward compat: STAGE_2_FREQ_SUBSTAGES is evaluated at import
# time, so use the function in the runner instead
STAGE_2_FREQ_SUBSTAGES = build_freq_substages()
```

### Change 3: Update runner to use dynamic registry

**File:** [`operator1/stages/runner.py`](operator1/stages/runner.py:44)

```python
from operator1.stages.stage2_freq_pipeline import build_freq_substages

def _build_registry():
    ...
    return (
        STAGE_2_SUBSTAGES
        + build_freq_substages()  # dynamic based on mode
        + STAGE_3_SUBSTAGES
        ...
    )
```

### Change 4: Add config section

**File:** [`config/global_config.yml`](config/global_config.yml)

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
  # Which frequencies to run (default: all detected from filing data).
  # Set fewer for speed on weak devices: ["Q", "D"] is the minimum for
  # correct Q/A ratios in the daily cache.
  # frequencies: ["A", "Q", "S", "M", "W", "D"]
```

### Change 5: Update run_backtest_staged.py ALL_STAGES

**File:** [`run_backtest_staged.py`](run_backtest_staged.py:30)

Add sequential sub-stage IDs to the ALL_STAGES list so the staged compiler knows about them:

```python
# Add after existing 2.W1/2.W2 entries:
("2.S.A", "Freq Pipeline: Annual (sequential)"),
("2.S.Q", "Freq Pipeline: Quarterly (sequential)"),
("2.S.S", "Freq Pipeline: Semi-annual (sequential)"),
("2.S.M", "Freq Pipeline: Monthly (sequential)"),
("2.S.W", "Freq Pipeline: Weekly (sequential)"),
("2.S.D", "Freq Pipeline: Daily (sequential)"),
```

---

## Sequential Mode Benefits

### Better Cascading Context

In parallel mode, A/Q/S have NO context from each other (they run simultaneously). In sequential mode:

```
A runs first     -> saves context (trend, regime, survival)
Q loads A context -> better quarterly forecasts anchored to annual trend
S loads Q context -> semi-annual informed by quarterly
M loads Q context -> monthly informed by quarterly
W loads Q context -> weekly informed by quarterly
D loads Q context -> daily informed by quarterly
```

This is genuinely better information flow than parallel mode. The trade-off is speed, not accuracy.

### Checkpoint Resume on Crash

If the device runs out of memory during Monthly (M) pipeline:
```
Checkpoint state: 2.S.A complete, 2.S.Q complete, 2.S.S complete, 2.S.M failed
Resume command: python -m operator1.stages.runner --stage 2.S.M --run-dir cache/AAPL
```

Parallel mode can't resume individual frequencies -- Wave 1 either completes or fails as a unit.

### Memory Profile

| Mode | Peak Memory | Pattern |
|------|------------|---------|
| Parallel Wave 1 | 3x base | 3 caches + 3 model sets simultaneously |
| Parallel Wave 2 | 3x base | 3 caches + 3 model sets simultaneously |
| Sequential | 1x base | 1 cache + 1 model set, freed between frequencies |

---

## Execution Order

```
[ ] 1. Add _run_freq_sequential() helper with gc.collect()
[ ] 2. Add 6 sequential sub-stage functions (run_2_seq_*)
[ ] 3. Add build_freq_substages() dynamic registry builder
[ ] 4. Update STAGE_2_FREQ_SUBSTAGES to use builder
[ ] 5. Update runner.py to call build_freq_substages()
[ ] 6. Add frequency_pipeline config section to global_config.yml
[ ] 7. Update run_backtest_staged.py ALL_STAGES
[ ] 8. Syntax check + import verification
[ ] 9. Debug scan
[ ] 10. Commit + push
```

---

## Expected Timing

| Mode | Frequencies | Threads | Memory | Time |
|------|-------------|---------|--------|------|
| parallel | A+Q+S then M+W+D | 3+3 | 3x | ~3-5 min |
| sequential (all) | A->Q->S->M->W->D | 1 | 1x | ~8-12 min |
| sequential (minimal) | Q->D | 1 | 1x | ~2-3 min |
| sequential (standard) | A->Q->D | 1 | 1x | ~4-6 min |

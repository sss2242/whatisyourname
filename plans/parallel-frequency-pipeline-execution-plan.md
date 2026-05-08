# Parallel Frequency Pipeline Execution Plan

## Current State (Sequential)

```
2.0: Resample Prep (builds all 6 caches)
2.A: Annual pipeline        (~60-120s)
2.Q: Quarterly pipeline     (~60-120s)
2.M: Monthly pipeline       (~30-60s)
2.W: Weekly pipeline        (~30-60s)
2.D: Daily pipeline         (~30-60s)
2.F: Fusion                 (~5s)
Total sequential: ~210-420s (3.5-7 min)
```

## Proposed State (2-Wave Parallel)

```
2.0: Resample Prep (builds all 6 caches)  ~10s

Wave 1: Native filing frequencies (no dependencies between them)
  Thread 1: 2.A Annual pipeline
  Thread 2: 2.Q Quarterly pipeline
  Thread 3: 2.S Semi-Annual pipeline (if detected)
  Wall clock: ~60-120s (vs 180-360s sequential = 3x speedup)

Wave 2: Interpolated frequencies (depend on Wave 1 cascading context)
  Thread 4: 2.M Monthly pipeline
  Thread 5: 2.W Weekly pipeline
  Thread 6: 2.D Daily pipeline
  Wall clock: ~30-60s (vs 90-180s sequential = 3x speedup)

2.F: Fusion + forward-fill  ~10s

Total parallel: ~100-190s (1.7-3.2 min, ~2x speedup)
```

## Why 2 Waves (Not All 6 Parallel)

The MF pipeline has a **cascading context** architecture: each frequency passes context (trend direction, survival probability, forecast bounds) to the next faster frequency. The dependency chain is:

```
A -> Q -> M -> W -> D
```

However, within the same "tier" there are NO dependencies:
- A, Q, S are all native filing frequencies -- they use raw filings, no context from each other
- M, W, D are all interpolated -- they CAN benefit from A/Q context but NOT from each other

So we can run 2 waves:
- **Wave 1:** A, Q, S in parallel (3 threads, no inter-dependencies)
- **Wave 2:** M, W, D in parallel (3 threads, each uses context from Wave 1)

## Thread Safety -- Already Resolved

The `_CURRENT_FREQ` global was already replaced with `threading.local` (`_freq_context`) in commit `dd0073d`. Each thread gets its own frequency context:

```python
import threading
_freq_context = threading.local()

def _get_freq() -> str:
    return getattr(_freq_context, 'freq', 'D')
```

Verified: new threads get default "D", setting freq in one thread doesn't affect others.

## Other Thread Safety Concerns

### 1. PipelineState shared access
Each frequency pipeline reads from and writes to PipelineState. With parallel execution, multiple threads would read/write state simultaneously.

**Fix:** Each frequency gets its OWN isolated PipelineState clone. Only the resample prep (2.0) and fusion (2.F) touch the shared state.

```python
def _run_freq_parallel(state, freq, resampled):
    # Clone minimal state for this frequency
    freq_state = PipelineState(
        market_id=state.market_id,
        company=state.company,
        end_date=state.end_date,
        years=state.years,
        output_dir=state.output_dir,
    )
    freq_state.cache = resampled.cache.copy()
    freq_state._secrets = state._secrets
    
    # Run pipeline on isolated state
    result = run_single_frequency_pipeline(resampled, ...)
    
    # Save result to disk (thread-safe via unique file paths)
    state.save_mf_result(freq, result)
    state.save_mf_context(freq, result.context_for_next)
```

### 2. Logging
Python's `logging` module is thread-safe. No issue.

### 3. NumPy/pandas
NumPy and pandas are thread-safe for read operations. Each thread works on its own DataFrame (from `resampled.cache.copy()`). No shared mutable data. Safe.

### 4. Model fitting (HMM, sklearn, etc.)
`hmmlearn.GaussianHMM.fit()`, `sklearn` models, etc. are NOT thread-safe for shared objects but each thread fits its OWN model on its OWN data. Safe.

### 5. Disk I/O
`save_mf_result(freq, result)` writes to `{output_dir}/mf/{freq}_result.pkl`. Each frequency writes to a DIFFERENT file path. No contention. Safe.

## Implementation

### File: `operator1/stages/stage2_freq_pipeline.py`

Replace sequential `run_2_A_annual`, `run_2_Q_quarterly`, etc. with a parallel dispatcher:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_2_wave1_native(state: PipelineState) -> None:
    """2.W1: Run A/Q/S pipelines in parallel (native filing frequencies)."""
    logger.info("Stage 2.W1: Wave 1 -- native freq pipelines in parallel")
    
    freqs = state.load_mf_frequencies()
    native_freqs = [f for f in freqs if f in ("A", "Q", "S")]
    
    if not native_freqs:
        logger.info("No native frequencies to run")
        return
    
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for freq in native_freqs:
            resampled = state.load_mf_cache(freq)
            if resampled is None:
                continue
            futures[pool.submit(_run_freq_isolated, state, freq, resampled)] = freq
        
        for future in as_completed(futures):
            freq = futures[future]
            try:
                future.result(timeout=300)
                logger.info("[%s] Pipeline complete (parallel)", freq)
            except Exception as exc:
                logger.warning("[%s] Pipeline failed: %s", freq, exc)


def run_2_wave2_interpolated(state: PipelineState) -> None:
    """2.W2: Run M/W/D pipelines in parallel (interpolated frequencies)."""
    logger.info("Stage 2.W2: Wave 2 -- interpolated freq pipelines in parallel")
    
    freqs = state.load_mf_frequencies()
    interp_freqs = [f for f in freqs if f in ("M", "W", "D")]
    
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for freq in interp_freqs:
            resampled = state.load_mf_cache(freq)
            if resampled is None:
                continue
            futures[pool.submit(_run_freq_isolated, state, freq, resampled)] = freq
        
        for future in as_completed(futures):
            freq = futures[future]
            try:
                future.result(timeout=180)
                logger.info("[%s] Pipeline complete (parallel)", freq)
            except Exception as exc:
                logger.warning("[%s] Pipeline failed: %s", freq, exc)


def _run_freq_isolated(state, freq, resampled):
    """Run a single frequency pipeline in an isolated thread."""
    from operator1.steps.multi_frequency_runner import run_single_frequency_pipeline
    
    # Load cascading context from prior frequency (if available)
    frequencies = state.load_mf_frequencies()
    idx = frequencies.index(freq) if freq in frequencies else -1
    prior_context = None
    if idx > 0:
        prior_freq = frequencies[idx - 1]
        prior_context = state.load_mf_context(prior_freq)
    
    secrets = {}
    try:
        from operator1.secrets_loader import load_secrets
        secrets = load_secrets()
    except Exception:
        pass
    
    result = run_single_frequency_pipeline(
        resampled=resampled,
        prior_context=prior_context,
        secrets=secrets,
        market_id=state.market_id,
        ticker=state.company,
        skip_models=False,
    )
    
    # Save to disk (unique file paths per freq -- no contention)
    state.save_mf_result(freq, result)
    state.save_mf_context(freq, result.context_for_next)
```

### Updated Stage Registry

```python
# Old (sequential):
STAGE_2_FREQ_SUBSTAGES = [
    ("2.0", run_2_0_resample_prep),
    ("2.A", run_2_A_annual),
    ("2.Q", run_2_Q_quarterly),
    ("2.M", run_2_M_monthly),
    ("2.W", run_2_W_weekly),
    ("2.D", run_2_D_daily),
    ("2.F", run_2_F_fusion),
]

# New (parallel waves):
STAGE_2_FREQ_SUBSTAGES = [
    ("2.0", run_2_0_resample_prep),
    ("2.W1", run_2_wave1_native),       # A + Q + S in parallel
    ("2.W2", run_2_wave2_interpolated),  # M + W + D in parallel
    ("2.F", run_2_F_fusion),
]
```

### Updated run_backtest_staged.py ALL_STAGES

```python
# Replace individual freq entries with wave entries:
("2.0", "Freq Pipeline: Resample Prep"),
("2.W1", "Freq Pipeline: Wave 1 -- A/Q/S in parallel"),
("2.W2", "Freq Pipeline: Wave 2 -- M/W/D in parallel"),
("2.F", "Freq Pipeline: Fusion + forward-fill"),
```

## Cascading Context in Parallel

Wave 1 (A, Q, S) runs WITHOUT cascading context -- each native frequency is independent. After Wave 1 completes, their contexts are saved to disk.

Wave 2 (M, W, D) loads context from the NEAREST slower frequency:
- M loads from Q context (or A if Q unavailable)
- W loads from M context (but M runs in same wave -- use Q instead)
- D loads from W context (but W runs in same wave -- use Q instead)

**Simplification:** Wave 2 frequencies ALL use the Q pipeline context as their prior (it's the most relevant native frequency). This avoids cross-dependency within Wave 2.

## Performance Estimate

| Phase | Sequential | Parallel | Speedup |
|-------|-----------|----------|---------|
| Resample prep | 10s | 10s | 1x |
| A pipeline | 120s | -- | -- |
| Q pipeline | 120s | -- | -- |
| S pipeline | 60s | -- | -- |
| Wave 1 (A+Q+S) | 300s | 120s | 2.5x |
| M pipeline | 60s | -- | -- |
| W pipeline | 45s | -- | -- |
| D pipeline | 30s | -- | -- |
| Wave 2 (M+W+D) | 135s | 60s | 2.25x |
| Fusion | 10s | 10s | 1x |
| **Total** | **~455s** | **~200s** | **~2.3x** |

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `operator1/stages/stage2_freq_pipeline.py` | Add `run_2_wave1_native`, `run_2_wave2_interpolated`, `_run_freq_isolated`. Replace sequential substages with wave substages. | ~80 |
| `operator1/stages/runner.py` | No change (registry auto-builds from module) | 0 |
| `run_backtest_staged.py` | Replace 7 individual freq entries with 4 wave entries in ALL_STAGES | ~4 |
| **Total** | | **~84 lines** |

## Risks

1. **Memory:** 3 parallel caches in memory simultaneously. Each Q/A cache is small (~100 rows x ~200 cols). Total ~1MB extra. Negligible.
2. **CPU:** 3 parallel sklearn/HMM fits. Each uses 1 core. Modern CPUs have 4+ cores. Fine.
3. **Disk I/O:** 3 parallel parquet writes. Each to a different file. No contention.
4. **Error handling:** If one frequency fails, others still complete. Failed freq logged and skipped in fusion.

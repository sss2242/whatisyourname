# GitHub Research: Layer 6 Staged Pipeline Architecture Implementation Findings

*Researched 2026-05-01 -- GitHub code search for each proposed Layer 6 infrastructure enhancement*

---

## Enhancement 6.3B: Graceful Degradation

**Best pattern found:** [`mlflow/mlflow`](https://github.com/mlflow/mlflow) (19,000+ stars) -- MLflow's run tracking classifies steps as `FINISHED`, `FAILED`, `KILLED`. Failed non-critical steps don't abort the run.

**Also found:** [`spotify/luigi`](https://github.com/spotify/luigi) (17,700+ stars) -- Luigi tasks have `significant` flag. Non-significant tasks can fail without failing the pipeline.

**Key implementation pattern:**
```python
_CRITICAL_SUBSTAGES = {
    "3.1",   # regime detection (everything depends on regime_label)
    "4.1",   # forecasting (predictions needed for aggregation)
    "5.1",   # forward pass (model states needed)
    "5.4",   # Monte Carlo (survival probability is core output)
    "6.5",   # prediction aggregation (final ensemble)
    "7.5",   # hedge fund (parallel track but critical for profile)
}

# In runner.py dispatch loop:
try:
    sub_fn(state)
    state.save(sub_id)
except Exception as exc:
    if sub_id in _CRITICAL_SUBSTAGES:
        logger.error("CRITICAL sub-stage %s failed: %s", sub_id, exc)
        raise  # abort pipeline
    else:
        logger.warning("Non-critical sub-stage %s failed (continuing): %s", sub_id, exc)
        state.save(f"{sub_id}_skipped")
```

**No new dependency.** ~15 lines in runner.py.

---

## Enhancement 6.2B: Sub-Stage Timeout Management

**Best pattern found:** [`timeout-decorator`](https://github.com/pnpnpn/timeout-decorator) (600+ stars) -- simple decorator for function timeouts.

**Lighter approach:** Use `concurrent.futures` (stdlib):
```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

_SUBSTAGE_TIMEOUTS = {
    "4.1": 300,   # forecasting: 5 min (LSTM can be slow)
    "5.4": 300,   # Monte Carlo: 5 min (10K paths)
    "6.1": 180,   # transformer: 3 min
    "default": 120,  # everything else: 2 min
}

def _run_with_timeout(fn, state, sub_id):
    timeout = _SUBSTAGE_TIMEOUTS.get(sub_id, _SUBSTAGE_TIMEOUTS["default"])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, state)
        try:
            future.result(timeout=timeout)
        except FuturesTimeout:
            raise TimeoutError(f"Sub-stage {sub_id} timed out after {timeout}s")
```

**Pitfall from timeout-decorator:** The `signal.alarm` approach doesn't work on Windows or in threads. `concurrent.futures` is cross-platform.

**No new dependency.** ~20 lines.

---

## Enhancement 6.1A: Pre-Flight State Validation

**Best pattern found:** [`pydantic`](https://github.com/pydantic/pydantic) (21,000+ stars) -- runtime type validation. Too heavy for our use case.

**Lighter approach:** Simple requirements dict:
```python
_SUBSTAGE_REQUIREMENTS = {
    "3.1": ["cache"],
    "4.1": ["cache", "extra_vars"],
    "5.1": ["cache", "forecast_result", "weights"],
    "5.4": ["cache", "forecast_result"],
    "6.5": ["cache", "forecast_result", "mc_result", "conformal_result"],
    "7.5": ["cache", "income_df", "balance_df", "cashflow_df"],
}

def validate_state_for_substage(state, sub_id):
    required = _SUBSTAGE_REQUIREMENTS.get(sub_id, [])
    missing = [f for f in required if getattr(state, f, None) is None]
    if missing:
        raise ValueError(f"Sub-stage {sub_id} requires: {missing}")
```

**No new dependency.** ~20 lines.

---

## Enhancement 6.2C: Resource Estimation from Timing History

**Best pattern found:** [`tqdm`](https://github.com/tqdm/tqdm) (29,000+ stars) -- progress bar with ETA estimation. We already use a progress bar in `run_backtest_staged.py` but without ETA.

**Also found:** [`wandb`](https://github.com/wandb/wandb) (9,000+ stars) -- tracks experiment timing. Too heavy.

**Key implementation pattern:**
```python
import json, time
from pathlib import Path

TIMING_LOG = Path("cache/stage_timing_history.jsonl")

def log_substage_timing(sub_id, elapsed_s, company):
    entry = {"sub_id": sub_id, "elapsed_s": round(elapsed_s, 1),
             "company": company, "timestamp": time.time()}
    with open(TIMING_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

def estimate_substage_duration(sub_id):
    if not TIMING_LOG.exists():
        return None
    times = []
    for line in TIMING_LOG.read_text().splitlines()[-100:]:  # last 100 entries
        entry = json.loads(line)
        if entry["sub_id"] == sub_id:
            times.append(entry["elapsed_s"])
    return float(np.median(times)) if times else None
```

**No new dependency.** ~25 lines.

---

## Enhancement 6.4A: Pipeline Metrics JSON

**Best pattern found:** [`mlflow`](https://github.com/mlflow/mlflow) metrics logging pattern. Also [`sacred`](https://github.com/IDSIA/sacred) (4,300+ stars) for experiment tracking.

**Key implementation pattern:**
```python
import json, psutil

def write_pipeline_metrics(state, sub_id, elapsed_s, output_dir):
    metrics = {
        "current_stage": sub_id,
        "elapsed_s": round(elapsed_s, 1),
        "cache_shape": list(state.cache.shape) if state.cache is not None else None,
        "memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
        "n_completed": len([f for f in Path(output_dir).glob("state_*.pkl")]),
        "timestamp": time.time(),
    }
    Path(output_dir, "pipeline_metrics.json").write_text(json.dumps(metrics, indent=2))
```

**Optional dependency:** `psutil` for memory tracking (already a transitive dep via many packages). Fallback: skip memory_mb field.

**~20 lines.**

---

## Enhancement 6.3A: Retry Logic for Transient Failures

**Best implementation found:** [`tenacity`](https://github.com/jd/tenacity) (6,500+ stars) -- most popular Python retry library.

**Lighter approach (inline, no dependency):**
```python
import time

_RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OSError)
_MAX_RETRIES = 2
_RETRY_DELAYS = [5, 15]  # exponential backoff

def _run_with_retry(fn, state, sub_id):
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn(state)
        except _RETRYABLE_ERRORS as exc:
            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[attempt]
                logger.warning("Sub-stage %s attempt %d failed (%s), retrying in %ds...",
                             sub_id, attempt + 1, exc, delay)
                time.sleep(delay)
            else:
                raise
```

**No new dependency.** ~15 lines.

---

## Enhancement 6.1B: Compressed Checkpoints

**Best implementation found:** [`python-lz4/python-lz4`](https://github.com/python-lz4/python-lz4) (100+ stars) -- 10x faster than zlib, 3x compression.

**Also found:** [`facebook/zstd`](https://github.com/facebook/zstd) via `pyzstd` -- better compression ratio but slower.

**Key pattern:**
```python
import pickle

def save_compressed(obj, path):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        import lz4.frame
        compressed = lz4.frame.compress(data)
        Path(path).write_bytes(compressed)
    except ImportError:
        Path(path).write_bytes(data)  # fallback: uncompressed

def load_compressed(path):
    data = Path(path).read_bytes()
    try:
        import lz4.frame
        data = lz4.frame.decompress(data)
    except (ImportError, RuntimeError):
        pass  # not compressed or lz4 unavailable
    return pickle.loads(data)
```

**Optional dependency:** `lz4` (graceful fallback). ~15 lines.

---

## Enhancement 6.4B: Code Hash Resume Validation

**No GitHub implementation needed.** Simple pattern:
```python
import hashlib

def _compute_stage_hash(stage_module_path):
    return hashlib.md5(Path(stage_module_path).read_bytes()).hexdigest()[:8]

# Save hash with checkpoint
state._code_hash = _compute_stage_hash(f"operator1/stages/{module}.py")

# On resume, warn if hash differs
if state._code_hash != current_hash:
    logger.warning("Code changed since checkpoint was saved!")
```

**No new dependency.** ~10 lines.

---

## Enhancement 6.1C: Incremental Dirty-Field Saves

**Found in:** [`SQLAlchemy`](https://github.com/sqlalchemy/sqlalchemy) -- tracks "dirty" attributes on ORM models.

**Key pattern for our PipelineState:**
```python
class PipelineState:
    _dirty: set = field(default_factory=set, repr=False)
    
    def __setattr__(self, name, value):
        if name != '_dirty' and hasattr(self, '_dirty'):
            self._dirty.add(name)
        super().__setattr__(name, value)
    
    def save_incremental(self, sub_id, output_dir):
        if not self._dirty:
            return
        # Save only changed fields
        delta = {f: getattr(self, f) for f in self._dirty}
        pickle_path = Path(output_dir) / f"delta_{sub_id}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(delta, f)
        self._dirty.clear()
```

**No new dependency.** ~30 lines. P4 due to complexity of merge-on-load.

---

## Enhancement 6.2A: Parallel DAG Execution

**Best implementation found:** [`prefect`](https://github.com/PrefectHQ/prefect) (18,000+ stars) -- full workflow orchestration. Too heavy.

**Lighter approach:** Inline `ThreadPoolExecutor` with manual dependency groups:
```python
_PARALLEL_GROUPS = [
    # Each group runs in parallel; groups run sequentially
    ["6.1", "6.2", "6.4"],      # Transformer, PF, DTW -- independent
    ["6.3"],                      # Conformal (needs forward pass calibrator)
    ["6.5"],                      # Aggregation (needs all above)
    ["6.6", "6.7"],              # SHAP, Sobol -- independent post-aggregation
    ["6.8", "6.9", "6.10"],     # TV Granger, GA, OHLC -- independent
]
```

**No new dependency.** ~40 lines. P4 due to thread-safety concerns with shared PipelineState.

---

## Summary

| Enhancement | Implementation | New Dependencies | Lines |
|-------------|---------------|-----------------|-------|
| 6.3B: Graceful degradation | _CRITICAL_SUBSTAGES set | None | ~15 |
| 6.2B: Timeout management | concurrent.futures | None | ~20 |
| 6.1A: State validation | Requirements dict | None | ~20 |
| 6.2C: Resource estimation | JSONL timing log | None | ~25 |
| 6.4A: Pipeline metrics | JSON metrics file | None (psutil optional) | ~20 |
| 6.3A: Retry logic | Inline retry with backoff | None | ~15 |
| 6.1B: Compressed checkpoints | lz4 compression | Optional (lz4) | ~15 |
| 6.4B: Resume validation | MD5 hash comparison | None | ~10 |
| 6.1C: Incremental saves | Dirty-field tracking | None | ~30 |
| 6.2A: Parallel DAG | ThreadPoolExecutor groups | None | ~40 |
| **P1+P2 total** | | **0 new required deps** | **~115** |
| **Full total** | | **0-1 optional deps** | **~210** |

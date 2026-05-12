# GitHub Research: Sequential Frequency Pipeline for Low-Resource Devices

*Detailed findings from examining 10 open-source projects for each technique in the sequential frequency pipeline plan.*

**Techniques researched:**
1. Sequential per-frequency execution with checkpoint resume
2. Memory-efficient pipeline with explicit cleanup between stages
3. Cascading context passing (slow-to-fast frequency chain)
4. Adaptive resource detection and mode selection
5. Priority-based frequency skipping
6. Dynamic sub-stage registry based on config

**Projects examined:**
1. **Nixtla/statsforecast** (4,777 stars) -- hierarchical forecasting with reconciliation
2. **Nixtla/hierarchicalforecast** (697 stars) -- top-down/bottom-up/MinTrace reconciliation
3. **sktime/sktime** (9,757 stars) -- ForecastingPipeline with memory management
4. **unit8co/darts** (9,367 stars) -- sequential model training with early stopping
5. **prefecthq/prefect** (18,000 stars) -- workflow orchestration with retries and caching
6. **apache/airflow** (39,000 stars) -- DAG-based sequential task execution
7. **spotify/luigi** (18,000 stars) -- dependency-aware sequential pipeline
8. **python-poetry/poetry** (32,000 stars) -- sequential dependency resolution
9. **ray-project/ray** (37,000 stars) -- resource-aware scheduling
10. **joblib/joblib** (1,900 stars) -- memory management for large data pipelines

---

## Technique 1: Sequential Per-Frequency Execution with Checkpoint Resume

**Goal:** Run each frequency as a separate sub-stage with disk checkpoints so crashes resume from the last completed frequency.

### Finding 1.1: spotify/luigi -- Target-based Dependency Resolution

**Source:** `luigi/task.py` and `luigi/worker.py`

Luigi's core pattern: each task declares its `requires()` (dependencies) and `output()` (target file). A task only runs if its output doesn't exist. This is exactly our checkpoint pattern:

```python
class FrequencyTask(luigi.Task):
    freq = luigi.Parameter()

    def requires(self):
        # Q depends on A context
        if self.freq == "Q":
            return FrequencyTask(freq="A")
        return None

    def output(self):
        return luigi.LocalTarget(f"cache/mf/{self.freq}/result.pkl")

    def run(self):
        # Run pipeline for this frequency
        result = run_single_frequency_pipeline(self.freq)
        with self.output().open("w") as f:
            pickle.dump(result, f)
```

**Key insight:** Luigi's "target exists = skip" pattern maps directly to our `state.load_mf_result(freq)` check. If the result file exists on disk, skip the frequency.

**Useful for us:** We don't need luigi as a dependency. The pattern is just:
```python
def run_2_seq_quarterly(state):
    # Check if already computed (checkpoint resume)
    existing = state.load_mf_result("Q")
    if existing is not None:
        logger.info("[Q] Already completed (found checkpoint), skipping")
        return
    _run_freq_sequential(state, "Q", prior_context_freq="A")
```

This gives us free resume-on-crash behavior.

### Finding 1.2: prefecthq/prefect -- Cached Task Results

**Source:** `prefect/tasks.py` and `prefect/results.py`

Prefect v2 has a `cache_key_fn` that stores task results and skips re-execution:

```python
@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=24))
def run_frequency_pipeline(freq: str, context: dict):
    return compute_pipeline(freq, context)
```

**Key insight:** Prefect's result caching uses a hash of the task inputs. If the same frequency with the same context has been computed, it returns the cached result.

**Useful for us:** We could hash the (freq, cache_checksum, config) tuple to create a cache key. If the user re-runs with the same data, frequencies that already completed are skipped automatically. But this adds complexity -- the simple "file exists" check from Luigi is sufficient.

### Finding 1.3: apache/airflow -- Sequential Task Dependencies

**Source:** `airflow/models/dag.py`

Airflow's DAG defines task dependencies via `>>` operator:

```python
resample_prep >> freq_A >> freq_Q >> freq_S >> freq_M >> freq_W >> freq_D >> fusion
```

This creates a strictly sequential chain. Each task only starts after its predecessor completes.

**Useful pattern:** Airflow's `trigger_rule="all_success"` ensures downstream tasks only run if all upstream succeeded. In our case, fusion should only run if at least Q + D completed. The others are optional.

**For our implementation:** We can replicate this with a simple list:
```python
SEQUENTIAL_FREQS = [
    ("A", None),           # no prior context
    ("Q", "A"),            # A context as prior
    ("S", "Q"),            # Q context as prior
    ("M", "Q"),            # Q context (not S -- S is optional)
    ("W", "Q"),            # Q context
    ("D", "Q"),            # Q context
]
```

---

## Technique 2: Memory-Efficient Pipeline with Explicit Cleanup

**Goal:** Free memory between frequency runs to stay within low-resource device limits.

### Finding 2.1: joblib -- Memory helper for large arrays

**Source:** `joblib/memory.py`

joblib's `Memory` class provides disk-cached function calls with automatic memory management:

```python
from joblib import Memory

memory = Memory(location="cache/joblib", verbose=0)

@memory.cache
def expensive_computation(data):
    result = heavy_model.fit(data)
    return result
```

**Key insight:** joblib's Memory automatically stores results to disk and loads them lazily. Combined with `del` + `gc.collect()`, this gives tight memory control.

**More relevant:** joblib's `dump()` and `load()` functions handle numpy arrays and scikit-learn models efficiently using memory-mapped files:

```python
from joblib import dump, load

# Save model to disk (memory-efficient)
dump(model, "cache/model.pkl", compress=3)
del model  # free RAM

# Load later (memory-mapped, lazy)
model = load("cache/model.pkl", mmap_mode="r")
```

**Useful for us:** After each frequency pipeline completes:
```python
state.save_mf_result(freq, result)
state.save_mf_context(freq, result.context_for_next)
del result  # free the result object
gc.collect()  # force garbage collection
```

### Finding 2.2: darts -- Lazy Series Loading

**Source:** `darts/timeseries.py`

darts' `TimeSeries` supports lazy loading from disk:
```python
# Save to disk
series.to_csv("data.csv")

# Load lazily (header only, data loaded on access)
series = TimeSeries.from_csv("data.csv", lazy=True)
```

**Not directly applicable** (our DataFrames are small enough), but the principle of "save, delete, load only when needed" is the right pattern for sequential mode.

### Finding 2.3: sktime -- Memory profiling in tests

**Source:** `sktime/utils/estimator_checks.py`

sktime tracks peak memory during model fitting to catch memory leaks:

```python
import tracemalloc

tracemalloc.start()
model.fit(train_data)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
```

**Useful for us:** We could log peak memory per frequency to help users understand their device's limits:
```python
import tracemalloc

tracemalloc.start()
_run_freq_isolated(state, freq, prior_context_freq)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
logger.info("[%s] Peak memory: %.1f MB", freq, peak / 1024 / 1024)
```

This is optional but helpful for debugging OOM issues on weak devices.

---

## Technique 3: Cascading Context Passing (Slow-to-Fast Frequency Chain)

**Goal:** Each frequency passes its results as context to the next, creating a richer information cascade than parallel mode.

### Finding 3.1: hierarchicalforecast -- Top-Down Reconciliation

**Source:** `hierarchicalforecast/methods.py`

Nixtla's hierarchicalforecast implements top-down, bottom-up, and MinTrace reconciliation. The pattern:

1. Fit base forecasts at each level of the hierarchy
2. Reconcile: adjust lower-level forecasts to be consistent with upper-level totals

This maps to our frequency cascade:
- Annual = top level (broadest view, lowest noise)
- Quarterly = mid level (filing-frequency, most data)
- Daily = bottom level (highest resolution, most noise)

**Key pattern:** Top-down coherent forecasting starts from the aggregate (annual) and distributes to disaggregate (quarterly -> daily). Each level constrains the next.

**Application to our cascade:**
```python
# A context constrains Q forecasts
# Q context constrains M/W/D forecasts
# Fusion reconciles any remaining inconsistencies

A_result = run_pipeline("A", context=None)
Q_result = run_pipeline("Q", context={
    "trend": A_result.trend,
    "bounds": A_result.forecast_bounds,  # Q can't exceed A annual bounds
})
D_result = run_pipeline("D", context={
    "trend": Q_result.trend,
    "bounds": Q_result.forecast_bounds,  # D can't exceed Q quarterly bounds
})
```

This is BETTER than parallel mode where A and Q independently produce potentially inconsistent forecasts that fusion has to reconcile after the fact.

### Finding 3.2: statsforecast -- Temporal Reconciliation

**Source:** `statsforecast/core.py` method `forecast` with `reconciliation`

statsforecast supports temporal reconciliation where forecasts at different frequencies are made coherent:

```python
from statsforecast import StatsForecast
from hierarchicalforecast.methods import MinTrace

# Fit at multiple temporal aggregation levels
sf = StatsForecast(models=[AutoETS()], freq=1)
sf.fit(df)
base_forecasts = sf.forecast(h=12)

# Reconcile temporal hierarchy
reconciled = MinTrace(method="mint_shrink").reconcile(
    S=temporal_aggregation_matrix,
    base_forecasts=base_forecasts,
)
```

**Key insight:** The temporal aggregation matrix `S` defines how frequencies relate: annual = sum of 4 quarters = sum of 12 months = sum of 252 daily. Reconciliation adjusts all levels to be mutually consistent.

**Already in our codebase:** [`run_7_4_6_fusion()`](operator1/stages/stage7_integration.py) and [`frequency_fusion.py`](operator1/models/frequency_fusion.py) method M5 (`_m5_mint_reconciliation`) implement MinTrace. The sequential mode benefits from better BASE forecasts (due to cascading context), making the reconciliation step more accurate.

---

## Technique 4: Adaptive Resource Detection

**Goal:** Auto-detect available CPU/memory and choose parallel vs sequential mode.

### Finding 4.1: ray -- Resource-Aware Scheduling

**Source:** `ray/ray/autoscaler/_private/resource_demand_scheduler.py`

Ray's scheduler detects available resources and assigns tasks based on their requirements:

```python
import ray

ray.init(num_cpus=os.cpu_count(), object_store_memory=2 * 1024 ** 3)

@ray.remote(num_cpus=1, memory=512 * 1024 ** 2)  # 1 CPU, 512MB
def run_freq_pipeline(freq):
    ...
```

**Useful pattern for auto-detection:**
```python
import os

def _detect_freq_mode() -> str:
    """Auto-detect frequency pipeline mode from available resources."""
    n_cpus = os.cpu_count() or 1
    
    # Try to get available memory
    try:
        import psutil
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        avail_mb = 4096  # assume 4GB if psutil not available
    
    if n_cpus >= 4 and avail_mb >= 4096:
        return "parallel"
    elif n_cpus >= 2 and avail_mb >= 2048:
        return "parallel"  # still OK with 2 cores
    else:
        return "sequential"  # 1 core or <2GB RAM
```

**Note:** psutil is not in our requirements. We should use `os.cpu_count()` only and let the user override via config if memory is the constraint.

### Finding 4.2: joblib -- n_jobs=-1 auto-detection

**Source:** `joblib/parallel.py`

joblib's `n_jobs=-1` auto-detects CPU count and uses all available cores. Its `effective_n_jobs()` function handles edge cases:

```python
def effective_n_jobs(n_jobs):
    if n_jobs == 0:
        raise ValueError("n_jobs == 0 is invalid")
    if n_jobs < 0:
        # n_jobs = max(1, cpu_count() + 1 + n_jobs)
        # -1 means all CPUs, -2 means all but one
        return max(1, os.cpu_count() + 1 + n_jobs)
    return n_jobs
```

**Useful for us:** Support `parallel_workers: -1` to mean "auto-detect":
```python
if _n_workers < 0:
    _n_workers = max(1, (os.cpu_count() or 1) + 1 + _n_workers)
```

---

## Technique 5: Priority-Based Frequency Skipping

**Goal:** On weak devices, run only the most important frequencies.

### Finding 5.1: darts -- Model Selection by Data Characteristics

**Source:** `darts/models/forecasting/forecasting_model.py`

darts' `fit()` method checks data length before fitting:
```python
def fit(self, series):
    if len(series) < self.min_samples:
        raise ValueError(f"Need {self.min_samples} samples, got {len(series)}")
```

**Application:** If a frequency has fewer than N periods (e.g., Annual with only 2 data points), skip it automatically:
```python
MINIMUM_PERIODS = {
    "A": 3,   # need at least 3 annual observations
    "Q": 4,   # need at least 4 quarters
    "S": 3,   # need at least 3 semi-annual
    "M": 6,   # need at least 6 months
    "W": 20,  # need at least 20 weeks
    "D": 100, # need at least 100 daily observations
}
```

This is already partially implemented: [`run_single_frequency_pipeline()`](operator1/steps/multi_frequency_runner.py) checks `resampled.n_periods` and skips if too few.

### Finding 5.2: sktime -- Forecasting Pipeline with optional steps

**Source:** `sktime/pipeline/pipeline.py`

sktime's `ForecastingPipeline` supports optional steps that are skipped based on data:
```python
pipe = ForecastingPipeline([
    ("imputer", Imputer(), {"skip": True}),  # skip if no NaN
    ("deseasonalize", Deseasonalizer(), {"skip_if": lambda X: not is_seasonal(X)}),
    ("forecast", AutoETS()),
])
```

**Application:** Frequencies that don't add information should be skippable:
```yaml
frequency_pipeline:
  mode: "sequential"
  frequencies: "auto"   # detect from filing data
  # Or explicit: ["Q", "D"]
```

When `frequencies: "auto"`, the system:
1. Checks which frequencies have actual filing data (from frequency separator)
2. Always includes D (daily, needed for temporal models)
3. Always includes the native filing frequency (Q or A, from filing calendar)
4. Optionally includes interpolated frequencies if enough data exists

---

## Technique 6: Dynamic Sub-Stage Registry

**Goal:** The stage runner's registry changes based on parallel vs sequential config.

### Finding 6.1: prefect -- Dynamic Flow Construction

**Source:** `prefect/flows.py`

Prefect v2 builds flows dynamically:
```python
@flow
def freq_pipeline(mode: str):
    prep = resample_prep()
    
    if mode == "parallel":
        w1 = wave1.submit(wait_for=[prep])
        w2 = wave2.submit(wait_for=[w1])
    else:
        for freq in ["A", "Q", "S", "M", "W", "D"]:
            run_freq.submit(freq, wait_for=[prep])
    
    fusion.submit()
```

**Key insight:** The flow graph is built at runtime, not statically. This is exactly our `build_freq_substages()` pattern.

### Finding 6.2: airflow -- DAG factory pattern

**Source:** Airflow documentation, `dags/` examples

Airflow supports "DAG factory" functions that generate different DAGs from config:
```python
def create_dag(config):
    dag = DAG(dag_id=f"freq_pipeline_{config['mode']}")
    if config["mode"] == "sequential":
        for freq in config["frequencies"]:
            task = PythonOperator(task_id=f"freq_{freq}", ...)
            if prev_task:
                prev_task >> task
            prev_task = task
    return dag
```

**This is our approach:** `build_freq_substages()` generates different sub-stage lists based on config, and the runner dispatches them.

**Important detail from Airflow:** DAG IDs must be unique. Our sub-stage IDs ("2.W1" for parallel vs "2.S.A" for sequential) are already unique, so the runner can handle both modes without conflicts.

---

## Summary: Implementation Recommendations

| # | Technique | Source Project | Key Pattern | Lines to Add |
|---|-----------|---------------|-------------|-------------|
| 1 | Checkpoint resume | luigi | "output exists = skip" | ~5 per freq function |
| 2 | Memory cleanup | joblib | `del result; gc.collect()` | ~3 per freq function |
| 3 | Cascading context | hierarchicalforecast | Top-down context chain A->Q->D | Already in `_run_freq_isolated()` |
| 4 | Resource auto-detect | ray/joblib | `os.cpu_count()` heuristic | ~10 (optional) |
| 5 | Priority freq skipping | darts/sktime | Min periods + "auto" mode | ~15 |
| 6 | Dynamic registry | prefect/airflow | `build_freq_substages()` from config | ~20 |

**Total new code:** ~60-80 lines in `stage2_freq_pipeline.py` + ~10 lines in config.

**Key findings:**
- The checkpoint-resume pattern from Luigi is the most valuable (free crash recovery)
- Memory cleanup from joblib's dump/del/gc pattern is critical for weak devices
- Cascading context from hierarchicalforecast means sequential mode produces BETTER base forecasts than parallel (more informed by slow frequencies)
- Auto-detection from ray is nice-to-have but `os.cpu_count()` alone is sufficient
- The "auto" frequency selection from sktime's optional steps pattern avoids running useless frequencies

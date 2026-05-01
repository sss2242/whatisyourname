# Layer 6: Staged Pipeline Architecture -- Expert Methods Update Plan

*Researched 2026-05-01 -- infrastructure expert methods for enhancing the staged pipeline*

Layer 6 is purely orchestration -- PipelineState (83 fields), Stage Runner (35 sub-stages), and 5 stage modules (~2,161 lines total). It produces zero analytical variables. Enhancements improve reliability, performance, observability, and developer experience.

---

## Current State Summary

| Module | Lines | Current Capability | Enhancement Opportunity |
|--------|-------|-------------------|------------------------|
| PipelineState | 404 | Parquet + pickle serialization | Typed validation, compression, incremental saves |
| Stage Runner | 281 | Sequential dispatch + checkpoint | Parallel execution, dependency DAG, timeout management |
| Stage modules (5) | 1,476 | Per-model sub-stage functions | Retry logic, resource estimation, progress reporting |

---

## Enhancement 6.1A: PipelineState Validation Layer

**Current:** PipelineState is a bag of `Any`-typed fields. No validation that required fields exist before a sub-stage runs.

**Expert method:** Add a **pre-flight validator** per sub-stage that checks required fields are populated and non-None before execution. Prevents cryptic errors deep inside model code.

**Implementation:** Dict of `sub_stage_id -> required_fields` checked in the runner loop before dispatching.

**New:** `_SUBSTAGE_REQUIREMENTS` dict, `validate_state_for_substage()` function (~30 lines)

### Enhancement 6.1B: Compressed Checkpoint Format

**Current:** Pickle for all non-DataFrame state. Large model results (MC terminal_values with 10K paths) produce multi-MB pickle files.

**Expert method:** Use `lz4` or `zstd` compression on pickle files. Typical 3-5x compression on numpy-heavy pickles.

**Implementation:** `pickle.dumps(obj)` -> `lz4.frame.compress(pickle.dumps(obj))` with graceful fallback. ~10 lines.

### Enhancement 6.1C: Incremental State Saves

**Current:** Full PipelineState serialized at each checkpoint.

**Expert method:** Only serialize **changed fields** since last checkpoint. Track dirty flags per field. Reduces I/O by 5-10x for late-stage sub-stages where only 1-2 fields change.

**Implementation:** Add `_dirty_fields: set` to PipelineState, mark fields on `__setattr__`, save only dirty fields + merge on load. ~40 lines.

---

## Enhancement 6.2A: Dependency DAG with Parallel Execution

**Current:** Sub-stages execute sequentially in fixed order. Stage 6 ensemble sub-stages (6.1-6.10) could run in parallel since they're independent.

**Expert method:** Build a **dependency DAG** from sub-stage requirements. Independent sub-stages run in parallel via `concurrent.futures.ThreadPoolExecutor`.

**Parallelizable groups:**
- 6.1 (Transformer) + 6.2 (Particle Filter) + 6.4 (DTW) -- independent
- 6.6 (SHAP) + 6.7 (Sobol) -- independent
- 7.4.1-7.4.5 (per-frequency pipelines) -- independent after 7.4.0

**Implementation:** `_build_dag()` from requirements dict, topological sort, parallel dispatch for same-level nodes. ~60 lines.

### Enhancement 6.2B: Sub-Stage Timeout Management

**Current:** No timeout on individual sub-stages. A hanging LSTM or GARCH can block the entire pipeline indefinitely.

**Expert method:** Per-sub-stage timeout with configurable limits. Default: 120s for most, 300s for MC (10K paths), 60s for simple models.

**Implementation:** `signal.alarm()` on Unix or `concurrent.futures.wait(timeout=)`. ~20 lines.

### Enhancement 6.2C: Resource Estimation

**Current:** No estimation of how long a sub-stage will take or how much memory it needs.

**Expert method:** Track historical timing per sub-stage in a JSONL file (`cache/stage_timing_history.jsonl`). Use rolling median to estimate expected duration. Display ETA in `run_backtest_staged.py` progress bar.

**Implementation:** Log timing after each sub-stage, load history for estimation. ~25 lines.

---

## Enhancement 6.3A: Retry Logic for Transient Failures

**Current:** Sub-stage failure saves a `{sub_id}_failed` checkpoint and aborts.

**Expert method:** **Exponential backoff retry** for transient failures (API timeouts, memory pressure). Classify errors as retryable vs permanent. Retry up to 2 times with 5s/15s delays.

**Implementation:** Wrap sub-stage dispatch in retry loop with error classification. ~25 lines.

### Enhancement 6.3B: Graceful Degradation Mode

**Current:** If a sub-stage fails, the pipeline stops.

**Expert method:** **Graceful degradation** -- if a non-critical sub-stage fails (e.g., SHAP, Sobol, DTW), continue with remaining sub-stages. Mark the failed result as `available=False` and proceed.

**Critical sub-stages** (must succeed): 3.1 (regime), 4.1 (forecasting), 5.4 (MC), 6.5 (aggregation)
**Non-critical** (can skip): 6.1 (transformer), 6.2 (PF), 6.4 (DTW), 6.6 (SHAP), 6.7 (Sobol), 6.8 (TV Granger)

**Implementation:** `_CRITICAL_SUBSTAGES` set + conditional abort logic. ~15 lines.

---

## Enhancement 6.4A: Pipeline Observability Dashboard

**Current:** Progress tracked via stdout logging in `run_backtest_staged.py`.

**Expert method:** Real-time observability via a **pipeline metrics file** (`cache/pipeline_metrics.json`) updated after each sub-stage with: current stage, elapsed time, memory usage, cache shape, n_models fitted, errors.

**Implementation:** Write metrics JSON after each sub-stage. Dashboard can read this for live status. ~20 lines.

### Enhancement 6.4B: Diff-Based Resume Validation

**Current:** Resume loads the latest checkpoint and continues. No validation that the checkpoint is compatible with the current code version.

**Expert method:** Store a **code hash** (hash of stage module source) in each checkpoint. On resume, if the hash differs, warn the user that code has changed since the checkpoint was saved.

**Implementation:** `hashlib.md5(open(stage_module).read())` stored in checkpoint metadata. ~15 lines.

---

## Implementation Priority Matrix

| Enhancement | Impact | Complexity | Priority |
|-------------|--------|------------|----------|
| 6.3B: Graceful degradation | HIGH | LOW | P1 |
| 6.2B: Timeout management | HIGH | LOW | P1 |
| 6.1A: State validation | MEDIUM | LOW | P1 |
| 6.2C: Resource estimation | MEDIUM | LOW | P2 |
| 6.4A: Pipeline metrics | MEDIUM | LOW | P2 |
| 6.3A: Retry logic | MEDIUM | MEDIUM | P2 |
| 6.1B: Compressed checkpoints | LOW | LOW | P3 |
| 6.4B: Resume validation | LOW | LOW | P3 |
| 6.1C: Incremental saves | LOW | MEDIUM | P4 |
| 6.2A: Parallel DAG execution | MEDIUM | HIGH | P4 |

---

## No New Variables

Layer 6 produces zero analytical variables. All enhancements are infrastructure improvements that make the pipeline more reliable, faster, and easier to debug. No changes to cache columns or result fields.

---

## GitHub Research Highlights

- **Retry logic:** [`tenacity`](https://github.com/jd/tenacity) (6,500+ stars) -- but inline is simpler for our 2-retry case
- **DAG execution:** [`prefect`](https://github.com/PrefectHQ/prefect) (18,000+ stars), [`dagster`](https://github.com/dagster-io/dagster) (12,000+ stars) -- too heavy; inline ThreadPoolExecutor is sufficient
- **Checkpoint compression:** [`lz4`](https://github.com/python-lz4/python-lz4) (100+ stars) -- 10x faster than zlib with 3x compression
- **Pipeline observability:** [`mlflow`](https://github.com/mlflow/mlflow) (19,000+ stars) metrics tracking pattern -- our lightweight JSON approach is simpler

All P1+P2 enhancements: **zero new dependencies**, ~130 lines total.

---

## Implementation Checklist

```
[ ] Phase 1: P1 (high impact, low complexity)
    [ ] 6.3B: Graceful degradation (_CRITICAL_SUBSTAGES set)
    [ ] 6.2B: Timeout per sub-stage
    [ ] 6.1A: Pre-flight state validation

[ ] Phase 2: P2
    [ ] 6.2C: Resource estimation from timing history
    [ ] 6.4A: Pipeline metrics JSON
    [ ] 6.3A: Retry logic with error classification

[ ] Phase 3-4: Deferred
    [ ] 6.1B: lz4 checkpoint compression
    [ ] 6.4B: Code hash resume validation
    [ ] 6.1C: Incremental dirty-field saves
    [ ] 6.2A: Parallel DAG execution
```

# Debug Scan: Layer 6 Edit Points, Inputs, Outputs & Full Implementation Plan

---

## 1. Files That Will Be DIRECTLY EDITED

### 1a. `operator1/stages/runner.py` (281 lines)

**All P1 enhancements modify the dispatch loop at lines 175-196:**

```python
# CURRENT (lines 175-196):
for i, (sub_id, func) in enumerate(substages, 1):
    logger.info("[%d/%d] Running sub-stage %s", i, total, sub_id)
    t0 = time.time()
    try:
        func(state)          # <-- P1 edits here: validation, timeout, graceful degradation
    except Exception as exc:
        logger.error("Sub-stage %s FAILED: %s", sub_id, exc)
        state.save(f"{sub_id}_failed")
        raise                 # <-- P1 edit: conditional raise (graceful degradation)
    elapsed = time.time() - t0
    logger.info("Sub-stage %s completed in %.1fs", sub_id, elapsed)
    state.save(sub_id)       # <-- P2 edit: log timing history
```

**Edit point for 6.1A (validation):** Add `validate_state_for_substage(state, sub_id)` call BEFORE `func(state)`.

**Edit point for 6.2B (timeout):** Wrap `func(state)` in `_run_with_timeout()`.

**Edit point for 6.3B (graceful degradation):** Replace `raise` with conditional: only raise if `sub_id in _CRITICAL_SUBSTAGES`.

**Edit point for 6.2C (timing):** Add `_log_substage_timing()` call after `elapsed` is computed.

**No changes to function signatures.** All edits are inside the existing `run_stages()` function body.

### 1b. `operator1/pipeline_state.py` (404 lines) -- P3/P4 only

Only touched for P3+ enhancements (compression, incremental saves). P1 edits are runner-only.

---

## 2. NO DOWNSTREAM CHANGES

Layer 6 is pure orchestration. All edits are internal to `runner.py`. No changes to:
- main.py, backtest_runner.py, run_backtest_staged.py
- Any model or feature module
- Profile builder, report generator, dashboard
- Any config file

---

## 3. COMPLETE FILE EDIT LIST

| # | File | Lines | Edit Type | Enhancement | New Lines |
|---|------|-------|-----------|-------------|-----------|
| 1 | `operator1/stages/runner.py` | 281 | MODERATE | 6.3B + 6.2B + 6.1A + 6.2C | ~80 |
| **Total** | | | | | **~80 lines** |

---

## 4. IMPLEMENTATION PHASES

### Phase 1 (~55 lines in runner.py)

**Step 1.1:** Add `_CRITICAL_SUBSTAGES` set and `_SUBSTAGE_REQUIREMENTS` dict at module level
**Step 1.2:** Add `_validate_state()` helper function
**Step 1.3:** Add `_SUBSTAGE_TIMEOUTS` dict
**Step 1.4:** Modify the dispatch loop:
  - Before `func(state)`: call `_validate_state()`
  - Wrap `func(state)` with timeout via `concurrent.futures`
  - In `except`: check `_CRITICAL_SUBSTAGES` before raising

### Phase 2 (~25 lines)

**Step 2.1:** Add `_log_substage_timing()` function
**Step 2.2:** Call after each sub-stage completes
**Step 2.3:** Add `_estimate_remaining()` for ETA

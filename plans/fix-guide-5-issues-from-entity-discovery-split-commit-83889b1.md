# Fix Guide: 5 Issues from Entity Discovery Split Debug Scan

*Created: 2026-05-14 | Source: Debug scan of commit 83889b1 | Priority: Medium*
*Affects: backtest_runner.py, run_backtest_staged.py, plans/fix-guide-split-entity-discovery-into-2-sub-stages.md*

## Summary

Commit `83889b1` correctly split the monolithic sub-stage 1.6 into 1.6a (entity discovery) and 1.6b (entity data fetch), but introduced 1 documentation mismatch and exposed 4 pre-existing sub-stage routing issues that affect `run_backtest_staged.py` individual sub-stage execution.

**Impact:** When `run_backtest_staged.py` runs sub-stages 1.4a, 1.4b, 1.8a, or 1.8b individually, they silently no-op (routed to temporal runner which ignores them, returns code 0). The staged compiler reports them as "completed" but no work was done. This only affects individual sub-stage execution -- running `--stage 1` (full Stage 1) or `--stage all` works correctly.

**No syntax errors. No data corruption. No model wiring bugs.**

---

## Issue 1: Documentation Mismatch -- 1.6a Description

**Severity:** Low | **Type:** Documentation | **Introduced by:** Commit 83889b1

### Problem

The fix guide and `run_backtest_staged.py` describe 1.6a as including "graph risk, game theory", but these models actually run inside the 1.6b block.

### Evidence

- `run_backtest_staged.py` line 38: `"Entity Discovery (LLM entities, GLEIF, graph risk, game theory)"`
- `backtest_runner.py` line 988-1009: `compute_graph_risk_metrics()` and `analyze_competitive_dynamics()` are AFTER the 1.6a checkpoint (line 893), inside the 1.6b `if state._llm_client is not None:` block (line 900)

### Fix

**File 1: `run_backtest_staged.py` line 38**
```python
# BEFORE:
("1.6a", "Entity Discovery (LLM entities, GLEIF, graph risk, game theory)"),

# AFTER:
("1.6a", "Entity Discovery (LLM entities, GLEIF corporate structure)"),
```

**File 2: `plans/fix-guide-split-entity-discovery-into-2-sub-stages.md` lines 18-21**

Remove "5. Initial graph risk (without linked_caches)" and "6. Initial game theory (without competitor_caches)" from the 1.6a operations list. Move them to the 1.6b operations list (they already appear there implicitly as "Graph risk" and "Game theory" in the data fetch section).

---

## Issue 2: `_STAGE1_SUBSTAGES` Missing 4 Sub-Stage IDs

**Severity:** Medium | **Type:** Wiring bug | **Pre-existing** (not introduced by 83889b1)

### Problem

`_STAGE1_SUBSTAGES` at `backtest_runner.py` line 2356 is missing `"1.4a"`, `"1.4b"`, `"1.8a"`, `"1.8b"`. These IDs exist in:
- `run_backtest_staged.py` ALL_STAGES (lines 35, 36, 41, 42)
- `backtest_runner.py` run_stage1() checkpoint checks (lines 622, 688, 1251)

When `backtest_runner.py --stage 1.4a` is called, it falls through to the Stage 2 temporal routing because "1.4a" is not in `_STAGE1_SUBSTAGES`, producing a silent no-op.

### Evidence

```
_STAGE1_SUBSTAGES = {"1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.6a", "1.6b", "1.7", "1.8"}
                                               ^                                          ^
                                          Missing 1.4a, 1.4b                        Missing 1.8a, 1.8b
```

### Fix

**File: `backtest_runner.py` line 2356**
```python
# BEFORE:
_STAGE1_SUBSTAGES = {"1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.6a", "1.6b", "1.7", "1.8"}

# AFTER:
_STAGE1_SUBSTAGES = {
    "1.1", "1.2", "1.3", "1.4", "1.4a", "1.4b",
    "1.5", "1.6", "1.6a", "1.6b",
    "1.7", "1.8", "1.8a", "1.8b",
}
```

---

## Issue 3: `_STAGE_FUNCS` and `_STAGE_DEPS` Missing 4 Entries Each

**Severity:** Medium | **Type:** Wiring bug | **Pre-existing**

### Problem

`_STAGE_FUNCS` (line 2370) and `_STAGE_DEPS` (line 2386) don't have entries for "1.4a", "1.4b", "1.8a", "1.8b". When these stage IDs are requested, they can't be dispatched or have their dependencies loaded.

### Fix

**File: `backtest_runner.py` -- `_STAGE_FUNCS` (after line 2375)**
```python
# BEFORE (lines 2374-2381):
    "1.4": lambda s: run_stage1(s, substage="1.4"),
    "1.5": lambda s: run_stage1(s, substage="1.5"),
    "1.6": lambda s: run_stage1(s, substage="1.6"),
    "1.6a": lambda s: run_stage1(s, substage="1.6a"),
    "1.6b": lambda s: run_stage1(s, substage="1.6b"),
    "1.7": lambda s: run_stage1(s, substage="1.7"),
    "1.8": lambda s: run_stage1(s, substage="1.8"),

# AFTER:
    "1.4": lambda s: run_stage1(s, substage="1.4"),
    "1.4a": lambda s: run_stage1(s, substage="1.4a"),
    "1.4b": lambda s: run_stage1(s, substage="1.4b"),
    "1.5": lambda s: run_stage1(s, substage="1.5"),
    "1.6": lambda s: run_stage1(s, substage="1.6"),
    "1.6a": lambda s: run_stage1(s, substage="1.6a"),
    "1.6b": lambda s: run_stage1(s, substage="1.6b"),
    "1.7": lambda s: run_stage1(s, substage="1.7"),
    "1.8": lambda s: run_stage1(s, substage="1.8"),
    "1.8a": lambda s: run_stage1(s, substage="1.8a"),
    "1.8b": lambda s: run_stage1(s, substage="1.8b"),
```

**File: `backtest_runner.py` -- `_STAGE_DEPS` (lines 2390-2397)**
```python
# BEFORE:
    "1.4": "1.3",
    "1.5": "1.4",
    "1.6": "1.5",
    "1.6a": "1.5",
    "1.6b": "1.6a",
    "1.7": "1.6b",
    "1.8": "1.7",

# AFTER:
    "1.3": "1.2",
    "1.4": "1.3",
    "1.4a": "1.3",
    "1.4b": "1.4a",
    "1.5": "1.4",
    "1.6": "1.5",
    "1.6a": "1.5",
    "1.6b": "1.6a",
    "1.7": "1.6b",
    "1.8": "1.7",
    "1.8a": "1.7",
    "1.8b": "1.8a",
```

Note: "1.4" still maps to "1.3" for backward compat. "1.4a" is the new first-half, "1.4b" is the new second-half. Same pattern for 1.8/1.8a/1.8b.

---

## Issue 4: `run_single_stage()` Missing Market/Company Params for Stage 1 Sub-Stages

**Severity:** Low | **Type:** Robustness | **Pre-existing**

### Problem

`run_single_stage()` in `run_backtest_staged.py` line 147 only passes `--market`, `--company`, `--end-date`, `--years` when `stage_id == "1"` (exact match). Stage 1 sub-stages like "1.6a" go to the else branch (line 159) which only passes `--run-dir`.

This works in practice because `backtest_runner.py` has argparse defaults and `state.load_checkpoint()` overrides from config.json, but it's fragile -- if config.json is missing or corrupted, the defaults ("us_sec_edgar", "AAPL", "2024-12-31") would be used silently for any company.

### Fix

**File: `run_backtest_staged.py` lines 147-161**
```python
# BEFORE:
if stage_id == "1":
    # Stage 1 needs market/company/end-date
    cmd.extend([
        "--stage", "1",
        "--market", market,
        "--company", company,
        "--end-date", end_date,
        "--years", str(years),
    ])
elif stage_id == "3:profile":
    # Stage 3 (profile build)
    cmd.extend(["--stage", "3", "--run-dir", run_dir])
else:
    # Sub-stage spec routed through Stage 2
    cmd.extend(["--stage", stage_id, "--run-dir", run_dir])

# AFTER:
if stage_id == "1" or stage_id.startswith("1."):
    # Stage 1 (full or sub-stage) needs market/company/end-date
    cmd.extend([
        "--stage", stage_id,
        "--market", market,
        "--company", company,
        "--end-date", end_date,
        "--years", str(years),
    ])
    if run_dir:
        cmd.extend(["--run-dir", run_dir])
elif stage_id == "3:profile":
    # Stage 3 (profile build)
    cmd.extend(["--stage", "3", "--run-dir", run_dir])
else:
    # Sub-stage spec routed through Stage 2
    cmd.extend(["--stage", stage_id, "--run-dir", run_dir])
```

---

## Issue 5: Stale Docstring in `run_stage1()`

**Severity:** Low | **Type:** Documentation | **Pre-existing, worsened by 83889b1**

### Problem

The `run_stage1()` docstring at `backtest_runner.py` line 77 says "1.1 through 1.6" and doesn't list 1.6a, 1.6b, 1.8a, 1.8b.

### Fix

**File: `backtest_runner.py` lines 77-90**
```python
# BEFORE:
    When substage is "all", runs everything (original behavior).
    When substage is "1.1" through "1.6", runs only that portion and saves
    a checkpoint so the next sub-stage can resume from disk.

    Sub-stages:
        1.1  -- Data fetch (PIT client, profile, holders, statements, OHLCV)
        1.2  -- Reconciliation + pivot + frequency separation + cache build
        1.3  -- OHLCV fallback + holders + segments
        1.4a -- Cache build (OHLCV spine, merge, benchmark, IV, cross-asset, options)
        1.4b -- Macro + risk (macro fetch, quadrant, conflict, buying power, pre-ratios)
        1.5  -- Estimation + SIX proxies + derived variables + survival + FH
        1.6  -- Entity discovery + graph risk + sentiment + catalysts
        1.7  -- Adaptive calibration (thresholds, model params, windows, signal IC)
        1.8a -- Regime detection + enriched timeline (HMM/GMM/PELT/BCP/ChangeFinder)
        1.8b -- Finalization (linked conflict, aggregates, peer ranking, behavioral, normalization)

# AFTER:
    When substage is "all", runs everything (original behavior).
    When substage is "1.1" through "1.8b", runs only that portion and saves
    a checkpoint so the next sub-stage can resume from disk.

    Sub-stages:
        1.1  -- Data fetch (PIT client, profile, holders, statements, OHLCV)
        1.2  -- Reconciliation + pivot + frequency separation + cache build
        1.3  -- OHLCV fallback + holders + segments
        1.4a -- Cache build (OHLCV spine, merge, benchmark, IV, cross-asset, options)
        1.4b -- Macro + risk (macro fetch, quadrant, conflict, buying power, pre-ratios)
        1.5  -- Estimation + SIX proxies + derived variables + survival + FH
        1.6a -- Entity discovery (LLM entities, GLEIF corporate structure)
        1.6b -- Entity data fetch + graph risk + game theory + contagion + sentiment
        1.7  -- Adaptive calibration (thresholds, model params, windows, signal IC)
        1.8a -- Regime detection + enriched timeline (HMM/GMM/PELT/BCP/ChangeFinder)
        1.8b -- Finalization (linked conflict, aggregates, peer ranking, behavioral, normalization)
```

---

## Data Flow Diagram

```
run_backtest_staged.py              backtest_runner.py main()
                                    
  ALL_STAGES list                   _STAGE1_SUBSTAGES set
  ["1.4a", "1.4b",                  --> must contain all IDs
   "1.6a", "1.6b",                      from ALL_STAGES
   "1.8a", "1.8b"]                 
        |                           _STAGE_FUNCS dict
        |  subprocess call          --> must map each ID to
        v  --stage {id}                 run_stage1 with substage
                                    
  run_single_stage()                _STAGE_DEPS dict  
  --> must pass market/             --> must map each ID to
      company/end-date                  its predecessor
      for 1.* stages                
                                    
                                    run_stage1(state, substage)
                                    --> checkpoint boundaries at
                                        1.4a, 1.4b, 1.6a, 1.6b,
                                        1.8a, 1.8b with if/return
```

## Files Changed Summary

| File | Lines Changed | Issues Fixed |
|------|--------------|-------------|
| `backtest_runner.py` line 2356 | 1 line | Issue 2 |
| `backtest_runner.py` lines 2374-2381 | 4 lines added | Issue 3 |
| `backtest_runner.py` lines 2390-2397 | 4 lines added | Issue 3 |
| `backtest_runner.py` lines 77-90 | 3 lines changed | Issue 5 |
| `run_backtest_staged.py` line 38 | 1 line | Issue 1 |
| `run_backtest_staged.py` lines 147-161 | 8 lines changed | Issue 4 |
| `plans/fix-guide-split-entity-discovery-into-2-sub-stages.md` lines 18-21 | 2 lines removed | Issue 1 |
| **Total** | **~23 lines** | **5 issues** |

## Risk Assessment

**Very low risk.** All changes are to routing tables, docstrings, and descriptions. No analytical model logic is modified. No data flow changes. The run_stage1() function body is untouched -- only the dispatch infrastructure that routes sub-stage IDs to it is fixed.

## Verification

After applying fixes, run:
```bash
python backtest_runner.py --stage 1.4a --run-dir cache/backtest_AAPL_2024-12-31
python backtest_runner.py --stage 1.8a --run-dir cache/backtest_AAPL_2024-12-31
```
Both should produce log output showing Stage 1 execution (not silent completion). The checkpoint files `state_1.4a.pkl` and `state_1.8a.pkl` should appear in the run directory.

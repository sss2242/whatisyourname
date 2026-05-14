# Fix Guide: Stage 1 Checkpoint Skip -- Stop Re-Running Prior Sub-Stages

*Updated: 2026-05-14 | Priority: High | Source: AAPL backtest -- 1.8b wastes 440s re-running 1.6a/1.6b*

## Problem

`run_stage1()` in `backtest_runner.py` is a single 1300-line waterfall function. When called with `--stage 1.8b`, it starts from line 1 and re-executes ALL prior sub-stages (1.1 through 1.8a) before reaching 1.8b. The `substage` parameter only controls where it STOPS, not where it STARTS.

The checkpoint files exist on disk (`state_1.6a.pkl`, `state_1.6b.pkl`, `state_1.8a.pkl`) but the function ignores them. Result: `--stage 1.8b` takes 470s to do 30s of work. The LLM entity discovery (1.6a) and entity data fetch + sentiment (1.6b) are re-invoked every time, wasting API credits and 5+ minutes.

## Solution: Load Dependency Checkpoint and Skip Prior Sections

Add ~30 lines at the top of `run_stage1()` that:
1. Check if a specific substage was requested (not "all")
2. Find the dependency checkpoint on disk
3. Load it into state
4. Recover local variables from state
5. Set a flag to skip prior sections

### File: `backtest_runner.py` -- Insert at line ~116 (after setup, before section 1.1)

The setup code at lines 96-115 MUST always run -- it creates `state._secrets`, `end_dt`, `market_info`, `pit_client` which are transient runtime objects not stored in checkpoints. The skip logic goes immediately after.

```python
def run_stage1(state: PipelineState, substage: str = "all") -> None:
    # ... existing setup (lines 96-115) stays unchanged ...
    
    # ---------------------------------------------------------------
    # Fast-path: skip prior sub-stages by loading dependency checkpoint
    # ---------------------------------------------------------------
    _SUBSTAGE_ORDER = [
        "1.1", "1.2", "1.3", "1.4a", "1.4b", "1.5",
        "1.6a", "1.6b", "1.7", "1.8a", "1.8b",
    ]
    _skip_to = None  # when set, skip all sections before this substage
    
    if substage != "all" and substage in _SUBSTAGE_ORDER:
        _idx = _SUBSTAGE_ORDER.index(substage)
        if _idx > 0:
            # Find the dependency checkpoint (the sub-stage right before target)
            _dep = _SUBSTAGE_ORDER[_idx - 1]
            _dep_pkl = Path(state.output_dir) / f"state_{_dep}.pkl"
            if _dep_pkl.exists():
                logger.info(
                    "Fast-path: loading checkpoint '%s', skipping to '%s'",
                    _dep, substage,
                )
                state.load_checkpoint(_dep)
                _skip_to = substage
                
                # Recover local variables from loaded state
                cache = state.cache
                ticker = state.target_profile.get("ticker", state.company)
                company_name = state.target_profile.get("name", ticker)
                identifier = state.target_profile.get("cik", ticker)
                
                # Rebuild _entity_groups from state.relationships
                # (not stored in state directly, built during 1.6b)
                _entity_groups = {}
                if state.relationships:
                    for _grp, _ents in state.relationships.items():
                        _ids = []
                        if isinstance(_ents, list):
                            for _e in _ents:
                                _eid = ""
                                if isinstance(_e, dict):
                                    _eid = _e.get("isin", "") or _e.get("ticker", "")
                                elif hasattr(_e, "isin"):
                                    _eid = _e.isin or getattr(_e, "ticker", "")
                                if _eid:
                                    _ids.append(_eid)
                        _entity_groups[_grp] = _ids
    
    # ---------------------------------------------------------------
    # Section guards: skip prior sections when fast-pathing
    # ---------------------------------------------------------------
```

Then wrap each section with a skip guard:

```python
    # --- Section 1.1: Profile fetch ---
    if _skip_to is None or _skip_to == "1.1":
        _skip_to = None  # stop skipping, execute from here
        # ... existing 1.1 code (lines 117-156) ...
        state.save("1.1")
    if substage == "1.1":
        return

    # --- Section 1.2: Financial statements ---
    if _skip_to is None or _skip_to == "1.2":
        _skip_to = None
        # ... existing 1.2 code (lines 163-262) ...
        state.save("1.2")
    if substage == "1.2":
        return

    # ... same pattern for each section ...

    # --- Section 1.6a: Entity discovery ---
    if _skip_to is None or _skip_to == "1.6a":
        _skip_to = None
        # ... existing 1.6a code (lines 849-893) ...
        state.save("1.6a")
    if substage == "1.6a":
        return

    # --- Section 1.6b: Entity data fetch ---
    if _skip_to is None or _skip_to == "1.6b":
        _skip_to = None
        # ... existing 1.6b code (lines 899-1096) ...
        state.save("1.6b")
    if substage in ("1.6", "1.6b"):
        return

    # ... continue for 1.7, 1.8a, 1.8b ...
```

## Local Variables That Need Recovery After Checkpoint Load

From the debug scan, these local variables are used across section boundaries:

| Variable | Where defined | Where used later | Recovery method |
|----------|--------------|------------------|-----------------|
| `pit_client` | Line 114 | 1.3, 1.6b | Always recreated at line 114 (setup code) |
| `ticker` | Line 127 | Many sections | `state.target_profile["ticker"]` |
| `company_name` | Line 128 | 1.6a LLM, profile | `state.target_profile["name"]` |
| `identifier` | Line 129 | 1.2, 1.3 | `state.target_profile["cik"]` or ticker |
| `market_info` | Line 107 | 1.4b macro, 1.3 OHLCV | Always recreated at line 107 (setup code) |
| `end_dt`/`start_dt` | Lines 101-102 | 1.4a cache build | Always computed from state (setup code) |
| `cache` | Line ~320+ | All sections after 1.4a | `state.cache` |
| `_entity_groups` | Line 912 | 1.6b, peer_ranking, adaptive_thresh | Rebuilt from `state.relationships` |
| `_rel_dicts` | Line 990 | graph_risk re-run | Rebuilt from `state.relationships` |

## What This Fixes

| Scenario | Before | After |
|----------|--------|-------|
| `--stage 1.8b` | 470s (re-runs 1.1-1.8a) | ~30s (loads 1.8a checkpoint, runs only 1.8b) |
| `--stage 1.7` | 414s (re-runs 1.1-1.6b) | ~30s (loads 1.6b checkpoint, runs only 1.7) |
| `--stage 1.8a` | ~400s | ~120s (loads 1.7 checkpoint, runs only 1.8a) |
| `--stage all` | Unchanged | Unchanged (no skipping) |
| `--stage 1.1` | Unchanged | Unchanged (no prior checkpoint) |

**LLM API savings:** Entity discovery (1.6a) and sentiment scoring (1.6b) no longer re-invoked when running later sub-stages. Saves 2-3 LLM calls per sub-stage execution.

## Files Changed

| File | Lines | Change |
|------|-------|--------|
| `backtest_runner.py` lines 116-160 | +35 lines | Skip logic + variable recovery block |
| `backtest_runner.py` 11 section starts | +22 lines (2 per section) | `if _skip_to is None or _skip_to == "X":` guards |
| **Total** | **~57 lines** | |

No other files affected. No changes to `main.py`, `run_backtest_staged.py`, `pipeline_state.py`, or any model/stage modules.

## Risk Assessment

**Low.** The checkpoint files already contain complete state from when each sub-stage originally ran. Loading them produces the exact same state as re-running. The only new code is the skip guards and variable recovery, which are straightforward conditional blocks. `--stage all` (the default) is completely unaffected since `_skip_to` stays None.

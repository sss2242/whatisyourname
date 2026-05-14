# Fix Guide: Re-Run Game Theory After Entity Data Fetch in main.py

*Created: 2026-05-14 | Priority: Medium | Source: Debug scan of PR #2 -- game_theory execution order*

## Problem

In `main.py`, game theory runs at **Step 5e** (line 1852) -- BEFORE entity financial data is fetched in **Step 5f** (line 1865). At execution time:

- `linked_caches` is an empty dict `{}` (initialized at line 1636, populated at line 1989 during Step 5f)
- `_entity_groups` does not exist yet (defined at line 1874 during Step 5f)
- Game theory receives `competitor_caches={}` and returns a default "monopoly" result with 0 competitors

This means **game theory never actually analyzes competitors in `main.py`**. It always reports monopoly market structure. The backtest_runner.py path does NOT have this problem -- it runs game theory AFTER entity data fetch with populated competitor caches.

### Execution Order in main.py

```
Line 1731: Step 5e -- Entity Discovery
Line 1744:   discover_linked_entities() -- LLM proposes entities
Line 1770:   GLEIF corporate structure
Line 1818:   graph_risk -- initial run (no linked_caches yet)
Line 1845:   game_theory -- RUNS HERE with empty linked_caches    <-- PROBLEM
Line 1864: Step 5f -- Entity Data Fetch
Line 1874:   _entity_groups built
Line 1989:   linked_caches populated (ThreadPoolExecutor fetch)
Line 2004: Step 5f.1 -- Competitor holders fetched
Line 2020: Step 5f.2 -- Ownership contagion
Line 2048:   graph_risk -- RE-RUN with ownership edge weights     <-- graph_risk is re-run, game_theory is not
```

### Contrast with backtest_runner.py

```
Line 854:  Entity discovery
Line 871:  GLEIF
Line 893:  Checkpoint 1.6a
Line 900:  Entity data fetch (inside same block)
Line 984:  linked_caches populated
Line 986:  graph_risk (WITH linked_caches)
Line 1003: game_theory (WITH competitor-filtered linked_caches)   <-- CORRECT
```

## Root Cause

The original code at line 1845 was written when game_theory + graph_risk were intended as "initial" analyses using relationship metadata only (entity names, not financial data). Graph risk was later fixed with a re-run at line 2048 after ownership data was available. Game theory never got the same re-run treatment.

## Solution: Re-Run Game Theory After Entity Data Fetch

Add a game_theory re-run after Step 5f.2 (line ~2072), using competitor-filtered `linked_caches`. This mirrors how graph_risk is already re-run at line 2048.

### Implementation

**File: `main.py` -- add after line ~2072 (after ownership contagion block)**

```python
            # Re-run game_theory with actual competitor financial data.
            # The initial run at Step 5e has empty linked_caches (data not
            # yet fetched). Now that linked_caches are populated, re-run
            # with competitor-only caches for accurate market structure.
            try:
                _competitor_caches = {eid: linked_caches[eid]
                                     for eid in _entity_groups.get("competitors", [])
                                     if eid in linked_caches}
                if _competitor_caches:
                    from operator1.models.game_theory import analyze_competitive_dynamics
                    game_theory_result = analyze_competitive_dynamics(
                        target_cache=cache,
                        target_name=target_profile.get("name", "target"),
                        competitor_caches=_competitor_caches,
                    )
                    logger.info(
                        "Game theory re-run with %d competitors: %s, pressure=%.3f",
                        len(_competitor_caches),
                        game_theory_result.market_structure,
                        game_theory_result.competitive_pressure,
                    )
            except Exception as exc:
                logger.debug("Game theory re-run failed: %s", exc)
```

### Where to Insert

After the graph_risk re-run block (line ~2070) and before Step 5g (linked aggregates, line 2077). The flow becomes:

```
Step 5f.2: Ownership contagion
  -> Graph risk re-run (existing, line 2048)
  -> Game theory re-run (NEW)
Step 5g: Linked aggregates
```

### What NOT to Change

- **Keep the initial game_theory call at line 1845** -- it provides a fallback result when entity data fetch fails or is skipped. The re-run overwrites it only when competitor caches are available.
- **Keep the `linked_caches if linked_caches else None` at line 1855** -- this is the pre-fetch call, linked_caches is always empty here. No need to filter since there is nothing to filter.
- **Do NOT move the initial call** -- the Step 5e block establishes `game_theory_result` variable that downstream code references. Moving it would break variable scoping.

## Files Changed

| File | Location | Change |
|------|----------|--------|
| `main.py` | After line ~2072 | Add game_theory re-run block (~15 lines) |

## Data Flow After Fix

```
Step 5e:
  game_theory(competitor_caches={})        -> default "monopoly" result
  
Step 5f:
  linked_caches populated: {MSFT: df, GOOG: df, TSMC: df, ...}
  _entity_groups = {competitors: [MSFT, GOOG], suppliers: [TSMC], ...}

Step 5f.2:
  graph_risk re-run (existing)             -> enhanced with ownership weights
  game_theory re-run (NEW)                 -> competitor-filtered: {MSFT: df, GOOG: df}
                                              -> accurate market structure: "oligopoly", 2 competitors

Step 5g:
  linked_aggregates                        -> per-group columns (already correct)
```

## Risk Assessment

**Very low.** The re-run is inside a try/except, uses the same function signature as the initial call, and only overwrites the result when competitor caches are available. If the re-run fails, the initial result (monopoly default) is preserved.

## Verification

After applying, run the AAPL pipeline and check the log for:
```
Game theory re-run with 5 competitors: oligopoly, pressure=0.XXX
```
The profile `game_theory.market_structure` should now report the correct market structure instead of "monopoly".

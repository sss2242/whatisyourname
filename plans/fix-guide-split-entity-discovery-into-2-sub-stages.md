# Fix Guide: Split Entity Discovery Into 2 Sub-Stages

*Created: 2026-05-14 | Priority: Medium | Source: AAPL backtest -- 1.6 takes 487s as one block*

## Problem

Sub-stage 1.6 currently runs as one monolithic block for 487 seconds:
- LLM entity discovery (~30-60s)
- GLEIF corporate structure (~5s)
- Graph risk + game theory (~5s)
- Entity data fetch via ThreadPoolExecutor (~380s)
- Ownership contagion + graph risk re-run (~10s)
- Linked aggregates + peer ranking (~5s)

If the data fetch fails at minute 6, the entire sub-stage must re-run including the LLM call. The LLM result is lost.

## Proposed Split

### Sub-stage 1.6a: Entity Discovery (LLM + Market Routing)

**Duration:** ~60s | **Checkpoint saves:** state.relationships, state._llm_client

**Operations:**
1. Create LLM client
2. `discover_linked_entities()` -- LLM proposes entity names per group
3. `_infer_market_id()` -- resolve each entity to a PIT market
4. GLEIF corporate structure -- parent/subsidiary enrichment
5. Initial graph risk (without linked_caches)
6. Initial game theory (without competitor_caches)

**State written:**
- `state.relationships` -- dict of groups with entity names/tickers/market_ids
- `state._llm_client` -- LLM client for sentiment later
- `state.graph_risk_result` -- initial (no edge weights yet)
- `state.game_theory_result` -- initial (no competitor data yet)

**Checkpoint:** `state.save("1.6a")` -- relationships persisted to disk

### Sub-stage 1.6b: Entity Data Fetch + Cross-Entity Analysis

**Duration:** ~400s | **Checkpoint saves:** state.linked_caches, state.contagion_result

**Operations:**
1. Read `state.relationships` from checkpoint
2. Apply per-group fetch caps from config
3. ThreadPoolExecutor(4) fetches financial data per entity (cross-region routing)
4. Compute derived variables per entity cache
5. Ownership contagion (competitor holders, MHHI, crowding)
6. Re-run graph risk with ownership edge weights
7. Linked aggregates + relative metrics

**State written:**
- `state.linked_caches` -- entity DataFrames (persisted to linked_caches/*.parquet)
- `state.contagion_result` -- MHHI, crowding, liquidation
- `state.linked_agg_df` -- cross-entity aggregate features
- `state.graph_risk_result` -- enhanced with ownership weights
- `state.cache` -- updated with contagion + aggregate columns

### Current 1.6b Renamed to 1.6c

The current sub-stage 1.6b (post-linked features: conflict propagation, sentiment, catalysts, behavioral, normalization) becomes 1.6c.

## Sub-Stage ID Mapping

| Before | After | Content |
|--------|-------|---------|
| 1.6 | **1.6a** | LLM discovery + GLEIF + initial graph/game |
| (part of 1.6) | **1.6b** | Entity data fetch + contagion + aggregates |
| 1.6b | **1.6c** | Post-linked features (conflict, sentiment, catalysts, normalization) |
| 1.7 | 1.7 | Adaptive calibration (unchanged) |
| 1.8a | 1.8a | Regime timeline (unchanged) |
| 1.8b | 1.8b | Finalization (unchanged) |

## Files to Modify

| File | Change |
|------|--------|
| `backtest_runner.py` lines 850-1037 | Split entity block into 2 functions, rename current run_stage1_6b |
| `run_backtest_staged.py` line 38 | Split `("1.6", ...)` into `("1.6a", ...)` and `("1.6b", ...)`, rename 1.6b to 1.6c |
| `main.py` lines 1731-2105 | Add checkpoint save between discovery and fetch (optional, main.py doesn't use staged runner for Stage 1) |

## Data Flow Diagram

```
1.6a: LLM Discovery
  IN:  state.target_profile, state._secrets, state._pit_client
  OUT: state.relationships, state._llm_client, state.graph_risk_result, state.game_theory_result
  SAVES: state_1.6a.pkl (relationships serialized)

        |
        v (checkpoint boundary -- can resume here)

1.6b: Entity Data Fetch
  IN:  state.relationships (from 1.6a checkpoint), state._pit_client, state._secrets
  OUT: state.linked_caches, state.contagion_result, state.linked_agg_df, state.cache
  SAVES: state_1.6b.pkl + linked_caches/*.parquet

        |
        v

1.6c: Post-Linked Features (was 1.6b)
  IN:  state.cache, state.relationships, state.linked_caches, state._llm_client
  OUT: state.cache (+ conflict, sentiment, catalysts, behavioral columns)
```

## Risk Assessment

**Low risk.** This is a structural split of existing code, not new logic. Both sub-stages already exist as code blocks within the monolithic Stage 1.6 -- they just need function boundaries and a checkpoint save between them.

The main risk is the sub-stage ID renaming (1.6b -> 1.6c) which affects `run_backtest_staged.py` ALL_STAGES. Existing checkpoints from prior runs won't match the new IDs, but that's expected for any structural change.

## Estimated Scope

- `backtest_runner.py`: ~30 lines refactored (split function, add checkpoint boundary)
- `run_backtest_staged.py`: 3 lines changed (rename + add new entry)
- `main.py`: 0 lines (optional: add logging boundary)
- Total: ~35 lines changed

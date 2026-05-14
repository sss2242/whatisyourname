# Fix Guide: Linked Entity Group Differentiation

*Created: 2026-05-14 | Priority: Medium-High | Source: Debug scan of linked entity data flow*
*Affects: main.py, backtest_runner.py, operator1/stages/stage6_ensemble.py*

## Problem

The entity discovery system correctly identifies linked entities by relationship group -- competitors, suppliers, customers, financial_institutions, logistics, regulators, parent_companies, subsidiaries. Each group has a distinct analytical purpose per the core idea document.

However, 5 downstream models receive the full `linked_caches` dict (ALL entities from ALL groups) instead of group-filtered subsets. This causes:

- Game theory treating suppliers as competitors in Cournot/Stackelberg analysis
- Peer ranking comparing the target against suppliers from different industries
- Adaptive threshold calibration using non-peer financial profiles for percentile calculation
- DTW analog search using irrelevant supplier/customer price histories

## What the Core Idea Says

Per `App core idea (initial).md` Section A, linked variables are tied to companies connected through specific relationships:

| Relationship | Analytical Purpose | Should Feed |
|-------------|-------------------|-------------|
| **Competitors** | Market structure, pricing power, relative positioning | Game theory, peer ranking, adaptive thresholds, DTW analogs |
| **Suppliers** | Supply chain risk, input cost pressure | Supply chain stress, conflict propagation, graph risk |
| **Customers** | Revenue exposure, demand risk | Revenue exposure score, demand forecasting, geographic risk |
| **Financial institutions** | Credit risk, contagion | Contagion analysis, graph risk, leverage stress |
| **Logistics** | Supply chain continuity | Supply chain stress, geographic concentration |
| **Regulators** | Policy risk | No financial data -- policy risk score only |
| **Parent companies** | Corporate structure, protection | Fuzzy protection inheritance, graph risk |
| **Subsidiaries** | No PIT data available | Skipped in data fetch (cap=0) |

## Current State: What Is Already Correct

These 4 consumption points properly differentiate by group:

| Model | File:Line | How It Filters |
|-------|-----------|---------------|
| Linked aggregates | `main.py:2080` | Passes `entity_groups=_entity_groups` -- produces per-group columns |
| Ownership contagion | `main.py:2006` | `_comp_ids = _entity_groups.get("competitors", [])` -- fetches only competitor holders |
| Conflict risk | `main.py:2147` | Passes `relationships` dict -- scores supply/revenue/competitive separately |
| Graph risk | `main.py:1833` | Passes `relationships=_rel_dicts` -- uses group-specific edge weights |

## What Needs Fixing: 5 Models x 8 Call Sites

### Fix 1: Game Theory -- competitors only

**Purpose:** Cournot/Stackelberg/Bertrand models analyze competitive dynamics. Only competitors should be in the analysis.

**main.py line 1851:**
```python
# BEFORE:
competitor_caches=linked_caches if linked_caches else None,

# AFTER:
competitor_caches={eid: linked_caches[eid]
                   for eid in _entity_groups.get("competitors", [])
                   if eid in linked_caches} or None,
```

**backtest_runner.py line 1008:**
```python
# BEFORE:
competitor_caches=state.linked_caches or None,

# AFTER:
competitor_caches={eid: state.linked_caches[eid]
                   for eid in _entity_groups.get("competitors", [])
                   if eid in (state.linked_caches or {})} or None,
```

### Fix 2: Peer Ranking -- competitors only

**Purpose:** Percentile ranking measures relative positioning within the same competitive set. Comparing against suppliers from different industries produces meaningless rankings.

**main.py line 2170:**
```python
# BEFORE:
cache, _pr_result = compute_peer_ranking(
    cache, linked_caches=linked_caches,
)

# AFTER:
_peer_caches = {eid: linked_caches[eid]
                for eid in _entity_groups.get("competitors", [])
                if eid in linked_caches}
cache, _pr_result = compute_peer_ranking(
    cache, linked_caches=_peer_caches if _peer_caches else linked_caches,
)
```

Note: Falls back to full linked_caches if no competitors found -- preserves existing behavior for companies without identified competitors.

**backtest_runner.py line 1086:**
```python
# BEFORE:
cache, pr = compute_peer_ranking(cache, linked_caches=state.linked_caches)

# AFTER:
_peer_caches = {eid: state.linked_caches[eid]
                for eid in _entity_groups.get("competitors", [])
                if eid in state.linked_caches}
cache, pr = compute_peer_ranking(
    cache, linked_caches=_peer_caches if _peer_caches else state.linked_caches,
)
```

### Fix 3: Adaptive Thresholds -- competitors only

**Purpose:** Peer percentile calibration of survival thresholds. A technology company's current_ratio P10 should be computed from other technology companies, not from bank suppliers or logistics partners.

**main.py line 2348:**
```python
# BEFORE:
linked_caches=linked_caches if linked_caches else None,

# AFTER:
linked_caches={eid: linked_caches[eid]
               for eid in _entity_groups.get("competitors", [])
               if eid in linked_caches} or None,
```

**backtest_runner.py line 1107:**
```python
# BEFORE:
cache, linked_caches=state.linked_caches or None,

# AFTER:
linked_caches={eid: state.linked_caches[eid]
               for eid in _entity_groups.get("competitors", [])
               if eid in (state.linked_caches or {})} or None,
```

### Fix 4: DTW Analogs -- competitors only

**Purpose:** Historical analog search looks for price patterns similar to the target. Only same-industry competitors have relevant price dynamics. A semiconductor supplier's price history is not a useful analog for a consumer tech company.

**operator1/stages/stage6_ensemble.py line 242:**
```python
# BEFORE:
linked_caches=state.linked_caches if state.linked_caches else None,

# AFTER:
_competitor_ids = set()
if state.relationships:
    for ent in state.relationships.get("competitors", []):
        eid = ent.get("isin", "") or ent.get("ticker", "") if isinstance(ent, dict) else getattr(ent, "isin", "") or getattr(ent, "ticker", "")
        if eid:
            _competitor_ids.add(eid)
_dtw_caches = {eid: state.linked_caches[eid]
               for eid in _competitor_ids
               if eid in (state.linked_caches or {})} or None
...
linked_caches=_dtw_caches,
```

Note: This one is in stage6_ensemble.py, not main.py/backtest_runner.py. The stage module doesn't have `_entity_groups` dict, so it must build competitor IDs from `state.relationships`.

## Helper Function (Optional Refactor)

To avoid repeating the dict comprehension 8 times, add a utility to PipelineState:

```python
# In operator1/pipeline_state.py:
def get_group_caches(self, group: str) -> dict[str, pd.DataFrame]:
    """Return linked_caches filtered to a specific relationship group."""
    if not self.linked_caches or not self.relationships:
        return {}
    group_entities = self.relationships.get(group, [])
    group_ids = set()
    for ent in group_entities:
        eid = ""
        if isinstance(ent, dict):
            eid = ent.get("isin", "") or ent.get("ticker", "")
        elif hasattr(ent, "isin"):
            eid = ent.isin or getattr(ent, "ticker", "")
        if eid:
            group_ids.add(eid)
    return {eid: self.linked_caches[eid] for eid in group_ids if eid in self.linked_caches}
```

Then all call sites simplify to:
```python
competitor_caches=state.get_group_caches("competitors") or None
```

For main.py which uses local variables instead of PipelineState:
```python
_competitor_caches = {eid: linked_caches[eid] for eid in _entity_groups.get("competitors", []) if eid in linked_caches}
```

## Data Flow Diagram

```
entity_discovery.py
  |
  v
state.relationships = {
  "competitors": [MSFT, GOOG, AMZN, META, NVDA],
  "suppliers": [TSMC, HON],
  "customers": [BBY, WMT],
  "financial_institutions": [JPM, GS],
  ...
}
  |
  v (data fetch with per-group caps)
  |
state.linked_caches = {
  "MSFT": DataFrame,   -- competitor
  "GOOG": DataFrame,   -- competitor
  "TSMC": DataFrame,   -- supplier
  "BBY": DataFrame,    -- customer
  "JPM": DataFrame,    -- financial
  ...
}
  |
  +-- game_theory:       SHOULD get only {MSFT, GOOG}        NOT {MSFT, GOOG, TSMC, BBY, JPM}
  +-- peer_ranking:      SHOULD get only {MSFT, GOOG}        NOT all
  +-- adaptive_thresh:   SHOULD get only {MSFT, GOOG}        NOT all
  +-- dtw_analogs:       SHOULD get only {MSFT, GOOG}        NOT all
  +-- linked_aggregates: CORRECT -- uses entity_groups dict
  +-- ownership_contagion: CORRECT -- filters to competitors
  +-- conflict_risk:     CORRECT -- uses relationships dict
  +-- graph_risk:        CORRECT -- uses relationships dict
```

## Files Changed Summary

| File | Lines Changed | Fixes |
|------|--------------|-------|
| `main.py` lines 1851, 2170, 2348 | 3 call sites (~9 lines each) | Game theory, peer ranking, adaptive thresholds |
| `backtest_runner.py` lines 1008, 1086, 1107 | 3 call sites (~9 lines each) | Same 3 models, backtest parity |
| `operator1/stages/stage6_ensemble.py` line 242 | 1 call site (~8 lines) | DTW analogs |
| `operator1/pipeline_state.py` | 1 new method (~12 lines) | Optional: `get_group_caches()` helper |
| **Total** | **~8 call sites, ~80 lines** | **5 models fixed** |

## Risk Assessment

**Low risk.** All changes are at the call site level -- filtering the input dict before passing to unchanged model functions. The model functions themselves (game_theory.py, peer_ranking.py, adaptive_thresholds.py, dtw_analogs.py) are not modified. Fallback to full linked_caches when no competitors found preserves existing behavior for edge cases.

## Verification

After applying fixes, run AAPL backtest and check:
1. Game theory log should show 5 competitors (not 5+suppliers+customers)
2. Peer ranking n_peers should match competitor count
3. Adaptive thresholds should calibrate from same-sector companies only
4. DTW analog search should report searching only competitor histories

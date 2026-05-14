# Fix Guide: Per-Group Entity Fetch Caps

*Created: 2026-05-14 | Priority: Medium | Source: AAPL backtest -- only competitors fetched, 0 suppliers/customers*

## Problem

The entity discovery finds entities across 7 relationship groups, but the data fetch phase has a single global cap of 10 entities (`_MAX_LINKED_ENTITIES = 10` in main.py:1865). Entities are flattened into a single list by group insertion order, so competitors (first group) fill the cap and suppliers/customers/financial_institutions get dropped.

For AAPL: 5 competitors found (MSFT, GOOG, AMZN, META, NVDA), 0 suppliers, 0 customers, 8 subsidiaries. Only 5 entities fetched because competitors consumed the cap before subsidiaries could be added (subsidiaries don't have PIT financial data anyway).

The real issue: even if the LLM found TSMC (supplier) and Best Buy (customer), they'd get dropped by the global 10-entity cap.

## Current Architecture

```
entity_discovery.py          main.py / backtest_runner.py
        |                              |
   LLM proposes per-group        Flatten all groups into 
   (budget_per_group: 15)        single list, cap at 10
        |                              |
   Search + resolve per-group    ThreadPoolExecutor(4) fetch
   (budget_global: 80 calls)     per entity (quotes + statements)
        |                              |
   state.relationships           state.linked_caches
   {group: [entities]}           {entity_id: DataFrame}
```

The bottleneck is in the flatten-and-cap step, not in discovery.

## Solution: Per-Group Fetch Caps

### Config (global_config.yml)

```yaml
# Per-group entity fetch caps (max entities to download data for)
# Total across all groups should stay under ~25 to respect API budgets.
entity_fetch_caps:
  competitors: 5         # most important for peer ranking + game theory
  suppliers: 3           # supply chain risk + contagion
  customers: 3           # revenue exposure
  financial_institutions: 2  # credit risk + contagion
  subsidiaries: 0        # skip (no PIT data for subsidiaries)
  parent_companies: 1    # GLEIF parent
  logistics: 1           # supply chain
  regulators: 0          # skip (no financial data)
  _default: 2            # for any unlisted group
```

### Implementation in main.py (replace lines 1860-1898)

Replace the flat cap:

```python
# BEFORE (flat cap):
_MAX_LINKED_ENTITIES = 10
_all_linked = _all_linked[:_MAX_LINKED_ENTITIES]

# AFTER (per-group cap):
from operator1.config_loader import get_global_config
_fetch_caps = get_global_config().get("entity_fetch_caps", {})
_default_cap = _fetch_caps.get("_default", 2)

_all_linked = []
_entity_groups = {}
for group_name, group_entities in relationships.items():
    _cap = _fetch_caps.get(group_name, _default_cap)
    group_ids = []
    if isinstance(group_entities, list):
        for ent in group_entities[:_cap]:  # PER-GROUP CAP
            ent_id = ""
            if hasattr(ent, "isin") and ent.isin:
                ent_id = ent.isin
            elif hasattr(ent, "ticker") and ent.ticker:
                ent_id = ent.ticker
            elif isinstance(ent, dict):
                ent_id = ent.get("isin", "") or ent.get("ticker", "")
            if ent_id and ent_id not in {e.get("id") for e in _all_linked}:
                _ent_market_id = ""
                if hasattr(ent, "market_id"):
                    _ent_market_id = ent.market_id
                elif isinstance(ent, dict):
                    _ent_market_id = ent.get("market_id", "")
                _all_linked.append({
                    "id": ent_id,
                    "name": getattr(ent, "name", "") if hasattr(ent, "name") else ent.get("name", ""),
                    "group": group_name,
                    "market_id": _ent_market_id,
                })
                group_ids.append(ent_id)
    _entity_groups[group_name] = group_ids

logger.info("Entity fetch: %d entities across %d groups (caps: %s)",
            len(_all_linked), len(_entity_groups),
            {g: len(ids) for g, ids in _entity_groups.items() if ids})
```

### Same change in backtest_runner.py

Replace the flat `_all_linked[:10]` with the identical per-group cap logic.

### Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `config/global_config.yml` | after line 26 | Add `entity_fetch_caps` section (~10 lines) |
| `main.py` | 1860-1898 | Replace flat cap with per-group cap (~20 lines changed) |
| `backtest_runner.py` | 900-913 | Same per-group cap logic (~20 lines changed) |

### What This Enables

With per-group caps of 5+3+3+2+1+1 = 15 max entities:

| Group | AAPL Expected | Source |
|-------|---------------|--------|
| competitors | MSFT, GOOG, AMZN, META, NVDA (5) | LLM + static registry |
| suppliers | TSMC, Broadcom, Qualcomm (3) | LLM (needs cross_region: true for TSMC) |
| customers | Best Buy, AT&T, Verizon (3) | LLM |
| financial_institutions | Goldman Sachs, JPMorgan (2) | LLM |
| parent_companies | (none for AAPL) | GLEIF |
| logistics | (1 if found) | LLM |

### Dependency: Cross-Region Discovery

For suppliers like TSMC (Taiwan) and Samsung (Korea), `cross_region_discovery` must be `true` in global_config.yml. Currently it's `false`. This is a separate config toggle -- the per-group cap fix works without it, but non-US suppliers won't be found.

### Risk Assessment

**Low risk.** The per-group cap is a config-driven replacement of a hardcoded constant. Downstream consumers already iterate by group. No model changes needed.

### Estimated Scope

- 3 files, ~50 lines changed
- Config-driven, backward compatible (default caps sum to ~15, close to the old 10)

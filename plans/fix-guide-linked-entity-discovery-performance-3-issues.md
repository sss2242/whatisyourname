# Fix Guide: Linked Entity Discovery Performance

*Created: 2026-05-13 | Based on live debug session with AAPL*

## Diagnosis Summary

The linked entity discovery pipeline **works correctly** -- the code logic is sound after commits `fa0c720` (cross-region routing) and `13950d0` (PR review fixes). However, **performance makes it impractical** within typical execution timeouts (300s).

**Live test results:**
- LLM entity proposal: 40.4s, returned 20 entities across 5 groups (competitors, suppliers, customers, financial_institutions, logistics)
- All 20 entities returned with **empty `market_id` and `ticker`** fields
- Entity resolution: times out at 300s because each entity searches up to 25 PIT clients sequentially

## Root Cause: 3 Performance Issues

### Issue 1: LLM Ignores market_id/ticker Routing Fields

**Location:** [`operator1/clients/llm_base.py:544-584`](operator1/clients/llm_base.py:544)

**Problem:** The prompt at line 544 asks the LLM to provide `market_id` and `ticker` with available markets injected at `{available_markets}`. However, the free nemotron-120b model returns empty strings for both fields despite the prompt explicitly requesting them.

**Evidence:**
```python
# LLM response for Samsung:
{'name': 'Samsung Electronics', 'market_id': '', 'ticker': ''}
# Expected:
{'name': 'Samsung Electronics', 'market_id': 'kr_dart', 'ticker': '005930'}
```

**Impact:** Path A (direct routing, 1-2 API calls) at [`entity_discovery.py:694`](operator1/steps/entity_discovery.py:694) is always skipped. Every entity falls through to Path B (cross-region search, up to 25 API calls).

**Root cause:** The nemotron-120b free model has weaker instruction following for structured JSON fields. It produces the `name` correctly but ignores `market_id`/`ticker` because:
1. The available markets list is at the end of the prompt (after the JSON format instructions)
2. Free-tier models optimize for speed, not precision on optional fields
3. The prompt says "If unsure of the market_id or ticker, omit them" -- the model takes the easy path

---

### Issue 2: Cross-Region Entity Search Is Sequential

**Location:** [`operator1/steps/entity_discovery.py:360-414`](operator1/steps/entity_discovery.py:360)

**Problem:** `_resolve_entity_cross_region()` searches PIT clients one at a time in a for loop:

```python
# Line 396-412: Sequential loop through all 25 clients
if all_clients:
    for client in all_clients:  # <-- Sequential!
        entity = _resolve_entity(query, group, client, ...)
        if entity is not None:
            return entity
```

Each `_resolve_entity()` call makes 1-2 HTTP API requests (search + optional short-query fallback). With 25 clients and ~1-2s per search, a single non-US entity like "Samsung" takes **25-50 seconds** to resolve (or until it's found in kr_dart).

**Impact:** 20 entities x 5-25 clients each = 100-500 seconds for the resolution phase.

---

### Issue 3: No Overall Timeout on Discovery

**Location:** [`operator1/steps/entity_discovery.py:497-786`](operator1/steps/entity_discovery.py:497)

**Problem:** `discover_linked_entities()` has per-group budget limits (`budget_per_group=10`, `budget_global=50`) but no wall-clock timeout. The budget counts API calls, not seconds. A single slow cross-region search consuming 50 seconds counts as just 1 API call.

**Impact:** Even with the budget mechanism, 50 API calls x 5-10s each = 250-500 seconds, exceeding the 300s terminal timeout.

---

## Fix Plan

### Fix 1: Client-Side market_id Inference (No LLM Change)

**Strategy:** Instead of relying on the LLM to provide `market_id`, infer it from the entity name + country using a fast heuristic lookup. This avoids the 25-client brute-force search entirely.

**Files to modify:**
- [`operator1/steps/entity_discovery.py`](operator1/steps/entity_discovery.py) -- add `_infer_market_id()` function

**Implementation:**

Add a new function `_infer_market_id(name, country_hint)` at line ~358 that maps well-known company names and country keywords to market_ids:

```
Samsung, Hyundai, LG, SK -> kr_dart
Toyota, Sony, Honda -> jp_jquants
TSMC, Foxconn, MediaTek -> tw_mops
Tencent, Alibaba, Baidu -> cn_sse
Unilever, BP, HSBC -> uk_companies_house
SAP, Siemens, BMW -> de_esef
Petrobras, Vale, Itau -> br_cvm
```

Also use the `country` field from the profile to route: US -> us_sec_edgar, etc.

**Code location for the change:**

In `discover_linked_entities()` at line 690, after extracting name/market_id/ticker from the LLM item, add:

```python
# Line ~693: After extracting _entity_market_id from LLM response
if not _entity_market_id:
    _entity_market_id = _infer_market_id(name, target_country)
```

This transforms empty `market_id` into a best-guess, enabling Path A (direct routing, 1-2 calls) instead of Path B (25-client loop).

**Fallback:** If inference returns empty, Path B still runs as before.

---

### Fix 2: Parallelize Cross-Region Search

**Strategy:** When Path B (cross-region search) must run, search multiple PIT clients in parallel using `ThreadPoolExecutor` with early termination on first match.

**Files to modify:**
- [`operator1/steps/entity_discovery.py:360-414`](operator1/steps/entity_discovery.py:360) -- modify `_resolve_entity_cross_region()`

**Implementation:**

Replace the sequential for loop (lines 396-412) with:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _resolve_entity_cross_region(query, group, primary_client, all_clients, ...):
    # Try primary first (unchanged)
    entity = _resolve_entity(query, group, primary_client, ...)
    if entity is not None:
        return entity

    # Parallel search across all other clients
    if all_clients:
        other_clients = [c for c in all_clients 
                        if getattr(c, 'market_id', '') != getattr(primary_client, 'market_id', '')]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_resolve_entity, query, group, c, target_country, target_sector): c
                for c in other_clients
            }
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)
                    if result is not None:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        return result
                except Exception:
                    continue
    return None
```

**Expected improvement:** 25 sequential searches (~25-50s) -> 8 parallel workers (~3-6s per entity).

---

### Fix 3: Add Wall-Clock Timeout to Discovery

**Strategy:** Wrap the entire resolution loop in a wall-clock timeout so it gracefully stops after a configurable duration.

**Files to modify:**
- [`operator1/steps/entity_discovery.py:658-743`](operator1/steps/entity_discovery.py:658) -- add timeout check in resolution loop

**Implementation:**

At the start of the resolution loop (line 658), record start time:

```python
import time
_discovery_start = time.time()
_discovery_timeout = cfg.get("entity_discovery_timeout_s", 120)  # 2 minutes
```

Inside the inner loop (line 668), add a wall-clock check:

```python
for item in items:
    # Wall-clock timeout check
    if time.time() - _discovery_start > _discovery_timeout:
        logger.warning(
            "Entity discovery wall-clock timeout (%.0fs) -- stopping resolution",
            time.time() - _discovery_start,
        )
        break
    # ... existing resolution code ...
```

Also add the config key to [`config/global_config.yml`](config/global_config.yml):

```yaml
entity_discovery_timeout_s: 120
```

---

## Execution Order

1. **Fix 1** (market_id inference) -- highest impact, eliminates the root cause for most entities
2. **Fix 3** (wall-clock timeout) -- safety net, prevents runaway execution
3. **Fix 2** (parallel search) -- optimization for the remaining entities that need cross-region search

## Files Modified (Complete List)

| # | File | Lines | Change |
|---|------|-------|--------|
| 1 | `operator1/steps/entity_discovery.py` | ~358 | Add `_infer_market_id()` function (~40 lines) |
| 2 | `operator1/steps/entity_discovery.py` | ~693 | Add market_id inference call (3 lines) |
| 3 | `operator1/steps/entity_discovery.py` | 360-414 | Parallelize `_resolve_entity_cross_region()` (~20 lines changed) |
| 4 | `operator1/steps/entity_discovery.py` | ~658-668 | Add wall-clock timeout check (~10 lines) |
| 5 | `config/global_config.yml` | end | Add `entity_discovery_timeout_s: 120` (1 line) |

## Testing Plan

After implementing all 3 fixes, run the same test:

```python
python -c "
from operator1.steps.entity_discovery import discover_linked_entities
result = discover_linked_entities(
    target_profile={'name': 'Apple Inc.', 'ticker': 'AAPL', 'country': 'US', 'sector': 'Technology'},
    llm_client=llm_client, pit_client=pit_client, secrets=secrets, force_rebuild=True,
)
print(f'Total: {sum(len(v) for v in result.linked.values())} entities')
print(f'Search calls: {result.search_calls_used}')
```

**Expected results:**
- Total time: ~50-60s (40s LLM + 10-20s resolution) instead of 300s+
- 15-20 entities resolved (vs timing out with partial results)
- market_id populated for most entities via inference

## Risk Assessment

- **Fix 1 (inference):** Low risk -- adds a fast pre-check, existing fallback paths unchanged. Only risk is wrong market_id inference (mitigated by Path B fallback).
- **Fix 2 (parallel):** Medium risk -- parallel HTTP requests may hit API rate limits. Mitigated by max_workers=8 and per-host rate limiting already in `http_utils.py`.
- **Fix 3 (timeout):** Low risk -- graceful degradation, partial results still usable. Static competitor fallback (P10) provides competitors even if resolution times out.

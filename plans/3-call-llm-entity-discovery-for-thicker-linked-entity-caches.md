# 3-Call LLM Entity Discovery for Thicker Linked Entity Caches

## Problem

The current entity discovery makes a single LLM call that asks for all relationship groups at once. This produces thin results because:
1. The LLM has to spread its attention across 6 categories in one response
2. It tends to list 2-3 entities per group, not 5-8
3. It does not distinguish international vs local peers (a Korean company gets mixed Korean + US competitors with no geographic strategy)
4. There is no gap-fill pass to catch entities the first call missed

## Current Flow (1 LLM call)

```
LLM Call 1: "Given this profile, suggest competitors, suppliers, customers,
             financial_institutions, logistics, regulators"
  -> Returns ~10-15 entity names across 6 groups
  -> Each name resolved via PIT client search
  -> Fallback: sector peer listing if no competitors found
```

## Proposed Flow (3 LLM calls)

### Call 1: International Scope Check + Global Peers

**Purpose:** Determine if the company operates internationally. If yes, discover international competitors, suppliers, and customers.

**Prompt logic:**
- Ask the LLM: "Does {company} have significant international operations, revenue, or supply chains? If yes, list international competitors, suppliers, and customers from different countries/regions."
- Include the company profile with country, sector, industry
- Request structured JSON with `is_international: true/false` and per-group entities with `country` field

**Output:**
- `is_international` flag (stored for downstream use)
- International entities (competitors from other markets, cross-border suppliers)
- Each entity tagged with its country so cross-region resolution knows where to search

**Skip condition:** If the company is clearly domestic-only (e.g. a regional utility), the LLM returns `is_international: false` and the call still produces value by providing a high-level assessment.

### Call 2: Local/Domestic Peers

**Purpose:** Deep dive into the company's home market for domestic competitors, local suppliers, local financial institutions, and regulators.

**Prompt logic:**
- "Focus on {country} market only. List domestic competitors in the same sector, local suppliers, domestic banks/lenders, and relevant regulatory bodies."
- Include results from Call 1 so the LLM does not repeat the same entities
- Request more entities per group: "List at least 5 competitors and 3 suppliers if possible"

**Output:**
- Domestic competitors (same exchange, same market)
- Local suppliers, banks, regulators
- Higher resolution count because the LLM is focused on one market

### Call 3: Gap-Fill + Backup

**Purpose:** Review what was found in Calls 1-2 and fill gaps in underrepresented groups.

**Prompt logic:**
- "Here is what we found so far: {summary of Calls 1+2 results}. The following groups are thin or empty: {list groups with <2 entities}. Please suggest additional entities for these groups specifically."
- Also ask for: "Any major entities we missed? Any recent changes in the competitive landscape, supplier relationships, or customer base in the last 12 months?"

**Output:**
- Gap-fill entities for underrepresented groups
- Recently changed relationships (new entrants, lost suppliers)
- This call acts as quality assurance on the first two calls

## Implementation Plan

```
[ ] Step 1: Add 3 new prompt templates to llm_base.py
    - _LINKED_ENTITIES_INTL_PROMPT (Call 1: international scope + global peers)
    - _LINKED_ENTITIES_LOCAL_PROMPT (Call 2: domestic deep dive)
    - _LINKED_ENTITIES_GAPFILL_PROMPT (Call 3: gap-fill)

[ ] Step 2: Add propose_linked_entities_3call() method to LLMClient
    - Orchestrates the 3-call flow
    - Merges results from all 3 calls (deduplicating by name)
    - Returns same dict format as current propose_linked_entities()
    - Tracks is_international flag in the result

[ ] Step 3: Update discover_linked_entities() in entity_discovery.py
    - Call propose_linked_entities_3call() when available
    - Fall back to single-call propose_linked_entities() if 3-call fails
    - Pass the is_international flag to profile for report narrative

[ ] Step 4: Update entity discovery budget in global_config.yml
    - Increase search_budget_per_group from 10 to 15
    - Increase search_budget_global from 50 to 80
    - (3 LLM calls produce more entity names to resolve)

[ ] Step 5: Test with synthetic profile and verify entity count improvement
[ ] Step 6: Commit and push
```

## Expected Impact

| Metric | Current (1 call) | Proposed (3 calls) |
|--------|------------------|--------------------|
| LLM calls | 1 | 3 |
| Entities per group | 2-3 | 5-8 |
| Total entities proposed | 10-15 | 25-40 |
| International coverage | Mixed | Explicit international vs domestic |
| Gap-fill | None | Dedicated pass |
| Cost | 1 LLM call (~$0.001 Gemini) | 3 LLM calls (~$0.003 Gemini) |

The 3x cost increase is negligible ($0.003 total for Gemini Flash). The benefit is substantially thicker linked entity caches, which directly improves: graph_risk, game_theory, peer_ranking, linked_aggregates, ownership_contagion, cross-company DTW, and the prediction aggregator's multi-entity signals.

## Prompt Design Notes

- Call 1 is the most important: it establishes whether the company is international and discovers cross-border relationships that the domestic-only PIT client cannot find
- Call 2 benefits from knowing the results of Call 1: the LLM avoids repeating entities and focuses on the gaps
- Call 3 is a quality assurance pass: it catches entities that fell through the cracks and identifies recent relationship changes
- All 3 calls ask for the same JSON format with temporal context (relationship_start, relationship_end, stability)
- The `country` field in each entity helps the cross-region resolver know which PIT client to search first

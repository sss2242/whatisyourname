# LLM Fallback for Missing Financial Fields

## Problem

When SEC EDGAR's structured XBRL data doesn't contain a field (goodwill, intangible_assets, rd_expenses), the pipeline leaves it as NaN and downstream models lose information. The pipeline has LLM capabilities that could fill these gaps but they're not wired into the per-field gap-filling path.

## Current Data Extraction Chain

```
edgartools XBRL -> CompanyFacts API -> Keyword Auto-Discovery -> STOP (field = NaN)
```

## Proposed Chain (4-tier fallback)

```
edgartools XBRL 
  -> CompanyFacts API (explicit XBRL concepts)
    -> Keyword Auto-Discovery (fuzzy match across all concepts)
      -> LLM Concept Resolution (ask LLM to map the canonical name to a concept)
        -> Zero-Default for Balance Sheet items that companies legitimately don't have
```

### Tier 3: LLM Concept Resolution

After keyword auto-discovery fails, ask the LLM: "What XBRL concept does Apple use for goodwill?" The pipeline already has `resolve_unmapped_concepts_llm()` in [`canonical_translator.py`](operator1/clients/canonical_translator.py). This can be called with the missing field names.

**Implementation:**

In [`backtest_runner.py`](backtest_runner.py) after the keyword auto-discovery block (line ~635), add:

```python
# Tier 3: LLM concept resolution for still-missing fields
_still_missing_after_auto = [
    f for f in _all_critical_fields
    if f not in cache.columns or cache[f].isna().all()
]
if _still_missing_after_auto and state._llm_client is not None:
    try:
        from operator1.clients.canonical_translator import resolve_unmapped_concepts_llm
        _llm_mappings = resolve_unmapped_concepts_llm(
            _still_missing_after_auto, state._llm_client
        )
        # _llm_mappings: {canonical_name: xbrl_concept_name}
        for _field, _concept in _llm_mappings.items():
            if _concept and _field in _still_missing_after_auto:
                _cdata = _usgaap.get(_concept, {}) or _dei.get(_concept, {})
                # ... same extraction logic as CompanyFacts ...
    except Exception as exc:
        logger.debug("LLM concept resolution failed: %s", exc)
```

**Cost:** 1 LLM call for all missing fields (batched). Already has disk caching in `cache/llm_concept_map.json`.

### Tier 4: Zero-Default for Balance Sheet Items

After all extraction tiers fail, certain balance sheet fields should default to 0 rather than NaN because "not reported" has a clear accounting meaning:

| Field | Not Reported Means | Default |
|-------|-------------------|---------|
| `goodwill` | No acquisitions | **0** |
| `intangible_assets` | No significant intangibles | **0** |
| `rd_expenses` | No separate R&D line | **NaN** (may be bundled into SGA) |
| `inventory` | Service company (no physical inventory) | **0** |
| `dividends_paid` | No dividends | **0** |
| `stock_buybacks` | No buybacks | **0** |

**Implementation:**

In [`backtest_runner.py`](backtest_runner.py) or in [`estimator.py`](operator1/estimation/estimator.py) Phase 1 (identity fill), add after all extraction:

```python
_ZERO_DEFAULT_FIELDS = {
    "goodwill", "intangible_assets", "inventory",
    "dividends_paid", "stock_buybacks",
}
for _field in _ZERO_DEFAULT_FIELDS:
    if _field not in cache.columns or cache[_field].isna().all():
        cache[_field] = 0.0
        logger.info("Zero-default: %s set to 0 (not reported = not applicable)", _field)
```

**NOT zero-defaulted:** `rd_expenses` (could be bundled into SGA -- should stay NaN and let the estimator handle it), `sga_expenses` (same logic), any flow variable where "not reported" doesn't mean "zero."

### Where to Place the Logic

**Option A: In backtest_runner.py sub-stage 1.4a** (after CompanyFacts fallback)
- Pro: Closest to the data source, earliest in pipeline
- Con: Only runs for US SEC EDGAR market

**Option B: In estimator.py Phase 1** (after identity fill)
- Pro: Runs for ALL markets, not just US
- Con: Estimator is meant for model-based imputation, not data source fallback

**Option C: In main.py Step 4b** (after all data sources exhausted)
- Pro: Market-agnostic, runs after all PIT clients have had their chance
- Con: Adds complexity to already-long main.py

**Recommendation:** Option A for the LLM resolution (market-specific XBRL), Option B for zero-defaults (universal accounting semantics).

## Implementation Steps

1. Add LLM concept resolution as Tier 3 in `backtest_runner.py` after auto-discovery (~15 lines)
2. Add zero-defaults for balance sheet fields in `estimator.py` Phase 1 (~10 lines)
3. Add `rd_expenses` to `ESTIMABLE_VARIABLES` in `estimator.py` so the MAR/MNAR engine can impute it (~1 line)

## Risk

- **Low:** LLM concept resolution already cached to disk. Won't re-call for same company.
- **Low:** Zero-defaults are conservative -- only applied to fields with clear "not applicable" semantics.
- **Medium:** LLM could return wrong XBRL concept mappings. Mitigated by the existing validation in CompanyFacts extraction (checks for valid `form` and `val` fields).

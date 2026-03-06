# Bug Fix Plan

Two bugs found during debug audit of the new filing discoverer and conflict risk code.

---

## Bug 1: Profile Builder Missing conflict_risk Wiring (HIGH)

**Problem:** The report section `_build_geopolitical_risk_section()` reads conflict data from `profile["conflict_risk"]`, but [`profile_builder.py`](operator1/report/profile_builder.py) never stores this key. The section will always render as "No geopolitical risk data available."

**Root cause:** `build_company_profile()` was not updated to include conflict risk data.

**Fix:** Add a `_build_conflict_risk_section()` helper in `profile_builder.py` that:
1. Reads conflict risk columns from the cache: `country_conflict_flag`, `conflict_intensity_score`, `sanctions_flag`, `fragile_state_flag`, `conflict_type`, `supply_chain_risk_score`, etc.
2. Packages them into a dict matching the `ConflictRiskResult.to_dict()` format
3. Stores it as `profile["conflict_risk"]`
4. Called from `build_company_profile()` alongside the existing sections

**Files to change:**
- [ ] `operator1/report/profile_builder.py` -- add `_build_conflict_risk_section()` and call it in `build_company_profile()`

---

## Bug 2: Filing Discoverer Redundant Extraction Calls (MEDIUM)

**Problem:** `in_bse.py` calls `try_filing_extraction()` three times (once each for income, balance, cashflow). Each call independently discovers filings, downloads PDFs, and runs LLM extraction. But `statement_type` is never used to filter the result -- the LLM extracts ALL financial fields from each PDF regardless.

This means:
- 3x BSE API calls to discover the same filings
- 3x PDF downloads of the same documents
- 3x LLM extraction calls on the same content
- The returned DataFrame contains all statement types mixed together each time

**Fix:** Add a per-ticker extraction cache in `try_filing_extraction()`:
1. On first call for a ticker, do the full discovery -> download -> extract pipeline
2. Cache the combined result DataFrame in a module-level dict keyed by `market_id:ticker`
3. On subsequent calls, return the cached result
4. Filter the result by `statement_type` using the canonical field names to return only relevant columns

**Files to change:**
- [ ] `operator1/clients/filing_discoverer.py` -- add extraction cache + statement_type filtering in `try_filing_extraction()`

---

## Implementation Order

1. Fix Bug 1 first (profile builder) -- this is the higher severity issue
2. Fix Bug 2 (extraction caching) -- performance optimization
3. Re-run all 53 tests
4. Push to PR

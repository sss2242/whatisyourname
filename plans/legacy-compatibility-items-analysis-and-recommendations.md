# Legacy Compatibility Items: Analysis and Recommendations

## Summary

Three legacy compatibility items were identified during the codebase review. After investigation, the recommendation is: **leave two as-is, wire one into production**.

---

## Item 1: `create_equity_provider()` -- LEAVE AS-IS

**Location:** `operator1/clients/equity_provider.py:282` (28 lines)

**What it does:** Backward-compatible wrapper around `create_pit_client()`. Accepts a legacy `provider` parameter from the old FMP-based architecture, ignores it with a log message, and delegates to `create_pit_client()`.

**Usage:**
- Production code: **0 callers** -- `main.py` and `run.py` both use `create_pit_client()` directly
- Tests: 1 test in `test_equity_provider_new.py:83`

**Decision: No action needed.**
- 28 lines, zero maintenance burden
- Has a passing test
- Protects any external scripts that may call the old function name
- Removing it saves nothing and risks breaking unknown consumers

---

## Item 2: `compute_vanity_percentage()` -- LEAVE AS-IS

**Location:** `operator1/analysis/vanity.py:450` (~85 lines)

**What it does:** Legacy 4-component vanity score (exec comp excess, SGA bloat, buyback waste, marketing excess). Produces the `vanity_percentage` column.

**Usage:**
- Production code: **Called internally** by `compute_vanity_score()` on line 573 -- it IS in the active pipeline path
- Tests: 7 references across `test_vanity_v2.py` and `test_phase4_analysis.py`
- The `vanity_percentage` column it produces is read by `_build_vanity_section()` in `profile_builder.py`

**Decision: No action needed.**
- Not dead code -- actively called by the new `compute_vanity_score()` as a sub-computation
- Removing it would require refactoring the vanity module internals and updating 7+ tests
- Zero functional benefit from removal

---

## Item 3: `profile_schema.py` -- WIRE INTO PIPELINE

**Location:** `operator1/report/profile_schema.py` (119 lines)

**What it does:** Defines 24 required profile keys and 10 optional keys. Provides `validate_profile()` (returns issues list) and `validate_profile_strict()` (raises ValueError) for validating the company profile dict between profile_builder and report_generator.

**Usage:**
- Production code: **0 callers** -- not imported by `main.py`, `profile_builder.py`, or `report_generator.py`
- Tests: 5 test methods in `test_pipeline_upgrades.py` exercise the validation functions

**Decision: Wire `validate_profile()` into `main.py` after `build_company_profile()`.**

This is a low-risk, high-value change. The module exists specifically to catch profile key mismatches at runtime before they surface as silent "No data available" report sections. Its own docstring says so. The implementation is clean and tested.

### Implementation Plan

**Single change in `main.py` around line 2530** (after `profile_path` is saved):

```python
# Validate profile completeness before report generation
try:
    from operator1.report.profile_schema import validate_profile
    _profile_issues = validate_profile(profile)
    if _profile_issues:
        logger.warning(
            "Profile validation: %d issues (report may have missing sections)",
            len(_profile_issues),
        )
except Exception as exc:
    logger.debug("Profile validation skipped: %s", exc)
```

**Why this approach:**
- Wrapped in try/except so validation failure never blocks the pipeline
- Logs at WARNING level so issues are visible but not fatal
- Placed after profile save but before report generation -- exactly where mismatches matter
- No new dependencies, no test changes needed

---

## Checklist

- [ ] Wire `profile_schema.validate_profile()` into `main.py` Step 7 after profile save
- [ ] Verify the 5 existing tests in `test_pipeline_upgrades.py` still pass
- [ ] No changes needed for `create_equity_provider()` or `compute_vanity_percentage()`

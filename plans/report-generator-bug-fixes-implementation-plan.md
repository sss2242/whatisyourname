# Report Generator Bug Fixes -- Implementation Plan

Four bugs and three minor issues found during debug scan of `operator1/report/report_generator.py` (4,151 lines). All are "data computed but not displayed" -- the pipeline produces correct results, the generator just doesn't access or render some of them properly.

---

## Fix 1: Delete 140 lines of dead code in `_build_economic_position()`

**Bug:** Lines 2214-2353 are unreachable code after a `return` statement at line 2210. The dead block builds a "Methodology Summary" that duplicates `_build_appendix()`.

**File:** `operator1/report/report_generator.py`

**Change:** Delete lines 2214-2353 (everything after `return "\n".join(lines)` at line 2210 and before the next `def` statement).

**Risk:** Zero -- dead code, never executes.

**Test:** `python3.12 -c "from operator1.report.report_generator import _build_economic_position; print('OK')"` should still work. Run `test_phase7_report.py` for regression.

---

## Fix 2: Wire `_build_economic_position()` into section builders

**Bug:** The function exists and renders useful content (5-plane economic model: Supply/Manufacturing/Consumption/Logistics/Financial Services) but is not in the `_section_builders` dict at line 2932. The profile's `economic_plane` data goes unused in reports.

**File:** `operator1/report/report_generator.py`

**Change:**
1. Add a new section number (e.g., `35` for "3.5" conceptually) or repurpose an unused number
2. Add to `_section_builders` dict: `35: ("3.5 Economic Position", _build_economic_position(profile)),`
3. Add section 35 to `TIER_SECTIONS` for PRO and PREMIUM tiers
4. OR: integrate the economic position content into Section 7 (Linked Variables & Market Context) since they're thematically related

**Recommended approach:** Add as section number `75` (after section 7) to keep existing section numbers stable:
```python
# In _section_builders dict:
75: ("7.5. Economic Position & Industry Classification", _build_economic_position(profile)),

# In TIER_SECTIONS:
ReportTier.PRO: {1, 2, 3, 4, 5, 6, 7, 75, 11, 14, 16, 17, 18, 195, 196, 197, 198, 20},
ReportTier.PREMIUM: set(range(1, 23)) | {75, 195, 196, 197, 198},
```

**Risk:** Low -- adds a new section, doesn't change existing ones.

**Test:** Generate a test report with a profile containing `economic_plane` data and verify the new section appears.

---

## Fix 3: Fix LLM validation auto-fix (implement or honest logging)

**Bug:** Lines 4049-4052 log "Attempting to append missing section" but never actually append the section content. The validation detects missing sections but the "fix" is a no-op.

**File:** `operator1/report/report_generator.py`

**Change -- Option A (implement auto-fix):**
```python
# Replace the no-op loop at lines 4049-4052 with:
for issue in validation_issues:
    if issue.startswith("Missing section:"):
        section_name = issue.replace("Missing section: '", "").rstrip("'")
        # Find matching section builder by header text
        for sec_num, (heading, content) in _section_builders.items():
            if section_name.lower() in heading.lower():
                markdown += f"\n\n---\n\n## {heading}\n\n{content}\n"
                logger.info("Appended missing section: %s", heading)
                break
```

**Change -- Option B (honest logging):**
```python
# Replace "Attempting to append" with honest log:
logger.info("Missing section '%s' -- not auto-patched (LLM output used as-is)", section_name)
```

**Recommended:** Option A -- actually append missing sections from the fallback template. The `_section_builders` dict is already available in scope (it's a local variable in `_build_fallback_report` but can be extracted).

**Complication:** `_section_builders` is built inside `_build_fallback_report()` as a local dict. To reuse it in `generate_report()`, either:
- Extract the section builder mapping into a module-level function `_get_section_builders(profile)` that both `_build_fallback_report` and `generate_report` can call
- Or build the needed sections inline using the individual `_build_*` functions

**Risk:** Low-Medium -- need to handle the case where the LLM report has different heading formatting than the fallback template.

**Test:** Mock an LLM that returns a report missing 2 sections; verify they get appended.

---

## Fix 4: Fix Section 9 profile key mismatches for conformal/SHAP/DTW

**Bug:** Three `profile.get()` calls in `_build_predictions_forecasts()` use top-level key names that don't match where main.py actually stores the data:

| Line | Generator reads | Actual location |
|------|----------------|-----------------|
| 1141 | `profile.get("conformal_intervals")` | `profile["extended_models"]["conformal_prediction"]` |
| 1159 | `profile.get("shap_explanations")` | `profile["extended_models"]["shap_explanations"]` |
| 1190 | `profile.get("historical_analogs")` | `profile["extended_models"]["dtw_analogs"]` |

**File:** `operator1/report/report_generator.py`

**Change:**
```python
# Line 1141: Change from
conformal = profile.get("conformal_intervals", {})
# To
conformal = profile.get("extended_models", {}).get("conformal_prediction", {})

# Line 1159: Change from
shap_data = profile.get("shap_explanations", {})
# To
shap_data = profile.get("extended_models", {}).get("shap_explanations", {})

# Line 1190: Change from
analogs = profile.get("historical_analogs", {})
# To
analogs = profile.get("extended_models", {}).get("dtw_analogs", {})
```

**Risk:** Zero -- the current code always gets empty dicts (the keys don't exist at top level). The fix reads from where the data actually lives.

**Test:** Create a profile with `extended_models.conformal_prediction`, `extended_models.shap_explanations`, and `extended_models.dtw_analogs` populated; verify Section 9 now renders conformal interval tables, SHAP driver narratives, and DTW analog forecasts.

---

## Minor Fix M1: Clean up stale `geopolitical_risk` fallback key

**File:** `operator1/report/report_generator.py`, line 2577

**Change:**
```python
# From:
conflict = profile.get("conflict_risk", profile.get("geopolitical_risk", {}))
# To:
conflict = profile.get("conflict_risk", {})
```

**Risk:** Zero -- `geopolitical_risk` is never produced by any module.

---

## Execution Order

1. **Fix 1** (dead code deletion) -- independent, zero risk
2. **Fix 4** (key mismatches) -- independent, zero risk, highest impact
3. **Fix 2** (wire economic position) -- independent, low risk
4. **Fix 3** (validation auto-fix) -- more complex, implement after others
5. **M1** (stale key cleanup) -- trivial, do alongside any fix

All fixes are in a single file (`report_generator.py`), so they can be done in one commit.

---

## Verification Checklist

- [ ] `python3.12 -m py_compile operator1/report/report_generator.py` -- syntax OK
- [ ] `python3.12 -m pytest tests/test_phase7_report.py -v` -- all existing tests pass
- [ ] Section 9 renders conformal intervals when profile has conformal data
- [ ] Section 9 renders SHAP narratives when profile has SHAP data
- [ ] Section 9 renders DTW analog forecasts when profile has DTW data
- [ ] Economic position section appears in PRO and PREMIUM reports
- [ ] Dead code removed (file shrinks by ~140 lines)
- [ ] LLM validation either auto-fixes or logs honestly

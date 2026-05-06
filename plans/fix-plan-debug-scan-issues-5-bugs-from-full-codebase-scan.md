# Fix Plan: Debug Scan Issues

5 issues found during comprehensive debug scan of 443 imports, 38 sub-stages, 50 scoring weight refs, 93 PipelineState fields, and all Python source files.

---

## Fix 1 (HIGH): Kelly criterion dead code -- `mc_result` NameError

**File:** `operator1/hedge_fund/engine.py`
**Line:** 1188
**Root cause:** `_compute_position_signal()` is a module-level function (not a closure) but references `mc_result` as a bare variable at line 1188. The function signature (line 1072) does not include `mc_result` as a parameter. The caller at line 1335 does not pass it. The Kelly criterion code (lines 1187-1220) always hits `NameError`, caught by try/except. `kelly_fraction`, `half_kelly_size`, `kelly_edge` are never computed.

**Fix:**
1. Add `mc_result: Any = None` parameter to `_compute_position_signal()` function signature
2. Update the call site at line 1335 to pass `mc_result`:
```python
hf.position = _compute_position_signal(
    hf, cache, signal_ic_result, survival_controller,
    forecast_result, filing_calendar_result, mc_result,
)
```

---

## Fix 2 (MEDIUM): `classify_economic_plane` not imported at main.py:2919

**File:** `main.py`
**Line:** 2919
**Root cause:** `classify_economic_plane()` is called in Step 7 profile building but the import from `operator1.analysis.economic_planes` was only done earlier in a different scope (inside PipelineState usage). The function is undefined at line 2919 scope. Falls through try/except to the except handler which sets `{"primary_plane": "unknown"}`.

**Fix:**
Add the import inside the try block at line 2917:
```python
try:
    from operator1.analysis.economic_planes import classify_economic_plane
    plane_info = classify_economic_plane(...)
```

---

## Fix 3 (LOW): `Any` type not imported in main.py

**File:** `main.py`
**Line:** 3145
**Root cause:** `_inst_analysis: dict[str, Any]` uses `Any` from typing but `Any` is never imported. Works in Python 3.12 because annotations are lazy-evaluated, but technically incorrect.

**Fix:**
Remove the type annotation (main.py is not a library, type annotations on locals are unnecessary):
```python
_inst_analysis = {"available": False}
```

---

## Fix 4 (LOW): `extra_vars` not copied back from PipelineState

**File:** `main.py`
**Line:** ~2679 (copy-back section)
**Root cause:** After `run_stages()` completes, `_ps.extra_vars` is not copied back to a local variable. All other stage result fields are copied. Minimal impact since `extra_vars` is only consumed by temporal models that have already run.

**Fix:**
Add to copy-back section (after `_economic_plane` line):
```python
_extra_vars = _ps.extra_vars
```
This is for completeness -- no downstream code uses it, but it keeps the copy-back section consistent.

---

## Fix 5 (LOW): `_build_macro_section()` dead code

**File:** `operator1/report/report_generator.py`
**Root cause:** `_build_macro_section()` is defined but never called. Macro data is rendered via `_build_macro_quadrant()` instead. Dead code from when macro rendering was split into two functions but the old one was never removed.

**Fix:**
Delete the `_build_macro_section()` function definition. It adds ~30 lines of dead code.

---

## Implementation Order

1. Fix 1 (Kelly criterion) -- highest impact, enables position sizing
2. Fix 2 (classify_economic_plane import) -- medium impact, enables profile economic plane
3. Fix 3 (Any annotation) -- trivial, one-line change
4. Fix 4 (extra_vars copy-back) -- trivial, one-line addition
5. Fix 5 (dead code removal) -- cleanup

## Files Modified

| # | File | Fix | Lines Changed |
|---|------|-----|---------------|
| 1 | `operator1/hedge_fund/engine.py` | Add mc_result param + pass at call site | ~3 lines |
| 2 | `main.py` | Add import inside try block | ~1 line |
| 3 | `main.py` | Remove `Any` annotation | ~1 line |
| 4 | `main.py` | Add extra_vars copy-back | ~1 line |
| 5 | `operator1/report/report_generator.py` | Delete dead function | ~-30 lines |

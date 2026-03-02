# Defensive Hardening Plan -- 7 Edge Cases

The debug audit found **no actual bugs**, but identified 7 areas where defensive hardening would make the code more resilient to future changes or unexpected data. Here is the plan for each.

---

## 1. `regime_label` "nan" string leaking

**Current state**: [`regime_detector.py:818`](operator1/models/regime_detector.py:818) uses `labels.astype(str).where(labels.notna(), "unknown")` which works correctly because `.where()` uses the pre-astype NaN mask.

**Risk**: A future developer might refactor this line to something like `labels.fillna("unknown").astype(str)` or change the order, which could produce "nan" strings if NaN leaks through a different code path.

**Hardening**:
- Add an explicit assertion after the conversion: `assert "nan" not in early.regime_labels.values`
- Add a sanitization step: replace any lingering "nan" strings with "unknown" as a safety net
- Add a comment explaining WHY the current ordering matters

**Where**: [`operator1/models/regime_detector.py`](operator1/models/regime_detector.py) line ~818

---

## 2. Column collision during enriched timeline merge

**Current state**: [`main.py:1228`](main.py:1228) uses `if col not in cache.columns` to skip existing columns.

**Risk**: This silently drops enriched values when a column already exists. If a future step accidentally adds a column with the same name (e.g., some other module adds `survival_mode`), the enriched timeline's version would be silently lost.

**Hardening**:
- Add a debug-level log when a column is skipped due to already existing in cache
- Consider whether certain columns (like `regime_state`, `survival_intensity`) should ALWAYS overwrite since they are authoritative from the enriched timeline

**Where**: [`main.py`](main.py) Step 5.5 merge loop, line ~1226

---

## 3. Index misalignment between enriched timeline and cache

**Current state**: No misalignment because `compute_survival_timeline()` copies `daily_cache` directly.

**Risk**: If `compute_survival_timeline()` ever filters rows (e.g., drops NaN-heavy days), the enriched timeline would have fewer rows than cache. The merge `cache[col] = enriched_timeline_result.timeline[col]` would then introduce NaN for missing index positions.

**Hardening**:
- Add an index length check before merging: `assert len(enriched_timeline_result.timeline) == len(cache)`
- If lengths differ, log a warning and use `.reindex(cache.index)` with explicit fill values

**Where**: [`main.py`](main.py) Step 5.5 merge loop, line ~1216

---

## 4. Double regime detection

**Current state**: Step 6 checks `if regime_detector is None` and skips if already set.

**Risk**: If Step 5.5 raises an exception partway through (after setting `regime_detector` but before completing the enriched timeline), Step 6 would skip regime detection because `regime_detector` is not None, but the detector might be in a partial state.

**Hardening**:
- Check `regime_detector is not None and hasattr(regime_detector, 'result')` instead of just `is not None`
- Also check that regime columns actually exist in cache: `"regime_label" in cache.columns`

**Where**: [`main.py`](main.py) Step 6 regime detection skip logic, line ~1307

---

## 5. `skip_models` path leaving variables undefined

**Current state**: All variables initialized to `None` before the conditional blocks.

**Risk**: Minimal. But if someone adds a new variable reference in Step 7 that depends on an enriched timeline field without checking for None, it would fail.

**Hardening**:
- No code change needed. The current pattern is correct.
- Add a comment block at the top of Step 7 listing all variables that may be None when `--skip-models` is used

**Where**: [`main.py`](main.py) Step 7, line ~1628

---

## 6. Non-standard HMM labels with n_regimes > 4

**Current state**: [`_map_combined_state()`](operator1/analysis/survival_timeline.py:500) falls back to the "unknown" regime row when it encounters a label not in the map.

**Risk**: If someone uses `n_regimes=6` for finer-grained regime detection, the enriched timeline would treat 2 of the 6 regimes as "unknown", losing information. The combined state and intensity would be less accurate.

**Hardening**:
- Add a log warning when a regime label is not found in `_COMBINED_STATE_MAP`, including the label name, so operators can see they need to extend the map
- Consider adding a config-driven extension point where additional regime labels can be mapped to combined states without code changes

**Where**: [`operator1/analysis/survival_timeline.py`](operator1/analysis/survival_timeline.py) `_map_combined_state()`, line ~500

---

## 7. `regime_confidence` column sourcing

**Current state**: HMM max posterior flows correctly through the chain.

**Risk**: If HMM produces posterior probabilities that are all very low (e.g., max = 0.3 for all days), the `regime_confidence` might not meaningfully distinguish "confident" from "uncertain" regimes. Also, if future code adds a `regime_confidence` column somewhere else in the pipeline before Step 5.5, the enriched timeline's version would be silently dropped by the merge guard.

**Hardening**:
- Log the min/max/mean of `regime_confidence` after computation to help diagnose low-confidence situations
- Consider normalizing confidence to [0, 1] explicitly if the raw max-posterior values have a compressed range

**Where**: [`operator1/analysis/survival_timeline.py`](operator1/analysis/survival_timeline.py) `compute_enriched_survival_timeline()`, around line ~583

---

## Implementation Priority

All 7 items are **low-severity defensive measures** -- the current code works correctly. The recommended implementation order:

- [ ] Item 1 -- "nan" string sanitization: add safety net replace + comment
- [ ] Item 6 -- Unknown regime label warning: add log.warning for unmapped labels  
- [ ] Item 3 -- Index length check: add assertion before merge
- [ ] Item 2 -- Column collision logging: add debug log for skipped columns
- [ ] Item 4 -- Partial detector state check: strengthen the None check
- [ ] Item 7 -- Confidence range logging: add summary stats log
- [ ] Item 5 -- Skip-models comment: documentation only

These are all small, isolated changes that can be done in a single follow-up commit.

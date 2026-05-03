# Fix Plan: 6 Confirmed Issues from Deep Scan

## Issue 1: Missing Balance Sheet Fields in CompanyFacts Fallback (HIGH)

**File:** `backtest_runner.py:384`
**Problem:** `_critical_balance_fields` dict is missing 5 fields that the SEC EDGAR wrapper can extract. When edgartools returns partial data due to SEC 429 rate limiting, the CompanyFacts fallback doesn't fill these fields.
**Impact:** `current_ratio = NaN` for all days -> survival probability stuck at 0.5 -> wrong regime -> cascading degradation of HF grade and prediction quality.

**Fix:**
Add 5 missing fields to `_critical_balance_fields` at line 384:
```python
_critical_balance_fields = {
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],  # ADD
    "cash_and_equivalents": [...],
    "total_liabilities": ["Liabilities"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "short_term_debt": ["ShortTermBorrowings", "DebtCurrent", "CommercialPaper"],  # ADD
    "receivables": ["AccountsReceivableNetCurrent"],  # ADD
    "inventory": ["InventoryNet"],  # ADD
    "payables": ["AccountsPayableCurrent"],  # ADD
}
```

**Verification:** Re-run Stage 1 for AAPL, check `current_liabilities` in saved `balance_df.parquet`.

---

## Issue 2: ForwardPassResult Not Surviving Pickle Serialization (HIGH)

**File:** `operator1/pipeline_state.py:196`
**Problem:** `ForwardPassResult.model_states` contains fitted sklearn/torch model objects with C-extension references that can't be pickled. The salvage logic at line 196 tries to strip non-picklable sub-fields, but the entire result fails to serialize.
**Impact:** `forward_pass_result = None` after checkpoint save -> conformal (6.3) skipped, SHAP (6.6) empty, burn-out (5.2) gets no forward pass to calibrate from.

**Fix:** Two-part approach:

**Part A:** In `ForwardPassResult` (inside `forecasting.py`), add a `__getstate__`/`__setstate__` method that strips `model_states` and `conformal_calibrator` before pickling and restores them as `None` on unpickle:
```python
def __getstate__(self):
    state = self.__dict__.copy()
    state['model_states'] = {}  # Not picklable (fitted models)
    state['conformal_calibrator'] = None  # Thread-local refs
    return state
```

**Part B:** In `stage5_forward.py run_5_1_forward_pass()`, after the forward pass completes, extract the conformal calibrator's residuals into a picklable list and store separately on PipelineState so conformal (6.3) can rebuild the calibrator:
```python
if state.forward_pass_result and hasattr(state.forward_pass_result, 'conformal_calibrator'):
    cal = state.forward_pass_result.conformal_calibrator
    if cal:
        state._conformal_residuals = list(cal.scores)  # picklable
```

Then in `stage6_ensemble.py run_6_3_conformal()`, rebuild the calibrator from saved residuals if `forward_pass_result` is None.

**Verification:** Run 5.1, check `state_5.1.pkl` contains non-None `forward_pass_result`. Run 6.3, check `conformal_result` is non-None.

---

## Issue 3: Survival Probability Returns 0.5 for NaN Triggers (MEDIUM)

**File:** `operator1/analysis/survival_mode.py`
**Problem:** The sigmoid survival probability formula computes one component per trigger variable. When a trigger variable is NaN (e.g., `current_ratio` missing), the sigmoid of NaN distance returns 0.5 (neutral). With 2 of 6 components returning 0.5, the overall probability is diluted toward 0.5 even when the 4 healthy components indicate ~0.95.

**Fix:** In `compute_survival_probability()`, skip NaN trigger variables instead of defaulting to 0.5:
```python
components = []
weights = []
for var, threshold, weight in trigger_configs:
    value = cache[var].iloc[-1] if var in cache.columns else np.nan
    if pd.isna(value):
        continue  # Skip, don't contribute 0.5
    distance = (value - threshold) / abs(threshold)
    prob = 1 / (1 + np.exp(-distance * scale))
    components.append(prob)
    weights.append(weight)

if components:
    total_weight = sum(weights)
    survival_prob = sum(c * w / total_weight for c, w in zip(components, weights))
else:
    survival_prob = 0.5  # Only if ALL triggers are NaN
```

**Verification:** With current_liabilities missing, survival_probability should be ~0.85-0.95 for AAPL (healthy on all other triggers), not 0.5.

---

## Issue 4: Operating Margin NaN Due to Scale Mismatch (MEDIUM)

**File:** `operator1/features/derived_variables.py`
**Problem:** `operating_margin = ebit / revenue` where `ebit` is $42.8B (raw quarterly from filing) and `revenue` is $299M (daily interpolated from quarterly total distributed across business days). The frequency interpolator distributes flow variables to daily rates but `ebit` wasn't processed by the interpolator (it came through a different merge path).

**Fix:** In `_compute_profitability()`, use `operating_income` (which went through the same merge path as `revenue`) instead of `ebit` for the operating_margin calculation. Or better, check if both numerator and denominator are on the same scale by comparing magnitudes:
```python
def _compute_profitability(df):
    # Use operating_income (same interpolation as revenue) for margin
    num = df.get("operating_income", df.get("ebit"))
    den = df.get("revenue")
    if num is not None and den is not None:
        # Scale check: if ratio > 100 or < -100, likely scale mismatch
        ratio = num / den.where(den.abs() > EPSILON)
        ratio = ratio.where(ratio.abs() < 100)  # Cap absurd ratios
        df["operating_margin"] = ratio
```

**Verification:** `operating_margin` should be ~0.30-0.35 for AAPL (not NaN or > 100).

---

## Issue 5: Dead Code -- Duplicate compute_granger_causality (LOW)

**File:** `operator1/models/causality.py:66`
**Problem:** `compute_granger_causality()` exists in both `causality.py` and `granger_causality.py`. The stage module imports from `granger_causality.py`. The `causality.py` version is dead code.

**Fix:** Remove `compute_granger_causality()` from `causality.py`. Keep only the `compute_transfer_entropy()` function which is the module's actual purpose. Add a deprecation comment or re-export for backward compatibility:
```python
# causality.py -- Transfer Entropy only
# compute_granger_causality moved to granger_causality.py
```

**Verification:** `grep -r "from operator1.models.causality import compute_granger" operator1/` should return 0 results.

---

## Issue 6: 61 Profile Sections Injected Post-Build (LOW / DESIGN)

**File:** `operator1/report/profile_builder.py`
**Problem:** `build_company_profile()` creates a base profile with ~26 sections, then `main.py` / `backtest_runner.py` inject 35+ additional sections (behavioral_signals, complexity_signals, conflict_risk, corporate_structure, etc.) after the function returns. The report_generator reads all 61+ keys expecting them to be present.

**Fix:** This is a design improvement, not a bug fix. The cleanest approach is to add optional kwargs to `build_company_profile()` for the post-injection sections, so they're all set in one place:
```python
def build_company_profile(
    ...,
    # NEW: sections previously injected post-build
    enriched_timeline_result=None,
    filing_calendar_result=None,
    conflict_result=None,
    behavioral_signals=None,
    complexity_signals=None,
    ...
) -> dict:
```

However, this is a large refactor with 35+ new parameters. A pragmatic alternative: add a `profile.setdefault(key, {"available": False})` for every key the report_generator reads, inside `build_company_profile()`, so the report never crashes on a missing key.

**Verification:** Call `build_company_profile()` without post-injection, verify no KeyError when generating reports.

---

## Implementation Order

```
1. Fix Issue 1 (backtest_runner.py) -- highest impact, simplest change
2. Fix Issue 3 (survival_mode.py) -- medium impact, small change
3. Fix Issue 4 (derived_variables.py) -- medium impact, small change
4. Fix Issue 2 (pipeline_state.py + forecasting.py + stage modules) -- highest impact, most complex
5. Fix Issue 5 (causality.py) -- cleanup
6. Fix Issue 6 (profile_builder.py) -- design improvement, optional
```

Fixes 1-3 are one-line to five-line changes. Fix 4 is a scale guard. Fix 2 requires changes across 3 files but is well-scoped. Fix 5 is deletion. Fix 6 is optional refactoring.

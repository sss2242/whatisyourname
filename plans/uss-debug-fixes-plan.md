# USS Debug Fixes Plan

10 issues found during systematic scan of the Unified Survival System implementation. Fixes ordered by file to minimize context switching.

---

## Fix 1: Unused imports in survival_regime_controller.py

**File:** `operator1/analysis/survival_regime_controller.py`
**Lines:** 26, 389

- Remove `import numpy as np` (line 26) -- not used anywhere in the file
- Remove `from operator1.constants import EPSILON` (line 389) -- imported but never referenced

---

## Fix 2: Cash runway day computation inconsistency

**File:** `operator1/report/triage_card.py`
**Line:** 252

Change `runway_days = -c / (monthly_burn / 30.0)` to `runway_days = -c / (monthly_burn / 21.0)` to match the business-day convention used in `scenario_engine.py` line 102.

Both modules should use 21 business days per month consistently.

---

## Fix 3: Profile schema update

**File:** `operator1/report/profile_schema.py`
**Line:** 53 (OPTIONAL_PROFILE_KEYS set)

Add `unified_survival_system`, `scenario_analysis`, `corporate_structure`, `market_buying_power`, `product_catalysts`, `macro_indicators`, `synergies_applied` to `OPTIONAL_PROFILE_KEYS`.

---

## Fix 4: Bound aggregated predictions

**File:** `main.py`
**Location:** After `run_prediction_aggregation` completes (around line 2795)

After the prediction aggregator runs, apply `bound_forecast_dict` to `pred_result.predictions` as well. Extract point forecasts from `HorizonPrediction` objects, bound them, and write back.

---

## Fix 5: Remove duplicate scenario engine log

**File:** `operator1/analysis/scenario_engine.py`
**Line:** 380

Remove the `logger.info` call at the end of `run_scenario_engine()` since `main.py` already logs the same data at line 2949.

---

## Fix 6: Scope triage card to company distress regimes only

**File:** `main.py`
**Line:** 3547

Change `survival_controller.is_survival` to check for company_survival or extreme_survival specifically (not modified_survival, which means the company is healthy but country is in crisis).

---

## Fix 7: Add modified_survival bounding for debt/margins

**File:** `operator1/analysis/survival_regime_controller.py`
**Line:** 297

Change the guard from `if regime not in ("company_survival", "extreme_survival")` to include `modified_survival` for debt and margin variables only (not revenue/cash, since the company itself is healthy in modified_survival).

---

## Fix 8: Add USS report sections

**File:** `operator1/report/report_generator.py`

This is the largest fix. Add two new section builders:
- `_build_unified_survival_system_section(profile)` -- renders regime, triage, horizons, correlation override, early warning
- `_build_scenario_analysis_section(profile)` -- renders 3-scenario comparison table

Wire them into `_section_builders` dict and add to PRO and PREMIUM tier section lists.

---

## Fix 9: Add USS to backtest_runner.py

**File:** `backtest_runner.py`

In `run_stage2()` after temporal models, add:
- Create `SurvivalRegimeController` from cache
- Apply forecast bounding if in survival mode
- Run scenario engine if in survival mode
- Store results in `BacktestState`

In `run_stage3()`:
- Include USS and scenario data in profile

---

## Fix 10: Document Tier 3 reduced model enforcement

**File:** `operator1/analysis/survival_regime_controller.py`

Add a TODO comment in the `_TRIAGE_TABLE` definition documenting that Tier 3 "active" should eventually enforce "reduced" model subset (Kalman + baseline only) per the architecture spec. This is a future enhancement, not a bug.

---

## Execution Order

1. Fix 1 (unused imports) -- trivial, 2 line deletions
2. Fix 2 (cash runway) -- 1 line change
3. Fix 3 (profile schema) -- 1 addition to set
4. Fix 5 (duplicate log) -- 1 deletion
5. Fix 6 (triage card scope) -- 1 line change
6. Fix 7 (modified_survival bounding) -- small logic change
7. Fix 10 (TODO comment) -- 1 comment addition
8. Fix 4 (bound aggregated predictions) -- medium, main.py edit
9. Fix 9 (backtest_runner) -- medium, ~30 lines
10. Fix 8 (report sections) -- largest, ~60 lines

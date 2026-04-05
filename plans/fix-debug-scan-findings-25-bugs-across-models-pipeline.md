# Fix Debug Scan Findings -- 38 Bugs Across Models Pipeline

*Created: 2026-04-05*
*Updated: 2026-04-05 (second-pass scan added 13 more findings)*

Full pyflakes static analysis + manual code review across all 60+ model modules, main.py, dashboard.py, and scoring_weights integration. 38 findings categorized by severity.

## Fix Status

| Severity | Total | Fixed | Remaining |
|----------|-------|-------|-----------|
| Critical (NameError crash) | 3 | 3 (PR #1) | 0 |
| High (wrong behavior) | 3 | 2 (PR #1) | 1 |
| Medium (report-profile mismatch) | 2 | 2 (PR #1) | 0 |
| Medium (config disconnection) | 10 | 0 | 10 |
| Medium (dead code/logic gaps) | 15 | 0 | 15 |
| Low (unused imports, etc.) | 5 | 0 | 5 |

## Second-Pass Findings (13 additional)

### Scoring Weights Config Disconnections (10 keys)

7 YAML keys in `config/scoring_weights.yml` are defined but never consumed:
- `conflict_propagation` -- linked conflict propagation uses hardcoded weights
- `forecasting_min_data` -- forecasting.py uses hardcoded min_periods
- `signal_filtering` -- signal_ic.py uses hardcoded IC threshold
- `uss_forecast_bounds` -- survival_regime_controller uses hardcoded bounds
- `uss_horizons` -- survival_regime_controller uses hardcoded horizons
- `vanity_labels` -- vanity.py uses hardcoded label breakpoints
- `vanity_weight_shift_pct` -- hierarchy_weights.py uses hardcoded 5% shift

3 YAML keys are partially connected (code exists but doesn't read from config):
- `fuzzy_protection` -- module uses hardcoded membership functions
- `scenario_engine` -- module uses hardcoded scenario parameters
- `model_regime_affinity` -- prediction_aggregator builds inline defaults

### Report-Profile Key Mismatches (2 keys) -- FIXED

- `technical_patterns` -- report reads this key but data stored in `extended_models.candlestick_patterns`. Fixed: added fallback chain.
- `linked_conflict` -- report reads this key but data stored as flat keys in `conflict_risk`. Fixed: added fallback to read flat keys.

### run.py CLI Arg Gap (1 finding)

- `run.py` has `--quiet` flag that `main.py` doesn't recognize (harmless)

---

## CRITICAL BUGS (3 Runtime NameErrors)

### Bug 1: `tier_map` and `_get_tier_for_variable` undefined in prediction_aggregator.py

**File:** `operator1/models/prediction_aggregator.py:2153`
**Error:** `NameError: name 'tier_map' is not defined` + `NameError: name '_get_tier_for_variable' is not defined`
**Impact:** When `survival_adjusted` is True (survival mode active), the per-tier confidence multiplier code at Phase 2.5 crashes. Prediction aggregation FAILS for any company in survival mode.

**Root cause:** `tier_map` is never passed as a parameter to `run_prediction_aggregation()` (not in function signature at line 1814) and never defined as a local variable. `_get_tier_for_variable()` is defined in `forecasting.py:211` but never imported into prediction_aggregator.py.

**Fix:**
1. Add import: `from operator1.models.forecasting import _get_tier_for_variable, _build_tier_map`
2. At the top of `run_prediction_aggregation()`, build the tier map: `tier_map = _build_tier_map()`
3. Alternatively, inline a simpler approach: load `config/survival_hierarchy.yml` directly inside the function.

---

### Bug 6: `np` undefined in survival_mode.py (Cox PH fallback path)

**File:** `operator1/analysis/survival_mode.py:245`
**Error:** `NameError: name 'np' is not defined`
**Impact:** When lifelines is not installed, the `except ImportError` branch at line 243 tries to return `pd.Series(np.nan, ...)` but `import numpy as np` is at line 247 (AFTER this line). This crashes the Cox PH fallback.

**Fix:** Replace `np.nan` with `float('nan')` at line 245, OR move `import numpy as np` before the try/except block (before line 241).

---

### Bug 7: `payout` undefined in six_derived_proxies.py (quarterly SIX path)

**File:** `operator1/features/six_derived_proxies.py:4094`
**Error:** `NameError: name 'payout' is not defined`
**Impact:** When `target_frequency == "Q"` (quarterly synthetic generation for Swiss companies), line 4094 passes `payout_ratio=payout` but `payout` is only defined at line 3904 in the ANNUAL code path below. The Q path at line 4081 executes first.

**Fix:** Compute payout before the frequency branch:
```python
# Before line 4081
payout = total_dividends / max(net_income, 1) if net_income > 0 else 0.7
```
Or use `payout_ratio` from `result.estimated_payout_ratio` (computed at line 688-689).

---

## HIGH SEVERITY (2 Incorrect Behavior Bugs)

### Bug 9: Dashboard shows pre-fusion HF signal instead of fused signal

**File:** `dashboard.py:567`
**Issue:** `pos.get("signal")` reads from `hf["position"]` (pre-fusion signal) instead of `hf["fusion"]["fused_signal"]` (cross-pipeline fused signal from 8 fusion methods).

**Fix:** Add a check for the fused signal:
```python
# After line 569, add:
fusion = hf.get("fusion", {})
if fusion.get("available"):
    _fused = fusion.get("fused_signal", 0)
    _flbl = fusion.get("fused_label", "hold").upper()
    _card("Fused Signal", f"{_fused:+.2f} - {_flbl}", 
          f"Conviction: {fusion.get('fused_conviction', 0):.0%}", "merge_type")
```

---

### Bug 10: Scoring weights panel missing HF weights

**File:** `dashboard.py:1124`
**Issue:** Panel loads from `config/scoring_weights.yml` but HF weights are in separate `config/hedge_fund_weights.yml`. HF-specific weights (tier weights, DCF WACC defaults, sector calibration) are not editable from the dashboard.

**Fix:** In `render_scoring_weights()`, add a second tab or section that loads and saves `config/hedge_fund_weights.yml` using the same pattern as the main weights panel.

---

## MEDIUM SEVERITY (15 Dead Code / Logic Gap Bugs)

### Bug 11: `zone_map` dead code in financial_health.py

**File:** `operator1/models/financial_health.py:1003`
**Issue:** `zone_map` dict is built but never used. The Altman Z-Score zone classification uses hardcoded if/elif instead.
**Fix:** Either use `zone_map` for the classification or delete it.

---

### Bug 12: `_equity_fraction` unused in financial_health.py

**File:** `operator1/models/financial_health.py:1047`
**Issue:** `_equity_fraction` is computed but never used -- suggests an incomplete EV/EBITDA calculation.
**Fix:** Investigate whether this was meant to feed into EV computation. If not needed, delete.

---

### Bug 14: `prev_pairs` unused in granger_causality.py

**File:** `operator1/models/granger_causality.py:446`
**Issue:** `prev_pairs` in time-varying Granger computation is assigned but never used. The disappearing pairs detection logic appears incomplete.
**Fix:** Complete the disappearing pairs logic: `disappearing = prev_pairs - current_pairs`.

---

### Bug 18: `cce_score` unused in accruals_forensics.py

**File:** `operator1/hedge_fund/accruals_forensics.py:168`
**Issue:** Cash Conversion Efficiency score is computed but never added to the composite. The 5-component composite at the top of the module lists CCE as component 5 (10% weight) but the score is discarded.
**Fix:** Include `cce_score` in the composite calculation with its 10% weight.

---

### Bug 20: `beneish_dict` unused in earnings_smoothing.py

**File:** `operator1/hedge_fund/earnings_smoothing.py:105`
**Issue:** `beneish_dict` extracted from `fh_result` but never used. Component 1 (Beneish M-Score probability, 25% weight) appears to not actually consume this data.
**Fix:** Wire `beneish_dict` values into the smoothing composite component 1.

---

### Bug 21: `roic_slope` unused in hedge_fund/engine.py

**File:** `operator1/hedge_fund/engine.py:472`
**Issue:** ROIC slope (trend direction) computed in return spread analysis but discarded. This would be useful for the return spread narrative.
**Fix:** Store in `ReturnSpreadResult` as `roic_trend`.

---

### Bug 22: `mean_surprise` unused in hedge_fund/engine.py

**File:** `operator1/hedge_fund/engine.py:561`
**Issue:** Mean standardized surprise computed but not used in surprise probability output.
**Fix:** Store in `SurpriseResult` as `historical_sue_mean`.

---

### Bug 23: `catalyst_weights` unused in hedge_fund/fusion.py

**File:** `operator1/hedge_fund/fusion.py:561`
**Issue:** Catalyst-driven temporal weights are computed (Method 5) but never applied to the final fused signal.
**Fix:** Apply catalyst weights to modulate the signal before the final clamp.

---

### Bugs 13, 15, 16, 17, 19, 24, 25 (Minor dead code)

These are computed-but-unused variables that don't affect correctness but should be cleaned up:
- `avg_price_proxy` in game_theory.py:217
- `total` in model_synergies.py:184
- `point_estimate` in monte_carlo.py:766
- `residual` in cycle_decomposition.py:310
- `rev_series` in earnings_smoothing.py:94
- `target_ts` in prediction_log.py:284
- `last_close` in scenario_engine.py:252

**Fix:** Either use these values or delete the assignments.

---

## LOW SEVERITY (Cleanup)

27 unused imports across model files. 4 f-strings with missing placeholders in report_generator.py (lines 3376, 3683, 3866, 3962). These should be cleaned up in a separate pass.

---

## Execution Order

1. **Fix Bug 1** (prediction_aggregator tier_map) -- highest impact, blocks all survival mode predictions
2. **Fix Bug 6** (survival_mode np.nan) -- blocks Cox PH fallback
3. **Fix Bug 7** (six_derived_proxies payout) -- blocks quarterly SIX proxy generation
4. **Fix Bug 9** (dashboard fused signal) -- shows wrong signal to users
5. **Fix Bug 10** (dashboard HF weights) -- missing configuration UI
6. **Fix Bugs 18, 20, 23** (HF composite gaps) -- silent quality degradation
7. **Fix Bug 14** (Granger disappearing pairs) -- incomplete analysis
8. **Fix remaining medium bugs** (11, 12, 13, 15-17, 19, 21-22, 24-25)
9. **Clean up unused imports and f-strings**

---

## Files Modified (Summary)

| File | Bugs Fixed |
|------|-----------|
| `operator1/models/prediction_aggregator.py` | Bug 1 |
| `operator1/analysis/survival_mode.py` | Bug 6 |
| `operator1/features/six_derived_proxies.py` | Bug 7 |
| `dashboard.py` | Bugs 9, 10 |
| `operator1/hedge_fund/accruals_forensics.py` | Bug 18 |
| `operator1/hedge_fund/earnings_smoothing.py` | Bug 20 |
| `operator1/hedge_fund/fusion.py` | Bug 23 |
| `operator1/models/granger_causality.py` | Bug 14 |
| `operator1/models/financial_health.py` | Bugs 11, 12 |
| `operator1/hedge_fund/engine.py` | Bugs 21, 22 |
| `operator1/models/game_theory.py` | Bug 13 |
| `operator1/models/model_synergies.py` | Bug 15 |
| `operator1/models/monte_carlo.py` | Bug 16 |
| `operator1/models/cycle_decomposition.py` | Bug 17 |
| `operator1/hedge_fund/earnings_smoothing.py` | Bug 19 |
| `operator1/analysis/prediction_log.py` | Bug 24 |
| `operator1/analysis/scenario_engine.py` | Bug 25 |

# Fix Scoring Weights Disconnection and Debug Scan Findings

## Problem Summary

The debug scan found that 13 of 24 `scoring_weights.yml` sections are disconnected -- the config exists and the dashboard panel renders edit controls, but analytical modules hardcode values instead of reading from config. Users editing weights via the dashboard get no effect.

## Approach

Each analytical module needs a 1-3 line change: replace the hardcoded default with a `get_weight()` call that falls back to the same hardcoded value. This is non-breaking -- if the config section is missing, the module behaves identically to today.

Pattern:
```python
# BEFORE (hardcoded)
THRESHOLD = 1.0

# AFTER (config-backed with same default)
from operator1.scoring_weights import get_weight
THRESHOLD = get_weight("survival_thresholds.current_ratio", 1.0)
```

## Tasks

### Task 1: Wire `frequency_fusion.py` to scoring_weights
- File: `operator1/models/frequency_fusion.py`
- Replace hardcoded `_HORIZON_WEIGHTS` dict with `get_weight("frequency_fusion", _HORIZON_WEIGHTS)` pattern
- The module-level dict stays as the default; at call time, check config first

### Task 2: Wire `conflict_risk.py` to scoring_weights  
- File: `operator1/features/conflict_risk.py`
- Replace hardcoded conflict component weights (0.40/0.20/0.25/0.15) with `get_weight("conflict_weights.*")` calls

### Task 3: Wire `vanity.py` to scoring_weights
- File: `operator1/analysis/vanity.py`
- Replace hardcoded component weights (0.15/0.25/0.30/0.15/0.15) with `get_weight("vanity_weights.*")` calls

### Task 4: Wire `survival_regime_controller.py` to scoring_weights
- File: `operator1/analysis/survival_regime_controller.py`
- Replace hardcoded USS model switching params with `get_weight("uss_model_switching.*")` calls

### Task 5: Wire `financial_health.py` to scoring_weights
- File: `operator1/models/financial_health.py`
- Replace hardcoded Altman Z zones (2.99/1.81) and Beneish M threshold (-2.22) with `get_weight("financial_health.*")` calls

### Task 6: Wire `fuzzy_protection.py` to scoring_weights
- File: `operator1/analysis/fuzzy_protection.py`
- Replace hardcoded fuzzy membership function parameters with `get_weight("fuzzy_protection.*")` calls

### Task 7: Wire `conformal.py` default coverage to scoring_weights
- File: `operator1/models/conformal.py`
- Replace hardcoded target_coverage=0.90 default with `get_weight("conformal.target_coverage", 0.90)`

### Task 8: Wire `main.py` survival_blend to scoring_weights
- File: `main.py` 
- Replace hardcoded `0.4 * _sig + 0.6 * _cox` with weights from `get_weight("survival_blend.*")`

### Task 9: Wire `hedge_fund/fusion.py` tables to scoring_weights
- File: `operator1/hedge_fund/fusion.py`
- Load `_REGIME_IC_TABLE` and `_CATALYST_TIER_WEIGHTS` from config with hardcoded fallback

### Task 10: Wire `graph_risk.py` edge weights to scoring_weights
- File: `operator1/models/graph_risk.py`
- Replace hardcoded edge weights with `get_weight("graph_edge_weights.*")` calls

### Task 11: Remove dead `combined_confidence` computation
- File: `main.py` lines 1228-1251
- Remove or comment out the unused `combined_confidence` computation (no downstream consumer)

### Task 12: Clean up duplicate `compute_granger_causality`
- File: `operator1/models/causality.py`
- Rename the legacy version to `compute_transfer_entropy_granger` or add a deprecation note

### Task 13: Fix cosmetic parameter naming
- File: `operator1/estimation/frequency_interpolator.py`
- Rename `daily_index` parameter to `target_index` in `_compute_interpolation_confidence()`

### Task 14: Remove 7 dead config sections or add placeholder readers
- File: `config/scoring_weights.yml`
- Add comments marking `vanity_weight_shift_pct`, `conflict_propagation`, `vanity_labels`, `uss_horizons`, `uss_forecast_bounds`, `signal_filtering`, `forecasting_min_data` as planned/unimplemented

## Priority Order

1. Tasks 1-10 (scoring_weights wiring) -- highest impact, fixes the dashboard disconnect
2. Task 11 (dead computation removal) -- cleanup
3. Task 12-13 (naming/duplication) -- cleanup
4. Task 14 (dead config annotation) -- documentation

## Risk Assessment

- All changes use the `get_weight(key, default)` pattern which falls back to the current hardcoded value when the config key is missing
- No behavioral change for users who haven't edited scoring_weights.yml
- Users who HAVE edited via the dashboard will finally see their changes take effect
- No new dependencies introduced

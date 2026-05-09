# Batch A: Full Implementation Plan -- ThresholdRegistry + Burn-Out Cleanup

*Problems 1 (threshold fragmentation), 2 (forward pass scale mixing), 6 (burn-out contamination)*
*Based on debug scan of all 6 threshold sources + 14 patterns from 7 open-source projects*

---

## Debug Scan Summary: Current Threshold Fragmentation

### 6 threshold sources that can disagree

| # | Source | Location | Format | Values |
|---|--------|----------|--------|--------|
| 1 | `_COMPANY_THRESHOLDS` | [`survival_mode.py:68`](operator1/analysis/survival_mode.py:68) | `{"current_ratio_lt": 1.0, ...}` | Reads from `scoring_weights.yml` via `get_weight()` |
| 2 | `DEFAULT_SURVIVAL_THRESHOLDS` | [`monte_carlo.py:150`](operator1/models/monte_carlo.py:150) | `{"current_ratio": ("lt", 1.0), ...}` | Hardcoded dict (different key format!) |
| 3 | `SECTOR_SURVIVAL_OVERRIDES` | [`monte_carlo.py:164`](operator1/models/monte_carlo.py:164) | `{"technology": {"current_ratio": ("lt", 0.7)}}` | Hardcoded dict (MC-only) |
| 4 | `sector_overrides` in `scoring_weights.yml` | [`scoring_weights.yml`](config/scoring_weights.yml) | YAML nested under `survival_thresholds` | Read by survival_mode.py only |
| 5 | `adaptive_thresholds` | [`main.py:2315`](main.py:2315) | `ThresholdSet` dataclass | Runtime-computed, passed to survival_mode only |
| 6 | `scenario_engine.py` hardcoded | [`scenario_engine.py:463`](operator1/analysis/scenario_engine.py:463) | `{"fcf_yield_lt": 0.0, ...}` | Fallback hardcoded dict |

### Key format mismatch

survival_mode uses `"current_ratio_lt": 1.0` (key has `_lt` suffix, value is bare float).
monte_carlo uses `"current_ratio": ("lt", 1.0)` (key is bare, value is (operator, float) tuple).
These are the SAME threshold expressed in incompatible formats.

### 5 consumers that read thresholds

| # | Consumer | File:Line | How it reads | Sector-aware? |
|---|----------|-----------|-------------|---------------|
| 1 | `compute_company_survival_flag()` | [`survival_mode.py:97`](operator1/analysis/survival_mode.py:97) | `dict(thresholds or _COMPANY_THRESHOLDS)` + sector_overrides from yml | Yes (since PR #4) |
| 2 | `run_monte_carlo()` | [`monte_carlo.py:915`](operator1/models/monte_carlo.py:915) | `survival_thresholds` param or `DEFAULT_SURVIVAL_THRESHOLDS` | Only via `get_sector_aware_thresholds()` in stage5 |
| 3 | `compute_reverse_stress_test()` | [`scenario_engine.py:458`](operator1/analysis/scenario_engine.py:458) | `_load_company_thresholds()` or hardcoded fallback | No |
| 4 | `compute_survival_probability()` | [`survival_mode.py:322`](operator1/analysis/survival_mode.py:322) | `thresholds or _COMPANY_THRESHOLDS` | No (missing sector) |
| 5 | MC multivariate `run_multivariate_monte_carlo()` | [`monte_carlo.py:1373`](operator1/models/monte_carlo.py:1373) | `survival_thresholds or DEFAULT_SURVIVAL_THRESHOLDS` | No |

### Burn-out learner data flow

```
forward_pass (day-by-day walk)
  |-- for each day: learner.update(regime, per_model_preds, actual, variable=var_name)
  |-- stores: self._regime_weighted_returns[regime].append({"variable": var, "actual": actual})
  |
  v
ExponentialGradientWeightLearner.get_regime_distributions(variable="return_1d")
  |-- filters entries where e["variable"] == "return_1d"
  |-- FALLBACK: if no match, uses |value| < 1.0 filter (the redundant guard)
  |
  v
stage5_forward.py:199  _burnout_dists = state.burnout_result.regime_distributions
  |
  v
run_monte_carlo(burnout_distributions=_burnout_dists)
  |-- monte_carlo.py:1436  _MAX_RETURN_MEAN/STD scale guard (REDUNDANT)
  |-- uses distributions for regime-specific path generation
```

---

## Implementation Plan

### Phase 1: ThresholdRegistry (new file, ~180 lines)

**New file:** `operator1/analysis/threshold_registry.py`

```python
"""Centralized threshold registry -- single source of truth.

Replaces 6 fragmented threshold sources with one layered composition:
  Code defaults -> scoring_weights.yml -> sector overrides -> adaptive -> company

Patterns: Hydra (composition), vectorbt (singleton), dynaconf (layered priority),
OpenBB (frozen dataclass), dynaconf (validation).
"""

from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType

@dataclass(frozen=True)
class SurvivalThresholds:
    """Immutable threshold contract. Field names match survival_mode.py keys."""
    current_ratio_lt: float = 1.0
    debt_to_equity_abs_gt: float = 3.0
    fcf_yield_lt: float = 0.0
    drawdown_252d_lt: float = -0.40
    conflict_intensity_gt: float = 0.70
    inst_flow_momentum_lt: float = -0.15

class ThresholdRegistry:
    """Single registry for all survival thresholds across the pipeline.
    
    Created once in main.py after adaptive calibration (Step 5j).
    Consumed via get_registry() by survival_mode, MC, scenario_engine, USS.
    """
    
    def __init__(self, base_config, sector="", adaptive_thresholds=None, company_adj=None):
        raw = self._compose(base_config, sector, adaptive_thresholds, company_adj)
        self._validate(raw)
        self._thresholds = SurvivalThresholds(**raw)
    
    @property
    def survival(self) -> SurvivalThresholds:
        """For survival_mode.py consumption."""
        return self._thresholds
    
    @property 
    def survival_dict(self) -> dict[str, float]:
        """For survival_mode.py backward compat (dict format with _lt/_gt keys)."""
        t = self._thresholds
        return {
            "current_ratio_lt": t.current_ratio_lt,
            "debt_to_equity_abs_gt": t.debt_to_equity_abs_gt,
            "fcf_yield_lt": t.fcf_yield_lt,
            "drawdown_252d_lt": t.drawdown_252d_lt,
            "conflict_intensity_gt": t.conflict_intensity_gt,
            "inst_flow_momentum_lt": t.inst_flow_momentum_lt,
        }
    
    @property
    def mc_dict(self) -> dict[str, tuple[str, float]]:
        """For monte_carlo.py consumption (variable -> (operator, value))."""
        t = self._thresholds
        return {
            "current_ratio": ("lt", t.current_ratio_lt),
            "debt_to_equity_abs": ("gt", t.debt_to_equity_abs_gt),
            "fcf_yield": ("lt", t.fcf_yield_lt),
            "drawdown_252d": ("lt", t.drawdown_252d_lt),
        }
    
    def _compose(self, base_config, sector, adaptive, company_adj):
        """Layered merge: defaults -> config -> sector -> adaptive -> company."""
        # Layer 1: code defaults
        merged = {f.name: f.default for f in SurvivalThresholds.__dataclass_fields__.values()}
        # Layer 2: scoring_weights.yml
        if base_config:
            st = base_config.get("survival_thresholds", {})
            KEY_MAP = {"current_ratio": "current_ratio_lt", ...}  # map yml keys to field names
            for yml_key, field_name in KEY_MAP.items():
                if yml_key in st:
                    merged[field_name] = float(st[yml_key])
        # Layer 3: sector overrides
        if sector and base_config:
            overrides = base_config.get("survival_thresholds", {}).get("sector_overrides", {})
            sector_key = sector.lower().replace(" ", "_")
            sector_cfg = overrides.get(sector_key, {})
            for yml_key, value in sector_cfg.items():
                field = KEY_MAP.get(yml_key)
                if field:
                    merged[field] = float(value)
        # Layer 4: adaptive (from compute_adaptive_thresholds)
        if adaptive and hasattr(adaptive, 'adapted') and adaptive.adapted:
            # ThresholdSet -> dict conversion
            ...
        # Layer 5: company-specific
        if company_adj:
            merged.update(company_adj)
        return merged
    
    def _validate(self, raw):
        assert 0 < raw.get("current_ratio_lt", 1.0) < 10
        assert 0 < raw.get("debt_to_equity_abs_gt", 3.0) < 50
        # ... bounds checks

# Module singleton
_registry: ThresholdRegistry | None = None

def init_registry(base_config, sector="", adaptive=None, company_adj=None):
    global _registry
    _registry = ThresholdRegistry(base_config, sector, adaptive, company_adj)
    return _registry

def get_registry() -> ThresholdRegistry:
    if _registry is None:
        # Fallback: create with defaults (for testing / backward compat)
        from operator1.scoring_weights import get_scoring_weights
        return ThresholdRegistry(get_scoring_weights())
    return _registry
```

### Phase 2: Wire registry into main.py (~15 lines)

**File:** [`main.py`](main.py) -- after Step 5j adaptive thresholds (~line 2345)

```python
# After adaptive thresholds computed, create the unified registry
from operator1.analysis.threshold_registry import init_registry
from operator1.scoring_weights import get_scoring_weights

_threshold_registry = init_registry(
    base_config=get_scoring_weights(),
    sector=target_profile.get("sector", ""),
    adaptive=_adaptive_thresholds,
    company_adj=None,  # future: per-company adjustments
)
logger.info("ThresholdRegistry initialized: %s", _threshold_registry.survival)
```

Also wire in [`backtest_runner.py`](backtest_runner.py) at equivalent location (~line 970).

### Phase 3: Rewire consumers to use registry (~40 lines across 4 files)

**File:** [`operator1/analysis/survival_mode.py`](operator1/analysis/survival_mode.py)
- Line 97: Replace `t = dict(thresholds or _COMPANY_THRESHOLDS)` with:
  ```python
  from operator1.analysis.threshold_registry import get_registry
  t = dict(thresholds or get_registry().survival_dict)
  ```
- Line 322 (`compute_survival_probability`): Same change
- Remove sector_overrides inline lookup (lines 99-110) -- registry already merged sector

**File:** [`operator1/models/monte_carlo.py`](operator1/models/monte_carlo.py)
- Line 915: Replace `survival_thresholds = DEFAULT_SURVIVAL_THRESHOLDS` with:
  ```python
  from operator1.analysis.threshold_registry import get_registry
  survival_thresholds = get_registry().mc_dict
  ```
- Line 1373 (multivariate MC): Same change
- Line 511 (`check_survival_triggers` default): Same change

**File:** [`operator1/analysis/scenario_engine.py`](operator1/analysis/scenario_engine.py)
- Line 458-463: Replace `_load_company_thresholds()` / hardcoded fallback with:
  ```python
  from operator1.analysis.threshold_registry import get_registry
  thresholds = get_registry().survival_dict
  ```

**File:** [`operator1/stages/stage5_forward.py`](operator1/stages/stage5_forward.py)
- Lines 169-183: Remove `get_sector_aware_thresholds()` call, replace with:
  ```python
  from operator1.analysis.threshold_registry import get_registry
  _mc_thresholds = get_registry().mc_dict
  ```

### Phase 4: Cleanup redundant code (~-50 lines)

**File:** [`operator1/models/monte_carlo.py`](operator1/models/monte_carlo.py)
- Delete `DEFAULT_SURVIVAL_THRESHOLDS` dict (line 150-155) -- replaced by registry
- Delete `SECTOR_SURVIVAL_OVERRIDES` dict (line 164-181) -- replaced by registry
- Delete `get_sector_aware_thresholds()` function (line 183-210) -- replaced by registry
- Delete `_MAX_RETURN_MEAN`/`_MAX_RETURN_STD` scale guard (lines 1436-1486) -- redundant now that burn-out per-variable filtering works

**File:** [`operator1/models/forecasting.py`](operator1/models/forecasting.py)
- In `get_regime_distributions()` (line 4647-4657): Remove the `|value| < 1.0` fallback filter block. Per-variable filtering is the primary path now; the fallback is dead code.

**File:** [`operator1/analysis/survival_mode.py`](operator1/analysis/survival_mode.py)
- Delete `_load_company_thresholds()` function (lines 47-67) -- replaced by registry
- Delete `_COMPANY_THRESHOLDS` module variable (line 68) -- replaced by registry

### Phase 5: Update scoring_weights.yml (~5 lines)

**File:** [`config/scoring_weights.yml`](config/scoring_weights.yml)

Restructure the `survival_thresholds` section to include all sector overrides in a clean hierarchy:

```yaml
survival_thresholds:
  current_ratio: 1.0
  debt_to_equity: 3.0
  fcf_yield: 0.0
  drawdown_252d: -0.40
  conflict_intensity: 0.70
  inst_flow_momentum: -0.15
  sector_overrides:
    technology:
      current_ratio: 0.7
      debt_to_equity: 5.0
    financial_services:
      current_ratio: 0.6
      debt_to_equity: 10.0
    communication_services:
      current_ratio: 0.8
      debt_to_equity: 5.0
    consumer_cyclical:
      current_ratio: 0.8
```

---

## Files Changed Summary

| File | Change | Lines +/- |
|------|--------|-----------|
| `operator1/analysis/threshold_registry.py` | **NEW** | +180 |
| `main.py` | Init registry after Step 5j | +15 |
| `backtest_runner.py` | Init registry at equivalent point | +10 |
| `operator1/analysis/survival_mode.py` | Use registry, delete _load/_COMPANY | +8, -25 |
| `operator1/models/monte_carlo.py` | Use registry, delete DEFAULT/SECTOR/guard | +6, -65 |
| `operator1/analysis/scenario_engine.py` | Use registry | +3, -5 |
| `operator1/stages/stage5_forward.py` | Use registry | +3, -8 |
| `operator1/models/forecasting.py` | Remove fallback filter in get_regime_distributions | -10 |
| `config/scoring_weights.yml` | Add missing sector overrides | +8 |
| **Total** | | **+233, -113 = ~120 net** |

---

## Validation Criteria

After implementation, ALL must pass:

1. `grep -rn "DEFAULT_SURVIVAL_THRESHOLDS" operator1/` returns 0 results
2. `grep -rn "SECTOR_SURVIVAL_OVERRIDES" operator1/` returns 0 results
3. `grep -rn "_COMPANY_THRESHOLDS" operator1/` returns only test files
4. `grep -rn "_MAX_RETURN_MEAN\|_MAX_RETURN_STD" operator1/` returns 0 results
5. `pytest tests/test_integration_threshold_consistency.py -v` -- all 4 tests pass
6. `pytest tests/test_integration_l2_analysis.py -v -k Golden` -- Apple survival prob > 0.60
7. `pytest tests/test_integration_l3_temporal.py -v -k Synthetic` -- MC drawdown not NaN
8. Running AAPL backtest: MC and survival_mode produce consistent results (same thresholds)

---

## Execution Order

1. Create `threshold_registry.py` (Phase 1)
2. Wire into `main.py` + `backtest_runner.py` (Phase 2)
3. Rewire 4 consumers (Phase 3)
4. Delete redundant code (Phase 4)
5. Update scoring_weights.yml (Phase 5)
6. Run integration tests to validate

# Batch A: Threshold Unification + Burn-Out Fix

*Problems 1 (threshold fragmentation), 2 (forward pass scale mixing), 6 (burn-out contamination)*

## Current State

Three interconnected bugs share a root cause: **modules that should communicate through structured contracts instead pass raw values through untyped channels.**

### What's broken

1. Survival thresholds exist in 6 places that can disagree
2. The burn-out weight learner mixes $30B cash with 0.001 returns in one list
3. Contaminated distributions flow from burn-out to MC, producing NaN/Inf

### Expert methods from different domains

**From distributed systems engineering (Martin Kleppmann, "Designing Data-Intensive Applications"):**
- **Single Leader pattern**: One authoritative source writes thresholds; all consumers read from it. No local copies. This is how ZooKeeper/etcd/Consul handle configuration in distributed systems.
- Applied here: A `ThresholdRegistry` singleton loaded once, consumed by every module that checks survival triggers.

**From control theory (Karl Astrom, "Feedback Systems"):**
- **State estimation with observer design**: The burn-out learner is a state estimator, but it's observing the wrong state space. A properly designed observer separates the observation equation (what you measure) from the state equation (what you're tracking). Returns and cash are different state variables with different observation equations.
- Applied here: Per-variable Kalman-style tracking in the burn-out learner, with the return distribution as the explicit output state.

**From Bayesian statistics (Andrew Gelman, "Bayesian Data Analysis"):**
- **Hierarchical priors**: Instead of fixed thresholds OR peer-calibrated thresholds OR sector overrides, use a hierarchical model: global prior (textbook thresholds) -> sector prior (peer P10/P90) -> company posterior (own history). Each level informs but doesn't override.
- Applied here: The ThresholdRegistry computes thresholds as a Bayesian update: `posterior = prior * likelihood(peer_data) * likelihood(own_history)`.

**From software architecture (Robert C. Martin, "Clean Architecture"):**
- **Dependency Inversion Principle**: High-level modules (MC, scenario engine) should not depend on low-level modules (hardcoded threshold dicts). Both should depend on abstractions (ThresholdRegistry interface).
- Applied here: Define a `ThresholdProvider` protocol that all consumers import. Swap implementations (static, adaptive, hierarchical) without changing consumers.

---

## Implementation Plan

### Phase 1: ThresholdRegistry (solves Problem 1)

**File:** `operator1/analysis/threshold_registry.py` (new, ~200 lines)

```
ThresholdRegistry:
    - Loads base thresholds from scoring_weights.yml (one read)
    - Accepts sector string for sector-aware adjustments
    - Accepts adaptive_thresholds (ThresholdSet) for peer calibration
    - Merges with Bayesian update: base -> sector -> adaptive -> company
    - Exposes get_survival_thresholds() returning unified dict
    - Exposes get_mc_thresholds() returning MC-format dict
    - Exposes get_scenario_thresholds() returning scenario-format dict
    - All formats are views of the same underlying thresholds
```

**Consumers to rewire:**
- `survival_mode.py` line 52: Replace `get_weight()` calls with `registry.get_survival_thresholds()`
- `monte_carlo.py` line 151: Replace `DEFAULT_SURVIVAL_THRESHOLDS` with `registry.get_mc_thresholds()`
- `stage5_forward.py` line 172: Remove inline threshold construction, use registry
- `scenario_engine.py`: Add registry consumption (currently uses hardcoded values)
- `survival_regime_controller.py`: Add registry consumption
- Delete my `SECTOR_SURVIVAL_OVERRIDES` from monte_carlo.py (subsumed by registry)

**Sector configuration** (in scoring_weights.yml):
```yaml
sector_threshold_overrides:
  technology:
    current_ratio: 0.7
    debt_to_equity: 5.0
  financial_services:
    current_ratio: 0.6
    debt_to_equity: 10.0
  communication_services:
    current_ratio: 0.8
    debt_to_equity: 5.0
```

**Wiring:**
- Created once in `main.py` after Step 5j (adaptive thresholds computed)
- Stored in `PipelineState.threshold_registry`
- Passed to all consumers via `state.threshold_registry`

### Phase 2: Scoped burn-out learner (solves Problems 2 and 6)

**File:** `operator1/models/forecasting.py` lines 4482-4660

**Current design flaw:**
```python
# Line 4608 -- called for EVERY variable (return, close, cash, revenue...)
self._regime_weighted_returns[regime].append(actual)
```

**Fix approach (control theory observer design):**

Replace single `_regime_weighted_returns` accumulator with per-variable tracking:

```python
class ExponentialGradientWeightLearner:
    def __init__(self, model_names, ...):
        # Per-variable per-regime tracking (instead of mixed accumulator)
        self._regime_per_var_returns: dict[str, dict[str, list[float]]] = {}
        # Explicit return variable name for distribution export
        self._return_var: str = "return_1d"
    
    def update(self, regime, per_model_preds, actual, variable_name):
        # ... weight update logic unchanged ...
        
        # Track per-variable (not mixed)
        if regime not in self._regime_per_var_returns:
            self._regime_per_var_returns[regime] = {}
        if variable_name not in self._regime_per_var_returns[regime]:
            self._regime_per_var_returns[regime][variable_name] = []
        self._regime_per_var_returns[regime][variable_name].append(actual)
    
    def get_regime_distributions(self) -> dict[str, dict[str, float]]:
        # Export ONLY the return variable distribution (not mixed)
        distributions = {}
        for regime, var_dict in self._regime_per_var_returns.items():
            returns = var_dict.get(self._return_var, [])
            if len(returns) >= 5:
                arr = np.array(returns)
                distributions[regime] = {
                    "mean": float(np.mean(arr)),
                    "std": max(float(np.std(arr, ddof=1)), 1e-6),
                    "n_obs": len(returns),
                }
            else:
                distributions[regime] = {"mean": 0.0, "std": 0.01, "n_obs": 0}
        return distributions
```

**Call site change** (burn-out loop, ~line 4905):
```python
# Before: learner.update(regime, per_model, actual)
# After:  learner.update(regime, per_model, actual, variable_name=var_name)
```

**Remove my band-aid filter** from `get_regime_distributions()` (the `|value| < 1.0` filter) -- no longer needed when variables are tracked separately.

**Remove MC scale guard** from `run_monte_carlo()` (the `_MAX_RETURN_MEAN` / `_MAX_RETURN_STD` rejection) -- no longer needed when burn-out only exports return distributions.

### Phase 3: Wire ThresholdRegistry into PipelineState

**File:** `operator1/pipeline_state.py`
- Add `threshold_registry: Any = None` field (83 -> 84 fields)

**File:** `operator1/stages/stage5_forward.py`
- Replace inline threshold construction with `state.threshold_registry.get_mc_thresholds()`

**File:** `main.py` Step 5j area (~line 2300)
- After adaptive thresholds are computed, create ThresholdRegistry:
```python
from operator1.analysis.threshold_registry import ThresholdRegistry
threshold_registry = ThresholdRegistry(
    base_config=scoring_weights,
    sector=target_profile.get("sector", ""),
    adaptive=_adaptive_thresholds,
)
```

---

## Validation criteria

After implementation, these must all be true:

1. `grep -rn "DEFAULT_SURVIVAL_THRESHOLDS" operator1/` returns only the definition in threshold_registry.py (not monte_carlo.py)
2. `grep -rn "SECTOR_SURVIVAL_OVERRIDES" operator1/` returns 0 results
3. `grep -rn "current_ratio.*lt.*1.0" operator1/` returns only threshold_registry.py and scoring_weights.yml
4. Running AAPL backtest: MC survival probability matches survival_mode.py survival probability (both use same thresholds)
5. Running AAPL backtest: MC drawdown is never NaN at any horizon
6. `learner.get_regime_distributions()` returns distributions with |mean| < 0.1 and std < 0.1 (return-scale)

---

## Files changed (estimated)

| File | Change | Lines |
|------|--------|-------|
| `operator1/analysis/threshold_registry.py` | NEW | ~200 |
| `operator1/models/forecasting.py` | Modify learner | ~40 |
| `operator1/models/monte_carlo.py` | Remove hardcoded thresholds + scale guard | -30 |
| `operator1/stages/stage5_forward.py` | Use registry | ~10 |
| `operator1/analysis/survival_mode.py` | Use registry | ~15 |
| `operator1/analysis/scenario_engine.py` | Use registry | ~10 |
| `operator1/pipeline_state.py` | Add field | ~2 |
| `main.py` | Create registry after Step 5j | ~15 |
| `config/scoring_weights.yml` | Add sector_threshold_overrides | ~15 |
| **Total** | | ~340 net |

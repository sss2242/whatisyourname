# Forecasting Speedup v2: Full Implementation Plan with Exact Code References

*From debug scan of all inputs, outputs, closures, and inter-model connections in the per-variable forecasting loop.*

---

## Debug Scan Summary

### Primary edit target
- [`operator1/models/forecasting.py`](operator1/models/forecasting.py) -- the per-variable loop at lines 2600-2970

### Config edit
- [`config/global_config.yml`](config/global_config.yml) -- update `forecasting:` section with `mode` and `parallel_workers`

### Files that are NOT edited (verified read-only consumers)
All downstream consumers verified compatible in PR #1 scan -- no changes needed.

---

## Scan Area 1: Closure Variables Used Inside the Per-Variable Loop

The per-variable loop body (lines 2600-2970) uses these variables from the enclosing `run_forecasting()` scope. Any function extracted for parallel execution must receive ALL of these as parameters or via a shared context object.

### Read-only (safe to share across threads)

| Variable | Type | Defined at | Used for |
|----------|------|-----------|----------|
| `cache` | DataFrame | Parameter | `_extract_series()`, `_extract_multivariate()`, `_extract_features()`, regime columns |
| `tier_map` | Dict | Line 2340 | `_get_tier_for_variable()` |
| `available_vars` | List[str] | Line 2365 | VAR multivariate extraction |
| `has_returns` | Bool | Line 2375 | Kalman regime check |
| `HORIZONS` | Dict | Module-level | Horizon labels + values |
| `random_state` | Int | Parameter | LSTM/tree seed |
| `enable_burnout` | Bool | Parameter | Burnout refinement gate |
| `extra_variables` | List[str] | Parameter | Feature extraction fallback |
| `model_feature_sets` | Dict | Parameter | Feature routing |
| `_forecasting_cfg` | Dict | Line 2580 | Mode, LSTM enabled, fast tiers |
| `_fast_tiers` | Set[int] | Line 2581 | Fast path check |
| `_distributional_vars` | Set[str] | Line 2583 | Distributional guard |
| `_lstm_enabled` | Bool | Line 2589 | LSTM skip |
| `_fast_path_excluded` | Frozenset | Line 2593 | close/return_1d exclusion |
| `_n_workers` | Int | NEW | Parallel worker count |

### Closure functions (must be passable to threads)

| Function | Defined at | Dependencies |
|----------|-----------|--------------|
| `_extract_series(var)` | Line 2385 | `cache` (read-only) |
| `_extract_multivariate(target)` | Line 2394 | `cache`, `available_vars`, `_get_model_features`, `extra_variables` |
| `_extract_features(target, model_type)` | Line 2440 | `cache`, `_get_model_features`, `extra_variables` |
| `_get_model_features(model_type)` | Line 2317 | `model_feature_sets`, `extra_variables` |

All closures are read-only on `cache` -- safe for thread-based parallel execution.

### Mutated state (MUST NOT be accessed from threads)

| Variable | Type | Mutated how | Thread-safe? |
|----------|------|-------------|-------------|
| `result` | ForecastResult | `.forecasts[var]`, `.metrics.append()`, `.model_used[var]`, `.model_failed_*` | NO -- list.append() is thread-safe in CPython but dict assignment is not |
| `kalman_attempted` | Bool | Set to True on first Kalman attempt | NO |
| `var_attempted` | Bool | Set to True | NO |
| `lstm_attempted` | Bool | Set to True | NO |
| `tree_attempted` | Bool | Set to True | NO |

**Solution:** Each worker returns a result dict. The main thread merges them sequentially.

---

## Scan Area 2: Post-Processing That Reads Cache (Sequential Only)

Lines 2800-2965 contain post-cascade logic that reads cache columns and modifies `_horizon_forecasts`. This MUST run sequentially after all model fits complete, because:

1. **Regime directional shift** (line 2803-2842): Reads `regime_hmm_prob_*`, `regime_hmm`, `return_1d`, `anchoring_52w_high`, `hurst_exponent_rolling` from cache
2. **Momentum overlay** (line 2848-2888): Reads `return_1d`, `close`, `anchoring_52w_high` from cache
3. **Residual feature adjustment** (line 2894-2918): Reads cache feature columns
4. **C1 horizon blending** (line 2920-2943): Calls `_extract_features()` + `fit_tree_ensemble()`
5. **ETS medium-horizon blend** (line 2944-2964): Calls `fit_ets()`

**Architecture decision:** Split the loop into two phases:
- **Phase A (parallelizable):** Model fitting -- Kalman/VAR/LSTM/Tree/Baseline cascade. Returns `(best_forecast, best_model_name, best_metrics, per_var_metrics)`.
- **Phase B (sequential):** Post-processing -- regime shift, momentum, C1 blending, ETS blend, distributional. Reads cache, modifies `_horizon_forecasts`, writes to `result`.

---

## Scan Area 3: GARCH Special Case (Before the Loop)

Lines 2498-2567: GARCH forecasting runs BEFORE the per-variable loop on `return_1d`. It writes to `result.forecasts["volatility_*"]`. This is independent and should NOT be parallelized (it's already fast at ~0.3s total).

---

## Scan Area 4: Parallel Tree Block (After the Loop)

Lines 2971-3014: Parallel tree ensemble runs AFTER the per-variable loop on `["close", "return_1d", "volatility_21d"]`. Uses `_extract_features()` closure and `result.model_used`. This should remain sequential and run after the parallel phase.

---

## Scan Area 5: Residual Collection (After the Loop)

Lines 3055-3073: Iterates over `result.metrics` to collect residuals for conformal calibration. Must run after ALL model fits (both parallel and post-processing) are complete.

---

## Implementation: 4 Changes

### Change 1: Extract `_forecast_single_variable()` function

**Location:** Before the per-variable loop (line ~2595)

This function encapsulates Phase A (model fitting) for one variable. It receives all needed parameters explicitly (no closures on mutable state) and returns a result dict.

```python
def _forecast_single_variable(
    var_name: str,
    series: np.ndarray,
    tier: str,
    max_horizon: int,
    tier_num: int,
    fast_tiers: set[int],
    fast_path_excluded: frozenset,
    lstm_enabled: bool,
    distributional_vars: set[str],
    random_state: int,
    # Feature extraction needs these:
    cache_columns: list[str],
    multivariate_df: pd.DataFrame | None,  # pre-extracted for VAR
    feature_df: pd.DataFrame | None,       # pre-extracted for Tree
    has_returns: bool,
    regime_data: dict | None,  # pre-extracted regime columns
) -> dict:
    """Fit models for ONE variable. Returns result dict.

    Thread-safe: no shared mutable state. All inputs are read-only
    copies or pre-extracted DataFrames.
    """
    metrics_list = []
    best_forecast = None
    best_model_name = ""
    best_metrics = None
    model_flags = {}  # e.g. {"kalman_failed": True, "kalman_error": "..."}

    # Express mode: fast ensemble
    if tier_num in fast_tiers and var_name not in fast_path_excluded:
        fcast, met = _fit_fast_ensemble(series, n_forecast=max_horizon)
        met.variable = var_name
        metrics_list.append(met)
        if fcast is not None:
            best_forecast = fcast
            best_model_name = met.model_name
            best_metrics = met
        # ... distributional guard ...
        if best_forecast is None:
            fcast, met = fit_baseline(series, n_forecast=max_horizon)
            met.variable = var_name
            metrics_list.append(met)
            best_forecast = fcast
            best_model_name = met.model_name
            best_metrics = met
        return {
            "var_name": var_name,
            "best_forecast": best_forecast,
            "best_model_name": best_model_name,
            "best_metrics": best_metrics,
            "metrics": metrics_list,
            "model_flags": model_flags,
            "horizon_forecasts": _build_horizon_dict(best_forecast, HORIZONS),
        }

    # Balanced/Full mode: cascade
    # Kalman (tier1/2 only)
    if tier in ("tier1", "tier2"):
        if regime_data is not None:
            fcast, met = fit_kalman_per_regime(
                series, regime_data["labels"],
                regime_probs=regime_data.get("probs"),
                n_forecast=max_horizon,
            )
        else:
            fcast, met = fit_kalman(series, n_forecast=max_horizon)
        met.variable = var_name
        metrics_list.append(met)
        if fcast is not None:
            best_forecast = fcast
            best_model_name = "kalman"
            best_metrics = met
        else:
            model_flags["kalman_failed"] = True
            model_flags["kalman_error"] = met.error

    # VAR
    if best_forecast is None and multivariate_df is not None:
        fcast, met = fit_var(multivariate_df, var_name, n_forecast=max_horizon)
        met.variable = var_name
        metrics_list.append(met)
        if fcast is not None:
            best_forecast = fcast
            best_model_name = met.model_name
            best_metrics = met
        else:
            model_flags["var_failed"] = True

    # LSTM (only in full mode)
    if best_forecast is None and lstm_enabled:
        fcast, met = fit_lstm(series, n_forecast=max_horizon, random_state=random_state)
        met.variable = var_name
        metrics_list.append(met)
        if fcast is not None:
            best_forecast = fcast
            best_model_name = met.model_name
            best_metrics = met
        else:
            model_flags["lstm_failed"] = True

    # Tree ensemble
    if best_forecast is None and feature_df is not None and not feature_df.empty:
        fcast, met = fit_tree_ensemble(
            feature_df, var_name, n_forecast=max_horizon, random_state=random_state,
        )
        met.variable = var_name
        metrics_list.append(met)
        if fcast is not None:
            best_forecast = fcast
            best_model_name = met.model_name
            best_metrics = met
        else:
            model_flags["tree_failed"] = True

    # Baseline (always succeeds)
    if best_forecast is None:
        fcast, met = fit_baseline(series, n_forecast=max_horizon)
        met.variable = var_name
        metrics_list.append(met)
        best_forecast = fcast
        best_model_name = met.model_name
        best_metrics = met

    # Burnout refinement (only for Kalman)
    if best_forecast is not None and best_model_name == "kalman":
        burnout_fcast, burnout_met = _burnout_refit(series, fit_kalman, n_forecast=max_horizon)
        if (burnout_fcast is not None and not np.isnan(burnout_met.rmse)
                and (best_metrics is None or np.isnan(best_metrics.rmse) or burnout_met.rmse < best_metrics.rmse)):
            best_forecast = burnout_fcast
            best_model_name = f"{best_model_name}_burnout"
            best_metrics = burnout_met

    return {
        "var_name": var_name,
        "best_forecast": best_forecast,
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "metrics": metrics_list,
        "model_flags": model_flags,
        "horizon_forecasts": _build_horizon_dict(best_forecast, HORIZONS),
    }
```

**Helper function:**
```python
def _build_horizon_dict(forecast, horizons):
    if forecast is None:
        return {}
    return {
        label: float(forecast[min(h - 1, len(forecast) - 1)])
        for label, h in horizons.items()
    }
```

---

### Change 2: Pre-extract data for each variable (before parallel phase)

**Location:** Before the parallel dispatch (line ~2600)

Extract multivariate DataFrames and feature DataFrames BEFORE spawning threads, because `_extract_multivariate()` and `_extract_features()` read from `cache` which is safe, but they also call `classify_column_frequency()` and `add_filing_timing_features()` which modify the cache by adding columns. Pre-extracting avoids concurrent modification.

```python
# Pre-extract data for each variable (sequential, safe)
_var_data = {}
for var_name in available_vars:
    series = _extract_series(var_name)
    tier = _get_tier_for_variable(var_name, tier_map)
    _tier_num = int(tier.replace("tier", "")) if tier.startswith("tier") else 0

    # Pre-extract multivariate for VAR
    multivariate_df = None
    if _tier_num not in _fast_tiers or var_name in _fast_path_excluded:
        if len(available_vars) >= 2:
            multivariate_df = _extract_multivariate(var_name)

    # Pre-extract features for Tree
    feature_df = None
    if _tier_num not in _fast_tiers or var_name in _fast_path_excluded:
        feature_df = _extract_features(var_name, model_type="tree")

    # Pre-extract regime data for Kalman
    regime_data = None
    if tier in ("tier1", "tier2"):
        _regime_labels_arr = cache.get("regime_label")
        if _regime_labels_arr is not None and has_returns:
            _rl = _regime_labels_arr.values if hasattr(_regime_labels_arr, "values") else _regime_labels_arr
            _rp = None
            _prob_cols = [c for c in cache.columns if c.startswith("regime_hmm_prob_")]
            if _prob_cols and len(cache) > 0:
                _last_probs = cache[_prob_cols].iloc[-1]
                _regime_map = {0: "bull", 1: "bear", 2: "high_vol", 3: "low_vol"}
                _rp = {_regime_map.get(i, str(i)): float(_last_probs.iloc[i])
                       for i in range(len(_last_probs)) if not np.isnan(_last_probs.iloc[i])}
            regime_data = {"labels": _rl, "probs": _rp}

    _var_data[var_name] = {
        "series": series,
        "tier": tier,
        "tier_num": _tier_num,
        "multivariate_df": multivariate_df,
        "feature_df": feature_df,
        "regime_data": regime_data,
    }
```

---

### Change 3: Parallel dispatch with joblib

**Location:** Replace the per-variable for-loop (lines 2600-2970)

```python
from joblib import Parallel, delayed

_n_workers = _forecasting_cfg.get("parallel_workers", 4)
_mode = _forecasting_cfg.get("mode", "balanced")

# Determine effective LSTM setting from mode
if _mode == "express":
    _lstm_enabled = False
elif _mode == "balanced":
    _lstm_enabled = False
elif _mode == "full":
    _lstm_enabled = _forecasting_cfg.get("lstm_enabled", True)

max_horizon = max(HORIZONS.values())

# Phase A: Parallel model fitting (read-only cache access)
if _n_workers > 1 and len(available_vars) > 1:
    raw_results = Parallel(n_jobs=_n_workers, backend="threading", prefer="threads")(
        delayed(_forecast_single_variable)(
            var_name=var_name,
            series=_var_data[var_name]["series"].copy(),  # numpy copy for safety
            tier=_var_data[var_name]["tier"],
            max_horizon=max_horizon,
            tier_num=_var_data[var_name]["tier_num"],
            fast_tiers=_fast_tiers,
            fast_path_excluded=_fast_path_excluded,
            lstm_enabled=_lstm_enabled,
            distributional_vars=_distributional_vars,
            random_state=random_state,
            cache_columns=list(cache.columns),
            multivariate_df=_var_data[var_name]["multivariate_df"],
            feature_df=_var_data[var_name]["feature_df"],
            has_returns=has_returns,
            regime_data=_var_data[var_name]["regime_data"],
        )
        for var_name in available_vars
    )
    raw_results_dict = {r["var_name"]: r for r in raw_results}
else:
    # Sequential fallback (n_workers=1 or single variable)
    raw_results_dict = {}
    for var_name in available_vars:
        vd = _var_data[var_name]
        raw_results_dict[var_name] = _forecast_single_variable(
            var_name=var_name, series=vd["series"],
            tier=vd["tier"], max_horizon=max_horizon,
            tier_num=vd["tier_num"], fast_tiers=_fast_tiers,
            fast_path_excluded=_fast_path_excluded,
            lstm_enabled=_lstm_enabled,
            distributional_vars=_distributional_vars,
            random_state=random_state,
            cache_columns=list(cache.columns),
            multivariate_df=vd["multivariate_df"],
            feature_df=vd["feature_df"],
            has_returns=has_returns,
            regime_data=vd["regime_data"],
        )

# Merge model flags
kalman_attempted = any(r.get("model_flags", {}).get("kalman_failed") for r in raw_results_dict.values())
# ... etc for var_attempted, lstm_attempted, tree_attempted ...

# Phase B: Sequential post-processing (reads cache, writes to result)
for var_name in available_vars:
    raw = raw_results_dict[var_name]
    result.metrics.extend(raw["metrics"])

    best_forecast = raw["best_forecast"]
    best_model_name = raw["best_model_name"]
    best_metrics = raw["best_metrics"]
    _horizon_forecasts = raw["horizon_forecasts"]

    if best_forecast is None:
        continue

    # Express mode vars already have horizon_forecasts set and skip post-processing
    if raw.get("express_path"):
        result.forecasts[var_name] = _horizon_forecasts
        result.model_used[var_name] = best_model_name
        continue

    # --- POST-PROCESSING (sequential, reads cache) ---

    # Regime directional shift for close (Method 5)
    # ... existing code from lines 2803-2842 ...

    # Momentum overlay for close (P1 fix)
    # ... existing code from lines 2844-2888 ...

    # Residual feature adjustment
    # ... existing code from lines 2894-2918 ...

    # C1 horizon-specific tree blending
    # ... existing code from lines 2920-2943 ...

    # ETS medium-horizon blend
    # ... existing code from lines 2944-2964 ...

    # Distributional model (guarded to key vars)
    # ... existing code from lines 2739-2758 ...

    result.forecasts[var_name] = _horizon_forecasts
    result.model_used[var_name] = best_model_name
```

---

### Change 4: Update config schema

**Location:** [`config/global_config.yml`](config/global_config.yml)

```yaml
forecasting:
  mode: "balanced"              # "express" | "balanced" | "full"
  parallel_workers: 4           # set to 1 for sequential (deterministic)
  fast_ensemble_tiers: [3, 4, 5]
  lstm_enabled: false
  distributional_vars:
    - close
    - return_1d
    - volatility_21d
    - revenue
    - fcf_yield
```

---

## Data Flow Verification

### Inputs to `_forecast_single_variable()` (all read-only)

| Parameter | Source | Type | Thread-safe? |
|-----------|--------|------|-------------|
| `series` | `cache[var].values.copy()` | numpy array (COPY) | Yes |
| `tier` | `tier_map` (read-only dict) | str | Yes |
| `multivariate_df` | Pre-extracted `.copy()` | DataFrame (COPY) | Yes |
| `feature_df` | Pre-extracted `.copy()` | DataFrame (COPY) | Yes |
| `regime_data` | Pre-extracted dict with numpy values | dict | Yes |
| `random_state` | int literal | int | Yes |

### Outputs from `_forecast_single_variable()` (per-variable result dict)

| Field | Type | Consumed by |
|-------|------|-------------|
| `best_forecast` | numpy array or None | Phase B post-processing |
| `best_model_name` | str | `result.model_used` |
| `best_metrics` | ModelMetrics | Phase B residual adjustment |
| `metrics` | List[ModelMetrics] | `result.metrics` (merged sequentially) |
| `model_flags` | Dict | `result.model_failed_*` flags |
| `horizon_forecasts` | Dict[str, float] | Phase B post-processing -> `result.forecasts` |

### Post-processing inputs (Phase B, sequential, reads cache)

| Input | Source | Used by |
|-------|--------|---------|
| `cache["regime_hmm_prob_*"]` | cache columns | Regime directional shift |
| `cache["return_1d"]` | cache column | Momentum overlay |
| `cache["anchoring_52w_high"]` | cache column | Regime/momentum dampening |
| `cache["hurst_exponent_rolling"]` | cache column | Regime dampening |
| `cache["close"]` | cache column | Momentum last close |
| `_extract_features(var_name)` | closure | C1 tree blending, distributional |

---

## Execution Order

```
[ ] 1. Add _build_horizon_dict() helper function
       Location: forecasting.py, before run_forecasting()

[ ] 2. Add _forecast_single_variable() function
       Location: forecasting.py, before run_forecasting()

[ ] 3. Add pre-extraction loop (_var_data construction)
       Location: forecasting.py, inside run_forecasting(), before per-variable loop

[ ] 4. Replace per-variable for-loop with:
       Phase A: joblib.Parallel dispatch
       Phase B: sequential post-processing loop

[ ] 5. Wire LSTM skip into _forecast_single_variable() (already in the function)

[ ] 6. Update config/global_config.yml with mode + parallel_workers

[ ] 7. Syntax check + import verification

[ ] 8. Debug scan for thread safety issues

[ ] 9. Commit, push, update PR
```

---

## Expected Timing

| Mode | Phase A (parallel) | Phase B (sequential) | Total |
|------|-------------------|---------------------|-------|
| express | 8s (ETS ensemble) | 2s (close/return_1d post-proc) | **~10s** |
| balanced | 15s (Kalman+Tree, 4 workers) | 5s (post-proc) | **~20s** |
| full | 75s (Kalman+Tree+LSTM, 4 workers) | 5s (post-proc) | **~80s** |
| current (no PR) | N/A (sequential) | N/A | **~700s** |

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Thread contention in sklearn/xgboost | Medium | Pre-set `OMP_NUM_THREADS=2` before parallel phase, restore after |
| Non-deterministic results from thread scheduling | Low | Set `parallel_workers: 1` for bit-identical results |
| `_extract_features` modifying cache (adding columns) | High | Pre-extract ALL features before parallel phase |
| LSTM CUDA issues in threads | Low | LSTM only enabled in "full" mode, runs sequentially per-device |
| Exception in one thread kills all | Low | joblib wraps exceptions per-job with traceback |

# Beyond Bands - Fix Guide for 5 Debug Scan Issues

*From debug scan 2026-05-07*

---

## Fix 1: Wire `fit_distributional()` into `run_forecasting()` cascade (CRITICAL)

**Severity:** Critical -- Method 2 never executes without this fix
**File:** `operator1/models/forecasting.py`
**Location:** Inside `run_forecasting()`, after the tree ensemble call in the per-variable model cascade

### Problem

`fit_distributional()` is defined at line 1828 but never called from `run_forecasting()`. The model cascade runs: Kalman -> GARCH -> VAR -> LSTM -> Tree -> Baseline. The distributional model is not in this sequence, so `ModelMetrics.conditional_sigma` is always `None` and Method 2 never contributes feature-dependent band width to the prediction aggregator.

### Fix

Find the section in `run_forecasting()` where the tree model is called for each variable. After the tree result is stored, add a `fit_distributional()` call. The distributional model should run AFTER tree (it uses the same features) and its `ModelMetrics` should be appended to `result.metrics`.

**Where to insert:** After the tree ensemble model result is appended to `result.metrics` for each variable. Look for the pattern where `fit_tree_ensemble()` is called and its metrics stored. The distributional call goes right after.

**Code to add:**

```python
# After tree ensemble result is stored for this variable:
# Beyond Bands Method 2: Distributional forecast (conditional sigma)
try:
    _dist_forecasts, _dist_metrics = fit_distributional(
        features=_build_tree_features(cache, var, extra_variables or []),
        target_col=var,
        random_state=random_state,
    )
    _dist_metrics.variable = var
    if _dist_metrics.fitted:
        result.metrics.append(_dist_metrics)
        logger.info(
            "Distributional model: %s sigma=%.6f",
            var, _dist_metrics.conditional_sigma or 0,
        )
except Exception as _de:
    logger.debug("Distributional model skipped for %s: %s", var, _de)
```

**Note:** The exact feature builder function name may differ. Check how `fit_tree_ensemble()` builds its `features` DataFrame and use the same logic. The key is that `fit_distributional()` receives the same feature matrix as the tree model.

### Verification

After fix, run the pipeline and check logs for "Distributional model: close sigma=X.XXXX". Also verify `conditional_sigma` is non-None in prediction aggregator by adding a temporary log line.

---

## Fix 2: Pass barriers through `run_simulation()` to survival paths (MEDIUM)

**Severity:** Medium -- barrier reflection only works for terminal values, not survival paths
**File:** `operator1/models/monte_carlo.py`
**Location:** `run_simulation()` function signature at line 793 and its two `simulate_return_paths()` calls

### Problem

`run_simulation()` at line 793 does NOT have `ath_barrier`/`support_barrier` parameters. The two `simulate_return_paths()` calls inside it (nominal paths at line 864, importance-sampled paths at line 895) use the default `ath_barrier=0.0, support_barrier=0.0`, disabling barrier reflection. Only the terminal_values paths at line 1586 pass barriers.

This means survival probability is computed WITHOUT barrier awareness, while terminal value distributions HAVE barrier awareness -- inconsistent behavior.

### Fix

**Step 1:** Add `ath_barrier` and `support_barrier` to `run_simulation()` signature:

```python
def run_simulation(
    n_paths: int,
    horizon_steps: int,
    current_regime_idx: int,
    transition_matrix: np.ndarray,
    regime_distributions: list[RegimeDistribution],
    initial_values: dict[str, float],
    rng: np.random.Generator,
    *,
    importance_fraction: float = 0.3,
    importance_tilt: float = DEFAULT_IS_TILT,
    survival_thresholds: dict[str, tuple[str, float]] | None = None,
    variable_sensitivities: dict[str, float] | None = None,
    jump_lambda: float = 0.0,
    jump_mean: float = 0.0,
    jump_std: float = 0.0,
    ath_barrier: float = 0.0,       # NEW
    support_barrier: float = 0.0,   # NEW
) -> tuple[np.ndarray, np.ndarray, float]:
```

**Step 2:** Pass barriers to both `simulate_return_paths()` calls inside `run_simulation()`:

At the nominal paths call (~line 864), add:
```python
ath_barrier=ath_barrier,
support_barrier=support_barrier,
```

At the importance-sampled paths call (~line 895), add:
```python
ath_barrier=ath_barrier,
support_barrier=support_barrier,
```

**Step 3:** Pass barriers from `run_monte_carlo()` to `run_simulation()`:

At the `run_simulation()` call (~line 1548), add:
```python
ath_barrier=_ath_barrier,
support_barrier=_support_barrier,
```

### Verification

After fix, survival probability and terminal values will use consistent barrier-reflected paths. Verify by checking that survival probability changes slightly when `_ath_barrier > 0` vs when it's forced to 0.

---

## Fix 3: Serialize new `HorizonPrediction` fields in profile builder (LOW)

**Severity:** Low -- data computed correctly but invisible in profile/report/dashboard
**File:** `operator1/report/profile_builder.py`
**Location:** Prediction section builder at line ~492-516

### Problem

The profile builder's prediction section only extracts `point_forecast`, `lower_ci`, `upper_ci`, `confidence` from `HorizonPrediction`. The 4 new Beyond Bands fields (`skew_signal`, `between_model_std`, `scenario_weighted_point`, `scenarios`) are NOT serialized. They're computed at runtime but vanish when the profile is saved to JSON.

### Fix

In the `else` branch (dataclass path, line 500-516), add 4 additional `getattr()` calls after `confidence`:

```python
else:
    entry = {
        "variable": var_name,
        "point_forecast": _safe_float(getattr(pred, "point_forecast", None)),
        "lower_ci": _safe_float(getattr(pred, "lower_ci", None)),
        "upper_ci": _safe_float(getattr(pred, "upper_ci", None)),
        "confidence": _safe_float(getattr(pred, "confidence", None)),
        # Beyond Bands distributional forecasting fields
        "skew_signal": _safe_float(getattr(pred, "skew_signal", None)),
        "between_model_std": _safe_float(getattr(pred, "between_model_std", None)),
        "scenario_weighted_point": _safe_float(getattr(pred, "scenario_weighted_point", None)),
        "scenarios": getattr(pred, "scenarios", None),
    }
```

Also do the same in the `if isinstance(pred, dict)` branch (line 492-499):
```python
entry = {
    ...existing fields...,
    "skew_signal": _safe_float(pred.get("skew_signal")),
    "between_model_std": _safe_float(pred.get("between_model_std")),
    "scenario_weighted_point": _safe_float(pred.get("scenario_weighted_point")),
    "scenarios": pred.get("scenarios"),
}
```

### Verification

After fix, `cache/company_profile.json` predictions section should contain `skew_signal`, `between_model_std`, `scenario_weighted_point`, and `scenarios` for close/5d horizon.

---

## Fix 4: Add scenario decomposition rendering to report generator (LOW)

**Severity:** Low -- data is in profile but not rendered in any report section
**File:** `operator1/report/report_generator.py`
**Location:** Section 9 builder (`_build_predictions_forecasts()`)

### Problem

The `scenario_decomposition` field on `PredictionAggregatorResult` and the `scenarios` list on individual `HorizonPrediction` objects are computed and serialized (after Fix 3) but have no rendering in the report. The user never sees the bear/base_low/base_high/bull scenario table.

### Fix

In the Section 9 builder function (`_build_predictions_forecasts()`), after the existing prediction tables, add a subsection for scenario decomposition:

```python
# Beyond Bands: Scenario Decomposition subsection
_scenario_md = ""
for h_label, preds in predictions.items():
    if isinstance(preds, dict):
        for var_name, pred_data in preds.items():
            scenarios = None
            if isinstance(pred_data, dict):
                scenarios = pred_data.get("scenarios")
            elif hasattr(pred_data, "scenarios"):
                scenarios = getattr(pred_data, "scenarios", None)
            
            if scenarios and len(scenarios) > 0:
                _scenario_md += f"\n#### Scenario Decomposition: {var_name} ({h_label})\n\n"
                _scenario_md += "| Scenario | Probability | Target Price |\n"
                _scenario_md += "|----------|-------------|-------------|\n"
                for s in scenarios:
                    _scenario_md += f"| {s['label'].title()} | {s['probability']:.0%} | ${s['target_price']:.2f} |\n"
                
                # Add skew signal if available
                skew = None
                if isinstance(pred_data, dict):
                    skew = pred_data.get("skew_signal")
                elif hasattr(pred_data, "skew_signal"):
                    skew = getattr(pred_data, "skew_signal", None)
                if skew is not None:
                    direction = "right-skewed, more upside" if skew > 0 else "left-skewed, more downside"
                    _scenario_md += f"\n**Skew Signal:** {skew:+.2f} ({direction})\n"

if _scenario_md:
    section_text += "\n### Distributional Forecast Analysis\n" + _scenario_md
```

### Verification

After fix, Premium report Section 9 should contain a "Distributional Forecast Analysis" subsection with scenario tables for the close variable.

---

## Fix 5: Add scenario display to dashboard Home page (LOW)

**Severity:** Low -- cosmetic, data available via profile JSON but not visualized
**File:** `dashboard.py`
**Location:** `render_home()` function, after existing prediction/signal cards

### Problem

The dashboard reads `company_profile.json` and displays cards for health score, survival probability, regime, position signal, etc. But there is no card or visualization for the new scenario decomposition data.

### Fix

In `render_home()`, after the "OHLC Predictions" section, add:

```python
# Beyond Bands: Scenario Decomposition
pred_data = profile.get("predictions", {})
_scenarios_shown = False
for h_key, h_preds in pred_data.items():
    if isinstance(h_preds, list):
        for p in h_preds:
            if isinstance(p, dict) and p.get("scenarios"):
                if not _scenarios_shown:
                    ui.separator()
                    ui.label("Scenario Decomposition").classes("text-lg font-bold mt-3")
                    _scenarios_shown = True
                with ui.row().classes("gap-4"):
                    for s in p["scenarios"]:
                        _card(
                            s["label"].title(),
                            f"${s['target_price']:.2f}",
                            f"{s['probability']:.0%}",
                            "casino",
                        )
                break
    if _scenarios_shown:
        break
```

### Verification

After fix, the dashboard Home page should show scenario cards (Bear/Base Low/Base High/Bull with probabilities and target prices) when scenario decomposition data is available in the profile.

---

## Execution Order

| Priority | Fix | File | Impact |
|----------|-----|------|--------|
| 1 | **Fix 1** | forecasting.py | Critical: enables Method 2 |
| 2 | **Fix 2** | monte_carlo.py | Medium: consistent barrier behavior |
| 3 | **Fix 3** | profile_builder.py | Low: makes data visible |
| 4 | **Fix 4** | report_generator.py | Low: renders in reports |
| 5 | **Fix 5** | dashboard.py | Low: renders in dashboard |

Fixes 1 and 2 are functional bugs. Fixes 3-5 are presentation/serialization gaps.

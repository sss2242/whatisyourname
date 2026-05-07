# Beyond Bands: Full Implementation Plan

*Based on code scan of 2026-05-07 + GitHub research findings*

This plan documents every edit location, data flow connection, input/output contract, and integration point for implementing the 6 distributional forecasting methods from the Beyond Bands proposal.

---

## Current Architecture (What Exists)

### Data Flow: Forecast -> Bands -> Profile

```
forecasting.py:run_forecasting()
    |
    | ForecastResult:
    |   .forecasts: {var: {horizon: point_float}}
    |   .metrics: [ModelMetrics(model_name, rmse, test_residuals)]
    |   .model_used: {var: model_name}
    |   .residuals: [float]
    |
    v
conformal.py:build_conformal_result()
    |
    | Uses ConformalPIDCalibrator (preferred) or ConformalCalibrator (fallback)
    | + QuantileRegressionCalibrator (already exists at line 920!)
    |
    | ConformalResult:
    |   .intervals: {var: {horizon: ConformalInterval(lower, upper, forecast, coverage)}}
    |
    v
prediction_aggregator.py:run_prediction_aggregation()
    |
    | Step 1: compute_ensemble_weights(metrics) -> inverse-RMSE weights (line 392)
    | Step 2: Regime blending via dual_regime_result
    | Step 3: Point forecast = weighted average of model forecasts
    | Step 4a: Try conformal intervals (line 2664-2689) -- re-center on ensemble point
    | Step 4b: Fallback to compute_uncertainty_bands() (line 2692) -- SYMMETRIC RMSE * sqrt(h) * z
    | Step 5: Copula widening
    | Step 6: DTW analog overlay
    | Step 7: Granger/PCMCI causal adjustment
    | Step 8: SHAP attachment
    | Step 9: Technical Alpha mask
    | Step 10: Walk-forward MCS weighting
    |
    | PredictionAggregatorResult:
    |   .predictions: {var: {horizon: HorizonPrediction}}
    |   HorizonPrediction: point_forecast, lower_ci, upper_ci, confidence, interval_source
    |
    v
monte_carlo.py:run_monte_carlo()
    |
    | MonteCarloResult:
    |   .terminal_values: {horizon: np.ndarray(n_paths)}  <-- 10K paths
    |   .survival_probability: {horizon: {mean, p5, p95}}
    |   .transition_matrix: np.ndarray
    |   .regime_distributions: [RegimeDistribution]
    |
    v
profile_builder.py -> report_generator.py
```

### Key Weakness Identified

1. **`compute_uncertainty_bands()`** (line 840-893): Uses `SYMMETRIC` bands: `point +/- z * rmse * sqrt(h)`. No asymmetry. No feature dependence.

2. **`compute_ensemble_weights()`** (line 392-434): Pure inverse-RMSE. No between-model variance term (BMA gap).

3. **Conformal re-centering** (line 2674-2681): Takes conformal `half_width` and applies symmetrically around ensemble point. Even when conformal itself is asymmetric, the re-centering SYMMETRIZES it.

4. **`QuantileRegressionCalibrator`** (line 920-1078): ALREADY EXISTS in conformal.py but is NOT WIRED into the prediction aggregator pipeline. It's fitted but never consumed.

5. **MC paths** (monte_carlo.py line 512-611): No barrier reflection at ATH/support levels. Paths can go to infinity or zero without boundary awareness.

6. **No scenario decomposition** in the output format. `HorizonPrediction` has point + lower + upper but no scenario probabilities.

---

## Implementation Plan: 6 Methods

### Method 1: Quantile Regression in Tree Cascade

**Goal:** Replace single-point tree forecast with simultaneous multi-quantile prediction.

**File:** `operator1/models/forecasting.py`

**Edit location:** `fit_tree_ensemble()` at line 1624

**Current contract:**
- Input: `features: DataFrame`, `target_col: str`
- Output: `(forecasts: np.ndarray, metrics: ModelMetrics)` -- point forecasts only

**New contract:**
- Output: `(forecasts: np.ndarray, metrics: ModelMetrics)` -- unchanged for backward compat
- **NEW field on ModelMetrics:** `quantile_forecasts: dict[float, float] | None` -- keyed by quantile (0.05, 0.25, 0.50, 0.75, 0.95)

**Implementation:**

```python
# In fit_tree_ensemble(), after model_obj.fit(X, y) at line 1681:

# Multi-quantile prediction via XGBoost custom objective
quantile_forecasts = None
if metrics.model_name == "xgboost":
    try:
        from xgboost import XGBRegressor
        _quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        quantile_forecasts = {}
        for q in _quantiles:
            _qmodel = XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=q,
                n_estimators=100, max_depth=5, learning_rate=0.05,
                random_state=random_state,
            )
            _qmodel.fit(X, y)
            quantile_forecasts[q] = float(_qmodel.predict(last_features)[0])
    except Exception:
        quantile_forecasts = None
elif metrics.model_name in ("gradient_boosting",):
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        _quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        quantile_forecasts = {}
        for q in _quantiles:
            _qmodel = GradientBoostingRegressor(
                loss="quantile", alpha=q, n_estimators=100,
                max_depth=3, learning_rate=0.05,
            )
            _qmodel.fit(X, y)
            quantile_forecasts[q] = float(_qmodel.predict(last_features)[0])
    except Exception:
        quantile_forecasts = None

metrics.quantile_forecasts = quantile_forecasts
```

**Changes to `ModelMetrics` dataclass** (line 90):
```python
# Add new field:
quantile_forecasts: dict[float, float] | None = None
```

**Downstream consumer:** `prediction_aggregator.py` will use `quantile_forecasts` for asymmetric bands (see Method 1 integration in aggregator below).

**Lines changed:** ~30 in forecasting.py, ~1 in ModelMetrics

---

### Method 2: Distributional Forecasting (Two-Tree NGBoost-Lite)

**Goal:** Train a conditional variance model alongside the mean model, giving feature-dependent uncertainty.

**File:** `operator1/models/forecasting.py`

**Edit location:** NEW function `fit_distributional()` added after `fit_tree_ensemble()` (after line 1718).

**New contract:**
- Input: Same as `fit_tree_ensemble()`
- Output: `(forecasts, metrics)` with new `ModelMetrics` field: `conditional_sigma: float | None`

**Implementation:**

```python
def fit_distributional(
    features: pd.DataFrame,
    target_col: str,
    random_state: int = 42,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """NGBoost-lite: two-tree distributional forecast (mean + variance)."""
    metrics = ModelMetrics(model_name="distributional")
    
    clean = features.dropna()
    if len(clean) < _MIN_OBS_TREE:
        metrics.error = f"Insufficient data ({len(clean)})"
        return None, metrics
    
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        feature_cols = [c for c in clean.columns if c != target_col]
        X = clean[feature_cols].values
        y = clean[target_col].values
        split = max(1, int(len(X) * 0.85))
        
        # Model 1: conditional mean
        mean_model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
        mean_model.fit(X[:split], y[:split])
        mu = float(mean_model.predict(X[-1:])[0])
        
        # Model 2: conditional log-variance (heteroscedastic)
        residuals = y[:split] - mean_model.predict(X[:split])
        var_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        var_model.fit(X[:split], residuals ** 2)
        sigma2 = max(float(var_model.predict(X[-1:])[0]), 1e-8)
        
        # Validation metrics
        if len(X) > split:
            preds = mean_model.predict(X[split:])
            mae, rmse = _compute_metrics(y[split:], preds)
            metrics.test_residuals = _compute_residuals(y[split:], preds)
        else:
            mae, rmse = float("nan"), float("nan")
        
        # Refit on all data
        mean_model.fit(X, y)
        residuals_all = y - mean_model.predict(X)
        var_model.fit(X, residuals_all ** 2)
        
        mu = float(mean_model.predict(X[-1:])[0])
        sigma2 = max(float(var_model.predict(X[-1:])[0]), 1e-8)
        
        metrics.mae = mae
        metrics.rmse = rmse
        metrics.fitted = True
        metrics.conditional_sigma = float(np.sqrt(sigma2))
        
        return np.array([mu]), metrics
    except Exception as exc:
        metrics.error = str(exc)
        return None, metrics
```

**Changes to `ModelMetrics`** (line 90):
```python
conditional_sigma: float | None = None  # Feature-dependent std from distributional model
```

**Wiring:** Called in `run_forecasting()` as an additional model in the cascade, after tree ensemble. The `conditional_sigma` flows to prediction_aggregator for data-driven band width.

**Lines changed:** ~55 new function, ~1 in ModelMetrics

---

### Method 3: Constrained Optimization (Bounds as Constraints)

**Goal:** Optimize the ensemble point forecast within conformal/MC bounds.

**File:** `operator1/models/prediction_aggregator.py`

**Edit location:** NEW function `_constrained_point_forecast()` called at line ~2660 (after all ensemble adjustments, before band computation).

**Implementation:**

```python
def _constrained_point_forecast(
    raw_forecast: float,
    lower_bound: float,
    upper_bound: float,
    last_close: float,
    momentum_5d: float,
    regime: str,
) -> float:
    """Optimize point forecast within bounds subject to momentum + mean-reversion."""
    if math.isnan(raw_forecast) or math.isnan(lower_bound) or math.isnan(upper_bound):
        return raw_forecast
    if lower_bound >= upper_bound:
        return raw_forecast
    
    try:
        from scipy.optimize import minimize_scalar
        
        # Regime-dependent mean reversion strength
        mr_weight = 0.15 if regime == "normal" else 0.05
        mom_weight = 0.30 if regime != "extreme_survival" else 0.10
        
        def objective(x):
            model_loss = (x - raw_forecast) ** 2
            momentum_loss = -(x - last_close) * momentum_5d * mom_weight
            mean_rev = (x - last_close) ** 2 * mr_weight
            return model_loss + momentum_loss + mean_rev
        
        result = minimize_scalar(
            objective, bounds=(lower_bound, upper_bound), method='bounded'
        )
        return float(result.x) if result.success else raw_forecast
    except Exception:
        return raw_forecast
```

**Integration point:** Called after Phase 1 conformal/RMSE band computation (line ~2700), using the computed lower/upper as bounds. The result replaces `point` before final HorizonPrediction construction.

```python
# After computing lower, upper at line ~2700:
if var_name == "close" and not math.isnan(point):
    _mom_5d = cache.get("return_5d", pd.Series([0.0])).iloc[-1] if "return_5d" in cache.columns else 0.0
    point = _constrained_point_forecast(
        point, lower, upper, _last_close, float(_mom_5d), result.current_regime,
    )
```

**Lines changed:** ~30 new function, ~5 integration lines

---

### Method 4: Scenario-Weighted MC Path Expectations (Entropy Pooling)

**Goal:** Reweight MC paths using model ensemble views, producing actionable scenario decomposition.

**File:** `operator1/models/prediction_aggregator.py`

**Edit location:** NEW function `_entropy_pooling_scenarios()` called after ensemble point computation.

**Implementation (inlined from fortitudo-tech, ~60 lines):**

```python
def _entropy_pooling_scenarios(
    mc_terminal_values: np.ndarray,
    model_forecast_return: float,
    last_close: float,
    n_scenarios: int = 4,
) -> dict:
    """Reweight MC paths to match model view via entropy pooling (Meucci 2010)."""
    if mc_terminal_values is None or len(mc_terminal_values) < 100:
        return {"available": False}
    
    try:
        from scipy.optimize import minimize, Bounds
        
        S = len(mc_terminal_values)
        p = np.ones(S) / S  # uniform prior
        log_p = np.log(p)
        
        # View: expected return = model_forecast_return
        A = mc_terminal_values.reshape(1, -1) - 1.0  # excess returns
        b = np.array([[model_forecast_return]])
        
        # Solve dual problem for posterior weights
        def dual_obj(lam):
            log_x = log_p.reshape(-1) - 1 - A.T.flatten() * lam[0]
            x = np.exp(np.clip(log_x, -500, 500))
            obj = x @ (log_x - log_p.flatten()) - lam[0] * (b[0,0] - A.flatten() @ x)
            grad = np.array([b[0,0] - A.flatten() @ x])
            return -obj, grad
        
        result = minimize(dual_obj, x0=np.array([0.0]), jac=True, method='L-BFGS-B')
        log_q = log_p.flatten() - 1 - A.T.flatten() * result.x[0]
        q = np.exp(np.clip(log_q, -500, 500))
        q = q / q.sum()  # normalize
        
        # Scenario decomposition
        mc_prices = last_close * mc_terminal_values
        pcts = [0, 25, 50, 75, 100]
        boundaries = np.percentile(mc_prices, pcts)
        
        scenarios = []
        labels = ["bear", "base_low", "base_high", "bull"]
        for i in range(len(pcts) - 1):
            mask = (mc_prices >= boundaries[i]) & (mc_prices < boundaries[i+1])
            if i == len(pcts) - 2:
                mask = (mc_prices >= boundaries[i])  # include upper boundary
            if mask.sum() > 0:
                scenarios.append({
                    "label": labels[i],
                    "probability": float(q[mask].sum()),
                    "target_price": float(np.average(mc_prices[mask], weights=q[mask])),
                    "n_paths": int(mask.sum()),
                })
        
        weighted_point = float(last_close * np.average(mc_terminal_values, weights=q))
        
        return {
            "available": True,
            "weighted_point": weighted_point,
            "scenarios": scenarios,
            "kl_divergence": float(np.sum(q * np.log(q / p.flatten() + 1e-30))),
        }
    except Exception:
        return {"available": False}
```

**Integration in run_prediction_aggregation():** After MC terminal values are used (line ~2710 area), call `_entropy_pooling_scenarios()` and store result in a new field on `PredictionAggregatorResult`.

**New field on `PredictionAggregatorResult`** (line 332):
```python
scenario_decomposition: dict = field(default_factory=dict)  # {horizon: {scenarios: [...]}}
```

**New field on `HorizonPrediction`** (line 296):
```python
scenario_weighted_point: float | None = None  # Entropy-pooling reweighted forecast
scenarios: list[dict] | None = None  # [{label, probability, target_price}]
```

**Lines changed:** ~65 new function, ~15 integration, ~3 dataclass fields

---

### Method 5: Reflected Brownian Motion (Barrier Reflection in MC)

**Goal:** Add price reflection at ATH and major support levels in MC path generation.

**File:** `operator1/models/monte_carlo.py`

**Edit location:** `simulate_return_paths()` at line 512, inside the per-step loop (line 576-611).

**Current code (line 599-600):**
```python
z = rng.normal(tilted_mean, tilted_std)
```

**New code (add after line 610, after jump component):**
```python
# Barrier reflection at ATH and support (Harrison 1985)
# Prevents paths from unrealistically penetrating strong
# technical levels. ATH acts as reflecting ceiling near
# all-time highs; support acts as reflecting floor near
# 252-day lows. Reflection creates naturally asymmetric
# distributions near boundaries.
if _barriers_active:
    cum_price = _start_price * np.exp(np.sum(return_paths[i, :t+1]))
    if cum_price > _ath_barrier:
        # Reflect: overshoot becomes undershoot
        excess = np.log(cum_price / _ath_barrier)
        return_paths[i, t] -= 2 * excess
    elif cum_price < _support_barrier:
        deficit = np.log(_support_barrier / cum_price)
        return_paths[i, t] += 2 * deficit
```

**New parameters for `simulate_return_paths()`:**
```python
ath_barrier: float = 0.0,       # ATH price (0 = disabled)
support_barrier: float = 0.0,   # Support price (0 = disabled)
```

**Wiring in `run_monte_carlo()`** (line 1170): Extract barriers from cache:
```python
_ath = float(cache["close"].max()) if "close" in cache.columns else 0.0
_support = float(cache["close"].rolling(252).min().iloc[-1]) if "close" in cache.columns else 0.0
```

Pass to `simulate_return_paths()` as `ath_barrier=_ath, support_barrier=_support`.

**Lines changed:** ~15 in simulate_return_paths, ~5 in run_monte_carlo

---

### Method 6: Bayesian Model Averaging (Between-Model Variance)

**Goal:** Add between-model variance to ensemble weight computation.

**File:** `operator1/models/prediction_aggregator.py`

**Edit location:** `compute_ensemble_weights()` at line 392 AND band computation at line 840.

**Current `compute_ensemble_weights()` returns:** `{model_name: weight}` (inverse-RMSE only)

**New function `compute_bma_weights()`:**

```python
def compute_bma_weights(
    metrics: list[ModelMetrics],
    forecasts: dict[str, dict[str, float]],
    variable: str,
    horizon: str,
) -> tuple[dict[str, float], float]:
    """Bayesian Model Averaging: weights + between-model variance.
    
    Returns
    -------
    (weights, between_model_std)
        weights: BMA posterior model weights (sum=1)
        between_model_std: std of point forecasts across models (0 if all agree)
    """
    base_weights = compute_ensemble_weights(metrics)
    if not base_weights:
        return {}, 0.0
    
    # Collect per-model forecasts for this variable/horizon
    model_forecasts = {}
    for m in metrics:
        if m.fitted and m.model_name in base_weights:
            fc = forecasts.get(m.variable, {}).get(horizon)
            if fc is not None and not math.isnan(fc):
                model_forecasts[m.model_name] = fc
    
    if len(model_forecasts) < 2:
        return base_weights, 0.0
    
    # BMA posterior mean
    mu_bma = sum(
        base_weights.get(name, 0) * fc
        for name, fc in model_forecasts.items()
    )
    
    # Between-model variance (Law of Total Variance)
    between_var = sum(
        base_weights.get(name, 0) * (fc - mu_bma) ** 2
        for name, fc in model_forecasts.items()
    )
    
    return base_weights, math.sqrt(max(between_var, 0.0))
```

**Integration in band computation** (modify `compute_uncertainty_bands()` at line 840):

Add `between_model_std` parameter:
```python
def compute_uncertainty_bands(
    point_forecast, rmse, horizon_days, *,
    survival_probability=1.0, survival_risk_multiplier=2.0, z_score=1.645,
    between_model_std: float = 0.0,  # NEW: BMA between-model uncertainty
) -> tuple[float, float]:
    # ... existing code ...
    # NEW: Add between-model variance (Law of Total Variance)
    within_var = base_spread ** 2
    between_var = (between_model_std * math.sqrt(max(horizon_days, 1))) ** 2
    total_spread = math.sqrt(within_var + between_var)
    adjusted_spread = total_spread * risk_factor
```

**Lines changed:** ~35 new function, ~5 modification to existing function

---

## Integration Points Summary

### ForecastResult (forecasting.py line 110)

| Field | Current | After |
|-------|---------|-------|
| `forecasts` | `{var: {horizon: float}}` | Unchanged |
| `metrics` | `[ModelMetrics]` | ModelMetrics gets `quantile_forecasts` + `conditional_sigma` |

### ModelMetrics (forecasting.py line 90)

| Field | Current | After |
|-------|---------|-------|
| `quantile_forecasts` | N/A | `dict[float, float] | None` -- P5/P25/P50/P75/P95 |
| `conditional_sigma` | N/A | `float | None` -- feature-dependent std from distributional model |

### HorizonPrediction (prediction_aggregator.py line 296)

| Field | Current | After |
|-------|---------|-------|
| `scenario_weighted_point` | N/A | `float | None` -- entropy-pooling reweighted |
| `scenarios` | N/A | `list[dict] | None` -- [{label, probability, target_price}] |
| `skew_signal` | N/A | `float | None` -- (P95-P50)-(P50-P5) from quantile regression |
| `between_model_std` | N/A | `float | None` -- BMA between-model disagreement |

### PredictionAggregatorResult (prediction_aggregator.py line 332)

| Field | Current | After |
|-------|---------|-------|
| `scenario_decomposition` | N/A | `dict` -- {horizon: {scenarios, kl_divergence}} |

### MonteCarloResult (monte_carlo.py)

Unchanged. Barrier reflection is internal to path generation.

### ConformalResult (conformal.py)

Unchanged. The existing `QuantileRegressionCalibrator` is already implemented but needs wiring into prediction_aggregator.

---

## Execution Order in Pipeline

```
Stage 4 (sub-stage 4.1): run_forecasting()
    |-- fit_tree_ensemble() now produces quantile_forecasts on ModelMetrics
    |-- fit_distributional() NEW -- produces conditional_sigma on ModelMetrics
    |
Stage 5 (sub-stage 5.4): run_monte_carlo()
    |-- simulate_return_paths() now has barrier reflection at ATH/support
    |-- terminal_values reflect boundary behavior (asymmetric near ATH)
    |
Stage 6 (sub-stage 6.3): build_conformal_result()
    |-- QuantileRegressionCalibrator already exists (line 920)
    |-- Wire into prediction flow (needs existing QRC.fit() call)
    |
Stage 6 (sub-stage 6.5): run_prediction_aggregation()
    |-- compute_bma_weights() replaces compute_ensemble_weights() for band width
    |-- _entropy_pooling_scenarios() produces scenario decomposition from MC paths
    |-- _constrained_point_forecast() optimizes point within bounds
    |-- Quantile forecasts from ModelMetrics -> asymmetric bands + skew signal
    |-- conditional_sigma from distributional model -> feature-dependent band width
```

---

## File-by-File Edit Summary

| File | Function | Change Type | Lines |
|------|----------|-------------|-------|
| `operator1/models/forecasting.py` | `ModelMetrics` dataclass | Add 2 fields | ~2 |
| `operator1/models/forecasting.py` | `fit_tree_ensemble()` | Add quantile prediction after main fit | ~25 |
| `operator1/models/forecasting.py` | `fit_distributional()` | NEW function (NGBoost-lite two-tree) | ~55 |
| `operator1/models/forecasting.py` | `run_forecasting()` | Call fit_distributional in cascade | ~10 |
| `operator1/models/monte_carlo.py` | `simulate_return_paths()` | Add barrier reflection params + logic | ~15 |
| `operator1/models/monte_carlo.py` | `run_monte_carlo()` | Extract ATH/support, pass to simulate | ~5 |
| `operator1/models/prediction_aggregator.py` | `HorizonPrediction` dataclass | Add 4 fields | ~4 |
| `operator1/models/prediction_aggregator.py` | `PredictionAggregatorResult` dataclass | Add 1 field | ~1 |
| `operator1/models/prediction_aggregator.py` | `compute_bma_weights()` | NEW function (BMA with between-model var) | ~35 |
| `operator1/models/prediction_aggregator.py` | `compute_uncertainty_bands()` | Add between_model_std parameter | ~5 |
| `operator1/models/prediction_aggregator.py` | `_entropy_pooling_scenarios()` | NEW function (Meucci entropy pooling) | ~65 |
| `operator1/models/prediction_aggregator.py` | `_constrained_point_forecast()` | NEW function (scipy optimize) | ~30 |
| `operator1/models/prediction_aggregator.py` | `run_prediction_aggregation()` | Wire all 6 methods into pipeline | ~40 |
| **Total** | | | **~292 lines** |

---

## Profile / Report Impact

### New Profile Fields

```json
{
  "predictions": {
    "close": {
      "5d": {
        "point_forecast": 245.30,
        "lower_ci": 238.50,
        "upper_ci": 249.80,
        "skew_signal": -2.3,
        "between_model_std": 1.85,
        "scenario_weighted_point": 244.10,
        "scenarios": [
          {"label": "bull", "probability": 0.35, "target_price": 252.0},
          {"label": "base_high", "probability": 0.30, "target_price": 245.0},
          {"label": "base_low", "probability": 0.25, "target_price": 239.0},
          {"label": "bear", "probability": 0.10, "target_price": 228.0}
        ]
      }
    }
  }
}
```

### Report Section Changes

The report generator reads `HorizonPrediction` fields. The new `scenarios` field enables a new report subsection in Section 9 (Predictions):

```markdown
### Scenario Decomposition (5-day horizon)
| Scenario | Probability | Target |
|----------|-------------|--------|
| Bull     | 35%         | $252   |
| Base+    | 30%         | $245   |
| Base-    | 25%         | $239   |
| Bear     | 10%         | $228   |

**Skew:** -2.3 (downside exceeds upside -- left-skewed distribution)
**Model Agreement:** Between-model std = $1.85 (moderate disagreement)
```

---

## Testing Strategy

Each method can be tested independently:

1. **Quantile Regression:** Verify P5 < P25 < P50 < P75 < P95 (no crossing). Verify asymmetry near ATH (P5 much further from P50 than P95).
2. **Distributional:** Verify conditional_sigma varies with features (higher during volatile periods, lower during stable).
3. **Constrained Optimization:** Verify point_forecast is always within [lower, upper] bounds.
4. **Entropy Pooling:** Verify scenario probabilities sum to 1.0. Verify KL divergence > 0 (paths were reweighted).
5. **Reflected BM:** Verify MC path distribution is left-skewed when close is near ATH; right-skewed near support.
6. **BMA:** Verify total_variance >= within_variance (between-model term is non-negative). Verify bands widen when models disagree.

---

## Dependencies

**Zero new pip dependencies.** All methods use numpy, scipy, sklearn, and xgboost -- all already installed.

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Quantile crossing (P25 > P75) | Use single QRF model (zillow pattern) or enforce monotonicity post-hoc |
| Entropy pooling numerical instability | Clip log-weights to [-500, 500], scale objective by 1000 (fortitudo pattern) |
| Constrained optimization slow | Use `minimize_scalar` (bounded, 1D) instead of `minimize` (multi-D) |
| Distributional variance model overfits | Cap variance at 10x the unconditional variance; min_samples_leaf=10 |
| Barrier reflection distorts MC survival probability | Only reflect for "close" variable paths, not for ratio evolution paths |
| BMA between-model variance too large when one model is outlier | Cap between_model_std at 3x the median model RMSE |

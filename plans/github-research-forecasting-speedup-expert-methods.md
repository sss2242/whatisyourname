# GitHub Research: Forecasting Speedup Expert Methods

*Detailed findings from examining 7 open-source projects for each of the 6 proposed acceleration methods.*

**Projects examined:**
1. **Nixtla/statsforecast** (4,777 stars) -- ETS, Croston, Theta, GARCH, intermittent demand models
2. **Nixtla/mlforecast** (1,222 stars) -- Global LightGBM cross-series forecasting with lag transforms
3. **unit8co/darts** (9,367 stars) -- NaiveEnsemble and RegressionEnsemble patterns
4. **uber/orbit** (2,050 stars) -- Bayesian structural time series (DLT/LGT)
5. **aj-cloete/pssa** (148 stars) -- Pure-numpy SSA implementation
6. **facebook/prophet** (20,174 stars) -- Structural decomposition + regressors
7. **sktime/sktime** (9,757 stars) -- Unified forecasting framework with model selection

---

## Method 1: Hurst-Based Model Routing

**Goal:** Profile each variable in ~1ms, skip the 6-model cascade entirely.

### GitHub Findings

No dedicated Hurst-routing library exists on GitHub. The Hurst exponent is computed in many projects but never used as a model router. This is a novel pattern.

**Key implementation insight from our own codebase:** `hurst_exponent_rolling` is already in the cache (computed in `derived_variables.py` Stage 19 via R/S analysis). `adaptive_model_params.py` also computes a Hurst exponent. We can read the latest value directly -- zero computation needed.

**Recommended routing table (from Lopez de Prado AFML Ch.17):**

```python
def _route_by_hurst(hurst: float, tier: int) -> str:
    """Route variable to optimal model based on Hurst exponent."""
    if tier <= 2:
        return "kalman"  # unchanged for survival-critical vars
    if hurst > 0.55:
        return "ets"     # trending: ETS captures momentum
    if hurst < 0.45:
        return "ar1"     # mean-reverting: AR(1) is optimal
    return "baseline"    # random walk: no model beats naive
```

**Useful pattern from sktime:** `sktime.registry.all_estimators(filter_tags={"capability:pred_int": True})` shows how to query model capabilities. We could add a `"best_for_hurst_range"` tag to our model wrappers.

**Implementation effort:** ~20 lines. Read cache column, return model name.

---

## Method 2: Croston/TSB for Intermittent Filing-Frequency Series

**Goal:** Handle forward-filled quarterly/annual financial statement variables that LSTM wastes 40s failing on.

### GitHub Findings from statsforecast

**Source:** `/tmp/statsforecast/python/statsforecast/models.py` lines 4600-5440

statsforecast provides 5 intermittent demand models:

| Model | Class | Description | Speed |
|-------|-------|-------------|-------|
| `CrostonClassic` | Line 4600 | Original 1972 method, fixed alpha=0.1 | ~0.01s |
| `CrostonOptimized` | Line 4781 | Optimized smoothing parameters | ~0.02s |
| `CrostonSBA` | Line 4932 | Syntetos-Boylan Approximation (bias-corrected) | ~0.02s |
| `IMAPA` | Line 5105 | Intermittent Multiple Aggregation Prediction Algorithm | ~0.03s |
| `TSB` | Line 5267 | Teunter-Syntetos-Babai with separate demand probability | ~0.02s |

**Key implementation detail:** All models use the same `fit(y) -> predict(h)` interface. The `y` array is the raw time series -- zeros are treated as zero-demand periods. For our use case, the forward-filled values should be converted to a sparse series where only filing dates have non-zero values (the actual reported number) and all other dates are zero.

**TSB is the best fit for our use case** because it models demand probability separately from demand size. For quarterly filers, the "demand probability" at any given day is ~1/63 (one filing per 63 trading days). TSB's `alpha_d` and `alpha_p` parameters control how quickly the model adapts to changes in filing frequency and filing magnitude.

**Recommended usage pattern:**
```python
from statsforecast.models import TSB

# Convert forward-filled to sparse (only filing dates have values)
sparse = cache[var].diff().fillna(0)  # non-zero only on change days
sparse[sparse == 0] = 0  # explicit zeros

model = TSB(alpha_d=0.1, alpha_p=0.1)
model.fit(sparse.values)
forecast = model.predict(h=horizons)
```

**Alternative approach (simpler):** Since our `_frequency_classifier.py` already identifies forward-filled columns, we can just use `AutoETS` on the sparse (filing-date-only) subset and interpolate between forecasted filing values. This avoids the Croston API entirely.

**Implementation effort:** ~30 lines. Detect filing columns, convert to sparse, fit TSB or ETS on sparse, interpolate.

---

## Method 3: Google BSTS / Structural Time Series

**Goal:** One flexible model replaces the 6-model cascade for Tier3 variables.

### GitHub Findings from uber/orbit

**Source:** `/tmp/orbit/orbit/models/dlt.py`

Orbit's DLT (Damped Local Trend) is their workhorse model. Key features:
- Decomposes into level + slope + seasonality + regressors
- `damped_factor=0.8` prevents trend explosion (critical for financial data)
- `global_trend_option="linear"` with cap/floor for bounded forecasts
- Supports regressor columns with sign constraints (`+`, `-`, `=`)
- Estimators: `stan-mcmc` (slow, full Bayesian), `stan-map` (fast, point estimate)

**Problem for us:** Orbit requires PyStan which is heavy and has C++ compilation requirements. Not suitable for our sandbox.

**Better alternative already in deps:** `statsmodels.tsa.statespace.structural.UnobservedComponents` provides the same local-level/local-linear-trend decomposition without PyStan:

```python
from statsmodels.tsa.statespace.structural import UnobservedComponents

model = UnobservedComponents(
    endog=series,
    level='llevel',     # local level
    trend=False,        # no trend component (financial data is noisy)
    irregular=True,     # observation noise
)
result = model.fit(disp=False, maxiter=50)
forecast = result.forecast(steps=h)
```

**Key insight from Prophet:** Prophet's success comes from decomposition + regressors, NOT from complex models. For our Tier3 variables (close, return_1d, volatility), a simple UnobservedComponents with `level='llevel'` (same as our existing Kalman filter) but without the VAR/LSTM cascade overhead is sufficient.

**Verdict:** BSTS adds complexity without clear benefit over our existing Kalman. Skip this method. Use ETS ensemble (Method 6) instead.

**Implementation effort:** N/A (skipped).

---

## Method 4: Global LightGBM Cross-Variable Model

**Goal:** Train ONE model across all 31 variables simultaneously instead of 31 separate fits.

### GitHub Findings from Nixtla/mlforecast

**Source:** `/tmp/mlforecast/mlforecast/forecast.py` line 921, `/tmp/mlforecast/mlforecast/core.py`

mlforecast's architecture is exactly what we need:

**Data format:** Long format DataFrame with `unique_id` (variable name), `ds` (date), `y` (value).

```python
# How mlforecast structures multi-series data:
df = pd.DataFrame({
    'unique_id': ['close'] * 500 + ['revenue'] * 500 + ...,
    'ds': dates * 31,
    'y': all_values,
})
```

**Lag feature construction:** `mlforecast/lag_transforms.py` provides:
- `RollingMean(window_size=21)` -- rolling average
- `RollingStd(window_size=21)` -- rolling volatility
- `ExponentiallyWeightedMean(alpha=0.1)` -- EWMA
- `Offset(n=1)` -- lag features (shift by n)
- All implemented in C via `coreforecast` for speed

**Model fitting pattern:**
```python
from mlforecast import MLForecast
from lightgbm import LGBMRegressor

mlf = MLForecast(
    models=[LGBMRegressor(n_estimators=100, learning_rate=0.05, verbosity=-1)],
    freq=1,
    lags=[1, 5, 21, 63],
    lag_transforms={
        1: [RollingMean(window_size=21), RollingStd(window_size=21)],
        21: [RollingMean(window_size=63)],
    },
)
mlf.fit(df)
forecasts = mlf.predict(h=5)
```

**Key insight: hyperparameter space from `auto.py`:**
```python
def lightgbm_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 20, 1000, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 4096, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "learning_rate": 0.05,
    }
```

**For our use case:** We don't need mlforecast as a dependency -- we can replicate the pattern in ~50 lines:
1. Stack all 31 variables into long format with lag features
2. One `LGBMRegressor.fit()` call on 31 x 500 = 15,500 rows x ~10 features
3. Predict all 31 variables in one `predict()` call

**Expected timing:** ~2s for fit + predict (vs 31 x 40s = 1,240s for individual LSTM fits).

**Implementation effort:** ~50 lines. Build stacked DataFrame, construct lag features, one LGB fit, unstack predictions.

---

## Method 5: Singular Spectrum Analysis (SSA)

**Goal:** Non-parametric decomposition forecast in ~0.05s per variable.

### GitHub Findings from aj-cloete/pssa

**Source:** `/tmp/pssa/mySSA.py`

The core SSA algorithm in ~100 lines:
1. **Embed:** Build Hankel trajectory matrix from time series (window size L = N/2)
2. **Decompose:** SVD of trajectory matrix -> singular values + left/right singular vectors
3. **Group:** Select top-K components (trend + oscillatory, discard noise)
4. **Reconstruct:** Anti-diagonal averaging of selected components
5. **Forecast:** Extend reconstructed components forward

**Key code patterns:**
```python
# Embedding: Hankel matrix construction
X = linalg.hankel(ts, np.zeros(L)).T[:,:K]

# Decomposition: SVD
U, s, Vt = np.linalg.svd(X, full_matrices=False)

# Reconstruction: select top-r components
X_reconstructed = sum(s[i] * np.outer(U[:,i], Vt[i,:]) for i in range(r))

# Diagonal averaging: convert back to time series
reconstructed_series = diagonal_averaging(X_reconstructed)
```

**For our use case:** SSA is well-suited for Tier4/5 variables (margins, PE, revenue growth) which have smooth long-term trends contaminated by quarterly filing noise. The embedding window should be set to `filing_period / 2` (Nyquist alignment, same as our adaptive_windows).

**No new dependencies needed.** Only numpy + scipy.linalg.

**Potential issue:** SSA's `diagonal_averaging` is O(L*K) which is fast for 500-row series but the SVD step is O(min(L,K)^2 * max(L,K)). For L=250 (half of 500), this is ~0.05s.

**Implementation effort:** ~40 lines. Inline from pssa (MIT license).

---

## Method 6: Simple Ensemble of 4 Fast Models

**Goal:** Run 4 fast models in parallel and average, replacing the cascade.

### GitHub Findings from darts

**Source:** `/tmp/darts/darts/models/forecasting/naive_ensemble_model.py` line 19

darts' `NaiveEnsembleModel` is exactly this pattern:
```python
class NaiveEnsembleModel(EnsembleModel):
    """Returns the average of all predictions of the constituent models"""
    
    def fit(self, series, ...):
        # Trains all constituent models
        super().fit(series, ...)
    
    def predict(self, n, ...):
        # Gets predictions from all models, returns average
        predictions = [model.predict(n) for model in self.models]
        return sum(predictions) / len(predictions)
```

**RegressionEnsembleModel** (line 28) goes further -- uses a meta-learner (Ridge/Linear regression) to learn optimal weights from holdout predictions instead of equal weights.

**For our use case with statsforecast models:**
```python
from statsforecast.models import AutoETS, Theta, Naive, WindowAverage

models = [
    AutoETS(season_length=1),           # ~0.05s
    Theta(season_length=1),             # ~0.03s  
    Naive(),                            # ~0.001s
    WindowAverage(window_size=21),      # ~0.001s
]

# Fit all, average predictions
forecasts = []
for model in models:
    model.fit(y)
    forecasts.append(model.predict(h=horizon)['mean'])
ensemble_forecast = np.mean(forecasts, axis=0)
```

**Key insight from CDC COVID hub (Ray et al. 2023):** Equal-weight ensemble consistently beat every individual model AND every weighted ensemble. The reason: weighted ensembles overfit to recent performance, while equal weights are robust to distribution shift. For financial data with regime changes, this robustness is valuable.

**darts RegressionEnsemble uses holdout validation for weights.** We could do the same using our walk-forward infrastructure:
```python
# Use walk-forward MAE per model as inverse-weight
weights = 1.0 / np.array([model_mae_1, model_mae_2, model_mae_3, model_mae_4])
weights /= weights.sum()
weighted_forecast = np.average(forecasts, axis=0, weights=weights)
```

But the CDC research says don't bother -- equal weights win.

**Implementation effort:** ~25 lines. Fit 4 statsforecast models, average predictions.

---

## Cross-Project Patterns Worth Adopting

### 1. Conformal Prediction Intervals (statsforecast)

statsforecast's `ConformalIntervals` class (used by all models) provides distribution-free prediction intervals via conformal prediction. This is exactly what our `conformal.py` does, but statsforecast bakes it into the model API:

```python
model = AutoETS(
    season_length=1,
    prediction_intervals=ConformalIntervals(h=5, n_windows=3),
)
model.fit(y)
result = model.predict(h=5, level=[90])
# result contains 'lo-90' and 'hi-90' columns
```

**Benefit for us:** If we use statsforecast models for the fast path, we get conformal intervals for free without running our separate conformal calibrator on those variables.

### 2. Direct Multi-Horizon (mlforecast)

mlforecast supports `max_horizon=N` which trains N separate models, one per horizon. This is better than recursive multi-step forecasting (our current approach in `recursive_aggregator.py`) because it avoids error accumulation:

```python
mlf = MLForecast(models=[LGBMRegressor()], freq=1, lags=[1,5,21])
mlf.fit(df, max_horizon=5)  # trains 5 models
forecasts = mlf.predict(h=5)  # each horizon from its own model
```

**Benefit for us:** The recursive aggregator (sub-stage 6.11) chains day-by-day predictions with cumulative error. Direct multi-horizon avoids this entirely.

### 3. Target Transforms (mlforecast)

mlforecast's `Differences` and `LocalStandardScaler` target transforms normalize the target before fitting and inverse-transform predictions. This is similar to our `feature_normalization.py` but applied at the model level:

```python
from mlforecast.target_transforms import Differences, LocalStandardScaler

mlf = MLForecast(
    models=[LGBMRegressor()],
    target_transforms=[Differences([1]), LocalStandardScaler()],
)
```

**Benefit for us:** Could replace our manual z-score normalization for tree models.

---

## Summary: What to Implement

| # | Method | Source Project | Lines | New Deps | Speedup |
|---|--------|---------------|-------|----------|---------|
| 1 | ETS+Theta+Naive+WindowAvg ensemble | statsforecast + darts pattern | ~25 | None (statsforecast in requirements) | 640s saved |
| 2 | Distributional model guard (5 vars) | N/A (config change) | ~5 | None | 55s saved |
| 3 | Hurst-based routing | Novel (our cache has the value) | ~20 | None | 100s saved |
| 4 | Global LightGBM | mlforecast pattern (inlined) | ~50 | None (lightgbm installed) | Optional (replaces ensemble) |
| 5 | SSA for Tier4/5 | pssa (inlined, MIT) | ~40 | None (numpy/scipy) | Alternative to ETS |
| 6 | Croston/TSB for filing vars | statsforecast | ~30 | None (statsforecast in requirements) | 200s at MF freq |
| ~~7~~ | ~~BSTS/Orbit~~ | ~~orbit~~ | ~~N/A~~ | ~~PyStan (heavy)~~ | ~~Skipped~~ |

**Priority order:** Method 1 (ensemble) + Method 2 (guard) = 90% of speedup in ~30 lines. Add Method 3 (Hurst routing) for intelligence. Method 4 (global LGB) is a future upgrade. Method 5 (SSA) is an alternative. Method 6 (Croston) matters at multi-frequency stages.

**BSTS/Orbit dropped** -- requires PyStan compilation, and UnobservedComponents (already installed) provides the same local-level model we already have via Kalman.

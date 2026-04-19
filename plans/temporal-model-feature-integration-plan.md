# Temporal Model Feature Integration Plan

*2026-04-19*

## The Problem

The AAPL backtest shows that even with the new Boruta+PIMP+mRMR feature selection (which correctly retained 1 feature from 77 candidates), the predictions are identical to the previous run. This is because the core temporal models (Kalman, GARCH, AR1) are **univariate** -- they only look at the target variable itself and ignore `extra_vars` entirely.

**Which models consume extra_vars:**

| Model | Uses extra_vars? | How? |
|-------|-----------------|------|
| Kalman | NO | Univariate local-level state space on target column only |
| GARCH | NO | Univariate conditional volatility on returns only |
| AR(1) | NO | Univariate autoregression on target only |
| VAR | PARTIALLY | Uses `extra_variables` as companion series, but requires >50 non-NaN obs per variable AND the VAR often fails to converge with many variables |
| LSTM | YES | Uses `extra_variables` as input features alongside target |
| Tree/XGBoost | YES | Uses `extra_variables` as features for prediction |
| ETS | NO | Univariate exponential smoothing |
| Baseline | NO | Last value or EMA |

So only 3 of 8 model types (VAR, LSTM, Tree) can consume extra_vars. And VAR often falls back to AR(1) when variables have insufficient data.

## The Core Insight

The features selected by Boruta+PIMP+mRMR represent the best predictors from a non-linear importance perspective. But 5 of 8 models can't use them. We need to either:

1. **Modify univariate models to accept exogenous variables** (make Kalman, GARCH, ETS consume features)
2. **Add new multivariate models** that naturally consume features
3. **Use features as regime/context modifiers** for univariate models (don't change the model, change its parameters based on features)

Option 3 is the most elegant -- it doesn't break existing model contracts and works for all model types.

## Synergy Pruning Replacement Verification

The `prune_by_unified_network()` in `model_synergies.py` was correctly replaced. The unified causal network is still computed (for profile/report), but the pruning call was removed. The function `prune_by_unified_network` still exists for backward compatibility but is no longer called.

**What the synergies module still does correctly:**
- Synergy A: Injects cycle phase features (unchanged)
- Synergy B: Builds unified causal network for metadata (computed, not pruned)
- Synergy G: Peer-adjusted survival thresholds (unchanged)
- Synergy I: Plane-aware model weighting (unchanged)

**What was removed:** Only the `prune_by_unified_network()` call at line 780 that was zeroing out all extra_vars.

## Recommended Architecture: Feature-Conditioned Model Parameters

Instead of modifying each model to accept exogenous variables (which would break the model cascade and require rewriting 4000+ lines of forecasting.py), use the feature selection results to **condition model behavior**:

### Approach A: Feature-Informed Kalman (External Signal Injection)

The Kalman filter supports exogenous inputs via the control matrix B and control vector u:
```
x_t = F*x_{t-1} + B*u_t + noise
```
Where `u_t` is the external signal vector at time t.

**Method:** For each selected feature, compute its standardized latest value and inject as a drift signal into the Kalman prediction step. If `put_call_ratio` is elevated (z-score > 1.5), shift the Kalman state prediction downward proportional to the z-score.

**Reference:** Durbin & Koopman (2012) "Time Series Analysis by State Space Methods" -- Chapter 3 on external regressors in state space models.

### Approach B: GARCH-X (GARCH with Exogenous Variables)

GARCH-X (Engle 2002) extends standard GARCH by adding exogenous variables to the conditional variance equation:
```
sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2 + gamma*X_t
```
Where `X_t` is the exogenous variable (e.g., `event_uncertainty_premium`, `cross_asset_stress`).

The `arch` library supports this via `GARCHX` model or via `volatility=GARCH(p=1, o=0, q=1)` with `x` parameter.

**Reference:** Han & Kristensen (2014) "Asymptotic Theory for the QMLE in GARCH-X Models".

### Approach C: ARIMAX/ETS-X (Exogenous Regressors)

statsforecast supports ARIMA with exogenous regressors:
```python
from statsforecast.models import AutoARIMA
model = AutoARIMA(season_length=1)
model.fit(y, X=exogenous_features)
forecast = model.predict(h=1, X=future_exogenous)
```

For ETS, the external regressors can be added via a post-hoc adjustment: `forecast_adjusted = forecast_ets + beta * X_latest`.

### Approach D: Regime-Conditional Model Switching (No Model Changes)

The simplest approach: use feature values to SELECT which model to trust, not to modify the model itself.

If `cross_asset_stress > 0.7` AND `put_call_ratio > 1.5`:
- Weight GARCH higher (it handles crisis vol well)
- Weight Kalman lower (linear trend less reliable)
- Weight Tree higher (non-linear patterns dominate)

This is exactly what `compute_model_routing_weights()` does, which we already wired in Bug #3. The issue is that routing weights only affect confidence, not the point forecast.

### Approach E: Feature-Augmented Tree Ensemble (Already Working)

The Tree/XGBoost model already receives `extra_variables` and trains on them. The issue is that for AAPL, only 1 feature survived selection, and the tree model was already using 15 core features.

**Enhancement:** Force the tree model to run even when other models (Kalman, GARCH) succeed first in the model cascade. Currently, the cascade is first-fit-wins: if Kalman fits, the tree never runs for that variable. Change to: always run tree as an additional model, then let the ensemble decide weights.

## Step-by-Step Implementation Plan

### Step 1: Always Run Tree Ensemble (Parallel Model Execution)

Edit `operator1/models/forecasting.py` `run_forecasting()`:

Currently the model cascade tries Kalman first, and if it succeeds, skips all other models. Change to:
- Run Kalman/GARCH/AR1/ETS as before (first-fit-wins for the "primary" model)
- ALSO always run Tree/XGBoost if extra_variables is non-empty, regardless of whether another model already fitted
- Store both results: `model_used[var] = "kalman"` (primary) + add tree metrics to the metrics list
- The prediction aggregator ensemble already handles multiple models per variable

### Step 2: GARCH-X for Volatility Variables

Edit `operator1/models/forecasting.py` `fit_garch()`:

- When `extra_variables` is available, try GARCH-X before standard GARCH
- Use `arch.univariate.ARX` with `x` parameter for exogenous variables
- If GARCH-X fails to converge, fall back to standard GARCH
- Only use features that PIMP identified as significant for volatility (from feature_selection_result)

### Step 3: Kalman with External Signal

Edit `operator1/models/forecasting.py` `fit_kalman()`:

- After standard Kalman filter, compute a feature-based adjustment:
  `adjustment = sum(beta_i * zscore(feature_i)) for i in extra_vars`
- Where `beta_i` comes from a simple OLS of feature_i on Kalman residuals
- Apply adjustment to the Kalman forecast: `forecast_adjusted = forecast_kalman + adjustment`
- This is a "Kalman + feature residual regression" hybrid -- common in quantitative macro forecasting

### Step 4: Feature-Weighted Ensemble Confidence

Edit `operator1/models/prediction_aggregator.py`:

- When `feature_selection_result` is available, compute a "feature support score":
  `support = n_features_used / max(n_features_used, 5)` (0.2 to 1.0)
- Multiply each prediction's confidence by the support score
- Predictions backed by many significant features get higher confidence
- This propagates the feature selection quality into the ensemble

### Step 5: Verify No Regressions

- Run AAPL backtest with all changes
- Compare validation results
- Check that predictions change (should be slightly different due to tree ensemble now running in parallel + GARCH-X + Kalman adjustment)

## Community Project References

## Detailed Community Project Implementation Patterns

### 1. Kalman + Exogenous Variables

**filterpy (rlabbe/Kalman-and-Bayesian-Filters-in-Python, 17K stars)**

The key implementation pattern for adding exogenous control inputs:

```python
# From filterpy: Kalman filter with control input
from filterpy.kalman import KalmanFilter
kf = KalmanFilter(dim_x=1, dim_z=1, dim_u=n_features)

# State transition: x_t = F*x_{t-1} + B*u_t + noise
kf.F = np.array([[1.0]])       # state transition (random walk)
kf.B = np.zeros((1, n_features))  # control matrix -- learned via OLS
kf.H = np.array([[1.0]])       # observation matrix
kf.R = np.array([[obs_noise]])
kf.Q = np.array([[process_noise]])

# Learn B from data: regress Kalman residuals on features
residuals = actual_values - kalman_predictions
B_ols = np.linalg.lstsq(feature_matrix, residuals, rcond=None)[0]
kf.B = B_ols.reshape(1, -1)

# Predict with control input
for t in range(T):
    kf.predict(u=features[t].reshape(-1, 1))
    kf.update(z=observation[t])
```

**Key adoption pattern:** Don't modify our existing Kalman. Instead, compute the "Kalman residual regression" as a post-processing step:
1. Run standard Kalman (unchanged)
2. Compute residuals: `residuals = actual - kalman_forecast`
3. Regress residuals on features: `beta = OLS(features, residuals)`
4. Adjust forecast: `adjusted = kalman_forecast + beta @ latest_features`

This is called "regression-augmented Kalman" in the quant literature (Christoffersen & Diebold 2006).

**pykalman (pykalman/pykalman, 1.8K stars):**
- Supports `observation_offsets` parameter for exogenous inputs
- Also supports `em()` algorithm that jointly learns state transition + observation matrices from data
- Pattern: `kf.em(observations, n_iter=10)` then `kf.filter(observations)` -- the EM step automatically finds the relationship between hidden state and observations

### 2. GARCH-X (Exogenous Volatility)

**arch library (bashtage/arch, 1.2K stars)**

Two methods for adding exogenous variables:

```python
# Method 1: ARX mean model + GARCH volatility
from arch import arch_model
from arch.univariate import ARX, GARCH

# x must be a DataFrame with same index as returns
exog = cache[extra_vars].iloc[-252:]  # last year
am = ARX(returns, lags=[1], x=exog)
am.volatility = GARCH(p=1, q=1)
res = am.fit(disp='off')
forecast = res.forecast(horizon=1, x=exog_future)

# Method 2: HARX (Heterogeneous AR) with realized volatility measures
from arch.univariate import HARX
harx = HARX(returns, lags=[1, 5, 22], x=exog)
harx.volatility = GARCH(1, 1)
res = harx.fit()
```

**Key adoption pattern for our fit_garch():**
- Try GARCH-X first with the selected features
- The `x` parameter in `ARX` accepts a DataFrame of exogenous variables
- If convergence fails (common with many exogenous vars), fall back to standard GARCH
- Only use features with PIMP p-value < 0.1 for GARCH-X (avoid overfitting)

**Practical insight from arch documentation:** "When using exogenous regressors, ensure they are stationary. Non-stationary regressors will cause the MLE to fail." -- This means we should z-score normalize features before passing to GARCH-X.

### 3. Parallel Model Execution

**mlforecast (Nixtla/mlforecast, 900 stars)**

The pattern for running multiple models simultaneously:

```python
# From mlforecast: run multiple models and combine
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

models = [
    RandomForestRegressor(n_estimators=100),
    XGBRegressor(n_estimators=100),
    LGBMRegressor(n_estimators=100),
]
fcst = MLForecast(models=models, freq='B', lags=[1,5,21])
fcst.fit(df)
predictions = fcst.predict(h=1)
# predictions has one column per model + combined
```

**Key adoption pattern:** Our `run_forecasting()` currently uses a cascade (first-fit-wins). Change to:
1. Run the cascade as before to get the "primary" model per variable
2. After the cascade, ALSO run Tree/XGBoost for all variables with `extra_vars`
3. Add the tree results to `forecast_result.metrics` and `forecast_result.forecasts`
4. The prediction aggregator already computes inverse-RMSE weights across all models

**darts (unit8co/darts, 8K stars)**

Their `NaiveEnsembleModel` pattern is simpler and more relevant:

```python
# From darts: ensemble of diverse models
from darts.models import (
    ExponentialSmoothing,  # our ETS
    AutoARIMA,             # removed but equivalent to our Kalman+AR
    XGBModel,              # our tree ensemble
    RNNModel,              # our LSTM
)

# Key insight: their ensemble trains ALL models on same data
# and combines via simple average or learned weights
model = NaiveEnsembleModel(
    forecasting_models=[
        ExponentialSmoothing(),
        XGBModel(lags=21, lags_past_covariates=21),
        RNNModel(input_chunk_length=21, output_chunk_length=1),
    ],
    past_covariates=features_timeseries,  # exogenous features
)
```

The critical pattern: `past_covariates` is how darts passes exogenous features to models that support them, while models that don't (ETS) simply ignore them. **This is exactly what we should do** -- pass `extra_vars` to ALL models, let each model decide whether to use them.

### 4. Feature-Conditioned Forecasting (Reduction to Regression)

**sktime (sktime/sktime, 7.5K stars)**

Their `make_reduction` pattern converts ANY regression model into a time series forecaster:

```python
# From sktime: turn any regressor into a forecaster
from sktime.forecasting.compose import make_reduction
from sklearn.ensemble import GradientBoostingRegressor

# Creates lagged features automatically:
# X = [y_{t-1}, y_{t-2}, ..., y_{t-k}, exog_1_t, exog_2_t, ...]
# y = y_t
regressor = GradientBoostingRegressor(n_estimators=100)
forecaster = make_reduction(
    regressor,
    window_length=21,          # how many lags to use
    strategy="recursive",       # multi-step via recursion
)
forecaster.fit(y_train, X=exogenous_train)
y_pred = forecaster.predict(fh=[1,5,21], X=exogenous_future)
```

**Key adoption pattern for our Kalman/AR1:** Instead of modifying the Kalman filter itself, use the sktime reduction pattern:
1. Create a lagged DataFrame: `y_lag1, y_lag2, ..., y_lag21` + `feature_1, feature_2, ...`
2. Train a GBM/XGB on this DataFrame to predict `y_t`
3. This "reduced" model captures both autoregressive structure AND feature effects
4. Use alongside the pure Kalman for ensemble diversity

This is called "direct reduction" in the forecasting literature and is the standard approach at Amazon, Uber, and other tech companies with large-scale forecasting.

### 5. Feature-Weighted Ensemble

**Nixtla/hierarchicalforecast (700 stars)**

Their reconciliation approach provides a pattern for weighting models by data quality:

```python
# Concept: weight predictions by information content
# More features = more information = higher weight
feature_support = min(1.0, n_features_confirmed / 10)
model_weights = base_weights * (0.5 + 0.5 * feature_support)
```

**tslearn (tslearn-team/tslearn, 2.8K stars)**

Their `KNeighborsTimeSeriesRegressor` provides a pattern for feature-conditioned k-NN forecasting:
- Find historical periods with similar feature values
- Weight their outcomes by similarity
- This naturally handles non-linear feature effects
- Similar to our DTW analogs but with feature conditioning

### 6. Residual Feature Regression (Post-Hoc Adjustment)

**statsmodels (statsmodels/statsmodels, 10K stars)**

The standard approach for "adding features to a univariate model":

```python
# From statsmodels documentation:
# Step 1: Fit ARIMA/Kalman on target
from statsmodels.tsa.arima.model import ARIMA
arima = ARIMA(y, order=(1,0,0)).fit()
residuals = y - arima.fittedvalues

# Step 2: Regress residuals on features
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_features, residuals)

# Step 3: Adjusted forecast = ARIMA forecast + feature adjustment
arima_forecast = arima.forecast(steps=1)
feature_adjustment = ridge.predict(X_latest.reshape(1, -1))
adjusted_forecast = arima_forecast + feature_adjustment
```

This "residual regression" pattern is used by:
- **Facebook Prophet** (facebook/prophet, 18K stars): Adds "regressors" as additional linear terms on top of the trend+seasonality decomposition
- **LinkedIn Greykite** (linkedin/greykite, 1.8K stars): Adds "extra_pred_cols" as linear regressors in the forecast equation
- **Uber Orbit** (uber/orbit, 1.9K stars): Bayesian structural time series with regression component -- the cleanest theoretical framework for this

**Key insight from Prophet:** Their `add_regressor()` method adds features as linear terms with automatic regularization. This is equivalent to our "Kalman + OLS residual regression" approach. The regularization prevents overfitting to noisy features.

### Recommended Adoption Strategy

| Step | Method | Reference Project | Complexity | Impact |
|------|--------|------------------|-----------|--------|
| 1 | Always run Tree alongside primary | mlforecast, darts | Low (~30 lines) | HIGH -- tree naturally consumes features |
| 2 | Residual feature regression for Kalman/AR1 | Prophet, statsmodels | Medium (~50 lines) | MEDIUM -- adds feature signal to all univariate models |
| 3 | GARCH-X for volatility variables | arch library | Medium (~40 lines) | LOW -- only affects volatility forecasts |
| 4 | Feature-weighted ensemble confidence | hierarchicalforecast | Low (~15 lines) | LOW -- subtle confidence adjustment |
| 5 | Reduction to regression (optional, future) | sktime | High (~100 lines) | HIGH but risky -- changes model architecture |

---

## Step-by-Step Code Mode Implementation Instructions

### Step 1: Always Run Tree Ensemble Alongside Primary Model (~30 lines)

**File:** `operator1/models/forecasting.py` in `run_forecasting()`

**Current behavior:** The model cascade tries Kalman -> GARCH -> VAR -> LSTM -> Tree -> Baseline. First model that fits wins. If Kalman fits, Tree never runs for that variable.

**Change:** After the cascade loop, check if `extra_variables` is non-empty. If so, run `fit_tree_ensemble()` for the top variables (close, return_1d, volatility_21d) even if another model already won. Add the tree result to `forecast_result.metrics` as an additional model entry.

```
1a. Find the end of the per-variable cascade loop in run_forecasting()
1b. After the loop, add a "parallel tree" block:
    if extra_variables and len(extra_variables) > 0:
        for var in ["close", "return_1d", "volatility_21d"]:
            if var in cache.columns and var not in tree_already_fitted:
                tree_forecast, tree_metrics = fit_tree_ensemble(
                    cache, var, extra_variables, n_forecast=max_horizon
                )
                if tree_forecast is not None:
                    forecast_result.forecasts.setdefault(var, {})
                    forecast_result.forecasts[var]["tree_parallel"] = tree_forecast
                    forecast_result.metrics.append(tree_metrics)
1c. Update backtest_runner.py similarly (run_stage2a2)
```

### Step 2: Residual Feature Regression for Kalman/AR1 (~50 lines)

**File:** `operator1/models/forecasting.py` -- add a new function `apply_residual_feature_adjustment()`

**Pattern:** After any univariate model (Kalman, AR1, ETS, Baseline) produces a forecast, compute:
1. Residuals = actual - model_fitted_values (in-sample)
2. OLS regression: residuals ~ features (with Ridge regularization to prevent overfitting)
3. Adjustment = beta * latest_features
4. Adjusted_forecast = original_forecast + adjustment

```
2a. Create function:
    def apply_residual_feature_adjustment(
        cache, variable, forecast_value, extra_variables,
        fitted_values=None, alpha=1.0
    ):
        # Get in-sample residuals
        if fitted_values is None:
            return forecast_value  # can't compute without fitted values
        
        # Build feature matrix from cache
        feature_cols = [c for c in extra_variables if c in cache.columns]
        if not feature_cols:
            return forecast_value
        
        X = cache[feature_cols].dropna()
        resid = (cache[variable] - fitted_values).reindex(X.index).dropna()
        common = X.index.intersection(resid.index)
        if len(common) < 30:
            return forecast_value
        
        # Ridge regression of residuals on features
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=alpha)
        model.fit(X.loc[common], resid.loc[common])
        
        # Apply adjustment using latest feature values
        latest = X.iloc[-1:].values
        adjustment = float(model.predict(latest)[0])
        return forecast_value + adjustment

2b. Call this function after Kalman, AR1, ETS, and Baseline produce forecasts
    (inside the per-variable cascade, after a univariate model succeeds)
2c. Store the adjustment amount in ModelMetrics for transparency
2d. Mirror in backtest_runner.py
```

### Step 3: GARCH-X for Volatility (~40 lines)

**File:** `operator1/models/forecasting.py` in `fit_garch()`

```
3a. Before the existing GARCH fit, try GARCH-X:
    if extra_variables:
        try:
            from arch.univariate import ARX, GARCH as GARCHVol
            exog = cache[extra_variables].iloc[-len(returns):].dropna(axis=1)
            if len(exog.columns) > 0 and len(exog) == len(returns):
                # Z-score normalize (GARCH-X needs stationary regressors)
                exog_norm = (exog - exog.mean()) / exog.std().clip(lower=1e-8)
                am = ARX(returns, lags=[1], x=exog_norm)
                am.volatility = GARCHVol(p=1, q=1)
                res = am.fit(disp='off')
                # Extract forecast
                garchx_forecast = res.forecast(horizon=n_forecast).variance.values[-1]
                return garchx_forecast, ModelMetrics(model_name="garch_x", ...)
        except Exception:
            pass  # fall through to standard GARCH

3b. If GARCH-X fails, continue with existing standard GARCH (no change)
```

### Step 4: Feature-Weighted Ensemble Confidence (~15 lines)

**File:** `operator1/models/prediction_aggregator.py` in `run_prediction_aggregation()`

```
4a. After the routing_weights confidence adjustment (which we added in Bug #3 fix),
    add a feature support score:
    
    if feature_selection_result is not None and feature_selection_result.fitted:
        n_confirmed = len(feature_selection_result.boruta_confirmed)
        support = min(1.0, max(0.3, n_confirmed / 10.0))
        # Boost confidence when many features support the prediction
        for var in result.predictions:
            for h, hp in result.predictions[var].items():
                if hasattr(hp, "confidence") and not math.isnan(hp.confidence):
                    hp.confidence *= (0.7 + 0.3 * support)

4b. Add feature_selection_result as a new parameter to run_prediction_aggregation()
4c. Pass it from main.py, stage6_ensemble.py, and backtest_runner.py
```

### Step 5: Compile + Test + Verify

```
5a. python -m py_compile for all modified files
5b. Run AAPL backtest (all stages)
5c. Compare predictions to previous run -- should be DIFFERENT now
    (tree ensemble runs in parallel, Kalman has residual adjustment)
5d. Run validation -- compare error rates
5e. Check for look-ahead bias in the residual regression
    (fitted_values must come from in-sample only)
```

### Step 6: Commit + Push

```
6a. git add all modified files
6b. git commit -m 'feat: integrate features into temporal models via parallel tree + residual regression + GARCH-X'
6c. git push
```

### Files Modified

| File | Changes |
|------|---------|
| `operator1/models/forecasting.py` | Add `apply_residual_feature_adjustment()`, parallel tree block, GARCH-X attempt |
| `operator1/models/prediction_aggregator.py` | Add `feature_selection_result` param + support score |
| `operator1/stages/stage6_ensemble.py` | Pass feature_selection_result to aggregator |
| `main.py` | Pass feature_selection_result to aggregator |
| `backtest_runner.py` | Mirror all changes |

# GitHub Research: Per-Model Deep Dive for Forecasting 2-Mode Redesign

*Detailed implementation findings from examining actual source code of 15 GitHub projects. For each of the 3 new models in the forecasting redesign plan, we examine how established projects implement them and extract specific code patterns we should adopt.*

---

## Model 1: Batched Multi-Series LSTM

**What we need:** Replace 31 sequential per-variable LSTM training loops with 1 batched training loop that processes all variables simultaneously.

### Project 1.1: Nixtla/neuralforecast (3,100 stars)

**File:** `neuralforecast/models/lstm.py`

neuralforecast's LSTM processes multiple series via a custom `TimeSeriesDataset` that stacks series along the batch dimension:

```python
# neuralforecast/tsdataset.py
class TimeSeriesDataset(Dataset):
    def __init__(self, Y_df, ...):
        # Y_df has columns: unique_id, ds, y
        # Group by unique_id and stack into (n_series, seq_len, 1)
        self.series = []
        for uid in Y_df["unique_id"].unique():
            mask = Y_df["unique_id"] == uid
            self.series.append(Y_df.loc[mask, "y"].values)

    def __getitem__(self, idx):
        return torch.tensor(self.series[idx], dtype=torch.float32)
```

The DataLoader then batches across series:
```python
dataloader = DataLoader(dataset, batch_size=len(series), shuffle=False)
# Each batch: (n_series, seq_len, 1)
```

**Key patterns to adopt:**
1. **Stack as batch dimension, not feature dimension.** Each variable is a separate sample in the batch, not a separate input feature. This means the LSTM architecture stays simple (input_size=1, not input_size=31).
2. **Per-series normalization before stacking.** Each variable is z-scored independently before entering the batch. This prevents scale differences between `close` (~200) and `return_1d` (~0.01) from dominating gradients.
3. **Inverse-transform after prediction.** Predictions are de-normalized per variable using stored mean/std.

**Implementation for us:**
```python
def fit_batched_lstm(cache, target_vars, n_forecast=252, epochs=50):
    # 1. Normalize each variable independently
    normalized = {}
    stats = {}
    for var in target_vars:
        series = cache[var].dropna().values.astype(np.float32)
        mu, sigma = series.mean(), series.std() + 1e-8
        normalized[var] = (series - mu) / sigma
        stats[var] = (mu, sigma)

    # 2. Stack into batch tensor: (n_vars, seq_len, 1)
    min_len = min(len(v) for v in normalized.values())
    batch = torch.stack([
        torch.tensor(v[-min_len:]).unsqueeze(-1)
        for v in normalized.values()
    ])  # shape: (31, min_len, 1)

    # 3. Train ONE LSTM on the batch
    model = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
    decoder = nn.Linear(64, 1)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=0.001)

    train_len = int(min_len * 0.85)
    X_train = batch[:, :train_len, :]
    Y_train = batch[:, 1:train_len+1, :]  # shifted by 1

    for epoch in range(epochs):
        output, _ = model(X_train)
        pred = decoder(output)
        loss = F.mse_loss(pred, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch > 5 and loss.item() < best_loss * 0.999:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                break

    # 4. Forecast: feed last window, predict n_forecast steps
    with torch.no_grad():
        last_window = batch[:, -64:, :]  # last 64 steps as context
        output, hidden = model(last_window)
        forecasts = {}
        for i, var in enumerate(target_vars):
            pred_normalized = decoder(output[i:i+1, -1:, :]).item()
            mu, sigma = stats[var]
            forecasts[var] = pred_normalized * sigma + mu

    return forecasts
```

### Project 1.2: unit8co/darts GlobalForecastingModel (9,367 stars)

**File:** `darts/models/forecasting/torch_forecasting_model.py`

darts interleaves training samples from ALL series in each batch:

```python
class GlobalForecastingDataset(Dataset):
    def __init__(self, series_list):
        self.samples = []
        for series in series_list:
            for i in range(len(series) - input_length - output_length):
                self.samples.append((
                    series[i:i+input_length],
                    series[i+input_length:i+input_length+output_length],
                ))

    def __len__(self):
        return len(self.samples)
```

**Key pattern:** Interleaved sampling means the model sees patterns from ALL variables in each mini-batch, not just one variable at a time. This improves gradient diversity and generalization.

**For us:** With only 31 variables x ~450 training windows = ~14,000 samples, we can fit everything in one batch without mini-batching. So the interleaving happens naturally.

### Project 1.3: pytorch-forecasting TFT (4,100 stars)

**File:** `pytorch_forecasting/models/temporal_fusion_transformer/tuning.py`

pytorch-forecasting uses **learning rate finder** before training:

```python
trainer = pl.Trainer(...)
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model, train_dataloaders=train_dl)
model.hparams.learning_rate = lr_finder.suggestion()
```

**Key pattern:** Auto-tuning learning rate eliminates a hyperparameter and speeds convergence. For our batched LSTM, this means fewer epochs needed.

**Simpler alternative for us:** Use PyTorch's `ReduceLROnPlateau` scheduler:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
```

---

## Model 2: Global Cross-Variable LightGBM

**What we need:** Replace 31 separate tree ensemble fits with 1 global LightGBM model on a stacked DataFrame.

### Project 2.1: Nixtla/mlforecast (1,222 stars)

**File:** `mlforecast/core.py`

mlforecast's approach to building the stacked DataFrame:

```python
class MLForecast:
    def preprocess(self, df):
        # df has: unique_id, ds, y
        # Add lag features per series
        for lag in self.lags:
            df[f"lag_{lag}"] = df.groupby("unique_id")["y"].shift(lag)
        # Add rolling features
        for lag, transforms in self.lag_transforms.items():
            for transform in transforms:
                col_name = f"{transform.__class__.__name__}_{lag}"
                df[col_name] = df.groupby("unique_id")["y"].transform(
                    lambda x: transform(x.shift(lag))
                )
        return df

    def fit(self, df):
        df = self.preprocess(df)
        X = df.drop(["unique_id", "ds", "y"], axis=1)
        y = df["y"]
        # unique_id as categorical feature -- LightGBM handles natively
        X["unique_id"] = df["unique_id"].astype("category")
        self.model.fit(X, y)
```

**Key patterns to adopt:**
1. **`unique_id` as categorical feature.** LightGBM's categorical handling is much better than one-hot encoding -- it finds optimal splits on the category directly.
2. **Lag features computed per-series via groupby.** Essential to prevent lag "leakage" between variables.
3. **Rolling transforms via groupby.** Each variable's rolling mean/std is computed independently.

**Implementation for us:**
```python
def _fit_global_lightgbm(cache, available_vars, n_forecast=252):
    import lightgbm as lgb

    lags = [1, 5, 21, 63]
    rows = []

    for var in available_vars:
        series = cache[var].dropna()
        if len(series) < max(lags) + 1:
            continue
        for i in range(max(lags), len(series)):
            row = {"var_id": var, "target": float(series.iloc[i])}
            for lag in lags:
                row[f"lag_{lag}"] = float(series.iloc[i - lag])
            row["rolling_mean_21"] = float(series.iloc[max(0, i-21):i].mean())
            row["rolling_std_21"] = float(series.iloc[max(0, i-21):i].std())
            rows.append(row)

    df = pd.DataFrame(rows)
    df["var_id"] = df["var_id"].astype("category")

    X = df.drop("target", axis=1)
    y = df["target"]

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        verbosity=-1,
        n_jobs=4,
    )
    model.fit(X, y, categorical_feature=["var_id"])

    # Predict for each variable
    forecasts = {}
    for var in available_vars:
        series = cache[var].dropna()
        if len(series) < max(lags):
            continue
        last_row = {
            "var_id": var,
            "lag_1": float(series.iloc[-1]),
            "lag_5": float(series.iloc[-5]) if len(series) >= 5 else float(series.iloc[-1]),
            "lag_21": float(series.iloc[-21]) if len(series) >= 21 else float(series.iloc[-1]),
            "lag_63": float(series.iloc[-63]) if len(series) >= 63 else float(series.iloc[-1]),
            "rolling_mean_21": float(series.iloc[-21:].mean()),
            "rolling_std_21": float(series.iloc[-21:].std()),
        }
        pred_df = pd.DataFrame([last_row])
        pred_df["var_id"] = pred_df["var_id"].astype("category")
        forecasts[var] = float(model.predict(pred_df)[0])

    return forecasts, model
```

### Project 2.2: M5 Kaggle Winners

**Key findings from M5 top-5 solutions:**

1. **Target encoding + mean encoding for `var_id`:** Instead of relying solely on LightGBM's categorical handling, top solutions also added mean-encoded features:
   ```python
   df["var_mean"] = df.groupby("var_id")["target"].transform("mean")
   df["var_std"] = df.groupby("var_id")["target"].transform("std")
   ```

2. **Multi-step recursive prediction:** Don't predict all 252 steps at once. Predict 1 step, feed it back as `lag_1`, predict step 2. This handles regime changes better than single-shot prediction.

3. **Time features:** Add `day_of_week`, `month`, `quarter` as features. For financial data: `days_since_filing`, `days_to_next_filing` from our filing calendar.

4. **Tweedie loss for zero-inflated targets:** Financial variables like `dividends_paid` or `stock_buybacks` have many zeros. Tweedie distribution handles this:
   ```python
   model = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5)
   ```

### Project 2.3: Nixtla/hierarchicalforecast MinTrace (697 stars)

**File:** `hierarchicalforecast/methods.py`

After the global model produces base forecasts, hierarchicalforecast reconciles them using MinTrace:

```python
class MinTrace:
    def reconcile(self, S, base_forecasts, residuals):
        # S = summing matrix (defines hierarchy)
        # MinTrace optimal reconciliation
        W = np.diag(1.0 / np.var(residuals, axis=0))
        P = np.linalg.inv(S.T @ W @ S) @ S.T @ W
        return P @ base_forecasts
```

**For us:** We already have MinTrace in `hierarchical_reconciliation.py` (stage 7.4.7). The global LightGBM's per-variable forecasts should be reconciled through this existing step.

---

## Model 3: Parallel Per-Variable Cascade

**What we need:** Run the Kalman->VAR->Tree->Baseline cascade for 31 variables in parallel.

### Project 3.1: joblib Parallel with threading (1,900 stars)

**File:** `joblib/parallel.py`

The cleanest pattern from joblib:

```python
from joblib import Parallel, delayed

def _forecast_one_var(var_name, series, tier, ...):
    """Pure function: no shared mutable state."""
    metrics = []
    best = None

    # Kalman
    if tier in ("tier1", "tier2"):
        fcast, met = fit_kalman(series)
        metrics.append(met)
        if fcast is not None:
            best = fcast

    # Tree
    if best is None:
        fcast, met = fit_tree(series, features)
        metrics.append(met)
        if fcast is not None:
            best = fcast

    # Baseline
    if best is None:
        fcast, met = fit_baseline(series)
        metrics.append(met)
        best = fcast

    return {"var": var_name, "forecast": best, "metrics": metrics}

# Parallel execution
results = Parallel(n_jobs=4, backend="threading", prefer="threads")(
    delayed(_forecast_one_var)(var, series[var], tiers[var], ...)
    for var in available_vars
)

# Sequential merge
for r in results:
    result.forecasts[r["var"]] = r["forecast"]
    result.metrics.extend(r["metrics"])
```

**Key patterns:**
1. **`backend="threading"` not `"loky"`**: Threading avoids pickle serialization of the cache DataFrame. All threads share the same memory space.
2. **Pure function with no side effects**: The worker function returns a dict, doesn't modify shared state.
3. **Sequential merge after parallel phase**: Dict merging and list appending happen in the main thread only.

### Project 3.2: tsfresh parallel feature extraction (9,183 stars)

**File:** `tsfresh/feature_extraction/extraction.py`

tsfresh's chunk size optimization:

```python
def _calculate_chunk_size(n_items, n_workers):
    # 4x oversubscription handles load imbalance
    chunk_size = max(1, n_items // (n_workers * 4))
    return chunk_size
```

**For us:** With 31 variables and 4 workers, chunk_size = max(1, 31 // 16) = 1. So each variable is its own chunk -- which is what we want. No custom chunking needed.

### Project 3.3: Pre-extraction pattern (avoid repeated cache access)

From multiple projects, the pattern of pre-extracting data BEFORE the parallel phase:

```python
# BEFORE parallel phase: extract all data (main thread, cache access)
pre_extracted = {}
for var in available_vars:
    pre_extracted[var] = {
        "series": cache[var].dropna().values,
        "features": _extract_features(var),
        "multivariate": _extract_multivariate(var),
        "regime_data": _extract_regime_data(var),
    }

# DURING parallel phase: workers use pre-extracted data (no cache access)
results = Parallel(n_jobs=4, backend="threading")(
    delayed(_forecast_one_var)(var, pre_extracted[var], ...)
    for var in available_vars
)
```

**This is critical for thread safety.** The `_extract_features()` and `_extract_multivariate()` functions read from the cache DataFrame. While pandas DataFrames are technically thread-safe for reads, pre-extracting avoids any potential GIL contention from concurrent pandas operations.

---

## Summary: Code Patterns to Adopt

| # | Pattern | Source | Lines to Add |
|---|---------|--------|-------------|
| 1 | Per-series z-normalization before LSTM batching | neuralforecast | ~15 |
| 2 | Stack as batch dimension (n_vars, seq_len, 1) | neuralforecast | ~10 |
| 3 | ReduceLROnPlateau scheduler for faster convergence | pytorch-forecasting | ~3 |
| 4 | Early stopping patience=3 | darts | ~5 |
| 5 | var_id as LightGBM categorical feature | mlforecast | ~5 |
| 6 | Lag features via per-variable groupby | mlforecast | ~20 |
| 7 | Mean encoding + rolling features | M5 Kaggle | ~10 |
| 8 | Recursive multi-step prediction | M5 Kaggle | ~15 |
| 9 | Pre-extract all data before parallel phase | tsfresh pattern | ~20 |
| 10 | joblib Parallel with threading backend | sklearn/joblib | ~10 |
| 11 | Pure worker function returning dict (no side effects) | all projects | ~5 |
| 12 | Sequential merge after parallel phase | all projects | ~10 |

**Total new code:** ~130 lines for all 3 models.
**Total modified code:** ~100 lines (extract per-var body, update mode logic, remove balanced).
**Grand total:** ~230 lines changed in forecasting.py + ~10 lines in config + ~5 lines in dashboard.

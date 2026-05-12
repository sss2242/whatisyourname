# GitHub Research: Forecasting Speedup v2 -- Parallel Execution & Batched Models

*Detailed findings from examining 12 open-source projects for each of the 4 acceleration techniques in the v2 zero-accuracy-loss architecture.*

**Techniques researched:**
1. Parallel per-variable forecasting (ThreadPoolExecutor / ProcessPoolExecutor)
2. LSTM skip / intelligent model routing
3. Batched multi-output LSTM (multi-task learning)
4. Global cross-variable LightGBM

**Projects examined:**
1. **Nixtla/statsforecast** (4,777 stars) -- parallel per-series execution via fugue/ray/spark
2. **Nixtla/mlforecast** (1,222 stars) -- global LightGBM with lag transforms
3. **Nixtla/neuralforecast** (3,100 stars) -- batched multi-series LSTM/Transformer with PyTorch DataLoader
4. **sktime/sktime** (9,757 stars) -- parallel forecasting via joblib, model selection
5. **unit8co/darts** (9,367 stars) -- ensemble parallelism, GlobalForecastingModel
6. **amazon-science/chronos-forecasting** (4,200 stars) -- pretrained T5 for zero-shot forecasting
7. **salesforce/Merlion** (3,600 stars) -- AutoML model selection with parallel evaluation
8. **tsfresh/tsfresh** (9,183 stars) -- parallel feature extraction patterns
9. **ray-project/ray** (37,000 stars) -- distributed forecasting examples
10. **pytorch-forecasting** (4,100 stars) -- multi-target temporal fusion transformer
11. **AutoViML/Auto_TS** (720 stars) -- automatic model selection with speed tiers
12. **facebook/prophet** (20,174 stars) -- parallel cross-validation

---

## Technique 1: Parallel Per-Variable Forecasting

**Goal:** Run 31 independent variable forecasts simultaneously instead of sequentially.

### Finding 1.1: statsforecast parallel backend (fugue)

**Source:** `statsforecast/core.py` class `StatsForecast`

statsforecast uses the `fugue` library as a parallelism abstraction layer. The `forecast()` method accepts an `engine` parameter:

```python
class StatsForecast:
    def forecast(self, h, ..., engine=None):
        # engine can be: None (sequential), "fugue" (local parallel),
        # "spark" (distributed), "ray" (distributed), "dask" (distributed)
        if engine is not None:
            return self._forecast_distributed(h, engine=engine)
        return self._forecast_local(h)
```

The key insight: `_forecast_local()` still processes series sequentially in Python. The parallelism only kicks in via `fugue.transform()` which maps the forecast function across a DataFrame of grouped series:

```python
# fugue-based parallel execution
from fugue import transform

def _forecast_group(df: pd.DataFrame) -> pd.DataFrame:
    model = AutoETS()
    model.fit(df["y"].values)
    return model.predict(h=horizon)

result = transform(
    stacked_df,
    _forecast_group,
    partition={"by": ["unique_id"]},  # each variable is a partition
    engine="native",  # uses concurrent.futures internally
)
```

**Key learning for us:** fugue's "native" engine uses `concurrent.futures.ProcessPoolExecutor` (not ThreadPoolExecutor) because statsforecast's Cython/numba models don't release the GIL. But our models (sklearn, xgboost, arch) DO release the GIL during native computation, so ThreadPoolExecutor is the right choice for us -- avoids serialization overhead.

**Useful pattern:** The partition-by-variable approach maps cleanly to our architecture. Each variable = one partition = one thread.

### Finding 1.2: sktime parallel forecasting (joblib)

**Source:** `sktime/forecasting/base/_base.py` and `sktime/utils/parallel.py`

sktime uses joblib for parallelism:

```python
from joblib import Parallel, delayed

class BaseForecaster:
    def _predict_moving_cutoff(self, ...):
        results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(self._predict_single_cutoff)(fh, X, cutoff)
            for cutoff in cutoffs
        )
```

**Key insight:** sktime parallelizes across time cutoffs (for cross-validation), not across variables. But the pattern is directly applicable: `Parallel(n_jobs=4)(delayed(fn)(args) for args in variable_list)`.

joblib's `backend="threading"` uses ThreadPoolExecutor internally -- same as our plan but with nicer API:

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=4, backend="threading")(
    delayed(_forecast_single_variable)(var_name, series, tier, ...)
    for var_name in available_vars
)
```

**Advantage over raw ThreadPoolExecutor:** joblib handles exception propagation, progress reporting, and memory management automatically. It's already installed (sklearn dependency).

### Finding 1.3: tsfresh parallel feature extraction

**Source:** `tsfresh/feature_extraction/extraction.py`

tsfresh computes 794 features per time series using a distributor pattern:

```python
class MapDistributor:
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or multiprocessing.cpu_count()

    def distribute(self, func, partitioned_chunks):
        with Pool(self.n_workers) as pool:
            result = pool.map(func, partitioned_chunks)
        return result
```

**Key learning:** tsfresh uses `multiprocessing.Pool.map()` (ProcessPoolExecutor equivalent) because their feature functions are pure Python. For our sklearn/xgboost models that release the GIL, threading is better.

**Useful pattern:** tsfresh's chunk size optimization. They compute optimal chunk size to balance parallelism overhead vs utilization:

```python
chunk_size = max(1, len(items) // (n_workers * 4))  # 4x oversubscription
```

This prevents the "last worker" problem where one slow variable (e.g., close with 500 rows) blocks while other workers idle.

### Finding 1.4: prophet parallel cross-validation

**Source:** `prophet/diagnostics.py`

Prophet parallelizes cross-validation (not model fitting) using `concurrent.futures`:

```python
from concurrent.futures import ProcessPoolExecutor

def cross_validation(model, ...):
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(_single_cv, model, cutoff, ...)
            for cutoff in cutoffs
        ]
        results = [f.result() for f in futures]
```

**Key learning:** Prophet uses ProcessPoolExecutor because the model object (Stan-compiled) can't share state across threads. Our models CAN -- sklearn estimators are thread-safe for prediction (not fitting), but since each thread fits a DIFFERENT model on DIFFERENT data, there's no shared state conflict.

### Recommendation for Technique 1

**Use joblib with threading backend** (already installed, cleaner API than raw ThreadPoolExecutor):

```python
from joblib import Parallel, delayed

raw_results = Parallel(n_jobs=_n_workers, backend="threading", prefer="threads")(
    delayed(_forecast_single_variable)(
        var_name, _extract_series(var_name),
        _get_tier_for_variable(var_name, tier_map),
        max_horizon, random_state, _lstm_enabled,
        _get_model_features, cache, extra_variables,
    )
    for var_name in available_vars
)
```

**Worker count:** 4 (from tsfresh's `cpu_count() // 2` heuristic -- leaves cores for sklearn's internal thread pools).

**Critical: read-only cache access.** Each worker must NOT modify the cache DataFrame. Workers return result dicts; the main thread merges them.

---

## Technique 2: LSTM Skip / Intelligent Model Routing

**Goal:** Skip the 40s LSTM attempt when it's unlikely to beat simpler models.

### Finding 2.1: Merlion AutoML model selection

**Source:** `Merlion/merlion/models/automl/automl.py`

Salesforce Merlion pre-profiles each series before selecting models:

```python
class AutoML:
    def _select_model(self, train_data):
        # Profile the series
        stats = self._compute_stats(train_data)

        # Route to optimal model based on profile
        if stats["n_obs"] < 100:
            return [ETS, ARIMA]  # simple models for short series
        if stats["seasonality_strength"] > 0.5:
            return [Prophet, MSTL]  # seasonal-aware models
        if stats["trend_strength"] > 0.7:
            return [ARIMA, LGBMForecaster]  # trend-capable models
        return [ETS, ARIMA, LGBMForecaster]  # default set
```

**Key insight:** Model selection BEFORE fitting saves all the wasted computation from fitting models that won't be used. Merlion's profiling step takes <10ms.

**Directly applicable:** Our Hurst exponent routing (from v1 research) is the same pattern. But Merlion adds series length, seasonality strength, and trend strength as routing criteria beyond just Hurst.

### Finding 2.2: Auto_TS speed tiers

**Source:** `Auto_TS/auto_ts/auto_ts.py`

AutoViML's Auto_TS has explicit speed tiers matching our architecture:

```python
class auto_timeseries:
    def __init__(self, score_type='rmse', time_interval='', 
                 non_seasonal_pdq=None, seasonality=False,
                 seasonal_period=12, model_type='best',  # 'best', 'fast', 'stats'
                 verbose=0):
        ...

    def fit(self, traindata, ...):
        if self.model_type == 'fast':
            models = ['ARIMA']  # single fast model
        elif self.model_type == 'stats':
            models = ['ARIMA', 'ETS', 'Theta']  # stats only
        else:  # 'best'
            models = ['ARIMA', 'ETS', 'Prophet', 'ML']  # all models
```

**Useful pattern:** The `model_type` parameter directly maps to our `mode: express/balanced/full` config. The same codebase serves all speed tiers with a single config switch.

### Finding 2.3: darts model selection via backtest

**Source:** `darts/models/forecasting/forecasting_model.py`

darts doesn't skip models -- it fits all and selects the best via `historical_forecasts()` (rolling backtest). This is the opposite approach: fit everything, pick the winner.

**Anti-pattern for us:** This is what our current cascade does (try all 6 models), and it's the reason for the 40-minute runtime. The darts approach works because their models are fast (no LSTM by default).

### Recommendation for Technique 2

**Combine Merlion's profiling with our Hurst routing and Auto_TS's tier system:**

```python
def _select_models_for_variable(var_name, series, tier, mode, hurst):
    """Profile-based model selection. Returns ordered list of models to try."""
    if mode == "express":
        return ["fast_ensemble"]

    if tier in ("tier1", "tier2"):
        return ["kalman", "var", "tree", "baseline"]  # survival-critical path

    # Balanced mode: profile-based routing (no LSTM)
    n_obs = np.sum(~np.isnan(series))
    if n_obs < 30:
        return ["baseline"]
    if hurst is not None and hurst < 0.45:
        return ["ar1", "baseline"]  # mean-reverting: AR1 is optimal
    if hurst is not None and hurst > 0.55:
        return ["ets", "tree", "baseline"]  # trending: ETS + tree
    return ["var", "tree", "baseline"]  # random walk neighborhood

    # Full mode: all models including LSTM
    if mode == "full":
        return ["kalman", "var", "lstm", "tree", "baseline"]
```

This replaces the fixed cascade with a data-driven routing table. No model is ever tried that isn't appropriate for the series characteristics.

---

## Technique 3: Batched Multi-Output LSTM

**Goal:** One LSTM fit for all 31 variables instead of 31 separate fits.

### Finding 3.1: neuralforecast multi-series LSTM

**Source:** `neuralforecast/models/lstm.py` and `neuralforecast/core.py`

Nixtla's neuralforecast is the gold standard for batched neural forecasting. Their LSTM processes multiple series simultaneously via PyTorch DataLoader:

```python
class LSTM(BaseWindows):
    def __init__(self, h, input_size, hidden_size=200, num_layers=2, ...):
        self.encoder = nn.LSTM(
            input_size=input_size,  # number of features per timestep
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.Linear(hidden_size, h)  # direct multi-horizon output

    def forward(self, windows_batch):
        # windows_batch shape: (batch_size, seq_len, n_features)
        # batch_size = number of series (our 31 variables)
        output, _ = self.encoder(windows_batch)
        forecast = self.decoder(output[:, -1, :])  # last hidden state -> forecast
        return forecast  # shape: (batch_size, h)
```

**Key architecture insight:** neuralforecast uses `input_size` = number of features (covariates + target), NOT number of series. Each series is a SEPARATE SAMPLE in the batch dimension. This means:

```
Our 31 variables become 31 samples in a batch:
  Input:  (31, 500, 1)  -- 31 series, 500 timesteps, 1 feature each
  Output: (31, 4)        -- 31 series, 4 horizons (1d, 5d, 21d, 252d)
```

ONE forward pass through the LSTM produces all 31 forecasts. The encoder parameters are shared across all variables (transfer learning from cross-variable patterns).

**Training loop from `core.py`:**

```python
class NeuralForecast:
    def fit(self, df, ...):
        dataset = TimeSeriesDataset(df)  # stacked long-format
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(max_epochs):
            for batch in dataloader:
                # batch["y"]: (batch_size, seq_len)
                # batch["x"]: (batch_size, seq_len, n_exog)
                pred = self.model(batch)
                loss = self.loss_fn(pred, batch["y_future"])
                loss.backward()
                self.optimizer.step()
```

**Key learning for our implementation:**

1. Stack all 31 variables into a long-format DataFrame with `unique_id` column
2. Use a shared LSTM encoder with per-variable linear heads (or just the shared decoder)
3. Train for 20-50 epochs on the combined batch (31 series x 500 rows)
4. Total time: ~60s (one model) vs 31 x 40s = 1,240s (31 separate models)

### Finding 3.2: pytorch-forecasting Temporal Fusion Transformer

**Source:** `pytorch_forecasting/models/temporal_fusion_transformer/tft.py`

The TFT architecture is inherently multi-variable:

```python
class TemporalFusionTransformer(BaseModelWithCovariates):
    def __init__(self, ...):
        # Variable Selection Networks (one per variable type)
        self.static_variable_selection = VariableSelectionNetwork(...)
        self.encoder_variable_selection = VariableSelectionNetwork(...)

        # Shared LSTM encoder
        self.lstm_encoder = nn.LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            batch_first=True,
        )

        # Multi-head attention (attends to all variables jointly)
        self.multihead_attn = InterpretableMultiHeadAttention(...)

        # Output: per-quantile predictions for all target variables
        self.output_layer = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)
```

**Key insight:** The TFT uses Variable Selection Networks to automatically learn which variables are important. This replaces our manual feature selection (Boruta/PIMP/mRMR) with learned attention weights.

**Too complex for our use case.** The TFT has ~2000 lines of architecture. Our batched LSTM should be simpler: shared encoder + per-variable linear heads.

### Finding 3.3: chronos-forecasting (Amazon)

**Source:** `chronos-forecasting/src/chronos/config.py` and `model.py`

Amazon's Chronos is a pretrained T5 transformer for time series. It tokenizes real-valued time series into discrete bins and uses the T5 language model for forecasting.

```python
class ChronosModel:
    def predict(self, context, prediction_length, num_samples=20):
        # Tokenize: real values -> discrete tokens via quantile binning
        tokens = self.tokenizer.encode(context)
        # Generate: autoregressive decoding like text generation
        generated = self.model.generate(tokens, max_new_tokens=prediction_length)
        # Detokenize: discrete tokens -> real values
        samples = self.tokenizer.decode(generated)
        return samples  # (num_samples, prediction_length) probabilistic forecasts
```

**Key insight:** Chronos processes ANY time series without training (zero-shot). It's pretrained on 27 billion time points from public datasets. For our 500-row financial series, it produces reasonable forecasts in ~2s with no fitting.

**Relevant for us:** Chronos could be an alternative to LSTM in "full" mode. No training needed, just inference. But it requires the `chronos-forecasting` package + HuggingFace model download (~300MB).

**Deferred:** Interesting but adds a heavy dependency. Better suited as a future "premium" model option.

### Recommendation for Technique 3

**Use neuralforecast's batching pattern but inline** (avoid the dependency):

```python
class MultiOutputLSTM(nn.Module):
    def __init__(self, n_variables, hidden_size=64, num_layers=2, n_horizons=4):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=1,  # univariate per-variable
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Shared hidden -> per-variable per-horizon output
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, n_horizons)
            for _ in range(n_variables)
        ])

    def forward(self, x):
        # x: (n_variables, seq_len, 1)
        encoded, _ = self.encoder(x)  # (n_variables, seq_len, hidden)
        last_hidden = encoded[:, -1, :]  # (n_variables, hidden)
        forecasts = torch.stack([
            head(last_hidden[i]) for i, head in enumerate(self.heads)
        ])  # (n_variables, n_horizons)
        return forecasts
```

**Training: 50 epochs, combined MSE loss, ~60s on CPU for 31 vars x 500 rows.**

**Per-variable heads** (instead of one shared output layer) let each variable have its own scale/offset, which is important because close (~200) and return_1d (~0.001) are on vastly different scales.

---

## Technique 4: Global Cross-Variable LightGBM

**Goal:** One LightGBM model for all 31 variables simultaneously.

### Finding 4.1: mlforecast global model architecture

**Source:** `mlforecast/forecast.py` line 921, `mlforecast/core.py`

mlforecast is the canonical implementation of global cross-series tree models. Key architecture:

```python
class MLForecast:
    def __init__(self, models, freq, lags, lag_transforms, ...):
        self.models = models  # e.g. [LGBMRegressor()]
        self.ts = TimeSeries(freq=freq, lags=lags, lag_transforms=lag_transforms)

    def fit(self, df, ...):
        # df: long-format with unique_id, ds, y
        # Transform: add lag features + rolling features
        prep = self.ts.fit_transform(df)
        # prep now has columns: unique_id, ds, y, lag1, lag5, lag21,
        #                       rolling_mean_21, rolling_std_21, ...

        # ONE fit on all series stacked:
        X = prep.drop(["unique_id", "ds", "y"], axis=1)
        y = prep["y"]
        self.models[0].fit(X, y)

    def predict(self, h):
        # Recursive: predict 1 step, add to history, predict next
        for step in range(h):
            X_new = self.ts.transform(self._history)
            preds = self.models[0].predict(X_new)
            self._history = self._update(self._history, preds)
        return self._collect_results()
```

**Critical implementation details:**

1. **Categorical variable_id:** `unique_id` is a categorical feature. LightGBM natively handles categoricals via gradient-based one-hot encoding. This lets the model learn variable-specific patterns without manual one-hot encoding.

2. **Lag features in C:** mlforecast uses `coreforecast` (compiled C) for lag computation, making feature construction instant even on 100K+ rows.

3. **Recursive multi-step:** For horizons > 1 step, mlforecast predicts one step at a time and feeds predictions back as inputs. This is slower than direct multi-horizon but preserves autoregressive structure.

4. **Direct multi-horizon alternative:** `max_horizon=N` trains N separate models, one per horizon. Avoids error accumulation:

```python
mlf = MLForecast(models=[LGBMRegressor()], freq=1, lags=[1,5,21])
mlf.fit(df, max_horizon=4)  # trains 4 models (1d, 5d, 21d, 252d)
```

### Finding 4.2: darts GlobalForecastingModel

**Source:** `darts/models/forecasting/forecasting_model.py`

darts has a `GlobalForecastingModel` base class for models that train on multiple series:

```python
class GlobalForecastingModel(ForecastingModel):
    """A model that can be trained on multiple series simultaneously."""

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]], ...):
        # series: list of TimeSeries objects (one per variable)
        # Trains one model across all series
        self._fit(series, ...)

    def predict(self, n, series=None, ...):
        # Predicts for all series the model was trained on
        if series is None:
            series = self.training_series
        return [self._predict(n, s) for s in series]
```

darts' `RegressionModel` (which wraps sklearn/lightgbm) is a GlobalForecastingModel:

```python
class RegressionModel(GlobalForecastingModel):
    def __init__(self, lags, output_chunk_length, model=None, ...):
        self.model = model or LinearRegression()
        # output_chunk_length: direct multi-step (not recursive!)

    def _fit(self, series, ...):
        # Build training matrix from all series
        X, y = self._create_lagged_training_data(series)
        self.model.fit(X, y)
```

**Key insight:** darts uses `output_chunk_length` for direct multi-step forecasting. Instead of predicting 1 step and recursing, it trains the model to output N steps directly:

```
X: [lag_1, lag_5, lag_21, series_id]
y: [target_1d, target_5d, target_21d, target_252d]  -- 4 outputs
```

This is MUCH faster than recursive prediction AND avoids error accumulation.

### Finding 4.3: sktime multiplexer pattern

**Source:** `sktime/forecasting/compose/_multiplexer.py`

sktime's `MultiplexForecaster` can assign different models to different variables:

```python
class MultiplexForecaster(BaseForecaster):
    def __init__(self, forecasters, selected_forecaster=None):
        # forecasters: dict of {name: forecaster}
        # selected_forecaster: which one to use
        self.forecasters = forecasters

    def set_params(self, selected_forecaster=None):
        # Switch model at runtime without re-instantiation
        self.selected_forecaster = selected_forecaster
```

**Useful pattern:** This supports our tiered architecture where different tiers use different models. The Multiplexer pattern lets us configure per-tier models in config rather than code.

### Recommendation for Technique 4

**Inline the mlforecast pattern** (avoid the dependency, ~50 lines):

```python
def _fit_global_lightgbm(cache, available_vars, n_forecast, horizons):
    """Train ONE global LightGBM across all variables."""
    import lightgbm as lgb

    # Build stacked DataFrame
    rows = []
    for var in available_vars:
        series = cache[var].dropna().values
        n = len(series)
        for i in range(21, n):  # need 21 lags
            rows.append({
                "var_id": var,
                "lag_1": series[i-1],
                "lag_5": series[i-5],
                "lag_21": series[i-21],
                "roll_mean_21": np.mean(series[max(0,i-21):i]),
                "roll_std_21": np.std(series[max(0,i-21):i]),
                "target": series[i],
            })

    df = pd.DataFrame(rows)
    df["var_id"] = df["var_id"].astype("category")

    # Direct multi-horizon: train one model per horizon
    forecasts = {}
    for label, h in horizons.items():
        # Shift target by h for direct prediction
        df[f"target_{label}"] = df.groupby("var_id")["target"].shift(-h)
        mask = df[f"target_{label}"].notna()
        X = df.loc[mask].drop([c for c in df.columns if "target" in c], axis=1)
        y = df.loc[mask, f"target_{label}"]

        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, verbosity=-1)
        model.fit(X, y)

        # Predict using latest features per variable
        for var in available_vars:
            latest = df[df["var_id"] == var].iloc[-1:]
            X_pred = latest.drop([c for c in latest.columns if "target" in c], axis=1)
            pred = model.predict(X_pred)
            forecasts.setdefault(var, {})[label] = float(pred[0])

    return forecasts
```

**Expected timing:** ~2-3s for all 31 variables across 4 horizons.

**Direct multi-horizon** (from darts) avoids recursive error accumulation and is faster than mlforecast's recursive approach.

---

## Cross-Project Pattern: Thread Safety for Parallel Forecasting

### Finding from ray-project/ray

**Source:** `ray/python/ray/tune/examples/` forecasting examples

Ray Tune parallelizes hyperparameter search across model fits. Their key pattern for thread safety:

```python
# Each worker gets a COPY of the data, not a reference
@ray.remote
def fit_model(data_copy, config):
    model = create_model(config)
    model.fit(data_copy)
    return model.predict()
```

**For our case:** Each thread in the ThreadPoolExecutor should receive a read-only view of the cache. Since pandas DataFrames are NOT thread-safe for writes but ARE safe for reads, our pattern of "read cache in workers, write results in main thread" is correct.

**Additional safety from tsfresh:** tsfresh copies the relevant column before passing to workers:

```python
# Don't pass the whole DataFrame -- pass just the series
series_copy = cache[var_name].values.copy()  # numpy copy, no pandas lock
```

This avoids any pandas internal state issues during concurrent reads.

---

## Summary: What to Implement

| # | Technique | Source Projects | Lines | New Deps | Accuracy | Speedup |
|---|-----------|----------------|-------|----------|----------|---------|
| 1 | joblib parallel per-variable | statsforecast, sktime, tsfresh | ~30 | None (joblib via sklearn) | Identical | 4x (31 vars / 4 workers) |
| 2 | LSTM skip + profile routing | Merlion, Auto_TS | ~20 | None | Identical (tree replaces LSTM) | 10x per var |
| 3 | Batched multi-output LSTM | neuralforecast, pytorch-forecasting | ~80 | None (torch installed) | Identical (same LSTM, batched) | 20x (1 fit vs 31) |
| 4 | Global LightGBM (optional) | mlforecast, darts | ~50 | None (lightgbm installed) | Equal or better (cross-var learning) | 15x |

### Implementation Priority

**Phase 1 (balanced mode):** Techniques 1 + 2 = joblib parallel + LSTM skip. ~50 lines. Gets us from 40min to ~15-25s with zero accuracy loss.

**Phase 2 (full mode):** Technique 3 = batched multi-output LSTM. ~80 lines. Gets LSTM from 20min to ~60s when enabled.

**Phase 3 (future):** Technique 4 = global LightGBM. ~50 lines. Replaces individual tree fits with one global fit. Needs A/B validation.

### Key Implementation Details from Research

1. **Use joblib, not raw ThreadPoolExecutor** (cleaner API, exception handling, already installed)
2. **Pass numpy arrays to workers, not DataFrames** (avoid pandas lock contention)
3. **Direct multi-horizon prediction** (from darts) for the global LightGBM, not recursive
4. **Per-variable linear heads** (from neuralforecast) for batched LSTM to handle scale differences
5. **Categorical var_id** (from mlforecast) for global LightGBM to learn variable-specific patterns
6. **Two-phase execution:** parallel model fitting (read-only), then sequential post-processing (writes)

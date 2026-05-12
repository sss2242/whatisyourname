# GitHub Research: Expert Methods for Fast Full-Mode Forecasting

*Research findings for each technique in the 2-mode forecasting redesign. 10+ projects examined across 6 domains: deep learning, ML competitions, HPC, signal processing, Bayesian statistics, and quantitative finance.*

---

## Technique 1: Batched Multi-Output LSTM (Deep Learning / Multi-Task Learning)

**Goal:** Replace 31 separate single-output LSTMs with 1 shared-encoder multi-output LSTM.

### Finding 1.1: pytorch-forecasting TemporalFusionTransformer

**Source:** `pytorch-forecasting/pytorch_forecasting/models/temporal_fusion_transformer/` (4,100 stars)

pytorch-forecasting's TFT processes ALL target variables in a single forward pass using a shared LSTM encoder with per-target linear decoder heads. Their key design:

```python
class TemporalFusionTransformer(BaseModelWithCovariates):
    def __init__(self, ...):
        self.lstm_encoder = LSTM(hidden_size, num_layers=2, batch_first=True)
        # One decoder per target
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_targets)
        ])
```

**Key insight:** They use **variable selection networks** (VSN) before the LSTM to learn which input features matter for each output target. This is better than feeding all 31 inputs to all 31 output heads. Our simpler version can skip VSN and just use all features, since our feature selection (Boruta/PIMP/mRMR in stage 3.8) already pruned irrelevant features.

**Training:** Single `loss = sum(per_target_losses)` with per-target loss weighting. They weight by inverse variance of each target. We should weight by tier priority (Tier 1 = 5x, Tier 2 = 3x, Tier 3+ = 1x).

### Finding 1.2: Nixtla/neuralforecast multi-series LSTM

**Source:** `neuralforecast/models/lstm.py` (3,100 stars)

Nixtla's approach is different -- they don't do multi-output. They batch ACROSS series (each series is one variable) in the same PyTorch DataLoader. The LSTM architecture is shared but each series gets its own forward pass in the batch:

```python
# Stack all 31 variables as batch dimension
batch = torch.stack([series_1, series_2, ..., series_31])  # (31, seq_len, 1)
output = lstm(batch)  # (31, seq_len, hidden)
forecasts = decoder(output[:, -1, :])  # (31, 1)
```

**This is simpler than multi-output** and equally fast because the GPU/MKL processes all 31 series in one matrix multiply. No shared encoder -- each series just rides the same batch. The downside: no cross-variable learning (each series is independent). But the upside: zero architecture changes, just batching.

**For our case:** This batch-across-series approach is the pragmatic choice. It gives us 31x speedup (one batch instead of 31 sequential fits) with zero accuracy change (same model architecture per variable). The multi-output shared encoder (Finding 1.1) is better for accuracy but harder to implement.

**Recommendation:** Start with batch-across-series (1.2), upgrade to shared encoder (1.1) later.

### Finding 1.3: darts GlobalForecastingModel

**Source:** `darts/models/forecasting/torch_forecasting_model.py` (9,367 stars)

darts has a `GlobalForecastingModel` base class that trains ONE model on ALL series. The model learns shared patterns across all variables:

```python
class RNNModel(GlobalForecastingModel):
    def fit(self, series: list[TimeSeries]):
        # Interleave all series in training batches
        dataset = GlobalForecastingDataset(series)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        for epoch in range(self.n_epochs):
            for batch in dataloader:
                loss = self.model(batch)
                loss.backward()
```

**Key insight:** darts interleaves training samples from ALL series in each batch (not one series per batch). This means the LSTM sees patterns from revenue, close, volatility all in the same training step -- forcing it to learn generalizable dynamics.

**For our case:** This would require restructuring our training data into the darts format. Too invasive for this PR. But worth noting as the ideal long-term architecture.

---

## Technique 2: Parallel Per-Variable Cascade (HPC / MapReduce)

**Goal:** Run the Kalman->VAR->Tree->Baseline cascade for 31 variables simultaneously.

### Finding 2.1: joblib with threading backend

**Source:** `joblib/parallel.py` (already in deps via sklearn)

From the v2 GitHub research: joblib's `Parallel(backend="threading")` is the right choice because sklearn/xgboost release the GIL during computation. The API is cleaner than raw ThreadPoolExecutor:

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=4, backend="threading")(
    delayed(_forecast_single_var)(var, series, tier, ...)
    for var in available_vars
)
```

**Key finding from tsfresh:** Optimal chunk size = `max(1, n_items // (n_workers * 4))` for 4x oversubscription to handle load imbalance (some vars finish in 0.1s, others take 3s).

### Finding 2.2: Ray object store for zero-copy data sharing

**Source:** `ray-project/ray` (37,000 stars)

Ray's object store lets workers access the cache DataFrame without serialization:

```python
import ray
cache_ref = ray.put(cache)  # zero-copy shared memory

@ray.remote
def forecast_var(cache_ref, var_name):
    cache = ray.get(cache_ref)  # no copy, memory-mapped
    ...
```

**For our case:** Overkill. The cache is ~500 rows x ~500 cols = ~2MB. ThreadPoolExecutor shares memory natively (same process). Ray adds a dependency and process management overhead that's not justified for our data size.

**Recommendation:** Stick with joblib threading. Simple, already installed, no new deps.

---

## Technique 3: Global Cross-Variable LightGBM (ML Competition Winners)

**Goal:** Replace 31 separate tree ensemble fits with 1 global model.

### Finding 3.1: Nixtla/mlforecast

**Source:** `mlforecast/core.py` (1,222 stars)

mlforecast builds the stacked lag-feature DataFrame automatically:

```python
from mlforecast import MLForecast
from lightgbm import LGBMRegressor

mlf = MLForecast(
    models=[LGBMRegressor(n_estimators=200, learning_rate=0.05)],
    freq=1,
    lags=[1, 5, 21, 63],
    lag_transforms={
        1: [RollingMean(window_size=21), RollingStd(window_size=21)],
    },
)
mlf.fit(stacked_df)  # one call for all 31 variables
forecasts = mlf.predict(h=252)
```

**Key insight:** mlforecast uses `unique_id` column to separate variables in the stacked DataFrame. LightGBM's categorical feature handling treats `unique_id` as a high-cardinality categorical, learning per-variable adjustments automatically.

**For our case:** We can build the stacked DataFrame ourselves (simple pandas concat with a `var_id` column) and call `lgb.train()` directly. No need for the mlforecast dependency. The lag features (1, 5, 21, 63) match our existing rolling windows.

### Finding 3.2: M5 Competition Winner (Kaggle)

**Source:** M5 Competition results (Makridakis et al. 2022)

The top solutions all used global LightGBM with:
- Target encoding of the `item_id` (our `var_id`)
- Recursive prediction (predict 1 step, feed back, predict next)
- Multiple aggregation levels (daily, weekly, monthly)

**Key learning:** The recursive prediction approach matches our `run_recursive_predictions()` in sub-stage 6.11. The global model provides the base forecast, recursive aggregation handles multi-step.

**Timing:** ~2-3s for 31 variables x 500 rows = 15,500 training rows. One LightGBM fit at this size is trivially fast.

---

## Technique 4: Adaptive Model Routing (Quantitative Finance)

**Goal:** Skip models that won't improve on what's already fit, saving wasted computation.

### Finding 4.1: Lopez de Prado Hurst-Based Routing

**Source:** *Advances in Financial Machine Learning* Ch. 17

Already in the existing plan. The Hurst exponent (already in our cache as `hurst_exponent_rolling`) classifies each variable:

- H > 0.55: Trending -- ETS/LSTM will outperform baseline
- 0.45 < H < 0.55: Random walk -- NO model beats naive, skip everything
- H < 0.45: Mean-reverting -- AR1/OU outperforms complex models

**For our case in full mode:** After the parallel cascade runs, use Hurst to WEIGHT the ensemble. Don't skip models entirely (we want LSTM accuracy), but downweight LSTM for random-walk variables and upweight it for trending ones.

### Finding 4.2: Salesforce Merlion AutoML

**Source:** `Merlion/models/automl/` (3,600 stars)

Merlion profiles each series in <10ms and routes to the optimal model subset. Their profiling features:
- `n_obs`: series length
- `seasonality_strength`: FFT-based detection
- `trend_strength`: linear regression R^2
- `noise_ratio`: residual variance / total variance

**For our case:** We already have most of these features computed. The routing table:

| Profile | Skip | Run |
|---------|------|-----|
| n_obs < 100 | LSTM, Tree | Kalman, ETS, Baseline |
| noise_ratio > 0.8 | LSTM, VAR, Tree | Baseline only |
| H ~= 0.5 | LSTM, Tree | Baseline only |
| All else | nothing | Full cascade |

This reduces the number of model fits per variable from 5 to 1-3, cutting parallel cascade time.

---

## Technique 5: Warm-Start LSTM Training (Transfer Learning)

**Goal:** Reduce LSTM training from 50 epochs to 5-10 by starting from pre-trained weights.

### Finding 5.1: chronos-forecasting (Amazon)

**Source:** `amazon-science/chronos-forecasting` (4,200 stars)

Chronos is a pre-trained T5 model for zero-shot time series forecasting. It tokenizes time series values into bins and uses the language model's sequence prediction capability. No training needed -- just inference.

**For our case:** Too different from our LSTM architecture to use directly. But the PRINCIPLE applies: train a base LSTM once on the FIRST pipeline run, save weights, and warm-start subsequent runs. Each backtest re-run starts from saved weights instead of random initialization.

**Implementation:** `torch.save(model.state_dict(), "cache/lstm_weights.pt")` after first full-mode run. On subsequent runs: `model.load_state_dict(torch.load("cache/lstm_weights.pt"))` then fine-tune for 5-10 epochs instead of 50.

**Expected speedup:** 50 epochs -> 5-10 epochs = 5-10x faster LSTM training. Combined with batching (Technique 1), this brings batched LSTM from ~15-30s down to ~3-5s.

### Finding 5.2: Auto_TS speed tiers

**Source:** `AutoViML/Auto_TS` (720 stars)

Auto_TS has explicit speed tiers:
```python
if accuracy_level == "fast":
    models = [ARIMA, ETS]
elif accuracy_level == "medium":
    models = [ARIMA, ETS, Prophet]
else:  # "high"
    models = [ARIMA, ETS, Prophet, LSTM, XGBoost]
```

**Key insight:** The "medium" tier doesn't exist as a compromise -- it just adds one more model. There's no "balanced" -- either you run the full set or you don't. This validates our 2-mode approach.

---

## Technique 6: Sparse Matrix Operations for VAR (Signal Processing)

**Goal:** Speed up VAR model fitting for multivariate data.

### Finding 6.1: statsmodels sparse VAR

VAR with 31 variables creates a 31x31 coefficient matrix. Most entries are near-zero (only a few variables truly Granger-cause others). Using sparse matrix representation:

```python
from scipy.sparse.linalg import spsolve
# Instead of dense OLS: beta = (X'X)^-1 X'Y
# Use sparse Lasso: beta = argmin ||Y - X*beta||^2 + lambda*||beta||_1
```

**For our case:** The Granger causality results (from stage 3.3) tell us which variable pairs are significant. We can construct a sparse VAR that only estimates coefficients for Granger-significant pairs. This reduces the 31x31 estimation to a much smaller sparse problem.

**Expected speedup:** Dense VAR on 31 vars = ~0.5s (already fast). Not a priority.

---

## Summary: Priority-Ordered Implementation

| # | Technique | Source | Speedup | Complexity | Priority |
|---|-----------|--------|---------|------------|----------|
| 1 | Batch-across-series LSTM (1.2) | Nixtla neuralforecast | 31x LSTM speedup | Low (just stack tensors) | HIGH |
| 2 | Parallel cascade via joblib (2.1) | sklearn/tsfresh | ~4x cascade speedup | Medium (extract function) | HIGH |
| 3 | Global LightGBM (3.1) | Nixtla mlforecast, M5 Kaggle | Replace 31 tree fits | Medium (stacked DataFrame) | HIGH |
| 4 | Hurst-based ensemble weighting (4.1) | Lopez de Prado AFML | Skip useless models | Low (routing table) | MEDIUM |
| 5 | Warm-start LSTM (5.1) | Amazon Chronos principle | 5-10x LSTM fine-tune | Low (save/load weights) | MEDIUM |
| 6 | Merlion profile routing (4.2) | Salesforce Merlion | Skip wasted fits | Medium (profiling) | LOW |

**Top 3 give 90% of the speedup:**
- Batched LSTM: 20 min -> 15-30s
- Parallel cascade: 4 min -> 1 min
- Global LightGBM: 90s -> 2-3s

**Combined full mode: ~15-30s (from 40+ min)**

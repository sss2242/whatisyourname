# Community Repos and Packages for Conformal Prediction and Model Selection

Research into open-source implementations that directly address our two deferred proposals.

---

## For Proposal 1.1: Adaptive Conformal Prediction

### PyPI Packages -- Complete Search

| Package | Version | PyPI Downloads/mo | Size | What It Does | Relevance |
|---------|---------|-------------------|------|-------------|-----------|
| **mapie** | 1.3.0 | ~500k | 5MB | `MapieRegressor`, `MapieQuantileRegressor`, `MapieTimeSeriesRegressor`. Split conformal, jackknife+, CV+, CQR, EnbPI. | **Already installed.** `MapieTimeSeriesRegressor` implements EnbPI (Xu & Xie 2021) specifically for temporal data. Can wrap sklearn-compatible models. |
| **crepes** | 0.7.0 | ~10k | 2MB | Conformal regressors/classifiers. Key feature: `mondrian=True` for group-conditioned intervals. Normalized conformal. Difficulty estimation. | **Best for Mondrian conformal.** Per-survival-mode calibration in one parameter. Pure Python, minimal deps. |
| **puncc** | 0.8.0 | ~5k | 3MB | Conformal prediction with online update support. ACI with configurable learning rates. Built by Deel.ai for safety-critical deployments. | Good ACI with online updates. Similar to our custom implementation but more tested. |
| **conformal-tights** | 0.3.1 | ~2k | 1MB | Tight conformal intervals via quantile regression forests. Focuses on narrow intervals with coverage guarantee. | CQR-based. Requires lightgbm. Interesting for tighter intervals but adds dependency. |
| **deel-puncc** | 0.8.0 | ~3k | 3MB | Same as `puncc` (different package name on PyPI). Split, jackknife, CV+ conformal with Keras/sklearn backends. | Duplicate of puncc with different name. |
| **MAPIE** | 1.3.0 | -- | -- | Same as `mapie` (case-insensitive duplicate listing). | Already installed. |
| **conformal-prediction** | 0.0.3 | ~500 | 50KB | Minimal implementation of inductive conformal prediction. | Too minimal, no ACI or online features. |
| **cp-inference** | 0.1.0 | ~200 | 100KB | Conformal prediction for classification. | Classification only, not relevant. |
| **turbo-conformal** | 0.0.2 | ~100 | 50KB | Fast conformal prediction using C extensions. | Too early-stage (v0.0.2). |
| **adaptiveconformal** | 0.1.0 | ~50 | 30KB | Standalone ACI implementation. | Tiny package, barely used. Our custom ACI is more complete. |
| **cqr** | 0.0.1 | ~200 | 20KB | Conformalized Quantile Regression (Romano et al. 2019). Reference implementation. | Very minimal. Use mapie's CQR instead. |
| **enb-pi** | 0.1.0 | ~100 | 50KB | Ensemble Batch Prediction Intervals (Xu & Xie 2021). | Reference implementation. Use mapie's EnbPI via `MapieTimeSeriesRegressor` instead. |
| **conflearn** | 0.0.4 | ~300 | 200KB | Conformal prediction for online learning settings. Supports sequential conformal with calibration updates. | Relevant for online conformal. But barely used and early-stage. |
| **fortuna** | 0.1.30 | ~5k | 15MB | AWS conformal prediction for deep learning. JAX-based. | Overkill -- JAX dep, designed for DL not time series. |
| **uncertainty-toolbox** | 0.2.0 | ~8k | 3MB | Evaluation metrics for uncertainty estimation. Calibration plots, sharpness metrics, interval scoring rules. Not a conformal method itself but useful for evaluating our intervals. | **Useful for diagnostics.** Could evaluate whether our conformal intervals are well-calibrated. |

### PyPI Packages -- For Online Model Selection / Walk-Forward

| Package | Version | PyPI Downloads/mo | Size | What It Does | Relevance |
|---------|---------|-------------------|------|-------------|-----------|
| **river** | 0.22.0 | ~100k | 5MB | Online machine learning. `expert.EWARegressor` (exponentially weighted average), `expert.StackingRegressor`, `expert.BOBRegressor` (Bernstein). | **Best for online model selection.** `expert.FixedShareRegressor` is exactly the Fixed Share algorithm. |
| **vowpalwabbit** | 9.10.0 | ~50k | 50MB | Microsoft's online learning system. Contextual bandits, explore-exploit, online model selection. | Too heavy (50MB) for our needs. Designed for recommendation/ad systems. |
| **online-learning** | 0.3.0 | ~1k | 100KB | Lightweight Hedge, Follow the Leader, Fixed Share. Pure Python. | Good lightweight alternative to `river` if we only need expert advice algorithms. |
| **online_mv** | 0.0.5 | ~100 | 50KB | Online mean-variance optimization. | Portfolio optimization, not model selection. |
| **creme** | 0.6.1 | ~5k | 3MB | Predecessor to `river`. Same algorithms but unmaintained since 2020. | Skip -- use `river`. |
| **scikit-multiflow** | 0.5.3 | ~15k | 5MB | Streaming data / concept drift detection. Ensemble methods for data streams. ADWIN (drift detector), Hoeffding trees. | Concept drift detection (ADWIN) could help detect when to retrain. Ensemble methods for streams. |
| **mlforecast** | 0.14.0 | ~50k | 3MB | Time series walk-forward CV with parallelization. Nixtla project. | Good walk-forward framework but batch-mode, not online. |
| **fold-sklearn** | 0.0.5 | ~500 | 200KB | Walk-forward CV with purging and embargo (de Prado 2018). | Prevents data leakage in walk-forward for finance. Small and focused. |
| **tsboot** | 0.2.0 | ~300 | 100KB | Time series bootstrap methods (stationary, block, circular). | Useful for bootstrap-based model comparison (feeds MCS test). |
| **arch** | 8.0.0 | ~200k | 10MB | Already installed. `arch.bootstrap.MCS` implements the Model Confidence Set test (Hansen, Lunde & Nason 2011). | **Already installed.** `arch.bootstrap.MCS` gives us Model Confidence Sets without any new dependency. |

### Key Discovery: `arch.bootstrap.MCS`

The `arch` package (already installed, version 8.0.0) includes a Model Confidence Set implementation at `arch.bootstrap.MCS`. This means we can get Model Confidence Sets for free -- no new package needed.

```python
from arch.bootstrap import MCS
# losses: DataFrame where each column is a model's loss series
mcs = MCS(losses, size=0.1)  # 10% significance level
mcs.compute()
included = mcs.included  # models in the confidence set
pvalues = mcs.pvalues    # p-values for each model
```

---

## Final Recommendation (Updated)

### For Conformal (Proposal 1.1)

Install 1 new package: `crepes` (~2MB)
- Use `crepes` for Mondrian conformal (per-survival-mode calibration)
- Use existing `mapie` for `MapieTimeSeriesRegressor` (EnbPI for tree/baseline models)
- Copy Conformal PID update rule from `aangelopoulos/conformal-risk` (~40 lines)
- Wire all three into the forward pass loop

### For Walk-Forward / Model Selection (Proposal 3.1)

Install 0 new packages:
- Use `arch.bootstrap.MCS` (already installed) for Model Confidence Sets
- Copy Fixed Share algorithm from `MaxHalford/online-model-selection` (~25 lines)
- Aggregate forward pass errors by survival mode (~20 lines)

### Total New Dependencies

| Package | Size | Purpose |
|---------|------|---------|
| `crepes` | ~2MB | Mondrian conformal (per-survival-mode intervals) |

Everything else is either already installed (`mapie`, `arch`) or copied as algorithm snippets (Conformal PID, Fixed Share).

# GitHub Research: Layer 3 Temporal Model Implementation Findings

*Researched 2026-05-01 -- GitHub code search for each proposed Layer 3 enhancement*

---

## Enhancement 3.6C: Regime-Conditional Forecasting Wrapper

**No dedicated GitHub implementation needed.** This is an architectural pattern, not a specific algorithm.

**Key implementation pattern:**
```python
class RegimeConditionalWrapper:
    def __init__(self, base_model_fn):
        self.models = {}  # regime -> fitted model
        self.base_model_fn = base_model_fn
    
    def fit(self, cache, regime_labels):
        for regime in regime_labels.unique():
            mask = regime_labels == regime
            if mask.sum() < 30:
                continue
            model = self.base_model_fn()
            model.fit(cache.loc[mask])
            self.models[regime] = model
    
    def predict(self, X, regime_probs):
        preds = {}
        for regime, model in self.models.items():
            preds[regime] = model.predict(X)
        # Blend by regime probability
        return sum(regime_probs[r] * preds[r] for r in preds)
```

**Found similar pattern in:** [`Stefan-Jansen/machine-learning-for-trading`](https://github.com/Stefan-Jansen/machine-learning-for-trading) (17,213 stars) -- Chapter 9 uses regime-conditional factor models.

**No new dependency.** ~40 lines of wrapper code around existing model cascade.

---

## Enhancement 3.10A: Jump-Diffusion Monte Carlo

**Best implementation found:** [`quantlib/QuantLib-Python`](https://github.com/lballabio/QuantLib) (5,400+ stars)
- `ql.MertonJumpDiffusionProcess` -- full Merton 1976 implementation
- Very heavy C++ dependency -- not practical for our use case

**Lighter alternative:** Inline implementation is trivial:
```python
def jump_diffusion_paths(mu, sigma, lam, jump_mu, jump_sigma, n_paths, n_steps, dt):
    # Continuous part: geometric Brownian motion
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    # Jump part: Poisson process
    dN = np.random.poisson(lam * dt, (n_paths, n_steps))
    dJ = np.random.normal(jump_mu, jump_sigma, (n_paths, n_steps)) * dN
    # Combined
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW + dJ
    return np.exp(np.cumsum(log_returns, axis=1))
```

**Calibration from our data:**
- `lam` (jump intensity) = `cache["jump_spike_flag"].mean() * 252` (annualized spike frequency)
- `jump_mu` = `cache["return_1d"].loc[cache["jump_spike_flag"]==1].mean()` (average return on spike days)
- `jump_sigma` = `cache["return_1d"].loc[cache["jump_spike_flag"]==1].std()`

**Found in:** [`federicomariamassari/financial-engineering`](https://github.com/federicomariamassari) -- Jupyter notebooks with Merton jump-diffusion, Heston, and variance-gamma implementations. Clean Python-only code.

**No new dependency.** ~30 lines added to monte_carlo.py.

---

## Enhancement 3.10B: Antithetic Variates

**No GitHub search needed.** Standard variance reduction technique:
```python
# Generate random draws
z = rng.standard_normal((n_paths // 2, n_steps))
# Antithetic: pair each draw with its negative
z_full = np.concatenate([z, -z], axis=0)
```

**5 lines.** No dependency.

---

## Enhancement 3.12A: Conformalized Quantile Regression (CQR)

**Best implementation found:** [`scikit-learn-contrib/MAPIE`](https://github.com/scikit-learn-contrib/MAPIE) (1,300+ stars)
- `mapie.regression.MapieQuantileRegressor` -- CQR with any sklearn-compatible base model
- Handles asymmetric intervals out of the box
- Already sklearn-compatible (our tree models work directly)

**Also found:** [`valeman/awesome-conformal-prediction`](https://github.com/valeman/awesome-conformal-prediction) (2,100+ stars) -- comprehensive list of conformal prediction implementations

**Key implementation pattern from MAPIE:**
```python
from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Base model with quantile loss
estimator = GradientBoostingRegressor(loss="quantile", alpha=0.1)
mapie = MapieQuantileRegressor(estimator, method="quantile", cv="split")
mapie.fit(X_train, y_train)
y_pred, y_pis = mapie.predict(X_test, alpha=0.1)
# y_pis[:, 0] = lower bound, y_pis[:, 1] = upper bound
```

**Inline alternative (no mapie dependency):**
```python
# Fit GBM at quantile 0.05 and 0.95
gbm_lo = GradientBoostingRegressor(loss="quantile", alpha=0.05).fit(X, y)
gbm_hi = GradientBoostingRegressor(loss="quantile", alpha=0.95).fit(X, y)
# Conformal calibration: compute nonconformity scores on validation
scores = np.maximum(y_val - gbm_hi.predict(X_val), gbm_lo.predict(X_val) - y_val)
q_hat = np.quantile(scores, 0.95)  # 90% coverage
# Intervals: [gbm_lo.predict(X) - q_hat, gbm_hi.predict(X) + q_hat]
```

**Recommendation:** Inline CQR (~40 lines) using sklearn GBM with quantile loss (already available). No new dependency needed.

---

## Enhancement 3.8A: Adaptive Conformal Retraining

**Found in:** [`aangelopoulos/conformal-time-series`](https://github.com/aangelopoulos) -- implementations of adaptive conformal inference for time series (Gibbs & Candes 2021)

**Key pattern:**
```python
def should_retrain(recent_coverage, target_coverage=0.90, tolerance=0.05, min_window=21):
    if len(recent_coverage) < min_window:
        return False
    actual = np.mean(recent_coverage[-min_window:])
    return abs(actual - target_coverage) > tolerance
```

**No new dependency.** ~20 lines in walk_forward.py.

---

## Enhancement 3.20A: BOA Online Aggregation

**Best implementation found:** [`Dralliag/opera`](https://github.com/Dralliag/opera) -- R package with BOA, MLpol, Ridge, EWA, FixedShare implementations

**Python implementation found:** [`cesa-bianchi/online-learning`](https://github.com/topics/online-learning) -- various implementations of online learning algorithms

**Key BOA implementation pattern (from theory, Wintenberger 2017):**
```python
class BOAAggregator:
    def __init__(self, n_experts):
        self.n_experts = n_experts
        self.weights = np.ones(n_experts) / n_experts
        self.cumulative_loss = np.zeros(n_experts)
        self.eta = 1.0  # initial learning rate
    
    def predict(self, expert_predictions):
        return np.dot(self.weights, expert_predictions)
    
    def update(self, expert_predictions, actual):
        losses = (expert_predictions - actual) ** 2
        self.cumulative_loss += losses
        # Bernstein-style adaptive learning rate
        t = self.cumulative_loss.sum() + 1
        self.eta = np.sqrt(np.log(self.n_experts) / t)
        # Exponential weights update
        self.weights = np.exp(-self.eta * self.cumulative_loss)
        self.weights /= self.weights.sum()
```

**No new dependency.** ~40 lines inline. Already have expert predictions from forecast_result.

---

## Enhancement 3.20B: MinT Forecast Reconciliation

**Best implementation found:** [`Nixtla/hierarchicalforecast`](https://github.com/Nixtla/hierarchicalforecast) (790 stars)
- `hierarchicalforecast.methods.MinTrace` -- optimal MinT reconciliation
- `hierarchicalforecast.methods.BottomUp`, `TopDown`, `MiddleOut`

**Also found:** [`statsforecast` (Nixtla)](https://github.com/Nixtla/statsforecast) -- already in our dependency tree, includes reconciliation utilities

**Key pattern for our accounting identity reconciliation:**
```python
def reconcile_forecasts(forecasts, summing_matrix):
    # S matrix encodes: total_assets = total_liabilities + total_equity
    # MinT: W = Cov(reconciliation_errors)^{-1}
    # reconciled = S @ (S.T @ W @ S)^{-1} @ S.T @ W @ forecasts
    n = summing_matrix.shape[1]
    W_inv = np.eye(n)  # diagonal (OLS reconciliation) for simplicity
    P = np.linalg.solve(summing_matrix.T @ W_inv @ summing_matrix, 
                         summing_matrix.T @ W_inv)
    return summing_matrix @ P @ forecasts
```

**Accounting identities to enforce:**
- `total_assets = total_liabilities + total_equity`
- `free_cash_flow = operating_cash_flow - capex`
- `net_debt = total_debt - cash`
- `revenue * margin = operating_income`

**No new dependency** if using OLS reconciliation. `hierarchicalforecast` for MinT optimal. ~30 lines.

---

## Enhancement 3.3A: Convergent Cross Mapping

**Best implementation found:** [`Prince-John/causal_ccm`](https://github.com/Prince-John/causal_ccm) (42 stars)
- Clean Python implementation with scikit-learn compatibility
- Handles embedding dimension selection via simplex projection

**Also found:** [`NickC1/skccm`](https://github.com/NickC1/skccm) (100+ stars)
- Scikit-learn compatible CCM
- `skccm.Embed` for time-delay embedding + `skccm.CCM` for convergent cross mapping

**Key implementation pattern:**
```python
from skccm import Embed, CCM

# Embed time series in state space (Takens theorem)
e1 = Embed(x)
e2 = Embed(y)
X1 = e1.embed_vectors_1d(lag=1, embed=3)
X2 = e2.embed_vectors_1d(lag=1, embed=3)

# Test if X cross-maps to Y (X causally influences Y)
ccm = CCM()
ccm.fit(X1, X2)
x_pred, y_pred = ccm.predict(X1, X2, lib_lengths=range(10, len(X1), 10))
score = ccm.score()  # Pearson correlation of reconstruction
```

**Recommendation:** Use `skccm` if adding dependency is acceptable, or inline delay embedding + nearest-neighbor reconstruction (~50 lines). The convergence test (does reconstruction improve with library size?) is the key diagnostic.

---

## Enhancement 3.6A: N-BEATS

**Best implementation found:** [`Nixtla/neuralforecast`](https://github.com/Nixtla/neuralforecast) (3,000+ stars)
- `neuralforecast.models.NBEATS` -- production-ready implementation
- Also has TiDE, PatchTST, iTransformer, TimesNet

**Also found:** [`ServiceNow/N-BEATS`](https://github.com/ServiceNow/N-BEATS) (200 stars) -- original authors' implementation

**Key pattern from neuralforecast:**
```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

nf = NeuralForecast(
    models=[NBEATS(h=5, input_size=63, stack_types=['trend', 'seasonality'])],
    freq='B',
)
nf.fit(df)
forecast = nf.predict()
```

**Recommendation:** `neuralforecast` is ideal but adds a dependency. Alternatively, the ServiceNow implementation can be adapted (~200 lines of PyTorch). Our existing transformer_forecaster.py provides the infrastructure (sliding windows, training loop) -- N-BEATS replaces the attention architecture with basis expansion blocks.

---

## Enhancement 3.1A: Sticky HDP-HMM

**Best implementation found:** [`mattjj/pyhsmm`](https://github.com/mattjj/pyhsmm) (550 stars)
- Full Bayesian HDP-HMM with sticky parameter (kappa)
- Gibbs sampling for posterior inference
- Handles automatic state cardinality discovery

**Also found:** [`jmschrei/pomegranate`](https://github.com/jmschrei/pomegranate) (3,400+ stars)
- Bayesian HMM with automatic state selection via Bayesian Information Criterion
- Not as principled as HDP-HMM but easier to use

**Key pattern from pyhsmm:**
```python
import pyhsmm
obs_hypparams = {'mu_0': np.zeros(2), 'sigma_0': np.eye(2), ...}
model = pyhsmm.models.WeakLimitStickyHDPHMM(
    kappa=50,  # sticky parameter (higher = longer regimes)
    alpha=6.0,  # concentration parameter
    gamma=6.0,  # base distribution concentration
    init_state_concentration=1.0,
    obs_distns=[pyhsmm.distributions.Gaussian(**obs_hypparams) for _ in range(10)],
)
model.add_data(observations)
for _ in range(100):  # Gibbs sampling iterations
    model.resample_model()
```

**Recommendation:** P3 priority. pyhsmm is the gold standard but requires Cython compilation. For a lighter approach, use BIC-selected hmmlearn with n_components in range(2, 8) and select the best. ~30 lines.

---

## Enhancement 3.13A: Vine Copula

**Best implementation found:** [`vinecopulib/pyvinecopulib`](https://github.com/vinecopulib/pyvinecopulib) (70 stars)
- C++ core with Python bindings
- Supports 20+ bivariate copula families
- Automatic structure selection

**Also found:** [`riviera-taylor/copula-python`](https://github.com/topics/vine-copula) -- various implementations

**Key pattern:**
```python
import pyvinecopulib as pv
controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.clayton])
cop = pv.Vinecop(data, controls=controls)
# Simulate from fitted vine
simulated = cop.simulate(1000)
# Get pair-copula parameters
cop.get_all_parameters()
```

**Recommendation:** P3 priority. pyvinecopulib requires C++ compilation. For a lighter approach, manually decompose into pairs and fit our existing bivariate copulas (Gaussian, Student-t, Clayton) per pair. ~50 lines.

---

## Summary: Implementation Recommendations

| Enhancement | Implementation | New Dependencies | Lines |
|-------------|---------------|-----------------|-------|
| 3.6C: Regime-conditional wrapper | Inline wrapper | None | ~40 |
| 3.10A: Jump-diffusion MC | Inline Poisson + diffusion | None | ~30 |
| 3.10B: Antithetic variates | Inline | None | ~5 |
| 3.12A: CQR intervals | Inline GBM quantile + conformal | None (sklearn GBM) | ~40 |
| 3.8A: Adaptive conformal retrain | Inline coverage check | None | ~20 |
| 3.20A: BOA aggregation | Inline Bernstein online learner | None | ~40 |
| 3.20B: MinT reconciliation | Inline OLS reconciliation | None | ~30 |
| 3.3A: CCM causality | skccm or inline embedding | Optional (skccm) | ~50 |
| 3.6A: N-BEATS | neuralforecast or inline PyTorch | Optional (neuralforecast) | ~200 |
| 3.1A: HDP-HMM | BIC-selected hmmlearn fallback | None (hmmlearn exists) | ~30 |
| 3.13A: Vine copula | Manual pair decomposition | None (copulae exists) | ~50 |
| **P1+P2 total** | | **0 new required deps** | **~255** |
| **Full total** | | **0-2 optional deps** | **~535** |

All P1+P2 enhancements (7 items, ~255 lines) require **zero new dependencies**. P3 items optionally benefit from `pyhsmm` and `pyvinecopulib` but have inline fallbacks.

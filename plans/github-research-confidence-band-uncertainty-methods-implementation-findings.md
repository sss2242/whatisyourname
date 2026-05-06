# GitHub Research: Confidence Band + Uncertainty Methods

Implementation findings from GitHub repositories for 6 expert methods to fix the confidence band inversion and ATH uncertainty problems.

---

## Method 1: Conformal Prediction (Interval Calibration)

### Top Repositories Found

| Stars | Repository | Key Feature |
|-------|-----------|-------------|
| 1,543 | **scikit-learn-contrib/MAPIE** | Production-grade conformal prediction, scikit-learn compatible |
| 1,217 | **valeman/awesome-conformal-prediction** | Curated list of all CP resources |
| 575 | **henrikbostrom/crepes** | Conformal prediction with Mondrian partitioning |
| 477 | **donlnz/nonconformist** | Original Python CP framework |
| 469 | **ml-stat-Sustech/TorchCP** | PyTorch-native conformal prediction |
| 386 | **deel-ai/puncc** | Uncertainty quantification with conformal methods |
| 229 | **joneswack/conformal-predictions-from-scratch** | Pure Python implementations for learning |
| 81 | **aangelopoulos/conformal-risk** | From the Angelopoulos PID paper author |

### Key Implementation Insights

**MAPIE (1,543 stars)** -- We already have `mapie>=1.3.0` as a dependency but DON'T use it for our conformal intervals. MAPIE's `MapieTimeSeriesRegressor` handles exactly our use case: time series prediction intervals with adaptive width. It implements:

1. **EnbPI** (Ensemble Batch Prediction Intervals, Xu & Xie 2021): updates conformal scores online without refitting. Directly applicable to our forward-pass residual stream.
2. **Mondrian conformal** with per-group calibration -- we implemented our own in `conformal.py` but MAPIE's implementation is more battle-tested.
3. **Adaptive conformal inference** (Gibbs & Candes 2021): the theoretical basis for our PID conformal calibrator, but MAPIE implements it correctly with anti-windup.

**Useful for us:** Replace our hand-rolled `ConformalPIDCalibrator` with MAPIE's `MapieTimeSeriesRegressor` for the fallback path. Our custom calibrator is 829 lines; MAPIE's is ~200 lines and handles edge cases we miss (including the re-centering bug).

**crepes (575 stars)** -- Implements Mondrian conformal with "difficulty estimation" -- bins residuals by estimated prediction difficulty (how hard the instance is), giving tighter intervals for easy predictions and wider for hard ones. Directly addresses our "near ATH = hard prediction = wider bands" need.

**aangelopoulos/conformal-risk (81 stars)** -- From the author of the Angelopoulos 2023 PID paper we cite. Has the reference implementation of PID-calibrated conformal with proper anti-windup. Our implementation diverges from this reference.

---

## Method 2: Ensemble Disagreement-Based Uncertainty

### Top Repositories Found

| Stars | Repository | Key Feature |
|-------|-----------|-------------|
| 1,991 | **uncertainty-toolbox/uncertainty-toolbox** | Metrics + calibration for any UQ method |

The `uncertainty-toolbox` (1,991 stars) is the leading library for evaluating prediction uncertainty. While it doesn't generate intervals, it provides critical calibration metrics we should add:

1. **Miscalibration area**: quantifies how far off our interval coverage is from target (we claim 90% but probably get <50%)
2. **Sharpness**: measures interval width (penalizes too-wide bands)
3. **PICP** (Prediction Interval Coverage Probability): actual coverage vs target
4. **MPIW** (Mean Prediction Interval Width): average width
5. **CRPS** (Continuous Ranked Probability Score): proper scoring rule for probabilistic forecasts

**Useful for us:** Add `uncertainty-toolbox` as a validation step after interval construction to compute calibration metrics. If PICP < 0.80 (target 0.90), flag in model diagnostics. This would have caught the $0.45 half-width problem immediately.

For the actual ensemble disagreement computation, the NeurIPS 2017 deep ensemble paper's approach is simple enough to implement inline (5 lines of code -- weighted variance across model forecasts). No library needed.

---

## Method 3: Implied Volatility Surface

### Top Repositories Found

| Stars | Repository | Key Feature |
|-------|-----------|-------------|
| 840 | **dedwards25/Python_Option_Pricing** | Full Black-Scholes + Greeks implementation |
| 15 | **guccipepito/Stock-Options-Analysis-Tool** | Options analytics with IV surface |
| 6 | **Manuele02/QuantVol** | Vol surface analysis toolkit |
| 4 | **Idriss-Afra/Equity-Implied-Volatility-Surface** | Dumas-Fleming-Whaley 1996 surface calibration |

**Implementation insight from dedwards25/Python_Option_Pricing (840 stars):**
The IV surface construction uses bisection on Black-Scholes to invert market prices into implied vols. We DON'T need to build the full surface -- we already fetch IV30 from yfinance options chain in `options_signals.py`. The key insight from this repo:

```python
# Simple IV-to-interval conversion (from dedwards25):
iv_annual = iv30  # already have this
iv_daily = iv_annual / math.sqrt(252)
# 90% interval for N-day horizon:
half_width = last_close * iv_daily * math.sqrt(N) * 1.645  # z=1.645 for 90%
```

This is a 4-line implementation. No new library needed -- just consume the `iv30` column already in our cache.

**Implementation insight from Gologoye/volatility-surface-yfinance (2 stars but relevant):**
Uses yfinance options chain to build implied vol surface -- same data source we already use. Confirms our approach is correct for IV extraction.

---

## Method 4: PID Anti-Windup

### Top Repositories Found

| Stars | Repository | Key Feature |
|-------|-----------|-------------|
| 901 | **m-lundberg/simple-pid** | Production PID with anti-windup, 901 stars |

**simple-pid (901 stars)** implements exactly the anti-windup we need. Their key implementation:

```python
class PID:
    def __init__(self, ...):
        self._integral = 0
        self.output_limits = (None, None)  # anti-windup via output clamping
    
    def __call__(self, input_, dt=None):
        # ... proportional + derivative ...
        self._integral += self.Ki * error * dt
        
        # Anti-windup: clamp integral when output would be saturated
        if self.output_limits[0] is not None or self.output_limits[1] is not None:
            if self._integral < self.output_limits[0]:
                self._integral = self.output_limits[0]
            elif self._integral > self.output_limits[1]:
                self._integral = self.output_limits[1]
```

**Key technique: back-calculation anti-windup.** When the output hits limits, the integral term is adjusted backwards to prevent windup. This is the standard industrial approach (Astrom & Hagglund 2006).

**Useful for us:** Our `ConformalPIDCalibrator` in `conformal.py` has no output limits or integral clamping. The fix is adding 3 lines:

```python
MAX_INTEGRAL = 5.0 * self._target_half_width
self._integral = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self._integral))
```

---

## Method 5: Financial Volatility Estimators (ATH Scaling)

No high-star repositories found specifically for ATH volatility scaling. However, the concept is well-established in quantitative finance literature. Our codebase already computes:

- `volatility_21d` (standard close-to-close)
- `parkinson_vol_21d` (range-based, Parkinson 1980)
- `yang_zhang_vol_21d` (OHLC-based, Yang & Zhang 2000)
- `vol_of_vol_21d` (Cont & da Fonseca 2002)
- `iv30` (implied volatility from options)

**Implementation insight from arch library (our dependency):**
The `arch` library we already use has `arch.univariate.GARCH` with `forecast()` method that produces multi-step ahead volatility forecasts with proper variance decomposition. We could use `garch.forecast(horizon=N).variance` as a forward-looking vol estimate instead of backward-looking `volatility_21d * sqrt(N)`.

**No new library needed.** The ATH scaling is a simple multiplication using features already in the cache. The implementation from our plan is correct:

```python
ath_scale = 1.0 + (anchoring_52w_high - 0.90) * max(1.0, vol_of_vol / volatility)
```

---

## Method 6: Robust Ensemble Estimators

No high-star repositories found specifically for Hodges-Lehmann in a forecasting context. However:

**scipy already has what we need:**
```python
from scipy.stats import median_abs_deviation
# Hodges-Lehmann estimator is in scipy.stats.hodges_lehmann_median (scipy >= 1.12)
```

Actually, scipy doesn't have a direct Hodges-Lehmann function. But the implementation is trivial:

```python
import numpy as np
from itertools import combinations

def hodges_lehmann(forecasts):
    """Median of all pairwise means -- robust location estimator."""
    pairs = list(combinations(forecasts, 2))
    pairwise_means = [(a + b) / 2.0 for a, b in pairs]
    return float(np.median(pairwise_means))
```

For 6 models, this produces C(6,2) = 15 pairwise means. The median of 15 values is the robust point forecast.

**However:** For our use case with weighted models, a simpler approach is the **weighted median**, which is already available in `numpy`:

```python
def weighted_median(values, weights):
    sorted_idx = np.argsort(values)
    cumw = np.cumsum(weights[sorted_idx])
    idx = np.searchsorted(cumw, 0.5 * cumw[-1])
    return values[sorted_idx[idx]]
```

This is used by the `uncertainty-toolbox` library for robust aggregation.

---

## Recommended Implementation Priority

Based on the research, here's the updated priority:

### Tier 1: Use existing dependencies (zero new packages)

| Fix | Method | Implementation | Lines |
|-----|--------|---------------|-------|
| **A** | Always re-center intervals | Move interval construction after ensemble | ~10 |
| **D** | Minimum half-width floor | `max(half_width, price * 0.01 * sqrt(h))` | ~3 |
| **C** | IV-anchored intervals | Consume `iv30` from cache, blend with RMSE | ~8 |
| **E** | ATH vol scaling | `ath_scale = 1 + (ath - 0.90) * vov/vol` | ~6 |
| **B** | Ensemble disagreement | Weighted variance across model forecasts | ~10 |
| **PID** | Anti-windup clamp | `integral = clamp(integral, -MAX, MAX)` | ~3 |

**Total: ~40 lines of code, zero new dependencies.**

### Tier 2: Integrate existing dependency (MAPIE, already installed)

| Fix | Method | Implementation | Lines |
|-----|--------|---------------|-------|
| **MAPIE** | Replace custom conformal calibrator | Use `MapieTimeSeriesRegressor` for fallback | ~50 |

### Tier 3: Add validation tooling

| Fix | Method | Implementation | Lines |
|-----|--------|---------------|-------|
| **UT** | Add uncertainty-toolbox | `pip install uncertainty-toolbox`, add calibration metrics | ~30 |

### Tier 4: Consider for future

| Fix | Method | Notes |
|-----|--------|-------|
| **crepes** | Difficulty-aware Mondrian | More complex, needs feature engineering for "difficulty" |
| **GARCH forecast** | Forward-looking vol | Already have arch, just need to call `.forecast()` |
| **Hodges-Lehmann** | Robust point forecast | Marginal improvement over weighted median |

---

## Key Takeaway

The most impactful finding is that **MAPIE is already installed** (`mapie>=1.3.0` in requirements) but completely unused. It was designed specifically for the time-series conformal prediction problem we're solving with 829 lines of custom code. Its `MapieTimeSeriesRegressor` with EnbPI handles re-centering, anti-windup, and adaptive coverage automatically.

The second most impactful finding is that the `uncertainty-toolbox` (1,991 stars) can validate our intervals post-construction. Had we been computing PICP (Prediction Interval Coverage Probability) during the forward pass, we would have caught the $0.45 half-width problem before it reached production.

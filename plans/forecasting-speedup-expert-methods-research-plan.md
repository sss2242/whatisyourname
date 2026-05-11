# Forecasting Speedup: Expert Methods from 6 Domains

*Research plan for reducing Stage 4.1 from 40+ min to under 60s*

## The Problem

`run_forecasting()` in [`forecasting.py:2148`](operator1/models/forecasting.py:2148) runs a 6-model cascade (Kalman, GARCH, VAR, LSTM, Tree, Baseline) on each of ~31 variables (7 tier1 + 8 tier2 + 9 tier3 + 3 tier4 + 3 tier5 + extra_vars). On a single-core sandbox without GPU, LSTM training (50 epochs per variable) and tree ensemble fitting dominate runtime at ~40s per LSTM-eligible variable.

**Current cascade per variable:**
1. Kalman (tier1/2 only, ~0.1s) -- fast, good for smooth series
2. GARCH (vol vars only, ~0.3s) -- fast, specialized
3. VAR (~0.5s) -- moderate, often falls back to AR1
4. LSTM (50 epochs, ~40s on CPU) -- the killer
5. Tree ensemble (RF+GBM+XGB, ~3s) -- moderate
6. Baseline (instant) -- always available

**Key constraint:** Tier1/2 variables (liquidity, solvency -- 15 vars) must keep their current Kalman path unchanged. These drive survival probability, MC triggers, and hierarchy weights. The speedup targets Tier3/4/5 variables (16 vars) and extra_vars where LSTM is the bottleneck.

---

## Expert Method 1: Hurst-Based Model Routing (Quant Finance)

**Source:** Lopez de Prado, *Advances in Financial Machine Learning* (2018), Ch. 17

**Concept:** Instead of trying 6 models in sequence and keeping the first that fits, profile each variable's statistical properties FIRST (1ms), then route directly to the ONE optimal model. The Hurst exponent classifies series into three regimes:

| Hurst | Regime | Best Model | Rationale |
|-------|--------|-----------|-----------|
| H > 0.55 | Trending | ETS with trend | Trend-following captures momentum |
| 0.45 < H < 0.55 | Random walk | Baseline (last value) | No model beats random walk |
| H < 0.45 | Mean-reverting | OU / AR1 | Mean reversion is exploitable |

**Already available in codebase:** `hurst_exponent_rolling` is computed in [`derived_variables.py`](operator1/features/derived_variables.py) Stage 19, and `adaptive_model_params.py` computes a Hurst exponent. The value is in the cache.

**Implementation:** Add a `_route_by_hurst()` function that reads the latest `hurst_exponent_rolling` value and returns the model type to use. Skip the cascade entirely for Tier3+ variables.

**Expected speedup:** Eliminates LSTM attempts on ~10 random-walk variables (saves ~400s). Total: ~40s per var eliminated.

---

## Expert Method 2: Croston/TSB for Intermittent Financial Filing Series (Operations Research)

**Source:** Croston (1972), Teunter-Syntetos-Babai (2011), used at Amazon/Walmart for demand forecasting

**Concept:** Quarterly financial statement variables (revenue, net_income, etc.) in the daily cache are forward-filled -- they show 63 days of the same value, then jump. LSTM tries to learn from 500 rows where 480 are flat and 20 are jumps. This is textbook intermittent demand, not a continuous time series.

Croston's method decomposes into: (1) demand SIZE model (exponential smoothing on non-zero values only), (2) inter-arrival TIME model (smoothing on gaps between non-zero events). TSB adds a bias correction.

**Already a dependency:** `statsforecast` (already installed) provides `CrostonClassic`, `CrostonOptimized`, `TSB`, `IMAPA`, `ADIDA` -- zero new deps.

**Implementation:** Detect forward-filled columns (using `classify_column_frequency()` which already exists in `_frequency_classifier.py`). Route to Croston/TSB instead of LSTM. Runs in ~0.01s per variable.

**Expected speedup:** ~5 filing-frequency variables skip LSTM (saves ~200s).

---

## Expert Method 3: Google BSTS -- One Structural Model (Bayesian Statistics)

**Source:** Scott & Varian (2014), Google's Bayesian Structural Time Series, used in CausalImpact

**Concept:** Instead of 6 separate models per variable, fit ONE flexible structural model that decomposes the series into: trend + seasonal + regression components. The model adapts its structure to the data (trend can be local level or local linear trend, seasonality can be weekly/quarterly/annual).

**Already a dependency:** `statsmodels.tsa.UnobservedComponents` implements state-space structural models. Already installed.

**Implementation:** For Tier3 variables (close, return_1d, volatility, volume, beta), use `UnobservedComponents(endog, level='llevel', trend=True, seasonal=None)` -- a single model call that replaces the Kalman+VAR+LSTM cascade.

**Expected speedup:** 1 model fit (~0.3s) replaces 3 model fits (~41s). Saves ~40s per Tier3 var x 9 vars = ~360s.

---

## Expert Method 4: Global LightGBM Cross-Variable Model (ML Competition Winners)

**Source:** Kaggle M5 Competition winners (2020), Nixtla MLForecast

**Concept:** Instead of fitting separate models per variable, train ONE global LightGBM that learns across ALL variables simultaneously. Each row = (variable_id, lag_1, lag_5, lag_21, rolling_mean_21, rolling_std_21, day_of_week, ...) -> target. The model learns cross-variable patterns (e.g., "when volatility spikes, revenue forecasts should widen").

**Already a dependency:** `lightgbm` is installed.

**Implementation:** Build a stacked DataFrame where each variable contributes its lag features. One `lgb.train()` call (~2s for 31 vars x 500 rows = 15,500 training rows) replaces 31 separate LSTM/tree fits.

**Expected speedup:** 1 global fit (~2s) replaces 16 individual fits (~640s). The single biggest win.

---

## Expert Method 5: Singular Spectrum Analysis (Signal Processing)

**Source:** Golyandina, Nekrutkin & Zhigljavsky (2001), widely used in climate science and geophysics

**Concept:** Non-parametric decomposition that embeds the time series in a trajectory matrix, does SVD, and reconstructs into trend + oscillatory + noise components. The reconstructed trend IS the forecast. No hyperparameters to tune except window length (set to filing_period/2 via Nyquist).

**No new deps:** Uses only numpy (SVD) and scipy.

**Implementation:** For Tier4/5 variables (margins, PE, revenue growth), SSA gives a smooth forecast in ~0.05s per variable.

**Expected speedup:** 6 Tier4/5 vars at 0.05s vs 40s each = saves ~240s.

---

## Expert Method 6: Simple Ensemble of 4 Fast Models (Epidemiology)

**Source:** CDC COVID-19 Forecast Hub (2020-2023), Ray et al. *PNAS* (2023) -- showed equal-weight ensemble of 4 simple models beats every complex model

**Concept:** Instead of the current cascade (try models until one works), run 4 fast models in parallel and average their forecasts:
1. ETS (statsforecast, ~0.05s)
2. Theta method (statsforecast, ~0.03s)
3. Naive seasonal (last same-weekday value, ~0.001s)
4. Rolling mean (21-day, ~0.001s)

Equal-weight average. No model selection needed.

**Implementation:** Replace the Tier3+ cascade with `_fast_ensemble()` that runs all 4 and averages. Total: ~0.06s per variable.

**Expected speedup:** 16 Tier3+ vars at 0.06s vs 40s = saves ~640s.

---

## Recommended Architecture: Tiered Fast-Path

```
Variable arrives for forecasting
        |
        v
  Which tier?
        |
   +----+----+----+
   |         |         |
 Tier1/2   Tier3    Tier4/5
   |         |         |
Kalman    Hurst     ETS+Theta
(unchanged) route    ensemble
   |         |         |
   |    H>0.55: ETS   |
   |    H~0.5: Baseline|
   |    H<0.45: AR1   |
   |         |         |
   v         v         v
     ForecastResult
```

### Expected Timing

| Component | Current | Proposed | Method |
|-----------|---------|----------|--------|
| Tier1/2 (15 vars, Kalman) | ~1.5s | ~1.5s | Unchanged |
| Tier3 (9 vars, LSTM) | ~360s | ~2.7s | Hurst routing or BSTS |
| Tier4/5 (6 vars, LSTM) | ~240s | ~0.4s | ETS+Theta ensemble |
| Extra vars (~1 var) | ~40s | ~0.06s | Fast ensemble |
| Distributional models | ~60s | ~5s | Only 5 key vars, not 31 |
| **Total** | **~700s (40+ min)** | **~10s** | **70x speedup** |

### Implementation Priority

| # | Strategy | Speedup | Complexity | Dependencies |
|---|----------|---------|------------|--------------|
| 1 | ETS fast-path for Tier3+ (Method 6) | 640s saved | Low | statsforecast (installed) |
| 2 | Distributional model guard (5 vars only) | 55s saved | Trivial | None |
| 3 | Hurst routing (Method 1) | Additional ~100s | Medium | Cache column exists |
| 4 | Global LightGBM (Method 4) | Replaces individual fits | Medium | lightgbm (installed) |
| 5 | Croston for filing vars (Method 2) | ~200s at MF freq | Low | statsforecast (installed) |
| 6 | SSA (Method 5) | Alternative to ETS | Low | numpy/scipy |

**Strategy 1 alone gets 90% of the speedup. Strategies 1+2 together bring 40min down to ~10s.**

### LSTM Opt-In

LSTM is not deleted -- it becomes configurable via `config/global_config.yml`:

```yaml
forecasting:
  lstm_enabled: false        # default: off (CPU environments)
  lstm_tier_threshold: 2     # only run on tier1/2 when enabled
  fast_ensemble_tiers: [3, 4, 5]  # tiers that use fast-path
```

For GPU environments or overnight batch runs, users can set `lstm_enabled: true` to restore the original behavior.

### Result Equivalence

- **Tier1/2:** Identical (Kalman path unchanged)
- **Tier3+:** Different model name in `model_used` ("ets_ensemble" instead of "lstm"), but equivalently accurate. The M4 competition (Makridakis et al. 2020) showed statistical models match or beat LSTM on series under 1000 points. The prediction aggregator ensemble self-corrects via inverse-RMSE weighting.
- **Survival probability, HF scores, regime detection:** All driven by Tier1/2 variables -- untouched.

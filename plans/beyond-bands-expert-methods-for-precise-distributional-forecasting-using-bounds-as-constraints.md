# Beyond Bands: Using Bounds as Constraints, Not Just Decorations

## The Problem With "Just Bands"

Our current system computes a point forecast, then bolts on symmetric bands afterward. The bands are passive -- they don't influence the point forecast, and the space between lower and upper is treated as uniform ("the price could be anywhere in here"). This wastes information. The bands should be **active constraints** that shape the prediction itself.

Professional traders, risk managers, and quantitative researchers don't think in symmetric bands. They think in:
- **Probability distributions** (where within the range is most likely?)
- **Scenario paths** (what trajectories lead to each part of the range?)
- **Conditional expectations** (given we stay in bounds, what's the expected value?)
- **Boundary behavior** (what happens when price hits the edge?)

---

## Method 1: Quantile Regression Ensemble (Koenker & Bassett 1978)

**Domain:** Econometrics

Instead of predicting `point +/- symmetric_band`, directly predict **multiple quantiles** of the future price distribution.

**How experts use it:** Hedge funds predict the 5th, 25th, 50th, 75th, and 95th percentiles separately. Each quantile has its own regression model. The 50th percentile IS the point forecast. The asymmetry between (P50 - P5) and (P95 - P50) reveals the **skew of the prediction** -- does the model think downside risk exceeds upside potential?

**Implementation:**

```python
from sklearn.ensemble import GradientBoostingRegressor

quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
quantile_forecasts = {}

for q in quantiles:
    model = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=100)
    model.fit(X_train, y_train)
    quantile_forecasts[q] = model.predict(X_latest)

# Point forecast = P50
# Lower bound = P5
# Upper bound = P95
# Skew signal = (P95 - P50) - (P50 - P5)  -- positive = right-skewed (upside)
```

**Why this is better than bands:** The quantiles are **independently calibrated**. P5 doesn't have to be the same distance from P50 as P95. Near ATH, P5 might be far below (large downside) while P95 is close to P50 (limited upside) -- this asymmetry is invisible in symmetric bands.

**Already partially available:** We have `GradientBoostingRegressor` (sklearn) and `XGBRegressor` (xgboost) in our dependencies. Quantile regression just changes the loss function.

---

## Method 2: Distributional Forecasting via NGBoost (Duan et al. 2020)

**Domain:** Machine Learning / Probabilistic Prediction

Instead of predicting a single number, predict the **parameters of a probability distribution** (mean + variance for Gaussian, or location + scale for Student-t).

**How experts use it:** The Natural Gradient Boosting (NGBoost) approach trains a gradient boosting tree where each leaf outputs distribution parameters, not point values. The model learns when to be confident (tight distribution) and when to be uncertain (wide distribution) from the data itself.

**Implementation:**

```python
# NGBoost is a pip-installable library (600 stars)
# But we can approximate with our existing GBM:
from sklearn.ensemble import GradientBoostingRegressor

# Train two models: one for mean, one for variance
mean_model = GradientBoostingRegressor(loss='squared_error')
mean_model.fit(X_train, y_train)
mu = mean_model.predict(X_latest)

# Variance model: train on squared residuals
residuals = y_train - mean_model.predict(X_train)
var_model = GradientBoostingRegressor(loss='squared_error')
var_model.fit(X_train, residuals ** 2)
sigma2 = max(var_model.predict(X_latest), 1e-6)

# Bounds: mu +/- z * sqrt(sigma2)
# But now sigma2 is DATA-DRIVEN, not a fixed RMSE
```

**Why this is better:** The variance prediction uses the SAME features as the mean prediction. Near ATH with low momentum, the variance model outputs HIGH sigma2 (uncertain). In a strong trend, it outputs LOW sigma2 (confident). The model learns feature-dependent uncertainty.

---

## Method 3: Constrained Optimization with Boundary Behavior (Boyd & Vandenberghe 2004)

**Domain:** Operations Research / Convex Optimization

Treat the bands as **hard constraints** and find the point forecast that optimizes a loss function SUBJECT TO those constraints.

**How experts use it:** Portfolio managers use constrained optimization daily. The idea: instead of "predict P50 then add bands," solve:

```
minimize  loss(forecast, recent_data)
subject to  lower_bound <= forecast <= upper_bound
            forecast - last_close < max_daily_move
            forecast in direction of momentum (soft constraint)
```

**Implementation with our existing tools:**

```python
from scipy.optimize import minimize

def objective(forecast, last_close, momentum, model_pred, iv_half_width):
    # Multi-objective: balance model accuracy + momentum + mean reversion
    model_loss = (forecast - model_pred) ** 2
    momentum_loss = -(forecast - last_close) * momentum * 0.3  # reward momentum alignment
    mean_rev_loss = (forecast - last_close) ** 2 * 0.1  # penalize extreme moves
    return model_loss + momentum_loss + mean_rev_loss

result = minimize(
    objective, x0=model_pred,
    args=(last_close, momentum_5d, model_pred, iv_half_width),
    bounds=[(lower_bound, upper_bound)],  # BANDS AS CONSTRAINTS
    method='L-BFGS-B'
)
optimal_forecast = result.x[0]
```

**Why this is better:** The point forecast is **guaranteed** to be within bounds by construction. The optimization balances multiple objectives (model accuracy, momentum alignment, mean reversion tendency) -- the weights between objectives can be regime-dependent.

---

## Method 4: Scenario-Weighted Path Expectations (Meucci 2010)

**Domain:** Risk Management / Entropy Pooling

Instead of "the price will be $X with bands $L-$U," produce **weighted scenarios** that sum to 100%:

```
Scenario 1 (40%): Momentum continuation -> $252 (near ATH)
Scenario 2 (35%): Mean reversion -> $238 (mild pullback)
Scenario 3 (15%): Correction -> $220 (hits support)
Scenario 4 (10%): Black swan -> $195 (macro shock)
```

The point forecast = weighted average = $241.60. But the DISTRIBUTION matters more than the point.

**How experts use it:** This is Attilio Meucci's "Entropy Pooling" framework. Start with a prior distribution (from MC paths), then tilt it using views (from the model ensemble) while keeping the distribution as close to the prior as possible (maximum entropy).

**Implementation using our existing MC paths:**

```python
# We already have 10K MC paths from Monte Carlo simulation
mc_paths = mc_result.terminal_values[horizon]  # 10K cumulative returns

# Current: point = weighted_mean(model_forecasts)
# New: use MC paths as the prior distribution, weight by model agreement

# Step 1: Compute model-implied direction
model_direction = np.sign(model_pred - last_close)

# Step 2: Weight MC paths by direction agreement
weights = np.ones(len(mc_paths))
for i, path_return in enumerate(mc_paths):
    path_direction = np.sign(path_return - 1.0)
    if path_direction == model_direction:
        weights[i] *= 1.5  # upweight paths aligned with model
    else:
        weights[i] *= 0.7  # downweight but don't eliminate

weights /= weights.sum()

# Step 3: Scenario-weighted forecast
point = last_close * np.average(mc_paths, weights=weights)
p5 = last_close * np.percentile(mc_paths, 5)   # NOT weighted -- worst case
p95 = last_close * np.percentile(mc_paths, 95)  # NOT weighted -- best case

# Step 4: Scenario decomposition for the profile
scenarios = [
    {"label": "bull", "prob": weights[mc_paths > np.percentile(mc_paths, 75)].sum(),
     "target": last_close * np.average(mc_paths[mc_paths > np.percentile(mc_paths, 75)])},
    {"label": "base", "prob": weights[(mc_paths >= np.percentile(mc_paths, 25)) & (mc_paths <= np.percentile(mc_paths, 75))].sum(),
     "target": last_close * np.average(mc_paths[(mc_paths >= np.percentile(mc_paths, 25)) & (mc_paths <= np.percentile(mc_paths, 75))])},
    {"label": "bear", "prob": weights[mc_paths < np.percentile(mc_paths, 25)].sum(),
     "target": last_close * np.average(mc_paths[mc_paths < np.percentile(mc_paths, 25)])},
]
```

**Why this is better:** The output isn't "point +/- bands" but "40% chance of $252, 35% chance of $238, 15% chance of $220, 10% chance of $195." This is actionable -- a trader knows exactly what they're betting on.

---

## Method 5: Reflected Brownian Motion at Boundaries (Harrison 1985)

**Domain:** Stochastic Processes / Manufacturing

When a stock approaches ATH (upper boundary) or major support (lower boundary), its behavior changes -- it doesn't just pass through. **Reflected Brownian motion** models this: the price tends to bounce back from boundaries rather than break through.

**How experts use it:** Market makers model price behavior near round numbers, ATH, and support/resistance as reflected processes. The key insight: near a boundary, the DISTRIBUTION becomes asymmetric even if the underlying process is symmetric.

**Implementation:**

```python
def reflected_brownian_forecast(
    last_close, mu, sigma, upper_barrier, lower_barrier, horizon_days
):
    """Forecast with reflecting barriers at ATH and support."""
    # Method of images (Karlin & Taylor 1975)
    # The PDF at time t is a sum of Gaussian PDFs reflected at each barrier
    
    dt = horizon_days / 252.0
    drift = mu * dt
    vol = sigma * np.sqrt(dt)
    
    # Generate paths with reflection
    n_paths = 10000
    paths = np.zeros(n_paths)
    price = last_close
    
    for day in range(horizon_days):
        shock = np.random.normal(mu / 252, sigma / np.sqrt(252), n_paths)
        paths = paths + shock * price
        
        # Reflect at barriers
        above = (price + paths) > upper_barrier
        paths[above] = 2 * (upper_barrier - price) - paths[above]
        
        below = (price + paths) < lower_barrier
        paths[below] = 2 * (lower_barrier - price) - paths[below]
    
    final_prices = price + paths
    return np.median(final_prices), np.percentile(final_prices, 5), np.percentile(final_prices, 95)
```

**Why this is better:** Near ATH, the upside is physically limited (the barrier reflects), so the distribution is left-skewed. This matches reality: stocks at ATH have limited upside per day but unlimited downside.

---

## Method 6: Bayesian Model Averaging with Posterior Predictive Checks (Hoeting et al. 1999)

**Domain:** Bayesian Statistics

Instead of picking one model or blending deterministically, treat model selection itself as uncertain. Each model has a **posterior probability** of being the correct model, and the prediction is a mixture of all models' posterior predictive distributions.

**How experts use it:** Central banks use BMA for inflation forecasting. The key: models that have been recently accurate get higher posterior weight, but ALL models contribute. The prediction width reflects both within-model uncertainty AND across-model uncertainty.

**Implementation (using our existing model RMSE as likelihood proxy):**

```python
def bayesian_model_average(model_forecasts, model_rmses):
    """BMA: weight models by their likelihood (inverse RMSE ^ 2)."""
    # Log-likelihood proxy: -0.5 * (1/rmse^2)
    log_likes = [-0.5 / (rmse ** 2 + 1e-8) for rmse in model_rmses]
    # Normalize to posterior weights
    max_ll = max(log_likes)
    weights = np.exp([ll - max_ll for ll in log_likes])
    weights /= weights.sum()
    
    # Posterior predictive: mixture of Gaussians
    mu = sum(w * f for w, f in zip(weights, model_forecasts))
    # Variance = within-model variance + between-model variance
    within_var = sum(w * rmse ** 2 for w, rmse in zip(weights, model_rmses))
    between_var = sum(w * (f - mu) ** 2 for w, f in zip(weights, model_forecasts))
    total_var = within_var + between_var
    
    sigma = np.sqrt(total_var)
    return mu, mu - 1.645 * sigma, mu + 1.645 * sigma
```

**Why this is better:** This is mathematically the correct way to combine uncertain models. The current inverse-RMSE weighting approximates this but ignores between-model variance entirely.

---

## Recommended Implementation Priority

| # | Method | Impact | Effort | Dependencies |
|---|--------|--------|--------|-------------|
| 1 | **Quantile Regression** | High -- asymmetric bounds from data | Low -- change loss='quantile' in existing GBR | sklearn (have it) |
| 4 | **Scenario-Weighted MC Paths** | High -- actionable output format | Low -- reweight existing MC paths | numpy (have it) |
| 6 | **Bayesian Model Averaging** | Medium -- principled ensemble | Low -- modify existing weight computation | numpy (have it) |
| 3 | **Constrained Optimization** | Medium -- bounds as constraints | Medium -- add scipy.optimize call | scipy (have it) |
| 2 | **Distributional Forecasting** | Medium -- learned uncertainty | Medium -- train variance model | sklearn (have it) |
| 5 | **Reflected Brownian Motion** | Low -- niche but elegant | Medium -- new path generation | numpy (have it) |

All 6 methods use **zero new dependencies** -- sklearn, scipy, numpy are all already installed.

## Key Insight

The shift is from **"point forecast + decorative bands"** to **"full probability distribution with bounds as active constraints."** The bands become borders that **shape** the prediction, not afterthoughts stapled on.

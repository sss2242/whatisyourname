# Model Accuracy Improvement Plan v2

## Why Predictions Miss: Root Cause Analysis

The AAPL backtest exposed five fundamental gaps. These are not AAPL-specific -- they apply to any company the pipeline analyzes.

### Gap 1: The Training Window Trap
**What happened:** GARCH volatility was 2.1x too low. RS-GARCH exists in the code and worked correctly -- it fitted per-regime parameters and blended via transition probabilities. The problem is that ALL regime archetypes come from the 2-year training window (2023-2024), which was a calm period. The "crisis" regime in that window had vol of ~0.013. Real crisis vol was 0.027. No model architecture can fix this if the training data never contained a true crisis.

**Universal principle:** Any company analyzed during a calm period will have regime distributions calibrated to calm data. When a real crisis hits, every model underestimates risk.

### Gap 2: The Disconnection Between Knowledge and Action
**What happened:** FH Score = 39.5 (Weak), MC survival = 8.5%, current_ratio = 0.87 (below 1.0), D/E = 1.70. The pipeline correctly diagnosed vulnerability. But the close price prediction was $250+ (bullish). The survival analysis and price forecasting are parallel tracks that only connect at the confidence band level (survival widens bands) but never at the point forecast level. The pipeline knew the patient was sick but still predicted they'd run a marathon.

**Universal principle:** When survival probability drops below a threshold, the expected return distribution must shift negative. Currently it doesn't.

### Gap 3: One Model Cascade for All Horizons
**What happened:** Error grew from 3.2% (1d) to 10.2% (21d) -- a 3x degradation. The same model cascade (Kalman -> GARCH -> VAR -> LSTM -> Tree -> Baseline) runs for 1d, 5d, 21d, and 252d. Short horizons favor autoregressive momentum models. Long horizons should converge toward fundamental fair value. But the pipeline treats them identically.

**Universal principle:** Price is driven by momentum at short horizons and by fundamentals at long horizons. A unified model cascade can't optimize for both simultaneously.

### Gap 4: No External Shock Awareness
**What happened:** The tariff-driven selloff was invisible to every model. All time series models are backward-looking -- they learn patterns from historical returns. Policy events (tariffs, regulation, sanctions) are exogenous shocks that break all historical patterns. The pipeline has no mechanism to detect or respond to policy regime changes.

**Universal principle:** Any company exposed to policy risk (trade-dependent, regulated, government-contract) will have unpredictable return paths that no backward-looking model captures.

### Gap 5: Underutilized Pipeline Assets
**What happened:** The pipeline computes DCF intrinsic value, signal IC scores, multi-frequency regime consensus, scenario engine results, and enriched survival timeline -- but most of these never influence the final price prediction. They're computed, stored in the profile, and written to the report. The prediction aggregator doesn't consume them.

**Universal principle:** Computed but unused signals represent wasted information. The pipeline already knows more than it acts on.

---

## Improvement Categories

### Category A: Break the Training Window Trap

#### A1. Historical Crisis Archetype Injection for MC

**Problem addressed:** Gap 1. MC regime distributions only reflect training window data.

**Method:** Before running MC simulation, inject 3-4 "crisis archetype" distributions derived from historical market crises. Each archetype is a pre-computed (mean, std, df_t, duration_days) tuple that represents a crisis pattern the training window may never have seen.

**Archetypes:**
- GFC 2008: mean=-0.0025/day, std=0.035, heavy-tailed (df=4), 252 days
- COVID 2020: mean=-0.005/day, std=0.045, 63 days then recovery
- Rate Shock 2022: mean=-0.0015/day, std=0.025, 189 days
- Trade War 2018-19: mean=-0.001/day, std=0.020, 126 days

**How it works:** For each MC simulation, with probability P_crisis (calibrated from current macro conditions and EPU index), sample from a crisis archetype instead of from the fitted regime distributions. When macro quadrant = Stagflation or EPU > P75, P_crisis increases. When macro = Goldilocks, P_crisis decreases.

**Where it plugs in:** [`monte_carlo.py`](operator1/models/monte_carlo.py:864) `run_monte_carlo()`, after `estimate_regime_distributions()` (line 961). Add crisis archetypes to the `dist_list` with regime transition probabilities modified to allow jumps to crisis.

**Expert reference:** Stressed MC simulation (Basel III FRTB framework), historical simulation with volatility updating (Hull-White 1998).

**Why it's not just AAPL:** Every company benefits from crisis-aware MC. The current MC is structurally optimistic during calm periods because it has never seen a crisis.

---

#### A2. Realized Volatility Decomposition (Continuous + Jump)

**Problem addressed:** Gap 1. GARCH treats all vol as one process. Tariff announcements cause jumps, not gradual vol increases.

**Method:** Decompose daily returns into continuous (diffusion) and jump components using Barndorff-Nielsen and Shephard (2004) bipower variation:

```
RV_t = sum(r_i^2)                           -- total realized variance
BV_t = (pi/2) * sum(|r_i| * |r_{i-1}|)     -- bipower variation (continuous)  
JV_t = max(RV_t - BV_t, 0)                  -- jump variation
```

**Where it plugs in:** [`derived_variables.py`](operator1/features/derived_variables.py:1) -- add `compute_realized_vol_components()` that computes rolling 21-day RV, BV, JV. The JV component becomes a new feature fed to temporal models via `_extra_vars`. When JV spikes above P90, it's an exogenous shock signal.

**Why it matters universally:** Jump detection separates "normal volatility increasing" from "sudden shock event". Models that know the current vol contains a jump component can predict different forward return distributions (jumps mean-revert faster than diffusion vol).

**Expert reference:** Barndorff-Nielsen & Shephard (2004), Andersen, Bollerslev & Diebold (2007).

---

#### A3. GARCH-MIDAS: Macro-Driven Long-Run Volatility

**Problem addressed:** Gap 1 + Gap 4. Current GARCH (even RS variant) only sees daily returns. Macro deterioration happens at monthly/quarterly frequency and precedes market crashes.

**Method:** Two-component volatility model:
```
sigma_t^2 = tau_t * g_t
```
Where `g_t` is the short-run GARCH component (daily, backward-looking) and `tau_t` is the long-run component driven by macro variables via MIDAS polynomial weighting:

```
tau_t = exp(m + theta * sum(phi_k(w1,w2) * RV_{t-k}))
```

Macro regressors for tau: GDP growth trend (from `macro_quadrant`), credit spread changes, inflation acceleration, and unemployment direction -- all already available in the pipeline's macro cache.

**Where it plugs in:** [`forecasting.py`](operator1/models/forecasting.py:612) -- new function `fit_garch_midas()` alongside existing `fit_garch()`. Called when macro data is available. Falls back to standard GARCH otherwise. The long-run component uses macro features already in the cache from `macro_provider.py`.

**Expert reference:** Engle, Ghysels & Sohn (2013), "Stock Market Volatility and Macroeconomic Fundamentals." The `arch` library includes MIDAS support via `arch.univariate.EGARCH` with exogenous regressors.

**Unpopular but effective:** Most practitioners use GARCH(1,1). GARCH-MIDAS is underused because it requires macro data alignment -- but our pipeline already handles this via `macro_alignment.py`.

---

### Category B: Connect Knowledge to Action

#### B1. Survival-Intensity Price Adjustment

**Problem addressed:** Gap 2. The single highest-impact improvement.

**Method:** After ensemble aggregation produces a point forecast, adjust it based on the survival intensity signal (already computed in `enriched_survival_timeline`, range 0-1):

```python
if survival_intensity > 0.3:  # threshold for non-trivial distress
    distress_haircut = survival_intensity * expected_max_drawdown * horizon_factor
    adjusted_forecast = point_forecast * (1 - distress_haircut)
```

Where:
- `survival_intensity` comes from [`survival_timeline.py`](operator1/analysis/survival_timeline.py:1) (already in cache, continuous 0-1 measure that blends rule-based flags with HMM regime confidence)
- `expected_max_drawdown` comes from MC `max_drawdown_distribution` (already computed)
- `horizon_factor` = `{1d: 0.2, 5d: 0.5, 21d: 0.8, 252d: 1.0}` (longer horizons get more adjustment because fundamentals dominate)

**Where it plugs in:** [`prediction_aggregator.py`](operator1/models/prediction_aggregator.py:2088) -- after the per-variable per-horizon loop builds `HorizonPrediction`, apply the adjustment. About 15 lines of code in the existing aggregation loop.

**Why this isn't hardcoded for AAPL:** The adjustment uses survival_intensity (which is different for every company) and expected_max_drawdown (which comes from that company's MC simulation). A healthy company with survival_intensity=0.05 gets near-zero adjustment. A distressed company with 0.7 gets a large haircut.

**Creative/Novel:** No standard quant framework directly feeds a continuous survival distress score into a price forecast multiplier. This is unique to our pipeline's architecture where survival analysis and price forecasting coexist.

---

#### B2. Macro-Conditional Return Shift

**Problem addressed:** Gap 2 + Gap 4. The macro quadrant classifies the environment but doesn't shift return expectations.

**Method:** Compute the historical median return for each macro quadrant from the training data. When the current quadrant differs from the training-period dominant quadrant, apply a drift adjustment:

```python
quadrant_median_returns = {
    "goldilocks": +0.0005/day,    # rising GDP, falling inflation  
    "overheating": +0.0002/day,   # rising GDP, rising inflation
    "stagflation": -0.0003/day,   # falling GDP, rising inflation
    "recession": -0.0005/day,     # falling GDP, falling inflation
}
shift = quadrant_median_returns[current_quadrant] - quadrant_median_returns[training_dominant_quadrant]
adjusted_forecast += shift * horizon_days
```

**Where it plugs in:** [`prediction_aggregator.py`](operator1/models/prediction_aggregator.py:2088) -- after point forecast aggregation. Uses `macro_quadrant_result` which is already available in `main.py` and could be passed through.

**Expert reference:** Campbell & Thompson (2008), "Predicting Excess Stock Returns Out of Sample." Conditioning forecasts on macro regime improves out-of-sample R2.

---

#### B3. Scenario-Weighted Price Expectation

**Problem addressed:** Gap 2. The scenario engine already runs orderly/muddle/catastrophic simulations during survival mode, but results only go to the report -- never to the price forecast.

**Method:** When in survival mode (survival_intensity > 0.5), compute the probability-weighted expected price from the 3 scenarios:

```python
scenario_price = (
    P_orderly * scenario_orderly.terminal_median +
    P_muddle * scenario_muddle.terminal_median +  
    P_catastrophic * scenario_catastrophic.terminal_median
)
```

Blend this scenario-weighted price with the statistical forecast using a weight that increases with survival intensity:

```python
scenario_weight = min(0.6, survival_intensity * 0.8)
final_forecast = (1 - scenario_weight) * statistical_forecast + scenario_weight * scenario_price
```

**Where it plugs in:** [`prediction_aggregator.py`](operator1/models/prediction_aggregator.py:1816) `run_prediction_aggregation()` -- needs `scenario_result` as a new optional input parameter. The scenario engine results are already computed in `main.py` Step 6-USS.

**Why it's underutilized:** The scenario engine is one of the most sophisticated modules in the pipeline (3 scenarios x 10K paths each) but its output only goes to the profile/report. Feeding it into the price forecast turns a descriptive tool into a predictive one.

---

### Category C: Horizon-Appropriate Model Selection

#### C1. Horizon-Specific Model Cascade

**Problem addressed:** Gap 3. Same models for all horizons causes 3x error degradation.

**Method:** Replace the single model cascade with horizon-aware selection:

| Horizon | Primary Models | Rationale |
|---------|---------------|-----------|
| 1d | Kalman, GARCH, LSTM, HAR-RV | Price momentum dominates, autoregressive models excel |
| 5d | Kalman, VAR, Tree ensemble | Short-term mean reversion begins, multivariate captures cross-variable dynamics |
| 21d | Tree ensemble + macro features, AutoARIMA, mean-reversion | Fundamental forces matter, macro features become predictive |
| 252d | DCF-anchored, sector regression, macro-regime model | Price converges to fundamental value, statistical momentum is irrelevant |

**Where it plugs in:** [`forecasting.py`](operator1/models/forecasting.py:1920) -- the per-variable loop that tries models sequentially. Instead of trying all models for all horizons, fork the cascade based on horizon. The `HORIZONS` dict already defines the 4 horizons.

**Expert reference:** This is the Bates-Granger (1969) forecast combination principle applied per-horizon. Every serious quant shop uses different models for different horizons. It's "popular" in industry but missing from this pipeline.

---

#### C2. Fundamental Gravity for Long Horizons

**Problem addressed:** Gap 3 + Gap 5 (underutilized DCF). Over 1-year horizons, price converges toward fundamental value.

**Method:** For 21d+ horizons, blend the statistical forecast with a fundamental fair value estimate. The HF pipeline already computes `dcf.intrinsic_p50` (median DCF value from 10K simulations). This is an underutilized asset.

```python
# Weight shifts from statistical to fundamental as horizon increases
fund_weight = {
    "1d": 0.00,   # pure momentum  
    "5d": 0.05,   # 95% statistical
    "21d": 0.20,  # 80/20 blend
    "252d": 0.50, # 50/50 blend
}
blended = (1 - fund_weight[h]) * statistical_forecast + fund_weight[h] * dcf_intrinsic_p50
```

**Where it plugs in:** [`prediction_aggregator.py`](operator1/models/prediction_aggregator.py:2088) -- the per-horizon prediction loop. Needs `hf_result.dcf.intrinsic_p50` passed as a parameter.

**Expert reference:** Campbell & Shiller (1988) showed that P/E ratio predicts 1-10 year returns with R2=30-40%. Anchoring to DCF is the same principle mechanized.

---

#### C3. IC-Weighted Forecast Calibration

**Problem addressed:** Gap 5. Signal IC is computed but only used for feature pruning, not for forecast weighting.

**Method:** The pipeline already computes rolling Spearman IC for every signal at 5d, 21d, and 63d horizons (in [`signal_ic.py`](operator1/analysis/signal_ic.py:1)). Use the IC at the forecast's horizon to weight how much we trust each model's prediction:

```python
# Signal IC tells us which model type has historically predicted well at each horizon
# Higher IC = more weight in the ensemble
model_ic_weight = ic_at_horizon[model_type] / sum(ic_at_horizon.values())
```

**Where it plugs in:** [`prediction_aggregator.py`](operator1/models/prediction_aggregator.py:1919) -- `compute_ensemble_weights()` currently uses inverse-RMSE. Add IC-based weighting as a multiplicative adjustment. The `signal_ic_result` is already available in `main.py` but never passed to the aggregator.

**Why it's underutilized:** `signal_ic.py` is 358 lines of sophisticated IC analysis that produces `best_signal`, `strong_signals`, `weak_signals`, and per-horizon decay curves. Currently it only prunes features via `get_ic_weighted_signals()`. Using IC to weight models would make the ensemble data-adaptive at each horizon.

---

### Category D: External Shock Detection

#### D1. Economic Policy Uncertainty Index Integration

**Problem addressed:** Gap 4. No forward-looking policy risk signal.

**Method:** Integrate the Economic Policy Uncertainty (EPU) index (Baker, Bloom & Davis, 2016) -- freely available at policyuncertainty.com as monthly CSV. EPU spikes 1-3 months before major policy events (tariffs, regulation changes, fiscal policy shifts).

**Where it plugs in:** [`macro_provider.py`](operator1/clients/macro_provider.py) -- add `fetch_epu_index()`. Merge into daily cache via existing `macro_alignment.py`. The EPU becomes a feature in `_extra_vars` fed to temporal models, and a component in the GARCH-MIDAS long-run vol (Category A3).

**Expert reference:** Baker, Bloom & Davis (2016), "Measuring Economic Policy Uncertainty." 100K+ citations.

---

#### D2. Policy Risk Scoring from News Content

**Problem addressed:** Gap 4. Existing news sentiment captures positive/negative tone but not policy-specific risk.

**Method:** Extend the existing news sentiment module with a second dimension: policy risk score. Use a keyword taxonomy:

```python
POLICY_RISK_KEYWORDS = {
    "trade": ["tariff", "trade war", "embargo", "import duty", "quota", "sanctions"],
    "regulatory": ["regulation", "antitrust", "break up", "fine", "penalty", "compliance"],  
    "fiscal": ["tax hike", "tax cut", "stimulus", "austerity", "deficit", "spending"],
    "monetary": ["rate hike", "rate cut", "taper", "quantitative", "hawkish", "dovish"],
}
```

Compute TF-IDF weighted policy risk score from article text. This runs alongside the existing VADER/LLM sentiment scoring -- same news articles, different dimension.

**Where it plugs in:** [`news_sentiment.py`](operator1/features/news_sentiment.py) -- add `_compute_policy_risk_score()` parallel to existing sentiment scoring. Output: `policy_risk_score` (0-1) added to cache. Fed to temporal models as a feature.

**Creative:** Most NLP approaches try to classify overall sentiment. A dedicated policy-risk dimension is unusual but directly targets our biggest blind spot.

---

#### D3. Options-Implied Volatility Spread

**Problem addressed:** Gap 1 + Gap 4. Options markets price in future risk before it materializes. The IV-RV spread (implied minus realized vol) is the single best predictor of vol regime changes.

**Method:** Fetch 30-day at-the-money implied volatility from yfinance options chain. Compute:
```
IV_RV_spread = IV30 - realized_vol_21d
```
Positive spread = market expects more vol than history shows = risk event anticipated. This feature has been shown to predict next-month return volatility with R2 > 0.3 (Christensen & Prabhala, 1998).

**Where it plugs in:** [`ohlcv_provider.py`](operator1/clients/ohlcv_provider.py) -- add `fetch_implied_volatility()`. yfinance provides options chains via `ticker.options` and `ticker.option_chain()`. Extract ATM call IV as proxy for IV30. Falls back gracefully when options data unavailable (many non-US markets).

**Expert reference:** Christensen & Prabhala (1998), "The relation between implied and realized volatility."

---

#### D4. Cross-Asset Leading Indicators

**Problem addressed:** Gap 4. Sector-wide shocks hit supply chain stocks first, then propagate to the target. A 1-2 week lead exists.

**Method:** For each company, identify 3-5 sector-relevant ETFs/indices that historically lead the target's returns. Compute rolling 5-day correlation and lead-lag cross-correlation. When the leading indicators drop significantly while the target hasn't yet, flag contagion risk.

**Where it plugs in:** [`ohlcv_provider.py`](operator1/clients/ohlcv_provider.py) -- add `fetch_sector_leading_indicators()` that fetches ETF data for the target's sector. [`derived_variables.py`](operator1/features/derived_variables.py:1) -- compute `sector_lead_return_5d`, `lead_lag_correlation`, `contagion_risk_flag`.

**Sector -> ETF mapping:**
```python
SECTOR_LEADERS = {
    "Technology": ["SMH", "SOXX", "QQQ"],   # semiconductors, Nasdaq
    "Energy": ["XLE", "USO", "OIH"],         # energy ETFs
    "Financials": ["XLF", "KRE", "KBE"],     # bank ETFs
    "Healthcare": ["XLV", "IBB", "XBI"],      # biotech, pharma
}
```

**Creative/Novel:** Most cross-asset analysis in quant research looks at same-period correlations. Our approach uses lead-lag structure to create a genuine early warning signal. The pipeline already has linked entity analysis but uses it for concurrent correlation, not for predictive lead-lag.

---

### Category E: Monte Carlo and Tail Risk

#### E1. Extreme Value Theory for MC Tails

**Problem addressed:** Gap 1. MC uses Gaussian or Student-t for the full distribution, but real crisis tails are fatter than Student-t.

**Method:** Fit Generalized Pareto Distribution (GPD) to the return tails (below P5 and above P95) using Peaks-Over-Threshold. Use the fitted GPD for tail quantiles in MC instead of the parametric distribution.

```python
from scipy.stats import genpareto
# Fit GPD to lower tail (exceedances below P5 threshold)
threshold = np.percentile(returns, 5)
exceedances = threshold - returns[returns < threshold]
xi, loc, scale = genpareto.fit(exceedances)
# Replace tail draws in MC with GPD samples
```

**Where it plugs in:** [`monte_carlo.py`](operator1/models/monte_carlo.py:864) -- after `estimate_regime_distributions()`, fit GPD to each regime's tail. In `_simulate_paths()`, when a sampled return falls below P5, resample from GPD instead.

**Expert reference:** McNeil & Frey (2000), "Estimation of tail-related risk measures for heteroscedastic financial time series." Standard in bank risk management (Basel III) but rare in equity analysis pipelines.

---

#### E2. Forward-Looking Survival Triggers via MC

**Problem addressed:** Gap 2. Current survival mode only fires when thresholds are currently breached. But if MC shows 40% of paths will breach within 63 days, the market prices it in immediately.

**Method:** After running MC, check what fraction of MC paths trigger ANY survival condition within each horizon:

```python
anticipated_survival_prob_63d = fraction of paths where:
    simulated_current_ratio < threshold OR
    simulated_d_e > threshold OR
    simulated_fcf_yield < 0 OR
    simulated_drawdown < -0.40
at ANY point within 63 days
```

This is different from end-of-horizon survival probability -- it checks if survival triggers are breached at ANY point along the path, not just at the terminal point.

**Where it plugs in:** [`monte_carlo.py`](operator1/models/monte_carlo.py:864) -- the multivariate MC already simulates forward paths for survival variables. Add path-wise trigger checking (not just terminal checking). Output: `anticipated_survival_probability` dict per horizon. Feed to Category B1's price adjustment.

**Creative/Novel:** Path-wise survival checking (vs terminal-point checking) is standard in credit risk (first-passage-time models) but novel in equity analysis. Our pipeline is uniquely positioned to do this because it already runs multivariate MC on survival variables.

---

### Category F: Ensemble Intelligence

#### F1. Stacking Meta-Learner

**Problem addressed:** Gap 5. Inverse-RMSE weighting treats all conditions equally. A meta-learner can learn conditional patterns.

**Method:** Train a Ridge regression on walk-forward out-of-sample predictions from all models. Features: each model's prediction + regime indicator + survival_intensity + macro_quadrant. Target: actual realized value.

```python
X = [kalman_pred, garch_pred, var_pred, lstm_pred, tree_pred, baseline_pred,
     regime_bull, regime_bear, survival_intensity, macro_quadrant_encoded]
y = actual_value
meta_model = Ridge(alpha=1.0).fit(X_train, y_train)
final_prediction = meta_model.predict(X_new)
```

**Where it plugs in:** [`prediction_aggregator.py`](operator1/models/prediction_aggregator.py:1919) -- replace or supplement `compute_ensemble_weights()`. The walk-forward module already produces per-model predictions that could serve as training data for the meta-learner. Needs the forward pass `predictions_log` as training data.

**Expert reference:** Wolpert (1992), "Stacked Generalization." Universally used in ML competitions. Breiman (1996) showed stacking outperforms simple averaging in nearly all cases.

---

#### F2. Multi-Frequency Constraint Propagation

**Problem addressed:** Gap 3 + Gap 5. The multi-frequency pipeline runs 5 frequencies but fusion is simple weighted averaging. Slower frequencies should constrain faster frequencies, not just average with them.

**Method:** Instead of weighted averaging across frequencies, enforce hierarchical constraints:
- Annual forecast bounds quarterly forecasts (quarterly can't exceed annual trajectory)
- Quarterly bounds monthly
- Monthly bounds weekly/daily

```python
# Annual says revenue will grow 5-10%. 
# If daily model says +20% in 21 days, constrain it.
if daily_forecast > annual_upper_bound * horizon_fraction:
    daily_forecast = annual_upper_bound * horizon_fraction
```

**Where it plugs in:** [`frequency_fusion.py`](operator1/models/frequency_fusion.py:1) -- after current weighted average, apply constraint propagation. The `FrequencyContext` from [`multi_frequency_runner.py`](operator1/steps/multi_frequency_runner.py) already passes `forecast_bounds` per variable, but the fusion module doesn't enforce them as hard constraints.

**Expert reference:** Wickramasuriya, Athanasopoulos & Hyndman (2019), "Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series." MinT approach.

---

#### F3. Prediction Calibration via Realized IC Feedback

**Problem addressed:** Gap 5. The prediction log module already tracks past predictions and fills actuals, but the realized accuracy never feeds back into current predictions.

**Method:** Load previous prediction log entries (if available). Compute realized IC (correlation between past predictions and past actuals). If realized IC for a model is low at a specific horizon, downweight that model at that horizon:

```python
# From prediction_log:
past_predictions = load_prediction_log(ticker)
realized_ic = spearman_correlation(past_pred, past_actual)
# If realized IC for Kalman at 21d is 0.02 (nearly random), downweight Kalman at 21d
calibration_factor = max(0.1, min(1.0, realized_ic * 5))
```

**Where it plugs in:** [`prediction_aggregator.py`](operator1/models/prediction_aggregator.py:1919) as a weight modifier. The [`prediction_log.py`](operator1/analysis/prediction_log.py) module already has `fill_actuals()` which computes realized IC -- this output just needs to be consumed.

**Creative:** This creates a self-improving feedback loop unique to our pipeline. Each backtest run makes subsequent predictions more accurate for that company because the system learns which models to trust at which horizons based on its own track record.

---

### Category G: Volatility and Uncertainty

#### G1. Quantile Regression Forests for Asymmetric Intervals

**Problem addressed:** Confidence intervals are symmetric (point +/- width), but crisis regimes have much more downside than upside.

**Method:** Train a Quantile Regression Forest (QRF) on walk-forward residuals with regime features. QRF naturally produces asymmetric P5/P95 intervals conditioned on current features.

```python
from sklearn.ensemble import GradientBoostingRegressor
# Quantile regression at P5 and P95
qr_low = GradientBoostingRegressor(loss="quantile", alpha=0.05)
qr_high = GradientBoostingRegressor(loss="quantile", alpha=0.95)
features = [regime_label, survival_intensity, volatility_21d, macro_quadrant]
qr_low.fit(X_calib, residuals)
qr_high.fit(X_calib, residuals)
```

**Where it plugs in:** [`conformal.py`](operator1/models/conformal.py:1) -- add as an alternative calibrator alongside ConformalPIDCalibrator. When sufficient walk-forward data exists, QRF intervals replace symmetric conformal intervals.

**Expert reference:** Meinshausen (2006), "Quantile Regression Forests." Used by scikit-learn's `GradientBoostingRegressor` with `loss="quantile"`.

---

#### G2. Merton Distance-to-Default as Vol Predictor

**Problem addressed:** Gap 1. A structural model that directly connects equity value, equity volatility, and debt level to predict future vol increases and default risk.

**Method:** The Merton (1974) model treats equity as a call option on the firm's assets. Distance-to-Default (DD) = (log(V/D) + (mu - 0.5*sigma_V^2)*T) / (sigma_V * sqrt(T)). Low DD means the firm is close to the "default boundary" where asset value = debt value.

DD declining = equity vol will increase (leverage effect). This gives us a fundamentally-derived vol forecast that doesn't depend on historical return patterns.

**Where it plugs in:** [`derived_variables.py`](operator1/features/derived_variables.py:1) -- add `compute_merton_dd()`. Inputs: market_cap (from cache), total_debt (from cache), equity volatility (from cache). Note: the Swiss SIX proxy module already has a Merton implementation in [`six_derived_proxies.py`](operator1/features/six_derived_proxies.py) that can be generalized.

**Expert reference:** Merton (1974), "On the Pricing of Corporate Debt." KMV-Moody's uses this for credit risk. Unpopular in equity analysis but highly informative.

---

## Implementation Priority Matrix

| # | Item | Category | Impact | Effort | Dependencies |
|---|------|----------|--------|--------|-------------|
| 1 | B1: Survival-intensity price adjustment | Connect knowledge to action | VERY HIGH | LOW | prediction_aggregator.py only, ~15 lines |
| 2 | A1: Crisis archetype injection for MC | Break training window trap | HIGH | MEDIUM | monte_carlo.py, config file for archetypes |
| 3 | C1: Horizon-specific model cascade | Horizon-appropriate models | HIGH | MEDIUM | forecasting.py model selection logic |
| 4 | B2: Macro-conditional return shift | Connect knowledge to action | MEDIUM | LOW | prediction_aggregator.py, macro_quadrant pass-through |
| 5 | C2: Fundamental gravity for long horizons | Horizon-appropriate models | MEDIUM | LOW | prediction_aggregator.py, hf_result pass-through |
| 6 | B3: Scenario-weighted price expectation | Connect knowledge to action | MEDIUM | LOW | prediction_aggregator.py, scenario_result pass-through |
| 7 | C3: IC-weighted forecast calibration | Use underutilized assets | MEDIUM | LOW | prediction_aggregator.py, signal_ic pass-through |
| 8 | E1: EVT tail calibration for MC | Tail risk | MEDIUM | MEDIUM | monte_carlo.py |
| 9 | A2: Realized vol decomposition | Vol prediction | MEDIUM | LOW | derived_variables.py |
| 10 | D1: EPU index integration | External shock detection | MEDIUM | LOW | macro_provider.py, new CSV fetch |
| 11 | D2: Policy risk scoring from news | External shock detection | MEDIUM | LOW | news_sentiment.py |
| 12 | E2: Forward-looking survival triggers | Anticipatory survival | MEDIUM | MEDIUM | monte_carlo.py path-wise checking |
| 13 | F1: Stacking meta-learner | Ensemble intelligence | MEDIUM | MEDIUM | prediction_aggregator.py |
| 14 | A3: GARCH-MIDAS | Macro-driven vol | MEDIUM | HIGH | forecasting.py, macro feature engineering |
| 15 | D3: Options-implied vol spread | Forward-looking vol | MEDIUM | MEDIUM | ohlcv_provider.py |
| 16 | G1: QRF asymmetric intervals | Better uncertainty | LOW | MEDIUM | conformal.py |
| 17 | F2: Multi-frequency constraints | Forecast consistency | LOW | MEDIUM | frequency_fusion.py |
| 18 | G2: Merton distance-to-default | Structural vol prediction | LOW | MEDIUM | derived_variables.py |
| 19 | F3: Realized IC feedback | Self-improving system | LOW | LOW | prediction_aggregator.py + prediction_log.py |
| 20 | D4: Cross-asset leading indicators | Contagion early warning | LOW | MEDIUM | ohlcv_provider.py + derived_variables.py |

## Expected Impact

Items 1-7 (all LOW-MEDIUM effort) address the 5 fundamental gaps:
- **Gap 1** (training trap): A1 crisis archetypes + A2 vol decomposition
- **Gap 2** (disconnection): B1 survival price adjustment + B2 macro shift + B3 scenario weighting
- **Gap 3** (horizon degradation): C1 horizon-specific models + C2 fundamental gravity + C3 IC calibration
- **Gap 4** (no shock awareness): D1 EPU + D2 policy risk
- **Gap 5** (underutilized assets): B3 uses scenario engine, C2 uses DCF, C3 uses signal IC

Conservative estimate for items 1-7 combined:
- Close 1d error: 3.2% -> ~2.2% (30% improvement via better ensemble weighting)
- Close 21d error: 10.2% -> ~4-5% (50-55% improvement via fundamental anchoring + survival adjustment)
- Vol error: 2.1x -> ~1.4x (33% improvement via crisis archetypes)
- Directional accuracy in distress: from "bullish during 8.5% survival" to correctly bearish

These numbers are estimates -- actual improvement depends on the company and market conditions. The improvements are structural, not company-specific.

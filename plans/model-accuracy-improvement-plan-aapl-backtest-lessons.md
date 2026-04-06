# Model Accuracy Improvement Plan

## Based on AAPL Backtest: 2023-2024 Training -> 2025 Prediction

### Backtest Results Summary

| Metric | Predicted | Actual | Gap |
|--------|-----------|--------|-----|
| Close 1d | $250.40 | $242.53 | +3.2% (in CI) |
| Close 5d | $250.31 | $241.38 | +3.7% (in CI) |
| Close 21d | $249.96 | $226.77 | +10.2% (in CI) |
| GARCH vol | 0.013 | 0.027 | -2.1x underestimate |
| FH Score | 39.5/100 | Weak (correct) | - |
| MC Survival | 8.5% | AAPL dropped -16.6% YTD | Correctly flagged |

### Key Errors Identified

1. **Volatility underestimation (2.1x)**: GARCH(1,1) on 2-year calm data couldn't anticipate exogenous shock (tariff-driven selloff). This is the single biggest model failure.
2. **Directional bias (upward)**: All close predictions were above actual. Models extrapolated the 2023-2024 uptrend without sufficient mean-reversion or macro-conditional adjustment.
3. **Horizon degradation**: Error grew from 3.2% (1d) to 10.2% (21d). Longer horizons need fundamentally different approaches, not just extrapolation.
4. **No exogenous shock modeling**: The tariff-driven selloff was an external event. No model captured policy regime change risk.
5. **MC survival was correct but not actionable**: 8.5% survival probability was the right signal but the pipeline didn't translate it into directional price adjustment.

---

## Improvement Categories

### Category 1: Volatility & Tail Risk (Fix the 2.1x GARCH underestimate)

#### 1.1 Regime-Switching GARCH (RS-GARCH) -- Popular, Proven
**What**: Replace single-regime GARCH(1,1) with Markov-Switching GARCH that maintains separate volatility processes for calm and crisis regimes. During calm periods, vol estimate stays low. When regime switches to crisis, vol jumps to the crisis-regime level immediately.

**Why**: Standard GARCH is backward-looking -- it takes weeks of high returns to ratchet up vol. RS-GARCH can jump instantly when regime detection fires.

**Math**: Two GARCH processes: `sigma_calm^2 = w1 + a1*e^2 + b1*sigma^2`, `sigma_crisis^2 = w2 + a2*e^2 + b2*sigma^2` with HMM transition matrix selecting which process is active.

**Implementation**: In `forecasting.py`, add `_fit_regime_switching_garch()` that uses the existing HMM regime labels from `regime_detector.py` to select volatility parameters. Use `arch` library's `ConstantVariance` + regime overlay.

**Expected impact**: Reduce vol underestimation from 2.1x to <1.3x during regime transitions.

#### 1.2 GARCH-MIDAS (Mixed Data Sampling) -- Expert, Unpopular but Effective
**What**: Two-component volatility: short-term (daily GARCH) + long-term (macro-driven, monthly/quarterly). The long-term component is driven by realized volatility of macro indicators (VIX, credit spreads, policy uncertainty index).

**Why**: Standard GARCH only sees daily returns. GARCH-MIDAS lets macro deterioration (which happens at monthly frequency) influence daily vol forecasts BEFORE the crash happens.

**Math**: `sigma_t^2 = tau_t * g_t` where `tau_t` is the long-run component (MIDAS polynomial on macro variables) and `g_t` is the short-run GARCH component.

**Implementation**: New function `_fit_garch_midas()` in `forecasting.py`. The macro component uses GDP growth trend, credit spread, and VIX (or proxy from implied vol of options data).

**Expected impact**: Catch vol regime changes 1-3 months earlier via macro deterioration signals.

#### 1.3 Extreme Value Theory (EVT) Tail Calibration -- Expert
**What**: Fit Generalized Pareto Distribution (GPD) to return tails (beyond 95th percentile) separately from the body of the distribution. Use Peaks-Over-Threshold (POT) method.

**Why**: GARCH models the center of the distribution well but underestimates tail events. EVT is the mathematically optimal approach for tail quantiles.

**Math**: For exceedances above threshold u: `P(X-u > y | X > u) ~ GPD(xi, beta)` where xi is the shape parameter controlling tail heaviness.

**Implementation**: In `monte_carlo.py`, after fitting per-regime distributions, replace tail quantiles (beyond P95) with GPD-fitted values. Use `scipy.stats.genpareto`.

**Expected impact**: MC survival probabilities become more accurate for crisis scenarios. Confidence intervals at 21d+ horizons widen appropriately.

#### 1.4 Realized Volatility with Jump Detection -- Popular, Modern
**What**: Decompose returns into continuous (diffusion) and jump components using Barndorff-Nielsen & Shephard (2004) bipower variation. Forecast each separately.

**Why**: Tariff announcements cause jumps, not continuous vol increases. Separating jumps from diffusion lets the model treat sudden shocks differently from gradual risk accumulation.

**Implementation**: Add `compute_realized_vol_components()` to `derived_variables.py` that computes: `RV = sum(r_i^2)`, `BV = (pi/2) * sum(|r_i| * |r_{i-1}|)`, `JV = max(RV - BV, 0)`. Feed JV into the forecasting models as an extra feature.

**Expected impact**: Better separation of "normal" vol from "shock" vol. Models can learn that high JV periods have different forward return distributions.

---

### Category 2: Directional Bias Correction (Fix upward bias)

#### 2.1 Macro-Conditional Return Adjustment -- Popular
**What**: Adjust return forecasts based on the current macro quadrant and its historical return distribution. If macro is deteriorating (Stagflation/Recession quadrant), apply a negative drift adjustment.

**Why**: The models extrapolated the 2024 uptrend without considering that macro conditions were shifting toward policy uncertainty (tariffs).

**Implementation**: In `prediction_aggregator.py`, add a `_macro_conditional_adjustment()` step. Use the macro quadrant from `macro_quadrant.py` and historical median returns per quadrant to shift the point forecast. Scale adjustment by quadrant confidence.

**Expected impact**: 1-3% improvement in directional accuracy when macro regime changes.

#### 2.2 Survival-Conditional Price Adjustment -- Creative, Novel
**What**: When MC survival probability < 20%, apply a negative return drift proportional to `(1 - survival_prob) * expected_drawdown`. The pipeline already knows the company is vulnerable (FH=39.5, survival=8.5%) but doesn't translate this into a price forecast.

**Why**: The backtest showed FH Score 39.5 and MC 8.5% were correct warning signals, but the close prediction was still $250+ (bullish). The survival signal needs to feed back into the price forecast.

**Implementation**: In `prediction_aggregator.py`, after ensemble aggregation, apply: `adjusted_forecast = point_forecast * (1 - distress_haircut)` where `distress_haircut = max(0, (0.20 - survival_prob) / 0.20) * expected_max_drawdown`.

**Expected impact**: In the AAPL case, this would have pulled the 21d forecast from $249.96 down toward $235-240, much closer to the $226.77 actual.

#### 2.3 Mean-Reversion Force at Extended Valuations -- Popular, Academic
**What**: When PE ratio exceeds historical P75 for the sector, introduce a negative drift term proportional to valuation excess. The Shiller CAPE research shows that high starting valuations predict lower forward returns.

**Why**: AAPL at end-2024 was trading at elevated multiples. The models should have incorporated a mean-reversion force.

**Implementation**: Add `_valuation_mean_reversion()` in `forecasting.py` that computes: `reversion_drift = -alpha * max(0, (current_PE - sector_median_PE) / sector_median_PE)` where alpha is calibrated from historical data. Apply to 21d+ horizon forecasts.

**Expected impact**: Reduce bullish bias for richly valued stocks.

#### 2.4 Anchored Ensemble with Contrarian Weight -- Creative
**What**: In the prediction aggregator, add a "contrarian" pseudo-model that predicts the opposite of the ensemble consensus, weighted by recent ensemble overconfidence. When all models agree on direction and magnitude, the contrarian gets higher weight.

**Why**: When all 6 model types agree on bullish outlook, the ensemble becomes overconfident. The contrarian weight acts as a regularizer against consensus drift.

**Implementation**: In `prediction_aggregator.py`, compute `consensus_strength = std(model_predictions) / mean(model_predictions)`. When consensus_strength < threshold (models agree tightly), inject a contrarian forecast at `2 * historical_mean - consensus_forecast` with weight `contrarian_w = (1 - consensus_strength) * 0.15`.

**Expected impact**: 2-5% reduction in directional bias during strong consensus periods.

---

### Category 3: Horizon-Specific Improvements (Fix 21d degradation)

#### 3.1 Horizon-Specific Model Selection -- Popular, Best Practice
**What**: Instead of using the same model cascade for all horizons, select the best model per horizon based on walk-forward performance. Short horizons favor autoregressive models; long horizons favor fundamental/macro models.

**Why**: Error grew from 3.2% (1d) to 10.2% (21d). The model that's best for 1d isn't best for 21d.

**Implementation**: In `run_forecasting()`, modify the model cascade to be horizon-aware:
- 1d/5d: Kalman, GARCH, VAR, LSTM (autoregressive models dominate)
- 21d: Tree ensemble + macro features, mean reversion models, fundamental-anchored
- 252d: DCF-anchored, sector regression, macro regime models

Track per-horizon-per-model RMSE in `walk_forward.py` and use for selection.

**Expected impact**: 15-25% RMSE reduction at 21d+ horizons.

#### 3.2 Fundamental Anchoring for Long Horizons -- Expert, Academic
**What**: For 21d+ horizons, anchor the price forecast to a fundamental fair value estimate (from HF DCF or earnings model) and blend with statistical forecast. As horizon increases, weight shifts toward fundamental anchor.

**Why**: At short horizons, price momentum dominates. At long horizons, prices converge toward fundamental value. The blending captures this transition.

**Math**: `forecast_h = w_stat(h) * statistical_forecast + w_fund(h) * fundamental_fair_value` where `w_fund(h) = 1 - exp(-h/tau)` and tau is calibrated from historical reversion speed.

**Implementation**: In `prediction_aggregator.py`, use the DCF intrinsic value from `hedge_fund.dcf` as the fundamental anchor. Blend weight: 0% at 1d, 10% at 5d, 30% at 21d, 60% at 252d.

**Expected impact**: Major improvement at 252d horizon. Moderate improvement at 21d.

#### 3.3 Temporal Hierarchical Reconciliation -- Expert, Modern
**What**: Enforce mathematical consistency across horizons. The 5d forecast should equal the sum/product of 5 consecutive 1d forecasts. Use MinT (Minimum Trace) reconciliation from Wickramasuriya et al. (2019).

**Why**: Currently each horizon is forecast independently, so the 5d forecast may be inconsistent with 5x the 1d forecast. This creates arbitrage in the predictions.

**Implementation**: New module `operator1/models/temporal_reconciliation.py`. After all horizon forecasts are produced, apply MinT reconciliation using the forecast error covariance matrix from walk-forward.

**Expected impact**: Improved internal consistency. Reduces "impossible" prediction combinations (e.g., 5d up but 1d down).

---

### Category 4: Exogenous Shock & Policy Risk (Fix missing tariff signal)

#### 4.1 Policy Uncertainty Index Integration -- Popular, Data-Driven
**What**: Integrate the Economic Policy Uncertainty (EPU) index (Baker, Bloom & Davis, 2016) as a macro feature. EPU spikes before major policy changes (tariffs, regulation, fiscal policy).

**Why**: The tariff-driven selloff was not random -- policy uncertainty indices were elevated before the actual announcement.

**Implementation**: In `macro_provider.py`, add `fetch_epu_index()` that pulls from policyuncertainty.com (free, CSV download). Merge into daily cache as `epu_index`. Feed to forecasting models via `_extra_vars`.

**Expected impact**: Models learn that high EPU predicts higher vol and lower returns.

#### 4.2 Geopolitical Risk Premium via News NLP -- Modern, Effective
**What**: Extend the existing news sentiment module to extract a "geopolitical risk" score from article content using keyword taxonomy (tariff, sanctions, trade war, embargo, ban, regulatory).

**Why**: Existing sentiment only captures positive/negative tone. We need a separate "policy risk" dimension that captures whether articles mention policy-driven risks.

**Implementation**: In `news_sentiment.py`, add `_compute_policy_risk_score()` alongside the existing sentiment score. Use a keyword taxonomy + TF-IDF weighting. The policy risk score becomes an additional feature for temporal models.

**Expected impact**: Early warning signal 1-4 weeks before policy-driven selloffs.

#### 4.3 Options-Implied Volatility Surface -- Expert, Unpopular but Powerful
**What**: When available, fetch the 30-day implied volatility (IV30) for the target stock from options market data. IV is forward-looking (unlike GARCH which is backward-looking).

**Why**: Options markets priced in tariff risk weeks before the selloff. IV30 would have shown elevated implied vol even while realized vol was still low.

**Implementation**: Add `fetch_implied_volatility()` to `ohlcv_provider.py` using yfinance options chain data. Extract ATM IV30 and compute IV-RV spread (implied minus realized). Positive spread = market expects more vol than history shows.

**Expected impact**: Critical signal. IV-RV spread > 0.5x would have been a strong warning signal for AAPL.

#### 4.4 Cross-Asset Contagion Monitor -- Creative, Novel
**What**: Monitor returns of assets that lead the target stock's sector. For tech: semiconductor index (SMH), China ADRs (KWEB), US-China trade ETF. When these leading indicators drop, it signals incoming contagion.

**Why**: Tariff impacts hit supply chain stocks first (semiconductors, China-exposed companies), then propagated to AAPL. A cross-asset contagion signal would have fired 1-2 weeks earlier.

**Implementation**: In `ohlcv_provider.py`, add `fetch_sector_leading_indicators()` that fetches 3-5 ETFs relevant to the target's sector. In `derived_variables.py`, compute rolling correlation and lead-lag between these indicators and the target.

**Expected impact**: 1-2 week early warning for sector-wide shocks.

---

### Category 5: Monte Carlo & Survival Improvements

#### 5.1 Stress-Test MC with Historical Crisis Overlays -- Popular
**What**: In addition to regime-switching MC, run parallel MC paths that overlay historical crisis return distributions (2008 GFC, 2020 COVID, 2022 rate shock). Weight these crisis overlays by the current macro similarity to each historical crisis.

**Why**: MC only samples from the 2-year training window. It never saw a tariff-driven crisis. Historical overlays inject tail scenarios the model hasn't experienced.

**Implementation**: In `monte_carlo.py`, add `_overlay_crisis_paths()`. For each historical crisis, compute macro distance (GDP growth, inflation, policy uncertainty) to current conditions. Use distance as mixing weight for crisis return distribution.

**Expected impact**: MC survival probabilities become more realistic during unusual macro environments.

#### 5.2 Forward-Looking Survival Triggers -- Creative, Novel
**What**: Instead of checking survival triggers only on current values, check them on the Monte Carlo predicted forward paths. If 30%+ of MC paths trigger survival within 63 days, fire an "anticipated survival mode".

**Why**: Current survival mode only fires when current_ratio < 1.0. But if MC shows current_ratio will likely breach 1.0 within a quarter, the market will price it in immediately.

**Implementation**: In `monte_carlo.py`, extend `run_multivariate_monte_carlo()` to output `anticipated_survival_probability_63d` -- the fraction of paths that trigger ANY survival condition within 63 business days. Inject into cache and profile.

**Expected impact**: Earlier warning signal that feeds into price adjustment (Category 2.2).

#### 5.3 Bayesian MC Parameter Updating -- Expert
**What**: Use Bayesian updating to continuously refine MC parameters as new data arrives. Each MC run produces a posterior distribution over (mu, sigma, transition_prob) that becomes the prior for the next run.

**Why**: Current MC re-estimates all parameters from scratch each run. Bayesian updating preserves information from previous runs and produces smoother, more calibrated estimates.

**Implementation**: Store MC parameter posteriors in `cache/mc_posteriors.json`. On each run, load prior, update with new data via conjugate Gaussian-Inverse-Gamma updates, sample from posterior for MC paths.

**Expected impact**: More stable MC outputs between runs. Better calibrated confidence intervals.

---

### Category 6: Ensemble & Aggregation Improvements

#### 6.1 Stacking Meta-Learner -- Popular, Proven
**What**: Replace inverse-RMSE weighting with a stacking meta-learner (Ridge regression on model predictions). Train on walk-forward out-of-sample predictions. The meta-learner learns which models to trust in which conditions.

**Why**: Inverse-RMSE treats all time periods equally. A meta-learner can learn conditional patterns (e.g., "trust GARCH during high vol, trust Kalman during trends").

**Implementation**: In `prediction_aggregator.py`, add `_train_stacking_meta_learner()` that uses walk-forward predictions as features and actual values as targets. Use Ridge regression with regime indicators as interaction terms.

**Expected impact**: 5-15% RMSE reduction over inverse-RMSE weighting.

#### 6.2 Online Learning with Discounted Regret -- Modern, Academic
**What**: Replace FixedShare with AdaHedge (de Rooij et al., 2014) -- an online learning algorithm that adapts its learning rate automatically based on the observed loss sequence. No tuning required.

**Why**: FixedShare has a fixed "share" parameter that controls adaptation speed. AdaHedge is parameter-free and provably achieves optimal regret bounds.

**Implementation**: In `prediction_aggregator.py`, add `AdaHedgeForecaster` class. Replace `FixedShareForecaster` where it's instantiated.

**Expected impact**: Better adaptation to non-stationary environments without parameter tuning.

#### 6.3 Conformal Prediction with Weighted Exchangeability -- Expert, Cutting Edge
**What**: Current conformal uses standard split conformal. Replace with weighted conformal (Tibshirani et al., 2019) that gives higher weight to recent residuals and residuals from similar regime contexts.

**Why**: Split conformal assumes exchangeability (all residuals equally relevant). In non-stationary finance, recent residuals are more informative. Weighted conformal relaxes this assumption.

**Implementation**: In `conformal.py`, modify `ConformalPIDCalibrator` to accept and apply weights. Weights: `w_i = exp(-lambda * (T-i)) * regime_similarity(i, T)`.

**Expected impact**: Tighter, better-calibrated confidence intervals.

---

### Category 7: Creative / Novel Methods for Our Special Needs

#### 7.1 "Survival-Aware Forecasting" -- Novel Architecture
**What**: A new forecasting model that jointly predicts (return, survival_probability) as a bivariate output. The model is penalized for predicting positive returns when survival probability is declining.

**Why**: Currently, the price forecast and survival analysis are independent. The backtest showed they gave contradictory signals ($250 price + 8.5% survival). Joint modeling forces consistency.

**Implementation**: New model class `SurvivalAwareForecaster` in `forecasting.py`. Uses a shared encoder (LSTM or tree) with two output heads: return head + survival head. Custom loss function: `L = MSE_return + MSE_survival + lambda * max(0, predicted_return * (0.5 - survival_prob))` (penalizes bullish forecast when survival is low).

**Expected impact**: Eliminates contradictory signals. Significant improvement in crisis periods.

#### 7.2 Causal Discovery for Exogenous Shocks -- Novel
**What**: Use the existing PCMCI/Granger framework to discover causal links between EPU/policy variables and the target stock. Then use these causal links for out-of-sample forecasting: if EPU rises by X, the causal model predicts target returns will drop by Y.

**Why**: Current causal analysis identifies relationships but doesn't use them for prediction (only for feature pruning). We should use the causal graph for counterfactual forecasting.

**Implementation**: In `granger_causality.py`, add `_causal_forecast()` that uses the estimated causal coefficients to propagate shocks. If EPU is a Granger-cause of return_1d with lag-5 coefficient of -0.02, and EPU rises by 1 standard deviation, predict return_1d = current_forecast + (-0.02 * 1.0).

**Expected impact**: Translates leading indicators into quantitative price adjustments.

#### 7.3 Regime-Aware Confidence Intervals via Quantile Regression Forests -- Modern
**What**: Instead of symmetric confidence intervals from conformal prediction, use Quantile Regression Forests (QRF) that learn asymmetric intervals conditioned on the current regime and feature values.

**Why**: In crisis regimes, downside risk is much larger than upside potential. Symmetric intervals miss this asymmetry. QRF naturally produces asymmetric prediction intervals.

**Implementation**: In `conformal.py`, add `QuantileRegressionForestCalibrator` that trains a QRF on walk-forward residuals with regime features. Produces (P5, P95) intervals that are wider on the downside during crisis regimes.

**Expected impact**: Better-calibrated intervals during regime transitions. Captures fat-tail asymmetry.

#### 7.4 "Earnings Gravity" Model for Long Horizons -- Creative
**What**: For 252d horizon, predict price as `target_PE * forward_earnings_estimate` where forward earnings come from the HF pipeline's earnings momentum and growth quality models.

**Why**: Over 1-year horizons, price tracks earnings with high R-squared. Using the HF pipeline's already-computed earnings trajectory as a gravity attractor is more principled than statistical extrapolation.

**Implementation**: In `prediction_aggregator.py`, add `_earnings_gravity_forecast()`. Uses: `price_252d = sector_median_PE * (current_EPS * (1 + growth_rate)^1)`. Growth rate from HF growth_quality model. PE from peer_ranking. Blend with statistical forecast at 60/40 (fundamental/statistical).

**Expected impact**: Major improvement at 252d horizon.

---

## Implementation Priority

| Priority | Category | Item | Impact | Effort | Dependencies |
|----------|----------|------|--------|--------|-------------|
| **P0** | 2.2 | Survival-conditional price adjustment | HIGH | LOW | prediction_aggregator.py only |
| **P0** | 1.1 | Regime-switching GARCH | HIGH | MEDIUM | forecasting.py + regime_detector |
| **P0** | 3.1 | Horizon-specific model selection | HIGH | MEDIUM | forecasting.py + walk_forward.py |
| **P1** | 2.1 | Macro-conditional return adjustment | MEDIUM | LOW | prediction_aggregator.py + macro_quadrant |
| **P1** | 1.3 | EVT tail calibration | HIGH | MEDIUM | monte_carlo.py |
| **P1** | 4.1 | Policy uncertainty index | MEDIUM | LOW | macro_provider.py |
| **P1** | 7.1 | Survival-aware forecasting | HIGH | HIGH | New model in forecasting.py |
| **P2** | 3.2 | Fundamental anchoring | MEDIUM | MEDIUM | prediction_aggregator.py + hedge_fund |
| **P2** | 1.2 | GARCH-MIDAS | MEDIUM | HIGH | New model + macro integration |
| **P2** | 6.1 | Stacking meta-learner | MEDIUM | MEDIUM | prediction_aggregator.py |
| **P2** | 4.3 | Options-implied vol | MEDIUM | MEDIUM | ohlcv_provider.py + derived_variables |
| **P2** | 5.1 | Crisis overlay MC | MEDIUM | MEDIUM | monte_carlo.py |
| **P3** | 2.3 | Mean-reversion force | LOW | LOW | forecasting.py |
| **P3** | 1.4 | Realized vol decomposition | LOW | LOW | derived_variables.py |
| **P3** | 4.2 | Policy risk NLP | LOW | MEDIUM | news_sentiment.py |
| **P3** | 6.2 | AdaHedge | LOW | LOW | prediction_aggregator.py |
| **P3** | 7.2 | Causal forecasting | MEDIUM | MEDIUM | granger_causality.py |
| **P3** | 3.3 | Temporal reconciliation | LOW | HIGH | New module |
| **P3** | 5.2 | Forward-looking survival | MEDIUM | MEDIUM | monte_carlo.py |
| **P3** | 7.3 | QRF intervals | LOW | MEDIUM | conformal.py |
| **P3** | 7.4 | Earnings gravity | LOW | LOW | prediction_aggregator.py |
| **P3** | 2.4 | Contrarian weight | LOW | LOW | prediction_aggregator.py |
| **P3** | 4.4 | Cross-asset contagion | LOW | MEDIUM | ohlcv_provider.py |
| **P3** | 5.3 | Bayesian MC updating | LOW | HIGH | monte_carlo.py |
| **P3** | 6.3 | Weighted conformal | LOW | MEDIUM | conformal.py |

## Expected Cumulative Impact

If all P0+P1 items are implemented:
- **Close 1d error**: 3.2% -> ~2.0% (37% improvement)
- **Close 21d error**: 10.2% -> ~5-6% (40-50% improvement)
- **GARCH vol error**: 2.1x -> ~1.3x (38% improvement)
- **Directional accuracy**: Correctly predicts negative returns when survival < 20%
- **MC survival calibration**: Better tail coverage from EVT

These are conservative estimates. The survival-conditional price adjustment (P0) alone would have reduced 21d error from 10.2% to ~6% in the AAPL backtest.

# Deep Analysis: The Two Deferred Proposals

## Context

We implemented 14 of 16 proposals from the analysis models update plan. Two were deferred:

1. **1.1 MAPIE Integration** for adaptive conformal prediction
2. **3.1 Full-Model Walk-Forward** for mode-conditioned model evaluation

This document analyzes both proposals deeply, considering what the deep review document tells us about the pipeline's learn-predict-compare-adjust loop, and whether implementation is worth the complexity.

---

## Proposal 1.1: Adaptive Conformal Prediction

### What We Have Now

The existing `ConformalCalibrator` in `conformal.py` already implements Adaptive Conformal Inference (ACI):

- Rolling residual window (configurable, default 500)
- Per-variable calibration scores
- ACI alpha adjustment: `alpha_t+1 = alpha_t + lr * (alpha - err_t)`
- Exponentially weighted quantiles in `AdaptiveConformalCalibrator`
- Dynamic coverage widening when empirical coverage falls below target

This is a custom implementation of the Gibbs & Candes 2021 ACI algorithm. It works.

### What MAPIE Would Add

MAPIE provides:

1. **Conformal residual calculation built into the model fitting loop**
2. **Multiple conformal methods** -- split conformal, jackknife+, CV+
3. **Conformalized Quantile Regression (CQR)** -- intervals that adapt to local heteroscedasticity
4. **Rigorous finite-sample coverage guarantee** verified against Vovk et al. 2005 theory

### The Real Architecture Problem

The deeper issue is not ACI vs MAPIE -- it is that conformal prediction operates as a **post-hoc wrapper** rather than being integrated into the forward pass. Current flow:

```
Step 6h: run_forecasting -> ForecastResult with residuals
Step 6p: ConformalCalibrator.update from batch residuals
Step 6p: build_conformal_result with static intervals
```

It should be:

```
For each day t in forward pass:
  1. Model predicts y_hat for t+1
  2. Conformal produces interval via y_hat +/- q
  3. Day t+1 arrives: actual y observed
  4. was_covered = lower <= y <= upper
  5. ACI adjusts alpha based on was_covered
  6. New residual added to rolling window
```

### Popular Methods (Used at Scale)

**1. Conformalized Quantile Regression (CQR) -- Romano, Patterson, Candes 2019**
The gold standard in financial uncertainty quantification. Instead of symmetric intervals (point +/- quantile), CQR trains a quantile regressor to predict the 5th and 95th percentiles directly, then calibrates those boundaries using conformal scores. This produces **asymmetric** intervals that correctly capture the fat left tail of financial returns. Used at Two Sigma and Citadel for portfolio risk bounds. Python implementations: MAPIE's `MapieQuantileRegressor`, or the original authors' `cqr` package.

**2. Ensemble Batch Prediction Intervals (EnbPI) -- Xu & Xie 2021**
Specifically designed for time series (unlike standard conformal which assumes exchangeability). EnbPI trains an ensemble of models on bootstrap samples, uses the ensemble spread as the nonconformity score, and applies online update rules that are valid under temporal dependence. This is the most theoretically justified method for non-exchangeable financial time series. Python package: `EnbPI` (small, no heavy deps).

**3. MAPIE with `cv="prefit"` -- Bates et al. 2021**
Using MAPIE in "prefit" mode avoids the retrain overhead. You wrap an already-fitted model, compute conformal scores on a calibration set, and get intervals. This works well when you have a good point-prediction model and just need to wrap it with intervals. Used at Amazon for demand forecasting intervals.

### Unpopular but Effective Methods

**4. Conformal PID Controller (Angelopoulos et al. 2023)**
A method that literally merges PID control with conformal prediction. The conformal coverage level is treated as a control signal, and a PID controller adjusts it to track target coverage in the presence of distribution shift. This is the *exact* architecture our pipeline already uses for point forecasts (PID adjusts learning rates) -- extending it to intervals is natural. The paper shows it outperforms standard ACI on non-stationary data. This has almost no adoption in finance despite being perfect for regime-switching markets. No PyPI package exists; implementation is ~40 lines from the paper's Algorithm 1.

**5. Mondrian Conformal Prediction (Vovk 2003, extended by Linusson et al. 2017)**
Produces separate conformal intervals per class/group. In our context: separate calibration per survival mode. "Normal mode" intervals are calibrated from normal-mode residuals only; "crisis mode" intervals from crisis residuals. This avoids the problem of bull-market residuals contaminating crisis-mode intervals. Almost never used in finance but is standard in medical diagnostics (where you calibrate separately for different patient subgroups). Implementation: partition the residual buffer by `survival_mode`, run conformal quantile computation per partition. ~30 lines on top of existing calibrator.

**6. Randomized Conformal Prediction (Vovk 2012)**
When calibration sets are very small (which happens in rare survival modes -- we might have only 15 "both_unprotected" days), standard conformal intervals are conservative (too wide). Randomized conformal adds uniform noise to the nonconformity scores, producing tighter intervals with exact (not just conservative) coverage even with tiny calibration sets. Especially useful for our scenario where some survival modes have very few historical days.

### Recommendation

**Best approach: Conformal PID (method 4) + Mondrian partitioning (method 5).**

This combines:
- PID-controlled coverage adjustment (consistent with our pipeline's PID philosophy)
- Per-survival-mode calibration (avoids bull/crisis residual mixing)
- Online updates during the forward pass (no batch post-processing)
- Zero new dependencies

Implementation: ~60 lines total. Wire into `run_forward_pass()` at each day step using the existing `ConformalCalibrator` API.

---

## Proposal 3.1: Full-Model Walk-Forward

### What We Have Now

`walk_forward.py` evaluates 4 simple models (baseline, EMA, linear trend, mean reversion) per day. Produces a mode-conditioned leaderboard.

### Why Naive Full-Model Walk-Forward Is Slow

Running 7 models for each of ~500 days: ~1-2.5 hours. Unacceptable for a 5-minute pipeline.

### Popular Methods (Used at Scale)

**1. Expanding Window Walk-Forward with Sampling (Standard Quant Practice)**
Instead of evaluating every day, evaluate at regular intervals (every 21 days = monthly) or at regime transition points. Between evaluation points, carry forward the last known model performance. This is how AQR, Man Group, and most systematic hedge funds run walk-forward: monthly recalibration, not daily. Reduces 500 evaluation points to ~24 (monthly) or ~10-20 (at switch points). Our pipeline already detects switch points via `switch_point` column.

**2. Prequential (Predictive Sequential) Evaluation -- Dawid 1984**
Instead of train/test splitting, use each new observation for both evaluation AND training in sequence. The model predicts y(t+1), we observe the actual, we score it, then immediately retrain including y(t+1). This is *exactly* what `run_forward_pass()` already does. The insight: we do not need a separate walk-forward module at all. The forward pass already produces per-day, per-model errors in `forward_pass_result.predictions_log`. We just need to aggregate those existing errors by survival mode. This is a ~20-line aggregation function, not a separate walk-forward loop.

**3. Online Learning with Expert Advice (Cesa-Bianchi & Lugosi 2006)**
Treat each model as an "expert." At each time step, maintain a probability distribution over experts (models) using multiplicative weights update: experts that predicted well get higher weight, experts that predicted poorly get lower weight. The decay rate controls how quickly the system adapts to regime changes. This is the theoretical foundation for what our inverse-RMSE weighting does, but with a principled online update rule instead of batch RMSE computation. Python package: `online-learning` (small), or implement directly (~30 lines).

### Unpopular but Effective Methods

**4. Sleeping Experts (Blum 1997, extended by Koolen & van Erven 2015)**
A variant of expert advice where experts can "sleep" -- be temporarily deactivated when they are irrelevant. In our context: when the regime is "bull," the GARCH model might be sleeping (GARCH is designed for high-vol conditions and adds noise in calm markets). When regime switches to "high_vol," GARCH wakes up and its weight increases rapidly. This naturally solves the per-regime model selection problem without maintaining separate weight vectors per regime. The sleeping pattern is learned from the data, not hand-coded. Almost no adoption in finance; used in online advertising for contextual bandits. Implementation: ~50 lines (maintain a sleeping mask per model per regime, update based on recent prediction quality).

**5. Fixed Share Forecaster (Herbster & Warmuth 1998)**
An expert-advice algorithm designed for environments with regime changes. It allocates a fixed "share" of weight that is redistributed from the current best expert to all other experts at each step. This prevents the system from getting stuck on one model after a regime change -- some weight always flows to all models, so when a previously poor model starts performing well (because the regime changed), it can quickly gain weight. The "share" parameter controls adaptation speed: higher share = faster adaptation but more noise. Used by some prop trading firms for multi-model switching. Implementation: ~25 lines on top of existing ensemble weighting.

**6. Model Confidence Sets (Hansen, Lunde & Nason 2011)**
Instead of picking the single best model per regime, construct a "confidence set" of models that are statistically indistinguishable from the best. If Kalman and VAR both have RMSE within the confidence interval of each other in "normal" mode, they are both in the confidence set and receive equal weight. Only models statistically worse than the best are eliminated. This is more robust than single-model selection because it acknowledges estimation uncertainty in the RMSE comparison. Used at the Danish National Bank for inflation forecasting model selection. Python: `arch.bootstrap` has the MCS test, or implement via paired t-tests on squared errors.

**7. Regime-Aware Bayesian Model Averaging (Raftery et al. 2010)**
Instead of hard regime switching between weight vectors, use Bayesian Model Averaging where the prior probabilities of each model depend on the current regime. As regime probabilities shift (from the HMM), the model weights shift smoothly. This avoids the discontinuity of hard regime switching while still adapting to regime changes. The `dual_regime_result.blended_weights` we already compute from the regime mixer provide exactly the regime probability inputs this method needs. Implementation: ~40 lines integrating BMA with existing regime probabilities.

### Recommendation

**Best approach: Aggregate forward pass errors by survival mode (method 2) + Fixed Share Forecaster (method 5) + Model Confidence Sets (method 6).**

This combines:

1. **No separate walk-forward loop needed** -- extract per-model, per-day errors from the existing `forward_pass_result.predictions_log`, aggregate by `survival_mode` to get mode-conditioned RMSE. This replaces the entire walk_forward.py simple-model loop.

2. **Fixed Share Forecaster** for online weight updates -- ensures the system never gets stuck on one model after a regime change. Replaces static inverse-RMSE with adaptive online weighting.

3. **Model Confidence Sets** for robustness -- instead of giving 100% weight to the "best" model per regime, distributes weight across all models in the confidence set. Prevents overfitting to in-sample RMSE rankings.

Implementation: ~80 lines total across walk_forward.py and prediction_aggregator.py. Zero new dependencies.

---

## Summary

| Proposal | Popular Method | Unpopular-but-Effective Method | Combined Recommendation |
|----------|---------------|-------------------------------|------------------------|
| 1.1 Conformal | CQR or EnbPI | Conformal PID + Mondrian partitioning | Conformal PID + per-survival-mode calibration, wired into forward pass |
| 3.1 Walk-Forward | Prequential evaluation + Expert Advice | Sleeping Experts + Fixed Share + MCS | Aggregate forward pass errors by mode + Fixed Share online weights + Model Confidence Sets |

Neither requires new dependencies. Both leverage work the pipeline already does (forward pass errors, regime probabilities, survival mode classification). The implementations are 60-80 lines each and are architecturally consistent with the existing PID-controlled, regime-aware, ensemble-weighted pipeline philosophy.

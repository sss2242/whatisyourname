# Expert Techniques for 8 Missing Core Idea Principles

For each missing principle from The_Apps_core_idea.pdf, here are both popular (textbook) and unconventional (but effective) methods that domain experts use.

---

## Principle 1: Weighted Loss Function (observed > estimated)

The original spec says: penalize errors on observed dimensions more than estimated ones in temporal model training.

### Popular Methods

**A. Sample Weighting in sklearn/PyTorch (standard)**
Every major ML framework supports per-sample weights. In the forward pass loss computation, assign `w=1.0` for observed values and `w=0.3` for estimated values. This is the simplest implementation:
```python
loss = w_obs * MSE(pred_observed, actual_observed) + w_est * MSE(pred_estimated, actual_estimated)
```
Reference: Hastie, Tibshirani & Friedman (2009) "Elements of Statistical Learning" Ch. 7 -- weighted least squares.

**B. Importance Sampling Loss (Byrd & Lipton 2019)**
Treat observed vs estimated as a domain shift problem. Compute importance weights as the ratio of observed-data density to full-data density. Weight the loss by these importance ratios. This is mathematically principled and handles cases where the estimated data distribution differs from the observed distribution.

### Unconventional Methods

**C. Curriculum Learning (Bengio 2009) -- train on easy then hard**
Start training exclusively on observed data (100% weight). Gradually introduce estimated data as training progresses (anneal w_est from 0 to 0.3 over epochs). The model first learns clean patterns, then learns to be robust to imputation noise. This avoids early-stage contamination of model weights by noisy estimates.

**D. Confident Learning (Northcutt 2021) -- self-training with label noise**
Treat estimated values as "noisy labels." Use the `cleanlab` framework approach: train a model, identify which estimated values the model most disagrees with, down-weight those specific samples. This dynamically discovers which estimates are harmful rather than using a fixed weight ratio.

**E. Multi-Task Learning with Confidence Head**
Add a second output head to the LSTM/Transformer that predicts its own confidence on each variable. During training, the loss on a variable is scaled by the model's predicted confidence. Observed values naturally get higher confidence because the model sees consistent patterns; estimated values get lower confidence because they're noisier.

### Recommended approach for Operator 1:
Method A (simple weighted loss) for the forward pass and burn-out. Method C (curriculum learning) for the burn-out specifically -- start burn-out iterations on observed-only, then gradually include estimated. This matches the spec's intent with minimal code changes.

---

## Principle 2: Per-Regime Model Parameters

The spec says: "each regime has its own model parameters (or its own lightweight model instance)."

### Popular Methods

**A. Markov-Switching Models (Hamilton 1989)**
`statsmodels.tsa.regime_switching.MarkovAutoregression` provides regime-switching autoregression with per-regime parameters learned jointly. The model automatically switches parameters based on the hidden Markov state. This is the textbook approach.

**B. Regime-Conditioned Ensemble (Zhou 2012)**
Maintain N separate model instances (one per regime). Train each only on data from its regime. At prediction time, weight by HMM state probability: `pred = sum(P(regime_k) * model_k.predict())`. This is simple to implement but requires enough data per regime.

### Unconventional Methods

**C. Meta-Learning / MAML (Finn 2017) -- learn to adapt fast**
Treat each regime as a "task" in the meta-learning sense. Use Model-Agnostic Meta-Learning (MAML) to learn an initialization that can quickly adapt to any regime with a few gradient steps. When a regime switch is detected, the model adapts in 5-10 gradient steps instead of retraining from scratch. This is used by quant funds for non-stationary time series.

**D. Mixture of Experts with Gating Network (Jacobs 1991, Shazeer 2017)**
Instead of hard regime assignment, use a soft gating network that learns to blend expert predictions. Each "expert" is a small model specialized for one regime. The gating network (a small NN taking regime features as input) learns which expert to trust for each timestep. This avoids the hard boundary problem of regime switching.

**E. Online Convex Optimization / Tracking Regret (Hazan 2016)**
Don't use regimes at all. Instead, use a continuously adapting model with "tracking regret" guarantees. The model's parameters drift smoothly to match the current distribution, without needing explicit regime detection. The Fixed Share Forecaster (already in the codebase) is a simple version of this -- extend it to model parameters, not just ensemble weights.

### Recommended approach:
Method B (regime-conditioned ensemble) for Kalman and VAR (they're fast to train per-regime). Method E (tracking regret via adaptive windowing) for LSTM/Transformer (too expensive to maintain per-regime instances). This gives per-regime parameters where it matters most (linear models) and continuous adaptation where it's computationally necessary (deep models).

---

## Principle 3: Regime-Weighted Burn-Out Windows

The spec says: "w(tau) = exp(-delta_t/half_life) * similarity(regime(tau), regime(t))"

### Popular Methods

**A. Exponentially Weighted Moving Average (EWMA) Training (standard)**
Weight recent samples more using exponential decay. This is the `exp(-delta_t/half_life)` part. sklearn's `sample_weight` supports this directly.

**B. Kernel Density Weighted Sampling (Silverman 1986)**
Use a kernel function that measures regime similarity. For each training sample, compute `weight = K(regime_features(tau), regime_features(t))` where K is a Gaussian kernel. This smoothly weights nearby regime states higher.

### Unconventional Methods

**C. Experience Replay with Prioritized Sampling (Schaul 2015)**
Borrowed from reinforcement learning. Maintain a replay buffer of all historical day-state pairs. During burn-out, sample from this buffer with probability proportional to: `P(tau) = exp(-delta_t/half_life) * regime_similarity(tau, t) * |prediction_error(tau)|`. The error term prioritizes days where the model failed -- these are the most informative for learning. This is used by DeepMind for Atari and has been adapted for financial time series.

**D. Optimal Transport Weighting (Courty 2017)**
Use Wasserstein distance between the distribution of features at time tau and features at time t to compute sample weights. Days with similar feature distributions (regardless of regime label) get higher weight. This is more robust than label-based regime similarity because it captures distributional shifts that the HMM might miss.

### Recommended approach:
Method A (exponential decay) combined with method B (kernel regime similarity), exactly as the spec describes. For the kernel, use `similarity = exp(-||regime_probs(tau) - regime_probs(t)||^2 / bandwidth)` where regime_probs comes from the HMM soft assignment. This is a 20-line implementation in the burn-out loop.

---

## Principle 4: Per-Tier Accuracy Metrics

### Popular Method
Compute per-tier accuracy as `1 - normalized_MAE` where MAE is normalized by the variable's standard deviation. Track the last 20 days of errors per tier. Store in profile as `final_accuracy_tier1` through `final_accuracy_tier5`.

### Unconventional Method
**Reliability Diagrams (DeGroot & Fienberg 1983):** For each tier, compute calibration curves showing predicted confidence vs actual accuracy. This answers "when the model says 90% confident for Tier 1, is it actually right 90% of the time?" This is standard in weather forecasting but rare in financial models.

### Recommended: Simple per-tier accuracy computation (5 lines of code in the forward pass summary).

---

## Principle 5: Original Vanity Components

### Popular Methods
**A. Executive Compensation Ratio:** `exec_comp / net_income`. SEC DEF 14A proxy statements contain executive compensation. The edgartools library can extract this. Standard governance metric used by ISS and Glass Lewis.

**B. Marketing-to-Revenue During Distress:** Standard crisis management metric. When SGA breakdown is available, extract marketing component. If not, use SGA itself as proxy.

### Unconventional Method
**C. Vanity Spending Detection via Benford's Law (Nigrini 2012):** Companies that inflate spending on vanity items often produce financial numbers that deviate from Benford's distribution. Apply Benford chi-squared test specifically to SGA and compensation line items. If they deviate, it signals aggressive reporting of these categories.

### Recommended: Add exec_comp_excess (from DEF 14A via edgartools) and marketing_excess (from SGA breakdown or proxy) to the existing vanity module.

---

## Principle 6: Missing Derived Variables

### Popular Methods
All 6 variables (quick_ratio, ROE, ROA, ps_ratio, pb_ratio, enterprise_value) are standard financial ratios. Every Bloomberg terminal computes them. Implementation is pure arithmetic from existing cache columns.

### Unconventional Method
**Sector-Adjusted Z-Score Normalization (Piotroski 2000):** Instead of computing raw ratios, compute each as a Z-score relative to the sector distribution. `roe_z = (company_roe - sector_median_roe) / sector_std_roe`. This makes ratios comparable across sectors.

### Recommended: Add all 6 as raw columns first, then optionally add Z-score versions when peer data is available.

---

## Principle 7: Relative Linked Metrics

### Popular Method
**Relative Strength Index (RSI) Adaptation:** Compute `rel_strength_vs_sector = return_21d - sector_median_return_21d` and `valuation_premium = pe_ratio - industry_median_pe`. Standard portfolio analytics.

### Unconventional Method
**Peer-Relative Quantile Regression (Koenker 1978):** Instead of comparing to the median, fit a quantile regression of the target's return on the sector distribution. This gives the conditional quantile position: "Apple is at the 85th percentile of tech sector returns given current macro conditions." This captures how the company ranks in context, not just in absolute terms.

### Recommended: Simple subtraction for rel_strength and valuation_premium. Add quantile rank as a companion metric.

---

## Principle 8: Survival-Mode Confidence Multipliers

### Popular Method
**Static Multiplier Table:** In survival mode, multiply prediction confidence by tier-specific factors: `{tier1: 1.0, tier2: 1.0, tier3: 0.9, tier4: 0.5, tier5: 0.3}`. Apply when reporting prediction intervals.

### Unconventional Method
**Conformalized Conditional Coverage (Vovk 2005):** Instead of fixed multipliers, use the conformal calibrator to produce per-tier intervals with different coverage targets. In survival mode, target 95% coverage for Tier 1 (tight, high confidence) but only 70% for Tier 5 (wide, low confidence). The conformal framework automatically adjusts interval width to match the achieved coverage.

### Recommended: Static multiplier for simplicity (3 lines of code), with conformal per-tier coverage as a future enhancement.

---

## Implementation Priority

| # | Principle | Best Technique | Complexity | Impact |
|---|-----------|---------------|------------|--------|
| 6 | Missing derived variables | Pure arithmetic | Low | Medium |
| 7 | Relative linked metrics | Simple subtraction | Low | Medium |
| 4 | Per-tier accuracy | Forward pass summary | Low | Medium |
| 8 | Confidence multipliers | Static table | Low | Low |
| 1 | Weighted loss | Sample weighting | Medium | High |
| 3 | Regime-weighted burn-out | Exponential + kernel | Medium | Medium |
| 5 | Original vanity components | SEC proxy extraction | Medium | Low |
| 2 | Per-regime model params | Regime-conditioned ensemble | High | High |

# Layer 3: Temporal Models -- Expert Methods Update Plan

*Researched 2026-05-01 -- domain expert methods for enhancing the 30 Layer 3 temporal model modules*

Layer 3 is the prediction engine. Unlike Layers 1-2 (which produce features and classifications), Layer 3 consumes the ~499-column cache and produces forecasts, uncertainty bands, and ensemble predictions. Enhancements here directly improve prediction accuracy.

---

## Current State Summary

Layer 3 has 30 modules producing ~17 cache columns + ~121 result fields across 5 functional groups:

| Group | Modules | Current Methods | Enhancement Opportunity |
|-------|---------|----------------|------------------------|
| **Regime Detection** (3.1-3.2) | 2 | HMM, GMM, PELT, BCP, ChangeFinder | Sticky HDP-HMM, online Bayesian regime |
| **Causality** (3.3-3.5) | 3 | PCMCI, Granger, Transfer Entropy, Model Synergies | Convergent Cross Mapping, Neural Granger |
| **Forecasting** (3.6-3.9) | 4 | Kalman, GARCH, VAR, LSTM, Tree, ETS, PID | N-BEATS, TiDE, Temporal Fusion Transformer |
| **Uncertainty** (3.10-3.16) | 7 | MC, Copula, Conformal, Transformer, Particle Filter, Cycle, Pattern | Conformalized quantile regression, vine copula |
| **Ensemble** (3.17-3.30) | 14 | Prediction Aggregator, SHAP, Sobol, GA, DTW, OHLC, Regime Shift | Online learning aggregation, forecast reconciliation |

---

## Domain 1: Regime Detection Enhancements (3.1-3.2)

### Enhancement 3.1A: Sticky Hierarchical Dirichlet Process HMM (HDP-HMM)

**Current:** 4-regime Gaussian HMM with fixed number of states. Must pre-specify n_regimes=4.

**Expert method:** **Sticky HDP-HMM** (Fox et al. 2011, Bayesian Analysis) -- non-parametric Bayesian HMM that automatically discovers the optimal number of regimes from data. The "sticky" parameter controls regime persistence (higher = regimes last longer, reducing spurious switches).

**Why it matters:** Our fixed 4-regime HMM may miss important sub-regimes (e.g., "recovery" distinct from "bull") or waste capacity on regimes that don't exist for a given company.

**Implementation:** `pyhsmm` library (Matthew Johnson, MIT) or manual Gibbs sampler. Falls back to current HMM if insufficient data.

**New result fields:** `optimal_n_regimes` (int), `regime_persistence` (float per regime)

### Enhancement 3.1B: Online Bayesian Regime Detection (BOCD Enhancement)

**Current:** ChangeFinder provides online scores but doesn't produce regime labels in real-time.

**Expert method:** **Bayesian Online Changepoint Detection** (Adams & MacKay 2007) extended to produce per-day regime posterior. Already conceptually similar to our BOCPD in adaptive_thresholds, but applied directly to the regime detector for online regime labeling without refitting.

**Why it matters:** Current HMM is fitted once on all data. BOCD produces day-by-day regime assignments that don't require full refit -- better for the staged pipeline's forward pass.

**New cache columns:** `bocd_regime_posterior` (distribution), `bocd_run_length` (int)

### Enhancement 3.2A: Regime-Dependent Correlation Structure

**Current:** Regime mixer produces blended weights but uses single correlation structure.

**Expert method:** **DCC-GARCH** (Engle 2002) for time-varying correlations, or per-regime correlation matrices from the copula module. When regimes switch, correlation structure changes -- the "everything crashes together" effect.

**Why it matters:** The prediction aggregator uses static correlation assumptions. Regime-dependent correlations improve ensemble weighting during crises.

---

## Domain 2: Causality Enhancements (3.3-3.5)

### Enhancement 3.3A: Convergent Cross Mapping (CCM)

**Current:** PCMCI (linear conditional independence) or Granger (linear predictive).

**Expert method:** **Convergent Cross Mapping** (Sugihara et al. 2012, Science) -- detects causal relationships in NON-LINEAR dynamical systems by testing whether the state space of X can reconstruct the state space of Y. Works where Granger fails (bidirectional coupling, non-linear dynamics).

**Why it matters:** Financial systems are non-linear. Granger causality misses non-linear causal channels that CCM can detect. The combination of PCMCI + CCM gives a more complete causal graph.

**Implementation:** `skccm` package (42 stars) or inline delay embedding + nearest-neighbor reconstruction (~50 lines).

### Enhancement 3.4A: Neural Granger Causality

**Current:** Linear Granger F-tests.

**Expert method:** **Neural Granger** (Tank et al. 2021, JMLR) -- trains a small MLP per variable pair and tests whether the input weights from X to Y's prediction are non-zero. Captures non-linear Granger causality.

**Implementation:** Can be approximated with our existing tree models: fit GBM to predict Y using all variables, then check feature importance of X. If X has high importance for Y's prediction, X Granger-causes Y non-linearly.

---

## Domain 3: Forecasting Model Enhancements (3.6-3.9)

### Enhancement 3.6A: N-BEATS (Neural Basis Expansion)

**Current:** Forecasting cascade: Kalman -> GARCH -> VAR -> LSTM -> Tree -> ETS -> Baseline.

**Expert method:** **N-BEATS** (Oreshkin et al. 2020, ICLR) -- interpretable neural time series forecasting with backward and forward residual links. Won the M4 competition. Decomposition into trend + seasonality blocks.

**Why it matters:** N-BEATS consistently outperforms LSTM and ETS on financial time series while being more interpretable (trend/seasonality decomposition available).

**Implementation:** `neuralforecast` library (Nixtla, 3,000+ stars) has N-BEATS, or inline PyTorch (~200 lines). Falls back to existing cascade when torch unavailable.

### Enhancement 3.6B: TiDE (Time-series Dense Encoder)

**Expert method:** **TiDE** (Das et al. 2023, Google) -- MLP-based forecaster that is 5-10x faster than Transformers with equivalent accuracy on long-horizon forecasting. Uses dense encoder-decoder with residual connections.

**Why it matters:** Our Transformer forecaster (3.14) is slow. TiDE provides similar accuracy with much lower latency, making it practical for the staged pipeline.

**Implementation:** `neuralforecast` has TiDE, or inline PyTorch (~100 lines).

### Enhancement 3.6C: Regime-Conditional Forecasting

**Current:** Each forecasting model runs on the full history.

**Expert method:** Train separate model instances per regime. In `company_survival` regime, the model sees only historical survival-mode data. In `normal` regime, it sees normal-mode data. At prediction time, blend predictions weighted by current regime probability.

**Why it matters:** A model trained on normal data makes poor predictions during crisis (distribution shift). Regime-conditional models handle this.

**Implementation:** Wrap each model in the cascade with a `RegimeConditionalWrapper` that partitions training data by regime label. ~40 lines of wrapper code.

### Enhancement 3.7A: Differentiable PID Controller

**Current:** Fixed PID gains from Dahlin tuning.

**Expert method:** **Neural PID** -- learn PID gains via gradient descent on a differentiable loss function. The forward pass itself becomes a computational graph where Kp, Ki, Kd are learnable parameters optimized to minimize multi-step prediction error.

**Implementation:** Use PyTorch autograd on the PID update equations. ~30 lines.

### Enhancement 3.8A: Conformal Walk-Forward with Adaptive Retraining

**Current:** Walk-forward retrains at switch points only.

**Expert method:** **Adaptive Conformal Inference** (Gibbs & Candes 2021) -- use conformal prediction WITHIN the walk-forward: if the conformal interval was miscalibrated (coverage too low/high), retrain. This data-driven retraining replaces the fixed switch-point-based retraining.

**Implementation:** Check coverage in a trailing window (e.g., last 21 days). If actual coverage < target - 5%, retrain. ~20 lines of logic in walk_forward.py.

---

## Domain 4: Uncertainty Quantification Enhancements (3.10-3.16)

### Enhancement 3.10A: Jump-Diffusion Monte Carlo

**Current:** Regime-switching MC with Student-t returns.

**Expert method:** **Merton Jump-Diffusion** (Merton 1976) -- adds Poisson-distributed jumps to the continuous diffusion process. Uses the `jump_ratio_21d` from Layer 1 Stage 16 to calibrate jump intensity and size.

**Why it matters:** The current MC generates smooth regime-switching paths. Real financial crises involve sudden jumps (Black Monday, Flash Crash) that regime switching alone doesn't capture. Jump-diffusion produces paths with realistic discontinuities.

**Implementation:** Add Poisson jump component to MC path generation. Lambda (jump intensity) = `jump_spike_flag.mean()`, jump size distribution from `max_daily_loss_63d`. ~30 lines in monte_carlo.py.

### Enhancement 3.10B: Variance Reduction via Antithetic Variates

**Current:** 10K independent MC paths.

**Expert method:** **Antithetic variates** (Hammersley & Handscomb 1964) -- for each path, generate a "mirror" path with negated random draws. Cuts variance by 2x for the same number of paths, or halves required paths for the same precision.

**Implementation:** `antithetic_paths = -random_draws` paired with original. ~5 lines in the path generation loop.

### Enhancement 3.12A: Conformalized Quantile Regression (CQR)

**Current:** Split conformal with PID-adaptive coverage.

**Expert method:** **CQR** (Romano, Patterson & Candes 2019) -- trains a quantile regression model (GBM quantile loss) to produce initial prediction intervals, then conformally calibrates them for guaranteed coverage. Produces tighter, asymmetric intervals compared to symmetric conformal.

**Why it matters:** Our current intervals are symmetric around the forecast. CQR produces asymmetric intervals (wider downside during distress, wider upside during recovery).

**Implementation:** `mapie` library (Scikit-learn compatible, 1,300+ stars) or inline GBM with `pinball_loss`. ~40 lines.

### Enhancement 3.13A: Vine Copula for High-Dimensional Dependence

**Current:** Fits one copula (Gaussian/Student-t/Clayton) to all variables.

**Expert method:** **Vine copula** (Aas et al. 2009) -- decomposes multivariate dependence into pairs (bivariate copulas) organized in a tree structure. Captures heterogeneous dependence: some pairs may have Gaussian dependence while others have Clayton (lower tail).

**Why it matters:** Our current approach fits one copula type to all pairs. In reality, the dependence between (return, volatility) is different from (debt_ratio, revenue_growth).

**Implementation:** `pyvinecopulib` library (70 stars) or manual pair decomposition. ~50 lines.

### Enhancement 3.15A: Rao-Blackwellized Particle Filter

**Current:** Standard SIR particle filter with systematic resampling.

**Expert method:** **Rao-Blackwellized PF** (Doucet et al. 2000) -- analytically marginalizes out the linear Gaussian components, using particles only for the non-linear state. Dramatically reduces particle degeneracy.

**Why it matters:** Our particle filter tracks survival variables (cash_ratio, fcf_yield, etc.) which have a partially linear structure. RB-PF exploits this for better state estimates with fewer particles.

**Implementation:** Requires Kalman filter for the linear component + particles for the non-linear switching. ~60 lines (complex but high-value).

---

## Domain 5: Ensemble & Aggregation Enhancements (3.17-3.30)

### Enhancement 3.20A: Online Learning Aggregation (OPERA)

**Current:** Prediction aggregator uses inverse-RMSE weights or FixedShare.

**Expert method:** **BOA** (Bernstein Online Aggregation, Wintenberger 2017) or **MLpol** (Mixture of Experts with polynomially decaying learning rate). These online learning algorithms optimally aggregate expert predictions without requiring a validation set.

**Why it matters:** FixedShare has a fixed learning rate. BOA/MLpol adapt their learning rate based on the cumulative loss, achieving optimal regret bounds.

**Implementation:** `opera` R package has these, or inline Python (~40 lines for BOA). Already have the expert predictions from forecast_result.

### Enhancement 3.20B: Forecast Reconciliation (MinT)

**Current:** Multi-frequency fusion handles cross-frequency reconciliation, but the prediction aggregator doesn't reconcile cross-variable forecasts.

**Expert method:** **MinT Reconciliation** (Wickramasuriya et al. 2019, JASA) -- ensures that related forecasts are coherent. E.g., `revenue_forecast * margin_forecast = operating_income_forecast`. Currently these can be inconsistent.

**Why it matters:** Inconsistent forecasts (revenue up but margin up AND operating income down) confuse the report narrative and reduce credibility.

**Implementation:** Set up a summing matrix S for accounting identities, then apply MinT optimal reconciliation. ~30 lines.

### Enhancement 3.24A: Shapelet-Based Analogs

**Current:** DTW distance for analog matching.

**Expert method:** **Shapelet Transform** (Hills et al. 2014) -- instead of matching entire windows, find discriminative sub-sequences (shapelets) that distinguish regime transitions. Then search for historical instances of the same shapelet.

**Why it matters:** DTW matches the full shape but may miss that a specific SHORT pattern (e.g., 5-day V-shaped recovery) is the predictive part. Shapelets isolate the signal.

**Implementation:** `tslearn` library (3,000+ stars) has `ShapeletModel` and `TimeSeriesScalerMeanVariance`. ~30 lines.

### Enhancement 3.25A: Bayesian Optimization for Ensemble Weights

**Current:** Optuna TPE or custom GA for weight optimization.

**Expert method:** **Gaussian Process Bayesian Optimization** (Snoek et al. 2012) via Optuna with GP sampler. GP-based BO has better convergence for expensive objective functions (our RMSE evaluation requires walk-forward).

**Implementation:** `optuna.samplers.GPSampler` (already in optuna). 1-line change: `sampler=optuna.samplers.GPSampler()`.

---

## Implementation Priority Matrix

| Enhancement | Impact | Complexity | Dependencies | Priority |
|-------------|--------|------------|-------------|----------|
| 3.6C: Regime-conditional forecasting | HIGH | LOW | Existing regime labels | P1 |
| 3.10A: Jump-diffusion MC | HIGH | LOW | Layer 1 jump_ratio_21d | P1 |
| 3.10B: Antithetic variates MC | MEDIUM | TRIVIAL | None | P1 |
| 3.12A: CQR intervals | HIGH | MEDIUM | mapie or inline GBM | P1 |
| 3.8A: Adaptive conformal retraining | MEDIUM | LOW | Conformal coverage stats | P1 |
| 3.20A: OPERA online aggregation | HIGH | MEDIUM | Inline BOA implementation | P2 |
| 3.20B: Forecast reconciliation | MEDIUM | MEDIUM | Accounting identity matrix | P2 |
| 3.3A: Convergent Cross Mapping | MEDIUM | MEDIUM | skccm or inline | P2 |
| 3.6A: N-BEATS | HIGH | MEDIUM | neuralforecast or inline | P2 |
| 3.1A: Sticky HDP-HMM | MEDIUM | HIGH | pyhsmm or Gibbs sampler | P3 |
| 3.13A: Vine copula | MEDIUM | HIGH | pyvinecopulib | P3 |
| 3.15A: Rao-Blackwellized PF | LOW | HIGH | Custom implementation | P4 |
| 3.24A: Shapelet analogs | LOW | MEDIUM | tslearn | P4 |
| 3.7A: Neural PID | LOW | MEDIUM | PyTorch autograd | P4 |

---

## Proposed New Variables

| # | Variable | Source Module | Type | Downstream |
|---|----------|-------------|------|-----------|
| 1 | `optimal_n_regimes` | 3.1 Regime Detector | Integer | Profile |
| 2 | `regime_persistence` | 3.1 Regime Detector | Float per regime | Profile |
| 3 | `bocd_run_length` | 3.1 Regime Detector | Integer | Extra vars |
| 4 | `ccm_causal_pairs` | 3.3 Granger Causality | List | Causal network |
| 5 | `regime_conditional_forecast` | 3.6 Forecasting | Per-var per-horizon | Prediction aggregator |
| 6 | `jump_intensity_lambda` | 3.10 Monte Carlo | Float | Profile |
| 7 | `cqr_lower`, `cqr_upper` | 3.12 Conformal | Per-var per-horizon | Prediction aggregator |
| 8 | `reconciled_forecasts` | 3.20 Prediction Aggregator | Per-var per-horizon | Profile |
| 9 | `boa_weights` | 3.20 Prediction Aggregator | Per-model | Profile |

---

## Implementation Checklist

```
[ ] Phase 1: P1 Enhancements (high impact, low complexity)
    [ ] 3.6C: Regime-conditional forecasting wrapper
    [ ] 3.10A: Jump-diffusion MC paths
    [ ] 3.10B: Antithetic variates (5 lines)
    [ ] 3.12A: CQR asymmetric intervals
    [ ] 3.8A: Adaptive conformal retraining trigger

[ ] Phase 2: P2 Enhancements (medium complexity)
    [ ] 3.20A: BOA online aggregation
    [ ] 3.20B: MinT forecast reconciliation
    [ ] 3.3A: Convergent Cross Mapping
    [ ] 3.6A: N-BEATS forecaster

[ ] Phase 3: P3 Enhancements (high complexity)
    [ ] 3.1A: Sticky HDP-HMM
    [ ] 3.13A: Vine copula

[ ] Phase 4: P4 Enhancements (deferred)
    [ ] 3.15A: Rao-Blackwellized PF
    [ ] 3.24A: Shapelet analogs
    [ ] 3.7A: Neural PID
```

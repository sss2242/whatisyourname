# Burn-Out Phase Redesign: True Online Learning for Superior Simulation Weights

## Current Architecture (What Exists)

The pipeline has a two-phase temporal learning design:

### Phase 1: Forward Pass (`run_forward_pass`)
- Walks day-by-day through the full 2-year cache
- 5 model wrappers per variable: Kalman, VAR, LSTM, Tree, Baseline
- Each wrapper has an `update(state, actual, weights)` method
- Kalman is truly online (single Kalman gain update per observation)
- LSTM does single-epoch SGD per observation
- VAR/Tree accumulate data and batch-refit every N steps
- PID controller adjusts learning rates per tier
- Produces `ForwardPassResult` with `model_states`, `predictions_log`, `pid_summary`

### Phase 2: Burn-Out (`run_burnout`)
- Takes the last 130 days of cache
- Runs `run_forward_pass` repeatedly (up to 10 iterations) on that window
- Each iteration: full forward-pass on the 130-day slice, measure validation RMSE on last 20 days
- Shrinks nothing, learns nothing new between iterations -- just re-runs the same forward pass with a different random seed
- Saves the model states from whichever iteration had lowest RMSE
- Converges via early stopping (patience=3)

### The Problem

The burn-out phase does not actually do online learning. It is a hyperparameter search over random seeds on the same forward pass logic. Each iteration is independent -- the model states from iteration N are discarded before iteration N+1. The "best" model states are picked by RMSE on the last 20 days, but there is no mechanism for the models to learn from their own mistakes across iterations.

More critically, **the burn-out model states are never used downstream**. Looking at `main.py`:
- `burnout_result.model_states` is stored but never consumed by Monte Carlo, prediction aggregator, or OHLC predictor
- `burnout_result.best_rmse_by_tier` is stored in the profile but not used for ensemble weighting
- The actual Monte Carlo simulation uses `forecast_result` (from Phase 1 forecasting), not burn-out states

So the burn-out is currently: repeated forward passes that pick the best random seed, store RMSE numbers in the profile, and throw away the model states.

## Proposed Redesign

The goal: make burn-out produce **calibrated per-model per-regime weights** that Monte Carlo and the prediction aggregator actually consume for better simulation results.

### Design Principle

The forward pass is the temporal learner (fit models to historical data). The burn-out should be the **weight calibrator** (learn which models to trust in which regimes, and by how much). This gives Monte Carlo better distribution parameters and the prediction aggregator better ensemble weights.

### New Burn-Out Architecture

```
Forward Pass (existing)
  |
  v  model_states + predictions_log
  |
Burn-Out Phase (redesigned)
  |
  +-- Step 1: Extract per-model per-regime error distributions
  |     from predictions_log
  |
  +-- Step 2: Online weight learning via exponential gradient
  |     (Vovk 1990 / Cesa-Bianchi & Lugosi 2006)
  |     - Walk through predictions_log day-by-day
  |     - Maintain per-regime weight vectors
  |     - Exponential gradient update: w_i *= exp(-eta * loss_i)
  |     - Normalize weights per regime after each step
  |     - eta (learning rate) decays over iterations
  |
  +-- Step 3: Convergence detection
  |     - Track weight vector stability (L2 norm of daily delta)
  |     - Stop when weights stabilize (delta < threshold)
  |     - Or max iterations reached
  |
  +-- Step 4: Regime-conditioned weight output
  |     - Per-regime weight vectors: {regime: {model: weight}}
  |     - Per-regime distribution parameters: {regime: {mean, std}}
  |     - Effective sample size per regime (Kish)
  |
  v
BurnoutResult (enhanced)
  |
  +-- regime_weights: dict[str, dict[str, float]]
  |     e.g. {"bull": {"kalman": 0.4, "lstm": 0.3, "tree": 0.2, "baseline": 0.1}}
  |
  +-- regime_distributions: dict[str, dict[str, float]]
  |     e.g. {"bull": {"mean": 0.0008, "std": 0.012}, "bear": {"mean": -0.003, "std": 0.025}}
  |
  +-- calibrated_model_states: dict[str, ModelWrapper]
  |     (forward pass states refined by burn-out learning)
  |
  +-- convergence_metrics: {iterations, converged, weight_stability}
```

### How Downstream Modules Consume the Output

**Monte Carlo** (`run_monte_carlo`):
- Currently uses `estimate_regime_distributions()` from raw returns
- With burn-out: use `burnout_result.regime_distributions` which are model-weighted (not raw)
- The distribution parameters incorporate model uncertainty: weighted mean/std across models, not just sample mean/std of raw returns
- This produces more realistic tail behavior because the model ensemble captures different aspects of the return dynamics

**Prediction Aggregator** (`run_prediction_aggregation`):
- Currently uses inverse-RMSE weights or FixedShare weights
- With burn-out: use `burnout_result.regime_weights` as the primary ensemble weights
- Regime-conditioned: in a bull market, trust the model that performed best in historical bull periods
- Falls back to inverse-RMSE when burn-out did not produce stable weights

**OHLC Predictor** (`predict_ohlc_series`):
- Currently uses `forecast_result` directly
- With burn-out: weight the forecast components by burn-out regime weights for the current regime

### Implementation Steps

```
[ ] Step 1: Add per-model prediction tracking to ForwardPassResult.predictions_log
    - Currently logs ensemble_pred only
    - Need to also log per-model predictions: {"kalman": 150.2, "lstm": 150.5, ...}
    - This is the raw data the burn-out weight learner needs

[ ] Step 2: Implement ExponentialGradientWeightLearner
    - New class in forecasting.py (or a new burn_out_calibrator.py)
    - walk through predictions_log day-by-day
    - maintain weight vectors per regime
    - exponential gradient update rule
    - convergence detection via weight stability

[ ] Step 3: Redesign BurnoutResult dataclass
    - Add regime_weights, regime_distributions, calibrated_model_states
    - Keep backward compat (iterations_completed, converged, rmse_history)

[ ] Step 4: Redesign run_burnout() to use ExponentialGradientWeightLearner
    - Phase A: Run forward pass once (no more repeated passes with different seeds)
    - Phase B: Feed predictions_log into weight learner
    - Phase C: Optionally re-run forward pass with learned weights as initial ensemble
    - Phase D: Extract regime-conditioned distributions using learned weights

[ ] Step 5: Wire burn-out outputs into Monte Carlo
    - In main.py: pass burnout_result.regime_distributions to run_monte_carlo()
    - Add regime_distributions parameter to run_monte_carlo()
    - Use burn-out distributions when available, fall back to raw estimation

[ ] Step 6: Wire burn-out outputs into Prediction Aggregator
    - In main.py: pass burnout_result.regime_weights to run_prediction_aggregation()
    - Add burnout_weights parameter to run_prediction_aggregation()
    - Use burn-out weights as primary source, FixedShare as secondary

[ ] Step 7: Store enhanced burn-out results in profile
    - Add regime_weights summary to profile extended_models.burnout
    - Add per-regime distribution parameters
    - Add weight stability metrics
```

### Why This Produces Superior Results

1. **Model-weighted distributions vs raw distributions**: Raw return distributions treat every historical day equally. Model-weighted distributions emphasize days where the ensemble was confident (low model disagreement) and down-weight days where models diverged. This produces tighter, more realistic distribution parameters for Monte Carlo.

2. **Regime-conditioned ensemble**: In a bull market, a momentum-following LSTM may be best. In a bear market, a mean-reverting Kalman may be best. Regime-conditioned weights select the right model for the right conditions instead of using a single global weight vector.

3. **Online learning with exponential gradient**: The exponential gradient algorithm has provable regret bounds (O(sqrt(T) log N) where T is time steps and N is number of models). It adapts faster than inverse-RMSE (which is a static batch estimator) and is more principled than FixedShare (which assumes a fixed switching rate).

4. **Single forward pass instead of 10 repeated passes**: The current burn-out wastes compute by re-running the forward pass up to 10 times. The redesigned version runs it once and extracts maximum information from the predictions_log via online learning. This is both faster and more informative.

### Risk Mitigation

- If the burn-out weight learner fails or produces degenerate weights (one model gets weight 1.0), fall back to inverse-RMSE weights
- Minimum weight floor of 0.05 per model to prevent complete exclusion
- Maximum weight cap of 0.6 per model to prevent over-reliance
- If fewer than 50 predictions_log entries exist, skip burn-out calibration entirely

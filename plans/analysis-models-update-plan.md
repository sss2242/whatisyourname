# Analysis Models Update Plan

Implementation roadmap for the 10 proposals and 6 library integrations identified in the deep review. Organized into 3 phases by dependency order and risk level.

---

## Phase 1: Low-Risk, High-Impact (Wire existing libraries + single-module changes)

These changes modify one module each, use libraries that are already installed or trivially small, and have zero risk of breaking existing functionality because they are additive (new functions alongside existing ones).

### 1.1 Adaptive Conformal via MAPIE (Proposal B)

**Module**: `operator1/models/conformal.py`
**Library**: `mapie` (ALREADY INSTALLED)
**What to do**:
- Add `AdaptiveConformalCalibrator` class that wraps `mapie.regression.MapieRegressor`
- Uses rolling window of recent residuals (N=100) with exponential weighting
- Dynamically adjusts coverage target based on recent empirical coverage
- Keep existing `ConformalCalibrator` as fallback

**Wiring change in main.py**: In Step 6p, pass `adaptive=True` to use the new calibrator.

**Dependencies**: None new (mapie already in stage2-ml.txt)

### 1.2 Continuous Survival Probability (Proposal A)

**Module**: `operator1/analysis/survival_mode.py`
**Library**: `sklearn` (ALREADY INSTALLED)
**What to do**:
- Add `compute_survival_probability()` function (sigmoid-based continuous score)
- Keep existing binary `compute_company_survival_flag()` for backward compat
- The probability (0-1) replaces the binary flag in hierarchy weight computation

**Wiring change in main.py**: In Step 5, add `cache["survival_probability"] = compute_survival_probability(cache)`. Update `compute_hierarchy_weights()` to accept the continuous probability and interpolate between normal and survival weight vectors.

**Dependencies**: None new

### 1.3 Regime-Switching Ensemble Weights (Proposal from Issue 7)

**Module**: `operator1/models/prediction_aggregator.py`
**Library**: None new
**What to do**:
- In `run_prediction_aggregation()`, check if `dual_regime_result` provides current regime
- Maintain separate weight vectors per regime (bull, bear, high_vol, crisis)
- On regime change, switch to the weight vector optimized for that regime
- GA optimizer (if available) produces per-regime weights instead of a single vector

**Wiring change in main.py**: None (prediction_aggregator already receives dual_regime_result)

**Dependencies**: None new

### 1.4 Cycle-Aware OHLC Prediction (Proposal I)

**Module**: `operator1/models/ohlc_predictor.py`
**Library**: None new
**What to do**:
- Accept optional `cycle_result` parameter
- If dominant cycle detected, modulate predicted daily return by cycle phase
- At cycle peak, bias return downward; at trough, bias upward; scale by amplitude

**Wiring change in main.py**: In Step 6v, pass `cycle_result=cycle_result` to `predict_ohlc_series()`.

**Dependencies**: None new

### 1.5 Sobol -> Hierarchy Feedback Loop (Proposal H)

**Module**: `operator1/models/sensitivity.py` + `operator1/analysis/hierarchy_weights.py`
**Library**: None new
**What to do**:
- After Sobol analysis, compare first-order indices to hierarchy tier weights
- If discrepancy > 20% for any tier, adjust weight by up to 10% toward data-driven value
- Store adjusted weights as `data_adjusted_tier{N}_weight` columns
- Cap adjustments to prevent oscillation

**Wiring change in main.py**: After Step 6t (Sobol), call a new function to compare and adjust.

**Dependencies**: None new

---

## Phase 2: Library Integration (New pip installs, moderate changes)

These require installing new PyPI packages and modifying module internals, but the module interfaces remain the same.

### 2.1 Student-t / Clayton Copulas via copulae (Proposal D)

**Module**: `operator1/models/copula.py`
**Install**: `pip install copulae`
**What to do**:
- Add `_fit_student_t_copula()` and `_fit_clayton_copula()` using `copulae` library
- Fit all 3 copula types (Gaussian, Student-t, Clayton)
- Select best by AIC
- Report tail dependence from the best-fitting copula
- Keep Gaussian as fallback if copulae not installed

**Profile/Report**: `extended_models.copula` already has `best_copula` and `tail_dependence` fields. Just populate them with the winning copula type.

### 2.2 EMD Cycle Decomposition via EMD-signal (Proposal from skimmed review)

**Module**: `operator1/models/cycle_decomposition.py`
**Install**: `pip install EMD-signal`
**What to do**:
- Add `_decompose_emd()` using `EMD-signal` library (CEEMDAN variant for noise robustness)
- Produces Intrinsic Mode Functions (IMFs) instead of fixed-frequency Fourier components
- Each IMF represents an adaptive cycle that varies in frequency over time
- Keep FFT as fallback if EMD-signal not installed

**Profile/Report**: `extended_models.cycle_decomposition` already has `dominant_cycles` list. Populate with IMF periods instead of FFT peaks.

### 2.3 scikit-fuzzy Rule Engine (Proposal J)

**Module**: `operator1/analysis/fuzzy_protection.py`
**Install**: `pip install scikit-fuzzy`
**What to do**:
- Define 3 Antecedents (sector, economic, policy) and 1 Consequent (protection) using skfuzzy
- Define 4-6 fuzzy rules capturing dimension interactions:
  - IF sector IS strategic AND economic IS significant THEN protection IS high
  - IF sector IS non_strategic AND policy IS emergency THEN protection IS moderate
  - IF sector IS semi_strategic OR economic IS significant THEN protection IS moderate
- Use centroid defuzzification
- Keep manual implementation as fallback if skfuzzy not installed

### 2.4 Matrix Profile Pattern Discovery via stumpy

**Module**: `operator1/models/pattern_detector.py`
**Install**: `pip install stumpy`
**What to do**:
- Add `_detect_motifs_stumpy()` that finds ALL recurring patterns using Matrix Profile
- O(n log n) algorithm discovers motifs (repeated patterns) and discords (anomalies)
- Complements existing candlestick detection (which only finds specific named patterns)
- Motifs represent any recurring shape, not just Japanese candlestick classifications

### 2.5 Systemic Risk Measures (CoVaR, SRISK)

**Module**: `operator1/models/graph_risk.py`
**Install**: `pip install systemic-risk`
**What to do**:
- Add CoVaR (Conditional Value-at-Risk) and SRISK (Systemic Risk) measures
- CoVaR measures how much the target's risk increases when a linked entity is in distress
- SRISK measures capital shortfall under stress scenarios
- These complement the existing SIR contagion model with market-based risk measures

---

## Phase 3: Architectural Changes (Multi-module, higher complexity)

These changes affect multiple modules and require careful testing of cross-module data flow.

### 3.1 Full-Model Walk-Forward (Proposal C)

**Modules**: `walk_forward.py`, `forecasting.py`, `prediction_aggregator.py`
**What to do**:
- Modify `run_walk_forward()` to accept model predict functions from the forecasting module
- Evaluate ALL models (Kalman, GARCH, VAR, LSTM, Tree, Transformer) per survival mode
- Output: per-model, per-mode RMSE matrix
- Feed this matrix to `prediction_aggregator` for mode-conditioned weighting
- GA optimizer produces per-regime weight vectors

**Risk**: Medium. Walk-forward is slow; running all models per day significantly increases runtime. Mitigate by sampling (evaluate every 5th day instead of every day).

### 3.2 Dynamic Factor Kalman (Proposal from Issue 1)

**Module**: `operator1/models/forecasting.py` (Kalman section)
**What to do**:
- Replace local-level state-space model with a multi-variable Dynamic Factor Model
- State vector: latent factors extracted from 40+ derived variables
- Transition matrix: encodes accounting relationships between variables
- Uses `statsmodels.tsa.statespace.dynamic_factor.DynamicFactor`

**Risk**: Medium. The transition matrix specification requires domain knowledge. Start with an unstructured DFM (let the model learn the factor loadings) before adding accounting constraints.

### 3.3 Multivariate Monte Carlo (Proposal E)

**Modules**: `monte_carlo.py`, `copula.py`
**What to do**:
- Instead of simulating returns alone, jointly simulate (return, delta_current_ratio, delta_fcf_yield, delta_debt_to_equity)
- Use the copula correlation structure from the copula module (Phase 2.1 must be done first)
- Check survival triggers on simulated ratio changes directly, not via return proxy

**Risk**: Medium. Requires the copula module to produce a multivariate distribution, which is a prerequisite.

### 3.4 Cross-Company DTW Analogs (Proposal F)

**Modules**: `dtw_analogs.py`, `main.py`
**What to do**:
- Accept `linked_caches` parameter
- Concat target + peer histories (tagged with company ID)
- Run DTW across expanded history
- Weight same-company analogs 1.0x, peer analogs 0.7x
- Requires linked entity caches to be available (Step 5f must have run)

### 3.5 Time-Varying Causal Discovery (Proposal from Issue 6)

**Modules**: `granger_causality.py`, `causality.py`, `model_synergies.py`
**What to do**:
- Run Granger/TE on rolling windows (e.g., 126-day windows, sliding by 21 days)
- Produce a temporal causal graph showing when relationships emerge/disappear
- In `model_synergies`, use the causal DAG for the CURRENT regime, not the static aggregate
- Feature pruning becomes regime-aware

### 3.6 Edge-Weighted Graph Risk (Proposal G)

**Module**: `graph_risk.py`
**What to do**:
- Accept revenue/supply exposure weights from linked_aggregates
- Weight contagion probability per edge by market_cap ratio
- Asymmetric risk: smaller entity depends more on the relationship

---

## Dependency Graph

```
Phase 1 (independent, can run in parallel):
  1.1 Adaptive Conformal (mapie)
  1.2 Continuous Survival
  1.3 Regime-Switching Weights
  1.4 Cycle-Aware OHLC
  1.5 Sobol Feedback Loop

Phase 2 (independent, can run in parallel after Phase 1):
  2.1 copulae Copulas          <-- prerequisite for 3.3
  2.2 EMD Cycles
  2.3 scikit-fuzzy Protection
  2.4 stumpy Patterns
  2.5 Systemic Risk Measures

Phase 3 (dependencies):
  3.1 Full-Model Walk-Forward  <-- can start after Phase 1
  3.2 Dynamic Factor Kalman    <-- independent
  3.3 Multivariate MC          <-- requires 2.1 (copulae)
  3.4 Cross-Company DTW        <-- independent
  3.5 Time-Varying Causality   <-- independent
  3.6 Edge-Weighted Graph      <-- independent
```

---

## New Dependencies Required

| Package | Version | Phase | Size | Install Command |
|---------|---------|-------|------|----------------|
| `copulae` | 0.8.0 | 2.1 | ~2MB | `pip install copulae` |
| `EMD-signal` | 1.9.0 | 2.2 | ~1MB | `pip install EMD-signal` |
| `scikit-fuzzy` | 0.5.0 | 2.3 | ~3MB | `pip install scikit-fuzzy` |
| `stumpy` | 1.14.1 | 2.4 | ~2MB | `pip install stumpy` |
| `systemic-risk` | 0.0.10 | 2.5 | ~1MB | `pip install systemic-risk` |

All existing dependencies (`mapie`, `sklearn`, `statsmodels`, `arch`, `torch`, `shap`, `hmmlearn`, `ruptures`) remain unchanged.

---

## Summary

| Phase | Items | New Deps | Risk | Impact |
|-------|-------|----------|------|--------|
| **1** | 5 changes | 0 | Low | High -- smoother survival, adaptive intervals, regime-aware weights |
| **2** | 5 changes | 5 packages (~9MB) | Low-Medium | Medium-High -- better copulas, cycles, patterns, fuzzy rules |
| **3** | 6 changes | 0 | Medium | Medium -- multivariate MC, cross-company DTW, full walk-forward |
| **Total** | **16 changes** | **5 packages** | | |

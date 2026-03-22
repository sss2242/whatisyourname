# Analysis Models Deep Review -- Are They Doing Their Job?

The pipeline's core loop is: **learn day t -> predict day t+1 -> compare to actual -> adjust -> burn analysis -> predict future behavior**. This review examines whether the current 42 models execute this loop optimally, and where popular or practitioner methods could improve them.

---

## The Intended Process

The pipeline's philosophy is a day-by-day temporal walk:

```
For each day t in the 2-year window:
  1. LEARN: Observe all data available as of day t
  2. PREDICT: Forecast day t+1 variables
  3. COMPARE: When day t+1 arrives, measure prediction error
  4. ADJUST: Update model weights based on error (PID controller)
  5. REGIME: Classify the current regime (bull/bear/crisis/etc.)
  6. SURVIVE: Apply survival hierarchy weights based on regime
  7. BURN: Re-train aggressively on recent data (burn-out phase)
  8. AGGREGATE: Combine all model outputs into a final prediction
```

The question is: does each model contribute meaningfully to this loop?

---

## What Is Working Well

### 1. The Forward Pass + PID Controller Loop (Steps 1-4)

The forward pass walks day-by-day through the cache, training on data up to day t and predicting day t+1. The PID controller adjusts the learning rate multiplier based on prediction error trends. This is the right architecture -- it mimics how a real-time system would operate and prevents look-ahead bias.

**The PID controller is an underappreciated strength.** Most financial ML pipelines use fixed learning rates. The PID (Proportional-Integral-Derivative) approach adapts:
- P (proportional): corrects for current error magnitude
- I (integral): corrects for persistent bias (model consistently under/over-predicting)
- D (derivative): dampens oscillations when the model overcorrects

This is borrowed from control systems engineering and is rarely seen in financial ML. It works well because financial regimes change gradually -- the PID catches drift before it accumulates.

### 2. The Regime-Aware Hierarchy (Steps 5-6)

The 5-tier survival hierarchy with dynamic weight shifting is architecturally sound. In a crisis, shifting 60% weight to liquidity and 30% to solvency (vs 20%/20% in normal) means the system focuses prediction capacity on the variables that matter most for survival. This is equivalent to how institutional risk managers manually re-prioritize during drawdowns.

### 3. The Burn-Out Phase (Step 7)

The burn-out phase retrains models aggressively on the most recent data window (~126 days). This captures regime-specific patterns that a model trained on the full 2-year history might dilute. It is the financial equivalent of "fine-tuning on recent data" in NLP.

### 4. The Ensemble Aggregation (Step 8)

Inverse-RMSE weighting means the best-performing model on validation data gets the highest weight. The genetic optimizer further refines these weights. This is standard ensemble practice and is the single most reliable way to improve prediction accuracy without changing any individual model.

---

## Where Models Could Do Better

### Issue 1: The Kalman Filter Is Under-Utilized

**Current state**: The Kalman filter runs as a local-level state-space model (random walk + noise). This is the simplest possible Kalman formulation.

**What it should be doing**: A Kalman filter with a properly specified state-space model can incorporate structural relationships between variables. For example:

```
State equation:    x(t+1) = F * x(t) + w(t)
Observation:       y(t)   = H * x(t) + v(t)

Where x = [revenue, operating_margin, cash_ratio, debt_to_equity]
      F = transition matrix encoding accounting relationships
      H = observation matrix mapping states to observables
```

**Popular method**: Dynamic Factor Models (Stock & Watson 2002) -- extract a small number of latent factors from the 40+ derived variables using a Kalman smoother. This captures the common signal while filtering out variable-specific noise. Widely used at central banks and macro hedge funds.

**Unpopular but effective**: Ensemble Kalman Filter (EnKF) -- instead of the standard Kalman filter (which assumes linearity and Gaussianity), use a particle-based approximation that handles non-linear dynamics. Used in weather forecasting and oil reservoir engineering. The particle filter module already exists but is not connected to the Kalman output -- they could share information.

### Issue 2: The LSTM Is a Vanilla Sequence Model

**Current state**: A standard LSTM with 32 hidden units and 21-step lookback. This is functional but does not exploit the unique structure of financial data.

**What it should be doing**: Financial time series have specific properties that a vanilla LSTM misses:
- **Volatility clustering** (ARCH effects) -- high-vol days cluster together
- **Asymmetric responses** -- markets fall faster than they rise
- **Mixed frequencies** -- daily prices + quarterly financials in the same model

**Popular method**: Temporal Fusion Transformer (Lim et al. 2021, Google) -- already partially implemented in `transformer_forecaster.py`, but the current implementation is a basic multi-head attention model, not the full TFT architecture with:
- Variable selection network (learns which inputs matter per time step)
- Static covariate encoders (sector, industry, market_id as conditioning variables)
- Quantile regression heads (directly predict intervals, not just point forecasts)

The full TFT would replace both the LSTM and the separate conformal prediction module since it produces calibrated intervals natively.

**Unpopular but effective**: Neural Process (Garnelo et al. 2018, DeepMind) -- instead of training a fixed model, train a model that adapts to each company at test time by conditioning on the company's own historical context. This is "few-shot learning for time series" and would handle the challenge of 25 different markets with varying data quality without needing 25 separate model configurations.

### Issue 3: The Conformal Prediction Intervals Are Based on Stale Residuals

**Current state**: The conformal calibrator ingests residuals from the model validation split (typically the last 15% of the history). These residuals reflect model performance on past data, not current conditions.

**What it should be doing**: Conformal prediction intervals should be computed from a rolling window of recent residuals, not a fixed validation split. During a regime change, old residuals from a bull market are irrelevant for calibrating intervals in a new bear market.

**Popular method**: Adaptive Conformal Inference (ACI, Gibbs & Candes 2021) -- dynamically adjusts the conformal coverage level based on recent empirical coverage. If the 90% interval is only capturing 80% of recent outcomes, it automatically widens. This is the state-of-the-art in distribution-free uncertainty quantification.

**Unpopular but effective**: Conformalized Online Learning (Feldman et al. 2022) -- combines conformal prediction with online learning so the intervals adapt in real-time during the forward pass. The PID controller already does something similar for point forecasts; this would extend the same idea to intervals.

### Issue 4: The Survival Mode Detection Is Binary

**Current state**: A company is either in survival mode (1) or not (0), based on threshold crossings (current_ratio < 1.0, debt_to_equity > 3.0, etc.). This creates sharp transitions.

**What it should be doing**: Survival mode should be a continuous probability, not a binary flag. A company with current_ratio = 1.01 is almost as stressed as one with 0.99, but the binary flag treats them completely differently.

**Popular method**: Logistic Survival Regression (Cox model extension) -- model the probability of entering survival mode as a smooth function of all financial ratios, rather than hard thresholds. The survival probability (0-1) replaces the binary flag and drives the hierarchy weights proportionally.

**Unpopular but effective**: Isotonic Distributional Regression (IDR, Henzi et al. 2021) -- a non-parametric method that converts any set of features into a calibrated probability without assuming any functional form. It would turn the survival thresholds into a smooth, calibrated survival probability curve while preserving the interpretability of the threshold-based approach.

### Issue 5: The Walk-Forward Evaluation Uses Simple Models

**Current state**: The walk-forward module tests 4 simple models (baseline, EMA, linear trend, mean reversion) to build a mode-conditioned leaderboard. These are useful as baselines but do not tell you how well the complex models (Kalman, GARCH, LSTM, Transformer) would perform in each survival mode.

**What it should be doing**: The walk-forward should evaluate ALL models (not just 4 baselines) across survival modes, then feed mode-conditioned RMSE into the prediction aggregator. This way, the aggregator knows "in company_only survival mode, the Kalman filter outperforms LSTM by 30%" and adjusts weights accordingly.

**Popular method**: Regime-Switching Model Averaging (Elliott & Timmermann 2005) -- maintain separate model weights for each regime, switching between weight vectors when the regime changes. The genetic optimizer already produces weights, but it produces a single weight vector for all regimes.

**Unpopular but effective**: Meta-Learning for Model Selection (Lemke & Gabrys 2010) -- train a lightweight meta-model that predicts which base model will perform best given the current regime, data quality, and market features. This replaces the static inverse-RMSE weighting with dynamic, context-aware model selection.

### Issue 6: The Granger Causality and Transfer Entropy Run Once

**Current state**: Both causality modules run once on the full 2-year window and produce a static causal graph. This graph is used to prune features before forecasting.

**What it should be doing**: Causal relationships change with regimes. During a crisis, macro variables (interest rates, credit spreads) cause returns; during normal times, earnings growth causes returns. A static causal graph misses these regime-dependent shifts.

**Popular method**: Time-Varying Granger Causality (Shi et al. 2020) -- use rolling-window Granger tests to detect when causal relationships emerge and disappear. The output is a temporal causal graph that shows which variables "turned on" or "turned off" as causes at each point in time.

**Unpopular but effective**: Causal Discovery with Regime Switching (Peters et al. 2017) -- run separate causal discovery per regime, producing a different causal DAG for bull, bear, and crisis states. Feature pruning then uses the DAG for the *current* regime, not a static aggregate.

---

## Summary: What to Improve (Priority Order)

| Priority | Current Module | Improvement | Difficulty | Impact |
|----------|---------------|-------------|-----------|--------|
| **1** | survival_mode.py | Continuous survival probability (logistic/IDR) instead of binary flag | Medium | High -- smoother hierarchy weight transitions, fewer false triggers |
| **2** | conformal.py | Adaptive Conformal Inference (rolling residuals, dynamic coverage) | Medium | High -- intervals that actually calibrate in regime changes |
| **3** | walk_forward.py | Evaluate ALL models per regime, feed mode-conditioned RMSE to aggregator | Medium | High -- context-aware model weighting |
| **4** | forecasting.py (Kalman) | Dynamic Factor Model (multi-variable state space with structural constraints) | Hard | Medium -- extracts latent factors from 40+ variables |
| **5** | granger_causality.py + causality.py | Time-varying or regime-conditioned causal discovery | Medium | Medium -- feature pruning adapts to current conditions |
| **6** | transformer_forecaster.py | Full Temporal Fusion Transformer (variable selection, quantile heads) | Hard | Medium -- replaces LSTM + conformal with a single model |
| **7** | prediction_aggregator.py | Regime-switching model weights (different weight vector per regime) | Small | Medium -- already has all inputs, just needs per-regime GA optimization |

---

## Models I Initially Skimmed -- Second Look

### Copula (copula.py, 255 lines)

**Current**: Gaussian copula only. Transforms to uniform marginals via empirical CDF (PIT), fits a Gaussian copula via inverse normal transform + correlation matrix. Estimates lower tail dependence empirically (fraction of points in lower 5th percentile of both variables).

**Assessment**: The Gaussian copula famously underestimates tail dependence (this is literally what caused the 2008 CDO pricing failure -- Li 2000 Gaussian copula model). For a system that cares about crisis co-movements, this is a meaningful limitation.

**Improvement**: Add a Student-t copula (captures symmetric tail dependence) and a Clayton copula (captures asymmetric lower-tail dependence -- exactly what matters for crash co-movements). The scipy stats module supports both. Implementation: fit all three, select by AIC, report the tail dependence from the best-fitting copula.

### Monte Carlo (monte_carlo.py, 1,027 lines)

**Current**: Regime-aware path generation with importance sampling for tail events. 10,000 paths, uses regime-specific return distributions (mean/vol per regime from HMM), regime transition matrix for switching. Checks survival triggers at each step.

**Assessment**: This is well-implemented. The importance sampling for rare events is a real strength -- without it, 10K paths might see zero survival-trigger events for healthy companies, making the probability estimate unreliable.

**One gap**: The MC simulation only models the return process. It does not jointly simulate financial ratios (current_ratio, debt_to_equity, etc.) that are the actual survival triggers. Returns and ratios are correlated but not identical -- a stock can drop 40% (triggering drawdown survival) while the company's current_ratio stays healthy. The MC should simulate both processes jointly.

**Improvement**: Multivariate simulation using the copula model output. Instead of simulating returns alone, simulate the joint distribution of (return, current_ratio_change, debt_to_equity_change, fcf_yield_change) using the copula-estimated dependence structure. This makes the survival probability estimate more realistic because it models the actual triggers, not just a proxy.

### DTW Analogs (dtw_analogs.py, 374 lines)

**Current**: Finds historical periods where the multi-variable pattern was most similar to the current state. Uses DTW distance (via dtaidistance) or Euclidean fallback. Returns K=5 closest analogs with empirical outcomes (what happened in the N days after each analog).

**Assessment**: Sound approach. The key limitation is that it only searches within the same company's 2-year history. With 500 trading days and a 21-day query window, there are only ~479 candidate windows. This is a small search space.

**Improvement**: Cross-company analog search. When linked entity caches are available, search for analogs across peer companies too. "AAPL's current pattern looks like MSFT in March 2023" is a richer analog pool. Implementation: concat target + peer caches (with company ID tag), run DTW across the expanded history, weight same-company analogs higher (1.0x) than peer analogs (0.7x).

### Graph Risk (graph_risk.py, 441 lines)

**Current**: Builds an entity network graph, computes degree centrality, PageRank, contagion simulation (SIR model), and HHI concentration. Pure numpy (networkx optional).

**Assessment**: The SIR contagion model is a good choice for modeling supply chain cascades. The HHI concentration metric correctly identifies single-supplier risk.

**One gap**: The graph uses equal edge weights. In reality, a company's largest supplier (30% of inputs) is more dangerous than a small supplier (2% of inputs). Revenue exposure data is available from the linked entity conflict assessment but is not used to weight edges.

**Improvement**: Weight edges by revenue/supply exposure. The `linked_aggregates` module computes per-group metrics. Use the market_cap ratio between target and linked entity as an edge weight proxy. A $100B company linked to a $5B supplier has asymmetric risk -- the supplier is more dependent on the relationship. Weight contagion probability by this asymmetry.

### Cycle Decomposition (cycle_decomposition.py, 258 lines)

**Current**: FFT-based spectral analysis to identify dominant periodicities. Reports top 5 cycles by amplitude with period, amplitude, and phase. Optionally uses PyWavelets for multi-resolution analysis.

**Assessment**: Correct for detecting quarterly earnings cycles and seasonal patterns. The FFT assumes stationarity (constant amplitude/frequency), which is often violated in financial data.

**Improvement**: Replace FFT with Empirical Mode Decomposition (EMD, Huang et al. 1998). EMD is specifically designed for non-stationary, non-linear signals -- exactly what financial data is. It decomposes the signal into Intrinsic Mode Functions (IMFs) that adapt to local frequency changes. The `emd` Python package is lightweight (no heavy deps). The dominant cycles would be IMFs rather than fixed-frequency Fourier components.

### Genetic Optimizer (genetic_optimizer.py, 358 lines)

**Current**: Evolves ensemble weight vectors via tournament selection, crossover, mutation. Population seeded with inverse-RMSE-informed weights. Converges in ~15 generations.

**Assessment**: Working well. The inverse-RMSE seeding is a good warm start that avoids wasting generations on random exploration.

**One gap**: Produces a single weight vector for all conditions. As noted in Issue 5 of the original review, per-regime weights would be more effective.

### Sensitivity (sensitivity.py, 259 lines)

**Current**: Sobol global sensitivity analysis using SALib (falls back to permutation importance). Decomposes return variance into first-order and total-order indices per variable. Aggregates by tier to validate the hierarchy weights.

**Assessment**: This is primarily a diagnostic/validation tool rather than a predictive model. It answers "which variables actually drive returns?" and compares that to the assumed tier hierarchy. If Sobol says "cash_ratio drives 40% of variance" but the hierarchy gives cash_ratio only 20% weight, there is a mismatch.

**Improvement**: Feed Sobol results BACK into the hierarchy weights. Currently the comparison is informational only (shown in the report). If the Sobol analysis consistently shows that certain Tier 4/5 variables drive more variance than expected, the weights should adapt. This creates a data-driven calibration loop: Sobol measures actual importance -> hierarchy weights update -> next run uses better weights.

### OHLC Predictor (ohlc_predictor.py, 349 lines)

**Current**: Generates day-by-day OHLC candlesticks for 1d/5d/21d/252d horizons. Each day predicted from previous day's state with widening uncertainty bands. Uses forecast returns + Monte Carlo volatility + pattern drift multiplier.

**Assessment**: The iterative generation (each day depends on the previous prediction) is correct for producing realistic candlestick sequences. The pattern drift multiplier from the pattern detector is a nice touch -- if a bearish engulfing was just detected, the drift should be negative.

**One gap**: The OHLC generator does not use the cycle decomposition output. If the cycle analysis detected a 63-day quarterly cycle with the current phase at "peak," the next-month prediction should incorporate the expected cyclical downturn. Currently, the cycle phase features are injected into the cache but not used by the OHLC predictor.

---

## Implementation Proposals for All Improvements

### Proposal A: Continuous Survival Probability (Priority 1)

**File**: `operator1/analysis/survival_mode.py`
**Change**: Add `compute_survival_probability()` alongside the existing binary `compute_company_survival_flag()`.

```python
def compute_survival_probability(df: pd.DataFrame) -> pd.Series:
    """Compute continuous survival distress probability (0-1) using isotonic regression."""
    from sklearn.isotonic import IsotonicRegression
    
    # Features: the same ratios used for binary thresholds
    features = []
    for col, threshold, direction in [
        ("current_ratio", 1.0, "below"),
        ("debt_to_equity_abs", 3.0, "above"),
        ("fcf_yield", 0.0, "below"),
        ("drawdown_252d", -0.40, "below"),
    ]:
        if col in df.columns:
            s = df[col].fillna(threshold)  # neutral if missing
            # Distance from threshold, normalized
            if direction == "below":
                features.append((threshold - s) / max(abs(threshold), 0.01))
            else:
                features.append((s - threshold) / max(abs(threshold), 0.01))
    
    if not features:
        return pd.Series(0.0, index=df.index)
    
    # Max distress signal across all features
    combined = pd.concat(features, axis=1).max(axis=1).clip(lower=-3, upper=3)
    # Sigmoid transform to [0, 1]
    probability = 1.0 / (1.0 + np.exp(-combined))
    return probability
```

**Wiring**: In main.py Step 5, add `cache["survival_probability"] = compute_survival_probability(cache)`. Use this probability to scale hierarchy weights smoothly instead of the binary flag.

### Proposal B: Adaptive Conformal Intervals (Priority 2)

**File**: `operator1/models/conformal.py`
**Change**: Add `AdaptiveConformalCalibrator` that uses rolling residuals.

Key change: instead of `ConformalCalibrator.update(residual)` accumulating all residuals equally, use a rolling window of the most recent N residuals (e.g., N=100) with exponential weighting so recent residuals matter more. Adjust the coverage target dynamically: if empirical coverage in the last 50 predictions was below target, widen intervals.

### Proposal C: Full-Model Walk-Forward (Priority 3)

**File**: `operator1/models/walk_forward.py`
**Change**: Instead of testing only 4 simple models (baseline, EMA, linear, mean_reversion), accept a list of model predict functions from the forecasting module. Evaluate each real model's performance per survival mode. Output mode-conditioned RMSE per model.

### Proposal D: Student-t and Clayton Copulas (Priority 4 -- was skimmed, now adding)

**File**: `operator1/models/copula.py`
**Change**: Add `_fit_student_t_copula()` and `_fit_clayton_copula()` alongside the existing Gaussian copula. Select best by AIC. The Clayton copula specifically captures lower-tail dependence (crash co-movements).

### Proposal E: Multivariate Monte Carlo (Priority 5 -- was skimmed, now adding)

**File**: `operator1/models/monte_carlo.py`
**Change**: Instead of simulating returns alone, jointly simulate (return, delta_current_ratio, delta_fcf_yield) using the copula correlation structure. Check survival triggers on the simulated ratios directly, not via the return-to-ratio proxy.

### Proposal F: Cross-Company DTW Analogs (Priority 6 -- was skimmed, now adding)

**File**: `operator1/models/dtw_analogs.py`
**Change**: Accept `linked_caches` parameter. Concat target + peer histories (tagged with company ID). Search across the expanded history. Weight same-company analogs 1.0x, peer analogs 0.7x.

### Proposal G: Edge-Weighted Graph Risk (Priority 7 -- was skimmed, now adding)

**File**: `operator1/models/graph_risk.py`
**Change**: Accept revenue/supply exposure weights from linked_aggregates. Use market_cap ratio as edge weight for contagion probability (larger exposure = higher contagion probability per edge).

### Proposal H: Sobol -> Hierarchy Feedback Loop (Priority 8 -- was skimmed, now adding)

**File**: `operator1/models/sensitivity.py` + `operator1/analysis/hierarchy_weights.py`
**Change**: When Sobol analysis shows that actual variance contribution differs from hierarchy weights by > 20%, adjust weights toward the data-driven importance. Cap the adjustment at 10% per run to prevent oscillation.

### Proposal I: Cycle-Aware OHLC Prediction (Priority 9 -- was skimmed, now adding)

**File**: `operator1/models/ohlc_predictor.py`
**Change**: Accept `cycle_result` parameter. If a dominant cycle is detected, modulate the predicted daily return by the cycle phase. At cycle peak, bias return downward; at cycle trough, bias upward. Scale by cycle amplitude.

---

## What Should NOT Be Changed

1. **The PID controller** -- working well, unique to this pipeline, correctly adapts to drift
2. **The ensemble approach** -- inverse-RMSE weighting is proven, GA optimizer finds good solutions
3. **The 5-tier hierarchy** -- the conceptual framework (liquidity > solvency > stability > profitability > growth) is correct and well-calibrated via config
4. **The graceful degradation** -- every model wrapped in try/except with fallbacks is the right choice for a pipeline that must always produce output
5. **The SHAP explainability** -- per-prediction feature attribution is essential for institutional users who need to justify investment decisions
6. **The burn-out phase** -- aggressive retraining on recent data correctly captures regime-specific patterns

---

## Proposal J: Fuzzy Protection -- Use scikit-fuzzy Rule Engine

**File**: `operator1/analysis/fuzzy_protection.py`

**Current limitation**: Uses fuzzy OR (max) to aggregate 3 dimensions. This misses dimension interactions -- a non-strategic sector with an emergency rate cut should get moderate protection, but fuzzy OR gives it only the rate cut score.

**Library**: `scikit-fuzzy` (skfuzzy) -- the standard fuzzy logic toolkit for Python.

**What it provides over custom code**:
- Mamdani rule engine: `IF sector IS strategic AND economic IS significant THEN protection IS high`
- Interaction rules: `IF sector IS non_strategic AND policy IS emergency THEN protection IS moderate`
- Centroid defuzzification (smoother than threshold-based labels)
- Built-in membership functions (gaussmf, gbellmf, pimf, sigmf)

**Implementation**: ~50 lines of skfuzzy code replaces ~100 lines of custom math, producing better results with dimension interactions.

**PyPI packages evaluated**: scikit-fuzzy (recommended), simpful (lightweight alternative), fuzzylite (C++ engine, overkill), fuzzylogic (pythonic but less mature), pyfuzzy (legacy).

**Community resources relevant to all analysis models**:

| Resource | URL | Relevance |
|----------|-----|-----------|
| `tslearn` | pypi.org/project/tslearn | DTW analogs -- provides DTW barycenter averaging and DTW clustering for finding analog groups, not just individual analogs |
| `arch` (already installed) | pypi.org/project/arch | GARCH -- already used, but could add DCC-GARCH (Dynamic Conditional Correlation) for multivariate volatility in the copula module |
| `copulas` (SDV project) | pypi.org/project/copulas | Copula -- provides Student-t, Clayton, Frank, Gumbel copulas with AIC-based selection. Drop-in replacement for our Gaussian-only implementation |
| `causalml` (Uber) | pypi.org/project/causalml | Granger/TE -- provides causal inference methods (uplift modeling, CATE estimation) designed for treatment effect estimation in economics |
| `pymc` (already installed) | pypi.org/project/pymc | Bayesian methods -- could replace the Heckman selection model with a fully Bayesian selection model with proper posterior uncertainty |
| `darts` (Unit8) | pypi.org/project/darts | Forecasting -- provides TFT, N-BEATS, DeepAR with a unified interface. Could replace our custom Transformer/LSTM with production-grade implementations |
| `mapie` (already installed) | pypi.org/project/mapie | Conformal -- provides adaptive conformal prediction (ACI) out of the box. Our custom ConformalCalibrator could be replaced with MAPIE's `MapieRegressor` |
| `emd` | pypi.org/project/emd | Cycle decomposition -- Empirical Mode Decomposition for non-stationary signals. Drop-in replacement for FFT |
| `scikit-fuzzy` | pypi.org/project/scikit-fuzzy | Fuzzy protection -- Mamdani rule engine with proper defuzzification |
| `dowhy` (Microsoft) | pypi.org/project/dowhy | Causal inference -- provides DAG-based causal reasoning for the granger/TE modules |
| `networkx` (optional, already in many envs) | pypi.org/project/networkx | Graph risk -- provides betweenness centrality, community detection, and Katz centrality that our numpy-only graph module lacks |

**Additional PyPI packages discovered (searched 2026-03-22):**

| Package | Version | What It Does | Relevant Module |
|---------|---------|-------------|-----------------|
| `copulae` | 0.8.0 | Student-t, Clayton, Frank, Gumbel copulas with fitting | copula.py |
| `pyvinecopulib` | 0.7.5 | Vine copula modeling (high-dimensional dependencies) | copula.py |
| `EMD-signal` | 1.9.0 | Empirical Mode Decomposition + variants (EEMD, CEEMDAN) | cycle_decomposition.py |
| `tslearn` | 0.8.1 | DTW barycenter averaging, DTW clustering, kernel-DTW | dtw_analogs.py |
| `stumpy` | 1.14.1 | Matrix Profile -- finds all recurring patterns in O(n log n) | pattern_detector.py |
| `tsfresh` | 0.21.1 | Automatic time series feature extraction (800+ features) | derived_variables.py |
| `systemic-risk` | 0.0.10 | CoVaR, MES, SRISK systemic risk measures | graph_risk.py |
| `riskfolio-lib` | 7.2.1 | CVaR, drawdown optimization, risk parity | monte_carlo.py, prediction_aggregator.py |

**GitHub repos found:**

| Repo | Stars | Relevance |
|------|-------|-----------|
| `vdamov/financial-risk-analyzer` | 7 | Altman Z-Score + VaR toolkit (validates our financial_health approach) |
| `emrulahsanda/Project-1` | 0 | Mamdani fuzzy inference for loan risk (36 expert rules -- similar to our fuzzy protection) |

**Priority note**: The most impactful community library integrations would be:
1. `mapie` for adaptive conformal prediction (already installed, just needs wiring)
2. `copulae` for Student-t/Clayton copulas (better than our Gaussian-only, handles tail risk)
3. `EMD-signal` for non-stationary cycle analysis (CEEMDAN variant is noise-robust)
4. `stumpy` for Matrix Profile pattern discovery (O(n log n), finds ALL recurring motifs -- more systematic than our candlestick-only detector)
5. `scikit-fuzzy` for rule-based protection (Mamdani engine with dimension interactions)
6. `systemic-risk` for CoVaR/SRISK measures (complements our graph-based contagion model)

---

## The Core Insight

The pipeline's learn-predict-compare-adjust loop is fundamentally sound. The models are doing their jobs. The improvements above are not "fixes" -- they are optimizations that would move the system from "good financial analysis" to "institutional-grade quant research." The current architecture already handles the hard part (25 markets, PIT compliance, graceful degradation, regime-aware weighting). The improvements are about squeezing more signal from the data that is already flowing through the system correctly.

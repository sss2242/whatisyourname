# Layer 2: Analysis Modules -- Expert Methods Update Plan

*Researched 2026-05-01 -- domain expert methods for enhancing the 13 Layer 2 analysis modules*

Following the same pattern as the Layer 1 Feature Engineering Expert Methods Update Plan, this document identifies expert-grade enhancements for each Layer 2 module. Each enhancement is grounded in published academic methods or battle-tested quantitative finance practice.

---

## Current State Summary

Layer 2 has 13 modules producing 61 cache columns + 71 result fields. These modules form the **interpretive framework** that controls how all downstream temporal models behave. They answer: "What regime is this company in?" and "How should models adapt?"

| Module | Lines | Current Methods | Enhancement Opportunity |
|--------|-------|----------------|------------------------|
| 2.1 Survival Mode | 664 | OR-based triggers, Cox PH, sigmoid | Gradient-based early warning, ensemble survival models |
| 2.2 Hierarchy Weights | 322 | Fixed regime-to-weight lookup tables | Data-driven weight optimization, entropy-based allocation |
| 2.3 Survival Timeline | 664 | Rule-based 6-mode + HMM bridge | Semi-Markov duration modeling, Bayesian state estimation |
| 2.4 Fuzzy Protection | 394 | Mamdani fuzzy inference (11 rules) | Adaptive fuzzy rules, Choquet integral aggregation |
| 2.5 Ethical Filters | 355 | Threshold-based PASS/WARN/FAIL | Evidential reasoning, belief functions |
| 2.6 Economic Planes | 135 | Static sector-to-plane lookup | Dynamic plane migration, input-output linkage |
| 2.7 Vanity | 636 | 5-component weighted composite | Anomaly detection, peer-conditional scoring |
| 2.8 Financial Health | 1,100 | Percentile rank + Altman Z + Beneish M | Ensemble distress models, time-varying scoring |
| 2.9 Adaptive Thresholds | 836 | Peer percentile, BOCPD, Jenks, HMM crossover | Quantile regression, conformal threshold calibration |
| 2.10 Adaptive Model Params | 1,185 | 10+ calibration methods | Bayesian hyperparameter optimization |
| 2.11 Adaptive Windows | 552 | Nyquist-anchored, scaling laws | Optimal bandwidth selection, cross-validation |
| 2.12 USS Controller | 653 | 5-dimension reconfiguration | Multi-objective optimization, Pareto-optimal configs |
| 2.13 Scenario Engine | 387 | 3-scenario MC with fixed assumptions | Conditional scenario generation, stress testing frameworks |

---

## Domain 1: Survival Mode Detection (2.1)

### Enhancement 1A: Gradient-Based Early Warning System

**Current:** Binary OR trigger -- fires when ANY threshold breached. The `covenant_proximity_score` from Layer 1 PR #1 helps, but survival_mode itself has no gradient awareness.

**Expert method:** Compute the **time derivative** of each trigger variable and fire a "deterioration alert" when the slope toward a threshold exceeds a critical rate, even if the threshold isn't breached yet.

**Formula:**
```
deterioration_rate_i = (value_i - value_i.shift(21)) / abs(threshold_i) 
trigger_velocity_flag = any(deterioration_rate_i < -0.3)  # 30% of threshold crossed in 21d
```

**Academic basis:** Duffie, Saita & Wang (2007) "Multi-period corporate default prediction with stochastic covariates" -- default intensity depends on both level AND trajectory of covariates.

**Variables produced:** `survival_velocity_flag` (binary), `survival_deterioration_rate` (continuous max across triggers)

### Enhancement 1B: Competing Risks Survival Model

**Current:** Cox PH treats all distress as one event. In reality, companies can fail via different mechanisms (liquidity crisis, solvency crisis, market crash).

**Expert method:** **Fine-Gray competing risks model** (Fine & Gray 1999) via `lifelines.AalenJohansenFitter` or custom implementation. Each failure mode (liquidity, solvency, market) gets its own subdistribution hazard.

**Formula:** Separate cause-specific hazards: `h_k(t) = h_0k(t) * exp(X * beta_k)` for k in liquidity, solvency, market.

**Variables produced:** `survival_prob_liquidity` (0-1), `survival_prob_solvency` (0-1), `survival_prob_market` (0-1), `dominant_risk_channel` (categorical)

### Enhancement 1C: Bayesian Survival Probability with Uncertainty

**Current:** Survival probability is a point estimate (0.4*sigmoid + 0.6*cox). No uncertainty quantification.

**Expert method:** **Bayesian Cox PH** via `lifelines` with Markov Chain Monte Carlo or variational inference. Produces posterior distribution over survival probability, enabling credible intervals.

**Variables produced:** `survival_probability_p10` (0-1), `survival_probability_p90` (0-1), `survival_uncertainty` (p90-p10 spread)

---

## Domain 2: Hierarchy Weights (2.2)

### Enhancement 2A: Entropy-Based Weight Allocation

**Current:** Fixed lookup table: normal=[20,20,20,20,20], company_survival=[50,30,15,4,1]. No data-driven adaptation within a regime.

**Expert method:** **Maximum Entropy principle** (Jaynes 1957) -- allocate weights to maximize information content given regime constraints. Within each regime, use the Shannon entropy of per-tier prediction errors to shift weight toward tiers where models are most uncertain (where attention is most needed).

**Formula:**
```
H_k = -sum(p_i * log(p_i))  # entropy of tier k's prediction error distribution
w_k = H_k / sum(H_j for all j)  # weight proportional to entropy (uncertainty)
w_k_final = alpha * regime_default_k + (1-alpha) * w_k  # blend with regime defaults
```

Where alpha decays from 1.0 (pure regime defaults) to 0.5 as more data accumulates.

**Academic basis:** Information-theoretic portfolio allocation (Cover & Thomas 1991), applied to attention allocation instead of capital.

**Variables produced:** `hierarchy_tier{1-5}_entropy` (5 continuous), updated `hierarchy_tier{1-5}_weight` (entropy-blended)

### Enhancement 2B: Sobol-Feedback Continuous Adjustment

**Current:** `adjust_hierarchy_from_sobol()` exists but runs once after Sobol sensitivity (Step 6t). It's a one-shot retroactive adjustment.

**Expert method:** Make the Sobol feedback **continuous** via exponentially weighted memory. Each time the pipeline runs, the previous Sobol importance rankings are loaded from disk and blended with regime defaults, gradually shifting weights toward empirically important tiers.

**Variables produced:** `hierarchy_sobol_memory` (dict of tier -> rolling importance), updated weights

---

## Domain 3: Survival Timeline (2.3)

### Enhancement 3A: Semi-Markov Duration Modeling

**Current:** The timeline uses Markov chain transitions where the probability of leaving a state is independent of how long you've been in it.

**Expert method:** **Semi-Markov model** where transition probabilities depend on dwell time (Barbu & Limnios 2008). Companies in distress for 90+ days have different recovery dynamics than those in distress for 10 days.

**Formula:**
```
P(exit | state=s, dwell=d) = h_s(d) * (1 - H_s(d))  # duration-dependent hazard
```

Where `h_s(d)` is the state-specific hazard function of dwell time, estimated from the company's own `days_in_mode` history.

**Variables produced:** `expected_remaining_days_in_mode` (integer), `mode_exit_probability_21d` (0-1 duration-aware)

### Enhancement 3B: Bayesian Online State Estimation

**Current:** Enriched timeline uses HMM posterior probabilities but treats them as fixed after fitting.

**Expert method:** **Bayesian online changepoint detection** (Adams & MacKay 2007) applied to the survival timeline itself. Maintains a posterior over "how many days since the last regime change" and updates it daily without refitting.

**Variables produced:** `run_length_posterior` (distribution), `changepoint_probability` (0-1 daily)

---

## Domain 4: Fuzzy Protection (2.4)

### Enhancement 4A: Choquet Integral Aggregation

**Current:** Mamdani fuzzy inference with 11 hand-crafted rules. The rule base is static.

**Expert method:** Replace the rule base with a **Choquet integral** (Choquet 1954, Grabisch 1996) which automatically captures interactions between fuzzy inputs without explicit rules. The Choquet integral uses a **fuzzy measure** (capacity) that assigns importance to every subset of inputs, not just individual inputs.

**Formula:**
```
C_mu(f) = sum_{i=1}^{n} (f_{sigma(i)} - f_{sigma(i-1)}) * mu(A_i)
```

Where sigma orders inputs by value and mu is the fuzzy measure learned from historical protection outcomes.

**Academic basis:** Grabisch & Labreuche (2010) "A decade of application of the Choquet and Sugeno integrals in multi-criteria decision making"

**Variables produced:** Updated `fuzzy_protection_degree` (Choquet-based), `fuzzy_interaction_index` (Shapley interaction between inputs)

### Enhancement 4B: Adaptive Rule Learning via ANFIS

**Current:** 11 static rules.

**Expert method:** **Adaptive Neuro-Fuzzy Inference System** (Jang 1993) -- learns optimal fuzzy membership functions and rule weights from data. Can be implemented as a simple gradient descent on membership function parameters.

**Practical approach:** Too complex for our pipeline. Instead, use a simpler approach: **rule weight adaptation** from historical prediction accuracy. Track which rules fire and whether the protection prediction was correct (did the government actually intervene?). Decay weights of rules with low accuracy.

---

## Domain 5: Financial Health (2.8)

### Enhancement 5A: Ensemble Distress Prediction

**Current:** Altman Z-Score (1968) + Beneish M-Score (1999) as standalone scores.

**Expert method:** Combine multiple distress models into a **stacked ensemble**:
1. Altman Z (already have)
2. Ohlson O-Score (Ohlson 1980) -- logistic model, better calibrated probabilities
3. Zmijewski Score (Zmijewski 1984) -- probit model, accounts for sample selection
4. Campbell-Hilscher-Szilagyi (CHS 2008) -- best academic model, uses market + accounting
5. Merton DD (already have)

**Note:** Some of these are already computed in HF `advanced_methods.py` (Piotroski, Altman Z''', Merton). The enhancement is to bring them INTO financial_health.py as an ensemble with proper probability calibration via **isotonic regression** or **Platt scaling**.

**Formula:**
```
p_distress = isotonic_calibrate(w1*Z_prob + w2*O_prob + w3*CHS_prob + w4*Merton_PD)
```

**Variables produced:** `fh_ensemble_distress_prob` (0-1), `fh_ensemble_distress_label` (safe/watch/warning/critical)

### Enhancement 5B: Time-Varying Composite Scoring

**Current:** Expanding percentile rank normalization. The rank is computed against ALL history, so recent deterioration is diluted by long healthy periods.

**Expert method:** **Exponentially weighted percentile rank** with half-life matched to filing frequency (from adaptive_windows). Recent observations get more weight in the ranking.

**Formula:**
```
ewm_rank_k = ewm_percentile_rank(tier_k_score, halflife=filing_period_days)
```

**Variables produced:** Updated `fh_{tier}_score` with recency-weighted ranking

### Enhancement 5C: Conditional Value-at-Risk Health Score

**Current:** Composite score is a weighted average -- insensitive to tail risk.

**Expert method:** Replace weighted average with **CVaR-weighted composite** (Rockafellar & Uryasev 2000). Instead of averaging tier scores, compute the expected score conditional on being in the worst 10% of outcomes. This makes the composite much more sensitive to the weakest tier.

**Formula:**
```
fh_cvar_composite = E[composite | composite < VaR_10%]
```

**Variables produced:** `fh_cvar_composite` (0-100), more sensitive to weak-tier deterioration

---

## Domain 6: Adaptive Thresholds (2.9)

### Enhancement 6A: Quantile Regression Thresholds

**Current:** Peer percentile P10/P90 for threshold calibration.

**Expert method:** **Quantile regression** (Koenker & Bassett 1978) models the conditional quantile of survival trigger variables given sector, size, and market regime. This produces regime-conditional thresholds instead of unconditional percentiles.

**Formula:**
```
Q_tau(Y | X) = X * beta_tau  # where tau = 0.10 for lower threshold, 0.90 for upper
```

Where X includes sector dummies, log(market_cap), macro_quadrant, and survival_regime.

**Variables produced:** Updated threshold calibration with regime-conditional quantiles

### Enhancement 6B: Conformal Threshold Calibration

**Current:** Jenks natural breaks for FH label breakpoints.

**Expert method:** Use the **conformal prediction** framework (already in our pipeline) to set threshold breakpoints with guaranteed coverage. The FH label should be "Critical" with at most 5% false negative rate (companies labeled "Fair" that actually default within 1 year).

---

## Domain 7: USS Controller (2.12)

### Enhancement 7A: Multi-Objective Regime Configuration

**Current:** Fixed lookup tables for model config per regime (Kalman noise 3-5x, LSTM lookback 5-10, etc.).

**Expert method:** **Pareto-optimal configuration search** across two objectives: (1) prediction accuracy and (2) computational cost. For each regime, maintain a Pareto front of configs discovered during retroactive calibration. Select the knee point.

**Practical approach:** Track per-regime per-config prediction errors over time. When a regime activates, select the config that was best last time this regime was active. Simple but effective.

### Enhancement 7B: Gradual Regime Transition

**Current:** Hard switching between regime configs. When survival mode activates, ALL parameters change simultaneously.

**Expert method:** **Soft switching** with exponential transition. When entering a new regime, interpolate between old and new configs over a transition window (e.g., 5 days). This prevents discontinuities in model behavior.

**Formula:**
```
config_t = (1 - lambda_t) * config_old + lambda_t * config_new
lambda_t = 1 - exp(-t / halflife)
```

---

## Domain 8: Scenario Engine (2.13)

### Enhancement 8A: Conditional Scenario Generation

**Current:** 3 fixed scenarios with hardcoded assumptions (-15%, status quo, -40%).

**Expert method:** **Conditional scenario generation** from the company's own factor exposures. Use the Granger causal graph (from Layer 3) to propagate shocks: if revenue drops 15%, what happens to margins given the historical relationship?

**Formula:**
```
scenario_margin = f(revenue_shock) via VAR impulse response function
scenario_debt = g(margin_shock, rate_shock) via historical regression
```

### Enhancement 8B: Reverse Stress Testing

**Current:** Forward scenarios only ("what if revenue drops X%?").

**Expert method:** **Reverse stress testing** (Basel III requirement) -- "what combination of shocks would cause the company to breach survival thresholds?" Work backwards from the survival trigger to find the minimum joint shock.

**Formula:**
```
argmin ||shock|| such that f(base + shock) triggers survival_mode
```

Solved via constrained optimization (scipy.minimize).

**Variables produced:** `reverse_stress_revenue_shock` (%), `reverse_stress_margin_shock` (%), `reverse_stress_rate_shock` (bps)

---

## Implementation Priority Matrix

| Enhancement | Impact | Complexity | Dependencies | Priority |
|-------------|--------|------------|-------------|----------|
| 1A: Gradient early warning | HIGH | LOW | Layer 1 PR #1 `covenant_proximity_score` | P1 |
| 5B: Time-varying FH scoring | HIGH | LOW | adaptive_windows filing period | P1 |
| 3A: Semi-Markov duration | HIGH | MEDIUM | survival_timeline existing `days_in_mode` | P1 |
| 2A: Entropy-based weights | MEDIUM | LOW | Forward pass prediction errors | P2 |
| 5A: Ensemble distress | HIGH | MEDIUM | HF advanced_methods overlap | P2 |
| 7B: Gradual regime transition | MEDIUM | LOW | USS controller | P2 |
| 8B: Reverse stress testing | HIGH | MEDIUM | scipy.minimize | P2 |
| 1B: Competing risks | MEDIUM | MEDIUM | lifelines | P3 |
| 6A: Quantile regression thresholds | MEDIUM | MEDIUM | statsmodels quantile regression | P3 |
| 4A: Choquet integral | LOW | HIGH | Custom implementation | P4 |
| 8A: Conditional scenarios | MEDIUM | HIGH | Granger causal graph from Layer 3 | P4 |
| 1C: Bayesian survival | LOW | HIGH | MCMC / variational inference | P4 |

---

## Proposed New Variables (from all enhancements)

| # | Variable | Source Module | Type | Tier |
|---|----------|-------------|------|------|
| 1 | `survival_velocity_flag` | 2.1 Survival Mode | Binary (0/1) | T1 |
| 2 | `survival_deterioration_rate` | 2.1 Survival Mode | Continuous | T1 |
| 3 | `survival_prob_liquidity` | 2.1 Survival Mode | Continuous (0-1) | T1 |
| 4 | `survival_prob_solvency` | 2.1 Survival Mode | Continuous (0-1) | T2 |
| 5 | `survival_prob_market` | 2.1 Survival Mode | Continuous (0-1) | T3 |
| 6 | `dominant_risk_channel` | 2.1 Survival Mode | Categorical | -- |
| 7 | `survival_probability_p10` | 2.1 Survival Mode | Continuous (0-1) | T1 |
| 8 | `survival_probability_p90` | 2.1 Survival Mode | Continuous (0-1) | T1 |
| 9 | `survival_uncertainty` | 2.1 Survival Mode | Continuous | T1 |
| 10 | `hierarchy_tier{1-5}_entropy` | 2.2 Hierarchy Weights | Continuous (x5) | -- |
| 11 | `expected_remaining_days_in_mode` | 2.3 Survival Timeline | Integer | T3 |
| 12 | `mode_exit_probability_21d` | 2.3 Survival Timeline | Continuous (0-1) | T3 |
| 13 | `changepoint_probability` | 2.3 Survival Timeline | Continuous (0-1) | T3 |
| 14 | `fuzzy_interaction_index` | 2.4 Fuzzy Protection | Continuous | -- |
| 15 | `fh_ensemble_distress_prob` | 2.8 Financial Health | Continuous (0-1) | T2 |
| 16 | `fh_ensemble_distress_label` | 2.8 Financial Health | Categorical | -- |
| 17 | `fh_cvar_composite` | 2.8 Financial Health | Continuous (0-100) | -- |
| 18 | `reverse_stress_revenue_shock` | 2.13 Scenario Engine | Continuous (%) | -- |
| 19 | `reverse_stress_margin_shock` | 2.13 Scenario Engine | Continuous (%) | -- |
| 20 | `reverse_stress_rate_shock` | 2.13 Scenario Engine | Continuous (bps) | -- |

**Total new cache columns:** ~15 (some are result-only)
**Total new result fields:** ~10

---

## Implementation Checklist

```
[ ] Phase 1: P1 Enhancements (low complexity, high impact)
    [ ] 1A: Gradient early warning in survival_mode.py
    [ ] 5B: Time-varying FH scoring in financial_health.py
    [ ] 3A: Semi-Markov duration in survival_timeline.py

[ ] Phase 2: P2 Enhancements (medium complexity)
    [ ] 2A: Entropy-based hierarchy weights
    [ ] 5A: Ensemble distress prediction
    [ ] 7B: Gradual regime transition in USS controller
    [ ] 8B: Reverse stress testing in scenario_engine.py

[ ] Phase 3: P3 Enhancements (medium complexity, specialized)
    [ ] 1B: Competing risks survival model
    [ ] 6A: Quantile regression thresholds

[ ] Phase 4: P4 Enhancements (high complexity, lower priority)
    [ ] 4A: Choquet integral aggregation
    [ ] 8A: Conditional scenario generation
    [ ] 1C: Bayesian survival probability

[ ] Downstream wiring
    [ ] stage3_temporal.py: _init_extra_vars for new columns
    [ ] signal_ic.py: new signals to _SIGNAL_SPEED
    [ ] survival_hierarchy.yml: tier membership for new vars
    [ ] model_tests.py: smoke tests for enhanced modules
```

# Scoring Weights Panel -- All Models (2026-04-03)

Comprehensive reference for every weight, score, and calibration parameter across all 50 analytical models in Operator 1. Each model's scoring mechanism is documented with its default values, optimization method, what it explains in the data, and how it adapts.

---

## Panel 1: Hierarchy Tier Weights (Pipeline-Wide)

Controls how much analytical attention each variable tier receives. Every downstream model consumes these weights.

| Regime | Tier 1 (Liquidity) | Tier 2 (Solvency) | Tier 3 (Stability) | Tier 4 (Profit) | Tier 5 (Growth) | Trigger |
|--------|-------------------|-------------------|---------------------|-----------------|-----------------|---------|
| `normal` | 20% | 20% | 20% | 20% | 20% | No flags |
| `company_survival` | 50% | 30% | 15% | 4% | 1% | Company distress |
| `modified_survival` | 40% | 35% | 20% | 4% | 1% | Country crisis, not protected |
| `extreme_survival` | 60% | 30% | 10% | 0% | 0% | Both company + country |

**Optimization:** Vanity adjustment shifts 5% from T4/T5 to T1 when `vanity_percentage > threshold` in survival. Sobol feedback nudges weights toward data-driven variable importance after Step 6t.

**What it explains:** Determines which financial dimensions matter most for the company's current state. In distress, liquidity (can the company pay bills?) dominates over growth (is the company growing?).

---

## Panel 2: Financial Health Scoring Weights

5-tier daily composite health score.

| Tier | Component | Weight | Scoring Method | Variables |
|------|-----------|--------|---------------|-----------|
| T1 | Liquidity | hierarchy_tier1_weight | Expanding percentile rank | cash_ratio, current_ratio, free_cash_flow |
| T2 | Solvency | hierarchy_tier2_weight | Expanding percentile rank | debt_to_equity, interest_coverage, net_debt_to_ebitda |
| T3 | Stability | hierarchy_tier3_weight | Expanding percentile rank | volatility_21d, drawdown_252d, volume_trend |
| T4 | Profitability | hierarchy_tier4_weight | Expanding percentile rank | gross_margin, operating_margin, net_margin |
| T5 | Growth | hierarchy_tier5_weight | Expanding percentile rank | revenue_growth, pe_ratio (adaptive cap), ev_to_ebitda (adaptive cap) |

**PE/EV Adaptive Caps:** 3-method consensus replacing fixed PE=200/EV=100:
- Log-Normal P99.5 (Aitchison & Brown 1957)
- Tukey Extreme Fence (Tukey 1977): Q3 + 3*IQR
- MAD-Based Cap (Iglewicz & Hoaglin 1993): median + 5*MAD

**Composite:** `fh_composite = sum(tier_score * tier_weight) / sum(tier_weights)`

**Labels (Jenks Natural Breaks):** Distress / Stressed / Fair / Good / Strong -- breakpoints computed via dynamic programming from actual score distribution.

---

## Panel 3: Survival Mode Detection Weights

7-trigger OR logic + continuous probability scoring.

| Trigger | Default Threshold | Adaptive Source | Severity Weight | Signal Speed |
|---------|------------------|-----------------|-----------------|-------------|
| current_ratio < X | 1.0 | Peer P10 (Huber) | 0.25 | Slow |
| debt_to_equity > X | 3.0 | Peer P90 (Huber) | 0.20 | Slow |
| fcf_yield < X | 0.0 | Peer P10 (Huber) | 0.20 | Medium |
| drawdown_252d < X | -0.40 | BOCPD tightening | 0.15 | Fast |
| conflict_intensity > 0.7 | 0.70 | Static | 0.10 | Slow |
| sanctions_flag == 1 | Binary | Static | 0.05 | Slow |
| inst_flow_momentum < X | -0.15 | Static | 0.05 | Fast |

**Continuous probability:** Sigmoid of distance to each threshold, weighted by severity:
```
survival_probability = 0.4 * sigmoid_score + 0.6 * cox_ph_score
```

**Cox PH blend weights** (adaptive, Cochrane 1954): Inverse-variance of each model's prediction error on historical distress episodes. Higher-precision model gets more weight.

---

## Panel 4: Conflict Risk Scoring Weights

Weighted intensity formula for geopolitical risk.

| Component | Weight | Source | What It Measures |
|-----------|--------|--------|-----------------|
| Event score | 40% | UCDP GED API | Armed conflict events near company's country |
| Fatality score | 20% | UCDP GED API | Conflict fatalities (severity indicator) |
| Flag score | 25% | Static lists (WB FCS, OFAC, wars) | Binary risk classification |
| News score | 15% | GDELT | Real-time news conflict mentions + tone |

**Linked entity propagation weights:**

| Relationship Group | Supply Chain Weight | Revenue Weight | Competitive Weight |
|-------------------|--------------------|--------------|--------------------|
| Suppliers | 1.0x (direct) | 0.3x | 0.0x |
| Customers | 0.3x | 1.0x (direct) | 0.0x |
| Competitors | 0.0x | 0.0x | 1.0x (inverse, benefits target) |
| Financial institutions | 0.5x | 0.2x | 0.0x |
| Logistics | 0.8x | 0.1x | 0.0x |

---

## Panel 5: Fuzzy Protection Scoring Weights

Mamdani fuzzy inference engine with 3 inputs, 11 rules.

| Input Variable | Membership Functions | Range | What It Measures |
|---------------|---------------------|-------|-----------------|
| Sector strategicness | low/medium/high (trapezoidal) | 0-1 | How strategic the sector is to the government |
| Economic significance | low/medium/high (trapezoidal) | 0-1 | market_cap / GDP ratio |
| Policy responsiveness | low/medium/high (trapezoidal) | 0-1 | Emergency rate cut magnitude |

**Key rules (11 total):**
- IF sector=high AND economic=high THEN protection=very_high
- IF sector=high AND economic=low THEN protection=moderate
- IF sector=low AND economic=high THEN protection=moderate
- IF sector=low AND economic=low THEN protection=very_low

**Defuzzification:** Centroid method. Fallback: fuzzy OR (max) if scikit-fuzzy unavailable.

---

## Panel 6: Vanity (Capital Allocation) Scoring Weights

5-component composite measuring management quality.

| Component | Weight | Range | What It Measures |
|-----------|--------|-------|-----------------|
| R&D Mismatch | 15% | 0-100 | High R&D with declining revenue |
| SGA Bloat | 25% | 0-100 | Overhead expenses vs industry benchmarks |
| Capital Misallocation | 30% | 0-100 | Debt-funded dividends/buybacks while FCF negative |
| Competitive Decay | 15% | 0-100 | Losing market share vs peers (from peer_ranking) |
| Sentiment Gap | 15% | 0-100 | Market sentiment diverging from fundamentals |

**Labels:** Disciplined (0-20), Moderate (20-40), Wasteful (40-70), Reckless (70-100)

---

## Panel 7: Adaptive Threshold Calibration Methods

5 methods replacing fixed textbook thresholds with data-driven values.

| Method | Citation | Input | Output | Safety Floor |
|--------|----------|-------|--------|-------------|
| (A) Peer Percentile | Huber 1981 | Peer cache distribution | Survival triggers at P10/P90 | current_ratio >= 0.3 |
| (D) BOCPD Tightening | Adams & MacKay 2007 | Own-history structural shift | 20% threshold tightening | Never below absolute floor |
| (E) Sector Z-Score | Iglewicz & Hoaglin 1993 | Peer vanity distribution | 2 modified-Z-score flag | Floor per variable |
| (H) Jenks Natural Breaks | Fisher 1958 | FH composite distribution | Optimal class boundaries (DP) | Min 3 classes |
| (J) HMM Emission Crossover | Rabiner 1989 | HMM Gaussian parameters | Regime mixer thresholds | HMM must be fitted |

---

## Panel 8: Adaptive Model Parameters (Tier 2)

10+ data-derived parameters replacing fixed constants.

| Parameter | Method | Citation | Default | Adaptive Range | What It Controls |
|-----------|--------|----------|---------|---------------|-----------------|
| blend_w_sig / blend_w_cox | Inverse-variance | Cochrane 1954 | 0.4 / 0.6 | 0.1-0.9 | Survival probability blend |
| survival_risk_multiplier | HMM vol ratio | -- | 1.0 | 0.5-5.0 | MC risk widening in crisis |
| intraday_low_factor | Garman-Klass | GK 1980 | 1.0 | 0.5-2.0 | OHLC Low prediction spread |
| transition_halflife | Switch durations | -- | 21 days | 5-63 | Regime blend ramp speed |
| mc_n_paths | Precision-targeted | Glasserman 2003 | 10,000 | 5,000-30,000 | MC simulation path count |
| mc_is_tilt | Importance sampling | -- | 1.5 | 1.0-3.0 | MC tail event tilt factor |
| participation_rate | Amihud illiquidity | Amihud 2002 | 0.10 | 0.01-0.30 | Liquidation rate estimation |
| PID gains (Kp, Ki, Kd) | Lambda tuning | Dahlin 1968 | Data-derived | ACF half-life based | Forward pass learning rate |
| n_eff (effective sample) | Kish ESS | Kish 1965 | Computed | 1 to N | True information content |
| hurst_exponent | R/S analysis | -- | 0.5 | 0-1 | Trend vs mean-reversion |

---

## Panel 9: Adaptive Windows (Tier 3)

Filing-frequency-anchored rolling windows.

| Window | Formula | Quarterly (63d) | Semi-Annual (126d) | Annual (252d) |
|--------|---------|-----------------|-------------------|---------------|
| short | base/3 | 21 | 42 | 84 |
| medium | base | 63 | 126 | 252 |
| long | 2*base | 126 | 252 | 504 |
| trend | 4*base | 252 | 504 | 1008 |

**NN Architecture (Kaplan 2020 scaling):**
- d_model = clip(int(n_eff^0.5), 16, 128)
- hidden_dim = 2 * d_model
- dropout = clip(0.3 - 0.1 * log2(n_eff/100), 0.05, 0.5)

**Pattern Thresholds (Bulkowski 2008):**
- doji_threshold = P10 of body/range ratio
- body_threshold = P50 of body/range ratio

---

## Panel 10: Plane-Aware Model Weights

Economic plane classification adjusts which models get priority.

| Model | Default | Supply | Manufacturing | Consumption | Logistics | Finance |
|-------|---------|--------|---------------|-------------|-----------|---------|
| Forecasting (6-cascade) | 1.0 | 1.0 | **1.4** | 1.1 | 1.1 | 1.0 |
| Monte Carlo | 1.0 | **1.3** | 1.0 | 1.0 | **1.5** | **1.3** |
| Transformer | 1.0 | 0.8 | 1.0 | **1.4** | 0.9 | 1.0 |
| Cycle Decomposition | 1.0 | **1.8** | 1.2 | 1.0 | 1.2 | 0.7 |
| Pattern Detector | 1.0 | 0.7 | 1.0 | **1.5** | 0.8 | 0.8 |
| Copula | 1.0 | 1.2 | 0.8 | 1.0 | 1.0 | **1.8** |
| Particle Filter | 1.0 | **1.4** | 1.0 | 0.7 | **1.4** | 1.0 |
| DTW Analogs | 1.0 | 1.0 | 1.1 | **1.3** | 1.0 | 1.0 |
| Granger Causality | 1.0 | 1.0 | **1.5** | 0.8 | 1.0 | **1.5** |
| Transfer Entropy | 1.0 | 1.0 | **1.3** | 1.0 | 1.0 | **1.4** |

**Rationale:**
- **Supply/Resources:** Commodity cycles dominate returns. CEEMDAN cycle decomposition + MC regime switching are the primary value-add.
- **Manufacturing:** Supply chain causality drives margins. Granger/PCMCI causal networks identify leading indicators.
- **Consumption:** Pattern-driven (sentiment, earnings surprises). Transformer attention captures consumer behavior regime shifts.
- **Logistics:** Thin margins, high noise. MC + particle filter handle stochastic thin-margin dynamics better.
- **Finance:** Tail risk correlation is everything. Copula (Clayton lower tail) + Granger cross-institution contagion.

---

## Panel 11: Ensemble Prediction Aggregation Weights

10-step aggregation pipeline with layered weight computation.

### Step 1: Base Ensemble Weights (Inverse-RMSE)
```
weight[model] = (1 / RMSE[model]) / sum(1 / RMSE[all_models])
```
Models with lower RMSE get proportionally higher weight.

### Step 2: FixedShare Forecaster (Herbster & Warmuth 1998)
Online weight update with switching:
```
w_new[model] = (1-share) * w_old[model] * exp(-eta * loss) + share * (1/n_models)
```
- `share` parameter: 0.05 (allows 5% weight shift per step)
- `eta` (learning rate): 0.1

### Step 3: Regime-Blended Weights
Soft blending using dual regime probabilities:
```
blended_weight[model] = sum(regime_prob[r] * affinity[model][r]) * base_weight[model]
```

**Model-Regime Affinity:**

| Model | Bull | Bear | High Vol | Low Vol |
|-------|------|------|----------|---------|
| Kalman | 1.0 | 0.6 | 0.5 | 1.2 |
| GARCH | 0.7 | 1.0 | 1.3 | 0.6 |
| VAR | 1.0 | 0.8 | 0.7 | 1.1 |
| LSTM | 0.9 | 0.9 | 0.8 | 1.0 |
| Tree (RF/GBM/XGB) | 0.8 | 1.1 | 1.2 | 0.9 |
| Baseline | 1.0 | 1.0 | 1.0 | 1.0 |

### Step 4: Survival-Aware Weights
Soft blending during mode transitions:
```
alpha = 1 - exp(-days_in_mode / transition_halflife)
weight = (1-alpha) * previous_mode_weight + alpha * current_mode_weight
```

### Step 5: Burn-Out Calibrated Weights
Exponential gradient learning (Vovk 1990):
```
w_new = w_old * exp(-eta * loss)
```
Per-regime weight vectors from intensive retraining.

### Step 6: Walk-Forward MCS Filtering
Model Confidence Sets (Hansen et al. 2011) via `arch.bootstrap.MCS`:
- Keeps only models statistically equivalent at 5% significance
- Per-survival-mode leaderboards
- Recency-weighted RMSE (recent errors weighted 2x)

---

## Panel 12: Confidence & Uncertainty Weights

### Prediction Confidence Score
```
confidence = 0.7 * model_quality + 0.3 * survival_factor
model_quality = exp(-RMSE / RMSE_reference)
survival_factor = survival_probability
```

### Confidence Interval Width
```
spread = z_score * RMSE * sqrt(horizon_days)
risk_factor = 1 + (1 - survival_prob) * risk_multiplier
adjusted_spread = spread * risk_factor
CI = [forecast - adjusted_spread, forecast + adjusted_spread]
```

### Conformal PID Coverage Control
```
q_new = q_old + Kp*(target_coverage - empirical_coverage) + Ki*integral + Kd*derivative
```
Mondrian partitioning: separate calibration per survival mode.

### Copula Tail Widening
```
widened_CI = CI * (1 + joint_crisis_probability * tail_amplification)
tail_amplification = 0.5 (default), adjusted by Clayton lambda
```

---

## Panel 13: Graph Risk & Contagion Weights

### Edge Weights by Relationship Type

| Relationship | Default Weight | Corporate Structure Weight | What It Models |
|-------------|---------------|---------------------------|----------------|
| Parent companies | -- | **2.8x** | Parent distress -> subsidiary impact |
| Subsidiaries | -- | **2.3x** | Subsidiary distress -> parent impact |
| Suppliers | **1.2x** | -- | Supply chain disruption |
| Financial institutions | **1.3x** | -- | Credit channel contagion |
| Customers | **1.1x** | -- | Revenue channel disruption |
| Competitors | **1.0x** | -- | Market share competition |
| Logistics | **0.8x** | -- | Logistics disruption (indirect) |
| Regulators | **0.5x** | -- | Regulatory change (low direct impact) |

### Contagion Probability (SIR Model)
```
infection_prob = 1 - (1 - base_infection_rate)^(edge_weight * n_infected_neighbors)
base_infection_rate = copula_tail_dependence (from adaptive_model_params)
```

---

## Panel 14: Monte Carlo Simulation Weights

### Regime-Switching Parameters

| Parameter | Normal | Company Survival | Extreme Survival |
|-----------|--------|-----------------|-----------------|
| n_paths | 10,000 | 20,000 | 30,000 |
| importance_tilt | 1.5 | 2.0 | 3.0 |
| correlation_override | -- | 0.85 | 0.90 |
| copula_type | Best AIC | Clayton | Clayton |

### Scenario Engine Weights (USS)

| Scenario | Revenue Shift | Debt Assumption | MC Returns Used | Daily Drift |
|----------|-------------|-----------------|-----------------|------------|
| Orderly Resolution | -15% | Renegotiated | P25-P75 | -0.1%/day |
| Muddle Through | 0% | Status quo | Full distribution | 0%/day |
| Catastrophic | -40% | All called | Bottom P10 | -0.3%/day |

---

## Panel 15: Frequency Fusion Weights

### Horizon-to-Frequency Contribution Weights

| Horizon | Daily | Weekly | Monthly | Quarterly | Annual |
|---------|-------|--------|---------|-----------|--------|
| 1d | **1.00** | -- | -- | -- | -- |
| 5d | **0.70** | 0.30 | -- | -- | -- |
| 1w | 0.40 | **0.60** | -- | -- | -- |
| 21d | 0.30 | 0.40 | **0.30** | -- | -- |
| 1m | -- | 0.30 | **0.70** | -- | -- |
| 3m | -- | 0.15 | 0.35 | **0.50** | -- |
| 6m | -- | -- | 0.25 | **0.50** | 0.25 |
| 1y | -- | -- | 0.15 | 0.35 | **0.50** |
| 2y | -- | -- | -- | 0.30 | **0.70** |

**Regime Consensus:** Majority vote across frequencies. Strong (>=80% agreement), Moderate (60-80%), Weak (<60%).

**Survival Fusion:** Harmonic mean (weakest-link principle):
```
fused_prob = n / sum(1/prob_i for each frequency)
```

---

## Panel 16: Signal IC Filtering Weights

### Inclusion Thresholds (from global_config.yml)

| Metric | Threshold | What It Filters |
|--------|-----------|----------------|
| |IC| (Spearman) | >= 0.02 | Minimum predictive signal strength |
| |ICIR| (IC/std) | >= 0.3 | Minimum signal consistency |

### Signal Decay Rates (by speed category)

| Category | Decay Rate | Example Signals |
|----------|-----------|-----------------|
| Slow | 0.3%/day | Altman Z-Score, balance sheet ratios, debt_to_equity |
| Medium | 5%/day | EPS revisions, PEAD, earnings momentum |
| Fast | 30%/day | RSI, MACD, candlestick patterns, intraday signals |

---

## Panel 17: Forecasting Model Cascade Priority

Models tried in order; first that fits successfully wins.

| Priority | Model | Best For | Min Data | Typical RMSE Rank |
|----------|-------|----------|----------|-------------------|
| 1 | Kalman (local-level) | Smooth trends | 30 rows | 2-3 |
| 2 | GARCH | Volatility clusters | 50 rows | 1-2 (for vol) |
| 3 | VAR / AR(1) | Cross-variable dynamics | 50 rows (VAR) / 20 (AR) | 3-4 |
| 4 | LSTM / GBM / LR | Non-linear patterns | 200 (LSTM) / 50 (GBM) | 1-2 (with data) |
| 5 | Tree (RF/GBM/XGB) | Feature interactions | 30 rows | 2-3 |
| 6 | Baseline (EMA) | Fallback | 1 row | 4-5 |

**AutoARIMA** (statsforecast): Standalone, 100x faster than pmdarima. Used alongside cascade.

**Dynamic Factor Model**: Multi-variable DFM via statsmodels. Captures latent factor structure.

---

## Panel 18: USS (Unified Survival System) Dimension Weights

5 reconfiguration dimensions activated in survival mode.

| Dimension | What Changes | Normal | Company Survival | Extreme Survival |
|-----------|-------------|--------|-----------------|-----------------|
| 1. Variable Triage | Active/frozen tiers | All active | T4-5 frozen | T3-5 frozen |
| 2. Model Switching | Kalman noise | 1x | 3x | 5x |
| | LSTM lookback | 60d | 10d | 5d |
| | MC paths | 10K | 20K | 30K |
| | Tree max depth | 10 | 5 | 3 |
| 3. Horizon Compression | Available horizons | 1d/5d/21d/252d | 1d/5d/21d | 1d/5d |
| 4. Correlation Switching | Cross-asset correlation | Data-driven | 0.85 override | 0.90 override |
| | Copula type | Best AIC | Clayton | Clayton |
| 5. Forecast Bounding | Revenue cap | None | Last actual | Last actual * 0.85 |
| | Debt floor | None | Current level | Current * 1.15 |
| | Cash bound | None | Max burn rate | Max burn * 1.5 |

---

## Summary: Weight Sources and Optimization Methods

| Weight System | Source | Optimization | Adaptivity |
|--------------|--------|-------------|-----------|
| Hierarchy tier weights | Rule-based + Sobol | Sobol variance feedback | Per-run |
| Financial health composite | Expanding percentile rank | Adaptive PE/EV caps (3-method) | Per-day |
| Survival thresholds | 5 calibration methods | Peer P10/P90 + BOCPD + HMM crossover | Per-run |
| Survival probability blend | Inverse-variance | Cochrane 1954 | Per-run |
| Model cascade priority | Static order | First-fit wins | Static |
| Ensemble weights | Inverse-RMSE | + FixedShare online + GA/Optuna | Per-day |
| Regime blend weights | Soft probability | Dual regime probs * affinity | Per-day |
| Mode-conditioned weights | Walk-forward MCS | Mode-specific leaderboard | Per-mode |
| Burn-out weights | Exponential gradient | Vovk 1990 | Per-regime |
| Plane-aware weights | Economic plane lookup | 5-plane adjustment table | Per-company |
| Frequency fusion | Horizon-matched | Inverse walk-forward MAE | Per-frequency |
| PID controller | Error ACF half-life | Dahlin tuning | Per-tier |
| Conformal coverage | PID-controlled | Angelopoulos 2023 | Per-day per-mode |
| Graph contagion | Edge type + copula tail | SIR infection model | Per-entity |
| MC survival | Regime transition matrix | HMM + importance sampling | Per-regime |
| Signal filtering | Spearman IC + ICIR | IC decay by speed category | Per-signal |
| Conflict intensity | 4-component weighted | Static weights | Per-country |
| Fuzzy protection | Mamdani inference | 11 fuzzy rules | Per-company |
| Vanity composite | 5-component weighted | Peer-relative | Per-day |
| USS dimensions | Regime lookup table | 5 dimensions * 3 regimes | Per-regime |

# Category D Adaptive Parameters: Expert Methods and Formulas

89 model weight/affinity matrices across 10 groups. These are fundamentally different from Tiers 1-3 -- they require post-Step-6 temporal model outputs to calibrate, making them a **retroactive calibration pass**.

---

## Architectural Design: Two-Pass Calibration

Category D parameters cannot be computed before Step 6 because they depend on model performance metrics that only exist after models have run. The solution is a **two-pass architecture**:

**Pass 1 (Current)**: Run Step 6 with default weights (current behavior).
**Pass 2 (New)**: After Step 6 completes, use model outputs to calibrate weights, then selectively re-run only the prediction aggregation (Step 6r) with calibrated weights.

This is analogous to **Empirical Bayes** (Robbins 1956) -- use data from the first pass to set priors for the second pass.

---

## D1. Sector Strategicness Scores (12 constants in fuzzy_protection.py)

### Current State
```python
"energy": 0.95, "banking": 0.90, "defense": 1.0, "utilities": 0.70,
"telecom": 0.65, "healthcare": 0.50, "insurance": 0.45, "technology": 0.30, ...
```

### Why These Are Wrong

These scores assume a **universal** sector importance ranking. But sector strategicness is deeply country-specific:
- Energy: 0.95 for Saudi Arabia (83% of GDP), 0.15 for Singapore (no domestic oil)
- Banking: 0.90 for Switzerland (financial hub), 0.40 for Brazil (commodity-driven)
- Technology: 0.30 universally, but 0.90 for Taiwan (TSMC is 25% of TAIEX)

### Proposed Method

#### Method AT: GDP-Weighted Sector Importance (Leontief 1936 Input-Output)

**Theory**: A sector's strategic importance to a country is proportional to its contribution to GDP and its interconnectedness with other sectors. Leontief's input-output model quantifies this as the column sum of the Leontief inverse matrix -- the "backward linkage" coefficient.

**Simplified formula** (without full I-O table):
```
strategicness(sector, country) = 
    w1 * gdp_share(sector, country) +
    w2 * employment_share(sector, country) +
    w3 * export_share(sector, country) +
    w4 * historical_bailout_indicator(sector, country)
```

Where w1=0.4, w2=0.2, w3=0.2, w4=0.2.

**Data source**: World Bank sectoral GDP data is available via `wbgapi` (already installed). The macro_provider already fetches GDP -- extend it to fetch sectoral composition.

**Simpler alternative** (no additional API calls):
```python
# Use the country's own market composition as proxy
# If 30% of the linked entities are in this sector, it's important
def sector_strategicness(sector, market_id, linked_caches):
    same_sector = sum(1 for cache in linked_caches.values() 
                      if cache_sector(cache) == sector)
    sector_concentration = same_sector / max(len(linked_caches), 1)
    
    # Combine with static baseline (defense/energy always strategic)
    static = STATIC_SCORES.get(sector, 0.3)
    return 0.6 * static + 0.4 * min(sector_concentration * 3, 1.0)
```

**Academic basis**: Leontief (1936, Review of Economics and Statistics) input-output analysis. Rasmussen (1957) backward/forward linkage indices.

---

## D2. Graph Risk Relationship Group Weights (7 constants in graph_risk.py)

### Current State
```python
"parent_companies": 2.8, "subsidiaries": 2.3, "competitors": 1.0,
"suppliers": 1.2, "customers": 1.1, "financial_institutions": 1.3,
"logistics": 0.8, "regulators": 0.5
```

### Proposed Method

#### Method AU: Copula Tail Dependence Per Relationship Group (Joe 2014 + Engle 2002)

**Theory**: The weight of each relationship group should reflect how strongly distress transmits through that channel. Measure this as the average lower tail dependence between the target and entities in each group.

**Formula**:
```
group_weight(g) = 1.0 + 5.0 * mean(lambda_L(target, entity_i)) for entity_i in group g
```

Where lambda_L is the Student-t copula lower tail dependence. Scale by 5.0 to map from [0, 0.3] typical tail dependence to [1.0, 2.5] weight range.

**Data source**: The copula module runs on the target cache. To get per-group tail dependence, run a mini copula between the target's return_1d and each linked entity's return_1d from linked_caches.

**Fallback when linked caches are empty**: Use the current static weights.

**Alternative**: DCC-GARCH (Engle 2002) dynamic conditional correlations per group. Captures time-varying dependence (correlations spike during crises). More accurate but computationally heavier.

**Academic basis**: Joe (2014) copula tail dependence. Engle (2002, JBES) Dynamic Conditional Correlation. Billio et al. (2012, JFE) "Econometric measures of connectedness."

---

## D3. Plane-Specific Model Weights (26 constants in model_synergies.py)

### Current State
A 5x6 matrix of model affinity weights per economic plane:
```python
"supply":        {"cycle": 1.8, "mc": 1.3, "pf": 1.4, "copula": 1.2, "pattern": 0.7, "transformer": 0.8}
"manufacturing": {"forecast": 1.4, "granger": 1.5, "te": 1.3, "cycle": 1.2, "dtw": 1.1, "copula": 0.8}
...
```

### Proposed Method

#### Method AV: Walk-Forward Per-Plane Model RMSE (Timmermann 2006)

**Theory**: Instead of hand-coded affinities, use the walk-forward evaluation's per-model MAE to determine which models perform best for each economic plane. The walk_forward module already tracks per-mode-per-model errors -- extend this to per-plane tracking.

**Formula** (Timmermann 2006 forecast combination):
```
weight(model, plane) = (1 / RMSE(model, plane)) / sum_m(1 / RMSE(m, plane))
affinity(model, plane) = weight(model, plane) / weight(model, overall)
```

If a model has 2x the relative weight for "supply" companies vs its overall weight, its affinity is 2.0.

**Implementation**:
```python
def compute_plane_affinities(walk_forward_result, economic_plane):
    # Get per-model MAE from walk-forward
    model_maes = extract_model_maes(walk_forward_result)
    
    # If we had cross-company walk-forward data (from linked entities),
    # group by plane and compute per-plane RMSE.
    # For single-company: use overall MAE and adjust by model type heuristic.
    
    inv_maes = {m: 1/max(mae, 1e-8) for m, mae in model_maes.items()}
    total = sum(inv_maes.values())
    base_weights = {m: w/total for m, w in inv_maes.items()}
    
    # Affinity = relative weight vs equal weighting
    n = len(base_weights)
    return {m: w * n for m, w in base_weights.items()}
```

**Academic basis**: Timmermann (2006, Handbook of Economic Forecasting) "Forecast Combinations." Bates & Granger (1969, Operational Research Quarterly) -- the original inverse-MSE combination paper.

---

## D4. Regime-Model Affinities (9 constants in prediction_aggregator.py)

### Current State
```python
# Kalman: {"bull": 1.2, "low_vol": 1.2, "bear": 0.8, "high_vol": 0.8, ...}
# Tree:   {"bull": 0.9, "low_vol": 0.9, "bear": 1.2, "high_vol": 1.3, ...}
# GARCH:  {"bull": 0.8, "low_vol": 0.8, "bear": 1.1, "high_vol": 1.4, ...}
```

### Proposed Method

#### Method AW: Walk-Forward Mode-Conditioned Model Scoring (Already Implemented, Not Wired)

**Theory**: The walk_forward module already computes `mode_scores` -- per-model, per-survival-mode MAE scores. The `best_model_by_mode` dict tells us exactly which model performs best under each condition. These are the empirically correct affinities.

**Formula**:
```
affinity(model, regime) = (1 / MAE(model, regime)) / (1 / MAE(model, overall))
```

If Kalman has MAE 0.01 in bull but 0.03 overall, its bull affinity is 3.0.
If Tree has MAE 0.008 in bear but 0.02 overall, its bear affinity is 2.5.

**Data source**: `walk_forward_result.mode_scores` -- already computed in Step 6k. Currently unused for regime affinity, only for overall_best_model. This is the most impactful wiring fix in Category D.

**Implementation**:
```python
def compute_regime_affinities(walk_forward_result):
    if not walk_forward_result or not walk_forward_result.mode_scores:
        return {}
    
    # Group scores by model
    model_overall_mae = {}
    model_mode_mae = {}
    for score in walk_forward_result.mode_scores:
        model_overall_mae.setdefault(score.model_name, []).append(score.mae)
        model_mode_mae.setdefault(score.model_name, {})[score.survival_mode] = score.mae
    
    # Compute affinities
    affinities = {}
    for model, mode_maes in model_mode_mae.items():
        overall = np.mean(model_overall_mae[model])
        affinities[model] = {}
        for mode, mae in mode_maes.items():
            if overall > 1e-10:
                affinities[model][mode] = overall / max(mae, 1e-10)
            else:
                affinities[model][mode] = 1.0
    
    return affinities
```

**Academic basis**: Standard in adaptive model selection. Geweke & Amisano (2011, JASA) "Optimal prediction pools." The key insight: model performance varies across regimes, so static affinities are suboptimal.

---

## D5. GA Optimizer Tier-Model Priors (5 constants in genetic_optimizer.py)

### Current State
```python
"tier1": {"kalman": 1.5, "var": 1.2},
"tier2": {"kalman": 1.3, "var": 1.3},
"tier3": {"garch": 2.0, "lstm": 1.2},
"tier4": {"tree": 1.3, "lstm": 1.2},
"tier5": {"lstm": 1.5, "transformer": 1.5, "tree": 1.2}
```

### Proposed Method

#### Method AX: Forward Pass Per-Variable RMSE Seeding

**Theory**: The forward pass in Step 6i produces `model_states` with per-variable fitted model objects and their validation RMSE. Use the inverse-RMSE as the GA prior instead of hand-coded tier-model affinities.

**Formula**:
```
prior_weight(model, tier) = mean(1/RMSE(model, var)) for var in tier_variables
                            / sum_m(mean(1/RMSE(m, var)))
affinity(model, tier) = prior_weight(model, tier) * n_models  # normalize to mean=1
```

**Data source**: `forward_pass_result.model_states` and `forecast_result.metrics` -- both already computed before the GA runs.

**Academic basis**: Empirical Bayes (Robbins 1956). Use the data to set priors instead of hand-coding them.

---

## D6. Private Company Proxy Confidence (11 constants in private_company_proxies.py)

### Current State
```python
"equity_value": 0.85, "equity_change_rate": 0.55,
"financial_volatility": 0.40, "equity_drawdown": 0.75, ...
```

### Proposed Method

#### Method AY: Source-Variable Interpolation Confidence Propagation

**Theory**: Each proxy's confidence should derive from the interpolation confidence of its source variables. `equity_value` comes directly from `total_equity` (balance sheet), so its confidence = `interp_confidence_total_equity`. `equity_change_rate` is the first difference of an interpolated series, so its confidence is lower (derivatives amplify uncertainty).

**Formula**:
```
proxy_conf(p) = product(interp_conf(source_var_i) for source_var_i in sources(p))
                * derivative_penalty(p)
```

Where `derivative_penalty` = 1.0 for level proxies, 0.7 for first-difference proxies, 0.5 for second-difference proxies.

**Data source**: `interp_confidence_*` columns already in the cache from Step 4.

**Academic basis**: Error propagation (Taylor 1997). Uncertainty of a derived quantity is the product of source uncertainties (for multiplicative compositions) or the quadrature sum (for additive).

---

## D7. Interpolator Base Confidence (5 constants in frequency_interpolator.py)

### Current State
```python
"quarterly": {"confidence_base": 0.85},
"semiannual": {"confidence_base": 0.70},
"annual": {"confidence_base": 0.50},
"unknown": {"confidence_base": 0.40}
```

### Proposed Method

#### Method AZ: Cross-Validated Interpolation Accuracy

**Theory**: The base confidence should be the actual accuracy of interpolation, measured by leave-one-out cross-validation on the filing observations. Remove one filing, interpolate, compare to actual.

**Formula**:
```
base_conf(freq) = 1.0 - mean(|interpolated_i - actual_i| / |actual_i|)
```

For quarterly filers with smooth trajectories: base_conf might be 0.92.
For annual filers with volatile jumps: base_conf might be 0.35.

**Academic basis**: LOO-CV is the gold standard for interpolation accuracy assessment. Stone (1974, JRSS) "Cross-Validatory Choice and Assessment of Statistical Predictions."

---

## D8. MNAR Estimator Ensemble Priorities (1 constant in hidden_data_estimator.py)

### Current State
```python
priority = {"heckman": 0.45, "pattern": 0.30, "gain": 0.25}
```

### Proposed Method

#### Method BA: Cross-Validated Ensemble Stacking (Wolpert 1992)

**Theory**: Instead of fixed priorities, use "stacking" -- train a meta-model on the cross-validated predictions of each base estimator. The stacking weights are learned, not hand-coded.

**Formula** (constrained linear stacking):
```
y_stacked = w_heckman * pred_heckman + w_pattern * pred_pattern + w_gain * pred_gain
minimize MSE(y_stacked, y_actual) subject to w_i >= 0, sum(w_i) = 1
```

This is a constrained least-squares problem solvable in one line with `scipy.optimize.minimize`.

**Academic basis**: Wolpert (1992, Neural Networks) "Stacked Generalization." Breiman (1996) "Stacked Regressions." Van der Laan et al. (2007) "Super Learner."

---

## D9. MC Variable Sensitivity Weights (4 constants in monte_carlo.py)

### Current State
```python
"current_ratio": 0.5, "debt_to_equity_abs": -0.8,
"fcf_yield": 0.3, "drawdown_252d": 1.0
```

### Proposed Method

#### Method BB: Sobol First-Order Sensitivity Indices

**Theory**: The sensitivity of each variable to return shocks should come from the Sobol sensitivity analysis already computed in Step 6t. The Sobol first-order index S_i measures the fraction of output variance attributable to variable i.

**Formula**:
```
sensitivity(var) = sign(correlation(var, return)) * sobol_S1(var)
```

**Data source**: `sobol_result.first_order` -- already computed in Step 6t.

**Academic basis**: Sobol (1993) global sensitivity analysis. Already implemented in the pipeline -- this wires the output back as an input.

---

## D10. SIX Proxy Calibrations (55 constants in six_derived_proxies.py)

### Current State
55 hardcoded constants including blend weights, confidence assignments, calibration factors, and implied cost-of-equity assumptions. This is the most specialized module (CH market only).

### Proposed Method

#### Method BC: Swiss Market Index (SMI) Cross-Calibration

**Theory**: Use the other SMI/SLI component companies as a calibration sample. For each proxy method, compute the error against known ESEF financial data for companies that file both via SIX and ESEF.

**Formula**:
```
# For companies listed on both SIX and ESEF (many Swiss multinationals)
for company in smi_companies_with_esef:
    actual = esef_financials(company)
    proxy = six_proxy_estimate(company)
    error(method) = |proxy - actual| / |actual|

calibration_factor(method) = 1.0 / (1.0 + mean_error(method))
```

**Data source**: 8 of 20 SMI components file ESEF (Roche, Novartis, ABB, etc.). Their ESEF financials provide ground truth for calibrating the SIX proxy formulas.

**Academic basis**: Calibration by cross-referencing (standard in measurement science). The dual-listed companies serve as natural calibration anchors.

---

## Summary: Selected Methods

| Group | Count | Method | Key Insight | Data Source |
|-------|-------|--------|-------------|-------------|
| D1 | 12 | AT: GDP-weighted + market composition | Country-specific, not universal | linked_caches sector distribution |
| D2 | 7 | AU: Copula tail dependence per group | Empirical contagion measurement | Target vs linked entity returns |
| D3 | 26 | AV: Walk-forward per-plane RMSE | Data beats expert opinion | walk_forward_result.mode_scores |
| D4 | 9 | AW: Walk-forward mode-conditioned scoring | Already computed, just not wired | walk_forward_result.mode_scores |
| D5 | 5 | AX: Forward pass per-variable RMSE | Empirical Bayes priors | forecast_result.metrics |
| D6 | 11 | AY: Source-variable confidence propagation | Error propagation theory | interp_confidence_* columns |
| D7 | 5 | AZ: Leave-one-out cross-validated accuracy | Gold standard for interpolation | Filing observations |
| D8 | 1 | BA: Stacked generalization (Wolpert 1992) | Learned meta-model weights | Cross-validated predictions |
| D9 | 4 | BB: Sobol first-order indices | Already computed, just not wired | sobol_result.first_order |
| D10 | 55 | BC: SMI dual-listed cross-calibration | Natural calibration anchors | ESEF financials for SMI companies |

### Implementation Architecture

**New module**: `operator1/analysis/retroactive_calibration.py`

**Pipeline integration**: New Step 6.5 (after Step 6t Sobol, before Step 7 profile):
1. Collect all Step 6 outputs (walk_forward, sobol, forecast_result, copula_result)
2. Compute calibrated weights for D1-D9
3. Re-run prediction aggregation (Step 6r only) with calibrated weights
4. D10 is CH-market-only, runs within six_derived_proxies

**Computational cost**: Minimal. The heavy models (HMM, LSTM, MC) are NOT re-run. Only the lightweight prediction aggregation step is repeated with better weights.

### New Dependencies: None

### Academic References
- Leontief (1936) -- Input-output sector importance
- Rasmussen (1957) -- Backward/forward linkage indices
- Joe (2014) -- Copula tail dependence per relationship
- Engle (2002) -- Dynamic Conditional Correlation
- Timmermann (2006) -- Forecast combination theory
- Bates & Granger (1969) -- Inverse-MSE combination
- Geweke & Amisano (2011) -- Optimal prediction pools
- Robbins (1956) -- Empirical Bayes
- Taylor (1997) -- Error propagation
- Stone (1974) -- Cross-validation for interpolation
- Wolpert (1992) -- Stacked generalization
- Sobol (1993) -- Global sensitivity analysis
- Billio et al. (2012) -- Econometric connectedness measures

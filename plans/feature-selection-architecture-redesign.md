# Feature Selection Architecture: Analysis and Redesign Plan

*2026-04-19*

## Part 1: Current System Diagnosis

### What PCMCI Does

The current feature selection at [`granger_causality.py`](operator1/models/granger_causality.py:258) uses PCMCI (Peter-Clark Momentary Conditional Independence, Runge 2019) via tigramite:

**Input:** 25 float columns from cache (auto-selected by non-NaN count > 50)
**Test:** ParCorr (partial correlation, linear, assumes Gaussian)
**Output:** Directed causal graph with p-values per (source, target, lag) triple
**Criterion:** p < 0.05 after conditioning on ALL other variables' pasts
**Pruning:** Variables with ZERO significant links in either direction are removed from `_extra_vars`

### Why It's Killing Gap Features

The AAPL run shows: *"PCMCI causality: 17 significant pairs (density=0.028), 5 retained, 20 pruned"* followed by *"Causality pruning: kept 0 of 67 variables (67 removed)"*.

The 67 `_extra_vars` are separate from the 25 PCMCI test variables. The pruning function at line 480 checks if each `_extra_var` appears in `granger_result.retained_variables` -- but the 25 PCMCI variables are auto-selected from cache columns with >50 non-NaN obs, which biases toward price/return columns and excludes most Gap features (which are constant or near-constant for AAPL because options data, sector ETFs, and events don't vary daily).

**Three fundamental problems:**

1. **Linear-only test:** ParCorr assumes linear relationships. Options signals (put/call ratio, skew) affect price through non-linear threshold effects -- they matter when extreme, not linearly.

2. **All-or-nothing pruning:** A variable is kept only if it appears as source OR target in ANY significant pair. Gap features like `event_uncertainty_premium` have conditional predictive power (they matter near earnings) but zero unconditional predictive power.

3. **Wrong variable pool:** PCMCI runs on 25 auto-selected cache columns, which are mostly price-derived (return_1d, volatility, close, etc.). Gap features are NOT in the 25 PCMCI test variables at all -- they're only in the separate 67-item `_extra_vars` list. The pruning compares `_extra_vars` against PCMCI's `retained_variables`, which came from a different set. Naturally, zero overlap.

### Should We Replace or Edit?

**Replace.** PCMCI is designed for climate science (slow-varying, high-dimensional, long time series). Financial time series have:
- Short samples (502 days)
- Non-linear effects (options signals)
- Regime-dependent relationships (features matter in one regime, not another)
- Many near-constant features (events, sector rotation signals change infrequently)

PCMCI's strengths (handling autocorrelation, confounders) don't compensate for its weaknesses (linear-only, aggressive pruning, wrong variable pool).

## Part 2: What Experts Use

### Tier 1: Best Practices from Quantitative Finance

**1. LASSO/Elastic Net Feature Selection (Tibshirani 1996)**
- Standard in quant finance. L1 regularization drives irrelevant feature coefficients to exactly zero.
- Handles multicollinearity (common in financial data).
- Cross-validated penalty selects the sparse model that minimizes out-of-sample error.
- Non-linear extension: use LASSO on polynomial/interaction features.

**2. Permutation Importance with Cross-Validation (Breiman 2001)**
- Shuffle each feature, measure prediction degradation. If shuffling doesn't hurt, the feature is useless.
- Model-agnostic (works with tree, linear, NN).
- Already have tree models from the forecasting step -- can extract importance directly.
- Avoids the PCMCI problem of testing features that aren't even in the model.

**3. Regime-Conditional Feature Selection**
- Run feature importance SEPARATELY per regime (bull, bear, high_vol, normal).
- A feature that's useless globally might be the #1 predictor in crisis.
- The HMM regime labels are already in the cache from Step 5.5.

### Tier 2: Advanced Methods Used by Top Quant Funds

**4. Boruta (Kursa & Rudnicki 2010)**
- All-relevant feature selection (not minimal-optimal like LASSO).
- Creates "shadow features" (shuffled copies of each real feature), trains Random Forest on real+shadow, compares importance. Features that never beat their shadow are removed.
- Particularly good at detecting features with non-linear or interaction effects.
- Used by Two Sigma, Renaissance Technologies.

**5. mRMR (Minimum Redundancy Maximum Relevance, Peng 2005)**
- Selects features that are maximally relevant to the target AND minimally redundant with each other.
- Uses mutual information (captures non-linear relationships).
- Avoids the "pick 5 correlated features" problem.

**6. PCMCI+ with CMIknn (Runge 2020)**
- The non-linear extension of our current PCMCI.
- Uses k-nearest-neighbor conditional mutual information instead of ParCorr.
- Captures non-linear causal relationships.
- Still in tigramite library.
- Problem: very slow (O(n^2 * k * d)) for 25+ variables.

### Tier 3: Unconventional but Effective

**7. Knockoff Filter (Barber & Candes 2015)**
- Creates synthetic "knockoff" copies of features that mimic the correlation structure but have no relationship to the target.
- Compares real feature importance vs knockoff importance.
- Provides finite-sample FDR (False Discovery Rate) control.
- Used by Citadel's systematic strategies group.

**8. Causal Forest Feature Importance (Athey & Imbens 2019)**
- Heterogeneous treatment effect estimation.
- Finds features where the effect of one variable on another DEPENDS on context.
- Perfect for "put_call_ratio matters when > 1.5 but not otherwise".

## Part 3: Recommended Architecture

### Replace the current single-method filter with a 3-layer selection system:

```
Layer 1: Boruta (all-relevant)
    Keeps: any feature that ever beats its shadow in RF
    Removes: pure noise features
    Time: ~30s (500 rows x 70 features)
    
Layer 2: Regime-Conditional Permutation Importance
    For each HMM regime:
        Train XGBoost on return_1d using features from Layer 1
        Compute permutation importance
        Keep features with importance > 0 in ANY regime
    Time: ~10s per regime x 4 regimes = 40s
    
Layer 3: mRMR Redundancy Removal (optional, for very large feature sets)
    Among features surviving Layers 1-2:
        Maximize mutual information with target
        Minimize mutual information between selected features
    Cap at top 30 features (prevent overfitting)
    Time: ~5s
```

### Why This Is Better

| Criterion | Current PCMCI | Proposed 3-Layer |
|-----------|--------------|-----------------|
| Non-linear relationships | No (ParCorr is linear) | Yes (RF + XGBoost) |
| Regime-conditional features | No (single global test) | Yes (per-regime importance) |
| Correct variable pool | No (tests 25, prunes 67) | Yes (tests exactly the features we want to select) |
| Gap feature survival | 0/22 retained for AAPL | Expected: 5-10 retained (options, events, cross-asset in relevant regimes) |
| Runtime | ~2s | ~75s (acceptable, runs once) |
| False positives | Very low (too conservative) | Controlled via Boruta shadow comparison |

### What Happens to PCMCI/Granger?

Keep it as **an informational module** (Stage 3.3) that produces the causal graph for the profile and report, but **remove it from the feature pruning pipeline**. The causal graph is valuable for explainability ("revenue Granger-causes close at lag 3") but should not gate which features enter the models.

### Implementation Changes

1. **New file:** `operator1/models/feature_selector.py` (~300 lines)
   - `select_features_boruta()` -- Layer 1
   - `select_features_regime_importance()` -- Layer 2
   - `select_features_mrmr()` -- Layer 3 (optional)
   - `run_feature_selection()` -- orchestrator

2. **Edit:** `operator1/stages/stage3_temporal.py` run_3_3_granger
   - Remove the `prune_features_by_causality()` call
   - Add `run_feature_selection()` call after 3.7 synergies

3. **Edit:** `main.py` Step 6c
   - Same change: remove pruning, add feature selection

4. **Edit:** `operator1/models/granger_causality.py`
   - Keep `compute_granger_causality()` for profile/report
   - Keep `compute_time_varying_granger()` for profile
   - Deprecate `prune_features_by_causality()` (keep for backward compat but don't call)

5. **New sub-stage:** `3.8` Feature Selection (after 3.7 synergies, before 4.1 forecasting)

### Dependencies

- `boruta` -- not currently installed. Alternative: implement shadow feature comparison directly using sklearn RandomForestRegressor (avoids new dependency).
- `mrmr-selection` -- not installed. Alternative: use `sklearn.feature_selection.mutual_info_regression` for MI computation + greedy mRMR loop.
- Both can be implemented with existing dependencies (sklearn, xgboost, numpy).

---

## Part 4: Community Project Reference Implementations

### Layer 1: Boruta -- Reference Implementations

**1. BorutaPy (scikit-learn-contrib/boruta_py)**
- GitHub: `scikit-learn-contrib/boruta_py` (1.8K stars)
- Pure sklearn wrapper around the original R Boruta algorithm
- Key implementation patterns we should adopt:
  - **Shadow feature generation:** `np.apply_along_axis(np.random.permutation, 0, X)` -- shuffle each column independently to create shadow copies
  - **Two-tail test:** Uses binomial test instead of simple max-comparison. A feature must beat its best shadow significantly (p < 0.05) across N iterations, not just once
  - **Tentative handling:** Features that are borderline (sometimes beat shadow, sometimes don't) get extra rounds instead of immediate rejection. BorutaPy uses 100 iterations by default with early stopping
  - **Correction:** Bonferroni correction for multiple testing (`alpha / n_features`)
  - **Max depth control:** Sets `max_depth=7` for the internal RF to prevent overfitting to noise features in small samples

**Key code pattern to adopt:**
```python
# From BorutaPy: the shadow feature comparison loop
for trial in range(n_trials):
    X_shadow = np.apply_along_axis(np.random.permutation, 0, X)
    X_combined = np.hstack([X, X_shadow])
    rf.fit(X_combined, y)
    importances = rf.feature_importances_
    real_imp = importances[:n_real]
    shadow_imp = importances[n_real:]
    shadow_max = shadow_imp.max()
    hits[real_imp > shadow_max] += 1
# Binomial test: feature significant if hits > expected by chance
from scipy.stats import binom
for i in range(n_real):
    p_value = 1 - binom.cdf(hits[i] - 1, n_trials, 0.5)
    if p_value < alpha / n_real:  # Bonferroni
        confirmed[i] = True
```

**2. SHAP-based Boruta (BorutaShap)**
- GitHub: `Ekeany/Boruta-Shap` (500+ stars)
- Replaces RF feature importance with SHAP values for the shadow comparison
- More accurate for correlated features (RF importance dilutes across correlated features, SHAP doesn't)
- We already have SHAP installed -- can use TreeExplainer for fast SHAP computation
- Slower than standard Boruta but more reliable for financial data where features are heavily correlated

**3. Regime-Aware Boruta (custom, from quant finance blogs)**
- Run Boruta separately per regime segment of the data
- Union the confirmed features across all regimes
- A feature confirmed in ANY regime survives (captures crisis-only predictors)
- Not available as a package -- needs custom implementation (~30 lines on top of standard Boruta)

### Layer 2: Regime-Conditional Permutation Importance -- Reference Implementations

**4. sklearn permutation_importance (built-in)**
- `sklearn.inspection.permutation_importance` -- already installed
- Key parameters: `n_repeats=10` (shuffle each feature 10 times, take mean degradation), `scoring='neg_mean_absolute_error'`
- Returns `importance_mean` and `importance_std` per feature
- We wrap this to run per-regime: split data by HMM regime labels, run importance on each segment

**5. ELI5 (TeamHG-Memex/eli5)**
- GitHub: `TeamHG-Memex/eli5` (2.7K stars)
- Extends sklearn permutation importance with:
  - **Target-specific importance:** Can compute importance for specific prediction targets (not just overall)
  - **Feature interaction detection:** Permutes pairs of features to find interaction effects
  - **Confidence intervals:** Bootstrap CIs on importance scores
- Not installing the package, but adopting the pairwise permutation pattern:

```python
# Pairwise feature interaction importance (from ELI5 concept)
for i in range(n_features):
    for j in range(i+1, n_features):
        X_shuffled = X.copy()
        X_shuffled[:, [i, j]] = np.random.permutation(X_shuffled[:, [i, j]])
        interaction_imp = base_score - model.score(X_shuffled, y)
        if interaction_imp > individual_imp[i] + individual_imp[j]:
            # Synergistic interaction detected
            interaction_features.add((features[i], features[j]))
```

**6. PIMP (Permutation Importance with P-values, Altmann 2010)**
- Computes null distribution of importance by permuting the TARGET (not features)
- Each permutation gives a "chance" importance value
- Real importance compared to null distribution gives a p-value
- Much more statistically rigorous than simple permutation importance
- Implementation pattern:

```python
# PIMP: null distribution approach
null_importances = []
for _ in range(100):
    y_perm = np.random.permutation(y)
    model_perm = clone(model).fit(X, y_perm)
    null_importances.append(model_perm.feature_importances_)
null_dist = np.array(null_importances)  # shape: (100, n_features)
real_imp = model.feature_importances_
p_values = np.mean(null_dist >= real_imp, axis=0)
significant = p_values < 0.05
```

### Layer 3: mRMR -- Reference Implementations

**7. mrmr-selection (smazzanti/mrmr)**
- GitHub: `smazzanti/mrmr` (1.1K stars)
- Pure pandas implementation, no heavy dependencies
- Key algorithm (greedy forward selection):

```python
# mRMR greedy selection (from smazzanti/mrmr)
def mrmr_regression(X, y, K):
    # Relevance: MI(feature, target) for each feature
    relevance = mutual_info_regression(X, y)
    selected = []
    remaining = list(range(X.shape[1]))
    
    # First feature: highest relevance
    best = remaining[np.argmax(relevance[remaining])]
    selected.append(best)
    remaining.remove(best)
    
    for _ in range(K - 1):
        scores = []
        for f in remaining:
            rel = relevance[f]
            # Redundancy: mean MI with already-selected features
            red = np.mean([mutual_info_regression(
                X[:, [f]], X[:, s].ravel()
            )[0] for s in selected])
            scores.append(rel - red)  # mRMR = max(relevance - redundancy)
        best = remaining[np.argmax(scores)]
        selected.append(best)
        remaining.remove(best)
    return selected
```

**8. FCBF (Fast Correlation-Based Filter, Yu & Liu 2003)**
- Alternative to mRMR that's faster for high-dimensional data
- Uses symmetrical uncertainty (normalized MI) instead of raw MI
- Two-phase: (1) rank by SU with target, (2) remove redundant features where SU(fi, fj) >= SU(fi, target)
- Runs in O(n * m) vs mRMR's O(K * m * n)
- Good pattern for our ~70 feature case

### Cross-Cutting: Financial-Specific Methods from Quant Projects

**9. Marcos Lopez de Prado's Feature Importance (mlfinlab)**
- GitHub: `hudson-and-thames/mlfinlab` (4K stars, now commercial)
- **Mean Decrease Impurity (MDI) with substitution effect correction:** Standard RF importance overestimates correlated features. MDI-corrected clusters correlated features first, computes importance per cluster, then distributes back
- **Clustered Feature Importance (CFI):** Groups features by correlation (agglomerative clustering at threshold 0.5), computes importance per cluster, assigns cluster importance to all members. Prevents correlated features from splitting importance
- **Single Feature Importance (SFI):** Trains a separate model per feature. Slow but finds features that work independently (useful for our regime-conditional analysis)

**Key pattern to adopt (CFI):**
```python
# Clustered Feature Importance (from mlfinlab concept)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

corr = X.corr().abs()
dist = 1 - corr
linkage_matrix = linkage(squareform(dist), method='average')
clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')

# Compute importance per cluster (average of member importances)
cluster_imp = {}
for c in np.unique(clusters):
    members = np.where(clusters == c)[0]
    cluster_imp[c] = np.mean(importances[members])

# Assign cluster importance back to individual features
adjusted_imp = np.zeros_like(importances)
for c in np.unique(clusters):
    members = np.where(clusters == c)[0]
    adjusted_imp[members] = cluster_imp[c]
```

**10. Orthogonal Features (from QuantConnect/Lean research)**
- Decorrelate features via PCA, but keep the mapping back to original features
- Run feature selection on the orthogonal components
- Map selected components back to original features via loading matrix
- Avoids the "two correlated features split importance" problem entirely

### Recommended Implementation Strategy

Implement in-house using sklearn + numpy (no new dependencies):

| Layer | Method | Source | New Dependencies |
|-------|--------|--------|-----------------|
| Layer 1 | BorutaPy-style shadow comparison | `scikit-learn-contrib/boruta_py` pattern | None (sklearn RF) |
| Layer 1b | CFI decorrelation | `mlfinlab` clustered feature importance pattern | None (scipy linkage) |
| Layer 2 | PIMP (permutation with null distribution) | Altmann 2010 pattern | None (sklearn) |
| Layer 2b | Regime-conditional split | Custom (split by HMM labels, run PIMP per segment) | None |
| Layer 3 | Greedy mRMR | `smazzanti/mrmr` pattern | None (sklearn mutual_info_regression) |

Total new code: ~300 lines in `operator1/models/feature_selector.py`
New dependencies: zero
Runtime: ~75s (acceptable, runs once per pipeline execution)

---

## Part 5: Input/Output Mapping -- Current Granger vs New System

### Current Granger System: Exact Inputs

**Sub-stage 3.3 reads from PipelineState:**

| Input | Source | Type | Description |
|-------|--------|------|-------------|
| `state.cache` | Steps 1-5k.2 | DataFrame 502x337 | Daily cache with all features |
| `state.extra_vars` | Sub-stage 3.1 `_init_extra_vars()` | list of 67 strings | Candidate feature column names for temporal models |
| `state.is_private` | Stage 1 | bool | Whether company has no OHLCV |

**Variables fed to PCMCI (auto-selected, NOT from extra_vars):**

```
gc_vars = top 25 float columns with >50 non-NaN observations
```

These are typically: `close`, `open`, `high`, `low`, `volume`, `return_1d`, `return_5d`, `return_21d`, `volatility_21d`, `volatility_63d`, `drawdown_252d`, `revenue`, `total_assets`, `total_equity`, `net_income`, `operating_cash_flow`, `fh_composite_score`, `survival_probability`, `adx_14`, `obv`, `bb_width_20`, `macd_histogram`, `beta_252d`, plus 2-3 others.

**Note:** Gap features (`put_call_ratio`, `sector_relative_strength`, `days_to_next_event`, etc.) often have <50 non-NaN observations (they're constant or sparse for many companies) and so are NOT included in the 25 PCMCI test variables.

### Current Granger System: Exact Outputs

| Output | Written to | Type | Consumers |
|--------|-----------|------|-----------|
| `state.granger_result.significant_pairs` | PipelineState | list of dicts | Profile builder (3.3 -> 7 profile), report (Section 19) |
| `state.granger_result.retained_variables` | PipelineState | list of 5 strings | Pruning function |
| `state.granger_result.pruned_variables` | PipelineState | list of 20 strings | Pruning function |
| `state.granger_result.network_density` | PipelineState | float | Profile, report |
| `state.granger_result.causality_matrix` | PipelineState | dict | Profile, TV Granger (6.8) |
| **`state.extra_vars` (PRUNED)** | PipelineState | **list of 0 strings** | **4.1 Forecasting, 5.1 Forward Pass, 5.2 Burnout** |

The critical output is `state.extra_vars` going from 67 to 0. This starves ALL downstream models of Gap features.

### New System: Sub-Stage Architecture

```
[3.3 Granger/PCMCI]  -- INFORMATIONAL ONLY (no pruning)
    Input: state.cache, 25 auto-selected columns
    Output: state.granger_result (causal graph for profile/report)
    Does NOT modify: state.extra_vars
    Saves: checkpoint 3.3
    |
    v
[3.4 Transfer Entropy] -- unchanged
    |
    v
[3.5 Cycle Decomposition] -- unchanged
    |
    v
[3.6 Pattern Detection] -- unchanged
    |
    v
[3.7 Pre-Forecasting Synergies] -- unchanged
    Input adds: cycle features, unified causal network
    Output: state.cache enriched, state.extra_vars updated with synergy features
    Saves: checkpoint 3.7
    |
    v
[3.8 Feature Selection]  -- NEW SUB-STAGE
    Input: state.cache, state.extra_vars (67 features),
           state.regime_detector (HMM labels for regime split),
           state.granger_result (causal graph for CFI clustering)
    Output: state.extra_vars (pruned to 15-30 features),
            state.feature_selection_result (new result object)
    Saves: checkpoint 3.8
    |
    v
[4.1 Forecasting]  -- unchanged, but now receives 15-30 features instead of 0
```

### Sub-Stage 3.8 Detailed Input/Output Contract

**Inputs:**

| Input | From | Purpose in New System |
|-------|------|----------------------|
| `state.cache` DataFrame | 3.7 checkpoint | Raw data for RF/XGBoost training |
| `state.extra_vars` list | 3.7 (67 items) | Candidate features to select from |
| `state.regime_detector` | 3.1 | HMM regime labels for regime-conditional importance (Layer 2) |
| `state.granger_result` | 3.3 | Causal graph for CFI correlation clustering (optional enrichment) |
| `state.is_private` | Stage 1 | Target variable selection (return_1d vs equity_change_rate) |

**Outputs:**

| Output | Type | Consumers | Description |
|--------|------|-----------|-------------|
| `state.extra_vars` | list of 15-30 strings | 4.1 Forecasting, 5.1 Forward Pass, 5.2 Burnout | Pruned feature list |
| `state.feature_selection_result` | FeatureSelectionResult | Profile builder, report | New result dataclass |

**FeatureSelectionResult dataclass:**

```python
@dataclass
class FeatureSelectionResult:
    # Layer 1: Boruta
    boruta_confirmed: list[str]       # Features confirmed by shadow comparison
    boruta_tentative: list[str]       # Borderline features
    boruta_rejected: list[str]        # Pure noise
    boruta_n_iterations: int

    # Layer 2: Regime-Conditional Importance
    regime_importances: dict[str, dict[str, float]]  # {regime: {feature: importance}}
    regime_selected: dict[str, list[str]]  # {regime: [features selected in this regime]}
    pimp_p_values: dict[str, float]   # Per-feature p-values from null distribution

    # Layer 3: mRMR
    mrmr_selected: list[str]          # Final ordered list (most relevant first)
    mrmr_scores: dict[str, float]     # Relevance - Redundancy score per feature

    # Combined output
    final_selected: list[str]         # Union of all layers
    n_input: int                      # How many candidates entered
    n_output: int                     # How many survived
    method_contributions: dict[str, int]  # How many features each layer contributed

    fitted: bool = False
    error: str | None = None
```

### Output Comparison: Granger vs New System

| Aspect | Current Granger Output | New System Output | Enrichment |
|--------|----------------------|-------------------|------------|
| **Feature list for models** | 0 features (all pruned) | 15-30 features | Models actually receive features |
| **Causal graph** | Yes (17 pairs) | Yes (unchanged, 3.3 still runs) | Same |
| **Per-regime importance** | No | Yes (importance per bull/bear/high_vol) | **NEW** -- downstream models could weight features differently per regime |
| **Feature p-values** | No (binary keep/prune) | Yes (PIMP null distribution p-values) | **NEW** -- confidence in each selected feature |
| **Redundancy scores** | No | Yes (mRMR relevance-redundancy) | **NEW** -- identifies the most independent features |
| **Feature clusters** | No | Yes (CFI correlation clusters) | **NEW** -- groups correlated features |

### New Entry Points Needed in Downstream Models

**Question:** Does the richer output from the new feature selection system require new entry points in downstream models?

**Answer:** Two optional enhancements, one required change:

**Required:**
1. **Forecasting (4.1):** No code change needed. `state.extra_vars` is already the input. It just receives 15-30 features instead of 0.

**Optional but high-value:**
2. **Prediction Aggregator (6.5):** Add a `feature_importance_weights` parameter. When `state.feature_selection_result.pimp_p_values` is available, weight the prediction confidence by how many statistically significant features contributed. A prediction driven by 10 confirmed features is more reliable than one driven by 3 tentative features.

3. **Monte Carlo (5.4):** Add regime-conditional feature importance as an input. When `state.feature_selection_result.regime_importances["crisis"]` shows that `put_call_ratio` is the top crisis feature, the MC stress test could use it to calibrate crisis path probabilities.

These are enhancements, not requirements. The system works without them -- models receive features via `state.extra_vars` and the tree/XGBoost models will learn the importance themselves. The enhancements just propagate the feature selection insights deeper into the pipeline.

### Sub-Stage Checkpoint Chain

```
3.3: Saves granger_result (causal graph, informational)
     Does NOT prune extra_vars anymore
     
3.4: Saves transfer_entropy_result
     
3.5: Saves cycle_result
     
3.6: Saves pattern_result
     
3.7: Saves enriched cache + synergy_meta + extra_vars (still 67 items)
     
3.8: Saves feature_selection_result + PRUNED extra_vars (15-30 items)
     This is the ONLY sub-stage that modifies extra_vars
     
4.1: Reads extra_vars (15-30 items) and trains models with them
```

Each sub-stage saves its results via `state.save(sub_id)` which pickles the entire PipelineState. The next sub-stage loads from the previous checkpoint. This means:
- Running `--stage 3.8` alone (after 3.7 completed) will load the full 67-item extra_vars, run selection, save the pruned list
- Running `--stage 4.1` after 3.8 will load the pruned 15-30 item list
- Running `--stage 3.3` no longer affects which features reach the models

---

## Part 6: Step-by-Step Implementation Instructions

These are the exact steps to follow in Code mode. Each step produces a testable artifact.

### Step 1: Create feature_selector.py (~300 lines)

Create `operator1/models/feature_selector.py` with:

- 1a. `FeatureSelectionResult` dataclass (from Part 5 spec)
- 1b. `select_features_boruta(cache, target_col, candidates, n_trials=50)` -- Shadow feature comparison with binomial test + Bonferroni correction. RF max_depth=7, n_estimators=100. Read `boruta_*` params from `scoring_weights.yml`.
- 1c. `select_features_regime_importance(cache, target_col, candidates, regime_labels)` -- Split cache by regime_label. For each segment >= 30 rows: train XGBoost, run PIMP (50 target permutations, build null importance, p < 0.05). Union features confirmed in ANY regime.
- 1d. `select_features_mrmr(cache, target_col, candidates, K=30)` -- Greedy forward selection via sklearn `mutual_info_regression`. max(relevance - mean_redundancy). Return top K ordered list.
- 1e. `run_feature_selection(cache, candidates, regime_labels, target_col, granger_result=None)` -- Orchestrator: Layer 1 Boruta -> Layer 2 PIMP -> Layer 3 mRMR. Optional CFI decorrelation using granger_result for cluster seeds. Populate and return FeatureSelectionResult.

### Step 2: Expand _extra_vars to include missing data (Gaps B + C)

Edit `operator1/stages/stage3_temporal.py` `_init_extra_vars()` AND `main.py` Step 6 `_extra_vars` construction:

- 2a. Add raw macro column names to the explicit list: `"gdp_growth"`, `"inflation_rate_yoy"`, `"real_interest_rate"`, `"unemployment_rate"`, `"official_exchange_rate_lcu_per_usd"`
- 2b. Add estimation confidence columns via suffix match: `or c.endswith("_confidence")`
- 2c. Add interpolation confidence columns via prefix match: `or c.startswith("interp_confidence_")`

### Step 3: Remove Granger pruning from sub-stage 3.3

Edit `operator1/stages/stage3_temporal.py` `run_3_3_granger()`:

- 3a. Remove the `prune_features_by_causality()` call (lines 159-161)
- 3b. Keep `compute_granger_causality()` for informational/profile purposes
- 3c. Granger result still saved to `state.granger_result`

Edit `main.py` Step 6c similarly:

- 3d. Remove `prune_features_by_causality()` call (around line 2746)
- 3e. Keep `compute_granger_causality()` and logging

### Step 4: Add sub-stage 3.8 (Feature Selection)

Edit `operator1/stages/stage3_temporal.py`:

- 4a. Add `run_3_8_feature_selection(state)` function that calls `run_feature_selection()` from Step 1
- 4b. Append `("3.8", run_3_8_feature_selection)` to `STAGE_3_SUBSTAGES` list

### Step 5: Add feature_selection_result to PipelineState

Edit `operator1/pipeline_state.py`:

- 5a. Add `self.feature_selection_result: Any = None` in `__init__`

### Step 6: Mirror changes in backtest_runner.py

Edit `backtest_runner.py`:

- 6a. In `_init_extra_vars()`: same changes as Step 2
- 6b. In `run_stage2a1()`: remove `prune_features_by_causality()` call
- 6c. Add feature selection call after synergies, before save

### Step 7: Wire into main.py inline path

Edit `main.py`:

- 7a. After Step 6g (synergies) and before Step 6h (forecasting): add feature selection call
- 7b. Store result in `feature_selection_result`
- 7c. Copy to PipelineState if using staged path

### Step 8: Add feature_selection_result to profile

Edit `main.py` Step 7:

- 8a. Inject `profile["feature_selection"]` with confirmed/selected/contributions data

### Step 9: Add to scoring_weights.yml

Edit `config/scoring_weights.yml`:

- 9a. Add `feature_selection:` section with `boruta_n_trials`, `boruta_alpha`, `boruta_rf_max_depth`, `pimp_n_permutations`, `pimp_significance`, `mrmr_max_features`, `min_regime_samples`

### Step 10: Add report section

Edit `operator1/report/report_generator.py`:

- 10a. Add `_build_feature_selection_section(profile)` builder
- 10b. Add section number 2035 to `TIER_SECTIONS` for Premium
- 10c. Add to `_section_builders` dict

### Step 11: Compile, test, verify

- 11a. `python -m py_compile` all modified files
- 11b. Run AAPL backtest Stage 1 + Stage 2a1 to verify feature selection works
- 11c. Compare extra_vars count: should be 15-30 (was 0)
- 11d. Run full backtest and validation to check prediction accuracy impact

### Step 12: Commit, push, create PR

- 12a. Branch: `feature/replace-granger-with-boruta-pimp-mrmr`
- 12b. Commit message: `feat: replace Granger feature pruning with Boruta + PIMP + mRMR 3-layer selection`
- 12c. Push and create PR

# Integration Plan: 8 Core Idea Principles into Operator 1

Concrete file-by-file implementation plan with function signatures, wiring points in main.py, and execution order.

---

## Phase 1: Quick Wins (Low complexity, no new deps)

### Item 6: Missing Derived Variables

**Files to modify:** `operator1/features/derived_variables.py`

**Add after the existing profitability section (~line 400):**

```python
def _compute_extended_ratios(cache: pd.DataFrame) -> pd.DataFrame:
    # quick_ratio = (cash + receivables) / current_liabilities
    if all(c in cache.columns for c in ["cash_and_equivalents", "receivables", "current_liabilities"]):
        cache["quick_ratio"] = safe_ratio(
            cache["cash_and_equivalents"] + cache.get("receivables", 0),
            cache["current_liabilities"]
        )
    
    # roe = net_income / total_equity (TTM if available)
    cache["roe"] = safe_ratio(cache.get("net_income"), cache.get("total_equity"))
    
    # roa = net_income / total_assets
    cache["roa"] = safe_ratio(cache.get("net_income"), cache.get("total_assets"))
    
    # ps_ratio = market_cap / revenue
    if "close" in cache.columns and "shares_outstanding" in cache.columns:
        mcap = cache["close"] * cache["shares_outstanding"]
        cache["ps_ratio_calc"] = safe_ratio(mcap, cache.get("revenue"))
        cache["pb_ratio"] = safe_ratio(mcap, cache.get("total_equity"))
        cache["enterprise_value"] = mcap + cache.get("total_debt", 0) - cache.get("cash_and_equivalents", 0)
    
    return cache
```

**Wiring:** Call from `compute_derived_variables()` before return.

---

### Item 7: Relative Linked Metrics

**Files to modify:** `operator1/features/linked_aggregates.py`

**Add new function:**

```python
def compute_relative_metrics(
    target_cache: pd.DataFrame,
    linked_agg_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute company-relative metrics vs sector/industry aggregates."""
    result = pd.DataFrame(index=target_cache.index)
    
    # rel_strength_vs_sector = company return_21d - sector median return_21d
    if "return_21d" in target_cache.columns:
        for prefix in ["competitors_avg_", "sector_peers_avg_"]:
            peer_col = f"{prefix}return_21d"
            if peer_col in linked_agg_df.columns:
                result["rel_strength_vs_sector"] = target_cache["return_21d"] - linked_agg_df[peer_col]
                break
    
    # valuation_premium_vs_industry = company PE - industry median PE
    if "pe_ratio_calc" in target_cache.columns:
        for prefix in ["competitors_avg_", "industry_peers_avg_"]:
            peer_col = f"{prefix}pe_ratio_calc"
            if peer_col in linked_agg_df.columns:
                result["valuation_premium_vs_industry"] = target_cache["pe_ratio_calc"] - linked_agg_df[peer_col]
                break
    
    return result
```

**Wiring:** Call in main.py Step 5g after `compute_linked_aggregates()`, merge result into cache.

---

### Item 4: Per-Tier Accuracy Metrics

**Files to modify:** `operator1/models/forecasting.py` (in `run_forward_pass()`)

**Add at the end of forward pass, before return:**

```python
# Compute per-tier accuracy from last 20 days of errors
tier_accuracy = {}
for tier_num in range(1, 6):
    tier_errors = errors_by_tier.get(tier_num, [])
    if len(tier_errors) >= 20:
        recent = tier_errors[-20:]
        # Normalized MAE: 1 - (MAE / variable_std)
        tier_accuracy[f"tier{tier_num}"] = max(0, 1.0 - np.mean(recent))
    else:
        tier_accuracy[f"tier{tier_num}"] = float("nan")

result.tier_accuracy = tier_accuracy
```

**Profile wiring:** In main.py Step 7, add to profile:
```python
if forward_pass_result is not None:
    profile["model_metrics"]["per_tier_accuracy"] = forward_pass_result.tier_accuracy
```

---

### Item 8: Survival Confidence Multipliers

**Files to modify:** `operator1/models/prediction_aggregator.py`

**Add constant and apply in the interval computation section (~line 2097):**

```python
_SURVIVAL_CONFIDENCE_MULTIPLIERS = {
    1: 1.0,   # Full confidence in liquidity
    2: 1.0,   # Full confidence in solvency
    3: 0.9,   # Slightly reduced for market stability
    4: 0.5,   # Low confidence in profitability during crisis
    5: 0.3,   # Very low confidence in growth during crisis
}

# In the per-variable interval computation:
if survival_mode_active:
    tier = _get_tier_for_variable(var_name, tier_map)
    tier_num = int(tier.replace("tier", "")) if tier else 3
    conf_mult = _SURVIVAL_CONFIDENCE_MULTIPLIERS.get(tier_num, 1.0)
    # Widen interval by inverse of confidence multiplier
    half_width = (upper - lower) / 2.0
    lower = point - half_width / conf_mult
    upper = point + half_width / conf_mult
```

---

## Phase 2: Medium Complexity (important for prediction quality)

### Item 1: Weighted Loss (observed > estimated)

**Files to modify:** `operator1/models/forecasting.py` (forward pass loop)

**Approach:** Add `_source` column awareness to the loss computation.

```python
def _weighted_prediction_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    sources: np.ndarray,  # 1.0 for observed, 0.3 for estimated
) -> float:
    """Compute weighted MSE where observed values count 3x more."""
    weights = np.where(sources == "observed", 1.0, 0.3)
    weighted_errors = weights * (predicted - actual) ** 2
    return float(np.mean(weighted_errors))
```

**Wiring in forward pass (Step D in the forward pass loop):**
```python
# For each variable, check if _source column exists
for var in tier_vars:
    source_col = f"{var}_source"
    if source_col in cache.columns:
        w = 1.0 if cache.iloc[t][source_col] == "observed" else 0.3
    else:
        w = 1.0
    tier_weighted_error += w * (pred[var] - actual[var]) ** 2
```

**Curriculum learning in burn-out:**
```python
# In run_burnout(), add iteration-dependent weight schedule
def _burnout_est_weight(iteration: int, max_iterations: int) -> float:
    """Anneal estimated-data weight from 0 to 0.3 over burn-out."""
    return 0.3 * min(1.0, iteration / max(max_iterations * 0.5, 1))
```

---

### Item 3: Regime-Weighted Burn-Out Windows

**Files to modify:** `operator1/models/forecasting.py` (in `run_burnout()`)

**Add regime-weighted sampling:**

```python
def _compute_burnout_sample_weights(
    cache: pd.DataFrame,
    current_regime_probs: np.ndarray,
    half_life: int = 63,
) -> np.ndarray:
    """Compute per-day weights for burn-out: recency * regime_similarity."""
    n = len(cache)
    
    # Recency: exponential decay
    days_ago = np.arange(n, 0, -1)
    recency = np.exp(-days_ago / half_life)
    
    # Regime similarity: Gaussian kernel on HMM probability vectors
    if "regime_hmm_prob_0" in cache.columns:
        regime_cols = [c for c in cache.columns if c.startswith("regime_hmm_prob_")]
        regime_matrix = cache[regime_cols].values
        current = regime_matrix[-1]
        similarity = np.exp(-np.sum((regime_matrix - current) ** 2, axis=1) / 0.5)
    else:
        similarity = np.ones(n)
    
    weights = recency * similarity
    return weights / weights.sum()
```

**Wiring:** In `run_burnout()`, replace fixed window slicing with weighted sampling:
```python
sample_weights = _compute_burnout_sample_weights(cache, current_regime_probs)
# Use these weights when computing burn-out errors
```

---

### Item 5: Original Vanity Components

**Files to modify:** `operator1/analysis/vanity.py`

**Add 2 new component functions:**

```python
def _compute_exec_comp_excess(cache: pd.DataFrame) -> pd.Series:
    """Executive compensation excess: comp > 5% of net_income."""
    # Try to get exec comp from cache (populated by us_edgar get_executives)
    if "executive_compensation" not in cache.columns:
        return pd.Series(0.0, index=cache.index)
    threshold = cache.get("net_income", pd.Series(0)).fillna(0).abs() * 0.05
    excess = (cache["executive_compensation"].fillna(0) - threshold).clip(lower=0)
    return excess

def _compute_marketing_excess(cache: pd.DataFrame) -> pd.Series:
    """Marketing excess during survival: marketing > 10% of revenue in distress."""
    if "marketing_expenses" not in cache.columns:
        return pd.Series(0.0, index=cache.index)
    threshold = cache.get("revenue", pd.Series(0)).fillna(0) * 0.10
    in_survival = cache.get("company_survival_mode_flag", pd.Series(0)).fillna(0)
    excess = (cache["marketing_expenses"].fillna(0) - threshold).clip(lower=0) * in_survival
    return excess
```

**Wiring:** Add to the existing vanity composite computation.

---

## Phase 3: Major Enhancement (per-regime model parameters)

### Item 2: Per-Regime Model Parameters

**Approach:** Regime-conditioned ensemble using existing models.

**Files to modify:** `operator1/models/forecasting.py`

**New function:**

```python
def _fit_per_regime_models(
    cache: pd.DataFrame,
    var_name: str,
    regime_labels: pd.Series,
    model_fn,  # e.g., fit_kalman, fit_var
) -> dict:
    """Train separate model instances per regime."""
    models = {}
    for regime in regime_labels.dropna().unique():
        regime_mask = regime_labels == regime
        regime_data = cache[regime_mask]
        if len(regime_data) < 30:  # minimum data requirement
            continue
        series = regime_data[var_name].values
        try:
            forecast, metrics = model_fn(series, n_forecast=1)
            if forecast is not None:
                models[regime] = {"forecast": forecast, "metrics": metrics}
        except Exception:
            pass
    return models
```

**Prediction blending:**

```python
def _predict_with_regime_models(
    regime_models: dict,
    current_regime_probs: dict,
    fallback_forecast: float,
) -> float:
    """Blend per-regime predictions using HMM probabilities."""
    if not regime_models:
        return fallback_forecast
    
    weighted_pred = 0.0
    total_weight = 0.0
    for regime, model_data in regime_models.items():
        prob = current_regime_probs.get(regime, 0.0)
        if prob > 0.01:
            weighted_pred += prob * model_data["forecast"][0]
            total_weight += prob
    
    if total_weight > 0:
        return weighted_pred / total_weight
    return fallback_forecast
```

**Wiring in the forecasting per-variable loop:**
For Tier 1-2 variables (Kalman), train per-regime Kalman models.
For Tier 3 (volatility), the GARCH is already single-variable -- train per-regime.
For Tier 4-5 and deep models (LSTM/Transformer), use the existing single model with regime-weighted burn-out (Item 3) instead of per-regime instances.

---

## Execution Order

```
Phase 1 (can be done in 1 session):
  [1] Item 6: Add 6 derived variables to derived_variables.py
  [2] Item 7: Add relative metrics to linked_aggregates.py
  [3] Item 4: Add per-tier accuracy to forward pass result
  [4] Item 8: Add confidence multipliers to prediction aggregator

Phase 2 (1-2 sessions):
  [5] Item 1: Weighted loss in forward pass + curriculum burn-out
  [6] Item 3: Regime-weighted sample weights in burn-out
  [7] Item 5: Exec comp + marketing excess in vanity module

Phase 3 (2-3 sessions):
  [8] Item 2: Per-regime Kalman/GARCH for Tier 1-3 variables
```

## Files Modified Summary

| File | Items | Lines Changed (est.) |
|------|-------|---------------------|
| `operator1/features/derived_variables.py` | 6 | +30 |
| `operator1/features/linked_aggregates.py` | 7 | +25 |
| `operator1/models/forecasting.py` | 1, 2, 3, 4 | +120 |
| `operator1/models/prediction_aggregator.py` | 8 | +15 |
| `operator1/analysis/vanity.py` | 5 | +30 |
| `main.py` | 4, 7 (wiring) | +10 |
| `operator1/report/profile_builder.py` | 4 (profile section) | +5 |
| **Total** | **8 items** | **~235 lines** |

## New Dependencies Required

None. All techniques use libraries already installed:
- `statsmodels` (MarkovAutoregression for Item 2)
- `sklearn` (KernelDensity for Item 3)
- `numpy` (exponential decay, kernel functions)
- `scipy` (norm for confidence multipliers)

## Validation

After implementation, re-run the AAPL backtest:
```bash
python backtest_runner.py --stage all --market us_sec_edgar --company AAPL --end-date 2024-12-31
python backtest_runner.py --validate --run-dir cache/backtest_AAPL_2024-12-31
```

Expected improvements:
- Per-tier accuracy numbers in the profile
- Wider Tier 4/5 prediction intervals during survival mode
- Better burn-out convergence from regime-weighted windows
- 6 new financial ratios visible in the profile
- Relative strength and valuation premium metrics

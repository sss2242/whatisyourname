# Round 2 Fix Plan: 8 Items

4 bugs from the debug scan + 4 remaining backtest fixes. Ordered by dependency and complexity.

---

## Group A: Quick Wiring Fixes (bugs with known locations)

### Fix A1: backtest_runner conformal bounds field names
- **Bug:** `extract_predictions()` reads `lower_bound`/`upper_bound` but `HorizonPrediction` uses `lower_ci`/`upper_ci`
- **File:** `backtest_runner.py`
- **Change:** Replace `lower_bound`/`upper_bound` with `lower_ci`/`upper_ci` in the aggregated predictions extraction loop

### Fix A2: Catalyst articles NaN filter
- **Bug:** When LLM sentiment returns NaN scores, `.articles` list contains NaN-scored entries. `detect_product_catalysts` treats them as 0 valid articles.
- **File:** `operator1/features/news_sentiment.py` line ~493
- **Change:** Filter out NaN-scored articles before storing in `result.articles`. Add: `articles = articles[articles["sentiment"].notna()]` before line 493.

### Fix A3: Wire adaptive_tier3 params to consuming models
- **Bug:** Adaptive Tier 3 params (windows, NN hyperparams, pattern thresholds) are computed but never passed to models
- **Files:** `main.py` (3 call sites)
- **Changes:**
  1. Pass `_adaptive_tier3.nn_params` to `train_transformer()` -- need to add `nn_params` parameter to `train_transformer()`
  2. Pass `_adaptive_tier3.pattern_body_threshold`/`pattern_doji_threshold` to `detect_patterns()` -- need to add threshold params
  3. Pass `_adaptive_tier3.windows` to `run_forecasting()` -- need to add `windows` parameter

### Fix A4: Wire retroactive_calibration output to profile
- **Bug:** `_retro_params` contains calibrated group weights but only `n_calibrated` is read
- **File:** `main.py` Step 7
- **Change:** Store retro calibration results in the profile JSON under `profile["retroactive_calibration"]` so the report can reference them. Also store the calibrated entity group weights for graph_risk edge weighting.

---

## Group B: Backtest Fix 3 -- Conformal bounds propagation

The conformal calibrator produces intervals. The aggregator reads them via `get_conformal_interval()` and applies them to `HorizonPrediction.lower_ci`/`upper_ci`. But the backtest showed all bounds as null.

**Root cause analysis needed:** The conformal result was built with 300 calibration scores and "method=adaptive_conformal". The aggregator should be reading them. The issue might be:
1. Variable name mismatch between conformal result's interval keys and aggregator's variable loop
2. The conformal result's `.intervals` dict is structured differently than `get_conformal_interval()` expects

**Fix approach:**
1. Add debug logging in `get_conformal_interval()` to trace why it returns NaN
2. Check if the conformal result's variable names match the aggregator's variable names
3. The conformal result was built from `forecast_result.forecasts` which has keys like `volatility_garch`, `cash_ratio` etc. but the aggregator loops over `available_vars` which are the raw cache column names. Mismatch possible.

**File:** `operator1/models/prediction_aggregator.py` and `operator1/models/conformal.py`

---

## Group C: Backtest Fix 5 -- Volatility blending

**Current state:** GARCH produces `volatility_garch` (single-regime, 0.0128). HAR-RV produces `volatility_har` (if fitted). VAR produces `volatility_21d` forecast. These are 3 separate, unblended estimates.

**Fix approach:** After all volatility models run, compute a blended estimate:
```
blended_vol = w_garch * garch_vol + w_har * har_vol + w_regime * regime_transition_vol
```
Where `w_garch = 0.4`, `w_har = 0.3`, `w_regime = 0.3` (default). Regime transition vol = `current_vol + P(switch) * (high_vol_regime - current_vol)` using HMM transition matrix.

**File:** `operator1/models/forecasting.py` -- add `_blend_volatility_forecasts()` after the GARCH and HAR-RV blocks (around line 1796). Store blended result as `result.forecasts["volatility_blended"]`.

---

## Group D: Backtest Fix 7 -- Whale competitors + single-call LLM

**Approach:** Create 2 new files + modify entity_discovery.py:

### D1: `config/whale_competitors.yml`
Static competitor registry for the top 5 sectors. Each entry has ticker, name, optional market_id. Covers:
- Technology (hardware, software, semiconductors)
- Energy (oil/gas, renewables)
- Finance (banks, insurance)
- Healthcare (pharma, biotech)
- Consumer (retail, luxury)

### D2: `operator1/steps/entity_discovery.py` modification
Add a fast-path for when:
1. The company's sector matches a `whale_competitors.yml` entry, OR
2. The company's market_cap exceeds $50B (from profile)

The fast-path:
- Loads competitors from YAML (0 LLM calls, 0 API calls)
- Makes 1 focused LLM call for top-3 suppliers + top-3 customers only
- Skips the expensive 25-exchange cross-region search
- Returns the merged entity dict

This does NOT require creating `whale_classifier.py` or `sec_13f.py` -- those are Phase 2. This is a minimal intervention that solves the "0 entities discovered" problem.

---

## Group E: Backtest Fix 8 -- FH debt serviceability + Merton DD

### E1: Debt serviceability override in `financial_health.py`
When both conditions are met:
- `interest_coverage > 10` (company easily services its debt)
- `cash_and_equivalents > total_debt` (more cash than debt)

Then cap the solvency tier penalty: `solvency_score = max(solvency_score, 40)`.

This prevents Apple-type companies (high debt for buybacks, but massive cash reserves and earnings) from being penalized as if they were financially stressed.

### E2: Merton Distance-to-Default in `financial_health.py`
Add `compute_merton_dd(cache)` function:
```python
V = market_cap + total_debt  # asset value proxy
D = total_debt               # default point
sigma_V = equity_vol * V / (V - D)  # asset volatility
DD = (ln(V/D) + (r - 0.5*sigma_V^2)*T) / (sigma_V * sqrt(T))
PD = N(-DD)  # probability of default
```
Store as `fh_merton_dd` and `fh_merton_pd` in cache. Use DD as a modifier on the composite score: if DD > 5 (extremely safe), add +10 to composite.

No new dependencies -- pure numpy math.

---

## Execution Order

```
1. Fix A1 (backtest_runner field names) -- trivial, 2 lines
2. Fix A2 (catalyst NaN filter) -- trivial, 1 line  
3. Fix A4 (retro calibration to profile) -- easy, ~10 lines in main.py
4. Fix B (conformal propagation debug) -- medium, needs investigation
5. Fix C (volatility blending) -- medium, ~40 lines in forecasting.py
6. Fix D (whale competitors) -- medium, new YAML + entity_discovery mod
7. Fix E1 (debt serviceability) -- easy, ~15 lines in financial_health.py
8. Fix E2 (Merton DD) -- medium, ~50 lines in financial_health.py
9. Fix A3 (tier3 param wiring) -- complex, touches 3 model files + main.py

Note: Fix A3 is last because it requires adding parameters to 3 model
function signatures, which could introduce regressions. All other fixes
are additive (new code paths, not changed interfaces).
```

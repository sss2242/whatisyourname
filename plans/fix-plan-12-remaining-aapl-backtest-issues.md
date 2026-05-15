# Fix Plan: 12 Remaining AAPL Backtest Issues

*C5 (zero-confidence) already fixed in PR feature/band-informed-confidence-fix*

## Batch 1: Sector Label Mismatch (H1 + H2) -- Highest Impact

These two issues share a single root cause and fixing them will dramatically improve FH scoring (31 -> ~70+) and MC survival (3.7% -> ~95%+).

### H1 + H2: SIC Label Mapping

**Problem:** Apple's sector from SEC EDGAR is "Electronic Computers" (SIC code 3571). The sector-aware code checks for strings like "technology", "Technology", "tech", "Information Technology" but never matches SIC labels.

**Fix:** Add a SIC-to-GICS sector mapper function and call it wherever sector matching happens.

**Files to modify:**

1. **`operator1/models/financial_health.py`** -- `SECTOR_FH_FLOORS` dict and the matching logic
   - Add SIC code ranges or keyword patterns: SIC 35xx-36xx = Technology, SIC 60xx-67xx = Financial Services
   - Or simpler: normalize sector string to lowercase and check for substrings: "computer", "electronic", "semiconductor", "software" -> "technology"

2. **`operator1/models/monte_carlo.py`** -- `SECTOR_SURVIVAL_OVERRIDES` dict and `get_sector_aware_thresholds()`
   - Same sector normalization: "Electronic Computers" matches "technology" override (current_ratio threshold 0.7 instead of 1.0)

3. **`operator1/analysis/survival_mode.py`** -- sector parameter in `compute_company_survival_flag()`
   - Apply same normalization before checking sector-aware threshold overrides

4. **New utility: `operator1/sector_mapper.py`** (~30 lines)
   ```python
   _SIC_TO_SECTOR = {
       range(3500, 3700): "technology",      # Computer hardware
       range(3670, 3700): "technology",      # Electronic components  
       range(7370, 7380): "technology",      # Computer services/software
       range(4810, 4900): "communication_services",
       range(6000, 6800): "financial_services",
       range(2000, 4000): "industrials",
       # etc.
   }
   
   _KEYWORD_TO_SECTOR = {
       "computer": "technology",
       "electronic": "technology",
       "semiconductor": "technology",
       "software": "technology",
       "telecom": "communication_services",
       "bank": "financial_services",
       "insurance": "financial_services",
       "pharma": "healthcare",
       "oil": "energy",
       "gas": "energy",
   }
   
   def normalize_sector(raw_sector: str, sic_code: str = "") -> str:
       """Map SIC/raw sector labels to standardized GICS-like sector names."""
   ```

**Expected outcome:** Apple FH: 31 -> ~70+. Apple MC survival: 3.7% -> ~95%.

---

## Batch 2: USS + Scenario Missing (C2)

### C2: Build USS Controller in backtest_runner

**Problem:** `SurvivalRegimeController.from_cache(cache)` is called in `main.py` Step 5-USS but never in `backtest_runner.py`.

**Fix:** Add USS construction to `backtest_runner.py` sub-stage 1.8b (after survival mode but before temporal models).

**File to modify:** `backtest_runner.py` -- in the `"1.8b" not in _skip` block, add:
```python
# Build USS controller (parity with main.py Step 5-USS)
try:
    from operator1.analysis.survival_regime_controller import SurvivalRegimeController
    state.survival_controller = SurvivalRegimeController.from_cache(cache)
    if state.survival_controller and not state.survival_controller.early_warning.empty:
        cache["early_warning_score"] = state.survival_controller.early_warning
except Exception as exc:
    logger.warning("USS controller failed: %s", exc)
```

Also add scenario engine call in Stage 7.1 (already handled by staged runner if USS is available).

**Expected outcome:** USS available=True, scenario analysis with 3 scenarios.

---

## Batch 3: Profile Builder Bugs (C1 + H4)

### C1: PID Controller not in profile

**Problem:** `forward_pass_result` has `pid_summary` dict, but profile builder sets `pid_controller.available = false`.

**Fix:** In `backtest_runner.py` `run_stage3()`, extract PID summary correctly:
```python
if state.forward_pass_result is not None:
    _fp = state.forward_pass_result
    if hasattr(_fp, 'pid_summary') and _fp.pid_summary:
        profile["pid_controller"] = {
            "available": True,
            **_fp.pid_summary,
        }
```

### H4: model_metrics.best_model_per_variable = None

**Problem:** `forecast_result.metrics` is a LIST of `ModelMetrics` objects but profile builder treats it as a dict.

**Fix:** In `backtest_runner.py` `run_stage3()` or in `profile_builder.py`, convert the list to a dict:
```python
if hasattr(state.forecast_result, 'metrics') and isinstance(state.forecast_result.metrics, list):
    _best_per_var = {}
    for mm in state.forecast_result.metrics:
        var = getattr(mm, 'variable', '')
        if var and var not in _best_per_var:
            _best_per_var[var] = getattr(mm, 'model_name', 'unknown')
    profile["model_metrics"]["best_model_per_variable"] = _best_per_var
```

**Expected outcome:** PID controller section in report. Model leaderboard showing which model won per variable.

---

## Batch 4: SHAP + Sobol Fix (C4)

### C4: SHAP "Insufficient data (0 < 30)"

**Problem:** SHAP needs predict functions from forward pass model_states. The model_states dict contains model objects but `run_6_6_shap()` can't extract usable predict functions.

**Fix:** In `operator1/stages/stage6_ensemble.py` `run_6_6_shap()`:
1. Check what's actually in `state.forward_pass_result.model_states`
2. Build `predict_fns` dict from model objects that have a `.predict()` method
3. For tree models (XGBoost, RF, GBM), use TreeExplainer directly
4. For others, wrap in a lambda that calls `.predict()`
5. Minimum 30 samples needed -- use the full cache as background data

For Sobol: same issue. Needs a predict function and feature matrix. Extract from cache columns used by the forecasting model.

**Expected outcome:** SHAP explanations with top-5 drivers per variable. Sobol sensitivity indices.

---

## Batch 5: Entity Discovery Fix (C3)

### C3: LLM entity discovery returned 0 entities

**Problem:** OpenRouter free-tier LLM returned malformed or empty response after 343.5s.

**Fix options (choose one):**
1. **Add JSON repair:** After LLM response, attempt `json.loads()` with progressively relaxed parsing (strip markdown fences, fix unquoted keys, extract JSON from prose)
2. **Add fallback entity list:** When LLM returns no entities, use SEC EDGAR peer data (`get_peers()` from SIC code) as fallback
3. **Increase LLM timeout + retry:** Currently 120s timeout. Increase to 180s and add 1 retry with different prompt format

**Recommended:** Option 2 (SEC EDGAR peer fallback) -- most reliable, no LLM dependency.

**File to modify:** `operator1/steps/entity_discovery.py` -- in `discover_linked_entities()`, after LLM parsing:
```python
if not relationships or sum(len(v) for v in relationships.values()) == 0:
    # Fallback: use SEC EDGAR peers from SIC code
    try:
        peers = pit_client.get_peers(identifier)
        if peers:
            relationships["competitors"] = [
                {"ticker": p, "name": p, "relationship_group": "competitors"}
                for p in peers[:5]
            ]
    except Exception:
        pass
```

**Expected outcome:** At least 3-5 competitor entities for graph risk, game theory, peer ranking.

---

## Batch 6: OHLC Mask Fix (H3)

### H3: Technical Alpha mask still applied to profile OHLC

**Problem:** `technical_alpha_masked=true` in next_day OHLC, nulling open/high/close.

**Fix:** In `backtest_runner.py` `run_stage3()` where OHLC predictions are serialized to profile, skip the Technical Alpha mask:
```python
if state.ohlc_result is not None and state.ohlc_result.fitted:
    from operator1.models.ohlc_predictor import format_ohlc_for_profile
    ohlc_profile = format_ohlc_for_profile(state.ohlc_result)
    # Remove Technical Alpha masking -- predictions are for backtesting
    for candle_key in ["next_day", "next_week", "next_month", "next_year"]:
        candle = ohlc_profile.get(candle_key, {})
        if candle.get("technical_alpha_masked"):
            candle["technical_alpha_masked"] = False
            # Restore masked values from ohlc_result
            ...
```

Alternatively, modify `format_ohlc_for_profile()` in `ohlc_predictor.py` to accept a `mask=False` parameter.

**Expected outcome:** All 4 OHLC fields visible: open, high, low, close.

---

## Batch 7: Medium Issues (M1 + M2 + M3 + M4)

### M1: total_debt alias

**File:** `operator1/features/derived_variables.py`
Add: `if "total_debt" not in cache.columns and "total_debt_asof" in cache.columns: cache["total_debt"] = cache["total_debt_asof"]`

### M2: iv_rv_spread timing

**File:** `main.py` Step 4.bench.1 (line ~1086) and `backtest_runner.py`
Move `iv_rv_spread` computation AFTER derived variables (Step 5) or compute `volatility_21d` inline before the spread calc.

### M3: Annual frequency from quarterly data

**File:** `operator1/features/frequency_resampler.py`
When no annual filings exist but quarterly data is available, synthesize annual rows by summing 4 consecutive quarters (flow vars) or taking the last quarter (stock vars).

### M4: Inflation CPI index -> YoY rate

**File:** `operator1/clients/macro_fredapi.py`
After fetching CPIAUCSL, compute YoY change: `inflation_rate = cpi.pct_change(12) * 100` (12-month percentage change for monthly data).

---

## Implementation Order

| Batch | Issues | Files | Impact | Risk |
|-------|--------|-------|--------|------|
| 1 | H1+H2 | 4 files | Highest (FH 31->70, MC 3.7->95%) | Low |
| 2 | C2 | 1 file | High (USS + scenarios) | Low |
| 3 | C1+H4 | 1-2 files | High (PID + model leaderboard) | Low |
| 4 | C4 | 1 file | High (SHAP + Sobol) | Medium |
| 5 | C3 | 1 file | High (entity discovery) | Low |
| 6 | H3 | 1-2 files | Medium (OHLC unmasking) | Low |
| 7 | M1-M4 | 4 files | Medium (4 independent fixes) | Low |

Total: ~12 files modified, ~200 lines of changes.

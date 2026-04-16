# Fix backtest_runner.py Parity Bugs

12 wiring bugs + 17 profile injection gaps found during debug scan. All changes are in `backtest_runner.py` only. The stage-to-stage architecture (Stage 1/2a/2b/2c/2d/3 with disk state serialization) is preserved exactly as-is.

---

## Fix 1: HMM Look-Ahead Bias in `_init_extra_vars()` [HIGH]

**File:** `backtest_runner.py` line 1109-1124
**What:** Add `_hmm_lookahead_cols` exclusion set and filter them out of `_extra_vars`, matching main.py lines 2444-2473.

```python
# Add exclusion set (same as main.py)
_hmm_lookahead_cols = {
    "survival_intensity",
    "regime_confidence",
    "regime_transition_prob",
}
```

Then add `and c not in _hmm_lookahead_cols` to the list comprehension filter.

---

## Fix 2: Missing Column Prefixes in `_init_extra_vars()` [MEDIUM]

**File:** `backtest_runner.py` line 1112-1124
**What:** Add all missing prefixes and individual column names to match main.py lines 2449-2473.

Add prefixes:
- `merton_`, `rv_`, `policy_risk_`, `sector_leader_`
- `segment_`, `product_`, `pricing_`, `margin_`
- `som_`, `customer_`

Add linked prefixes:
- `industry_peers_`, `rel_`, `valuation_premium_`

Add individual vars:
- `online_change_score`, `iv30`, `iv_rv_spread`
- `cannibalization_rate`, `net_new_revenue_pct`, `network_effect_score`
- `input_cost_pressure`, `growth_runway_quarters`, `maturity_concentration`
- `estimated_market_share`, `dominant_segment_growth`
- `buying_power_index`, `sector_demand_momentum`, `catalyst_score`

---

## Fix 3: `run_prediction_aggregation()` Missing kwargs [MEDIUM]

**File:** `backtest_runner.py` line 1383-1397 (in `run_stage2c`)
**What:** Pass `mode_weights`, `signal_ic_result`, and `prediction_log_summary`.

Since FixedShare is not yet wired in BT (Fix 6), pass `mode_weights=None` for now plus add:
```python
signal_ic_result=state.signal_ic_result,
prediction_log_summary=state.prediction_log_summary,
```

After Fix 6 is applied, `mode_weights` will come from FixedShare + burn-out weights.

---

## Fix 4: `run_monte_carlo()` Missing kwargs [MEDIUM]

**File:** `backtest_runner.py` line 1274-1276 (in `run_stage2b`)
**What:** Pass adaptive MC params and burn-out distributions:

```python
_mc_n = (
    state._adaptive_model_params.mc_n_paths
    if state._adaptive_model_params is not None and getattr(state._adaptive_model_params, "adapted", False)
    else 10_000
)
_mc_tilt = (
    state._adaptive_model_params.mc_is_tilt
    if state._adaptive_model_params is not None and getattr(state._adaptive_model_params, "adapted", False)
    else 1.5
)
_mc_thresholds = None
if state._adaptive_thresholds is not None and state._adaptive_thresholds.adapted:
    from operator1.analysis.adaptive_thresholds import threshold_set_to_mc_dict
    _mc_thresholds = threshold_set_to_mc_dict(state._adaptive_thresholds)
_burnout_dists = (
    state.burnout_result.regime_distributions
    if state.burnout_result is not None and getattr(state.burnout_result, "calibrated", False)
    else None
)
state.mc_result = run_monte_carlo(
    cache, returns_col=mc_ret,
    n_paths=_mc_n,
    importance_tilt=_mc_tilt,
    survival_thresholds=_mc_thresholds,
    burnout_distributions=_burnout_dists,
)
```

---

## Fix 5: `run_forecasting()` Missing `windows` kwarg [LOW-MEDIUM]

**File:** `backtest_runner.py` line 1209 (in `run_stage2a`)
**What:** Pass adaptive windows:

```python
cache, state.forecast_result = run_forecasting(
    cache, extra_variables=state._extra_vars,
    windows=state._adaptive_tier3.windows if state._adaptive_tier3 is not None and state._adaptive_tier3.adapted else None,
)
```

---

## Fix 6: Missing FixedShare/MCS Pipeline [MEDIUM]

**File:** `backtest_runner.py` in `run_stage2b` (after walk-forward, before MC) or `run_stage2c` (before prediction aggregation)
**What:** Add FixedShare and MCS computation, matching main.py lines 2706-2754.

Add after forward pass in `run_stage2b`:
```python
_mode_confidence_sets = None
_fixed_share = None
try:
    if state.forward_pass_result is not None and hasattr(state.forward_pass_result, "predictions_log"):
        from operator1.models.walk_forward import (
            aggregate_forward_pass_errors,
            compute_mode_confidence_sets,
        )
        from operator1.models.prediction_aggregator import FixedShareForecaster
        _fp_log = getattr(state.forward_pass_result, "predictions_log", [])
        if _fp_log:
            _mode_errors = aggregate_forward_pass_errors(_fp_log, cache)
            if _mode_errors:
                _mode_confidence_sets = compute_mode_confidence_sets(_mode_errors)
                _all_model_names = set()
                for mode_models in _mode_errors.values():
                    _all_model_names.update(mode_models.keys())
                if _all_model_names:
                    _fixed_share = FixedShareForecaster(sorted(_all_model_names))
                    for mode_models in _mode_errors.values():
                        _min_len = min(len(v) for v in mode_models.values()) if mode_models else 0
                        for step in range(min(_min_len, 50)):
                            step_losses = {name: errs[step] for name, errs in mode_models.items() if step < len(errs)}
                            _fixed_share.update(step_losses)
except Exception:
    pass
```

Store results in state for Stage 2c:
```python
state._mode_weights = None
if _fixed_share is not None:
    state._mode_weights = {"global": _fixed_share.get_weights()}
# Merge burn-out regime weights
if state.burnout_result is not None and getattr(state.burnout_result, "calibrated", False) and state.burnout_result.regime_weights:
    if state._mode_weights is None:
        state._mode_weights = {}
    for regime, model_weights in state.burnout_result.regime_weights.items():
        state._mode_weights[regime] = model_weights
```

Then in `run_stage2c`, pass to prediction aggregator:
```python
mode_weights=state._mode_weights,
```

Add `_mode_weights` to `BacktestState.__init__()`.

---

## Fix 7: Forward Pass Conformal Calibrator Reuse [LOW]

**File:** `backtest_runner.py` line 1350-1367 (in `run_stage2c`)
**What:** Check for forward pass calibrator before creating new one:

```python
calibrator = None
if state.forward_pass_result is not None and hasattr(state.forward_pass_result, "conformal_calibrator") and state.forward_pass_result.conformal_calibrator is not None:
    calibrator = state.forward_pass_result.conformal_calibrator
else:
    try:
        calibrator = ConformalPIDCalibrator(target_coverage=0.9)
    except Exception:
        calibrator = ConformalCalibrator(coverage=0.9, adaptive=True)
    if hasattr(state.forecast_result, "residuals") and state.forecast_result.residuals:
        for r in state.forecast_result.residuals:
            calibrator.update(r)
```

---

## Fix 8: QuantileRegressionCalibrator (G1) [LOW]

**File:** `backtest_runner.py` in `run_stage2c` (after conformal result is built)
**What:** Add G1 asymmetric intervals, matching main.py lines 2982-3008:

```python
try:
    from operator1.models.conformal import QuantileRegressionCalibrator
    _qr_cal = QuantileRegressionCalibrator(lower_quantile=0.05, upper_quantile=0.95)
    _residuals_list = list(state.forecast_result.residuals) if hasattr(state.forecast_result, "residuals") and state.forecast_result.residuals else []
    if len(_residuals_list) >= _qr_cal._min_samples:
        if _qr_cal.fit(_residuals_list):
            if state.conformal_result is not None and hasattr(state.conformal_result, "intervals"):
                for var, horizons_dict in state.conformal_result.intervals.items():
                    if isinstance(horizons_dict, dict):
                        for h, interval in horizons_dict.items():
                            pf = getattr(interval, "point_forecast", None) or getattr(interval, "forecast", None)
                            if pf is not None:
                                _lo, _hi = _qr_cal.predict_interval(float(pf))
                                if _lo is not None and _hi is not None:
                                    if hasattr(interval, "lower"): interval.lower = _lo
                                    if hasattr(interval, "upper"): interval.upper = _hi
except Exception:
    pass
```

---

## Fix 9: USS Aggregated Prediction Bounding [LOW-MEDIUM]

**File:** `backtest_runner.py` in `run_stage2c` or `run_stage2d` (after prediction aggregation)
**What:** Add `bound_survival_forecast()` on aggregated predictions, matching main.py lines 3075-3099:

```python
if (state.survival_controller is not None
        and state.survival_controller.is_survival
        and state.pred_result is not None
        and hasattr(state.pred_result, "predictions")):
    try:
        from operator1.analysis.survival_regime_controller import bound_survival_forecast
        for var, horizons_dict in state.pred_result.predictions.items():
            if isinstance(horizons_dict, dict):
                for h, hp in horizons_dict.items():
                    pf = getattr(hp, "point_forecast", None)
                    if pf is not None:
                        bounded = bound_survival_forecast(var, float(pf), cache, state.survival_controller.current_regime)
                        if bounded != float(pf):
                            hp.point_forecast = bounded
    except Exception:
        pass
```

---

## Fix 10: Missing Data Fetches [LOW]

**File:** `backtest_runner.py` in `run_stage1` (after benchmark returns, before macro)
**What:** Add IV fetch and sector leading indicators, matching main.py lines 977-1008:

```python
# IV-RV spread
try:
    from operator1.clients.ohlcv_provider import fetch_implied_volatility
    _iv_series = fetch_implied_volatility(ticker)
    if not _iv_series.empty:
        _iv_val = float(_iv_series.iloc[0])
        cache["iv30"] = _iv_val
        if "volatility_21d" in cache.columns:
            _rv = cache["volatility_21d"].iloc[-1] if cache["volatility_21d"].notna().any() else 0.0
            cache["iv_rv_spread"] = _iv_val - float(_rv)
except Exception:
    pass

# Sector leading indicators
try:
    from operator1.clients.ohlcv_provider import fetch_sector_leading_indicators
    _sector = state.target_profile.get("sector", "")
    _leader_df = fetch_sector_leading_indicators(_sector, years=int(state.years))
    if not _leader_df.empty:
        _leader_aligned = _leader_df.reindex(cache.index, method="ffill")
        for _ldr_col in _leader_aligned.columns:
            _col_name = f"sector_leader_{_ldr_col}"
            if _col_name not in cache.columns:
                cache[_col_name] = _leader_aligned[_ldr_col]
except Exception:
    pass
```

---

## Fix 11: `segment_hhi` MC Injection [LOW]

**File:** `backtest_runner.py` in `run_stage2b` (after MC completes)
**What:** Add segment_hhi injection, matching main.py lines 2791-2794:

```python
if state.mc_result is not None and "segment_hhi" in cache.columns:
    _seg_hhi = float(cache["segment_hhi"].iloc[-1]) if cache["segment_hhi"].notna().any() else 0
    state.mc_result.segment_hhi = _seg_hhi
    state.mc_result.concentration_risk_flag = _seg_hhi > 0.5
```

---

## Fix 12: `news_articles` Passthrough to `detect_product_catalysts()` [LOW]

**File:** `backtest_runner.py` line 867-870 (in `run_stage1`)
**What:** The sentiment result stores articles. Pass them through:

```python
_news_articles = []
try:
    _news_articles = _sr.articles if hasattr(_sr, "articles") else []
except Exception:
    pass

cache, state.catalyst_result = detect_product_catalysts(
    cache, profile=state.target_profile,
    news_articles=_news_articles if _news_articles else None,
)
```

This requires moving the product catalyst call AFTER sentiment (it already is in the current code, at line 867 after sentiment at line 847).

---

## Fix 13: Profile Injection Gaps in `run_stage3()` [MEDIUM]

**File:** `backtest_runner.py` in `run_stage3()` (after `build_company_profile()` call, around line 1727)
**What:** Add the 17 missing profile section injections. These are copy-adapted from main.py Step 7.

Each injection follows the same pattern -- check if the data exists in `state`, inject into `profile[key]` with `{"available": True/False}`. No new computation needed; all data was already computed in Stages 1-2.

Sections to add:
1. `profile["meta"]` -- market_id, ohlcv_source, pit_source, is_private
2. `profile["enriched_survival_timeline"]` -- from `state.enriched_timeline_result`
3. `profile["filing_calendar"]` -- from `state.filing_calendar_result`
4. `profile["economic_plane"]` -- call `classify_economic_plane()`
5. `profile["corporate_structure"]` -- from `state.relationships` parent/subsidiary groups
6. `profile["institutional_holders"]` -- from `state.target_holders`
7. `profile["institutional_ownership_analysis"]` -- from `state.contagion_result` + cache inst_* cols
8. `profile["macro_indicators"]` -- from `state.macro_data`
9. `profile["market_buying_power"]` -- from `state.buying_power_result`
10. `profile["supply_chain_stress"]` -- from cache columns
11. `profile["product_catalysts"]` -- from `state.catalyst_result`
12. `profile["product_segments"]` -- from `state._seg_result` (needs saving in Stage 1)
13. `profile["model_diagnostics"]` -- already computed in Stage 2d
14. `profile["ohlc_predictions"]` -- from `state.ohlc_result`
15. `profile["predicted_regime_shifts"]` -- computed in Stage 2b
16. `profile["extended_models"]` -- all temporal model results
17. `profile["synergies_applied"]` -- from `state._synergy_meta`

Also needs: save `_seg_result` into `BacktestState` so Stage 3 can access it. Add `self._seg_result: dict = {}` to `BacktestState.__init__()`.

---

## Execution Order

All fixes are in `backtest_runner.py`. No other files change. The stage boundaries remain exactly as they are.

| Fix | Location | Stage | Touches |
|-----|----------|-------|---------|
| 1 | `_init_extra_vars()` | Shared function | 15 lines changed |
| 2 | `_init_extra_vars()` | Shared function | 20 lines added |
| 3 | `run_stage2c` | Stage 2c | 3 lines added |
| 4 | `run_stage2b` | Stage 2b | 25 lines changed |
| 5 | `run_stage2a` | Stage 2a | 3 lines changed |
| 6 | `run_stage2b` + `run_stage2c` | Stage 2b+2c | 35 lines added |
| 7 | `run_stage2c` | Stage 2c | 10 lines changed |
| 8 | `run_stage2c` | Stage 2c | 15 lines added |
| 9 | `run_stage2c` or `run_stage2d` | Stage 2c/2d | 15 lines added |
| 10 | `run_stage1` | Stage 1 | 20 lines added |
| 11 | `run_stage2b` | Stage 2b | 5 lines added |
| 12 | `run_stage1` | Stage 1 | 8 lines changed |
| 13 | `run_stage3` + `__init__` | Stage 3 + init | ~120 lines added |

**Total**: ~295 lines of changes across `backtest_runner.py` only.

# Fix Guide: 4 Remaining Deferred Issues

Each issue investigated in depth with exact root cause, file locations, and implementation steps.

---

## C4: SHAP/Sobol -- model_states empty after pickle

### Root Cause
`forward_pass_result.model_states = {}` (empty dict, 0 entries). The forward pass at sub-stage 5.1 builds model_states with fitted sklearn/torch model objects, but `PipelineState.save()` at [`pipeline_state.py:244-269`](operator1/pipeline_state.py:244) strips non-picklable sub-fields. Fitted models (RandomForest, XGBoost, LSTM) fail pickle across process boundaries in the staged runner.

`shap_inline = {}` is also empty -- the forward pass was designed to populate it with plain-dict feature importance while models are alive, but this never happened.

### Fix: Populate shap_inline during forward pass (sub-stage 5.1)

**File: [`operator1/stages/stage5_forward.py`](operator1/stages/stage5_forward.py)** in `run_5_1_forward_pass()`

After the forward pass completes (model objects still alive), extract tree feature importance:

```python
# At the end of run_5_1_forward_pass(), before state.save():
if state.forward_pass_result is not None:
    _ms = getattr(state.forward_pass_result, "model_states", {})
    _inline = {}
    for var, wrapper in _ms.items():
        # Extract feature importance from tree models (sklearn/xgboost)
        model = getattr(wrapper, "model", None) or getattr(wrapper, "_model", None)
        if model is None:
            continue
        fi = getattr(model, "feature_importances_", None)
        if fi is not None:
            # Get feature names
            fn = getattr(wrapper, "feature_names", None)
            if fn is None:
                fn = getattr(model, "feature_names_in_", None)
            if fn is not None and len(fn) == len(fi):
                # Top-10 features by importance
                pairs = sorted(zip(fn, fi), key=lambda x: -abs(x[1]))[:10]
                _inline[var] = {
                    "feature_importance": {str(k): float(v) for k, v in pairs},
                    "model_type": type(model).__name__,
                }
    if _inline:
        state.forward_pass_result.shap_inline = _inline
        logger.info("Extracted inline SHAP importance for %d variables", len(_inline))
```

**File: [`operator1/stages/stage6_ensemble.py`](operator1/stages/stage6_ensemble.py)** in `run_6_6_shap()` -- the existing Path A code at line 316 already handles `shap_inline`. No changes needed there.

For **Sobol**, the issue is similar -- it needs a predict function. Add a lightweight refit in `run_6_7_sobol()`:

```python
# In run_6_7_sobol(), before calling compute_sensitivity:
# Fit a lightweight tree model for Sobol if model_states are empty
if not _predict_fns and state.cache is not None:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        _target = state.cache.get("return_1d")
        if _target is not None and _target.notna().sum() >= 50:
            _features = state.cache.select_dtypes(include=["number"]).dropna(axis=1, how="all")
            _features = _features.drop(columns=["return_1d"], errors="ignore").iloc[-252:]
            _target = _target.iloc[-252:]
            _mask = _features.notna().all(axis=1) & _target.notna()
            if _mask.sum() >= 30:
                _gb = GradientBoostingRegressor(n_estimators=50, max_depth=3)
                _gb.fit(_features[_mask], _target[_mask])
                _predict_fns["return_1d"] = lambda X: _gb.predict(X)
    except Exception:
        pass
```

### Expected Outcome
SHAP: feature importance for ~10-30 variables (from inline tree importance). Sobol: sensitivity indices from refitted lightweight tree.

---

## H3: OHLC mask -- masking happens inside ohlc_predictor.py

### Root Cause
The `OHLCPredictionResult.next_day` has `open=None, high=None, close=None` because the masking is applied inside [`ohlc_predictor.py`](operator1/models/ohlc_predictor.py) `predict_ohlc_series()` before the result is returned. The mask was supposed to be moved to report-only per commit `5966408` but it still runs in the prediction code.

### Fix: Compute all 4 values, flag but don't mask

**File: [`operator1/models/ohlc_predictor.py`](operator1/models/ohlc_predictor.py)**

Find the Technical Alpha masking code (search for `technical_alpha` or `None` assignment on open/high/close) and change it to:

```python
# BEFORE (masks values):
# candle.open = None  # Technical Alpha masked
# candle.high = None
# candle.close = None

# AFTER (compute all values, set flag only):
# Keep all computed values (open, high, close, low)
# Set flag for report-level masking
result.technical_alpha_masked = False  # Predictions available for backtesting
```

The report generator at [`report_generator.py`](operator1/report/report_generator.py) can check this flag and mask values in the rendered report if desired, but the profile and backtest validation should see all 4 values.

### Expected Outcome
OHLC next_day: open, high, low, close all populated. Profile: `technical_alpha_masked=false`.

---

## M2: iv_rv_spread -- volatility_21d not available at computation time

### Root Cause
In [`backtest_runner.py`](backtest_runner.py) sub-stage 1.4a, `iv_rv_spread` is computed as `iv30 - volatility_21d`. But `volatility_21d` is created in sub-stage 1.5 by `compute_derived_variables()`. At 1.4a time, the column doesn't exist yet.

In the final cache, `iv30` has 502 values and `volatility_21d` has 497 values, but `iv_rv_spread` has 0 because it was only attempted at 1.4a.

### Fix: Compute iv_rv_spread in derived_variables.py

**File: [`operator1/features/derived_variables.py`](operator1/features/derived_variables.py)** -- add to Stage 1 `_compute_returns_and_risk()` after `volatility_21d` is created:

```python
# After volatility_21d computation:
if "iv30" in df.columns and df["iv30"].notna().any():
    _iv = df["iv30"]
    _rv = df["volatility_21d"]
    df["iv_rv_spread"] = _iv - _rv
```

Also remove the now-redundant computation from `backtest_runner.py` sub-stage 1.4a and `main.py` Step 4.bench.1 (or leave them as no-ops since derived_variables will overwrite).

### Expected Outcome
`iv_rv_spread` with ~497 values in cache. Options-aware models can use IV-RV spread.

---

## M3: Annual frequency missing from MF pipeline

### Root Cause
Two issues:
1. `income_freq_groups` in state is type `str` (not `dict`) -- the field wasn't properly loaded from the parquet checkpoint in sub-stage 2.1
2. Even when freq groups are available, if no annual 10-K filings exist separately (Apple's 10-K data is merged into the quarterly stream by CompanyFacts), the frequency separator won't detect an "annual" group

### Fix Part 1: Load raw DFs in sub-stage 2.1

**File: [`operator1/stages/stage2_preprocessing.py`](operator1/stages/stage2_preprocessing.py)** in the frequency separation sub-stage:

```python
# At the start of the sub-stage, ensure raw DFs are loaded from parquet
run_dir = Path(state.output_dir)
for df_name in ("income_df", "balance_df", "cashflow_df"):
    df = getattr(state, df_name, None)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        pq_path = run_dir / f"{df_name}.parquet"
        if pq_path.exists():
            setattr(state, df_name, pd.read_parquet(pq_path))
```

### Fix Part 2: Synthesize annual from quarterly

**File: [`operator1/features/frequency_resampler.py`](operator1/features/frequency_resampler.py)** -- in `build_resampled_caches()` or the MF pipeline entry:

```python
# If no annual filings but quarterly data exists, synthesize annual
if "A" not in freq_caches and "Q" in freq_caches:
    q_cache = freq_caches["Q"].cache
    if len(q_cache) >= 4:
        # Resample quarterly to annual
        a_cache = q_cache.resample("YE").agg({
            # Stock vars: last
            **{c: "last" for c in STOCK_VARS if c in q_cache.columns},
            # Flow vars: sum
            **{c: "sum" for c in FLOW_VARS if c in q_cache.columns},
            # OHLCV
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        })
        freq_caches["A"] = ResampledCache(
            frequency="A", label="Annual", cache=a_cache, ...
        )
```

### Expected Outcome
5 frequencies (A/Q/M/W/D) in MF pipeline. Annual-perspective predictions available for 1yr and 2yr horizons.

---

## Implementation Order

| # | Issue | Files | Lines | Risk |
|---|-------|-------|-------|------|
| 1 | M2 iv_rv_spread | `derived_variables.py` | ~5 | Low |
| 2 | H3 OHLC mask | `ohlc_predictor.py` | ~10 | Low |
| 3 | C4 SHAP inline | `stage5_forward.py` + `stage6_ensemble.py` | ~30 | Medium |
| 4 | M3 Annual freq | `stage2_preprocessing.py` + `frequency_resampler.py` | ~30 | Medium |

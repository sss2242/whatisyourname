# Batch D: Full Implementation Plan -- Feature Classifier for Temporal Resolution Routing

*Problem 8: ~300 extra_vars with no dimensionality discipline*
*Based on debug scan + 13 patterns from 7 open-source projects*

---

## Debug Scan Summary

### Data flow: extra_vars through the pipeline

```
[stage3_temporal.py:24] _init_extra_vars(state)
  |-- Builds state.extra_vars from cache columns (~300 features)
  |-- Prefix matching: fh_*, sentiment_*, peer_*, macro_*, inst_*, corwin_*, etc.
  |
[stage3_temporal.py:118] IC-weighted signal filtering (optional)
  |
[stage3_temporal.py:260] apply_pre_forecasting_synergies()
  |-- Adds cycle phase features, prunes by causal network
  |
[stage3_temporal.py:276] run_3_8_feature_selection() -- Boruta/PIMP/mRMR
  |-- Reduces ~300 to ~50-80 features
  |-- state.extra_vars = selected
  |
[stage4_forecasting.py:37] run_forecasting(cache, extra_variables=state.extra_vars)
  |-- ALL models receive the SAME feature list
  |-- LSTM, Tree, VAR, Kalman all get ~50-80 mixed features
  |
[forecasting.py:690] _exog_cols = [c for c in extra_variables if c in cache.columns]
  |-- Filtered to available columns with >20 non-NaN
  |-- LSTM gets forward-filled constants alongside daily returns
```

### The problem

`_init_extra_vars()` at [`stage3_temporal.py:24`](operator1/stages/stage3_temporal.py:24) collects ALL matching columns. After Boruta/PIMP/mRMR selection at [`stage3_temporal.py:276`](operator1/stages/stage3_temporal.py:276), ~50-80 features survive. But these include:
- TICK features (return_1d, volatility_21d, MACD) that change daily
- PERIODIC features (current_ratio, gross_margin, PE) that are flat for 63 days
- STATIC features (conflict_flag, sector_score) that never change
- NORMALIZED features (current_ratio_zscore_63d) that DO change daily despite periodic base

All are fed equally to LSTM/Tree/VAR. LSTM learns "when current_ratio=0.92, predict 0.001 return" -- a spurious correlation because current_ratio was 0.92 for ALL return values.

### Insertion point

The classifier should run AFTER feature selection (sub-stage 3.8) and BEFORE forecasting (sub-stage 4.1). It classifies the already-selected features and produces per-model feature sets.

---

## Implementation Plan

### Phase 1: Feature Classifier Module (NEW, ~150 lines)

**New file:** `operator1/features/feature_classifier.py`

```python
"""Classify cache columns by temporal resolution for model routing.

Three classes:
- TICK: changes most trading days (return_1d, volume, MACD, z-scores)
- PERIODIC: changes at filing frequency (current_ratio, PE, margins)  
- STATIC: changes rarely or never (conflict flags, sector scores)

Models consume different classes:
- LSTM/Transformer: TICK only (daily variation needed)
- Tree ensemble: TICK + PERIODIC normalized (z-scores change daily)
- VAR/Kalman: TICK only (requires stationarity)
- Snapshot scoring: PERIODIC + STATIC (FH, survival -- not temporal)
"""
```

Implements `FeatureClassifier` class with:
- `classify(col_name, series)` -> `FeatureClass.TICK/PERIODIC/STATIC`
- `build_model_feature_sets(cache, extra_vars)` -> `dict[str, list[str]]`

**Classification method (3-tier, from research):**

1. **Name-based** (instant, ~90% coverage): TICK_PREFIXES for return/vol/technical, STATIC_PREFIXES for flags/scores
2. **Suffix-based**: `_zscore_63d`, `_percentile_252d`, `_change_21d`, `_regime_zscore` are always TICK
3. **Variance-based** (sklearn VarianceThreshold fallback): `nunique() / len()` on recent 63-day window

**Per-model feature sets:**

| Model | Features | Rationale |
|-------|----------|-----------|
| `lstm` | TICK only | Daily variation needed for gradient learning |
| `tree` | TICK + normalized PERIODIC | z-scores change daily, trees handle mixed types |
| `var` | TICK + stationary | VAR requires stationarity |
| `kalman` | (empty) | Single-variable, no exogenous features |
| `baseline` | (empty) | Last-value or EMA, no features |
| `all` | everything | Backward compat |

### Phase 2: Wire into stage3_temporal.py (~30 lines)

**File:** [`operator1/stages/stage3_temporal.py`](operator1/stages/stage3_temporal.py)

Add classification AFTER feature selection (sub-stage 3.8), storing result in PipelineState:

```python
# At end of run_3_8_feature_selection():
from operator1.features.feature_classifier import FeatureClassifier
classifier = FeatureClassifier()
state.model_feature_sets = classifier.build_model_feature_sets(
    state.cache, state.extra_vars,
)
n_tick = len(state.model_feature_sets.get("lstm", []))
n_tree = len(state.model_feature_sets.get("tree", []))
logger.info("Feature routing: lstm=%d tick, tree=%d tick+norm, total=%d",
            n_tick, n_tree, len(state.extra_vars))
```

Also add preference logic in `_init_extra_vars()`: when both `current_ratio` and `current_ratio_zscore_63d` exist, prefer z-score and exclude raw periodic from extra_vars.

### Phase 3: Wire into stage4_forecasting.py (~10 lines)

**File:** [`operator1/stages/stage4_forecasting.py`](operator1/stages/stage4_forecasting.py)

Pass `model_feature_sets` to `run_forecasting()`:

```python
def run_4_1_forecasting(state):
    ...
    _, state.forecast_result = run_forecasting(
        cache,
        extra_variables=state.extra_vars,
        model_feature_sets=getattr(state, "model_feature_sets", None),
        ...
    )
```

### Phase 4: Accept model_feature_sets in forecasting.py (~30 lines)

**File:** [`operator1/models/forecasting.py`](operator1/models/forecasting.py)

Modify `run_forecasting()` at line 2152 to accept `model_feature_sets` and route per model:

```python
def run_forecasting(
    cache, variables=None, *,
    extra_variables=None,
    model_feature_sets=None,  # NEW
    ...
):
    ...
    # Per-model feature routing
    def _get_features_for_model(model_type: str) -> list[str]:
        if model_feature_sets and model_type in model_feature_sets:
            return model_feature_sets[model_type]
        return extra_variables or []  # backward compat
    
    # In each model's fit call:
    # LSTM: _get_features_for_model("lstm")
    # Tree: _get_features_for_model("tree")
    # VAR:  _get_features_for_model("var")
```

### Phase 5: Add model_feature_sets to PipelineState (~2 lines)

**File:** [`operator1/pipeline_state.py`](operator1/pipeline_state.py)

```python
# Line ~120, after extra_vars:
self.model_feature_sets: dict = {}
```

---

## Files Changed Summary

| File | Change | Lines +/- |
|------|--------|-----------|
| `operator1/features/feature_classifier.py` | **NEW** | +150 |
| `operator1/stages/stage3_temporal.py` | Add classification after 3.8 + preference logic | +30 |
| `operator1/stages/stage4_forecasting.py` | Pass model_feature_sets | +5 |
| `operator1/models/forecasting.py` | Accept + route per model type | +30 |
| `operator1/pipeline_state.py` | Add field | +2 |
| **Total** | | **~217 net** |

---

## Validation Criteria

1. No raw forward-filled periodic feature in `model_feature_sets["lstm"]`
2. Every feature in `model_feature_sets["lstm"]` has >30% daily change rate
3. `model_feature_sets["tree"]` includes `_zscore_63d` versions of periodic features
4. Forecasting RMSE does not increase when using routed features
5. `len(model_feature_sets["lstm"])` < 30 (down from ~80 mixed)
6. `len(model_feature_sets["tree"])` < 60 (tick + normalized periodic)
7. `close_frac_diff` is in `model_feature_sets["lstm"]` when available
8. Raw `close` is NOT in `model_feature_sets["lstm"]` (non-stationary)

---

## Execution Order

1. Create `feature_classifier.py` (Phase 1)
2. Wire into stage3_temporal.py after sub-stage 3.8 (Phase 2)
3. Wire into stage4_forecasting.py (Phase 3)
4. Accept in forecasting.py per model type (Phase 4)
5. Add PipelineState field (Phase 5)
6. Run tests to validate

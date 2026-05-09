# Batch D: Extra Vars Dimensionality Fix

*Problem 8: ~300 extra_vars are selected for forecasting with no dimensionality discipline*

---

## Current state

`_init_extra_vars()` in `stage3_temporal.py` is a 60+ line prefix-matching function that collects every column matching patterns like `fh_*`, `sentiment_*`, `peer_*`, `macro_*`, `inst_*`, `corwin_*`, `anchoring_*`, etc. This produces ~300 features.

Boruta/PIMP/mRMR (sub-stage 3.8) reduces to ~50-80, but the fundamental problem remains: **most of these features are forward-filled constants that don't change daily.**

A quarterly `current_ratio` is the same value for 63 consecutive trading days, then jumps. A tree model trained on this learns "predict tomorrow's return = f(current_ratio)" but current_ratio didn't change from yesterday to today. The model effectively learns "predict same as yesterday" because the input signal is flat.

### The three types of features by temporal resolution

| Type | Examples | Daily change rate | Useful for |
|------|----------|-------------------|------------|
| **Tick-level** (changes every day) | return_1d, volatility_21d, volume, MACD, RSI, OBV | 100% | Temporal models (LSTM, VAR, tree) |
| **Periodic** (changes every 63-252 days) | current_ratio, PE, gross_margin, fcf_yield, Altman Z | 0.4-1.6% | Snapshot scoring (FH, survival), level anchoring |
| **Static** (changes once or never) | conflict_flag, sector scores, macro quadrant, geo_hhi | 0% | Conditioning/regime classification |

Currently all three types are dumped into the same `extra_vars` list and fed to every temporal model equally.

---

## Expert methods

**From signal processing (Nyquist-Shannon Sampling Theorem):**
- A signal can only be reconstructed if sampled at 2x its frequency. A quarterly ratio sampled daily is 63x oversampled -- the daily "observations" contain zero new information between filings. A model trained on this oversampled signal is fitting noise in the forward-fill, not the actual ratio dynamics.
- Applied here: Features should be matched to models by their update frequency. Daily-changing features go to daily temporal models. Quarterly-changing features should only inform models at the quarterly frequency (which the MF pipeline already handles).

**From feature engineering for time series (Hyndman & Athanasopoulos, "Forecasting: Principles and Practice"):**
- **Lagged features**: Instead of feeding the raw forward-filled `current_ratio` (flat for 63 days), use `current_ratio_change` (0 for 62 days, non-zero on 1 day) or `days_since_current_ratio_change` (1,2,...,63,1,2,...). These have daily variation and carry the actual information content.
- **Rolling features of periodic data**: Instead of raw `gross_margin` (flat 63 days), use `gross_margin_zscore_252d` (percentile rank changes daily as the window slides) or `gross_margin_surprise` (deviation from expected value).
- Applied here: The feature_normalization module already computes `{var}_zscore_63d`, `{var}_percentile_252d`, `{var}_change_21d`. These normalized versions DO change daily and should be preferred over raw periodic features.

**From ML feature selection (Christoph Molnar, "Interpretable Machine Learning"):**
- **Variance threshold**: Features with near-zero variance over the prediction window carry no information. A column that's 0.92 for 63 days has zero variance -- it should be excluded from temporal models even if Boruta says it's "important" (Boruta tests importance against shadow features, which are also constant, making the comparison meaningless).
- Applied here: Add a variance threshold filter that excludes features with < 1% of rows changing within the prediction lookback window.

**From quantitative finance (Lopez de Prado, "Advances in Financial Machine Learning"):**
- **Triple-barrier labeling + meta-labeling**: Instead of predicting the raw next-day return from 300 features, predict a binary outcome (price crosses upper/lower barrier within T days) from a curated feature set. This eliminates the "predict same as yesterday" failure mode because the label itself has meaningful variation.
- **Fractional differentiation**: `close_frac_diff` (already computed in derived_variables) is the stationary version of close that preserves long-range memory. This should be the primary price feature for temporal models, not raw close.
- Applied here: Use `close_frac_diff` and normalized features as primary inputs, with raw periodic features only as conditioning variables.

**From deep learning (Bengio, "Representation Learning"):**
- **Feature hierarchy**: In CNNs, low-level features (edges) and high-level features (objects) serve different purposes. Mixing them in the same layer hurts performance. Similarly, tick-level features (return, volume) and periodic features (ratios) serve different purposes in financial prediction.
- Applied here: Split features into tiers and route each tier to the appropriate model type.

---

## Implementation Plan

### Phase 1: Feature Classification System

**File:** `operator1/features/feature_classifier.py` (new, ~150 lines)

```python
"""Classify cache columns by temporal resolution for model routing.

Three classes:
- TICK: changes most trading days (return_1d, volume, MACD, etc.)
- PERIODIC: changes at filing frequency (current_ratio, PE, margins, etc.)
- STATIC: changes rarely or never (conflict flags, sector scores, geo_hhi)

Models consume different classes:
- LSTM/Transformer: TICK features only (daily variation needed)
- Tree ensemble: TICK + PERIODIC normalized (z-scores change daily)
- VAR/Kalman: TICK features only (requires stationarity)
- Regime detector: TICK features (return_1d, volatility_21d)
- Snapshot scoring: PERIODIC + STATIC (FH, survival -- not temporal models)
"""

from enum import Enum

class FeatureClass(Enum):
    TICK = "tick"          # changes daily
    PERIODIC = "periodic"  # changes at filing frequency
    STATIC = "static"      # changes rarely

def classify_feature(col_name: str, series: pd.Series, 
                     filing_frequency: str = "quarterly") -> FeatureClass:
    """Classify a feature by its temporal resolution.
    
    Uses two signals:
    1. Name-based heuristics (fast, covers known features)
    2. Variance-based detection (handles unknown features)
    """
    # Name-based classification (covers ~90% of known features)
    TICK_PREFIXES = (
        "return_", "log_return", "volatility_", "volume", "close", "open",
        "high", "low", "macd", "rsi", "adx", "obv", "bb_", "sma_",
        "benchmark_return", "online_change_score", "sentiment_score",
    )
    STATIC_PREFIXES = (
        "country_conflict", "company_conflict", "sanctions_flag",
        "fragile_state", "conflict_type", "geo_hhi", "china_revenue",
        "macro_quadrant", "fuzzy_protection", "fuzzy_sector",
    )
    
    col_lower = col_name.lower()
    if any(col_lower.startswith(p) for p in TICK_PREFIXES):
        return FeatureClass.TICK
    if any(col_lower.startswith(p) for p in STATIC_PREFIXES):
        return FeatureClass.STATIC
    # Normalized features (_zscore, _percentile, _change) are TICK
    # even if the underlying variable is PERIODIC
    if col_lower.endswith("_zscore_63d") or col_lower.endswith("_percentile_252d") or col_lower.endswith("_change_21d"):
        return FeatureClass.TICK
    
    # Variance-based fallback: count distinct values in recent window
    recent = series.tail(63).dropna()
    if len(recent) < 10:
        return FeatureClass.STATIC
    n_distinct = recent.nunique()
    change_rate = n_distinct / len(recent)
    if change_rate > 0.3:
        return FeatureClass.TICK
    elif change_rate > 0.02:
        return FeatureClass.PERIODIC
    else:
        return FeatureClass.STATIC


def build_model_feature_sets(
    cache: pd.DataFrame,
    extra_vars: list[str],
    filing_frequency: str = "quarterly",
) -> dict[str, list[str]]:
    """Build per-model-type feature sets from classified features.
    
    Returns:
        {
            "lstm": [...tick features...],
            "tree": [...tick + periodic_normalized...],
            "var": [...tick features...],
            "kalman": [...tick features...],
            "all": [...everything for backward compat...],
        }
    """
```

### Phase 2: Route features to models

**File:** `operator1/models/forecasting.py`

Modify `run_forecasting()` to accept and use model-specific feature sets:

```python
def run_forecasting(cache, extra_variables=None, 
                    model_feature_sets=None, ...):
    """
    When model_feature_sets is provided, each model type uses
    only its appropriate features:
    - LSTM: model_feature_sets["lstm"] (tick-level only)
    - Tree: model_feature_sets["tree"] (tick + normalized periodic)
    - VAR: model_feature_sets["var"] (tick-level, stationary)
    - Kalman: single variable (no extra features)
    - Baseline: no features (last-value or EMA)
    
    When model_feature_sets is None (backward compat), uses
    extra_variables for all models (current behavior).
    """
```

### Phase 3: Wire into stage3_temporal.py

**File:** `operator1/stages/stage3_temporal.py`

After `_init_extra_vars()` and feature selection (sub-stage 3.8), classify and route:

```python
def run_3_8_feature_selection(state):
    # ... existing Boruta/PIMP/mRMR ...
    
    # NEW: Classify selected features by temporal resolution
    from operator1.features.feature_classifier import build_model_feature_sets
    state.model_feature_sets = build_model_feature_sets(
        state.cache, state.extra_vars,
        filing_frequency=state.filing_calendar_result.detected_frequency 
            if state.filing_calendar_result else "quarterly",
    )
    logger.info(
        "Feature routing: tick=%d, periodic_norm=%d, static=%d",
        len(state.model_feature_sets.get("lstm", [])),
        len(state.model_feature_sets.get("tree", [])) - len(state.model_feature_sets.get("lstm", [])),
        len([v for v in state.extra_vars if v not in state.model_feature_sets.get("tree", [])]),
    )
```

**File:** `operator1/stages/stage4_forecasting.py`

Pass model_feature_sets to forecasting:

```python
def run_4_1_forecasting(state):
    cache, state.forecast_result = run_forecasting(
        cache,
        extra_variables=state.extra_vars,
        model_feature_sets=getattr(state, "model_feature_sets", None),
        ...
    )
```

### Phase 4: Prefer normalized features over raw periodic

**File:** `operator1/stages/stage3_temporal.py` -- `_init_extra_vars()`

Add preference logic: when both `current_ratio` and `current_ratio_zscore_63d` exist, prefer the z-score for temporal models and exclude the raw periodic feature:

```python
# After building extra_vars, replace raw periodic with normalized versions
_normalized_suffixes = ("_zscore_63d", "_percentile_252d", "_change_21d")
_raw_periodic = set()
_normalized_available = set()
for v in state.extra_vars:
    for suffix in _normalized_suffixes:
        if v.endswith(suffix):
            base = v[:len(v)-len(suffix)]
            _normalized_available.add(base)

# Remove raw periodic features that have normalized versions
state.extra_vars = [
    v for v in state.extra_vars
    if v not in _normalized_available  # keep if no normalized version
    or any(v.endswith(s) for s in _normalized_suffixes)  # keep normalized
]
```

---

## Expected impact

| Metric | Before | After |
|--------|--------|-------|
| Features to LSTM | ~80 (mixed) | ~20 (tick only) |
| Features to Tree | ~80 (mixed) | ~45 (tick + normalized periodic) |
| Features to VAR | ~80 (mixed) | ~15 (tick, stationary) |
| Forward-filled constants in temporal models | ~60% | ~5% |
| Tree model "same as yesterday" predictions | High (flat inputs) | Low (varying inputs) |
| LSTM training convergence | Slow (noisy gradients from flat features) | Faster (meaningful gradients) |

---

## Validation criteria

1. No raw forward-filled periodic feature (current_ratio, gross_margin, PE, etc.) appears in `model_feature_sets["lstm"]`
2. Every feature in `model_feature_sets["lstm"]` has >30% daily change rate over 63-day window
3. Forecasting RMSE does not increase (or improves) when using routed features vs all features
4. `model_feature_sets["tree"]` includes `{var}_zscore_63d` versions of periodic features

---

## Files changed

| File | Change | Lines |
|------|--------|-------|
| `operator1/features/feature_classifier.py` | NEW | ~150 |
| `operator1/stages/stage3_temporal.py` | Add classification + preference logic | ~40 |
| `operator1/stages/stage4_forecasting.py` | Pass model_feature_sets | ~5 |
| `operator1/models/forecasting.py` | Accept model_feature_sets, route per model type | ~50 |
| `operator1/pipeline_state.py` | Add model_feature_sets field | ~2 |
| **Total** | | ~250 net |

---

## Why this works

The core insight is that **temporal frequency mismatch is a form of data leakage**. When you train a tree model on `current_ratio = 0.92` (constant for 63 days) to predict `return_1d = 0.001` (changes daily), the model learns a spurious association: "when current_ratio is 0.92, returns are about 0.001." But current_ratio was 0.92 on days with +3% returns AND -2% returns. The model is fitting noise.

By routing only tick-level features to temporal models and using normalized versions of periodic features (which DO change daily as the rolling window slides), every input to the model carries actual daily information content. The model can then learn "when the z-score of current_ratio is falling AND volatility is rising, returns tend to be negative" -- a meaningful pattern, not noise.

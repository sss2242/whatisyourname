# GitHub Research: Feature Classifier Patterns from Open-Source Projects

*Research for Batch D: Extra Vars Dimensionality Fix (Problem 8)*
*Date: 2026-05-10*

Research across 7 open-source projects to identify implementation patterns for feature temporal resolution classification, per-model feature routing, variance-based filtering, and time-series-aware feature engineering.

---

## Batch D has 3 components:

1. **Feature classification** -- separate tick (daily-changing) vs periodic (quarterly-changing) vs static features
2. **Per-model routing** -- LSTM gets tick-only, Tree gets tick+normalized, VAR gets stationary
3. **Preference for normalized over raw** -- when both `current_ratio` and `current_ratio_zscore_63d` exist, temporal models use the z-score

---

## Project 1: tsfresh (9,205 stars)

**Repo:** `blue-yonder/tsfresh`
**Relevance:** The standard for time series feature extraction and selection. Their relevance testing framework directly addresses "which features carry information for this prediction task."

### Key Patterns Found

**Pattern 1: Benjamini-Hochberg Feature Relevance Testing**

tsfresh's `calculate_relevance_table()` computes a p-value for each feature's association with the target using univariate statistical tests. Then applies Benjamini-Hochberg multiple testing correction to control the false discovery rate. Features below the FDR threshold are marked "relevant."

```python
# tsfresh approach: univariate significance test per feature
p_values = [test_feature_relevance(X[:, i], y) for i in range(n_features)]
relevant = benjamini_hochberg(p_values, fdr_level=0.05)
```

**What we can adopt:** After classifying features by temporal resolution, run a quick p-value relevance test on the TICK features to confirm they actually predict the target. A tick feature with p > 0.1 is noise even though it changes daily. This is a lightweight validation layer on top of the classification.

**Pattern 2: Feature Type Inference from Distribution**

tsfresh infers whether a feature is binary, categorical, or continuous from its value distribution. This determines which statistical test to use for relevance.

**What we can adopt:** Our classifier should infer feature type from the value distribution within a recent window:
- **Change rate > 30%** of rows have distinct values -> TICK
- **Change rate 2-30%** -> PERIODIC 
- **Change rate < 2%** -> STATIC
- **Exactly 2 unique values** -> BINARY (flag/indicator)

---

## Project 2: scikit-learn VarianceThreshold (66,018 stars)

**Repo:** `scikit-learn/scikit-learn`
**Relevance:** The simplest feature selection method -- remove features with variance below a threshold.

### Key Patterns Found

**Pattern 3: VarianceThreshold as First-Pass Filter**

sklearn's `VarianceThreshold` removes features where `variance < threshold`. With `threshold=0`, it removes constant features. This is O(n) and can be applied before any expensive selection method.

```python
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
X_reduced = sel.fit_transform(X)
```

**What we can adopt:** Before any classification, apply a variance threshold to the recent window (63 days) of each feature. A `current_ratio` that's been 0.92 for 63 days has variance ~0. It should be excluded from LSTM/VAR regardless of its name-based classification.

Key insight from sklearn: **variance threshold is faster than any ML-based selection and catches the most obvious waste** (forward-filled constants). Use it as a pre-filter, then classify what remains.

**Pattern 4: NaN-Aware Variance Computation**

sklearn's VarianceThreshold handles NaN by computing variance on non-NaN values only. Features that are mostly NaN with a few non-NaN values can still pass the threshold.

**What we can adopt:** When computing change rate for classification, use `s.dropna().nunique()` not `s.nunique()` -- NaN should not count as a "unique value" that inflates the change rate.

---

## Project 3: mlfinlab / AFML (4,708 stars)

**Repo:** `hudson-and-thames/mlfinlab`
**Relevance:** Lopez de Prado's "Advances in Financial Machine Learning" implementation. Directly addresses the problem of feature importance in financial time series.

### Key Patterns Found

**Pattern 5: Mean Decrease Impurity with Anti-Masking**

mlfinlab's `mean_decrease_impurity()` implements MDI from AFML Chapter 8 with a critical fix: `max_features=1` to prevent masking effects where correlated features cancel each other out.

**What we can adopt:** When routing features to Tree models, the MDI importance from the tree itself can validate the classification. If a TICK feature has 0 MDI importance, it's not helping the tree despite changing daily. Conversely, if a PERIODIC feature (via its z-score) has high MDI, it's providing useful context.

**Pattern 6: Clustered Feature Importance (CFI)**

AFML's CFI groups correlated features into clusters, then measures importance per cluster instead of per feature. This prevents the "two identical features both get 50% importance" dilution problem.

**What we can adopt:** Our `feature_normalization.py` already creates `{var}_zscore_63d`, `{var}_percentile_252d`, and `{var}_change_21d` for each periodic feature. These 3 are highly correlated (all derived from the same raw series). The classifier should pick ONE representative per group:
- For temporal models: prefer `_zscore_63d` (changes daily, centered)
- For snapshot models: prefer raw value (actual level matters)

**Pattern 7: Fractional Differentiation (fracdiff)**

mlfinlab implements AFML's fractional differentiation (`fracdiff.py`) which makes price series stationary while preserving memory. `close_frac_diff` (already in our `derived_variables.py`) is the stationary version of close.

**What we can adopt:** `close_frac_diff` should be the PRIMARY price feature for LSTM/VAR (stationary), not raw `close` (non-stationary). Raw `close` should be EXCLUDED from temporal models. This is already suggested in the Batch D plan but mlfinlab confirms the mathematical basis.

---

## Project 4: featuretools (7,640 stars)

**Repo:** `alteryx/featuretools`
**Relevance:** Automated feature engineering. Their "entity set" architecture shows how to model features with different temporal resolutions.

### Key Patterns Found

**Pattern 8: Entity Sets with Temporal Relationships**

featuretools models data as "entity sets" where entities can have different update frequencies. A customer entity updates rarely; a transaction entity updates per-event. Features are generated at each entity's natural frequency and joined via temporal relationships.

**What we can adopt:** Our cache mixes entities with different frequencies: OHLCV data (daily entity), financial statements (quarterly entity), macro data (annual entity), conflict flags (static entity). The feature classifier should respect these entity boundaries:

```python
ENTITY_MAP = {
    "daily": ["return_1d", "close", "volume", "macd", ...],       # changes every day
    "quarterly": ["current_ratio", "gross_margin", "PE", ...],     # changes at filing freq
    "annual": ["gdp_growth", "inflation_rate", ...],               # changes yearly
    "static": ["conflict_flag", "sector_score", "geo_hhi", ...],   # rarely changes
}
```

This is more principled than the variance-based classification because it uses domain knowledge about WHERE the data comes from.

**Pattern 9: Primitive-Based Feature Generation**

featuretools generates features using "primitives" (sum, mean, trend, etc.) applied to time-windowed data. Each primitive produces a feature at the granularity of its input entity.

**What we can adopt:** Our `_zscore_63d` and `_percentile_252d` suffixes are essentially featuretools-style primitives applied to periodic features. The classifier should recognize these suffixes as "primitives that convert periodic to tick-frequency" and always prefer them for temporal models.

---

## Project 5: sktime (9,757 stars)

**Repo:** `sktime/sktime`
**Relevance:** Unified time series ML framework. Their transformer architecture shows how to compose feature extraction with model-specific routing.

### Key Patterns Found

**Pattern 10: Transformer Pipeline with Type Dispatch**

sktime uses a type system where transformers declare what types of data they accept and produce. A transformer that accepts `pd.Series` is a univariate transformer; one that accepts `pd.DataFrame` is multivariate. Models declare their input requirements and the framework routes data accordingly.

**What we can adopt:** Our forecasting models should declare their input requirements:

```python
MODEL_INPUT_REQUIREMENTS = {
    "lstm": {"type": "tick", "min_change_rate": 0.3, "stationary_preferred": True},
    "tree": {"type": "tick+periodic_normalized", "min_change_rate": 0.02},
    "var": {"type": "tick", "min_change_rate": 0.3, "stationary_required": True},
    "kalman": {"type": "tick", "single_variable": True},
    "baseline": {"type": "any"},
}
```

The classifier checks each feature against the model's requirements and builds a per-model feature set.

---

## Project 6: feature_engine (2,236 stars)

**Repo:** `feature-engine/feature_engine`
**Relevance:** sklearn-compatible feature engineering. Their `LagFeatures` and `WindowFeatures` show how to create temporal features from periodic data.

### Key Patterns Found

**Pattern 11: LagFeatures as Periodic-to-Tick Converter**

feature_engine's `LagFeatures` creates `t-1`, `t-2`, ... `t-k` lagged versions of a feature. For a quarterly feature, these lags represent "current_ratio 1 quarter ago", "2 quarters ago", etc. The CHANGE between lags is a tick-frequency signal.

**What we can adopt:** Instead of feeding raw `current_ratio` (constant for 63 days) to LSTM, feed `current_ratio_change_21d` (changes daily as the diff window slides). Our `feature_normalization.py` already computes `{var}_change_21d`. The classifier should recognize `_change_21d` suffixed features as TICK-class derivatives of PERIODIC features.

**Pattern 12: WindowFeatures for Rolling Statistics**

feature_engine's `WindowFeatures` computes rolling mean, std, min, max over configurable windows. These convert any-frequency features into daily-varying signals.

**What we can adopt:** `_zscore_63d` is a rolling z-score (WindowFeature). `_percentile_252d` is a rolling percentile (WindowFeature). Both convert periodic features into daily-varying signals suitable for temporal models.

---

## Project 7: Kats / Meta (6,301 stars)

**Repo:** `facebookresearch/Kats`
**Relevance:** Meta's time series analysis toolkit. Their feature extraction module classifies features by temporal characteristics.

### Key Patterns Found

**Pattern 13: TsFeatures -- Automated Time Series Characteristic Extraction**

Kats' `TsFeatures` extracts characteristics like trend strength, seasonality strength, entropy, and stationarity from a time series. These characteristics describe HOW the series behaves over time.

**What we can adopt:** Instead of just classifying by change rate, characterize each feature's temporal behavior:
- **Entropy** (from `complexity_signals.py`): high entropy = unpredictable = less useful for deterministic models
- **Stationarity** (from `derived_variables.py` Hurst exponent): H > 0.5 = trending = good for LSTM, H < 0.5 = mean-reverting = good for mean-reversion models
- **Autocorrelation**: high lag-1 autocorrelation = predictable sequence = good for VAR/AR models

This is a richer classification than binary tick/periodic -- it tells WHICH temporal model benefits most from each feature.

---

## Synthesis: Recommended Implementation

### Architecture (combining best patterns)

```python
# operator1/features/feature_classifier.py

from enum import Enum

class FeatureClass(Enum):
    TICK = "tick"          # changes most trading days
    PERIODIC = "periodic"  # changes at filing frequency
    STATIC = "static"      # changes rarely or never

class FeatureClassifier:
    """Classify cache columns by temporal resolution for model routing.
    
    Three classification methods, in priority order:
    1. Name-based heuristics (Pattern 8: entity map, instant)
    2. Suffix-based inference (Pattern 9/11/12: _zscore, _change, _percentile are TICK)
    3. Variance-based detection (Pattern 3/4: sklearn VarianceThreshold, for unknowns)
    """
    
    TICK_PREFIXES = ("return_", "log_return", "volatility_", "volume", "close_frac_diff",
                     "macd", "rsi", "adx", "obv", "bb_", "sma_", "online_change",
                     "sentiment_score", "benchmark_return")
    
    STATIC_PREFIXES = ("country_conflict", "company_conflict", "sanctions_flag",
                       "fragile_state", "conflict_type", "geo_hhi", "china_revenue",
                       "macro_quadrant", "fuzzy_protection", "fuzzy_sector")
    
    TICK_SUFFIXES = ("_zscore_63d", "_percentile_252d", "_change_21d", "_regime_zscore")
    
    def classify(self, col_name: str, series: pd.Series) -> FeatureClass:
        # 1. Name-based (instant, covers ~90% of known features)
        col_lower = col_name.lower()
        if any(col_lower.startswith(p) for p in self.TICK_PREFIXES):
            return FeatureClass.TICK
        if any(col_lower.startswith(p) for p in self.STATIC_PREFIXES):
            return FeatureClass.STATIC
        
        # 2. Suffix-based (normalized versions of periodic features ARE tick)
        if any(col_lower.endswith(s) for s in self.TICK_SUFFIXES):
            return FeatureClass.TICK
        
        # 3. Variance-based (sklearn Pattern 3)
        recent = series.tail(63).dropna()
        if len(recent) < 10:
            return FeatureClass.STATIC
        change_rate = recent.nunique() / len(recent)
        if change_rate > 0.30:
            return FeatureClass.TICK
        elif change_rate > 0.02:
            return FeatureClass.PERIODIC
        return FeatureClass.STATIC
    
    def build_model_feature_sets(self, cache, extra_vars, filing_frequency="quarterly"):
        """Route features to model-appropriate subsets (sktime Pattern 10)."""
        tick, periodic_norm, static = [], [], []
        for var in extra_vars:
            if var not in cache.columns:
                continue
            cls = self.classify(var, cache[var])
            if cls == FeatureClass.TICK:
                tick.append(var)
            elif cls == FeatureClass.PERIODIC:
                # Check if normalized version exists
                for suffix in self.TICK_SUFFIXES:
                    norm_var = var + suffix.replace("_63d", "_63d").replace("_252d", "_252d")
                    if norm_var in cache.columns:
                        tick.append(norm_var)
                        break
                else:
                    periodic_norm.append(var)
            else:
                static.append(var)
        
        return {
            "lstm": tick,                         # TICK only (Pattern 7: stationary preferred)
            "tree": tick + periodic_norm,          # TICK + normalized periodic (Pattern 5)
            "var": [v for v in tick if ...],       # TICK + stationary (Pattern 10)
            "kalman": [],                          # single variable (no extra features)
            "baseline": [],                        # no features
            "all": extra_vars,                     # backward compat
        }
```

### Priority Ranking

| Priority | Pattern | Source | Why |
|----------|---------|--------|-----|
| P1 | Name-based classification (entity map) | featuretools | Instant, covers 90% of known features |
| P1 | Suffix-based inference | feature_engine | Normalized features ARE tick-frequency |
| P1 | VarianceThreshold as first-pass filter | sklearn | Catches forward-filled constants cheaply |
| P2 | Prefer close_frac_diff over raw close | mlfinlab/AFML | Stationary + memory-preserving |
| P2 | Per-model input requirements | sktime | Clean model-feature contract |
| P2 | One representative per correlated group | mlfinlab CFI | Prevents dilution from 3x normalized variants |
| P3 | p-value relevance confirmation | tsfresh | Validates tick features actually predict target |
| P3 | Temporal characteristic extraction | Kats | Entropy/stationarity for model-feature matching |

### Estimated Implementation Size

| Component | Lines | Source Patterns |
|-----------|-------|-----------------|
| `operator1/features/feature_classifier.py` (NEW) | ~150 | sklearn, featuretools, sktime, mlfinlab |
| `operator1/stages/stage3_temporal.py` (modify) | ~30 | Wire classifier after feature selection |
| `operator1/stages/stage4_forecasting.py` (modify) | ~10 | Pass model_feature_sets |
| `operator1/models/forecasting.py` (modify) | ~30 | Accept model_feature_sets per model type |
| `operator1/pipeline_state.py` (modify) | ~2 | Add model_feature_sets field |
| **Total** | **~222 net** | |

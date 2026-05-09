# Full Implementation Plan: Multi-Frequency Model Adaptation

*17 modules, 61 hardcoded-252 refs, 6 hardcoded rolling windows, 2 min_obs thresholds. All line numbers verified via debug scan.*

---

## Step 0: Create `operator1/freq_constants.py` (NEW FILE)

Central frequency constants module that all models import:

```python
"""Frequency-aware constants for multi-frequency pipeline.

Every model replaces hardcoded 252 with:
    from operator1.freq_constants import get_periods_per_year
    ann = get_periods_per_year()  # returns 252 for D, 52 for W, 4 for Q, etc.
"""
import threading

_freq_context = threading.local()

PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "Q": 4, "S": 2, "A": 1}
VOL_ANNUALIZATION = {"D": 252**0.5, "W": 52**0.5, "M": 12**0.5, "Q": 2.0, "S": 2**0.5, "A": 1.0}
PERIODS_PER_QUARTER = {"D": 63, "W": 13, "M": 3, "Q": 1, "S": 0.5, "A": 0.25}
PERIODS_PER_MONTH = {"D": 21, "W": 4.3, "M": 1, "Q": 0.33, "S": 0.17, "A": 0.08}

# Models that should not run at certain frequencies
SKIP_AT_FREQ = {
    "pattern_detector": {"Q", "S", "A", "M"},
    "ohlc_predictor": {"Q", "S", "A", "M", "W"},
    "options_signals": {"Q", "S", "A", "M", "W"},
    "cross_asset_signals": {"Q", "S", "A", "M"},
    "behavioral_signals": {"Q", "S", "A", "M"},
    "news_sentiment": {"Q", "S", "A", "M"},
    "institutional_flow": {"Q", "S", "A", "M"},
    "accruals_forensics": {"D", "W", "M"},
    "earnings_smoothing": {"D", "W", "M"},
    "fcf_quality": {"D", "W", "M"},
}

# Model compatibility: min observations required
MIN_OBS = {
    "garch": {"D": 100, "W": 30, "M": 12, "Q": 0, "A": 0},
    "hmm": {"D": 50, "W": 20, "M": 8, "Q": 0, "A": 0},
    "lstm": {"D": 100, "W": 30, "M": 0, "Q": 0, "A": 0},
    "tree": {"D": 30, "W": 15, "M": 8, "Q": 0, "A": 0},
    "var": {"D": 50, "W": 20, "M": 0, "Q": 0, "A": 0},
}

def set_freq(freq: str) -> None:
    _freq_context.freq = freq.upper()

def get_freq() -> str:
    return getattr(_freq_context, "freq", "D")

def get_periods_per_year(freq: str | None = None) -> int:
    return PERIODS_PER_YEAR.get(freq or get_freq(), 252)

def get_vol_annualization(freq: str | None = None) -> float:
    return VOL_ANNUALIZATION.get(freq or get_freq(), 252**0.5)

def get_rolling_quarter(freq: str | None = None) -> int:
    return max(1, int(PERIODS_PER_QUARTER.get(freq or get_freq(), 63)))

def get_rolling_year(freq: str | None = None) -> int:
    return max(1, get_periods_per_year(freq))

def should_skip(model_name: str, freq: str | None = None) -> bool:
    f = freq or get_freq()
    return f in SKIP_AT_FREQ.get(model_name, set())
```

---

## Step 1: Replace 252 in Priority 1 Modules (14 files, 61 refs)

### 1.1 `operator1/hedge_fund/advanced_methods.py` (8 refs, 6 sqrt)

| Line | Current | Replace With |
|------|---------|-------------|
| L564 | `returns.iloc[-5:].std() * np.sqrt(252)` | `returns.iloc[-5:].std() * _vol_ann` |
| L565 | `returns.iloc[-21:].std() * np.sqrt(252)` | `returns.iloc[-21:].std() * _vol_ann` |
| L566 | `returns.iloc[-63:].std() * np.sqrt(252)` | `returns.iloc[-63:].std() * _vol_ann` |
| L567 | `returns.std() * np.sqrt(252)` | `returns.std() * _vol_ann` |
| L788 | `returns.iloc[-21:].std() * np.sqrt(252)` | `returns.iloc[-21:].std() * _vol_ann` |

Add at top: `from operator1.freq_constants import get_vol_annualization; _vol_ann = get_vol_annualization()`

### 1.2 `operator1/models/monte_carlo.py` (7 refs)

| Line | Current | Replace With |
|------|---------|-------------|
| L125 | `"252d": 252` | `"252d": _ppy` (in horizon dict) |
| L568 | `_dt = 1.0 / 252.0` | `_dt = 1.0 / _ppy` |
| L729 | `"annual": 252` | `"annual": _ppy` |

Add at top: `from operator1.freq_constants import get_periods_per_year; _ppy = get_periods_per_year()`

### 1.3 `operator1/models/prediction_aggregator.py` (7 refs, 2 sqrt)

| Line | Current | Replace With |
|------|---------|-------------|
| L2155 | `vol.iloc[-1]) / np.sqrt(252)` | `vol.iloc[-1]) / _vol_ann` |
| L2157 | `"next_year": 252` | `"next_year": _ppy` |
| L2870 | `horizon_days / 252` | `horizon_days / _ppy` |
| L3094 | `_iv30_val / math.sqrt(252)` | `_iv30_val / _vol_ann` |

### 1.4 `operator1/analysis/scenario_engine.py` (5 refs)

| Line | Current | Replace With |
|------|---------|-------------|
| L29 | `_DEFAULT_HORIZON_DAYS = 252` | `_DEFAULT_HORIZON_DAYS = get_periods_per_year()` |
| L290,329,368 | `up_to_day=min(252, horizon_days)` | `up_to_day=min(_ppy, horizon_days)` |

### 1.5 `operator1/models/ohlc_predictor.py` (5 refs, 1 sqrt)

| Line | Current | Replace With |
|------|---------|-------------|
| L182 | `sigma / math.sqrt(252)` | `sigma / _vol_ann` |
| L199 | `"252d": 252` | `"252d": _ppy` |
| L304 | `_generate_series(252, "252d")` | `_generate_series(_ppy, "252d")` |

### 1.6 `operator1/analysis/adaptive_model_params.py` (4 refs, 1 sqrt)

| Line | Current | Replace With |
|------|---------|-------------|
| L682 | `vol * math.sqrt(252)` | `vol * _vol_ann` |
| L1094,1138 | `/ 252.0` | `/ _ppy` |
| L1167 | `* 252` | `* _ppy` |

### 1.7 `operator1/features/product_metrics.py` (3 refs)

| Line | Current | Replace With |
|------|---------|-------------|
| L284 | `.tail(252)` | `.tail(_ppy)` |
| L296 | `slope * 252` | `slope * _ppy` |

### 1.8 `operator1/analysis/vanity.py` (3 refs)

| Line | Current | Replace With |
|------|---------|-------------|
| L115 | `.pct_change(periods=min(252, ...))` | `.pct_change(periods=min(_ppy, ...))` |
| L238 | `.pct_change(252, ...)` | `.pct_change(_ppy, ...)` |

### 1.9 `operator1/features/behavioral_signals.py` (2 refs + 3 windows)

| Line | Current | Replace With |
|------|---------|-------------|
| L56 | `close.rolling(252, ...)` | `close.rolling(_ppy, ...)` |
| L57 | `close.rolling(252, ...)` | `close.rolling(_ppy, ...)` |
| L83 | `volume.rolling(21, ...)` | `volume.rolling(_rpm, ...)` |
| L102 | `residual.rolling(63, ...)` | `residual.rolling(_rpq, ...)` |
| L105 | `ret.rolling(63, ...)` | `ret.rolling(_rpq, ...)` |

### 1.10 Remaining modules (L1 fixes each)

| Module | Line | Change |
|--------|------|--------|
| `models/forecasting.py` | L59 | `"252d": 252` -> `"252d": _ppy` |
| `models/transformer_forecaster.py` | L423 | `"252d": 252` -> `"252d": _ppy` |
| `models/copula.py` | (annualization) | Replace `sqrt(252)` with `_vol_ann` |
| `hedge_fund/accruals_forensics.py` | L315 | `len(so) >= 252` -> `len(so) >= _ppy` |
| `features/feature_normalization.py` | L82-83 | `.rolling(63, ...)` -> `.rolling(_rpq, ...)` |
| `models/regime_detector.py` | L986 | `reference_window: int = 252` -> `= _ppy` |

---

## Step 2: Model Routing Table (3 files)

### 2.1 `operator1/steps/multi_frequency_runner.py`

Before running each model in `run_single_frequency_pipeline()`, check `should_skip()`:

```python
from operator1.freq_constants import should_skip
if should_skip("pattern_detector", freq):
    logger.info("[%s] Skipping pattern_detector (D-only model)", freq)
else:
    detect_patterns(cache)
```

### 2.2 `operator1/stages/stage3_temporal.py` through `stage6_ensemble.py`

Add skip guards for freq-incompatible sub-stages. Example for 3.6 (pattern detection):

```python
def run_3_6_pattern(state):
    from operator1.freq_constants import should_skip, get_freq
    if should_skip("pattern_detector"):
        logger.info("Skipping pattern detection at freq=%s", get_freq())
        return
    # ... existing code
```

---

## Step 3: Hurst-Based Vol Scaling (5 files)

Replace `sqrt(T)` vol scaling with `T^H`:

```python
from operator1.freq_constants import get_periods_per_year

H = cache.get("hurst_exponent", pd.Series(0.5, index=cache.index))
H_latest = float(H.iloc[-1]) if H.notna().any() else 0.5
ppy = get_periods_per_year()
vol_annualized = vol_daily * (ppy ** H_latest)
```

Apply in: `advanced_methods.py`, `adaptive_model_params.py`, `monte_carlo.py`, `prediction_aggregator.py`, `copula.py`

---

## Step 4: MinTrace Hierarchical Reconciliation (1 new file)

### New: `operator1/models/hierarchical_reconciliation.py` (~200 lines)

Based on Nixtla/hierarchicalforecast MinTrace with mint_shrink:

```python
def reconcile_temporal_forecasts(
    forecasts: dict,  # {freq: {variable: {horizon: value}}}
    residuals: dict,  # {freq: {variable: [residual_series]}}
    method: str = "mint_shrink",
) -> dict:
    # Build summing matrix S encoding temporal hierarchy
    # A = sum of Q = sum of M
    S = _build_temporal_summing_matrix(forecasts.keys())
    
    # Estimate covariance W from residuals
    W = _estimate_covariance(residuals, method=method)
    
    # Reconciliation: P = (S'WS)^{-1} S'W^{-1}
    P = np.linalg.solve(S.T @ W @ S, S.T @ W)
    
    # Apply: reconciled = S @ P @ base_forecasts
    reconciled = S @ P @ base_vector
    return _unpack_reconciled(reconciled, forecasts.keys())
```

Wire into `frequency_fusion.py` after the existing fusion step.

---

## Step 5: Discrete-Time Survival at Q/A (1 file)

### `operator1/analysis/survival_mode.py`

Add `compute_discrete_time_survival()` for Q/A frequencies:

```python
def compute_discrete_time_survival(df, freq="Q"):
    """Discrete-time survival for quarterly/annual data.
    Uses complementary log-log link instead of continuous Cox PH.
    """
    # Data expansion: each at-risk observation generates a row
    expanded = _expand_discrete_data(df)
    # Fit with lifelines CoxPHFitter (cloglog link approximation)
    # or direct logistic regression with time dummies
    ...
```

Call from `compute_cox_survival_score()` when `freq in ("Q", "A", "S")`.

---

## Execution Checklist

```
Phase 1: freq_constants.py + 252 replacement (~700 lines across 17 files)
[ ] Create operator1/freq_constants.py
[ ] advanced_methods.py: 8 refs
[ ] monte_carlo.py: 7 refs
[ ] prediction_aggregator.py: 7 refs  
[ ] scenario_engine.py: 5 refs
[ ] ohlc_predictor.py: 5 refs
[ ] adaptive_model_params.py: 4 refs
[ ] product_metrics.py: 3 refs
[ ] vanity.py: 3 refs
[ ] behavioral_signals.py: 5 refs (252 + windows)
[ ] feature_normalization.py: 2 windows
[ ] forecasting.py: 1 ref
[ ] transformer_forecaster.py: 1 ref
[ ] copula.py: sqrt(252)
[ ] accruals_forensics.py: 1 ref
[ ] regime_detector.py: 1 ref
[ ] Walk-forward.py: min_obs

Phase 2: Model routing (~100 lines across 3 files)
[ ] multi_frequency_runner.py: skip guards
[ ] stage3_temporal.py: skip pattern at non-D
[ ] stage6_ensemble.py: skip OHLC at non-D

Phase 3: Hurst vol scaling (~30 lines across 5 files)
[ ] advanced_methods.py
[ ] adaptive_model_params.py
[ ] monte_carlo.py
[ ] prediction_aggregator.py
[ ] copula.py

Phase 4: MinTrace reconciliation (~200 lines, 1 new file)
[ ] hierarchical_reconciliation.py
[ ] Wire into frequency_fusion.py

Phase 5: Discrete-time survival (~100 lines, 1 file)
[ ] survival_mode.py: add compute_discrete_time_survival()
```

## Total Estimated Changes

| Phase | Files | Lines | Impact |
|-------|-------|-------|--------|
| 1 | 17 + 1 new | ~700 | Fixes all 61 annualization errors |
| 2 | 3 | ~100 | Eliminates wasted computation at wrong freqs |
| 3 | 5 | ~30 | Correct vol scaling using Hurst exponent |
| 4 | 1 new + 1 | ~220 | Coherent cross-frequency forecasts |
| 5 | 1 | ~100 | Correct survival analysis at Q/A |
| **Total** | **24 files** | **~1,150 lines** | |

# Debug Scan Findings and Fix Plan

Results of a comprehensive debug scan across 146 Python files in the Operator 1 pipeline. 5 issues found: 2 bugs and 3 data flow gaps. Bugs 1, 2, and Gap 4 are fixed. Gaps 3 and 5 analyzed below.

---

## Fixed Issues

| # | Issue | Status |
|---|-------|--------|
| Bug 1 | `persist_linked_aggregates` dead code in `linked_aggregates.py` | FIXED |
| Bug 2 | `enriched_survival_timeline` not rendered in reports | FIXED |
| Gap 4 | `_linked_prefixes` missing `rel_` prefix | FIXED |

---

## Gap 3: Adaptive Windows Not Passed to run_forecasting

### Current State

`_adaptive_tier3.windows` is computed in main.py Step 5k.2 via `compute_adaptive_windows()`. It produces a `WindowSet` with four filing-frequency-anchored values:

- `short` = filing_period / 3 (e.g., 21 for quarterly)
- `medium` = filing_period (e.g., 63 for quarterly)
- `long` = 2 * filing_period (e.g., 126 for quarterly)
- `trend` = 4 * filing_period (e.g., 252 for quarterly)

These are computed but never consumed by `run_forecasting()`. The forecasting module uses these hardcoded constants:

| Constant | Location | Value | Used By |
|----------|----------|-------|---------|
| `_LSTM_LOOKBACK` | `forecasting.py:77` | 21 | LSTMWrapper, TFTWrapper |
| `_BURNOUT_WINDOW` | `forecasting.py:71` | 126 | run_burnout window |
| `query_window=21` | `forecasting.py:3508` | 21 | DTW analog check in forward pass |

Additionally, `derived_variables.py` uses:
- `window=21` for volatility rolling std
- `window=252` for max drawdown

### Why This Matters

For a **semi-annual filer** (UK Companies House, some EU), the natural filing period is ~126 business days. Using a 21-day lookback for LSTM means the model sees roughly 1/6 of a filing period -- it's fitting sub-period noise. The Nyquist-anchored window would give `short=42` (1/3 of 126), which is long enough to capture one full statement transition without overfitting to intra-period interpolated data.

For an **annual filer** (some EU/SIX markets), the filing period is ~252 days. A 21-day LSTM lookback sees ~1/12 of a filing period -- almost pure noise from interpolation artifacts.

### Implementation Plan

The fix is straightforward because `run_forecasting` already accepts keyword arguments via `**kwargs`. The approach:

**Step 1: Add `windows` parameter to `run_forecasting`**

```python
def run_forecasting(
    cache: pd.DataFrame,
    variables: list[str] | None = None,
    *,
    extra_variables: list[str] | None = None,
    windows: Any | None = None,    # <-- NEW: AdaptiveTier3Params.windows
    random_state: int = 42,
    enable_burnout: bool = True,
) -> tuple[pd.DataFrame, ForecastResult]:
```

**Step 2: Inside run_forecasting, override hardcoded constants when windows provided**

Near line 1760, after the variable list is built:

```python
# Override hardcoded window sizes with adaptive values when available
_lstm_lookback = _LSTM_LOOKBACK
_burnout_win = _BURNOUT_WINDOW
if windows is not None:
    _lstm_lookback = getattr(windows, 'short', _LSTM_LOOKBACK)
    _burnout_win = getattr(windows, 'long', _BURNOUT_WINDOW)
    logger.info("Using adaptive windows: lstm_lookback=%d, burnout=%d",
                _lstm_lookback, _burnout_win)
```

Then pass `_lstm_lookback` to `LSTMWrapper` and `_burnout_win` to `_refinement_burnout`.

**Step 3: In main.py, pass the windows**

At line ~2388 where `run_forecasting` is called:

```python
cache, forecast_result = run_forecasting(
    cache,
    extra_variables=_extra_vars,
    windows=_adaptive_tier3.windows if _adaptive_tier3 is not None and _adaptive_tier3.adapted else None,
)
```

**Risk assessment:** LOW. The `windows` parameter defaults to `None`, meaning existing behavior is unchanged when adaptive windows aren't available. The `getattr` calls with defaults ensure backward compatibility even if the WindowSet shape changes.

**Files changed:** 2 (forecasting.py signature + internal routing, main.py call site)

---

## Gap 5: Mondrian Partitioning Not Wired

### Current State

`ConformalPIDCalibrator` (conformal.py:581) already supports Mondrian partitioning. Its `add_score()` method accepts a `mode` parameter:

```python
def add_score(self, variable, predicted, actual, mode="normal"):
```

And it maintains separate calibration buckets per `{variable}:{mode}` with hierarchical fallback:

```
both_unprotected -> crisis -> _global
company_only -> crisis -> _global
normal -> _global
```

The calibrator is already fully implemented. The problem is upstream: scores are only fed via the flat `update(residual)` method which routes everything to `_global`:

1. **In the forward pass** (forecasting.py:3462): `_conformal_calibrator.add_score(var_name, ensemble_pred, actual)` -- no `mode` parameter, defaults to `"normal"`.

2. **In main.py** (line 2632-2634): `calibrator.update(r)` feeds flat residuals from `forecast_result.residuals` -- no variable or mode info.

### Why This Matters

During a survival crisis, prediction errors are typically 2-5x larger than during normal operations (fat tails, regime switches, structural breaks). If the calibrator uses one global score pool, intervals in normal mode are too wide (contaminated by crisis residuals) and intervals in crisis mode are too narrow (dominated by normal residuals). The Mondrian approach would produce:

- Normal mode: intervals from normal-era residuals (tight, appropriate)
- Crisis mode: intervals from crisis-era residuals (wide, appropriate)
- Rare modes: hierarchical fallback to crisis or global pool

### Implementation Plan

**Step 1: Feed survival_mode to the forward pass conformal calibrator**

In `run_forward_pass` (forecasting.py), the `cache` DataFrame already contains `survival_mode` from the enriched timeline (Step 5.5). At line 3462:

```python
# CURRENT:
_conformal_calibrator.add_score(var_name, ensemble_pred, float(actual_t1[0]))

# FIXED:
_mode = "normal"
if "survival_mode" in cache.columns:
    _sm = cache["survival_mode"].iloc[t]
    if pd.notna(_sm):
        _mode = str(_sm)
if isinstance(_conformal_calibrator, ConformalPIDCalibrator):
    _conformal_calibrator.add_score(var_name, ensemble_pred, float(actual_t1[0]), mode=_mode)
else:
    _conformal_calibrator.add_score(var_name, ensemble_pred, float(actual_t1[0]))
```

**Step 2: Feed survival_mode to the main.py conformal calibrator**

In main.py ~line 2632, the flat residual feeding doesn't have mode information. But we can change the approach: instead of feeding flat residuals from `forecast_result`, use the forward pass calibrator directly. The forward pass already creates a `_conformal_calibrator` and feeds it per-prediction scores with variable names. We just need to:

(a) Store the forward pass calibrator in `ForwardPassResult`:
```python
result.conformal_calibrator = _conformal_calibrator
```

This already happens at line 3533.

(b) In main.py, use the forward pass calibrator instead of creating a new one:

```python
# CURRENT (creates new calibrator, feeds flat residuals):
calibrator = ConformalPIDCalibrator(target_coverage=0.9)
for r in forecast_result.residuals:
    calibrator.update(r)

# FIXED (reuse forward pass calibrator which has per-variable per-mode scores):
if forward_pass_result is not None and hasattr(forward_pass_result, 'conformal_calibrator'):
    calibrator = forward_pass_result.conformal_calibrator
    logger.info("Reusing forward pass conformal calibrator (per-variable per-mode scores)")
else:
    calibrator = ConformalPIDCalibrator(target_coverage=0.9)
    if hasattr(forecast_result, 'residuals') and forecast_result.residuals:
        for r in forecast_result.residuals:
            calibrator.update(r)
```

**Step 3: Add isinstance check in forward pass**

The forward pass currently imports `ConformalCalibrator` but needs to also handle `ConformalPIDCalibrator` for the `mode` parameter. Add:

```python
from operator1.models.conformal import ConformalCalibrator
try:
    from operator1.models.conformal import ConformalPIDCalibrator
    _use_pid_calibrator = True
except ImportError:
    _use_pid_calibrator = False
```

Then create the PID version when available:
```python
if _use_pid_calibrator:
    _conformal_calibrator = ConformalPIDCalibrator(target_coverage=0.9)
else:
    _conformal_calibrator = ConformalCalibrator(coverage=0.9, adaptive=True)
```

**Risk assessment:** LOW-MEDIUM. The forward pass calibrator already exists and is stored in the result. The main change is adding `mode` parameter to `add_score` calls and reusing the calibrator. Fallback behavior (global bucket) is unchanged when `survival_mode` column is absent.

**Files changed:** 2 (forecasting.py forward pass loop + calibrator init, main.py conformal block)

---

## Implementation Priority

| # | Issue | Severity | Risk | Status |
|---|-------|----------|------|--------|
| Bug 1 | persist_linked_aggregates dead code | Medium | None | FIXED |
| Gap 4 | rel_ columns not in _extra_vars | Low | None | FIXED |
| Bug 2 | enriched_survival_timeline not in reports | Low | None | FIXED |
| Gap 3 | Adaptive windows not passed to forecasting | Low | Low | Ready to implement |
| Gap 5 | Mondrian partitioning not wired | Low | Low-Med | Ready to implement |

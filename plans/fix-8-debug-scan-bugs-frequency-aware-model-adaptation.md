# Fix 8 Debug Scan Bugs + Frequency-Aware Model Adaptation

## Overview

8 bugs found during a full debug scan of 170 Python files after a live AAPL backtest (2024 data predicting 2025). 3 are critical (cause runtime failures), 3 are medium (functionality gaps), 2 are low (dead code). Plus 1 enhancement to make all models frequency-aware during multi-frequency pipeline execution.

---

## Fix 1: prediction_aggregator.py `h_days` typo (CRITICAL)

**File:** `operator1/models/prediction_aggregator.py`
**Line:** 2563
**Bug:** `h_days` is undefined. Line 2359 defines `horizon_days = HORIZONS.get(h_label, 1)` but line 2563 uses `h_days`.
**Impact:** `NameError` crashes `run_prediction_aggregation()` on every call. The entire 10-step ensemble never runs.

**Fix:**
```
Line 2563: Change h_days -> horizon_days
```

**Verification:** Prediction aggregation completes without NameError. Ensemble weights, copula widening, DTW overlay, SHAP attachment all execute.

---

## Fix 2: backtest_runner.py forward pass missing traceback (CRITICAL)

**File:** `backtest_runner.py`
**Line:** 1336
**Bug:** `except Exception as exc: logger.warning("Forward pass failed: %s", exc)` -- no traceback printed, making the root cause invisible. The exception value `0` is likely a `KeyError(0)` from indexing.

**Fix:**
```python
# Line 1336: Add traceback
import traceback
logger.warning("Forward pass failed: %s\n%s", exc, traceback.format_exc())
```

Also apply the same pattern to the other bare `except Exception` blocks in backtest_runner.py that suppress tracebacks (lines ~1344, ~1480, ~1506, etc. -- any model call that catches and swallows).

**Verification:** Re-run the AAPL backtest Stage 2 and capture the full stack trace to identify the actual IndexError/KeyError source.

---

## Fix 3: VAROnlineWrapper maxlags cap (CRITICAL)

**File:** `operator1/models/forecasting.py`
**Line:** 2711
**Bug:** `VAROnlineWrapper._refit()` passes `maxlags=self._max_lag` (=10) directly without the `len // 3` cap that `fit_var()` uses. With ~40 columns, VAR needs 401+ observations minimum.

**Fix:**
```python
# Line 2711: Add cap matching fit_var() logic
lag_order = min(self._max_lag, max(1, len(self._buffer) // (3 * max(1, self._buffer.shape[1]))))
self._result = model.fit(maxlags=max(1, lag_order), ic="aic")
```

**Verification:** VAR should fit successfully for at least some variables on daily data instead of falling back to AR(1) for all 40+.

---

## Fix 4: survival_mode.py read thresholds from scoring_weights (MEDIUM)

**File:** `operator1/analysis/survival_mode.py`
**Bug:** Hardcoded thresholds (`current_ratio < 1.0`, `debt_to_equity_abs > 3.0`, `fcf_yield < 0`, `drawdown_252d < -0.40`) ignore `config/scoring_weights.yml` edits from the dashboard.

**Fix:**
In `compute_company_survival_flag()`, replace hardcoded values with:
```python
from operator1.scoring_weights import get_weight

_thresh_cr = float(get_weight("survival_thresholds.current_ratio", 1.0))
_thresh_de = float(get_weight("survival_thresholds.debt_to_equity", 3.0))
_thresh_fcf = float(get_weight("survival_thresholds.fcf_yield", 0.0))
_thresh_dd = float(get_weight("survival_thresholds.drawdown_252d", -0.40))
_thresh_ci = float(get_weight("survival_thresholds.conflict_intensity", 0.70))
_thresh_ifm = float(get_weight("survival_thresholds.inst_flow_momentum", -0.15))
```

Similarly update `compute_survival_probability()` to read thresholds from config.

Accept optional `thresholds` dict parameter (already exists for adaptive override) -- the config values become the base defaults that adaptive can further tune.

**Verification:** Edit `survival_thresholds.current_ratio` in `config/scoring_weights.yml` to 0.8, re-run, verify fewer survival days flagged.

---

## Fix 5: Frequency-aware regime detection for MF pipeline (MEDIUM)

**File:** `operator1/models/regime_detector.py`
**Bug:** Regime detection requires `return_1d` column. Resampled caches (A/Q/M/W) from raw filings do not have this column. Regime detection skips silently for 4 of 5 frequencies.

**Fix:**
1. In `run_early_regime_detection()` and `detect_regimes_and_breaks()`, add frequency-aware return column detection:

```python
# Detect the appropriate return column for this frequency
_return_col = None
for candidate in ["return_1d", "return_1w", "return_1m", "close"]:
    if candidate in cache.columns and cache[candidate].notna().sum() > 10:
        _return_col = candidate
        break

# If no return column exists but close does, compute it
if _return_col is None and "close" in cache.columns:
    cache["_freq_return"] = cache["close"].pct_change()
    _return_col = "_freq_return"
```

2. In `operator1/features/frequency_resampler.py`, when building resampled caches, compute period-appropriate returns from close:
```python
if "close" in resampled.columns:
    resampled["return_1d"] = resampled["close"].pct_change()  # period return, not daily
```

3. In `operator1/features/derived_variables.py`, add a guard that computes `volatility_21d` from whatever return column is available, not just `return_1d`.

**Verification:** Re-run MF pipeline. All 5 frequencies should produce HMM regime labels and structural break detection.

---

## Fix 6: Remove dead scoring_weights typed accessors (LOW)

**File:** `operator1/scoring_weights.py`
**Bug:** 12 typed accessor functions (`get_survival_thresholds()`, `get_hierarchy_weights()`, etc.) are defined but never called. All consumers use `get_weight("dotted.path", default)` directly.

**Fix:** Two options:
- **Option A (preferred):** Keep the typed accessors but wire them into the consuming modules (survival_mode, hierarchy_weights, monte_carlo, etc.) to replace `get_weight()` calls. This provides better type safety and centralized defaults.
- **Option B:** Delete the 12 unused functions and their docstrings (~120 lines).

**Recommendation:** Option A -- the typed accessors have correct defaults and would improve maintainability if wired in. But this is low priority since `get_weight()` works correctly.

---

## Fix 7: PipelineState / BacktestState attribute alignment (LOW)

**File:** `operator1/pipeline_state.py`, `backtest_runner.py`
**Bug:** 24 attributes have naming mismatches (PipelineState uses `extra_vars`, BacktestState uses `_extra_vars`). Stage modules reference PipelineState-style names.

**Fix:** Not a runtime bug since they use separate code paths. Document the naming convention:
- PipelineState (for `stages/runner.py`): clean public names
- BacktestState (for `backtest_runner.py`): underscore-prefixed private names

If future refactoring merges the two, align to PipelineState naming.

---

## Fix 8: stage7_integration.py MF helper method verification (LOW)

**File:** `operator1/stages/stage7_integration.py`
**Bug:** References 9 MF helper methods (`load_mf_cache`, `save_mf_result`, etc.) that the automated scan flagged as "unknown attrs" on PipelineState.

**Fix:** Verify these methods exist on PipelineState (likely defined in the class body but not captured by the simple regex scan). If missing, add them as methods on PipelineState that delegate to the MF sub-directory helpers already in the `save()`/`load()` methods.

---

## Execution Order

1. **Fix 1** (h_days typo) -- 1 line change, highest impact
2. **Fix 3** (VAR maxlags cap) -- 2 line change, restores multivariate forecasting
3. **Fix 2** (traceback in backtest_runner) -- 5-10 line changes, enables debugging
4. **Fix 4** (survival_mode config) -- 15-20 lines, connects dashboard to survival detection
5. **Fix 5** (frequency-aware regime detection) -- 30-50 lines across 3 files, enables regime detection for A/Q/M/W
6. **Fix 6** (dead code cleanup) -- optional, 0-120 lines
7. **Fix 7** (attribute naming docs) -- documentation only
8. **Fix 8** (MF helper verification) -- investigation + 0-20 lines

Fixes 1-3 should be done together as they are all critical runtime failures. Fix 4-5 form a natural second batch. Fixes 6-8 are optional cleanup.

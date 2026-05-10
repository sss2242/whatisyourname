# Batch B: Full Implementation Plan -- FH Scoring Redesign + OHLC Predictor Replacement

*Problems 3 (self-referential FH scoring), 4 (OHLC random walk)*
*Based on debug scan + 13 patterns from 7 open-source projects*

---

## Debug Scan Summary

### Problem 3: FH Scoring Self-Referential

**Root cause:** [`_normalize_series()`](operator1/models/financial_health.py:340) computes expanding percentile rank of each variable against the company's OWN history. Apple's current_ratio=0.92 ranks ~50th of Apple's own current_ratio (which has always been ~0.9).

**Called from 5 tier scorers:**
- [`_score_liquidity()`](operator1/models/financial_health.py:430) -- cash_ratio, current_ratio, free_cash_flow
- [`_score_solvency()`](operator1/models/financial_health.py:465) -- debt_to_equity, interest_coverage, net_debt_to_ebitda
- [`_score_stability()`](operator1/models/financial_health.py:507) -- volatility, drawdown, volume
- [`_score_profitability()`](operator1/models/financial_health.py:549) -- gross_margin, operating_margin, net_margin
- [`_score_growth()`](operator1/models/financial_health.py:570) -- revenue_growth, PE, EV/EBITDA

**FH result consumed by:**
- [`compute_hierarchy_weights()`](operator1/analysis/hierarchy_weights.py:289) -- via `fh_composite_score` in cache
- [`run_hedge_fund_analysis()`](operator1/hedge_fund/engine.py:1523) -- `fh_result` parameter
- [`compute_earnings_smoothing()`](operator1/hedge_fund/earnings_smoothing.py:69) -- Beneish from fh_result
- [`stage2_freq_pipeline.py`](operator1/stages/stage2_freq_pipeline.py:338) -- per-freq FH
- [`stage7_integration.py`](operator1/stages/stage7_integration.py:535) -- HF analysis
- [`build_company_profile()`](operator1/report/profile_builder.py:1121) -- profile `financial_health` section

### Problem 4: OHLC Random Walk

**Root cause:** [`predict_ohlc_series()`](operator1/models/ohlc_predictor.py:97) generates single-seed `rng.normal()` paths with custom drift (`mu` from 63-day average) and noise model. It does NOT use MC's 10,000-path regime-switching distributions.

**MC terminal_values structure:** `mc_result.terminal_values` is `dict[str, np.ndarray]` where key=horizon label ("1d", "252d"), value=array of cumulative returns across paths ([`monte_carlo.py:292`](operator1/models/monte_carlo.py:292)).

**OHLC result consumed by:**
- [`profile["ohlc_predictions"]`](main.py:3277) -- via `format_ohlc_for_profile()`
- [`stage6_ensemble.py`](operator1/stages/stage6_ensemble.py) -- run_6_10_ohlc()
- [`detect_patterns_on_predicted_ohlc()`](operator1/models/pattern_detector.py) -- predicted patterns

---

## Implementation Plan

### Phase 1: Sector Reference Ranges Config (~100 lines)

**New file:** `config/sector_reference_ranges.yml`

```yaml
# Sector reference ranges for cross-sectional FH scoring.
# Each sector: {variable: [p10, p25, median, p75, p90]}
# Sources: Damodaran NYU sector averages (2025), S&P Capital IQ.

technology:
  current_ratio: [0.7, 0.9, 1.2, 1.8, 2.5]
  debt_to_equity: [0.3, 0.8, 1.5, 2.5, 4.0]
  gross_margin: [0.35, 0.50, 0.62, 0.72, 0.82]
  operating_margin: [0.05, 0.12, 0.22, 0.30, 0.40]
  net_margin: [0.03, 0.08, 0.18, 0.25, 0.35]
  fcf_yield: [-0.02, 0.01, 0.03, 0.05, 0.08]
  revenue_growth_yoy: [-0.05, 0.02, 0.08, 0.15, 0.30]
  volatility_21d: [0.01, 0.015, 0.02, 0.03, 0.05]
  interest_coverage: [3, 8, 15, 30, 60]
  
financial_services:
  current_ratio: [0.3, 0.5, 0.8, 1.2, 1.5]
  debt_to_equity: [2.0, 5.0, 8.0, 12.0, 18.0]
  gross_margin: [0.20, 0.35, 0.50, 0.65, 0.80]
  # ... banks are inherently leveraged
  
healthcare:
  current_ratio: [1.0, 1.5, 2.2, 3.0, 4.5]
  gross_margin: [0.40, 0.55, 0.65, 0.75, 0.85]
  
consumer_cyclical:
  current_ratio: [0.8, 1.0, 1.3, 1.8, 2.5]
  gross_margin: [0.25, 0.35, 0.45, 0.55, 0.65]

energy:
  current_ratio: [0.8, 1.0, 1.2, 1.5, 2.0]
  debt_to_equity: [0.5, 1.0, 1.5, 2.5, 4.0]

industrials:
  current_ratio: [1.0, 1.3, 1.6, 2.0, 2.8]
  
communication_services:
  current_ratio: [0.6, 0.8, 1.0, 1.4, 2.0]
  
_default:
  current_ratio: [0.5, 0.8, 1.2, 2.0, 3.0]
  debt_to_equity: [0.3, 0.8, 1.5, 3.0, 5.0]
  gross_margin: [0.20, 0.35, 0.50, 0.65, 0.80]
  operating_margin: [0.02, 0.08, 0.15, 0.22, 0.30]
  net_margin: [0.01, 0.05, 0.10, 0.18, 0.25]
  fcf_yield: [-0.03, 0.00, 0.03, 0.06, 0.10]
  revenue_growth_yoy: [-0.10, 0.00, 0.05, 0.12, 0.25]
  volatility_21d: [0.008, 0.012, 0.018, 0.028, 0.045]
  interest_coverage: [1, 3, 8, 15, 30]
```

### Phase 2: Cross-Sectional Scoring Function (~80 lines)

**File:** [`operator1/models/financial_health.py`](operator1/models/financial_health.py)

Add `_cross_sectional_score()` and `_load_sector_ranges()` functions. Modify each `_score_*` function to use cross-sectional scoring when sector ranges are available.

**New function at ~line 335:**
```python
_SECTOR_RANGES: dict | None = None

def _load_sector_ranges() -> dict:
    global _SECTOR_RANGES
    if _SECTOR_RANGES is not None:
        return _SECTOR_RANGES
    try:
        from operator1.config_loader import load_config
        _SECTOR_RANGES = load_config("sector_reference_ranges")
    except Exception:
        _SECTOR_RANGES = {}
    return _SECTOR_RANGES

def _cross_sectional_score(
    value: float,
    sector_range: list[float],
    higher_is_better: bool = True,
) -> float:
    """Score 0-100 based on sector reference range (alphalens pattern).
    
    sector_range: [p10, p25, median, p75, p90]
    Returns score where median -> 50, p90 -> 90, etc.
    """
    if value is None or np.isnan(value):
        return np.nan
    bp = list(sector_range)
    if not higher_is_better:
        bp = bp[::-1]
    score_anchors = [10, 25, 50, 75, 90]
    # Linear interpolation between breakpoints
    for i in range(len(bp) - 1):
        if bp[i] <= value <= bp[i + 1]:
            frac = (value - bp[i]) / max(bp[i+1] - bp[i], 1e-9)
            return score_anchors[i] + frac * (score_anchors[i+1] - score_anchors[i])
    if value < bp[0]:
        return max(0, 10 * value / max(abs(bp[0]), 1e-9))
    return min(100, 90 + 10)

def _get_sector_range(sector: str, variable: str) -> list[float] | None:
    ranges = _load_sector_ranges()
    sector_key = sector.lower().replace(" ", "_") if sector else "_default"
    section = ranges.get(sector_key, ranges.get("_default", {}))
    return section.get(variable)
```

**Modify each `_score_*` function:** Add `sector` parameter. For each variable, try cross-sectional first, fall back to self-percentile:

```python
def _score_liquidity(cache: pd.DataFrame, sector: str = "") -> pd.Series:
    scores = []
    for var, higher_better in [("cash_ratio", True), ("current_ratio", True), ("free_cash_flow", True)]:
        s = cache.get(var)
        if s is None or s.isna().all():
            continue
        sr = _get_sector_range(sector, var)
        if sr:
            # Cross-sectional: 70% sector-relative + 30% own trend
            cross = s.apply(lambda v: _cross_sectional_score(v, sr, higher_better))
            trend = _normalize_series(s, invert=not higher_better)
            score = 0.7 * cross + 0.3 * trend
        else:
            score = _normalize_series(s, invert=not higher_better)
        scores.append(score)
    return pd.concat(scores, axis=1).mean(axis=1) if scores else pd.Series(50, index=cache.index)
```

### Phase 3: Wire sector into compute_financial_health (~10 lines)

**File:** [`operator1/models/financial_health.py`](operator1/models/financial_health.py:909)

The `compute_financial_health()` function already accepts `sector` parameter (added in PR #4). Pass it through to each `_score_*` function:

```python
# Line ~940: currently calls _score_liquidity(cache)
# Change to: _score_liquidity(cache, sector=sector)
```

Same for `_score_solvency`, `_score_stability`, `_score_profitability`, `_score_growth`.

### Phase 4: Rewrite OHLC Predictor (~120 lines)

**File:** [`operator1/models/ohlc_predictor.py`](operator1/models/ohlc_predictor.py:97)

Replace the random walk in `predict_ohlc_series()` with MC-derived predictions:

```python
def predict_ohlc_series(cache, forecast_result=None, mc_result=None, ...):
    # ... existing setup code (last_close, returns, mu, sigma) ...
    
    # NEW: Derive predictions from MC terminal values when available
    for horizon_key, n_days in HORIZONS.items():
        terminal = None
        if mc_result is not None:
            terminal = getattr(mc_result, 'terminal_values', {}).get(horizon_key)
        
        if terminal is not None and hasattr(terminal, '__len__') and len(terminal) > 50:
            # Close: MC median (vectorbt pattern)
            close_est = last_close * float(np.median(terminal))
            
            # Range: Parkinson sigma from MC distribution (arch pattern)
            log_terminal = np.log(np.maximum(terminal, 1e-10))
            sigma_est = float(np.std(log_terminal))
            expected_range = sigma_est * np.sqrt(8 / np.pi)
            
            # Skewness-adjusted H/L (QuantLib pattern)
            skew = float(cache.get("skewness_63d", pd.Series(0)).dropna().iloc[-1]) if "skewness_63d" in cache.columns else 0.0
            skew_adj = max(-1.0, min(1.0, skew))
            up_ratio = 0.5 + 0.1 * skew_adj
            down_ratio = 0.5 - 0.1 * skew_adj
            
            high_est = close_est * (1 + expected_range * up_ratio)
            low_est = close_est * (1 - expected_range * down_ratio)
            open_est = prev_close  # or last_close for day 1
        else:
            # FALLBACK: bounded forecast (no random walk)
            forecast_return = mu * n_days
            forecast_return = max(-0.30, min(0.30, forecast_return))  # cap at 30%
            close_est = last_close * (1 + forecast_return)
            sigma_scaled = sigma_daily * np.sqrt(n_days)
            high_est = close_est * (1 + sigma_scaled)
            low_est = close_est * (1 - sigma_scaled)
            open_est = prev_close
        
        # Enforce H >= C >= L
        high_est = max(high_est, close_est)
        low_est = min(low_est, close_est)
```

### Phase 5: Update stage6_ensemble OHLC call (~5 lines)

**File:** [`operator1/stages/stage6_ensemble.py`](operator1/stages/stage6_ensemble.py)

Ensure MC result is passed to OHLC predictor (it already is in most paths, but verify):

```python
def run_6_10_ohlc(state):
    state.ohlc_result = predict_ohlc_series(
        state.cache,
        forecast_result=state.forecast_result,
        mc_result=state.mc_result,  # <-- ensure this is passed
        ...
    )
```

---

## Files Changed Summary

| File | Change | Lines +/- |
|------|--------|-----------|
| `config/sector_reference_ranges.yml` | **NEW** | +100 |
| `operator1/models/financial_health.py` | Add cross-sectional scoring | +80, -0 |
| `operator1/models/ohlc_predictor.py` | Replace random walk with MC-derived | +60, -40 |
| `operator1/stages/stage6_ensemble.py` | Verify MC passthrough | +2 |
| **Total** | | **~245 net** |

---

## Validation Criteria

1. Apple FH composite > 50 (currently 28.97, will improve with cross-sectional scoring)
2. Apple FH label not "Critical" or "Poor"
3. Technology companies with >30% gross margin score >40 on profitability tier
4. Banks with high D/E don't score 0 on solvency (sector range accommodates leverage)
5. OHLC year-end close within +/-30% of current price for S&P 500 companies
6. OHLC year-end close equals MC median terminal value (when MC available)
7. High > Close > Low for all predicted candles
8. No `rng.normal()` random walk code remains in ohlc_predictor.py (replaced by MC)
9. `pytest tests/test_financial_health.py -v` -- all existing tests still pass
10. Integration test `test_apple_fh_composite_above_minimum` passes with min=40

---

## Execution Order

1. Create `sector_reference_ranges.yml` (Phase 1)
2. Add `_cross_sectional_score()` to financial_health.py (Phase 2)
3. Wire sector through _score_* functions (Phase 3)
4. Rewrite OHLC predictor (Phase 4)
5. Verify MC passthrough (Phase 5)
6. Run tests to validate

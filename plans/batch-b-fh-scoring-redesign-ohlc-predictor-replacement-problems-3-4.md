# Batch B: FH Scoring Redesign + OHLC Predictor Replacement

*Problems 3 (self-referential financial health scoring), 4 (OHLC predictor duplicates MC)*

---

## Problem 3: Financial Health Scoring Is Self-Referential

### Current behavior

`_normalize_series()` in financial_health.py line 340 computes expanding percentile rank of each variable against the company's own history. Apple's `current_ratio = 0.92` ranks ~50th percentile of Apple's own current_ratio history (which has always been ~0.9). Apple's `gross_margin = 0.77` ranks LOW when it drops from 0.78 to 0.77 even though 77% is exceptional by any absolute standard.

Result: Apple FH = 28.97 "Weak" despite being one of the most financially healthy companies on Earth.

### Expert methods

**From credit risk modeling (Edward Altman, "Predicting Financial Distress of Companies"):**
- Altman Z-Score uses **absolute coefficients** derived from cross-sectional data (1,000+ companies). The thresholds (Z > 2.99 = safe, Z < 1.81 = distress) are calibrated from bankruptcy prediction across industries. Altman never compares a company to itself.
- Applied here: FH tier scoring should use cross-sectional baselines, not self-referential percentiles.

**From quantitative equity research (Barra/MSCI Factor Models):**
- **Cross-sectional z-score normalization**: Every factor (value, momentum, quality) is computed as a z-score relative to the investable universe or sector peer group at each point in time. A company's quality score is its distance from the sector median in standard deviations.
- Applied here: Replace expanding self-percentile with **sector cross-sectional z-score** using peer data from `peer_ranking.py` when available, sector medians from a reference table when not.

**From clinical medicine (diagnostic reference ranges):**
- Blood test results are always interpreted against **population reference ranges** stratified by age/sex, not against the patient's own history. A hemoglobin of 12 is "normal" for a woman but "low" for a man. The reference range is the external anchor.
- Applied here: Each financial ratio needs a **sector reference range** that defines what "healthy" means for that sector. Technology companies with current_ratio < 1.0 are in the normal range for tech; the same value for a utility is concerning.

**From behavioral finance (Daniel Kahneman, "Thinking Fast and Slow"):**
- **Anchoring bias**: The current system anchors health scores to the company's own history, which is exactly the anchoring bias Kahneman warns about. When you evaluate Apple's margins against Apple's own history, you're anchored to Apple's exceptional past. Any small decline looks like deterioration even when the company remains in the top 1% globally.
- Applied here: The scoring function should anchor to external sector norms, not self-history.

### Implementation

**Phase 1: Sector Reference Ranges** (new config)

**File:** `config/sector_reference_ranges.yml` (new, ~100 lines)

```yaml
# Sector reference ranges for FH cross-sectional scoring.
# Each range defines [p10, p25, median, p75, p90] from industry data.
# Sources: Damodaran NYU sector averages, S&P Capital IQ aggregates.

technology:
  current_ratio: [0.7, 0.9, 1.2, 1.8, 2.5]
  debt_to_equity: [0.3, 0.8, 1.5, 2.5, 4.0]
  gross_margin: [0.35, 0.50, 0.62, 0.72, 0.82]
  operating_margin: [0.05, 0.12, 0.22, 0.30, 0.40]
  net_margin: [0.03, 0.08, 0.18, 0.25, 0.35]
  fcf_yield: [-0.02, 0.01, 0.03, 0.05, 0.08]
  revenue_growth_yoy: [-0.05, 0.02, 0.08, 0.15, 0.30]
  
financial_services:
  current_ratio: [0.3, 0.5, 0.8, 1.2, 1.5]
  debt_to_equity: [2.0, 5.0, 8.0, 12.0, 18.0]
  # ... banks are inherently leveraged
  
healthcare:
  current_ratio: [1.0, 1.5, 2.2, 3.0, 4.5]
  gross_margin: [0.40, 0.55, 0.65, 0.75, 0.85]
  # ...

_default:  # fallback when sector not matched
  current_ratio: [0.5, 0.8, 1.2, 2.0, 3.0]
  debt_to_equity: [0.3, 0.8, 1.5, 3.0, 5.0]
  # ...
```

**Phase 2: Cross-Sectional Scoring Function**

**File:** `operator1/models/financial_health.py`

Replace `_normalize_series()` with `_cross_sectional_score()`:

```python
def _cross_sectional_score(
    value: float,
    sector_range: list[float],  # [p10, p25, median, p75, p90]
    higher_is_better: bool = True,
) -> float:
    """Score a value 0-100 based on sector reference range.
    
    Uses linear interpolation between sector percentile breakpoints.
    Value at sector median = 50. Value at sector p90 = 90.
    Value beyond p90 = 90-100 (capped). Below p10 = 0-10.
    """
```

**Phase 3: Hybrid scoring** (cross-sectional primary, self-history secondary)

The new scoring pipeline for each tier:
1. **Cross-sectional score** (weight: 0.7): Where does this value sit in the sector reference range?
2. **Trend score** (weight: 0.3): Is the value improving or deteriorating vs own recent history? (This keeps the expanding percentile rank but as a secondary signal, not the primary score.)

When peer data from `peer_ranking.py` is available, use live peer distribution instead of static reference ranges.

**Expected result for Apple (technology sector):**
- `current_ratio = 0.92`: Cross-sectional = ~55 (between p25=0.9 and median=1.2). Trend = ~50 (stable). Weighted = 53.
- `gross_margin = 0.77`: Cross-sectional = ~85 (between p75=0.72 and p90=0.82). Trend = ~45 (slight decline). Weighted = 73.
- Composite would be ~55-65 "Good" instead of 28 "Weak".

---

## Problem 4: OHLC Predictor Runs Its Own Random Walk

### Current behavior

`predict_ohlc_series()` in ohlc_predictor.py generates a single-seed random walk using `rng.normal()` for each day. It has its own noise model (`sigma_daily * survival_mult * growth_factor`), its own drift (`mu` from 63-day average), and produces a single path that can produce absurd year-end prices ($78 for Apple from $245).

Meanwhile, `monte_carlo.py` already generates 10,000 regime-switching paths with importance sampling, producing `terminal_values` arrays with proper distributional shape.

### Expert methods

**From quantitative finance (Paul Glasserman, "Monte Carlo Methods in Financial Engineering"):**
- The canonical approach to OHLC prediction is to **derive intraday statistics from the return distribution**, not generate a parallel simulation. Given a distribution of daily returns, the expected High = Close * (1 + E[max intraday excursion]) and expected Low = Close * (1 - E[max intraday drawdown]). Parkinson (1980) and Garman-Klass (1980) provide the exact relationships.
- Applied here: Use MC terminal distributions for Close, derive High/Low from the return distribution's intraday range characteristics.

**From ensemble forecasting (meteorology, ECMWF):**
- Weather forecasting uses **ensemble percentiles** for uncertainty communication. The 50th percentile of the ensemble is the point forecast. The 10th and 90th are the confidence bands. You never run a separate single-path simulation alongside the ensemble.
- Applied here: OHLC predictions should be percentiles of the MC ensemble, not a separate random walk.

**From options pricing (Black-Scholes barrier options):**
- Barrier option pricing gives the **exact probability distribution of the path maximum and minimum** given the terminal distribution. The expected high/low of a price path over N days is a function of the drift, volatility, and time.
- Applied here: Use the MC regime distributions to compute expected path extremes analytically, then convert to OHLC.

### Implementation

**Phase 1: Replace random walk with MC-derived predictions**

**File:** `operator1/models/ohlc_predictor.py` -- rewrite `predict_ohlc_series()`

```python
def predict_ohlc_series(
    cache, forecast_result=None, mc_result=None, ...
) -> OHLCPredictionResult:
    
    # Close: median of MC terminal values (or forecast if MC unavailable)
    for horizon_key, n_days in horizons.items():
        terminal = mc_result.terminal_values.get(horizon_key)
        if terminal is not None and len(terminal) > 0:
            close_p50 = last_close * float(np.median(terminal))
            close_p10 = last_close * float(np.percentile(terminal, 10))
            close_p90 = last_close * float(np.percentile(terminal, 90))
        else:
            # Fallback to forecast_result point forecast
            close_p50 = forecast_close
            
        # High/Low: derived from return distribution path statistics
        # Parkinson (1980): E[H-L] = sigma * sqrt(8/pi) * sqrt(T)
        sigma = mc_distribution.std
        expected_range = sigma * np.sqrt(8 / np.pi) * np.sqrt(n_days)
        
        # Garman-Klass intraday factor (from adaptive_model_params)
        gk_factor = adaptive_params.intraday_low_factor
        
        high_estimate = close_p50 * (1 + expected_range * 0.6)
        low_estimate = close_p50 * (1 - expected_range * gk_factor)
        
        # Open: close of previous candle (for multi-day) or last actual close
        open_estimate = prev_candle_close
```

**Phase 2: Generate per-day candles from MC path statistics**

For multi-day horizons (week/month/year), instead of iterating day-by-day with random noise:

```python
# Use MC path quantiles at each time step
# mc_result already has full (n_paths, n_steps) arrays in terminal_values
# Extract per-step statistics instead of simulating new paths

for day_i in range(n_days):
    # Slice MC paths at step day_i
    step_returns = mc_paths[:, day_i]  # all 10K paths at this step
    step_prices = last_close * np.exp(np.cumsum(mc_paths[:, :day_i+1], axis=1)[:, -1])
    
    candle_close = float(np.median(step_prices))
    candle_high = float(np.percentile(step_prices, 95))
    candle_low = float(np.percentile(step_prices, 5))
```

**Phase 3: Fallback for no-MC case**

When MC hasn't run (--skip-models), keep a simplified version of the current random walk but with proper scaling:
- Use forecast_result for drift
- Use `volatility_21d` for noise (no survival_mult, no growth factor)
- Cap year-end at +/-50% of current price (sanity bound)

---

## Validation criteria

**Problem 3 (FH scoring):**
1. Apple FH composite > 50 (currently 28.97)
2. JPMorgan FH composite > 45 (banks should not score as "distressed" for high leverage)
3. Tesla FH composite in 35-55 range (volatile but growing)
4. A company actually in distress (e.g., Bed Bath & Beyond 2022) scores < 25

**Problem 4 (OHLC predictor):**
1. OHLC year-end close within +/-30% of current price for S&P 500 companies
2. OHLC year-end close is the median of MC terminal values (verifiable)
3. No single-path random walk code remains in ohlc_predictor.py
4. High > Close > Low for all candles (consistency)

---

## Files changed (estimated)

| File | Change | Lines |
|------|--------|-------|
| `config/sector_reference_ranges.yml` | NEW | ~100 |
| `operator1/models/financial_health.py` | Replace _normalize_series with cross-sectional | ~80 |
| `operator1/models/ohlc_predictor.py` | Rewrite to use MC distributions | ~120 |
| `operator1/stages/stage6_ensemble.py` | Update OHLC call to pass MC paths | ~10 |
| **Total** | | ~310 net |

# Multi-Scope Multi-Frequency Forecasting Architecture

## Concept

Run 5 separate pipeline instances at different time frequencies, each with its own optimal lookback window. Each pipeline produces independent survival analysis, regime detection, and forecasts. A final cross-frequency fusion layer reconciles all 5 into a single coherent prediction set.

```
Daily Pipeline   -- 2yr lookback, ~504 points  -- captures short-term momentum, volatility, intraday patterns
Weekly Pipeline  -- 3yr lookback, ~156 points  -- captures medium-term trends, earnings cycles
Monthly Pipeline -- 5yr lookback, ~60 points   -- captures macro cycles, sector rotation
Quarterly Pipe   -- 6yr lookback, ~24 points   -- captures business cycle turns, structural shifts
Annual Pipeline  -- 8yr lookback, ~8 points    -- captures secular trends, decade-scale regime changes
```

Each pipeline resamples OHLCV and financial data to its target frequency before running the full analytical stack.

---

## Architecture

```
                    [Raw Data Fetch -- max lookback 8yr]
                         |
            [Cache Build -- daily spine, full history]
                         |
      [1. Annual Pipeline -- 8yr, ~8 points]
      Secular trends, competitive moat, decade-scale regimes
                         |
      [2. Quarterly Pipeline -- 6yr, ~24 points]
      Inherits: annual regime context, secular trend direction
                         |
      [3. Monthly Pipeline -- 5yr, ~60 points]
      Inherits: quarterly earnings trajectory, annual growth trend
                         |
      [4. Weekly Pipeline -- 3yr, ~156 points]
      Inherits: monthly macro regime, quarterly fundamentals
                         |
      [5. Daily Pipeline -- 2yr, ~504 points]
      Inherits: weekly trend, monthly regime, quarterly survival
                         |
              [Cross-Frequency Fusion]
              All 5 results reconciled into final output
                         |
              [Unified Multi-Scale Profile]
              [Report with frequency breakdown]
```

**Key design: sequential execution with cascading context.**

Each pipeline runs after the previous one completes, and receives the prior
pipeline's insights as additional input context. This means:

- The daily pipeline knows the annual secular trend before it starts
- The weekly pipeline knows the quarterly earnings trajectory
- The monthly pipeline knows the long-term competitive moat assessment
- Context flows from slow to fast: annual -> quarterly -> monthly -> weekly -> daily

This is more powerful than running them independently because longer-horizon
insights constrain shorter-horizon predictions, preventing the daily pipeline
from making predictions that contradict the quarterly fundamental reality.

---

## Frequency-Specific Design

### 1. Daily Pipeline -- 2 years / ~504 business days

This is the current pipeline. No changes needed except labeling its outputs as "daily_scope".

**Best at:** Short-term momentum, intraday patterns, candlestick formations, volatility clustering, news sentiment impact, institutional flow detection.

**Models that excel at daily:** GARCH, LSTM, candlestick patterns, DTW analogs, PID controller, online change detection.

**Prediction horizons:** 1d, 5d, 21d.

### 2. Weekly Pipeline -- 3 years / ~156 weeks

Resample daily cache to weekly OHLCW (open=Monday open, high=week high, low=week low, close=Friday close, volume=week sum). Financial statements stay as-is (already periodic).

**Best at:** Earnings cycle patterns, medium-term trend identification, sector rotation signals, weekly momentum.

**Models that excel at weekly:** Kalman filter (smoother signal), VAR (captures inter-variable dynamics over weeks), Granger causality (more statistically powerful with weekly aggregation), cycle decomposition (cleaner periodic signals).

**Prediction horizons:** 1w, 4w, 13w (1 quarter).

### 3. Monthly Pipeline -- 5 years / ~60 months

Resample to monthly OHLCV. Financial statements: use as-reported (quarterly/annual periods map naturally to monthly).

**Best at:** Macro cycle alignment, inflation impact on real returns, long-term valuation mean reversion, sector-level competitive dynamics.

**Models that excel at monthly:** VAR with macro variables (GDP, inflation, rates align naturally to monthly), copula (crisis co-movement over months), adaptive thresholds (5yr of peer data gives robust percentiles).

**Prediction horizons:** 1m, 3m, 6m, 12m.

### 4. Quarterly Pipeline -- 6 years / ~24 quarters

Resample to quarterly. This naturally aligns with financial statement reporting periods.

**Best at:** Earnings trajectory, fundamental regime classification, balance sheet deterioration trends, cash flow sustainability.

**Models that excel at quarterly:** Financial health scoring (no interpolation noise), Altman Z trend, Beneish M trend, survival probability (filing-aligned, no forward-fill distortion), Sobol sensitivity (which fundamentals actually drive outcomes over quarters).

**Prediction horizons:** 1Q, 2Q, 4Q (1 year).

### 5. Annual Pipeline -- 8 years / ~8 annual observations

Resample to annual. Limited data points but captures structural trends.

**Best at:** Secular growth trends, competitive moat durability, capital allocation quality over full cycles, decade-scale regime identification (GFC recovery, COVID, rate hiking cycle).

**Models that work with 8 points:** Simple trend extrapolation, growth rate persistence, mean reversion to historical norms, Bayesian priors from sector/industry medians (compensates for small N).

**Prediction horizons:** 1Y, 2Y.

---

## Cross-Frequency Fusion Layer

The fusion layer reconciles predictions from all 5 frequencies into a single coherent output.

### Fusion Method: Inverse-Variance Weighted Reconciliation

For each prediction horizon, find which frequencies contribute:

| Horizon | Contributing Frequencies | Primary Weight |
|---------|------------------------|---------------|
| 1 day   | Daily only             | 100% daily    |
| 1 week  | Daily + Weekly         | 60% daily, 40% weekly |
| 1 month | Daily + Weekly + Monthly | 30/30/40    |
| 1 quarter | Weekly + Monthly + Quarterly | 20/30/50 |
| 1 year  | Monthly + Quarterly + Annual | 20/30/50  |
| 2 years | Quarterly + Annual     | 40/60        |

Weights are then adjusted by each frequency's walk-forward accuracy (inverse-variance weighting). A frequency that predicted well gets more weight.

### Regime Consensus

Each frequency detects regimes independently. The fusion layer builds a consensus:

- If 4/5 frequencies agree on "bull" -- high confidence bull
- If daily says "bear" but weekly/monthly say "bull" -- short-term correction in a longer uptrend
- If monthly/quarterly say "bear" but daily says "bull" -- bear market rally (dangerous)

This multi-scale regime view is extremely valuable -- it distinguishes between short-term noise and structural regime changes.

### Survival Probability Fusion

Each frequency computes its own survival probability. Fusion:

```
P_survival_fused = weighted_harmonic_mean across frequencies
```

Harmonic mean is used instead of arithmetic because survival is a "weakest link" problem -- if ANY frequency shows high distress probability, that matters more than others showing safety.

### Confidence Aggregation

Prediction confidence combines:
- Walk-forward accuracy per frequency
- Regime agreement across frequencies
- Data density (daily has 504 points, annual has 8 -- daily gets higher base confidence for short horizons)
- Conformal interval width (narrower = more confident)

---

## Implementation Plan

### Phase 1: Multi-Frequency Cache Resampler

New module: `operator1/features/frequency_resampler.py`

```python
def resample_cache_to_frequency(
    daily_cache: pd.DataFrame,
    frequency: str,  # 'W', 'M', 'Q', 'A'
    lookback_years: int,
) -> pd.DataFrame:
    # OHLCV resampling rules
    # Financial statement alignment
    # Derived variable recomputation
```

### Phase 2: Sequential Pipeline Runner with Cascading Context

New module: `operator1/steps/multi_frequency_runner.py`

```python
def run_multi_frequency_pipeline(
    market_id: str,
    company: str,
    secrets: dict,
    frequencies: list[str] = ['A', 'Q', 'M', 'W', 'D'],  # slow-to-fast order
) -> dict[str, PipelineResult]:
    # Sequential execution: slow frequencies first, fast frequencies last.
    # Each pipeline receives the previous pipeline's output as context.
    #
    # 1. Run annual pipeline (8yr) -- produces secular_trend, moat_assessment
    # 2. Run quarterly pipeline (6yr) -- receives annual context, adds earnings_trajectory
    # 3. Run monthly pipeline (5yr) -- receives quarterly context, adds macro_regime
    # 4. Run weekly pipeline (3yr) -- receives monthly context, adds medium_term_trend
    # 5. Run daily pipeline (2yr) -- receives all prior context, full model suite
    #
    # Context propagation:
    #   prior_context = {
    #       'regime_from_slower': prior_result.regime_labels,
    #       'survival_from_slower': prior_result.survival_probability,
    #       'trend_direction': prior_result.trend_direction,
    #       'forecast_bounds': prior_result.forecast_bounds,  # constrains faster predictions
    #   }
```

### Phase 3: Cross-Frequency Fusion

New module: `operator1/models/frequency_fusion.py`

```python
def fuse_multi_frequency_results(
    results: dict[str, PipelineResult],
) -> FusedResult:
    # 1. Reconcile predictions across frequencies
    # 2. Build regime consensus
    # 3. Fuse survival probabilities
    # 4. Aggregate confidence scores
    # 5. Generate multi-scale narrative
```

### Phase 4: Profile and Report Integration

- Profile gets a `multi_frequency` section with per-frequency summaries
- Report gets a "Multi-Scale Analysis" section showing frequency agreement/disagreement
- Charts show multi-frequency overlay (daily + weekly + monthly trendlines)

---

## No-Look-Ahead Guarantees

The multi-frequency architecture introduces new look-ahead risk vectors.
Every one must be explicitly guarded.

### Risk 1: Resampling look-ahead

When resampling daily to weekly/monthly, the last period may be incomplete.
Example: if today is Wednesday, the current week's OHLCV should only use
Mon-Wed data, not the full Mon-Fri week.

**Guard:** `resample_cache_to_frequency()` must truncate the last period to
only include data up to and including today. Incomplete periods are marked
with `is_partial_period = True`.

### Risk 2: Financial statement look-ahead in longer lookbacks

The 8-year annual pipeline fetches data from 8 years ago. Financial
statements from that period must use their original `filing_date`, not
`report_date`. A Q4 2017 earnings report filed in February 2018 must not
appear in the December 2017 row.

**Guard:** All frequencies use the same PIT alignment rule as daily:
`filing_date <= period_end_date` (not `report_date <= period_end_date`).
The `--pit-mode filing_date` flag applies to all frequencies.

### Risk 3: Cascading context look-ahead

When the annual pipeline passes regime context to the quarterly pipeline,
the annual regime labels were fitted on the full 8-year history. If the
quarterly pipeline uses these labels as features in its forward pass, it
has future information from the annual pipeline's full-history HMM fit.

**Guard:** Cascading context must be limited to:
- `trend_direction` (single value: up/down/flat, from historical data only)
- `secular_regime` (single label, not a time series)
- `survival_probability_latest` (single value, latest point only)
- `forecast_bounds` (min/max constraints derived from historical range)

Do NOT pass: time-series regime labels, daily/weekly regime probabilities,
or any per-period prediction from a slower frequency. Only pass summary
statistics that represent the current state, not historical time series.

### Risk 4: Macro data look-ahead in longer windows

World Bank/FRED macro data for 8 years ago may include revisions that
were not available at the time. GDP figures are revised months/years later.

**Guard:** Use vintage (first-release) data when available. When not
available, flag macro values older than 2 years with
`is_potentially_revised = True` and give them lower confidence weight.

### Risk 5: Forward-fill across frequency boundaries

When resampling financial statements to monthly frequency, forward-filling
a quarterly filing across 3 months is correct. But forward-filling an
annual filing across 12 months gives each month the same value, creating
artificial smoothness that masks actual quarterly variation.

**Guard:** The `frequency_interpolator` already handles stock vs flow
variable types. For lower frequencies, use the same interpolation logic
but with wider filing gaps. The interpolation confidence will naturally
be lower for annual filings (confidence decays with distance from filing).

---

## Compute Cost Analysis

| Frequency | Cache Size | Models to Run | Estimated Time |
|-----------|-----------|---------------|---------------|
| Daily     | 504 rows  | All 50 modules | ~15-30 min |
| Weekly    | 156 rows  | All 50 modules | ~5-10 min |
| Monthly   | 60 rows   | 40 modules (skip LSTM, pattern, DTW) | ~3-5 min |
| Quarterly | 24 rows   | 25 modules (fundamentals only) | ~1-2 min |
| Annual    | 8 rows    | 10 modules (trend + baseline only) | ~30 sec |
| **Total** | | | **~25-50 min** |

The 5x figure is generous -- lower frequencies run much faster due to fewer data points and simpler model subsets. Real-world total is closer to 2-3x the daily-only time.

---

## What Makes This Powerful

1. **Noise filtering:** Daily data is noisy. Weekly/monthly smooth it out. You see the actual trend without daily noise.

2. **Regime disambiguation:** Is a 10% drop in 2 weeks a crash or a buying opportunity? Daily says "crash." Monthly says "normal pullback in a 3-year uptrend." Quarterly says "still in growth phase." That context changes everything.

3. **Survival probability grounding:** A company's quarterly cash flow tells you more about survival than daily stock price movements. The quarterly pipeline's survival analysis is fundamentally more reliable.

4. **Forecast horizon matching:** A 1-year prediction from a daily pipeline (iterating 252 steps, error compounding each day) is inherently worse than a 1-year prediction from an annual pipeline (single step, 8 data points). Each frequency forecasts best at its natural horizon.

5. **Earnings pattern detection:** Quarterly data naturally aligns with earnings reports. The quarterly pipeline can detect earnings trajectory changes that are invisible in daily noise.

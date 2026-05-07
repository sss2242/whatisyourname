# Frequency-First Pipeline Architecture

## Core Principle

Each frequency runs its OWN complete pipeline with variables and ratios at their native scale. No mixing interpolated daily values with quarterly formulas.

## Current Architecture (BROKEN)

```
Wrappers download Q/A filings
    |
    v
Interpolate ALL flow variables to daily rates    <-- THIS IS THE PROBLEM
    |
    v
Compute ALL ratios on daily cache                <-- Ratios designed for Q/A break
    |
    v
Run temporal models on daily cache
    |
    v
Stage 7.4: Multi-frequency pipeline (runs again at A/Q/M/W/D)  <-- Too late, damage done
```

## New Architecture

```
Wrappers download source filings
    |
    v
Label each filing: A, Q, or S (detect from filing dates)
    |
    v
Check which freqs have native data: {Q: yes, A: yes, S: no}
    |
    v
Interpolate ONLY for missing freqs:
  - W: always interpolated from Q/A (no wrapper provides weekly filings)
  - M: always interpolated from Q/A (no wrapper provides monthly filings)
  - D: interpolated from Q/A for time-series models ONLY (OHLCV is native daily)
  - Q: use raw quarterly data if available, else interpolate from A
  - A: use raw annual data if available, else aggregate from Q
  - S: use raw semi-annual if available, else interpolate from A or aggregate from Q
    |
    v
Each freq runs its OWN full pipeline:
  A pipeline: derived_variables(A_cache) -> survival(A) -> FH(A) -> regime(A) -> forecast(A) -> MC(A)
  Q pipeline: derived_variables(Q_cache) -> survival(Q) -> FH(Q) -> regime(Q) -> forecast(Q) -> MC(Q)
  M pipeline: derived_variables(M_cache) -> survival(M) -> FH(M) -> regime(M) -> forecast(M) -> MC(M)
  W pipeline: derived_variables(W_cache) -> survival(W) -> FH(W) -> regime(W) -> forecast(W) -> MC(W)
  D pipeline: derived_variables(D_cache) -> survival(D) -> regime(D) -> forecast(D) -> MC(D)
             NOTE: D pipeline uses ONLY daily-safe variables (41 from scan)
                   D does NOT compute PE, ROA, gross_margin, etc.
    |
    v
Resemble: collect results from all freq pipelines
    |
    v
Fuse: 13-method frequency fusion (already exists in Stage 7.4.6)
    |
    v
HF Analysis: runs on fused results + raw Q/A statement DataFrames
```

## Implementation Plan

### Phase 1: Source Labeling (backtest_runner.py, main.py)

After downloading each statement, detect and label frequency:

```python
from operator1.estimation.frequency_interpolator import detect_frequency_from_dates

# After fetching income/balance/cashflow
for label, stmt_df in statements:
    dates = pd.to_datetime(stmt_df[date_col]).dropna().sort_values()
    freq = detect_frequency_from_dates(list(dates))
    # freq = "quarterly" / "semiannual" / "annual" / "unknown"
    state.source_frequencies[label] = freq
    logger.info(f"{label}: {freq} ({len(dates)} filings)")
```

Store in PipelineState:
```python
source_frequencies: dict = {}  # {"income": "quarterly", "balance": "quarterly", "cashflow": "quarterly"}
native_freqs: set = set()       # {"Q", "A"} -- which freqs have native data
```

### Phase 2: Build Per-Frequency Caches (new module or extend frequency_resampler.py)

For each native frequency, build a cache at that frequency using raw filing values:

```python
# Q cache: one row per quarter, values are quarterly totals (flow) or end-of-quarter snapshots (stock)
Q_cache = build_native_freq_cache(income_df, balance_df, cashflow_df, freq="Q", ohlcv=quotes_df)

# A cache: one row per year, values are annual totals (flow) or year-end snapshots (stock)
A_cache = build_native_freq_cache(income_df, balance_df, cashflow_df, freq="A", ohlcv=quotes_df)

# D cache: OHLCV spine (daily) + stock variables forward-filled + flow variables NOT merged
# (flow variables live only in Q/A caches, not in D)
D_cache = build_daily_ohlcv_cache(quotes_df, balance_df)  # stock vars only
```

`build_cache_from_raw_filings()` already exists in [`frequency_resampler.py`](operator1/features/frequency_resampler.py) -- it constructs Q/A caches directly from raw statement DataFrames without going through daily forward-fill. This is already the correct approach. It just needs to be the PRIMARY path, not a secondary path used only by Stage 7.4.

### Phase 3: Per-Frequency Pipeline Runs

Each frequency runs the full pipeline independently. This is already implemented in Stage 7.4 sub-stages (7.4.1-7.4.5). The change is to make this the PRIMARY execution path, not an afterthought.

**What each frequency pipeline computes:**

| Variable Group | D | W | M | Q | A |
|----------------|---|---|---|---|---|
| Returns, vol, drawdown | Native | Resampled | Resampled | Resampled | Resampled |
| Technical indicators | Native | Resampled | Skip | Skip | Skip |
| Balance sheet ratios (current_ratio, D/E) | FFill from Q | FFill from Q | FFill from Q | Native | Native |
| Flow/Stock ratios (PE, ROA, EV/EBITDA) | **SKIP** | **SKIP** | Interpolated | **Native** | **Native** |
| Flow/Flow ratios (margins) | **SKIP** | **SKIP** | Interpolated | **Native** | **Native** |
| TTM values | **SKIP** | **SKIP** | Interpolated | Sum 4Q | = Annual |
| Survival mode | D-safe triggers only | Same | Full | Full | Full |
| Financial Health | T1-T3 only | Same | Full | Full | Full |
| HMM regime detection | Native | Resampled | Resampled | Resampled | Resampled |
| Forecasting | Full | Full | Full | Full | Full |
| Monte Carlo | D-safe triggers | Same | Full triggers | Full triggers | Full triggers |

The D pipeline skips all 18 broken ratios and uses only the 41 daily-safe + 6 stock/stock variables. PE, ROA, gross_margin, etc. come from the Q and A pipeline results.

### Phase 4: Resemble + Fuse

After all frequency pipelines complete:

1. **Resemble:** Collect results from all pipelines into a `MultiFrequencyResult`
2. **Fuse:** Use the existing 13-method fusion in [`frequency_fusion.py`](operator1/models/frequency_fusion.py):
   - Regime consensus (majority vote across freqs)
   - Survival probability (harmonic mean -- weakest link)
   - Prediction reconciliation (inverse-variance weighted)
   - Cross-frequency momentum signal
3. **Forward-fill Q/A ratios into daily timeline:** After fusion, the Q/A-computed PE, ROA, gross_margin etc. get forward-filled onto the daily timeline for the profile and report. These are now CORRECT values at their native frequency, forward-filled to daily for display only.

### Phase 5: HF Analysis

HF engine already uses raw statement DataFrames (`income_df`, `balance_df`, `cashflow_df`) for most calculations. The fix is to route the few cache-dependent values (PE, fcf_yield, market_cap) from the Q pipeline results instead of the broken daily cache.

## What Changes vs Current Code

### Files that change:

| File | Change | Effort |
|------|--------|--------|
| `backtest_runner.py` | Add source frequency labeling after data fetch | Small |
| `main.py` | Same source labeling | Small |
| `pipeline_state.py` | Add `source_frequencies`, `native_freqs` fields | Tiny |
| `stages/runner.py` | Reorder: run per-freq pipelines BEFORE daily pipeline | Medium |
| `stages/stage7_integration.py` | Move 7.4 sub-stages to be PRIMARY path (run earlier) | Medium |
| `features/derived_variables.py` | Add `freq` parameter; skip broken ratios when freq="D" | Medium |
| `analysis/survival_mode.py` | Accept freq parameter; use only D-safe triggers on daily | Small |
| `models/financial_health.py` | Accept freq parameter; skip T5 on daily | Small |

### Files that DON'T change:
- `frequency_resampler.py` -- already has `build_cache_from_raw_filings()` 
- `frequency_fusion.py` -- already has 13-method fusion
- `multi_frequency_runner.py` -- already runs per-freq pipeline
- All temporal model files -- they don't care about frequency
- HF engine -- already uses raw DFs

## Stage Execution Order (New)

```
Stage 1: Data Fetch (unchanged)
  1.1  Profile + search
  1.2  Financial statements (income, balance, cashflow)
  1.3  OHLCV + holders + segments  
  1.4  Source frequency labeling (NEW: detect Q/A/S from filing dates)
  1.5  Build per-frequency caches: Q_cache, A_cache, D_cache (MOVED from 7.4.0)

Stage 2: Per-Frequency Pipeline Runs (REORDERED -- was Stage 7.4)
  2.A   Annual pipeline (full: derived vars + survival + FH + regime + forecast + MC)
  2.Q   Quarterly pipeline (full)
  2.M   Monthly pipeline (interpolated from Q/A)
  2.W   Weekly pipeline (interpolated from Q/A)
  2.D   Daily pipeline (OHLCV + stock vars + D-safe features ONLY)

Stage 3: Fusion + Integration
  3.1   Resemble per-freq results
  3.2   13-method frequency fusion
  3.3   Forward-fill Q/A ratios into daily timeline (for profile/report display)

Stage 4: Extended Models (on daily cache with correct Q/A ratios)
  4.1   Forecasting (on D cache with fused context)
  4.2   Forward pass, burn-out, walk-forward
  4.3   Monte Carlo (with correct survival triggers from Q pipeline)
  4.4   Ensemble models (transformer, particle filter, conformal, DTW, etc.)
  4.5   Prediction aggregation

Stage 5: HF Analysis + Profile + Report
  5.1   Hedge Fund (uses raw Q/A DFs + fused results)
  5.2   Profile build
  5.3   Report generation
```

## Expected Results After Implementation

| Metric | Current (BROKEN) | After Fix |
|--------|-----------------|-----------|
| PE | 178.87 | ~40 |
| EV/EBITDA | 89.97 | ~25 |
| P/S | 12,639 | ~9.7 |
| gross_margin | 0.775 | 0.469 |
| fcf_yield | 0.04% | 2.86% |
| roa | 0.03% | 28.3% |
| Survival mode | 80% of days | ~0% (correct for AAPL) |
| MC survival | 47% | 97%+ |
| DCF intrinsic | $32 | ~$200-280 |
| HF grade | C | B+/A |
| 1-day close error | 3.2% | TBD (should improve with PE anchor) |

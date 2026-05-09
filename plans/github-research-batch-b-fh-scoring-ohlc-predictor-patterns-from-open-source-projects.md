# GitHub Research: FH Scoring Redesign + OHLC Predictor Patterns from Open-Source Projects

*Research for Batch B: FH Scoring Redesign + OHLC Predictor Replacement (Problems 3, 4)*
*Date: 2026-05-09*

Research across 7 open-source projects to identify implementation patterns for cross-sectional factor scoring, sector reference ranges, MC-derived price predictions, and barrier option mathematics for OHLC estimation.

---

## Batch B has 2 components:

1. **Cross-sectional FH scoring** -- replace self-referential percentile rank with sector reference ranges (Problem 3: Apple FH=28.97 "Weak")
2. **MC-derived OHLC predictor** -- replace single-seed random walk with MC terminal distributions + Parkinson/GK volatility (Problem 4: OHLC year-end $78 crash)

---

## Project 1: Quantopian alphalens (4,260 stars)

**Repo:** `quantopian/alphalens`
**Relevance:** The standard for cross-sectional factor analysis. Their `quantize_factor` and `demean_forward_returns` functions show exactly how to score a value relative to a cross-section.

### Key Patterns Found

**Pattern 1: Period-wise Cross-Sectional Quantile Bucketing**

alphalens' `quantize_factor()` computes quantile buckets across the entire universe at each time period. A company's factor value is compared to ALL other companies in that period, not to its own history:

```python
def quantize_factor(factor_data, quantiles=5, by_group=False):
    """
    Computes period wise factor quantiles.
    by_group: If True, compute quantiles within each group (sector).
    """
    grouper = [factor_data.index.get_level_values('date')]
    if by_group:
        grouper.append('group')
    
    factor_quantile = factor_data.groupby(grouper)['factor'].transform(
        lambda x: pd.qcut(x, quantiles, labels=False, duplicates='drop') + 1
    )
```

**What we can adopt:** Our FH scoring should use `by_group=True` (sector-level quantiles). Apple's current_ratio=0.92 should be scored against other technology companies, not against all sectors combined or its own history.

When live peer data is available (from `peer_ranking.py`), use it as the cross-section. When unavailable, use static sector reference ranges from a YAML config (Damodaran-sourced).

**Pattern 2: Group-Adjusted Demeaning**

alphalens' `demean_forward_returns()` subtracts the group (sector) mean from each company's return. This converts absolute values to relative: "AAPL's return is +0.1% but tech sector average is +0.5%, so AAPL's group-adjusted return is -0.4%."

**What we can adopt:** Each FH tier score should be expressed as deviation from sector median, not absolute percentile:

```python
# Current (self-referential):
score = expanding_percentile_rank(current_ratio_history)  # 50th of own history

# Proposed (cross-sectional):
score = sector_percentile(current_ratio, sector_reference_range)  # vs sector peers
# Apple's 0.92 vs tech range [0.7, 0.9, 1.2, 1.8, 2.5] -> between p25 and median -> score ~40
```

---

## Project 2: OpenBB (67,265 stars)

**Repo:** `OpenBB-finance/OpenBB`
**Relevance:** Financial data platform with standardized ratio models. Their `FinancialRatiosData` Pydantic model shows how to structure sector-specific ratio benchmarks.

### Key Patterns Found

**Pattern 3: Standardized Financial Ratio Data Models**

OpenBB defines `FinancialRatiosData` with typed fields for every standard ratio (current_ratio, debt_to_equity, gross_margin, etc.). Each ratio has a standard name and description. Multiple data providers (FMP, Intrinio, Polygon) implement the same model.

**What we can adopt:** Our `sector_reference_ranges.yml` should use the same canonical ratio names that OpenBB standardizes. This makes the config self-documenting and prevents name mismatches between the reference ranges and `financial_health.py`.

**Pattern 4: Provider-Agnostic Scoring**

OpenBB's architecture separates data fetching from data scoring. The scoring logic doesn't know where the data came from -- it operates on the standardized model.

**What we can adopt:** The `_cross_sectional_score()` function should accept a generic `(value, reference_range)` tuple, not care whether the reference came from live peers, static YAML, or a third-party API. This makes it easy to swap between sources.

---

## Project 3: finviz (1,271 stars)

**Repo:** `mariostoev/finviz`
**Relevance:** Financial screener that provides sector-level ratio statistics. Their filter system shows how to express financial health criteria as sector-relative conditions.

### Key Patterns Found

**Pattern 5: Sector-Filtered Screening**

finviz allows filtering stocks by sector + financial metric combinations. The filters express relative criteria: "technology stocks with current_ratio > sector P50" rather than absolute thresholds.

**What we can adopt:** Our FH scoring should think in terms of sector-relative screening. Instead of "score = percentile of own history," score = "where does this value fall in the sector distribution?" This is the fundamental shift from self-referential to cross-sectional.

**Pattern 6: Pre-Computed Sector Aggregates**

finviz pre-computes sector averages, medians, and percentile ranges for all financial metrics. These are updated periodically (weekly/monthly) and cached.

**What we can adopt:** Our `sector_reference_ranges.yml` is the static equivalent of finviz's pre-computed sector aggregates. It should contain [p10, p25, median, p75, p90] for each ratio per sector, sourced from Damodaran's annual sector data or S&P Capital IQ.

---

## Project 4: quantstats (7,084 stars)

**Repo:** `ranaroussi/quantstats`
**Relevance:** Portfolio analytics with benchmark-relative metrics. Shows how to compute relative scoring against a benchmark.

### Key Patterns Found

**Pattern 7: Benchmark-Relative Performance Metrics**

quantstats computes every metric relative to a benchmark: excess return, information ratio, tracking error. The key insight: **a metric without a benchmark is meaningless for comparison.**

**What we can adopt:** FH scoring without a benchmark (sector reference range) is meaningless for comparison -- that's exactly our bug. Apple FH=28 is "weak" because it's comparing Apple to Apple. The benchmark should be the sector median.

The hybrid scoring formula should be:
```python
final_score = 0.7 * cross_sectional_score + 0.3 * trend_score
```
Where `cross_sectional_score` anchors to the sector benchmark and `trend_score` captures own-history improvement/deterioration.

---

## Project 5: arch (1,516 stars)

**Repo:** `bashtage/arch`
**Relevance:** ARCH/GARCH volatility modeling library. Contains Parkinson and Garman-Klass volatility estimators that are critical for the OHLC predictor redesign.

### Key Patterns Found

**Pattern 8: Range-Based Volatility Estimators**

The `arch` library implements several range-based volatility estimators that use OHLC data:

- **Parkinson (1980):** Uses high-low range: `sigma = sqrt(1/(4n*ln2) * sum(ln(H/L)^2))`
- **Garman-Klass (1980):** Uses OHLC: `sigma = sqrt(0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2)`
- **Rogers-Satchell:** Drift-independent range estimator
- **Yang-Zhang (2000):** Combines overnight + open-close + range components

These estimators are 5-8x more efficient than close-to-close volatility, meaning they estimate the true volatility with fewer observations.

**What we can adopt:** Instead of generating random OHLC paths with `rng.normal()`, derive High/Low from the MC-estimated return distribution using Parkinson's relationship:

```python
# Expected daily range from volatility
expected_range = sigma * sqrt(8 / pi)  # Parkinson 1980

# High = Close * (1 + range_fraction * up_ratio)
# Low = Close * (1 - range_fraction * down_ratio)
# where up_ratio, down_ratio derived from return skewness
```

**Pattern 9: Multi-Estimator Consensus**

`arch` allows fitting multiple volatility models and comparing them. The best-fitting model is selected by AIC/BIC.

**What we can adopt:** For OHLC prediction, use the best available volatility estimate: Garman-Klass if OHLC history exists, Parkinson if only HL available, close-to-close as fallback. The `adaptive_model_params.intraday_low_factor` already computes GK -- use it directly in the OHLC predictor.

---

## Project 6: vectorbt (7,452 stars)

**Repo:** `polakowo/vectorbt`
**Relevance:** Backtesting engine with OHLCV simulation capabilities. Their `ohlcv_accessors.py` shows how to generate synthetic OHLC candles from return distributions.

### Key Patterns Found

**Pattern 10: OHLCV from Return Distributions**

vectorbt generates synthetic OHLC data for backtesting by:
1. Sampling returns from a distribution (normal, Student-t, or empirical)
2. Computing Close from cumulative returns
3. Deriving High/Low from intraday return range statistics
4. Setting Open = previous Close (with optional gap)

```python
# vectorbt approach:
close = last_close * np.cumprod(1 + sampled_returns)
# Range statistics from the sampled distribution:
high = close * (1 + abs_max_intraday_excursion)
low = close * (1 - abs_max_intraday_drawdown)
```

**What we can adopt:** This is exactly what our OHLC predictor should do, but using MC terminal distributions instead of fresh random samples:

```python
# Proposed: derive OHLC from existing MC paths
for horizon_key, n_days in horizons.items():
    terminal = mc_result.terminal_values.get(horizon_key)
    close_p50 = last_close * np.median(terminal)
    
    # H/L from distribution range statistics
    sigma = np.std(np.log(terminal))
    expected_range = sigma * np.sqrt(8 / np.pi)  # Parkinson
    high_est = close_p50 * (1 + expected_range * 0.6)
    low_est = close_p50 * (1 - expected_range * gk_factor)
```

**Pattern 11: Per-Step Path Statistics**

vectorbt extracts per-step statistics from simulation paths (not just terminal values). For each day d in the forecast horizon, it computes the distribution of prices at that day across all paths.

**What we can adopt:** For multi-day OHLC candles (week/month/year), extract per-day statistics from MC paths:

```python
# MC paths shape: (n_paths, n_steps)
for day_i in range(n_days):
    step_prices = last_close * np.exp(np.cumsum(mc_paths[:, :day_i+1], axis=1)[:, -1])
    candle_close = np.median(step_prices)
    candle_high = np.percentile(step_prices, 95)
    candle_low = np.percentile(step_prices, 5)
```

This requires MC to store per-step paths (not just terminal), which `mc_result.terminal_values` may or may not support. If not, use the analytical Parkinson formula as fallback.

---

## Project 7: QuantLib-SWIG (388 stars) + QuantLib (C++)

**Repo:** `lballabio/QuantLib-SWIG`
**Relevance:** The definitive quantitative finance library. Contains barrier option pricing that gives the exact probability distribution of path maximum and minimum.

### Key Patterns Found

**Pattern 12: Barrier Option Mathematics for Expected Path Extremes**

QuantLib's barrier option pricing gives the analytical distribution of the maximum (high) and minimum (low) of a Brownian motion path over a time interval. Given (mu, sigma, T):

```
E[max(S_t, 0 <= t <= T)] = S_0 * exp(mu*T) * N(d1) + S_0 * N(d2) * exp(mu*T)
```

where d1, d2 are functions of mu, sigma, T (Black-Scholes-Merton framework).

**What we can adopt:** For the OHLC predictor, instead of simulating paths to find H/L, use the analytical formulas for expected path maximum and minimum:

```python
# Analytical expected high/low (simplified Parkinson approximation)
# Full barrier formulas available in QuantLib for exact computation

mu = mc_regime_mean  # from regime distribution
sigma = mc_regime_std
T = n_days / 252  # time in years

# Expected maximum excursion (Parkinson 1980 simplified)
expected_high_pct = sigma * np.sqrt(T) * np.sqrt(8 / np.pi) * 0.6
# Expected minimum excursion
expected_low_pct = sigma * np.sqrt(T) * np.sqrt(8 / np.pi) * 0.4

high_est = close_p50 * (1 + expected_high_pct)
low_est = close_p50 * (1 - expected_low_pct)
```

The 0.6/0.4 split reflects the empirical observation that price paths tend to have larger upward excursions than downward (positive skewness). Adjust by `skewness_63d` from cache when available.

**Pattern 13: Return Skewness-Adjusted Range**

QuantLib's skew-adjusted option pricing shows that skewness shifts the expected high/low asymmetrically. Positive skewness -> higher expected high, lower expected low shift.

**What we can adopt:**
```python
skew = cache.get("skewness_63d", pd.Series(0)).iloc[-1]
up_ratio = 0.5 + 0.1 * max(-1, min(1, skew))    # 0.4-0.6
down_ratio = 0.5 - 0.1 * max(-1, min(1, skew))   # 0.4-0.6
high_est = close_p50 * (1 + expected_range * up_ratio)
low_est = close_p50 * (1 - expected_range * down_ratio)
```

---

## Synthesis: Recommended Implementation

### Problem 3: Cross-Sectional FH Scoring

**Architecture (combining Patterns 1-7):**

```python
def _cross_sectional_score(
    value: float,
    sector_range: list[float],  # [p10, p25, median, p75, p90]
    higher_is_better: bool = True,
) -> float:
    """Score 0-100 based on sector reference range (alphalens Pattern 1)."""
    breakpoints = sector_range
    if not higher_is_better:
        breakpoints = breakpoints[::-1]
    
    # Linear interpolation between breakpoints (dynaconf Pattern 6)
    # p10 -> score 10, p25 -> 25, median -> 50, p75 -> 75, p90 -> 90
    score_map = [10, 25, 50, 75, 90]
    for i in range(len(breakpoints) - 1):
        if breakpoints[i] <= value <= breakpoints[i + 1]:
            frac = (value - breakpoints[i]) / max(breakpoints[i+1] - breakpoints[i], 1e-9)
            return score_map[i] + frac * (score_map[i+1] - score_map[i])
    
    # Below p10 or above p90
    if value < breakpoints[0]:
        return max(0, 10 * value / max(breakpoints[0], 1e-9))
    return min(100, 90 + 10 * (value - breakpoints[-1]) / max(abs(breakpoints[-1]), 1e-9))


def _hybrid_score(
    value: float,
    sector_range: list[float],
    own_history_percentile: float,
    higher_is_better: bool = True,
    cross_weight: float = 0.7,
) -> float:
    """70% cross-sectional + 30% trend (quantstats Pattern 7)."""
    cross = _cross_sectional_score(value, sector_range, higher_is_better)
    trend = own_history_percentile * 100  # 0-100
    return cross_weight * cross + (1 - cross_weight) * trend
```

### Problem 4: MC-Derived OHLC Predictor

**Architecture (combining Patterns 8-13):**

```python
def predict_ohlc_from_mc(
    cache, mc_result, gk_factor, skewness=0.0,
) -> OHLCPredictionResult:
    """Derive OHLC from MC distributions (no random walk)."""
    last_close = float(cache["close"].dropna().iloc[-1])
    
    for horizon_key, n_days in HORIZONS.items():
        terminal = mc_result.terminal_values.get(horizon_key)
        if terminal is not None and len(terminal) > 100:
            # Close: MC median (vectorbt Pattern 10)
            close_p50 = last_close * float(np.median(terminal))
            
            # Range: Parkinson (arch Pattern 8)
            sigma = float(np.std(np.log(terminal)))
            expected_range = sigma * np.sqrt(8 / np.pi)
            
            # Skew-adjusted split (QuantLib Pattern 13)
            skew_adj = max(-1, min(1, skewness))
            up_ratio = 0.5 + 0.1 * skew_adj
            down_ratio = 0.5 - 0.1 * skew_adj
            
            high_est = close_p50 * (1 + expected_range * up_ratio)
            low_est = close_p50 * (1 - expected_range * down_ratio * gk_factor)
        else:
            # Fallback: forecast + bounded drift
            close_p50 = last_close * (1 + forecast_return)
            high_est = close_p50 * 1.02
            low_est = close_p50 * 0.98
```

### Priority Ranking

| Priority | Pattern | Source | Component |
|----------|---------|--------|-----------|
| P1 | Cross-sectional quantile scoring | alphalens | FH scoring |
| P1 | Sector reference ranges config | finviz, Damodaran | FH scoring |
| P1 | MC median for Close | vectorbt | OHLC predictor |
| P1 | Parkinson range for H/L | arch, QuantLib | OHLC predictor |
| P2 | Hybrid 70/30 cross+trend | quantstats | FH scoring |
| P2 | Skewness-adjusted range split | QuantLib | OHLC predictor |
| P2 | GK factor for Low | arch | OHLC predictor |
| P3 | Per-step MC path statistics | vectorbt | Multi-day OHLC |
| P3 | Benchmark-relative demeaning | alphalens | FH scoring |
| P3 | Provider-agnostic data model | OpenBB | Config structure |

### Estimated Implementation Size

| Component | Lines | Source Patterns |
|-----------|-------|-----------------|
| `config/sector_reference_ranges.yml` (new) | ~100 | finviz, Damodaran |
| `financial_health.py` (_cross_sectional_score) | ~80 | alphalens, quantstats |
| `ohlc_predictor.py` (rewrite) | ~120 | arch, vectorbt, QuantLib |
| `stage6_ensemble.py` (OHLC call update) | ~10 | Wiring |
| **Total** | **~310 net** | |

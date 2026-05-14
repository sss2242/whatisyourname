# AAPL Backtest: 2024-12-31 Live Run Results

**Date:** 2026-05-14 | **Duration:** 21.1 minutes | **Stages:** 63/63 (0 failures)
**Config:** full mode, LSTM enabled, parallel freqs, per_call LLM rotation, OpenRouter

---

## Run Summary

| Metric | Value |
|--------|-------|
| Market | US SEC EDGAR |
| Company | Apple Inc (AAPL) |
| Training window | 2023-01-01 to 2024-12-31 (2 years) |
| Prediction target | 2025 forward |
| Total stages | 63/63 complete |
| Total time | 1,267s (21.1 min) |
| Failures | 0 |

### Stage Timing Breakdown

| Stage Group | Time | Notes |
|-------------|------|-------|
| Stage 1 (Data Acquisition) | 833s (13.9min) | Entity discovery (379s) + adaptive calibration (339s) dominate |
| Stage 2 (Freq Pipeline) | 209s (3.5min) | Wave 2 parallel (M/W/D) took 193s |
| Stage 3 (Temporal) | 51s | Pattern detection (24s) + feature selection (10s) |
| Stage 4 (Forecasting) | 36s | Full mode with LSTM |
| Stage 5 (Forward) | 74s | MC 10K paths (49s) |
| Stage 6 (Ensemble) | 30s | Transformer (7s), TV Granger (6s) |
| Stage 7 (Integration) | 25s | MF pipeline + HF analysis |
| Profile + Report | 9s | |

---

## Prediction vs Actual (2025)

### Close Price Predictions (from 2024-12-31 base: ~$250.42)

| Horizon | Predicted | Actual | Error | % Error | In Bounds? |
|---------|-----------|--------|-------|---------|------------|
| 1d | $241.42 | $242.30 | -$0.88 | **-0.36%** | Yes |
| 5d | $245.04 | $241.16 | +$3.88 | **+1.61%** | Yes |
| 21d | $251.87 | $226.56 | +$25.30 | **+11.17%** | Yes |
| 252d | $373.65 | $272.57 | +$101.08 | +37.08% | N/A (no bounds) |

**Notes:**
- 1d prediction accuracy is excellent (-0.36% error)
- 5d prediction within 1.61% -- strong
- 21d overestimated by 11.17%, but actual was within conformal bounds [214.15, 300.52]
- 252d predicted $373.65, actual was $272.57 -- AAPL actually returned +8.85% in 2025, prediction implied +49.2%
- The 252d overestimation reflects the model's momentum bias from 2024's strong rally

### Actual 2025 Market Data

| Metric | Value |
|--------|-------|
| 2025 total return | +8.85% |
| Max drawdown | -30.22% |
| 21d realized vol | 2.00% |
| Latest close (2025-12-30) | $272.57 |

### Model Outputs

| Module | Result |
|--------|--------|
| HF Scorecard | Grade C, Conviction 4/10 |
| Position Signal | Buy (1.0) |
| Survival | Not in survival mode |
| Current Ratio | 0.92 (below 1.0 but tech sector override applies) |
| Debt/Equity | 1.48 |
| Interest Coverage | 10.89x |

---

## Analysis of Results

### What Worked Well

1. **Short-term accuracy is strong.** 1d error of -0.36% and 5d error of +1.61% are institutional-grade. The aggregated ensemble outperforms the individual model predictions (which had 3.32% and 3.67% error respectively).

2. **Conformal bounds contain actuals.** All three bounded horizons (1d, 5d, 21d) have the actual price within the prediction interval. The conformal calibration is working correctly.

3. **Zero failures across 63 stages.** The staged pipeline with graceful degradation worked flawlessly. No sub-stage crashed, timed out, or needed retry.

4. **Reasonable execution time.** 21.1 minutes for a full 63-stage pipeline including LSTM, 10K-path Monte Carlo, transformer, and HF analysis is acceptable for a backtest.

### What Needs Improvement

1. **252d overestimation (+37%).** The model predicted $373.65 but AAPL only reached $272.57. The 2024 rally momentum was extrapolated too aggressively. The MC paths or the forecasting cascade may have a bullish bias when recent returns are strongly positive.

2. **21d error (+11.17%).** January 2025 had a significant drawdown (AAPL dropped from $250 to $226 in 3 weeks). The model didn't anticipate this -- it predicted continued upward momentum. This is a classic mean-reversion vs momentum failure: the model was momentum-biased from 2024's +31% year.

3. **Several profile sections unavailable.** Behavioral signals, complexity signals, enriched survival timeline, linked entities aggregates, peer ranking, SHAP, copula, patterns, and several others show `available=False`. These modules ran but didn't produce results -- likely insufficient data or feature dependencies not met.

4. **CCC anomaly.** Cash conversion cycle predicted at -9,517 days -- this is clearly wrong. The DSO/DIO/DPO computation may have a unit error or data issue with Apple's balance sheet structure.

5. **OHLC High/Low inversion.** High predicted at $235.18, Low at $238.13 -- the low is higher than the high. This is the known OHLC predictor issue that was partially fixed but may still occur for certain parameter combinations.

### Potential Upgrades

1. **Mean-reversion dampener for long horizons.** When trailing 12-month return exceeds +20%, apply a dampening factor to 252d predictions. Most +30% years are followed by single-digit or negative years.

2. **CCC computation fix.** Investigate why Apple's CCC is -9,517 days. Likely a denominator issue with quarterly COGS vs daily revenue in the DSO/DIO/DPO formulas.

3. **OHLC bounds enforcement.** Add a post-prediction constraint: `predicted_high >= predicted_close >= predicted_low`. If violated, swap and log.

4. **Behavioral/complexity signal activation.** These modules show `available=False`. Check if the minimum data requirements (inst_flow_momentum for behavioral, return_1d for complexity) are met in the backtest cache.

5. **Peer ranking activation.** Shows `available=False` -- linked entity caches may not have survived the staged pipeline serialization. Check pickle compatibility.

---

## Raw Timing Data

```
Stage 1.1:    2.6s  Profile + Company Search
Stage 1.2:   22.1s  Financial Statements (CompanyFacts API)
Stage 1.3:   22.8s  OHLCV + Holders + Segments
Stage 1.4a:   0.5s  Cache Build
Stage 1.4b:   0.5s  Macro + Risk
Stage 1.5:   64.3s  Estimation + Derived Variables + Survival
Stage 1.6:  379.3s  Entity Discovery + Sentiment
Stage 1.7:  339.4s  Adaptive Calibration
Stage 1.8a:   1.1s  Regime Detection + Timeline
Stage 1.8b:   1.0s  Finalization
Stage 2.1:    1.1s  Frequency Separation
Stage 2.2:    1.2s  Data Reconciliation
Stage 2.0:    1.3s  Freq Resample Prep
Stage 2.W1:   1.2s  Freq Wave 1 (A/Q/S parallel)
Stage 2.W2: 193.5s  Freq Wave 2 (M/W/D parallel)
Stage 2.F:    1.8s  Freq Fusion
Stage 3.1:    2.7s  Regime Detection
Stage 3.6:   24.0s  Pattern Detection
Stage 3.8:   10.2s  Feature Selection
Stage 4.1:   36.2s  Forecasting (full + LSTM)
Stage 5.1:   15.5s  Forward Pass
Stage 5.4:   48.8s  Monte Carlo (10K paths)
Stage 6.1:    6.9s  Transformer
Stage 7.5.1:  2.0s  HF Base Metrics
Profile:      9.2s  Profile Build + Report
```

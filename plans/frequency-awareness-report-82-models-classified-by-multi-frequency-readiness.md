# Frequency Awareness Report: 82 Models Classified

*Debug scan of all analytical modules. Only 4/82 are frequency-aware. 73 are frequency-blind with hardcoded daily assumptions.*

## Annualization Factor Map

Every model that uses 252 (trading days/year) needs a frequency-aware factor:

| Frequency | Periods/Year | Annualization Factor | Rolling Window Divisor |
|-----------|-------------|---------------------|----------------------|
| D (Daily) | 252 | sqrt(252) for vol, *252 for returns | 1 |
| W (Weekly) | 52 | sqrt(52) | ~5 |
| M (Monthly) | 12 | sqrt(12) | ~21 |
| Q (Quarterly) | 4 | sqrt(4) = 2 | ~63 |
| S (Semi-Annual) | 2 | sqrt(2) | ~126 |
| A (Annual) | 1 | 1 | ~252 |

---

## Category 1: FREQ-AWARE (4 modules) -- Working Correctly

These modules accept a frequency parameter AND branch their logic based on it.

| Module | Freq Param | How It Adapts |
|--------|-----------|---------------|
| `derived_variables.py` | Via `_get_freq()` thread-local | Adjusts rolling windows, TTM computations, return periods. Some hardcoded 252 remains in beta/drawdown. |
| `frequency_resampler.py` | `frequency` param | Core resampler -- OHLCV agg rules, stock vs flow handling, period truncation all freq-specific. |
| `survival_mode.py` | `freq` param | Gates triggers by category: Q/A-only (fcf_yield, revenue decline, Altman Z), D/W-only (institutional, Merton DD, vol-of-vol). |
| `financial_health.py` | `freq` param | Accepts freq but only uses it for logging; actual tier scoring uses same formulas regardless. Needs real freq adaptation. |

---

## Category 2: FREQ-PARAM but No Branching (5 modules)

Accept frequency info but don't change calculations.

| Module | Freq Param | What Needs to Change |
|--------|-----------|---------------------|
| `six_derived_proxies.py` | `frequency` | Uses 252 for volatility annualization. At Q/A should use 4 or 1. |
| `adaptive_windows.py` | `filing_frequency` | Already freq-aware for window sizing. The 252 usage is for internal conversion -- OK. |
| `frequency_fusion.py` | `frequency` per result | Uses freq for weight matrix. The 252 is for annualizing fused predictions. Needs freq-specific annualization. |
| `engine.py` (HF) | `freq` in some calls | HF engine operates on raw Q DFs, not the cache freq. Mostly correct but DCF WACC and vol calculations use 252. |
| `frequency_interpolator.py` | `market_id` | Detects filing frequency from gap analysis. No annualization issue -- operates on raw values. |

---

## Category 3: FREQ-BLIND (73 modules) -- Need Fixes

### Priority 1: Models with Hardcoded 252 Annualization (HIGH IMPACT)

These produce WRONG values at non-daily frequencies because they multiply/divide by 252 when they should use the frequency-specific factor.

| Module | Hardcoded 252 Usage | Fix Needed |
|--------|-------------------|-----------|
| `monte_carlo.py` | `jump_lambda * 252` (annualize jump intensity), return mean/std scaling | Replace 252 with `PERIODS_PER_YEAR[freq]` lookup |
| `forecasting.py` | GARCH vol annualization, LSTM scaling, baseline EMA | Replace 252 with freq-aware factor |
| `prediction_aggregator.py` | Confidence band annualization, return forecast scaling | Replace 252 with freq-aware factor |
| `transformer_forecaster.py` | Feature scaling, loss normalization | Replace 252 with freq-aware factor |
| `copula.py` | Return annualization for joint crisis probability | Replace 252 with freq-aware factor |
| `ohlc_predictor.py` | High/low tail estimation uses daily return distribution | Should skip at non-daily freq (OHLC is daily concept) |
| `scenario_engine.py` | Cash runway = cash / daily_burn, return shift per day | Replace with period-appropriate burn rate |
| `vanity.py` | Vol calculations, trend windows | Replace 252 with freq-aware factor |
| `accruals_forensics.py` | Jones model regression uses annual-scaled variables | Should only run at Q/A; skip at D/W/M |
| `advanced_methods.py` | Merton vol, OU mean-reversion half-life, GARCH term structure | Replace 252 with freq-aware factor |
| `product_metrics.py` | Growth runway quarters uses 252 trading days | Replace with freq-appropriate period count |
| `filing_calendar.py` | Staleness threshold uses 252 for annual | Already handles this via market-specific thresholds -- OK |
| `adaptive_model_params.py` | Regime risk multiplier, Garman-Klass, MC path count | Replace 252 with freq-aware factor |
| `_frequency_classifier.py` | Period classification thresholds | Internal utility -- OK |

### Priority 2: Models with Hardcoded Rolling Windows (MEDIUM IMPACT)

These use fixed rolling window sizes (e.g., `.rolling(21)` for 1-month, `.rolling(63)` for 1-quarter) that are meaningless at non-daily frequencies.

| Module | Hardcoded Windows | Fix Needed |
|--------|-----------------|-----------|
| `derived_variables.py` | `.rolling(2)`, `.rolling(20)` for some calculations | Already partially freq-aware; remaining hardcoded windows need conversion |
| `behavioral_signals.py` | 52-week high/low uses `.rolling(252)` implicitly | Replace with `PERIODS_PER_YEAR[freq]` |
| `complexity_signals.py` | 21-day sample entropy, permutation entropy | Replace 21 with `PERIODS_PER_MONTH[freq]` |
| `feature_normalization.py` | 63d zscore, 252d percentile, 21d change | Replace all with freq-scaled equivalents |
| `institutional_flow.py` | EMA spans (21, 63) | Replace with freq-scaled periods |
| `regime_detector.py` | HMM min observations, PELT penalty | Scale min_obs by `PERIODS_PER_YEAR[freq]` |
| `walk_forward.py` | Min 10 observations | Scale by freq |
| `pattern_detector.py` | Candlestick patterns are daily OHLC | Should SKIP at non-daily frequencies |

### Priority 3: Models with Min Observation Requirements (LOW IMPACT)

These check `len(data) >= N` with hardcoded N. At quarterly frequency (8 rows), many models incorrectly refuse to run.

| Module | Min Obs | Quarterly Rows | Should Run? |
|--------|---------|---------------|------------|
| `monte_carlo.py` | 400 | 8 | No -- MC needs many paths but few obs is OK for distribution estimation |
| `granger_causality.py` | 50 | 8 | No -- PCMCI needs 50+ for lag testing. Skip at Q/A. |
| `walk_forward.py` | 10 | 8 | Marginal -- could work with 8 but barely |
| `genetic_optimizer.py` | 5 | 8 | Yes -- GA only needs enough for fitness evaluation |
| `forecasting.py` (LSTM) | 100 | 8 | No -- LSTM needs 100+. Falls back correctly to simpler models. |
| `forecasting.py` (VAR) | 50 | 8 | No -- VAR needs 50+. Falls back to AR(1). |
| `forecasting.py` (Tree) | 30 | 8 | No -- Tree needs 30+. Falls back to baseline. |

---

## Category 4: Models That Should SKIP at Certain Frequencies

| Module | Valid Frequencies | Reason |
|--------|-----------------|--------|
| `pattern_detector.py` | D only | Candlestick patterns are defined for daily OHLC candles |
| `ohlc_predictor.py` | D only | Predicts next-day open/high/low/close |
| `institutional_flow.py` | D, W only | Intraday/daily volume and holder data |
| `options_signals.py` | D only | Options surface is a daily snapshot |
| `cross_asset_signals.py` | D, W only | ETF/Treasury data is daily |
| `behavioral_signals.py` | D, W only | 52-week high, volume spikes are daily concepts |
| `news_sentiment.py` | D, W only | News articles have daily timestamps |
| `accruals_forensics.py` | Q, A only | Modified Jones model needs real Q/Q deltas |
| `earnings_smoothing.py` | Q, A only | Benford's law needs 30+ real filings |
| `fcf_quality.py` | Q, A only | OCF/NI ratio needs same-period values |

---

## Implementation: Frequency-Aware Constant Map

Add a single shared module that all models can import:

```python
# operator1/freq_constants.py

PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "Q": 4, "S": 2, "A": 1}
PERIODS_PER_QUARTER = {"D": 63, "W": 13, "M": 3, "Q": 1, "S": 0.5, "A": 0.25}
PERIODS_PER_MONTH = {"D": 21, "W": 4.3, "M": 1, "Q": 0.33, "S": 0.17, "A": 0.08}

ANNUALIZATION_FACTOR = {
    "D": 252, "W": 52, "M": 12, "Q": 4, "S": 2, "A": 1,
}
VOL_ANNUALIZATION = {
    "D": 252**0.5, "W": 52**0.5, "M": 12**0.5, "Q": 2.0, "S": 2**0.5, "A": 1.0,
}

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
```

Then each freq-blind model replaces `252` with:
```python
from operator1.freq_constants import ANNUALIZATION_FACTOR, _get_freq
_ann = ANNUALIZATION_FACTOR.get(_get_freq(), 252)
```

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| FREQ-AWARE (working) | 4 | None needed |
| FREQ-PARAM (no branching) | 5 | Add freq-conditional logic |
| FREQ-BLIND with 252 annualization | 14 | Replace 252 with freq-aware constant |
| FREQ-BLIND with hardcoded windows | 8 | Replace fixed windows with freq-scaled |
| FREQ-BLIND should skip at certain freqs | 10 | Add early-return guard |
| FREQ-BLIND (no freq-sensitive operations) | ~41 | No change needed (pure data transforms) |
| **Total modules** | **82** | **~37 need changes** |

# Fix Guide: 252d Momentum Bias -- Mean-Reversion Dampener

*Created: 2026-05-14 | Priority: Medium (model improvement, not a bug)*
*Source: AAPL backtest 2024-12-31 -- predicted +49.2%, actual +8.8%*

## Problem

After a strong year (AAPL +34.9% in 2024), the model extrapolates momentum into the 252d forecast, predicting +49.2% for the next year. Actual 2025 return was +8.8%. This is a well-documented phenomenon: trailing 12-month returns have near-zero correlation with forward 12-month returns for large-cap equities (DeBondt & Thaler 1985, Fama & French 1988).

The model has a partial dampener in `forecasting.py` line 3199-3206 that reduces regime-shift predictions near 52-week highs, but this only affects one component. The main 252d forecast from the model cascade (Kalman/Tree/Baseline) has no such correction.

## Where 252d Predictions Originate

The 252d close forecast flows through 5 modules:

```
1. forecasting.py (Stage 4.1)
   - Each model (Kalman, Tree, Baseline) produces a 252d point forecast
   - HORIZONS = {"1d": 1, "5d": 5, "21d": 21, "252d": 252}
   - Tree models fit on recent data, extrapolate recent trends
   - Baseline uses EMA which carries momentum forward

2. prediction_aggregator.py (Stage 6.5)
   - Inverse-RMSE weighted ensemble of all model 252d forecasts
   - No trailing-return awareness -- just weights + blend
   - This is where the dampener should be applied

3. ohlc_predictor.py (Stage 6.10)
   - Uses MC terminal values for 252d candle
   - MC paths carry regime drift (momentum-biased if current regime is bullish)

4. frequency_fusion.py (Stage 7.4.6)
   - 252d/1y: 50% Annual + 35% Quarterly + 15% Monthly weights
   - Annual frequency forecast may be less momentum-biased
   - But fusion doesn't explicitly dampen momentum

5. recursive_aggregator.py (Stage 6.11)
   - Day-by-day chaining for 252 steps
   - Compounds small daily momentum biases into large annual ones
```

## Existing Partial Dampening

`forecasting.py` line 3199-3210 already has:

```python
# Dampen regime shift near 52-week extremes
# and when Hurst exponent signals mean reversion
_regime_dampen = 1.0
if "anchoring_52w_high" in cache.columns:
    _ath_v = float(_ath.iloc[-1])
    if _ath_v > 0.90 and _rshift > 0:
        _regime_dampen = max(0.1, 1.0 - (_ath_v - 0.90) / 0.10)
if "hurst_exponent_rolling" in cache.columns:
    # H < 0.5 = mean-reverting: dampen momentum
    ...
```

This only affects the regime-shift adjustment, not the base model forecasts.

## Fix: Mean-Reversion Shrinkage in Prediction Aggregator

### Location

`operator1/models/prediction_aggregator.py` -- after the ensemble aggregation loop (around line 3330, before OHLC constraint).

### Logic

For horizons >= 21d, compute a shrinkage factor based on trailing 12-month return:

```python
# Mean-reversion shrinkage for long-horizon forecasts (DeBondt & Thaler 1985)
# When trailing 12m return is extreme (>20% or <-20%), shrink the forecasted
# return toward the historical mean (long-run equity premium ~8-10%/yr).
#
# Shrinkage formula:
#   trailing_12m = (close[-1] / close[-252]) - 1
#   excess = abs(trailing_12m) - 0.20  (only shrink if |return| > 20%)
#   shrinkage = min(0.5, excess * 2.0)  (max 50% shrinkage at +70% trailing)
#   target_return = config.get("mean_reversion_anchor", 0.08)  (8% long-run)
#   adjusted_forecast = (1 - shrinkage) * model_forecast + shrinkage * anchor
```

### Implementation

Add to `prediction_aggregator.py` after the main aggregation loop, before the OHLC constraint:

```python
# ------------------------------------------------------------------
# Mean-reversion shrinkage for long-horizon forecasts
# ------------------------------------------------------------------
# After strong years (trailing 12m > 20%), shrink 21d/252d forecasts
# toward the long-run equity premium (DeBondt & Thaler 1985).
if "close" in cache.columns and cache["close"].notna().sum() >= 252:
    _last_close = float(cache["close"].dropna().iloc[-1])
    _close_252_ago = float(cache["close"].dropna().iloc[-252])
    _trailing_12m = (_last_close / _close_252_ago) - 1.0 if _close_252_ago > 0 else 0.0
    _mr_anchor_annual = 0.08  # long-run equity premium ~8%/yr

    if abs(_trailing_12m) > 0.20:
        _excess = abs(_trailing_12m) - 0.20
        _shrinkage = min(0.50, _excess * 2.0)  # max 50% at +70%

        for _h_label, _h_days in [("21d", 21), ("252d", 252)]:
            _c_pred = aggregated.get("close", {}).get(_h_label)
            if _c_pred is not None:
                _pf = getattr(_c_pred, "point_forecast", None)
                if _pf is not None and _last_close > 0:
                    _pred_return = (_pf / _last_close) - 1.0
                    _anchor_return = _mr_anchor_annual * (_h_days / 252.0)
                    _adj_return = (1.0 - _shrinkage) * _pred_return + _shrinkage * _anchor_return
                    _c_pred.point_forecast = _last_close * (1.0 + _adj_return)
                    logger.info(
                        "Mean-reversion shrinkage at %s: trailing_12m=%.1f%%, "
                        "shrinkage=%.2f, pred_return=%.1f%%->%.1f%%",
                        _h_label, _trailing_12m * 100, _shrinkage,
                        _pred_return * 100, _adj_return * 100,
                    )
```

### Config

Add to `config/scoring_weights.yml`:

```yaml
# Mean-reversion dampener for long-horizon predictions
mean_reversion:
  enabled: true
  trailing_return_threshold: 0.20   # only shrink when |trailing 12m| > 20%
  max_shrinkage: 0.50               # max 50% blend toward anchor
  shrinkage_rate: 2.0               # shrinkage = min(max, excess * rate)
  anchor_return_annual: 0.08        # long-run equity premium (8%/yr)
  min_horizon_days: 21              # only apply to 21d+ horizons
```

### Files to Modify

1. `operator1/models/prediction_aggregator.py` -- add shrinkage block (~25 lines)
2. `config/scoring_weights.yml` -- add mean_reversion config section (~7 lines)

### Downstream Impact

| Consumer | Effect |
|----------|--------|
| `profile["predictions"]["close"]["252d"]` | Lower point forecast after strong years |
| `report_generator.py` investment recommendation | More conservative 1-year target |
| `ohlc_predictor.py` next_year candles | Uses prediction_aggregator close, so inherits dampening |
| `hedge_fund/engine.py` DCF | DCF uses MC growth, not aggregator -- NOT affected |
| `recursive_aggregator.py` | Runs independently -- NOT affected (has own confidence decay) |
| `monte_carlo.py` survival | MC uses per-regime distributions -- NOT affected |
| `frequency_fusion.py` | Fusion runs on per-frequency forecasts -- NOT affected |

### Expected Result

For AAPL 2024-12-31 backtest with trailing 12m = +34.9%:
- `excess = 0.349 - 0.20 = 0.149`
- `shrinkage = min(0.50, 0.149 * 2.0) = 0.298`
- `pred_return = 0.492` (original +49.2%)
- `anchor_return = 0.08` (8% long-run)
- `adj_return = 0.702 * 0.492 + 0.298 * 0.08 = 0.369` (+36.9%)
- Still overestimates vs actual +8.8%, but reduced from +49.2% to +36.9% (-25% error reduction)

For a more extreme case (trailing +60%):
- `shrinkage = min(0.50, 0.40 * 2.0) = 0.50`
- `adj_return = 0.50 * pred + 0.50 * 0.08` (50/50 blend with anchor)

### Risk Assessment

**Low risk.** Additive post-prediction adjustment, doesn't change any model internals. Only affects 21d+ horizons. Configurable via scoring_weights.yml (can be disabled with `enabled: false`). No cross-file dependencies beyond the prediction_aggregator insertion point.

### Estimated Scope

- 2 files, ~35 lines total
- Can be tested immediately by re-running AAPL backtest

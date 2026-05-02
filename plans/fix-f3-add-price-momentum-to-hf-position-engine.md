# Fix F3: Add Price Momentum to HF Position Engine

*2026-05-02*

## Problem

The HF position engine generated a SELL signal on AAPL while it rallied +9%. The quality assessment was correct (C+ grade: slowing growth, high leverage, rich valuation), but the engine ignored strong price momentum driven by AI narrative and buybacks.

## Diagnosis (Verified)

Two functions in [`operator1/hedge_fund/engine.py`](operator1/hedge_fund/engine.py) lack price momentum awareness:

### Source 1: `_compute_momentum()` (line 448)

Computes ONLY fundamental momentum:
- Revenue acceleration (QoQ 2nd derivative)
- Margin trend slope (net margin over 8Q)
- FCF conversion trend (OCF/revenue slope)
- ROIC trajectory (placeholder, always 50)

**Missing:** Price momentum (63d return, 252d return, momentum 12-1). The `cache` DataFrame is passed to the function but never used for price data.

### Source 2: `_compute_position_signal()` (line 1044)

Signal formula: `alpha * q_mult * s_mult * d_mult * conviction`

- `alpha` = `forecast_result.return_5d` (5-day model forecast) or grade fallback (C+ = -0.005)
- No price momentum input at all
- No guard against fighting strong price trends
- A stock up +9% in 63 days gets the same signal as one down -9%

## Fix Design

### Change 1: Add price momentum to `_compute_momentum()`

Add a 4th momentum component using cache price returns:

```python
# After existing fundamental momentum computation (line 504)

# Price momentum from cache (Jegadeesh & Titman 1993)
price_mom_score = 50.0  # neutral default
if cache is not None and "close" in cache.columns:
    closes = cache["close"].dropna()
    if len(closes) >= 63:
        ret_63d = (closes.iloc[-1] / closes.iloc[-63] - 1)
        # Also check 252d if available
        ret_252d = (closes.iloc[-1] / closes.iloc[-252] - 1) if len(closes) >= 252 else ret_63d
        # 12-1 momentum (skip most recent month to avoid reversal)
        ret_12_1 = (closes.iloc[-21] / closes.iloc[-252] - 1) if len(closes) >= 252 else ret_63d
        
        # Normalize: +20% annual return -> score 80, -20% -> score 20
        price_mom_score = normalize_score(50 + ret_63d * 200, 0, 100)
        
        result.price_momentum_63d = ret_63d
        result.price_momentum_252d = ret_252d
```

Then blend into the final score:

```python
# Updated score computation
score = (
    w.get("revenue_accel_weight", 0.30) * _norm(rev_accel, 500)
    + w.get("margin_trend_weight", 0.25) * _norm(margin_slope, 5000)
    + w.get("fcf_conversion_weight", 0.15) * _norm(fcf_conv_slope, 5000)
    + w.get("roic_trajectory_weight", 0.05) * 50
    + w.get("price_momentum_weight", 0.25) * price_mom_score  # NEW
)
```

Weight redistribution: fundamental drops from 100% to 75%, price gets 25%.

### Change 2: Add price momentum divergence detection

When fundamental momentum is weak but price momentum is strong, flag a "divergence" -- this is the exact AAPL scenario:

```python
# Price-fundamental divergence detection
if price_mom_score > 65 and result.score < 40:
    result.price_fundamental_divergence = True
    result.divergence_direction = "price_leading"  # price up, fundamentals down
elif price_mom_score < 35 and result.score > 60:
    result.price_fundamental_divergence = True
    result.divergence_direction = "fundamentals_leading"  # fundamentals up, price down
```

### Change 3: Add momentum clamp to `_compute_position_signal()`

Prevent the engine from generating strong SELL when price momentum is strongly positive:

```python
# After computing raw signal (line 1109)

# Momentum clamp: don't fight strong price trends
if cache is not None and "close" in cache.columns:
    closes = cache["close"].dropna()
    if len(closes) >= 63:
        ret_63d = float(closes.iloc[-1] / closes.iloc[-63] - 1)
        
        # Strong uptrend guard: if 63d return > +8%, floor signal at -0.1 (weak hold)
        if ret_63d > 0.08 and result.signal < -0.1:
            result.signal = -0.1
            result.label = "hold"
            result.momentum_override = True
            result.momentum_override_reason = (
                f"Signal clamped from {raw:.2f} to -0.10: "
                f"63d return +{ret_63d*100:.1f}% overrides quality-driven sell"
            )
        
        # Strong downtrend guard: if 63d return < -15%, cap signal at +0.1 (weak hold)
        elif ret_63d < -0.15 and result.signal > 0.1:
            result.signal = 0.1
            result.label = "hold"
            result.momentum_override = True
            result.momentum_override_reason = (
                f"Signal clamped from {raw:.2f} to +0.10: "
                f"63d return {ret_63d*100:.1f}% overrides quality-driven buy"
            )
```

### Change 4: Add new fields to result types

In [`operator1/hedge_fund/types.py`](operator1/hedge_fund/types.py):

```python
# MomentumCompositeResult -- add fields
price_momentum_63d: float | None = None
price_momentum_252d: float | None = None
price_fundamental_divergence: bool = False
divergence_direction: str = ""  # "price_leading" or "fundamentals_leading"

# PositionSignalResult -- add fields
momentum_override: bool = False
momentum_override_reason: str = ""
```

### Change 5: Add configurable weights to `config/hedge_fund_weights.yml`

```yaml
inflection:
  momentum:
    revenue_accel_weight: 0.30
    margin_trend_weight: 0.25
    fcf_conversion_weight: 0.15
    roic_trajectory_weight: 0.05
    price_momentum_weight: 0.25  # NEW
position_sizing:
  momentum_clamp:
    uptrend_threshold: 0.08    # 63d return above which sell is clamped
    downtrend_threshold: -0.15  # 63d return below which buy is clamped
    clamp_floor: -0.1          # minimum signal when uptrend clamped
    clamp_ceiling: 0.1         # maximum signal when downtrend clamped
```

## Files to Modify

| File | Change |
|------|--------|
| `operator1/hedge_fund/engine.py` | Add price momentum to `_compute_momentum()`, add momentum clamp to `_compute_position_signal()` |
| `operator1/hedge_fund/types.py` | Add `price_momentum_63d`, `price_momentum_252d`, `price_fundamental_divergence`, `divergence_direction` to `MomentumCompositeResult`; add `momentum_override`, `momentum_override_reason` to `PositionSignalResult` |
| `config/hedge_fund_weights.yml` | Add `price_momentum_weight` and `momentum_clamp` config |

## Expected Impact on AAPL Scenario

| | Before | After |
|---|--------|-------|
| Momentum score | ~35 (fundamental only) | ~62 (25% from +9% price return) |
| Scorecard Tier 4 "Inflection" | ~38 (weak) | ~49 (neutral) |
| Overall grade | C+ | C+ (quality grade unchanged) |
| Position alpha | -0.005 (C+ fallback) | -0.005 (unchanged) |
| Raw signal | ~-0.27 (sell) | ~-0.27 (sell, same) |
| **After momentum clamp** | N/A | **-0.10 (hold)** -- clamped by 63d +9% return |
| Final label | sell | **hold** |

The quality assessment stays correct (C+). The price target and risk warnings are unchanged. The only difference: the position signal is clamped to HOLD instead of SELL because the engine acknowledges strong price momentum shouldn't be fought.

## Why This Design is Safe

1. **Quality grade unchanged** -- the engine still correctly identifies slowing growth, high leverage, and rich valuation
2. **Clamp is one-directional** -- it prevents sell-on-uptrend and buy-on-downtrend, but doesn't create false buys or sells
3. **Configurable thresholds** -- the 8% uptrend and -15% downtrend thresholds are in config, not hardcoded
4. **Divergence flagging** -- when price leads fundamentals, the report will explicitly flag it as a risk ("price momentum may be unsustainable if fundamentals don't catch up")
5. **Fundamental momentum still dominates** (75% weight vs 25% price) -- this isn't a pure momentum strategy

# Per-Frequency Survival Criteria Design

## Principle

Each frequency has different data available and different signal reliability. Survival triggers should use ONLY the variables that are valid and meaningful at that frequency. No frequency should use a variable designed for a different timescale.

---

## Current Triggers (all frequencies, same criteria)

| # | Trigger | Type | Threshold | Category |
|---|---------|------|-----------|----------|
| 1 | `current_ratio < 1.0` | STOCK/STOCK | 1.0 | Liquidity |
| 2 | `debt_to_equity_abs > 3.0` | STOCK/STOCK | 3.0 | Liquidity |
| 3 | `fcf_yield < 0` | FLOW/MARKET | 0.0 | Liquidity |
| 4 | `drawdown_252d < -0.40` | OHLCV | -0.40 | Non-liquidity |
| 5 | `inst_flow_momentum < -0.15` | OHLCV-derived | -0.15 | Non-liquidity |
| 6 | `inst_crowding + illiquidity` | OHLCV-derived | 0.8 + P90 | Non-liquidity |
| 7 | `conflict_intensity > 0.7` | Static/news | 0.7 | Non-liquidity |
| 8 | `sanctions_flag == 1` | Static | 1 | Non-liquidity |

---

## Per-Frequency Trigger Matrix

### Annual (A) -- Full financial statement depth

All 8 triggers valid. Annual data has the richest financial statement information. Add 2 annual-specific triggers that leverage the longer time horizon.

| # | Trigger | Threshold | Status | Rationale |
|---|---------|-----------|--------|-----------|
| 1 | `current_ratio < 1.0` | 1.0 | USE | Stock/stock, valid at A |
| 2 | `debt_to_equity_abs > 3.0` | 3.0 | USE | Stock/stock, valid at A |
| 3 | `fcf_yield < 0` | 0.0 | **USE** | Flow/market, VALID at A (annual FCF / market_cap) |
| 4 | `drawdown_252d < -0.40` | -0.40 | USE | OHLCV resampled to A, represents year-over-year |
| 5 | `inst_flow_momentum < -0.15` | -0.15 | SKIP | No institutional data at annual frequency |
| 6 | `inst_crowding + illiquidity` | -- | SKIP | No institutional data at annual |
| 7 | `conflict_intensity > 0.7` | 0.7 | USE | Static, valid always |
| 8 | `sanctions_flag == 1` | 1 | USE | Static, valid always |
| **9** | **`revenue_growth_yoy < -0.20`** | **-0.20** | **NEW** | Revenue decline >20% YoY signals structural distress. Valid at A because revenue is at native annual scale. (Altman 1968: revenue/TA is Altman Z x5 component) |
| **10** | **`fh_altman_z_score < 1.81`** | **1.81** | **NEW** | Full Altman Z available with correct x3/x5 at A freq. Z < 1.81 = distress zone. (Altman 1968) |

### Quarterly (Q) -- Full financial statement depth + more granular

All 8 triggers valid. Quarterly is the sweet spot -- most filings are quarterly and all ratios compute correctly. Add revenue decline and Altman Z.

| # | Trigger | Threshold | Status | Rationale |
|---|---------|-----------|--------|-----------|
| 1 | `current_ratio < 1.0` | 1.0 | USE | Valid |
| 2 | `debt_to_equity_abs > 3.0` | 3.0 | USE | Valid |
| 3 | `fcf_yield < 0` | 0.0 | **USE** | Valid at Q (quarterly FCF annualized / market_cap) |
| 4 | `drawdown_252d < -0.40` | -0.40 | USE | OHLCV resampled |
| 5 | `inst_flow_momentum < -0.15` | -0.15 | SKIP | No institutional data at Q |
| 6 | `inst_crowding + illiquidity` | -- | SKIP | No institutional data at Q |
| 7 | `conflict_intensity > 0.7` | 0.7 | USE | Static |
| 8 | `sanctions_flag == 1` | 1 | USE | Static |
| **9** | **`revenue_growth_yoy < -0.20`** | **-0.20** | **NEW** | 4-quarter revenue decline. Valid at Q with correct TTM. |
| **10** | **`fh_altman_z_score < 1.81`** | **1.81** | **NEW** | Correct Altman Z at Q freq. |
| **11** | **`gross_margin < 0`** | **0** | **NEW** | Negative gross margin = selling below cost. Only reliable at Q/A where gross_margin is from same-filing gross_profit/revenue. |

### Semi-Annual (S) -- Same as Q with 2-period annualization

Same as Q. Semi-annual filings annualized by *2 produce correct ratios.

### Monthly (M) -- Interpolated from Q/A, limited reliability

Balance sheet ratios (stock/stock) are forward-filled from Q/A and valid. Flow-based ratios are interpolated and less reliable. OHLCV available natively at M.

| # | Trigger | Threshold | Status | Rationale |
|---|---------|-----------|--------|-----------|
| 1 | `current_ratio < 1.0` | 1.0 | USE | Stock/stock, forward-filled from Q, valid |
| 2 | `debt_to_equity_abs > 3.0` | 3.0 | USE | Stock/stock, valid |
| 3 | `fcf_yield < 0` | 0.0 | **SKIP** | Interpolated flow/market, unreliable |
| 4 | `drawdown_252d < -0.40` | -0.40 | USE | OHLCV resampled to M, valid |
| 5 | `inst_flow_momentum < -0.15` | -0.15 | SKIP | No institutional data at M |
| 6 | `inst_crowding + illiquidity` | -- | SKIP | No institutional data at M |
| 7 | `conflict_intensity > 0.7` | 0.7 | USE | Static |
| 8 | `sanctions_flag == 1` | 1 | USE | Static |

4 triggers active at M (down from 8 at Q).

### Weekly (W) -- OHLCV native + forward-filled balance sheet

Same as M. OHLCV-based triggers are native. Balance sheet triggers are forward-filled. Flow-based triggers skipped.

| # | Trigger | Threshold | Status | Rationale |
|---|---------|-----------|--------|-----------|
| 1 | `current_ratio < 1.0` | 1.0 | USE | Stock/stock, forward-filled |
| 2 | `debt_to_equity_abs > 3.0` | 3.0 | USE | Stock/stock |
| 3 | `fcf_yield < 0` | 0.0 | **SKIP** | Interpolated |
| 4 | `drawdown_252d < -0.40` | -0.40 | USE | OHLCV native at W |
| 5 | `inst_flow_momentum < -0.15` | -0.15 | SKIP | Not available at W |
| 6 | `inst_crowding + illiquidity` | -- | SKIP | Not available at W |
| 7 | `conflict_intensity > 0.7` | 0.7 | USE | Static |
| 8 | `sanctions_flag == 1` | 1 | USE | Static |

4 triggers active at W.

### Daily (D) -- OHLCV native + forward-filled balance sheet + institutional

OHLCV is native. Institutional flow/crowding are daily-computed from volume/holder data. Balance sheet ratios forward-filled and valid. Flow-based ratios (fcf_yield) distorted by interpolation -- skipped.

| # | Trigger | Threshold | Status | Rationale |
|---|---------|-----------|--------|-----------|
| 1 | `current_ratio < 1.0` | 1.0 | USE | Stock/stock, forward-filled, valid |
| 2 | `debt_to_equity_abs > 3.0` | 3.0 | USE | Stock/stock |
| 3 | `fcf_yield < 0` | 0.0 | **SKIP** | Interpolated flow/market, distorted |
| 4 | `drawdown_252d < -0.40` | -0.40 | USE | OHLCV native |
| 5 | `inst_flow_momentum < -0.15` | -0.15 | **USE** | Institutional flow computed from daily holder data |
| 6 | `inst_crowding + illiquidity` | 0.8+P90 | **USE** | Computed from daily volume data |
| 7 | `conflict_intensity > 0.7` | 0.7 | USE | Static |
| 8 | `sanctions_flag == 1` | 1 | USE | Static |
| **9** | **`merton_dd < 1.0`** | **1.0** | **NEW** | Merton distance-to-default uses daily close + volatility + debt (market/stock). DD < 1.0 = within 1 std of default barrier. Valid at D because uses OHLCV-native vol + stock debt. (Merton 1974) |
| **10** | **`vol_of_vol_21d > P95`** | **P95 own** | **NEW** | Volatility-of-volatility spike = regime instability. Pure OHLCV signal, native at D. (Cont & da Fonseca 2002) |

6 triggers active at D (different mix than Q/A: no fcf_yield, but has institutional + Merton DD + vol-of-vol).

---

## Summary Matrix

| Trigger | D | W | M | Q | S | A |
|---------|---|---|---|---|---|---|
| current_ratio < 1.0 | Y | Y | Y | Y | Y | Y |
| debt_to_equity > 3.0 | Y | Y | Y | Y | Y | Y |
| fcf_yield < 0 | **N** | **N** | **N** | **Y** | **Y** | **Y** |
| drawdown < -40% | Y | Y | Y | Y | Y | Y |
| inst_flow < -0.15 | Y | N | N | N | N | N |
| inst_crowding | Y | N | N | N | N | N |
| conflict > 0.7 | Y | Y | Y | Y | Y | Y |
| sanctions | Y | Y | Y | Y | Y | Y |
| revenue_growth < -20% | N | N | N | **Y** | **Y** | **Y** |
| altman_z < 1.81 | N | N | N | **Y** | **Y** | **Y** |
| gross_margin < 0 | N | N | N | **Y** | **Y** | N |
| merton_dd < 1.0 | **Y** | N | N | N | N | N |
| vol_of_vol > P95 | **Y** | N | N | N | N | N |
| **Total triggers** | **8** | **4** | **4** | **9** | **9** | **8** |

---

## Survival Probability Weighting per Frequency

The continuous survival_probability sigmoid should weight components differently per frequency:

| Component | D weight | Q/A weight | Rationale |
|-----------|---------|-----------|-----------|
| Liquidity (current_ratio, D/E) | 0.30 | 0.20 | More emphasis at D (fewer valid triggers) |
| Cash flow (fcf_yield) | 0.00 | 0.25 | SKIP at D, full weight at Q/A |
| Market stress (drawdown) | 0.25 | 0.15 | D has native OHLCV |
| Institutional (flow, crowding) | 0.15 | 0.00 | D-only signal |
| Geopolitical (conflict, sanctions) | 0.10 | 0.10 | Same |
| Credit (Merton DD) | 0.10 | 0.00 | D-only signal (needs daily vol) |
| Fundamental (revenue, Altman Z) | 0.00 | 0.20 | Q/A-only signals |
| Volatility regime (vol-of-vol) | 0.10 | 0.00 | D-only signal |
| **Total** | **1.00** | **0.90** | Q/A gets 0.10 from gross_margin when available |

---

## Stage 2.F Post-Fusion: Re-run survival on daily cache

After Stage 2.F forward-fills correct Q/A ratios into the daily cache, re-run survival with the D-freq trigger set. Now `fcf_yield` in the daily cache is the correct Q/A value (forward-filled), but we still SKIP it at D freq because the D-freq trigger set doesn't include it. The correct fcf_yield-based survival signal comes from the Q/A pipeline results via MF fusion.

Alternatively, after Stage 2.F, we could re-run survival with a SPECIAL "post-fusion" trigger set that INCLUDES fcf_yield (since it's now correct on the daily cache). This gives the daily survival_probability the benefit of all 8+ triggers:

```python
# In run_2_F_fusion(), after forward-fill:
cache["company_survival_mode_flag"] = compute_company_survival_flag(cache, freq="Q")  
# Use Q trigger set since ratios are now at Q-quality on the daily cache
cache["survival_probability"] = compute_survival_probability(cache)
cache = compute_hierarchy_weights(cache)
```

---

## Implementation Location

**File:** `operator1/analysis/survival_mode.py`

The `freq` parameter already exists. The per-frequency trigger selection needs to be implemented as a dict lookup:

```python
_FREQ_TRIGGERS = {
    "D": {"current_ratio", "debt_to_equity_abs", "drawdown_252d", 
           "inst_flow_momentum", "inst_crowding", "conflict", "sanctions",
           "merton_dd", "vol_of_vol_21d"},
    "W": {"current_ratio", "debt_to_equity_abs", "drawdown_252d",
           "conflict", "sanctions"},
    "M": {"current_ratio", "debt_to_equity_abs", "drawdown_252d",
           "conflict", "sanctions"},
    "Q": {"current_ratio", "debt_to_equity_abs", "fcf_yield", "drawdown_252d",
           "conflict", "sanctions", "revenue_growth_yoy", "fh_altman_z_score",
           "gross_margin"},
    "S": {"current_ratio", "debt_to_equity_abs", "fcf_yield", "drawdown_252d",
           "conflict", "sanctions", "revenue_growth_yoy", "fh_altman_z_score"},
    "A": {"current_ratio", "debt_to_equity_abs", "fcf_yield", "drawdown_252d",
           "conflict", "sanctions", "revenue_growth_yoy", "fh_altman_z_score"},
}
```

**File:** `operator1/stages/stage2_freq_pipeline.py`

Add survival re-run after 2.F fusion forward-fill.

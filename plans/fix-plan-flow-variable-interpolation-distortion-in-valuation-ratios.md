# Fix Plan: Flow-Variable Interpolation Distortion

## Evidence Summary

The frequency interpolator distributes quarterly flow variables into daily rates. This is correct for time-series models but breaks all ratios involving market_cap/flow or stock/flow:

| Ratio | Current Value | Expected Value | Distortion Factor |
|-------|--------------|----------------|-------------------|
| PE | 178.87 | ~44.7 | 4x (quarterly EPS not annualized) |
| EV/EBITDA | 89.97 | ~17-25 | 5.1x (TTM distorted) |
| P/S | 12,639.69 | ~9.9 | 1277x (daily revenue rate) |
| fcf_yield | 0.04% | ~3.3% | ~82x too small |
| roa | 0.03% | ~28% | ~1000x too small |
| revenue_ttm | $1.18B | $383B | 325x too small |
| net_income_ttm | $336M | $94B | 280x too small |

Key insight: **flow/flow ratios (gross_margin=77.5%, net_margin=28.1%) are CORRECT** because both sides scaled same way. Only cross-type ratios (market_cap/flow, stock/flow) are broken.

## Root Cause

1. `interpolate_statement_to_daily()` converts quarterly net_income ($24B) to daily rate ($84M/day * 63 business days per quarter, but actually just $84M forward-filled with 1 transition)
2. `_rolling_4q_ttm()` detects only 1 transition for net_income (constant value), sums 4x that = $336M instead of $94B
3. PE uses `eps_diluted` (1.4 per quarter) without annualizing: 250/1.4 = 178.9 instead of 250/5.6 = 44.7

## Fix Strategy: Preserve Raw Filing Values

### Change 1: Save raw filing values before interpolation
**Files:** `backtest_runner.py` L385-410, `main.py` L937-1002

Before calling `interpolate_statement_to_daily()`, forward-fill the raw statement values into `{col}_filing` columns:

```python
# Save raw filing values (forward-filled, NOT interpolated)
for col in ncols:
    filing_col = f"{col}_filing"
    if filing_col not in cache.columns:
        _raw = si[col].reindex(cache.index.union(si.index).sort_values()).ffill().reindex(cache.index)
        cache[filing_col] = _raw
```

This adds ~18 `_filing` columns (one per flow variable that has data).

### Change 2: Fix `_rolling_4q_ttm()` to use `_filing` columns
**File:** `derived_variables.py` L31-80, L529-534

```python
def _compute_ttm_and_growth(df):
    # Use _filing columns for TTM (raw quarterly values, not interpolated)
    revenue_raw = df.get("revenue_filing", df.get("revenue"))
    net_income_raw = df.get("net_income_filing", df.get("net_income"))
    ebitda_raw = df.get("ebitda_filing", df.get("ebitda"))
    
    df["revenue_ttm_asof"] = _rolling_4q_ttm(revenue_raw)
    df["net_income_ttm_asof"] = _rolling_4q_ttm(net_income_raw)
    df["ebitda_ttm_asof"] = _rolling_4q_ttm(ebitda_raw)
```

### Change 3: Fix PE to annualize quarterly EPS
**File:** `derived_variables.py` L389-413

```python
# eps_diluted is quarterly -- annualize by multiplying by 4
# (or use net_income_ttm_asof / shares once TTM is fixed)
eps_from_filings = df.get("eps_diluted", pd.Series(np.nan, index=df.index))
if eps_from_filings.notna().any():
    # Annualize quarterly EPS
    eps_annual = eps_from_filings * 4
    eps_calc = eps_annual
```

### Change 4: Fix fcf_yield to use filing TTM
**File:** `derived_variables.py` L290-299 (already partially fixed)

```python
# Use raw filing FCF for TTM, not interpolated daily FCF
_fcf_raw = df.get("free_cash_flow_filing", df.get("free_cash_flow"))
df["free_cash_flow_ttm_asof"] = _rolling_4q_ttm(_fcf_raw)
_fcf_for_yield = df["free_cash_flow_ttm_asof"]
```

### Change 5: Fix EV/EBITDA and P/S to use filing TTM
**File:** `derived_variables.py` L448-455 (already partially fixed for EV/EBITDA)

The fix from the previous commit already uses `ebitda_ttm_asof` and `revenue_ttm_asof`, but those TTM values are wrong because they're computed from interpolated daily values. Fixing Change 2 above will cascade-fix these.

### Change 6: Fix roa to use filing values
**File:** `derived_variables.py` L491-502

```python
def _compute_roa(df):
    # Use filing-frequency net_income (quarterly value, not daily rate)
    net_income_raw = df.get("net_income_filing", df.get("net_income"))
    total_assets = df.get("total_assets")  # stock variable, already correct
    result, ism, inv = safe_ratio(net_income_raw, total_assets, "roa")
```

## Affected Downstream Consumers (58 references across 19 files)

### CRITICAL (directly read distorted values):
| File | Refs | Impact |
|------|------|--------|
| `derived_variables.py` | 29 | Source of all distorted ratios |
| `survival_mode.py` | 6 | `fcf_yield < 0` trigger |
| `financial_health.py` | 8 | T1 (fcf_yield), T5 (PE, EV/EBITDA) scores |
| `monte_carlo.py` | 1 | `fcf_yield` in survival triggers |
| `hedge_fund/engine.py` | 3 | PE, fcf_yield for valuation |
| `hedge_fund/advanced_methods.py` | 6 | PE for earnings torpedo, capital cycle |
| `regime_mixer.py` | 2 | `fcf_yield < 0` for distress score |
| `profile_builder.py` | 9 | TTM values in profile |

### MODERATE (read values but tolerate NaN):
| File | Refs | Impact |
|------|------|--------|
| `adaptive_thresholds.py` | 2 | fcf_yield in peer calibration |
| `ethical_filters.py` | 2 | fcf_yield for cash filter |
| `linked_aggregates.py` | 2 | PE in relative metrics |
| `particle_filter.py` | 1 | free_cash_flow_ttm in state tracking |
| `prediction_aggregator.py` | 1 | fcf_yield in tree variable list |
| `scenario_engine.py` | 3 | fcf_yield in reverse stress |

### NO CHANGE NEEDED (flow/flow ratios are correct):
- `gross_margin` (77.5%) -- flow/flow, correct
- `operating_margin` (31.6%) -- flow/flow, correct
- `net_margin` (28.1%) -- flow/flow, correct
- `interest_coverage` (10.89) -- flow/flow, correct (both ebit and interest_expense scaled same)

## Expected Results After Fix

| Metric | Before | After |
|--------|--------|-------|
| PE | 178.87 | ~44.7 |
| EV/EBITDA | 89.97 | ~17-25 |
| P/S | 12,639 | ~9.9 |
| fcf_yield | 0.04% | ~3.3% |
| roa | 0.03% | ~28% |
| revenue_ttm | $1.18B | ~$383B |
| net_income_ttm | $336M | ~$94B |
| fcf_yield survival trigger | Never fires (tiny positive) | Correctly evaluates |
| FH T5 score | Garbage (PE=179, EV=90) | Correct |
| HF valuation | DCF $32 | ~$200-280 |
| Survival mode | 80% (current_ratio only) | ~0% (cash adequacy floor works with correct fcf_yield) |

## Execution Order

1. Add `_filing` columns in cache builder (backtest_runner.py + main.py) -- ~15 lines each
2. Fix `_rolling_4q_ttm()` to use `_filing` columns -- ~5 lines
3. Fix PE annualization -- ~3 lines
4. Fix fcf_yield TTM source -- ~3 lines (already partially done)
5. Fix roa to use filing values -- ~3 lines
6. Re-run backtest
7. Verify all 58 consumer references produce correct values

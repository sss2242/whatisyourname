# Fix Guide: DCF Annualization Bug

*Created: 2026-05-13 | Based on AAPL backtest validation*

## Diagnosis

The DCF Monte Carlo valuation in `_compute_dcf()` produces $32.33/share for AAPL instead of the expected ~$130-180/share. The stock was trading at $250.42.

**Root cause:** Line 690-695 of [`operator1/hedge_fund/engine.py`](operator1/hedge_fund/engine.py:690) extracts a single quarterly FCF value ($27.8B) and uses it as the annual base for a 5-year DCF projection without annualizing it.

## Evidence

```
cashflow_df: 9 rows x 6 columns (quarterly filings)
free_cash_flow column: NOT in cashflow_df
operating_cash_flow latest quarter: $29.7B
capex latest quarter: $1.97B
computed quarterly FCF: $27.8B

Actual Apple annual FCF (TTM): ~$111B ($27.8B x 4 quarters)
```

The DCF formula at line 769 uses `fcf_latest` directly:
```python
pv_fcf = sum(fcf_latest * (1 + g) ** t / (1 + wacc) ** t for t in range(1, years + 1))
```

With quarterly FCF ($27.8B) instead of annual ($111B), the enterprise value is 4x too low, producing $32/share instead of ~$130/share.

## Fix

**File:** [`operator1/hedge_fund/engine.py`](operator1/hedge_fund/engine.py:690)

**Strategy:** Sum the last 4 quarters of (OCF - capex) from `cashflow_df` to compute TTM FCF, instead of using a single quarter's value. Fall back to 4x latest quarter if fewer than 4 quarters are available.

### Exact Change (lines 690-695)

**Before:**
```python
fcf_latest = extract_latest_value(cashflow_df, "free_cash_flow")
if fcf_latest is None:
    ocf = extract_latest_value(cashflow_df, "operating_cash_flow")
    capex = extract_latest_value(cashflow_df, "capex")
    if ocf is not None and capex is not None:
        fcf_latest = ocf - abs(capex)
```

**After:**
```python
# Compute TTM (trailing twelve months) FCF by summing last 4 quarters.
# Using a single quarterly value produces 1/4 of the correct annual FCF,
# resulting in ~4x undervaluation (e.g., $32/share instead of $130 for AAPL).
fcf_latest = None
try:
    n_q = get_hf_weight("data_windows.quarterly_lookback", 8)
    ocf_series = extract_quarterly_series(cashflow_df, "operating_cash_flow", n_q)
    capex_series = extract_quarterly_series(cashflow_df, "capex", n_q)
    if len(ocf_series) >= 2 and len(capex_series) >= 2:
        # Align indices and compute per-quarter FCF
        common = ocf_series.index.intersection(capex_series.index)
        if len(common) >= 2:
            q_fcf = ocf_series.loc[common] - capex_series.loc[common].abs()
            # Sum last 4 quarters (or fewer if unavailable) and annualize
            n_periods = min(4, len(q_fcf))
            ttm_fcf = float(q_fcf.iloc[-n_periods:].sum())
            # Annualize if fewer than 4 quarters available
            if n_periods < 4:
                ttm_fcf = ttm_fcf * (4 / n_periods)
            fcf_latest = ttm_fcf
except Exception:
    pass

# Fallback: single quarter x 4 (annualized)
if fcf_latest is None:
    ocf = extract_latest_value(cashflow_df, "operating_cash_flow")
    capex = extract_latest_value(cashflow_df, "capex")
    if ocf is not None and capex is not None:
        fcf_latest = (ocf - abs(capex)) * 4  # annualize single quarter
```

### Expected Impact

With the fix, AAPL DCF should produce:
- TTM FCF = ~$111B (sum of 4 quarters)
- 5-year PV at 5% growth, 9% WACC = ~$520B
- Terminal PV = ~$1.5T
- Total EV = ~$2.0T
- Minus net debt ($68B) = ~$1.9T equity
- Per share: $1.9T / 15.1B shares = ~$128/share

This is still below market ($250) because Apple trades at a premium, but it's a reasonable DCF fundamental value -- not the absurd $32 from before.

## Files Modified

| # | File | Lines | Change |
|---|------|-------|--------|
| 1 | `operator1/hedge_fund/engine.py` | 690-695 | Replace single-quarter FCF extraction with TTM sum (~20 lines) |

## Risk Assessment

**Low risk.** The change only affects the FCF input to the DCF -- all other inputs (growth rate, WACC, terminal growth, shares, debt, cash) are unchanged. The fix uses `extract_quarterly_series()` which is already battle-tested by 6 other HF metrics (FCF quality, dividend burn, accruals, etc.).

## Testing

After fix, re-run stage 7.5.1 and verify:
```python
dcf = profile["hedge_fund"]["dcf"]
assert dcf["intrinsic_p50"] > 80  # should be ~$130, not $32
assert dcf["upside_pct"] > -60    # should be ~-48%, not -87%
```

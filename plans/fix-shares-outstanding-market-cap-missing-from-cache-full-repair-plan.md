# Fix Plan: shares_outstanding/market_cap Missing from Cache

## Root Cause

`shares_outstanding` is fetched into the profile dict by `us_edgar.py` line 410, but never transferred into the daily cache DataFrame. `derived_variables.py` lines 381-382 and 942-945 look for `shares_outstanding` or `market_cap` in the cache, find neither, and every downstream valuation computation returns NaN.

## Impact Chain

```
shares_outstanding NOT IN CACHE
  -> market_cap MISSING (derived_variables.py L276, L382)
     -> pe_ratio_calc ALL NaN (L381-395)
     -> ev_to_ebitda ALL NaN (L420-430)
     -> fcf_yield ALL NaN (L290-294)
     -> ps_ratio, pb_ratio ALL NaN
     -> Altman Z x4 coefficient NaN (financial_health.py L660-661)
     -> cash adequacy floor can't fire (survival_mode.py L166-168)
        -> current_ratio < 1.0 false positive on AAPL (378/502 days)
           -> hierarchy [50/30/15/4/1] crushes Tier 4/5
           -> MC survival = 47% (monte_carlo.py L83-84 market_cap floor disabled)
           -> DCF = $33 (hedge_fund/engine.py L1004-1005 no market_cap)
           -> HF grade = C instead of B+/A
     -> close prediction: AR(1) (no valuation mean-reversion anchor)
```

## Files to Edit (7 files, ~15 changes)

### Fix 1: Inject shares_outstanding from profile into cache
**Files:** `backtest_runner.py`, `main.py`

#### backtest_runner.py (3 changes)

**Change 1a** -- Add `shares_outstanding` to CompanyFacts fallback (after line 434):
```python
_critical_balance_fields = {
    ...existing fields...
    "shares_outstanding": [
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
}
```
Note: `EntityCommonStockSharesOutstanding` is in the DEI namespace, not us-gaap. The CompanyFacts fallback at line 476 fetches from `facts.us-gaap`. Need to also check `facts.dei` for this concept.

**Change 1b** -- Inject profile shares_outstanding into cache (after line 563, after cache build):
```python
# Inject shares_outstanding from profile if not already in cache
if "shares_outstanding" not in cache.columns or cache["shares_outstanding"].isna().all():
    _shares = state.target_profile.get("shares_outstanding")
    if _shares and float(_shares) > 0:
        cache["shares_outstanding"] = float(_shares)
        logger.info("Injected shares_outstanding from profile: %s", _shares)
```

**Change 1c** -- Add DEI namespace check in CompanyFacts fallback (around line 478):
After fetching `_usgaap`, also try `_dei`:
```python
_dei = _facts_resp.json().get("facts", {}).get("dei", {})
```
And when processing `shares_outstanding`, check `_dei` first (it's the most reliable source).

#### main.py (2 changes)

**Change 1d** -- Same injection after cache build (after line 1010):
```python
# Inject shares_outstanding from profile if missing from cache
if "shares_outstanding" not in cache.columns or cache["shares_outstanding"].isna().all():
    _shares = target_profile.get("shares_outstanding")
    if _shares and float(_shares) > 0:
        cache["shares_outstanding"] = float(_shares)
        logger.info("Injected shares_outstanding from profile: %s", _shares)
```

**Change 1e** -- Compute market_cap immediately after injection:
```python
if "shares_outstanding" in cache.columns and "close" in cache.columns:
    if "market_cap" not in cache.columns or cache["market_cap"].isna().all():
        cache["market_cap"] = cache["close"] * cache["shares_outstanding"]
        logger.info("Computed market_cap from close * shares_outstanding")
```

### Fix 2: Add DEI namespace to CompanyFacts fallback
**File:** `backtest_runner.py`

The CompanyFacts API returns `facts.dei.EntityCommonStockSharesOutstanding` which is the most reliable shares_outstanding source for many companies. The current fallback only checks `facts.us-gaap`.

**Change 2a** -- After line 478 (`_usgaap = ...`), add:
```python
_dei = _facts_resp.json().get("facts", {}).get("dei", {})
```

**Change 2b** -- In the concept loop (line 485), for `shares_outstanding` field, check `_dei` first:
```python
if _field == "shares_outstanding":
    # DEI namespace has the most reliable shares data
    for _concept in _concepts:
        _cdata = _dei.get(_concept, {})
        _entries = _cdata.get("units", {}).get("shares", [])
        if _entries:
            # Build series from DEI data
            ...
```

### Fix 3: Verify ohlcv_yfinance provides market_cap
**File:** `operator1/clients/ohlcv_yfinance.py`

Check if yfinance's `download()` or `Ticker.info` provides `shares_outstanding` or `marketCap`. If yes, include it in the returned quotes DataFrame. This would make the profile injection a fallback rather than the only path.

**Inspection needed:** Lines around the `yf.download()` call and return DataFrame construction.

### Fix 4: Downstream verification points (no code changes, just validation)

After Fixes 1-3, verify these downstream consumers now work:

| Consumer | File | Lines | What to verify |
|----------|------|-------|---------------|
| `pe_ratio_calc` | `derived_variables.py` | 381-395 | Non-NaN values |
| `enterprise_value` | `derived_variables.py` | 420-430 | Non-NaN values |
| `fcf_yield` | `derived_variables.py` | 290-294 | Non-NaN values |
| `ps_ratio_calc` | `derived_variables.py` | 417 | Non-NaN values |
| `pb_ratio` | `derived_variables.py` | 435 | Non-NaN values |
| Cash adequacy floor | `survival_mode.py` | 166-168 | AAPL should NOT trigger survival |
| Altman Z x4 | `financial_health.py` | 660-661 | Non-NaN Z-score |
| MC market_cap floor | `monte_carlo.py` | 83-84 | $3.8T AAPL gets 0.97+ floor |
| DCF current_price | `hedge_fund/engine.py` | 1004-1005 | Uses market_cap |
| Merton DD | `derived_variables.py` | 942-945 | market_cap for equity value |
| HF DCF | `hedge_fund/engine.py` | 690-693 | shares for per-share calcs |

### Fix 5: Prevent future regressions

Add `shares_outstanding` and `market_cap` to the data quality check in `operator1/quality/data_quality.py` as CRITICAL fields. If either is missing for a company with OHLCV data, log an ERROR (not just a warning).

## Execution Order

1. Edit `backtest_runner.py` (Changes 1a, 1b, 1c, 2a, 2b)
2. Edit `main.py` (Changes 1d, 1e)
3. Inspect and potentially edit `ohlcv_yfinance.py` (Fix 3)
4. Re-run the AAPL backtest
5. Verify all downstream consumers (Fix 4)
6. Add regression guard (Fix 5)

## Expected Results After Fix

| Metric | Before Fix | After Fix (expected) |
|--------|-----------|---------------------|
| `shares_outstanding` | MISSING | ~15.1B (AAPL) |
| `market_cap` | MISSING | ~$3.78T |
| `pe_ratio_calc` | ALL NaN | ~33x |
| `fcf_yield` | ALL NaN | ~3.3% |
| Survival mode days | 378/502 (76%) | ~0/502 (0%) |
| MC survival prob | 47.3% | ~97%+ |
| DCF intrinsic P50 | $33.27 | ~$200-280 |
| HF grade | C | B+/A |
| Close prediction model | AR(1) | Kalman/LSTM/Tree (with PE anchor) |
| 1-day error | 3.2% | Expected <2% |
| 21-day error | 11.5% | Expected <7% |

# Plan: Fix All Pipeline Bugs from Pipeline Runs

## Bug Inventory

Bugs discovered from the three pipeline runs (SEC EDGAR/AAPL, DART/LF, J-Quants/Toyota) plus user-reported CVM/EU ESEF search failures and chart date bug.

| # | Bug | Severity | Source | Files |
|---|-----|----------|--------|-------|
| B1 | `gemini-1.5-flash` deprecated (404 from Gemini API) | HIGH | DART run | `llm_base.py`, `run.py` |
| B2 | `torch.elu` moved in PyTorch 2.10 -- TFT model broken | MEDIUM | DART run | `forecasting.py` |
| B3 | Graph risk expects dict but receives `LinkedEntity` dataclass | MEDIUM | DART run | `main.py`, `graph_risk.py` |
| B4 | Walk-forward fails: `SurvivalTimelineResult has no len()` | MEDIUM | DART run | `walk_forward.py` or `main.py` |
| B5 | Conformal calibrator: `ConformalCalibrator has no attribute update` | MEDIUM | DART run | `conformal.py` or `main.py` |
| B6 | J-Quants empty identifier passed to API | HIGH | J-Quants run | `jp_jquants_wrapper.py`, `equity_provider.py` |
| B7 | CVM search fails to find companies | HIGH | User report | `br_cvm_wrapper.py` |
| B8 | EU ESEF search fails to find companies | HIGH | User report | `eu_esef_wrapper.py` |
| B9 | Forward pass returns 0 silently | MEDIUM | DART run | `forecasting.py`, `main.py` |
| B10 | HMM full covariance fails, diagonal fallback used | LOW | Both runs | `regime_detector.py` |
| B11 | Charts show 1959-1960 dates instead of actual 2024-2026 dates | HIGH | User report | `report_generator.py`, cache building |

---

## B1: `gemini-1.5-flash` Deprecated (404)

### Problem
The DART run shows repeated 404 errors:
```
models/gemini-1.5-flash is not found for API version v1beta
```
Google has deprecated `gemini-1.5-flash` from the v1beta API. The user's `.env` or config specifies this model, but the default in [`gemini.py:23`](operator1/clients/gemini.py:23) is already `gemini-2.0-flash` (correct). The issue is that when the user passes a model name via config/env that no longer exists, the retry loop burns all 5 retries on a non-retryable 404.

### Fix
1. **In [`llm_base.py`](operator1/clients/llm_base.py)**: Remove `gemini-1.5-flash` and `gemini-1.5-pro` from `GEMINI_MODELS` registry (they are deprecated)
2. **In [`llm_base.py:297-299`](operator1/clients/llm_base.py:297)**: The error handling already treats non-retryable status codes differently, but 404 is not in `retryable_codes`. The issue is that the `_execute_request` method still retries 404s. Add 404 to the non-retryable fast-fail path (it already should be -- verify the condition at line 297 actually breaks the retry loop)
3. **In [`validate_model()`](operator1/clients/llm_base.py:161)**: When a model is not in the registry, log a clearer warning about deprecation and auto-select the best available model (this already works but the user override from `.env` may bypass it)

### Files to modify
- [`operator1/clients/llm_base.py`](operator1/clients/llm_base.py) -- Remove deprecated models from registry, verify 404 non-retry behavior

---

## B2: `torch.elu` Moved in PyTorch 2.10

### Problem
TFT (Temporal Fusion Transformer) model fails with:
```
module 'torch' has no attribute 'elu'
```
In PyTorch 2.10, `torch.elu` was removed from the top-level namespace. The correct import is `torch.nn.functional.elu`.

### Fix
Search for all occurrences of `torch.elu` in [`forecasting.py`](operator1/models/forecasting.py) and replace with `torch.nn.functional.elu` (or `F.elu` with the standard `import torch.nn.functional as F` alias).

### Files to modify
- [`operator1/models/forecasting.py`](operator1/models/forecasting.py) -- Replace `torch.elu` with `torch.nn.functional.elu`

---

## B3: Graph Risk Expects Dict, Receives LinkedEntity Dataclass

### Problem
DART run shows:
```
Graph risk computation failed: 'LinkedEntity' object has no attribute 'get'
```
[`graph_risk.py`](operator1/models/graph_risk.py) expects entity data as dicts (using `.get()` method), but [`main.py`](main.py) passes `LinkedEntity` dataclass instances from entity discovery.

### Fix
Two options (prefer option A):

**Option A**: Convert `LinkedEntity` instances to dicts before passing to graph risk:
```python
# In main.py where graph_risk is called
from dataclasses import asdict
entity_dicts = {group: [asdict(e) for e in entities] for group, entities in relationships.items()}
graph_risk_result = compute_graph_risk_metrics(target_isin, entity_dicts)
```

**Option B**: Update `graph_risk.py` to handle both dicts and dataclasses by using `getattr(entity, field, default)` instead of `entity.get(field, default)`.

### Files to modify
- [`main.py`](main.py) -- Convert LinkedEntity to dict before passing to graph_risk (Option A)
- OR [`operator1/models/graph_risk.py`](operator1/models/graph_risk.py) -- Use `getattr` instead of `.get()` (Option B)

---

## B4: Walk-Forward Fails on SurvivalTimelineResult

### Problem
DART run shows:
```
Walk-forward failed: object of type 'SurvivalTimelineResult' has no len()
```
[`walk_forward.py`](operator1/models/walk_forward.py) calls `len()` on the survival timeline result, but `SurvivalTimelineResult` is a dataclass without `__len__`.

### Fix
1. Check what [`run_walk_forward()`](operator1/models/walk_forward.py) expects as input -- it likely expects the `.timeline` DataFrame, not the result wrapper
2. In `main.py`, pass `enriched_timeline_result.timeline` (the DataFrame) instead of the full `SurvivalTimelineResult` object
3. OR add a `__len__` method to `SurvivalTimelineResult` that delegates to `len(self.timeline)`

### Files to modify
- [`main.py`](main.py) -- Pass `.timeline` DataFrame instead of the full result object
- OR [`operator1/analysis/survival_timeline.py`](operator1/analysis/survival_timeline.py) -- Add `__len__` to `SurvivalTimelineResult`

---

## B5: Conformal Calibrator Missing `update` Method

### Problem
DART run shows:
```
Conformal prediction failed: 'ConformalCalibrator' object has no attribute 'update'
```
The main.py code calls `calibrator.update(r)` but `ConformalCalibrator` does not have an `update` method.

### Fix
1. Read [`operator1/models/conformal.py`](operator1/models/conformal.py) to find the correct method name (likely `add_residual()`, `calibrate()`, or `feed()`)
2. Update the call site in `main.py` to use the correct method name
3. OR add an `update()` alias in `ConformalCalibrator` that delegates to the existing method

### Files to modify
- [`main.py`](main.py) -- Fix method call to match `ConformalCalibrator` API
- OR [`operator1/models/conformal.py`](operator1/models/conformal.py) -- Add `update()` method

---

## B6: J-Quants Empty Identifier

### Problem
J-Quants run shows Toyota (7203) selected, but the adapter passes an empty identifier to the API:
```
J-Quants get_financials failed for : 400 Client Error: Bad Request for url: ...date=20210308
```
Also, the profile loaded the wrong company (KYOKUYO CO.,LTD. instead of Toyota).

### Root Cause
The `_JPJquantsAdapter` in [`equity_provider.py`](operator1/clients/equity_provider.py:62) wraps `JPJquantsClient`. The issue is likely in how the company code is passed through the adapter -- the adapter may not be forwarding the identifier correctly to `get_financials()`.

### Fix
1. In [`jp_jquants_wrapper.py`](operator1/clients/jp_jquants_wrapper.py): Verify that `get_financials(identifier)` correctly uses the company code (e.g., "7203") in the J-Quants API call
2. In [`equity_provider.py`](operator1/clients/equity_provider.py): Check that `_JPJquantsAdapter._get_financials(identifier)` passes the identifier through to the underlying client
3. Fix the company list search -- the profile fetch returned KYOKUYO (1301) instead of Toyota (7203), suggesting the list_companies search or profile lookup uses the wrong index

### Files to modify
- [`operator1/clients/jp_jquants_wrapper.py`](operator1/clients/jp_jquants_wrapper.py) -- Fix identifier passthrough in financials API calls
- [`operator1/clients/equity_provider.py`](operator1/clients/equity_provider.py) -- Verify adapter forwards identifier correctly

---

## B7: CVM Search Fails to Find Companies

### Problem
User reports that CVM (Brazil) search does not find companies.

### Root Cause Analysis
Looking at [`br_cvm_wrapper.py:86-120`](operator1/clients/br_cvm_wrapper.py:86):
1. `list_companies()` fetches the full CSV from `https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv`
2. It filters by `SIT == "ATIVO"` (active companies)
3. Search matches on `name.lower()` or `ticker.lower()`

**Likely issues**:
- The CSV URL may have changed or returned an error
- The CSV delimiter is `;` (semicolon) -- if the format changed, the parser breaks silently
- The `CD_CVM` field is used as "ticker" but B3 tickers (PETR4, VALE3) are not in the CVM registry -- users search by B3 ticker and get no results
- The `get_profile()` method at line 130 tries to strip trailing digits from B3 tickers, but the remaining base ("PETR", "VALE") may not match the Portuguese company name ("PETROLEO BRASILEIRO S.A.", "VALE S.A.")

### Fix
1. Add fallback search strategies:
   - Search by CNPJ if the query looks like a CNPJ
   - Search by commercial name (`DENOM_COMERC`) in addition to legal name (`DENOM_SOCIAL`)
   - Add common B3 ticker-to-CVM-code mapping for major companies (PETR4->9512, VALE3->4170, ITUB4->19348)
2. Add error logging when the CSV fetch fails (currently swallowed by the try/except)
3. Add a debug log showing how many companies were loaded from CSV and how many matched the query

### Files to modify
- [`operator1/clients/br_cvm_wrapper.py`](operator1/clients/br_cvm_wrapper.py) -- Improve search with multiple strategies, add logging

---

## B8: EU ESEF Search Fails to Find Companies

### Problem
User reports that EU ESEF search does not find companies.

### Root Cause Analysis
Looking at [`eu_esef_wrapper.py:118-152`](operator1/clients/eu_esef_wrapper.py:118):
1. `list_companies()` calls `_get_recent_filings()` which fetches from `filings.xbrl.org/api/filings`
2. It extracts entity names from the filing metadata
3. Search matches on `name.lower()` or `lei.lower()`

**Likely issues**:
- The `filings.xbrl.org` API may return filings without embedded entity data (the `entity` field may be empty)
- The search method at line 154 (`_search_entities_api`) is defined but `list_companies()` does NOT call it -- it only uses `_get_recent_filings()` which may return a limited set
- Company names in ESEF filings are often in local languages (e.g., "Bayerische Motoren Werke AG" instead of "BMW"), making English-language searches fail
- The paginated entities search at line 189 fetches only 100 entities per page -- if the target company is not in the first page, it will not be found

### Fix
1. Make `search_company()` call `_search_entities_api()` (the dedicated search method) instead of just delegating to `list_companies()`
2. In `_search_entities_api()`, increase page size and add multi-page iteration
3. Add ticker/ISIN-based search -- many EU companies can be found by LEI or ISIN
4. Add English alias matching for major EU companies
5. Implement the `/api/filings` search with `filter[entity_name]` parameter if the API supports it

### Files to modify
- [`operator1/clients/eu_esef_wrapper.py`](operator1/clients/eu_esef_wrapper.py) -- Wire `_search_entities_api()` into `search_company()`, improve search strategies

---

## B9: Forward Pass Returns 0 Silently

### Problem
DART run shows:
```
Forward pass failed: 0
```
The forward pass function returns 0 (or an integer) instead of a `ForwardPassResult`, and main.py logs this as a failure without a stack trace.

### Root Cause
The TFT failures (`torch.elu` -- B2) may cause the forward pass to fail completely. The error handling in main.py likely catches the exception and logs the return value (0) without the traceback.

### Fix
1. Fix B2 first (torch.elu) -- this likely resolves the forward pass failure
2. In `main.py`, add `traceback.format_exc()` to the forward pass error handler so failures are not silent
3. In [`forecasting.py`](operator1/models/forecasting.py), ensure `run_forward_pass()` returns a valid `ForwardPassResult` even when individual models fail (the existing try/except per-model should handle this, but verify)

### Files to modify
- [`main.py`](main.py) -- Add traceback logging for forward pass failures
- [`operator1/models/forecasting.py`](operator1/models/forecasting.py) -- Verify graceful degradation when TFT fails

---

## B10: HMM Full Covariance Fails

### Problem
Both runs show:
```
HMM full covariance failed, retrying with diagonal covariance
```
The HMM fitting fails with `covars must be symmetric, positive-definite` when using full covariance.

### Fix (Low Priority -- Already Has Fallback)
The diagonal covariance fallback is already implemented and works. To improve:
1. Add diagonal regularization before attempting full covariance: `covars += 1e-6 * np.eye(n)`
2. This is already noted in the [`forecasting-models-full-potential-architecture-plan.md`](plans/forecasting-models-full-potential-architecture-plan.md) as fix A2

### Files to modify
- [`operator1/models/regime_detector.py`](operator1/models/regime_detector.py) -- Add covariance regularization before HMM fit

---

## B11: Charts Show 1959-1960 Dates Instead of 2024-2026

### Problem
User reports that generated charts display dates from 1959-1960 instead of the correct 2024-2026 date range. This affects all time-series charts (price history, survival timeline, volatility, financial health, etc.).

### Root Cause Analysis
The chart code in [`report_generator.py:2986-2988`](operator1/report/report_generator.py:2986) plots using `cache.index`:
```python
ax.plot(cache.index, cache["close"], ...)
```
And [`_apply_brand_style`](operator1/report/report_generator.py:2961) applies date formatting:
```python
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
```

**The bug**: If `cache.index` is an integer-based index (0, 1, 2, ...) or dates stored as integers (like YYYYMMDD format), matplotlib's `DateFormatter` interprets them as **ordinal days since year 0001**. Integer values around 715,000-716,000 (typical for ~1960 dates in matplotlib ordinal format) would appear as 1959-1960.

This happens when:
1. The cache DataFrame loses its DatetimeIndex during processing (e.g., after a `.reset_index()` without reassigning)
2. The OHLCV data from yfinance has a `Date` column but it gets stored as an integer column instead of DatetimeIndex
3. The as-of merge in main.py's Step 4 produces a RangeIndex instead of DatetimeIndex

Additionally, the `filing_calendar.py` module uses `date.today()` which is correct, but if the cache index is wrong, the calendar analysis would also produce wrong date ranges.

### Fix

**Primary fix** -- In [`report_generator.py`](operator1/report/report_generator.py) `generate_charts()` function:
1. Before any plotting, ensure `cache.index` is a proper `DatetimeIndex`:
```python
if not isinstance(cache.index, pd.DatetimeIndex):
    # Try common date column names
    for col in ["date", "Date", "timestamp"]:
        if col in cache.columns:
            cache = cache.set_index(pd.to_datetime(cache[col]))
            break
    else:
        # If index is integer-like, try converting
        try:
            cache.index = pd.to_datetime(cache.index)
        except Exception:
            logger.warning("Cannot convert cache index to DatetimeIndex for charts")
            return []
```

**Secondary fix** -- In `main.py` Step 4 (cache building):
1. After the OHLCV merge, verify the cache has a DatetimeIndex and log if it does not
2. After the as-of statement merge, verify the index is still DatetimeIndex (merge operations can reset it)

**Tertiary fix** -- In the OHLCV providers (yfinance, pykrx, baostock, twstock, nselib):
1. Ensure all providers return DataFrames with a `date` column as proper datetime type, not strings or integers

### Files to modify
- [`operator1/report/report_generator.py`](operator1/report/report_generator.py) -- Add DatetimeIndex validation before charting
- [`main.py`](main.py) -- Add index type assertion after cache building
- Optionally: OHLCV provider wrappers -- Ensure date columns are proper datetime

---

## Execution Order

The bugs should be fixed in this order (dependencies noted):

```
B1  (gemini-1.5-flash deprecated)      -- standalone, highest user impact
B11 (chart dates 1959-1960)             -- standalone, high user impact
B2  (torch.elu)                         -- standalone, unblocks B9
B9  (forward pass silent failure)       -- depends on B2
B3  (graph risk dict vs dataclass)      -- standalone
B4  (walk-forward len)                  -- standalone
B5  (conformal update method)           -- standalone
B6  (J-Quants empty identifier)         -- standalone
B7  (CVM search)                        -- standalone
B8  (EU ESEF search)                    -- standalone
B10 (HMM covariance)                    -- standalone, low priority
```

## Todo List for Implementation

```
[ ] B1: Remove deprecated Gemini models from registry, verify 404 fast-fail
[ ] B11: Fix chart date rendering -- ensure cache.index is DatetimeIndex before plotting
[ ] B2: Replace torch.elu with torch.nn.functional.elu in forecasting.py
[ ] B9: Add traceback logging to forward pass error handler in main.py
[ ] B3: Convert LinkedEntity dataclass to dict before passing to graph_risk
[ ] B4: Pass .timeline DataFrame to walk-forward instead of SurvivalTimelineResult
[ ] B5: Fix conformal calibrator method name mismatch
[ ] B6: Fix J-Quants identifier passthrough in adapter and wrapper
[ ] B7: Improve CVM search with multi-strategy matching and better logging
[ ] B8: Wire _search_entities_api into ESEF search_company, improve EU search
[ ] B10: Add covariance regularization to HMM fitting
[ ] Run tests to verify all fixes
[ ] Create PR with all fixes
```

# Fix Product Segment Data Pipeline -- 3 Bug Fixes

*Based on debug scan findings from 2026-04-13*

---

## Overview

The debug scan found 3 connected bugs in the product/segment data pipeline. Segment revenue data is extracted from XBRL/PDF sources for some markets but:
1. Never converted to daily cache columns for quantitative models
2. Not extracted for 2 markets that have working code (MX, SA)  
3. Not extracted for 8 markets that have configured PDF keywords

These fixes restore the full data flow: **extraction -> cache injection -> temporal model consumption -> Monte Carlo concentration risk -> report rendering**.

---

## Bug 1: Product Segment Data Never Reaches Quantitative Models

**Root cause:** No module exists to convert the segment revenue dict from `extract_segment_data()` into daily cache columns.

**Current flow (broken):**
```
extract_segment_data() -> profile["product_segments"] -> report Section 30
                       X-> cache columns (MISSING)
                       X-> _extra_vars (columns don't exist)
                       X-> Monte Carlo segment_hhi (always 0)
```

**Target flow (fixed):**
```
extract_segment_data() -> compute_product_metrics() -> cache columns -> _extra_vars -> temporal models
                       |                            |-> MC concentration risk
                       +-> profile["product_segments"] -> report Section 30
```

### Implementation Steps

#### Step 1: Create `operator1/features/product_metrics.py`

New module (~200 lines) that converts segment revenue data into daily cache columns.

**Function:** `compute_product_metrics(cache, segment_data) -> cache`

**Input:** 
- `cache`: daily DataFrame
- `segment_data`: dict from `extract_segment_data()` with keys `segments` (name->revenue), `descriptions`, `n_segments`

**Columns produced (10):**
- `segment_hhi`: Herfindahl-Hirschman Index of segment revenue concentration (0-1)
- `dominant_segment_growth`: YoY growth rate of the largest segment (from multiple filings if available)
- `estimated_market_share`: dominant segment revenue / sector total revenue (from linked_agg if available)
- `cannibalization_rate`: overlap coefficient between new and old segment revenue (0 if single snapshot)
- `net_new_revenue_pct`: fraction of total revenue from segments that didn't exist in prior period
- `network_effect_score`: 0-1 score based on whether dominant segment shows accelerating growth (proxy for network effects)
- `input_cost_pressure`: COGS/revenue trend as proxy for input cost pressure on segments
- `growth_runway_quarters`: estimated quarters of above-average growth remaining (from segment growth trajectory)
- `maturity_concentration`: fraction of revenue from segments with decelerating growth
- `segment_count`: number of reporting segments

**Logic:**
- HHI = sum of squared revenue shares: `sum((rev_i / total_rev)^2)`
- For single-snapshot data (most common): static values broadcast across daily index
- For multi-period data (if `extract_segment_data` returns historical): compute YoY changes

#### Step 2: Wire `compute_product_metrics()` in `main.py`

After the existing `extract_segment_data()` call at line ~3655, add:

```python
# Inject segment metrics into cache for temporal models
if _seg_result and _seg_result.get("n_segments", 0) >= 2:
    try:
        from operator1.features.product_metrics import compute_product_metrics
        cache = compute_product_metrics(cache, _seg_result)
        logger.info("Product metrics injected: %d columns", 
                     sum(1 for c in cache.columns if c.startswith("segment_")))
    except Exception as exc:
        logger.debug("Product metrics computation failed: %s", exc)
```

**Location:** After line ~3685 in main.py (after profile["product_segments"] is set).

#### Step 3: Wire in `backtest_runner.py`

Add the same `compute_product_metrics()` call in `run_stage3()` after profile building, or in `run_stage1()` after segment extraction (if segment extraction is added to stage 1).

#### Step 4: Verify `_extra_vars` consumption

The existing `_extra_vars` list at main.py:2441-2444 already references these column names. Once they exist in the cache, they'll be automatically included in temporal model features. No changes needed to the `_extra_vars` list.

#### Step 5: Verify Monte Carlo `segment_hhi` consumption

The existing check at main.py:2767 (`if "segment_hhi" in cache.columns`) will automatically work once the column exists. No changes needed.

---

## Bug 2: MX BMV and SA Tadawul Segment Extraction Unwired

**Root cause:** Standalone segment extraction functions exist but aren't exposed as `extract_segment_data()` methods on the PIT client classes.

### Implementation Steps

#### Step 6: Add `extract_segment_data()` to `MXBmvClient`

In `operator1/clients/mx_bmv.py`, add method to the `MXBmvClient` class:

```python
def extract_segment_data(self, identifier: str) -> dict:
    """Extract segment data via BMV XBRL JSON."""
    try:
        return extract_bmv_segment_data(identifier)
    except Exception as exc:
        logger.debug("BMV segment extraction failed: %s", exc)
        return {"n_segments": 0, "segments": {}, "descriptions": {}}
```

The `extract_bmv_segment_data()` function already exists at line 1231 in the same file.

#### Step 7: Add `extract_segment_data()` to `SATadawulClient`

In `operator1/clients/sa_tadawul.py`, add method to the `SATadawulClient` class. This requires:
1. Fetching the XBRL HTML for the company (reuse existing `_fetch_xbrl_html()` or equivalent from the financial statements flow)
2. Calling `_extract_segment_revenue_from_xbrl(html_text)` (exists at line 634)
3. Wrapping result in the standard segment dict format

```python
def extract_segment_data(self, identifier: str) -> dict:
    """Extract segment data from Tadawul XBRL HTML."""
    try:
        # Reuse existing XBRL fetch path
        html_text = self._fetch_xbrl_statements(identifier, "income")
        if not html_text:
            return {"n_segments": 0, "segments": {}, "descriptions": {}}
        segments = _extract_segment_revenue_from_xbrl(html_text)
        return {
            "n_segments": len(segments),
            "segments": segments,
            "descriptions": {},
            "has_revenue": len(segments) >= 2,
            "has_descriptions": False,
            "source": "tadawul_xbrl",
        }
    except Exception as exc:
        logger.debug("Tadawul segment extraction failed: %s", exc)
        return {"n_segments": 0, "segments": {}, "descriptions": {}}
```

---

## Bug 3: 8 Markets Have Unused Fuzzy PDF Parser Segment Keywords

**Root cause:** The `fuzzy_pdf_parser` has market-specific segment keywords for IN, AU, CA, HK, SG, ZA, AE, but none of these wrappers have `extract_segment_data()`.

### Implementation Steps

#### Step 8: Create generic `_extract_segment_data_via_pdf()` helper

In `operator1/clients/filing_discoverer.py`, add a new function:

```python
def try_segment_extraction(
    ticker: str,
    market_id: str,
    llm_client=None,
) -> dict:
    """Extract segment data from PDF filings via fuzzy parser.
    
    Chains: discover filings -> download PDF -> extract_segments_from_pdf().
    """
    # Reuse existing filing discovery + download infrastructure
    # Call extract_segments_from_pdf(pdf_bytes, market_id=market_id)
    # Return standard segment dict format
```

This reuses the existing `try_filing_extraction()` pattern but calls `extract_segments_from_pdf()` instead of `extract_financials_from_pdf()`.

#### Step 9: Add `extract_segment_data()` to 8 wrapper classes

For each of these wrappers, add an `extract_segment_data()` method that calls the generic helper:

| # | File | Class | Method Body |
|---|------|-------|------------|
| 1 | `in_bse.py` | `INBseClient` | `try_segment_extraction(identifier, "in_bse")` |
| 2 | `au_asx.py` | `AUAsxClient` | `try_segment_extraction(identifier, "au_asx")` |
| 3 | `ca_sedar.py` | `CASedarClient` | `try_segment_extraction(identifier, "ca_sedar")` |
| 4 | `hk_hkex.py` | `HKHkexClient` | `try_segment_extraction(identifier, "hk_hkex")` |
| 5 | `sg_sgx.py` | `SGSgxClient` | `try_segment_extraction(identifier, "sg_sgx")` |
| 6 | `za_jse.py` | `ZAJseClient` | `try_segment_extraction(identifier, "za_jse")` |
| 7 | `ae_dfm.py` | `AEDfmClient` | `try_segment_extraction(identifier, "ae_dfm")` |
| 8 | `mx_bmv.py` | Already done in Step 6 | -- |

Each method wraps the call in try/except and returns the empty segment dict on failure.

---

## Execution Order

```
Step 1:  Create operator1/features/product_metrics.py (new file)
Step 2:  Wire compute_product_metrics() in main.py after segment extraction
Step 3:  Wire compute_product_metrics() in backtest_runner.py
Step 4:  Verify _extra_vars consumption (no code change needed)
Step 5:  Verify MC segment_hhi consumption (no code change needed)
Step 6:  Add extract_segment_data() to MXBmvClient
Step 7:  Add extract_segment_data() to SATadawulClient
Step 8:  Create try_segment_extraction() in filing_discoverer.py
Step 9:  Add extract_segment_data() to 7 remaining wrapper classes (IN, AU, CA, HK, SG, ZA, AE)
```

## Files Modified

| File | Change Type | Lines Changed (est.) |
|------|------------|---------------------|
| `operator1/features/product_metrics.py` | **NEW** | ~200 |
| `main.py` | Modified | ~10 |
| `backtest_runner.py` | Modified | ~10 |
| `operator1/clients/mx_bmv.py` | Modified | ~10 |
| `operator1/clients/sa_tadawul.py` | Modified | ~20 |
| `operator1/clients/filing_discoverer.py` | Modified | ~50 |
| `operator1/clients/in_bse.py` | Modified | ~10 |
| `operator1/clients/au_asx.py` | Modified | ~10 |
| `operator1/clients/ca_sedar.py` | Modified | ~10 |
| `operator1/clients/hk_hkex.py` | Modified | ~10 |
| `operator1/clients/sg_sgx.py` | Modified | ~10 |
| `operator1/clients/za_jse.py` | Modified | ~10 |
| `operator1/clients/ae_dfm.py` | Modified | ~10 |
| **Total** | 1 new + 12 modified | ~370 |

## Risk Assessment

- **Low risk:** All changes are additive (no existing behavior modified)
- **Graceful degradation:** Every new call is wrapped in try/except; if segment extraction fails, pipeline continues with `{"available": False}`
- **No new dependencies:** Uses existing fuzzy_pdf_parser, filing_discoverer, and XBRL parsing infrastructure
- **Backward compatible:** Markets that already have `extract_segment_data()` (US, EU, BR, CH) are unaffected

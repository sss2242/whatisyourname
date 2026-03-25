# Filing Discoverer Dual-Category Enhancement

Add shareholding pattern discovery alongside financial results discovery
for all 10 filing discoverers. Each discoverer currently only queries
financial result filings -- this plan adds a parallel query for
shareholding/ownership filings.

---

## Current State

The [`FilingDiscoverer`](operator1/clients/filing_discoverer.py:83) protocol has `discover_filings()` which returns a `FilingDiscovery` with filings of type annual/quarterly/interim. Each market's discoverer filters for financial results only.

The [`FilingMetadata`](operator1/clients/filing_discoverer.py:44) dataclass already has a `filing_type` field. We need to add a new filing type: `"shareholding"`.

---

## Changes Per Discoverer

### 1. BSE India (`BSEFilingDiscoverer`)

**Current**: Queries `strCat=Result` (financial results category)

**Change**: Add a second query with `strCat=Shareholding` or `strCat=Board Meeting` to discover shareholding pattern filings. SEBI mandates quarterly shareholding pattern disclosures.

```python
# Current
params["strCat"] = "Result"

# Add
params["strCat"] = "Shareholding"  # or "Corp. Governance"
```

**Filing type**: `"shareholding"`
**Title pattern**: "Shareholding Pattern" in subject line

---

### 2. ASX Australia (`ASXFilingDiscoverer`)

**Current**: Filters for `PERIODIC REPORTS` and `ANNUAL REPORT` announcement types

**Change**: Add `SUBSTANTIAL SHAREHOLDER NOTICE` and `BECOMING A SUBSTANTIAL HOLDER` announcement types

```python
# Current headline filters
if "appendix 4e" in lower or "financial" in lower: ...

# Add
if "substantial" in lower and "holder" in lower:
    filing_type = "shareholding"
```

---

### 3. HKEX Hong Kong (`HKEXFilingDiscoverer`)

**Current**: Delegates to `HKEXScraper` with date-windowed search

**Change**: Add title filter for `"Disclosure of Interests"` and `"Major Transaction"`. HKEX filing types include `"DISCLOSURE OF INTERESTS"` category.

---

### 4. SGX Singapore (`SGXFilingDiscoverer`)

**Current**: Queries `financialreports` endpoint filtered by company name

**Change**: Also query the `announcements` endpoint with `categoryId` for `"DIRECTORS_DEALINGS"` and `"SUBSTANTIAL_SHAREHOLDERS"`. The SGX announcements API supports category filtering.

---

### 5. Tadawul Saudi Arabia (`TadawulFilingDiscoverer`)

**Current**: Queries with `category=FINANCIAL`

**Change**: Add `category=BOARD_REPORT` or `category=OWNERSHIP` to discover ownership structure disclosures. Tadawul requires listed companies to disclose major shareholders.

---

### 6. SEDAR+ Canada (`SEDARFilingDiscoverer`)

**Current**: Searches for `"Annual financial statements"` and `"Interim financial statements"` filing types

**Change**: Add `"Early warning report"` filing type (Canadian securities law requires >10% shareholders to file early warning reports on SEDAR+).

```python
for filing_type_label in [
    "Annual financial statements",
    "Interim financial statements",
    "Early warning report",        # NEW: major shareholder disclosures
]:
```

---

### 7. JSE South Africa (`JSEFilingDiscoverer`)

**Current**: Queries SENS announcements filtered by `SensAnnouncementTypeId` for financial results

**Change**: JSE SENS also publishes `"DIRECTORS DEALINGS"` and `"SHAREHOLDER SPREAD"` announcement types. Add these type IDs to the query.

---

### 8. BMV Mexico (`BMVFilingDiscoverer`)

**Current**: Searches via `busquedaPanel` for financial documents

**Change**: Also search for `"Estructura Accionaria"` (shareholder structure) documents. BMV XBRL ZIP files may already contain ownership data in some filings.

---

### 9. DFM UAE (`DFMFilingDiscoverer`)

**Current**: Queries `announcement_type=Disclosure` with financial keywords

**Change**: Also query with `announcement_type=Ownership` or filter for `"Board of Directors Report"` which typically includes ownership data.

---

### 10. SIX Switzerland (`SIXFilingDiscoverer`)

**Current**: Queries official notices API for all notice types

**Change**: The SIX Disclosure Office publishes significant shareholding notifications. Filter for `notice_type` containing `"MANAGEMENT_TRANSACTION"` or `"SIGNIFICANT_SHAREHOLDING"`. Note: the pagination bug returns 0 items for some queries -- may need date windowing.

---

## Protocol Enhancement

Add `discover_shareholding_filings()` method or add a `filing_category` parameter to `discover_filings()`:

### Option A: New method (cleaner, no breaking changes)

```python
class FilingDiscoverer:
    def discover_filings(self, ticker, years=2) -> FilingDiscovery: ...
    def discover_shareholding_filings(self, ticker, years=2) -> FilingDiscovery: ...
```

### Option B: Category parameter (more flexible)

```python
class FilingDiscoverer:
    def discover_filings(self, ticker, years=2, categories=None) -> FilingDiscovery: ...
    # categories: ["financial", "shareholding", "all"]
```

**Recommendation**: Option B is more flexible. Add `categories: list[str] | None = None` parameter with default `["financial"]` for backward compatibility. When `categories` includes `"shareholding"`, add the market-specific shareholding query.

---

## Integration with `try_filing_extraction()`

The [`try_filing_extraction()`](operator1/clients/filing_discoverer.py:2159) function currently chains: discover -> download -> extract financials. Add a parallel path:

```python
def try_shareholding_extraction(ticker, market_id, llm_client=None):
    discoverer = _get_discoverer(market_id)
    filings = discoverer.discover_filings(ticker, categories=["shareholding"])
    for filing in filings:
        pdf_bytes = discoverer.download_filing(filing)
        holders = extract_shareholders_from_pdf(pdf_bytes, market_id=market_id)
        if holders:
            return holders
    return []
```

---

## New `FilingMetadata.filing_type` Values

| Value | Description |
|-------|-------------|
| `"annual"` | Annual financial results (existing) |
| `"interim"` | Half-year / semi-annual (existing) |
| `"quarterly"` | Quarterly financial results (existing) |
| `"shareholding"` | Shareholding pattern / ownership disclosure (NEW) |
| `"insider"` | Insider / director dealings (NEW) |
| `"corporate_action"` | Mergers, acquisitions, capital changes (future) |

---

## Priority Order

| # | Market | Filing Category Param | Native Holder API Alternative |
|---|--------|-----------------------|-------------------------------|
| 1 | India BSE | `strCat=Shareholding` | BSE SAST API (already native) |
| 2 | Canada SEDAR | `Early warning report` | TMX GraphQL (already native) |
| 3 | Australia ASX | `Substantial Shareholder` | ASX substantial notices (already in discoverer) |
| 4 | Singapore SGX | `SUBSTANTIAL_SHAREHOLDERS` | SGX DOI (already native) |
| 5 | South Africa JSE | `DIRECTORS DEALINGS` | JSE SENS (already native) |
| 6 | Saudi Arabia | `category=OWNERSHIP` | Tadawul foreign ownership (already native) |
| 7 | UAE DFM | `announcement_type=Ownership` | DFM eFsah (already native) |
| 8 | Mexico BMV | `Estructura Accionaria` | BMV filing + fuzzy PDF |
| 9 | Hong Kong HKEX | `Disclosure of Interests` | akshare/EastMoney (already native) |
| 10 | Switzerland SIX | `SIGNIFICANT_SHAREHOLDING` | yfinance fallback (already native) |

**Note**: Most markets already have native holder APIs implemented. The filing discoverer shareholding category is a **supplementary path** for markets where native APIs fail or for enriching existing data with filing-level detail.

---

## Implementation Steps

1. Add `categories` parameter to `FilingDiscoverer` protocol
2. Update `FilingMetadata.filing_type` to accept `"shareholding"` and `"insider"`
3. Add `FilingDiscovery.shareholding_filings()` convenience method
4. Update each of the 10 discoverers to handle `categories=["shareholding"]`
5. Create `try_shareholding_extraction()` in filing_discoverer.py
6. Wire into each wrapper's `get_holders()` as a fallback path

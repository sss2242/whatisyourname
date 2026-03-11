# PIT Source Research: Endpoints, Taxonomy, and LLM-Aided Filing Extraction

## Part 1: Exchange Disclosure API Endpoints (Per Market)

### 1. Canada -- SEDAR+ (sedarplus.ca)

**API Endpoints:**
- Company search: `GET https://www.sedarplus.ca/csa-party/searchCompany?searchText={query}`
- Filing list: `GET https://www.sedarplus.ca/csa-party/records/filing?companyName={name}&dateFrom={YYYY-MM-DD}&dateTo={YYYY-MM-DD}`
- Filing document: `GET https://www.sedarplus.ca/csa-party/records/filing/{filingId}/document`

**Filing format:** PDF + increasingly iXBRL (Inline XBRL). iXBRL filings are machine-parseable.
**Taxonomy:** IFRS-based (Canada adopted IFRS in 2011). Tags follow IFRS taxonomy.
**Key filing types:** Annual Information Form, Financial Statements (quarterly/annual), MD&A
**Auth:** No API key required. Rate limiting appears to be ~60 req/min.
**Status:** Search endpoint already partially implemented in `ca_sedar.py`.

### 2. Australia -- ASX (asx.com.au)

**API Endpoints:**
- Company info: `GET https://www.asx.com.au/asx/1/share/{ticker}`
- Company list: `GET https://www.asx.com.au/asx/1/share/list`
- Announcements: `GET https://www.asx.com.au/asx/statistics/announcements.do?searchText={ticker}&page={n}`

**Filing format:** PDF only (Australia does not require XBRL/iXBRL for exchange filings).
**Taxonomy:** AASB (Australian Accounting Standards Board) -- equivalent to IFRS.
**Key filing types:** Annual Report, Half-Year Report, Appendix 4D/4E (results), Appendix 3Z
**Auth:** No API key. Headers must include Accept: application/json.
**Note:** Financial statement data is in PDF only -- requires LLM extraction.

### 3. Hong Kong -- HKEX (hkexnews.hk)

**API Endpoints:**
- Filing search: `GET https://www1.hkexnews.hk/search/titlesearch.xhtml?query={company}&dateRange=5y&docType=Financial%20Statements`
- Filing list: `GET https://www1.hkexnews.hk/app/appbrowse.html?doctype=financial&stock={code}`
- Document download: Direct PDF/HTML links from search results

**Filing format:** PDF + HTML. Some companies file in iXBRL (HKEX started mandating XBRL in 2022 for large-caps).
**Taxonomy:** HKFRS (Hong Kong Financial Reporting Standards) -- substantively identical to IFRS.
**Key filing types:** Annual Results, Interim Results, Annual Report
**Auth:** No API key. Web scraping may require session handling.

### 4. Singapore -- SGX (sgx.com)

**API Endpoints:**
- SGXNET announcements: `GET https://api2.sgx.com/sites/default/files/reports/full-listing-companies/sgx_full_listing.csv`
- Company announcements: `GET https://links.sgx.com/1.0.0/corporate-actions/{company_id}`

**Filing format:** PDF only.
**Taxonomy:** SFRS (Singapore Financial Reporting Standards) -- identical to IFRS.
**Key filing types:** Annual Report, Half-Year Financial Statements, Full Year Results
**Auth:** No API key.
**Note:** SGX API is poorly documented. PDF extraction required.

### 5. Mexico -- BMV/CNBV

**API Endpoints:**
- CNBV EMMA system: `https://www.bmv.com.mx/en/issuers/issuers-profile/{ticker}`
- CNBV filings: `https://emisnet.bmv.com.mx/informacion-financiera`

**Filing format:** PDF + some structured data via CNBV API.
**Taxonomy:** NIF (Normas de Informacion Financiera) -- Mexico's local GAAP, converging with IFRS.
**Auth:** No API key.
**Note:** Some structured financial data available via BMV issuer profiles.

### 6. South Africa -- JSE

**API Endpoints:**
- JSE SENS: `https://senspdf.jse.co.za/` (announcement system)
- Company page: `https://www.jse.co.za/instrument/{ticker}`

**Filing format:** PDF announcements via SENS.
**Taxonomy:** IFRS (South Africa adopted IFRS fully).
**Auth:** No API key.

### 7. Switzerland -- SIX

**API Endpoints:**
- Company search: `https://www.six-group.com/en/products-services/the-swiss-stock-exchange/market-data/shares/share-explorer.html`
- Ad-hoc disclosures: `https://www.six-group.com/en/products-services/the-swiss-stock-exchange/market-data/ad-hoc-disclosures.html`

**Filing format:** PDF. Switzerland does not require XBRL.
**Taxonomy:** Swiss GAAP FER or IFRS (issuer's choice).
**Auth:** No API key.

### 8. Saudi Arabia -- Tadawul

**API Endpoints:**
- Company disclosures: `https://www.saudiexchange.sa/wps/portal/saudiexchange/pages/companies/company-info/{id}`
- Financial statements: Available via company disclosure pages

**Filing format:** PDF (often in Arabic with English translation).
**Taxonomy:** IFRS (Saudi Arabia adopted IFRS in 2017 for listed companies).
**Auth:** No API key. Arabic language handling required.

### 9. UAE -- DFM/ADX

**API Endpoints:**
- DFM company list: `https://www.dfm.ae/listed-companies`
- ADX company disclosures: `https://www.adx.ae/English/pages/listedcompanies/`

**Filing format:** PDF (bilingual Arabic/English).
**Taxonomy:** IFRS (UAE adopted IFRS for listed companies).
**Auth:** No API key.

---

## Part 2: Taxonomy Mapping

All Tier 2 markets use either IFRS or an IFRS-equivalent standard:

| Market | Standard | Mapping to Canonical |
|--------|----------|---------------------|
| CA | IFRS | Direct -- same tags as EU ESEF |
| AU | AASB/IFRS | Direct -- same concepts |
| HK | HKFRS/IFRS | Direct -- substantively identical |
| SG | SFRS/IFRS | Direct -- identical |
| MX | NIF | Needs NIF-to-IFRS mapping (partial overlap) |
| ZA | IFRS | Direct |
| CH | IFRS or Swiss GAAP FER | IFRS direct; Swiss GAAP needs mapping |
| SA | IFRS | Direct |
| AE | IFRS | Direct |

The canonical translator already has IFRS concept mappings via [`canonical_translator.py`](operator1/clients/canonical_translator.py). For most markets, the IFRS mapping works directly. Only MX (NIF) and CH (Swiss GAAP FER) need additional concept maps.

---

## Part 3: LLM-Aided Financial Filing Extraction

### The Problem
Most Tier 2 exchanges publish financial statements as PDF documents, not structured data. Extracting revenue, total_assets, net_income etc. from PDFs is the core challenge.

### Architecture: LLM Extraction Pipeline

```
PDF/HTML Filing --> Text Extraction --> LLM Structured Extraction --> Canonical Translation --> Cache
```

**Step 1: Text Extraction**
- PDF: Use `pdfplumber` or `PyMuPDF` to extract text and tables
- HTML: Use `beautifulsoup4` (already installed) to extract tables
- iXBRL: Use `ixbrl-parse` (already installed) for tagged data

**Step 2: LLM Structured Extraction**
The app already has Gemini and Claude clients. Use them to extract structured financials:

```python
PROMPT = """Extract the following financial data from this filing text.
Return a JSON object with these fields (use null if not found):
- revenue (total revenue/sales)
- cost_of_revenue
- gross_profit
- operating_income
- net_income
- total_assets
- total_liabilities
- total_equity
- current_assets
- current_liabilities
- cash_and_equivalents
- operating_cash_flow
- capex
- free_cash_flow
- report_date (YYYY-MM-DD format)
- filing_date (YYYY-MM-DD format)
- currency (3-letter code)
- period_type (annual/quarterly/semiannual)

Filing text:
{text}
"""
```

**Step 3: Validation**
- Check accounting identity: total_assets = total_liabilities + total_equity
- Check cash flow: operating_cash_flow - capex ~= free_cash_flow
- Flag suspicious values (negative revenue, equity > assets)
- Compare with yfinance data as a cross-check

### Implementation: `llm_filing_extractor.py`

```python
class LLMFilingExtractor:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def extract_from_text(self, text: str, market_id: str) -> dict:
        """Send filing text to LLM, get structured financials back."""
        
    def extract_from_pdf(self, pdf_path: str, market_id: str) -> dict:
        """Extract text from PDF, then send to LLM."""
        
    def extract_from_html(self, html: str, market_id: str) -> dict:
        """Extract tables from HTML, then send to LLM."""
        
    def validate(self, data: dict) -> dict:
        """Validate accounting identities and flag issues."""
```

### Cost and Rate Considerations
- Gemini Flash: ~$0.075/1M tokens (very cheap for extraction)
- A typical quarterly filing is ~5,000-15,000 tokens of relevant text
- 8 filings per company (2-year quarterly) = ~80,000-120,000 tokens
- Cost per company: ~$0.01-0.02 with Gemini Flash
- Claude Haiku: ~$0.25/1M tokens (still cheap)

### Hybrid Approach (Recommended)

For each market, use the best available method in priority order:

1. **iXBRL parsing** (Canada, Hong Kong large-caps) -- structured, no LLM needed
2. **HTML table extraction** (HKEX HTML filings) -- semi-structured, light LLM validation
3. **PDF + LLM extraction** (all others) -- unstructured, full LLM extraction
4. **yfinance fallback** (when LLM extraction fails) -- non-PIT but functional

### Per-Market Implementation Priority

| Market | Method | Effort | Value |
|--------|--------|--------|-------|
| CA | iXBRL parsing + SEDAR+ API | Medium | High ($3T market) |
| HK | HTML extraction + LLM validation | Medium | High ($4.5T market) |
| AU | PDF + LLM extraction | Medium | Medium ($1.8T) |
| SG | PDF + LLM extraction | Medium | Low ($0.6T) |
| MX | PDF + LLM extraction | Medium | Low ($0.5T) |
| ZA | PDF + LLM extraction | Medium | Medium ($1T) |
| CH | PDF + LLM extraction | Medium | Medium ($1.8T) |
| SA | PDF + LLM extraction (Arabic) | High | Medium ($2.7T) |
| AE | PDF + LLM extraction (Arabic) | High | Low ($0.8T) |

---

## Part 4: Implementation Plan

### Phase 2A: LLM Filing Extractor (shared infrastructure)
1. Create `operator1/clients/llm_filing_extractor.py`
2. Add PDF text extraction using pdfplumber
3. Add HTML table extraction using BeautifulSoup
4. Build LLM prompt for structured financial extraction
5. Add validation layer (accounting identities)
6. Add `pdfplumber` to `requirements/stage4-wrappers.txt`

### Phase 2B: Canada (SEDAR+ iXBRL)
1. Add filing list endpoint to `ca_sedar.py`
2. Parse iXBRL filings using `ixbrl-parse` (already installed)
3. Fall back to LLM extraction for PDF-only filings
4. Map IFRS tags to canonical fields

### Phase 2C: Hong Kong (HKEX HTML)
1. Add filing search to `hk_hkex.py`
2. Download HTML filings from HKEX
3. Extract tables from HTML
4. Use LLM to structure financial data from tables
5. Map HKFRS tags to canonical fields

### Phase 2D: Other Markets (PDF + LLM)
- For each market: add filing list endpoint, download PDF, extract via LLM
- Priority order: AU, CH, ZA, SG, MX, SA, AE

### Files to Create/Modify

| File | Change |
|------|--------|
| `operator1/clients/llm_filing_extractor.py` | **New** -- shared LLM extraction infrastructure |
| `operator1/clients/ca_sedar.py` | Add iXBRL parsing for financials |
| `operator1/clients/hk_hkex.py` | Add HKEX filing search + HTML extraction |
| `operator1/clients/au_asx.py` | Add ASX announcement search + PDF extraction |
| `requirements/stage4-wrappers.txt` | Add pdfplumber |
| Other market clients | Add filing download + LLM extraction |

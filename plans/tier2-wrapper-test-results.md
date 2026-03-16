# Tier 2 Wrapper Test Results

Comprehensive test of all 15 Tier 2 market wrappers run on 2026-03-16.

**Environment:** Python 3.12.3, all 4 dependency stages installed.

---

## Summary

| Market | Country | Import | Instantiation | Profile | Financial Statements | OHLCV | Company Search |
|--------|---------|--------|--------------|---------|---------------------|-------|----------------|
| `ca_sedar` | Canada | OK | OK | PARTIAL (SEDAR+ API down) | EMPTY (no discoverer PDF + no LLM) | OK (yfinance) | FAIL (SEDAR+ API returns non-JSON) |
| `au_asx` | Australia | OK | OK | PARTIAL (ASX API down) | EMPTY (discoverer exists, no LLM) | OK (yfinance) | FAIL (ASX API returns non-JSON) |
| `in_bse` | India | OK | OK | WORKING (yfinance .NS) | WORKING (BSE API, 40 rows) | OK (yfinance) | OK (BSE API) |
| `cn_sse` | China | OK | OK | WORKING (baostock + yfinance) | EMPTY (baostock socket errors) | OK (baostock) | OK (baostock, by ticker only) |
| `hk_hkex` | Hong Kong | OK | OK | WORKING (yfinance .HK) | EMPTY (HKEX scraper broken, see below) | OK (yfinance) | FAIL (yfinance search broken) |
| `sg_sgx` | Singapore | OK | OK | WORKING (yfinance .SI) | EMPTY (discoverer exists, no LLM) | OK (yfinance) | FAIL (SGX API down, yfinance fallback broken) |
| `mx_bmv` | Mexico | OK | OK | WORKING (yfinance .MX) | EMPTY (discoverer exists, no LLM) | OK (yfinance) | FAIL (BMV API down, yfinance fallback broken) |
| `za_jse` | South Africa | OK | OK | WORKING (yfinance .JO) | EMPTY (discoverer exists, no LLM) | OK (yfinance) | FAIL (JSE API down, yfinance fallback broken) |
| `ch_six` | Switzerland | OK | OK | WORKING (yfinance .SW) | EMPTY (no discoverer registered) | OK (yfinance) | FAIL (SIX API down, yfinance fallback broken) |
| `sa_tadawul` | Saudi Arabia | OK | OK | WORKING (yfinance .SR) | EMPTY (discoverer exists, no LLM) | OK (yfinance) | FAIL (Tadawul API down, yfinance fallback broken) |
| `ae_dfm` | UAE | OK | OK | WORKING (yfinance .AE) | EMPTY (discoverer exists, no LLM) | OK (yfinance) | FAIL (DFM API down, yfinance fallback broken) |
| `nl_esef` | Netherlands | OK | OK | PARTIAL (ESEF API) | EMPTY (ESEF API) | OK (yfinance) | FAIL (ESEF API returns 0 results) |
| `es_esef` | Spain | OK | OK | PARTIAL (ESEF API) | EMPTY (ESEF API) | OK (yfinance) | FAIL (ESEF API returns 0 results) |
| `it_esef` | Italy | OK | OK | PARTIAL (ESEF API) | EMPTY (ESEF API) | OK (yfinance) | FAIL (ESEF API returns 0 results) |
| `se_esef` | Sweden | OK | OK | PARTIAL (ESEF API) | EMPTY (ESEF API) | OK (yfinance) | FAIL (ESEF API returns 0 results) |

---

## Capability Breakdown

### What Works (all 15 markets)

1. **Import**: All 11 unique client modules import without errors (11/11).
2. **Instantiation**: All clients instantiate correctly with proper PITClient methods (11/11).
3. **OHLCV Data**: All 15 markets return 1,000+ rows of daily price data via `ohlcv_provider.py`. Sources:
   - yfinance: CA (.TO), AU (.AX), IN (.NS), HK (.HK), SG (.SI), MX (.MX), ZA (.JO), CH (.SW), SA (.SR), AE (.AE)
   - baostock: CN (600519 format)
4. **Profile (9/15)**: Company profiles work via yfinance for 9 markets (IN, CN, HK, SG, MX, ZA, CH, SA, AE). Returns name, sector, industry, market cap.

### What Partially Works

5. **Profile (6/15)**: CA and AU return skeleton profiles (empty name) because their native APIs (SEDAR+, ASX) return non-JSON responses. NL, ES, IT, SE return partial profiles from ESEF/XBRL API.
6. **Financial Statements (1/15)**: Only India (`in_bse`) returns financial statement data, via the BSE India FinancialResult API (40 income rows, 36 balance sheet rows). This is the only Tier 2 market with a native financial API that works without an LLM.
7. **Company Search (2/15)**: Only India (BSE API) and China (baostock, ticker-only) have working company search.

### What Does Not Work

8. **Financial Statements (14/15)**: All markets except India return empty DataFrames because:
   - 9 markets have filing discoverers registered (`in_bse`, `au_asx`, `hk_hkex`, `sg_sgx`, `sa_tadawul`, `ca_sedar`, `za_jse`, `mx_bmv`, `ae_dfm`) but require an LLM client for PDF extraction. When `llm_client=None`, they return empty.
   - `ch_six` and the 4 ESEF markets (NL, ES, IT, SE) have no discoverer registered.
   - `cn_sse` uses baostock for financials but gets socket errors ("Bad file descriptor") when fetching financial data.
   - yfinance is intentionally NOT used for financial statements because it doesn't provide true filing dates (violates PIT compliance).

9. **Company Search (13/15)**: Most native API endpoints return errors:
   - SEDAR+ (`sedarplus.ca`): Returns non-JSON HTML (API may require different endpoint)
   - ASX (`asx.com.au/asx/1/share/list`): Returns non-JSON HTML
   - SGX (`api.sgx.com`): API unreachable
   - Tadawul, BMV, JSE, SIX, DFM: APIs return HTML or are blocked
   - ESEF (`filings.xbrl.org/api`): Returns 0 results for company name search
   - yfinance search fallback: Broken (returns 404 for `COMPANY_NAME.SUFFIX` format)

---

## Root Cause Analysis

### Financial Statements: Structural Limitation

The Tier 2 wrappers were designed with a deliberate architecture:
- **PIT compliance requires filing dates**, which yfinance does not provide.
- The solution is **filing discovery + LLM PDF extraction**: discover filings from the exchange's announcement API, download the PDF, and use an LLM (Gemini/Claude) to extract structured financial data.
- Without an LLM client, all filing discoverers return empty results.
- This is **by design**, not a bug -- the wrappers prioritize data integrity (PIT compliance) over availability.

### Company Search: API Endpoint Issues

Most Tier 2 native APIs either:
1. Have changed their endpoints since the wrappers were written
2. Require browser-like headers/cookies that `cached_get()` doesn't provide
3. Return HTML instead of JSON (anti-scraping measures)
4. Are geo-blocked or rate-limited

The yfinance search fallback (`yf_search`) is broken because it constructs tickers as `QUERY_TEXT.SUFFIX` (e.g., "Tencent.HK") which yfinance interprets as a ticker lookup, not a search.

---

## Filing Discoverer Registry

9 of 15 Tier 2 markets have filing discoverers registered in `DISCOVERER_REGISTRY`:

| Market | Discoverer Class | Status |
|--------|-----------------|--------|
| `in_bse` | `BSEFilingDiscoverer` | Has native BSE API (works without LLM) |
| `au_asx` | `ASXFilingDiscoverer` | Discovery works, PDF download returns 404 |
| `hk_hkex` | `HKEXFilingDiscoverer` | Requires HKEX News parsing |
| `sg_sgx` | `SGXFilingDiscoverer` | Requires SGX announcement parsing |
| `sa_tadawul` | `TadawulFilingDiscoverer` | Requires Tadawul disclosure parsing |
| `ca_sedar` | `SEDARFilingDiscoverer` | Requires SEDAR+ filing parsing |
| `za_jse` | `JSEFilingDiscoverer` | Requires JSE SENS parsing |
| `mx_bmv` | `BMVFilingDiscoverer` | Requires BMV filing parsing |
| `ae_dfm` | `DFMFilingDiscoverer` | Requires DFM/ADX filing parsing |

**Not registered:** `ch_six` (Switzerland), `nl_esef`, `es_esef`, `it_esef`, `se_esef` (all ESEF-based)

---

## Observations

- India (`in_bse`) is the only Tier 2 market with a fully working native financial API that doesn't need an LLM.
- Most native API endpoints (SEDAR+, ASX, SGX, Tadawul, BMV, JSE, SIX, DFM) return HTML instead of JSON -- likely changed endpoints or anti-scraping measures.
- The yfinance search fallback in `yfinance_backed.py` constructs invalid ticker strings (e.g. "Tencent.HK") causing 404 errors.
- baostock (China) has a session management issue -- socket errors on financial data calls after the initial login/logout cycle.
- `ch_six` (Switzerland) and the 4 ESEF-based markets (NL, ES, IT, SE) have no filing discoverer registered.
- All wrappers correctly refuse to use yfinance for financial statements (no filing dates = no PIT compliance).

### HKEX Scraper Deep Dive

`hkex_scraper.py` was built following the MIT-licensed `hkex-filing-scraper` approach. After reverse-engineering the actual HKEX page JavaScript (`titlesearch_research.js`), the real architecture is:

**Actual HKEX architecture** (NOT JSF as previously assumed):
- Page is a jQuery/AJAX app, not a JSF application
- `titlesearch_research.js` defines `loadMore()` -> `getTitleSearchCriteria()` -> `searchAgain()`
- `getTitleSearchCriteria()` reads hidden inputs: `#startDate`, `#endDate`, `#stockId`, `#selectedSecurities`, `#selectedDocType`, `#newsTitle`, `#searchTypeInt`, `#tierOneId`, `#tierTwoGpId`, `#tierTwoId`, `#lang`
- `searchAgain()` fires `$.ajax GET` to `titleSearchServlet.do` with those params
- Date format in hidden fields: `YYYY-MM-DD` (dashes stripped to `YYYYMMDD` before sending)

**Status: GEO-BLOCKED.** Tested with Playwright (full headless Chrome with JS execution):
- Set all hidden input fields correctly via `document.getElementById()`
- Called `loadMore()` which triggered real AJAX request with correct params
- Server returned `recordCnt: 0` and `result: "null"`
- Even loading the page normally in headless Chrome shows "Total records found: 0"
- **HKEX blocks non-HK IP addresses** at the server level. Returns 200 with empty results for foreign IPs.

**Code fixes applied:**
- Fixed JSF form ID extraction (now dynamic instead of hardcoded `j_idt10`)

**Notes:**
- `hk_hkex.py` doesn't use this scraper -- routes through `filing_discoverer.py` instead
- To make HKEX work: needs to be run from an HK-based IP or via HK proxy

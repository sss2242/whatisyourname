# Tier 2 Wrapper Status Review (2026-03-18)

Current state of all 15 Tier 2 market wrappers based on code review.

## Status Table

| # | Market | Profile | Company Search | Financial Statements | OHLCV | Filing Discovery |
|---|--------|---------|---------------|---------------------|-------|-----------------|
| 1 | **India** `in_bse` | BSE ComHeadernew API | BSE ListofScripData (4,800+ scrips) | BSE FinancialResult API (native) + filing discovery + LLM | yfinance (.NS) / nselib | BSEFilingDiscoverer (works globally) |
| 2 | **Australia** `au_asx` | ASX MarkitDigital API | ASX MarkitDigital directory (2,200+ companies) | Filing discovery + LLM only | yfinance (.AX) | ASXFilingDiscoverer (discovery + PDF download work) |
| 3 | **Canada** `ca_sedar` | TMX GraphQL API | TMX Directory (2,900 TSX + 1,800 TSXV) | Filing discovery + LLM only | yfinance (.TO) | SEDARFilingDiscoverer (Catalyst form POST, bypasses WAF) |
| 4 | **Hong Kong** `hk_hkex` | yfinance (.HK) | yfinance (.HK) | Filing discovery + LLM only | yfinance (.HK) | HKEXScraper (date-windowed JSON API, works globally) |
| 5 | **China** `cn_sse` | baostock + yfinance | baostock (8,600+ securities, ticker only) | akshare/Sina Finance (primary, PIT dates) + baostock fallback | baostock | N/A (uses akshare directly) |
| 6 | **Singapore** `sg_sgx` | **SGX Securities API + yfinance** | **SGX Securities API (561 stocks)** | **SGX Financial Reports API (discovery) + LLM extraction** | yfinance (.SI) | **SGXFilingDiscoverer (financialreports API, PDF download confirmed)** |
| 7 | **Saudi Arabia** `sa_tadawul` | yfinance (.SR) | Tadawul API (broken) | Filing discovery + LLM only | yfinance (.SR) | TadawulFilingDiscoverer (registered) |
| 8 | **Switzerland** `ch_six` | yfinance (.SW) | Broken | None | yfinance (.SW) | None registered |
| 9 | **South Africa** `za_jse` | **JSE WCF API (ISIN, sector, market cap) + yfinance** | **JSE Issuer Directory (298 equity issuers)** | **JSE SENS per-issuer API (instant) + LLM extraction** | yfinance (.JO) | **JSEFilingDiscoverer (WCF SENS API, per-issuer fast path, PDF download confirmed)** |
| 10 | **Mexico** `mx_bmv` | **BMV WSO2 Search API + yfinance** | **BMV Search API (244 issuers, token-based)** | **BMV XBRL JSON (native, no LLM needed, 5,817 ZIPs)** | yfinance (.MX) | **BMVFilingDiscoverer (token-based search API) + XBRL fast path** |
| 11 | **UAE** `ae_dfm` | yfinance (.AE) | DFM API (broken) | Filing discovery + LLM only | yfinance (.AE) | DFMFilingDiscoverer (registered, untested) |
| 12 | **Netherlands** `nl_esef` | ESEF partial | ESEF API (0 results) | ESEF API (0 results) | yfinance | None registered |
| 13 | **Spain** `es_esef` | ESEF partial | ESEF API (0 results) | ESEF API (0 results) | yfinance | None registered |
| 14 | **Italy** `it_esef` | ESEF partial | ESEF API (0 results) | ESEF API (0 results) | yfinance | None registered |
| 15 | **Sweden** `se_esef` | ESEF partial | ESEF API (0 results) | ESEF API (0 results) | yfinance | None registered |

---

## What Works

### Fully functional (native APIs, no LLM needed)

**India (`in_bse`)** -- best Tier 2 wrapper
- Profile: BSE ComHeadernew API (sector, industry, ISIN, EPS, PE, market cap)
- Search: BSE ListofScripData (4,800+ active equity scrips, fuzzy search by name/ticker/code)
- Financials: BSE FinancialResult API (quarterly results HTML table, ~40 income rows, ~36 balance sheet rows)
- Filing discovery: BSEFilingDiscoverer discovers and downloads PDFs from bseindia.com
- All BSE endpoints work globally without geo-blocking

**China (`cn_sse`)** -- akshare fixed the baostock socket issue
- Financials: akshare (Sina Finance API) as primary -- returns full raw line items (revenue, total_assets, cash, etc.) with PIT announcement dates (公告日期). ~100 quarterly rows per statement.
- Fallback: baostock for financial ratios when akshare fails
- Search: baostock query_stock_basic (8,600+ securities)
- Profile: baostock basic info + yfinance supplement
- Chinese field names mapped to canonical schema (48 fields across 3 statements)
- Works globally (Sina Finance is not geo-blocked)

**Singapore (`sg_sgx`)** -- newly fixed
- Profile: SGX Securities API (name, ticker, currency, board) + yfinance supplement for sector/industry
- Search: SGX Securities API directory (561 stocks, client-side fuzzy match by ticker or name)
- Filing discovery: SGX Financial Reports API (12,674 reports, `companyname` filter). Two-step PDF download (HTML page -> PDF links). DBS Annual Report 2025 = 8.3MB valid PDF.
- Financials: need LLM client for PDF content extraction (discovery + download both work)

### Profile + Search working (native APIs)

**Australia (`au_asx`)**
- Profile: ASX MarkitDigital header API (name, sector, industry, market cap, listing date). No yfinance dependency.
- Search: ASX MarkitDigital directory API (2,200+ companies, client-side filtering)
- Filing discovery: ASXFilingDiscoverer works, PDF download confirmed (3.5MB)
- Financials: need LLM client for PDF extraction

**Canada (`ca_sedar`)**
- Profile: TMX GraphQL API (sector, industry, name, price). Works globally without auth.
- Search: TMX Company Directory (2,900 TSX + 1,800 TSXV companies, paginated A-Z, client-side fuzzy search)
- Filing discovery: SEDAR+ Catalyst form POST (bypasses F5 WAF). Tested live: Royal Bank of Canada returns 60 filings. PDF download confirmed (142KB).
- Financials: need LLM client for PDF extraction
- First pure-requests SEDAR+ implementation (community projects all use Selenium)

### Profile + Search + Filing discovery working (native API, J-Quants speed)

**South Africa (`za_jse`)** -- newly fixed (WCF API discovery)
- Profile: JSE Issuer Directory (298 equity issuers, name, ticker, MasterID, registration, contacts) + Instruments API (ISIN, sector, industry, market cap, price, board, listing date). Richer than yfinance.
- Search: JSE CustomerRoleService.svc/GetAllIssuers (client-side fuzzy match, cached). Confirmed: NPN (Naspers), SOL (Sasol), SBK (Standard Bank).
- Filing discovery: JSE SENSService.svc/GetSensAnnouncementsByIssuerMasterId (single API call per issuer, instant, like J-Quants). Returns 15 announcements with PDFPath URLs.
- PDF download: senspdf.jse.co.za/documents/SENS_*.pdf (165KB Naspers PDF confirmed)
- No authentication required (public SharePoint WCF endpoints)
- Financials: need LLM client for PDF extraction (discovery + download both work)

**Mexico (`mx_bmv`)** -- newly fixed (HKEX pattern + XBRL extraction)
- Profile: BMV WSO2 Search API (name, ticker, series, market, status, company IDs) + yfinance supplement for sector/industry
- Search: BMV Search API via ElasticSearch backend (244 issuers, dual search type: `busquedaClaveCotizacion` for instruments, `busquedaPanel` for documents). All major tickers confirmed: AMX, WALMEX, CEMEX, BIMBO, FEMSA.
- **Financials: BMV XBRL JSON extraction (fast path, NO LLM NEEDED)**. The BMV XBRL page contains 5,817 XBRL JSON ZIPs for 244 issuers. Each ZIP (~767KB) contains structured IFRS financial data extracted directly. Tested: WALMEX (115 income rows, 114 balance rows, 55 cashflow rows), AMX, CEMEX -- all with revenue, total_assets, cash, EPS, etc.
- Filing discovery (fallback): BMVFilingDiscoverer uses `busquedaPanel` search type for PDF documents.
- Token-based auth: `GET /rest/tokenservice/token` returns Bearer token (no API key needed).

### Filing discovery working (scraper fixed)

**Hong Kong (`hk_hkex`)**
- HKEX scraper FIXED: uses date-windowed queries (2-week windows) to undocumented JSON API (`titleSearchServlet.do`). NOT geo-blocked as previously thought -- the server-side date range limit of ~15 days was the issue, not IP blocking.
- Profile + search: yfinance (.HK) fallback
- Financials: need LLM client for PDF extraction from discovered filings

### OHLCV only (yfinance fallback)

All 15 markets return daily OHLCV data via yfinance or regional wrappers (baostock for CN).

---

## What Doesn't Work

### Financial statements (10/15 need LLM)
- AU, CA, HK, SG, SA, ZA, MX, AE: have filing discoverers (discovery + PDF download work) but need an LLM client (Gemini/Claude/OpenRouter API key) to extract structured data from PDFs
- CH, NL, ES, IT, SE: no filing discoverer registered, no native financial API
- yfinance intentionally NOT used for financials (no filing dates = no PIT compliance)

### Company search (4/15 broken)
- SA (Tadawul returns HTML), AE (DFM API broken), CH (SIX API broken): native APIs return HTML or are unreachable
- NL, ES, IT, SE: ESEF API returns 0 results for company name searches
- MX: **FIXED** -- BMV WSO2 Search API now works (token-based, no key needed)
- ZA: **FIXED** -- JSE WCF CustomerRoleService returns 298 equity issuers

### ESEF markets (4/15 barely functional)
- Netherlands, Spain, Italy, Sweden all use the same `eu_esef_wrapper` which queries `filings.xbrl.org/api`
- The API returns partial profiles but 0 results for company search and financial statements
- No filing discoverer registered for any of these markets

---

## Summary by Capability

| Capability | Working | Partial | Broken |
|-----------|---------|---------|--------|
| Profile | IN, CN, AU, CA, SG, **MX**, **ZA** + 4 via yfinance | NL, ES, IT, SE (ESEF partial) | -- |
| Company Search | IN, CN, AU, CA, SG, **MX**, **ZA** | -- | HK (yfinance only), SA, CH, AE, NL, ES, IT, SE |
| Financial Statements | IN (native), CN (akshare), **MX (XBRL JSON)** | AU, CA, HK, SG, SA, **ZA**, AE (need LLM) | CH, NL, ES, IT, SE (no path) |
| OHLCV | All 15 | -- | -- |
| Filing Discovery | IN, AU, CA, HK, SG, **MX**, SA, **ZA**, AE | -- | CH, NL, ES, IT, SE (none registered) |

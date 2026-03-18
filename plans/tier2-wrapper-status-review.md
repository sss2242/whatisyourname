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
| 6 | **Singapore** `sg_sgx` | yfinance (.SI) | SGX API (broken) | Filing discovery + LLM only | yfinance (.SI) | SGXFilingDiscoverer (registered) |
| 7 | **Saudi Arabia** `sa_tadawul` | yfinance (.SR) | Tadawul API (broken) | Filing discovery + LLM only | yfinance (.SR) | TadawulFilingDiscoverer (registered) |
| 8 | **Switzerland** `ch_six` | yfinance (.SW) | Broken | None | yfinance (.SW) | None registered |
| 9 | **South Africa** `za_jse` | yfinance (.JO) | JSE API (broken) | Filing discovery + LLM only | yfinance (.JO) | JSEFilingDiscoverer (registered, untested) |
| 10 | **Mexico** `mx_bmv` | yfinance (.MX) | BMV API (broken) | Filing discovery + LLM only | yfinance (.MX) | BMVFilingDiscoverer (registered, untested) |
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
- AU, CA, HK, SG, SA, ZA, MX, AE: have filing discoverers but need an LLM client (Gemini/Claude/OpenRouter API key) to extract structured data from PDFs
- CH, NL, ES, IT, SE: no filing discoverer registered, no native financial API
- yfinance intentionally NOT used for financials (no filing dates = no PIT compliance)

### Company search (7/15 broken)
- SG (SGX API unreachable), SA (Tadawul returns HTML), ZA (JSE API broken), MX (BMV API broken), AE (DFM API broken), CH (SIX API broken): native APIs return HTML or are unreachable
- NL, ES, IT, SE: ESEF API returns 0 results for company name searches

### ESEF markets (4/15 barely functional)
- Netherlands, Spain, Italy, Sweden all use the same `eu_esef_wrapper` which queries `filings.xbrl.org/api`
- The API returns partial profiles but 0 results for company search and financial statements
- No filing discoverer registered for any of these markets

---

## Summary by Capability

| Capability | Working | Partial | Broken |
|-----------|---------|---------|--------|
| Profile | IN, CN, AU, CA + 7 via yfinance | NL, ES, IT, SE (ESEF partial) | -- |
| Company Search | IN, CN, AU, CA | -- | HK (yfinance only), SG, SA, CH, ZA, MX, AE, NL, ES, IT, SE |
| Financial Statements | IN (native), CN (akshare) | AU, CA, HK, SG, SA, ZA, MX, AE (need LLM) | CH, NL, ES, IT, SE (no path) |
| OHLCV | All 15 | -- | -- |
| Filing Discovery | IN, AU, CA, HK, SG, SA, ZA, MX, AE | -- | CH, NL, ES, IT, SE (none registered) |

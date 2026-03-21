# Tier 2 Wrapper Status Review (2026-03-19)

Current state of all 15 Tier 2 market wrappers based on code review.

## Status Table

| # | Market | Profile | Company Search | Financial Statements | OHLCV | Filing Discovery |
|---|--------|---------|---------------|---------------------|-------|-----------------|
| 1 | **India** `in_bse` | BSE ComHeadernew API | BSE ListofScripData (4,800+ scrips) | BSE FinancialResult API (native) + filing discovery + LLM | yfinance (.NS) / nselib | BSEFilingDiscoverer (works globally) |
| 2 | **Australia** `au_asx` | ASX MarkitDigital API | ASX MarkitDigital directory (2,200+ companies) | LLMFilingExtractor + fuzzy_pdf_parser fallback (no LLM needed) | yfinance (.AX) | ASXFilingDiscoverer (discovery + PDF download work) |
| 3 | **Canada** `ca_sedar` | TMX GraphQL API | TMX Directory (2,900 TSX + 1,800 TSXV) | LLMFilingExtractor + fuzzy_pdf_parser fallback (no LLM needed) | yfinance (.TO) | SEDARFilingDiscoverer (Catalyst form POST, bypasses WAF) |
| 4 | **Hong Kong** `hk_hkex` | **akshare/EastMoney profile + yfinance enrichment** | yfinance (.HK) | **akshare/EastMoney (native, no LLM, ~4,000 rows/company)** + HKEX filing discovery fallback | yfinance (.HK) | HKEXScraper (date-windowed JSON API, works globally) |
| 5 | **China** `cn_sse` | baostock + yfinance | baostock (8,600+ securities, ticker only) | akshare/Sina Finance (primary, PIT dates) + baostock fallback | baostock | N/A (uses akshare directly) |
| 6 | **Singapore** `sg_sgx` | **SGX Securities API + yfinance** | **SGX Securities API (561 stocks)** | **LLMFilingExtractor + fuzzy_pdf_parser fallback (no LLM needed)** | yfinance (.SI) | **SGXFilingDiscoverer (financialreports API, PDF download confirmed)** |
| 7 | **Saudi Arabia** `sa_tadawul` | **ThemeSearchUtilityServlet (1,949 companies, ISIN) + TASI sector enrichment + TickerServlet (live prices)** | **ThemeSearchUtilityServlet (1,949 companies) + TASI WPS portal (269 with sectors), fuzzy search** | **XBRL HTML extraction (native, no LLM needed, IFRS income/balance/cashflow with PIT dates)** | yfinance (.SR) | TadawulFilingDiscoverer (fallback for LLM extraction) |
| 8 | **Switzerland** `ch_six` | **SIX FQS + Share Details APIs (no auth)** | **SIX FQS ref.json (110K+ securities)** | **Synthetic from SIX dividends/capital/notices + Kalman/PELT/Merton/L1 (22 fields, no LLM)** | **SIX historic CSV (close+volume, ~5mo)** | ESEF crossover + LLM fallback |
| 9 | **South Africa** `za_jse` | **JSE WCF API (ISIN, sector, market cap) + yfinance** | **JSE Issuer Directory (298 equity issuers)** | **LLMFilingExtractor + fuzzy_pdf_parser fallback (no LLM needed)** | yfinance (.JO) | **JSEFilingDiscoverer (WCF SENS API, per-issuer fast path, PDF download confirmed)** |
| 10 | **Mexico** `mx_bmv` | **BMV WSO2 Search API + yfinance** | **BMV Search API (244 issuers, token-based)** | **BMV XBRL JSON (native, no LLM needed, 5,817 ZIPs)** | yfinance (.MX) | **BMVFilingDiscoverer (token-based search API) + XBRL fast path** |
| 11 | **UAE** `ae_dfm` | **DFM api2 stocks + yfinance** | **DFM api2 (461 securities) + Nuxt SSR (510 companies)** | **LLMFilingExtractor + fuzzy_pdf_parser fallback (no LLM needed)** | yfinance (.AE) | **DFMFilingDiscoverer (eFsah API, PDF download confirmed)** |
| 12 | **Netherlands** `nl_esef` | **ESEF entity (filings.xbrl.org)** | **ESEF directory (157 entities, fuzzy match)** | **XBRL JSON native (no LLM, 32 IFRS fields)** | yfinance | ESEF XBRL JSON (inline) |
| 13 | **Spain** `es_esef` | **ESEF entity** | **ESEF directory (133 entities, fuzzy match)** | **XBRL JSON native (no LLM, 32 IFRS fields)** | yfinance | ESEF XBRL JSON (inline) |
| 14 | **Italy** `it_esef` | **ESEF entity** | **ESEF directory (202 entities, fuzzy match)** | **XBRL JSON native (no LLM, 32 IFRS fields)** | yfinance | ESEF XBRL JSON (inline) |
| 15 | **Sweden** `se_esef` | **ESEF entity** | **ESEF directory (343 entities, fuzzy match)** | **XBRL JSON native (no LLM, 32 IFRS fields)** | yfinance | ESEF XBRL JSON (inline) |

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
- Financials: LLMFilingExtractor (primary) + fuzzy_pdf_parser fallback (no LLM needed, camelot-py + fuzzy match)

### Profile + Search working (native APIs)

**Australia (`au_asx`)**
- Profile: ASX MarkitDigital header API (name, sector, industry, market cap, listing date). No yfinance dependency.
- Search: ASX MarkitDigital directory API (2,200+ companies, client-side filtering)
- Filing discovery: ASXFilingDiscoverer works, PDF download confirmed (3.5MB)
- Financials: LLMFilingExtractor (primary) + fuzzy_pdf_parser fallback (no LLM needed)

**Canada (`ca_sedar`)**
- Profile: TMX GraphQL API (sector, industry, name, price). Works globally without auth.
- Search: TMX Company Directory (2,900 TSX + 1,800 TSXV companies, paginated A-Z, client-side fuzzy search)
- Filing discovery: SEDAR+ Catalyst form POST (bypasses F5 WAF). Tested live: Royal Bank of Canada returns 60 filings. PDF download confirmed (142KB).
- Financials: LLMFilingExtractor (primary) + fuzzy_pdf_parser fallback (no LLM needed)
- First pure-requests SEDAR+ implementation (community projects all use Selenium)

### Profile + Search + Filing discovery working (native API, J-Quants speed)

**South Africa (`za_jse`)** -- newly fixed (WCF API discovery)
- Profile: JSE Issuer Directory (298 equity issuers, name, ticker, MasterID, registration, contacts) + Instruments API (ISIN, sector, industry, market cap, price, board, listing date). Richer than yfinance.
- Search: JSE CustomerRoleService.svc/GetAllIssuers (client-side fuzzy match, cached). Confirmed: NPN (Naspers), SOL (Sasol), SBK (Standard Bank).
- Filing discovery: JSE SENSService.svc/GetSensAnnouncementsByIssuerMasterId (single API call per issuer, instant, like J-Quants). Returns 15 announcements with PDFPath URLs.
- PDF download: senspdf.jse.co.za/documents/SENS_*.pdf (165KB Naspers PDF confirmed)
- No authentication required (public SharePoint WCF endpoints)
- Financials: LLMFilingExtractor (primary) + fuzzy_pdf_parser fallback (no LLM needed)

**Mexico (`mx_bmv`)** -- newly fixed (HKEX pattern + XBRL extraction)
- Profile: BMV WSO2 Search API (name, ticker, series, market, status, company IDs) + yfinance supplement for sector/industry
- Search: BMV Search API via ElasticSearch backend (244 issuers, dual search type: `busquedaClaveCotizacion` for instruments, `busquedaPanel` for documents). All major tickers confirmed: AMX, WALMEX, CEMEX, BIMBO, FEMSA.
- **Financials: BMV XBRL JSON extraction (fast path, NO LLM NEEDED)**. The BMV XBRL page contains 5,817 XBRL JSON ZIPs for 244 issuers. Each ZIP (~767KB) contains structured IFRS financial data extracted directly. Tested: WALMEX (115 income rows, 114 balance rows, 55 cashflow rows), AMX, CEMEX -- all with revenue, total_assets, cash, EPS, etc.
- Filing discovery (fallback): BMVFilingDiscoverer uses `busquedaPanel` search type for PDF documents.
- Token-based auth: `GET /rest/tokenservice/token` returns Bearer token (no API key needed).

### Profile + Search + Synthetic Financials (native APIs, no LLM needed)

**Switzerland (`ch_six`)** -- fully native SIX APIs, no yfinance dependency
- Profile: SIX FQS ref.json (ValorId, ISIN, ticker, name for 110K+ securities) + Share Details API (shares outstanding, indices, regulatory standard, nominal value, capital structure, dividends, auditor, accounting standard) + historic CSV (latest close for market cap). Sector inferred from SMI/SLI index membership + regulatory standard + name heuristics.
- Search: SIX FQS ref.json (ticker exact match, ISIN lookup, name substring search). Covers 110K+ securities on SIX. No auth required.
- OHLCV: SIX historic CSV (`/sheldon/market_data/v1/{ValorId}/historic.csv`) provides ~5 months of close+volume. PIT-compliant exchange data. No open/high/low (set to close for compatibility).
- Financials: Synthetic generation via `six_derived_proxies.py` -- derives 22 canonical fields mathematically from 18 years of dividend history, capital structure, buyback notices, and sector-calibrated ratios. Uses Kalman filter (Harvey 1989), PELT regime detection (Killick 2012), UKF two-factor earnings, Merton structural debt model (1974), L1 balance sheet reconstruction (Candes & Tao 2005), EBO reverse earnings, Ohlson residual income equity. Fallbacks: EU ESEF crossover, then LLM filing extraction.
- TDA: Persistent homology on dividend trajectories via ripser (Bauer 2021) for monotonicity/cyclicality detection.
- No authentication required for any SIX endpoint.

### Profile + Search + Native XBRL Financials (curl_cffi, no LLM needed)

**Saudi Arabia (`sa_tadawul`)** -- newly fixed (full native implementation)
- Profile: ThemeSearchUtilityServlet (1,949 companies with ISIN, companyNameEN, tradingNameEn) + TASI WPS portal (269 companies with sectorName) + TickerServlet (live prices, volume, change%). All via curl_cffi Chrome TLS impersonation.
- Search: ThemeSearchUtilityServlet directory (exact symbol + substring + rapidfuzz WRatio fuzzy match). Covers 1,949 TASI + NOMU companies.
- Financials: XBRL HTML extraction from `/Resources/XBRL_DOCS/` via statementsTabData portal resource. Structured IFRS tables (income, balance, cashflow) with PIT filing dates. 48+ canonical fields mapped (revenue, total_assets, eps, operating_cash_flow, etc.). NO LLM NEEDED.
- OHLCV: yfinance (.SR suffix) via ohlcv_provider
- Filing discovery: TadawulFilingDiscoverer as fallback (PDF links from /Resources/fsPdf/)
- All endpoints require curl_cffi with Chrome TLS impersonation (plain requests gets 403 from Akamai WAF)

### Profile + Financials working (akshare/EastMoney native, no LLM needed)

**Hong Kong (`hk_hkex`)** -- newly upgraded with akshare/EastMoney
- Profile: akshare stock_hk_company_profile_em (English name, Chinese name, registration, industry, chairman, employees, website, auditor, fiscal year end) + yfinance enrichment (sector, industry, market cap, shares outstanding, ISIN)
- Search: yfinance (.HK) fallback
- Financials: akshare stock_financial_hk_report_em as primary (EastMoney structured data, ~4,000 rows per company, balance/income/cashflow with STD_ITEM_CODE mapping to 50+ canonical fields). Fast (~2s), no LLM needed. Works globally.
- Filing discovery fallback: HKEX scraper uses date-windowed queries (2-week windows) to undocumented JSON API (`titleSearchServlet.do`). NOT geo-blocked -- the server-side date range limit of ~15 days was the issue, not IP blocking.
- Note: EastMoney data doesn't include filing_date (announcement date). For true PIT compliance, the HKEX filing discovery path provides filing_dates from HKEX announcement timestamps.

### OHLCV (all 15 markets)

All 15 markets return daily OHLCV data via yfinance or regional wrappers (baostock for CN, SIX CSV for CH).

---

## What Doesn't Work

### Financial statements (0/15 broken -- all have extraction paths)
- IN (native BSE API), CN (akshare), **MX (XBRL JSON)**, **SA (XBRL HTML)**, **CH (synthetic)**, **NL/ES/IT/SE (ESEF XBRL JSON)**: fully native, no LLM needed
- AU, CA, HK, SG, **ZA**, **AE**: filing discovery + PDF download work. Extraction via **LLMFilingExtractor** (primary, uses Gemini/Claude/OpenRouter with market-specific taxonomy hints) OR **fuzzy_pdf_parser** (automatic fallback, no LLM needed -- uses camelot-py + pdfplumber + fuzzy string matching with ~98% accuracy on SEBI/IFRS format results)
- The `try_filing_extraction()` function automatically chains: discovery -> download -> LLM extraction -> fuzzy PDF fallback. All 6 filing-discoverer-backed markets work without an LLM key via the fuzzy parser.
- yfinance intentionally NOT used for financials (no filing dates = no PIT compliance)

### Company search (0/15 broken)
- SA: **FIXED** -- ThemeSearchUtilityServlet (1,949 companies) + TASI WPS portal (269 with sectors), curl_cffi Chrome impersonation, fuzzy search via rapidfuzz
- AE: **FIXED** -- DFM api2 stocks (461 securities) + Nuxt SSR (510 companies)
- NL, ES, IT, SE: **FIXED** -- ESEF entity directory (133-343 entities per country, fuzzy match with rapidfuzz WRatio)
- CH: **FIXED** -- SIX FQS ref.json (110K+ securities, ticker/ISIN/name search, no auth)
- MX: **FIXED** -- BMV WSO2 Search API now works (token-based, no key needed)
- ZA: **FIXED** -- JSE WCF CustomerRoleService returns 298 equity issuers

### ESEF markets (4/15 now functional with XBRL JSON)
- Netherlands (599 filings), Spain (542), Italy (754), Sweden (1,415) now have **structured IFRS financial data** via XBRL JSON extraction from filings.xbrl.org
- Entity search uses `include=entity` sideload + rapidfuzz WRatio fuzzy matching
- 32 IFRS concepts mapped to canonical fields (revenue, net_income, total_assets, equity, cash, etc.)
- Uses `xbrl-filings-api==1.0` library for entity.name exact match + raw API fallback
- Tested: Heineken NL (42 income + 64 balance facts), BNP PARIBAS FR (47+82), Unilever NL (32+41)
- **Germany (DE): 0 filings** on filings.xbrl.org -- not available via ESEF

---

## Summary by Capability

| Capability | Working | Partial | Broken |
|-----------|---------|---------|--------|
| Profile | IN, CN, AU, CA, SG, **MX**, **ZA**, **CH**, **AE**, **SA**, **HK** (akshare+yfinance), **NL**, **ES**, **IT**, **SE** | -- | -- |
| Company Search | IN, CN, AU, CA, SG, **MX**, **ZA**, **CH**, **AE**, **SA**, **NL**, **ES**, **IT**, **SE** | HK (yfinance only) | -- |
| Financial Statements | IN (native), CN (akshare), **HK (akshare/EastMoney)**, **MX (XBRL JSON)**, **CH (synthetic)**, **SA (XBRL HTML)**, **NL/ES/IT/SE (ESEF XBRL JSON)** | AU, CA, SG, **ZA**, **AE** (LLM primary + fuzzy PDF fallback) | -- |
| OHLCV | All 15 (**CH**: SIX CSV, **AE**: api2 snapshot + yfinance) | -- | -- |
| Filing Discovery | IN, AU, CA, HK, SG, **MX**, **SA**, **ZA**, **AE**, **NL/ES/IT/SE** (ESEF XBRL) | **CH** (ESEF + LLM) | -- |

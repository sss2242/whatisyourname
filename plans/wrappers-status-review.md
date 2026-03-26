# Wrappers Status Review (2026-03-21)

Current state of all Tier 1 and Tier 2 market wrappers based on code review.

---

# Tier 1 Wrappers (10 markets, $91T+ coverage)

## Working Tier 1 Wrappers

| # | Market | ID | Profile | Company Search | Financial Statements | OHLCV | Key Required |
|---|--------|----|---------|---------------|---------------------|-------|-------------|
| 1 | **United States** | `us_sec_edgar` | edgartools (XBRL-parsed profile, sector, industry, CIK, SIC) | edgartools full-text search + sec-edgar-api EFTS | edgartools XBRL (income, balance, cashflow with filing_date + report_date, PIT-compliant) | yfinance (fallback via ohlcv_provider) | No (email User-Agent only) |
| 2 | **United Kingdom** | `uk_companies_house` | Companies House REST API (name, SIC codes, registered office, accounts) | Companies House search API (5M+ companies) | iXBRL document parsing (UK-GAAP/FRS 102/IFRS) + filing history | yfinance (.L) | Yes (free API key) |
| 3 | **EU (pan-EU)** | `eu_esef` | ESEF entity from filings.xbrl.org | ESEF entity directory (7,200+ filers, fuzzy match) | XBRL JSON native (32 IFRS fields, no LLM needed) | yfinance | No |
| 4 | **France** | `fr_esef` | ESEF entity (FR country filter) | ESEF directory (FR entities) | XBRL JSON native (32 IFRS fields) | yfinance (.PA) | No |
| 5 | **Germany** | `de_esef` | ESEF entity (DE country filter) | ESEF directory (DE entities) | **Bundesanzeiger scraper (ONNX captcha solver + HTML table extraction, 60+ German field mappings)** -- filings.xbrl.org has 0 DE filings, falls back to Bundesanzeiger. Tested: Siemens Pensionsfonds AG, 7 balance fields x 3 periods. | yfinance (.DE) | No |
| 6 | **Japan** | `jp_jquants` | J-Quants company info API (name, sector, industry, market cap) | J-Quants listed info (3,800+ TSE companies) | J-Quants financial summary (structured, quarterly/annual) via _JPJquantsAdapter | yfinance (.T) via ohlcv_provider | Yes (free registration) |
| 7 | **South Korea** | `kr_dart` | dart-fss corporate code lookup + profile | dart-fss corp code search (2,600+ listed) | dart-fss XBRL financial statements (income, balance, cashflow with filing_date) | pykrx (primary) / yfinance (.KS) | Yes (free DART API key) |
| 8 | **Taiwan** | `tw_mops` | MOPS form POST (company basic info, ROC date conversion) | MOPS company list scraping (1,700+ TWSE/TPEX) | MOPS form POST scraping (quarterly financials, ROC date -> Gregorian conversion) | twstock (primary) / yfinance (.TW) | No |
| 9 | **Brazil** | `br_cvm` | CVM CSV registry (2,600+ companies) + FCA ZIP (sector, industry, ownership, website, founding date) | CVM registry search (name, ticker, CNPJ, CD_CVM) + B3 ticker resolution | **CVM ZIP archives (native, no LLM, DFP annual + ITR quarterly, 32+ canonical fields, DT_RECEB PIT dates)** -- live tested: Petrobras 12 periods income, 9 periods balance, 12 periods cashflow | yfinance (.SA) | No |
| 10 | **Chile** | `cl_cmf` | yfinance (.SN) fallback for profile | yfinance (.SN) fallback | **US ADR fallback** -- CMF opendata/FECU all return 404 (site restructuring since 2025). Major Chilean companies are fetched via their NYSE ADR tickers (SQM, LTM, BSAC, BCH, CCU, ENIC, CNCO) through SEC EDGAR (primary, PIT) or yfinance (fallback). Tested: SQM 106 balance + 88 income rows. | yfinance (.SN) | No |

## Not Working Tier 1 Wrappers

| # | Market | ID | Issue |
|---|--------|----|-------|
| -- | -- | -- | No Tier 1 wrappers are fully broken as of 2026-03-23. Chile uses US ADR fallback. |

## Recently Fixed Tier 1 Wrappers

| # | Market | ID | Fix |
|---|--------|----|-----|
| 1 | **Germany** | `de_esef` | **FIXED (2026-03-23)** -- Added Bundesanzeiger scraper as fallback when filings.xbrl.org returns 0 DE filings. Uses community-trained ONNX neural network (dre808/bundesanzeiger-scraper, MIT) to solve Bundesanzeiger CAPTCHAs automatically (~60-70% per attempt, 3 retries). Extracts financial data from HTML tables with 60+ German field name mappings (Bilanz/GuV/Kapitalflussrechnung to canonical English). HKEX-style AJAX bypass was attempted (all 5 patterns: session init, XMLHttpRequest, date-windowed queries, internal Wicket endpoints, NLP/CSV direct access) -- Bundesanzeiger enforces captcha server-side on ALL document paths unlike HKEX. Tested: Siemens Pensionsfonds AG, 7 balance fields x 3 periods. |
| 2 | **Chile** | `cl_cmf` | **FIXED (2026-03-23)** -- CMF website restructuring broke all data endpoints (opendata, FECU, RGALS, SEIL all 404). No community scrapers exist. Bolsa de Santiago is behind hCaptcha. Solution: US ADR fallback chain. Major Chilean companies (SQM, LATAM, Santander Chile, Banco de Chile, CCU, Enel Chile, Cencosud) have NYSE ADR listings. Fallback: (1) SEC EDGAR 20-F filings via edgartools (IFRS, PIT dates), (2) yfinance on ADR ticker (no PIT but full data). ADR resolution covers ~95% of Santiago market cap. Tested: SQM 106 balance + 88 income rows, LATAM 111 balance rows, BCH 52 income rows. |

**Note:** 10 of 10 Tier 1 wrappers now have financial statement extraction paths. Germany uses Bundesanzeiger with ONNX captcha solver. Chile uses US ADR fallback (SEC EDGAR primary, yfinance secondary) while CMF rebuilds their data services.

---

# Tier 2 Wrappers (15 markets)

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
- **Germany (DE): 0 filings** on filings.xbrl.org -- **FIXED**: falls back to Bundesanzeiger scraper (ONNX captcha + HTML table extraction, 60+ German field mappings)

---

## Portfolio / Holder Data Coverage (updated 2026-03-25)

The `PITClient` protocol defines three holder-related methods: `get_holders()`, `get_holder_history()`, and `get_insider_transactions()`. All 25 wrappers now implement these methods, with varying levels of native vs stub coverage.

### Table 1: Wrappers WITH Native Portfolio/Holder Data (16 markets)

| # | Market | ID | Tier | `get_holders()` | Native Source | `get_holder_history()` | Native Source | `get_insider_transactions()` | Native Source |
|---|--------|----|------|-----------------|---------------|------------------------|---------------|------------------------------|---------------|
| 1 | **United States** | `us_sec_edgar` | 1 | Yes | SEC EDGAR SC 13D/13G filings (>5% beneficial owners) + DEF 14A proxy via LLM/fuzzy PDF | Yes | SEC EDGAR filing-derived snapshot | Yes | SEC EDGAR Form 4 filings |
| 2 | **United Kingdom** | `uk_companies_house` | 1 | Yes | **Native PSC API** (>25% control, name, natures_of_control) | Yes | PSC-derived snapshot | No | -- |
| 3 | **South Korea** | `kr_dart` | 1 | Yes | **Native DART API** (hyslr_sttus, >5% shareholders) | Yes | **Native DART quarterly** (up to 8 quarters, per-period HHI) | No | -- |
| 4 | **Japan** | `jp_jquants` | 1 | Yes | J-Quants financial summary (major shareholder data) | Yes | J-Quants / filing-derived snapshot | Yes | J-Quants / filing discovery |
| 5 | **Taiwan** | `tw_mops` | 1 | Yes | **TWSE OpenAPI** (director/supervisor shareholding, major shareholders) | Yes | TWSE director shareholding data | Yes | **TWSE OpenAPI t187ap12_L** (insider transfer notifications) |
| 6 | **Brazil** | `br_cvm` | 1 | Yes | **CVM FRE archives** (posicao_acionaria CSV -- name, shares, %, controlling flag) | Yes | CVM FRE-derived snapshot | Yes | CVM FRE archives (insider transactions) |
| 7 | **India** | `in_bse` | 2 | Yes | **BSE SAST disclosures** (Reg. 29/31) + BSE BoardMeetings PDF scraping (SEBI shareholding patterns) | Yes | BSE SAST-derived snapshot | Yes | BSE SAST announcements |
| 8 | **Hong Kong** | `hk_hkex` | 2 | Yes | **akshare/EastMoney** (stock_hk_holders_em, structured) | Yes | akshare/EastMoney-derived snapshot | Yes | HKEX disclosure filings |
| 9 | **Australia** | `au_asx` | 2 | Yes | ASX substantial holder notices via filing discovery + LLM/fuzzy PDF | Yes | ASX filing-derived snapshot | Yes | ASX director dealings announcements |
| 10 | **Canada** | `ca_sedar` | 2 | Yes | **TMX GraphQL API** (insider activity summary, SEDI data) | Yes | TMX insider activity-derived snapshot | Yes | TMX GraphQL individual insider transactions |
| 11 | **Saudi Arabia** | `sa_tadawul` | 2 | Yes | **Tadawul native HTML** (foreign ownership table, 411 companies) via curl_cffi | Yes | Tadawul foreign ownership-derived snapshot | Yes | Tadawul statementsTabData type=5 (board reports) |
| 12 | **South Africa** | `za_jse` | 2 | Yes | **JSE SENS** shareholding announcements | Yes | JSE SENS-derived snapshot | Yes | JSE SENS director dealings |
| 13 | **UAE** | `ae_dfm` | 2 | Yes | **DFM eFsah API** (holder-related disclosures) | Yes | DFM eFsah-derived snapshot | Yes | DFM eFsah BOD meeting results |
| 14 | **Mexico** | `mx_bmv` | 2 | Yes | BMV filing discovery + LLM/fuzzy PDF extraction | Yes | BMV filing-derived snapshot | Yes | BMV filing discovery (insider filings) |
| 15 | **Singapore** | `sg_sgx` | 2 | Yes | **SGX DOI announcements** (Section 137 SFA) + **two-stage annual report PDF extraction** (Stage 1: fuzzy_pdf_parser for financials, Stage 2: page-level shareholding with section-isolated regex). **No yfinance** (removed 2026-03-25). **FIXED (2026-03-26)**: was extracting 27 garbage holders from full 107-page PDF; now extracts 20 legitimate top-20 shareholders via page-level isolation + stricter regex. | Yes | Derived from get_holders() aggregation (includes top20 holder_type). **FIXED (2026-03-26)**: was 0 rows; now returns 86.78% ownership, HHI=0.2177 for DBS (D05). | Yes | **SGX Financial Reports API** (director/board announcements) |
| 16 | **Switzerland** | `ch_six` | 2 | **Partial** | **yfinance .SW** (no native SIX holder API). Native `get_insider_transactions()` from SIX management transactions API. | **yfinance** | yfinance .SW snapshot only | Yes | **SIX native** management transactions API |
| 17 | **China** | `cn_sse` | 2 | Yes | **akshare/Sina Finance** (`stock_main_stock_holder`, top shareholders with name, shares, %, holder type) + circulating shareholder fallback | Yes | **akshare/EastMoney** (`stock_zh_a_gdhs_detail_em`, quarterly shareholder count + avg shares + change ratio) | Yes | **akshare/THS** (`stock_shareholder_change_ths`, shareholder buy/sell changes) |
| 18 | **EU (pan-EU)** | `eu_esef` | 1 | Yes | **GLEIF API** (corporate ownership: ultimate parent, direct parent, subsidiaries via LEI relationships) | Yes | GLEIF-derived snapshot | No | -- (national regulators, not accessible) |
| 19 | **France** | `fr_esef` | 1 | Yes | **GLEIF API** (via EUEsefClient) | Yes | GLEIF-derived snapshot | No | -- (AMF BDIF backend down) |
| 20 | **Germany** | `de_esef` | 1 | Yes | **GLEIF API** (via EUEsefClient) | Yes | GLEIF-derived snapshot | No | -- (BaFin portal moved) |
| 21 | **Netherlands** | `nl_esef` | 2 | Yes | **GLEIF API** (via EUEsefClient) | Yes | GLEIF-derived snapshot | No | -- (AFM HTML only) |
| 22 | **Spain** | `es_esef` | 2 | Yes | **GLEIF API** (via EUEsefClient) | Yes | GLEIF-derived snapshot | No | -- (CNMV WAF blocked) |
| 23 | **Italy** | `it_esef` | 2 | Yes | **GLEIF API** (via EUEsefClient) | Yes | GLEIF-derived snapshot | No | -- (CONSOB captcha) |
| 24 | **Sweden** | `se_esef` | 2 | Yes | **GLEIF API** (via EUEsefClient) | Yes | GLEIF-derived snapshot | No | -- (FI SPA app) |
| 25 | **Chile** | `cl_cmf` | 1 | Yes | **SEC EDGAR ADR fallback** (SC 13D/13G + DEF 14A via USEdgarClient for SQM, LTM, BSAC, BCH, CCU, ENIC etc.) + GLEIF corporate ownership | Yes | SEC EDGAR ADR-derived or GLEIF snapshot | Yes | **SEC EDGAR Form 4** via ADR ticker |

**Notes:**
- South Korea (DART) is the only market with genuine quarterly historical ownership data across multiple periods.
- China (`cn_sse`) now provides shareholder count history via akshare/EastMoney (`stock_zh_a_gdhs_detail_em`), which tracks quarterly holder counts and average shares per holder -- useful for detecting institutional accumulation/distribution patterns.
- All other `get_holder_history()` implementations return single-row snapshots.
- Switzerland (`ch_six`) is the only remaining wrapper that uses yfinance for holder data. SIX has no free native holder API; the Ownership/Ownership.svc is behind a paid Refinitiv subscription.
- Singapore (`sg_sgx`) was fully de-yfinanced on 2026-03-25. All holder data now comes from native SGX DOI announcements + annual report PDF extraction.
- Singapore (`sg_sgx`) **FIXED (2026-03-26)**: `_parse_sgx_shareholding_text()` was applying loose regex to entire 107-page annual report, producing 27 garbage holders ("Wholesale funding 16%", etc.). Fix: (1) page-level PDF extraction isolates only shareholding-relevant pages, (2) section heading isolation restricts regex to bounded 3000-char windows after headings, (3) stricter `_is_valid_name()` validator rejects table headers/labels/sentence fragments, (4) `get_holder_history()` now includes `"top20"` holder_type.
- China (`cn_sse`) holder methods added on 2026-03-25 using akshare: `stock_main_stock_holder` (Sina Finance top shareholders), `stock_zh_a_gdhs_detail_em` (EastMoney shareholder count history), `stock_shareholder_change_ths` (THS insider changes).

### Live Holder Data Test Results (2026-03-26)

Live probe results for 8 markets with known holder data issues. All tests run from this container (no VPN, no geo-bypass).

| # | Market | Ticker | `get_holders` | `get_holder_history` | Source | Root Cause | Status |
|---|--------|--------|--------------|---------------------|--------|------------|--------|
| 1 | **SG SGX** | D05 (DBS) | **20 holders** (was 27 garbage) | **1 row** (86.78% ownership, HHI=0.2177) | Two-stage PDF: fuzzy_pdf_parser (financials) + page-level shareholding regex | `_parse_sgx_shareholding_text()` regex was too loose; applied to full 107-page PDF instead of isolated shareholding sections | **FIXED** |
| 2 | **US EDGAR** | AAPL | **5 holders** (was 0) | **1 row** (15.17% ownership) | SEC EDGAR SC 13D/13G + DEF 14A proxy | **FIXED (2026-03-26)**: (1) `filings[:20]` raised pyarrow `ChunkedArray.as_py` error in edgartools v5.19.1, silently caught; fix: `list(filings)[:20]`. (2) `filing.company` returned issuer not filer; fix: parse SEC-HEADER from `text_url`. (3) `seen_filers` reset per form_type; fix: moved outside loop. | **FIXED** |
| 3 | **BR CVM** | PETR4 | **8 holders** (was 0) | **1 row** | CVM FRE 2025 posicao_acionaria (8 MB ZIP, 30K rows, 697 companies) | **FIXED (2026-03-26)**: (1) `_resolve_cd_cvm()` didn't strip B3 digits (PETR4->PETR). (2) `get_holders()` CNPJ resolution same issue. (3) Started from 2026 (18 companies) not 2025 (697). (4) Used bare `requests.get` instead of `_download_zip()`. | **FIXED** |
| 4 | **IN BSE** | 500325 | **1 holder** (was 0) | **1 row** | BSE SAST disclosures (Reg. 29/31) | **FIXED (2026-03-26)**: `_extract_holders_from_sast()` skipped Reg. 29(2) "Disclosures" because "closure" is a substring of "Dis-closure-s". Fix: check for "closure of trading" not just "closure". Only 1 SAST disclosure for Reliance in 2yr (major holders stable). | **FIXED** |
| 5 | AU ASX | BHP | 0 | 0 | ASX MarkitDigital substantial holder notices | **Probed (2026-03-26)**: MarkitDigital announcements API only returns 5 items for BHP regardless of `count` param (CEO change, dividends, results). The `announcementTypes`, date range, and `count=200` params are all ignored by the API -- appears to be a hard API limit or endpoint change. Substantial holder notices exist on ASX website but are NOT returned by this API endpoint. **Needs new data source** (e.g. ASX website announcements search, or yfinance institutional_holders fallback). | Not fixable via current API |
| 6 | **CA SEDAR** | RY | **27 insiders** (was 0) | **1 row** | TMX GraphQL SEDI transactions | **FIXED (2026-03-26)**: `getCompanyInsidersActivities` schema changed (removed `numberOfTransactions`/`averagePrice`/`totalValue`), always returned 400. Fix: use `getInsiderTransactions` as primary source, aggregate by unique holder name. Returns 550 SEDI transactions -> 27 unique insiders (Royal Bank, David McKay CEO, etc.). | **FIXED** |
| 7 | ZA JSE | NPN | 0 | 0 | JSE SENS WCF shareholding announcements | `_resolve_master_id("NPN")` may fail if JSE issuer directory search doesn't match the short ticker code | Needs MasterID resolution fix |
| 8 | **HK HKEX** | 00700 | **4 disclosure filings** (was 0) | **1 row** (4 filings counted) | HKEX date-windowed news search (titleSearchServlet.do, title="disclosure") | **FIXED (2026-03-26)**: Three bugs: (1) `stock_hk_main_board_stock_holder_em` REMOVED from akshare 1.18.43 (AttributeError). (2) `scraper.search_announcements()` doesn't exist on HKEXScraper (only `search_filings` + `download_pdf`). (3) EastMoney HK F10 shareholder PageAjax endpoints all 404. Fix: replaced broken akshare + broken search_announcements with HKEX date-windowed scraper using "disclosure" title filter. Returns Next Day Disclosure Return filings with dates and PDF URLs. No yfinance dependency. | **FIXED** |

**Two-stage PDF extraction architecture** (all 6 filing-discoverer-backed wrappers):
- The `fuzzy_pdf_parser.extract_shareholders_from_pdf()` function already implements page-level isolation: scores each page against shareholding keywords, processes top-5 scoring pages, extracts tables with pdfplumber, identifies name/shares/% columns from headers.
- The `try_shareholding_extraction()` function in `filing_discoverer.py` chains: discover filings -> filter for shareholding filings -> download PDFs -> call `extract_shareholders_from_pdf()`. This is called by 6 wrappers (SG, ZA, MX, CA, AU, AE) as a final fallback.
- The SGX fix adds a wrapper-specific improvement: the `_parse_sgx_shareholding_text()` regex parser (used before the `try_shareholding_extraction` fallback) now has section isolation and name validation. This catches SGX's specific format (substantial shareholder table + top-20 list) that the generic `extract_shareholders_from_pdf()` table parser may miss.

### Table 2: Wrappers WITHOUT Portfolio/Holder Data (9 markets)

| # | Market | ID | Tier | Reason | Potential Native Source |
|---|--------|----|------|--------|------------------------|
| -- | -- | -- | -- | **All 25 markets now have holder data methods** | -- |

**Coverage summary:** 25 of 25 markets (100%) have holder data methods. Of those, 23 are fully native (no yfinance). 1 market (Switzerland) still depends on yfinance for holders. 1 market (Chile) has no holder methods at all. China was moved to Table 1 on 2026-03-25 via akshare integration. EU ESEF markets (EU, FR, DE, NL, ES, IT, SE) were moved to Table 1 on 2026-03-25 via GLEIF API integration (corporate ownership structure: parent/subsidiary relationships, 2.6M+ LEI records globally).

---

## Summary by Capability

| Capability | Working | Partial | Broken |
|-----------|---------|---------|--------|
| Profile | IN, CN, AU, CA, SG, **MX**, **ZA**, **CH**, **AE**, **SA**, **HK** (akshare+yfinance), **NL**, **ES**, **IT**, **SE** | -- | -- |
| Company Search | IN, CN, AU, CA, SG, **MX**, **ZA**, **CH**, **AE**, **SA**, **NL**, **ES**, **IT**, **SE** | HK (yfinance only) | -- |
| Financial Statements | IN (native), CN (akshare), **HK (akshare/EastMoney)**, **MX (XBRL JSON)**, **CH (synthetic)**, **SA (XBRL HTML)**, **NL/ES/IT/SE (ESEF XBRL JSON)** | AU, CA, SG, **ZA**, **AE** (LLM primary + fuzzy PDF fallback) | -- |
| OHLCV | All 15 (**CH**: SIX CSV, **AE**: api2 snapshot + yfinance) | -- | -- |
| Filing Discovery | IN, AU, CA, HK, SG, **MX**, **SA**, **ZA**, **AE**, **NL/ES/IT/SE** (ESEF XBRL) | **CH** (ESEF + LLM) | -- |

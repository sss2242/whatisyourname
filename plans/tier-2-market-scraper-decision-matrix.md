# Tier 2 Market Scraper Decision Matrix

## Current State of Each Tier 2 Market

Each Tier 2 client falls into one of three categories:

1. **Already upgraded** -- filing discovery wired, no yfinance for financials
2. **Pure yfinance** -- uses yfinance for financial statements (NOT PIT-compliant)
3. **EU ESEF crossover** -- uses EU ESEF wrapper for dual-listed companies

---

## Assessment Matrix

| Market | Cap | Current Financial Data Path | Filing Discoverer? | Profile Source | Scraper Needed? | Recommendation |
|--------|-----|---------------------------|:--:|----------------|:--:|----------------|
| HK HKEX | $4.5T | Filing discovery only | Yes | yfinance | **No** | Already done |
| SG SGX | $0.6T | Filing discovery only | Yes | SGX API + yfinance | **No** | Already done |
| SA Tadawul | $2.7T | Filing discovery only | Yes | yfinance | **No** | Already done |
| CA SEDAR+ | $3T | Filing discovery only | Yes | SEDAR+ search + canonical | **No** | Already done |
| CH SIX | $1.8T | EU ESEF crossover | No | yfinance | **No** | EU ESEF is the right path |
| AU ASX | $1.8T | **yfinance only** | Yes, but discovery-only | ASX API + canonical | **Yes** | Wire filing discovery into client |
| IN BSE | $4T | Filing discovery + yfinance fallback | Yes | BSE API + canonical | **No** | Already done |
| CN SSE | $10T | SSE/baostock + canonical | No | SSE API + canonical | **No** | Has its own API wrappers |
| MX BMV | $0.5T | **yfinance only** | No | yfinance | **Research** | No known free API; keep yfinance or add filing discovery |
| ZA JSE | $1T | **yfinance only** | No | yfinance | **Research** | JSE SENS might be scrapeable |
| AE DFM | $0.8T | **yfinance only** | No | yfinance | **Research** | DFM/ADX disclosure pages might work |

---

## Detailed Analysis Per Market

### Already Upgraded (5 markets) -- No Action Needed

#### Hong Kong HKEX ($4.5T)
- `hk_hkex.py` uses `try_filing_extraction` as primary
- `HKEXFilingDiscoverer` in `filing_discoverer.py` searches HKEX News, downloads PDFs
- LLM taxonomy hints exist for HKFRS
- `_IFRS_MAP` covers HKFRS concepts (substantively identical)
- **Status: Complete**

#### Singapore SGX ($0.6T)
- `sg_sgx.py` uses `try_filing_extraction` as primary
- `SGXFilingDiscoverer` in `filing_discoverer.py` queries SGX announcements API
- SGX API also used for company search and profile
- `_IFRS_MAP` covers SFRS concepts (identical to IFRS)
- **Status: Complete**

#### Saudi Arabia Tadawul ($2.7T)
- `sa_tadawul.py` uses `try_filing_extraction` as primary
- `TadawulFilingDiscoverer` in `filing_discoverer.py` queries Tadawul disclosure API
- Tadawul API also used for company search
- `_IFRS_MAP` covers Saudi IFRS (adopted 2017)
- **Status: Complete**

#### Canada SEDAR+ ($3T)
- `ca_sedar.py` uses `try_filing_extraction` as primary
- `SEDARFilingDiscoverer` in `filing_discoverer.py` resolves entity ID then searches documents
- SEDAR+ API used for company search and profile
- `_IFRS_MAP` covers Canadian IFRS (adopted 2011)
- **Status: Complete**

#### India BSE ($4T)
- `in_bse.py` uses `try_filing_extraction` as primary with a duplicate `_fetch_financials` method
- `BSEFilingDiscoverer` in `filing_discoverer.py` is the most mature implementation
- Downloads actual PDFs from BSE, extracts via LLM
- **Status: Complete**

### EU ESEF Crossover (1 market) -- No Scraper Needed

#### Switzerland SIX ($1.8T)
- `ch_six.py` tries EU ESEF wrapper first (for dual-listed Swiss blue chips)
- No yfinance fallback for financials
- Returns empty DataFrame if ESEF lookup fails
- This is the correct approach: most major Swiss companies (Nestle, Novartis, Roche, ABB, UBS) file ESEF reports
- Smaller Swiss companies without EU listings will get no financial data
- **No scraper needed** -- SIX has no free structured API; ESEF crossover is the best available path

### Has Own API (1 market) -- No Scraper Needed

#### China SSE ($10T)
- `cn_sse.py` has its own SSE/SZSE API integration with multiple endpoints
- Uses `baostock` for OHLCV data
- Has `canonical_translator` mappings for CAS (Chinese Accounting Standards)
- **No scraper needed** -- already has native integration

### Needs Upgrade (1 market) -- Wire Existing Discoverer

#### Australia ASX ($1.8T)
- `au_asx.py` currently uses yfinance for ALL financial statements
- `ASXFilingDiscoverer` EXISTS in `filing_discoverer.py` but is **not wired** into the client
- ASX API works for company search and profile (native)
- The discoverer is discovery-only (document download returns 404)
- **Action: Wire `try_filing_extraction` into `au_asx.py` as primary, keep yfinance as fallback only if discovery returns empty**
- **Note: Since ASX document download often fails, this may still return empty. ASX is PDF-only with no structured filing API.**

### Needs Research (3 markets) -- Pure yfinance Today

#### Mexico BMV ($0.5T)
- `mx_bmv.py` is pure yfinance (69 lines)
- No filing discoverer exists
- **Options:**
  - a) Create BMV filing discoverer using CNBV (https://emisnet.bmv.com.mx) disclosure pages
  - b) Keep yfinance -- small market, limited data availability
  - c) Remove financial statements entirely (return empty, like SIX does)
- **Recommendation: Option (c)** -- remove yfinance for financials to maintain PIT compliance. BMV is a small market and CNBV has no documented free API.

#### South Africa JSE ($1T)
- `za_jse.py` is pure yfinance (69 lines)
- No filing discoverer exists
- **Options:**
  - a) Create JSE SENS filing discoverer (https://senspdf.jse.co.za) for announcement PDFs
  - b) Keep yfinance
  - c) Remove financial statements entirely
- **Recommendation: Option (a) is viable** -- JSE SENS has a searchable announcement system. SENS announcements include annual/interim results with PDF attachments. Filing dates are available from the announcement dates. Worth implementing a discoverer.

#### UAE DFM/ADX ($0.8T)
- `ae_dfm.py` is pure yfinance (69 lines)
- No filing discoverer exists
- **Options:**
  - a) Create DFM/ADX filing discoverer from disclosure pages
  - b) Keep yfinance
  - c) Remove financial statements entirely
- **Recommendation: Option (c)** -- remove yfinance for financials. Both DFM and ADX have web portals but no documented APIs. Arabic-language filing PDFs add extraction complexity. Small market.

---

## Summary Decision Table

| Market | Action | Implementation |
|--------|--------|---------------|
| HK HKEX | None | Already complete |
| SG SGX | None | Already complete |
| SA Tadawul | None | Already complete |
| CA SEDAR+ | None | Already complete |
| IN BSE | None | Already complete |
| CH SIX | None | EU ESEF crossover is correct |
| CN SSE | None | Has own API integration |
| AU ASX | Wire existing discoverer | Change `au_asx.py` to use `try_filing_extraction` as primary |
| MX BMV | Remove yfinance for financials | Return empty DataFrame for PIT compliance |
| ZA JSE | Create JSE SENS discoverer | New `JSEFilingDiscoverer` + wire into `za_jse.py` |
| AE DFM | Remove yfinance for financials | Return empty DataFrame for PIT compliance |

## Implementation Priority

1. **MX BMV + AE DFM** -- Quick wins: remove non-PIT yfinance from financial statements
2. **AU ASX** -- Wire existing discoverer into client
3. **ZA JSE** -- Research and implement JSE SENS filing discoverer

## What Does NOT Need a Scraper

- **HKEX, SGX, Tadawul, SEDAR+** -- Already have filing discoverers using exchange announcement APIs
- **SIX** -- EU ESEF crossover covers major companies
- **SSE** -- Has native Chinese API integration
- **BSE** -- Has the most mature filing discovery + LLM extraction pipeline

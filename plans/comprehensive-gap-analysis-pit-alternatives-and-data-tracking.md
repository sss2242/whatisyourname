# Comprehensive Gap Analysis: PIT Alternatives and Data Tracking

## Part 1: How the App Tracks Data Per Day

The app uses a **reactive** model, not a predictive one:

1. `DATE_START` = today - 730 days, `DATE_END` = today
2. A business-day date index is generated (~520 trading days)
3. OHLCV data fills the price spine (1 row per trading day)
4. Financial statements are forward-filled via `merge_asof(direction=backward)` -- each day gets the latest filing whose `report_date <= that day`
5. `is_missing_<col>` flags track which values are NaN
6. The estimation module fills NaN values using statistical methods
7. Data quality module computes coverage percentages after the fact

**What the app does NOT do:**
- Does not pre-calculate expected filing dates
- Does not track staleness (how old the latest filing is)
- Does not know how many filings it should have received
- Does not distinguish "data not yet filed" from "data that will never exist"
- Does not adjust behavior based on filing frequency (quarterly vs annual)

---

## Part 2: What the App Lacks (Complete List)

### A. Data Tracking Gaps

1. **Filing calendar awareness** -- No knowledge of expected filing dates per region (e.g., US 10-Q due 40 days after quarter end, Japan semiannual reports)
2. **Staleness detection** -- No flag when the latest filing is older than expected
3. **Coverage forecasting** -- Cannot predict "this company should have 8 quarterly filings in 2 years but only has 5"
4. **Filing frequency detection** -- Doesn't auto-detect if a company reports quarterly, semiannually, or annually
5. **As-of date tracking** -- The inline cache builder in main.py doesn't track `filing_date` vs `report_date` distinction properly

### B. PIT Micro Gaps (Tier 2 Markets)

9 markets currently use yfinance for financials (not PIT-grade):

| Market | yfinance Problem | PIT Alternative |
|--------|-----------------|-----------------|
| **CA** | Restated data | SEDAR+ has a filing API (partially implemented, search works, filings endpoint needs scraping) |
| **AU** | Restated data | ASX has company announcements API; ASIC has financial reports |
| **HK** | Restated data | HKEX News (hkexnews.hk) has filing disclosure search API |
| **SG** | Restated data | SGX has SGXNET disclosure system |
| **MX** | Restated data | BMV/CNBV has EMMA reporting system |
| **ZA** | Restated data | JSE SENS announcements |
| **CH** | Restated data | SIX Exchange Regulation has ad-hoc disclosures |
| **SA** | Restated data | Tadawul has Nomu/disclosure platform |
| **AE** | Restated data | DFM has listed company disclosures |

**Reality check**: Most of these exchanges provide disclosure/announcement systems, not structured financial statement APIs. Getting structured data from them typically requires:
- HTML/PDF scraping of published reports
- XBRL parsing (where available)
- Natural language extraction

The practical alternative to yfinance for structured financials without scraping is **OpenBB** or **SimFin** (both have free tiers with filing-date-aware data).

### C. Macro Gaps

| Market | Gap | Fix |
|--------|-----|-----|
| CA, AU, HK, SG, ZA, CN, IN | No primary macro fetcher -- annual-only from wbgapi | Add FRED series for these countries (FRED covers most of them with monthly data) |
| SA, AE | No FRED coverage | Keep wbgapi; consider IMF IFS API as supplement |
| All regions | No credit spread data | Add FRED credit spread series per market |
| All regions | No yield curve data | Add government bond yield series |

### D. Model/Pipeline Gaps

1. **Filing calendar awareness** not implemented
2. **Staleness detection** not implemented
3. **cache_builder.py not used** by main.py (Bug 10, deferred from PR #2)
4. **Conformal prediction has circular dependency** with prediction aggregation (partially addressed in PR #2)

---

## Part 3: PIT Alternatives to yfinance for Tier 2 Markets

### Option A: Exchange Disclosure APIs (per-market, most work)

Each exchange has a disclosure/announcement system. The approach is:
1. Fetch filing list from disclosure API
2. Download the filing document (HTML/PDF/XBRL)
3. Parse financial statements from the document
4. Store with filing_date for PIT compliance

**Feasibility per market:**

| Market | API | Filing Format | Parsing Difficulty | Free? |
|--------|-----|---------------|-------------------|-------|
| CA (SEDAR+) | REST API at sedarplus.ca/csa-party | PDF + iXBRL | Medium (iXBRL parseable) | Yes |
| AU (ASX) | asx.com.au announcements | PDF | Hard (no XBRL) | Yes |
| HK (HKEX) | hkexnews.hk search | PDF + some XBRL | Medium | Yes |
| SG (SGX) | sgx.com SGXNET | PDF | Hard | Yes |
| MX (BMV) | CNBV/BMV disclosures | PDF | Hard | Yes |
| ZA (JSE) | JSE SENS system | PDF | Hard | Yes |
| CH (SIX) | six-group.com exchange regulation | PDF | Hard | Yes |
| SA (Tadawul) | saudiexchange.sa disclosures | PDF + Arabic | Very hard | Yes |
| AE (DFM) | dfm.ae disclosures | PDF + Arabic | Very hard | Yes |

### Option B: Third-party structured data providers (less work)

| Provider | Coverage | Free Tier | PIT-grade? | Notes |
|----------|----------|-----------|------------|-------|
| **SimFin** | 5000+ companies globally | Yes (limited) | Yes (has filing dates) | simfin.com, Python SDK |
| **OpenBB** | Global via multiple backends | Yes | Varies by backend | openbb.co, aggregator |
| **Financial Modeling Prep** | 40k+ companies | 250 requests/day | Partial | financialmodelingprep.com |
| **Alpha Vantage** | Global | 25 requests/day | No | Too rate-limited |

### Recommended Approach

**Phase 1 (quick wins):**
1. Add FRED macro series for CA, AU, HK, SG, ZA, CN, IN (monthly data instead of annual wbgapi)
2. Add filing calendar module with expected filing dates per region/frequency
3. Add staleness detection to data quality module

**Phase 2 (medium effort):**
4. Replace yfinance financials with SimFin for markets where SimFin has coverage (most developed markets)
5. Implement SEDAR+ iXBRL parsing for Canada (partially built already)
6. Implement HKEX filing search + PDF extraction for Hong Kong

**Phase 3 (heavy effort):**
7. Implement exchange disclosure scrapers for remaining markets (SG, MX, ZA, CH, SA, AE)
8. Add credit spread and yield curve data to macro provider
9. Migrate main.py cache building to use cache_builder.py properly

---

## Files That Would Change

### Phase 1
- New: `operator1/features/filing_calendar.py` -- filing frequency detection and expected dates
- New: `operator1/quality/staleness_detector.py` -- flag stale filings
- Modified: `operator1/clients/macro_fredapi.py` -- add CA/AU/HK/SG/ZA/CN/IN series
- Modified: `operator1/clients/macro_provider.py` -- add primary fetcher entries

### Phase 2
- New: `operator1/clients/simfin_wrapper.py` -- SimFin integration
- Modified: `operator1/clients/ca_sedar.py` -- add iXBRL parsing
- Modified: `operator1/clients/hk_hkex.py` -- add disclosure API
- Modified: 7 stub clients to use SimFin instead of yfinance

### Phase 3
- New: per-market disclosure scrapers (SG, MX, ZA, CH, SA, AE)
- Modified: `main.py` -- use cache_builder.py instead of inline logic
- Modified: `operator1/clients/macro_fredapi.py` -- add credit spread/yield curve series

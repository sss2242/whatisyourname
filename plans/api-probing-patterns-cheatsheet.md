# API Probing Patterns Cheatsheet

Quick reference for the four proven patterns used in Operator 1 wrappers.
Paste this into any new chat to restore context instantly.

---

## Pattern 1: HKEX Date-Windowed Scraper

**When to use:** API silently rejects large date ranges or returns empty for wide queries.

**Key steps:**
1. **Session init**: GET the search/landing page first to obtain session cookies (JSESSIONID)
2. **Date windowing**: Split multi-year searches into 2-week (14-day) windows. Iterate from newest to oldest
3. **Headers**: Always include `X-Requested-With: XMLHttpRequest` and `Referer: <search_page_url>`
4. **Client-side filtering**: Fetch ALL results per window (`stockId=-1`), filter by target in Python
5. **Double JSON parse**: Some APIs return a JSON string inside the JSON response (`json.loads(response["result"])`)
6. **Rate limit**: 0.5s sleep between requests

**Code location:** `operator1/clients/hkex_scraper.py`

```python
session = requests.Session()
session.get(SEARCH_PAGE_URL, timeout=30)  # get cookies
for from_date, to_date in date_windows:
    resp = session.get(API_ENDPOINT, params={
        "fromDate": from_yyyymmdd, "toDate": to_yyyymmdd,
        "searchType": "1", "title": "results",
        "stockId": "-1", "rowRange": "5000", "lang": "E",
    }, headers={"X-Requested-With": "XMLHttpRequest", "Referer": SEARCH_PAGE})
```

---

## Pattern 2: SGX Free-Key (No Auth API)

**When to use:** Exchange provides a public REST API that works globally without authentication.

**Key steps:**
1. **Direct GET**: No token, no cookies, no auth. Just `requests.get(url, params, headers)`
2. **Module-level caching**: Cache the full directory in-memory with TTL (24h). Avoids re-fetching on every call
3. **Short field names**: Exchange APIs often use abbreviated keys (`n`=name, `nc`=code). Map them in Python
4. **Client-side search**: Fetch the full directory once, search/filter locally (exact match -> substring -> fuzzy)
5. **Paginated iteration**: Use `pagestart`/`pagesize` params when results are paginated
6. **Two-step document download**: API returns HTML pages with embedded PDF links, not direct PDF URLs

**Code location:** `operator1/clients/sg_sgx.py`

```python
resp = requests.get(f"{BASE}/securities/v1.1",
    params={"type": "stocks", "pagestart": "0", "pagesize": "2000"},
    headers={"User-Agent": CHROME_UA, "Accept": "application/json"},
    timeout=30)
stocks = resp.json()["data"]["prices"]
```

---

## Pattern 3: BMV Token/Build (WSO2 API Manager)

**When to use:** Site uses a gateway that issues free Bearer tokens without credentials. Common with WSO2, Azure APIM, AWS API Gateway behind CDN.

**Key steps:**
1. **Session bootstrap**: Load main page first for JSESSIONID + CDN cookies
2. **Token acquisition**: `GET /rest/tokenservice/token` (or similar) returns a Bearer token. No API key needed
3. **Token refresh**: Cache token with TTL (1 hour). Track failed attempts, skip retries for 5 min on failure
4. **Search via POST**: `POST /api/searchservice/v1` with JSON body + `Authorization: Bearer {token}` + `Content-Type: application/json` + `Referer` + `Origin` headers
5. **ElasticSearch response navigation**: Results are deeply nested: `response.busquedaClaveCotizacion.emisorasSinSerie.hits[]._source`
6. **XBRL bulk extraction**: Parse HTML table of ZIP links, build ticker->URL index (cached 24h), download ZIPs, extract JSON with concept mappings

**Code location:** `operator1/clients/mx_bmv.py`

```python
# Token acquisition
session = requests.Session()
session.get(BASE_URL, timeout=15)  # cookies
resp = session.get(TOKEN_URL, headers={
    "Referer": f"{BASE_URL}/", "Accept": "application/json",
    "X-Requested-With": "XMLHttpRequest"})
token = resp.json()["response"]["access_token"]

# Search
resp = session.post(SEARCH_URL, json={
    "lang": "en", "payload": {"term": query, "searchType": "busquedaClaveCotizacion"}
}, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
```

---

## Pattern 4: Tadawul WAF-Skip (curl_cffi Chrome Impersonation)

**When to use:** Site uses Akamai/Cloudflare/Imperva WAF that blocks plain `requests`. TLS fingerprinting required.

**Key steps:**
1. **curl_cffi session**: `from curl_cffi import requests as cf_requests; s = cf_requests.Session(impersonate="chrome")`
2. **Multiple directory sources**: Combine several endpoints for full coverage (search servlet for IDs + sector page for enrichment + ticker servlet for prices)
3. **Base64-decoded URLs**: Some community-maintained gists store encoded WPS portal URLs. Decode at runtime
4. **WebSphere Portal (WPS) pattern**: Load the company profile page first (WPS session init), then hit relative resource paths like `p0/IZ7_...=CZ6_...=NJstatementsTabData=/` with query params
5. **Server-rendered HTML tables**: Some exchanges render full data tables in HTML (no JS/API). Parse with `pd.read_html()` or regex
6. **Foreign ownership page scraping**: Full 411-company table in one 760KB page. Cache aggressively (1h TTL)
7. **Fuzzy search**: `rapidfuzz.fuzz.WRatio` with 70% cutoff for company name matching

**Code location:** `operator1/clients/sa_tadawul.py`

```python
from curl_cffi import requests as cf_requests
s = cf_requests.Session(impersonate="chrome")

# Directory fetch
r = s.get(f"{BASE}/tadawul.eportal.theme.helper/ThemeSearchUtilityServlet",
    params={"searchText": ""}, timeout=20)
companies = r.json()

# WPS financial data
s.get(profile_url, timeout=25)  # WPS session init
r = s.get(profile_base + "p0/IZ7_.../NJstatementsTabData=/",
    params={"statementType": "6", "reportType": "Q", "symbol": symbol})
```

---

## Pattern 5: ASX MarkitDigital (Third-Party Exchange Data Provider)

**When to use:** Exchange outsources its data API to a third-party provider (MarkitDigital, Refinitiv, FactSet). The third-party API is public, versioned, and JSON-based but lives on a different domain from the exchange website.

**Key steps:**
1. **Identify the actual API domain**: The exchange website (asx.com.au) may return 404 on old API paths. The real API lives on a provider domain (asx.api.markitdigital.com). Find it by inspecting the exchange website's network requests
2. **Versioned REST endpoints**: Provider APIs use versioned paths like `/asx-research/1.0/companies/{ticker}/header`. Try version bumps (`1.0`, `1.1`, `2.0`) when endpoints fail
3. **Full directory fetch + client-side search**: The directory endpoint (`/companies/directory`) returns ALL companies in one paginated call. No server-side search parameter -- filter locally by ticker/name substring match
4. **Announcement-based filing discovery**: The announcements endpoint (`/companies/{ticker}/announcements`) returns filing metadata (headline, date, documentKey). Filter by announcement type keywords ("PERIODIC REPORTS", "ANNUAL REPORT", "substantial holder"). Note: some APIs hard-cap results (ASX returns max 5 items regardless of `count` param)
5. **Two-step document download**: Announcements return document keys, not direct PDF URLs. First GET the document page (HTML), then extract the actual PDF link from it. Validate PDF magic bytes (`%PDF`) before processing
6. **Headline parsing for holder data**: Substantial holder notices and director interest changes are announced via headlines. Parse holder names and percentages from headline text using regex (e.g., "Becoming a substantial holder from BlackRock" -> name=BlackRock)
7. **Multi-source holder strategy**: MarketScreener (primary, global coverage) -> Exchange announcements (fallback, limited to >5% holders) -> PDF extraction (last resort). This pattern applies to any exchange where holder APIs are limited

**Code location:** `operator1/clients/au_asx.py`

```python
# Directory fetch (full list, client-side search)
resp = requests.get(f"{MARKIT_BASE}/companies/directory",
    params={"page": 0, "itemsPerPage": 2500},
    headers={"Accept": "application/json", "User-Agent": "Operator1/1.0"},
    timeout=20)
items = resp.json()["data"]["items"]

# Profile from header endpoint
resp = requests.get(f"{MARKIT_BASE}/companies/{ticker}/header",
    headers=HEADERS, timeout=10)
data = resp.json()["data"]
# Fields: displayName, sector, industryGroup, marketCap, dateListed

# Announcement-based filing discovery
resp = requests.get(f"{MARKIT_BASE}/companies/{ticker}/announcements",
    params={"count": "50"},
    headers=HEADERS, timeout=15)
items = resp.json()["data"]["items"]
# Filter: headline contains "substantial", "annual report", "appendix 3y"

# Holder extraction from announcement headlines
import re
for item in items:
    headline = item["headline"]
    name_match = re.search(r"(?:from|by|for)\s+([A-Z][\w\s&]+?)(?:\s*-|\s*$)", headline)
    pct_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", headline)
```

---

## CRITICAL: Mix All Patterns Together

**Do NOT use patterns in isolation.** Real-world probing requires combining
multiple patterns in a single probe script. The patterns are building blocks,
not standalone recipes.

**Example: Probing SGX for holder endpoints**

A single probe script might use:
- **Pattern 2** (free-key GET) to hit `api.sgx.com` REST endpoints
- **Pattern 4** (curl_cffi) if any endpoint returns 403 from WAF
- **Pattern 1** (date-windowed) if results are empty for large date ranges
- **Pattern 3** (token) if you discover a gateway token endpoint
- **Pattern 5** (third-party provider) if the exchange outsources its API to MarkitDigital/Refinitiv

```python
import requests
from curl_cffi import requests as cf_requests

BASE = 'https://api.sgx.com'
SITE = 'https://www.sgx.com'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 ... Chrome/120',
    'Accept': 'application/json',
    'Referer': f'{SITE}/',
    'X-Requested-With': 'XMLHttpRequest',
}

# Phase 1: Probe with plain requests (Pattern 2)
for endpoint in ['/shareholders/v1.0', '/disclosures/v1.0', ...]:
    r = requests.get(BASE + endpoint, headers=HEADERS, timeout=10)
    print(f'{r.status_code} | {endpoint} | {r.text[:100]}')

# Phase 2: If 403, retry with curl_cffi (Pattern 4)
if r.status_code == 403:
    s = cf_requests.Session(impersonate='chrome')
    r = s.get(BASE + endpoint, timeout=10)

# Phase 3: If empty, try date-windowed (Pattern 1)
# Phase 4: If gateway detected, get token first (Pattern 3)

# Phase 5: Probe the website itself for server-rendered HTML tables
# (like Tadawul's foreign ownership page)
r = requests.get(f'{SITE}/some/page', headers=HEADERS)
# Parse HTML tables with pd.read_html() or regex

# Phase 6: Check for undocumented JSON endpoints behind AJAX
# (like HKEX's titleSearchServlet.do)
# Look in browser DevTools Network tab for XHR requests
```

**The probing mindset:**
1. Start broad: hit many endpoints quickly with Pattern 2
2. Adapt: if blocked, escalate to Pattern 4 (curl_cffi)
3. If API exists but returns empty: try Pattern 1 (date windows) or Pattern 3 (token)
4. If no API: scrape HTML tables from the website (Pattern 4 + regex/pd.read_html)
5. Always check for XBRL/structured data before resorting to PDF parsing
6. Check community packages (akshare, edgartools, dart-fss, pykrx etc.)
7. Filing discovery + fuzzy PDF as last resort (works without LLM via camelot-py)

---

## General Probing Checklist

When probing a new exchange/API:

1. **Probe REST API endpoints** (Pattern 2) -- try `/securities/`, `/shareholders/`, `/disclosures/`, `/ownership/`, `/insider/`, `/directors/`, `/announcements/`, `/corporateactions/` with version suffixes `/v1.0`, `/v1.1`
2. **If 403** -- switch to curl_cffi Chrome impersonation (Pattern 4)
3. **If gateway** -- look for token endpoints like `/rest/tokenservice/token` (Pattern 3)
4. **If empty results** -- try date-windowed queries or smaller page sizes (Pattern 1)
5. **Check for HTML pages** -- some exchanges serve full data tables in server-rendered HTML (scrape with regex + pd.read_html)
6. **Check AJAX/XHR** -- undocumented JSON endpoints behind XMLHttpRequest headers
7. **Look for XBRL/structured data** -- always prefer over PDF parsing
8. **Check community packages** -- akshare, edgartools, dart-fss, pykrx etc.
9. **Filing discovery + fuzzy PDF** as last resort -- works without LLM via camelot-py

**Common headers for all patterns:**
```python
headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "<site_main_page>",
}
```

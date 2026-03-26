# Per-Wrapper Deep Probing with Pattern-Specific Health Checks

## Problem

The current health check system uses the same generic 4-level probe (L0 DNS, L1 Search, L2 Profile, L3 Financials) for all 25 markets. This misses wrapper-specific failure modes:

- **HKEX**: Date-windowed API that silently returns 0 results when the window is wrong, needs JSESSIONID from search page, geo-blocks non-HK IPs
- **SGX**: Free key-less API with Securities v1.1 endpoint that can change structure
- **Tadawul**: Akamai WAF requires curl_cffi Chrome TLS fingerprint impersonation; plain requests gets 403
- **BMV Mexico**: Needs Bearer token from public token endpoint (JSESSIONID pattern)
- **CVM Brazil**: Downloads ZIP archives per year; year boundary issues
- **BSE India**: Requires Referer header; rate limiting patterns
- **JSE South Africa**: WCF services with specific endpoint patterns
- **ESEF**: filings.xbrl.org API may return 0 filings for certain countries while working fine for others

## Existing Wrapper Patterns Discovered

| Market | Pattern | Unique Technical Requirement |
|--------|---------|------------------------------|
| HKEX | Date-windowed JSON API + JSESSIONID | Must load search page first to get session cookie; 2-week max window; geo-blocks non-HK |
| SGX | REST API + session cookies | Securities v1.1 endpoint; Financial Reports API paginated; no auth needed |
| Tadawul | curl_cffi Chrome impersonation | Akamai WAF blocks plain requests; WPS portal resource URLs with base64 gist |
| BMV Mexico | Bearer token auth | Must obtain JWT from token endpoint before API calls |
| BSE India | Referer header required | All BSE APIs need Referer: https://www.bseindia.com |
| CVM Brazil | Year-windowed ZIP downloads | One ZIP per year; structured CSV inside; year boundary issues |
| JSE SA | WCF services | SensAnnouncements.js pattern; session-less but specific endpoint format |
| ESEF | REST API per country | Same API but different country codes; one country can be down while others work |

## Design: Per-Wrapper Probe Registry

Add a `WRAPPER_PROBES` registry mapping each market_id to its specific probe configuration. Each probe tests the unique technical requirement of that wrapper.

```python
WRAPPER_PROBES = {
    'hk_hkex': {
        'probe_type': 'session_api',
        'steps': [
            # Step 1: Load search page to get JSESSIONID
            {'action': 'GET', 'url': 'https://www1.hkexnews.hk/search/titlesearch.xhtml',
             'expect': 'cookie:JSESSIONID'},
            # Step 2: Date-windowed API query with session
            {'action': 'GET', 'url': 'https://www1.hkexnews.hk/search/titleSearchServlet.do',
             'params': {'searchType': 1, 'stockCode': '00700',
                        'from': '{today_minus_14d}', 'to': '{today}'},
             'headers': {'X-Requested-With': 'XMLHttpRequest',
                         'Referer': 'https://www1.hkexnews.hk/search/titlesearch.xhtml'},
             'expect': 'json_key:recordCnt'},
        ],
        'status_map': {
            'recordCnt > 0': 'working',
            'recordCnt == 0': 'restructured',  # API changed format or window logic
            'HTTP 403': 'geo_blocked',
            'connection_error': 'down',
        },
    },
    'sa_tadawul': {
        'probe_type': 'waf_bypass',
        'steps': [
            # Step 1: Test plain requests (should fail with 403)
            {'action': 'GET_plain', 'url': 'https://www.saudiexchange.sa/wps/portal/...',
             'expect': 'HTTP 403',
             'label': 'waf_active'},
            # Step 2: Test curl_cffi impersonation (should work)
            {'action': 'GET_curl_cffi', 'url': '{tadawul_search_url}',
             'expect': 'json_key:data',
             'label': 'bypass_working'},
        ],
        'status_map': {
            'waf_active + bypass_working': 'working',
            'waf_active + bypass_failed': 'restructured',
            '!waf_active': 'waf_removed',  # Tadawul removed WAF
        },
    },
    'sg_sgx': {
        'probe_type': 'free_api',
        'steps': [
            {'action': 'GET', 'url': 'https://api.sgx.com/securities/v1.1',
             'params': {'type': 'stocks', 'pagestart': '0', 'pagesize': '5'},
             'expect': 'json_path:data.prices[0].n'},
        ],
    },
    ...
}
```

## Implementation Plan

```
[ ] Step 1: Create operator1/monitoring/wrapper_probes.py
    - WRAPPER_PROBES registry with per-market probe configs
    - ProbeExecutor class that runs the steps sequentially
    - Support for: GET, GET_with_session, GET_curl_cffi, POST
    - Status detection: working, restructured, down, geo_blocked, waf_removed

[ ] Step 2: Add deep probe runner to health_check.py
    - New probe level L4: wrapper-specific deep probe
    - Runs after L3 passes (or as an alternative when L3 fails)
    - Returns WrapperProbeResult with detailed status per step

[ ] Step 3: Add per-market probe configs for all 25 markets
    - HKEX: session + date-windowed query + geo-block detection
    - SGX: securities directory API structure check
    - Tadawul: WAF bypass test (plain vs curl_cffi)
    - BMV: token endpoint + JWT auth flow
    - BSE India: Referer header test
    - CVM Brazil: ZIP download + structure check
    - JSE SA: WCF endpoint pattern test
    - ESEF: per-country filing count (detect country-specific outages)
    - SEC EDGAR: company_tickers.json structure check
    - DART: API key validation + corp_code search
    - And remaining 15 markets

[ ] Step 4: Update dashboard Health tab
    - Show per-market deep probe results
    - Color-coded status: working (green), restructured (orange), down (red), geo_blocked (yellow)
    - Expand card to show probe step details
    - Show technical pattern info (WAF, session, geo-block)

[ ] Step 5: Add community probe data sources
    - isitdown.com / downdetector status for major exchange websites
    - uptimerobot-style HTTP checks via built-in scheduler
    - GitHub status pages for open-source APIs (SEC EDGAR, XBRL)

[ ] Step 6: Test with live probes on 3-5 markets
[ ] Step 7: Commit and push
```

## Status Categories

| Status | Meaning | Color | Action |
|--------|---------|-------|--------|
| working | API responding correctly with expected data format | Green | None |
| restructured | API responds but format/endpoints changed | Orange | Wrapper needs update |
| down | API not responding (timeout, 5xx, DNS failure) | Red | Use fallback path |
| geo_blocked | API blocks based on IP geography | Yellow | Need proxy or alternative |
| waf_blocked | Web Application Firewall rejecting requests | Orange | Need fingerprint update |
| rate_limited | API responding with 429 | Yellow | Reduce request rate |
| auth_changed | Authentication requirements changed | Orange | Update auth flow |
| partial | Some endpoints work, others broken | Orange | Per-endpoint routing |

## Technical Notes

- `curl_cffi` is already a dependency (used by Tadawul wrapper) -- reuse for WAF bypass probes
- Session management (JSESSIONID) reuses existing `HKEXScraper._get_session()` pattern
- Probes should be lightweight: 1-2 API calls max per market, timeout 15s
- Results stored in extended `wrapper_health.json` under a `deep_probes` key per market
- Dashboard shows probe details on click/expand (NiceGUI expansion panel)

# Enhanced Wrapper Monitoring with Community Tools and Techniques

## Community Tools We Can Leverage

### Already Installed (zero-cost integration)

| Tool | What it gives us | Where used |
|------|-----------------|------------|
| **curl_cffi** (installed) | Chrome TLS fingerprint impersonation for WAF bypass probing. Already used by Tadawul, MOPS, MarketScreener. Can probe whether a site's WAF is up/down/changed. | Tadawul, MOPS, MarketScreener, ASX |
| **dnspython** (installed) | DNS record probing (A, CNAME, NS, SOA). Detects CDN changes, DNS-level blocks, domain migrations. More informative than raw socket connect. | L0 enhancement |
| **httpx** (installed via edgartools) | HTTP/2 support, connection pooling, response timing, redirect chain inspection. Better diagnostics than requests for probing. | All markets |
| **requests** (installed) | Session management, cookie tracking, header inspection. Already used extensively. | All markets |
| **beautifulsoup4** (installed) | HTML parsing for detecting page structure changes (restructured status). | HKEX, BSE, Tadawul |

### New Packages (small, focused)

| Tool | What it gives us | Size | License |
|------|-----------------|------|---------|
| **httpx** (enhanced use) | Already installed. Use `httpx.Client(http2=True)` for HTTP/2 probing; `.stream()` for response inspection without downloading full body; timing breakdown (DNS, connect, TLS, transfer). | 0 | BSD |

### Community Patterns We Can Adopt

| Pattern | Source | Application |
|---------|--------|-------------|
| **Synthetic monitoring** (uptime-kuma style) | UptimeKuma, Blackbox Exporter | Run canary transactions at fixed intervals, track latency percentiles, detect drift |
| **TLS fingerprint rotation** | curl_cffi community | Test multiple fingerprint profiles (Chrome 120, Firefox, Safari) to detect which bypass still works |
| **API structure diff** | API monitoring tools (Akita, Optic) | Compare current API response schema against last-known-good schema. Detect field renames, missing keys, type changes |
| **WAF detection** (Tadawul pattern) | wafw00f community | Probe with plain requests first (expect 403), then curl_cffi (expect 200). Classifies WAF type by response headers |
| **Session token validation** (HKEX/BMV pattern) | Selenium/Playwright community | Cookie lifecycle probing: obtain session -> validate session -> make API call with session -> verify data in response |
| **CDN cache probing** | Varnish/Cloudflare community | Check `Age`, `X-Cache`, `CF-Cache-Status` headers to detect whether responses are cached or live |
| **Certificate chain inspection** | certifi/ssl community | Detect cert expiry approaching, CA changes, or self-signed certs that indicate infrastructure changes |

## Implementation Design

### Core: `wrapper_probes.py`

A single new file with a `ProbeExecutor` that runs per-market probe sequences using the tools above:

```python
class ProbeExecutor:
    def __init__(self):
        self._sessions = {}  # Reuse sessions across probes
    
    def run_probe(self, market_id: str) -> WrapperProbeResult:
        config = WRAPPER_PROBES[market_id]
        results = []
        for step in config['steps']:
            result = self._execute_step(step)
            results.append(result)
            if result.blocking and not result.passed:
                break
        return WrapperProbeResult(
            market_id=market_id,
            steps=results,
            status=self._classify_status(results, config),
        )
    
    def _execute_step(self, step):
        if step['method'] == 'curl_cffi':
            return self._probe_curl_cffi(step)
        elif step['method'] == 'session':
            return self._probe_session(step)
        elif step['method'] == 'dns':
            return self._probe_dns(step)
        elif step['method'] == 'schema_diff':
            return self._probe_schema(step)
        ...
```

### Probe Types (using community patterns)

**1. DNS Probe** (dnspython)
```python
# Detect domain migrations, CDN changes, DNS blocks
import dns.resolver
answers = dns.resolver.resolve('api.sgx.com', 'A')
# Check if IPs changed from last-known-good
# Check if CNAME points to expected CDN
```

**2. TLS Probe** (ssl + certifi)
```python
# Detect cert changes, expiry approaching, CA switches
import ssl, socket
ctx = ssl.create_default_context()
with ctx.wrap_socket(socket.socket(), server_hostname=host) as s:
    s.connect((host, 443))
    cert = s.getpeercert()
    # Check expiry, issuer, subject
```

**3. WAF Detection** (curl_cffi + requests comparison)
```python
# Tadawul pattern: compare plain requests vs curl_cffi
import requests
from curl_cffi import requests as cf_requests

# Plain requests (should get 403 if WAF active)
r_plain = requests.get(url, timeout=10)
# curl_cffi Chrome (should get 200 if bypass works)
s = cf_requests.Session(impersonate='chrome')
r_cf = s.get(url, timeout=10)

# Classify: waf_active + bypass_working, waf_removed, bypass_broken
```

**4. Session Cookie Probe** (requests.Session)
```python
# HKEX/BMV pattern: full session lifecycle
session = requests.Session()
# Step 1: Load page to get session cookie
r1 = session.get(search_page_url)
assert 'JSESSIONID' in session.cookies
# Step 2: Use session cookie for API call
r2 = session.get(api_url, params=..., headers={'X-Requested-With': 'XMLHttpRequest'})
assert r2.json().get('recordCnt', 0) > 0
```

**5. Schema Diff** (json comparison)
```python
# Compare current API response structure against saved baseline
current = response.json()
baseline = load_baseline(market_id)
diff = compute_schema_diff(current, baseline)
# Detects: new fields, removed fields, type changes, structural changes
```

**6. Referer Gate Probe** (BSE India pattern)
```python
# Test with and without Referer header
r_no_ref = requests.get(url, timeout=10)  # Expect 403/empty
r_with_ref = requests.get(url, headers={'Referer': 'https://www.bseindia.com'}, timeout=10)
# Classify: referer_required, referer_not_required, api_down
```

### Dashboard Integration

The dashboard Health tab gets expanded with:
- Clickable market cards that show probe step details
- Technical pattern badges: "WAF", "Session", "Geo-Block", "Referer"
- Schema diff alerts: "API structure changed 2 days ago"
- Response time trends (latency history from JSONL)

## Implementation Steps

```
[ ] Step 1: Create operator1/monitoring/wrapper_probes.py
    - ProbeExecutor with 6 probe types (DNS, TLS, WAF, Session, Schema, Referer)
    - WRAPPER_PROBES registry for all 25 markets
    - WrapperProbeResult with 8 status categories
    - Schema baseline save/load for drift detection

[ ] Step 2: Wire into health_check.py as L4 deep probe
    - run_deep_probes() function called after standard L0-L3
    - Results stored in wrapper_health.json under 'deep_probes' key

[ ] Step 3: Update dashboard Health tab
    - Expandable market cards with probe step details
    - Technical pattern badges
    - Schema diff alerts
    - Latency trend sparklines

[ ] Step 4: Add schema baselines for all 25 markets
    - Capture current API response structure as baseline JSON
    - Store in config/probe_baselines/ directory
    - Auto-update baseline on explicit 'accept changes' action

[ ] Step 5: Test live probes on 5 markets (HKEX, SGX, Tadawul, BSE, SEC)
[ ] Step 6: Commit and push
```

## Why This Is Better Than Starting from Scratch

1. **curl_cffi is already installed** and battle-tested across 4 wrappers. We reuse it for WAF probing rather than writing our own TLS fingerprinting.
2. **dnspython is already installed**. We use it for DNS probing rather than raw socket queries.
3. **The HKEX session pattern** is already implemented in `hkex_scraper.py`. We extract it into a reusable probe pattern.
4. **The Tadawul WAF bypass** is already working in `sa_tadawul.py`. We turn it into a diagnostic probe.
5. **Schema diffing** uses plain JSON comparison (no new dependencies). The baseline files serve as documentation of each API's expected structure.
6. **httpx** is already installed via edgartools. We use it for HTTP/2 probing and timing breakdown.

Total new dependencies: **zero**. Everything builds on tools already in the dependency tree.

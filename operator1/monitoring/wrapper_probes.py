"""Per-wrapper deep probing with pattern-specific health checks.

Uses community tools already in the dependency tree (curl_cffi, dnspython,
httpx, requests, ssl) to probe each market wrapper's unique technical
requirements:

- **DNS probe**: detect domain migrations, CDN changes, DNS blocks
- **TLS probe**: detect cert expiry, CA changes, infrastructure shifts
- **WAF probe**: detect WAF presence + bypass status (curl_cffi vs plain)
- **Session probe**: validate session cookie lifecycle (JSESSIONID pattern)
- **Schema probe**: detect API response structure changes vs baseline
- **Referer probe**: detect Referer-gated APIs (BSE India pattern)

Each market gets probes matching its actual wrapper implementation pattern.

Usage:
    from operator1.monitoring.wrapper_probes import run_deep_probes
    results = run_deep_probes()  # all markets
    result = run_deep_probe("hk_hkex")  # single market
"""

from __future__ import annotations

import json
import logging
import socket
import ssl
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASELINE_DIR = Path("config/probe_baselines")
_PROBE_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ProbeStepResult:
    """Result of a single probe step."""
    step_name: str = ""
    method: str = ""          # dns, tls, waf, session, schema, referer, http
    passed: bool = False
    latency_ms: int = 0
    detail: str = ""
    error: str = ""


@dataclass
class WrapperProbeResult:
    """Deep probe result for a single market."""
    market_id: str = ""
    status: str = "unknown"   # working, restructured, down, geo_blocked,
                              # waf_blocked, rate_limited, auth_changed, partial
    pattern: str = ""         # Technical pattern label (waf, session, free_api, etc.)
    steps: list[dict] = field(default_factory=list)
    total_latency_ms: int = 0
    probe_timestamp: str = ""
    schema_drift: bool = False
    schema_diff_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Probe implementations (using community tools)
# ---------------------------------------------------------------------------

def _probe_dns(host: str) -> ProbeStepResult:
    """DNS resolution probe using dnspython if available, socket fallback."""
    result = ProbeStepResult(step_name="dns_resolve", method="dns")
    t0 = time.time()
    try:
        try:
            import dns.resolver
            answers = dns.resolver.resolve(host, "A")
            ips = [str(r) for r in answers]
            result.detail = f"A records: {', '.join(ips[:3])}"
            result.passed = True
        except ImportError:
            # Fallback to socket
            ip = socket.gethostbyname(host)
            result.detail = f"Resolved: {ip}"
            result.passed = True
    except Exception as exc:
        result.error = f"DNS failed: {exc}"
    result.latency_ms = int((time.time() - t0) * 1000)
    return result


def _probe_tls(host: str, port: int = 443) -> ProbeStepResult:
    """TLS certificate inspection."""
    result = ProbeStepResult(step_name="tls_cert", method="tls")
    t0 = time.time()
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                not_after = cert.get("notAfter", "")
                issuer = dict(x[0] for x in cert.get("issuer", []))
                issuer_org = issuer.get("organizationName", "unknown")
                # Parse expiry
                try:
                    from email.utils import parsedate_to_datetime
                    expiry = parsedate_to_datetime(not_after)
                    days_left = (expiry - datetime.now(timezone.utc)).days
                    result.detail = f"Issuer: {issuer_org}, expires in {days_left}d"
                    if days_left < 14:
                        result.detail += " WARNING: expiring soon!"
                except Exception:
                    result.detail = f"Issuer: {issuer_org}, expires: {not_after}"
                result.passed = True
    except Exception as exc:
        result.error = f"TLS failed: {exc}"
    result.latency_ms = int((time.time() - t0) * 1000)
    return result


def _probe_http(url: str, headers: dict | None = None, expect_status: int = 200) -> ProbeStepResult:
    """Standard HTTP GET probe."""
    result = ProbeStepResult(step_name="http_get", method="http")
    t0 = time.time()
    try:
        import requests
        resp = requests.get(url, headers=headers or {}, timeout=_PROBE_TIMEOUT, allow_redirects=True)
        result.latency_ms = int((time.time() - t0) * 1000)
        result.detail = f"HTTP {resp.status_code}, {len(resp.content)} bytes"
        if resp.status_code == expect_status:
            result.passed = True
        elif resp.status_code == 403:
            result.error = "HTTP 403 Forbidden (possible WAF or geo-block)"
        elif resp.status_code == 429:
            result.error = "HTTP 429 Rate Limited"
        else:
            result.error = f"Expected {expect_status}, got {resp.status_code}"
    except Exception as exc:
        result.latency_ms = int((time.time() - t0) * 1000)
        result.error = f"HTTP failed: {str(exc)[:100]}"
    return result


def _probe_waf(url: str, label: str = "waf_bypass") -> ProbeStepResult:
    """WAF detection: compare plain requests vs curl_cffi Chrome impersonation."""
    result = ProbeStepResult(step_name=label, method="waf")
    t0 = time.time()
    plain_status = 0
    cf_status = 0

    # Step A: Plain requests (should get 403 if WAF active)
    try:
        import requests
        r = requests.get(url, timeout=_PROBE_TIMEOUT,
                         headers={"User-Agent": "Mozilla/5.0"})
        plain_status = r.status_code
    except Exception:
        plain_status = 0

    # Step B: curl_cffi Chrome (should get 200 if bypass works)
    try:
        from curl_cffi import requests as cf_requests
        s = cf_requests.Session(impersonate="chrome")
        r = s.get(url, timeout=_PROBE_TIMEOUT)
        cf_status = r.status_code
    except ImportError:
        result.error = "curl_cffi not installed"
        result.latency_ms = int((time.time() - t0) * 1000)
        return result
    except Exception as exc:
        result.error = f"curl_cffi failed: {str(exc)[:100]}"
        cf_status = 0

    result.latency_ms = int((time.time() - t0) * 1000)

    if plain_status == 403 and cf_status == 200:
        result.passed = True
        result.detail = "WAF active, curl_cffi bypass working"
    elif plain_status == 200 and cf_status == 200:
        result.passed = True
        result.detail = "No WAF detected (both methods work)"
    elif plain_status == 403 and cf_status != 200:
        result.error = f"WAF active, bypass FAILED (curl_cffi got {cf_status})"
    elif plain_status == 0 and cf_status == 0:
        result.error = "Both methods failed (site down)"
    else:
        result.detail = f"plain={plain_status}, curl_cffi={cf_status}"
        result.passed = cf_status == 200

    return result


def _probe_session(
    page_url: str,
    api_url: str,
    api_params: dict | None = None,
    api_headers: dict | None = None,
    cookie_name: str = "JSESSIONID",
    expect_json_key: str = "",
) -> ProbeStepResult:
    """Session cookie lifecycle probe (HKEX/BMV pattern)."""
    result = ProbeStepResult(step_name="session_lifecycle", method="session")
    t0 = time.time()
    try:
        import requests
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        })

        # Step 1: Load page to get session cookie
        r1 = session.get(page_url, timeout=_PROBE_TIMEOUT)
        has_cookie = cookie_name in session.cookies
        if not has_cookie:
            # Some sites use different cookie names
            has_cookie = len(session.cookies) > 0

        if not has_cookie and r1.status_code != 200:
            result.error = f"Page load failed: HTTP {r1.status_code}"
            result.latency_ms = int((time.time() - t0) * 1000)
            return result

        # Step 2: API call with session
        hdrs = api_headers or {}
        r2 = session.get(api_url, params=api_params, headers=hdrs, timeout=_PROBE_TIMEOUT)

        if r2.status_code != 200:
            result.error = f"API call failed: HTTP {r2.status_code}"
            result.latency_ms = int((time.time() - t0) * 1000)
            return result

        # Step 3: Validate response
        if expect_json_key:
            try:
                data = r2.json()
                if expect_json_key in str(data):
                    result.passed = True
                    result.detail = f"Session OK, got {expect_json_key} in response"
                else:
                    result.error = f"Response missing '{expect_json_key}'"
            except Exception:
                result.error = "Response not valid JSON"
        else:
            result.passed = True
            result.detail = f"Session OK, HTTP {r2.status_code}"

    except Exception as exc:
        result.error = f"Session probe failed: {str(exc)[:100]}"
    result.latency_ms = int((time.time() - t0) * 1000)
    return result


def _probe_referer(
    url: str,
    required_referer: str,
    params: dict | None = None,
) -> ProbeStepResult:
    """Referer-gated API probe (BSE India pattern)."""
    result = ProbeStepResult(step_name="referer_gate", method="referer")
    t0 = time.time()
    try:
        import requests
        # Without Referer
        r1 = requests.get(url, params=params, timeout=_PROBE_TIMEOUT)
        no_ref_status = r1.status_code
        no_ref_len = len(r1.content)

        # With Referer
        r2 = requests.get(url, params=params, timeout=_PROBE_TIMEOUT,
                          headers={"Referer": required_referer})
        with_ref_status = r2.status_code
        with_ref_len = len(r2.content)

        if with_ref_status == 200 and with_ref_len > no_ref_len * 2:
            result.passed = True
            result.detail = f"Referer required (without: {no_ref_len}B, with: {with_ref_len}B)"
        elif with_ref_status == 200:
            result.passed = True
            result.detail = f"API working (Referer may not be strictly required)"
        else:
            result.error = f"API failed even with Referer: HTTP {with_ref_status}"

    except Exception as exc:
        result.error = f"Referer probe failed: {str(exc)[:100]}"
    result.latency_ms = int((time.time() - t0) * 1000)
    return result


def _probe_schema(
    url: str,
    market_id: str,
    headers: dict | None = None,
    params: dict | None = None,
    json_path: str = "",
) -> ProbeStepResult:
    """API schema diff probe -- detect structural changes vs baseline."""
    result = ProbeStepResult(step_name="schema_diff", method="schema")
    t0 = time.time()
    try:
        import requests
        resp = requests.get(url, headers=headers or {}, params=params,
                            timeout=_PROBE_TIMEOUT)
        if resp.status_code != 200:
            result.error = f"HTTP {resp.status_code}"
            result.latency_ms = int((time.time() - t0) * 1000)
            return result

        try:
            data = resp.json()
        except Exception:
            result.error = "Response not JSON"
            result.latency_ms = int((time.time() - t0) * 1000)
            return result

        # Extract schema (top-level keys + types)
        current_schema = _extract_schema(data, json_path)

        # Compare against baseline
        baseline = _load_baseline(market_id)
        if baseline is None:
            # No baseline yet -- save current as baseline
            _save_baseline(market_id, current_schema)
            result.passed = True
            result.detail = "Baseline created (first probe)"
        else:
            drift_fields = _compare_schemas(baseline, current_schema)
            if drift_fields:
                result.passed = True  # API works, just changed
                result.detail = f"Schema drift detected: {', '.join(drift_fields[:5])}"
            else:
                result.passed = True
                result.detail = "Schema matches baseline"

    except Exception as exc:
        result.error = f"Schema probe failed: {str(exc)[:100]}"
    result.latency_ms = int((time.time() - t0) * 1000)
    return result


def _extract_schema(data: Any, json_path: str = "") -> dict:
    """Extract a simplified schema from JSON data."""
    if json_path:
        for key in json_path.split("."):
            if isinstance(data, dict):
                data = data.get(key, {})
            elif isinstance(data, list) and data:
                data = data[0]
    if isinstance(data, dict):
        return {k: type(v).__name__ for k, v in data.items()}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return {k: type(v).__name__ for k, v in data[0].items()}
    return {"_type": type(data).__name__}


def _load_baseline(market_id: str) -> dict | None:
    path = _BASELINE_DIR / f"{market_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def _save_baseline(market_id: str, schema: dict) -> None:
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = _BASELINE_DIR / f"{market_id}.json"
    path.write_text(json.dumps(schema, indent=2))


def _compare_schemas(baseline: dict, current: dict) -> list[str]:
    """Compare two schemas and return list of changed field names."""
    diffs = []
    all_keys = set(baseline.keys()) | set(current.keys())
    for key in all_keys:
        if key not in baseline:
            diffs.append(f"+{key}")
        elif key not in current:
            diffs.append(f"-{key}")
        elif baseline[key] != current[key]:
            diffs.append(f"~{key}")
    return diffs


# ---------------------------------------------------------------------------
# Per-market probe configurations
# ---------------------------------------------------------------------------

def _today_str():
    return date.today().strftime("%Y%m%d")

def _days_ago_str(n: int):
    return (date.today() - timedelta(days=n)).strftime("%Y%m%d")


def _build_probe_configs() -> dict[str, dict[str, Any]]:
    """Build the probe configuration registry for all 25 markets."""
    return {
        # === Tier 1 ===
        "us_sec_edgar": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "data.sec.gov"}},
                {"fn": _probe_tls, "args": {"host": "data.sec.gov"}},
                {"fn": _probe_http, "args": {
                    "url": "https://www.sec.gov/files/company_tickers.json",
                    "headers": {"User-Agent": "Operator1/1.0 probe@example.com"},
                }},
                {"fn": _probe_schema, "args": {
                    "url": "https://www.sec.gov/files/company_tickers.json",
                    "market_id": "us_sec_edgar",
                    "headers": {"User-Agent": "Operator1/1.0 probe@example.com"},
                    "json_path": "0",
                }},
            ],
        },
        "uk_companies_house": {
            "pattern": "api_key",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "api.company-information.service.gov.uk"}},
                {"fn": _probe_tls, "args": {"host": "api.company-information.service.gov.uk"}},
            ],
        },
        "eu_esef": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "filings.xbrl.org"}},
                {"fn": _probe_http, "args": {
                    "url": "https://filings.xbrl.org/api/filings",
                    "headers": {"Accept": "application/json"},
                }},
                {"fn": _probe_schema, "args": {
                    "url": "https://filings.xbrl.org/api/filings?entity_name=Unilever&limit=1",
                    "market_id": "eu_esef",
                    "json_path": "data",
                }},
            ],
        },
        "fr_esef": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_http, "args": {
                    "url": "https://filings.xbrl.org/api/filings?country=FR&limit=1",
                }},
            ],
        },
        "de_esef": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_http, "args": {
                    "url": "https://filings.xbrl.org/api/filings?country=DE&limit=1",
                }},
            ],
        },
        "jp_jquants": {
            "pattern": "api_key",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "api.jquants.com"}},
                {"fn": _probe_tls, "args": {"host": "api.jquants.com"}},
            ],
        },
        "kr_dart": {
            "pattern": "api_key",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "opendart.fss.or.kr"}},
                {"fn": _probe_http, "args": {
                    "url": "https://opendart.fss.or.kr",
                }},
            ],
        },
        "tw_mops": {
            "pattern": "waf",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "mops.twse.com.tw"}},
                {"fn": _probe_waf, "args": {
                    "url": "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
                    "label": "twse_waf",
                }},
            ],
        },
        "br_cvm": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "dados.cvm.gov.br"}},
                {"fn": _probe_http, "args": {
                    "url": "https://dados.cvm.gov.br/api/3/action/package_list",
                }},
            ],
        },
        "cl_cmf": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "www.cmfchile.cl"}},
                {"fn": _probe_http, "args": {"url": "https://www.cmfchile.cl"}},
            ],
        },

        # === Tier 2 ===
        "hk_hkex": {
            "pattern": "session",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "www1.hkexnews.hk"}},
                {"fn": _probe_tls, "args": {"host": "www1.hkexnews.hk"}},
                {"fn": _probe_session, "args": {
                    "page_url": "https://www1.hkexnews.hk/search/titlesearch.xhtml",
                    "api_url": "https://www1.hkexnews.hk/search/titleSearchServlet.do",
                    "api_params": {
                        "searchType": "1",
                        "stockCode": "00700",
                        "t1code": "50000",
                        "t2Gcode": "-2",
                        "t2code": "-2",
                        "rowRange": "10",
                        "from": _days_ago_str(14),
                        "to": _today_str(),
                    },
                    "api_headers": {
                        "X-Requested-With": "XMLHttpRequest",
                        "Referer": "https://www1.hkexnews.hk/search/titlesearch.xhtml",
                        "Accept": "application/json, text/javascript, */*; q=0.01",
                    },
                    "cookie_name": "JSESSIONID",
                    "expect_json_key": "recordCnt",
                }},
            ],
        },
        "sg_sgx": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "api.sgx.com"}},
                {"fn": _probe_http, "args": {
                    "url": "https://api.sgx.com/securities/v1.1?type=stocks&pagestart=0&pagesize=5",
                    "headers": {"Accept": "application/json"},
                }},
                {"fn": _probe_schema, "args": {
                    "url": "https://api.sgx.com/securities/v1.1?type=stocks&pagestart=0&pagesize=5",
                    "market_id": "sg_sgx",
                    "headers": {"Accept": "application/json"},
                    "json_path": "data.prices",
                }},
            ],
        },
        "sa_tadawul": {
            "pattern": "waf",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "www.saudiexchange.sa"}},
                {"fn": _probe_tls, "args": {"host": "www.saudiexchange.sa"}},
                {"fn": _probe_waf, "args": {
                    "url": "https://www.saudiexchange.sa/wps/portal/saudiexchange/ourmarkets/main-market/market-watch",
                    "label": "tadawul_waf",
                }},
            ],
        },
        "in_bse": {
            "pattern": "referer",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "api.bseindia.com"}},
                {"fn": _probe_referer, "args": {
                    "url": "https://api.bseindia.com/BseIndiaAPI/api/GetSubCodesByGroup/w",
                    "required_referer": "https://www.bseindia.com",
                    "params": {"Group": "A", "language": "en"},
                }},
            ],
        },
        "cn_sse": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "www.sse.com.cn"}},
                {"fn": _probe_http, "args": {"url": "http://www.sse.com.cn"}},
            ],
        },
        "ca_sedar": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "www.sedarplus.ca"}},
                {"fn": _probe_http, "args": {"url": "https://www.sedarplus.ca"}},
            ],
        },
        "au_asx": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "asx.api.markitdigital.com"}},
                {"fn": _probe_http, "args": {
                    "url": "https://asx.api.markitdigital.com/asx-research/1.0/companies/BHP/header",
                    "headers": {"Accept": "application/json"},
                }},
            ],
        },
        "za_jse": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "www.jse.co.za"}},
                {"fn": _probe_http, "args": {"url": "https://www.jse.co.za"}},
            ],
        },
        "mx_bmv": {
            "pattern": "session",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "www.bmv.com.mx"}},
                {"fn": _probe_http, "args": {"url": "https://www.bmv.com.mx"}},
            ],
        },
        "ae_dfm": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "www.dfm.ae"}},
                {"fn": _probe_http, "args": {"url": "https://www.dfm.ae"}},
            ],
        },
        "ch_six": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_dns, "args": {"host": "www.six-group.com"}},
                {"fn": _probe_http, "args": {"url": "https://www.six-group.com"}},
            ],
        },
        "nl_esef": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_http, "args": {
                    "url": "https://filings.xbrl.org/api/filings?country=NL&limit=1",
                }},
            ],
        },
        "es_esef": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_http, "args": {
                    "url": "https://filings.xbrl.org/api/filings?country=ES&limit=1",
                }},
            ],
        },
        "it_esef": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_http, "args": {
                    "url": "https://filings.xbrl.org/api/filings?country=IT&limit=1",
                }},
            ],
        },
        "se_esef": {
            "pattern": "free_api",
            "probes": [
                {"fn": _probe_http, "args": {
                    "url": "https://filings.xbrl.org/api/filings?country=SE&limit=1",
                }},
            ],
        },
    }


WRAPPER_PROBES = _build_probe_configs()


# ---------------------------------------------------------------------------
# Probe execution
# ---------------------------------------------------------------------------

def run_deep_probe(market_id: str) -> WrapperProbeResult:
    """Run deep probes for a single market."""
    config = WRAPPER_PROBES.get(market_id)
    if not config:
        return WrapperProbeResult(
            market_id=market_id,
            status="unknown",
            probe_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    result = WrapperProbeResult(
        market_id=market_id,
        pattern=config.get("pattern", "unknown"),
        probe_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    all_passed = True
    any_passed = False
    schema_drift = False

    for probe_cfg in config.get("probes", []):
        fn = probe_cfg["fn"]
        args = probe_cfg.get("args", {})
        try:
            step_result = fn(**args)
        except Exception as exc:
            step_result = ProbeStepResult(
                step_name=fn.__name__,
                error=f"Probe crashed: {str(exc)[:100]}",
            )

        result.steps.append(asdict(step_result))
        result.total_latency_ms += step_result.latency_ms

        if step_result.passed:
            any_passed = True
        else:
            all_passed = False

        # Detect schema drift
        if step_result.method == "schema" and "drift" in step_result.detail.lower():
            schema_drift = True
            result.schema_drift = True
            # Extract drift fields from detail
            if ":" in step_result.detail:
                fields_str = step_result.detail.split(":", 1)[1].strip()
                result.schema_diff_fields = [f.strip() for f in fields_str.split(",")]

    # Classify status
    if all_passed and not schema_drift:
        result.status = "working"
    elif all_passed and schema_drift:
        result.status = "restructured"
    elif any_passed:
        # Check for specific failure patterns
        step_errors = " ".join(s.get("error", "") for s in result.steps)
        if "403" in step_errors and "WAF" in step_errors:
            result.status = "waf_blocked"
        elif "403" in step_errors and "geo" in step_errors.lower():
            result.status = "geo_blocked"
        elif "429" in step_errors:
            result.status = "rate_limited"
        else:
            result.status = "partial"
    else:
        result.status = "down"

    return result


def run_deep_probes(
    markets: list[str] | None = None,
) -> dict[str, WrapperProbeResult]:
    """Run deep probes on all (or specified) markets.

    Returns dict mapping market_id -> WrapperProbeResult.
    """
    if markets is None:
        markets = list(WRAPPER_PROBES.keys())

    results: dict[str, WrapperProbeResult] = {}
    for market_id in markets:
        logger.info("Deep probing %s ...", market_id)
        try:
            results[market_id] = run_deep_probe(market_id)
            logger.info(
                "  [%s] %s: %s (%dms, pattern=%s)",
                "+" if results[market_id].status == "working" else "!",
                market_id,
                results[market_id].status,
                results[market_id].total_latency_ms,
                results[market_id].pattern,
            )
        except Exception as exc:
            results[market_id] = WrapperProbeResult(
                market_id=market_id,
                status="down",
                probe_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            logger.warning("  Deep probe crashed for %s: %s", market_id, exc)

    return results

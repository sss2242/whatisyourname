"""Wrapper health check, auto-healing, and alerting system.

Monitors all 25 PIT market wrappers and detects when an API breaks
(like CMF Chile going 404 or filings.xbrl.org having 0 DE filings).
When a primary path fails, automatically routes to the best available
fallback and emits alerts.

Probe levels:
  L0: DNS/TLS -- can we connect?           (0 API calls)
  L1: Search  -- does company search work?  (1 API call)
  L2: Profile -- can we fetch a profile?    (1 API call)
  L3: Financials -- do we get data?         (1 API call)

Status values:
  healthy  -- Primary path works at L3
  degraded -- Primary broken, fallback works
  critical -- All paths failing
  unknown  -- Not yet checked

Usage:
    # CLI: check all markets
    python -m operator1.monitoring.health_check

    # CLI: check specific market
    python -m operator1.monitoring.health_check --market us_sec_edgar

    # Python API
    from operator1.monitoring.health_check import run_health_check
    report = run_health_check()
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_HEALTH_FILE = Path("cache/wrapper_health.json")
_HISTORY_FILE = Path("cache/wrapper_health_history.jsonl")


# ---------------------------------------------------------------------------
# Canary companies -- one well-known company per market for health probes
# ---------------------------------------------------------------------------

CANARY_COMPANIES: dict[str, dict[str, str]] = {
    # Tier 1
    "us_sec_edgar": {"identifier": "AAPL", "name": "Apple"},
    "uk_companies_house": {"identifier": "Unilever", "name": "Unilever"},
    "eu_esef": {"identifier": "Unilever", "name": "Unilever NV"},
    "fr_esef": {"identifier": "BNP Paribas", "name": "BNP Paribas"},
    "de_esef": {"identifier": "Siemens Aktiengesellschaft", "name": "Siemens AG"},
    "jp_jquants": {"identifier": "7203", "name": "Toyota Motor"},
    "kr_dart": {"identifier": "005930", "name": "Samsung Electronics"},
    "tw_mops": {"identifier": "2330", "name": "TSMC"},
    "br_cvm": {"identifier": "PETR4", "name": "Petrobras"},
    "cl_cmf": {"identifier": "SQM-B", "name": "SQM"},
    # Tier 2
    "in_bse": {"identifier": "500325", "name": "Reliance Industries"},
    "cn_sse": {"identifier": "600519", "name": "Kweichow Moutai"},
    "hk_hkex": {"identifier": "00700", "name": "Tencent"},
    "ca_sedar": {"identifier": "RY", "name": "Royal Bank of Canada"},
    "au_asx": {"identifier": "BHP", "name": "BHP Group"},
    "sg_sgx": {"identifier": "D05", "name": "DBS Group"},
    "sa_tadawul": {"identifier": "2222", "name": "Saudi Aramco"},
    "ch_six": {"identifier": "NESN", "name": "Nestle"},
    "za_jse": {"identifier": "NPN", "name": "Naspers"},
    "mx_bmv": {"identifier": "WALMEX", "name": "Walmart Mexico"},
    "ae_dfm": {"identifier": "EMAAR", "name": "Emaar Properties"},
    "nl_esef": {"identifier": "Heineken", "name": "Heineken NV"},
    "es_esef": {"identifier": "Iberdrola", "name": "Iberdrola"},
    "it_esef": {"identifier": "Enel", "name": "Enel SpA"},
    "se_esef": {"identifier": "Volvo", "name": "Volvo AB"},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Result of a single probe level."""
    level: str = ""           # L0, L1, L2, L3
    passed: bool = False
    latency_ms: int = 0
    error: str = ""
    detail: str = ""


@dataclass
class MarketHealth:
    """Health status for a single market."""
    market_id: str = ""
    status: str = "unknown"       # healthy, degraded, critical, unknown
    level_passed: str = ""        # highest level passed (L0-L3)
    active_path: str = ""         # which extraction path is working
    fallback_available: bool = False
    last_success: str = ""
    last_failure: str = ""
    latency_ms: int = 0
    consecutive_failures: int = 0
    degraded_reason: str = ""
    probes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("probes", None)
        return d


@dataclass
class HealthReport:
    """Full health report across all markets."""
    last_check: str = ""
    total_markets: int = 0
    healthy: int = 0
    degraded: int = 0
    critical: int = 0
    unknown: int = 0
    markets: dict[str, MarketHealth] = field(default_factory=dict)

    def summary_line(self) -> str:
        return (
            f"Health: {self.healthy} healthy, {self.degraded} degraded, "
            f"{self.critical} critical, {self.unknown} unknown "
            f"(of {self.total_markets})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_check": self.last_check,
            "total_markets": self.total_markets,
            "healthy": self.healthy,
            "degraded": self.degraded,
            "critical": self.critical,
            "unknown": self.unknown,
            "markets": {
                mid: mh.to_dict() for mid, mh in self.markets.items()
            },
        }


# ---------------------------------------------------------------------------
# Probe implementations
# ---------------------------------------------------------------------------

def _probe_l0_connectivity(market_id: str) -> ProbeResult:
    """L0: DNS/TLS connectivity check (no API calls)."""
    from operator1.clients.pit_registry import MARKETS

    result = ProbeResult(level="L0")
    market = MARKETS.get(market_id)
    if not market:
        result.error = f"Unknown market: {market_id}"
        return result

    url = market.pit_api_url
    if not url:
        result.error = "No API URL configured"
        return result

    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        t0 = time.time()
        sock = socket.create_connection((host, port), timeout=10)
        sock.close()
        result.latency_ms = int((time.time() - t0) * 1000)
        result.passed = True
        result.detail = f"Connected to {host}:{port}"
    except Exception as exc:
        result.error = f"Connection failed: {exc}"
        result.latency_ms = 10000

    return result


def _probe_l1_search(market_id: str, secrets: dict | None = None) -> ProbeResult:
    """L1: Company search probe (1 API call)."""
    result = ProbeResult(level="L1")
    canary = CANARY_COMPANIES.get(market_id)
    if not canary:
        result.error = f"No canary company for {market_id}"
        return result

    try:
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client(market_id, secrets or {})

        t0 = time.time()
        results = client.search_company(canary["identifier"])
        result.latency_ms = int((time.time() - t0) * 1000)

        if results and len(results) > 0:
            result.passed = True
            result.detail = f"Found {len(results)} results for '{canary['identifier']}'"
        else:
            result.error = f"Search returned 0 results for '{canary['identifier']}'"
    except Exception as exc:
        result.error = f"Search failed: {str(exc)[:200]}"

    return result


def _probe_l2_profile(market_id: str, secrets: dict | None = None) -> ProbeResult:
    """L2: Profile fetch probe (1 API call)."""
    result = ProbeResult(level="L2")
    canary = CANARY_COMPANIES.get(market_id)
    if not canary:
        result.error = f"No canary company for {market_id}"
        return result

    try:
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client(market_id, secrets or {})

        t0 = time.time()
        profile = client.get_profile(canary["identifier"])
        result.latency_ms = int((time.time() - t0) * 1000)

        if profile and profile.get("name"):
            result.passed = True
            result.detail = f"Profile: {profile.get('name', '?')}"
        else:
            result.error = "Profile returned empty or no name"
    except Exception as exc:
        result.error = f"Profile failed: {str(exc)[:200]}"

    return result


def _probe_l3_financials(market_id: str, secrets: dict | None = None) -> ProbeResult:
    """L3: Financial data probe (1 API call)."""
    result = ProbeResult(level="L3")
    canary = CANARY_COMPANIES.get(market_id)
    if not canary:
        result.error = f"No canary company for {market_id}"
        return result

    try:
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client(market_id, secrets or {})

        t0 = time.time()
        # Try balance sheet first (most commonly available)
        df = client.get_balance_sheet(canary["identifier"])
        result.latency_ms = int((time.time() - t0) * 1000)

        if df is not None and not df.empty:
            result.passed = True
            result.detail = f"Balance sheet: {len(df)} rows"
        else:
            # Try income statement as fallback
            df = client.get_income_statement(canary["identifier"])
            if df is not None and not df.empty:
                result.passed = True
                result.detail = f"Income statement: {len(df)} rows"
            else:
                result.error = "Financials returned empty for both balance sheet and income statement"
    except Exception as exc:
        result.error = f"Financials failed: {str(exc)[:200]}"

    return result


# ---------------------------------------------------------------------------
# Market health assessment
# ---------------------------------------------------------------------------

def check_market_health(
    market_id: str,
    secrets: dict | None = None,
    max_level: str = "L3",
    previous: MarketHealth | None = None,
) -> MarketHealth:
    """Run health probes on a single market.

    Parameters
    ----------
    market_id : str
        Market identifier from pit_registry.
    secrets : dict, optional
        API keys for markets that require them.
    max_level : str
        Highest probe level to run ("L0", "L1", "L2", "L3").
    previous : MarketHealth, optional
        Previous health state for tracking consecutive failures.

    Returns
    -------
    MarketHealth with status, probes, and routing info.
    """
    health = MarketHealth(market_id=market_id)
    now = datetime.now(timezone.utc).isoformat()

    if previous:
        health.consecutive_failures = previous.consecutive_failures

    levels = ["L0", "L1", "L2", "L3"]
    target_idx = levels.index(max_level) if max_level in levels else 3

    probes: list[ProbeResult] = []
    highest_passed = ""

    for i, level in enumerate(levels[:target_idx + 1]):
        if level == "L0":
            probe = _probe_l0_connectivity(market_id)
        elif level == "L1":
            probe = _probe_l1_search(market_id, secrets)
        elif level == "L2":
            probe = _probe_l2_profile(market_id, secrets)
        elif level == "L3":
            probe = _probe_l3_financials(market_id, secrets)
        else:
            continue

        probes.append(probe)
        health.probes.append(asdict(probe))

        if probe.passed:
            highest_passed = level
            health.latency_ms = max(health.latency_ms, probe.latency_ms)
        else:
            # Don't probe higher levels if a lower one failed
            # (except L0 -> L1 since some wrappers use curl_cffi not TCP)
            if level != "L0":
                break

    health.level_passed = highest_passed

    # Determine status
    if highest_passed == "L3":
        health.status = "healthy"
        health.active_path = "primary"
        health.last_success = now
        health.consecutive_failures = 0
    elif highest_passed in ("L1", "L2"):
        health.status = "degraded"
        health.degraded_reason = (
            f"Primary works up to {highest_passed} but financials (L3) failed: "
            + (probes[-1].error if probes else "unknown")
        )
        health.active_path = "fallback"
        health.fallback_available = True
        health.last_failure = now
        health.consecutive_failures += 1
    elif highest_passed == "L0":
        health.status = "degraded"
        health.degraded_reason = "Connected but search/profile failed"
        health.active_path = "fallback"
        health.last_failure = now
        health.consecutive_failures += 1
    else:
        health.status = "critical"
        health.degraded_reason = "Cannot connect to API"
        health.active_path = "none"
        health.last_failure = now
        health.consecutive_failures += 1

    return health


# ---------------------------------------------------------------------------
# Full health check runner
# ---------------------------------------------------------------------------

def run_health_check(
    markets: list[str] | None = None,
    secrets: dict | None = None,
    max_level: str = "L3",
    quick: bool = False,
) -> HealthReport:
    """Run health checks on all (or specified) markets.

    Parameters
    ----------
    markets : list[str], optional
        Specific markets to check. Defaults to all registered markets.
    secrets : dict, optional
        API keys dictionary.
    max_level : str
        Highest probe level ("L0" for quick, "L3" for full).
    quick : bool
        If True, only run L0+L1 (no API-intensive probes).

    Returns
    -------
    HealthReport with per-market status.
    """
    if quick:
        max_level = "L1"

    from operator1.clients.pit_registry import MARKETS

    if markets is None:
        markets = list(MARKETS.keys())

    # Load previous state for tracking
    previous_report = load_health_report()
    previous_states: dict[str, MarketHealth] = {}
    if previous_report:
        previous_states = previous_report.markets

    report = HealthReport(
        last_check=datetime.now(timezone.utc).isoformat(),
        total_markets=len(markets),
    )

    for market_id in markets:
        logger.info("Checking %s ...", market_id)
        try:
            health = check_market_health(
                market_id,
                secrets=secrets,
                max_level=max_level,
                previous=previous_states.get(market_id),
            )
        except Exception as exc:
            health = MarketHealth(
                market_id=market_id,
                status="critical",
                degraded_reason=f"Health check crashed: {exc}",
            )

        report.markets[market_id] = health

        # Count statuses
        if health.status == "healthy":
            report.healthy += 1
        elif health.status == "degraded":
            report.degraded += 1
        elif health.status == "critical":
            report.critical += 1
        else:
            report.unknown += 1

        # Log result
        icon = {"healthy": "+", "degraded": "~", "critical": "!", "unknown": "?"}.get(health.status, "?")
        logger.info(
            "  [%s] %s: %s (level=%s, latency=%dms)",
            icon, market_id, health.status, health.level_passed, health.latency_ms,
        )

    # Run deep probes (L4: per-wrapper pattern-specific probes)
    try:
        from operator1.monitoring.wrapper_probes import run_deep_probes
        deep_results = run_deep_probes(markets)
        # Attach deep probe results to each market health entry
        for mid, probe_result in deep_results.items():
            if mid in report.markets:
                report.markets[mid].probes.append({
                    "level": "L4_deep",
                    "status": probe_result.status,
                    "pattern": probe_result.pattern,
                    "total_latency_ms": probe_result.total_latency_ms,
                    "schema_drift": probe_result.schema_drift,
                    "steps": probe_result.steps,
                })
                # Upgrade degraded to restructured if schema drift detected
                if probe_result.schema_drift and report.markets[mid].status == "healthy":
                    report.markets[mid].status = "degraded"
                    report.markets[mid].degraded_reason = (
                        f"API schema drift detected: {', '.join(probe_result.schema_diff_fields[:3])}"
                    )
        logger.info("Deep probes complete: %d markets probed", len(deep_results))
    except Exception as exc:
        logger.warning("Deep probes failed (non-fatal): %s", exc)

    # Detect status changes and alert
    _detect_and_alert(report, previous_report)

    # Save report
    save_health_report(report)
    _append_history(report)

    logger.info(report.summary_line())
    return report


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_health_report(report: HealthReport) -> None:
    """Save health report to disk."""
    _HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_HEALTH_FILE, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)


def load_health_report() -> HealthReport | None:
    """Load the most recent health report from disk."""
    if not _HEALTH_FILE.exists():
        return None
    try:
        with open(_HEALTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        report = HealthReport(
            last_check=data.get("last_check", ""),
            total_markets=data.get("total_markets", 0),
            healthy=data.get("healthy", 0),
            degraded=data.get("degraded", 0),
            critical=data.get("critical", 0),
            unknown=data.get("unknown", 0),
        )
        for mid, mdata in data.get("markets", {}).items():
            report.markets[mid] = MarketHealth(**{
                k: v for k, v in mdata.items()
                if k in MarketHealth.__dataclass_fields__
            })
        return report
    except Exception:
        return None


def _append_history(report: HealthReport) -> None:
    """Append a summary line to the health history log."""
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": report.last_check,
        "healthy": report.healthy,
        "degraded": report.degraded,
        "critical": report.critical,
        "markets": {
            mid: mh.status for mid, mh in report.markets.items()
        },
    }
    with open(_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Auto-healing: path routing
# ---------------------------------------------------------------------------

def get_active_path(market_id: str) -> str:
    """Get the recommended extraction path for a market.

    Wrappers can call this before attempting extraction to skip
    known-dead paths and go straight to the working fallback.

    Returns
    -------
    str
        "primary", "fallback", or "none"
    """
    report = load_health_report()
    if not report:
        return "primary"  # no health data, try primary

    health = report.markets.get(market_id)
    if not health:
        return "primary"

    return health.active_path or "primary"


def is_market_healthy(market_id: str) -> bool:
    """Quick check if a market's primary path is working."""
    report = load_health_report()
    if not report:
        return True  # assume healthy if no data
    health = report.markets.get(market_id)
    if not health:
        return True
    return health.status == "healthy"


def get_degraded_markets() -> list[str]:
    """Return list of market_ids that are degraded or critical."""
    report = load_health_report()
    if not report:
        return []
    return [
        mid for mid, mh in report.markets.items()
        if mh.status in ("degraded", "critical")
    ]


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

def _detect_and_alert(
    current: HealthReport,
    previous: HealthReport | None,
) -> None:
    """Detect status changes and emit alerts."""
    if not previous:
        return

    for market_id, health in current.markets.items():
        prev = previous.markets.get(market_id)
        if not prev:
            continue

        if prev.status != health.status:
            _emit_alert(market_id, prev.status, health.status, health.degraded_reason)


def _emit_alert(
    market_id: str,
    old_status: str,
    new_status: str,
    reason: str = "",
) -> None:
    """Emit an alert for a status change."""
    icons = {"healthy": "+", "degraded": "~", "critical": "!", "unknown": "?"}
    old_icon = icons.get(old_status, "?")
    new_icon = icons.get(new_status, "?")

    msg = f"[{new_icon}] {market_id}: {old_status} -> {new_status}"
    if reason:
        msg += f" ({reason})"

    if new_status == "critical":
        logger.error("ALERT: %s", msg)
    elif new_status == "degraded":
        logger.warning("ALERT: %s", msg)
    elif new_status == "healthy" and old_status in ("degraded", "critical"):
        logger.info("RECOVERED: %s", msg)

    # Optional webhook
    webhook_url = os.environ.get("HEALTH_WEBHOOK_URL")
    if webhook_url:
        _send_webhook(webhook_url, market_id, old_status, new_status, reason)


def _send_webhook(
    url: str,
    market_id: str,
    old_status: str,
    new_status: str,
    reason: str = "",
) -> None:
    """Send a webhook notification for status changes."""
    try:
        import requests
        payload = {
            "text": (
                f"Wrapper health change: **{market_id}** "
                f"{old_status} -> {new_status}"
                + (f"\nReason: {reason}" if reason else "")
            ),
            "market_id": market_id,
            "old_status": old_status,
            "new_status": new_status,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        requests.post(url, json=payload, timeout=10)
    except Exception as exc:
        logger.debug("Webhook notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Pre-flight check for pipeline
# ---------------------------------------------------------------------------

def preflight_check(
    market_id: str,
    secrets: dict | None = None,
) -> tuple[bool, str]:
    """Quick pre-flight health check before running a pipeline.

    Called from run.py to warn the user if the target market has issues.

    Parameters
    ----------
    market_id : str
        Target market for the pipeline run.
    secrets : dict, optional
        API keys.

    Returns
    -------
    (ok, message)
        ok is True if the market is healthy or degraded-with-fallback.
        message describes the status.
    """
    report = load_health_report()

    if report:
        health = report.markets.get(market_id)
        if health:
            age_hours = 0
            try:
                last = datetime.fromisoformat(report.last_check)
                age_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
            except Exception:
                pass

            if age_hours < 24:
                if health.status == "healthy":
                    return True, f"{market_id}: healthy (last checked {age_hours:.0f}h ago)"
                elif health.status == "degraded":
                    return True, (
                        f"{market_id}: degraded -- using fallback. "
                        f"Reason: {health.degraded_reason}"
                    )
                elif health.status == "critical":
                    return False, (
                        f"{market_id}: CRITICAL -- all paths failing. "
                        f"Reason: {health.degraded_reason}"
                    )

    # No recent health data -- run a quick L1 probe
    logger.info("No recent health data for %s, running quick probe...", market_id)
    health = check_market_health(market_id, secrets=secrets, max_level="L1")

    if health.status in ("healthy", "degraded"):
        return True, f"{market_id}: {health.status} (fresh probe)"
    return False, f"{market_id}: {health.status} -- {health.degraded_reason}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for health checks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check health of all PIT market wrappers",
    )
    parser.add_argument(
        "--market", "-m",
        help="Check a specific market (e.g. us_sec_edgar)",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: L0+L1 only (no financial data probes)",
    )
    parser.add_argument(
        "--level",
        choices=["L0", "L1", "L2", "L3"],
        default="L3",
        help="Maximum probe level (default: L3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report to stdout",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    markets = [args.market] if args.market else None
    report = run_health_check(
        markets=markets,
        max_level=args.level,
        quick=args.quick,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print()
        print(f"  {report.summary_line()}")
        print()
        for mid, mh in sorted(report.markets.items()):
            icon = {"healthy": "+", "degraded": "~", "critical": "!", "unknown": "?"}.get(mh.status, "?")
            line = f"  [{icon}] {mid:25s} {mh.status:10s} {mh.level_passed:3s} {mh.latency_ms:5d}ms"
            if mh.degraded_reason:
                line += f"  -- {mh.degraded_reason[:60]}"
            print(line)
        print()


if __name__ == "__main__":
    main()

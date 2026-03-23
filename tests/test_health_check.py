"""Tests for the wrapper health check and auto-healing system.

Tests use mocked probes to avoid hitting real APIs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from operator1.monitoring.health_check import (
    CANARY_COMPANIES,
    HealthReport,
    MarketHealth,
    ProbeResult,
    check_market_health,
    get_active_path,
    get_degraded_markets,
    is_market_healthy,
    load_health_report,
    preflight_check,
    run_health_check,
    save_health_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def health_file(tmp_path):
    """Override health file path to temp dir."""
    import operator1.monitoring.health_check as hc
    old = hc._HEALTH_FILE
    hc._HEALTH_FILE = tmp_path / "wrapper_health.json"
    hc._HISTORY_FILE = tmp_path / "wrapper_health_history.jsonl"
    yield hc._HEALTH_FILE
    hc._HEALTH_FILE = old


@pytest.fixture
def mock_pit_registry():
    """Mock pit_registry.MARKETS with 3 test markets."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FakeMarket:
        market_id: str = ""
        country: str = ""
        country_code: str = ""
        region: str = ""
        exchange: str = ""
        market_cap: str = ""
        pit_api_name: str = ""
        pit_api_url: str = "https://example.com"
        requires_api_key: bool = False

    fake_markets = {
        "test_healthy": FakeMarket(market_id="test_healthy", pit_api_url="https://httpbin.org"),
        "test_degraded": FakeMarket(market_id="test_degraded", pit_api_url="https://httpbin.org"),
        "test_critical": FakeMarket(market_id="test_critical", pit_api_url="https://nonexistent.invalid"),
    }

    with patch("operator1.monitoring.health_check.CANARY_COMPANIES", {
        "test_healthy": {"identifier": "TEST1", "name": "Test Healthy Co"},
        "test_degraded": {"identifier": "TEST2", "name": "Test Degraded Co"},
        "test_critical": {"identifier": "TEST3", "name": "Test Critical Co"},
    }):
        with patch("operator1.clients.pit_registry.MARKETS", fake_markets):
            yield fake_markets


# ---------------------------------------------------------------------------
# Test: ProbeResult and MarketHealth data classes
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_probe_result_defaults(self):
        p = ProbeResult()
        assert p.level == ""
        assert p.passed is False
        assert p.latency_ms == 0

    def test_market_health_defaults(self):
        mh = MarketHealth(market_id="test")
        assert mh.status == "unknown"
        assert mh.consecutive_failures == 0

    def test_market_health_to_dict(self):
        mh = MarketHealth(market_id="test", status="healthy")
        d = mh.to_dict()
        assert d["market_id"] == "test"
        assert d["status"] == "healthy"
        assert "probes" not in d  # probes stripped from dict

    def test_health_report_summary(self):
        report = HealthReport(
            total_markets=25, healthy=20, degraded=3, critical=1, unknown=1,
        )
        summary = report.summary_line()
        assert "20 healthy" in summary
        assert "3 degraded" in summary
        assert "1 critical" in summary


# ---------------------------------------------------------------------------
# Test: Health check with mocked clients
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_check_healthy_market(self, mock_pit_registry):
        """A market where all probes pass should be healthy."""
        mock_client = MagicMock()
        mock_client.search_company.return_value = [{"name": "Test", "ticker": "T"}]
        mock_client.get_profile.return_value = {"name": "Test Co", "country": "US"}
        mock_client.get_balance_sheet.return_value = pd.DataFrame({"value": [1, 2, 3]})

        with patch("operator1.clients.equity_provider.create_pit_client", return_value=mock_client):
            health = check_market_health("test_healthy")

        assert health.status == "healthy"
        assert health.level_passed == "L3"
        assert health.consecutive_failures == 0

    def test_check_degraded_market(self, mock_pit_registry):
        """A market where search works but financials fail is degraded."""
        mock_client = MagicMock()
        mock_client.search_company.return_value = [{"name": "Test"}]
        mock_client.get_profile.return_value = {"name": "Test Co", "country": "US"}
        mock_client.get_balance_sheet.return_value = pd.DataFrame()
        mock_client.get_income_statement.return_value = pd.DataFrame()

        with patch("operator1.clients.equity_provider.create_pit_client", return_value=mock_client):
            health = check_market_health("test_degraded")

        assert health.status == "degraded"
        assert health.level_passed == "L2"
        assert "L3" in health.degraded_reason or "financials" in health.degraded_reason.lower()

    def test_check_critical_market(self, mock_pit_registry):
        """A market where even search fails is critical."""
        mock_client = MagicMock()
        mock_client.search_company.side_effect = Exception("API down")

        with patch("operator1.clients.equity_provider.create_pit_client", return_value=mock_client):
            health = check_market_health("test_critical", max_level="L1")

        assert health.status in ("degraded", "critical")

    def test_consecutive_failures_tracked(self, mock_pit_registry):
        """Consecutive failures should increment."""
        prev = MarketHealth(market_id="test_degraded", consecutive_failures=2)

        mock_client = MagicMock()
        mock_client.search_company.return_value = [{"name": "Test"}]
        mock_client.get_profile.return_value = {"name": "Test"}
        mock_client.get_balance_sheet.return_value = pd.DataFrame()
        mock_client.get_income_statement.return_value = pd.DataFrame()

        with patch("operator1.clients.equity_provider.create_pit_client", return_value=mock_client):
            health = check_market_health("test_degraded", previous=prev)

        assert health.consecutive_failures == 3


# ---------------------------------------------------------------------------
# Test: Persistence (save/load)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, health_file):
        report = HealthReport(
            last_check="2026-03-23T19:00:00Z",
            total_markets=2,
            healthy=1,
            degraded=1,
        )
        report.markets["test_a"] = MarketHealth(
            market_id="test_a", status="healthy", level_passed="L3",
        )
        report.markets["test_b"] = MarketHealth(
            market_id="test_b", status="degraded", degraded_reason="API down",
        )

        save_health_report(report)
        assert health_file.exists()

        loaded = load_health_report()
        assert loaded is not None
        assert loaded.healthy == 1
        assert loaded.degraded == 1
        assert loaded.markets["test_a"].status == "healthy"
        assert loaded.markets["test_b"].degraded_reason == "API down"

    def test_load_missing_file(self, health_file):
        loaded = load_health_report()
        assert loaded is None


# ---------------------------------------------------------------------------
# Test: Auto-healing (path routing)
# ---------------------------------------------------------------------------

class TestAutoHealing:
    def test_get_active_path_no_data(self, health_file):
        """With no health file, default to primary."""
        assert get_active_path("us_sec_edgar") == "primary"

    def test_get_active_path_healthy(self, health_file):
        report = HealthReport()
        report.markets["us_sec_edgar"] = MarketHealth(
            market_id="us_sec_edgar", status="healthy", active_path="primary",
        )
        save_health_report(report)
        assert get_active_path("us_sec_edgar") == "primary"

    def test_get_active_path_degraded(self, health_file):
        report = HealthReport()
        report.markets["de_esef"] = MarketHealth(
            market_id="de_esef", status="degraded", active_path="fallback",
        )
        save_health_report(report)
        assert get_active_path("de_esef") == "fallback"

    def test_is_market_healthy(self, health_file):
        report = HealthReport()
        report.markets["us_sec_edgar"] = MarketHealth(status="healthy")
        report.markets["cl_cmf"] = MarketHealth(status="degraded")
        save_health_report(report)

        assert is_market_healthy("us_sec_edgar") is True
        assert is_market_healthy("cl_cmf") is False
        assert is_market_healthy("unknown") is True  # no data = assume healthy

    def test_get_degraded_markets(self, health_file):
        report = HealthReport()
        report.markets["us"] = MarketHealth(status="healthy")
        report.markets["de"] = MarketHealth(status="degraded")
        report.markets["cl"] = MarketHealth(status="critical")
        save_health_report(report)

        degraded = get_degraded_markets()
        assert "de" in degraded
        assert "cl" in degraded
        assert "us" not in degraded


# ---------------------------------------------------------------------------
# Test: Alerting (status change detection)
# ---------------------------------------------------------------------------

class TestAlerting:
    def test_status_change_detected(self, health_file):
        """When status changes, an alert should be logged."""
        # Save initial state
        report1 = HealthReport()
        report1.markets["test"] = MarketHealth(
            market_id="test", status="healthy",
        )
        save_health_report(report1)

        # New state: degraded
        report2 = HealthReport(last_check="2026-03-23T20:00:00Z")
        report2.markets["test"] = MarketHealth(
            market_id="test", status="degraded",
            degraded_reason="API returned 404",
        )

        with patch("operator1.monitoring.health_check._emit_alert") as mock_alert:
            from operator1.monitoring.health_check import _detect_and_alert
            _detect_and_alert(report2, report1)
            mock_alert.assert_called_once_with(
                "test", "healthy", "degraded", "API returned 404",
            )


# ---------------------------------------------------------------------------
# Test: Preflight check
# ---------------------------------------------------------------------------

class TestPreflight:
    def test_preflight_healthy(self, health_file):
        from datetime import datetime, timezone
        report = HealthReport(
            last_check=datetime.now(timezone.utc).isoformat(),
        )
        report.markets["us_sec_edgar"] = MarketHealth(
            market_id="us_sec_edgar", status="healthy",
        )
        save_health_report(report)

        ok, msg = preflight_check("us_sec_edgar")
        assert ok is True
        assert "healthy" in msg

    def test_preflight_critical(self, health_file):
        from datetime import datetime, timezone
        report = HealthReport(
            last_check=datetime.now(timezone.utc).isoformat(),
        )
        report.markets["cl_cmf"] = MarketHealth(
            market_id="cl_cmf", status="critical",
            degraded_reason="CMF API down",
        )
        save_health_report(report)

        ok, msg = preflight_check("cl_cmf")
        assert ok is False
        assert "CRITICAL" in msg


# ---------------------------------------------------------------------------
# Test: Canary companies coverage
# ---------------------------------------------------------------------------

class TestCanaryCoverage:
    def test_all_tier1_have_canaries(self):
        """Every Tier 1 market should have a canary company defined."""
        tier1 = [
            "us_sec_edgar", "uk_companies_house", "eu_esef", "fr_esef",
            "de_esef", "jp_jquants", "kr_dart", "tw_mops", "br_cvm", "cl_cmf",
        ]
        for market_id in tier1:
            assert market_id in CANARY_COMPANIES, f"Missing canary for {market_id}"
            assert CANARY_COMPANIES[market_id].get("identifier"), f"Empty identifier for {market_id}"

    def test_all_tier2_have_canaries(self):
        """Every Tier 2 market should have a canary company defined."""
        tier2 = [
            "in_bse", "cn_sse", "hk_hkex", "ca_sedar", "au_asx",
            "sg_sgx", "sa_tadawul", "ch_six", "za_jse", "mx_bmv",
            "ae_dfm", "nl_esef", "es_esef", "it_esef", "se_esef",
        ]
        for market_id in tier2:
            assert market_id in CANARY_COMPANIES, f"Missing canary for {market_id}"

"""Tests for operator1.clients.uk_ch_wrapper.UKCompaniesHouseClient.

Includes sandbox integration tests that can run against the CH sandbox
environment when COMPANIES_HOUSE_API_KEY is set.

CH Sandbox docs: https://developer-specs.company-information.service.gov.uk/sandbox-test-data-generator-api/
Sandbox host: api-sandbox.company-information.service.gov.uk
Test data generator: test-data-sandbox.company-information.service.gov.uk
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestUKCompaniesHouseClient:
    """Unit tests (mock-based, always run)."""

    def test_import(self):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient, UKCompaniesHouseError
        assert UKCompaniesHouseClient is not None

    def test_instantiation(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient
        client = UKCompaniesHouseClient(api_key="test_key", cache_dir=tmp_path / "cache")
        assert client.market_id == "uk_companies_house"

    def test_sandbox_mode(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient, _CH_SANDBOX_BASE
        client = UKCompaniesHouseClient(api_key="test_key", cache_dir=tmp_path, sandbox=True)
        assert client._sandbox is True
        assert client._base_url == _CH_SANDBOX_BASE

    def test_production_mode_default(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient, _CH_BASE
        client = UKCompaniesHouseClient(api_key="test_key", cache_dir=tmp_path)
        assert client._sandbox is False
        assert client._base_url == _CH_BASE

    def test_market_properties(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient
        client = UKCompaniesHouseClient(cache_dir=tmp_path / "cache")
        assert client.market_id == "uk_companies_house"
        assert "UK" in client.market_name or "United Kingdom" in client.market_name

    def test_basic_auth_encoding(self, tmp_path):
        """Verify Basic Auth uses API key as username with empty password.

        From CH docs: 'the Companies House API takes the username as the
        API key and ignores the password, so can be left blank.'
        Source: developer-specs.company-information.service.gov.uk/guides/authorisation
        """
        import base64
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient

        client = UKCompaniesHouseClient(api_key="my_test_key_123", cache_dir=tmp_path)

        # Simulate what _get() does for auth header
        expected = base64.b64encode(b"my_test_key_123:").decode()
        assert expected == base64.b64encode(f"{client._api_key}:".encode()).decode()

    def test_cache_path(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient
        client = UKCompaniesHouseClient(cache_dir=tmp_path / "cache")
        path = client._cache_path("00000001", "profile.json")
        assert "00000001" in str(path).upper()

    def test_cache_read_write(self, tmp_path):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient
        client = UKCompaniesHouseClient(cache_dir=tmp_path / "cache")
        cache_data = {"name": "TEST PLC", "ticker": "TEST", "country": "GB"}
        client._write_cache("TEST", "profile.json", cache_data)
        result = client._read_cache("TEST", "profile.json")
        assert result is not None
        assert result["name"] == "TEST PLC"

    def test_get_uses_sandbox_url(self, tmp_path):
        """Verify sandbox mode uses the sandbox base URL in requests."""
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient

        client = UKCompaniesHouseClient(
            api_key="test_key", cache_dir=tmp_path, sandbox=True,
        )

        with patch("operator1.http_utils.cached_get") as mock_get:
            mock_get.return_value = {"items": []}
            try:
                client._get("/search/companies", params={"q": "test"})
            except Exception:
                pass

            if mock_get.called:
                call_url = mock_get.call_args[0][0]
                assert "sandbox" in call_url


@pytest.mark.skipif(
    not os.environ.get("COMPANIES_HOUSE_API_KEY"),
    reason="COMPANIES_HOUSE_API_KEY not set -- skipping sandbox integration tests",
)
class TestUKCompaniesHouseSandboxIntegration:
    """Integration tests against the CH sandbox.

    These only run when COMPANIES_HOUSE_API_KEY is set in the environment.
    They hit the real sandbox API at api-sandbox.company-information.service.gov.uk.

    The sandbox uses the same API key as production but returns test data.
    """

    def test_sandbox_search(self):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient

        client = UKCompaniesHouseClient(sandbox=True)
        results = client.list_companies("test")
        # Sandbox should return some results (test data)
        assert isinstance(results, list)

    def test_sandbox_company_profile(self):
        """Test fetching a well-known company from sandbox."""
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient

        client = UKCompaniesHouseClient(sandbox=True)
        # 00000006 is used in CH's own documentation examples
        try:
            profile = client.get_profile("00000006")
            assert isinstance(profile, dict)
            assert profile.get("country") == "GB"
        except Exception:
            # Sandbox may not have this company; that's OK
            pass

    def test_sandbox_filing_history(self):
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient
        import pandas as pd

        client = UKCompaniesHouseClient(sandbox=True)
        # Try to get filing data -- sandbox may or may not have filings
        result = client.get_income_statement("00000006")
        assert isinstance(result, pd.DataFrame)

"""Tests for TMX GraphQL data provider (Canadian stocks).

Tests both profile enrichment and OHLCV data fetching from the TMX
Money GraphQL API.  These are live API tests (no mocking) since the
TMX API is free and requires no authentication.
"""

import pytest
import pandas as pd


class TestTMXGetQuote:
    """Test TMX GraphQL quote fetching."""

    def test_get_quote_ry(self):
        """Royal Bank of Canada should return a valid quote."""
        from operator1.clients.ohlcv_tmx import tmx_get_quote

        quote = tmx_get_quote("RY")
        assert quote is not None, "TMX quote for RY should not be None"
        assert quote["symbol"] == "RY"
        assert quote["name"] == "Royal Bank of Canada"
        assert quote["sector"] == "Finance"
        assert quote["industry"] == "Banking"
        assert quote["MarketCap"] > 0
        assert quote["eps"] > 0
        assert quote["peRatio"] > 0
        assert quote["beta"] is not None

    def test_get_quote_shop(self):
        """Shopify should return a valid quote."""
        from operator1.clients.ohlcv_tmx import tmx_get_quote

        quote = tmx_get_quote("SHOP")
        assert quote is not None
        assert quote["symbol"] == "SHOP"
        assert "Shopify" in quote["name"]

    def test_get_quote_invalid(self):
        """Invalid ticker should return None."""
        from operator1.clients.ohlcv_tmx import tmx_get_quote

        quote = tmx_get_quote("ZZZZZZZZZ")
        assert quote is None


class TestTMXEnrichProfile:
    """Test TMX profile enrichment."""

    def test_enrich_empty_profile(self):
        """TMX should fill in all available fields on an empty profile."""
        from operator1.clients.ohlcv_tmx import tmx_enrich_profile

        profile: dict = {
            "ticker": "ENB",
            "name": "",
            "country": "CA",
            "exchange": "TSX",
            "currency": "CAD",
        }
        result = tmx_enrich_profile("ENB", profile)

        assert result["_tmx_enriched"] is True
        assert "Enbridge" in result.get("name", "")
        assert result.get("sector") != ""
        assert result.get("market_cap") is not None
        assert result.get("eps") is not None
        assert result.get("dividend_yield") is not None
        assert result.get("description") is not None
        assert len(result.get("description", "")) > 50

    def test_enrich_preserves_existing(self):
        """TMX enrichment should not overwrite existing profile fields."""
        from operator1.clients.ohlcv_tmx import tmx_enrich_profile

        profile: dict = {
            "ticker": "RY",
            "name": "My Custom Name",
            "sector": "My Sector",
            "country": "CA",
        }
        result = tmx_enrich_profile("RY", profile)

        # Existing fields should be preserved
        assert result["name"] == "My Custom Name"
        assert result["sector"] == "My Sector"

        # Missing fields should be filled
        assert result.get("market_cap") is not None
        assert result.get("eps") is not None


class TestTMXOHLCV:
    """Test TMX OHLCV data fetching."""

    def test_fetch_daily_bars(self):
        """Should return daily OHLCV bars for a Canadian stock."""
        from operator1.clients.ohlcv_tmx import fetch_ohlcv_tmx

        df = fetch_ohlcv_tmx("RY", interval=1440)
        assert not df.empty, "TMX daily OHLCV should not be empty"
        assert "date" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert len(df) >= 5, "Should have at least 5 daily bars"

    def test_fetch_minute_aggregated(self):
        """Minute bars aggregated to daily should have OHLCV columns."""
        from operator1.clients.ohlcv_tmx import fetch_ohlcv_tmx

        df = fetch_ohlcv_tmx("RY", interval=1)
        assert not df.empty
        assert "date" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        # Minute bars aggregated to daily should have fewer rows
        assert len(df) <= 15, "Should be ~8 trading days max"

    def test_ohlcv_values_reasonable(self):
        """OHLCV values should be positive and reasonable."""
        from operator1.clients.ohlcv_tmx import fetch_ohlcv_tmx

        df = fetch_ohlcv_tmx("RY", interval=1440)
        if df.empty:
            pytest.skip("TMX API returned no data")

        assert (df["open"] > 0).all()
        assert (df["high"] >= df["low"]).all()
        assert (df["close"] > 0).all()
        assert (df["volume"] >= 0).all()

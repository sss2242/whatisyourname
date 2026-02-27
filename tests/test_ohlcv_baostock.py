"""Tests for the baostock China OHLCV wrapper."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class TestBaostockCodeConversion:
    """Unit tests for _to_baostock_code helper."""

    def test_plain_shanghai(self):
        from operator1.clients.ohlcv_baostock import _to_baostock_code
        assert _to_baostock_code("600519") == "sh.600519"

    def test_plain_shenzhen_0(self):
        from operator1.clients.ohlcv_baostock import _to_baostock_code
        assert _to_baostock_code("000001") == "sz.000001"

    def test_plain_shenzhen_3(self):
        from operator1.clients.ohlcv_baostock import _to_baostock_code
        assert _to_baostock_code("300750") == "sz.300750"

    def test_yfinance_ss_suffix(self):
        from operator1.clients.ohlcv_baostock import _to_baostock_code
        assert _to_baostock_code("600519.SS") == "sh.600519"

    def test_yfinance_sz_suffix(self):
        from operator1.clients.ohlcv_baostock import _to_baostock_code
        assert _to_baostock_code("000001.SZ") == "sz.000001"

    def test_already_baostock_format(self):
        from operator1.clients.ohlcv_baostock import _to_baostock_code
        assert _to_baostock_code("sh.600519") == "sh.600519"
        assert _to_baostock_code("sz.000001") == "sz.000001"


class TestBaostockFetch:
    """Live tests -- these hit the baostock server."""

    def test_fetch_moutai(self):
        """Fetch Kweichow Moutai (Shanghai 600519)."""
        from operator1.clients.ohlcv_baostock import fetch_ohlcv_baostock
        df = fetch_ohlcv_baostock("600519", years=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 50, f"Expected >= 50 rows, got {len(df)}"
        assert "date" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        # Bonus columns from baostock
        assert "turn" in df.columns, "Missing turnover rate column"
        assert "pct_chg" in df.columns, "Missing percent change column"
        assert "amount" in df.columns, "Missing amount column"
        logger.info("Moutai: %d rows, close range: %.1f - %.1f",
                     len(df), df["close"].min(), df["close"].max())

    def test_fetch_ping_an(self):
        """Fetch Ping An Bank (Shenzhen 000001)."""
        from operator1.clients.ohlcv_baostock import fetch_ohlcv_baostock
        df = fetch_ohlcv_baostock("000001", years=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 50, f"Expected >= 50 rows, got {len(df)}"
        logger.info("Ping An: %d rows", len(df))

    def test_fetch_with_yfinance_suffix(self):
        """Accepts yfinance-style ticker (600519.SS)."""
        from operator1.clients.ohlcv_baostock import fetch_ohlcv_baostock
        df = fetch_ohlcv_baostock("600519.SS", years=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 50

    def test_invalid_ticker_returns_empty(self):
        """Invalid ticker returns empty DataFrame, doesn't crash."""
        from operator1.clients.ohlcv_baostock import fetch_ohlcv_baostock
        df = fetch_ohlcv_baostock("INVALID_TICKER_9999", years=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

"""Tests for operator1.clients.ohlcv_twstock wrapper (Taiwan TWSE)."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestOHLCVTwstock:
    """Tests for operator1.clients.ohlcv_twstock."""

    def test_import(self):
        from operator1.clients.ohlcv_twstock import fetch_ohlcv_twstock
        assert callable(fetch_ohlcv_twstock)

    def test_fetch_with_mock(self):
        from operator1.clients.ohlcv_twstock import fetch_ohlcv_twstock
        from collections import namedtuple

        mock_stock_instance = MagicMock()
        mock_stock_instance.date = [date(2023, 1, 2), date(2023, 1, 3)]
        mock_stock_instance.open = [500, 510]
        mock_stock_instance.high = [520, 530]
        mock_stock_instance.low = [490, 500]
        mock_stock_instance.close = [510, 520]
        mock_stock_instance.capacity = [10000, 20000]

        mock_twstock = MagicMock()
        mock_twstock.Stock.return_value = mock_stock_instance

        # The wrapper also does `import twstock.stock as _ts_mod` and reads
        # _ts_mod.DATATUPLE._fields, so we need to mock the submodule too.
        mock_stock_mod = MagicMock()
        mock_stock_mod.DATATUPLE = namedtuple("Data", [
            "date", "capacity", "turnover", "open", "high", "low",
            "close", "change", "transaction",
        ])
        mock_twstock.stock = mock_stock_mod

        with patch.dict("sys.modules", {
            "twstock": mock_twstock,
            "twstock.stock": mock_stock_mod,
        }):
            result = fetch_ohlcv_twstock("2330", years=1)

        assert not result.empty
        assert "close" in result.columns
        assert len(result) == 2

    def test_fetch_empty_on_no_data(self):
        from operator1.clients.ohlcv_twstock import fetch_ohlcv_twstock

        mock_stock_instance = MagicMock()
        mock_stock_instance.date = []

        mock_twstock = MagicMock()
        mock_twstock.Stock.return_value = mock_stock_instance

        with patch.dict("sys.modules", {"twstock": mock_twstock}):
            result = fetch_ohlcv_twstock("0000")
        assert result.empty

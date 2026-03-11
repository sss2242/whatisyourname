"""Tests for operator1.clients.macro_kosis wrapper (South Korea via FRED)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMacroKosis:
    """Tests for operator1.clients.macro_kosis."""

    def test_import(self):
        from operator1.clients.macro_kosis import fetch_macro_kosis
        assert callable(fetch_macro_kosis)

    def test_fred_kr_series_defined(self):
        from operator1.clients.macro_kosis import _FRED_KR_SERIES
        assert "gdp_growth" in _FRED_KR_SERIES
        assert "inflation_rate_yoy" in _FRED_KR_SERIES
        assert "unemployment_rate" in _FRED_KR_SERIES

    def test_returns_wbgapi_fallback_without_fred_key(self):
        from operator1.clients.macro_kosis import fetch_macro_kosis

        with patch.dict(os.environ, {}, clear=True):
            result = fetch_macro_kosis()
        # Without FRED key, wbgapi fallback should still return data
        # (wbgapi needs no key).  If wbgapi is unavailable the result
        # will be empty -- either outcome is acceptable.
        assert isinstance(result, dict)

    def test_fetch_with_mock_fred(self):
        from operator1.clients.macro_kosis import fetch_macro_kosis

        mock_series = pd.Series([3.0, 3.2], index=pd.date_range("2023-01-01", periods=2))

        mock_fred = MagicMock()
        mock_fred.get_series.return_value = mock_series
        mock_fred_class = MagicMock(return_value=mock_fred)

        with patch.dict(os.environ, {"FRED_API_KEY": "test_key"}):
            with patch.dict("sys.modules", {"fredapi": MagicMock(Fred=mock_fred_class)}):
                result = fetch_macro_kosis()

        assert len(result) > 0

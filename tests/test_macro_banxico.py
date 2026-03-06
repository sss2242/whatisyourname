"""Tests for operator1.clients.macro_banxico wrapper (Mexico)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMacroBanxico:
    """Tests for operator1.clients.macro_banxico."""

    def test_import(self):
        from operator1.clients.macro_banxico import fetch_macro_banxico
        assert callable(fetch_macro_banxico)

    def test_returns_empty_without_token(self):
        from operator1.clients.macro_banxico import fetch_macro_banxico

        with patch.dict(os.environ, {}, clear=True):
            result = fetch_macro_banxico(api_token="")
        assert result == {}

    def test_series_ids_defined(self):
        from operator1.clients.macro_banxico import _BANXICO_SERIES
        assert "interest_rate" in _BANXICO_SERIES
        assert "inflation_rate_yoy" in _BANXICO_SERIES
        assert "exchange_rate" in _BANXICO_SERIES

    def test_fetch_with_mock(self):
        from operator1.clients.macro_banxico import fetch_macro_banxico

        # Mock the banxicoapi.BanxicoApi.get() response format:
        # Returns a list of dicts with "idSerie" and "datos" keys.
        mock_api_response = [
            {
                "idSerie": "SF61745",
                "datos": [
                    {"fecha": "01/01/2023", "dato": "11.25"},
                    {"fecha": "01/02/2023", "dato": "11.25"},
                    {"fecha": "01/03/2023", "dato": "11.00"},
                ],
            }
        ]

        mock_api = MagicMock()
        mock_api.get.return_value = mock_api_response

        mock_banxicoapi = MagicMock()
        mock_banxicoapi.BanxicoApi.return_value = mock_api

        with patch.dict("sys.modules", {"banxicoapi": mock_banxicoapi}):
            result = fetch_macro_banxico(api_token="test_token")

        assert len(result) > 0

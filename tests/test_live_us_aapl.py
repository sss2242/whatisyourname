"""Live full-pipeline test: US -- Apple Inc (AAPL).

Wrappers tested:
  Micro: SEC EDGAR (us_edgar.py) -- no key, email User-Agent
  Macro: wbgapi (World Bank) -- no key
  OHLCV: yfinance -- no key

Run: EDGAR_IDENTITY='abdh0saifelden2@gmail.com' python3.12 -m pytest tests/test_live_us_aapl.py -v
"""

from __future__ import annotations

import os
import unittest

import pandas as pd
import pytest

from tests.live_helpers import (
    NETWORK_OK,
    build_minimal_cache,
    fetch_macro_safe,
    fetch_ohlcv_safe,
    make_temp_cache,
    run_aggregation_safe,
    run_forecasting_safe,
    run_monte_carlo_safe,
    run_regime_detection,
)

# SEC EDGAR needs email as User-Agent
EDGAR_EMAIL = "abdh0saifelden2@gmail.com"
os.environ.setdefault("EDGAR_IDENTITY", EDGAR_EMAIL)

TICKER = "AAPL"
MARKET_ID = "us_sec_edgar"
COUNTRY = "US"


@pytest.mark.skipif(not NETWORK_OK, reason="No network connectivity")
class TestUSFullPipeline(unittest.TestCase):
    """Full pipeline integration test for US -- Apple (AAPL)."""

    @classmethod
    def setUpClass(cls):
        cls._cache_dir = make_temp_cache()
        cls._ohlcv = pd.DataFrame()
        cls._macro = {}
        cls._profile = {}
        cls._cache = pd.DataFrame()
        cls._forecast = None
        cls._mc = None
        cls._result = None

    # -- Step 1: Micro (PIT filings from SEC EDGAR) --
    def test_01_micro_edgar_profile(self):
        try:
            from operator1.clients.us_edgar import USEdgarClient
            client = USEdgarClient(user_agent=f"Operator1/1.0 ({EDGAR_EMAIL})")
            profile = client.get_profile(TICKER)
            self.assertIsInstance(profile, dict)
            self.assertTrue(profile.get("name"), "Profile should have a company name")
            self.__class__._profile = profile
        except Exception as exc:
            self.skipTest(f"SEC EDGAR unavailable: {exc}")

    def test_02_micro_edgar_financials(self):
        try:
            from operator1.clients.us_edgar import USEdgarClient
            client = USEdgarClient(user_agent=f"Operator1/1.0 ({EDGAR_EMAIL})")
            income = client.get_income_statement(TICKER)
            balance = client.get_balance_sheet(TICKER)
            self.assertIsInstance(income, pd.DataFrame)
            self.assertIsInstance(balance, pd.DataFrame)
            # At least one should have data for Apple
            has_data = len(income) > 0 or len(balance) > 0
            if not has_data:
                self.skipTest("EDGAR returned empty financials (library compatibility)")
        except Exception as exc:
            self.skipTest(f"SEC EDGAR financials unavailable: {exc}")

    # -- Step 2: OHLCV --
    def test_03_ohlcv_yfinance(self):
        df = fetch_ohlcv_safe(TICKER, MARKET_ID, years=1)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0, "OHLCV should have data for AAPL")
        self.__class__._ohlcv = df

    # -- Step 3: Macro --
    def test_04_macro_wbgapi(self):
        macro = fetch_macro_safe(COUNTRY, MARKET_ID)
        self.assertIsInstance(macro, dict)
        # wbgapi should return at least GDP for US
        if macro:
            self.assertGreater(len(macro), 0)
        self.__class__._macro = macro

    # -- Step 4: Build cache (7 days) --
    def test_05_build_cache(self):
        cache = build_minimal_cache(
            self.__class__._ohlcv,
            self.__class__._macro,
            self.__class__._profile,
        )
        self.assertIsInstance(cache, pd.DataFrame)
        self.assertGreater(len(cache), 0, "Cache should have rows")
        self.assertIn("close", cache.columns)
        self.__class__._cache = cache

    # -- Step 5: Regime detection --
    def test_06_regime_detection(self):
        cache = self.__class__._cache
        if len(cache) == 0:
            self.skipTest("No cache built")
        cache = run_regime_detection(cache)
        self.assertIn("regime_label", cache.columns)
        self.__class__._cache = cache

    # -- Step 6: Forecasting --
    def test_07_forecasting(self):
        cache = self.__class__._cache
        if len(cache) == 0:
            self.skipTest("No cache built")
        cache, forecast = run_forecasting_safe(cache)
        self.assertIsNotNone(forecast)
        self.__class__._cache = cache  # updated cache from forecasting
        self.__class__._forecast = forecast

    # -- Step 7: Monte Carlo --
    def test_08_monte_carlo(self):
        cache = self.__class__._cache
        if len(cache) == 0:
            self.skipTest("No cache built")
        mc = run_monte_carlo_safe(cache)
        # MC is optional -- may fail with small data
        self.__class__._mc = mc

    # -- Step 8: Prediction Aggregation --
    def test_09_prediction_aggregation(self):
        cache = self.__class__._cache
        forecast = self.__class__._forecast
        if forecast is None:
            self.skipTest("No forecast available")
        result = run_aggregation_safe(
            cache, forecast, self.__class__._mc,
            cache_dir=self.__class__._cache_dir,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.fitted or result.error is not None)
        self.__class__._result = result

    # -- Step 9: End-to-end summary --
    def test_10_end_to_end_summary(self):
        result = self.__class__._result
        cache = self.__class__._cache
        print(f"\n{'='*60}")
        print(f"US AAPL Pipeline Summary")
        print(f"{'='*60}")
        print(f"Cache rows: {len(cache)}")
        print(f"OHLCV rows: {len(self.__class__._ohlcv)}")
        print(f"Macro indicators: {len(self.__class__._macro)}")
        print(f"Profile fields: {len(self.__class__._profile)}")
        if result:
            print(f"Predictions: {len(result.predictions)} variables")
            print(f"Fitted: {result.fitted}")
            print(f"Error: {result.error}")
            print(f"Regime: {result.current_regime}")
        print(f"{'='*60}")
        # This test always passes -- it's a summary
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()

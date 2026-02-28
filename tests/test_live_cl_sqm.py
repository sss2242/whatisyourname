"""Live full-pipeline test: Chile -- SQM (SQM).

Wrappers tested:
  Micro: cl_cmf_wrapper (CMF) -- no key
  Macro: wbgapi (World Bank) -- no key
  OHLCV: yfinance (SQM) -- no key

Run: python3.12 -m pytest tests/test_live_cl_sqm.py -v
"""

from __future__ import annotations

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

TICKER = "SQM"
YFINANCE_TICKER = "SQM"
MARKET_ID = "cl_cmf"
COUNTRY = "CL"


@pytest.mark.skipif(not NETWORK_OK, reason="No network connectivity")
class TestCLFullPipeline(unittest.TestCase):
    """Full pipeline integration test for Chile -- SQM."""

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

    def test_01_micro_cmf_profile(self):
        try:
            from operator1.clients.cl_cmf_wrapper import CLCmfClient
            client = CLCmfClient()
            profile = client.get_profile(TICKER)
            self.assertIsInstance(profile, dict)
            self.__class__._profile = profile
        except Exception as exc:
            self.skipTest(f"CMF unavailable: {exc}")

    def test_02_ohlcv_yfinance(self):
        df = fetch_ohlcv_safe(YFINANCE_TICKER, MARKET_ID, years=1)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0, "OHLCV should have data for SQM")
        self.__class__._ohlcv = df

    def test_03_macro_wbgapi(self):
        macro = fetch_macro_safe(COUNTRY, MARKET_ID)
        self.assertIsInstance(macro, dict)
        self.__class__._macro = macro

    def test_04_build_cache(self):
        cache = build_minimal_cache(
            self.__class__._ohlcv, self.__class__._macro, self.__class__._profile,
        )
        self.assertGreater(len(cache), 0)
        self.__class__._cache = cache

    def test_05_regime_detection(self):
        if len(self.__class__._cache) == 0:
            self.skipTest("No cache")
        self.__class__._cache = run_regime_detection(self.__class__._cache)

    def test_06_forecasting(self):
        if len(self.__class__._cache) == 0:
            self.skipTest("No cache")
        self.__class__._forecast = run_forecasting_safe(self.__class__._cache)

    def test_07_monte_carlo(self):
        if len(self.__class__._cache) == 0:
            self.skipTest("No cache")
        self.__class__._mc = run_monte_carlo_safe(self.__class__._cache)

    def test_08_prediction_aggregation(self):
        if self.__class__._forecast is None:
            self.skipTest("No forecast")
        self.__class__._result = run_aggregation_safe(
            self.__class__._cache, self.__class__._forecast, self.__class__._mc,
            cache_dir=self.__class__._cache_dir,
        )
        self.assertIsNotNone(self.__class__._result)

    def test_09_summary(self):
        r = self.__class__._result
        print(f"\n{'='*60}\nCL SQM Pipeline Summary\n{'='*60}")
        print(f"Cache: {len(self.__class__._cache)} rows, OHLCV: {len(self.__class__._ohlcv)}")
        print(f"Macro: {len(self.__class__._macro)} indicators")
        if r:
            print(f"Predictions: {len(r.predictions)}, Fitted: {r.fitted}, Error: {r.error}")
        print(f"{'='*60}")


if __name__ == "__main__":
    unittest.main()

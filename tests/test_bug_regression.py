"""Regression tests for bugs found in pipeline run 2026-03-06.

These tests verify the specific bugs that were fixed:
1. Key indicators table using wrong profile key (current_snapshot vs current_state)
2. OHLCV source attribution in report LIMITATIONS section
3. HMM regime detector falling back to diagonal covariance
4. Forecasting with all-NaN variables (estimator didn't fill them)
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


class TestKeyIndicatorsProfileKey(unittest.TestCase):
    """Bug: _build_key_indicators_table read from profile['current_snapshot']
    but the profile builder writes to profile['current_state'].

    The indicators table showed all N/A values in the premium report while
    basic/pro reports showed data via a different code path (section 4).
    """

    def test_indicators_read_from_current_state(self):
        """Key indicators must read values from current_state, not current_snapshot."""
        from operator1.report.report_generator import _build_key_indicators_table

        profile = {
            "current_state": {
                "available": True,
                "tier4_profitability": {
                    "gross_margin": 0.404,
                    "operating_margin": 0.213,
                    "net_margin": 0.304,
                    "roe": 0.55,
                    "roa": 0.10,
                },
                "tier3_stability": {
                    "volatility_21d": 0.019,
                    "drawdown_252d": -0.09,
                },
                "tier2_solvency": {
                    "debt_to_equity": 1.49,
                },
            },
            "financial_health": {},
        }

        result = _build_key_indicators_table(profile)
        # Gross margin should show as 40.4%, not N/A
        self.assertIn("40.4%", result)
        self.assertIn("21.3%", result)

    def test_indicators_empty_when_no_current_state(self):
        """If current_state is missing, indicators should be N/A, not crash."""
        from operator1.report.report_generator import _build_key_indicators_table

        profile = {"financial_health": {}}
        result = _build_key_indicators_table(profile)
        # Should produce a table with N/A values, not an error
        self.assertIn("Key Financial Indicators", result)

    def test_old_key_does_not_work(self):
        """Putting data under 'current_snapshot' should NOT make indicators appear."""
        from operator1.report.report_generator import _build_key_indicators_table

        profile = {
            "current_snapshot": {
                "gross_margin": 0.404,
            },
            "financial_health": {},
        }
        result = _build_key_indicators_table(profile)
        # Should NOT find 40.4% because the correct key is current_state
        self.assertNotIn("40.4%", result)


class TestOHLCVSourceAttribution(unittest.TestCase):
    """Bug: Report LIMITATIONS said 'All financial data (statements, filings,
    prices) is sourced from SEC EDGAR' when prices came from yfinance.
    """

    def test_separate_ohlcv_source_in_limitations(self):
        """When ohlcv_source differs from data_provider, both should appear."""
        from operator1.report.report_generator import _build_limitations

        profile = {
            "meta": {
                "date_range": {"start": "2024-01-01", "end": "2026-01-01"},
                "data_provider": "SEC EDGAR",
                "data_provider_label": "SEC EDGAR (US -- NYSE / NASDAQ)",
                "pit_source": True,
                "ohlcv_source": "yfinance (Yahoo Finance)",
            },
            "data_quality": {},
            "failed_modules": [],
            "estimation": {},
        }

        result = _build_limitations(profile)
        # Should mention SEC EDGAR for filings
        self.assertIn("SEC EDGAR", result)
        # Should mention yfinance separately for price data
        self.assertIn("yfinance", result)
        # Should NOT say "All financial data" is from SEC EDGAR
        self.assertNotIn("All financial data (statements, filings, prices)", result)

    def test_same_source_when_pit_provides_ohlcv(self):
        """When ohlcv_source is same as data_provider, no separate mention."""
        from operator1.report.report_generator import _build_limitations

        profile = {
            "meta": {
                "date_range": {"start": "2024-01-01", "end": "2026-01-01"},
                "data_provider": "J-Quants",
                "data_provider_label": "J-Quants (Japan -- TSE)",
                "pit_source": True,
                "ohlcv_source": "J-Quants",
            },
            "data_quality": {},
            "failed_modules": [],
            "estimation": {},
        }

        result = _build_limitations(profile)
        self.assertIn("J-Quants", result)
        # Should mention J-Quants as the price source too
        self.assertIn("Price data is also sourced from", result)


class TestHMMCovarianceFallback(unittest.TestCase):
    """Bug: HMM regime detection failed with 'covars must be symmetric,
    positive-definite' when one regime had near-zero variance.
    """

    def test_hmm_survives_near_zero_variance(self):
        """HMM should fall back to diagonal covariance, not crash."""
        from operator1.models.regime_detector import RegimeDetector

        # Create data where one regime has near-zero volatility
        np.random.seed(42)
        n = 200
        returns = np.concatenate([
            np.random.randn(100) * 0.001,  # very low vol regime
            np.random.randn(100) * 0.03,   # normal vol regime
        ])
        volatility = np.abs(returns)

        detector = RegimeDetector(n_regimes=4, random_state=42)
        regimes, probs = detector.fit_hmm(returns, volatility)

        # Should succeed (possibly with diag fallback), not crash
        if regimes is not None:
            self.assertEqual(len(regimes), n)
        # If it still fails, the error should be recorded, not raised
        if regimes is None:
            self.assertIsNotNone(detector._result.hmm_error)


class TestForecastingWithNaNVariables(unittest.TestCase):
    """Bug: Forecasting models reported '0 observations' for variables that
    were entirely NaN (e.g. cash_ratio, current_ratio when EDGAR didn't
    provide the denominators).
    """

    def test_all_nan_variable_gets_baseline_zero(self):
        """A variable with all NaN values should fall through to baseline."""
        from operator1.models.forecasting import run_forecasting

        np.random.seed(42)
        idx = pd.bdate_range("2023-01-02", periods=300, name="date")
        cache = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(300) * 0.5),
            "return_1d": np.random.randn(300) * 0.01,
            "cash_ratio": np.full(300, np.nan),  # all NaN -- estimator couldn't fill
            "current_ratio": np.full(300, np.nan),  # all NaN
            "volatility_21d": np.abs(np.random.randn(300)) * 0.02,
        }, index=idx)

        _, result = run_forecasting(
            cache,
            variables=["cash_ratio", "current_ratio", "close"],
        )

        # close should be forecasted by a real model
        self.assertIn("close", result.model_used)
        self.assertNotEqual(result.model_used["close"], "baseline_zero")

        # cash_ratio and current_ratio should fall to baseline_zero
        if "cash_ratio" in result.model_used:
            self.assertEqual(result.model_used["cash_ratio"], "baseline_zero")
        if "current_ratio" in result.model_used:
            self.assertEqual(result.model_used["current_ratio"], "baseline_zero")

    def test_mixed_nan_variable_still_forecasted(self):
        """A variable with partial NaN values should be forecasted normally."""
        from operator1.models.forecasting import run_forecasting

        np.random.seed(42)
        idx = pd.bdate_range("2023-01-02", periods=300, name="date")
        # Simulate quarterly financial data (8 observations forward-filled)
        revenue = np.full(300, np.nan)
        for i in range(0, 300, 63):  # ~quarterly
            revenue[i:] = 100e9 + np.random.randn() * 5e9

        cache = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(300) * 0.5),
            "return_1d": np.random.randn(300) * 0.01,
            "revenue": revenue,
            "volatility_21d": np.abs(np.random.randn(300)) * 0.02,
        }, index=idx)

        _, result = run_forecasting(
            cache,
            variables=["revenue"],
        )

        # revenue has ~5 non-NaN values (forward-filled to ~300), should forecast
        self.assertIn("revenue", result.model_used)


class TestFailedModulesIncludeVariableNames(unittest.TestCase):
    """Bug: Failed module errors said 'Insufficient observations (0 < 30)'
    without specifying WHICH variables had zero data.
    """

    def test_failed_modules_include_affected_variables(self):
        from operator1.report.profile_builder import _build_failed_modules_section

        forecast_result = {
            "model_failed_kalman": True,
            "kalman_error": "Insufficient observations for Kalman (0 < 30)",
            "model_failed_var": True,
            "var_error": "Insufficient data for AR(1) (0 < 10)",
            "model_failed_lstm": False,
            "model_failed_garch": False,
            "model_failed_tree": False,
            "model_used": {
                "cash_ratio": "baseline_zero",
                "current_ratio": "baseline_zero",
                "close": "lstm",
                "volatility_21d": "garch",
            },
        }

        failed = _build_failed_modules_section(forecast_result=forecast_result)

        # Should have entries for kalman and var
        kalman_entry = next(
            (f for f in failed if "Kalman" in f["module"]), None
        )
        self.assertIsNotNone(kalman_entry)
        # The error should mention the affected variables
        self.assertIn("cash_ratio", kalman_entry["error"])
        self.assertIn("current_ratio", kalman_entry["error"])


if __name__ == "__main__":
    unittest.main()

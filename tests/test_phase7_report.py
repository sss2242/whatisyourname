"""Phase 7 tests -- Report assembly (T7.1, T7.2).

Tests the profile JSON builder and report generator:
  - Profile builder aggregates all pipeline sections
  - Profile JSON is valid and serialisable
  - Fallback report template produces all required sections
  - LIMITATIONS section is always present
  - Charts generate or skip gracefully
  - Missing/partial inputs handled with "available: False"
  - Gemini integration (mocked)
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_index(days: int = 250) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2023-01-02", periods=days, name="date")


def _make_cache(days: int = 250, *, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic daily cache with survival flags and regime labels."""
    np.random.seed(seed)
    idx = _make_daily_index(days)
    mid = days // 2

    returns = np.concatenate([
        np.random.randn(mid) * 0.005 + 0.001,
        np.random.randn(days - mid) * 0.03 - 0.002,
    ])
    close = 100 + np.cumsum(returns * 100)

    regime_labels = np.concatenate([
        np.full(mid, "bull"),
        np.full(days - mid, "bear"),
    ])

    return pd.DataFrame({
        "close": close,
        "open": close + np.random.randn(days) * 0.5,
        "high": close + np.abs(np.random.randn(days)) * 1.0,
        "low": close - np.abs(np.random.randn(days)) * 1.0,
        "return_1d": returns,
        "volatility_21d": np.abs(np.random.randn(days)) * 0.15,
        "regime_label": regime_labels,
        "structural_break": np.where(np.random.rand(days) > 0.95, 1, 0),
        "company_survival_mode_flag": np.where(
            np.arange(days) > mid, 1, 0,
        ),
        "country_survival_mode_flag": np.zeros(days, dtype=int),
        "country_protected_flag": np.zeros(days, dtype=int),
        "survival_regime": np.where(
            np.arange(days) > mid, "company_survival", "normal",
        ),
        "hierarchy_tier1_weight": np.full(days, 0.35),
        "hierarchy_tier2_weight": np.full(days, 0.25),
        "hierarchy_tier3_weight": np.full(days, 0.20),
        "hierarchy_tier4_weight": np.full(days, 0.12),
        "hierarchy_tier5_weight": np.full(days, 0.08),
        "vanity_percentage": np.clip(np.random.randn(days) * 5 + 3, 0, 100),
        "current_ratio": 1.5 + np.random.randn(days) * 0.2,
        "debt_to_equity_abs": 1.0 + np.random.randn(days) * 0.3,
    }, index=idx)


def _make_verified_target() -> dict:
    return {
        "isin": "US0378331005",
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "country": "US",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "sub_industry": "Smartphones",
        "fmp_symbol": "AAPL",
        "currency": "USD",
        "exchange": "NASDAQ",
    }


def _make_forecast_result() -> dict:
    return {
        "forecasts": {
            "close": {"1d": 155.0, "5d": 156.0, "21d": 160.0, "252d": 170.0},
            "return_1d": {"1d": 0.001, "5d": 0.005, "21d": 0.02, "252d": 0.08},
        },
        "model_failed_kalman": False,
        "model_failed_garch": True,
        "model_failed_var": False,
        "model_failed_lstm": True,
        "model_failed_tree": False,
        "kalman_error": None,
        "garch_error": "arch not installed",
        "var_error": None,
        "lstm_error": "torch not installed",
        "tree_error": None,
        "metrics": [
            {
                "model_name": "kalman",
                "variable": "close",
                "mae": 1.5,
                "rmse": 2.1,
                "n_train": 200,
                "n_test": 50,
                "fitted": True,
                "error": None,
            },
            {
                "model_name": "baseline",
                "variable": "close",
                "mae": 2.0,
                "rmse": 3.0,
                "n_train": 200,
                "n_test": 50,
                "fitted": True,
                "error": None,
            },
        ],
        "model_used": {"close": "kalman", "return_1d": "baseline"},
    }


def _make_mc_result() -> dict:
    return {
        "n_paths": 10000,
        "n_paths_importance": 2000,
        "survival_probability": {
            "1d": 0.99,
            "5d": 0.97,
            "21d": 0.92,
            "252d": 0.75,
        },
        "survival_probability_mean": 0.85,
        "survival_probability_p5": 0.65,
        "survival_probability_p95": 0.95,
        "current_regime": "bull",
        "importance_sampling_used": True,
        "error": None,
    }


def _make_prediction_result() -> dict:
    return {
        "predictions": {
            "close": {
                "1d": {
                    "variable": "close",
                    "horizon": "1d",
                    "point_forecast": 155.0,
                    "lower_ci": 150.0,
                    "upper_ci": 160.0,
                    "confidence": 0.8,
                },
                "5d": {
                    "variable": "close",
                    "horizon": "5d",
                    "point_forecast": 156.0,
                    "lower_ci": 148.0,
                    "upper_ci": 164.0,
                    "confidence": 0.7,
                },
            },
        },
        "technical_alpha": {
            "next_day_open": None,
            "next_day_high": None,
            "next_day_low": 149.5,
            "next_day_close": None,
            "mask_applied": True,
        },
        "ensemble_weights": {"kalman": 0.6, "baseline": 0.4},
        "n_models_available": 3,
        "n_models_failed": 2,
        "survival_probability_mean": 0.85,
        "error": None,
    }


# ===================================================================
# T7.1 -- Profile JSON builder tests
# ===================================================================


class TestProfileBuilder(unittest.TestCase):
    """Tests for operator1.report.profile_builder."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.cache = _make_cache()
        self.verified = _make_verified_target()
        self.forecast = _make_forecast_result()
        self.mc = _make_mc_result()
        self.prediction = _make_prediction_result()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build(self, **kwargs):
        from operator1.report.profile_builder import build_company_profile

        defaults = {
            "verified_target": self.verified,
            "cache": self.cache,
            "forecast_result": self.forecast,
            "mc_result": self.mc,
            "prediction_result": self.prediction,
        }
        defaults.update(kwargs)
        return build_company_profile(**defaults)

    def test_profile_is_valid_json(self) -> None:
        profile = self._build()
        # build_company_profile returns a dict; verify it's JSON-serialisable
        self.assertIsInstance(profile, dict)
        text = json.dumps(profile, default=str)
        loaded = json.loads(text)
        self.assertIsInstance(loaded, dict)

    def test_all_sections_present(self) -> None:
        profile = self._build()
        # Core sections that must always be present
        required_keys = {
            "meta", "identity", "survival", "vanity",
            "linked_entities", "regimes", "predictions",
            "monte_carlo", "model_metrics", "data_quality",
            "estimation", "failed_modules", "filters",
            "graph_risk", "game_theory", "fuzzy_protection", "pid_controller",
            "financial_health", "sentiment", "peer_ranking", "macro_quadrant",
        }
        # Profile may contain additional keys from extended models,
        # canonical translation, or pipeline enrichment -- that's fine.
        missing = required_keys - set(profile.keys())
        self.assertEqual(missing, set(), f"Missing required profile sections: {missing}")

    def test_identity_section(self) -> None:
        profile = self._build()
        identity = profile["identity"]
        self.assertTrue(identity["available"])
        self.assertEqual(identity["isin"], "US0378331005")
        self.assertEqual(identity["ticker"], "AAPL")
        self.assertEqual(identity["country"], "US")

    def test_survival_section(self) -> None:
        profile = self._build()
        survival = profile["survival"]
        self.assertTrue(survival["available"])
        self.assertIn("company_survival_mode_flag", survival)
        self.assertIn("survival_regime", survival)
        self.assertIn("hierarchy_weights", survival)

    def test_vanity_section(self) -> None:
        profile = self._build()
        vanity = profile["vanity"]
        self.assertTrue(vanity["available"])
        self.assertIn("mean", vanity)
        self.assertIn("latest", vanity)
        self.assertIsNotNone(vanity["mean"])

    def test_monte_carlo_section(self) -> None:
        profile = self._build()
        mc = profile["monte_carlo"]
        self.assertTrue(mc["available"])
        self.assertAlmostEqual(mc["survival_probability_mean"], 0.85, places=2)
        self.assertEqual(mc["n_paths"], 10000)

    def test_predictions_section(self) -> None:
        profile = self._build()
        preds = profile["predictions"]
        self.assertTrue(preds["available"])
        self.assertIn("horizons", preds)
        self.assertIn("technical_alpha", preds)

    def test_model_metrics_section(self) -> None:
        profile = self._build()
        metrics = profile["model_metrics"]
        self.assertTrue(metrics["available"])
        self.assertIn("model_status", metrics)
        self.assertTrue(metrics["model_status"]["model_failed_garch"])
        self.assertFalse(metrics["model_status"]["model_failed_kalman"])

    def test_failed_modules_listed(self) -> None:
        profile = self._build()
        failed = profile["failed_modules"]
        self.assertIsInstance(failed, list)
        # Should have failures for garch and lstm
        module_names = [f["module"] for f in failed]
        self.assertTrue(
            any("GARCH" in m for m in module_names),
            f"Expected GARCH failure in {module_names}",
        )
        self.assertTrue(
            any("LSTM" in m for m in module_names),
            f"Expected LSTM failure in {module_names}",
        )

    def test_missing_target_produces_unavailable(self) -> None:
        profile = self._build(verified_target=None)
        self.assertFalse(profile["identity"]["available"])

    def test_missing_cache_produces_unavailable(self) -> None:
        profile = self._build(cache=None)
        self.assertFalse(profile["survival"]["available"])
        self.assertFalse(profile["vanity"]["available"])

    def test_missing_mc_produces_unavailable(self) -> None:
        profile = self._build(mc_result=None)
        self.assertFalse(profile["monte_carlo"]["available"])

    def test_empty_cache(self) -> None:
        profile = self._build(cache=pd.DataFrame())
        self.assertFalse(profile["survival"]["available"])

    def test_no_nan_in_json(self) -> None:
        """Verify serialised JSON contains no NaN values."""
        profile = self._build()
        text = json.dumps(profile, default=str)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)

    def test_meta_section(self) -> None:
        profile = self._build()
        meta = profile["meta"]
        self.assertIn("generated_at", meta)
        self.assertIn("date_range", meta)
        self.assertEqual(meta["pipeline_version"], "1.0.0")


# ===================================================================
# T7.2 -- Report generation tests
# ===================================================================


class TestReportGenerator(unittest.TestCase):
    """Tests for operator1.report.report_generator."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.cache = _make_cache()
        self.profile = self._make_profile()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_profile(self) -> dict:
        """Build a profile directly for report tests."""
        return {
            "meta": {
                "generated_at": "2024-01-15T12:00:00Z",
                "pipeline_version": "1.0.0",
                "date_range": {"start": "2022-01-15", "end": "2024-01-15"},
            },
            "identity": {
                "available": True,
                "isin": "US0378331005",
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "country": "US",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "sub_industry": "Smartphones",
                "fmp_symbol": "AAPL",
                "currency": "USD",
                "exchange": "NASDAQ",
            },
            "survival": {
                "available": True,
                "company_survival_mode_flag": 0,
                "country_survival_mode_flag": 0,
                "country_protected_flag": 0,
                "survival_regime": "normal",
                "regime_distribution_pct": {"normal": 0.8, "company_survival": 0.2},
                "hierarchy_weights": {
                    "tier1": 0.35, "tier2": 0.25, "tier3": 0.20,
                    "tier4": 0.12, "tier5": 0.08,
                },
            },
            "vanity": {
                "available": True,
                "mean": 3.5,
                "median": 3.0,
                "min": 0.0,
                "max": 12.0,
                "latest": 4.2,
                "n_observed": 250,
                "n_missing": 0,
            },
            "linked_entities": {"available": False, "groups": {}},
            "regimes": {
                "available": True,
                "current_regime": "bull",
                "regime_distribution_pct": {"bull": 0.5, "bear": 0.5},
                "n_structural_breaks": 3,
                "structural_break_dates": ["2023-06-15"],
            },
            "predictions": {
                "available": True,
                "horizons": {
                    "1d": [{
                        "variable": "close",
                        "point_forecast": 155.0,
                        "lower_ci": 150.0,
                        "upper_ci": 160.0,
                        "confidence": 0.8,
                    }],
                },
                "technical_alpha": {"next_day_low": 149.5, "mask_applied": True},
                "ensemble_weights": {"kalman": 0.6, "baseline": 0.4},
                "n_models_available": 3,
                "n_models_failed": 2,
            },
            "monte_carlo": {
                "available": True,
                "survival_probability_mean": 0.85,
                "survival_probability_p5": 0.65,
                "survival_probability_p95": 0.95,
                "n_paths": 10000,
                "importance_sampling_used": True,
                "current_regime": "bull",
                "survival_by_horizon": {"1d": 0.99, "252d": 0.75},
            },
            "model_metrics": {
                "available": True,
                "model_status": {
                    "model_failed_kalman": False,
                    "model_failed_garch": True,
                },
                "model_best_rmse": {"kalman": 2.1, "baseline": 3.0},
                "model_used": {"close": "kalman"},
            },
            "data_quality": {"available": False},
            "estimation": {"available": False},
            "failed_modules": [
                {
                    "module": "Forecasting (GARCH)",
                    "error": "arch not installed",
                    "mitigation": "Baseline model used.",
                },
            ],
        }

    def test_fallback_report_generated(self) -> None:
        from operator1.report.report_generator import generate_report

        result = generate_report(
            self.profile,
            gemini_client=None,
            cache=self.cache,
            output_dir=self.tmpdir,
            generate_chart_images=False,
            generate_pdf=False,
        )

        self.assertIn("markdown", result)
        self.assertTrue(len(result["markdown"]) > 100)
        self.assertTrue(Path(result["markdown_path"]).exists())

    def test_limitations_section_present(self) -> None:
        from operator1.report.report_generator import generate_report

        result = generate_report(
            self.profile,
            output_dir=self.tmpdir,
            generate_chart_images=False,
        )

        md = result["markdown"].upper()
        self.assertIn("LIMITATIONS", md)

    def test_limitations_covers_required_topics(self) -> None:
        from operator1.report.report_generator import generate_report

        result = generate_report(
            self.profile,
            output_dir=self.tmpdir,
            generate_chart_images=False,
        )

        md = result["markdown"]
        # Data window
        self.assertIn("Data Window", md)
        # OHLCV source
        self.assertIn("OHLCV", md)
        # Macro data
        self.assertIn("Macro", md)
        # Failed modules
        self.assertIn("Failed Modules", md)

    def test_all_report_sections_present(self) -> None:
        from operator1.report.report_generator import generate_report

        result = generate_report(
            self.profile,
            output_dir=self.tmpdir,
            generate_chart_images=False,
        )

        md = result["markdown"]
        for section in [
            "Executive Summary",
            "Company Overview",
            "Financial Health",
            "Survival Mode Analysis",
            "Linked Variables",
            "Temporal Analysis",
            "LIMITATIONS",
        ]:
            self.assertIn(
                section,
                md,
                f"Section '{section}' not found in report",
            )

    def test_report_with_gemini_client(self) -> None:
        from operator1.report.report_generator import generate_report

        mock_gemini = MagicMock()
        mock_gemini.generate_report.return_value = (
            "# Gemini Report\n\n## LIMITATIONS\n\nSome limitations."
        )

        result = generate_report(
            self.profile,
            gemini_client=mock_gemini,
            output_dir=self.tmpdir,
            generate_chart_images=False,
        )

        self.assertIn("Gemini Report", result["markdown"])
        mock_gemini.generate_report.assert_called_once()

    def test_gemini_failure_falls_back(self) -> None:
        from operator1.report.report_generator import generate_report

        mock_gemini = MagicMock()
        mock_gemini.generate_report.side_effect = RuntimeError("API down")

        result = generate_report(
            self.profile,
            gemini_client=mock_gemini,
            output_dir=self.tmpdir,
            generate_chart_images=False,
        )

        # Should still produce a report via fallback
        self.assertTrue(len(result["markdown"]) > 100)
        self.assertIn("LIMITATIONS", result["markdown"].upper())

    def test_gemini_empty_response_falls_back(self) -> None:
        from operator1.report.report_generator import generate_report

        mock_gemini = MagicMock()
        mock_gemini.generate_report.return_value = ""

        result = generate_report(
            self.profile,
            gemini_client=mock_gemini,
            output_dir=self.tmpdir,
            generate_chart_images=False,
        )

        self.assertTrue(len(result["markdown"]) > 100)

    def test_limitations_appended_when_gemini_omits(self) -> None:
        from operator1.report.report_generator import generate_report

        mock_gemini = MagicMock()
        mock_gemini.generate_report.return_value = (
            "# Report\n\nSome analysis without limitations."
        )

        result = generate_report(
            self.profile,
            gemini_client=mock_gemini,
            output_dir=self.tmpdir,
            generate_chart_images=False,
        )

        self.assertIn("LIMITATIONS", result["markdown"].upper())

    def test_chart_generation(self) -> None:
        from operator1.report.report_generator import generate_charts

        chart_dir = Path(self.tmpdir) / "charts"
        paths = generate_charts(self.cache, self.profile, chart_dir)

        # Should generate at least one chart if matplotlib is available
        try:
            import matplotlib
            self.assertTrue(len(paths) > 0, "Expected at least one chart")
            for p in paths:
                self.assertTrue(Path(p).exists(), f"Chart not found: {p}")
        except ImportError:
            self.assertEqual(paths, [])

    def test_chart_generation_no_cache(self) -> None:
        from operator1.report.report_generator import generate_charts

        chart_dir = Path(self.tmpdir) / "charts"
        paths = generate_charts(None, self.profile, chart_dir)
        self.assertEqual(paths, [])

    def test_chart_generation_empty_cache(self) -> None:
        from operator1.report.report_generator import generate_charts

        chart_dir = Path(self.tmpdir) / "charts"
        paths = generate_charts(pd.DataFrame(), self.profile, chart_dir)
        self.assertEqual(paths, [])

    def test_report_with_all_unavailable(self) -> None:
        from operator1.report.report_generator import generate_report

        empty_profile = {
            "meta": {"generated_at": "2024-01-15T12:00:00Z", "date_range": {}},
            "identity": {"available": False},
            "survival": {"available": False},
            "vanity": {"available": False},
            "linked_entities": {"available": False, "groups": {}},
            "regimes": {"available": False},
            "predictions": {"available": False},
            "monte_carlo": {"available": False},
            "model_metrics": {"available": False},
            "data_quality": {"available": False},
            "estimation": {"available": False},
            "failed_modules": [],
        }

        result = generate_report(
            empty_profile,
            output_dir=self.tmpdir,
            generate_chart_images=False,
        )

        self.assertTrue(len(result["markdown"]) > 50)
        self.assertIn("LIMITATIONS", result["markdown"].upper())


# ===================================================================
# Integration tests
# ===================================================================


class TestProfileToReport(unittest.TestCase):
    """End-to-end: profile builder -> report generator."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_pipeline(self) -> None:
        from operator1.report.profile_builder import build_company_profile
        from operator1.report.report_generator import generate_report

        cache = _make_cache()
        profile = build_company_profile(
            verified_target=_make_verified_target(),
            cache=cache,
            forecast_result=_make_forecast_result(),
            mc_result=_make_mc_result(),
            prediction_result=_make_prediction_result(),
        )

        result = generate_report(
            profile,
            cache=cache,
            output_dir=Path(self.tmpdir) / "report",
            generate_chart_images=False,
        )

        # Verify profile is a valid dict
        self.assertIsInstance(profile, dict)

        # Verify report markdown exists
        self.assertTrue(Path(result["markdown_path"]).exists())

        # Verify LIMITATIONS section
        self.assertIn("LIMITATIONS", result["markdown"].upper())

        # Verify company name appears in report
        self.assertIn("Apple", result["markdown"])

    def test_minimal_pipeline(self) -> None:
        """Test with only required inputs (everything else None)."""
        from operator1.report.profile_builder import build_company_profile
        from operator1.report.report_generator import generate_report

        profile = build_company_profile(
            verified_target=_make_verified_target(),
        )

        result = generate_report(
            profile,
            output_dir=Path(self.tmpdir) / "report",
            generate_chart_images=False,
        )

        self.assertTrue(len(result["markdown"]) > 50)
        self.assertIn("LIMITATIONS", result["markdown"].upper())


# ===================================================================
# Helper function tests
# ===================================================================


class TestHelpers(unittest.TestCase):
    """Tests for profile_builder helper functions."""

    def test_safe_float_nan(self) -> None:
        from operator1.report.profile_builder import _safe_float
        self.assertIsNone(_safe_float(float("nan")))
        self.assertIsNone(_safe_float(float("inf")))
        self.assertIsNone(_safe_float(None))
        self.assertEqual(_safe_float(3.14), 3.14)
        self.assertEqual(_safe_float(0), 0.0)

    def test_safe_str(self) -> None:
        from operator1.report.profile_builder import _safe_str
        self.assertIsNone(_safe_str(None))
        self.assertIsNone(_safe_str(float("nan")))
        self.assertEqual(_safe_str("hello"), "hello")
        self.assertEqual(_safe_str(42), "42")

    def test_series_summary(self) -> None:
        from operator1.report.profile_builder import _series_summary

        s = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
        result = _series_summary(s)
        self.assertEqual(result["n_observed"], 4)
        self.assertEqual(result["n_missing"], 1)
        self.assertIsNotNone(result["mean"])
        self.assertAlmostEqual(result["mean"], 2.75, places=2)

    def test_series_summary_empty(self) -> None:
        from operator1.report.profile_builder import _series_summary

        result = _series_summary(pd.Series([], dtype=float))
        self.assertEqual(result["n_observed"], 0)
        self.assertIsNone(result["mean"])

    def test_json_serialisable(self) -> None:
        from operator1.report.profile_builder import _json_serialisable

        data = {
            "int": np.int64(42),
            "float": np.float64(3.14),
            "nan": float("nan"),
            "array": np.array([1, 2, 3]),
            "bool": np.bool_(True),
        }
        result = _json_serialisable(data)
        self.assertEqual(result["int"], 42)
        self.assertAlmostEqual(result["float"], 3.14)
        self.assertIsNone(result["nan"])
        self.assertEqual(result["array"], [1, 2, 3])
        self.assertTrue(result["bool"])

        # Verify it's JSON-serialisable
        text = json.dumps(result)
        self.assertNotIn("NaN", text)


if __name__ == "__main__":
    unittest.main()

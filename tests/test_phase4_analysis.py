"""Phase 4 tests -- data quality, survival mode, vanity, hierarchy weights."""

from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers for building synthetic test data
# ---------------------------------------------------------------------------

def _make_daily_index(days: int = 20) -> pd.DatetimeIndex:
    """Return a short business-day index for testing."""
    return pd.bdate_range(start="2024-01-02", periods=days, name="date")


def _make_feature_df(days: int = 20, **overrides) -> pd.DataFrame:
    """Build a synthetic daily feature table with typical columns."""
    idx = _make_daily_index(days)
    np.random.seed(42)

    defaults = {
        "close": 100 + np.cumsum(np.random.randn(days) * 2),
        "return_1d": np.concatenate([[np.nan], np.random.randn(days - 1) * 0.02]),
        "current_ratio": np.full(days, 1.5),
        "debt_to_equity_abs": np.full(days, 1.0),
        "fcf_yield": np.full(days, 0.05),
        "drawdown_252d": np.full(days, -0.10),
        "revenue": np.full(days, 1_000_000.0),
        "net_income": np.full(days, 200_000.0),
        "gross_profit": np.full(days, 600_000.0),
        "ebit": np.full(days, 300_000.0),
        "ebitda": np.full(days, 350_000.0),
        "total_equity": np.full(days, 3_000_000.0),
        "total_debt_asof": np.full(days, 500_000.0),
        "cash_and_equivalents": np.full(days, 400_000.0),
        "market_cap": np.full(days, 5_000_000.0),
        "free_cash_flow": np.full(days, 200_000.0),
        "free_cash_flow_ttm_asof": np.full(days, 200_000.0),
        "operating_cash_flow": np.full(days, 300_000.0),
        "net_debt": np.full(days, 100_000.0),
        "net_debt_to_ebitda": np.full(days, 0.3),
        "quick_ratio": np.full(days, 1.2),
        "cash_ratio": np.full(days, 0.8),
        "gross_margin": np.full(days, 0.6),
        "operating_margin": np.full(days, 0.3),
        "net_margin": np.full(days, 0.2),
        "roe": np.full(days, 0.067),
        "pe_ratio_calc": np.full(days, 25.0),
        "ev_to_ebitda": np.full(days, 14.0),
        "enterprise_value": np.full(days, 5_100_000.0),
        "volatility_21d": np.full(days, 0.15),
        "log_return_1d": np.concatenate([[np.nan], np.random.randn(days - 1) * 0.02]),
        "earnings_yield_calc": np.full(days, 0.04),
        "ps_ratio_calc": np.full(days, 5.0),
    }

    # Apply overrides
    for k, v in overrides.items():
        defaults[k] = v

    df = pd.DataFrame(defaults, index=idx)

    # Add is_missing flags
    for col in list(df.columns):
        if not col.startswith("is_missing_") and not col.startswith("invalid_math_"):
            df[f"is_missing_{col}"] = df[col].isna().astype(int)

    return df


# ===========================================================================
# T4.4 -- Data quality enforcement tests
# ===========================================================================

class TestLookAheadCheck(unittest.TestCase):
    """Test look-ahead violation detection."""

    def test_clean_data_no_violations(self):
        from operator1.quality.data_quality import check_look_ahead
        df = _make_feature_df(10)
        violations = check_look_ahead(df, "test")
        self.assertEqual(violations, [])

    def test_report_date_violation_detected(self):
        from operator1.quality.data_quality import check_look_ahead
        idx = _make_daily_index(5)
        df = pd.DataFrame({
            "close": [100.0] * 5,
            "_report_date": pd.to_datetime([
                "2024-01-01",  # before index[0] -- OK
                "2024-01-03",  # == index[1] -- OK
                "2024-01-05",  # > index[2] 2024-01-04 -- VIOLATION
                "2024-01-05",  # == index[3] -- OK
                "2024-01-08",  # == index[4] -- OK
            ]),
        }, index=idx)
        violations = check_look_ahead(df, "test_entity")
        self.assertGreater(len(violations), 0)
        self.assertEqual(violations[0]["entity"], "test_entity")


class TestMissingFlagAudit(unittest.TestCase):
    """Test is_missing_* flag consistency audit."""

    def test_correct_flags_pass(self):
        from operator1.quality.data_quality import audit_missing_flags
        df = pd.DataFrame({
            "x": [1.0, np.nan, 3.0],
            "is_missing_x": [0, 1, 0],
        })
        issues = audit_missing_flags(df)
        self.assertEqual(issues, [])

    def test_false_negative_detected(self):
        from operator1.quality.data_quality import audit_missing_flags
        df = pd.DataFrame({
            "x": [1.0, np.nan, 3.0],
            "is_missing_x": [0, 0, 0],  # bug: should be 1 for row 1
        })
        issues = audit_missing_flags(df)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["false_negatives"], 1)

    def test_false_positive_detected(self):
        from operator1.quality.data_quality import audit_missing_flags
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "is_missing_x": [0, 1, 0],  # bug: row 1 is not missing
        })
        issues = audit_missing_flags(df)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["false_positives"], 1)


class TestInvalidMathAudit(unittest.TestCase):
    """Test invalid_math_* flag consistency."""

    def test_correct_flags_pass(self):
        from operator1.quality.data_quality import audit_invalid_math_flags
        df = pd.DataFrame({
            "ratio": [np.nan, 5.0, np.nan],
            "invalid_math_ratio": [1, 0, 1],
        })
        issues = audit_invalid_math_flags(df)
        self.assertEqual(issues, [])

    def test_inconsistency_detected(self):
        from operator1.quality.data_quality import audit_invalid_math_flags
        df = pd.DataFrame({
            "ratio": [3.0, 5.0],  # both have values
            "invalid_math_ratio": [1, 0],  # but first says invalid
        })
        issues = audit_invalid_math_flags(df)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["invalid_math_flagged_but_value_present"], 1)


class TestCoverage(unittest.TestCase):
    """Test coverage computation."""

    def test_full_coverage(self):
        from operator1.quality.data_quality import compute_coverage
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        cov = compute_coverage(df, ["a", "b"])
        self.assertAlmostEqual(cov["a"], 1.0)
        self.assertAlmostEqual(cov["b"], 1.0)

    def test_partial_coverage(self):
        from operator1.quality.data_quality import compute_coverage
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        cov = compute_coverage(df, ["a"])
        self.assertAlmostEqual(cov["a"], 2 / 3)

    def test_missing_column(self):
        from operator1.quality.data_quality import compute_coverage
        df = pd.DataFrame({"a": [1.0]})
        cov = compute_coverage(df, ["a", "b"])
        self.assertAlmostEqual(cov["a"], 1.0)
        self.assertAlmostEqual(cov["b"], 0.0)


class TestRunQualityChecks(unittest.TestCase):
    """Test the full quality check pipeline."""

    def test_clean_data_passes(self):
        from operator1.quality.data_quality import run_quality_checks
        df = _make_feature_df(10)
        report = run_quality_checks(df, "test", fail_on_look_ahead=True)
        self.assertTrue(report.passed)
        self.assertEqual(report.look_ahead_violations, [])
        self.assertGreater(report.overall_coverage_pct, 0)

    def test_report_to_dict(self):
        from operator1.quality.data_quality import run_quality_checks
        df = _make_feature_df(5)
        report = run_quality_checks(df, "test")
        d = report.to_dict()
        self.assertIn("total_rows", d)
        self.assertIn("passed", d)
        self.assertIn("coverage_per_variable", d)


# ===========================================================================
# T4.1 -- Survival mode detection tests
# ===========================================================================

class TestCompanySurvivalFlag(unittest.TestCase):
    """Test company survival mode flag triggers."""

    def test_normal_company_no_flag(self):
        from operator1.analysis.survival_mode import compute_company_survival_flag
        df = _make_feature_df(10)
        flag = compute_company_survival_flag(df)
        self.assertTrue((flag == 0).all())

    def test_low_current_ratio_triggers(self):
        from operator1.analysis.survival_mode import compute_company_survival_flag
        # cash_and_equivalents=0 disables cash adequacy floor (P6) so trigger fires
        df = _make_feature_df(10, current_ratio=np.full(10, 0.8),
                              cash_and_equivalents=np.full(10, 0.0))
        flag = compute_company_survival_flag(df)
        self.assertTrue((flag == 1).all())

    def test_high_debt_equity_triggers(self):
        from operator1.analysis.survival_mode import compute_company_survival_flag
        df = _make_feature_df(10, debt_to_equity_abs=np.full(10, 4.0),
                              cash_and_equivalents=np.full(10, 0.0))
        flag = compute_company_survival_flag(df)
        self.assertTrue((flag == 1).all())

    def test_negative_fcf_yield_triggers(self):
        from operator1.analysis.survival_mode import compute_company_survival_flag
        df = _make_feature_df(10, fcf_yield=np.full(10, -0.05),
                              cash_and_equivalents=np.full(10, 0.0))
        flag = compute_company_survival_flag(df)
        self.assertTrue((flag == 1).all())

    def test_deep_drawdown_triggers(self):
        from operator1.analysis.survival_mode import compute_company_survival_flag
        df = _make_feature_df(10, drawdown_252d=np.full(10, -0.50))
        flag = compute_company_survival_flag(df)
        self.assertTrue((flag == 1).all())

    def test_missing_data_no_trigger(self):
        from operator1.analysis.survival_mode import compute_company_survival_flag
        df = _make_feature_df(5, current_ratio=np.full(5, np.nan))
        flag = compute_company_survival_flag(df)
        # NaN current_ratio should NOT trigger (only non-null below threshold triggers)
        self.assertTrue((flag == 0).all())

    def test_single_trigger_sufficient(self):
        """Only one condition needs to be true."""
        from operator1.analysis.survival_mode import compute_company_survival_flag
        df = _make_feature_df(10)
        # All healthy except drawdown
        df["drawdown_252d"] = -0.50
        flag = compute_company_survival_flag(df)
        self.assertTrue((flag == 1).all())


class TestCountrySurvivalFlag(unittest.TestCase):
    """Test country survival mode flag."""

    def test_no_macro_data_defaults_zero(self):
        from operator1.analysis.survival_mode import compute_country_survival_flag
        df = _make_feature_df(10)
        flag = compute_country_survival_flag(df)
        self.assertTrue((flag == 0).all())

    def test_high_credit_spread_triggers(self):
        from operator1.analysis.survival_mode import compute_country_survival_flag
        df = _make_feature_df(10)
        df["credit_spread"] = 6.0  # above 5% threshold
        flag = compute_country_survival_flag(df)
        self.assertTrue((flag == 1).all())

    def test_config_driven_thresholds(self):
        from operator1.analysis.survival_mode import compute_country_survival_flag
        df = _make_feature_df(5)
        df["credit_spread"] = 4.0  # below default 5%
        config = {
            "country_survival": {
                "credit_spread_pct": 3.0,  # lower threshold
            }
        }
        flag = compute_country_survival_flag(df, config=config)
        self.assertTrue((flag == 1).all())


class TestCountryProtectedFlag(unittest.TestCase):
    """Test country protection flag."""

    def test_strategic_sector_triggers(self):
        from operator1.analysis.survival_mode import compute_country_protected_flag
        df = _make_feature_df(10)
        flag = compute_country_protected_flag(df, target_sector="Energy")
        self.assertTrue((flag == 1).all())

    def test_non_strategic_sector(self):
        from operator1.analysis.survival_mode import compute_country_protected_flag
        df = _make_feature_df(10)
        flag = compute_country_protected_flag(df, target_sector="Retail")
        # No GDP data in default feature df, so only sector check matters
        self.assertTrue((flag == 0).all())

    def test_market_cap_gdp_triggers(self):
        from operator1.analysis.survival_mode import compute_country_protected_flag
        df = _make_feature_df(10)
        df["market_cap"] = 1_000_000_000.0  # 1B
        df["gdp_current_usd"] = 100_000_000_000.0  # 100B
        # 1B / 100B = 0.01 > 0.001 threshold
        flag = compute_country_protected_flag(df, target_sector="Retail")
        self.assertTrue((flag == 1).all())


class TestComputeSurvivalFlags(unittest.TestCase):
    """Test the combined survival flag computation."""

    def test_all_flags_computed(self):
        from operator1.analysis.survival_mode import compute_survival_flags
        df = _make_feature_df(10)
        result = compute_survival_flags(df, target_sector="Technology")
        self.assertIn("company_survival_mode_flag", result.columns)
        self.assertIn("country_survival_mode_flag", result.columns)
        self.assertIn("country_protected_flag", result.columns)


# ===========================================================================
# T4.2 -- Vanity percentage tests
# ===========================================================================

class TestVanityComponents(unittest.TestCase):
    """Test individual vanity components."""

    def test_exec_comp_excess(self):
        from operator1.analysis.vanity import _exec_comp_excess
        df = pd.DataFrame({
            "exec_compensation": [100_000.0],
            "net_income": [1_000_000.0],
        })
        # 5% of 1M = 50K; excess = 100K - 50K = 50K
        result = _exec_comp_excess(df, threshold_pct=5.0)
        self.assertAlmostEqual(result.iloc[0], 50_000.0)

    def test_exec_comp_no_excess(self):
        from operator1.analysis.vanity import _exec_comp_excess
        df = pd.DataFrame({
            "exec_compensation": [10_000.0],
            "net_income": [1_000_000.0],
        })
        # 5% of 1M = 50K; 10K < 50K -> 0
        result = _exec_comp_excess(df, threshold_pct=5.0)
        self.assertAlmostEqual(result.iloc[0], 0.0)

    def test_buyback_waste_during_negative_fcf(self):
        from operator1.analysis.vanity import _buyback_waste
        df = pd.DataFrame({
            "share_buybacks": [100_000.0],
            "free_cash_flow_ttm_asof": [-50_000.0],
        })
        result = _buyback_waste(df)
        self.assertAlmostEqual(result.iloc[0], 100_000.0)

    def test_buyback_ok_during_positive_fcf(self):
        from operator1.analysis.vanity import _buyback_waste
        df = pd.DataFrame({
            "share_buybacks": [100_000.0],
            "free_cash_flow_ttm_asof": [200_000.0],
        })
        result = _buyback_waste(df)
        self.assertAlmostEqual(result.iloc[0], 0.0)

    def test_marketing_excess_during_survival(self):
        from operator1.analysis.vanity import _marketing_excess
        df = pd.DataFrame({
            "marketing_expense": [150_000.0],
            "revenue": [1_000_000.0],
            "company_survival_mode_flag": [1],
        })
        # 10% of 1M = 100K; excess = 150K - 100K = 50K
        result = _marketing_excess(df, threshold_pct=10.0)
        self.assertAlmostEqual(result.iloc[0], 50_000.0)

    def test_marketing_not_counted_outside_survival(self):
        from operator1.analysis.vanity import _marketing_excess
        df = pd.DataFrame({
            "marketing_expense": [150_000.0],
            "revenue": [1_000_000.0],
            "company_survival_mode_flag": [0],
        })
        result = _marketing_excess(df, threshold_pct=10.0)
        self.assertAlmostEqual(result.iloc[0], 0.0)


class TestVanityPercentage(unittest.TestCase):
    """Test the full vanity percentage computation."""

    def test_vanity_with_no_components(self):
        from operator1.analysis.vanity import compute_vanity_percentage
        df = _make_feature_df(5)
        result = compute_vanity_percentage(df)
        self.assertIn("vanity_percentage", result.columns)
        self.assertIn("is_missing_vanity_percentage", result.columns)

    def test_vanity_clipped_0_100(self):
        from operator1.analysis.vanity import compute_vanity_percentage
        df = _make_feature_df(3)
        # Huge exec comp -> huge vanity spend
        df["exec_compensation"] = 2_000_000.0  # 200% of net income
        df["net_income"] = 100_000.0
        df["revenue"] = 100_000.0
        result = compute_vanity_percentage(df)
        vp = result["vanity_percentage"]
        valid = vp.dropna()
        if len(valid) > 0:
            self.assertTrue((valid >= 0).all())
            self.assertTrue((valid <= 100).all())

    def test_vanity_null_when_no_revenue(self):
        from operator1.analysis.vanity import compute_vanity_percentage
        df = _make_feature_df(3, revenue=np.full(3, np.nan))
        result = compute_vanity_percentage(df)
        self.assertTrue(result["vanity_percentage"].isna().all())

    def test_vanity_null_when_zero_revenue(self):
        from operator1.analysis.vanity import compute_vanity_percentage
        df = _make_feature_df(3, revenue=np.full(3, 0.0))
        result = compute_vanity_percentage(df)
        self.assertTrue(result["vanity_percentage"].isna().all())


# ===========================================================================
# T4.3 -- Hierarchy weights tests
# ===========================================================================

class TestRegimeSelection(unittest.TestCase):
    """Test the regime selection logic."""

    def test_normal_regime(self):
        from operator1.analysis.hierarchy_weights import _select_regime
        self.assertEqual(_select_regime(0, 0, 0), "normal")

    def test_company_survival_regime(self):
        from operator1.analysis.hierarchy_weights import _select_regime
        self.assertEqual(_select_regime(1, 0, 0), "company_survival")

    def test_modified_survival_regime(self):
        from operator1.analysis.hierarchy_weights import _select_regime
        self.assertEqual(_select_regime(0, 1, 0), "modified_survival")

    def test_extreme_survival_regime(self):
        from operator1.analysis.hierarchy_weights import _select_regime
        self.assertEqual(_select_regime(1, 1, 0), "extreme_survival")

    def test_protected_uses_normal(self):
        from operator1.analysis.hierarchy_weights import _select_regime
        # Country crisis + protected -> "normal" (shielded by government)
        self.assertEqual(_select_regime(0, 1, 1), "normal")

    def test_not_protected_uses_modified(self):
        from operator1.analysis.hierarchy_weights import _select_regime
        # Country crisis + NOT protected -> "modified_survival"
        self.assertEqual(_select_regime(0, 1, 0), "modified_survival")


class TestRegimeSeriesSelection(unittest.TestCase):
    """Test vectorised regime selection."""

    def test_mixed_regimes(self):
        from operator1.analysis.hierarchy_weights import select_regime_series
        df = pd.DataFrame({
            "company_survival_mode_flag": [0, 1, 0, 0, 1],
            "country_survival_mode_flag": [0, 0, 1, 1, 1],
            "country_protected_flag":     [0, 0, 0, 1, 0],
        }, index=_make_daily_index(5))
        regimes = select_regime_series(df)
        self.assertEqual(regimes.iloc[0], "normal")
        self.assertEqual(regimes.iloc[1], "company_survival")
        self.assertEqual(regimes.iloc[2], "modified_survival")
        # Country crisis + protected -> normal (shielded)
        self.assertEqual(regimes.iloc[3], "normal")
        self.assertEqual(regimes.iloc[4], "extreme_survival")


class TestHierarchyWeights(unittest.TestCase):
    """Test the full hierarchy weight computation."""

    def test_normal_weights(self):
        from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
        df = _make_feature_df(10)
        df["company_survival_mode_flag"] = 0
        df["country_survival_mode_flag"] = 0
        df["country_protected_flag"] = 0
        df["vanity_percentage"] = 0.0

        result = compute_hierarchy_weights(df)
        self.assertIn("survival_regime", result.columns)
        self.assertIn("hierarchy_tier1_weight", result.columns)
        self.assertIn("hierarchy_tier5_weight", result.columns)

        # All days should be normal
        self.assertTrue((result["survival_regime"] == "normal").all())

        # Normal weights from config: [0.20, 0.20, 0.20, 0.20, 0.20]
        self.assertAlmostEqual(result["hierarchy_tier1_weight"].iloc[0], 0.20)
        self.assertAlmostEqual(result["hierarchy_tier5_weight"].iloc[0], 0.20)

    def test_weights_sum_to_one(self):
        from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
        df = _make_feature_df(10)
        df["company_survival_mode_flag"] = 0
        df["country_survival_mode_flag"] = 0
        df["country_protected_flag"] = 0
        df["vanity_percentage"] = 0.0

        result = compute_hierarchy_weights(df)
        weight_cols = [f"hierarchy_tier{i}_weight" for i in range(1, 6)]
        sums = result[weight_cols].sum(axis=1)
        for s in sums:
            self.assertAlmostEqual(s, 1.0, places=5)

    def test_extreme_survival_weights(self):
        from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
        df = _make_feature_df(5)
        df["company_survival_mode_flag"] = 1
        df["country_survival_mode_flag"] = 1
        df["country_protected_flag"] = 0
        df["vanity_percentage"] = 0.0

        result = compute_hierarchy_weights(df)
        self.assertTrue((result["survival_regime"] == "extreme_survival").all())

        # Extreme weights: [0.60, 0.30, 0.10, 0.00, 0.00]
        self.assertAlmostEqual(result["hierarchy_tier1_weight"].iloc[0], 0.60)
        self.assertAlmostEqual(result["hierarchy_tier2_weight"].iloc[0], 0.30)

    def test_vanity_adjustment_shifts_weights_in_survival(self):
        from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
        df = _make_feature_df(5)
        df["company_survival_mode_flag"] = 1  # must be in survival for vanity adj
        df["country_survival_mode_flag"] = 0
        df["country_protected_flag"] = 0
        df["vanity_percentage"] = 15.0  # above 10% threshold

        result = compute_hierarchy_weights(df)

        # Company survival weights: [0.50, 0.30, 0.15, 0.04, 0.01]
        # After vanity adj: tier1 += 0.05, tier4 -= 0.02, tier5 -= 0.03
        # -> [0.55, 0.30, 0.15, 0.02, -0.02] -> clamped -> [0.55, 0.30, 0.15, 0.02, 0.00]
        # Sum = 1.02, normalised -> [0.5392, 0.2941, 0.1471, 0.0196, 0.0]
        self.assertGreater(result["hierarchy_tier1_weight"].iloc[0], 0.50)
        self.assertAlmostEqual(result["hierarchy_tier5_weight"].iloc[0], 0.0, places=4)

    def test_vanity_no_adjustment_in_normal_regime(self):
        from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
        df = _make_feature_df(5)
        df["company_survival_mode_flag"] = 0
        df["country_survival_mode_flag"] = 0
        df["country_protected_flag"] = 0
        df["vanity_percentage"] = 15.0  # above threshold, but normal regime

        result = compute_hierarchy_weights(df)
        # Vanity adjustment should NOT apply in normal regime.
        # Should be unmodified normal weights: [0.20, 0.20, 0.20, 0.20, 0.20]
        self.assertAlmostEqual(result["hierarchy_tier1_weight"].iloc[0], 0.20)
        self.assertAlmostEqual(result["hierarchy_tier5_weight"].iloc[0], 0.20)

    def test_vanity_below_threshold_no_change(self):
        from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
        df = _make_feature_df(5)
        df["company_survival_mode_flag"] = 1
        df["country_survival_mode_flag"] = 0
        df["country_protected_flag"] = 0
        df["vanity_percentage"] = 5.0  # below 10% threshold

        result = compute_hierarchy_weights(df)
        # Should be unmodified company_survival weights: [0.50, 0.30, 0.15, 0.04, 0.01]
        self.assertAlmostEqual(result["hierarchy_tier1_weight"].iloc[0], 0.50)
        self.assertAlmostEqual(result["hierarchy_tier4_weight"].iloc[0], 0.04)

    def test_all_four_regimes_in_one_table(self):
        from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
        df = _make_feature_df(4)
        df["company_survival_mode_flag"] = [0, 1, 0, 1]
        df["country_survival_mode_flag"] = [0, 0, 1, 1]
        df["country_protected_flag"] = [0, 0, 0, 0]
        df["vanity_percentage"] = 0.0

        result = compute_hierarchy_weights(df)
        self.assertEqual(result["survival_regime"].iloc[0], "normal")
        self.assertEqual(result["survival_regime"].iloc[1], "company_survival")
        self.assertEqual(result["survival_regime"].iloc[2], "modified_survival")
        self.assertEqual(result["survival_regime"].iloc[3], "extreme_survival")

        # Verify different weight profiles
        self.assertNotAlmostEqual(
            result["hierarchy_tier1_weight"].iloc[0],
            result["hierarchy_tier1_weight"].iloc[3],
        )


# ===========================================================================
# Integration: Phase 4 pipeline smoke test
# ===========================================================================

class TestPhase4Integration(unittest.TestCase):
    """End-to-end: quality -> survival -> vanity -> weights."""

    def test_full_pipeline(self):
        from operator1.quality.data_quality import run_quality_checks
        from operator1.analysis.survival_mode import compute_survival_flags
        from operator1.analysis.vanity import compute_vanity_percentage
        from operator1.analysis.hierarchy_weights import compute_hierarchy_weights

        df = _make_feature_df(30)

        # 1. Quality check
        report = run_quality_checks(df, "target")
        self.assertTrue(report.passed)

        # 2. Survival flags
        df = compute_survival_flags(df, target_sector="Technology")
        self.assertIn("company_survival_mode_flag", df.columns)
        self.assertIn("country_survival_mode_flag", df.columns)
        self.assertIn("country_protected_flag", df.columns)

        # 3. Vanity percentage
        df = compute_vanity_percentage(df)
        self.assertIn("vanity_percentage", df.columns)

        # 4. Hierarchy weights
        df = compute_hierarchy_weights(df)
        self.assertIn("survival_regime", df.columns)
        self.assertIn("hierarchy_tier1_weight", df.columns)
        self.assertIn("hierarchy_tier5_weight", df.columns)

        # Weights should sum to 1
        weight_cols = [f"hierarchy_tier{i}_weight" for i in range(1, 6)]
        sums = df[weight_cols].sum(axis=1)
        for s in sums:
            self.assertAlmostEqual(s, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

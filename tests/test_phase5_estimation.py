"""Phase 5 tests -- Post-cache Sudoku inference (estimation engine)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_index(days: int = 30) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=days, name="date")


def _make_feature_df(days: int = 30, **overrides) -> pd.DataFrame:
    """Build a synthetic daily feature table for estimation tests."""
    idx = _make_daily_index(days)
    np.random.seed(42)

    defaults = {
        "close": 100 + np.cumsum(np.random.randn(days) * 2),
        "return_1d": np.concatenate([[np.nan], np.random.randn(days - 1) * 0.02]),
        "revenue": np.full(days, 1_000_000.0),
        "gross_profit": np.full(days, 600_000.0),
        "ebit": np.full(days, 300_000.0),
        "ebitda": np.full(days, 350_000.0),
        "net_income": np.full(days, 200_000.0),
        "interest_expense": np.full(days, 10_000.0),
        "taxes": np.full(days, 50_000.0),
        "total_assets": np.full(days, 5_000_000.0),
        "total_liabilities": np.full(days, 2_000_000.0),
        "total_equity": np.full(days, 3_000_000.0),
        "current_assets": np.full(days, 1_500_000.0),
        "current_liabilities": np.full(days, 800_000.0),
        "cash_and_equivalents": np.full(days, 500_000.0),
        "short_term_debt": np.full(days, 100_000.0),
        "long_term_debt": np.full(days, 400_000.0),
        "receivables": np.full(days, 200_000.0),
        "operating_cash_flow": np.full(days, 300_000.0),
        "capex": np.full(days, -50_000.0),
        "free_cash_flow": np.full(days, 250_000.0),
        "free_cash_flow_ttm_asof": np.full(days, 250_000.0),
        "total_debt_asof": np.full(days, 500_000.0),
        "net_debt": np.full(days, 0.0),
        "market_cap": np.full(days, 5_000_000.0),
        "volatility_21d": np.full(days, 0.15),
        # Hierarchy weights (needed for Pass 2 tier weighting)
        "hierarchy_tier1_weight": np.full(days, 0.15),
        "hierarchy_tier2_weight": np.full(days, 0.15),
        "hierarchy_tier3_weight": np.full(days, 0.20),
        "hierarchy_tier4_weight": np.full(days, 0.25),
        "hierarchy_tier5_weight": np.full(days, 0.25),
    }

    for k, v in overrides.items():
        defaults[k] = v

    df = pd.DataFrame(defaults, index=idx)

    # Add is_missing flags
    for col in list(df.columns):
        if not col.startswith("is_missing_") and not col.startswith("invalid_math_") \
                and not col.startswith("hierarchy_tier"):
            df[f"is_missing_{col}"] = df[col].isna().astype(int)

    return df


# ===========================================================================
# Pass 1: Deterministic identity fill tests
# ===========================================================================

class TestPass1IdentityFill(unittest.TestCase):
    """Test deterministic accounting identity fill."""

    def test_total_assets_from_components(self):
        """total_assets = total_liabilities + total_equity."""
        from operator1.estimation.estimator import run_pass1_identity_fill

        df = _make_feature_df(10)
        # Remove total_assets -- should be filled from liabilities + equity
        df["total_assets"] = np.nan
        df["is_missing_total_assets"] = 1

        result = run_pass1_identity_fill(df)

        self.assertGreater(result.total_filled, 0)
        self.assertIn("total_assets", result.fills)
        # Should be 2M + 3M = 5M
        self.assertAlmostEqual(df["total_assets"].iloc[0], 5_000_000.0)
        # is_missing flag should be updated
        self.assertEqual(df["is_missing_total_assets"].iloc[0], 0)

    def test_total_equity_from_identity(self):
        """total_equity = total_assets - total_liabilities."""
        from operator1.estimation.estimator import run_pass1_identity_fill

        df = _make_feature_df(10)
        df["total_equity"] = np.nan
        df["is_missing_total_equity"] = 1

        run_pass1_identity_fill(df)

        # 5M - 2M = 3M
        self.assertAlmostEqual(df["total_equity"].iloc[0], 3_000_000.0)

    def test_total_liabilities_from_identity(self):
        """total_liabilities = total_assets - total_equity."""
        from operator1.estimation.estimator import run_pass1_identity_fill

        df = _make_feature_df(10)
        df["total_liabilities"] = np.nan
        df["is_missing_total_liabilities"] = 1

        run_pass1_identity_fill(df)

        # 5M - 3M = 2M
        self.assertAlmostEqual(df["total_liabilities"].iloc[0], 2_000_000.0)

    def test_net_debt_from_components(self):
        """net_debt = total_debt_asof - cash_and_equivalents."""
        from operator1.estimation.estimator import run_pass1_identity_fill

        df = _make_feature_df(10)
        df["net_debt"] = np.nan
        df["is_missing_net_debt"] = 1

        run_pass1_identity_fill(df)

        # 500K - 500K = 0
        self.assertAlmostEqual(df["net_debt"].iloc[0], 0.0)

    def test_total_debt_from_components(self):
        """total_debt_asof = short_term_debt + long_term_debt."""
        from operator1.estimation.estimator import run_pass1_identity_fill

        df = _make_feature_df(10)
        df["total_debt_asof"] = np.nan
        df["is_missing_total_debt_asof"] = 1

        run_pass1_identity_fill(df)

        # 100K + 400K = 500K
        self.assertAlmostEqual(df["total_debt_asof"].iloc[0], 500_000.0)

    def test_no_fill_when_all_present(self):
        """No fills should occur when everything is observed."""
        from operator1.estimation.estimator import run_pass1_identity_fill

        df = _make_feature_df(10)
        result = run_pass1_identity_fill(df)
        self.assertEqual(result.total_filled, 0)

    def test_cascading_fill(self):
        """Filling one variable enables filling another."""
        from operator1.estimation.estimator import run_pass1_identity_fill

        df = _make_feature_df(10)
        # Remove both total_debt_asof AND net_debt
        # total_debt_asof can be filled from short+long debt,
        # then net_debt can be filled from total_debt - cash
        df["total_debt_asof"] = np.nan
        df["net_debt"] = np.nan
        df["is_missing_total_debt_asof"] = 1
        df["is_missing_net_debt"] = 1

        result = run_pass1_identity_fill(df)

        self.assertIn("total_debt_asof", result.fills)
        self.assertIn("net_debt", result.fills)
        self.assertAlmostEqual(df["total_debt_asof"].iloc[0], 500_000.0)
        self.assertAlmostEqual(df["net_debt"].iloc[0], 0.0)

    def test_partial_missing_in_rows(self):
        """Only rows with missing values should be filled."""
        from operator1.estimation.estimator import run_pass1_identity_fill

        df = _make_feature_df(10)
        # Only make first 3 rows missing for total_assets
        df.loc[df.index[:3], "total_assets"] = np.nan

        result = run_pass1_identity_fill(df)

        # Only 3 rows should be filled
        self.assertEqual(result.fills.get("total_assets", 0), 3)
        # Last rows should be unchanged
        self.assertAlmostEqual(df["total_assets"].iloc[5], 5_000_000.0)


# ===========================================================================
# Pass 2: Rolling imputer tests
# ===========================================================================

class TestPass2RollingImputer(unittest.TestCase):
    """Test the regime-weighted rolling imputer."""

    def test_estimate_fills_missing(self):
        from operator1.estimation.estimator import _estimate_variable_rolling

        df = _make_feature_df(50)
        # Introduce some missing values in the latter half
        missing_idx = df.index[40:]
        df.loc[missing_idx, "revenue"] = np.nan

        estimated, confidence = _estimate_variable_rolling(df, "revenue")

        # Should have estimates for the missing positions
        n_estimated = estimated.loc[missing_idx].notna().sum()
        self.assertGreater(n_estimated, 0)

        # Confidence should be between 0 and 1
        valid_conf = confidence.dropna()
        if len(valid_conf) > 0:
            self.assertTrue((valid_conf >= 0).all())
            self.assertTrue((valid_conf <= 1).all())

    def test_no_estimation_when_all_observed(self):
        from operator1.estimation.estimator import _estimate_variable_rolling

        df = _make_feature_df(30)
        # No missing values
        estimated, confidence = _estimate_variable_rolling(df, "revenue")
        self.assertTrue(estimated.isna().all())

    def test_observed_values_never_overwritten(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(30)
        original_revenue = df["revenue"].copy()

        # Introduce some missing values
        df.loc[df.index[25:], "revenue"] = np.nan

        result, coverage = run_estimation(df, variables=["revenue"])

        # Check that observed values are preserved
        observed_mask = original_revenue.notna() & (df.index < df.index[25])
        revenue_final = result["revenue_final"]
        for idx in df.index[observed_mask]:
            self.assertAlmostEqual(
                revenue_final.loc[idx],
                original_revenue.loc[idx],
                msg=f"Observed value overwritten at {idx}",
            )

    def test_source_column_values(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(30)
        df.loc[df.index[25:], "revenue"] = np.nan

        result, _ = run_estimation(df, variables=["revenue"])

        self.assertIn("revenue_source", result.columns)
        # First 25 days should be "observed"
        observed_sources = result["revenue_source"].iloc[:25]
        self.assertTrue((observed_sources == "observed").all())

    def test_confidence_for_observed_is_one(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(30)
        df.loc[df.index[25:], "revenue"] = np.nan

        result, _ = run_estimation(df, variables=["revenue"])

        # Observed days should have confidence = 1.0
        observed_conf = result["revenue_confidence"].iloc[:25]
        self.assertTrue((observed_conf == 1.0).all())


# ===========================================================================
# Estimation output columns
# ===========================================================================

class TestEstimationColumns(unittest.TestCase):
    """Test that output columns are correctly structured."""

    def test_all_output_columns_present(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(20)
        df.loc[df.index[15:], "revenue"] = np.nan

        result, _ = run_estimation(df, variables=["revenue"])

        self.assertIn("revenue_observed", result.columns)
        self.assertIn("revenue_estimated", result.columns)
        self.assertIn("revenue_final", result.columns)
        self.assertIn("revenue_source", result.columns)
        self.assertIn("revenue_confidence", result.columns)

    def test_observed_column_matches_original(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(20)
        original = df["revenue"].copy()
        df.loc[df.index[15:], "revenue"] = np.nan

        result, _ = run_estimation(df, variables=["revenue"])

        # revenue_observed should match the input (with NaN where missing)
        np.testing.assert_array_equal(
            result["revenue_observed"].iloc[:15].values,
            original.iloc[:15].values,
        )
        self.assertTrue(result["revenue_observed"].iloc[15:].isna().all())

    def test_final_prefers_observed_over_estimated(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(30)
        original = df["revenue"].copy()
        df.loc[df.index[25:], "revenue"] = np.nan

        result, _ = run_estimation(df, variables=["revenue"])

        # For observed rows, final should equal observed
        for i in range(25):
            self.assertAlmostEqual(
                result["revenue_final"].iloc[i],
                original.iloc[i],
            )

    def test_columns_for_fully_observed_variable(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(20)
        # No missing values in revenue
        result, _ = run_estimation(df, variables=["revenue"])

        self.assertIn("revenue_final", result.columns)
        self.assertIn("revenue_source", result.columns)
        # All should be observed
        self.assertTrue((result["revenue_source"] == "observed").all())
        self.assertTrue((result["revenue_confidence"] == 1.0).all())


# ===========================================================================
# Coverage tracking
# ===========================================================================

class TestEstimationCoverage(unittest.TestCase):
    """Test coverage statistics."""

    def test_coverage_before_after(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(30)
        df.loc[df.index[20:], "revenue"] = np.nan

        _, coverage = run_estimation(df, variables=["revenue"])

        self.assertIn("revenue", coverage.coverage_before)
        self.assertIn("revenue", coverage.coverage_after)

        # Before should be ~20/30
        self.assertAlmostEqual(coverage.coverage_before["revenue"], 20 / 30, places=2)

        # After should be >= before (estimation fills gaps)
        self.assertGreaterEqual(
            coverage.coverage_after["revenue"],
            coverage.coverage_before["revenue"],
        )

    def test_coverage_to_dict(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(20)
        _, coverage = run_estimation(df, variables=["revenue"])

        d = coverage.to_dict()
        self.assertIn("pass1_fills", d)
        self.assertIn("pass2_estimates", d)
        self.assertIn("coverage_before", d)
        self.assertIn("coverage_after", d)

    def test_full_coverage_no_estimation(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(20)
        _, coverage = run_estimation(df, variables=["revenue"])

        # No missing values -> no estimation needed
        self.assertEqual(coverage.pass2_estimates.get("revenue", 0), 0)
        self.assertAlmostEqual(coverage.coverage_before["revenue"], 1.0)
        self.assertAlmostEqual(coverage.coverage_after["revenue"], 1.0)


# ===========================================================================
# Tier membership
# ===========================================================================

class TestTierMembership(unittest.TestCase):
    """Test tier membership lookup from config."""

    def test_membership_from_config(self):
        from operator1.estimation.estimator import _build_tier_membership

        membership = _build_tier_membership()

        # Per corrected survival_hierarchy.yml:
        # tier1 = Liquidity & Cash (cash_ratio, etc.)
        # tier2 = Debt & Solvency (current_ratio, debt_to_equity, etc.)
        # tier3 = Market Stability (volatility_21d, drawdown_252d, volume)
        # tier4 = Profitability (gross_margin, operating_margin, net_margin)
        # tier5 = Growth & Valuation (revenue_asof, pe_ratio, ev_to_ebitda)

        # current_ratio is in tier 2 (Debt & Solvency)
        self.assertEqual(membership.get("current_ratio"), 2)

        # cash_ratio is in tier 1 (Liquidity & Cash)
        self.assertEqual(membership.get("cash_ratio"), 1)

        # gross_margin should be in tier 4
        self.assertEqual(membership.get("gross_margin"), 4)

        # pe_ratio_calc should be in tier 5 (canonical computed name from derived_variables)
        self.assertEqual(membership.get("pe_ratio_calc"), 5)


# ===========================================================================
# Integration: full estimation pipeline
# ===========================================================================

class TestEstimationIntegration(unittest.TestCase):
    """End-to-end estimation pipeline test."""

    def test_full_pipeline_with_missing_data(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(50)

        # Introduce various missing patterns
        df.loc[df.index[40:], "total_assets"] = np.nan
        df.loc[df.index[35:], "net_debt"] = np.nan
        df.loc[df.index[30:], "revenue"] = np.nan

        result, coverage = run_estimation(df)

        # Pass 1 should fill total_assets from identity
        self.assertIn("total_assets", coverage.pass1_fills)

        # Final columns should exist
        self.assertIn("total_assets_final", result.columns)
        self.assertIn("revenue_final", result.columns)

        # Coverage should improve
        self.assertGreaterEqual(
            coverage.coverage_after.get("total_assets", 0),
            coverage.coverage_before.get("total_assets", 0),
        )

    def test_pipeline_with_no_missing(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(20)
        result, coverage = run_estimation(df)

        # No fills or estimates needed
        self.assertEqual(coverage.pass1_fills, {})

    def test_pipeline_preserves_original_data(self):
        from operator1.estimation.estimator import run_estimation

        df = _make_feature_df(30)
        original_values = df["total_equity"].iloc[:20].copy()

        df.loc[df.index[20:], "total_equity"] = np.nan
        result, _ = run_estimation(df, variables=["total_equity"])

        # Original observed values should be untouched in _final
        final = result["total_equity_final"]
        for i in range(20):
            self.assertAlmostEqual(
                final.iloc[i], original_values.iloc[i],
                msg=f"Original value modified at position {i}",
            )


if __name__ == "__main__":
    unittest.main()

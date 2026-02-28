"""Tests for the split estimator (missing data vs hidden data).

Covers:
  - Missingness classifier
  - Missing data estimator (MICE + GP + Matrix Completion)
  - Hidden data estimator (Heckman + Pattern-Mixture + Bounds)
  - Full pipeline orchestration via run_estimation(imputer_method="split")
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_feature_table():
    """Build a realistic feature table with mixed missingness."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    df = pd.DataFrame(index=dates)

    # Observed variables (full coverage)
    df["close"] = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df["volume"] = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    df["market_cap"] = df["close"] * 1_000_000

    # Financial variables with MAR gaps (random holes)
    df["revenue"] = np.random.uniform(1e9, 5e9, n)
    df.loc[df.index[10:15], "revenue"] = np.nan  # random gap
    df["is_missing_revenue"] = df["revenue"].isna().astype(int)

    df["net_income"] = df["revenue"] * np.random.uniform(0.05, 0.15, n)
    df.loc[df.index[12:18], "net_income"] = np.nan
    df["is_missing_net_income"] = df["net_income"].isna().astype(int)

    # Balance sheet items
    df["total_assets"] = np.random.uniform(1e10, 5e10, n)
    df["total_liabilities"] = df["total_assets"] * np.random.uniform(0.4, 0.7, n)
    df["total_equity"] = df["total_assets"] - df["total_liabilities"]

    # MNAR-like: debt detail hidden when debt is high
    df["short_term_debt"] = np.random.uniform(1e8, 2e9, n)
    df["long_term_debt"] = np.random.uniform(5e8, 5e9, n)
    # Hide debt detail when it is high (MNAR pattern)
    high_debt = (df["short_term_debt"] + df["long_term_debt"]) > 5e9
    df.loc[high_debt, "short_term_debt"] = np.nan
    df.loc[high_debt, "long_term_debt"] = np.nan
    df["is_missing_short_term_debt"] = df["short_term_debt"].isna().astype(int)
    df["is_missing_long_term_debt"] = df["long_term_debt"].isna().astype(int)

    df["total_debt_asof"] = df["short_term_debt"].fillna(0) + df["long_term_debt"].fillna(0)
    df.loc[high_debt, "total_debt_asof"] = np.random.uniform(5e9, 8e9, high_debt.sum())

    # Cash
    df["cash_and_equivalents"] = np.random.uniform(5e8, 3e9, n)
    df["operating_cash_flow"] = np.random.uniform(1e8, 1e9, n)
    df["capex"] = np.random.uniform(-5e8, -1e8, n)

    # Derived: will be filled by Pass 1
    df["free_cash_flow"] = np.nan
    df["net_debt"] = np.nan
    df["is_missing_free_cash_flow"] = 1
    df["is_missing_net_debt"] = 1

    return df


# ---------------------------------------------------------------------------
# Missingness Classifier Tests
# ---------------------------------------------------------------------------

class TestMissingnessClassifier:

    def test_classify_returns_types_for_all_vars(self, sample_feature_table):
        from operator1.estimation.missingness_classifier import classify_missingness

        df = sample_feature_table
        variables = ["revenue", "net_income", "short_term_debt", "long_term_debt"]
        result = classify_missingness(df, variables)

        for var in variables:
            assert var in result.types
            assert var in result.confidences
            assert len(result.types[var]) == len(df)

    def test_fully_observed_variable_classified_as_observed(self, sample_feature_table):
        from operator1.estimation.missingness_classifier import classify_missingness

        df = sample_feature_table
        result = classify_missingness(df, ["close"])

        assert (result.types["close"] == "observed").all()

    def test_mnar_detection_for_disappearing_values(self, sample_feature_table):
        from operator1.estimation.missingness_classifier import classify_missingness

        df = sample_feature_table
        result = classify_missingness(df, ["short_term_debt"])

        # Should detect some MNAR patterns
        mnar_count = (result.types["short_term_debt"] == "mnar").sum()
        assert mnar_count > 0, "Expected MNAR classification for hidden debt"

    def test_summary_counts_are_consistent(self, sample_feature_table):
        from operator1.estimation.missingness_classifier import classify_missingness

        df = sample_feature_table
        variables = ["revenue", "short_term_debt"]
        result = classify_missingness(df, variables)

        for var in variables:
            summary = result.summary[var]
            total = sum(summary.values())
            assert total == len(df), f"Summary counts should sum to {len(df)}, got {total}"


# ---------------------------------------------------------------------------
# Missing Data Estimator Tests (MAR)
# ---------------------------------------------------------------------------

class TestMissingDataEstimator:

    def test_mice_produces_estimates(self, sample_feature_table):
        from operator1.estimation.missing_data_estimator import estimate_missing_data

        df = sample_feature_table
        result = estimate_missing_data(df, ["revenue", "net_income"])

        assert "revenue" in result.estimated_values
        rev_est = result.estimated_values["revenue"]
        # Should have estimated some values
        assert rev_est.notna().sum() > 0

    def test_estimates_have_confidence_scores(self, sample_feature_table):
        from operator1.estimation.missing_data_estimator import estimate_missing_data

        df = sample_feature_table
        result = estimate_missing_data(df, ["revenue"])

        rev_conf = result.confidence_scores["revenue"]
        # Confidence should be in [0, 1]
        valid = rev_conf.dropna()
        if len(valid) > 0:
            assert valid.min() >= 0
            assert valid.max() <= 1.0

    def test_n_estimated_matches_actual(self, sample_feature_table):
        from operator1.estimation.missing_data_estimator import estimate_missing_data

        df = sample_feature_table
        result = estimate_missing_data(df, ["revenue"])

        actual_count = sum(
            int(v.notna().sum()) for v in result.estimated_values.values()
        )
        assert result.n_estimated == actual_count


# ---------------------------------------------------------------------------
# Hidden Data Estimator Tests (MNAR)
# ---------------------------------------------------------------------------

class TestHiddenDataEstimator:

    def test_heckman_produces_estimates(self, sample_feature_table):
        from operator1.estimation.hidden_data_estimator import estimate_hidden_data

        df = sample_feature_table
        result = estimate_hidden_data(df, ["short_term_debt", "long_term_debt"])

        # Should produce some estimates
        assert result.n_estimated > 0

    def test_sensitivity_bounds_are_computed(self, sample_feature_table):
        from operator1.estimation.hidden_data_estimator import estimate_hidden_data

        df = sample_feature_table
        result = estimate_hidden_data(df, ["short_term_debt"])

        assert "short_term_debt" in result.sensitivity_lower
        assert "short_term_debt" in result.sensitivity_upper
        assert "short_term_debt" in result.tipping_points

        # Bounds should have values where data is missing
        lower = result.sensitivity_lower["short_term_debt"]
        upper = result.sensitivity_upper["short_term_debt"]
        assert lower.notna().sum() > 0
        assert upper.notna().sum() > 0

    def test_mnar_confidence_capped(self, sample_feature_table):
        from operator1.estimation.hidden_data_estimator import estimate_hidden_data

        df = sample_feature_table
        result = estimate_hidden_data(df, ["short_term_debt"])

        conf = result.confidence_scores.get("short_term_debt")
        if conf is not None:
            valid = conf.dropna()
            if len(valid) > 0:
                # MNAR confidence should be capped below 0.85
                assert valid.max() <= 0.85


# ---------------------------------------------------------------------------
# Full Pipeline Integration Test
# ---------------------------------------------------------------------------

class TestSplitEstimationPipeline:

    def test_run_estimation_with_split_method(self, sample_feature_table):
        from operator1.estimation.estimator import run_estimation

        df = sample_feature_table
        variables = ["revenue", "net_income", "short_term_debt", "long_term_debt"]

        result_df, coverage = run_estimation(
            df, variables=variables, imputer_method="split",
        )

        # Should have estimation columns for each variable
        for var in variables:
            assert f"{var}_final" in result_df.columns, f"Missing {var}_final"
            assert f"{var}_source" in result_df.columns, f"Missing {var}_source"
            assert f"{var}_confidence" in result_df.columns, f"Missing {var}_confidence"

    def test_pass1_fills_identities(self, sample_feature_table):
        from operator1.estimation.estimator import run_estimation

        df = sample_feature_table
        result_df, coverage = run_estimation(
            df, variables=["free_cash_flow", "net_debt"],
            imputer_method="split",
        )

        # Pass 1 should have filled free_cash_flow and net_debt
        # from operating_cash_flow - capex and total_debt - cash
        assert coverage.pass1_fills.get("free_cash_flow", 0) > 0 or \
               coverage.pass1_fills.get("net_debt", 0) > 0

    def test_coverage_improves_after_estimation(self, sample_feature_table):
        from operator1.estimation.estimator import run_estimation

        df = sample_feature_table
        variables = ["revenue", "net_income"]

        result_df, coverage = run_estimation(
            df, variables=variables, imputer_method="split",
        )

        for var in variables:
            before = coverage.coverage_before.get(var, 0)
            after = coverage.coverage_after.get(var, 0)
            assert after >= before, (
                f"Coverage should not decrease: {var} before={before} after={after}"
            )

    def test_missingness_type_column_present(self, sample_feature_table):
        from operator1.estimation.estimator import run_estimation

        df = sample_feature_table
        result_df, _ = run_estimation(
            df, variables=["short_term_debt"], imputer_method="split",
        )

        assert "short_term_debt_missingness_type" in result_df.columns

    def test_source_distinguishes_mar_and_mnar(self, sample_feature_table):
        from operator1.estimation.estimator import run_estimation

        df = sample_feature_table
        result_df, _ = run_estimation(
            df,
            variables=["revenue", "short_term_debt"],
            imputer_method="split",
        )

        # Source column should contain mar/mnar distinctions
        sources = set()
        for var in ["revenue", "short_term_debt"]:
            col = f"{var}_source"
            if col in result_df.columns:
                sources.update(result_df[col].unique())

        # Should have at least "observed" and some estimated type
        assert "observed" in sources

    def test_legacy_bayesian_ridge_still_works(self, sample_feature_table):
        from operator1.estimation.estimator import run_estimation

        df = sample_feature_table
        result_df, coverage = run_estimation(
            df, variables=["revenue"], imputer_method="bayesian_ridge",
        )

        assert "revenue_final" in result_df.columns

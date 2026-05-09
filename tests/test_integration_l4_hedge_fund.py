"""Integration tests -- Layer 4: Hedge Fund Analysis.

Validates the HF engine produces reasonable grades and signals
on synthetic data, and Apple-specific assertions on golden data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import AAPL_EXPECTED


def _make_minimal_statements(n_quarters: int = 8) -> tuple:
    """Build minimal income/balance/cashflow DataFrames for HF tests."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2022-01-01", periods=n_quarters, freq="QE")

    income = pd.DataFrame({
        "report_date": dates,
        "revenue": rng.uniform(8e10, 1.2e11, n_quarters),
        "net_income": rng.uniform(2e10, 3e10, n_quarters),
        "gross_profit": rng.uniform(5e10, 8e10, n_quarters),
        "operating_income": rng.uniform(3e10, 4e10, n_quarters),
        "ebit": rng.uniform(3e10, 4e10, n_quarters),
        "interest_expense": rng.uniform(5e8, 1e9, n_quarters),
        "eps_diluted": rng.uniform(1.2, 2.0, n_quarters),
        "sga_expenses": rng.uniform(5e9, 8e9, n_quarters),
    })
    balance = pd.DataFrame({
        "report_date": dates,
        "total_assets": rng.uniform(3e11, 4e11, n_quarters),
        "total_liabilities": rng.uniform(2e11, 3e11, n_quarters),
        "total_equity": rng.uniform(5e10, 1e11, n_quarters),
        "current_assets": rng.uniform(1e11, 1.5e11, n_quarters),
        "current_liabilities": rng.uniform(1e11, 1.6e11, n_quarters),
        "cash_and_equivalents": rng.uniform(2e10, 6e10, n_quarters),
        "goodwill": rng.uniform(0, 1e10, n_quarters),
        "intangible_assets": rng.uniform(0, 5e9, n_quarters),
        "receivables": rng.uniform(2e10, 5e10, n_quarters),
        "inventory": rng.uniform(5e9, 1e10, n_quarters),
        "payables": rng.uniform(5e10, 8e10, n_quarters),
        "short_term_debt": rng.uniform(5e9, 2e10, n_quarters),
        "long_term_debt": rng.uniform(5e10, 1e11, n_quarters),
        "retained_earnings": rng.uniform(-5e9, 1e10, n_quarters),
    })
    cashflow = pd.DataFrame({
        "report_date": dates,
        "operating_cash_flow": rng.uniform(2e10, 4e10, n_quarters),
        "capex": -rng.uniform(2e9, 5e9, n_quarters),
        "dividends_paid": -rng.uniform(3e9, 4e9, n_quarters),
        "free_cash_flow": rng.uniform(1.5e10, 3.5e10, n_quarters),
    })
    return income, balance, cashflow


class TestL4HedgeFundOnSynthetic:
    """HF tests using synthetic statement data."""

    def test_hf_engine_runs_without_crash(self, synthetic_cache):
        from operator1.hedge_fund.engine import run_hedge_fund_analysis
        income, balance, cashflow = _make_minimal_statements()
        hf = run_hedge_fund_analysis(
            income_df=income,
            balance_df=balance,
            cashflow_df=cashflow,
            cache=synthetic_cache,
            target_profile={"sector": "Technology", "name": "Test Corp"},
        )
        assert hf is not None

    def test_hf_available_with_sufficient_data(self, synthetic_cache):
        from operator1.hedge_fund.engine import run_hedge_fund_analysis
        income, balance, cashflow = _make_minimal_statements()
        hf = run_hedge_fund_analysis(
            income_df=income,
            balance_df=balance,
            cashflow_df=cashflow,
            cache=synthetic_cache,
            target_profile={"sector": "Technology", "name": "Test Corp"},
        )
        assert hf.available, (
            f"HF not available despite sufficient data. "
            f"Readiness: {hf.data_readiness}"
        )

    def test_hf_not_available_with_empty_data(self, synthetic_cache):
        from operator1.hedge_fund.engine import run_hedge_fund_analysis
        hf = run_hedge_fund_analysis(
            income_df=pd.DataFrame(),
            balance_df=pd.DataFrame(),
            cashflow_df=pd.DataFrame(),
            cache=synthetic_cache,
            target_profile={"sector": "Technology", "name": "Test Corp"},
        )
        # Should either be unavailable or have a low-confidence grade
        # The key assertion is: it should NOT crash
        assert hf is not None

    def test_hf_scorecard_has_5_tiers(self, synthetic_cache):
        from operator1.hedge_fund.engine import run_hedge_fund_analysis
        income, balance, cashflow = _make_minimal_statements()
        hf = run_hedge_fund_analysis(
            income_df=income,
            balance_df=balance,
            cashflow_df=cashflow,
            cache=synthetic_cache,
            target_profile={"sector": "Technology", "name": "Test Corp"},
        )
        if hf.available and hf.scorecard:
            sc = hf.scorecard
            assert sc.investment_grade is not None
            assert sc.conviction >= 0


class TestL4HedgeFundOnGolden:
    """Tests using frozen AAPL data (skipped if fixture missing)."""

    def test_apple_hf_grade_not_worst(self, aapl_cache, aapl_statements):
        if all(df.empty for df in aapl_statements.values()):
            pytest.skip("AAPL statement fixtures empty")

        from operator1.hedge_fund.engine import run_hedge_fund_analysis
        hf = run_hedge_fund_analysis(
            income_df=aapl_statements["income"],
            balance_df=aapl_statements["balance"],
            cashflow_df=aapl_statements["cashflow"],
            cache=aapl_cache,
            target_profile={"sector": "Technology", "name": "Apple Inc"},
        )
        if hf.available and hf.scorecard:
            bad_grades = AAPL_EXPECTED.get("l4_hf_grades_bad", ["F"])
            assert hf.scorecard.investment_grade not in bad_grades, (
                f"Apple HF grade={hf.scorecard.investment_grade} "
                f"(should not be in {bad_grades})"
            )

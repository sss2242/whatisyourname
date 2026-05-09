"""Integration tests -- Graceful Degradation.

Parametrized fault injection: verifies that pipeline modules produce
reasonable output (or fail gracefully) when given incomplete data.

Pattern: Evidently parametrized edge cases + Netflix fault injection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _minimal_cache(n: int = 100, **overrides) -> pd.DataFrame:
    """Build a minimal cache with optional column overrides."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2023-06-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"close": 100 + np.cumsum(rng.randn(n) * 0.3)},
        index=dates,
    )
    df["return_1d"] = df["close"].pct_change()
    df["volume"] = rng.uniform(1e6, 1e8, n)
    for k, v in overrides.items():
        df[k] = v
    df.index.name = "date"
    return df


class TestFHGracefulDegradation:
    """Financial health with missing/partial data."""

    @pytest.mark.parametrize(
        "extra_cols,expected_min",
        [
            ({"current_ratio": 1.5, "gross_margin": 0.5}, 0),
            ({"volatility_21d": 0.02}, 0),
            ({}, 0),  # only close + return_1d
        ],
        ids=["partial_ratios", "only_vol", "bare_minimum"],
    )
    def test_fh_with_partial_data(self, extra_cols, expected_min):
        from operator1.models.financial_health import compute_financial_health
        cache = _minimal_cache(**extra_cols)
        cache_out, result = compute_financial_health(cache)
        assert result is not None
        assert result.latest_composite >= expected_min

    def test_fh_with_all_nan_ratios(self):
        from operator1.models.financial_health import compute_financial_health
        cache = _minimal_cache(
            current_ratio=float("nan"),
            gross_margin=float("nan"),
        )
        cache_out, result = compute_financial_health(cache)
        assert result is not None  # should not crash


class TestSurvivalGracefulDegradation:
    """Survival mode with missing trigger variables."""

    def test_survival_with_no_trigger_columns(self):
        from operator1.analysis.survival_mode import compute_company_survival_flag
        cache = _minimal_cache()
        flag = compute_company_survival_flag(cache)
        assert isinstance(flag, pd.Series)
        assert len(flag) == len(cache)

    def test_survival_probability_with_minimal_data(self):
        from operator1.analysis.survival_mode import compute_survival_probability
        cache = _minimal_cache()
        prob = compute_survival_probability(cache)
        assert isinstance(prob, pd.Series)
        valid = prob.dropna()
        if len(valid) > 0:
            assert valid.min() >= 0.0
            assert valid.max() <= 1.0


class TestMCGracefulDegradation:
    """Monte Carlo with edge case inputs."""

    def test_mc_with_no_regime_labels(self):
        """MC should run with single 'unknown' regime when labels missing."""
        from operator1.models.monte_carlo import run_monte_carlo
        cache = _minimal_cache(n=200)
        mc = run_monte_carlo(cache, n_paths=100, random_state=42)
        assert mc is not None
        assert mc.survival_probability is not None

    def test_mc_with_short_history(self):
        """MC should handle very short history (50 days)."""
        from operator1.models.monte_carlo import run_monte_carlo
        cache = _minimal_cache(n=50)
        cache["regime_label"] = "unknown"
        mc = run_monte_carlo(cache, n_paths=100, random_state=42)
        assert mc is not None

    def test_mc_with_constant_returns(self):
        """MC should handle zero-variance returns (constant price)."""
        from operator1.models.monte_carlo import run_monte_carlo
        cache = _minimal_cache(n=200)
        cache["return_1d"] = 0.0
        cache["regime_label"] = "bull"
        mc = run_monte_carlo(cache, n_paths=100, random_state=42)
        assert mc is not None


class TestHFGracefulDegradation:
    """Hedge fund with empty/partial statement data."""

    def test_hf_with_empty_statements(self):
        from operator1.hedge_fund.engine import run_hedge_fund_analysis
        cache = _minimal_cache(n=200)
        hf = run_hedge_fund_analysis(
            income_df=pd.DataFrame(),
            balance_df=pd.DataFrame(),
            cashflow_df=pd.DataFrame(),
            cache=cache,
            target_profile={"sector": "Technology", "name": "Empty Corp"},
        )
        assert hf is not None  # must not crash

    def test_hf_with_single_quarter(self):
        """HF with only 1 quarter of data (below 4Q minimum for most metrics)."""
        from operator1.hedge_fund.engine import run_hedge_fund_analysis
        cache = _minimal_cache(n=200)
        income = pd.DataFrame({
            "report_date": [pd.Timestamp("2024-03-31")],
            "revenue": [1e10],
            "net_income": [2e9],
        })
        balance = pd.DataFrame({
            "report_date": [pd.Timestamp("2024-03-31")],
            "total_assets": [5e10],
            "total_equity": [2e10],
        })
        hf = run_hedge_fund_analysis(
            income_df=income,
            balance_df=balance,
            cashflow_df=pd.DataFrame(),
            cache=cache,
            target_profile={"sector": "Technology", "name": "Single Q Corp"},
        )
        assert hf is not None

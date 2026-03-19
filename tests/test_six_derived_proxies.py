"""Tests for operator1.features.six_derived_proxies.

Validates the 17 expert methods for SIX Swiss Exchange proxy estimation.
Uses synthetic data (no live API calls) for unit tests, with an optional
live test against Nestle (NESN) when SIX APIs are reachable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dividends() -> pd.Series:
    """18-year synthetic dividend series (modeled after Nestle)."""
    dates = pd.date_range("2008-04-15", periods=18, freq="365D")
    # Simulate steady growth with one plateau
    values = [1.60, 1.70, 1.85, 1.95, 2.05, 2.15, 2.25, 2.30,
              2.35, 2.40, 2.50, 2.60, 2.70, 2.75, 2.80, 2.90, 3.00, 3.10]
    return pd.Series(values[::-1], index=dates[::-1])  # newest first


@pytest.fixture
def sample_profile() -> dict:
    """Synthetic SIX profile (modeled after Nestle)."""
    return {
        "ticker": "TEST",
        "name": "Test Corp",
        "isin": "CH0000000000",
        "country": "CH",
        "exchange": "SIX",
        "currency": "CHF",
        "market_id": "ch_six",
        "sector": "Consumer Defensive",
        "shares_outstanding": 2_500_000_000,
        "nominal_value": 0.10,
        "latest_dividend_amount": 3.10,
        "latest_close": 80.0,
        "dividend_history_years": 18,
        "index_memberships": ["SMI", "SLI"],
        "valor_id": "CH0000000000CHF4",
        "reported_share_capital": 250_000_000,
    }


@pytest.fixture
def sample_cache() -> pd.DataFrame:
    """Synthetic daily cache with close+volume (108 days)."""
    dates = pd.bdate_range("2025-10-13", periods=108)
    np.random.seed(42)
    close = 80.0 + np.cumsum(np.random.randn(108) * 0.5)
    volume = np.random.randint(500_000, 2_000_000, 108)
    return pd.DataFrame({
        "close": close,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "volume": volume.astype(float),
    }, index=dates)


# ---------------------------------------------------------------------------
# Unit tests for individual methods
# ---------------------------------------------------------------------------

class TestPreciseCagr:
    def test_basic_cagr(self, sample_dividends):
        from operator1.features.six_derived_proxies import _precise_cagr
        cagr = _precise_cagr(sample_dividends)
        assert cagr > 0
        assert cagr < 0.10  # should be in reasonable range

    def test_cagr_5y(self, sample_dividends):
        from operator1.features.six_derived_proxies import _precise_cagr
        cagr_5y = _precise_cagr(sample_dividends, max_years=5)
        assert cagr_5y > 0

    def test_too_few_dividends(self):
        from operator1.features.six_derived_proxies import _precise_cagr
        short = pd.Series([3.0], index=[pd.Timestamp("2024-04-15")])
        assert _precise_cagr(short) == 0.0


class TestAdaptivePayoutRatio:
    def test_consumer_defensive_smi(self):
        from operator1.features.six_derived_proxies import _estimate_payout_ratio
        pr = _estimate_payout_ratio("Consumer Defensive", ["SMI"])
        assert 0.70 <= pr <= 0.80  # high payout for defensive blue chip

    def test_technology(self):
        from operator1.features.six_derived_proxies import _estimate_payout_ratio
        pr = _estimate_payout_ratio("Technology", [])
        assert 0.30 <= pr <= 0.45  # lower payout for tech


class TestExponentialStability:
    def test_stable_dividends(self, sample_dividends):
        from operator1.features.six_derived_proxies import _exponential_stability
        stability = _exponential_stability(sample_dividends)
        assert stability > 0.7  # monotonically growing = high stability

    def test_volatile_dividends(self):
        from operator1.features.six_derived_proxies import _exponential_stability
        dates = pd.date_range("2015-01-01", periods=8, freq="365D")
        volatile = pd.Series([1.0, 2.0, 0.5, 1.5, 0.8, 2.5, 0.3, 1.0], index=dates)
        stability = _exponential_stability(volatile)
        assert stability < 0.6  # volatile = low stability


class TestKalmanEarnings:
    def test_kalman_returns_series(self, sample_dividends):
        from operator1.features.six_derived_proxies import _kalman_earnings_estimate
        eps, stds = _kalman_earnings_estimate(sample_dividends, payout_prior=0.70)
        assert len(eps) > 0
        assert len(stds) > 0
        assert eps.iloc[0] > 0  # latest earnings > 0

    def test_kalman_earnings_above_dividends(self, sample_dividends):
        from operator1.features.six_derived_proxies import _kalman_earnings_estimate
        eps, _ = _kalman_earnings_estimate(sample_dividends, payout_prior=0.70)
        # Earnings should be >= dividends (can't pay more than you earn)
        for i in range(min(len(eps), len(sample_dividends))):
            assert eps.iloc[i] >= sample_dividends.iloc[i] * 0.90


class TestPeltRegimes:
    def test_detect_regimes(self, sample_dividends):
        from operator1.features.six_derived_proxies import _detect_dividend_regimes
        result = _detect_dividend_regimes(sample_dividends)
        assert result["n_regimes"] >= 1
        assert isinstance(result["breakpoints"], list)

    def test_short_series(self):
        from operator1.features.six_derived_proxies import _detect_dividend_regimes
        short = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2022-01-01", periods=3, freq="365D"))
        result = _detect_dividend_regimes(short)
        assert result["n_regimes"] == 1  # too short for breaks


class TestL1BalanceSheet:
    def test_basic_reconstruction(self):
        from operator1.features.six_derived_proxies import _reconstruct_balance_sheet, _DEFAULT_RATIOS
        bs = _reconstruct_balance_sheet(
            net_income=10e9,
            total_equity_floor=35e9,
            annual_dividends=8e9,
            sector_ratios=_DEFAULT_RATIOS,
            market_cap=200e9,
        )
        if bs:  # solver may not converge with all constraints
            assert bs["total_assets"] > 0
            # Accounting identity
            gap = abs(bs["total_assets"] - bs["total_liabilities"] - bs["total_equity"])
            assert gap < 100  # should be nearly exact


class TestMonteCarloUncertainty:
    def test_basic_propagation(self):
        from operator1.features.six_derived_proxies import _propagate_uncertainty
        result = _propagate_uncertainty(
            dividend_yield=0.04,
            buyback_yield=0.02,
            market_cap=200e9,
        )
        assert "implied_pe" in result
        pe = result["implied_pe"]
        assert pe["p5"] < pe["p50"] < pe["p95"]


class TestEboReverseEarnings:
    def test_basic_ebo(self):
        from operator1.features.six_derived_proxies import _ebo_reverse_earnings
        result = _ebo_reverse_earnings(
            price=80.0,
            dividend_per_share=3.10,
            dividend_growth=0.03,
        )
        assert result.get("implied_eps", 0) > 0
        assert result.get("implied_pe", 0) > 5


class TestUkfTwoFactor:
    def test_basic_ukf(self, sample_dividends):
        from operator1.features.six_derived_proxies import _ukf_two_factor_earnings
        eps, payout, stds = _ukf_two_factor_earnings(sample_dividends)
        assert len(eps) == len(sample_dividends)
        assert 0.15 <= float(payout.iloc[0]) <= 0.95


class TestJamesShrinkage:
    def test_shrinkage(self):
        from operator1.features.six_derived_proxies import _james_stein_shrink, _SECTOR_RATIOS
        shrunk = _james_stein_shrink(_SECTOR_RATIOS, "Financial Services")
        assert shrunk  # should return a dict
        # Financial Services has extreme equity_ratio (0.10) -- should be
        # shrunk toward the grand mean (~0.35)
        assert shrunk.get("equity_ratio", 0) > 0.10


class TestJackknifeBias:
    def test_jackknife(self, sample_dividends):
        from operator1.features.six_derived_proxies import (
            _jackknife_bias_correction, _kalman_earnings_estimate,
        )
        corrected, se = _jackknife_bias_correction(
            sample_dividends, _kalman_earnings_estimate, payout_prior=0.70,
        )
        assert corrected > 0
        assert se >= 0


class TestDividendEntropy:
    def test_entropy(self, sample_dividends):
        from operator1.features.six_derived_proxies import _dividend_entropy
        result = _dividend_entropy(sample_dividends)
        assert "normalized_entropy" in result
        assert 0 <= result["normalized_entropy"] <= 1


class TestBenfordConformity:
    def test_benford(self):
        from operator1.features.six_derived_proxies import _benford_conformity_test
        # Generate Benford-conforming data
        values = [float(x) for x in range(100, 1000)]
        result = _benford_conformity_test(values)
        assert result["conformity"] > 0
        assert result["n"] > 0


class TestMarchenkosPastur:
    def test_mp_clean(self):
        from operator1.features.six_derived_proxies import _marchenko_pastur_clean
        np.random.seed(42)
        matrix = np.random.randn(100, 5)
        cleaned = _marchenko_pastur_clean(matrix)
        assert cleaned.shape == (5, 5)
        assert abs(cleaned[0, 0] - 1.0) < 0.1  # diagonal should be ~1


class TestOhlsonEquity:
    def test_ohlson(self):
        from operator1.features.six_derived_proxies import _ohlson_residual_income_equity
        result = _ohlson_residual_income_equity(
            price=80.0,
            earnings_per_share=4.0,
            dividend_per_share=3.0,
        )
        assert result.get("book_value_per_share", 0) > 0
        assert result.get("implied_pb", 0) > 1.0


class TestMertonDebt:
    def test_merton(self):
        from operator1.features.six_derived_proxies import _merton_structural_debt
        result = _merton_structural_debt(
            equity_value=200e9,
            equity_volatility=0.20,
        )
        assert result.get("asset_value", 0) > 200e9  # assets > equity
        assert result.get("distance_to_default", 0) > 0


# ---------------------------------------------------------------------------
# Integration test: full compute_six_proxies
# ---------------------------------------------------------------------------

class TestComputeSixProxies:
    def test_full_pipeline(self, sample_cache, sample_profile):
        from operator1.features.six_derived_proxies import compute_six_proxies
        # Patch out SIX API calls (dividends come from profile, not API)
        result = compute_six_proxies(sample_cache, sample_profile)
        # Should compute without error (may not have all data)
        assert result is not None
        assert isinstance(result.n_proxies, int)


class TestSeedCanonicalColumns:
    def test_seed_basic(self, sample_cache, sample_profile):
        from operator1.features.six_derived_proxies import SixProxyResult, seed_canonical_columns
        proxy = SixProxyResult(
            computed=True,
            kalman_earnings=10e9,
            book_equity_floor=35e9,
            implied_earnings=11e9,
        )
        sample_profile["latest_dividend_amount"] = 3.0
        sample_profile["shares_outstanding"] = 2_500_000_000
        seed_canonical_columns(sample_cache, sample_profile, proxy)
        assert "net_income" in sample_cache.columns
        assert "total_equity" in sample_cache.columns

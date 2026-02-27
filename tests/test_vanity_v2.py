"""Tests for vanity v2 scoring system.

Covers all five proxy components, the composite score, trend detection,
label assignment, legacy backward compatibility, and integration with
profile_builder and hierarchy_weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from operator1.analysis.vanity import (
    _rnd_growth_mismatch,
    _sga_bloat_v2,
    _capital_misallocation,
    _competitive_decay,
    _sentiment_reality_gap,
    _assign_label,
    _compute_trend,
    _normalize_score,
    compute_vanity_percentage,
    compute_vanity_score,
    _DEFAULT_LABELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cache(days: int = 100, **overrides) -> pd.DataFrame:
    """Build a minimal daily cache DataFrame for testing."""
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    rng = np.random.RandomState(42)

    base = {
        "revenue": 1_000_000 + rng.randn(days).cumsum() * 10_000,
        "rd_expenses": rng.uniform(50_000, 200_000, days),
        "sga_expenses": rng.uniform(200_000, 400_000, days),
        "operating_margin": rng.uniform(0.05, 0.25, days),
        "total_debt": 500_000 + rng.randn(days).cumsum() * 5_000,
        "fh_liquidity_score": rng.uniform(20, 80, days),
        "fh_solvency_score": rng.uniform(20, 80, days),
        "dividends_paid": rng.uniform(0, 50_000, days),
        "net_debt_to_ebitda": rng.uniform(1, 6, days),
        "competitive_pressure_index": rng.uniform(0, 1, days),
        "peer_composite_rank": rng.uniform(30, 70, days),
        "news_sentiment_score": rng.uniform(-1, 1, days),
        "fh_composite_score": rng.uniform(30, 70, days),
        "company_survival_mode_flag": np.zeros(days, dtype=int),
        "vanity_percentage": rng.uniform(0, 10, days),
    }
    base.update(overrides)
    return pd.DataFrame(base, index=idx)


# ---------------------------------------------------------------------------
# Component tests
# ---------------------------------------------------------------------------

class TestRnDGrowthMismatch:
    """Tests for _rnd_growth_mismatch component."""

    def test_returns_series(self):
        cache = _make_cache()
        result = _rnd_growth_mismatch(cache)
        assert isinstance(result, pd.Series)
        assert len(result) == len(cache)

    def test_score_range(self):
        cache = _make_cache()
        result = _rnd_growth_mismatch(cache)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_high_rnd_negative_growth_scores_high(self):
        """High R&D with negative revenue growth should score high."""
        days = 50
        cache = _make_cache(
            days=days,
            rd_expenses=np.full(days, 250_000),
            revenue=np.linspace(1_000_000, 800_000, days),  # declining
        )
        result = _rnd_growth_mismatch(cache)
        # Last values should have non-zero scores
        valid = result.dropna()
        assert valid.iloc[-1] > 0

    def test_low_rnd_returns_zero(self):
        """Low R&D should not trigger mismatch."""
        days = 50
        cache = _make_cache(
            days=days,
            rd_expenses=np.full(days, 10_000),
            revenue=np.full(days, 1_000_000),
        )
        result = _rnd_growth_mismatch(cache)
        valid = result.dropna()
        assert (valid == 0).all() or valid.mean() < 5

    def test_missing_inputs_return_nan(self):
        cache = _make_cache(days=10)
        cache["rd_expenses"] = np.nan
        result = _rnd_growth_mismatch(cache)
        assert result.isna().all()


class TestSGABloatV2:
    """Tests for _sga_bloat_v2 component."""

    def test_returns_series(self):
        cache = _make_cache()
        result = _sga_bloat_v2(cache)
        assert isinstance(result, pd.Series)

    def test_score_range(self):
        cache = _make_cache()
        result = _sga_bloat_v2(cache)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_high_sga_scores_high(self):
        days = 50
        cache = _make_cache(
            days=days,
            sga_expenses=np.full(days, 600_000),
            revenue=np.full(days, 1_000_000),
            operating_margin=np.linspace(0.20, 0.05, days),
        )
        result = _sga_bloat_v2(cache)
        valid = result.dropna()
        assert valid.iloc[-1] > 0

    def test_missing_sga_returns_nan(self):
        cache = _make_cache(days=10)
        cache["sga_expenses"] = np.nan
        result = _sga_bloat_v2(cache)
        assert result.isna().all()


class TestCapitalMisallocation:
    """Tests for _capital_misallocation component."""

    def test_returns_series(self):
        cache = _make_cache()
        result = _capital_misallocation(cache)
        assert isinstance(result, pd.Series)

    def test_score_range(self):
        cache = _make_cache()
        result = _capital_misallocation(cache)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_debt_up_liquidity_down_scores_high(self):
        days = 50
        cache = _make_cache(
            days=days,
            total_debt=np.linspace(500_000, 800_000, days),
            fh_liquidity_score=np.linspace(70, 20, days),
        )
        result = _capital_misallocation(cache)
        valid = result.dropna()
        # Later values should be higher
        assert valid.iloc[-1] >= valid.iloc[0] or valid.iloc[-1] > 0

    def test_dividends_while_critical_solvency(self):
        days = 50
        cache = _make_cache(
            days=days,
            dividends_paid=np.full(days, 100_000),
            fh_solvency_score=np.full(days, 15.0),  # critical
        )
        result = _capital_misallocation(cache)
        valid = result.dropna()
        assert valid.mean() > 0


class TestCompetitiveDecay:
    """Tests for _competitive_decay component."""

    def test_returns_series(self):
        cache = _make_cache()
        result = _competitive_decay(cache)
        assert isinstance(result, pd.Series)

    def test_score_range(self):
        cache = _make_cache()
        result = _competitive_decay(cache)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_high_pressure_scores_high(self):
        days = 50
        cache = _make_cache(
            days=days,
            competitive_pressure_index=np.full(days, 0.9),
            peer_composite_rank=np.linspace(60, 30, days),
            operating_margin=np.linspace(0.20, 0.05, days),
        )
        result = _competitive_decay(cache)
        valid = result.dropna()
        assert valid.mean() > 10

    def test_missing_all_returns_nan(self):
        cache = _make_cache(days=10)
        for col in ["competitive_pressure_index", "peer_composite_rank", "operating_margin"]:
            cache[col] = np.nan
        result = _competitive_decay(cache)
        assert result.isna().all()


class TestSentimentRealityGap:
    """Tests for _sentiment_reality_gap component."""

    def test_returns_series(self):
        cache = _make_cache()
        result = _sentiment_reality_gap(cache)
        assert isinstance(result, pd.Series)

    def test_score_range(self):
        cache = _make_cache()
        result = _sentiment_reality_gap(cache)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_positive_sentiment_declining_health(self):
        days = 50
        cache = _make_cache(
            days=days,
            news_sentiment_score=np.full(days, 0.8),
            fh_composite_score=np.linspace(70, 30, days),
        )
        result = _sentiment_reality_gap(cache)
        valid = result.dropna()
        # Should have some non-zero scores
        assert valid.iloc[-1] > 0

    def test_negative_sentiment_returns_low(self):
        days = 50
        cache = _make_cache(
            days=days,
            news_sentiment_score=np.full(days, -0.5),
            fh_composite_score=np.linspace(70, 30, days),
        )
        result = _sentiment_reality_gap(cache)
        valid = result.dropna()
        # Negative sentiment shouldn't produce high vanity
        assert valid.mean() < 10


# ---------------------------------------------------------------------------
# Label and trend tests
# ---------------------------------------------------------------------------

class TestLabelAssignment:
    """Tests for _assign_label."""

    def test_disciplined(self):
        assert _assign_label(10.0, _DEFAULT_LABELS) == "Disciplined"

    def test_moderate(self):
        assert _assign_label(30.0, _DEFAULT_LABELS) == "Moderate"

    def test_wasteful(self):
        assert _assign_label(50.0, _DEFAULT_LABELS) == "Wasteful"

    def test_reckless(self):
        assert _assign_label(80.0, _DEFAULT_LABELS) == "Reckless"

    def test_nan_returns_unknown(self):
        assert _assign_label(np.nan, _DEFAULT_LABELS) == "Unknown"

    def test_boundary_20(self):
        assert _assign_label(20.0, _DEFAULT_LABELS) == "Disciplined"

    def test_just_above_20(self):
        assert _assign_label(20.1, _DEFAULT_LABELS) == "Moderate"


class TestNormalizeScore:
    """Tests for _normalize_score."""

    def test_clips_to_range(self):
        s = pd.Series([-10, 50, 150])
        result = _normalize_score(s)
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 50.0
        assert result.iloc[2] == 100.0


class TestComputeTrend:
    """Tests for _compute_trend."""

    def test_rising_trend(self):
        # Score increasing over time
        s = pd.Series(np.linspace(10, 80, 100))
        trend = _compute_trend(s, short_window=5, long_window=20)
        # Last values should be "rising"
        assert trend.iloc[-1] == "rising"

    def test_falling_trend(self):
        s = pd.Series(np.linspace(80, 10, 100))
        trend = _compute_trend(s, short_window=5, long_window=20)
        assert trend.iloc[-1] == "falling"

    def test_stable_trend(self):
        s = pd.Series(np.full(100, 40.0))
        trend = _compute_trend(s, short_window=5, long_window=20)
        assert trend.iloc[-1] == "stable"


# ---------------------------------------------------------------------------
# Composite score tests
# ---------------------------------------------------------------------------

class TestComputeVanityScore:
    """Tests for compute_vanity_score (main entry point)."""

    def test_all_columns_present(self):
        cache = _make_cache(days=100)
        result = compute_vanity_score(cache)
        expected_cols = [
            "vanity_rnd_mismatch",
            "vanity_sga_bloat_v2",
            "vanity_capital_misallocation",
            "vanity_competitive_decay",
            "vanity_sentiment_gap",
            "vanity_score",
            "vanity_score_21d",
            "vanity_trend",
            "vanity_label",
            # Legacy columns
            "vanity_percentage",
            "vanity_total_spend",
            "is_missing_vanity_percentage",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_score_range(self):
        cache = _make_cache(days=100)
        result = compute_vanity_score(cache)
        score = result["vanity_score"].dropna()
        assert (score >= 0).all()
        assert (score <= 100).all()

    def test_label_values(self):
        cache = _make_cache(days=100)
        result = compute_vanity_score(cache)
        valid_labels = {"Disciplined", "Moderate", "Wasteful", "Reckless", "Unknown"}
        labels = set(result["vanity_label"].dropna().unique())
        assert labels.issubset(valid_labels)

    def test_trend_values(self):
        cache = _make_cache(days=100)
        result = compute_vanity_score(cache)
        valid_trends = {"rising", "stable", "falling"}
        trends = set(result["vanity_trend"].dropna().unique())
        assert trends.issubset(valid_trends)

    def test_custom_config(self):
        cache = _make_cache(days=100)
        config = {
            "weights": {
                "rnd_mismatch": 0.5,
                "sga_bloat": 0.1,
                "capital_misallocation": 0.1,
                "competitive_decay": 0.2,
                "sentiment_gap": 0.1,
            },
            "labels": [(50, "OK"), (100, "Bad")],
            "trend_window_short": 10,
            "trend_window_long": 30,
        }
        result = compute_vanity_score(cache, config=config)
        assert "vanity_score" in result.columns
        valid_labels = {"OK", "Bad", "Unknown"}
        labels = set(result["vanity_label"].dropna().unique())
        assert labels.issubset(valid_labels)

    def test_empty_cache(self):
        cache = pd.DataFrame()
        result = compute_vanity_score(cache)
        assert "vanity_score" in result.columns

    def test_preserves_original_columns(self):
        cache = _make_cache(days=50)
        original_cols = set(cache.columns)
        result = compute_vanity_score(cache)
        for col in original_cols:
            assert col in result.columns


# ---------------------------------------------------------------------------
# Legacy backward compatibility
# ---------------------------------------------------------------------------

class TestLegacyVanityPercentage:
    """Tests for compute_vanity_percentage (legacy API)."""

    def test_produces_legacy_columns(self):
        cache = _make_cache(days=50)
        result = compute_vanity_percentage(cache)
        assert "vanity_percentage" in result.columns
        assert "vanity_total_spend" in result.columns
        assert "is_missing_vanity_percentage" in result.columns

    def test_vanity_percentage_range(self):
        cache = _make_cache(days=50)
        result = compute_vanity_percentage(cache)
        vp = result["vanity_percentage"].dropna()
        assert (vp >= 0).all()
        assert (vp <= 100).all()


# ---------------------------------------------------------------------------
# Integration: profile_builder
# ---------------------------------------------------------------------------

class TestProfileBuilderVanitySection:
    """Test that profile_builder handles v2 vanity data."""

    def test_v2_section_populated(self):
        from operator1.report.profile_builder import _build_vanity_section
        cache = _make_cache(days=100)
        cache = compute_vanity_score(cache)
        section = _build_vanity_section(cache)
        assert section.get("v2_available") is True
        assert "v2_score" in section
        assert "v2_label" in section
        assert "v2_trend" in section
        assert "v2_breakdown" in section

    def test_v2_breakdown_has_components(self):
        from operator1.report.profile_builder import _build_vanity_section
        cache = _make_cache(days=100)
        cache = compute_vanity_score(cache)
        section = _build_vanity_section(cache)
        breakdown = section.get("v2_breakdown", {})
        assert "vanity_rnd_mismatch" in breakdown
        assert "vanity_capital_misallocation" in breakdown

    def test_legacy_only_cache(self):
        from operator1.report.profile_builder import _build_vanity_section
        cache = _make_cache(days=50)
        cache = compute_vanity_percentage(cache)
        section = _build_vanity_section(cache)
        assert section.get("available") is True
        assert section.get("v2_available") is False

    def test_empty_cache(self):
        from operator1.report.profile_builder import _build_vanity_section
        section = _build_vanity_section(None)
        assert section.get("available") is False

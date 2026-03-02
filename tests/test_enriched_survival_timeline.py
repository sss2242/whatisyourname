"""Tests for the enriched survival timeline (Step 5.5 bridge layer).

Covers:
- Base survival timeline computation.
- Combined state mapping from (survival_mode, market_regime) pairs.
- Enriched timeline with and without HMM regime data.
- Early regime detection wrapper.
- Graceful degradation when HMM data is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from operator1.analysis.survival_timeline import (
    COMBINED_STATES,
    SURVIVAL_MODES,
    EnrichedTimelineResult,
    SurvivalTimelineResult,
    _map_combined_state,
    classify_survival_mode,
    compute_enriched_survival_timeline,
    compute_survival_timeline,
    get_mode_at_date,
    get_switch_dates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache(
    n_days: int = 100,
    company_flag_days: list[int] | None = None,
    country_flag_days: list[int] | None = None,
    protected: bool = False,
) -> pd.DataFrame:
    """Build a minimal daily cache with survival flags and market data."""
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    rng = np.random.default_rng(42)

    close = 100 + np.cumsum(rng.normal(0, 1, n_days))
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    vol = pd.Series(returns).rolling(21, min_periods=1).std().values

    df = pd.DataFrame(
        {
            "close": close,
            "return_1d": returns,
            "volatility_21d": vol,
            "current_ratio": rng.uniform(0.5, 3.0, n_days),
            "debt_to_equity_abs": rng.uniform(0.5, 4.0, n_days),
            "fcf_yield": rng.uniform(-0.05, 0.10, n_days),
            "drawdown_252d": rng.uniform(-0.50, 0.0, n_days),
        },
        index=dates,
    )

    df["company_survival_mode_flag"] = 0
    if company_flag_days:
        for d in company_flag_days:
            if d < n_days:
                df.iloc[d, df.columns.get_loc("company_survival_mode_flag")] = 1

    df["country_survival_mode_flag"] = 0
    if country_flag_days:
        for d in country_flag_days:
            if d < n_days:
                df.iloc[d, df.columns.get_loc("country_survival_mode_flag")] = 1

    df["country_protected_flag"] = 1 if protected else 0

    return df


# ---------------------------------------------------------------------------
# Tests: classify_survival_mode (unit)
# ---------------------------------------------------------------------------

class TestClassifySurvivalMode:
    def test_normal(self):
        assert classify_survival_mode(0, 0, 0) == "normal"

    def test_company_only(self):
        assert classify_survival_mode(1, 0, 0) == "company_only"

    def test_country_protected(self):
        assert classify_survival_mode(0, 1, 1) == "country_protected"

    def test_country_exposed(self):
        assert classify_survival_mode(0, 1, 0) == "country_exposed"

    def test_both_unprotected(self):
        assert classify_survival_mode(1, 1, 0) == "both_unprotected"

    def test_both_protected(self):
        assert classify_survival_mode(1, 1, 1) == "both_protected"


# ---------------------------------------------------------------------------
# Tests: _map_combined_state
# ---------------------------------------------------------------------------

class TestCombinedStateMapping:
    def test_all_survival_modes_covered(self):
        """Every survival mode has an entry for every known market regime."""
        market_regimes = ["bull", "bear", "high_vol", "low_vol", "unknown"]
        for sm in SURVIVAL_MODES:
            for mr in market_regimes:
                state, intensity = _map_combined_state(sm, mr)
                assert isinstance(state, str), f"No state for ({sm}, {mr})"
                assert 0.0 <= intensity <= 1.0, f"Bad intensity for ({sm}, {mr})"

    def test_normal_bull_is_lowest_intensity(self):
        state, intensity = _map_combined_state("normal", "bull")
        assert state == "stable_growth"
        assert intensity == 0.0

    def test_extreme_crisis(self):
        state, intensity = _map_combined_state("both_unprotected", "bear")
        assert state == "extreme_crisis"
        assert intensity == 1.0

    def test_unknown_survival_mode_fallback(self):
        state, intensity = _map_combined_state("made_up_mode", "bull")
        assert state == "unknown"
        assert intensity == 0.5

    def test_intensity_ordering(self):
        """More severe states should have higher intensity."""
        _, i_normal = _map_combined_state("normal", "bull")
        _, i_company = _map_combined_state("company_only", "bear")
        _, i_crisis = _map_combined_state("both_unprotected", "bear")
        assert i_normal < i_company < i_crisis


# ---------------------------------------------------------------------------
# Tests: compute_survival_timeline (base)
# ---------------------------------------------------------------------------

class TestBaseSurvivalTimeline:
    def test_normal_cache(self):
        cache = _make_cache(50)
        result = compute_survival_timeline(cache)
        assert result.fitted
        assert len(result.timeline) == 50
        assert "survival_mode" in result.timeline.columns
        assert result.mode_distribution.get("normal", 0) > 0

    def test_with_company_flags(self):
        cache = _make_cache(50, company_flag_days=[10, 11, 12])
        result = compute_survival_timeline(cache)
        assert result.fitted
        assert result.n_switches > 0
        modes = result.timeline["survival_mode"]
        assert "company_only" in modes.values

    def test_empty_cache(self):
        cache = pd.DataFrame()
        result = compute_survival_timeline(cache)
        assert not result.fitted
        assert result.error is not None

    def test_switch_points(self):
        cache = _make_cache(50, company_flag_days=list(range(20, 30)))
        result = compute_survival_timeline(cache)
        switch_dates = get_switch_dates(result)
        assert len(switch_dates) >= 2  # at least enter and exit

    def test_stability_score_range(self):
        cache = _make_cache(50)
        result = compute_survival_timeline(cache)
        stability = result.timeline["stability_score_21d"]
        assert stability.min() >= 0.0
        assert stability.max() <= 1.0


# ---------------------------------------------------------------------------
# Tests: compute_enriched_survival_timeline
# ---------------------------------------------------------------------------

class TestEnrichedSurvivalTimeline:
    def test_without_regime_data(self):
        """Enriched timeline should work without HMM regime data."""
        cache = _make_cache(80)
        result = compute_enriched_survival_timeline(cache)
        assert result.fitted
        assert not result.regime_available
        assert "regime_state" in result.timeline.columns
        assert "survival_intensity" in result.timeline.columns
        assert "regime_confidence" in result.timeline.columns
        assert "regime_transition_prob" in result.timeline.columns
        # Without regime, market_regime should be "unknown".
        assert (result.timeline["market_regime"] == "unknown").all()
        # Intensity should be consistent with normal mode.
        assert result.mean_intensity >= 0.0

    def test_with_regime_data(self):
        """Enriched timeline should incorporate HMM regime labels."""
        cache = _make_cache(80)
        # Simulate HMM regime labels.
        regime_labels = pd.Series(
            np.random.default_rng(42).choice(
                ["bull", "bear", "high_vol", "low_vol"], size=80
            ),
            index=cache.index,
        )
        regime_confidence = pd.Series(
            np.random.default_rng(42).uniform(0.6, 0.99, size=80),
            index=cache.index,
        )
        result = compute_enriched_survival_timeline(
            cache,
            regime_labels=regime_labels,
            regime_confidence=regime_confidence,
        )
        assert result.fitted
        assert result.regime_available
        # market_regime should not be all "unknown" anymore.
        assert not (result.timeline["market_regime"] == "unknown").all()
        # Combined states should include more variety than just stable_growth.
        unique_states = result.timeline["regime_state"].nunique()
        assert unique_states >= 2

    def test_with_company_survival_and_regime(self):
        """When company is in survival AND market is bearish, intensity should be high."""
        cache = _make_cache(80, company_flag_days=list(range(30, 50)))
        regime_labels = pd.Series("bear", index=cache.index)
        result = compute_enriched_survival_timeline(
            cache, regime_labels=regime_labels,
        )
        assert result.fitted
        # Days 30-49 are company_only + bear = company_distress_severe (0.70).
        survival_days = result.timeline.iloc[30:50]
        assert (survival_days["survival_intensity"] >= 0.60).all()

    def test_regime_switch_computation(self):
        """regime_switch should flag days where combined state changes."""
        cache = _make_cache(80, company_flag_days=list(range(40, 80)))
        result = compute_enriched_survival_timeline(cache)
        assert result.fitted
        switches = result.timeline["regime_switch"]
        # There should be at least one switch at the boundary.
        assert switches.sum() >= 1

    def test_empty_cache_graceful(self):
        """Empty cache should not crash, just return unfitted result."""
        cache = pd.DataFrame()
        result = compute_enriched_survival_timeline(cache)
        assert not result.fitted
        assert result.error is not None

    def test_combined_state_distribution(self):
        """Distribution values should sum to ~1.0."""
        cache = _make_cache(100)
        result = compute_enriched_survival_timeline(cache)
        assert result.fitted
        total = sum(result.combined_state_distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_all_combined_states_are_known(self):
        """Every combined state produced should be in COMBINED_STATES or 'unknown'."""
        cache = _make_cache(
            100,
            company_flag_days=list(range(20, 40)),
            country_flag_days=list(range(50, 70)),
        )
        regime_labels = pd.Series(
            np.random.default_rng(99).choice(
                ["bull", "bear", "high_vol", "low_vol"], size=100
            ),
            index=cache.index,
        )
        result = compute_enriched_survival_timeline(
            cache, regime_labels=regime_labels,
        )
        assert result.fitted
        unique_states = set(result.timeline["regime_state"].unique())
        allowed = set(COMBINED_STATES) | {"unknown"}
        assert unique_states.issubset(allowed), (
            f"Unexpected states: {unique_states - allowed}"
        )


# ---------------------------------------------------------------------------
# Tests: run_early_regime_detection
# ---------------------------------------------------------------------------

class TestEarlyRegimeDetection:
    def test_basic_run(self):
        """Early regime detection should run and return labels + confidence."""
        from operator1.models.regime_detector import run_early_regime_detection

        cache = _make_cache(120)
        cache, early = run_early_regime_detection(cache)

        assert early.regime_labels is not None
        assert early.regime_confidence is not None
        assert len(early.regime_labels) == len(cache)
        assert len(early.regime_confidence) == len(cache)

        # Regime columns should be added to cache.
        assert "regime_hmm" in cache.columns
        assert "regime_label" in cache.columns

    def test_insufficient_data_graceful(self):
        """With too few observations, should still return safe defaults."""
        from operator1.models.regime_detector import run_early_regime_detection

        cache = _make_cache(10)  # too few for HMM (needs 60)
        cache, early = run_early_regime_detection(cache)

        # Should have defaults even if HMM didn't fit.
        assert early.regime_labels is not None
        assert early.regime_confidence is not None

    def test_integration_with_enriched_timeline(self):
        """Full pipeline: early regime -> enriched timeline."""
        from operator1.models.regime_detector import run_early_regime_detection

        cache = _make_cache(150, company_flag_days=list(range(60, 90)))
        cache, early = run_early_regime_detection(cache)

        result = compute_enriched_survival_timeline(
            cache,
            regime_labels=early.regime_labels,
            regime_confidence=early.regime_confidence,
        )
        assert result.fitted
        assert "regime_state" in result.timeline.columns
        assert "survival_intensity" in result.timeline.columns
        # With 150 days of data, HMM should have fitted.
        if early.fitted and early.detector and early.detector.result.hmm_fitted:
            assert result.regime_available

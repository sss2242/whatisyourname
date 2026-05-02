"""Tests for the pipeline quality upgrades.

1. Profile schema validation
2. Historical conflict time-varying
3. Cache column namespacing
4. Estimation-conflict integration (placeholder)

Note: Parallel executor tests removed -- parallel_executor module was
replaced by inline ThreadPoolExecutor usage in the estimator (Fix 5).
"""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Upgrade 2: Profile Schema Validation
# ---------------------------------------------------------------------------

class TestProfileSchema:
    """Test profile schema validation."""

    def test_valid_profile(self):
        """Test that a complete profile passes validation."""
        from operator1.report.profile_schema import validate_profile, REQUIRED_PROFILE_KEYS

        profile = {key: {"available": True} for key in REQUIRED_PROFILE_KEYS}
        profile["meta"] = {"generated_at": "2025-01-01"}
        # Schema also checks 'available' flag on these optional sections
        for opt_key in ("predicted_regime_shifts", "model_diagnostics", "supply_chain_stress"):
            profile[opt_key] = {"available": False}

        issues = validate_profile(profile)
        assert len(issues) == 0

    def test_missing_required_key(self):
        """Test that missing keys are caught."""
        from operator1.report.profile_schema import validate_profile, REQUIRED_PROFILE_KEYS

        profile = {key: {"available": True} for key in REQUIRED_PROFILE_KEYS}
        del profile["conflict_risk"]

        issues = validate_profile(profile)
        assert len(issues) >= 1
        assert any("conflict_risk" in i for i in issues)

    def test_none_value_caught(self):
        """Test that None values are caught."""
        from operator1.report.profile_schema import validate_profile, REQUIRED_PROFILE_KEYS

        profile = {key: {"available": True} for key in REQUIRED_PROFILE_KEYS}
        profile["meta"] = {"generated_at": "2025-01-01"}
        profile["survival"] = None

        issues = validate_profile(profile)
        assert any("None" in i and "survival" in i for i in issues)

    def test_missing_available_flag(self):
        """Test that optional sections missing 'available' flag are caught."""
        from operator1.report.profile_schema import validate_profile, REQUIRED_PROFILE_KEYS

        profile = {key: {"available": True} for key in REQUIRED_PROFILE_KEYS}
        profile["meta"] = {"generated_at": "2025-01-01"}
        profile["conflict_risk"] = {"intensity": 0.5}  # no 'available' key

        issues = validate_profile(profile)
        assert any("available" in i and "conflict_risk" in i for i in issues)

    def test_validate_strict_raises(self):
        """Test that strict validation raises ValueError."""
        from operator1.report.profile_schema import validate_profile_strict

        with pytest.raises(ValueError, match="Profile validation failed"):
            validate_profile_strict({})


# ---------------------------------------------------------------------------
# Upgrade 3: Historical Conflict Time-Varying
# ---------------------------------------------------------------------------

class TestHistoricalConflictTimeVarying:
    """Test time-varying conflict flags."""

    def test_ukraine_pre_war_peaceful(self):
        """Ukraine cache should show flag=0 before 2022-02-24."""
        from operator1.features.conflict_risk import (
            assess_conflict_risk, inject_conflict_risk_into_cache,
        )

        # Cache spanning 2021-2023 (before and after war)
        dates = pd.bdate_range("2021-06-01", "2023-06-01")
        cache = pd.DataFrame({"close": np.linspace(100, 80, len(dates))}, index=dates)

        result = assess_conflict_risk("UA", skip_ucdp=True, skip_gdelt=True)
        cache = inject_conflict_risk_into_cache(cache, result)

        war_start = pd.Timestamp("2022-02-24")
        pre_war = cache.loc[cache.index < war_start]
        post_war = cache.loc[cache.index >= war_start]

        # Before war: should be peaceful
        assert (pre_war["country_conflict_flag"] == 0).all()
        assert (pre_war["conflict_intensity_score"] == 0).all()

        # After war: should be flagged
        assert (post_war["country_conflict_flag"] == 1).all()

    def test_peaceful_country_unchanged(self):
        """US cache should have flag=0 for all dates (no time-varying change)."""
        from operator1.features.conflict_risk import (
            assess_conflict_risk, inject_conflict_risk_into_cache,
        )

        dates = pd.bdate_range("2021-01-01", "2025-01-01")
        cache = pd.DataFrame({"close": [100] * len(dates)}, index=dates)

        result = assess_conflict_risk("US", skip_ucdp=True, skip_gdelt=True)
        cache = inject_conflict_risk_into_cache(cache, result)

        assert (cache["country_conflict_flag"] == 0).all()

    def test_recent_cache_all_flagged(self):
        """If cache is entirely after the conflict start, all days should be flagged."""
        from operator1.features.conflict_risk import (
            assess_conflict_risk, inject_conflict_risk_into_cache,
        )

        # Cache starting after Ukraine war
        dates = pd.bdate_range("2023-01-01", "2025-01-01")
        cache = pd.DataFrame({"close": [100] * len(dates)}, index=dates)

        result = assess_conflict_risk("UA", skip_ucdp=True, skip_gdelt=True)
        cache = inject_conflict_risk_into_cache(cache, result)

        # All dates after 2022-02-24, so all flagged
        assert (cache["country_conflict_flag"] == 1).all()


# ---------------------------------------------------------------------------
# Upgrade 4: Cache Column Namespacing
# ---------------------------------------------------------------------------

class TestCacheColumnNamespacing:
    """Test cache column namespace validation."""

    def test_known_columns_pass(self):
        """Test that registered columns are accepted."""
        from operator1.steps.cache_builder import validate_cache_columns

        cache = pd.DataFrame({
            "close": [100],
            "revenue": [1000],
            "fh_composite_score": [75],
            "regime_label": ["bull"],
            "country_conflict_flag": [0],
            "is_missing_revenue": [0],
        })

        unknown = validate_cache_columns(cache)
        assert len(unknown) == 0

    def test_unknown_columns_detected(self):
        """Test that unregistered columns are flagged."""
        from operator1.steps.cache_builder import validate_cache_columns

        cache = pd.DataFrame({
            "close": [100],
            "mystery_column": [42],
            "xyz_unknown_prefix": [0],
        })

        unknown = validate_cache_columns(cache)
        assert "mystery_column" in unknown
        assert "xyz_unknown_prefix" in unknown

    def test_derived_columns_accepted(self):
        """Test that derived variable columns are accepted."""
        from operator1.steps.cache_builder import validate_cache_columns

        cache = pd.DataFrame({
            "return_1d": [0.01],
            "volatility_21d": [0.15],
            "current_ratio": [1.5],
            "debt_to_equity_abs": [0.8],
        })

        unknown = validate_cache_columns(cache)
        assert len(unknown) == 0

    def test_prefix_columns_accepted(self):
        """Test that columns matching registered prefixes are accepted."""
        from operator1.steps.cache_builder import validate_cache_columns

        cache = pd.DataFrame({
            "is_missing_net_income": [0],
            "invalid_math_pe_ratio": [0],
            "fh_liquidity_score": [80],
            "peer_return_rank": [0.75],
            "sentiment_momentum_21d": [0.3],
            "vanity_score": [60],
            "competitors_avg_return_1d": [0.005],
        })

        unknown = validate_cache_columns(cache)
        assert len(unknown) == 0


# ---------------------------------------------------------------------------
# Upgrade 5: Estimation-Conflict Integration (placeholder)
# ---------------------------------------------------------------------------

class TestEstimationConflictIntegration:
    """Placeholder tests for estimation-conflict integration.

    The full integration requires modifying the estimator's missingness
    classifier, which is a larger change. These tests verify the
    foundation is in place.
    """

    def test_conflict_columns_available_for_estimator(self):
        """Verify conflict columns exist in cache before estimation runs."""
        from operator1.features.conflict_risk import (
            assess_conflict_risk, inject_conflict_risk_into_cache,
        )

        dates = pd.bdate_range("2025-01-01", periods=50)
        cache = pd.DataFrame({
            "close": [100] * 50,
            "revenue": [np.nan] * 20 + [1000] * 30,
        }, index=dates)

        result = assess_conflict_risk("UA", skip_ucdp=True, skip_gdelt=True)
        cache = inject_conflict_risk_into_cache(cache, result)

        # Estimator can now see conflict columns
        assert "country_conflict_flag" in cache.columns
        assert "conflict_intensity_score" in cache.columns

        # In a conflict zone, revenue missingness could be MNAR
        conflict_period = cache["country_conflict_flag"] == 1
        missing_during_conflict = cache.loc[conflict_period, "revenue"].isna().sum()
        # This is just verifying the data is available -- actual MNAR
        # classification will be in the estimator upgrade
        assert isinstance(missing_during_conflict, (int, np.integer))

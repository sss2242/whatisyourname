"""Tests for the war/conflict risk assessment module.

Tests static list checks, UCDP event analysis, intensity scoring,
and cache injection with mocked API responses.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from operator1.features.conflict_risk import (
    ACTIVE_WAR_COUNTRIES,
    FRAGILE_CONFLICT_STATES,
    SANCTIONED_COUNTRIES,
    ConflictRiskResult,
    assess_conflict_risk,
    inject_conflict_risk_into_cache,
    _analyze_ucdp_events,
    _compute_intensity_score,
)


# ---------------------------------------------------------------------------
# Static list tests
# ---------------------------------------------------------------------------

class TestStaticLists:
    """Test the hardcoded conflict/sanctions/fragile state lists."""

    def test_ukraine_is_active_war(self):
        assert "UA" in ACTIVE_WAR_COUNTRIES

    def test_russia_is_sanctioned(self):
        assert "RU" in SANCTIONED_COUNTRIES

    def test_iran_is_sanctioned(self):
        assert "IR" in SANCTIONED_COUNTRIES

    def test_afghanistan_is_fragile(self):
        assert "AF" in FRAGILE_CONFLICT_STATES

    def test_us_not_in_any_list(self):
        assert "US" not in ACTIVE_WAR_COUNTRIES
        assert "US" not in SANCTIONED_COUNTRIES
        assert "US" not in FRAGILE_CONFLICT_STATES

    def test_japan_not_in_any_list(self):
        assert "JP" not in ACTIVE_WAR_COUNTRIES
        assert "JP" not in SANCTIONED_COUNTRIES
        assert "JP" not in FRAGILE_CONFLICT_STATES


# ---------------------------------------------------------------------------
# UCDP event analysis tests
# ---------------------------------------------------------------------------

class TestUCDPAnalysis:
    """Test UCDP event analysis logic."""

    def test_empty_events(self):
        result = _analyze_ucdp_events([])
        assert result["events_30d"] == 0
        assert result["fatalities_30d"] == 0
        assert result["conflict_type"] == "none"
        assert result["trend"] == "stable"

    def test_recent_events(self):
        today = date.today()
        events = [
            {
                "date_start": (today - timedelta(days=5)).isoformat(),
                "best": 10,
                "type_of_violence": 1,
            },
            {
                "date_start": (today - timedelta(days=15)).isoformat(),
                "best": 5,
                "type_of_violence": 1,
            },
            {
                "date_start": (today - timedelta(days=60)).isoformat(),
                "best": 20,
                "type_of_violence": 2,
            },
        ]
        result = _analyze_ucdp_events(events)

        assert result["events_30d"] == 2
        assert result["events_90d"] == 3
        assert result["fatalities_30d"] == 15
        assert result["fatalities_90d"] == 35
        # Type 1 (state-based) present, fewer than 50 events -> civil_war
        assert result["conflict_type"] == "civil_war"

    def test_non_state_conflict(self):
        today = date.today()
        events = [
            {
                "date_start": (today - timedelta(days=10)).isoformat(),
                "best": 3,
                "type_of_violence": 2,  # non-state
            },
        ]
        result = _analyze_ucdp_events(events)
        assert result["conflict_type"] == "low_intensity"

    def test_escalation_trend(self):
        today = date.today()
        # 20 events in last 90 days, 5 in previous 90 days -> escalating
        events = []
        for i in range(20):
            events.append({
                "date_start": (today - timedelta(days=i * 3)).isoformat(),
                "best": 1,
                "type_of_violence": 1,
            })
        for i in range(5):
            events.append({
                "date_start": (today - timedelta(days=100 + i * 10)).isoformat(),
                "best": 1,
                "type_of_violence": 1,
            })
        result = _analyze_ucdp_events(events)
        assert result["trend"] == "escalating"


# ---------------------------------------------------------------------------
# Intensity score tests
# ---------------------------------------------------------------------------

class TestIntensityScore:
    """Test the conflict intensity scoring function."""

    def test_peaceful_country(self):
        score = _compute_intensity_score(
            events_30d=0, events_90d=0, fatalities_30d=0,
            is_active_war=False, is_sanctioned=False, is_fragile=False,
            news_mentions=0,
        )
        assert score == 0.0

    def test_active_war_country(self):
        score = _compute_intensity_score(
            events_30d=100, events_90d=300, fatalities_30d=500,
            is_active_war=True, is_sanctioned=True, is_fragile=True,
            news_mentions=50,
        )
        assert score > 0.7
        assert score <= 1.0

    def test_sanctioned_only(self):
        score = _compute_intensity_score(
            events_30d=0, events_90d=0, fatalities_30d=0,
            is_active_war=False, is_sanctioned=True, is_fragile=False,
            news_mentions=5,
        )
        assert 0.0 < score < 0.3  # sanctions alone = moderate score

    def test_fragile_with_events(self):
        score = _compute_intensity_score(
            events_30d=15, events_90d=40, fatalities_30d=20,
            is_active_war=False, is_sanctioned=False, is_fragile=True,
            news_mentions=10,
        )
        assert 0.2 < score < 0.7

    def test_score_bounded(self):
        """Verify score is always in [0, 1]."""
        # Extreme values
        score = _compute_intensity_score(
            events_30d=10000, events_90d=50000, fatalities_30d=100000,
            is_active_war=True, is_sanctioned=True, is_fragile=True,
            news_mentions=1000,
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Main assessment function tests
# ---------------------------------------------------------------------------

class TestAssessConflictRisk:
    """Test the main assess_conflict_risk() function."""

    def test_peaceful_country_static_only(self):
        """Test assessment of a peaceful country using static lists only."""
        result = assess_conflict_risk("US", skip_ucdp=True, skip_gdelt=True)

        assert isinstance(result, ConflictRiskResult)
        assert result.country_iso2 == "US"
        assert not result.country_conflict_flag
        assert not result.sanctions_flag
        assert not result.fragile_state_flag
        assert result.conflict_type == "none"
        assert result.conflict_intensity_score == 0.0
        assert result.assessment_date == date.today().isoformat()
        assert "static_lists" in result.data_sources_used

    def test_war_country_static_only(self):
        """Test assessment of an active war country using static lists only."""
        result = assess_conflict_risk("UA", skip_ucdp=True, skip_gdelt=True)

        assert result.country_conflict_flag
        assert result.conflict_type == "interstate_war"
        assert result.conflict_intensity_score > 0.0

    def test_sanctioned_country_static_only(self):
        """Test assessment of a sanctioned country."""
        result = assess_conflict_risk("RU", skip_ucdp=True, skip_gdelt=True)

        assert result.sanctions_flag
        assert result.country_conflict_flag  # RU is both sanctioned and active war
        assert result.conflict_intensity_score > 0.0

    def test_fragile_state_static_only(self):
        """Test assessment of a fragile state."""
        result = assess_conflict_risk("HT", skip_ucdp=True, skip_gdelt=True)

        assert result.fragile_state_flag
        assert result.conflict_intensity_score > 0.0

    def test_company_flag_in_conflict_country(self):
        """Test that company flag is set when country is in conflict."""
        result = assess_conflict_risk("UA", company_name="MHP SE", skip_ucdp=True, skip_gdelt=True)

        assert result.company_conflict_flag

    def test_company_flag_in_peaceful_country(self):
        """Test that company flag is not set in peaceful country."""
        result = assess_conflict_risk("US", company_name="Apple Inc", skip_ucdp=True, skip_gdelt=True)

        assert not result.company_conflict_flag

    @patch("operator1.features.conflict_risk._fetch_ucdp_events")
    def test_with_ucdp_events(self, mock_ucdp):
        """Test assessment with mocked UCDP events."""
        today = date.today()
        mock_ucdp.return_value = [
            {
                "date_start": (today - timedelta(days=5)).isoformat(),
                "best": 25,
                "type_of_violence": 1,
            },
            {
                "date_start": (today - timedelta(days=20)).isoformat(),
                "best": 10,
                "type_of_violence": 1,
            },
        ]

        result = assess_conflict_risk("UA", skip_gdelt=True)

        assert result.recent_events_30d == 2
        assert result.recent_fatalities_30d == 35
        assert "ucdp_ged" in result.data_sources_used

    @patch("operator1.features.conflict_risk._fetch_gdelt_conflict_news")
    def test_with_gdelt_news(self, mock_gdelt):
        """Test assessment with mocked GDELT news data."""
        mock_gdelt.return_value = {
            "mentions_count": 45,
            "avg_tone": -5.2,
        }

        result = assess_conflict_risk("UA", skip_ucdp=True)

        assert result.news_conflict_mentions_7d == 45
        assert result.news_conflict_tone == -5.2
        assert "gdelt" in result.data_sources_used

    def test_to_dict(self):
        """Test serialization to dict."""
        result = assess_conflict_risk("JP", skip_ucdp=True, skip_gdelt=True)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["country_iso2"] == "JP"
        assert isinstance(d["conflict_intensity_score"], float)
        assert isinstance(d["data_sources_used"], list)


# ---------------------------------------------------------------------------
# Cache injection tests
# ---------------------------------------------------------------------------

class TestCacheInjection:
    """Test injection of conflict risk columns into the daily cache."""

    def test_inject_peaceful(self):
        """Test injection for a peaceful country."""
        cache = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0]},
            index=pd.bdate_range("2025-01-01", periods=3),
        )
        result = ConflictRiskResult(
            country_iso2="US",
            country_conflict_flag=False,
            company_conflict_flag=False,
            conflict_intensity_score=0.0,
            sanctions_flag=False,
            fragile_state_flag=False,
            conflict_type="none",
        )

        cache = inject_conflict_risk_into_cache(cache, result)

        assert "country_conflict_flag" in cache.columns
        assert "conflict_intensity_score" in cache.columns
        assert "sanctions_flag" in cache.columns
        assert cache["country_conflict_flag"].iloc[0] == 0
        assert cache["conflict_intensity_score"].iloc[0] == 0.0

    def test_inject_conflict(self):
        """Test injection for a conflict-affected country."""
        cache = pd.DataFrame(
            {"close": [50.0, 45.0, 40.0]},
            index=pd.bdate_range("2025-01-01", periods=3),
        )
        result = ConflictRiskResult(
            country_iso2="UA",
            country_conflict_flag=True,
            company_conflict_flag=True,
            conflict_intensity_score=0.85,
            sanctions_flag=False,
            fragile_state_flag=False,
            conflict_type="interstate_war",
        )

        cache = inject_conflict_risk_into_cache(cache, result)

        assert cache["country_conflict_flag"].iloc[0] == 1
        assert cache["company_conflict_flag"].iloc[0] == 1
        assert cache["conflict_intensity_score"].iloc[0] == 0.85
        assert cache["conflict_type"].iloc[0] == "interstate_war"

"""Integration tests -- Layer 2: Analysis Modules.

Validates survival mode detection, financial health scoring, and
hierarchy weights produce sensible results on synthetic data.
Golden AAPL tests verify that Apple is NOT classified as distressed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import AAPL_EXPECTED


class TestL2SurvivalOnSynthetic:
    """Survival mode tests using always-available synthetic data."""

    def test_survival_flag_returns_series(self, synthetic_cache):
        from operator1.analysis.survival_mode import compute_company_survival_flag
        flag = compute_company_survival_flag(synthetic_cache)
        assert isinstance(flag, pd.Series)
        assert len(flag) == len(synthetic_cache)
        assert set(flag.dropna().unique()).issubset({0, 1})

    def test_survival_probability_in_range(self, synthetic_cache):
        from operator1.analysis.survival_mode import compute_survival_probability
        prob = compute_survival_probability(synthetic_cache)
        assert isinstance(prob, pd.Series)
        valid = prob.dropna()
        if len(valid) > 0:
            assert valid.min() >= 0.0, f"Survival prob min={valid.min()}"
            assert valid.max() <= 1.0, f"Survival prob max={valid.max()}"

    def test_hierarchy_weights_sum_approximately_100(self, synthetic_cache):
        from operator1.analysis.survival_mode import compute_company_survival_flag
        from operator1.analysis.hierarchy_weights import compute_hierarchy_weights
        synthetic_cache["company_survival_mode_flag"] = compute_company_survival_flag(
            synthetic_cache
        )
        synthetic_cache["survival_probability"] = 0.9
        result = compute_hierarchy_weights(synthetic_cache.copy())
        weight_cols = [c for c in result.columns if c.startswith("hierarchy_tier")]
        if weight_cols:
            row_sums = result[weight_cols].sum(axis=1).dropna()
            # Weights may be fractions (sum~1.0) or percentages (sum~100)
            mean_sum = row_sums.mean()
            is_fraction = mean_sum < 5.0
            expected = 1.0 if is_fraction else 100.0
            assert mean_sum == pytest.approx(expected, rel=0.1), (
                f"Weight sum mean={mean_sum:.2f}, expected ~{expected}"
            )


class TestL2FinancialHealthOnSynthetic:
    """FH scoring tests using synthetic data."""

    def test_financial_health_produces_composite(self, synthetic_cache):
        from operator1.models.financial_health import compute_financial_health
        cache, result = compute_financial_health(synthetic_cache.copy())
        assert "fh_composite_score" in cache.columns
        assert not np.isnan(result.latest_composite), "FH composite is NaN"
        assert 0 <= result.latest_composite <= 100, (
            f"FH composite={result.latest_composite} out of 0-100 range"
        )

    def test_financial_health_label_not_empty(self, synthetic_cache):
        from operator1.models.financial_health import compute_financial_health
        _, result = compute_financial_health(synthetic_cache.copy())
        assert result.latest_label != "", "FH label is empty"
        assert result.latest_label != "Unknown", "FH label is Unknown"


class TestL2AnalysisOnGolden:
    """Tests using frozen AAPL data (skipped if fixture missing)."""

    def test_apple_not_in_survival_most_days(self, aapl_cache):
        from operator1.analysis.survival_mode import compute_company_survival_flag
        flag = compute_company_survival_flag(aapl_cache, sector="Technology")
        flag_rate = flag.sum() / len(flag)
        max_rate = AAPL_EXPECTED.get("l2_survival_flag_rate_max", 0.15)
        assert flag_rate < max_rate, (
            f"Apple survival flag rate={flag_rate:.1%} (expected <{max_rate:.0%}). "
            f"Flagged {int(flag.sum())}/{len(flag)} days"
        )

    def test_apple_survival_probability_high(self, aapl_cache):
        from operator1.analysis.survival_mode import compute_survival_probability
        prob = compute_survival_probability(aapl_cache)
        latest = float(prob.dropna().iloc[-1])
        min_prob = AAPL_EXPECTED.get("l2_survival_prob_min", 0.60)
        assert latest > min_prob, (
            f"Apple survival prob={latest:.3f} (expected >{min_prob})"
        )

    def test_apple_fh_composite_above_minimum(self, aapl_cache):
        from operator1.models.financial_health import compute_financial_health
        _, result = compute_financial_health(
            aapl_cache.copy(), sector="Technology"
        )
        fh_min = AAPL_EXPECTED.get("l2_fh_composite_min", 15)
        fh_max = AAPL_EXPECTED.get("l2_fh_composite_max", 95)
        assert result.latest_composite > fh_min, (
            f"Apple FH={result.latest_composite:.1f} (expected >{fh_min}). "
            f"Label={result.latest_label}. "
            f"Tier means: {result.tier_means}"
        )
        assert result.latest_composite < fh_max, (
            f"Apple FH={result.latest_composite:.1f} suspiciously high (>{fh_max})"
        )

    def test_apple_fh_label_not_critical(self, aapl_cache):
        from operator1.models.financial_health import compute_financial_health
        _, result = compute_financial_health(
            aapl_cache.copy(), sector="Technology"
        )
        bad_labels = AAPL_EXPECTED.get("l2_fh_labels_bad", ["Critical"])
        assert result.latest_label not in bad_labels, (
            f"Apple FH label='{result.latest_label}' (should not be in {bad_labels})"
        )

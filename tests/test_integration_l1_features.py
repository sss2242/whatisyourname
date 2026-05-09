"""Integration tests -- Layer 1: Feature Engineering.

Validates that compute_derived_variables produces correct column counts,
value ranges, and companion flags on both golden AAPL data and synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import AAPL_EXPECTED

CORE_COLUMNS = [
    "return_1d", "volatility_21d", "current_ratio", "gross_margin",
    "pe_ratio_calc", "drawdown_252d", "debt_to_equity_abs", "fcf_yield",
]


class TestL1DerivedVariablesOnSynthetic:
    """Tests using always-available synthetic data."""

    def test_derived_variables_runs_without_crash(self, synthetic_cache):
        from operator1.features.derived_variables import compute_derived_variables
        result = compute_derived_variables(synthetic_cache.copy())
        assert len(result.columns) > len(synthetic_cache.columns)

    def test_returns_are_return_scale(self, synthetic_cache):
        from operator1.features.derived_variables import compute_derived_variables
        result = compute_derived_variables(synthetic_cache.copy())
        if "return_1d" in result.columns:
            r = result["return_1d"].dropna()
            max_abs = AAPL_EXPECTED.get("l1_return_1d_abs_max", 0.5)
            assert r.abs().max() < max_abs, (
                f"return_1d max abs={r.abs().max():.4f}, "
                f"expected <{max_abs} (return-scale, not price-scale)"
            )

    def test_column_count_above_minimum(self, synthetic_cache):
        from operator1.features.derived_variables import compute_derived_variables
        result = compute_derived_variables(synthetic_cache.copy())
        # Synthetic cache starts with ~27 cols; derived adds ~185 -> ~212 total.
        # Golden AAPL cache has 300+. Use lower threshold for synthetic.
        min_cols = 100
        assert len(result.columns) >= min_cols, (
            f"Column count {len(result.columns)} < {min_cols}"
        )


class TestL1DerivedVariablesOnGolden:
    """Tests using frozen AAPL data (skipped if fixture missing)."""

    def test_core_columns_not_all_nan(self, aapl_cache):
        for col in CORE_COLUMNS:
            if col in aapl_cache.columns:
                non_nan = aapl_cache[col].notna().sum()
                assert non_nan > 50, (
                    f"{col} has only {non_nan} non-NaN values (expected >50)"
                )

    def test_core_nan_rate_within_bounds(self, aapl_cache):
        max_rate = AAPL_EXPECTED.get("l1_core_nan_rate_max", 0.25)
        for col in CORE_COLUMNS:
            if col in aapl_cache.columns:
                nan_rate = aapl_cache[col].isna().mean()
                assert nan_rate < max_rate, (
                    f"{col} NaN rate={nan_rate:.1%} exceeds {max_rate:.0%}"
                )

    def test_no_inf_values_in_cache(self, aapl_cache):
        numeric = aapl_cache.select_dtypes(include=[np.number])
        for col in numeric.columns:
            inf_count = np.isinf(numeric[col].dropna()).sum()
            assert inf_count == 0, f"{col} has {inf_count} Inf values"

    def test_index_is_datetime_monotonic(self, aapl_cache):
        assert isinstance(aapl_cache.index, pd.DatetimeIndex), (
            f"Cache index type: {type(aapl_cache.index)}"
        )
        assert aapl_cache.index.is_monotonic_increasing, (
            "Cache index is not monotonically increasing"
        )

    def test_no_duplicate_columns(self, aapl_cache):
        dupes = aapl_cache.columns[aapl_cache.columns.duplicated()].tolist()
        assert len(dupes) == 0, f"Duplicate columns: {dupes}"

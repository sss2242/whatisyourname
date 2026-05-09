"""Integration tests -- Regression Guards.

Tracks cache column counts, NaN rates, and structural properties.
Catches silent data loss and column regressions.

Pattern: QuantConnect/Lean DataPoints regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import AAPL_EXPECTED


class TestCacheStructureOnSynthetic:
    """Structure tests using synthetic data."""

    def test_derived_variables_adds_columns(self, synthetic_cache):
        from operator1.features.derived_variables import compute_derived_variables
        before = len(synthetic_cache.columns)
        result = compute_derived_variables(synthetic_cache.copy())
        after = len(result.columns)
        assert after > before, (
            f"derived_variables added 0 columns ({before} -> {after})"
        )

    def test_no_all_nan_columns_after_derivation(self, synthetic_cache):
        from operator1.features.derived_variables import compute_derived_variables
        result = compute_derived_variables(synthetic_cache.copy())
        all_nan_cols = [
            c for c in result.columns
            if result[c].isna().all() and result[c].dtype in ("float64", "float32")
        ]
        # Some companion flags may be all-NaN for synthetic data; allow up to 30%
        max_all_nan_rate = 0.30
        rate = len(all_nan_cols) / max(len(result.columns), 1)
        assert rate < max_all_nan_rate, (
            f"{len(all_nan_cols)}/{len(result.columns)} columns all-NaN "
            f"({rate:.0%}). First 10: {all_nan_cols[:10]}"
        )


class TestCacheStructureOnGolden:
    """Structure tests using frozen AAPL data."""

    def test_column_count_above_minimum(self, aapl_cache):
        min_cols = AAPL_EXPECTED.get("l1_column_count_min", 300)
        n_cols = len(aapl_cache.columns)
        assert n_cols >= min_cols, (
            f"Column count regressed: {n_cols} (expected >={min_cols}). "
            f"Columns may have been accidentally dropped."
        )

    def test_no_duplicate_columns(self, aapl_cache):
        dupes = aapl_cache.columns[aapl_cache.columns.duplicated()].tolist()
        assert len(dupes) == 0, f"Duplicate columns found: {dupes}"

    def test_index_is_datetime_and_monotonic(self, aapl_cache):
        assert isinstance(aapl_cache.index, pd.DatetimeIndex)
        assert aapl_cache.index.is_monotonic_increasing

    def test_core_columns_exist(self, aapl_cache):
        required = [
            "close", "return_1d", "volume",
            "current_ratio", "gross_margin",
        ]
        missing = [c for c in required if c not in aapl_cache.columns]
        assert len(missing) == 0, f"Required columns missing: {missing}"

    def test_no_inf_in_numeric_columns(self, aapl_cache):
        numeric = aapl_cache.select_dtypes(include=[np.number])
        inf_cols = []
        for col in numeric.columns:
            if np.isinf(numeric[col].dropna()).any():
                inf_cols.append(col)
        assert len(inf_cols) == 0, f"Inf values found in: {inf_cols}"

    def test_close_price_positive(self, aapl_cache):
        if "close" in aapl_cache.columns:
            close = aapl_cache["close"].dropna()
            assert (close > 0).all(), (
                f"Negative/zero close prices found: min={close.min()}"
            )

    def test_volume_non_negative(self, aapl_cache):
        if "volume" in aapl_cache.columns:
            vol = aapl_cache["volume"].dropna()
            assert (vol >= 0).all(), (
                f"Negative volume found: min={vol.min()}"
            )

"""Shared utility for classifying cache column data frequencies.

The daily cache mixes three frequencies:
- Daily (prices): ~252 unique values per year
- Quarterly (financials): ~4 unique values per year, forward-filled
- Annual (macro): ~1-2 unique values per year, forward-filled

Models that receive mixed-frequency data need to adapt their behavior
based on the true frequency of each column. This module provides the
classification logic used by all adapted models.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def classify_column_frequency(series: pd.Series) -> str:
    """Classify a cache column's true data frequency.

    Parameters
    ----------
    series:
        A column from the daily cache DataFrame.

    Returns
    -------
    One of: ``'daily'``, ``'quarterly'``, ``'annual'``, ``'constant'``
    """
    n = len(series.dropna())
    if n == 0:
        return "constant"

    n_unique = series.nunique()
    if n_unique <= 1:
        return "constant"

    unique_ratio = n_unique / n
    if unique_ratio > 0.10:
        return "daily"
    elif unique_ratio > 0.01:
        return "quarterly"
    elif unique_ratio > 0.002:
        return "annual"
    return "constant"


def detect_filing_change_days(series: pd.Series) -> pd.Series:
    """Return a boolean mask where True = new filing value appeared.

    For forward-filled quarterly/annual data, this identifies the
    ~4-8 days per year when the value actually changes (new filing).
    """
    changes = series.diff().abs() > 0
    # First non-NaN value is also a "change" (first filing)
    first_valid = series.first_valid_index()
    if first_valid is not None:
        changes[first_valid] = True
    return changes


def get_quarterly_change_variance(series: pd.Series) -> float:
    """Compute variance from quarter-to-quarter value changes only.

    For forward-filled data, this extracts the ~4-8 actual changes
    per year and computes their variance, giving a realistic measure
    of the variable's true volatility (not the artificial zero-variance
    from repeated daily values).

    Returns 0.0 if fewer than 2 changes are detected.
    """
    changes = series.diff()
    non_zero = changes[changes.abs() > 0]
    if len(non_zero) < 2:
        return 0.0
    return float(non_zero.var())


def add_filing_timing_features(
    cache: pd.DataFrame,
    var: str,
) -> list[str]:
    """Add filing-timing features for a quarterly/annual variable.

    Creates two new columns in the cache:
    - ``{var}_days_since_filing``: integer counter resetting to 0
      when the value changes
    - ``{var}_filing_arrived``: 1.0 on days when value changes, else 0.0

    These features help LSTM/TFT/Transformer models understand
    *when* new information arrived, not just the stale value.

    Returns the list of new column names added.
    """
    new_cols = []

    change_mask = detect_filing_change_days(cache[var])

    # Days since last filing change
    dsf_col = f"{var}_days_since_filing"
    if dsf_col not in cache.columns:
        groups = change_mask.cumsum()
        cache[dsf_col] = groups.groupby(groups).cumcount()
        new_cols.append(dsf_col)

    # Filing arrival flag
    fa_col = f"{var}_filing_arrived"
    if fa_col not in cache.columns:
        cache[fa_col] = change_mask.astype(float)
        new_cols.append(fa_col)

    return new_cols


def get_filing_change_derivative(
    cache: pd.DataFrame,
    var: str,
) -> str:
    """Create a filing-change derivative column for a forward-filled variable.

    The derivative is the absolute change on filing days, zero between
    filings. Useful for VAR and Tree models that need non-constant
    columns with meaningful variation.

    Returns the name of the new column (created in-place in cache).
    """
    col_name = f"{var}_filing_change"
    if col_name not in cache.columns:
        cache[col_name] = cache[var].diff().fillna(0)
    return col_name

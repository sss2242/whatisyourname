"""Missingness classifier -- routes NaN values to the correct estimator.

Classifies each missing value as one of:
  - ``"mcar"`` (Missing Completely At Random): structural data gap,
    missingness unrelated to any variable.
  - ``"mar"`` (Missing At Random): missingness depends on *observed*
    variables but not on the missing value itself.
  - ``"mnar"`` (Missing Not At Random): missingness depends on the
    missing value itself -- company is likely hiding data.

The classification drives which estimator handles the imputation:
  - MCAR / MAR  -> ``missing_data_estimator.py`` (MICE + GP + Matrix Completion)
  - MNAR        -> ``hidden_data_estimator.py`` (Heckman + Pattern-Mixture + Bounds)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification result container
# ---------------------------------------------------------------------------

@dataclass
class MissingnessClassification:
    """Per-variable missingness classification for the full feature table."""

    # var -> Series of missingness types ("mar", "mcar", "mnar") per row
    types: dict[str, pd.Series] = field(default_factory=dict)
    # var -> Series of classification confidence [0, 1]
    confidences: dict[str, pd.Series] = field(default_factory=dict)
    # Summary counts
    summary: dict[str, dict[str, int]] = field(default_factory=dict)

    def mar_mask(self, var: str, df: pd.DataFrame) -> pd.Series:
        """Boolean mask: rows where var is missing AND classified MAR/MCAR."""
        if var not in self.types:
            return df[var].isna()
        t = self.types[var]
        return df[var].isna() & t.isin(["mar", "mcar"])

    def mnar_mask(self, var: str, df: pd.DataFrame) -> pd.Series:
        """Boolean mask: rows where var is missing AND classified MNAR."""
        if var not in self.types:
            return pd.Series(False, index=df.index)
        t = self.types[var]
        return df[var].isna() & (t == "mnar")


# ---------------------------------------------------------------------------
# Classification heuristics
# ---------------------------------------------------------------------------

# Peer coverage threshold: if >80% of peers report a field but this
# company does not, the omission is likely deliberate (MNAR).
_PEER_COVERAGE_HIGH = 0.80
# If <30% of peers report a field, the gap is market-wide (MAR).
_PEER_COVERAGE_LOW = 0.30

# If a variable was reported in prior periods but vanishes, that is
# suspicious (MNAR).  We look back this many periods.
_LOOKBACK_PERIODS = 4


def _compute_peer_coverage(
    df: pd.DataFrame,
    variables: list[str],
) -> dict[str, float]:
    """Compute the fraction of non-null values per variable.

    This is a proxy for "peer coverage" -- in a single-company context,
    it measures how often the variable is reported across the timeline.
    """
    coverage: dict[str, float] = {}
    n = len(df)
    if n == 0:
        return {v: 0.0 for v in variables}
    for var in variables:
        if var in df.columns:
            coverage[var] = float(df[var].notna().sum() / n)
        else:
            coverage[var] = 0.0
    return coverage


def _detect_reporting_disappearance(
    series: pd.Series,
    lookback: int = _LOOKBACK_PERIODS,
) -> pd.Series:
    """Detect rows where a previously-reported value suddenly vanishes.

    Returns a boolean Series: True where the value was observed in at
    least ``lookback`` of the prior periods but is now NaN.
    """
    is_null = series.isna()
    # Count non-null values in rolling window ending at previous row
    prior_observed = series.notna().shift(1).rolling(
        window=lookback, min_periods=1,
    ).sum()
    # Disappearance: currently null but was observed in most prior periods
    threshold = max(1, lookback * 0.5)
    return is_null & (prior_observed >= threshold)


def _detect_suspicious_aggregation(
    df: pd.DataFrame,
    var: str,
) -> pd.Series:
    """Detect rows where a variable might be hidden via aggregation.

    Heuristic: if a component variable is missing but the aggregate
    is present and unusually large relative to history, the company
    may be lumping line items.
    """
    result = pd.Series(False, index=df.index)

    # Known decomposition relationships
    decompositions = {
        "short_term_debt": "total_debt_asof",
        "long_term_debt": "total_debt_asof",
        "cost_of_revenue": "revenue",
        "operating_expenses": "revenue",
        "interest_expense": "ebit",
    }

    aggregate = decompositions.get(var)
    if aggregate and aggregate in df.columns:
        agg_series = df[aggregate]
        var_missing = df[var].isna() if var in df.columns else pd.Series(True, index=df.index)
        agg_present = agg_series.notna()

        # Aggregate is unusually large (> 1.5 std above rolling mean)
        rolling_mean = agg_series.rolling(window=12, min_periods=3).mean()
        rolling_std = agg_series.rolling(window=12, min_periods=3).std()
        unusually_large = agg_series > (rolling_mean + 1.5 * rolling_std.fillna(0))

        result = var_missing & agg_present & unusually_large

    return result


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------


def classify_missingness(
    df: pd.DataFrame,
    variables: list[str],
    peer_coverage: dict[str, float] | None = None,
) -> MissingnessClassification:
    """Classify why each value is missing in the feature table.

    Parameters
    ----------
    df:
        The full feature table from the cache builder.
    variables:
        List of variable names to classify.
    peer_coverage:
        Optional externally-computed peer coverage rates.
        If None, computed from the dataframe itself.

    Returns
    -------
    MissingnessClassification
        Per-variable, per-row classification of missingness type.
    """
    result = MissingnessClassification()

    if peer_coverage is None:
        peer_coverage = _compute_peer_coverage(df, variables)

    for var in variables:
        if var not in df.columns:
            continue

        series = df[var]
        is_missing = series.isna()
        n_missing = int(is_missing.sum())

        if n_missing == 0:
            # No missing values -- skip
            result.types[var] = pd.Series("observed", index=df.index)
            result.confidences[var] = pd.Series(1.0, index=df.index)
            result.summary[var] = {"observed": len(df), "mar": 0, "mnar": 0, "mcar": 0}
            continue

        # Initialize all missing as MAR (default assumption)
        types = pd.Series("observed", index=df.index)
        types[is_missing] = "mar"
        conf = pd.Series(1.0, index=df.index)
        conf[is_missing] = 0.5  # default confidence for MAR

        # --- Rule 1: Low peer coverage -> MCAR (structural gap) ---
        cov = peer_coverage.get(var, 0.0)
        if cov < _PEER_COVERAGE_LOW:
            types[is_missing] = "mcar"
            conf[is_missing] = 0.7
            logger.debug(
                "%s: coverage=%.2f < %.2f -- classifying as MCAR",
                var, cov, _PEER_COVERAGE_LOW,
            )

        # --- Rule 2: High coverage but this entity missing -> MNAR ---
        # (applied per-row: if the variable is observed most of the time
        #  but suddenly vanishes, that's suspicious)
        disappearance = _detect_reporting_disappearance(series)
        types[disappearance] = "mnar"
        conf[disappearance] = 0.75

        # --- Rule 3: High peer coverage + missing -> MNAR ---
        if cov >= _PEER_COVERAGE_HIGH:
            # Most of the time this variable IS reported; missing rows
            # are likely deliberate omissions
            still_mar = is_missing & (types != "mnar")
            types[still_mar] = "mnar"
            conf[still_mar] = 0.6

        # --- Rule 4: Suspicious aggregation -> MNAR ---
        aggregation_flag = _detect_suspicious_aggregation(df, var)
        types[aggregation_flag] = "mnar"
        conf[aggregation_flag] = 0.8

        # --- Rule 5: All missing (never reported) -> MCAR ---
        if n_missing == len(df):
            types[is_missing] = "mcar"
            conf[is_missing] = 0.9

        result.types[var] = types
        result.confidences[var] = conf

        # Summary
        counts = types.value_counts().to_dict()
        result.summary[var] = {
            k: int(counts.get(k, 0))
            for k in ("observed", "mar", "mcar", "mnar")
        }

    n_mnar_total = sum(
        s.get("mnar", 0) for s in result.summary.values()
    )
    n_mar_total = sum(
        s.get("mar", 0) + s.get("mcar", 0)
        for s in result.summary.values()
    )
    logger.info(
        "Missingness classification: %d MAR/MCAR cells, %d MNAR cells "
        "across %d variables",
        n_mar_total, n_mnar_total, len(variables),
    )

    return result

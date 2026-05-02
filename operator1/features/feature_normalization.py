"""Feature normalization -- regime-aware z-scores and percentiles.

MUST run LAST in the feature pipeline (Step 5i.9), after all other
feature modules have populated the cache, and after hierarchy_weights
has computed survival_regime.

Produces ~40 normalized columns (10 key variables x 4 normalization types):
  1. {var}_zscore_63d: Rolling z-score (regime-relative positioning)
  2. {var}_percentile_252d: Expanding percentile (historical context)
  3. {var}_change_21d: 21-day level change (gradient/slope for tree models)
  4. {var}_regime_zscore: Regime-conditional z-score (for 5 survival vars only)

References:
  - DeMiguel et al. 2009 (portfolio optimization with normalized features)
  - Standard quantitative practice for ML feature preprocessing

Pipeline step: Step 5i.9 (MUST be LAST feature module before temporal models)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from operator1.scoring_weights import get_weight

from operator1.constants import EPSILON

logger = logging.getLogger(__name__)

# Variables to normalize with z-score, percentile, and change
_NORMALIZE_VARS: list[str] = [
    "current_ratio",
    "debt_to_equity_abs",
    "fcf_yield",
    "gross_margin",
    "volatility_21d",
    "pe_ratio_calc",
    "revenue_growth_yoy",
    "sentiment_score",
    "merton_dd",
    "fh_composite_score",
]

# Survival-critical variables for regime-conditional z-scores
_REGIME_VARS: list[str] = [
    "current_ratio",
    "debt_to_equity_abs",
    "fcf_yield",
    "drawdown_252d",
    "volatility_21d",
]


def compute_feature_normalization(cache: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized feature variants for ML model consumption.

    Parameters
    ----------
    cache:
        Daily cache with feature columns already computed. Must include
        ``survival_regime`` for regime-conditional z-scores (from
        ``hierarchy_weights.py`` Step 5).

    Returns
    -------
    pd.DataFrame
        Cache augmented with ~40 normalized columns.
    """
    result = cache.copy()
    n_added = 0

    for var in _NORMALIZE_VARS:
        if var not in result.columns or result[var].notna().sum() < 10:
            continue

        s = result[var].astype(float)

        # 1. Rolling z-score (63d)
        mean_63 = s.rolling(63, min_periods=10).mean()
        std_63 = s.rolling(63, min_periods=10).std().clip(lower=EPSILON)
        col_z = f"{var}_zscore_63d"
        if col_z not in result.columns:
            result[col_z] = (s - mean_63) / std_63
            n_added += 1

        # 2. Expanding percentile rank (252d window)
        col_p = f"{var}_percentile_252d"
        if col_p not in result.columns:
            result[col_p] = s.rolling(252, min_periods=20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                raw=False,
            )
            n_added += 1

        # 3. Level change over 21 days
        col_c = f"{var}_change_21d"
        if col_c not in result.columns:
            result[col_c] = s.diff(21)
            n_added += 1

    # 4. Regime-conditional z-score (for survival variables only)
    regime_col = result.get("survival_regime")
    if regime_col is not None and regime_col.notna().sum() > 20:
        for var in _REGIME_VARS:
            if var not in result.columns or result[var].notna().sum() < 10:
                continue

            col_rz = f"{var}_regime_zscore"
            if col_rz in result.columns:
                continue

            s = result[var].astype(float)
            result[col_rz] = np.nan

            for regime in regime_col.dropna().unique():
                mask = regime_col == regime
                if mask.sum() < 5:
                    continue
                s_regime = s.loc[mask]
                rmean = s_regime.expanding(min_periods=5).mean()
                rstd = s_regime.expanding(min_periods=5).std().clip(lower=EPSILON)
                result.loc[mask, col_rz] = (s_regime - rmean) / rstd

            n_added += 1

    if n_added > 0:
        logger.info("Feature normalization computed: %d columns", n_added)

    return result

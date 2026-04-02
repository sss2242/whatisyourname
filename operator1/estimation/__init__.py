"""Post-cache estimation and imputation engine.

Supports three imputer backends (configured via ``global_config.yml``
key ``estimation_imputer``):

  - ``"split"`` (default): Classifies missingness as MAR or MNAR,
    then routes to specialized estimators:
      * MAR: MICE + Gaussian Process + Matrix Completion ensemble
      * MNAR: Heckman Selection + Pattern-Mixture + Sensitivity Bounds + GAIN
  - ``"bayesian_ridge"``: Legacy per-variable BayesianRidge (linear)
  - ``"vae"``: Legacy Variational Autoencoder (nonlinear, requires torch)
"""

from __future__ import annotations

import pandas as pd


def select_feature_columns(
    df: pd.DataFrame,
    target_vars: list[str],
    *,
    max_features: int = 15,
    min_coverage: float = 0.5,
) -> list[str]:
    """Select numeric, non-flag columns with decent coverage for estimation features.

    Shared utility used by both ``missing_data_estimator`` (MAR path)
    and ``hidden_data_estimator`` (MNAR path) to select predictor
    columns for imputation models.

    Parameters
    ----------
    df:
        The daily cache DataFrame.
    target_vars:
        Variables being estimated (excluded from features).
    max_features:
        Maximum number of feature columns to return (default 15).
    min_coverage:
        Minimum non-null fraction required (default 0.5).

    Returns
    -------
    list[str]
        Selected feature column names, capped at *max_features*.
    """
    candidates = []
    for col in df.columns:
        if col in target_vars:
            continue
        if col.startswith("is_missing_") or col.startswith("invalid_math_"):
            continue
        if col.endswith("_source") or col.endswith("_confidence"):
            continue
        if col.endswith("_observed") or col.endswith("_estimated"):
            continue
        if col.endswith("_missingness_type"):
            continue
        if df[col].dtype not in ("float64", "float32", "int64", "int32"):
            continue
        if df[col].notna().mean() < min_coverage:
            continue
        candidates.append(col)

    return candidates[:max_features]

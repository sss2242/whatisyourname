"""Missing Data Estimator -- handles MAR/MCAR missingness.

Handles data that is genuinely absent: API gaps, filing periods not
yet arrived, market-wide coverage holes.  The missingness mechanism
is independent of the missing value itself.

Three-method ensemble:
  1. **MICE** (Multiple Imputation by Chained Equations):
     Iterative multivariate imputation capturing cross-variable
     dependencies.
  2. **Gaussian Process Regression**:
     Provides calibrated posterior uncertainty for each imputed value.
  3. **Matrix Completion** (nuclear norm / Soft-Impute):
     Exploits the low-rank structure of financial data (most variation
     explained by a few latent factors).

Final estimate = weighted median of the three methods, weighted by
inverse variance.  Confidence is driven by inter-method agreement.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_OBSERVED_FOR_MODEL = 10
_MIN_TRAIN_WINDOW = 20
_MICE_MAX_ITER = 25
_MICE_N_NEAREST = 10
_GP_MAX_TRAIN = 500      # Limit GP training set to avoid O(n^3) blowup
_SOFTIMPUTE_MAX_ITER = 50
_SOFTIMPUTE_TOL = 1e-5
_SOFTIMPUTE_MAX_RANK = 20


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MAREstimationResult:
    """Result from the missing-data (MAR) estimator."""

    estimated_values: dict[str, pd.Series] = field(default_factory=dict)
    confidence_scores: dict[str, pd.Series] = field(default_factory=dict)
    methods_used: dict[str, str] = field(default_factory=dict)
    n_estimated: int = 0


# ---------------------------------------------------------------------------
# Method 1: MICE (Multiple Imputation by Chained Equations)
# ---------------------------------------------------------------------------


def _run_mice(
    df: pd.DataFrame,
    variables: list[str],
    feature_cols: list[str],
    mar_masks: dict[str, pd.Series],
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Run MICE imputation using miceforest (LightGBM-based) or sklearn fallback.

    Prefers miceforest when available: 5-10x faster than sklearn MICE and
    captures non-linear relationships via LightGBM decision trees.

    Returns dict of var -> (estimated_values, confidence_scores).
    """
    results: dict[str, tuple[pd.Series, pd.Series]] = {}

    # Build the joint matrix: features + targets
    all_cols = list(set(feature_cols + variables))
    all_cols = [c for c in all_cols if c in df.columns]

    X = df[all_cols].copy()
    # Forward-fill feature columns to reduce sparsity
    for col in feature_cols:
        if col in X.columns:
            X[col] = X[col].ffill().bfill()

    X_imputed = None

    # Try miceforest first (faster, non-linear)
    try:
        import miceforest as mf
        kernel = mf.ImputationKernel(X, save_all_iterations=False, random_state=42)
        kernel.mice(iterations=min(_MICE_MAX_ITER, 3))
        X_imputed = kernel.complete_data().values
        logger.info("MICE imputation via miceforest (LightGBM): %d cols", len(all_cols))
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("miceforest failed, falling back to sklearn: %s", exc)

    # Fallback to sklearn IterativeImputer with BayesianRidge
    if X_imputed is None:
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            from sklearn.linear_model import BayesianRidge
        except ImportError:
            logger.warning("sklearn IterativeImputer not available -- skipping MICE")
            return results

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imputer = IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=_MICE_MAX_ITER,
                    n_nearest_features=min(_MICE_N_NEAREST, len(all_cols) - 1),
                    sample_posterior=True,
                    random_state=42,
                    verbose=0,
                )
                X_imputed = imputer.fit_transform(X.values)
        except Exception as exc:
            logger.warning("MICE failed: %s", exc)
            return results

    X_imputed_df = pd.DataFrame(X_imputed, index=df.index, columns=all_cols)

    for var in variables:
        if var not in all_cols:
            continue
        mask = mar_masks.get(var, df[var].isna())
        estimated = pd.Series(np.nan, index=df.index, dtype=float)
        confidence = pd.Series(np.nan, index=df.index, dtype=float)

        estimated[mask] = X_imputed_df.loc[mask, var]

        # MICE confidence: based on how many iterations converged and
        # the fraction of observed data available for this variable
        obs_frac = float(df[var].notna().sum() / max(len(df), 1))
        base_conf = min(0.85, 0.4 + obs_frac * 0.5)
        confidence[mask] = base_conf

        results[var] = (estimated, confidence)

    return results


# ---------------------------------------------------------------------------
# Method 2: Gaussian Process Regression
# ---------------------------------------------------------------------------


def _run_gp(
    df: pd.DataFrame,
    variables: list[str],
    feature_cols: list[str],
    mar_masks: dict[str, pd.Series],
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Run Gaussian Process regression per variable.

    Returns dict of var -> (estimated_values, confidence_scores).
    GP posterior variance provides calibrated uncertainty.
    """
    results: dict[str, tuple[pd.Series, pd.Series]] = {}

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    except ImportError:
        logger.warning("sklearn GP not available -- skipping GP method")
        return results

    # Prepare feature matrix (forward-filled)
    X_full = df[feature_cols].ffill().bfill()
    # Limit features to avoid GP slowdown
    usable_cols = [c for c in feature_cols if X_full[c].notna().mean() > 0.8]
    usable_cols = usable_cols[:10]  # GP is O(n^3), keep features lean

    if not usable_cols:
        return results

    X_full = X_full[usable_cols]

    for var in variables:
        if var not in df.columns:
            continue

        mask = mar_masks.get(var, df[var].isna())
        if not mask.any():
            continue

        target = df[var]
        obs_mask = target.notna()

        if obs_mask.sum() < _MIN_OBSERVED_FOR_MODEL:
            continue

        estimated = pd.Series(np.nan, index=df.index, dtype=float)
        confidence = pd.Series(np.nan, index=df.index, dtype=float)

        # Subsample training data if too large
        obs_indices = df.index[obs_mask]
        if len(obs_indices) > _GP_MAX_TRAIN:
            rng = np.random.RandomState(42)
            obs_indices = rng.choice(obs_indices, _GP_MAX_TRAIN, replace=False)
            obs_indices = np.sort(obs_indices)

        X_train = X_full.loc[obs_indices].values
        y_train = target.loc[obs_indices].values

        missing_indices = df.index[mask]
        X_pred = X_full.loc[missing_indices].values

        # Standardize
        y_mean = np.nanmean(y_train)
        y_std = np.nanstd(y_train)
        if y_std < 1e-10:
            y_std = 1.0
        y_train_norm = (y_train - y_mean) / y_std

        try:
            kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=2,
                random_state=42,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X_train, y_train_norm)
                y_pred_norm, y_std_pred = gp.predict(X_pred, return_std=True)

            # Denormalize
            y_pred = y_pred_norm * y_std + y_mean

            estimated.loc[missing_indices] = y_pred

            # Confidence from posterior std: lower std = higher confidence
            max_std = max(float(y_std_pred.max()), 1e-6)
            conf_scores = 1.0 - np.clip(y_std_pred / (max_std * 2), 0, 1)
            conf_scores = np.clip(conf_scores, 0.1, 0.95)
            confidence.loc[missing_indices] = conf_scores

        except Exception as exc:
            logger.debug("GP failed for '%s': %s", var, exc)
            continue

        results[var] = (estimated, confidence)

    return results


# ---------------------------------------------------------------------------
# Method 3: Matrix Completion (Soft-Impute / nuclear norm)
# ---------------------------------------------------------------------------


def _soft_threshold_svd(
    M: np.ndarray,
    lam: float,
    max_rank: int = _SOFTIMPUTE_MAX_RANK,
) -> np.ndarray:
    """Apply soft-thresholded SVD for nuclear norm minimization.

    Computes truncated SVD of M, soft-thresholds singular values,
    and reconstructs.
    """
    try:
        # Truncated SVD for efficiency
        rank = min(max_rank, min(M.shape) - 1)
        if rank < 1:
            return M

        from scipy.sparse.linalg import svds
        U, sigma, Vt = svds(M.astype(float), k=rank)

        # Sort by descending singular value
        idx = np.argsort(-sigma)
        sigma = sigma[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]

        # Soft threshold
        sigma_thresh = np.maximum(sigma - lam, 0)

        return U @ np.diag(sigma_thresh) @ Vt

    except Exception:
        # Fallback to full SVD if svds fails
        U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
        sigma_thresh = np.maximum(sigma - lam, 0)
        return U @ np.diag(sigma_thresh) @ Vt


def _run_matrix_completion(
    df: pd.DataFrame,
    variables: list[str],
    feature_cols: list[str],
    mar_masks: dict[str, pd.Series],
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Run Soft-Impute matrix completion.

    Returns dict of var -> (estimated_values, confidence_scores).
    """
    results: dict[str, tuple[pd.Series, pd.Series]] = {}

    # Build the joint matrix: features + targets
    all_cols = list(set(feature_cols + variables))
    all_cols = [c for c in all_cols if c in df.columns]

    X = df[all_cols].copy()
    # Forward-fill features
    for col in feature_cols:
        if col in X.columns:
            X[col] = X[col].ffill().bfill()

    # Standardize columns
    means = X.mean()
    stds = X.std()
    stds[stds < 1e-10] = 1.0
    X_norm = (X - means) / stds

    # Create observation mask
    observed = X_norm.notna()
    X_filled = X_norm.fillna(0).values
    obs_mask = observed.values.astype(bool)

    # Soft-Impute iterations
    # Choose lambda as fraction of largest singular value
    try:
        from scipy.sparse.linalg import svds
        _, s0, _ = svds(X_filled, k=1)
        lam = float(s0[0]) * 0.1
    except Exception:
        lam = 1.0

    M = X_filled.copy()
    for iteration in range(_SOFTIMPUTE_MAX_ITER):
        M_old = M.copy()

        # SVD soft-threshold
        M_new = _soft_threshold_svd(M, lam)

        # Replace observed entries back
        M_new[obs_mask] = X_filled[obs_mask]

        # Check convergence
        diff = np.linalg.norm(M_new - M_old) / max(np.linalg.norm(M_old), 1e-10)
        M = M_new

        if diff < _SOFTIMPUTE_TOL:
            logger.debug("Soft-Impute converged at iteration %d", iteration)
            break

    # Denormalize
    M_denorm = M * stds.values + means.values
    M_df = pd.DataFrame(M_denorm, index=df.index, columns=all_cols)

    for var in variables:
        if var not in all_cols:
            continue

        mask = mar_masks.get(var, df[var].isna())
        estimated = pd.Series(np.nan, index=df.index, dtype=float)
        confidence = pd.Series(np.nan, index=df.index, dtype=float)

        estimated[mask] = M_df.loc[mask, var]

        # Confidence: based on reconstruction error on observed data
        obs_var = df[var].notna()
        if obs_var.sum() > 5:
            recon_error = np.abs(M_df.loc[obs_var, var].values - df.loc[obs_var, var].values)
            mean_error = float(np.nanmean(recon_error))
            var_range = float(df[var].max() - df[var].min()) if df[var].notna().any() else 1.0
            if var_range < 1e-10:
                var_range = 1.0
            relative_error = mean_error / var_range
            conf = max(0.1, min(0.9, 1.0 - relative_error * 2))
        else:
            conf = 0.3

        confidence[mask] = conf
        results[var] = (estimated, confidence)

    return results


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


def _ensemble_estimates(
    mice_results: dict[str, tuple[pd.Series, pd.Series]],
    gp_results: dict[str, tuple[pd.Series, pd.Series]],
    mc_results: dict[str, tuple[pd.Series, pd.Series]],
    variables: list[str],
    df: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, str]]:
    """Combine estimates from three methods via confidence-weighted average.

    Returns (estimated_values, confidence_scores, methods_used) dicts.
    """
    estimated_out: dict[str, pd.Series] = {}
    confidence_out: dict[str, pd.Series] = {}
    methods_out: dict[str, str] = {}

    for var in variables:
        estimates = []
        weights = []
        method_names = []

        for name, res in [("mice", mice_results), ("gp", gp_results), ("mc", mc_results)]:
            if var in res:
                est, conf = res[var]
                estimates.append(est)
                weights.append(conf)
                method_names.append(name)

        if not estimates:
            estimated_out[var] = pd.Series(np.nan, index=df.index)
            confidence_out[var] = pd.Series(np.nan, index=df.index)
            methods_out[var] = "none"
            continue

        if len(estimates) == 1:
            estimated_out[var] = estimates[0]
            confidence_out[var] = weights[0]
            methods_out[var] = method_names[0]
            continue

        # Weighted average: weight by confidence
        est_stack = pd.DataFrame(
            {n: e for n, e in zip(method_names, estimates)},
            index=df.index,
        )
        wt_stack = pd.DataFrame(
            {n: w for n, w in zip(method_names, weights)},
            index=df.index,
        )

        # For each row, compute weighted average across available methods
        wt_sum = wt_stack.sum(axis=1).replace(0, np.nan)
        weighted_est = (est_stack * wt_stack).sum(axis=1) / wt_sum

        # Confidence: average confidence penalized by disagreement
        avg_conf = wt_stack.mean(axis=1)
        # Disagreement: normalized std of estimates
        est_std = est_stack.std(axis=1)
        est_mean_abs = est_stack.mean(axis=1).abs()
        relative_disagreement = est_std / est_mean_abs.replace(0, 1)
        disagreement_penalty = np.clip(relative_disagreement, 0, 0.5)
        final_conf = (avg_conf - disagreement_penalty).clip(0.05, 0.95)

        estimated_out[var] = weighted_est
        confidence_out[var] = final_conf
        methods_out[var] = "+".join(method_names)

    return estimated_out, confidence_out, methods_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_missing_data(
    df: pd.DataFrame,
    variables: list[str],
    mar_masks: dict[str, pd.Series] | None = None,
    tier_membership: dict[str, int] | None = None,
) -> MAREstimationResult:
    """Estimate missing values classified as MAR/MCAR.

    Parameters
    ----------
    df:
        Full feature table from the cache builder.
    variables:
        Variable names to estimate.
    mar_masks:
        Per-variable boolean masks indicating which rows are MAR.
        If None, all NaN values are treated as MAR.
    tier_membership:
        Variable -> tier number mapping for weighting.

    Returns
    -------
    MAREstimationResult
        Estimated values and confidence scores per variable.
    """
    result = MAREstimationResult()

    # Filter to variables with missing data
    vars_with_missing = [
        v for v in variables
        if v in df.columns and df[v].isna().any()
    ]
    if not vars_with_missing:
        logger.info("No MAR missing values to estimate")
        return result

    # Default: all NaN is MAR
    if mar_masks is None:
        mar_masks = {v: df[v].isna() for v in vars_with_missing}

    # Select feature columns (numeric, non-flag, decent coverage)
    feature_cols = _select_feature_columns(df, vars_with_missing)
    if not feature_cols:
        logger.warning("No suitable feature columns for MAR estimation")
        return result

    logger.info(
        "Running MAR estimator for %d variables using %d features",
        len(vars_with_missing), len(feature_cols),
    )

    # Run all three methods
    mice_results = _run_mice(df, vars_with_missing, feature_cols, mar_masks)
    gp_results = _run_gp(df, vars_with_missing, feature_cols, mar_masks)
    mc_results = _run_matrix_completion(df, vars_with_missing, feature_cols, mar_masks)

    # Ensemble
    est, conf, methods = _ensemble_estimates(
        mice_results, gp_results, mc_results,
        vars_with_missing, df,
    )

    result.estimated_values = est
    result.confidence_scores = conf
    result.methods_used = methods
    result.n_estimated = sum(
        int(v.notna().sum()) for v in est.values()
    )

    logger.info(
        "MAR estimation complete: %d values estimated across %d variables",
        result.n_estimated, len(vars_with_missing),
    )

    return result


def _select_feature_columns(
    df: pd.DataFrame,
    target_vars: list[str],
) -> list[str]:
    """Select numeric, non-flag columns with decent coverage for features.

    Delegates to the shared ``select_feature_columns`` in the estimation
    package ``__init__`` to avoid code duplication with hidden_data_estimator.
    """
    from operator1.estimation import select_feature_columns
    return select_feature_columns(df, target_vars)

"""T5.1 -- Post-cache Sudoku inference (estimation engine).

Three-phase estimation that fills missing values in the full feature
table **without** any additional API calls.

**Phase 1 -- Deterministic identity fill:**
Uses accounting identities to solve for missing values when two of
three related variables are observed:
  - ``total_assets = total_liabilities + total_equity``
  - ``free_cash_flow = operating_cash_flow - abs(capex)``
  - ``net_debt = total_debt_asof - cash_and_equivalents``
  - ``gross_profit = revenue - cost_of_revenue`` (if available)
  - ``ebit = revenue - operating_expenses`` (if available)

**Phase 2 -- Missingness classification:**
Classifies remaining NaN values as MAR (Missing At Random) or
MNAR (Missing Not At Random / hidden data) using peer coverage,
reporting history, and aggregation heuristics.

**Phase 3 -- Specialized estimation:**

  *MAR/MCAR path* (``missing_data_estimator``):
    - MICE (Multiple Imputation by Chained Equations)
    - Gaussian Process regression with posterior uncertainty
    - Matrix Completion via nuclear norm minimization (Soft-Impute)
    - Ensemble: confidence-weighted average of the three methods

  *MNAR path* (``hidden_data_estimator``):
    - Heckman Selection Model (two-stage, corrects selection bias)
    - Pattern-Mixture Model (separate distributions per pattern)
    - Sensitivity Bounds (tipping-point analysis)
    - GAIN adversarial imputation (optional, requires torch)
    - Ensemble: priority-weighted (Heckman > Pattern > GAIN)

  Legacy backends (``"bayesian_ridge"``, ``"vae"``) are still
  supported via ``estimation_imputer`` config key for backward
  compatibility.

Output columns per estimated variable ``x``:
  - ``x_observed``: original observed value (NaN if was missing)
  - ``x_estimated``: model-estimated value (NaN if was observed)
  - ``x_final``: best available (observed preferred over estimated)
  - ``x_source``: ``"observed"``, ``"estimated_mar"``, or ``"estimated_mnar"``
  - ``x_confidence``: confidence score in [0, 1]
  - ``x_missingness_type``: ``"mar"``, ``"mcar"``, ``"mnar"``, or ``"observed"``
  - ``x_estimation_method``: which model(s) produced the estimate
  - ``x_sensitivity_lower``: lower bound (MNAR only)
  - ``x_sensitivity_upper``: upper bound (MNAR only)

Observed values are **never** overwritten.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import CACHE_DIR
from operator1.config_loader import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Accounting identities for Pass 1
# ---------------------------------------------------------------------------

# Each identity is (result_var, component_a, component_b, operation).
# If operation is "add": result = a + b
# If operation is "sub": result = a - b
# Solving: if any ONE is missing but the other two are present, fill it.

_ACCOUNTING_IDENTITIES: list[tuple[str, str, str, str]] = [
    # total_assets = total_liabilities + total_equity
    ("total_assets", "total_liabilities", "total_equity", "add"),
    # free_cash_flow = operating_cash_flow - abs(capex)
    ("free_cash_flow", "operating_cash_flow", "capex_abs", "sub"),
    # net_debt = total_debt_asof - cash_and_equivalents
    ("net_debt", "total_debt_asof", "cash_and_equivalents", "sub"),
    # total_debt_asof = short_term_debt + long_term_debt
    ("total_debt_asof", "short_term_debt", "long_term_debt", "add"),
]


# Variables eligible for estimation in Pass 2
ESTIMABLE_VARIABLES: tuple[str, ...] = (
    # Statement fields
    "revenue", "gross_profit", "ebit", "ebitda", "net_income",
    "interest_expense", "taxes",
    "total_assets", "total_liabilities", "total_equity",
    "current_assets", "current_liabilities",
    "cash_and_equivalents", "short_term_debt", "long_term_debt",
    "receivables",
    "operating_cash_flow", "capex",
    # Derived variables
    "total_debt_asof", "net_debt",
    "free_cash_flow", "free_cash_flow_ttm_asof",
)

# Minimum number of observed data points required before we attempt
# model-based estimation for a variable.
_MIN_OBSERVED_FOR_MODEL = 10

# Minimum rolling window for training
_MIN_TRAIN_WINDOW = 20


# ---------------------------------------------------------------------------
# Tier membership lookup
# ---------------------------------------------------------------------------


def _build_tier_membership(
    config: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Build a mapping from variable name to tier number (1-5).

    Variables not in any tier get tier 0 (lowest priority).
    """
    if config is None:
        config = load_config("survival_hierarchy")

    tiers = config.get("tiers", {})
    membership: dict[str, int] = {}
    for tier_key, tier_info in tiers.items():
        tier_num = int(tier_key.replace("tier", ""))
        for var in tier_info.get("variables", []):
            membership[var] = tier_num

    return membership


# ---------------------------------------------------------------------------
# Pass 1: Deterministic identity fill
# ---------------------------------------------------------------------------


@dataclass
class IdentityFillResult:
    """Summary of Pass 1 deterministic fills."""
    fills: dict[str, int] = field(default_factory=dict)  # var -> count of rows filled
    total_filled: int = 0


def _fill_identity(
    df: pd.DataFrame,
    result_var: str,
    comp_a: str,
    comp_b: str,
    operation: str,
) -> int:
    """Apply a single accounting identity to fill missing values.

    Returns the number of rows filled.
    """
    # Handle capex_abs specially
    if comp_b == "capex_abs":
        if "capex" in df.columns:
            b_series = df["capex"].abs()
        else:
            return 0
    else:
        if comp_b not in df.columns:
            return 0
        b_series = df[comp_b]

    if comp_a not in df.columns or result_var not in df.columns:
        return 0

    a_series = df[comp_a]
    r_series = df[result_var]

    filled = 0

    if operation == "add":
        # result = a + b
        # If result missing but a and b present: fill result
        mask_r = r_series.isna() & a_series.notna() & b_series.notna()
        if mask_r.any():
            df.loc[mask_r, result_var] = a_series[mask_r] + b_series[mask_r]
            filled += mask_r.sum()

        # If a missing but result and b present: a = result - b
        mask_a = a_series.isna() & r_series.notna() & b_series.notna()
        if mask_a.any():
            df.loc[mask_a, comp_a] = r_series[mask_a] - b_series[mask_a]
            filled += mask_a.sum()

        # If b missing but result and a present: b = result - a
        if comp_b != "capex_abs":
            mask_b = b_series.isna() & r_series.notna() & a_series.notna()
            if mask_b.any():
                df.loc[mask_b, comp_b] = r_series[mask_b] - a_series[mask_b]
                filled += mask_b.sum()

    elif operation == "sub":
        # result = a - b
        # If result missing but a and b present: fill result
        mask_r = r_series.isna() & a_series.notna() & b_series.notna()
        if mask_r.any():
            df.loc[mask_r, result_var] = a_series[mask_r] - b_series[mask_r]
            filled += mask_r.sum()

        # If a missing but result and b present: a = result + b
        mask_a = a_series.isna() & r_series.notna() & b_series.notna()
        if mask_a.any():
            df.loc[mask_a, comp_a] = r_series[mask_a] + b_series[mask_a]
            filled += mask_a.sum()

        # If b missing but result and a present: b = a - result
        if comp_b != "capex_abs":
            mask_b = b_series.isna() & r_series.notna() & a_series.notna()
            if mask_b.any():
                df.loc[mask_b, comp_b] = a_series[mask_b] - r_series[mask_b]
                filled += mask_b.sum()

    return filled


def run_pass1_identity_fill(df: pd.DataFrame) -> IdentityFillResult:
    """Run Pass 1: deterministic accounting identity fill.

    Iterates through all accounting identities multiple times until
    no more fills are possible (cascading fills).

    Parameters
    ----------
    df:
        Feature table (modified in place).

    Returns
    -------
    IdentityFillResult
        Summary of fills performed.
    """
    result = IdentityFillResult()
    max_iterations = 5  # prevent infinite loops

    for iteration in range(max_iterations):
        round_fills = 0
        for res_var, comp_a, comp_b, op in _ACCOUNTING_IDENTITIES:
            n = _fill_identity(df, res_var, comp_a, comp_b, op)
            if n > 0:
                result.fills[res_var] = result.fills.get(res_var, 0) + n
                round_fills += n
                logger.debug(
                    "Pass 1 iter %d: filled %d rows for '%s'",
                    iteration, n, res_var,
                )

        if round_fills == 0:
            break
        result.total_filled += round_fills

    # Update is_missing flags for filled variables
    for var in result.fills:
        flag_col = f"is_missing_{var}"
        if flag_col in df.columns:
            df[flag_col] = df[var].isna().astype(int)

    logger.info(
        "Pass 1 complete: %d total cells filled across %d variables",
        result.total_filled, len(result.fills),
    )

    return result


# ---------------------------------------------------------------------------
# Pass 2: Regime-weighted rolling imputer
# ---------------------------------------------------------------------------


def _get_feature_columns(
    df: pd.DataFrame,
    target_var: str,
) -> list[str]:
    """Select feature columns for predicting ``target_var``.

    Uses numeric non-flag columns that are not the target itself and
    have reasonable coverage.
    """
    candidates = []
    for col in df.columns:
        if col == target_var:
            continue
        if col.startswith("is_missing_") or col.startswith("invalid_math_"):
            continue
        if col.endswith("_source") or col.endswith("_confidence"):
            continue
        if col.endswith("_observed") or col.endswith("_estimated"):
            continue
        if df[col].dtype not in ("float64", "float32", "int64", "int32"):
            continue
        # Need at least 50% coverage
        coverage = df[col].notna().mean()
        if coverage < 0.5:
            continue
        candidates.append(col)

    return candidates


def _estimate_variable_rolling(
    df: pd.DataFrame,
    var: str,
    tier_weights: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Estimate missing values for a single variable using rolling BayesianRidge.

    Parameters
    ----------
    df:
        Feature table.
    var:
        Target variable to estimate.
    tier_weights:
        Per-day sample weights for the tier this variable belongs to.

    Returns
    -------
    (estimated_values, confidence_scores)
        Both indexed like ``df``.  NaN where not estimated.
    """
    estimated = pd.Series(np.nan, index=df.index, dtype=float)
    confidence = pd.Series(np.nan, index=df.index, dtype=float)

    target = df[var]
    missing_mask = target.isna()

    if not missing_mask.any():
        return estimated, confidence

    feature_cols = _get_feature_columns(df, var)
    if not feature_cols:
        logger.debug("No feature columns for '%s' -- skipping estimation", var)
        return estimated, confidence

    # Limit to a manageable number of features
    feature_cols = feature_cols[:15]

    try:
        from sklearn.linear_model import BayesianRidge
    except ImportError:
        logger.warning("sklearn not available -- falling back to mean imputation")
        mean_val = target.mean()
        estimated[missing_mask] = mean_val
        confidence[missing_mask] = 0.3
        return estimated, confidence

    # Forward pass: for each missing day, train on observed data up to that point
    n = len(df)
    observed_indices = df.index[~missing_mask]
    missing_indices = df.index[missing_mask]

    if len(observed_indices) < _MIN_OBSERVED_FOR_MODEL:
        # Too few observations -- use simple mean
        mean_val = target.mean()
        estimated[missing_mask] = mean_val
        confidence[missing_mask] = 0.2
        return estimated, confidence

    # Build feature matrix (forward-filled to handle NaN features)
    X_full = df[feature_cols].ffill().bfill()

    # For efficiency, we don't retrain for every single day.
    # Instead, we train once using all observed data up to each
    # missing day's position, but batch missing days that share
    # the same training window.
    #
    # Simplified approach: train on all observed data (no look-ahead
    # since observed data is by definition available), then predict
    # missing days. Confidence is based on how much training data
    # was available relative to the missing day's position.

    # Find positional indices
    all_positions = {idx: pos for pos, idx in enumerate(df.index)}

    # For a stricter no-look-ahead approach, we'd train per-day.
    # For efficiency with large datasets, we group by "training window"
    # and retrain periodically.

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for mi in missing_indices:
            pos = all_positions[mi]

            # Training data: only observed rows BEFORE this position
            train_mask = (~missing_mask) & (pd.Series(range(n), index=df.index) < pos)

            if train_mask.sum() < _MIN_OBSERVED_FOR_MODEL:
                # Not enough prior data -- use expanding mean
                prior_data = target.iloc[:pos].dropna()
                if len(prior_data) > 0:
                    estimated.loc[mi] = prior_data.mean()
                    confidence.loc[mi] = min(0.3, len(prior_data) / 50)
                continue

            X_train = X_full.loc[train_mask]
            y_train = target.loc[train_mask]

            # Apply tier weights if available
            sample_weight = None
            if tier_weights is not None:
                sw = tier_weights.loc[train_mask]
                if sw.notna().any() and (sw > 0).any():
                    sample_weight = sw.fillna(sw.mean()).values

            try:
                model = BayesianRidge(
                    max_iter=100,
                    tol=1e-4,
                    compute_score=False,
                )
                model.fit(
                    X_train.values, y_train.values,
                    sample_weight=sample_weight,
                )

                X_pred = X_full.loc[[mi]].values
                y_pred = model.predict(X_pred)
                estimated.loc[mi] = y_pred[0]

                # Confidence based on model score and training size
                train_frac = min(1.0, train_mask.sum() / max(n, 1))
                try:
                    r2 = model.score(X_train.values, y_train.values)
                    conf = max(0.1, min(0.95, r2 * train_frac))
                except Exception:
                    conf = train_frac * 0.5
                confidence.loc[mi] = conf

            except Exception as exc:
                logger.debug(
                    "BayesianRidge failed for '%s' at pos %d: %s",
                    var, pos, exc,
                )
                # Fallback: expanding mean
                prior_data = target.iloc[:pos].dropna()
                if len(prior_data) > 0:
                    estimated.loc[mi] = prior_data.mean()
                    confidence.loc[mi] = 0.2

    n_estimated = estimated.notna().sum()
    if n_estimated > 0:
        logger.debug(
            "Estimated %d values for '%s' (avg confidence: %.3f)",
            n_estimated, var,
            confidence.dropna().mean() if confidence.notna().any() else 0,
        )

    return estimated, confidence


# ---------------------------------------------------------------------------
# Output column builders
# ---------------------------------------------------------------------------


def _build_estimation_columns(
    df: pd.DataFrame,
    var: str,
    estimated: pd.Series,
    confidence: pd.Series,
    missingness_type: pd.Series | None = None,
    estimation_method: str | None = None,
    sensitivity_lower: pd.Series | None = None,
    sensitivity_upper: pd.Series | None = None,
) -> None:
    """Add estimation output columns for a variable.

    Core columns (always present):
      - ``{var}_observed``: original value
      - ``{var}_estimated``: model estimate (NaN where observed)
      - ``{var}_final``: best available
      - ``{var}_source``: ``"observed"``, ``"estimated_mar"``, or ``"estimated_mnar"``
      - ``{var}_confidence``: [0, 1]

    Extended columns (when split estimator is active):
      - ``{var}_missingness_type``: ``"mar"``, ``"mcar"``, ``"mnar"``, ``"observed"``
      - ``{var}_estimation_method``: model(s) that produced the estimate
      - ``{var}_sensitivity_lower``: pessimistic bound (MNAR only)
      - ``{var}_sensitivity_upper``: optimistic bound (MNAR only)
    """
    observed = df[var].copy()

    # Final = observed where available, estimated where not
    final = observed.copy()
    still_missing = final.isna()
    final[still_missing] = estimated[still_missing]

    # Source: distinguish MAR vs MNAR estimates
    source = pd.Series("observed", index=df.index, dtype="object")
    estimated_mask = still_missing & estimated.notna()
    if missingness_type is not None:
        mar_est = estimated_mask & missingness_type.isin(["mar", "mcar"])
        mnar_est = estimated_mask & (missingness_type == "mnar")
        source[mar_est] = "estimated_mar"
        source[mnar_est] = "estimated_mnar"
        # Any estimated without classification -> generic
        other_est = estimated_mask & ~mar_est & ~mnar_est
        source[other_est] = "estimated"
    else:
        source[estimated_mask] = "estimated"

    # Confidence: 1.0 for observed, model confidence for estimated
    conf = pd.Series(1.0, index=df.index)
    conf[estimated_mask] = confidence[estimated_mask]
    conf[still_missing & estimated.isna()] = 0.0

    # Build all columns at once for performance
    cols_dict = {
        f"{var}_observed": observed,
        f"{var}_estimated": estimated,
        f"{var}_final": final,
        f"{var}_source": source,
        f"{var}_confidence": conf,
    }

    # Extended columns
    if missingness_type is not None:
        cols_dict[f"{var}_missingness_type"] = missingness_type
    if estimation_method is not None:
        method_series = pd.Series("observed", index=df.index, dtype="object")
        method_series[estimated_mask] = estimation_method
        cols_dict[f"{var}_estimation_method"] = method_series
    if sensitivity_lower is not None:
        cols_dict[f"{var}_sensitivity_lower"] = sensitivity_lower
    if sensitivity_upper is not None:
        cols_dict[f"{var}_sensitivity_upper"] = sensitivity_upper

    new_cols = pd.DataFrame(cols_dict, index=df.index)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")
        try:
            df[new_cols.columns] = new_cols
        except ValueError:
            # Fallback: assign columns one at a time to avoid shape mismatch
            # when existing columns overlap with new ones in unexpected ways.
            for col_name in new_cols.columns:
                try:
                    df[col_name] = new_cols[col_name]
                except Exception as exc:
                    logger.warning("Failed to assign estimation column %s: %s", col_name, exc)


# ---------------------------------------------------------------------------
# Coverage logging
# ---------------------------------------------------------------------------


@dataclass
class EstimationCoverage:
    """Coverage statistics for the estimation process."""

    pass1_fills: dict[str, int] = field(default_factory=dict)
    pass2_estimates: dict[str, int] = field(default_factory=dict)
    coverage_before: dict[str, float] = field(default_factory=dict)
    coverage_after: dict[str, float] = field(default_factory=dict)
    coverage_per_tier: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass1_fills": self.pass1_fills,
            "pass2_estimates": self.pass2_estimates,
            "coverage_before": {
                k: round(v, 4) for k, v in self.coverage_before.items()
            },
            "coverage_after": {
                k: round(v, 4) for k, v in self.coverage_after.items()
            },
            "coverage_per_tier": self.coverage_per_tier,
        }


# ---------------------------------------------------------------------------
# Pass 2 backend helpers
# ---------------------------------------------------------------------------


def _run_pass2_bayesian_ridge(
    result: pd.DataFrame,
    variables: list[str],
    tier_membership: dict[str, int],
    tier_weight_columns: dict[int, str],
    coverage: EstimationCoverage,
) -> None:
    """Run the BayesianRidge rolling imputer for *variables* in-place."""
    for var in variables:
        if var not in result.columns:
            continue

        tier_num = tier_membership.get(var, 0)
        tier_weights = None
        if tier_num > 0 and tier_num in tier_weight_columns:
            tier_weights = result[tier_weight_columns[tier_num]]

        estimated, confidence = _estimate_variable_rolling(
            result, var, tier_weights,
        )

        n_estimated = estimated.notna().sum()
        if n_estimated > 0:
            coverage.pass2_estimates[var] = int(n_estimated)

        _build_estimation_columns(result, var, estimated, confidence)


def _run_pass2_vae(
    result: pd.DataFrame,
    variables: list[str],
    tier_membership: dict[str, int],
    tier_weight_columns: dict[int, str],
    coverage: EstimationCoverage,
) -> bool:
    """Run the VAE imputer for *variables* in-place.

    Returns True if the VAE succeeded, False if it fell back.
    """
    try:
        from operator1.estimation.vae_imputer import (
            train_and_impute_vae,
            _check_torch_available,
        )
    except ImportError:
        logger.warning("VAE imputer module not available -- falling back to BayesianRidge")
        _run_pass2_bayesian_ridge(
            result, variables, tier_membership, tier_weight_columns, coverage,
        )
        return False

    if not _check_torch_available():
        logger.warning("torch not installed -- falling back to BayesianRidge")
        _run_pass2_bayesian_ridge(
            result, variables, tier_membership, tier_weight_columns, coverage,
        )
        return False

    # Select feature columns (same logic as BayesianRidge path)
    feature_cols = _get_feature_columns(result, variables[0])

    # Build tier weight mapping
    tier_weights_map: dict[str, pd.Series] = {}
    for var in variables:
        tier_num = tier_membership.get(var, 0)
        if tier_num > 0 and tier_num in tier_weight_columns:
            tier_weights_map[var] = result[tier_weight_columns[tier_num]]

    # Load VAE hyperparameters from config
    try:
        global_cfg = load_config("global_config")
    except Exception:
        global_cfg = {}

    try:
        vae_result = train_and_impute_vae(
            df=result,
            target_vars=variables,
            feature_cols=feature_cols,
            tier_weights=tier_weights_map if tier_weights_map else None,
            latent_dim=int(global_cfg.get("vae_latent_dim", 8)),
            hidden_dim=int(global_cfg.get("vae_hidden_dim", 32)),
            epochs=int(global_cfg.get("vae_epochs", 80)),
            kl_weight=float(global_cfg.get("vae_kl_weight", 0.5)),
        )
    except Exception as exc:
        logger.warning("VAE training failed (%s) -- falling back to BayesianRidge", exc)
        _run_pass2_bayesian_ridge(
            result, variables, tier_membership, tier_weight_columns, coverage,
        )
        return False

    if vae_result.fallback_used:
        logger.info("VAE had insufficient data -- falling back to BayesianRidge")
        _run_pass2_bayesian_ridge(
            result, variables, tier_membership, tier_weight_columns, coverage,
        )
        return False

    # Apply VAE estimates to the result DataFrame
    for var in variables:
        estimated = vae_result.estimated_values.get(
            var, pd.Series(np.nan, index=result.index),
        )
        confidence = vae_result.confidence_scores.get(
            var, pd.Series(np.nan, index=result.index),
        )

        n_estimated = estimated.notna().sum()
        if n_estimated > 0:
            coverage.pass2_estimates[var] = int(n_estimated)

        _build_estimation_columns(result, var, estimated, confidence)

    logger.info(
        "VAE Pass 2 complete: train_loss=%.6f, %d train rows, %d features",
        vae_result.train_loss_final,
        vae_result.n_train_rows,
        vae_result.n_features_used,
    )
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_estimation(
    df: pd.DataFrame,
    variables: tuple[str, ...] | list[str] | None = None,
    hierarchy_config: dict[str, Any] | None = None,
    imputer_method: str | None = None,
) -> tuple[pd.DataFrame, EstimationCoverage]:
    """Run the full three-phase estimation pipeline.

    Parameters
    ----------
    df:
        Full feature table.  Modified in-place for Phase 1; a copy is
        returned with estimation columns added.
    variables:
        List of variable names to estimate.  Defaults to
        ``ESTIMABLE_VARIABLES``.
    hierarchy_config:
        Override survival hierarchy config.
    imputer_method:
        Imputer backend.  Options:
          - ``"split"`` (default): classify missingness, route to
            MAR estimator (MICE+GP+MatrixCompletion) or MNAR estimator
            (Heckman+PatternMixture+GAIN).
          - ``"bayesian_ridge"``: legacy per-variable BayesianRidge.
          - ``"vae"``: legacy VAE imputer.
        When *None*, reads from ``global_config.yml`` key
        ``estimation_imputer``.

    Returns
    -------
    (result_df, coverage)
        Augmented DataFrame and coverage statistics.
    """
    if variables is None:
        variables = list(ESTIMABLE_VARIABLES)

    # Filter to variables actually present
    variables = [v for v in variables if v in df.columns]

    # Resolve imputer method from config if not specified
    if imputer_method is None:
        try:
            global_cfg = load_config("global_config")
            imputer_method = global_cfg.get("estimation_imputer", "split")
        except Exception:
            imputer_method = "split"

    result = df.copy()
    coverage = EstimationCoverage()

    # Record pre-estimation coverage
    n = len(result)
    for var in variables:
        if n > 0:
            coverage.coverage_before[var] = float(result[var].notna().sum() / n)
        else:
            coverage.coverage_before[var] = 0.0

    # ------------------------------------------------------------------
    # Phase 1: Deterministic identity fill
    # ------------------------------------------------------------------
    logger.info("Running Phase 1: Deterministic identity fill ...")
    p1_result = run_pass1_identity_fill(result)
    coverage.pass1_fills = p1_result.fills

    # ------------------------------------------------------------------
    # Build tier membership for weighting
    # ------------------------------------------------------------------
    tier_membership = _build_tier_membership(hierarchy_config)
    tier_weight_columns: dict[int, str] = {}
    for i in range(1, 6):
        col = f"hierarchy_tier{i}_weight"
        if col in result.columns:
            tier_weight_columns[i] = col

    # Determine which variables need imputation
    vars_needing_imputation = [
        v for v in variables
        if v in result.columns and result[v].isna().any()
    ]
    vars_already_complete = [
        v for v in variables
        if v in result.columns and not result[v].isna().any()
    ]

    # Set observation columns for complete variables
    for var in vars_already_complete:
        _build_estimation_columns(
            result, var,
            pd.Series(np.nan, index=result.index),
            pd.Series(np.nan, index=result.index),
        )

    # ------------------------------------------------------------------
    # Phase 2 + 3: Split estimator (new default)
    # ------------------------------------------------------------------
    if imputer_method == "split" and vars_needing_imputation:
        _run_split_estimation(
            result, vars_needing_imputation, tier_membership,
            tier_weight_columns, coverage,
        )

    # ------------------------------------------------------------------
    # Legacy paths (backward compatibility)
    # ------------------------------------------------------------------
    elif imputer_method == "vae" and vars_needing_imputation:
        logger.info(
            "Running legacy VAE imputer for %d variables ...",
            len(vars_needing_imputation),
        )
        vae_succeeded = _run_pass2_vae(
            result, vars_needing_imputation, tier_membership,
            tier_weight_columns, coverage,
        )
        if not vae_succeeded:
            remaining = [
                v for v in vars_needing_imputation
                if f"{v}_final" not in result.columns
            ]
            if remaining:
                _run_pass2_bayesian_ridge(
                    result, remaining, tier_membership,
                    tier_weight_columns, coverage,
                )

    elif imputer_method == "bayesian_ridge" and vars_needing_imputation:
        logger.info(
            "Running legacy BayesianRidge imputer for %d variables ...",
            len(vars_needing_imputation),
        )
        _run_pass2_bayesian_ridge(
            result, vars_needing_imputation, tier_membership,
            tier_weight_columns, coverage,
        )

    # ------------------------------------------------------------------
    # Record post-estimation coverage
    # ------------------------------------------------------------------
    for var in variables:
        final_col = f"{var}_final"
        if final_col in result.columns and n > 0:
            coverage.coverage_after[var] = float(
                result[final_col].notna().sum() / n
            )
        else:
            coverage.coverage_after[var] = coverage.coverage_before.get(var, 0.0)

    # Coverage per tier
    for tier_num in range(1, 6):
        tier_key = f"tier{tier_num}"
        tier_vars = [
            v for v, t in tier_membership.items()
            if t == tier_num and f"{v}_final" in result.columns
        ]
        if tier_vars and n > 0:
            tier_cov = {
                v: float(result[f"{v}_final"].notna().sum() / n)
                for v in tier_vars
            }
            coverage.coverage_per_tier[tier_key] = {
                k: round(v, 4) for k, v in tier_cov.items()
            }

    logger.info(
        "Estimation complete: Phase1 filled %d cells, Phase2+3 estimated %d variables",
        p1_result.total_filled,
        len(coverage.pass2_estimates),
    )

    # Defragment the DataFrame after adding many estimation columns.
    # This eliminates PerformanceWarning from downstream consumers.
    result = result.copy()

    return result, coverage


# ---------------------------------------------------------------------------
# Split estimation orchestrator
# ---------------------------------------------------------------------------


def _run_split_estimation(
    result: pd.DataFrame,
    variables: list[str],
    tier_membership: dict[str, int],
    tier_weight_columns: dict[int, str],
    coverage: EstimationCoverage,
) -> None:
    """Run the split estimator: classify -> route -> merge.

    Phase 2: Classify each NaN as MAR or MNAR.
    Phase 3a: Route MAR values to missing_data_estimator.
    Phase 3b: Route MNAR values to hidden_data_estimator.
    Merge: Combine results into the estimation output columns.
    """
    from operator1.estimation.missingness_classifier import classify_missingness
    from operator1.estimation.missing_data_estimator import estimate_missing_data
    from operator1.estimation.hidden_data_estimator import estimate_hidden_data

    # --- Phase 2: Classify missingness ---
    logger.info(
        "Running Phase 2: Missingness classification for %d variables ...",
        len(variables),
    )
    classification = classify_missingness(result, variables)

    # Split variables by missingness type
    mar_masks: dict[str, pd.Series] = {}
    mnar_masks: dict[str, pd.Series] = {}
    vars_with_mar: list[str] = []
    vars_with_mnar: list[str] = []

    for var in variables:
        mar_m = classification.mar_mask(var, result)
        mnar_m = classification.mnar_mask(var, result)

        if mar_m.any():
            mar_masks[var] = mar_m
            vars_with_mar.append(var)
        if mnar_m.any():
            mnar_masks[var] = mnar_m
            vars_with_mnar.append(var)

    logger.info(
        "Classification result: %d vars with MAR, %d vars with MNAR",
        len(vars_with_mar), len(vars_with_mnar),
    )

    # --- Phase 3a: MAR estimation ---
    mar_result = None
    if vars_with_mar:
        logger.info(
            "Running Phase 3a: MAR estimator (MICE+GP+MatrixCompletion) "
            "for %d variables ...", len(vars_with_mar),
        )
        mar_result = estimate_missing_data(
            result, vars_with_mar, mar_masks, tier_membership,
        )

    # --- Phase 3b: MNAR estimation ---
    mnar_result = None
    if vars_with_mnar:
        logger.info(
            "Running Phase 3b: MNAR estimator (Heckman+Pattern+GAIN) "
            "for %d variables ...", len(vars_with_mnar),
        )
        mnar_result = estimate_hidden_data(
            result, vars_with_mnar, mnar_masks,
        )

    # --- Merge results into output columns ---
    for var in variables:
        # Combine MAR and MNAR estimates for this variable
        estimated = pd.Series(np.nan, index=result.index, dtype=float)
        confidence = pd.Series(np.nan, index=result.index, dtype=float)
        sensitivity_lower = None
        sensitivity_upper = None
        method_name = "none"
        methods_list = []

        # MAR estimates
        if mar_result and var in mar_result.estimated_values:
            mar_est = mar_result.estimated_values[var]
            mar_conf = mar_result.confidence_scores[var]
            mar_mask = mar_masks.get(var, pd.Series(False, index=result.index))

            fill_mask = mar_mask & mar_est.notna()
            estimated[fill_mask] = mar_est[fill_mask]
            confidence[fill_mask] = mar_conf[fill_mask]
            methods_list.append(mar_result.methods_used.get(var, "mar"))

        # MNAR estimates
        if mnar_result and var in mnar_result.estimated_values:
            mnar_est = mnar_result.estimated_values[var]
            mnar_conf = mnar_result.confidence_scores[var]
            mnar_mask = mnar_masks.get(var, pd.Series(False, index=result.index))

            fill_mask = mnar_mask & mnar_est.notna()
            estimated[fill_mask] = mnar_est[fill_mask]
            confidence[fill_mask] = mnar_conf[fill_mask]
            methods_list.append(mnar_result.methods_used.get(var, "mnar"))

            # Sensitivity bounds (MNAR only)
            if var in mnar_result.sensitivity_lower:
                sensitivity_lower = mnar_result.sensitivity_lower[var]
            if var in mnar_result.sensitivity_upper:
                sensitivity_upper = mnar_result.sensitivity_upper[var]

        method_name = "+".join(methods_list) if methods_list else "none"

        # Track coverage
        n_estimated = int(estimated.notna().sum())
        if n_estimated > 0:
            coverage.pass2_estimates[var] = n_estimated

        # Get missingness type from classifier
        miss_type = classification.types.get(var)

        _build_estimation_columns(
            result, var, estimated, confidence,
            missingness_type=miss_type,
            estimation_method=method_name,
            sensitivity_lower=sensitivity_lower,
            sensitivity_upper=sensitivity_upper,
        )


def save_estimation_coverage(
    coverage: EstimationCoverage,
    output_path: str | None = None,
) -> str:
    """Persist estimation coverage to ``cache/estimation_coverage.json``.

    Parameters
    ----------
    coverage:
        Output of ``run_estimation()``.
    output_path:
        Override output file path.

    Returns
    -------
    str
        Path to the written JSON file.
    """
    if output_path is None:
        output_path = str(Path(CACHE_DIR) / "estimation_coverage.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(coverage.to_dict(), fh, indent=2)

    logger.info("Estimation coverage saved to %s", output_path)
    return output_path

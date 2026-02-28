"""Hidden Data Estimator -- handles MNAR (Missing Not At Random) missingness.

Handles data that is deliberately concealed, aggregated, or omitted.
The missingness mechanism is *informative*: companies that hide data
tend to have worse underlying values.

Four-pass approach:
  1. **Heckman Selection Model**: Two-stage estimator that models the
     selection mechanism (why data is hidden) and corrects for
     selection bias.
  2. **Pattern-Mixture Model**: Fits separate distributions for each
     missingness pattern, recognizing that hidden-data companies form
     a distinct population.
  3. **Sensitivity Bounds**: Computes tipping-point analysis -- what
     value would the hidden data need to be to change conclusions?
  4. **GAIN** (Generative Adversarial Imputation): Adversarial
     network for robust MNAR imputation.  Falls back gracefully
     if torch is unavailable.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_OBSERVED = 15
_HECKMAN_MIN_SELECTION_VARS = 3


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MNAREstimationResult:
    """Result from the hidden-data (MNAR) estimator."""

    estimated_values: dict[str, pd.Series] = field(default_factory=dict)
    confidence_scores: dict[str, pd.Series] = field(default_factory=dict)
    sensitivity_lower: dict[str, pd.Series] = field(default_factory=dict)
    sensitivity_upper: dict[str, pd.Series] = field(default_factory=dict)
    tipping_points: dict[str, pd.Series] = field(default_factory=dict)
    methods_used: dict[str, str] = field(default_factory=dict)
    n_estimated: int = 0


# ---------------------------------------------------------------------------
# Pass 1: Heckman Selection Model (two-stage)
# ---------------------------------------------------------------------------


def _build_selection_features(
    df: pd.DataFrame,
    var: str,
) -> pd.DataFrame | None:
    """Build the Z matrix for the Heckman selection equation.

    Selection variables should predict *whether* a value is observed,
    not the value itself.  Good candidates:
      - Company size proxy (market_cap, total_assets)
      - Filing frequency / data source coverage
      - Prior period reporting history
      - Peer reporting rates
    """
    candidates = []

    # Size proxy: larger companies disclose more
    for size_col in ["market_cap", "total_assets", "revenue"]:
        if size_col in df.columns and size_col != var:
            filled = df[size_col].ffill().bfill()
            if filled.notna().mean() > 0.5:
                candidates.append(filled.rename(f"sel_{size_col}"))

    # Prior period reporting: was this variable observed recently?
    if var in df.columns:
        prior_obs = df[var].notna().astype(float).shift(1).rolling(
            window=4, min_periods=1,
        ).mean()
        candidates.append(prior_obs.rename("sel_prior_reporting_rate"))

    # Overall data completeness of this row (proxy for filing quality)
    numeric_cols = df.select_dtypes(include=["float64", "float32", "int64"]).columns
    flag_cols = [c for c in numeric_cols if not c.startswith("is_missing_") and not c.startswith("invalid_math_")]
    if flag_cols:
        row_completeness = df[flag_cols].notna().mean(axis=1)
        candidates.append(row_completeness.rename("sel_row_completeness"))

    # Time trend (later periods may have better disclosure)
    time_index = pd.Series(range(len(df)), index=df.index, dtype=float)
    time_norm = time_index / max(len(df), 1)
    candidates.append(time_norm.rename("sel_time_trend"))

    if len(candidates) < _HECKMAN_MIN_SELECTION_VARS:
        return None

    Z = pd.concat(candidates, axis=1).ffill().bfill()
    return Z


def _run_heckman(
    df: pd.DataFrame,
    var: str,
    feature_cols: list[str],
    mnar_mask: pd.Series,
) -> tuple[pd.Series, pd.Series] | None:
    """Run the Heckman two-stage selection model for a single variable.

    Stage 1: Probit regression to model P(observed | Z)
    Stage 2: OLS with inverse Mills ratio correction

    Returns (estimated, confidence) or None if the model fails.
    """
    try:
        from statsmodels.discrete.discrete_model import Probit
        import statsmodels.api as sm
    except ImportError:
        logger.debug("statsmodels not available -- skipping Heckman")
        return None

    target = df[var]
    observed_mask = target.notna()

    if observed_mask.sum() < _MIN_OBSERVED:
        return None

    # Stage 1: Selection equation
    Z = _build_selection_features(df, var)
    if Z is None:
        return None

    # Dependent variable: 1 if observed, 0 if missing
    y_sel = observed_mask.astype(float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_with_const = sm.add_constant(Z.fillna(0))
            probit = Probit(y_sel, Z_with_const)
            probit_result = probit.fit(disp=0, maxiter=50)
    except Exception as exc:
        logger.debug("Heckman Stage 1 (Probit) failed for '%s': %s", var, exc)
        return None

    # Compute inverse Mills ratio: lambda = phi(Z*gamma) / Phi(Z*gamma)
    from scipy.stats import norm as normal_dist
    Z_gamma = probit_result.predict(Z_with_const)
    # Clip to avoid numerical issues at tails
    Z_gamma = np.clip(Z_gamma, 0.001, 0.999)
    inv_mills = normal_dist.pdf(normal_dist.ppf(Z_gamma)) / Z_gamma

    # Stage 2: OLS on observed data with inverse Mills ratio
    X_feat = df[feature_cols].ffill().bfill()
    usable_features = [c for c in feature_cols if c in X_feat.columns and X_feat[c].notna().mean() > 0.6]
    usable_features = usable_features[:10]

    if not usable_features:
        return None

    X_stage2 = X_feat[usable_features].copy()
    X_stage2["inv_mills_ratio"] = inv_mills
    X_stage2 = sm.add_constant(X_stage2.fillna(0))

    X_train = X_stage2.loc[observed_mask]
    y_train = target.loc[observed_mask]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols = sm.OLS(y_train, X_train)
            ols_result = ols.fit()
    except Exception as exc:
        logger.debug("Heckman Stage 2 (OLS) failed for '%s': %s", var, exc)
        return None

    # Predict missing values (with selection correction)
    X_pred = X_stage2.loc[mnar_mask]
    estimated = pd.Series(np.nan, index=df.index, dtype=float)
    confidence = pd.Series(np.nan, index=df.index, dtype=float)

    if len(X_pred) > 0:
        try:
            y_pred = ols_result.predict(X_pred)
            estimated.loc[mnar_mask] = y_pred.values

            # Confidence: based on selection model fit + OLS R-squared
            probit_accuracy = float((probit_result.predict() > 0.5).astype(float).eq(y_sel).mean())
            ols_r2 = max(0, float(ols_result.rsquared))
            # Selection correction significance
            mills_pval = float(ols_result.pvalues.get("inv_mills_ratio", 1.0))
            mills_significant = mills_pval < 0.1

            base_conf = 0.3 + probit_accuracy * 0.2 + ols_r2 * 0.2
            if mills_significant:
                base_conf += 0.1  # bonus: selection correction is meaningful
            base_conf = min(base_conf, 0.8)  # cap for MNAR (inherent uncertainty)

            confidence.loc[mnar_mask] = base_conf
        except Exception as exc:
            logger.debug("Heckman prediction failed for '%s': %s", var, exc)
            return None

    return estimated, confidence


# ---------------------------------------------------------------------------
# Pass 2: Pattern-Mixture Model
# ---------------------------------------------------------------------------


def _run_pattern_mixture(
    df: pd.DataFrame,
    variables: list[str],
    mnar_masks: dict[str, pd.Series],
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Fit separate distributions per missingness pattern.

    Companies that hide certain combinations of variables form
    distinct "patterns" with different underlying distributions.
    """
    results: dict[str, tuple[pd.Series, pd.Series]] = {}

    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        logger.debug("sklearn GMM not available -- skipping pattern-mixture")
        return results

    # Build binary missingness pattern matrix from is_missing_ columns
    missing_pattern_cols = [
        f"is_missing_{v}" for v in variables
        if f"is_missing_{v}" in df.columns
    ]

    if len(missing_pattern_cols) < 2:
        # Not enough pattern information
        return results

    pattern_matrix = df[missing_pattern_cols].fillna(1)

    # Cluster missingness patterns
    n_patterns = min(5, max(2, len(df) // 50))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm = GaussianMixture(
                n_components=n_patterns,
                covariance_type="diag",
                max_iter=50,
                random_state=42,
            )
            pattern_labels = gmm.fit_predict(pattern_matrix.values)
    except Exception as exc:
        logger.debug("Pattern-mixture clustering failed: %s", exc)
        return results

    pattern_series = pd.Series(pattern_labels, index=df.index)

    for var in variables:
        if var not in df.columns:
            continue

        mnar_mask = mnar_masks.get(var, df[var].isna())
        if not mnar_mask.any():
            continue

        estimated = pd.Series(np.nan, index=df.index, dtype=float)
        confidence = pd.Series(np.nan, index=df.index, dtype=float)

        target = df[var]

        for pattern_id in range(n_patterns):
            pattern_mask = pattern_series == pattern_id
            pattern_observed = pattern_mask & target.notna()
            pattern_missing = pattern_mask & mnar_mask

            if not pattern_missing.any():
                continue

            if pattern_observed.sum() >= 5:
                # Use pattern-specific distribution
                pattern_mean = float(target.loc[pattern_observed].mean())
                pattern_std = float(target.loc[pattern_observed].std())

                # For MNAR: shift estimate toward the pessimistic end
                # (hidden values tend to be worse than observed)
                pessimism_shift = -0.3 * pattern_std if pattern_std > 0 else 0
                adjusted_mean = pattern_mean + pessimism_shift

                estimated.loc[pattern_missing] = adjusted_mean
                # Confidence based on pattern sample size
                n_in_pattern = int(pattern_observed.sum())
                conf = min(0.7, 0.3 + n_in_pattern / 100)
                confidence.loc[pattern_missing] = conf
            else:
                # Too few observations in this pattern; use global stats
                global_mean = float(target.mean())
                global_std = float(target.std())
                pessimism_shift = -0.3 * global_std if global_std > 0 else 0
                estimated.loc[pattern_missing] = global_mean + pessimism_shift
                confidence.loc[pattern_missing] = 0.2

        results[var] = (estimated, confidence)

    return results


# ---------------------------------------------------------------------------
# Pass 3: Sensitivity Bounds (Tipping Point Analysis)
# ---------------------------------------------------------------------------


def _compute_sensitivity_bounds(
    df: pd.DataFrame,
    var: str,
    mnar_mask: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute pessimistic/optimistic bounds and tipping points.

    Returns (lower_bound, upper_bound, tipping_point).
    """
    lower = pd.Series(np.nan, index=df.index, dtype=float)
    upper = pd.Series(np.nan, index=df.index, dtype=float)
    tipping = pd.Series(np.nan, index=df.index, dtype=float)

    if not mnar_mask.any() or var not in df.columns:
        return lower, upper, tipping

    target = df[var]
    observed_vals = target.dropna()

    if len(observed_vals) < 5:
        return lower, upper, tipping

    # Bounds from observed distribution
    p5 = float(observed_vals.quantile(0.05))
    p25 = float(observed_vals.quantile(0.25))
    p75 = float(observed_vals.quantile(0.75))
    p95 = float(observed_vals.quantile(0.95))
    obs_std = float(observed_vals.std())
    obs_mean = float(observed_vals.mean())

    # Pessimism factor: hidden values are likely worse
    # For MNAR, shift bounds toward the unfavorable direction
    # Determine "unfavorable" direction from variable semantics
    unfavorable_low = _is_higher_better(var)

    if unfavorable_low:
        # Higher is better (e.g. revenue, cash) -> hidden values likely lower
        lower.loc[mnar_mask] = p5 - 0.5 * obs_std
        upper.loc[mnar_mask] = p75
    else:
        # Lower is better (e.g. debt_to_equity) -> hidden values likely higher
        lower.loc[mnar_mask] = p25
        upper.loc[mnar_mask] = p95 + 0.5 * obs_std

    # Tipping point: the value at which the variable would cross
    # from "normal" to "concerning" (using 1 std below/above mean)
    if unfavorable_low:
        tipping.loc[mnar_mask] = obs_mean - 1.5 * obs_std
    else:
        tipping.loc[mnar_mask] = obs_mean + 1.5 * obs_std

    return lower, upper, tipping


def _is_higher_better(var: str) -> bool:
    """Heuristic: is a higher value generally favorable for this variable?"""
    higher_is_better = {
        "revenue", "gross_profit", "ebit", "ebitda", "net_income",
        "total_assets", "total_equity", "current_assets",
        "cash_and_equivalents", "operating_cash_flow",
        "free_cash_flow", "free_cash_flow_ttm_asof",
        "current_ratio", "cash_ratio",
        "gross_margin", "operating_margin", "net_margin",
        "interest_coverage",
    }
    lower_is_better = {
        "total_liabilities", "current_liabilities",
        "short_term_debt", "long_term_debt", "total_debt_asof",
        "net_debt", "debt_to_equity", "net_debt_to_ebitda",
        "interest_expense",
    }
    if var in higher_is_better:
        return True
    if var in lower_is_better:
        return False
    # Default: assume higher is better
    return True


# ---------------------------------------------------------------------------
# Pass 4: GAIN adversarial imputation
# ---------------------------------------------------------------------------


def _run_gain(
    df: pd.DataFrame,
    variables: list[str],
    feature_cols: list[str],
    mnar_masks: dict[str, pd.Series],
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Run GAIN adversarial imputer.

    Falls back gracefully if torch is not available.
    """
    results: dict[str, tuple[pd.Series, pd.Series]] = {}

    try:
        from operator1.estimation.gain_imputer import (
            train_and_impute_gain,
            _check_torch,
        )
    except ImportError:
        logger.debug("GAIN imputer module not available")
        return results

    if not _check_torch():
        logger.debug("torch not available -- skipping GAIN")
        return results

    try:
        gain_result = train_and_impute_gain(
            df=df,
            target_vars=variables,
            feature_cols=feature_cols,
            mnar_masks=mnar_masks,
            epochs=80,
        )
    except Exception as exc:
        logger.warning("GAIN training failed: %s", exc)
        return results

    if gain_result.fallback_used:
        return results

    for var in variables:
        if var in gain_result.imputed_values:
            results[var] = (
                gain_result.imputed_values[var],
                gain_result.confidence_scores[var],
            )

    return results


# ---------------------------------------------------------------------------
# Ensemble for MNAR
# ---------------------------------------------------------------------------


def _ensemble_mnar(
    heckman_results: dict[str, tuple[pd.Series, pd.Series]],
    pattern_results: dict[str, tuple[pd.Series, pd.Series]],
    gain_results: dict[str, tuple[pd.Series, pd.Series]],
    variables: list[str],
    df: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, str]]:
    """Combine MNAR estimates: prioritize Heckman (theoretically grounded),
    with pattern-mixture and GAIN as supporting evidence."""
    estimated_out: dict[str, pd.Series] = {}
    confidence_out: dict[str, pd.Series] = {}
    methods_out: dict[str, str] = {}

    for var in variables:
        sources = []
        for name, res in [
            ("heckman", heckman_results),
            ("pattern", pattern_results),
            ("gain", gain_results),
        ]:
            if var in res:
                sources.append((name, res[var][0], res[var][1]))

        if not sources:
            estimated_out[var] = pd.Series(np.nan, index=df.index)
            confidence_out[var] = pd.Series(np.nan, index=df.index)
            methods_out[var] = "none"
            continue

        # Priority weights: Heckman > Pattern-Mixture > GAIN
        priority = {"heckman": 0.45, "pattern": 0.30, "gain": 0.25}

        est_stack = pd.DataFrame(
            {n: e for n, e, _ in sources}, index=df.index,
        )
        wt_stack = pd.DataFrame(
            {n: c * priority.get(n, 0.2) for n, _, c in sources}, index=df.index,
        )

        wt_sum = wt_stack.sum(axis=1).replace(0, np.nan)
        weighted_est = (est_stack * wt_stack).sum(axis=1) / wt_sum

        # MNAR confidence is capped lower than MAR (inherent uncertainty)
        avg_conf = wt_stack.mean(axis=1).clip(0.05, 0.75)

        estimated_out[var] = weighted_est
        confidence_out[var] = avg_conf
        methods_out[var] = "+".join(n for n, _, _ in sources)

    return estimated_out, confidence_out, methods_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_hidden_data(
    df: pd.DataFrame,
    variables: list[str],
    mnar_masks: dict[str, pd.Series] | None = None,
    feature_cols: list[str] | None = None,
) -> MNAREstimationResult:
    """Estimate hidden (MNAR) values using selection-corrected models.

    Parameters
    ----------
    df:
        Full feature table from the cache builder.
    variables:
        Variable names to estimate.
    mnar_masks:
        Per-variable boolean masks indicating MNAR rows.
    feature_cols:
        Predictor columns.  If None, auto-selected.

    Returns
    -------
    MNAREstimationResult
        Estimated values, confidence, sensitivity bounds per variable.
    """
    result = MNAREstimationResult()

    vars_with_mnar = [
        v for v in variables
        if v in df.columns and (
            mnar_masks.get(v, df[v].isna()).any() if mnar_masks else df[v].isna().any()
        )
    ]

    if not vars_with_mnar:
        logger.info("No MNAR values to estimate")
        return result

    if mnar_masks is None:
        mnar_masks = {v: df[v].isna() for v in vars_with_mnar}

    if feature_cols is None:
        feature_cols = _select_feature_columns(df, vars_with_mnar)

    logger.info(
        "Running MNAR estimator for %d variables using %d features",
        len(vars_with_mnar), len(feature_cols),
    )

    # Pass 1: Heckman selection model
    heckman_results: dict[str, tuple[pd.Series, pd.Series]] = {}
    for var in vars_with_mnar:
        mnar_mask = mnar_masks.get(var, df[var].isna())
        heckman_out = _run_heckman(df, var, feature_cols, mnar_mask)
        if heckman_out is not None:
            heckman_results[var] = heckman_out

    # Pass 2: Pattern-mixture model
    pattern_results = _run_pattern_mixture(df, vars_with_mnar, mnar_masks)

    # Pass 3: Sensitivity bounds (always computed, independent of model)
    for var in vars_with_mnar:
        mnar_mask = mnar_masks.get(var, df[var].isna())
        lower, upper, tipping = _compute_sensitivity_bounds(df, var, mnar_mask)
        result.sensitivity_lower[var] = lower
        result.sensitivity_upper[var] = upper
        result.tipping_points[var] = tipping

    # Pass 4: GAIN adversarial imputation
    gain_results = _run_gain(df, vars_with_mnar, feature_cols, mnar_masks)

    # Ensemble
    est, conf, methods = _ensemble_mnar(
        heckman_results, pattern_results, gain_results,
        vars_with_mnar, df,
    )

    result.estimated_values = est
    result.confidence_scores = conf
    result.methods_used = methods
    result.n_estimated = sum(
        int(v.notna().sum()) for v in est.values()
    )

    logger.info(
        "MNAR estimation complete: %d values estimated across %d variables",
        result.n_estimated, len(vars_with_mnar),
    )

    return result


def _select_feature_columns(
    df: pd.DataFrame,
    target_vars: list[str],
) -> list[str]:
    """Select numeric, non-flag columns for features."""
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
        if df[col].dtype not in ("float64", "float32", "int64", "int32"):
            continue
        if df[col].notna().mean() < 0.5:
            continue
        candidates.append(col)
    return candidates[:15]

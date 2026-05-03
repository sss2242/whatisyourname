"""C6 -- Sobol global sensitivity analysis.

Decomposes prediction variance to identify which decision/linked
variables are the key drivers of the model output. Validates the
survival hierarchy by comparing Sobol indices against tier weights.

Spec reference: The_Apps_core_idea.pdf Section E.2 Category 4.

Falls back to a permutation-based importance measure if SALib is
not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SobolResult:
    """Result from Sobol sensitivity analysis."""
    # First-order Sobol indices: {variable: S1}
    first_order: dict[str, float] = field(default_factory=dict)
    # Total-order Sobol indices: {variable: ST}
    total_order: dict[str, float] = field(default_factory=dict)
    # Tier-level aggregated importance: {tier: importance}
    tier_importance: dict[str, float] = field(default_factory=dict)
    available: bool = True
    error: str = ""
    method: str = "sobol"  # "sobol" or "permutation_fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "method": self.method,
            "first_order": self.first_order,
            "total_order": self.total_order,
            "tier_importance": self.tier_importance,
        }


def _load_tier_map() -> dict[str, int]:
    """Load variable -> tier mapping from config."""
    try:
        from operator1.config_loader import load_config
        cfg = load_config("survival_hierarchy")
        tier_map: dict[str, int] = {}
        for tier_key, tier_data in cfg.get("tiers", {}).items():
            tier_num = int(tier_key.replace("tier", ""))
            for var in tier_data.get("variables", []):
                tier_map[var] = tier_num
        return tier_map
    except Exception:
        return {}


def _try_sobol(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: list[str],
    n_samples: int = 512,
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Attempt SALib-based Sobol analysis. Returns None if SALib unavailable."""
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol as sobol_analyze
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        logger.info("SALib not available, falling back to permutation importance")
        return None

    n_vars = X.shape[1]
    if n_vars < 2 or len(y) < 50:
        return None

    # Define the problem for SALib
    problem = {
        "num_vars": n_vars,
        "names": variable_names,
        "bounds": [
            [float(X[:, i].min()), float(X[:, i].max())]
            for i in range(n_vars)
        ],
    }

    # Sanitise bounds (SALib requires lb < ub)
    for i, (lb, ub) in enumerate(problem["bounds"]):
        if lb >= ub:
            problem["bounds"][i] = [lb - 1.0, ub + 1.0]

    try:
        # Train a surrogate model
        model = GradientBoostingRegressor(
            n_estimators=50, max_depth=4, random_state=42,
        )
        model.fit(X, y)

        # Generate Sobol sample
        param_values = saltelli.sample(problem, n_samples, calc_second_order=False)

        # Clip to training bounds
        for i in range(n_vars):
            param_values[:, i] = np.clip(
                param_values[:, i],
                problem["bounds"][i][0],
                problem["bounds"][i][1],
            )

        # Evaluate surrogate
        Y = model.predict(param_values)

        # Analyse
        Si = sobol_analyze.analyze(problem, Y, calc_second_order=False)

        first_order = {
            name: max(0.0, float(Si["S1"][i]))
            for i, name in enumerate(variable_names)
        }
        total_order = {
            name: max(0.0, float(Si["ST"][i]))
            for i, name in enumerate(variable_names)
        }

        return first_order, total_order

    except Exception as exc:
        logger.warning("Sobol analysis failed: %s", exc)
        return None


def _try_morris(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: list[str],
    n_trajectories: int = 15,
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Morris screening method (Morris 1991, Campolongo et al. 2007).

    Needs only r*(p+1) samples where r=trajectories, p=parameters.
    For 30 features with r=15: 15*31 = 465 samples (fits 502 rows).
    Compare to Saltelli: N*(2D+2) = 512*(62) = 31,744 (way too many).

    Returns (first_order_proxy, total_order_proxy) using mu_star and sigma.
    mu_star (mean absolute elementary effect) is a ranking-equivalent
    substitute for Sobol S1 indices -- sufficient for hierarchy nudging.
    """
    try:
        from SALib.sample import morris as morris_sample
        from SALib.analyze import morris as morris_analyze
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        logger.info("SALib.sample.morris not available")
        return None

    n_vars = X.shape[1]
    if n_vars < 2 or len(y) < 50:
        return None

    # Cap trajectories to fit available data: r*(p+1) <= n_samples
    max_r = max(4, len(y) // (n_vars + 1))
    n_trajectories = min(n_trajectories, max_r)

    if n_trajectories < 4:
        return None

    problem = {
        "num_vars": n_vars,
        "names": variable_names,
        "bounds": [
            [float(X[:, i].min()), float(X[:, i].max())]
            for i in range(n_vars)
        ],
    }

    # Sanitise bounds
    for i, (lb, ub) in enumerate(problem["bounds"]):
        if lb >= ub:
            problem["bounds"][i] = [lb - 1.0, ub + 1.0]

    try:
        # Train a surrogate model (same as Sobol path)
        model = GradientBoostingRegressor(
            n_estimators=50, max_depth=4, random_state=42,
        )
        model.fit(X, y)

        # Generate Morris samples
        X_morris = morris_sample.sample(
            problem, N=n_trajectories, num_levels=4,
        )

        # Clip to training bounds
        for i in range(n_vars):
            X_morris[:, i] = np.clip(
                X_morris[:, i],
                problem["bounds"][i][0],
                problem["bounds"][i][1],
            )

        # Evaluate surrogate on Morris samples
        Y_morris = model.predict(X_morris)

        # Analyse using Morris method
        Si = morris_analyze.analyze(problem, X_morris, Y_morris)

        # mu_star = mean absolute elementary effect (proxy for S1)
        # sigma = std of elementary effects (proxy for ST / interaction)
        mu_star = Si.get("mu_star", Si.get("mu_star_conf", None))
        sigma = Si.get("sigma", None)

        if mu_star is None:
            return None

        # Normalize mu_star to sum to 1 (like Sobol S1)
        total_mu = float(np.sum(np.abs(mu_star)))
        if total_mu <= 0:
            return None

        first_order = {
            name: max(0.0, float(mu_star[i]) / total_mu)
            for i, name in enumerate(variable_names)
        }

        if sigma is not None:
            total_sigma = float(np.sum(np.abs(sigma)))
            total_order = {
                name: max(0.0, float(sigma[i]) / total_sigma) if total_sigma > 0 else first_order[name]
                for i, name in enumerate(variable_names)
            }
        else:
            total_order = dict(first_order)

        logger.info(
            "Morris screening: %d trajectories, %d variables, %d surrogate samples",
            n_trajectories, n_vars, len(X_morris),
        )

        return first_order, total_order

    except Exception as exc:
        logger.warning("Morris screening failed: %s", exc)
        return None


def _permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: list[str],
    n_repeats: int = 5,
) -> dict[str, float]:
    """Simple permutation importance as a Sobol fallback."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error

    model = GradientBoostingRegressor(
        n_estimators=50, max_depth=4, random_state=42,
    )
    model.fit(X, y)

    baseline_mse = mean_squared_error(y, model.predict(X))
    importances: dict[str, float] = {}

    rng = np.random.RandomState(42)

    for i, name in enumerate(variable_names):
        mse_increases = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, i])
            perm_mse = mean_squared_error(y, model.predict(X_perm))
            mse_increases.append(perm_mse - baseline_mse)
        importances[name] = max(0.0, float(np.mean(mse_increases)))

    # Normalise to sum to 1
    total = sum(importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}

    return importances


def run_sensitivity_analysis(
    cache: pd.DataFrame,
    target_variable: str = "return_1d",
    feature_variables: list[str] | None = None,
) -> SobolResult:
    """Run global sensitivity analysis on the cache.

    Parameters
    ----------
    cache:
        Daily feature table.
    target_variable:
        The output variable to decompose variance for.
    feature_variables:
        Input variables. If None, auto-detect from tier config.

    Returns
    -------
    SobolResult
    """
    try:
        return _run_sensitivity_impl(cache, target_variable, feature_variables)
    except Exception as exc:
        logger.warning("Sensitivity analysis failed: %s", exc)
        return SobolResult(available=False, error=str(exc))


def _run_sensitivity_impl(
    cache: pd.DataFrame,
    target_variable: str,
    feature_variables: list[str] | None,
) -> SobolResult:
    tier_map = _load_tier_map()

    if feature_variables is None:
        feature_variables = [v for v in tier_map if v in cache.columns]

    if not feature_variables:
        return SobolResult(available=False, error="No feature variables found")

    if target_variable not in cache.columns:
        return SobolResult(
            available=False, error=f"Target variable '{target_variable}' not in cache",
        )

    # Prepare clean data
    cols = feature_variables + [target_variable]
    df = cache[cols].dropna()
    if len(df) < 50:
        return SobolResult(available=False, error="Insufficient data for sensitivity analysis")

    X = df[feature_variables].values
    y = df[target_variable].values

    # Cascade: Morris (cheap, fits small samples) -> Sobol (exact but data-hungry)
    #          -> permutation importance (always works)
    first_order = None
    total_order = None
    method = ""

    # Step 1: Try Morris screening (needs r*(p+1) samples, typically 300-500)
    morris_result = _try_morris(X, y, feature_variables)
    if morris_result is not None:
        first_order, total_order = morris_result
        method = "morris_screening"
        logger.info("Sensitivity: Morris screening succeeded")
    else:
        # Step 2: Try Sobol (needs N*(2D+2) samples, typically 1000+)
        sobol_result = _try_sobol(X, y, feature_variables)
        if sobol_result is not None:
            first_order, total_order = sobol_result
            method = "sobol"
            logger.info("Sensitivity: Sobol analysis succeeded")

    # Step 3: Permutation importance fallback (always works)
    if first_order is None:
        perm_imp = _permutation_importance(X, y, feature_variables)
        first_order = perm_imp
        total_order = perm_imp
        method = "permutation_fallback"
        logger.info("Sensitivity: using permutation importance fallback")

    # Aggregate by tier
    tier_importance: dict[str, float] = {}
    for var, imp in total_order.items():
        tier = tier_map.get(var)
        if tier is not None:
            key = f"tier{tier}"
            tier_importance[key] = tier_importance.get(key, 0.0) + imp

    logger.info(
        "Sensitivity analysis (%s): %d variables, tier importance: %s",
        method, len(feature_variables), tier_importance,
    )

    return SobolResult(
        first_order=first_order,
        total_order=total_order,
        tier_importance=tier_importance,
        method=method,
    )


def adjust_hierarchy_from_sobol(
    sobol_result: SobolResult | None,
    current_weights: dict[str, float],
    max_adjustment: float = 0.10,
    threshold: float = 0.20,
) -> dict[str, float]:
    """Adjust hierarchy weights based on Sobol sensitivity analysis.

    Creates a data-driven calibration loop: Sobol measures actual
    variance contribution per tier -> weights adjust toward empirical
    importance -> next pipeline run uses better-calibrated weights.

    Parameters
    ----------
    sobol_result:
        Output from ``run_sensitivity_analysis()``.
    current_weights:
        Current tier weights, e.g. ``{"tier1": 20.0, ...}``.
    max_adjustment:
        Maximum weight adjustment per tier per run (prevents oscillation).
    threshold:
        Minimum discrepancy (as fraction) to trigger adjustment.

    Returns
    -------
    Adjusted weights dict. Returns original weights if Sobol result
    is unavailable or adjustment is not warranted.
    """
    if sobol_result is None or not sobol_result.tier_importance:
        return dict(current_weights)

    # Normalize Sobol tier importance to sum to 100
    total_sobol = sum(sobol_result.tier_importance.values())
    if total_sobol <= 0:
        return dict(current_weights)

    sobol_pct = {
        k: (v / total_sobol) * 100.0
        for k, v in sobol_result.tier_importance.items()
    }

    adjusted = dict(current_weights)
    any_adjusted = False

    for tier in sorted(current_weights.keys()):
        current = current_weights.get(tier, 20.0)
        target = sobol_pct.get(tier, current)

        discrepancy = abs(target - current) / max(current, 1.0)
        if discrepancy > threshold:
            # Move toward target, capped at max_adjustment * 100
            direction = 1.0 if target > current else -1.0
            adjustment = min(abs(target - current), max_adjustment * 100)
            adjusted[tier] = current + direction * adjustment
            any_adjusted = True

    # Renormalize to sum to 100
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total * 100.0 for k, v in adjusted.items()}

    if any_adjusted:
        logger.info(
            "Sobol-adjusted hierarchy weights: %s (from %s)",
            {k: f"{v:.1f}" for k, v in adjusted.items()},
            {k: f"{v:.1f}" for k, v in current_weights.items()},
        )

    return adjusted

"""C7 -- Copula models for tail dependency structure.

Models complex dependency structures between variables using copulas,
which separate marginal distributions from the dependence structure.
Critical for capturing crisis co-movements (variables crash together).

Spec reference: The_Apps_core_idea.pdf Section E.2 Category 3.

Uses scipy for marginal fitting and a Gaussian copula implementation.
Falls back to simple correlation if fitting fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CopulaResult:
    """Result from copula dependency analysis."""
    # Correlation matrix of the copula (uniform marginals)
    copula_correlation: dict[str, dict[str, float]] = field(default_factory=dict)
    # Tail dependence coefficients: {(var_i, var_j): lower_tail_dep}
    tail_dependence: dict[str, float] = field(default_factory=dict)
    # Joint crisis probability estimates
    joint_crisis_probability: float = 0.0
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "copula_correlation": self.copula_correlation,
            "tail_dependence": self.tail_dependence,
            "joint_crisis_probability": self.joint_crisis_probability,
        }


def _to_uniform_marginals(data: np.ndarray) -> np.ndarray:
    """Transform data to uniform marginals using the empirical CDF (PIT)."""
    n, d = data.shape
    uniform = np.zeros_like(data)
    for j in range(d):
        col = data[:, j]
        ranks = stats.rankdata(col, method="average")
        uniform[:, j] = ranks / (n + 1)  # avoid 0 and 1
    return uniform


def _fit_gaussian_copula(uniform_data: np.ndarray) -> np.ndarray:
    """Fit a Gaussian copula by inverting uniform marginals to normal
    and computing the correlation matrix."""
    # Transform uniform -> standard normal
    normal_data = stats.norm.ppf(np.clip(uniform_data, 1e-6, 1 - 1e-6))

    # Handle any remaining NaN/Inf from ppf
    mask = np.isfinite(normal_data).all(axis=1)
    normal_data = normal_data[mask]

    if len(normal_data) < 10:
        return np.eye(uniform_data.shape[1])

    # Guard against zero-variance columns (constant values after PIT
    # transform) which cause NaN in np.corrcoef.
    col_std = np.std(normal_data, axis=0)
    zero_var_mask = col_std < 1e-10
    if zero_var_mask.any():
        # Add tiny noise to constant columns to avoid NaN correlation
        normal_data[:, zero_var_mask] += np.random.default_rng(42).normal(
            0, 1e-6, size=(len(normal_data), int(zero_var_mask.sum()))
        )

    # Correlation matrix of the normal-transformed data = copula parameter
    corr = np.corrcoef(normal_data, rowvar=False)

    # Final NaN guard: replace any remaining NaN with identity
    if np.isnan(corr).any():
        np.fill_diagonal(corr, 1.0)
        corr = np.nan_to_num(corr, nan=0.0)

    return corr


def _estimate_tail_dependence(
    data: np.ndarray,
    variable_names: list[str],
    quantile: float = 0.05,
) -> dict[str, float]:
    """Estimate lower tail dependence coefficient for each pair.

    Tail dependence measures the probability that both variables are
    in their extreme lower tail simultaneously.
    """
    n, d = data.shape
    tail_dep: dict[str, float] = {}

    for i in range(d):
        for j in range(i + 1, d):
            # Empirical lower tail dependence:
            # P(Y <= q | X <= q) where q is the quantile threshold
            xi = data[:, i]
            xj = data[:, j]

            qi = np.quantile(xi, quantile)
            qj = np.quantile(xj, quantile)

            both_below = np.sum((xi <= qi) & (xj <= qj))
            i_below = np.sum(xi <= qi)

            if i_below > 0:
                dep = both_below / i_below
            else:
                dep = 0.0

            key = f"{variable_names[i]}|{variable_names[j]}"
            tail_dep[key] = round(float(dep), 4)

    return tail_dep


def _estimate_joint_crisis_prob(
    data: np.ndarray,
    variable_names: list[str],
    crisis_quantile: float = 0.10,
) -> float:
    """Estimate the probability that ALL variables are simultaneously
    in their lower tail (joint crisis scenario)."""
    n, d = data.shape
    if d == 0 or n == 0:
        return 0.0

    thresholds = np.quantile(data, crisis_quantile, axis=0)
    all_below = np.all(data <= thresholds, axis=1)
    return float(np.mean(all_below))


def run_copula_analysis(
    cache: pd.DataFrame,
    variables: list[str] | None = None,
    max_variables: int = 10,
) -> CopulaResult:
    """Run copula dependency analysis on the cache.

    Parameters
    ----------
    cache:
        Daily feature table.
    variables:
        Variables to include. If None, auto-detect from tier config.
    max_variables:
        Cap on number of variables (copula becomes expensive with many).

    Returns
    -------
    CopulaResult
    """
    try:
        return _run_copula_impl(cache, variables, max_variables)
    except Exception as exc:
        logger.warning("Copula analysis failed: %s", exc)
        return CopulaResult(available=False, error=str(exc))


def _run_copula_impl(
    cache: pd.DataFrame,
    variables: list[str] | None,
    max_variables: int,
) -> CopulaResult:
    if variables is None:
        # Pick representative variables across tiers
        candidates = [
            "cash_ratio", "debt_to_equity_abs", "volatility_21d",
            "gross_margin", "pe_ratio_calc", "return_1d",
            "free_cash_flow_ttm_asof", "drawdown_252d",
        ]
        variables = [v for v in candidates if v in cache.columns]

    variables = variables[:max_variables]

    if len(variables) < 2:
        return CopulaResult(available=False, error="Need >= 2 variables for copula")

    df = cache[variables].dropna()
    if len(df) < 30:
        return CopulaResult(available=False, error="Insufficient data for copula fitting")

    data = df.values
    var_names = list(variables)

    # Step 1: Transform to uniform marginals
    uniform = _to_uniform_marginals(data)

    # Step 2: Fit Gaussian copula
    copula_corr = _fit_gaussian_copula(uniform)

    # Step 3: Estimate tail dependence
    tail_dep = _estimate_tail_dependence(data, var_names)

    # Step 4: Joint crisis probability
    joint_crisis = _estimate_joint_crisis_prob(data, var_names)

    # Format correlation as nested dict
    corr_dict: dict[str, dict[str, float]] = {}
    for i, vi in enumerate(var_names):
        corr_dict[vi] = {}
        for j, vj in enumerate(var_names):
            corr_dict[vi][vj] = round(float(copula_corr[i, j]), 4)

    logger.info(
        "Copula analysis: %d variables, joint crisis prob = %.4f",
        len(var_names), joint_crisis,
    )

    return CopulaResult(
        copula_correlation=corr_dict,
        tail_dependence=tail_dep,
        joint_crisis_probability=round(joint_crisis, 4),
    )

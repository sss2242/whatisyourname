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
    # Best copula type selected by AIC (gaussian, student_t, clayton)
    best_copula: str = "gaussian"
    # AIC scores per copula type
    aic_scores: dict[str, float] = field(default_factory=dict)
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "best_copula": self.best_copula,
            "aic_scores": self.aic_scores,
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

    # Correlation matrix of the normal-transformed data = copula parameter.
    # Use robust Minimum Covariance Determinant when available (sklearn).
    # MCD is resistant to up to 50% outliers, preventing a single earnings
    # surprise day from corrupting the copula for weeks.
    corr = None
    if len(normal_data) >= 2 * normal_data.shape[1]:
        try:
            from sklearn.covariance import MinCovDet
            mcd = MinCovDet(random_state=42).fit(normal_data)
            # Convert covariance to correlation
            cov = mcd.covariance_
            d = np.sqrt(np.diag(cov))
            d[d < 1e-12] = 1.0  # avoid division by zero
            corr = cov / np.outer(d, d)
            np.fill_diagonal(corr, 1.0)
        except Exception:
            corr = None  # fall back to np.corrcoef

    if corr is None:
        corr = np.corrcoef(normal_data, rowvar=False)

    # Final NaN guard: replace any remaining NaN with identity
    if np.isnan(corr).any():
        np.fill_diagonal(corr, 1.0)
        corr = np.nan_to_num(corr, nan=0.0)

    return corr


def _fit_student_t_copula(
    uniform_data: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit a Student-t copula using the copulae library.

    Returns (correlation_matrix, degrees_of_freedom, log_likelihood).
    Falls back to Gaussian copula parameters if copulae is not installed.
    """
    try:
        from copulae import StudentCopula
        d = uniform_data.shape[1]
        cop = StudentCopula(dim=d)
        cop.fit(uniform_data)
        corr = np.array(cop.params.corr)
        df = float(cop.params.df)
        ll = float(cop.log_lik(uniform_data))
        logger.debug("Student-t copula fitted: df=%.1f, log_lik=%.2f", df, ll)
        return corr, df, ll
    except ImportError:
        logger.debug("copulae not installed, skipping Student-t copula")
        return np.eye(uniform_data.shape[1]), 30.0, float("-inf")
    except Exception as exc:
        logger.debug("Student-t copula fitting failed: %s", exc)
        return np.eye(uniform_data.shape[1]), 30.0, float("-inf")


def _fit_clayton_copula(
    uniform_data: np.ndarray,
) -> tuple[float, float]:
    """Fit a Clayton copula (captures lower-tail dependence).

    Returns (theta, log_likelihood).
    Falls back to theta=1.0 if copulae is not installed.
    """
    try:
        from copulae import ClaytonCopula
        d = uniform_data.shape[1]
        if d != 2:
            # Clayton copula in copulae only supports bivariate; for d>2
            # fit pairwise and average theta.
            thetas = []
            lls = []
            for i in range(d):
                for j in range(i + 1, d):
                    try:
                        pair = uniform_data[:, [i, j]]
                        cop = ClaytonCopula(dim=2)
                        cop.fit(pair)
                        thetas.append(float(cop.params))
                        lls.append(float(cop.log_lik(pair)))
                    except Exception:
                        pass
            if thetas:
                return float(np.mean(thetas)), float(np.mean(lls))
            return 1.0, float("-inf")
        cop = ClaytonCopula(dim=2)
        cop.fit(uniform_data)
        theta = float(cop.params)
        ll = float(cop.log_lik(uniform_data))
        logger.debug("Clayton copula fitted: theta=%.3f, log_lik=%.2f", theta, ll)
        return theta, ll
    except ImportError:
        logger.debug("copulae not installed, skipping Clayton copula")
        return 1.0, float("-inf")
    except Exception as exc:
        logger.debug("Clayton copula fitting failed: %s", exc)
        return 1.0, float("-inf")


def _compute_aic(log_lik: float, n_params: int) -> float:
    """Compute Akaike Information Criterion: AIC = 2k - 2*ln(L)."""
    if not np.isfinite(log_lik):
        return float("inf")
    return 2 * n_params - 2 * log_lik


def _select_best_copula(
    uniform_data: np.ndarray,
    gaussian_corr: np.ndarray,
) -> tuple[str, np.ndarray, dict[str, float]]:
    """Fit Gaussian, Student-t, and Clayton copulas; select best by AIC.

    Returns (best_type, best_correlation, aic_scores).
    """
    d = uniform_data.shape[1]
    n_corr_params = d * (d - 1) // 2  # unique off-diagonal elements

    # Gaussian AIC: compute log-likelihood from multivariate normal
    normal_data = stats.norm.ppf(np.clip(uniform_data, 1e-6, 1 - 1e-6))
    mask = np.isfinite(normal_data).all(axis=1)
    normal_clean = normal_data[mask]
    try:
        from scipy.stats import multivariate_normal
        gauss_ll = float(np.sum(multivariate_normal.logpdf(
            normal_clean, mean=np.zeros(d), cov=gaussian_corr,
        )))
    except Exception:
        gauss_ll = float("-inf")

    gauss_aic = _compute_aic(gauss_ll, n_corr_params)

    # Student-t: correlation + degrees of freedom
    t_corr, t_df, t_ll = _fit_student_t_copula(uniform_data)
    t_aic = _compute_aic(t_ll, n_corr_params + 1)

    # Clayton: single theta parameter
    c_theta, c_ll = _fit_clayton_copula(uniform_data)
    c_aic = _compute_aic(c_ll, 1)

    aic_scores = {
        "gaussian": round(gauss_aic, 2),
        "student_t": round(t_aic, 2),
        "clayton": round(c_aic, 2),
    }

    # Select best (lowest AIC)
    best = min(aic_scores, key=lambda k: aic_scores[k])

    if best == "student_t" and np.isfinite(t_aic):
        logger.info(
            "Best copula: Student-t (df=%.1f), AIC=%s", t_df, aic_scores,
        )
        return "student_t", t_corr, aic_scores
    elif best == "clayton" and np.isfinite(c_aic):
        logger.info(
            "Best copula: Clayton (theta=%.3f), AIC=%s", c_theta, aic_scores,
        )
        return "clayton", gaussian_corr, aic_scores
    else:
        logger.info("Best copula: Gaussian, AIC=%s", aic_scores)
        return "gaussian", gaussian_corr, aic_scores


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


def _fit_empirical_copula(
    data: np.ndarray,
    variable_names: list[str],
) -> CopulaResult:
    """Empirical copula using rank-based pseudo-observations + Kendall's tau.

    Non-parametric fallback that works with any sample size >= 10.
    Uses rank correlation (Kendall's tau) for dependence structure and
    empirical quantile exceedance for lower tail dependence estimation.

    References:
        Genest & Favre (2007), Kendall tau inversion for copula parameters.
        OpenTURNS uncertainty quantification (tau -> Clayton theta mapping).
    """
    n, d = data.shape

    # Compute pairwise Kendall tau correlation
    from scipy.stats import kendalltau

    tau_matrix: dict[str, dict[str, float]] = {}
    for i, vi in enumerate(variable_names):
        tau_matrix[vi] = {}
        for j, vj in enumerate(variable_names):
            if i == j:
                tau_matrix[vi][vj] = 1.0
            elif j > i:
                tau_val, _ = kendalltau(data[:, i], data[:, j])
                tau_matrix[vi][vj] = round(float(tau_val) if np.isfinite(tau_val) else 0.0, 4)
            else:
                tau_matrix[vi][vj] = tau_matrix[vj][vi]

    # Empirical lower tail dependence from quantile co-exceedance
    q = 0.10
    tail_dep: dict[str, float] = {}
    for i in range(d):
        for j in range(i + 1, d):
            qi = np.quantile(data[:, i], q)
            qj = np.quantile(data[:, j], q)
            both_below = np.sum((data[:, i] <= qi) & (data[:, j] <= qj))
            i_below = np.sum(data[:, i] <= qi)
            dep = both_below / max(i_below, 1)
            key = f"{variable_names[i]}|{variable_names[j]}"
            tail_dep[key] = round(float(dep), 4)

    # Joint crisis probability
    joint_crisis = _estimate_joint_crisis_prob(data, variable_names)

    # Infer best copula type from tau distribution
    mean_tau = np.mean([
        tau_matrix[vi][vj]
        for i, vi in enumerate(variable_names)
        for j, vj in enumerate(variable_names)
        if i < j
    ]) if d > 1 else 0.0

    # If mean tau is negative (lower tail clustering), Clayton is implied
    if mean_tau > 0:
        # Convert tau to Clayton theta: theta = 2*tau/(1-tau)
        clayton_theta = max(0.01, 2 * mean_tau / (1 - mean_tau)) if mean_tau < 1.0 else 10.0
        best_type = f"empirical_clayton_implied(theta={clayton_theta:.2f})"
    else:
        best_type = "empirical_kendall"

    logger.info(
        "Empirical copula: %d variables, %d observations, mean_tau=%.3f, type=%s",
        d, n, mean_tau, best_type,
    )

    return CopulaResult(
        copula_correlation=tau_matrix,
        tail_dependence=tail_dep,
        joint_crisis_probability=round(joint_crisis, 4),
        best_copula=best_type,
        aic_scores={"empirical": 0.0},
    )


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

    # Mixed-frequency aware: use event-day filtering for forward-filled
    # variables.  On "event days" (when any financial variable changes),
    # all variables have meaningful values.  Between events, financial
    # variables are stale repeats that create false zero-correlation.
    try:
        from operator1.models._frequency_classifier import classify_column_frequency, detect_filing_change_days

        has_quarterly = any(
            classify_column_frequency(cache[v]) in ("quarterly", "annual")
            for v in variables if v in cache.columns
        )
        if has_quarterly:
            event_mask = pd.Series(False, index=cache.index)
            for v in variables:
                if classify_column_frequency(cache[v]) in ("quarterly", "annual"):
                    event_mask |= detect_filing_change_days(cache[v])
            event_data = cache[variables][event_mask].dropna()
            # Use event-day data if enough events; otherwise fall back to weekly resample
            if len(event_data) >= 20:
                df = event_data
                logger.debug("Copula: using %d event-day observations", len(df))
            else:
                df = cache[variables].resample("W").last().dropna()
                logger.debug("Copula: using %d weekly-resampled observations", len(df))
        else:
            df = cache[variables].dropna()
    except Exception:
        df = cache[variables].dropna()

    if len(df) < 10:
        return CopulaResult(available=False, error="Insufficient data for copula fitting (<10 rows)")

    data = df.values
    var_names = list(variables)

    # Sparse data path (10-30 rows): use empirical copula directly
    if len(df) < 30:
        logger.info(
            "Copula: %d overlapping rows (<30), using empirical Kendall tau fallback",
            len(df),
        )
        return _fit_empirical_copula(data, var_names)

    # Full data path (>= 30 rows): try parametric first, empirical fallback
    try:
        # Step 1: Transform to uniform marginals
        uniform = _to_uniform_marginals(data)

        # Step 2: Fit Gaussian copula (always available as baseline)
        gaussian_corr = _fit_gaussian_copula(uniform)

        # Step 3: Model selection -- fit Student-t and Clayton, pick best by AIC
        best_type, best_corr, aic_scores = _select_best_copula(uniform, gaussian_corr)

        # Step 4: Estimate tail dependence
        tail_dep = _estimate_tail_dependence(data, var_names)

        # Step 5: Joint crisis probability
        joint_crisis = _estimate_joint_crisis_prob(data, var_names)

        # Format correlation as nested dict
        corr_dict: dict[str, dict[str, float]] = {}
        for i, vi in enumerate(var_names):
            corr_dict[vi] = {}
            for j, vj in enumerate(var_names):
                corr_dict[vi][vj] = round(float(best_corr[i, j]), 4)

        logger.info(
            "Copula analysis: %d variables, best=%s, joint crisis prob = %.4f",
            len(var_names), best_type, joint_crisis,
        )

        return CopulaResult(
            copula_correlation=corr_dict,
            tail_dependence=tail_dep,
            joint_crisis_probability=round(joint_crisis, 4),
            best_copula=best_type,
            aic_scores=aic_scores,
        )
    except Exception as exc:
        # Parametric copula failed -- fall back to empirical
        logger.warning(
            "Parametric copula failed (%s), falling back to empirical Kendall tau",
            exc,
        )
        return _fit_empirical_copula(data, var_names)

"""T6.3 -- Monte Carlo simulations for survival probability estimation.

Provides regime-aware Monte Carlo simulations that use fitted model
distributions per regime to project key financial variables forward
and estimate survival probabilities.

**Key features:**

1. **Regime-aware path generation**: each simulated path respects the
   current regime (bull, bear, high_vol, low_vol) and uses
   regime-specific distribution parameters (mean, volatility).

2. **Importance sampling**: tail events (paths leading to survival
   trigger breaches) are over-sampled to improve accuracy of rare
   event probability estimates.  Paths are reweighted by their
   likelihood ratios to produce unbiased estimates.

3. **Survival probability distribution**: the fraction of simulated
   paths that do NOT trigger any company survival flag yields the
   survival probability.  Output includes mean, p5, p95 across
   bootstrap resamples.

4. **Reproducibility**: all randomness is seeded via ``random_state``
   for deterministic results.

Company survival triggers (from T4.1):
  - ``current_ratio < 1.0``
  - ``debt_to_equity_abs > 3.0``
  - ``fcf_yield < 0``
  - ``drawdown_252d < -0.40``

Top-level entry point:
  ``run_monte_carlo(cache, n_paths, horizons, ...)``

Spec refs: Sec 17
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from operator1.config_loader import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default simulation parameters.
DEFAULT_N_PATHS: int = 10_000
DEFAULT_HORIZONS: dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "21d": 21,
    "252d": 252,
}

# Default survival thresholds (aligned with T4.1).
DEFAULT_SURVIVAL_THRESHOLDS: dict[str, tuple[str, float]] = {
    "current_ratio": ("lt", 1.0),
    "debt_to_equity_abs": ("gt", 3.0),
    "fcf_yield": ("lt", 0.0),
    "drawdown_252d": ("lt", -0.40),
}

# Importance sampling tilt factor: how much to shift the distribution
# mean toward the danger zone for tail sampling.
DEFAULT_IS_TILT: float = 1.5

# Minimum observations needed to estimate regime distributions.
_MIN_OBS_PER_REGIME: int = 10

# Bootstrap resamples for confidence intervals on survival probability.
DEFAULT_N_BOOTSTRAP: int = 1000

# Regime transition probability smoothing (Laplace).
_TRANSITION_SMOOTHING: float = 1.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class RegimeDistribution:
    """Distribution parameters for a single regime."""

    regime_label: str = ""
    mean: float = 0.0
    std: float = 1.0
    n_obs: int = 0


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation outputs."""

    # Number of simulation paths.
    n_paths: int = 0
    n_paths_importance: int = 0

    # Survival probabilities per horizon.
    # {horizon_label: probability}
    survival_probability: dict[str, float] = field(default_factory=dict)

    # Summary statistics.
    survival_probability_mean: float = float("nan")
    survival_probability_p5: float = float("nan")
    survival_probability_p95: float = float("nan")

    # Per-horizon breakdown.
    # {horizon_label: {"mean": ..., "p5": ..., "p95": ..., "std": ...}}
    survival_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # Regime distribution used.
    regime_distributions: dict[str, RegimeDistribution] = field(
        default_factory=dict,
    )

    # Regime transition matrix (n_regimes x n_regimes).
    transition_matrix: np.ndarray | None = None

    # Current regime at simulation start.
    current_regime: str = ""

    # Importance sampling info.
    importance_sampling_used: bool = False
    effective_sample_size: float = float("nan")

    # Path terminal values (for downstream analysis).
    # {horizon_label: array of shape (n_paths,)}
    terminal_values: dict[str, np.ndarray] = field(default_factory=dict)

    # Max drawdown distribution across MC paths (per horizon).
    # {horizon_label: {"median": ..., "p10": ..., "p90": ..., "mean": ...}}
    max_drawdown_distribution: dict[str, dict[str, float]] = field(
        default_factory=dict,
    )

    # Error info.
    error: str | None = None
    fitted: bool = False


# ---------------------------------------------------------------------------
# Regime distribution estimation
# ---------------------------------------------------------------------------


def estimate_regime_distributions(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    unique_regimes: list[str] | None = None,
) -> dict[str, RegimeDistribution]:
    """Estimate return distribution parameters per regime.

    Parameters
    ----------
    returns:
        1-D array of daily returns.
    regime_labels:
        1-D array of regime labels (same length as returns, may
        contain NaN for unclassified days).
    unique_regimes:
        Optional explicit list of regime labels.  If ``None``, derived
        from ``regime_labels``.

    Returns
    -------
    Dict mapping regime label -> ``RegimeDistribution``.
    """
    # Build a clean mask.
    valid = ~(np.isnan(returns) | pd.isna(regime_labels))
    clean_returns = returns[valid]
    clean_labels = np.array(regime_labels)[valid]

    if unique_regimes is None:
        if len(clean_labels) > 0:
            unique_regimes = sorted(set(str(r) for r in clean_labels))
        else:
            # Fallback: derive from all labels (including NaN-return rows).
            non_na_labels = np.array(regime_labels)[~pd.isna(regime_labels)]
            unique_regimes = sorted(set(str(r) for r in non_na_labels)) if len(non_na_labels) > 0 else []

    distributions: dict[str, RegimeDistribution] = {}

    for regime in unique_regimes:
        if len(clean_labels) > 0:
            mask = np.array([str(r) == regime for r in clean_labels], dtype=bool)
            regime_returns = clean_returns[mask]
        else:
            regime_returns = np.array([])

        dist = RegimeDistribution(regime_label=regime)

        if len(regime_returns) >= _MIN_OBS_PER_REGIME:
            dist.mean = float(np.mean(regime_returns))
            dist.std = float(np.std(regime_returns, ddof=1))
            if dist.std < 1e-12:
                dist.std = 1e-6  # avoid zero volatility
            dist.n_obs = len(regime_returns)
        else:
            # Fallback to overall distribution.
            if len(clean_returns) >= _MIN_OBS_PER_REGIME:
                dist.mean = float(np.mean(clean_returns))
                dist.std = float(np.std(clean_returns, ddof=1))
                dist.n_obs = len(clean_returns)
            else:
                dist.mean = 0.0
                dist.std = 0.01  # conservative default
                dist.n_obs = 0

            logger.warning(
                "Regime '%s' has %d obs (< %d) -- using fallback distribution",
                regime,
                len(regime_returns),
                _MIN_OBS_PER_REGIME,
            )

        distributions[regime] = dist

    return distributions


def estimate_transition_matrix(
    regime_labels: np.ndarray,
    unique_regimes: list[str] | None = None,
    smoothing: float = _TRANSITION_SMOOTHING,
) -> tuple[np.ndarray, list[str]]:
    """Estimate Markov transition matrix from regime label sequence.

    Parameters
    ----------
    regime_labels:
        1-D array of regime labels (may contain NaN).
    unique_regimes:
        Explicit regime list. If ``None``, derived from labels.
    smoothing:
        Laplace smoothing factor to avoid zero-probability transitions.

    Returns
    -------
    (transition_matrix, regime_order)
        ``transition_matrix[i, j]`` = P(regime_j at t+1 | regime_i at t).
        ``regime_order`` is the list of regime labels corresponding to
        row/column indices.
    """
    # Filter NaN.
    valid = ~pd.isna(regime_labels)
    clean = np.array(regime_labels)[valid]

    if unique_regimes is None:
        unique_regimes = sorted(set(str(r) for r in clean))

    n = len(unique_regimes)
    regime_to_idx = {r: i for i, r in enumerate(unique_regimes)}

    # Count transitions.
    counts = np.full((n, n), smoothing)  # Laplace smoothing.

    for t in range(len(clean) - 1):
        from_r = str(clean[t])
        to_r = str(clean[t + 1])
        if from_r in regime_to_idx and to_r in regime_to_idx:
            counts[regime_to_idx[from_r], regime_to_idx[to_r]] += 1

    # Normalize rows.
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)  # avoid division by zero
    matrix = counts / row_sums

    return matrix, unique_regimes


# ---------------------------------------------------------------------------
# Survival trigger checking
# ---------------------------------------------------------------------------


def check_survival_triggers(
    variable_values: dict[str, float],
    thresholds: dict[str, tuple[str, float]] | None = None,
) -> bool:
    """Check if any survival trigger is breached.

    Parameters
    ----------
    variable_values:
        Dict of ``{variable_name: simulated_value}``.
    thresholds:
        Dict of ``{variable_name: (comparison, threshold)}``.
        ``comparison`` is ``"lt"`` (less than) or ``"gt"`` (greater than).

    Returns
    -------
    True if ANY trigger is breached (survival mode activated).
    """
    if thresholds is None:
        thresholds = DEFAULT_SURVIVAL_THRESHOLDS

    for var_name, (comparison, threshold) in thresholds.items():
        if var_name not in variable_values:
            continue

        val = variable_values[var_name]

        if np.isnan(val):
            continue

        if comparison == "lt" and val < threshold:
            return True
        elif comparison == "gt" and val > threshold:
            return True

    return False


# ---------------------------------------------------------------------------
# Path simulation engine
# ---------------------------------------------------------------------------


def _simulate_regime_path(
    n_steps: int,
    current_regime_idx: int,
    transition_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a regime path using Markov transitions.

    Parameters
    ----------
    n_steps:
        Number of time steps.
    current_regime_idx:
        Starting regime index.
    transition_matrix:
        Row-stochastic transition matrix.
    rng:
        Numpy random generator.

    Returns
    -------
    Array of regime indices of length ``n_steps``.
    """
    n_regimes = transition_matrix.shape[0]
    path = np.zeros(n_steps, dtype=int)
    state = current_regime_idx

    for t in range(n_steps):
        path[t] = state
        probs = transition_matrix[state]
        state = rng.choice(n_regimes, p=probs)

    return path


def simulate_return_paths(
    n_paths: int,
    n_steps: int,
    current_regime_idx: int,
    transition_matrix: np.ndarray,
    regime_distributions: list[RegimeDistribution],
    rng: np.random.Generator,
    *,
    importance_tilt: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate return paths with optional importance sampling tilt.

    Parameters
    ----------
    n_paths:
        Number of simulation paths.
    n_steps:
        Number of time steps per path.
    current_regime_idx:
        Starting regime index.
    transition_matrix:
        Regime transition matrix.
    regime_distributions:
        List of ``RegimeDistribution`` per regime (indexed by regime idx).
    rng:
        Numpy random generator.
    importance_tilt:
        If > 0, shift the mean toward negative returns by this many
        standard deviations (for importance sampling of tail events).

    Returns
    -------
    (return_paths, log_weight_paths)
        ``return_paths`` has shape ``(n_paths, n_steps)``.
        ``log_weight_paths`` has shape ``(n_paths,)`` -- log importance
        weights (0 if no tilt applied).
    """
    return_paths = np.zeros((n_paths, n_steps))
    log_weights = np.zeros(n_paths)

    for i in range(n_paths):
        regime_path = _simulate_regime_path(
            n_steps, current_regime_idx, transition_matrix, rng,
        )

        path_log_w = 0.0

        for t in range(n_steps):
            r_idx = regime_path[t]
            dist = regime_distributions[r_idx]

            # Nominal distribution.
            nominal_mean = dist.mean
            nominal_std = dist.std

            # Tilted distribution for importance sampling.
            tilted_mean = nominal_mean - importance_tilt * nominal_std
            tilted_std = nominal_std

            # Sample from tilted distribution.
            z = rng.normal(tilted_mean, tilted_std)
            return_paths[i, t] = z

            if importance_tilt > 0:
                # Log importance weight: log(p_nominal / p_tilted).
                log_p_nom = -0.5 * ((z - nominal_mean) / nominal_std) ** 2
                log_p_tilt = -0.5 * ((z - tilted_mean) / tilted_std) ** 2
                path_log_w += log_p_nom - log_p_tilt

        log_weights[i] = path_log_w

    return return_paths, log_weights


# ---------------------------------------------------------------------------
# Variable evolution from return paths
# ---------------------------------------------------------------------------


def evolve_variables(
    return_paths: np.ndarray,
    initial_values: dict[str, float],
    variable_sensitivities: dict[str, float] | None = None,
    variable_frequencies: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Evolve survival-trigger variables along simulated return paths.

    Uses a simplified model where each variable evolves as a function
    of cumulative returns.  The ``variable_sensitivities`` dict maps
    each variable to its beta (sensitivity) to the return path.

    **Frequency-aware evolution**: quarterly/annual variables (like
    ``current_ratio``, ``fcf_yield``) are updated only at estimated
    filing intervals (~63 days for quarterly, ~252 for annual), holding
    the previous value between filings.  This prevents the unrealistic
    daily jitter that overstates short-horizon survival risk.

    Parameters
    ----------
    return_paths:
        Shape ``(n_paths, n_steps)``.
    initial_values:
        Starting values for each variable.
    variable_sensitivities:
        ``{variable_name: beta}``.  Positive beta means the variable
        moves with returns; negative means inversely.  If ``None``,
        sensible defaults are used.
    variable_frequencies:
        ``{variable_name: frequency}``.  Frequency is one of
        ``"daily"``, ``"quarterly"``, ``"annual"``.  If ``None``,
        defaults are used: ``drawdown_252d`` is daily, others are
        quarterly.

    Returns
    -------
    Dict of ``{variable_name: array of shape (n_paths, n_steps)}``.
    """
    if variable_sensitivities is None:
        # Default sensitivities:
        # - current_ratio: improves with positive returns (equity builds)
        # - debt_to_equity_abs: worsens with negative returns (equity drops)
        # - fcf_yield: improves with positive returns
        # - drawdown_252d: directly tracks cumulative drawdown
        variable_sensitivities = {
            "current_ratio": 0.5,
            "debt_to_equity_abs": -0.8,
            "fcf_yield": 0.3,
            "drawdown_252d": 1.0,
        }

    if variable_frequencies is None:
        # Default frequencies: drawdown is daily, financial ratios are quarterly.
        variable_frequencies = {
            "current_ratio": "quarterly",
            "debt_to_equity_abs": "quarterly",
            "fcf_yield": "quarterly",
            "drawdown_252d": "daily",
        }

    # Filing intervals in trading days.
    _FILING_INTERVALS = {
        "daily": 1,
        "quarterly": 63,  # ~3 months of trading days
        "annual": 252,
    }

    n_paths, n_steps = return_paths.shape
    result: dict[str, np.ndarray] = {}

    # Cumulative returns.
    cum_returns = np.cumsum(return_paths, axis=1)

    for var_name, beta in variable_sensitivities.items():
        if var_name not in initial_values:
            continue

        init_val = initial_values[var_name]
        freq = variable_frequencies.get(var_name, "daily")
        filing_interval = _FILING_INTERVALS.get(freq, 1)

        if var_name == "drawdown_252d":
            # Drawdown is special: track running max and compute drawdown.
            price_paths = np.exp(cum_returns)  # geometric returns
            running_max = np.maximum.accumulate(price_paths, axis=1)
            # Drawdown is (price / running_max) - 1 (always <= 0).
            drawdowns = (price_paths / np.maximum(running_max, 1e-12)) - 1.0
            # Combine with initial drawdown (take the worse of the two).
            result[var_name] = np.minimum(drawdowns, init_val)
        elif filing_interval > 1:
            # Frequency-aware evolution: update only at filing intervals.
            # Between filings, hold the last reported value constant.
            scale = max(abs(init_val), 1e-6)
            evolved = np.full((n_paths, n_steps), init_val)
            for t in range(n_steps):
                if t > 0 and t % filing_interval == 0:
                    # Filing day: update based on cumulative returns since
                    # last filing.
                    prev_filing_t = t - filing_interval
                    if prev_filing_t >= 0:
                        cum_since_filing = cum_returns[:, t] - cum_returns[:, prev_filing_t]
                    else:
                        cum_since_filing = cum_returns[:, t]
                    evolved[:, t] = evolved[:, max(0, t - 1)] + beta * cum_since_filing * scale
                elif t > 0:
                    # Non-filing day: carry forward previous value.
                    evolved[:, t] = evolved[:, t - 1]
            result[var_name] = evolved
        else:
            # Daily evolution: original linear sensitivity model.
            scale = max(abs(init_val), 1e-6)
            noise = cum_returns * beta * scale
            result[var_name] = init_val + noise

    return result


# ---------------------------------------------------------------------------
# Core Monte Carlo engine
# ---------------------------------------------------------------------------


def run_simulation(
    n_paths: int,
    horizon_steps: int,
    current_regime_idx: int,
    transition_matrix: np.ndarray,
    regime_distributions: list[RegimeDistribution],
    initial_values: dict[str, float],
    rng: np.random.Generator,
    *,
    importance_fraction: float = 0.3,
    importance_tilt: float = DEFAULT_IS_TILT,
    survival_thresholds: dict[str, tuple[str, float]] | None = None,
    variable_sensitivities: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run Monte Carlo simulation for a single horizon.

    Combines nominal and importance-sampled paths, computes survival
    across all paths, and returns weighted survival indicators.

    Parameters
    ----------
    n_paths:
        Total number of simulation paths.
    horizon_steps:
        Number of steps to simulate.
    current_regime_idx:
        Starting regime.
    transition_matrix:
        Regime transition matrix.
    regime_distributions:
        Distribution per regime.
    initial_values:
        Starting variable values for survival checks.
    rng:
        Random generator.
    importance_fraction:
        Fraction of paths allocated to importance sampling.
    importance_tilt:
        Tilt factor for importance sampling.
    survival_thresholds:
        Override survival trigger thresholds.
    variable_sensitivities:
        Variable sensitivity to returns.

    Returns
    -------
    (survival_flags, weights, effective_sample_size)
        ``survival_flags[i]`` is 1.0 if path ``i`` survives, else 0.0.
        ``weights[i]`` is the importance weight for path ``i``.
        ``effective_sample_size`` is the ESS of the weighted sample.
    """
    if survival_thresholds is None:
        survival_thresholds = DEFAULT_SURVIVAL_THRESHOLDS

    n_nominal = max(1, int(n_paths * (1 - importance_fraction)))
    n_importance = n_paths - n_nominal

    all_survival = []
    all_weights = []

    # Nominal paths (no tilt).
    if n_nominal > 0:
        ret_nom, lw_nom = simulate_return_paths(
            n_nominal,
            horizon_steps,
            current_regime_idx,
            transition_matrix,
            regime_distributions,
            rng,
            importance_tilt=0.0,
        )

        vars_nom = evolve_variables(
            ret_nom, initial_values, variable_sensitivities,
        )

        for i in range(n_nominal):
            terminal = {
                var: float(vals[i, -1])
                for var, vals in vars_nom.items()
            }
            breached = check_survival_triggers(terminal, survival_thresholds)
            all_survival.append(0.0 if breached else 1.0)
            all_weights.append(1.0)

    # Importance-sampled paths (tilted toward danger).
    if n_importance > 0:
        ret_is, lw_is = simulate_return_paths(
            n_importance,
            horizon_steps,
            current_regime_idx,
            transition_matrix,
            regime_distributions,
            rng,
            importance_tilt=importance_tilt,
        )

        vars_is = evolve_variables(
            ret_is, initial_values, variable_sensitivities,
        )

        for i in range(n_importance):
            terminal = {
                var: float(vals[i, -1])
                for var, vals in vars_is.items()
            }
            breached = check_survival_triggers(terminal, survival_thresholds)
            all_survival.append(0.0 if breached else 1.0)
            # Importance weight = exp(log_weight).
            all_weights.append(float(np.exp(lw_is[i])))

    survival_arr = np.array(all_survival)
    weight_arr = np.array(all_weights)

    # Normalize weights.
    w_sum = weight_arr.sum()
    if w_sum > 0:
        weight_arr = weight_arr / w_sum
    else:
        weight_arr = np.ones(len(weight_arr)) / len(weight_arr)

    # Effective sample size.
    ess = 1.0 / np.sum(weight_arr ** 2) if np.sum(weight_arr ** 2) > 0 else 0.0

    return survival_arr, weight_arr, ess


def bootstrap_survival_probability(
    survival_flags: np.ndarray,
    weights: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Compute bootstrapped survival probability with confidence intervals.

    Parameters
    ----------
    survival_flags:
        1-D array of survival indicators (1=survived, 0=breached).
    weights:
        Normalized importance weights.
    n_bootstrap:
        Number of bootstrap resamples.
    rng:
        Random generator.

    Returns
    -------
    Dict with ``mean``, ``std``, ``p5``, ``p25``, ``median``, ``p75``,
    ``p95`` of the survival probability distribution.
    """
    n = len(survival_flags)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p5": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
        }

    # Weighted point estimate.
    point_estimate = float(np.dot(weights, survival_flags))

    # Bootstrap.
    boot_probs = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_w = weights[idx]
        boot_s = survival_flags[idx]
        # Renormalize.
        bw_sum = boot_w.sum()
        if bw_sum > 0:
            boot_w = boot_w / bw_sum
        boot_probs[b] = float(np.dot(boot_w, boot_s))

    return {
        "mean": float(np.mean(boot_probs)),
        "std": float(np.std(boot_probs)),
        "p5": float(np.percentile(boot_probs, 5)),
        "p25": float(np.percentile(boot_probs, 25)),
        "median": float(np.percentile(boot_probs, 50)),
        "p75": float(np.percentile(boot_probs, 75)),
        "p95": float(np.percentile(boot_probs, 95)),
    }


# ---------------------------------------------------------------------------
# Initial value extraction
# ---------------------------------------------------------------------------


def extract_initial_values(
    cache: pd.DataFrame,
    variables: list[str] | None = None,
) -> dict[str, float]:
    """Extract the most recent values for survival-trigger variables.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    variables:
        Variables to extract.  Defaults to survival trigger variables.

    Returns
    -------
    Dict of ``{variable_name: latest_non_nan_value}``.
    """
    if variables is None:
        variables = list(DEFAULT_SURVIVAL_THRESHOLDS.keys())

    result: dict[str, float] = {}

    for var in variables:
        if var not in cache.columns:
            continue

        series = cache[var].dropna()
        if len(series) > 0:
            result[var] = float(series.iloc[-1])
        else:
            result[var] = float("nan")

    return result


def detect_current_regime(
    cache: pd.DataFrame,
    regime_col: str = "regime_label",
) -> str:
    """Determine the current (most recent) regime from the cache.

    Parameters
    ----------
    cache:
        Daily cache with regime labels.
    regime_col:
        Column name for the regime label.

    Returns
    -------
    The most recent non-NaN regime label, or ``"unknown"`` if
    unavailable.
    """
    if regime_col not in cache.columns:
        return "unknown"

    labels = cache[regime_col].dropna()
    if len(labels) == 0:
        return "unknown"

    return str(labels.iloc[-1])


# ===========================================================================
# Pipeline entry point
# ===========================================================================


def run_monte_carlo(
    cache: pd.DataFrame,
    *,
    n_paths: int = DEFAULT_N_PATHS,
    horizons: dict[str, int] | None = None,
    importance_fraction: float = 0.3,
    importance_tilt: float = DEFAULT_IS_TILT,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_state: int = 42,
    survival_thresholds: dict[str, tuple[str, float]] | None = None,
    variable_sensitivities: dict[str, float] | None = None,
    regime_col: str = "regime_label",
    returns_col: str = "return_1d",
    burnout_distributions: dict[str, dict[str, float]] | None = None,
) -> MonteCarloResult:
    """Run the full Monte Carlo simulation pipeline.

    Estimates survival probabilities at multiple horizons by simulating
    regime-aware return paths and checking whether survival triggers
    are breached at each horizon endpoint.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with regime labels and financial variables.
    n_paths:
        Total number of simulation paths per horizon.
    horizons:
        ``{label: n_steps}`` horizons to simulate.  Defaults to
        ``DEFAULT_HORIZONS`` (1d, 5d, 21d, 252d).
    importance_fraction:
        Fraction of paths using importance sampling.
    importance_tilt:
        Importance sampling tilt factor.
    n_bootstrap:
        Number of bootstrap resamples for confidence intervals.
    random_state:
        Seed for reproducibility.
    survival_thresholds:
        Override survival trigger thresholds.
    variable_sensitivities:
        Override variable sensitivity to returns.
    regime_col:
        Column name for regime labels.
    returns_col:
        Column name for daily returns.

    Returns
    -------
    ``MonteCarloResult`` with survival probabilities and diagnostics.
    """
    logger.info(
        "Starting Monte Carlo simulation: %d paths, %d horizons...",
        n_paths,
        len(horizons or DEFAULT_HORIZONS),
    )

    result = MonteCarloResult(n_paths=n_paths)
    rng = np.random.default_rng(random_state)

    if horizons is None:
        horizons = DEFAULT_HORIZONS

    if survival_thresholds is None:
        survival_thresholds = DEFAULT_SURVIVAL_THRESHOLDS

    # ------------------------------------------------------------------
    # Extract regime information
    # ------------------------------------------------------------------
    current_regime = detect_current_regime(cache, regime_col)
    result.current_regime = current_regime

    if returns_col not in cache.columns:
        result.error = f"Column '{returns_col}' not found in cache"
        logger.warning(result.error)
        return result

    returns = cache[returns_col].values

    # Get regime labels.
    if regime_col in cache.columns:
        regime_labels = cache[regime_col].values
    else:
        # No regime labels -- use a single "unknown" regime.
        regime_labels = np.full(len(cache), "unknown")

    # Determine unique regimes.
    valid_labels = pd.Series(regime_labels).dropna()
    if len(valid_labels) == 0:
        unique_regimes = ["unknown"]
        regime_labels = np.full(len(cache), "unknown")
    else:
        unique_regimes = sorted(valid_labels.unique().astype(str).tolist())

    # ------------------------------------------------------------------
    # Estimate distributions and transitions
    # ------------------------------------------------------------------
    distributions = estimate_regime_distributions(
        returns, regime_labels, unique_regimes,
    )

    # Override with burn-out calibrated distributions when available.
    # Burn-out distributions are model-weighted (incorporating ensemble
    # quality) rather than raw sample statistics, producing more realistic
    # tail behavior for survival probability estimation.
    if burnout_distributions:
        _n_overrides = 0
        for regime, params in burnout_distributions.items():
            if regime in distributions and params.get("n_obs", 0) >= 10:
                old = distributions[regime]
                distributions[regime] = RegimeDistribution(
                    regime_label=regime,
                    mean=params["mean"],
                    std=params["std"],
                    n_obs=params["n_obs"],
                )
                _n_overrides += 1
                logger.info(
                    "MC: burn-out override for regime '%s': "
                    "mean %.6f->%.6f, std %.6f->%.6f",
                    regime, old.mean, params["mean"],
                    old.std, params["std"],
                )
        if _n_overrides > 0:
            logger.info(
                "MC: %d/%d regime distributions overridden by burn-out calibration",
                _n_overrides, len(distributions),
            )

    result.regime_distributions = distributions

    transition_matrix, regime_order = estimate_transition_matrix(
        regime_labels, unique_regimes,
    )
    result.transition_matrix = transition_matrix

    # Map current regime to index.
    regime_to_idx = {r: i for i, r in enumerate(regime_order)}
    current_idx = regime_to_idx.get(current_regime, 0)

    # Build ordered distribution list.
    dist_list = [distributions.get(r, RegimeDistribution()) for r in regime_order]

    # ------------------------------------------------------------------
    # Extract initial variable values
    # ------------------------------------------------------------------
    initial_values = extract_initial_values(cache)
    if not initial_values:
        logger.warning(
            "No survival-trigger variables found in cache -- "
            "using neutral defaults",
        )
        initial_values = {
            "current_ratio": 1.5,
            "debt_to_equity_abs": 1.0,
            "fcf_yield": 0.05,
            "drawdown_252d": -0.10,
        }

    # ------------------------------------------------------------------
    # Run simulations per horizon
    # ------------------------------------------------------------------
    all_survival_probs: list[float] = []

    for h_label, h_steps in sorted(horizons.items(), key=lambda x: x[1]):
        logger.info(
            "Simulating horizon '%s' (%d steps)...", h_label, h_steps,
        )

        survival_flags, weights, ess = run_simulation(
            n_paths=n_paths,
            horizon_steps=h_steps,
            current_regime_idx=current_idx,
            transition_matrix=transition_matrix,
            regime_distributions=dist_list,
            initial_values=initial_values,
            rng=rng,
            importance_fraction=importance_fraction,
            importance_tilt=importance_tilt,
            survival_thresholds=survival_thresholds,
            variable_sensitivities=variable_sensitivities,
        )

        # Weighted survival probability (clip to [0, 1] for float safety).
        surv_prob = float(np.clip(np.dot(weights, survival_flags), 0.0, 1.0))
        result.survival_probability[h_label] = surv_prob
        all_survival_probs.append(surv_prob)

        # Bootstrap CI.
        stats = bootstrap_survival_probability(
            survival_flags, weights, n_bootstrap, rng,
        )
        result.survival_stats[h_label] = stats

        result.n_paths_importance = int(n_paths * importance_fraction)

        # Store terminal cumulative return ratios for drawdown distribution.
        # Generate a lightweight batch of return paths (100 paths) to get
        # terminal price ratios without the overhead of full simulation.
        try:
            _n_tv = min(n_paths, 500)
            _tv_returns, _ = simulate_return_paths(
                _n_tv, h_steps, current_idx,
                transition_matrix, dist_list, rng,
                importance_tilt=0.0,
            )
            # Terminal cumulative return as price ratio (e^sum(log_returns))
            _tv_cum = np.exp(np.sum(_tv_returns, axis=1))
            result.terminal_values[h_label] = _tv_cum
        except Exception:
            pass  # non-critical

        logger.info(
            "Horizon '%s': survival_prob=%.4f (p5=%.4f, p95=%.4f), ESS=%.1f",
            h_label,
            surv_prob,
            stats["p5"],
            stats["p95"],
            ess,
        )

    # ------------------------------------------------------------------
    # Summary statistics (across horizons)
    # ------------------------------------------------------------------
    if all_survival_probs:
        probs = np.array(all_survival_probs)
        result.survival_probability_mean = float(np.mean(probs))
        result.survival_probability_p5 = float(np.percentile(probs, 5))
        result.survival_probability_p95 = float(np.percentile(probs, 95))

    result.importance_sampling_used = importance_fraction > 0
    result.effective_sample_size = ess if all_survival_probs else float("nan")
    result.fitted = True

    # ------------------------------------------------------------------
    # Max drawdown distribution across MC paths (per horizon)
    # ------------------------------------------------------------------
    for h_label, tv in result.terminal_values.items():
        if tv is not None and len(tv) > 0:
            try:
                # terminal_values are cumulative return ratios (path / start)
                # For max drawdown, we need the full path -- approximate from
                # terminal values using the worst-case peak-to-trough ratio.
                # Since we don't store full paths, use: max_dd ~ 1 - min(tv)/max(tv)
                # For more accurate estimates, compute drawdown from the
                # terminal distribution assuming geometric Brownian motion.
                tv_arr = np.asarray(tv, dtype=float)
                # Approximate max drawdown from terminal returns:
                # Use the Magdon-Ismail formula: E[MDD] ~ sqrt(2 * T * sigma^2 / pi)
                # where sigma is the per-step vol and T is the horizon.
                h_steps = {"1d": 1, "5d": 5, "21d": 21, "252d": 252}.get(h_label, 21)
                log_returns = np.log(np.maximum(tv_arr, 1e-10))
                sigma = float(np.std(log_returns)) / max(np.sqrt(h_steps), 1)
                # Simulated max drawdowns: each path's worst loss from peak
                # Approximate: max_dd per path ~ -|min(0, log_return)| scaled
                path_drawdowns = np.minimum(0, log_returns)
                # Scale by sqrt(T) for multi-step paths
                scale = max(np.sqrt(h_steps / 252), 0.1)
                path_max_dd = path_drawdowns * scale
                # For terminal-only data, estimate drawdown distribution
                result.max_drawdown_distribution[h_label] = {
                    "median": round(float(np.median(path_max_dd)), 4),
                    "p10": round(float(np.percentile(path_max_dd, 10)), 4),
                    "p90": round(float(np.percentile(path_max_dd, 90)), 4),
                    "mean": round(float(np.mean(path_max_dd)), 4),
                    "worst": round(float(np.min(path_max_dd)), 4),
                }
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Summary log
    # ------------------------------------------------------------------
    logger.info(
        "Monte Carlo complete: %d paths x %d horizons, "
        "overall survival_mean=%.4f, p5=%.4f, p95=%.4f, "
        "current_regime='%s', IS_used=%s",
        n_paths,
        len(horizons),
        result.survival_probability_mean,
        result.survival_probability_p5,
        result.survival_probability_p95,
        current_regime,
        result.importance_sampling_used,
    )

    return result


# ---------------------------------------------------------------------------
# Multivariate Monte Carlo (Proposal 3.3)
# ---------------------------------------------------------------------------


def run_multivariate_monte_carlo(
    cache: pd.DataFrame,
    copula_correlation: np.ndarray | None = None,
    *,
    n_paths: int = 5000,
    horizon_steps: int = 252,
    random_state: int = 42,
    survival_thresholds: dict[str, tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Multivariate Monte Carlo that jointly simulates financial ratios.

    Instead of simulating returns alone and using a proxy mapping to ratios,
    this jointly simulates (return, delta_current_ratio, delta_fcf_yield,
    delta_debt_to_equity) using the copula correlation structure.

    Survival triggers are checked on the simulated ratios directly.

    Parameters
    ----------
    cache:
        Daily cache with return_1d and financial ratio columns.
    copula_correlation:
        Correlation matrix from copula analysis. If None, uses empirical
        correlation from the cache.
    n_paths:
        Number of simulation paths.
    horizon_steps:
        Number of days to simulate.
    random_state:
        Seed for reproducibility.
    survival_thresholds:
        Survival trigger thresholds.

    Returns
    -------
    Dict with multivariate survival probability and per-variable terminal
    distributions.
    """
    if survival_thresholds is None:
        survival_thresholds = DEFAULT_SURVIVAL_THRESHOLDS

    rng = np.random.default_rng(random_state)

    # Variables to jointly simulate
    sim_vars = ["return_1d", "current_ratio", "debt_to_equity_abs", "fcf_yield"]
    available = [v for v in sim_vars if v in cache.columns and cache[v].notna().sum() > 30]

    if len(available) < 2:
        return {"available": False, "error": "Insufficient variables for multivariate MC"}

    # Compute daily changes for ratio variables
    change_data = pd.DataFrame(index=cache.index)
    var_is_change: dict[str, bool] = {}
    for v in available:
        if v == "return_1d":
            change_data[v] = cache[v]
            var_is_change[v] = False
        else:
            change_data[v] = cache[v].diff()
            var_is_change[v] = True

    clean = change_data[available].dropna()
    if len(clean) < 50:
        return {"available": False, "error": "Insufficient clean data for multivariate MC"}

    # Correlation structure
    if copula_correlation is not None and copula_correlation.shape[0] >= len(available):
        corr = copula_correlation[:len(available), :len(available)]
    else:
        corr = clean[available].corr().values
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() <= 0:
            corr += np.eye(len(available)) * (abs(eigvals.min()) + 0.01)

    # Per-variable mean and std
    means = clean[available].mean().values
    stds = clean[available].std().values
    stds[stds < 1e-10] = 1e-6

    # Initial values for ratios
    initial: dict[str, float] = {}
    for v in available:
        if v == "return_1d":
            continue
        col = cache[v].dropna()
        initial[v] = float(col.iloc[-1]) if len(col) > 0 else 1.0

    # Simulate paths using multivariate normal with copula correlation
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Fallback: add diagonal noise
        corr_safe = corr + np.eye(len(available)) * 0.01
        L = np.linalg.cholesky(corr_safe)

    survival_count = 0
    terminal_values: dict[str, list[float]] = {v: [] for v in available if v != "return_1d"}

    for _ in range(n_paths):
        # Simulate correlated changes
        z = rng.standard_normal((horizon_steps, len(available)))
        correlated = z @ L.T  # apply correlation structure
        changes = correlated * stds + means

        # Track ratio levels
        ratios: dict[str, float] = dict(initial)
        triggered = False

        for t in range(horizon_steps):
            for vi, v in enumerate(available):
                if v == "return_1d":
                    continue
                ratios[v] = ratios.get(v, 1.0) + float(changes[t, vi])

            # Check survival triggers on actual ratios
            if check_survival_triggers(ratios, survival_thresholds):
                triggered = True
                break

        if not triggered:
            survival_count += 1

        for v in terminal_values:
            terminal_values[v].append(ratios.get(v, 0.0))

    survival_prob = survival_count / n_paths

    # Terminal distribution statistics
    terminal_stats: dict[str, dict[str, float]] = {}
    for v, vals in terminal_values.items():
        arr = np.array(vals)
        terminal_stats[v] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "p5": round(float(np.percentile(arr, 5)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
        }

    logger.info(
        "Multivariate MC: %d paths x %d steps, survival=%.4f, vars=%s",
        n_paths, horizon_steps, survival_prob, available,
    )

    return {
        "available": True,
        "survival_probability": round(survival_prob, 4),
        "n_paths": n_paths,
        "horizon_steps": horizon_steps,
        "variables_simulated": available,
        "terminal_distributions": terminal_stats,
    }

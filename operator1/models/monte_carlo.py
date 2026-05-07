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


# ---------------------------------------------------------------------------
# Market-cap survival floor (Fama-French size quintile calibration)
# ---------------------------------------------------------------------------

def get_mcap_survival_floor(market_cap: float) -> float:
    """Return minimum credible survival probability based on market cap quintile.

    Based on Fama-French size quintile analysis: mega-caps almost never
    experience >40% drawdowns that trigger survival mode.
    """
    if market_cap >= 200e9:
        return 0.92   # mega-cap: >$200B
    elif market_cap >= 10e9:
        return 0.82   # large-cap: $10B-$200B
    elif market_cap >= 2e9:
        return 0.70   # mid-cap: $2B-$10B
    elif market_cap >= 300e6:
        return 0.55   # small-cap: $300M-$2B
    else:
        return 0.40   # micro-cap: <$300M


def anchor_mc_survival(
    mc_result: "MonteCarloResult | None",
    market_cap: float | None = None,
    merton_pd: float | None = None,
) -> None:
    """Anchor MC survival probabilities using market-cap floor and Merton default.

    Modifies mc_result.survival_probability in place. Applied after MC simulation
    and optionally again after HF analysis produces Merton default probability.
    """
    if mc_result is None:
        return

    floor = 0.0
    source = ""
    if market_cap is not None and market_cap > 0:
        _mcap_floor = get_mcap_survival_floor(market_cap)
        if _mcap_floor > floor:
            floor = _mcap_floor
            source = "mcap"
    if merton_pd is not None and 0 < merton_pd < 1:
        _merton_floor = 1.0 - merton_pd
        if _merton_floor > floor:
            floor = _merton_floor
            source = "merton"

    if floor <= 0:
        return

    _logger = logging.getLogger(__name__)
    for horizon, data in mc_result.survival_probability.items():
        if isinstance(data, dict) and "mean" in data:
            if data["mean"] < floor:
                _logger.info(
                    "MC survival anchor [%s]: %.1f%% -> %.1f%% (floor=%s)",
                    horizon, data["mean"] * 100, floor * 100, source,
                )
                data["mean"] = floor
                data["floor_source"] = source

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


def _build_frequency_aware_horizons(cache_len: int) -> dict[str, int]:
    """Build horizons scaled to the cache length (frequency-adaptive).

    At daily (502 rows): {1d:1, 5d:5, 21d:21, 252d:252} -- standard
    At weekly (158 rows): {1p:1, 4p:4, 13p:13, 52p:52} -- ~week/month/quarter/year
    At monthly (61 rows): {1p:1, 3p:3, 6p:6, 12p:12}
    At quarterly (15 rows): {1p:1, 2p:2, 4p:4, 8p:8}
    At annual (6 rows): {1p:1, 2p:2, 3p:3, 5p:5}
    """
    if cache_len >= 400:
        return DEFAULT_HORIZONS  # daily, use standard
    # Scale horizons proportionally to available data
    spy = max(4, cache_len)  # steps per ~year equivalent
    return {
        "1p": 1,
        "short": max(1, spy // 12),
        "mid": max(1, spy // 4),
        "long": max(2, spy),
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
    use_student_t: bool = False  # True when Jarque-Bera rejects normality
    df_t: float = 30.0           # degrees of freedom for Student-t
    # E1: EVT (Extreme Value Theory) tail parameters via GPD
    evt_fitted: bool = False
    evt_xi: float = 0.0          # shape parameter (tail heaviness)
    evt_scale: float = 0.01      # scale parameter
    evt_threshold: float = -0.03  # P5 threshold for exceedances


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

    # Regime label order corresponding to transition matrix rows/columns.
    regime_order: list[str] = field(default_factory=list)

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

    # E2: Forward-looking survival -- fraction of paths that trigger
    # ANY survival condition at ANY point along the path (not just terminal).
    # Standard in credit risk (first-passage-time models) but novel in equity.
    anticipated_survival: dict[str, float] = field(default_factory=dict)

    # Product concentration risk: when segment_hhi > 0.5, the dominant
    # product drives most of the company's risk.  Flag for downstream
    # consumers to apply concentrated-risk adjustments.
    concentration_risk_flag: bool = False
    segment_hhi: float = 0.0

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

            # Jarque-Bera test: if normality is rejected (p < 0.05),
            # fit Student-t which captures fat tails (Mandelbrot 1963).
            # Financial returns in crisis regimes often have kurtosis > 3
            # causing Normal MC to underestimate tail probability by 10-100x.
            if len(regime_returns) >= 30:
                try:
                    from scipy.stats import jarque_bera, t as t_dist
                    _jb_stat, _jb_p = jarque_bera(regime_returns)
                    if _jb_p < 0.05:
                        _t_params = t_dist.fit(regime_returns)
                        dist.use_student_t = True
                        dist.df_t = max(2.1, float(_t_params[0]))  # df >= 2.1 for finite variance
                        logger.info(
                            "Regime '%s': Jarque-Bera p=%.4f, using Student-t(df=%.1f)",
                            regime, _jb_p, dist.df_t,
                        )
                except Exception as _exc:
                    logger.debug("Student-t fit skipped for regime '%s': %s", regime, _exc)

            # E1: EVT tail calibration via GPD (Peaks-Over-Threshold).
            # Fits Generalized Pareto Distribution to return tails (below P5)
            # for more accurate crisis probability estimation.
            # McNeil & Frey (2000), standard in bank risk management.
            if len(regime_returns) >= 50:
                try:
                    from scipy.stats import genpareto
                    _lower_threshold = float(np.percentile(regime_returns, 5))
                    _exceedances = _lower_threshold - regime_returns[regime_returns < _lower_threshold]
                    if len(_exceedances) >= 5:
                        _xi, _loc, _scale = genpareto.fit(_exceedances, floc=0)
                        dist.evt_xi = float(_xi)
                        dist.evt_scale = float(_scale)
                        dist.evt_threshold = float(_lower_threshold)
                        dist.evt_fitted = True
                        logger.debug(
                            "EVT GPD for regime '%s': xi=%.3f, scale=%.4f, threshold=%.4f, n_exceed=%d",
                            regime, _xi, _scale, _lower_threshold, len(_exceedances),
                        )
                except Exception as _evt_exc:
                    logger.debug("EVT GPD skipped for regime '%s': %s", regime, _evt_exc)
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
    jump_lambda: float = 0.0,
    jump_mean: float = 0.0,
    jump_std: float = 0.01,
    antithetic: bool = False,
    ath_barrier: float = 0.0,
    support_barrier: float = 0.0,
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
    # Antithetic variates (Hammersley & Handscomb 1964): generate half
    # the paths, then mirror random draws for the other half. Cuts
    # variance by ~2x for the same computational cost. Only used when
    # importance sampling is NOT active (antithetic and IS conflict).
    effective_n = n_paths
    use_antithetic = antithetic and importance_tilt <= 0 and n_paths >= 4
    if use_antithetic:
        effective_n = (n_paths + 1) // 2  # generate half, mirror the rest

    return_paths = np.zeros((effective_n, n_steps))
    log_weights = np.zeros(effective_n)

    # Jump-diffusion parameters (Merton 1976): dt = 1 day
    _dt = 1.0 / 252.0
    _jump_active = jump_lambda > 0 and jump_std > 0

    # Beyond Bands Method 5: Reflected Brownian Motion at boundaries (Harrison 1985).
    # When price approaches ATH or major support, it tends to bounce back rather
    # than break through. Reflection creates naturally asymmetric distributions
    # near boundaries -- left-skewed near ATH (limited upside), right-skewed
    # near support (limited downside). Only active when both barriers are set.
    _barriers_active = (ath_barrier > 0 and support_barrier > 0
                        and ath_barrier > support_barrier)
    if _barriers_active:
        # Convert price barriers to cumulative log-return barriers
        # relative to current price (mid-point between barriers as reference)
        _start_price_ref = (ath_barrier + support_barrier) / 2.0
        _ath_log = np.log(ath_barrier / _start_price_ref)
        _support_log = np.log(support_barrier / _start_price_ref)

    for i in range(effective_n):
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
            # Use Student-t when Jarque-Bera rejected normality for this
            # regime (fat tails -- Mandelbrot 1963).  Student-t with low df
            # generates more extreme returns, improving tail probability
            # estimates that Normal MC underestimates by 10-100x.
            if getattr(dist, "use_student_t", False) and dist.df_t > 2.0:
                try:
                    from scipy.stats import t as t_dist
                    z = t_dist.rvs(dist.df_t, loc=tilted_mean, scale=tilted_std, random_state=rng)
                except Exception:
                    z = rng.normal(tilted_mean, tilted_std)
            else:
                z = rng.normal(tilted_mean, tilted_std)

            # Jump-diffusion component (Merton 1976): add Poisson-distributed
            # jumps to the continuous diffusion. Calibrated from Layer 1
            # jump_spike_flag statistics. Captures sudden discontinuities
            # (Black Monday, Flash Crash) that regime switching alone misses.
            if _jump_active:
                n_jumps = rng.poisson(jump_lambda * _dt)
                if n_jumps > 0:
                    jump_return = rng.normal(jump_mean, jump_std, size=n_jumps).sum()
                    z += jump_return

            return_paths[i, t] = z

            # Beyond Bands Method 5: Barrier reflection (Harrison 1985).
            # Reflect cumulative return if the implied price breaches ATH
            # or support. This creates naturally asymmetric terminal
            # distributions near boundaries.
            if _barriers_active:
                _cum_log_return = np.sum(return_paths[i, :t + 1])
                if _cum_log_return > _ath_log:
                    _excess = _cum_log_return - _ath_log
                    return_paths[i, t] -= 2.0 * _excess
                elif _cum_log_return < _support_log:
                    _deficit = _support_log - _cum_log_return
                    return_paths[i, t] += 2.0 * _deficit

            if importance_tilt > 0:
                # Log importance weight: log(p_nominal / p_tilted).
                log_p_nom = -0.5 * ((z - nominal_mean) / nominal_std) ** 2
                log_p_tilt = -0.5 * ((z - tilted_mean) / tilted_std) ** 2
                path_log_w += log_p_nom - log_p_tilt

        log_weights[i] = path_log_w

    # Apply antithetic mirroring: negate return paths for the second half
    if use_antithetic:
        mirror_paths = -return_paths  # mirror all draws
        return_paths = np.concatenate([return_paths, mirror_paths], axis=0)[:n_paths]
        log_weights = np.concatenate([log_weights, log_weights], axis=0)[:n_paths]

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
            # Cap filing_interval to n_steps so short simulations still
            # produce meaningful terminal values (not flat lines).
            effective_interval = min(filing_interval, max(1, n_steps - 1))
            scale = max(abs(init_val), 1e-6)
            evolved = np.full((n_paths, n_steps), init_val)
            last_filing_t = 0
            for t in range(1, n_steps):
                if t % effective_interval == 0:
                    # Filing day: update based on cumulative returns since
                    # last filing.
                    cum_since_filing = cum_returns[:, t] - cum_returns[:, last_filing_t]
                    evolved[:, t] = evolved[:, t - 1] + beta * cum_since_filing * scale
                    last_filing_t = t
                else:
                    # Non-filing day: carry forward previous value.
                    evolved[:, t] = evolved[:, t - 1]
            # Always apply a final-step update if the last step wasn't a
            # filing day, so terminal values reflect cumulative returns.
            if (n_steps - 1) % effective_interval != 0 and n_steps > 1:
                cum_since_filing = cum_returns[:, -1] - cum_returns[:, last_filing_t]
                evolved[:, -1] = evolved[:, -2] + beta * cum_since_filing * scale
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
    jump_lambda: float = 0.0,
    jump_mean: float = 0.0,
    jump_std: float = 0.0,
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
    jump_lambda:
        Jump-diffusion Poisson intensity (annualized).
    jump_mean:
        Mean of log-normal jump size.
    jump_std:
        Std of log-normal jump size.

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
            jump_lambda=jump_lambda,
            jump_mean=jump_mean,
            jump_std=jump_std,
            antithetic=True,
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
    # Jump params passed (jumps are real events), but antithetic=False
    # because antithetic variates conflict with importance sampling.
    if n_importance > 0:
        ret_is, lw_is = simulate_return_paths(
            n_importance,
            horizon_steps,
            current_regime_idx,
            transition_matrix,
            regime_distributions,
            rng,
            importance_tilt=importance_tilt,
            jump_lambda=jump_lambda,
            jump_mean=jump_mean,
            jump_std=jump_std,
            antithetic=False,
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


# ---------------------------------------------------------------------------
# E2: Forward-looking (path-wise) survival trigger checking
# ---------------------------------------------------------------------------


def compute_anticipated_survival(
    cache: pd.DataFrame,
    mc_result: "MonteCarloResult",
    horizon_days: int = 63,
    n_paths: int = 5000,
    random_state: int = 42,
) -> float:
    """E2: Compute fraction of MC paths that trigger ANY survival condition
    at ANY point along the path (not just terminal).

    Standard in credit risk as 'first-passage-time' but novel in equity.
    If 40% of paths trigger survival within 63 days, the market will
    price the distress in immediately.

    Parameters
    ----------
    cache:
        Daily cache with survival trigger variables.
    mc_result:
        Completed MC result with regime distributions and transition matrix.
    horizon_days:
        How far forward to check (default 63 = one quarter).
    n_paths:
        Number of simulation paths.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    Float in [0, 1]: fraction of paths that trigger survival at any point.
    """
    if not mc_result.fitted or mc_result.transition_matrix is None:
        return 0.0

    rng = np.random.default_rng(random_state + 999)

    # Get initial variable values
    initial = extract_initial_values(cache)
    if not initial:
        return 0.0

    # Get current regime
    regime_to_idx = {r: i for i, r in enumerate(mc_result.regime_order)}
    current_idx = regime_to_idx.get(mc_result.current_regime, 0)
    n_regimes = len(mc_result.regime_order)

    # Build distribution parameters
    dist_list = [
        mc_result.regime_distributions.get(r, RegimeDistribution())
        for r in mc_result.regime_order
    ]

    # Default thresholds
    thresholds = DEFAULT_SURVIVAL_THRESHOLDS

    # Default variable sensitivities to returns
    sensitivities = {
        "current_ratio": -0.5,
        "debt_to_equity_abs": 0.3,
        "fcf_yield": -0.8,
        "drawdown_252d": 1.0,
    }

    n_triggered = 0

    for _p in range(n_paths):
        regime_idx = current_idx
        cumulative_return = 0.0
        triggered = False

        # Copy initial values
        _vals = dict(initial)

        for _t in range(horizon_days):
            # Sample regime transition
            probs = mc_result.transition_matrix[regime_idx]
            regime_idx = int(rng.choice(n_regimes, p=probs))

            # Sample return from regime distribution
            dist = dist_list[regime_idx]
            if dist.use_student_t and dist.df_t > 2:
                from scipy.stats import t as t_dist
                r = float(t_dist.rvs(dist.df_t, loc=dist.mean, scale=dist.std, random_state=rng))
            else:
                r = float(rng.normal(dist.mean, dist.std))

            cumulative_return += r

            # Update variable proxies
            for var, (direction, threshold) in thresholds.items():
                if var not in _vals:
                    continue
                sens = sensitivities.get(var, 0.0)
                if var == "drawdown_252d":
                    _vals[var] = min(_vals[var], cumulative_return)
                else:
                    _vals[var] += sens * r * _vals[var]

                # Check trigger
                if direction == "lt" and _vals[var] < threshold:
                    triggered = True
                    break
                elif direction == "gt" and _vals[var] > threshold:
                    triggered = True
                    break

            if triggered:
                break

        if triggered:
            n_triggered += 1

    anticipated = n_triggered / max(n_paths, 1)
    logger.info(
        "E2 anticipated survival: %.1f%% of paths trigger within %d days",
        anticipated * 100, horizon_days,
    )
    return anticipated


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
    jump_params: dict[str, float] | None = None,
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

    # Jump-diffusion parameters (Merton 1976): calibrate from cache if
    # jump_spike_flag is available (from Layer 1 Stage 16 vol decomposition).
    # When jump_params is None, auto-calibrate from data; when explicitly
    # passed, use the provided values.
    jump_lambda = 0.0
    jump_mean = 0.0
    jump_std = 0.01
    if jump_params is not None:
        jump_lambda = jump_params.get("lambda", 0.0)
        jump_mean = jump_params.get("mean", 0.0)
        jump_std = jump_params.get("std", 0.01)
    elif "jump_spike_flag" in cache.columns:
        _spikes = cache["jump_spike_flag"].fillna(0)
        if _spikes.sum() > 0:
            jump_lambda = float(_spikes.mean() * 252)  # annualized
            _spike_mask = _spikes.astype(bool)
            if returns_col in cache.columns:
                _spike_returns = cache[returns_col].loc[_spike_mask].dropna()
                if len(_spike_returns) > 2:
                    jump_mean = float(_spike_returns.mean())
                    jump_std = float(max(_spike_returns.std(), 0.005))
            # Cap lambda at 50 (max ~1 jump per 5 trading days)
            jump_lambda = min(jump_lambda, 50.0)
            logger.info(
                "Jump-diffusion calibrated: lambda=%.1f/yr, jump_mu=%.4f, jump_sigma=%.4f",
                jump_lambda, jump_mean, jump_std,
            )

    if horizons is None:
        horizons = _build_frequency_aware_horizons(len(cache))

    if survival_thresholds is None:
        survival_thresholds = DEFAULT_SURVIVAL_THRESHOLDS

    # Beyond Bands Method 5: Extract ATH and support barriers for reflected BM.
    # ATH = 252-day rolling max; support = 252-day rolling min.
    # Barriers are only active when we have sufficient price history.
    _ath_barrier = 0.0
    _support_barrier = 0.0
    if "close" in cache.columns and cache["close"].notna().sum() >= 63:
        _close = cache["close"].dropna()
        _ath_barrier = float(_close.max())
        _support_barrier = float(_close.rolling(252, min_periods=63).min().iloc[-1])
        if _ath_barrier > _support_barrier > 0:
            logger.info(
                "MC barrier reflection: ATH=%.2f, support=%.2f",
                _ath_barrier, _support_barrier,
            )

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

    # ------------------------------------------------------------------
    # A1: Crisis archetype injection (Plan v2 Category A).
    # Inject historical crisis distributions as additional regimes so MC
    # can simulate tail events the training window never saw. The crisis
    # probability is calibrated from current macro conditions.
    # ------------------------------------------------------------------
    _crisis_archetypes_injected = 0
    try:
        from operator1.config_loader import load_config as _load_cfg
        _crisis_cfg = _load_cfg("crisis_archetypes")
        _archetypes = _crisis_cfg.get("archetypes", {})
        _base_p_crisis = float(_crisis_cfg.get("base_crisis_probability", 0.05))
        _max_p_crisis = float(_crisis_cfg.get("max_crisis_probability", 0.30))

        if _archetypes:
            # Calibrate crisis probability from current conditions
            _p_crisis = _base_p_crisis

            # Increase if survival intensity is elevated
            if "survival_intensity" in cache.columns:
                _si = cache["survival_intensity"].dropna()
                if len(_si) > 0:
                    _p_crisis += float(_si.iloc[-1]) * 0.15

            # Increase if conflict risk is elevated
            if "conflict_intensity_score" in cache.columns:
                _ci = cache["conflict_intensity_score"].dropna()
                if len(_ci) > 0 and float(_ci.iloc[-1]) > 0.3:
                    _p_crisis += 0.05

            # Increase if macro quadrant is adverse
            if "macro_quadrant" in cache.columns:
                _mq = cache["macro_quadrant"].dropna()
                if len(_mq) > 0:
                    _latest_q = str(_mq.iloc[-1]).lower()
                    if _latest_q in ("stagflation", "recession"):
                        _p_crisis += 0.10
                    elif _latest_q == "overheating":
                        _p_crisis += 0.03

            _p_crisis = min(_p_crisis, _max_p_crisis)

            if _p_crisis > 0.01:
                # Select 1-2 most relevant archetypes based on conditions
                _selected = list(_archetypes.items())[:2]
                for _arch_name, _arch in _selected:
                    _crisis_regime = f"crisis_{_arch_name}"
                    distributions[_crisis_regime] = RegimeDistribution(
                        regime_label=_crisis_regime,
                        mean=float(_arch.get("mean_daily_return", -0.002)),
                        std=float(_arch.get("std_daily_return", 0.03)),
                        n_obs=int(_arch.get("typical_duration_days", 126)),
                        use_student_t=True,
                        df_t=float(_arch.get("df_t", 5.0)),
                    )
                    unique_regimes.append(_crisis_regime)
                    _crisis_archetypes_injected += 1

                if _crisis_archetypes_injected > 0:
                    logger.info(
                        "A1 crisis archetypes: injected %d archetypes, P(crisis)=%.3f",
                        _crisis_archetypes_injected, _p_crisis,
                    )
    except FileNotFoundError:
        pass  # crisis_archetypes.yml not present -- skip gracefully
    except Exception as _exc:
        logger.debug("Crisis archetype injection skipped: %s", _exc)

    transition_matrix, regime_order = estimate_transition_matrix(
        regime_labels, unique_regimes,
    )

    # Patch transition matrix for injected crisis archetypes: add rows/cols
    # with small transition probability from all regimes to crisis.
    if _crisis_archetypes_injected > 0 and transition_matrix is not None:
        try:
            _n_orig = transition_matrix.shape[0] - _crisis_archetypes_injected
            _p_to_crisis = _p_crisis / max(_crisis_archetypes_injected, 1)
            for _ci in range(_n_orig, transition_matrix.shape[0]):
                # From any normal regime -> crisis: small probability
                for _ri in range(_n_orig):
                    transition_matrix[_ri, _ci] = _p_to_crisis
                # From crisis -> crisis: high persistence (crises last)
                transition_matrix[_ci, _ci] = 0.92
                # From crisis -> normal regimes: small recovery probability
                for _ri in range(_n_orig):
                    transition_matrix[_ci, _ri] = (1.0 - 0.92) / max(_n_orig, 1)
            # Renormalize rows
            for _ri in range(transition_matrix.shape[0]):
                _row_sum = transition_matrix[_ri].sum()
                if _row_sum > 0:
                    transition_matrix[_ri] /= _row_sum
        except Exception:
            pass  # matrix patching failed -- use unpatched version

    result.transition_matrix = transition_matrix
    result.regime_order = list(regime_order)

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
    # Options + cross-asset stress boost for importance sampling tilt.
    # When options market signals stress (high put/call ratio, VIX
    # backwardation) or cross-asset stress is elevated, increase the
    # IS tilt toward crisis paths to better sample tail scenarios.
    # ------------------------------------------------------------------
    _stress_tilt_boost = 0.0
    try:
        if "put_call_ratio" in cache.columns and "vix_term_structure" in cache.columns:
            _pcr = cache["put_call_ratio"].iloc[-1] if cache["put_call_ratio"].notna().any() else 0
            _vts = cache["vix_term_structure"].iloc[-1] if cache["vix_term_structure"].notna().any() else 1.0
            if isinstance(_pcr, (int, float)) and isinstance(_vts, (int, float)):
                # PCR > 1.5 = heavy put buying (hedging/fear)
                # VTS > 1.0 = backwardation (near-term fear > long-term)
                if float(_pcr) > 1.5 and float(_vts) > 1.0:
                    _stress_tilt_boost += 0.5  # strong options stress
                    logger.info("MC options stress boost: PCR=%.2f, VTS=%.2f -> +0.5 tilt", _pcr, _vts)
                elif float(_pcr) > 1.2 or float(_vts) > 1.0:
                    _stress_tilt_boost += 0.25  # moderate options stress
        if "cross_asset_stress" in cache.columns:
            _cas = cache["cross_asset_stress"].iloc[-1] if cache["cross_asset_stress"].notna().any() else 0
            if isinstance(_cas, (int, float)) and float(_cas) > 0.7:
                _stress_tilt_boost += 0.25  # cross-asset contagion signal
                logger.info("MC cross-asset stress boost: stress=%.2f -> +0.25 tilt", _cas)
    except Exception:
        pass

    if _stress_tilt_boost > 0:
        importance_tilt = importance_tilt + _stress_tilt_boost
        logger.info("MC importance_tilt boosted to %.2f (base + %.2f stress)", importance_tilt, _stress_tilt_boost)

    # ------------------------------------------------------------------
    # Run simulations per horizon
    # ------------------------------------------------------------------
    all_survival_probs: list[float] = []

    for h_label, h_steps in sorted(horizons.items(), key=lambda x: x[1]):
        logger.info(
            "Simulating horizon '%s' (%d steps)...", h_label, h_steps,
        )

        # P3: Scale IS tilt by horizon to prevent ESS collapse at short
        # horizons. With full tilt at 1-5 steps, all importance weight
        # concentrates on a single particle (ESS=1), producing degenerate
        # survival estimates (0% or 100%). Scale: no tilt below 10 steps,
        # linear ramp from 10 to 63 steps, full tilt above 63 steps.
        _horizon_tilt = importance_tilt
        if h_steps < 10:
            _horizon_tilt = 0.0  # no IS for very short horizons
        elif h_steps < 63:
            _horizon_tilt = importance_tilt * (h_steps - 10) / 53.0
        # else: full tilt for 63+ steps

        survival_flags, weights, ess = run_simulation(
            n_paths=n_paths,
            horizon_steps=h_steps,
            current_regime_idx=current_idx,
            transition_matrix=transition_matrix,
            regime_distributions=dist_list,
            initial_values=initial_values,
            rng=rng,
            importance_fraction=importance_fraction,
            importance_tilt=_horizon_tilt,
            survival_thresholds=survival_thresholds,
            variable_sensitivities=variable_sensitivities,
            jump_lambda=jump_lambda,
            jump_mean=jump_mean,
            jump_std=jump_std,
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

        # Store terminal cumulative return ratios AND full-path max drawdowns.
        # Generate paths to compute both terminal values and realistic
        # max drawdown distribution (not just terminal approximation).
        try:
            _n_tv = min(n_paths, 500)
            _tv_returns, _ = simulate_return_paths(
                _n_tv, h_steps, current_idx,
                transition_matrix, dist_list, rng,
                importance_tilt=0.0,
                ath_barrier=_ath_barrier,
                support_barrier=_support_barrier,
            )
            # Terminal cumulative return as price ratio (e^sum(log_returns))
            _tv_cum = np.exp(np.sum(_tv_returns, axis=1))
            result.terminal_values[h_label] = _tv_cum

            # Compute TRUE max drawdown from full paths (not approximation)
            if h_steps >= 5:
                _cum_returns = np.exp(np.cumsum(_tv_returns, axis=1))  # (n_paths, h_steps)
                _running_max = np.maximum.accumulate(_cum_returns, axis=1)
                _drawdowns = (_cum_returns - _running_max) / np.maximum(_running_max, 1e-10)
                _path_max_dd = np.min(_drawdowns, axis=1)  # worst drawdown per path
                result.max_drawdown_distribution[h_label] = {
                    "median": round(float(np.median(_path_max_dd)), 4),
                    "p10": round(float(np.percentile(_path_max_dd, 10)), 4),
                    "p90": round(float(np.percentile(_path_max_dd, 90)), 4),
                    "mean": round(float(np.mean(_path_max_dd)), 4),
                    "worst": round(float(np.min(_path_max_dd)), 4),
                }
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
    # Max drawdown distribution (backfill for short horizons not computed above)
    # ------------------------------------------------------------------
    for h_label, tv in result.terminal_values.items():
        if h_label not in result.max_drawdown_distribution and tv is not None and len(tv) > 0:
            try:
                # Fallback for 1d horizon where full-path DD was skipped
                tv_arr = np.asarray(tv, dtype=float)
                log_returns = np.log(np.maximum(tv_arr, 1e-10))
                path_drawdowns = np.minimum(0, log_returns)
                result.max_drawdown_distribution[h_label] = {
                    "median": round(float(np.median(path_drawdowns)), 4),
                    "p10": round(float(np.percentile(path_drawdowns, 10)), 4),
                    "p90": round(float(np.percentile(path_drawdowns, 90)), 4),
                    "mean": round(float(np.mean(path_drawdowns)), 4),
                    "worst": round(float(np.min(path_drawdowns)), 4),
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
    # Cap horizon to cache length for lower-frequency caches
    horizon_steps = min(horizon_steps, max(4, len(cache)))

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

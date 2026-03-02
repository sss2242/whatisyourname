"""Particle Filter (Sequential Monte Carlo) for non-linear state estimation.

Spec ref: Core Idea PDF, Section E.2, Module Category 2.

The Particle Filter handles non-linear, non-Gaussian state transitions
that the Kalman filter cannot capture.  It is particularly useful for
survival mode prediction where extreme events cause non-linear jumps.

Key features:
- Maintains a swarm of weighted particles representing possible states.
- Propagates particles through a configurable state transition model.
- Updates weights using observation likelihood.
- Systematic resampling to avoid particle degeneracy.
- Returns full probability distribution, not just point estimate.

Integration:
- Called alongside Kalman filter in the forecasting pipeline.
- Especially useful for Tier 1-2 variables during survival mode.
- Falls back to Kalman if particle count is insufficient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_N_PARTICLES: int = 1000
_MIN_PARTICLES: int = 50
_EFFECTIVE_SAMPLE_THRESHOLD: float = 0.5  # resample when ESS < 50% of N
_STATE_NOISE_SCALE: float = 0.02  # default process noise (2% of state)
_OBS_NOISE_SCALE: float = 0.05  # default observation noise (5% of state)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ParticleFilterResult:
    """Output from particle filter estimation."""

    filtered_states: np.ndarray  # (T, n_states) -- weighted mean per step
    particles_final: np.ndarray  # (N, n_states) -- final particle cloud
    weights_final: np.ndarray  # (N,) -- final normalised weights
    effective_sample_sizes: list[float] = field(default_factory=list)
    n_resamples: int = 0
    n_particles: int = 0
    n_states: int = 0
    fitted: bool = False
    error: str | None = None

    # Distribution outputs
    percentiles: dict[str, np.ndarray] = field(default_factory=dict)
    # {"p5": array, "p50": array, "p95": array} for each state


# ---------------------------------------------------------------------------
# Core Particle Filter
# ---------------------------------------------------------------------------


class ParticleFilter:
    """Sequential Monte Carlo (Particle Filter) for state estimation.

    Parameters
    ----------
    n_particles:
        Number of particles in the swarm.
    n_states:
        Dimensionality of the state vector.
    state_noise_scale:
        Standard deviation of process noise (fraction of state magnitude).
    obs_noise_scale:
        Standard deviation of observation noise (fraction of obs magnitude).
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_particles: int = _DEFAULT_N_PARTICLES,
        n_states: int = 1,
        state_noise_scale: float = _STATE_NOISE_SCALE,
        obs_noise_scale: float = _OBS_NOISE_SCALE,
        random_state: int = 42,
    ) -> None:
        self.n_particles = max(n_particles, _MIN_PARTICLES)
        self.n_states = n_states
        self.state_noise_scale = state_noise_scale
        self.obs_noise_scale = obs_noise_scale
        self._rng = np.random.default_rng(random_state)

        # Particle cloud: (N, n_states)
        self.particles = np.zeros((self.n_particles, self.n_states))
        # Normalised weights: (N,)
        self.weights = np.ones(self.n_particles) / self.n_particles

        self._initialized = False

    def initialize(self, initial_state: np.ndarray, spread: float = 0.1) -> None:
        """Initialise particles around an initial state estimate.

        Parameters
        ----------
        initial_state:
            (n_states,) array -- initial best guess.
        spread:
            How widely to scatter particles around the initial state.
        """
        initial_state = np.asarray(initial_state, dtype=np.float64)
        noise = self._rng.normal(
            0,
            np.abs(initial_state) * spread + 1e-6,
            size=(self.n_particles, self.n_states),
        )
        self.particles = initial_state[np.newaxis, :] + noise
        self.weights = np.ones(self.n_particles) / self.n_particles
        self._initialized = True

    def predict(self, transition_fn=None) -> np.ndarray:
        """Propagate all particles through the state transition model.

        Parameters
        ----------
        transition_fn:
            Callable(particles: ndarray) -> ndarray.
            If None, uses a simple random-walk model with noise.

        Returns
        -------
        Predicted weighted mean state.
        """
        if transition_fn is not None:
            self.particles = transition_fn(self.particles)
        else:
            # Default: random walk with Gaussian noise
            noise = self._rng.normal(
                0,
                np.abs(self.particles) * self.state_noise_scale + 1e-8,
                size=self.particles.shape,
            )
            self.particles = self.particles + noise

        return self.get_state_estimate()

    def update(self, observation: np.ndarray) -> np.ndarray:
        """Update particle weights given a new observation.

        Uses Gaussian likelihood: p(obs | particle) ~ N(particle, sigma).

        Parameters
        ----------
        observation:
            (n_states,) array of observed values.  NaN values are skipped
            (partial observations supported).

        Returns
        -------
        Updated weighted mean state.
        """
        observation = np.asarray(observation, dtype=np.float64)

        # Compute log-likelihood for each particle
        log_weights = np.zeros(self.n_particles)

        for dim in range(self.n_states):
            if np.isnan(observation[dim]):
                continue  # Skip missing observations

            obs_val = observation[dim]
            particle_vals = self.particles[:, dim]
            sigma = np.abs(obs_val) * self.obs_noise_scale + 1e-8

            # Gaussian log-likelihood
            log_weights += -0.5 * ((particle_vals - obs_val) / sigma) ** 2

        # Update weights (log-sum-exp for numerical stability)
        log_weights += np.log(self.weights + 1e-300)
        log_weights -= np.max(log_weights)  # shift for stability
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum() + 1e-300

        return self.get_state_estimate()

    def resample_if_needed(self) -> bool:
        """Systematic resampling when effective sample size drops too low.

        Returns True if resampling was performed.
        """
        ess = self.effective_sample_size()
        if ess < _EFFECTIVE_SAMPLE_THRESHOLD * self.n_particles:
            self._systematic_resample()
            return True
        return False

    def effective_sample_size(self) -> float:
        """Compute ESS = 1 / sum(w_i^2). Higher is better."""
        return 1.0 / (np.sum(self.weights ** 2) + 1e-300)

    def get_state_estimate(self) -> np.ndarray:
        """Return weighted mean of particle cloud."""
        return np.average(self.particles, weights=self.weights, axis=0)

    def get_distribution(self) -> dict[str, np.ndarray]:
        """Return percentile estimates from the particle distribution.

        Returns dict with keys: p5, p25, p50, p75, p95.
        Each value is (n_states,) array.
        """
        result: dict[str, np.ndarray] = {}
        for pct_name, pct_val in [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]:
            vals = np.zeros(self.n_states)
            for dim in range(self.n_states):
                # Weighted percentile using sorted particles
                sorted_idx = np.argsort(self.particles[:, dim])
                sorted_particles = self.particles[sorted_idx, dim]
                sorted_weights = self.weights[sorted_idx]
                cum_weights = np.cumsum(sorted_weights)
                idx = np.searchsorted(cum_weights, pct_val / 100.0)
                idx = min(idx, len(sorted_particles) - 1)
                vals[dim] = sorted_particles[idx]
            result[pct_name] = vals
        return result

    def _systematic_resample(self) -> None:
        """Systematic resampling: low-variance resampling algorithm."""
        n = self.n_particles
        positions = (self._rng.random() + np.arange(n)) / n
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # ensure sum is exactly 1

        indices = np.searchsorted(cumulative_sum, positions)
        indices = np.clip(indices, 0, n - 1)

        self.particles = self.particles[indices].copy()
        self.weights = np.ones(n) / n


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def run_particle_filter(
    cache: "pd.DataFrame",
    variables: list[str] | None = None,
    n_particles: int = _DEFAULT_N_PARTICLES,
) -> ParticleFilterResult:
    """Run particle filter on cache data for specified variables.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with feature columns.
    variables:
        Variables to track. If None, uses Tier 1 liquidity variables.
    n_particles:
        Number of particles.

    Returns
    -------
    ParticleFilterResult
    """
    import pandas as pd

    if variables is None:
        variables = ["cash_ratio", "free_cash_flow_ttm_asof", "current_ratio"]

    # Filter to available variables
    available = [v for v in variables if v in cache.columns]
    if not available:
        return ParticleFilterResult(
            filtered_states=np.array([]),
            particles_final=np.array([]),
            weights_final=np.array([]),
            error="No target variables found in cache",
        )

    n_states = len(available)
    data = cache[available].values  # (T, n_states)
    T = len(data)

    if T < 10:
        return ParticleFilterResult(
            filtered_states=np.array([]),
            particles_final=np.array([]),
            weights_final=np.array([]),
            error=f"Insufficient data: {T} rows (need >= 10)",
        )

    pf = ParticleFilter(n_particles=n_particles, n_states=n_states)

    # Initialize from first valid observation
    first_valid = data[0].copy()
    for dim in range(n_states):
        if np.isnan(first_valid[dim]):
            col_vals = data[:, dim]
            valid = col_vals[~np.isnan(col_vals)]
            first_valid[dim] = valid[0] if len(valid) > 0 else 0.0

    pf.initialize(first_valid, spread=0.1)

    # Run forward pass
    filtered_states = np.zeros((T, n_states))
    ess_history: list[float] = []
    n_resamples = 0

    for t in range(T):
        # Predict
        pf.predict()

        # Update with observation
        obs = data[t]
        pf.update(obs)

        # Record
        filtered_states[t] = pf.get_state_estimate()
        ess_history.append(pf.effective_sample_size())

        # Resample if needed
        if pf.resample_if_needed():
            n_resamples += 1

    # Get final distribution
    distribution = pf.get_distribution()

    logger.info(
        "Particle filter: %d steps, %d states, %d resamples, final ESS=%.0f",
        T, n_states, n_resamples, ess_history[-1] if ess_history else 0,
    )

    return ParticleFilterResult(
        filtered_states=filtered_states,
        particles_final=pf.particles.copy(),
        weights_final=pf.weights.copy(),
        effective_sample_sizes=ess_history,
        n_resamples=n_resamples,
        n_particles=n_particles,
        n_states=n_states,
        fitted=True,
        percentiles=distribution,
    )


def predict_next_step(pf: ParticleFilter, n_steps: int = 1) -> dict[str, np.ndarray]:
    """Generate multi-step ahead predictions from current particle state.

    Returns percentile forecasts for each step ahead.
    """
    predictions: dict[str, list] = {
        "mean": [], "p5": [], "p25": [], "p50": [], "p75": [], "p95": [],
    }

    # Save state for rollback
    saved_particles = pf.particles.copy()
    saved_weights = pf.weights.copy()

    for step in range(n_steps):
        pf.predict()
        predictions["mean"].append(pf.get_state_estimate())
        dist = pf.get_distribution()
        for key in ["p5", "p25", "p50", "p75", "p95"]:
            predictions[key].append(dist[key])

    # Restore state
    pf.particles = saved_particles
    pf.weights = saved_weights

    return {k: np.array(v) for k, v in predictions.items()}

"""Model synergies: cross-model integration for higher prediction accuracy.

Implements all identified synergies between the 25+ mathematical models:

A. Particle Filter + Kalman fusion (regime-switched state estimation)
B. Cycle phase injection (add cyclical features before forward pass)
C. Pattern detector + OHLC predictor coupling (pattern-informed candles)
D. Unified causal network (Granger + Transfer Entropy combined pruning)
E. Copula-correlated Monte Carlo sampling (joint tail simulation)
F. DTW analog-weighted training windows (analog-boosted learning)
G. Peer-relative survival thresholds (dynamic sector-aware triggers)
H. GA optimizer in burn-out loop (evolving ensemble weights per iteration)

Each synergy is implemented as a standalone function that can be called
from the main pipeline at the appropriate point.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synergy A: Particle Filter + Kalman regime-switched fusion
# ---------------------------------------------------------------------------


def fuse_kalman_particle_estimates(
    kalman_estimate: float,
    particle_estimate: float,
    regime_probabilities: dict[str, float] | None = None,
    *,
    kalman_regimes: tuple[str, ...] = ("bull", "low_vol"),
    particle_regimes: tuple[str, ...] = ("bear", "high_vol", "crisis"),
) -> float:
    """Blend Kalman and Particle Filter estimates based on current regime.

    In calm regimes (bull, low-vol), Kalman is optimal (linear-Gaussian).
    In turbulent regimes (bear, high-vol, crisis), the Particle Filter
    handles fat tails and non-linearity better.

    Parameters
    ----------
    kalman_estimate:
        Point estimate from Kalman filter.
    particle_estimate:
        Point estimate from Particle filter.
    regime_probabilities:
        {regime_name: probability} for the current day.  If None,
        defaults to equal weighting.

    Returns
    -------
    Blended estimate.
    """
    if regime_probabilities is None:
        # Equal blend if no regime info
        return 0.5 * kalman_estimate + 0.5 * particle_estimate

    kalman_weight = sum(
        regime_probabilities.get(r, 0) for r in kalman_regimes
    )
    particle_weight = sum(
        regime_probabilities.get(r, 0) for r in particle_regimes
    )

    total = kalman_weight + particle_weight
    if total <= 0:
        return 0.5 * kalman_estimate + 0.5 * particle_estimate

    kalman_weight /= total
    particle_weight /= total

    return kalman_weight * kalman_estimate + particle_weight * particle_estimate


# ---------------------------------------------------------------------------
# Synergy B: Cycle phase injection
# ---------------------------------------------------------------------------


def inject_cycle_phase_features(
    cache: pd.DataFrame,
    cycle_result: Any | None = None,
) -> pd.DataFrame:
    """Add cycle phase columns to the cache before the forward pass.

    For each dominant cycle found by cycle decomposition, computes
    a phase indicator (0 to 2*pi) showing where in the cycle the
    current day falls.  This lets forecasting models learn cyclical
    patterns (e.g., quarterly earnings seasonality).

    Parameters
    ----------
    cache:
        Daily cache DataFrame (modified in-place).
    cycle_result:
        Output from ``run_cycle_decomposition()``.

    Returns
    -------
    The cache with added ``cycle_phase_*`` columns.
    """
    if cycle_result is None:
        return cache

    dominant_cycles = getattr(cycle_result, "dominant_cycles", [])
    if not dominant_cycles:
        return cache

    n = len(cache)
    t = np.arange(n, dtype=float)

    for i, cycle in enumerate(dominant_cycles[:3]):  # top 3 cycles
        if isinstance(cycle, dict):
            period = cycle.get("period_days", cycle.get("period"))
        else:
            period = getattr(cycle, "period_days", getattr(cycle, "period", None))

        if period is None or period <= 0:
            continue

        period = float(period)
        # Phase: where are we in this cycle? (0 = trough, pi = peak)
        phase = (2 * np.pi * t / period) % (2 * np.pi)
        col_name = f"cycle_phase_{int(period)}d"
        cache[col_name] = np.sin(phase)  # sin transform for smooth feature

        # Also add cosine component for full phase representation
        cache[f"cycle_cos_{int(period)}d"] = np.cos(phase)

        logger.info("Injected cycle phase feature: %s (period=%dd)", col_name, int(period))

    return cache


# ---------------------------------------------------------------------------
# Synergy C: Pattern detector -> OHLC predictor coupling
# ---------------------------------------------------------------------------


def compute_pattern_drift_adjustment(
    pattern_result: Any | None = None,
    *,
    lookback_days: int = 5,
) -> float:
    """Compute a drift adjustment for the OHLC predictor from detected patterns.

    If the pattern detector finds bullish patterns, the drift is biased
    upward; bearish patterns bias it downward.

    Returns a multiplier: >1.0 = bullish bias, <1.0 = bearish bias, 1.0 = neutral.
    """
    if pattern_result is None:
        return 1.0

    # Check for recent signals
    recent_bullish = 0
    recent_bearish = 0

    # Try various attribute names the pattern detector might use
    for attr in ("recent_patterns", "signals", "detected_patterns"):
        patterns = getattr(pattern_result, attr, None)
        if patterns and isinstance(patterns, list):
            for p in patterns[-lookback_days:]:
                p_str = str(p).lower() if not isinstance(p, dict) else str(p.get("pattern", "")).lower()
                if any(kw in p_str for kw in ("bullish", "hammer", "morning", "engulfing_up", "piercing")):
                    recent_bullish += 1
                elif any(kw in p_str for kw in ("bearish", "shooting", "evening", "engulfing_down", "dark_cloud")):
                    recent_bearish += 1

    if recent_bullish == 0 and recent_bearish == 0:
        return 1.0

    # Net signal: positive = bullish, negative = bearish
    net = recent_bullish - recent_bearish
    total = recent_bullish + recent_bearish

    # Convert to multiplier: max +/-20% drift adjustment
    adjustment = 1.0 + 0.04 * net  # each net signal = 4% drift
    return max(0.8, min(1.2, adjustment))


# ---------------------------------------------------------------------------
# Synergy D: Unified causal network (Granger + Transfer Entropy)
# ---------------------------------------------------------------------------


def build_unified_causal_network(
    granger_result: Any | None = None,
    transfer_entropy_result: Any | None = None,
    *,
    significance_threshold: float = 0.05,
) -> dict[str, Any]:
    """Combine Granger causality and Transfer Entropy into a unified network.

    Takes the union of significant causal pairs from both methods.
    Granger captures linear causation, Transfer Entropy captures
    non-linear information flow.

    Returns
    -------
    Dict with:
    - ``all_pairs``: list of {source, target, method, strength}
    - ``retained_variables``: union of variables with causal relevance
    - ``pruned_variables``: variables with no causal link in either method
    - ``network_density``: fraction of possible links that are significant
    """
    all_pairs: list[dict[str, Any]] = []
    all_variables: set[str] = set()
    causal_variables: set[str] = set()

    # Granger pairs
    if granger_result is not None and getattr(granger_result, "fitted", False):
        for pair in getattr(granger_result, "significant_pairs", []):
            if isinstance(pair, dict):
                all_pairs.append({
                    "source": pair["source"],
                    "target": pair["target"],
                    "method": "granger",
                    "strength": pair.get("f_statistic", pair.get("correlation", 0)),
                    "p_value": pair.get("p_value", 0),
                })
                causal_variables.add(pair["source"])
                causal_variables.add(pair["target"])

        for v in getattr(granger_result, "retained_variables", []):
            all_variables.add(v)
        for v in getattr(granger_result, "pruned_variables", []):
            all_variables.add(v)

    # Transfer entropy pairs
    if transfer_entropy_result is not None:
        te_pairs = getattr(transfer_entropy_result, "top_pairs", [])
        if not te_pairs:
            te_pairs = getattr(transfer_entropy_result, "significant_pairs", [])

        for pair in te_pairs:
            if isinstance(pair, dict):
                src = pair.get("source", pair.get("from", ""))
                tgt = pair.get("target", pair.get("to", ""))
                if src and tgt:
                    # Check if this pair was already found by Granger
                    existing = any(
                        p["source"] == src and p["target"] == tgt
                        for p in all_pairs
                    )
                    all_pairs.append({
                        "source": src,
                        "target": tgt,
                        "method": "transfer_entropy" if not existing else "both",
                        "strength": pair.get("te_value", pair.get("strength", 0)),
                    })
                    causal_variables.add(src)
                    causal_variables.add(tgt)
                    all_variables.add(src)
                    all_variables.add(tgt)

    # Compute unified pruning
    pruned = all_variables - causal_variables
    n_vars = len(all_variables) if all_variables else 1
    max_possible = n_vars * (n_vars - 1)
    density = len(all_pairs) / max_possible if max_possible > 0 else 0

    result = {
        "all_pairs": sorted(all_pairs, key=lambda x: x.get("strength", 0), reverse=True),
        "retained_variables": sorted(causal_variables),
        "pruned_variables": sorted(pruned),
        "network_density": round(density, 4),
        "n_granger_pairs": sum(1 for p in all_pairs if p["method"] in ("granger", "both")),
        "n_te_pairs": sum(1 for p in all_pairs if p["method"] in ("transfer_entropy", "both")),
        "n_both": sum(1 for p in all_pairs if p["method"] == "both"),
    }

    logger.info(
        "Unified causal network: %d pairs (%d Granger, %d TE, %d both), "
        "%d retained, %d pruned, density=%.3f",
        len(all_pairs), result["n_granger_pairs"], result["n_te_pairs"],
        result["n_both"], len(causal_variables), len(pruned), density,
    )

    return result


def prune_by_unified_network(
    variables: list[str],
    unified_network: dict[str, Any],
    always_keep: list[str] | None = None,
) -> list[str]:
    """Prune variables using the unified causal network."""
    retained = set(unified_network.get("retained_variables", []))
    always_keep_set = set(always_keep or [])

    pruned = [v for v in variables if v in retained or v in always_keep_set]

    n_removed = len(variables) - len(pruned)
    if n_removed > 0:
        logger.info(
            "Unified causal pruning: kept %d of %d (%d removed)",
            len(pruned), len(variables), n_removed,
        )

    return pruned


# ---------------------------------------------------------------------------
# Synergy E: Copula-correlated Monte Carlo sampling
# ---------------------------------------------------------------------------


def generate_copula_correlated_samples(
    n_samples: int,
    n_variables: int,
    copula_result: Any | None = None,
    *,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate correlated random samples using fitted copula structure.

    Instead of independent sampling (which underestimates joint tail
    risk), this uses the copula's dependence structure to produce
    realistic correlated scenarios.

    Parameters
    ----------
    n_samples:
        Number of scenarios to generate.
    n_variables:
        Number of variables to sample.
    copula_result:
        Output from ``run_copula_analysis()``.

    Returns
    -------
    Array of shape (n_samples, n_variables) with correlated [0, 1] samples.
    """
    rng = np.random.default_rng(random_seed)

    if copula_result is None or not getattr(copula_result, "available", False):
        # Fallback: independent uniform samples
        return rng.random((n_samples, n_variables))

    # Extract correlation matrix from copula if available
    corr_matrix = getattr(copula_result, "correlation_matrix", None)
    if corr_matrix is None:
        # Try to build from tail dependence
        tail_dep = getattr(copula_result, "tail_dependence", 0.0)
        if isinstance(tail_dep, (int, float)) and tail_dep > 0:
            # Build a simple equicorrelated matrix from tail dependence
            rho = min(0.95, tail_dep * 1.5)  # approximate correlation from tail dep
            corr_matrix = np.full((n_variables, n_variables), rho)
            np.fill_diagonal(corr_matrix, 1.0)
        else:
            return rng.random((n_samples, n_variables))

    # Ensure matrix is valid
    if isinstance(corr_matrix, np.ndarray):
        # Truncate or pad to match n_variables
        if corr_matrix.shape[0] < n_variables:
            padded = np.eye(n_variables)
            s = corr_matrix.shape[0]
            padded[:s, :s] = corr_matrix
            corr_matrix = padded
        elif corr_matrix.shape[0] > n_variables:
            corr_matrix = corr_matrix[:n_variables, :n_variables]

    try:
        # Cholesky decomposition for correlated normal generation
        L = np.linalg.cholesky(corr_matrix)
        z = rng.standard_normal((n_samples, n_variables))
        correlated_normal = z @ L.T

        # Transform to uniform [0, 1] via CDF
        from scipy.stats import norm
        correlated_uniform = norm.cdf(correlated_normal)
        return correlated_uniform

    except (np.linalg.LinAlgError, ImportError):
        # Matrix not positive definite or scipy missing; fallback
        logger.warning("Copula correlation matrix not PD; using independent samples")
        return rng.random((n_samples, n_variables))


# ---------------------------------------------------------------------------
# Synergy F: DTW analog-weighted training windows
# ---------------------------------------------------------------------------


def compute_dtw_training_boost(
    cache: pd.DataFrame,
    dtw_result: Any | None = None,
    *,
    boost_factor: float = 3.0,
) -> np.ndarray:
    """Compute per-day training weight boosts from DTW analog matches.

    Days that fall within DTW-matched historical analog periods get
    higher training weight, complementing the regime-weighted windows.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    dtw_result:
        Output from ``find_historical_analogs()``.
    boost_factor:
        Weight multiplier for days in analog periods (default 3x).

    Returns
    -------
    Array of per-day boost factors (1.0 = no boost, boost_factor = in analog).
    """
    n = len(cache)
    boosts = np.ones(n, dtype=float)

    if dtw_result is None or not getattr(dtw_result, "fitted", False):
        return boosts

    analogs = getattr(dtw_result, "analogs", [])
    if not analogs:
        return boosts

    for analog in analogs:
        # Extract start/end indices of the matched period
        start_idx = None
        end_idx = None

        if isinstance(analog, dict):
            start_idx = analog.get("start_idx", analog.get("match_start"))
            end_idx = analog.get("end_idx", analog.get("match_end"))
        else:
            start_idx = getattr(analog, "start_idx", getattr(analog, "match_start", None))
            end_idx = getattr(analog, "end_idx", getattr(analog, "match_end", None))

        if start_idx is not None and end_idx is not None:
            start_idx = max(0, int(start_idx))
            end_idx = min(n - 1, int(end_idx))
            boosts[start_idx:end_idx + 1] = boost_factor

    boosted_days = int((boosts > 1.0).sum())
    if boosted_days > 0:
        logger.info(
            "DTW analog boost: %d days get %.1fx training weight",
            boosted_days, boost_factor,
        )

    return boosts


# ---------------------------------------------------------------------------
# Synergy G: Peer-relative survival thresholds
# ---------------------------------------------------------------------------


def compute_peer_adjusted_thresholds(
    cache: pd.DataFrame,
    peer_result: Any | None = None,
    linked_caches: dict[str, pd.DataFrame] | None = None,
) -> dict[str, float]:
    """Compute sector-aware survival thresholds based on peer data.

    Instead of using static thresholds (current_ratio < 1.0, D/E > 3.0),
    adjusts them based on what's normal for the sector.

    If all peers have current_ratio < 1.5, then a ratio of 1.1 isn't
    really distress -- it's sector-normal.  The threshold becomes:
    max(absolute_threshold, peer_25th_percentile).

    Returns
    -------
    Dict of adjusted thresholds: {metric: threshold_value}
    """
    # Default static thresholds (from spec)
    thresholds = {
        "current_ratio": 1.0,
        "debt_to_equity_abs": 3.0,
        "fcf_yield": 0.0,
        "drawdown_252d": -0.40,
    }

    if not linked_caches:
        return thresholds

    # Collect peer values for each threshold metric
    for metric, static_threshold in thresholds.items():
        peer_values = []
        for isin, peer_cache in linked_caches.items():
            if metric in peer_cache.columns:
                latest = peer_cache[metric].dropna()
                if not latest.empty:
                    peer_values.append(float(latest.iloc[-1]))

        if len(peer_values) < 3:
            continue  # Not enough peers

        # 25th percentile of peer distribution
        p25 = float(np.percentile(peer_values, 25))

        # For "higher is safer" metrics (current_ratio, fcf_yield):
        # threshold = max(static, peer_25th_pct * 0.8)
        # For "lower is safer" metrics (debt_to_equity, drawdown):
        # threshold = min(static, peer_75th_pct * 1.2)

        if metric in ("current_ratio", "fcf_yield"):
            # These metrics: lower = worse
            adjusted = max(static_threshold, p25 * 0.8)
            if adjusted != static_threshold:
                logger.info(
                    "Peer-adjusted threshold: %s = %.2f (was %.2f, peer P25=%.2f)",
                    metric, adjusted, static_threshold, p25,
                )
                thresholds[metric] = adjusted
        else:
            # These metrics: higher = worse (debt, drawdown)
            p75 = float(np.percentile(peer_values, 75))
            adjusted = min(static_threshold, p75 * 1.2)
            if adjusted != static_threshold:
                logger.info(
                    "Peer-adjusted threshold: %s = %.2f (was %.2f, peer P75=%.2f)",
                    metric, adjusted, static_threshold, p75,
                )
                thresholds[metric] = adjusted

    return thresholds


# ---------------------------------------------------------------------------
# Synergy H: GA optimizer in burn-out loop
# ---------------------------------------------------------------------------


def run_mini_ga_for_burnout(
    cache: pd.DataFrame,
    forecast_result: Any | None = None,
    previous_weights: dict[str, float] | None = None,
    *,
    n_generations: int = 8,
    population_size: int = 30,
    random_seed: int = 42,
) -> dict[str, float]:
    """Run a lightweight GA within the burn-out loop to evolve ensemble weights.

    This is a "mini-GA" that warm-starts from the previous iteration's
    best weights and evolves for a few generations.  It runs after each
    burn-out iteration to adapt the ensemble blending.

    Parameters
    ----------
    cache:
        Recent portion of daily cache used in burn-out.
    forecast_result:
        Current ForecastResult with model metrics.
    previous_weights:
        Best weights from the previous burn-out iteration (warm start).

    Returns
    -------
    Evolved ensemble weight dict: {model_name: weight}.
    """
    try:
        from operator1.models.genetic_optimizer import run_genetic_optimization
    except ImportError:
        return previous_weights or {}

    result = run_genetic_optimization(
        cache,
        forecast_result=forecast_result,
        population_size=population_size,
        n_generations=n_generations,
        random_seed=random_seed,
    )

    if result.fitted:
        logger.info(
            "Mini-GA burn-out: evolved weights in %d generations (fitness=%.6f)",
            result.n_generations, result.best_fitness,
        )
        return result.best_weights

    return previous_weights or {}


# ---------------------------------------------------------------------------
# Synergy I: Plane-aware model weighting
# ---------------------------------------------------------------------------

# Default model weights (uniform across planes)
_BASE_MODEL_WEIGHTS: dict[str, float] = {
    "forecasting": 1.0,
    "monte_carlo": 1.0,
    "transformer": 1.0,
    "cycle_decomposition": 1.0,
    "pattern_detector": 1.0,
    "copula": 1.0,
    "particle_filter": 1.0,
    "dtw_analogs": 1.0,
    "granger_causality": 1.0,
    "transfer_entropy": 1.0,
}

# Plane-specific weight adjustments.  Values > 1.0 boost the model,
# < 1.0 dampen it.  The intuition:
#   - Supply (commodities): cycles and regime detection matter most
#   - Manufacturing: forecasting and causal networks are key
#   - Consumption: pattern detection and sentiment drive returns
#   - Logistics: Monte Carlo and particle filters (noisy, thin margins)
#   - Finance: copula tail risk and Granger causality dominate
_PLANE_WEIGHT_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "supply": {
        "cycle_decomposition": 1.8,
        "monte_carlo": 1.3,
        "particle_filter": 1.4,
        "copula": 1.2,
        "pattern_detector": 0.7,
        "transformer": 0.8,
    },
    "manufacturing": {
        "forecasting": 1.4,
        "granger_causality": 1.5,
        "transfer_entropy": 1.3,
        "cycle_decomposition": 1.2,
        "dtw_analogs": 1.1,
        "copula": 0.8,
    },
    "consumption": {
        "pattern_detector": 1.5,
        "transformer": 1.4,
        "dtw_analogs": 1.3,
        "forecasting": 1.1,
        "particle_filter": 0.7,
        "granger_causality": 0.8,
    },
    "logistics": {
        "monte_carlo": 1.5,
        "particle_filter": 1.4,
        "cycle_decomposition": 1.2,
        "forecasting": 1.1,
        "pattern_detector": 0.8,
        "transformer": 0.9,
    },
    "finance": {
        "copula": 1.8,
        "granger_causality": 1.5,
        "transfer_entropy": 1.4,
        "monte_carlo": 1.3,
        "cycle_decomposition": 0.7,
        "pattern_detector": 0.8,
        "dtw_analogs": 0.9,
    },
}


def compute_plane_aware_weights(
    economic_plane: str | dict[str, Any] | None = None,
    secondary_planes: list[str] | None = None,
) -> dict[str, float]:
    """Compute model weights adjusted for the company's economic plane.

    A company in the 'supply' plane benefits more from cycle analysis
    (commodity cycles), while a 'finance' company benefits more from
    copula tail-risk modelling.

    Parameters
    ----------
    economic_plane:
        Primary plane name (e.g. "supply", "manufacturing") or a dict
        with ``primary_plane`` key (from ``classify_economic_plane()``).
    secondary_planes:
        Optional list of secondary plane names.  Their adjustments
        are blended at 30% weight.

    Returns
    -------
    Dict of ``{model_name: weight}`` with plane-aware adjustments applied.
    """
    weights = _BASE_MODEL_WEIGHTS.copy()

    # Resolve primary plane
    primary = ""
    if isinstance(economic_plane, dict):
        primary = (economic_plane.get("primary_plane") or "").lower()
        if not secondary_planes:
            secondary_planes = economic_plane.get("secondary_planes", [])
    elif isinstance(economic_plane, str):
        primary = economic_plane.lower()

    if not primary or primary == "unknown":
        return weights

    # Apply primary plane adjustments (full weight)
    primary_adj = _PLANE_WEIGHT_ADJUSTMENTS.get(primary, {})
    for model, adj in primary_adj.items():
        if model in weights:
            weights[model] *= adj

    # Apply secondary plane adjustments (30% blend)
    if secondary_planes:
        for sec_plane in secondary_planes:
            sec_adj = _PLANE_WEIGHT_ADJUSTMENTS.get(sec_plane.lower(), {})
            for model, adj in sec_adj.items():
                if model in weights:
                    # Blend: 30% of the adjustment delta
                    delta = adj - 1.0
                    weights[model] *= (1.0 + 0.3 * delta)

    # Normalize so the mean weight stays at 1.0
    if weights:
        mean_w = sum(weights.values()) / len(weights)
        if mean_w > 0:
            weights = {k: v / mean_w for k, v in weights.items()}

    logger.info(
        "Plane-aware weights: plane=%s, top_boosted=%s, top_dampened=%s",
        primary,
        sorted(weights, key=weights.get, reverse=True)[:3],
        sorted(weights, key=weights.get)[:3],
    )

    return weights


# ---------------------------------------------------------------------------
# Master synergy orchestrator
# ---------------------------------------------------------------------------


def apply_pre_forecasting_synergies(
    cache: pd.DataFrame,
    *,
    cycle_result: Any | None = None,
    granger_result: Any | None = None,
    transfer_entropy_result: Any | None = None,
    peer_result: Any | None = None,
    linked_caches: dict[str, pd.DataFrame] | None = None,
    extra_variables: list[str] | None = None,
    economic_plane: str | dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Apply all pre-forecasting synergies in one call.

    This should be called AFTER regime detection and causality analysis,
    but BEFORE forecasting starts.

    Parameters
    ----------
    economic_plane:
        Primary plane (or dict from ``classify_economic_plane()``) used
        for plane-aware model weighting (Synergy I).

    Returns
    -------
    (cache, pruned_variables, synergy_metadata)
    """
    metadata: dict[str, Any] = {}

    # Synergy B: Inject cycle phase features
    cache = inject_cycle_phase_features(cache, cycle_result)
    cycle_cols = [c for c in cache.columns if c.startswith("cycle_phase_") or c.startswith("cycle_cos_")]
    if cycle_cols:
        metadata["cycle_features_added"] = cycle_cols

    # Synergy D: Unified causal network
    unified = build_unified_causal_network(granger_result, transfer_entropy_result)
    metadata["unified_causal_network"] = unified

    # Note: Unified causal pruning REMOVED -- feature selection is now handled
    # by sub-stage 3.8 (Boruta + PIMP + mRMR) which runs after synergies.
    # The unified causal network is still computed and stored in metadata
    # for informational purposes (profile, report).
    pruned_vars = extra_variables or []
    metadata["variables_after_pruning"] = len(pruned_vars)

    # Synergy G: Peer-relative survival thresholds
    adjusted_thresholds = compute_peer_adjusted_thresholds(
        cache, peer_result, linked_caches,
    )
    metadata["adjusted_survival_thresholds"] = adjusted_thresholds

    # Synergy I: Plane-aware model weighting
    plane_weights = compute_plane_aware_weights(economic_plane)
    metadata["plane_model_weights"] = plane_weights

    return cache, pruned_vars, metadata

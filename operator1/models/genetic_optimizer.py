"""Genetic Algorithm for ensemble weight and hyperparameter meta-optimization.

Uses evolutionary search to find optimal:
1. Ensemble weights across model types (Kalman, GARCH, VAR, LSTM, Tree, Transformer)
2. Per-tier model preferences (e.g., Kalman higher for Tier 1, LSTM for Tier 5)

The GA maintains a population of weight vectors, evaluates fitness based on
prediction accuracy on a validation window, and evolves better configurations.

Spec refs: Sec E.2 Module Category 4 (Genetic Algorithm for Meta-Optimization)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Model names that participate in ensemble
MODEL_NAMES = ["kalman", "garch", "var", "lstm", "tree", "baseline", "transformer"]
N_MODELS = len(MODEL_NAMES)


@dataclass
class GAResult:
    """Container for Genetic Algorithm optimization outputs."""

    # Best ensemble weights found: {model_name: weight}
    best_weights: dict[str, float] = field(default_factory=dict)

    # Per-tier best weights: {tier: {model_name: weight}}
    tier_weights: dict[str, dict[str, float]] = field(default_factory=dict)

    # Fitness history (best fitness per generation)
    fitness_history: list[float] = field(default_factory=list)

    # GA statistics
    n_generations: int = 0
    population_size: int = 0
    best_fitness: float = float("-inf")
    converged: bool = False

    fitted: bool = False
    error: str | None = None


def _random_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random weight vector that sums to 1."""
    w = rng.dirichlet(np.ones(n))
    return w


def _crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Single-point crossover between two weight vectors."""
    n = len(parent1)
    point = rng.integers(1, n)
    child = np.concatenate([parent1[:point], parent2[point:]])
    # Renormalize
    total = child.sum()
    if total > 0:
        child /= total
    else:
        child = np.ones(n) / n
    return child


def _mutate(
    weights: np.ndarray,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mutate a weight vector by adding small perturbations."""
    n = len(weights)
    mask = rng.random(n) < mutation_rate
    noise = rng.normal(0, 0.1, n) * mask
    mutated = weights + noise
    mutated = np.clip(mutated, 0, None)
    total = mutated.sum()
    if total > 0:
        mutated /= total
    else:
        mutated = np.ones(n) / n
    return mutated


def _evaluate_fitness(
    weights: np.ndarray,
    model_predictions: dict[str, np.ndarray],
    actuals: np.ndarray,
    model_names: list[str],
) -> float:
    """Evaluate fitness of a weight vector based on prediction accuracy.

    Fitness = -RMSE of the weighted ensemble prediction vs actuals.
    Higher (less negative) is better.
    """
    n = len(actuals)
    ensemble_pred = np.zeros(n)

    for i, name in enumerate(model_names):
        if name in model_predictions:
            pred = model_predictions[name]
            if len(pred) == n:
                ensemble_pred += weights[i] * pred

    # RMSE
    residuals = actuals - ensemble_pred
    valid = ~np.isnan(residuals)
    if valid.sum() < 5:
        return float("-inf")

    rmse = np.sqrt(np.mean(residuals[valid] ** 2))
    return -rmse  # Negative RMSE as fitness (maximize = minimize RMSE)


def run_genetic_optimization(
    cache: pd.DataFrame,
    forecast_result: Any | None = None,
    *,
    population_size: int = 50,
    n_generations: int = 30,
    mutation_rate: float = 0.15,
    elite_fraction: float = 0.1,
    validation_window: int = 63,  # ~3 months
    random_seed: int = 42,
) -> GAResult:
    """Run Genetic Algorithm to optimize ensemble weights.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    forecast_result:
        Output from ``run_forecasting()`` containing model metrics.
    population_size:
        Number of individuals in each generation.
    n_generations:
        Maximum generations to evolve.
    mutation_rate:
        Probability of mutating each gene.
    elite_fraction:
        Fraction of top individuals carried to next generation unchanged.
    validation_window:
        Number of recent days to use for fitness evaluation.
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    GAResult with optimized ensemble weights.
    """
    result = GAResult()
    result.population_size = population_size
    rng = np.random.default_rng(random_seed)

    if cache is None or cache.empty:
        result.error = "No data for genetic optimization"
        return result

    # Build per-model prediction arrays from available forecast metrics
    # Use return_1d as the target variable for optimization
    target_col = "return_1d"
    if target_col not in cache.columns:
        result.error = "No return_1d column for fitness evaluation"
        return result

    actuals = cache[target_col].values[-validation_window:]
    if np.isnan(actuals).all():
        result.error = "All NaN in validation window"
        return result

    # Build model prediction proxies from cache columns.
    # Prefer real forecast_<model>_<var> columns stored by the forecasting
    # pipeline; fall back to EWM/shifted proxies for models without stored
    # predictions.
    model_predictions: dict[str, np.ndarray] = {}

    for model_name in MODEL_NAMES:
        pred_col = f"forecast_{model_name}_{target_col}"
        if pred_col in cache.columns:
            model_predictions[model_name] = cache[pred_col].values[-validation_window:]
        elif model_name == "baseline":
            # Baseline: last-value carry forward (return = 0 assumption)
            model_predictions["baseline"] = np.zeros(validation_window)
        elif model_name == "kalman" and "return_1d" in cache.columns:
            # Proxy: smoothed returns (EWM)
            ew = cache["return_1d"].ewm(span=10).mean()
            model_predictions["kalman"] = ew.values[-validation_window:]
        elif model_name == "var" and "return_1d" in cache.columns:
            # Proxy: simple AR(1) prediction
            shifted = cache["return_1d"].shift(1)
            model_predictions["var"] = shifted.values[-validation_window:]

    # Seed initial population with inverse-RMSE weights from forecast_result
    # so the GA starts from an informed position rather than pure random.
    _informed_weights: dict[str, float] | None = None
    if forecast_result is not None:
        metrics_list = getattr(forecast_result, "metrics", [])
        if metrics_list:
            model_rmse: dict[str, float] = {}
            for met in metrics_list:
                if met.fitted and np.isfinite(met.rmse) and met.rmse > 0:
                    name = met.model_name
                    if name not in model_rmse or met.rmse < model_rmse[name]:
                        model_rmse[name] = met.rmse
            if model_rmse:
                inv_rmse = {k: 1.0 / v for k, v in model_rmse.items()}
                total = sum(inv_rmse.values())
                _informed_weights = {k: v / total for k, v in inv_rmse.items()}

    if len(model_predictions) < 2:
        # Not enough model predictions, use inverse-RMSE fallback
        result.error = "Insufficient model predictions for GA; need at least 2"
        # Return uniform weights as fallback
        result.best_weights = {name: 1.0 / N_MODELS for name in MODEL_NAMES}
        return result

    active_models = list(model_predictions.keys())
    n_active = len(active_models)

    # Initialize population -- seed half with informed weights (from inverse-RMSE)
    # and half with random Dirichlet to maintain diversity.
    population: list[np.ndarray] = []
    if _informed_weights:
        # Build a weight vector aligned to active_models
        seed_vec = np.array([
            _informed_weights.get(m, 1.0 / n_active) for m in active_models
        ])
        seed_vec = seed_vec / seed_vec.sum()
        # Seed ~half the population with perturbed versions of informed weights
        n_seeded = population_size // 2
        for _ in range(n_seeded):
            perturbed = seed_vec + rng.normal(0, 0.05, n_active)
            perturbed = np.clip(perturbed, 0, None)
            total = perturbed.sum()
            population.append(perturbed / total if total > 0 else seed_vec.copy())
    # Fill the rest with random weights for diversity
    while len(population) < population_size:
        population.append(_random_weights(n_active, rng))
    n_elite = max(1, int(population_size * elite_fraction))

    best_ever_fitness = float("-inf")
    best_ever_weights = population[0].copy()
    no_improve_count = 0

    for gen in range(n_generations):
        # Evaluate fitness
        fitness_scores = [
            _evaluate_fitness(ind, model_predictions, actuals, active_models)
            for ind in population
        ]

        # Sort by fitness (descending)
        ranked = sorted(
            zip(fitness_scores, population),
            key=lambda x: x[0],
            reverse=True,
        )

        gen_best = ranked[0][0]
        result.fitness_history.append(gen_best)

        if gen_best > best_ever_fitness:
            best_ever_fitness = gen_best
            best_ever_weights = ranked[0][1].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= 5:
            result.converged = True
            break

        # Selection: elite + tournament
        new_population: list[np.ndarray] = []

        # Elitism
        for i in range(n_elite):
            new_population.append(ranked[i][1].copy())

        # Tournament selection + crossover
        while len(new_population) < population_size:
            # Tournament selection (k=3)
            candidates = rng.choice(len(population), size=3, replace=False)
            parent1_idx = max(candidates, key=lambda i: fitness_scores[i])

            candidates = rng.choice(len(population), size=3, replace=False)
            parent2_idx = max(candidates, key=lambda i: fitness_scores[i])

            child = _crossover(
                population[parent1_idx],
                population[parent2_idx],
                rng,
            )
            child = _mutate(child, mutation_rate, rng)
            new_population.append(child)

        population = new_population[:population_size]

    result.n_generations = len(result.fitness_history)
    result.best_fitness = best_ever_fitness

    # Map weights back to model names
    for i, name in enumerate(active_models):
        result.best_weights[name] = round(float(best_ever_weights[i]), 6)

    # Fill missing models with 0
    for name in MODEL_NAMES:
        if name not in result.best_weights:
            result.best_weights[name] = 0.0

    # Per-tier optimization (simplified: use same weights but adjust
    # based on tier-model affinity heuristic from spec)
    _tier_affinity = {
        "tier1": {"kalman": 1.5, "var": 1.2},  # Kalman dominates for liquidity
        "tier2": {"kalman": 1.3, "var": 1.3},   # Solvency: Kalman + VAR
        "tier3": {"garch": 2.0, "lstm": 1.2},   # Volatility: GARCH dominates
        "tier4": {"tree": 1.3, "lstm": 1.2},    # Profitability: tree + LSTM
        "tier5": {"lstm": 1.5, "transformer": 1.5, "tree": 1.2},  # Growth: deep learning
    }

    for tier, affinities in _tier_affinity.items():
        tier_w = result.best_weights.copy()
        for model, mult in affinities.items():
            if model in tier_w:
                tier_w[model] *= mult
        # Renormalize
        total = sum(tier_w.values())
        if total > 0:
            tier_w = {k: round(v / total, 6) for k, v in tier_w.items()}
        result.tier_weights[tier] = tier_w

    result.fitted = True
    logger.info(
        "GA optimization complete: %d generations, best_fitness=%.6f, "
        "converged=%s, weights=%s",
        result.n_generations, result.best_fitness,
        result.converged, result.best_weights,
    )

    return result

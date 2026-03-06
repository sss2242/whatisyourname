"""Parallel model executor for Phase F temporal modeling.

Runs independent models concurrently using a thread pool to reduce
pipeline execution time. Models in the parallel group must be read-only
on the cache DataFrame -- they can read from it but must not write
columns back until all parallel tasks complete.

Usage:
    results = run_parallel_models(cache, profile, [
        ("copula", run_copula_analysis, {"variables": vars}),
        ("particle_filter", run_particle_filter, {"variables": pf_vars}),
        ("dtw_analogs", find_historical_analogs, {}),
        ("sobol", run_sensitivity_analysis, {"target_variable": "return_1d"}),
    ])
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Default max workers for parallel model execution.
# Conservative: 4 threads avoids memory pressure from large models.
DEFAULT_MAX_WORKERS: int = 4


def run_parallel_models(
    cache: Any,
    tasks: list[tuple[str, Callable, dict[str, Any]]],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    """Run independent model tasks in parallel using a thread pool.

    Each task is a tuple of (name, callable, kwargs). The callable
    receives ``cache=cache`` plus any extra kwargs.

    Parameters
    ----------
    cache:
        Daily cache DataFrame. Must be treated as READ-ONLY by all
        parallel tasks. Models that need to write columns should
        do so after parallel execution completes.
    tasks:
        List of (name, function, kwargs) tuples. Each function must
        accept ``cache`` as a keyword argument.
    max_workers:
        Maximum number of concurrent threads.

    Returns
    -------
    Dict mapping task name to its result (or None if it failed).
    """
    if not tasks:
        return {}

    results: dict[str, Any] = {}
    start = time.time()

    logger.info(
        "Starting parallel execution of %d models (max_workers=%d)",
        len(tasks), max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for name, fn, kwargs in tasks:
            future = executor.submit(_run_task, name, fn, cache, kwargs)
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                logger.info("Parallel model '%s' completed.", name)
            except Exception as exc:
                logger.warning("Parallel model '%s' failed: %s", name, exc)
                results[name] = None

    elapsed = time.time() - start
    succeeded = sum(1 for v in results.values() if v is not None)
    logger.info(
        "Parallel execution complete: %d/%d succeeded in %.1fs",
        succeeded, len(tasks), elapsed,
    )

    return results


def _run_task(
    name: str,
    fn: Callable,
    cache: Any,
    kwargs: dict[str, Any],
) -> Any:
    """Execute a single model task with timing."""
    t0 = time.time()
    try:
        result = fn(cache=cache, **kwargs)
        elapsed = time.time() - t0
        logger.debug("Model '%s' finished in %.1fs", name, elapsed)
        return result
    except Exception as exc:
        elapsed = time.time() - t0
        logger.warning("Model '%s' failed after %.1fs: %s", name, elapsed, exc)
        raise

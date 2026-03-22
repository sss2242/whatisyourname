"""Granger causality testing and causality-based feature pruning.

Identifies which variables actually have predictive power over other
variables using pairwise Granger causality tests.  The results are
used to prune weak/spurious relationships before training VAR/LSTM
models, reducing overfitting and improving out-of-sample accuracy.

Implements Synergy 3 from the spec: Causality Pruning -> Feature Selection.

Spec refs: Sec E.2 Module Category 3, Synergy 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GrangerResult:
    """Container for Granger causality analysis outputs."""

    # Causality matrix: {(source, target): p_value}
    causality_matrix: dict[tuple[str, str], float] = field(default_factory=dict)

    # Significant causal pairs at given threshold
    significant_pairs: list[dict[str, Any]] = field(default_factory=list)

    # Variables that survived pruning (have at least one causal link)
    retained_variables: list[str] = field(default_factory=list)

    # Variables pruned (no causal link to any target)
    pruned_variables: list[str] = field(default_factory=list)

    # Network density (fraction of possible links that are significant)
    network_density: float = 0.0

    # Number of tests performed
    n_tests: int = 0

    fitted: bool = False
    error: str | None = None


def compute_granger_causality(
    cache: pd.DataFrame,
    variables: list[str] | None = None,
    *,
    max_lag: int = 5,
    significance_level: float = 0.05,
    min_observations: int = 50,
) -> GrangerResult:
    """Run pairwise Granger causality tests across variables.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with numeric columns.
    variables:
        List of column names to test.  If None, selects float columns
        with sufficient data.
    max_lag:
        Maximum lag order for the F-test.
    significance_level:
        P-value threshold for declaring a causal link.
    min_observations:
        Minimum non-NaN observations required per variable.

    Returns
    -------
    GrangerResult with causality matrix, significant pairs, and
    pruning recommendations.
    """
    result = GrangerResult()

    if cache is None or cache.empty:
        result.error = "No data available for Granger causality"
        return result

    # Select variables
    if variables is None:
        variables = [
            c for c in cache.columns
            if cache[c].dtype in ("float64", "float32")
            and cache[c].notna().sum() >= min_observations
        ][:30]  # Cap at 30 to keep runtime reasonable

    if len(variables) < 2:
        result.error = "Need at least 2 variables for Granger causality"
        return result

    # Try PCMCI first (handles autocorrelation and confounders)
    pcmci_result = _try_pcmci(cache, variables, max_lag, significance_level)
    if pcmci_result is not None and pcmci_result.fitted:
        return pcmci_result

    # Fallback to standard Granger F-tests
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        logger.warning("statsmodels not installed; using simplified F-test fallback")
        return _fallback_granger(cache, variables, max_lag, significance_level)

    # Prepare data: drop rows with any NaN in selected variables
    df = cache[variables].dropna()
    if len(df) < min_observations:
        result.error = f"Insufficient clean data ({len(df)} < {min_observations})"
        return result

    # Pairwise Granger tests
    n_vars = len(variables)
    n_tests = 0

    for i, target in enumerate(variables):
        for j, source in enumerate(variables):
            if i == j:
                continue

            try:
                pair_data = df[[target, source]].values
                test_result = grangercausalitytests(
                    pair_data, maxlag=max_lag, verbose=False,
                )

                # Get minimum p-value across all lags
                min_p = min(
                    test_result[lag][0]["ssr_ftest"][1]
                    for lag in range(1, max_lag + 1)
                )

                result.causality_matrix[(source, target)] = min_p
                n_tests += 1

                if min_p < significance_level:
                    # Find best lag
                    best_lag = min(
                        range(1, max_lag + 1),
                        key=lambda l: test_result[l][0]["ssr_ftest"][1],
                    )
                    result.significant_pairs.append({
                        "source": source,
                        "target": target,
                        "p_value": round(min_p, 6),
                        "best_lag": best_lag,
                        "f_statistic": round(
                            test_result[best_lag][0]["ssr_ftest"][0], 4
                        ),
                    })

            except Exception as exc:
                logger.debug(
                    "Granger test failed for %s -> %s: %s",
                    source, target, exc,
                )
                continue

    result.n_tests = n_tests

    # Determine retained vs pruned variables
    causal_vars = set()
    for pair in result.significant_pairs:
        causal_vars.add(pair["source"])
        causal_vars.add(pair["target"])

    result.retained_variables = sorted(causal_vars)
    result.pruned_variables = sorted(set(variables) - causal_vars)

    # Network density
    max_possible = n_vars * (n_vars - 1)
    if max_possible > 0:
        result.network_density = len(result.significant_pairs) / max_possible

    result.fitted = True
    logger.info(
        "Granger causality: %d tests, %d significant pairs (density=%.3f), "
        "%d variables retained, %d pruned",
        n_tests, len(result.significant_pairs), result.network_density,
        len(result.retained_variables), len(result.pruned_variables),
    )

    return result


def _fallback_granger(
    cache: pd.DataFrame,
    variables: list[str],
    max_lag: int,
    significance_level: float,
) -> GrangerResult:
    """Simplified correlation-based fallback when statsmodels is unavailable."""
    result = GrangerResult()
    df = cache[variables].dropna()

    if len(df) < 30:
        result.error = "Insufficient data for fallback causality test"
        return result

    # Use lagged cross-correlation as a proxy for Granger causality
    n_tests = 0
    for target in variables:
        for source in variables:
            if target == source:
                continue

            max_corr = 0.0
            best_lag = 1
            for lag in range(1, max_lag + 1):
                shifted = df[source].shift(lag)
                valid = pd.concat([df[target], shifted], axis=1).dropna()
                if len(valid) < 20:
                    continue
                corr = abs(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
                if corr > max_corr:
                    max_corr = corr
                    best_lag = lag

            # Convert correlation to pseudo p-value
            n = len(df)
            t_stat = max_corr * np.sqrt(n - 2) / np.sqrt(1 - max_corr ** 2 + 1e-10)
            pseudo_p = 2 * (1 - min(0.9999, abs(t_stat) / (abs(t_stat) + 1)))

            result.causality_matrix[(source, target)] = pseudo_p
            n_tests += 1

            if pseudo_p < significance_level and max_corr > 0.1:
                result.significant_pairs.append({
                    "source": source,
                    "target": target,
                    "p_value": round(pseudo_p, 6),
                    "best_lag": best_lag,
                    "correlation": round(max_corr, 4),
                })

    result.n_tests = n_tests

    causal_vars = set()
    for pair in result.significant_pairs:
        causal_vars.add(pair["source"])
        causal_vars.add(pair["target"])

    result.retained_variables = sorted(causal_vars)
    result.pruned_variables = sorted(set(variables) - causal_vars)

    max_possible = len(variables) * (len(variables) - 1)
    if max_possible > 0:
        result.network_density = len(result.significant_pairs) / max_possible

    result.fitted = True
    return result


def _try_pcmci(
    cache: pd.DataFrame,
    variables: list[str],
    max_lag: int = 5,
    significance_level: float = 0.05,
) -> GrangerResult | None:
    """Run PCMCI causal discovery (handles autocorrelation, confounders).

    PCMCI (Runge et al. 2019) is specifically designed for time series
    causal discovery. Unlike Granger (which inflates significance when
    variables share autocorrelation), PCMCI conditions on the past of
    ALL variables simultaneously, producing a true causal graph.

    Returns GrangerResult or None if tigramite is not available.
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError:
        return None

    df = cache[variables].dropna()
    if len(df) < 50 or len(variables) < 2:
        return None

    try:
        data = df.values
        var_names = list(variables)
        dataframe = pp.DataFrame(data, var_names=var_names)

        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
        results = pcmci.run_pcmci(tau_max=max_lag, alpha_level=significance_level)

        # Extract significant links
        p_matrix = results["p_matrix"]
        val_matrix = results["val_matrix"]

        result = GrangerResult()
        n_vars = len(var_names)

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                for lag in range(1, max_lag + 1):
                    p_val = float(p_matrix[i, j, lag])
                    if p_val < significance_level:
                        pair_key = (var_names[i], var_names[j])
                        # Keep the best (lowest p-value) lag
                        existing_p = result.causality_matrix.get(pair_key, 1.0)
                        if p_val < existing_p:
                            result.causality_matrix[pair_key] = p_val
                            result.significant_pairs.append({
                                "source": var_names[i],
                                "target": var_names[j],
                                "p_value": round(p_val, 6),
                                "best_lag": lag,
                                "val_statistic": round(float(val_matrix[i, j, lag]), 4),
                                "method": "pcmci",
                            })

        # Deduplicate significant pairs (keep lowest p-value per pair)
        seen = {}
        for pair in result.significant_pairs:
            key = (pair["source"], pair["target"])
            if key not in seen or pair["p_value"] < seen[key]["p_value"]:
                seen[key] = pair
        result.significant_pairs = list(seen.values())

        causal_vars = set()
        for pair in result.significant_pairs:
            causal_vars.add(pair["source"])
            causal_vars.add(pair["target"])

        result.retained_variables = sorted(causal_vars)
        result.pruned_variables = sorted(set(variables) - causal_vars)
        result.n_tests = n_vars * (n_vars - 1) * max_lag

        max_possible = n_vars * (n_vars - 1)
        if max_possible > 0:
            result.network_density = len(result.significant_pairs) / max_possible

        result.fitted = True
        logger.info(
            "PCMCI causality: %d significant pairs (density=%.3f), %d retained, %d pruned",
            len(result.significant_pairs), result.network_density,
            len(result.retained_variables), len(result.pruned_variables),
        )
        return result

    except Exception as exc:
        logger.debug("PCMCI failed: %s", exc)
        return None


def compute_time_varying_granger(
    cache: pd.DataFrame,
    variables: list[str] | None = None,
    *,
    window_size: int = 126,
    step_size: int = 21,
    max_lag: int = 5,
    significance_level: float = 0.05,
    min_observations: int = 50,
) -> dict[str, Any]:
    """Run rolling-window Granger causality to detect time-varying causal links.

    Instead of a single static causal graph, this produces a temporal causal
    graph showing when relationships emerge and disappear. Feature pruning
    can then use the DAG for the CURRENT regime, not a static aggregate.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    variables:
        Columns to test (auto-selected if None).
    window_size:
        Rolling window size in days (default ~6 months).
    step_size:
        How many days to slide between windows (default ~1 month).
    max_lag:
        Maximum lag order for Granger F-tests.
    significance_level:
        P-value threshold.
    min_observations:
        Minimum non-NaN observations per window.

    Returns
    -------
    Dict with:
        - ``windows``: list of {start_date, end_date, significant_pairs, density}
        - ``emerging_pairs``: pairs that became significant in recent windows
        - ``disappearing_pairs``: pairs that were significant but are no longer
        - ``current_window_result``: GrangerResult for the most recent window
    """
    if cache is None or cache.empty:
        return {"error": "No data", "windows": []}

    if variables is None:
        variables = [
            c for c in cache.columns
            if cache[c].dtype in ("float64", "float32")
            and cache[c].notna().sum() >= min_observations
        ][:20]

    if len(variables) < 2:
        return {"error": "Need >= 2 variables", "windows": []}

    n_rows = len(cache)
    if n_rows < window_size + step_size:
        return {"error": "Insufficient data for rolling windows", "windows": []}

    windows_results: list[dict[str, Any]] = []
    prev_pairs: set[tuple[str, str]] = set()

    # Slide windows
    for start in range(0, n_rows - window_size + 1, step_size):
        end = start + window_size
        window_cache = cache.iloc[start:end]

        # Run Granger on this window
        window_result = compute_granger_causality(
            window_cache,
            variables=variables,
            max_lag=max_lag,
            significance_level=significance_level,
            min_observations=min(min_observations, window_size // 2),
        )

        current_pairs = {
            (p["source"], p["target"])
            for p in window_result.significant_pairs
        }

        start_date = str(cache.index[start])[:10] if hasattr(cache.index, "strftime") else str(start)
        end_date = str(cache.index[end - 1])[:10] if hasattr(cache.index, "strftime") else str(end)

        windows_results.append({
            "start_date": start_date,
            "end_date": end_date,
            "n_significant": len(window_result.significant_pairs),
            "density": window_result.network_density,
            "pairs": list(current_pairs),
        })

        prev_pairs = current_pairs

    # Identify emerging and disappearing pairs from the last two windows
    emerging: list[tuple[str, str]] = []
    disappearing: list[tuple[str, str]] = []
    if len(windows_results) >= 2:
        prev_set = set(tuple(p) for p in windows_results[-2].get("pairs", []))
        curr_set = set(tuple(p) for p in windows_results[-1].get("pairs", []))
        emerging = [p for p in curr_set - prev_set]
        disappearing = [p for p in prev_set - curr_set]

    # Current window result (most recent)
    current_result = compute_granger_causality(
        cache.iloc[-window_size:],
        variables=variables,
        max_lag=max_lag,
        significance_level=significance_level,
        min_observations=min(min_observations, window_size // 2),
    )

    logger.info(
        "Time-varying Granger: %d windows, %d emerging pairs, %d disappearing",
        len(windows_results), len(emerging), len(disappearing),
    )

    return {
        "windows": windows_results,
        "emerging_pairs": [{"source": p[0], "target": p[1]} for p in emerging],
        "disappearing_pairs": [{"source": p[0], "target": p[1]} for p in disappearing],
        "current_window_result": current_result,
        "n_windows": len(windows_results),
    }


def prune_features_by_causality(
    variables: list[str],
    granger_result: GrangerResult,
    *,
    always_keep: list[str] | None = None,
) -> list[str]:
    """Prune variables that have no causal link to any other variable.

    Parameters
    ----------
    variables:
        Full list of candidate variables.
    granger_result:
        Output from ``compute_granger_causality()``.
    always_keep:
        Variables that should never be pruned (e.g., close, return_1d).

    Returns
    -------
    Pruned list of variables with causal relevance.
    """
    if not granger_result.fitted or not granger_result.retained_variables:
        return variables  # No pruning if causality analysis failed

    always_keep = set(always_keep or [])
    retained = set(granger_result.retained_variables)

    pruned = [
        v for v in variables
        if v in retained or v in always_keep
    ]

    n_removed = len(variables) - len(pruned)
    if n_removed > 0:
        logger.info(
            "Causality pruning: kept %d of %d variables (%d removed)",
            len(pruned), len(variables), n_removed,
        )

    return pruned

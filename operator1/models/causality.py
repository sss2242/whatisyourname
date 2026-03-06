"""T6.1 (cont.) -- Causality analysis and variable pruning.

Provides Granger-causality testing and relationship pruning that
feed into the forecasting models (T6.2).

**Granger Causality:**
  Tests whether the past values of variable X improve the prediction
  of variable Y beyond what Y's own past provides.  The result is a
  pairwise causality matrix indicating statistically significant
  (p < 0.05) causal links.

**Variable Pruning:**
  Removes weakly connected variables from the modelling set to reduce
  overfitting and speed up downstream VAR/LSTM fitting.  The pruning
  threshold is configurable.

**Tier-Aware Pruning (Sec 10.4):**
  Variables belonging to Tier 1-2 (liquidity, solvency) are never
  pruned regardless of their Granger score, because survival mode
  logic depends on them unconditionally.

Output:
  - ``causality_matrix``: DataFrame of shape (n_vars, n_vars) where
    entry (y, x) = 1 if x Granger-causes y at p < 0.05.
  - ``pruned_variables``: list of variable names retained after pruning.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from operator1.config_loader import load_config

logger = logging.getLogger(__name__)

# Default Granger test settings.
DEFAULT_MAX_LAG: int = 5
DEFAULT_SIGNIFICANCE: float = 0.05

# Default pruning threshold: minimum incoming causal links to keep a var.
DEFAULT_PRUNE_THRESHOLD: float = 1.0

# Maximum variables for VAR to remain numerically stable.
DEFAULT_MAX_VARS_FOR_VAR: int = 20


# ---------------------------------------------------------------------------
# Granger causality matrix
# ---------------------------------------------------------------------------


def compute_granger_causality(
    cache: pd.DataFrame,
    variables: list[str],
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> pd.DataFrame:
    """Compute pairwise Granger-causality matrix.

    .. deprecated::
        This is a backward-compatible wrapper. The canonical implementation
        lives in :mod:`operator1.models.granger_causality` and returns a
        ``GrangerResult`` dataclass.  This wrapper converts the result to a
        DataFrame matrix for backward compatibility with existing callers.

    Parameters
    ----------
    cache:
        Daily cache DataFrame containing all ``variables`` as columns.
    variables:
        List of variable names to test.
    max_lag:
        Maximum lag order for the Granger test.
    significance:
        p-value threshold for declaring significance.

    Returns
    -------
    causality_matrix:
        DataFrame of shape ``(len(variables), len(variables))`` where
        entry ``[y, x]`` is 1 if ``x`` Granger-causes ``y`` at the
        given significance level, else 0.
    """
    from operator1.models.granger_causality import (
        compute_granger_causality as _canonical_granger,
    )

    # Filter to columns actually present in the cache (backward compat --
    # the old implementation silently dropped missing columns).
    available = [v for v in variables if v in cache.columns]
    missing = set(variables) - set(available)
    if missing:
        logger.info(
            "Granger: %d variables not in cache, skipping: %s",
            len(missing),
            sorted(missing)[:5],
        )

    # Delegate to the canonical implementation
    result = _canonical_granger(
        cache,
        variables=available,
        max_lag=max_lag,
        significance_level=significance,
    )
    matrix = pd.DataFrame(
        np.zeros((len(available), len(available))),
        index=available,
        columns=available,
    )

    if result.fitted and result.significant_pairs:
        for pair in result.significant_pairs:
            source = pair.get("source", "")
            target = pair.get("target", "")
            if source in matrix.columns and target in matrix.index:
                matrix.loc[target, source] = 1

    return matrix


# ---------------------------------------------------------------------------
# Transfer Entropy / Information Flow (C5)
# ---------------------------------------------------------------------------


def compute_transfer_entropy(
    cache: pd.DataFrame,
    variables: list[str],
    *,
    lag: int = 1,
    n_bins: int = 8,
) -> pd.DataFrame:
    """Compute pairwise transfer entropy between variables.

    Transfer entropy measures the directional information flow from
    variable X to variable Y: how much knowing past X reduces
    uncertainty about future Y beyond what past Y already tells us.

    Spec reference: The_Apps_core_idea.pdf Section E.2 Category 3
    (Module 9).

    Uses a binning-based estimator for speed and robustness.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    variables:
        List of column names to analyse.
    lag:
        Time lag for conditional entropy computation.
    n_bins:
        Number of histogram bins for discretisation.

    Returns
    -------
    pd.DataFrame
        Square matrix where entry (i, j) = TE from variable j -> i.
        Higher values mean variable j provides more information about
        variable i's future.
    """
    available = [v for v in variables if v in cache.columns]
    data = cache[available].dropna()

    if len(data) < 30 or len(available) < 2:
        logger.warning("Insufficient data for transfer entropy (%d rows, %d vars)", len(data), len(available))
        return pd.DataFrame(0.0, index=available, columns=available)

    n = len(data)
    te_matrix = pd.DataFrame(0.0, index=available, columns=available)

    for target_var in available:
        y_future = data[target_var].values[lag:]
        y_past = data[target_var].values[:-lag]

        for source_var in available:
            if source_var == target_var:
                continue

            x_past = data[source_var].values[:-lag]

            try:
                te = _binned_transfer_entropy(y_future, y_past, x_past, n_bins)
                te_matrix.loc[target_var, source_var] = max(0.0, te)
            except Exception:
                te_matrix.loc[target_var, source_var] = 0.0

    logger.info(
        "Transfer entropy computed: %d variables, mean TE = %.4f",
        len(available), te_matrix.values.mean(),
    )

    return te_matrix


def _binned_transfer_entropy(
    y_future: np.ndarray,
    y_past: np.ndarray,
    x_past: np.ndarray,
    n_bins: int,
) -> float:
    """Compute TE(X -> Y) using histogram binning.

    TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    where H is conditional entropy computed via joint histograms.
    """
    # Digitise each variable into bins
    yf_bins = _digitise(y_future, n_bins)
    yp_bins = _digitise(y_past, n_bins)
    xp_bins = _digitise(x_past, n_bins)

    # H(Y_future | Y_past)
    h_yf_given_yp = _conditional_entropy(yf_bins, yp_bins, n_bins)

    # H(Y_future | Y_past, X_past) -- condition on joint (yp, xp)
    joint_cond = yp_bins * n_bins + xp_bins  # unique joint index
    h_yf_given_yp_xp = _conditional_entropy(yf_bins, joint_cond, n_bins * n_bins)

    return h_yf_given_yp - h_yf_given_yp_xp


def _digitise(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Digitise values into n_bins equal-width bins."""
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        return np.zeros(len(x), dtype=int)
    edges = np.linspace(xmin, xmax, n_bins + 1)
    return np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)


def _conditional_entropy(
    target: np.ndarray,
    condition: np.ndarray,
    n_cond_states: int,
) -> float:
    """Compute H(target | condition) using frequency counting."""
    n = len(target)
    if n == 0:
        return 0.0

    # Joint counts
    cond_unique = np.unique(condition)
    h_cond = 0.0

    for c in cond_unique:
        mask = condition == c
        p_c = mask.sum() / n
        if p_c == 0:
            continue

        # Conditional distribution of target given condition = c
        t_vals = target[mask]
        _, counts = np.unique(t_vals, return_counts=True)
        probs = counts / counts.sum()
        h_given_c = -np.sum(probs * np.log2(probs + 1e-12))
        h_cond += p_c * h_given_c

    return h_cond


# ---------------------------------------------------------------------------
# Variable pruning
# ---------------------------------------------------------------------------


def _get_protected_variables() -> set[str]:
    """Return variable names that must never be pruned.

    Tier 1 and Tier 2 variables from the survival hierarchy config are
    always retained because survival mode depends on them.
    """
    try:
        cfg = load_config("survival_hierarchy")
    except FileNotFoundError:
        logger.warning("survival_hierarchy config not found -- no protected vars")
        return set()

    protected: set[str] = set()
    tiers = cfg.get("tiers", {})

    for tier_key in ("tier1", "tier2"):
        tier_data = tiers.get(tier_key, {})
        tier_vars = tier_data.get("variables", [])
        protected.update(tier_vars)

    return protected


def prune_weak_relationships(
    causality_matrix: pd.DataFrame,
    *,
    threshold: float = DEFAULT_PRUNE_THRESHOLD,
    max_vars: int = DEFAULT_MAX_VARS_FOR_VAR,
    protect_tiers_1_2: bool = True,
) -> list[str]:
    """Remove weakly-connected variables from the modelling set.

    Parameters
    ----------
    causality_matrix:
        Square DataFrame from ``compute_granger_causality``.
    threshold:
        Minimum number of incoming causal links to retain a variable.
    max_vars:
        Hard cap on variables retained (for VAR numerical stability).
    protect_tiers_1_2:
        If *True*, variables in Tier 1-2 of the survival hierarchy are
        never pruned.

    Returns
    -------
    List of variable names to keep, ordered by causal strength
    (descending).
    """
    if causality_matrix.empty:
        return []

    # Count incoming causal links per variable.
    incoming = causality_matrix.sum(axis=1)

    # Protected variables (never pruned).
    protected = _get_protected_variables() if protect_tiers_1_2 else set()

    # Select variables meeting the threshold.
    strong_vars = set(incoming[incoming >= threshold].index)

    # Always include protected variables.
    all_vars = strong_vars | (protected & set(causality_matrix.index))

    # Sort by causal strength descending.
    sorted_vars = (
        incoming.loc[list(all_vars)]
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Apply hard cap.
    if len(sorted_vars) > max_vars:
        # Keep protected variables and fill remaining slots by strength.
        protected_in_list = [v for v in sorted_vars if v in protected]
        non_protected = [v for v in sorted_vars if v not in protected]
        remaining_slots = max(0, max_vars - len(protected_in_list))
        sorted_vars = protected_in_list + non_protected[:remaining_slots]

    n_pruned = len(causality_matrix.index) - len(sorted_vars)
    logger.info(
        "Variable pruning: kept %d / %d variables (%d pruned, "
        "%d protected, threshold=%.1f, cap=%d)",
        len(sorted_vars),
        len(causality_matrix.index),
        n_pruned,
        len(protected & set(sorted_vars)),
        threshold,
        max_vars,
    )

    return sorted_vars


# ---------------------------------------------------------------------------
# Convenience: combined pipeline step
# ---------------------------------------------------------------------------


def run_causality_analysis(
    cache: pd.DataFrame,
    variables: list[str],
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    significance: float = DEFAULT_SIGNIFICANCE,
    prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
    max_vars: int = DEFAULT_MAX_VARS_FOR_VAR,
) -> tuple[pd.DataFrame, list[str]]:
    """Run Granger causality and variable pruning in one call.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    variables:
        Candidate variable names to analyse.
    max_lag:
        Maximum lag for Granger test.
    significance:
        p-value threshold.
    prune_threshold:
        Minimum incoming causal links to keep.
    max_vars:
        Hard cap on retained variables.

    Returns
    -------
    (causality_matrix, strong_variables)
    """
    logger.info("Running causality analysis pipeline...")

    matrix = compute_granger_causality(
        cache,
        variables,
        max_lag=max_lag,
        significance=significance,
    )

    strong_vars = prune_weak_relationships(
        matrix,
        threshold=prune_threshold,
        max_vars=max_vars,
    )

    return matrix, strong_vars

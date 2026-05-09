"""Hierarchical temporal forecast reconciliation (MinTrace method).

Ensures coherence across frequency forecasts: the annual forecast must
equal the sum of quarterly forecasts, which must equal the sum of monthly
forecasts, etc.  Without reconciliation, each frequency pipeline produces
independent forecasts that may contradict each other.

Based on:
- Wickramasuriya, Athanasopoulos & Hyndman (2019) "Optimal forecast
  reconciliation through trace minimization"
- Nixtla/hierarchicalforecast MinTrace implementation (MIT license)
- sktime ReconcilerForecaster pattern

Usage::

    from operator1.models.hierarchical_reconciliation import (
        reconcile_temporal_forecasts,
    )

    reconciled = reconcile_temporal_forecasts(
        forecasts={"A": {...}, "Q": {...}, "M": {...}},
        residuals={"A": [...], "Q": [...], "M": [...]},
    )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal summing matrix builder
# ---------------------------------------------------------------------------

def _build_temporal_summing_matrix(
    frequencies: list[str],
) -> np.ndarray:
    """Build the summing matrix S for temporal hierarchy.

    For frequencies [A, Q, M], the hierarchy is:
        A = sum of 4 Q = sum of 12 M

    The summing matrix maps bottom-level (highest freq) forecasts
    to all levels.

    Parameters
    ----------
    frequencies:
        Ordered list of frequencies from slowest to fastest.
        E.g., ["A", "Q", "M"] or ["A", "Q"].

    Returns
    -------
    S matrix (n_total x n_bottom) where n_bottom is the number of
    bottom-level periods and n_total includes all aggregation levels.
    """
    # Periods per year for each frequency
    from operator1.freq_constants import PERIODS_PER_YEAR

    periods = [PERIODS_PER_YEAR.get(f, 1) for f in frequencies]

    # Bottom level is the highest frequency (most periods per year)
    n_bottom = max(periods)
    bottom_freq = frequencies[periods.index(n_bottom)]

    # Build S: each row maps a bottom-level forecast to an aggregate
    rows = []

    for freq in frequencies:
        n_periods = PERIODS_PER_YEAR.get(freq, 1)
        if n_periods == n_bottom:
            # Bottom level: identity rows
            for i in range(n_bottom):
                row = np.zeros(n_bottom)
                row[i] = 1.0
                rows.append(row)
        else:
            # Aggregate level: sum groups of bottom periods
            group_size = n_bottom // n_periods
            for g in range(n_periods):
                row = np.zeros(n_bottom)
                start = g * group_size
                end = start + group_size
                row[start:end] = 1.0
                rows.append(row)

    return np.array(rows)


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def _estimate_covariance(
    residuals: dict[str, list[float]],
    method: str = "mint_shrink",
) -> np.ndarray:
    """Estimate the error covariance matrix W for reconciliation.

    Parameters
    ----------
    residuals:
        Per-frequency list of in-sample forecast residuals.
    method:
        One of "ols" (identity), "wls_var" (diagonal variance),
        "mint_shrink" (Schaefer-Strimmer shrinkage).

    Returns
    -------
    Positive-definite covariance matrix W.
    """
    # Flatten residuals into a matrix (n_obs x n_series)
    all_residuals = []
    for freq, res_list in sorted(residuals.items()):
        if res_list:
            all_residuals.append(np.array(res_list))

    if not all_residuals:
        # No residuals: fall back to identity (OLS)
        n = sum(len(r) for r in residuals.values()) or 1
        return np.eye(n)

    if method == "ols":
        n = sum(len(r) for r in all_residuals)
        return np.eye(n)

    if method == "wls_var":
        # Diagonal: variance of each series' residuals
        variances = []
        for r in all_residuals:
            v = np.var(r) if len(r) > 1 else 1.0
            variances.append(max(v, 1e-10))
        return np.diag(variances)

    # mint_shrink: Schaefer-Strimmer shrinkage
    # Shrinks sample covariance toward diagonal target
    try:
        # Stack residuals into matrix
        max_len = max(len(r) for r in all_residuals)
        n_series = len(all_residuals)

        # Pad shorter series with zeros
        R = np.zeros((max_len, n_series))
        for i, r in enumerate(all_residuals):
            R[:len(r), i] = r

        # Sample covariance
        S = np.cov(R.T)
        if S.ndim == 0:
            S = np.array([[float(S)]])

        # Target: diagonal of S
        T = np.diag(np.diag(S))

        # Shrinkage intensity (Ledoit-Wolf)
        n = R.shape[0]
        # Frobenius norm of off-diagonal
        off_diag = S - T
        num = np.sum(off_diag ** 2) / n
        denom = np.sum((S - T) ** 2)

        if denom > 0:
            alpha = max(0.0, min(1.0, num / denom))
        else:
            alpha = 0.5

        W = (1 - alpha) * S + alpha * T

        # Ensure positive definite with ridge
        W += np.eye(W.shape[0]) * 2e-8

        logger.debug(
            "MinTrace covariance: %dx%d, shrinkage=%.4f",
            W.shape[0], W.shape[1], alpha,
        )
        return W

    except Exception as exc:
        logger.warning("Covariance estimation failed: %s -- falling back to OLS", exc)
        n = sum(len(r) for r in all_residuals)
        return np.eye(max(n, 1))


# ---------------------------------------------------------------------------
# Main reconciliation function
# ---------------------------------------------------------------------------

def reconcile_temporal_forecasts(
    forecasts: dict[str, dict[str, Any]],
    residuals: dict[str, list[float]] | None = None,
    method: str = "mint_shrink",
) -> dict[str, dict[str, Any]]:
    """Reconcile forecasts across temporal frequencies.

    Ensures additive coherence: annual forecast = sum of quarterly
    forecasts = sum of monthly forecasts.

    Uses the MinTrace method (Wickramasuriya et al. 2019):
        P = (S'WS)^{-1} S'W^{-1}
        reconciled = S @ P @ base_forecasts

    Parameters
    ----------
    forecasts:
        Per-frequency forecast dicts.  Keys are frequency labels
        ("A", "Q", "M", etc.), values are dicts of
        {variable: {horizon: value}}.
    residuals:
        Per-frequency in-sample residual lists for covariance estimation.
        If None, uses OLS (identity covariance).
    method:
        Covariance method: "ols", "wls_var", or "mint_shrink".

    Returns
    -------
    Reconciled forecasts in the same format as input.
    """
    if len(forecasts) < 2:
        return forecasts  # nothing to reconcile with 1 frequency

    frequencies = sorted(forecasts.keys())
    logger.info(
        "Reconciling forecasts across %d frequencies: %s (method=%s)",
        len(frequencies), frequencies, method,
    )

    # Extract variables that appear in multiple frequencies
    all_vars: set[str] = set()
    for freq_forecasts in forecasts.values():
        all_vars.update(freq_forecasts.keys())

    reconciled = {f: dict(forecasts[f]) for f in frequencies}
    n_reconciled = 0

    for var in all_vars:
        # Collect base forecasts for this variable across frequencies
        base_values = {}
        for freq in frequencies:
            val = forecasts[freq].get(var)
            if val is not None:
                if isinstance(val, dict):
                    # Use first horizon value
                    first_h = next(iter(val.values()), None)
                    if first_h is not None:
                        base_values[freq] = float(first_h)
                elif isinstance(val, (int, float)):
                    base_values[freq] = float(val)

        if len(base_values) < 2:
            continue

        try:
            freq_list = sorted(base_values.keys())
            base_vector = np.array([base_values[f] for f in freq_list])

            # Build summing matrix for these frequencies
            S = _build_temporal_summing_matrix(freq_list)

            # Estimate covariance
            var_residuals = {}
            if residuals:
                for f in freq_list:
                    if f in residuals:
                        var_residuals[f] = residuals[f]

            W = _estimate_covariance(var_residuals or {}, method=method)

            # Reconciliation: P = (S'WS)^{-1} S'W^{-1}
            # Use solve instead of inverse for numerical stability
            if W.shape[0] == S.shape[0] and S.shape[0] == len(base_vector):
                StW = S.T @ np.linalg.solve(W, np.eye(W.shape[0]))
                P = np.linalg.solve(StW @ S, StW)
                recon_vector = S @ P @ base_vector

                # Update reconciled forecasts
                for i, freq in enumerate(freq_list):
                    if i < len(recon_vector):
                        old_val = base_values[freq]
                        new_val = float(recon_vector[i])
                        # Apply reconciliation as adjustment to all horizons
                        if abs(old_val) > 1e-10:
                            ratio = new_val / old_val
                        else:
                            ratio = 1.0

                        orig = reconciled[freq].get(var)
                        if isinstance(orig, dict):
                            reconciled[freq][var] = {
                                h: v * ratio for h, v in orig.items()
                            }
                        else:
                            reconciled[freq][var] = new_val

                n_reconciled += 1

        except Exception as exc:
            logger.debug("Reconciliation failed for %s: %s", var, exc)
            continue

    logger.info(
        "Reconciliation complete: %d/%d variables reconciled across %d frequencies",
        n_reconciled, len(all_vars), len(frequencies),
    )
    return reconciled

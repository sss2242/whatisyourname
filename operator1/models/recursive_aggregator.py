"""Recursive day-by-day prediction aggregation.

Instead of predicting each horizon (1d, 5d, 21d, 252d) independently,
this module focuses on maximizing 1d accuracy and then chains predictions
recursively: predict day 1, append it to the cache, re-derive features,
predict day 2, and so on up to the target horizon.

This mirrors how markets actually evolve (each day depends on the previous)
and avoids the compounding extrapolation error of direct multi-step-ahead
forecasts.

Top-level entry point:
    ``run_recursive_predictions(cache, model_states, ...) -> RecursivePredictionResult``

Spec refs: Phase F enhancement, recursive aggregation architecture.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default snapshot days (horizons extracted from the recursive chain)
# ---------------------------------------------------------------------------
DEFAULT_SNAPSHOT_DAYS: dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "21d": 21,
    "63d": 63,
    "252d": 252,
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RecursiveSnapshot:
    """A single horizon snapshot extracted from the recursive chain."""

    day: int
    """Recursive step number (1-based)."""

    date: str
    """Predicted date (ISO format string)."""

    predictions: dict[str, float]
    """Per-variable point forecasts at this step."""

    uncertainty: dict[str, tuple[float, float]]
    """Per-variable (lower, upper) confidence bounds."""

    regime: str
    """Predicted regime label at this step."""

    cumulative_confidence: float
    """Decaying confidence score (1.0 at day 0, decreasing)."""


@dataclass
class RecursivePredictionResult:
    """Container for the full recursive prediction output."""

    snapshots: dict[str, RecursiveSnapshot] = field(default_factory=dict)
    """Keyed by horizon label ('1d', '5d', '21d', etc.)."""

    full_trajectory: dict[str, list[float]] | None = None
    """Per-variable list of all recursive predictions (for charting)."""

    trajectory_dates: list[str] | None = None
    """ISO date strings for each step in the trajectory."""

    method: str = "recursive_1d"

    total_steps: int = 0
    """Number of recursive steps completed."""

    confidence_decay_rate: float = 0.0
    """Empirical per-step confidence decay."""

    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for profile storage."""
        d: dict[str, Any] = {
            "method": self.method,
            "total_steps": self.total_steps,
            "confidence_decay_rate": round(self.confidence_decay_rate, 6),
            "available": self.available,
            "snapshots": {},
        }
        for label, snap in self.snapshots.items():
            d["snapshots"][label] = {
                "day": snap.day,
                "date": snap.date,
                "predictions": {
                    k: round(v, 6) if isinstance(v, float) else v
                    for k, v in snap.predictions.items()
                },
                "uncertainty": {
                    k: (round(lo, 6), round(hi, 6))
                    for k, (lo, hi) in snap.uncertainty.items()
                },
                "regime": snap.regime,
                "cumulative_confidence": round(snap.cumulative_confidence, 4),
            }
        if self.full_trajectory is not None:
            d["trajectory_variables"] = list(self.full_trajectory.keys())
            d["trajectory_length"] = (
                len(next(iter(self.full_trajectory.values())))
                if self.full_trajectory
                else 0
            )
        return d


# ---------------------------------------------------------------------------
# Helper: build a synthetic cache row from predictions
# ---------------------------------------------------------------------------


def _build_synthetic_row(
    extended_cache: pd.DataFrame,
    predicted_values: dict[str, float],
    next_date: pd.Timestamp,
) -> pd.Series:
    """Build a synthetic daily cache row from 1d predictions.

    Recomputes rolling-window features (returns, volatility, drawdown)
    from the extended cache + new predicted close. Financial statement
    columns are carried forward from the last actual row.

    Parameters
    ----------
    extended_cache:
        The cache including any previously appended synthetic rows.
    predicted_values:
        Dict of {variable: predicted_value} from the ensemble.
    next_date:
        The business date for this synthetic row.

    Returns
    -------
    A pandas Series indexed by column names.
    """
    # Start with forward-fill of the last row (carries financials, etc.)
    last_row = extended_cache.iloc[-1].copy()
    row = last_row.copy()

    # Overwrite with model predictions where available
    for var, val in predicted_values.items():
        if var in row.index:
            row[var] = val

    # Derive return_1d from close
    if "close" in predicted_values and "close" in extended_cache.columns:
        prev_close = extended_cache["close"].dropna().iloc[-1]
        new_close = predicted_values["close"]
        if prev_close > 0 and not np.isnan(prev_close):
            row["return_1d"] = (new_close - prev_close) / prev_close
            row["log_return_1d"] = math.log(new_close / prev_close) if new_close > 0 else 0.0
        else:
            row["return_1d"] = 0.0
            row["log_return_1d"] = 0.0

    # Update rolling volatility from the return series
    if "return_1d" in extended_cache.columns:
        returns = list(extended_cache["return_1d"].dropna().values[-20:])
        returns.append(row.get("return_1d", 0.0))
        if len(returns) >= 5:
            row["volatility_21d"] = float(np.std(returns))

    # Update rolling drawdown from close series
    if "close" in extended_cache.columns and "close" in predicted_values:
        closes = list(extended_cache["close"].dropna().values[-251:])
        closes.append(predicted_values["close"])
        rolling_max = max(closes) if closes else predicted_values["close"]
        if rolling_max > 0:
            row["drawdown_252d"] = (predicted_values["close"] - rolling_max) / rolling_max

    # Update 5d and 21d returns if enough history
    if "close" in extended_cache.columns and "close" in predicted_values:
        close_series = extended_cache["close"].dropna()
        if len(close_series) >= 5:
            row["return_5d"] = (
                (predicted_values["close"] - float(close_series.iloc[-5]))
                / float(close_series.iloc[-5])
            )
        if len(close_series) >= 21:
            row["return_21d"] = (
                (predicted_values["close"] - float(close_series.iloc[-21]))
                / float(close_series.iloc[-21])
            )

    row.name = next_date
    return row


# ---------------------------------------------------------------------------
# Helper: ensemble predict 1 step ahead
# ---------------------------------------------------------------------------


def _ensemble_predict_1d(
    model_states: dict[str, Any],
    extended_cache: pd.DataFrame,
    variables: list[str],
    momentum_blend: float = 0.3,
) -> dict[str, float]:
    """Produce a 1-step-ahead prediction for each variable using the ensemble.

    Uses the single best fitted model per variable from the forward pass
    model_states. For close predictions, applies a momentum overlay
    (P1 fix: 70% model + 30% momentum signal).

    Parameters
    ----------
    model_states:
        Dict of {variable: BaseModelWrapper} from ForwardPassResult.
    extended_cache:
        Current cache (real + synthetic rows).
    variables:
        Which variables to predict.
    momentum_blend:
        Weight for momentum overlay on close predictions (P1 fix).

    Returns
    -------
    Dict of {variable: predicted_value}.
    """
    predictions: dict[str, float] = {}

    for var in variables:
        wrapper = model_states.get(var)
        if wrapper is None:
            # No model fitted for this variable -- carry forward
            if var in extended_cache.columns and extended_cache[var].notna().any():
                predictions[var] = float(extended_cache[var].dropna().iloc[-1])
            continue

        # Build state vector from latest cache row
        try:
            last_vals = extended_cache[var].dropna().values
            if len(last_vals) == 0:
                continue
            state_t = last_vals[-1:]

            # Model predict
            pred = wrapper.predict(state_t)
            if isinstance(pred, np.ndarray):
                pred_val = float(pred[0]) if len(pred) > 0 else float(pred)
            else:
                pred_val = float(pred)

            # Sanity check -- reject NaN/Inf
            if np.isnan(pred_val) or np.isinf(pred_val):
                pred_val = float(last_vals[-1])

            # Momentum overlay for close (P1 fix: 70% model + 30% momentum)
            if var == "close" and len(last_vals) >= 5 and momentum_blend > 0:
                momentum_5d = float(last_vals[-1]) - float(last_vals[-5])
                momentum_signal = float(last_vals[-1]) + momentum_5d / 5.0
                pred_val = (1.0 - momentum_blend) * pred_val + momentum_blend * momentum_signal

            predictions[var] = pred_val

        except Exception as exc:
            logger.debug("Recursive predict failed for %s: %s", var, exc)
            if var in extended_cache.columns and extended_cache[var].notna().any():
                predictions[var] = float(extended_cache[var].dropna().iloc[-1])

    return predictions


# ---------------------------------------------------------------------------
# Helper: evolve regime label
# ---------------------------------------------------------------------------


def _evolve_regime(
    current_regime: str,
    transition_matrix: np.ndarray | None,
    regime_order: list[str] | None,
    rng: np.random.Generator,
) -> str:
    """Sample the next regime from the HMM transition matrix.

    If no transition matrix is available, carries forward the current regime.
    """
    if transition_matrix is None or regime_order is None:
        return current_regime

    try:
        idx = regime_order.index(current_regime)
        row = transition_matrix[idx]
        # Normalize (safety)
        row = np.array(row, dtype=float)
        row_sum = row.sum()
        if row_sum > 0:
            row = row / row_sum
        else:
            return current_regime
        next_idx = rng.choice(len(regime_order), p=row)
        return regime_order[int(next_idx)]
    except (ValueError, IndexError):
        return current_regime


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_recursive_predictions(
    cache: pd.DataFrame,
    model_states: dict[str, Any],
    *,
    horizon_days: int = 252,
    snapshot_days: dict[str, int] | None = None,
    transition_matrix: np.ndarray | None = None,
    regime_order: list[str] | None = None,
    momentum_blend: float = 0.3,
    conformal_width_1d: float | None = None,
    drift_correction_interval: int = 5,
    random_state: int = 42,
) -> RecursivePredictionResult:
    """Run recursive day-by-day prediction aggregation.

    Focuses on maximizing 1d accuracy, then chains predictions
    recursively to build multi-horizon forecasts.

    Parameters
    ----------
    cache:
        Full daily cache up to the reference date (DatetimeIndex).
    model_states:
        Dict of {variable: BaseModelWrapper} from ForwardPassResult.
        These are the fitted models from the forward pass. One model
        per variable (the best model that fitted successfully).
    horizon_days:
        Maximum number of recursive steps (default 252 = 1 year).
    snapshot_days:
        Which recursive steps to extract as horizon predictions.
        Default: {"1d": 1, "5d": 5, "21d": 21, "63d": 63, "252d": 252}.
    transition_matrix:
        HMM transition matrix from Monte Carlo for regime evolution.
    regime_order:
        Ordered list of regime labels matching transition_matrix rows.
    momentum_blend:
        Weight for momentum overlay on close predictions (0.0 to 1.0).
    conformal_width_1d:
        Base conformal interval half-width for 1d predictions.
        If None, estimated from recent prediction residuals.
    drift_correction_interval:
        Every N steps, apply a small drift correction toward the
        historical trend to prevent runaway trajectories.
    random_state:
        Random seed for regime sampling reproducibility.

    Returns
    -------
    RecursivePredictionResult with snapshots at each horizon and full trajectory.
    """
    if cache is None or cache.empty or not model_states:
        logger.warning("Recursive predictions: empty cache or no model states")
        return RecursivePredictionResult(available=False)

    if snapshot_days is None:
        snapshot_days = dict(DEFAULT_SNAPSHOT_DAYS)

    # Cap horizon_days to max snapshot
    max_snap = max(snapshot_days.values()) if snapshot_days else 252
    horizon_days = min(horizon_days, max_snap)

    rng = np.random.default_rng(random_state)

    # Identify variables we can predict (those with fitted models)
    predictable_vars = [v for v in model_states if v in cache.columns]
    if not predictable_vars:
        logger.warning("Recursive predictions: no predictable variables")
        return RecursivePredictionResult(available=False)

    # Ensure "close" is first if available (other vars may depend on it)
    if "close" in predictable_vars:
        predictable_vars.remove("close")
        predictable_vars.insert(0, "close")

    logger.info(
        "Starting recursive predictions: %d steps, %d variables",
        horizon_days,
        len(predictable_vars),
    )

    # Estimate 1d conformal width from recent model residuals if not provided
    if conformal_width_1d is None and "close" in cache.columns:
        close_vals = cache["close"].dropna().values
        if len(close_vals) >= 30:
            recent_returns = np.diff(close_vals[-30:]) / close_vals[-30:-1]
            conformal_width_1d = float(np.std(recent_returns) * abs(close_vals[-1]))
        else:
            conformal_width_1d = abs(float(close_vals[-1])) * 0.015 if len(close_vals) > 0 else 1.0

    # Get current regime
    current_regime = "unknown"
    if "regime_label" in cache.columns and cache["regime_label"].notna().any():
        current_regime = str(cache["regime_label"].dropna().iloc[-1])

    # Get historical trend for drift correction
    trend_slope = 0.0
    if "close" in cache.columns:
        close_vals = cache["close"].dropna().values
        if len(close_vals) >= 63:
            # 63-day linear trend slope (daily increment)
            x = np.arange(63, dtype=float)
            y = close_vals[-63:]
            try:
                slope = float(np.polyfit(x, y, 1)[0])
                trend_slope = slope
            except Exception:
                pass

    # Pre-allocate synthetic rows list (B4 fix: avoid O(n^2) concat in loop).
    # We collect rows in a list and only concat once at the end of each step
    # into the extended cache. We also maintain the extended_cache incrementally
    # by appending one row at a time using loc assignment on a pre-grown index.
    extended_cache = cache.copy()

    # Determine next business date
    last_date = cache.index[-1]
    if isinstance(last_date, pd.Timestamp):
        next_date = last_date + pd.offsets.BDay(1)
    else:
        next_date = pd.Timestamp(last_date) + pd.offsets.BDay(1)

    # Pre-generate all future business dates
    future_dates = pd.bdate_range(start=next_date, periods=horizon_days)

    # Pre-allocate the extended cache with NaN rows for all future dates
    # This avoids O(n^2) concat -- we fill rows by index assignment instead
    empty_rows = pd.DataFrame(
        index=future_dates,
        columns=extended_cache.columns,
        dtype=float,
    )
    extended_cache = pd.concat([extended_cache, empty_rows])

    # Trajectory storage
    trajectory: dict[str, list[float]] = {v: [] for v in predictable_vars}
    trajectory_dates: list[str] = []

    # Snapshot collection
    snapshots: dict[str, RecursiveSnapshot] = {}

    # Base confidence for decay calculation
    base_confidence = 1.0
    # Per-step decay: confidence halves in ~60 steps
    decay_rate = 0.012

    for step in range(1, horizon_days + 1):
        current_date = future_dates[step - 1]

        # Step 1: Ensemble predict 1 step ahead
        # Use only the already-filled portion of extended_cache (up to previous row)
        filled_end = future_dates[step - 2] if step > 1 else last_date
        working_cache = extended_cache.loc[:filled_end]

        predicted = _ensemble_predict_1d(
            model_states=model_states,
            extended_cache=working_cache,
            variables=predictable_vars,
            momentum_blend=momentum_blend,
        )

        if not predicted:
            logger.warning("Recursive step %d: no predictions produced, stopping", step)
            break

        # Step 2: Drift correction every N steps
        if (
            drift_correction_interval > 0
            and step % drift_correction_interval == 0
            and "close" in predicted
            and trend_slope != 0
        ):
            # Nudge close prediction slightly toward historical trend
            expected_by_trend = float(cache["close"].dropna().iloc[-1]) + trend_slope * step
            current_pred = predicted["close"]
            # Blend: 95% recursive + 5% trend anchor
            predicted["close"] = 0.95 * current_pred + 0.05 * expected_by_trend

        # Step 3: Build synthetic row and fill into pre-allocated slot
        synthetic_row = _build_synthetic_row(working_cache, predicted, current_date)
        for col in synthetic_row.index:
            if col in extended_cache.columns:
                extended_cache.at[current_date, col] = synthetic_row[col]

        # B1 FIX: Do NOT update model wrappers with synthetic predictions.
        # Feeding a model its own prediction as "ground truth" corrupts the
        # Kalman gain (innovation -> 0, gain -> 0, model ignores new data).
        # Models retain their calibration from the real forward pass instead.

        # Step 4: Evolve regime
        current_regime = _evolve_regime(
            current_regime, transition_matrix, regime_order, rng
        )

        # Step 5: Store trajectory
        step_date = current_date.strftime("%Y-%m-%d")
        trajectory_dates.append(step_date)
        for var in predictable_vars:
            trajectory[var].append(predicted.get(var, float("nan")))

        # Step 6: Snapshot if this step matches a horizon
        step_confidence = base_confidence * math.exp(-decay_rate * step)
        for label, snap_day in snapshot_days.items():
            if step == snap_day:
                # Compute uncertainty bands (sqrt-t scaling from 1d width)
                uncertainty: dict[str, tuple[float, float]] = {}
                for var in predictable_vars:
                    val = predicted.get(var, 0.0)
                    if conformal_width_1d is not None and var == "close":
                        half_w = conformal_width_1d * math.sqrt(step)
                        uncertainty[var] = (val - half_w, val + half_w)
                    elif val != 0:
                        # For non-close vars: scale by relative uncertainty
                        rel_unc = 0.02 * math.sqrt(step)  # ~2% per sqrt(day)
                        half_w = abs(val) * rel_unc
                        uncertainty[var] = (val - half_w, val + half_w)
                    else:
                        uncertainty[var] = (0.0, 0.0)

                snapshots[label] = RecursiveSnapshot(
                    day=step,
                    date=step_date,
                    predictions=dict(predicted),
                    uncertainty=uncertainty,
                    regime=current_regime,
                    cumulative_confidence=step_confidence,
                )
                logger.info(
                    "Recursive snapshot '%s' (day %d): close=%.2f, confidence=%.3f",
                    label,
                    step,
                    predicted.get("close", 0.0),
                    step_confidence,
                )

    result = RecursivePredictionResult(
        snapshots=snapshots,
        full_trajectory=trajectory,
        trajectory_dates=trajectory_dates,
        method="recursive_1d",
        total_steps=len(trajectory_dates),
        confidence_decay_rate=decay_rate,
        available=len(snapshots) > 0,
    )

    logger.info(
        "Recursive predictions complete: %d steps, %d snapshots captured",
        result.total_steps,
        len(result.snapshots),
    )

    return result


# ---------------------------------------------------------------------------
# Forecast-based fallback (Fix 10: works without fitted model objects)
# ---------------------------------------------------------------------------


def _interpolate_horizon(
    forecasts_for_var: dict[str, float],
    day: int,
) -> float | None:
    """Interpolate a point prediction for an arbitrary day from known horizons.

    Uses linear interpolation between the two nearest horizon anchors.
    Known horizons: 1d, 5d, 21d, 63d, 252d.
    """
    # Build sorted list of (horizon_days, value) from available horizons
    horizon_map = {"1d": 1, "5d": 5, "21d": 21, "63d": 63, "252d": 252}
    points: list[tuple[int, float]] = []
    for label, h_days in horizon_map.items():
        val = forecasts_for_var.get(label)
        if val is not None:
            try:
                points.append((h_days, float(val)))
            except (TypeError, ValueError):
                pass

    if not points:
        return None

    points.sort(key=lambda x: x[0])

    # Clamp to nearest endpoint if day is outside range
    if day <= points[0][0]:
        return points[0][1]
    if day >= points[-1][0]:
        return points[-1][1]

    # Find bracketing pair and interpolate
    for i in range(len(points) - 1):
        d0, v0 = points[i]
        d1, v1 = points[i + 1]
        if d0 <= day <= d1:
            w = (day - d0) / (d1 - d0)
            return v0 * (1 - w) + v1 * w

    return points[-1][1]


def run_recursive_from_forecasts(
    cache: pd.DataFrame,
    forecasts: dict[str, dict[str, float]],
    *,
    horizon_days: int = 252,
    snapshot_days: dict[str, int] | None = None,
    transition_matrix: np.ndarray | None = None,
    regime_order: list[str] | None = None,
    random_state: int = 42,
) -> RecursivePredictionResult:
    """Forecast-based recursive predictions (no fitted model objects needed).

    Interpolates between known forecast horizons (1d/5d/21d/63d/252d) and
    applies sqrt(t) confidence decay + MC regime drift. Produces the same
    RecursivePredictionResult shape as the model-based path.

    This is the fallback for staged pipeline mode where model_states are
    lost to pickle serialization between sub-stages (Fix 10).

    Parameters
    ----------
    cache:
        Daily cache up to reference date.
    forecasts:
        ``{variable: {horizon_label: value}}`` from ForecastResult.
    horizon_days:
        Maximum number of days to project.
    snapshot_days:
        Which days to extract as horizon snapshots.
    transition_matrix:
        HMM transition matrix from Monte Carlo.
    regime_order:
        Ordered regime labels matching transition_matrix rows.
    random_state:
        Random seed for regime sampling.
    """
    if cache is None or cache.empty or not forecasts:
        return RecursivePredictionResult(available=False)

    if snapshot_days is None:
        snapshot_days = dict(DEFAULT_SNAPSHOT_DAYS)

    max_snap = max(snapshot_days.values()) if snapshot_days else 252
    horizon_days = min(horizon_days, max_snap)

    rng = np.random.default_rng(random_state)

    # Identify variables with at least one horizon forecast
    predictable_vars = [v for v in forecasts if forecasts[v]]
    if not predictable_vars:
        return RecursivePredictionResult(available=False)

    # Current regime
    current_regime = "unknown"
    if "regime_label" in cache.columns and cache["regime_label"].notna().any():
        current_regime = str(cache["regime_label"].dropna().iloc[-1])

    # Estimate 1d conformal width from recent returns
    conformal_width_1d = 1.0
    if "close" in cache.columns:
        close_vals = cache["close"].dropna().values
        if len(close_vals) >= 30:
            recent_returns = np.diff(close_vals[-30:]) / close_vals[-30:-1]
            conformal_width_1d = float(np.std(recent_returns) * abs(close_vals[-1]))
        elif len(close_vals) > 0:
            conformal_width_1d = abs(float(close_vals[-1])) * 0.015

    # Generate future dates
    last_date = cache.index[-1]
    next_date = pd.Timestamp(last_date) + pd.offsets.BDay(1)
    future_dates = pd.bdate_range(start=next_date, periods=horizon_days)

    # Trajectory storage
    trajectory: dict[str, list[float]] = {v: [] for v in predictable_vars}
    trajectory_dates: list[str] = []

    # Snapshot collection
    snapshots: dict[str, RecursiveSnapshot] = {}

    # Per-step confidence decay: halves in ~60 steps
    decay_rate = 0.012

    for step in range(1, horizon_days + 1):
        current_date = future_dates[step - 1]
        step_date = current_date.strftime("%Y-%m-%d")

        # Interpolate predictions for this day from known horizons
        predicted: dict[str, float] = {}
        for var in predictable_vars:
            val = _interpolate_horizon(forecasts[var], step)
            if val is not None:
                predicted[var] = val

        if not predicted:
            break

        # Evolve regime
        current_regime = _evolve_regime(
            current_regime, transition_matrix, regime_order, rng,
        )

        # Store trajectory
        trajectory_dates.append(step_date)
        for var in predictable_vars:
            trajectory[var].append(predicted.get(var, float("nan")))

        # Snapshot if this step matches a horizon
        step_confidence = math.exp(-decay_rate * step)
        for label, snap_day in snapshot_days.items():
            if step == snap_day:
                uncertainty: dict[str, tuple[float, float]] = {}
                for var in predictable_vars:
                    val = predicted.get(var, 0.0)
                    if var == "close" and conformal_width_1d > 0:
                        half_w = conformal_width_1d * math.sqrt(step)
                        uncertainty[var] = (val - half_w, val + half_w)
                    elif val != 0:
                        rel_unc = 0.02 * math.sqrt(step)
                        half_w = abs(val) * rel_unc
                        uncertainty[var] = (val - half_w, val + half_w)
                    else:
                        uncertainty[var] = (0.0, 0.0)

                snapshots[label] = RecursiveSnapshot(
                    day=step,
                    date=step_date,
                    predictions=dict(predicted),
                    uncertainty=uncertainty,
                    regime=current_regime,
                    cumulative_confidence=step_confidence,
                )
                logger.info(
                    "Forecast-recursive snapshot '%s' (day %d): confidence=%.3f",
                    label, step, step_confidence,
                )

    result = RecursivePredictionResult(
        snapshots=snapshots,
        full_trajectory=trajectory,
        trajectory_dates=trajectory_dates,
        method="forecast_interpolation",
        total_steps=len(trajectory_dates),
        confidence_decay_rate=decay_rate,
        available=len(snapshots) > 0,
    )

    logger.info(
        "Forecast-recursive predictions: %d steps, %d snapshots",
        result.total_steps, len(result.snapshots),
    )

    return result

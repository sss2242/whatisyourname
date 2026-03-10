"""Phase 3 -- Walk-forward prediction loop.

Iterates day-by-day through the 2-year daily cache.  For each day *t*,
the ensemble of models predicts day *t+1*, then the prediction is
compared against the actual value from the cache.

**Key behaviours:**

1. **Per-model error tracking** -- each model's cumulative and rolling
   MAE/RMSE are recorded per survival mode.

2. **Retrain at switch points** -- when the survival timeline reports a
   mode transition, all models are re-fitted on the history up to day *t*
   so they adapt to the new regime.

3. **Mode-conditioned leaderboard** -- after the walk-forward pass, each
   model receives a score for each survival mode, identifying which
   model performs best under each condition.

**Relationship to ``run_forward_pass()`` in forecasting.py:**

Both modules implement day-by-day predict-compare-update loops, but
they serve different purposes:

- ``run_forward_pass()`` (forecasting.py, Phase D): the *main* temporal
  engine with online model updates, PID-adjusted learning rates,
  conformal calibration, candlestick pattern injection, cycle
  decomposition, and DTW analog signals.  Its output model states
  feed into prediction aggregation.

- ``run_walk_forward()`` (this module, Phase 3): a *diagnostic* loop
  focused on per-survival-mode model scoring.  Its main output is the
  ``WalkForwardResult.mode_scores`` leaderboard which tells the
  prediction aggregator which models perform best under each survival
  condition (normal, company_only, country_exposed, etc.).

In short: the forward pass *learns*, the walk-forward *evaluates*.
Their outputs are complementary and both feed into the prediction
aggregator.

Top-level entry point:
    ``run_walk_forward(daily_cache, timeline_result, models) -> WalkForwardResult``

Spec refs: Phase 3 of survival-time-series-and-predictive-learning architecture.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of history days before we start predicting.
_MIN_HISTORY_DAYS: int = 60

# Rolling window for error tracking (business days).
_ERROR_ROLLING_WINDOW: int = 63  # ~3 months

# Variables to predict by default (close and key financials).
_DEFAULT_PREDICT_VARIABLES: list[str] = [
    "close",
    "current_ratio",
    "debt_to_equity_abs",
    "fcf_yield",
    "volatility_21d",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ModelDayError:
    """Error record for one model on one day."""

    date: str = ""
    model_name: str = ""
    variable: str = ""
    predicted: float = float("nan")
    actual: float = float("nan")
    error: float = float("nan")       # actual - predicted
    abs_error: float = float("nan")
    sq_error: float = float("nan")
    survival_mode: str = "normal"


@dataclass
class ModelModeScore:
    """Aggregated score for one model in one survival mode."""

    model_name: str = ""
    survival_mode: str = ""
    variable: str = ""
    n_days: int = 0
    mae: float = float("nan")
    rmse: float = float("nan")
    # Rank within this mode (1 = best).
    rank: int = 0


@dataclass
class WalkForwardResult:
    """Output of the walk-forward prediction loop."""

    # Per-day, per-model error records.
    day_errors: list[ModelDayError] = field(default_factory=list)

    # Per-model, per-mode aggregated scores.
    mode_scores: list[ModelModeScore] = field(default_factory=list)

    # Best model per mode per variable.
    # {(mode, variable): model_name}
    best_model_by_mode: dict[tuple[str, str], str] = field(default_factory=dict)

    # Days where retraining occurred.
    retrain_dates: list[str] = field(default_factory=list)

    # Summary statistics.
    total_days_evaluated: int = 0
    total_predictions: int = 0
    n_retrains: int = 0

    # Overall best model (across all modes).
    overall_best_model: str = ""
    overall_mae: float = float("nan")

    fitted: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Simple model wrappers for walk-forward
# ---------------------------------------------------------------------------


class WFBaselineModel:
    """Last-value carry-forward baseline."""

    name: str = "baseline_last"

    def predict_next(self, history: pd.Series) -> float:
        """Predict next value as the last observed value."""
        clean = history.dropna()
        if len(clean) == 0:
            return float("nan")
        return float(clean.iloc[-1])

    def fit(self, history: pd.Series) -> None:
        """No-op for baseline."""
        pass


class WFEMAModel:
    """Exponential moving average model."""

    name: str = "ema_21"

    def __init__(self, span: int = 21) -> None:
        self._span = span

    def predict_next(self, history: pd.Series) -> float:
        """Predict next value as the current EMA."""
        clean = history.dropna()
        if len(clean) < 2:
            return float(clean.iloc[-1]) if len(clean) == 1 else float("nan")
        ema = clean.ewm(span=self._span, adjust=False).mean()
        return float(ema.iloc[-1])

    def fit(self, history: pd.Series) -> None:
        """No-op -- EMA is non-parametric."""
        pass


class WFLinearTrendModel:
    """Simple linear trend extrapolation."""

    name: str = "linear_trend"

    def __init__(self) -> None:
        self._slope: float = 0.0
        self._intercept: float = 0.0
        self._fitted: bool = False

    def predict_next(self, history: pd.Series) -> float:
        """Predict next value by extending the linear trend."""
        if not self._fitted:
            self.fit(history)
        if not self._fitted:
            clean = history.dropna()
            return float(clean.iloc[-1]) if len(clean) > 0 else float("nan")
        n = len(history)
        return self._intercept + self._slope * n

    def fit(self, history: pd.Series) -> None:
        """Fit a simple OLS line to the history."""
        clean = history.dropna()
        if len(clean) < 5:
            self._fitted = False
            return
        y = clean.values.astype(float)
        x = np.arange(len(y), dtype=float)
        try:
            coeffs = np.polyfit(x, y, 1)
            self._slope = float(coeffs[0])
            self._intercept = float(coeffs[1])
            self._fitted = True
        except Exception:
            self._fitted = False


class WFMeanReversionModel:
    """Mean reversion model: predict toward the rolling mean."""

    name: str = "mean_reversion"

    def __init__(self, window: int = 63, speed: float = 0.1) -> None:
        self._window = window
        self._speed = speed

    def predict_next(self, history: pd.Series) -> float:
        """Predict next value by reverting toward the rolling mean."""
        clean = history.dropna()
        if len(clean) == 0:
            return float("nan")
        last_val = float(clean.iloc[-1])
        window_data = clean.iloc[-self._window:]
        mean_val = float(window_data.mean())
        # Partial reversion toward the mean
        predicted = last_val + self._speed * (mean_val - last_val)
        return predicted

    def fit(self, history: pd.Series) -> None:
        """No-op."""
        pass


# ---------------------------------------------------------------------------
# Default model factory
# ---------------------------------------------------------------------------


def get_default_wf_models() -> list[Any]:
    """Return the default set of walk-forward models."""
    return [
        WFBaselineModel(),
        WFEMAModel(span=21),
        WFLinearTrendModel(),
        WFMeanReversionModel(window=63, speed=0.1),
    ]


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------


def _compute_day_errors(
    models: list[Any],
    history: pd.Series,
    actual_value: float,
    day_str: str,
    variable: str,
    survival_mode: str,
) -> list[ModelDayError]:
    """Run all models on history and compute errors against actual."""
    errors: list[ModelDayError] = []
    for model in models:
        try:
            predicted = model.predict_next(history)
        except Exception:
            predicted = float("nan")

        if math.isnan(predicted) or math.isnan(actual_value):
            err = float("nan")
            abs_err = float("nan")
            sq_err = float("nan")
        else:
            err = actual_value - predicted
            abs_err = abs(err)
            sq_err = err ** 2

        errors.append(ModelDayError(
            date=day_str,
            model_name=model.name,
            variable=variable,
            predicted=predicted,
            actual=actual_value,
            error=err,
            abs_error=abs_err,
            sq_error=sq_err,
            survival_mode=survival_mode,
        ))

    return errors


def _aggregate_mode_scores(
    day_errors: list[ModelDayError],
) -> tuple[list[ModelModeScore], dict[tuple[str, str], str]]:
    """Aggregate per-day errors into per-mode scores and rankings.

    Returns
    -------
    (mode_scores, best_model_by_mode)
    """
    # Group by (model, mode, variable)
    groups: dict[tuple[str, str, str], list[ModelDayError]] = {}
    for de in day_errors:
        key = (de.model_name, de.survival_mode, de.variable)
        groups.setdefault(key, []).append(de)

    scores: list[ModelModeScore] = []
    for (model_name, mode, variable), errs in groups.items():
        abs_errors = [e.abs_error for e in errs if not math.isnan(e.abs_error)]
        sq_errors = [e.sq_error for e in errs if not math.isnan(e.sq_error)]

        if not abs_errors:
            continue

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(sq_errors)))

        scores.append(ModelModeScore(
            model_name=model_name,
            survival_mode=mode,
            variable=variable,
            n_days=len(abs_errors),
            mae=mae,
            rmse=rmse,
        ))

    # Rank within each (mode, variable) group
    mode_var_groups: dict[tuple[str, str], list[ModelModeScore]] = {}
    for s in scores:
        key = (s.survival_mode, s.variable)
        mode_var_groups.setdefault(key, []).append(s)

    best_model_by_mode: dict[tuple[str, str], str] = {}
    for (mode, variable), group in mode_var_groups.items():
        group.sort(key=lambda s: s.mae if not math.isnan(s.mae) else 1e10)
        for rank, s in enumerate(group, 1):
            s.rank = rank
        if group:
            best_model_by_mode[(mode, variable)] = group[0].model_name

    return scores, best_model_by_mode


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_walk_forward(
    daily_cache: pd.DataFrame,
    survival_modes: pd.Series | None = None,
    switch_points: pd.Series | None = None,
    models: list[Any] | None = None,
    variables: list[str] | None = None,
    min_history: int = _MIN_HISTORY_DAYS,
) -> WalkForwardResult:
    """Run the walk-forward prediction loop.

    Parameters
    ----------
    daily_cache:
        Full 2-year daily cache DataFrame.
    survival_modes:
        Series of survival mode labels (from ``SurvivalTimelineResult``).
        If None, defaults to "normal" for all days.
    switch_points:
        Binary series indicating mode switch days.  Models are
        retrained on these days.
    models:
        List of model instances (each must have ``.name``,
        ``.predict_next(history)``, ``.fit(history)``).
        Defaults to ``get_default_wf_models()``.
    variables:
        Variables to predict.  Defaults to ``_DEFAULT_PREDICT_VARIABLES``.
    min_history:
        Minimum history days before starting predictions.

    Returns
    -------
    WalkForwardResult
        Contains per-day errors, mode scores, and best-model rankings.
    """
    result = WalkForwardResult()

    if daily_cache.empty:
        result.error = "Empty daily cache"
        logger.warning(result.error)
        return result

    if models is None:
        models = get_default_wf_models()

    if variables is None:
        variables = [v for v in _DEFAULT_PREDICT_VARIABLES if v in daily_cache.columns]

    if not variables:
        result.error = "No predictable variables found in cache"
        logger.warning(result.error)
        return result

    if survival_modes is None:
        survival_modes = pd.Series("normal", index=daily_cache.index)

    if switch_points is None:
        switch_points = pd.Series(0, index=daily_cache.index, dtype=int)

    n_days = len(daily_cache)
    dates = daily_cache.index

    logger.info(
        "Starting walk-forward: %d days, %d variables, %d models",
        n_days, len(variables), len(models),
    )

    all_day_errors: list[ModelDayError] = []

    try:
        for t in range(min_history, n_days - 1):
            day_t = dates[t]
            day_t1 = dates[t + 1]
            day_str = str(day_t.date()) if hasattr(day_t, "date") else str(day_t)

            mode = str(survival_modes.iloc[t]) if t < len(survival_modes) else "normal"

            # Check if we need to retrain (switch point)
            is_switch = bool(switch_points.iloc[t]) if t < len(switch_points) else False
            if is_switch:
                result.retrain_dates.append(day_str)
                for var in variables:
                    if var not in daily_cache.columns:
                        continue
                    history = daily_cache[var].iloc[:t + 1]
                    for model in models:
                        try:
                            model.fit(history)
                        except Exception as exc:
                            logger.debug(
                                "Retrain failed for %s on %s: %s",
                                model.name, var, exc,
                            )

            # Predict day t+1 for each variable
            for var in variables:
                if var not in daily_cache.columns:
                    continue

                history = daily_cache[var].iloc[:t + 1]
                actual_value = float(daily_cache[var].iloc[t + 1])

                day_errors = _compute_day_errors(
                    models, history, actual_value, day_str, var, mode,
                )
                all_day_errors.extend(day_errors)

        result.day_errors = all_day_errors
        result.total_days_evaluated = n_days - min_history - 1
        result.total_predictions = len(all_day_errors)
        result.n_retrains = len(result.retrain_dates)

        # Aggregate mode scores
        result.mode_scores, result.best_model_by_mode = _aggregate_mode_scores(
            all_day_errors,
        )

        # Overall best model (lowest MAE across all modes)
        model_overall_errors: dict[str, list[float]] = {}
        for de in all_day_errors:
            if not math.isnan(de.abs_error):
                model_overall_errors.setdefault(de.model_name, []).append(de.abs_error)

        if model_overall_errors:
            best_name = ""
            best_mae = float("inf")
            for name, errs in model_overall_errors.items():
                mae = float(np.mean(errs))
                if mae < best_mae:
                    best_mae = mae
                    best_name = name
            result.overall_best_model = best_name
            result.overall_mae = best_mae

        result.fitted = True

        logger.info(
            "Walk-forward complete: %d days evaluated, %d predictions, "
            "%d retrains, overall_best=%s (MAE=%.6f), "
            "%d mode-variable combinations scored",
            result.total_days_evaluated,
            result.total_predictions,
            result.n_retrains,
            result.overall_best_model,
            result.overall_mae if not math.isnan(result.overall_mae) else -1,
            len(result.best_model_by_mode),
        )

    except Exception as exc:
        result.error = f"Walk-forward failed: {exc}"
        logger.error(result.error)

    return result


def get_mode_weights_from_walk_forward(
    wf_result: WalkForwardResult,
) -> dict[str, dict[str, float]]:
    """Extract per-mode model weights from walk-forward results.

    Converts the mode-conditioned leaderboard into inverse-MAE weights
    suitable for the survival-aware prediction aggregator.

    Returns
    -------
    Dict mapping survival_mode -> {model_name: weight}.
    Weights are normalised to sum to 1.0 within each mode.
    """
    if not wf_result.fitted or not wf_result.mode_scores:
        return {}

    # Group scores by (mode, variable)
    mode_model_mae: dict[str, dict[str, list[float]]] = {}
    for score in wf_result.mode_scores:
        mode_model_mae.setdefault(score.survival_mode, {})
        mode_model_mae[score.survival_mode].setdefault(score.model_name, []).append(
            score.mae if not math.isnan(score.mae) else 1e10,
        )

    result: dict[str, dict[str, float]] = {}
    for mode, model_maes in mode_model_mae.items():
        # Average MAE per model across variables
        avg_mae: dict[str, float] = {}
        for model_name, maes in model_maes.items():
            avg_mae[model_name] = float(np.mean(maes))

        # Inverse-MAE weighting
        min_mae = 1e-10
        inv_mae = {
            name: 1.0 / max(mae, min_mae) for name, mae in avg_mae.items()
        }
        total = sum(inv_mae.values())
        if total > 0:
            result[mode] = {name: w / total for name, w in inv_mae.items()}
        else:
            n = len(inv_mae)
            result[mode] = {name: 1.0 / n for name in inv_mae}

    return result

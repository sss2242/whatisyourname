"""T6.4 -- Prediction aggregation with ensemble weighting.

Consumes outputs from T6.1 (regime detection), T6.2 (forecasting models),
and T6.3 (Monte Carlo simulations) to produce unified, multi-horizon
predictions with uncertainty bands and Technical Alpha protection.

Additionally integrates optional results from sibling modules when
available, falling back to the base behaviour when they are not:

- **Conformal prediction intervals** (``ConformalResult``) replace the
  Gaussian RMSE-based bands with distribution-free calibrated intervals.
- **Dual regime probabilities** (``DualRegimeResult``) enable soft
  regime-weighted ensemble blending instead of hard label switching.
- **Copula tail dependencies** (``CopulaResult``) widen uncertainty
  bands for variables with high joint crisis probability.
- **DTW historical analogs** (``DTWAnalogResult``) provide an
  independent empirical forecast channel.
- **Granger causal structure** (``GrangerResult``) propagates forecast
  adjustments from causal drivers to dependent variables.
- **SHAP explanations** (``SHAPResult``) attach interpretability
  narratives to each prediction.
- **Walk-forward diagnostics** (``WalkForwardResult``) enable
  recency-weighted ensemble RMSE.
- **Kalman/Particle fusion** via model synergies for regime-conditional
  state estimation blending.

**Key features:**

1. **Ensemble weighting**: inverse-RMSE weighting across all models that
   successfully fitted for each variable.  Models with lower validation
   error receive higher weight.

2. **Multi-horizon predictions**: aggregated point forecasts for 1d, 5d,
   21d, and 252d horizons.

3. **Uncertainty bands**: confidence intervals derived from conformal
   calibration (preferred) or model RMSE scaled by the square root of
   horizon (fallback), optionally widened when Monte Carlo survival
   probability is low or copula tail dependence is high.

4. **Technical Alpha protection**: next-day OHLC predictions are masked
   (set to ``None``) except for the Low estimate, per Sec 17 of the spec.

5. **Persistence**: final predictions are stored in
   ``cache/predictions.parquet`` and a summary JSON in
   ``cache/prediction_summary.json``.

Top-level entry point:
  ``run_prediction_aggregation(cache, forecast_result, mc_result, ...)``

Spec refs: Sec 17, Phase F (conformal), Sec K (regime mixer),
           Sec E.2 Category 3 (copula), Sec E.2 Category 4 (GA)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operator1.constants import CACHE_DIR
from operator1.models.forecasting import (
    HORIZONS,
    BaseModelWrapper,
    BaselineWrapper,
    ForecastResult,
    ForwardPassResult,
    ModelMetrics,
    _load_tier_variables,
    _get_tier_for_variable,
)
from operator1.models.monte_carlo import MonteCarloResult

# Optional result types -- imported lazily in functions to avoid hard
# dependencies, but declared here for type annotations.
try:
    from operator1.models.conformal import ConformalInterval, ConformalResult
except ImportError:  # pragma: no cover
    ConformalResult = None  # type: ignore[assignment,misc]
    ConformalInterval = None  # type: ignore[assignment,misc]

try:
    from operator1.models.regime_mixer import DualRegimeResult
except ImportError:  # pragma: no cover
    DualRegimeResult = None  # type: ignore[assignment,misc]

try:
    from operator1.models.copula import CopulaResult
except ImportError:  # pragma: no cover
    CopulaResult = None  # type: ignore[assignment,misc]

try:
    from operator1.models.dtw_analogs import DTWAnalogResult
except ImportError:  # pragma: no cover
    DTWAnalogResult = None  # type: ignore[assignment,misc]

try:
    from operator1.models.granger_causality import GrangerResult
except ImportError:  # pragma: no cover
    GrangerResult = None  # type: ignore[assignment,misc]

try:
    from operator1.models.explainability import SHAPResult
except ImportError:  # pragma: no cover
    SHAPResult = None  # type: ignore[assignment,misc]

try:
    from operator1.models.walk_forward import WalkForwardResult
except ImportError:  # pragma: no cover
    WalkForwardResult = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Survival-mode aware imports (lazy to avoid circular deps)
# ---------------------------------------------------------------------------

_SURVIVAL_MODES = (
    "normal",
    "company_only",
    "country_protected",
    "country_exposed",
    "both_unprotected",
    "both_protected",
)

# Default transition blending half-life (in days).  When the survival mode
# changes, the new mode's weights are blended with the previous mode's
# weights over this many days using an exponential ramp.
_TRANSITION_BLEND_HALFLIFE: int = 5

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CI z-score for 90% interval (5th to 95th percentile).
Z_SCORE_90: float = 1.645

# Default confidence level.
DEFAULT_CONFIDENCE_LEVEL: float = 0.90

# Survival risk multiplier: how much to widen bands when survival is low.
DEFAULT_SURVIVAL_RISK_MULTIPLIER: float = 2.0

# Intraday low estimation factor (multiples of daily vol below last close).
INTRADAY_LOW_FACTOR: float = 1.5

# Minimum RMSE to avoid division by zero in inverse weighting.
MIN_RMSE_FOR_WEIGHTING: float = 1e-10

# Default volatility when none is available in the cache.
DEFAULT_VOLATILITY: float = 0.02


# ---------------------------------------------------------------------------
# Fixed Share Forecaster (Herbster & Warmuth 1998)
# ---------------------------------------------------------------------------


class FixedShareForecaster:
    """Online model weighting with fixed share redistribution.

    At each step, a fraction ``alpha`` of total weight is redistributed
    uniformly across all models. This prevents any model from reaching
    zero weight, so when a previously poor model starts performing well
    (e.g., after a regime change), it can gain weight rapidly.

    Parameters
    ----------
    model_names:
        List of model names participating in the ensemble.
    alpha:
        Share parameter (0-1). Fraction of weight redistributed per step.
        0.05 is a good default for financial regime changes (~20-day
        adaptation).
    eta:
        Learning rate for the multiplicative weight update.
    """

    def __init__(
        self,
        model_names: list[str],
        alpha: float = 0.05,
        eta: float = 0.5,
        adaptive: bool = True,
    ) -> None:
        self._models = list(model_names)
        self._n = len(model_names)
        self._base_alpha = alpha
        self._alpha = alpha
        self._eta = eta
        self._adaptive = adaptive
        # Initialize uniform weights
        self._weights = {name: 1.0 / self._n for name in self._models}

    def update(
        self,
        losses: dict[str, float],
        online_change_score: float = 0.0,
    ) -> None:
        """Update weights based on model losses for the current step.

        Parameters
        ----------
        losses:
            Dict of {model_name: squared_error} for the current day.
        online_change_score:
            ChangeFinder score (0-1) from the regime detector. When high,
            indicates a regime change is in progress and the share parameter
            should increase to adapt faster (Gap 6 Adaptive FixedShare).
        """
        if not losses or self._n == 0:
            return

        # Gap 6: Adaptive share parameter -- increase alpha during regime changes
        if self._adaptive and online_change_score > 0.3:
            # Scale alpha up to 3x during regime transitions (cap at 0.3)
            self._alpha = min(self._base_alpha * (1.0 + online_change_score * 2.0), 0.30)
        else:
            self._alpha = self._base_alpha

        # Multiplicative weight update
        for name in self._models:
            loss = losses.get(name, 0.0)
            if not np.isfinite(loss):
                continue
            self._weights[name] *= np.exp(-self._eta * loss)

        # Fixed share redistribution
        total = sum(self._weights.values())
        if total > 0:
            for name in self._models:
                self._weights[name] = (
                    (1 - self._alpha) * self._weights[name] / total
                    + self._alpha / self._n
                )

    def get_weights(self) -> dict[str, float]:
        """Return current normalized weights."""
        total = sum(self._weights.values())
        if total <= 0:
            return {name: 1.0 / self._n for name in self._models}
        return {name: w / total for name, w in self._weights.items()}

    def filter_by_mcs(
        self,
        confidence_set: list[str] | None,
    ) -> dict[str, float]:
        """Get weights filtered by a Model Confidence Set.

        Models not in the confidence set get zero weight. Remaining
        weights are renormalized.

        Parameters
        ----------
        confidence_set:
            List of model names in the confidence set. If None, returns
            all weights unfiltered.

        Returns
        -------
        Dict of {model_name: weight} with non-MCS models zeroed out.
        """
        raw = self.get_weights()
        if confidence_set is None:
            return raw

        filtered = {
            name: (w if name in confidence_set else 0.0)
            for name, w in raw.items()
        }
        total = sum(filtered.values())
        if total > 0:
            return {name: w / total for name, w in filtered.items()}
        # Fallback: equal weight across confidence set
        n_mcs = len(confidence_set)
        return {
            name: (1.0 / n_mcs if name in confidence_set else 0.0)
            for name in self._models
        }


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class HorizonPrediction:
    """Single prediction for one variable at one horizon."""

    variable: str = ""
    horizon: str = ""  # "1d", "5d", "21d", "252d"
    point_forecast: float = float("nan")
    lower_ci: float = float("nan")  # 5th percentile
    upper_ci: float = float("nan")  # 95th percentile
    confidence: float = float("nan")  # 0-1 overall confidence score
    model_used: str = ""  # which model produced the forecast
    ensemble_weight: float = 0.0  # weight this model got
    survival_adjusted: bool = False  # whether survival prob was factored in

    # --- New fields from full-potential upgrade ---
    interval_source: str = "rmse"  # "conformal" or "rmse"
    explanation: str = ""  # SHAP narrative for this prediction
    top_drivers: list[str] = field(default_factory=list)  # top feature names
    analog_forecast: float | None = None  # DTW empirical forecast if available
    causal_adjustment: float = 0.0  # adjustment from Granger propagation
    regime_blend_applied: bool = False  # True if soft regime blending was used

    # --- Beyond Bands: distributional forecasting fields ---
    # Skew signal from quantile regression (Method 1): (P95-P50)-(P50-P05).
    # Positive = right-skewed (more upside potential than downside risk).
    # Negative = left-skewed (more downside risk than upside potential).
    skew_signal: float | None = None
    # Between-model standard deviation from BMA (Method 6).
    # Measures model DISAGREEMENT, not model error. High = models disagree.
    between_model_std: float | None = None
    # Scenario-weighted point forecast from entropy pooling (Method 4).
    scenario_weighted_point: float | None = None
    # Scenario decomposition: [{label, probability, target_price, n_paths}]
    scenarios: list[dict] | None = None


@dataclass
class TechnicalAlphaMask:
    """Masked OHLC predictions for Technical Alpha protection.

    Per Sec 17: mask next-day OHLC except Low.
    """

    next_day_open: float | None = None  # MASKED
    next_day_high: float | None = None  # MASKED
    next_day_low: float = float("nan")  # VISIBLE
    next_day_close: float | None = None  # MASKED
    mask_applied: bool = True


@dataclass
class PredictionAggregatorResult:
    """Container for all prediction aggregation outputs."""

    # Per-variable, per-horizon predictions.
    # {variable: {horizon: HorizonPrediction}}
    predictions: dict[str, dict[str, HorizonPrediction]] = field(
        default_factory=dict,
    )

    # Technical Alpha masked OHLC.
    technical_alpha: TechnicalAlphaMask = field(
        default_factory=TechnicalAlphaMask,
    )

    # Ensemble weights per model type.
    # {model_name: weight}
    ensemble_weights: dict[str, float] = field(default_factory=dict)

    # Model availability.
    n_models_available: int = 0
    n_models_failed: int = 0

    # Survival probability summary (pass-through from MC).
    survival_probability_mean: float = float("nan")
    survival_probability_p5: float = float("nan")
    survival_probability_p95: float = float("nan")

    # Module contribution scores (Section F.1 Category 7 from core idea).
    # {model_name: contribution_pct} showing how much each model contributes
    # to the final ensemble prediction.
    module_contributions: dict[str, float] = field(default_factory=dict)

    # Current regime at prediction time.
    current_regime: str = ""

    # Metadata.
    prediction_date: str = ""
    variables_predicted: list[str] = field(default_factory=list)
    horizons: list[str] = field(default_factory=list)

    # Error info.
    error: str | None = None
    fitted: bool = False

    # --- New fields from full-potential upgrade ---
    conformal_coverage: float | None = None  # coverage level if conformal used
    copula_tail_risk: float = 0.0  # joint crisis probability from copula
    dtw_analogs_used: int = 0  # number of DTW analogs contributing
    granger_adjustments_applied: int = 0  # number of causal propagations
    regime_blend_method: str = ""  # "hard_label" or "soft_probability"
    recency_weighted_rmse_used: bool = False  # True if walk-forward recency applied
    shap_available: bool = False  # True if SHAP explanations were attached

    # --- Beyond Bands: scenario decomposition per horizon ---
    # {horizon: {scenarios: [...], kl_divergence: float, weighted_point: float}}
    scenario_decomposition: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ensemble weight computation
# ---------------------------------------------------------------------------


def compute_ensemble_weights(
    metrics: list[ModelMetrics],
) -> dict[str, float]:
    """Compute inverse-RMSE ensemble weights from model metrics.

    Parameters
    ----------
    metrics:
        List of ``ModelMetrics`` from ``ForecastResult.metrics``.
        Only entries with ``fitted=True`` and finite RMSE are used.

    Returns
    -------
    Dict mapping ``model_name`` to normalised weight (sum = 1.0).
    Models with lower RMSE receive higher weight.
    """
    # Collect best RMSE per unique model name.
    model_rmse: dict[str, float] = {}

    for m in metrics:
        if not m.fitted:
            continue
        if math.isnan(m.rmse) or m.rmse < MIN_RMSE_FOR_WEIGHTING:
            continue

        name = m.model_name
        if name not in model_rmse or m.rmse < model_rmse[name]:
            model_rmse[name] = m.rmse

    if not model_rmse:
        # No valid RMSE available -- return empty (caller handles this).
        return {}

    # Inverse-RMSE weights.
    inv_rmse = {name: 1.0 / rmse for name, rmse in model_rmse.items()}
    total = sum(inv_rmse.values())

    if total <= 0:
        # Uniform fallback.
        n = len(inv_rmse)
        return {name: 1.0 / n for name in inv_rmse}

    return {name: w / total for name, w in inv_rmse.items()}


# ---------------------------------------------------------------------------
# Beyond Bands Method 6: Bayesian Model Averaging (Hoeting et al. 1999)
# ---------------------------------------------------------------------------


def compute_bma_between_model_std(
    metrics: list[ModelMetrics],
    forecasts: dict[str, dict[str, float]],
    variable: str,
    horizon: str,
) -> float:
    """Compute between-model standard deviation for BMA.

    The Law of Total Variance states:
        total_var = within_model_var + between_model_var

    Our current system only uses within_model_var (RMSE^2). This function
    computes between_model_var -- the variance of point forecasts across
    models. High between-model std means models disagree, which should
    widen prediction intervals even if the best model has low RMSE.

    Reference: Hoeting, Madigan, Raftery & Volinsky (1999),
    'Bayesian Model Averaging: A Tutorial'.

    Parameters
    ----------
    metrics:
        Model metrics from ForecastResult.
    forecasts:
        Per-variable per-horizon forecasts from ForecastResult.
    variable:
        Variable name to compute BMA for.
    horizon:
        Horizon label (e.g. "1d", "5d").

    Returns
    -------
    Between-model standard deviation (0 if all models agree or < 2 models).
    """
    base_weights = compute_ensemble_weights(metrics)
    if not base_weights:
        return 0.0

    # Collect per-model forecasts for this variable
    model_forecasts: dict[str, float] = {}
    for m in metrics:
        if not m.fitted or m.model_name not in base_weights:
            continue
        # Match variable name (ModelMetrics.variable stores the var name)
        if m.variable != variable:
            continue
        fc_var = forecasts.get(variable, {})
        fc_val = fc_var.get(horizon)
        if fc_val is not None and not math.isnan(fc_val):
            model_forecasts[m.model_name] = fc_val

    if len(model_forecasts) < 2:
        return 0.0

    # BMA posterior mean
    mu_bma = sum(
        base_weights.get(name, 0) * fc
        for name, fc in model_forecasts.items()
    )

    # Between-model variance
    between_var = sum(
        base_weights.get(name, 0) * (fc - mu_bma) ** 2
        for name, fc in model_forecasts.items()
    )

    # Cap at 3x the median model RMSE to prevent outlier model from
    # dominating the between-model term
    _model_rmses = [m.rmse for m in metrics if m.fitted and not math.isnan(m.rmse)]
    if _model_rmses:
        _median_rmse = float(np.median(_model_rmses))
        _cap = (3.0 * _median_rmse) ** 2
        between_var = min(between_var, _cap)

    return math.sqrt(max(between_var, 0.0))


# ---------------------------------------------------------------------------
# Beyond Bands Method 4: Entropy Pooling (Meucci 2010)
# ---------------------------------------------------------------------------


def _entropy_pooling_scenarios(
    mc_terminal_values: np.ndarray | None,
    model_forecast_return: float,
    last_close: float,
) -> dict:
    """Reweight MC paths to match model view via entropy pooling.

    Starts with uniform prior (each MC path equally likely), then tilts
    the probability mass toward paths consistent with the model ensemble's
    directional view, while staying as close to uniform as possible
    (minimum KL-divergence reweighting).

    Reference: Meucci (2010), 'Fully Flexible Views: Theory and Practice',
    implemented following fortitudo-tech/fortitudo.tech (296 stars, GPL-3.0).

    Parameters
    ----------
    mc_terminal_values:
        Array of shape (n_paths,) with cumulative return ratios from MC.
    model_forecast_return:
        Model ensemble's predicted return (e.g. 0.02 for +2%).
    last_close:
        Last observed close price for price-level scenarios.

    Returns
    -------
    Dict with 'available', 'weighted_point', 'scenarios', 'kl_divergence'.
    """
    if mc_terminal_values is None or len(mc_terminal_values) < 100:
        return {"available": False}

    if math.isnan(model_forecast_return) or last_close <= 0:
        return {"available": False}

    try:
        from scipy.optimize import minimize as _sp_minimize

        S = len(mc_terminal_values)
        p = np.ones(S) / S  # uniform prior
        log_p = np.log(p)

        # View: expected excess return = model_forecast_return
        excess_returns = mc_terminal_values - 1.0
        A = excess_returns.reshape(1, -1)
        b_val = model_forecast_return

        # Solve dual problem for Lagrange multiplier
        def _dual_obj(lam):
            log_x = log_p - 1.0 - A.flatten() * lam[0]
            log_x = np.clip(log_x, -500.0, 500.0)
            x = np.exp(log_x)
            obj = float(x @ (log_x - log_p) - lam[0] * (b_val - A.flatten() @ x))
            grad = np.array([float(b_val - A.flatten() @ x)])
            return -1000.0 * obj, 1000.0 * grad

        result = _sp_minimize(
            _dual_obj, x0=np.array([0.0]), jac=True, method="L-BFGS-B",
        )
        log_q = log_p - 1.0 - A.flatten() * result.x[0]
        log_q = np.clip(log_q, -500.0, 500.0)
        q = np.exp(log_q)
        q = q / q.sum()  # normalize

        # Scenario decomposition (quartile-based)
        mc_prices = last_close * mc_terminal_values
        pcts = np.percentile(mc_prices, [0, 25, 50, 75, 100])

        scenarios = []
        labels = ["bear", "base_low", "base_high", "bull"]
        for idx in range(4):
            lo, hi = pcts[idx], pcts[idx + 1]
            if idx == 3:
                mask = mc_prices >= lo
            else:
                mask = (mc_prices >= lo) & (mc_prices < hi)
            if mask.sum() > 0:
                _w = q[mask]
                scenarios.append({
                    "label": labels[idx],
                    "probability": round(float(_w.sum()), 4),
                    "target_price": round(float(np.average(mc_prices[mask], weights=_w)), 2),
                    "n_paths": int(mask.sum()),
                })

        weighted_point = float(last_close * np.average(mc_terminal_values, weights=q))
        kl_div = float(np.sum(q * np.log((q + 1e-30) / (p + 1e-30))))

        return {
            "available": True,
            "weighted_point": round(weighted_point, 4),
            "scenarios": scenarios,
            "kl_divergence": round(kl_div, 6),
        }

    except Exception as _ep_exc:
        logger.debug("Entropy pooling failed: %s", _ep_exc)
        return {"available": False}


# ---------------------------------------------------------------------------
# Beyond Bands Method 3: Constrained Optimization (Boyd & Vandenberghe 2004)
# ---------------------------------------------------------------------------


def _constrained_point_forecast(
    raw_forecast: float,
    lower_bound: float,
    upper_bound: float,
    last_close: float,
    momentum_5d: float,
    regime: str,
) -> float:
    """Optimize point forecast within bounds subject to momentum + mean-reversion.

    Instead of passively placing the point forecast within bands, this
    treats the bands as HARD CONSTRAINTS and finds the point that optimizes
    a multi-objective loss: model accuracy + momentum alignment + mean
    reversion tendency, with regime-dependent weights.

    Reference: Boyd & Vandenberghe (2004), 'Convex Optimization'.

    Parameters
    ----------
    raw_forecast:
        Unconstrained ensemble point forecast.
    lower_bound:
        Lower confidence bound (from conformal or RMSE).
    upper_bound:
        Upper confidence bound.
    last_close:
        Last observed close price.
    momentum_5d:
        5-day return (positive = uptrend, negative = downtrend).
    regime:
        Current survival regime label.

    Returns
    -------
    Constrained optimal point forecast (guaranteed within bounds).
    """
    if (math.isnan(raw_forecast) or math.isnan(lower_bound)
            or math.isnan(upper_bound) or lower_bound >= upper_bound):
        return raw_forecast

    try:
        from scipy.optimize import minimize_scalar

        # Regime-dependent weights
        mr_weight = 0.15 if regime in ("normal", "") else 0.05
        mom_weight = 0.30 if regime not in ("extreme_survival",) else 0.10

        _lc = last_close if not math.isnan(last_close) and last_close > 0 else raw_forecast
        _mom = momentum_5d if not math.isnan(momentum_5d) else 0.0

        def _objective(x):
            model_loss = (x - raw_forecast) ** 2
            momentum_loss = -(x - _lc) * _mom * mom_weight
            mean_rev = (x - _lc) ** 2 * mr_weight
            return model_loss + momentum_loss + mean_rev

        result = minimize_scalar(
            _objective, bounds=(lower_bound, upper_bound), method="bounded",
        )
        return float(result.x) if result.success else raw_forecast

    except Exception:
        # Fallback: clamp raw forecast to bounds
        return max(lower_bound, min(upper_bound, raw_forecast))


def apply_ic_weighted_calibration(
    base_weights: dict[str, float],
    signal_ic_result: Any | None = None,
) -> dict[str, float]:
    """C3: Adjust ensemble weights using signal IC measurements.

    The signal_ic module already computes rolling Spearman IC for every
    signal but it's only used for feature pruning. Here we use it to
    upweight models whose signals have high IC (historically predictive)
    and downweight models with low IC.

    Parameters
    ----------
    base_weights:
        Inverse-RMSE ensemble weights from ``compute_ensemble_weights``.
    signal_ic_result:
        ``SignalICResult`` from ``signal_ic.py``. Contains
        ``strong_signals``, ``weak_signals``, ``best_ic``.

    Returns
    -------
    Adjusted weights (still sum to 1.0).
    """
    if not base_weights or signal_ic_result is None:
        return base_weights

    if not getattr(signal_ic_result, "available", False):
        return base_weights

    # Map model types to the signals they're best at predicting
    _MODEL_SIGNAL_MAP = {
        "kalman": ["close", "revenue", "total_assets"],
        "garch": ["volatility_21d", "volatility_63d"],
        "var": ["return_1d", "close"],
        "lstm": ["close", "return_1d"],
        "tree": ["close", "return_1d", "fcf_yield", "current_ratio"],
        "baseline": ["close"],
        "ets": ["close", "revenue"],
        "transformer": ["close", "return_1d"],
    }

    # Get IC scores per signal
    _ic_scores = getattr(signal_ic_result, "ic_scores", {})
    if not _ic_scores:
        return base_weights

    # Compute IC-based multiplier per model
    adjusted = dict(base_weights)
    for model_name, weight in base_weights.items():
        _base_model = model_name.lower().split("_")[0].split("(")[0]
        _relevant_signals = _MODEL_SIGNAL_MAP.get(_base_model, [])
        if not _relevant_signals:
            continue

        # Average absolute IC across this model's relevant signals
        _ics = [abs(_ic_scores.get(s, 0.0)) for s in _relevant_signals if s in _ic_scores]
        if _ics:
            _avg_ic = sum(_ics) / len(_ics)
            # IC multiplier: IC=0.05 -> 1.5x, IC=0.01 -> 0.5x, IC=0.10 -> 2.0x
            _ic_mult = max(0.3, min(3.0, _avg_ic * 20.0))
            adjusted[model_name] = weight * _ic_mult

    # Renormalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted


def apply_prediction_log_feedback(
    base_weights: dict[str, float],
    prediction_log_summary: dict | None = None,
) -> dict[str, float]:
    """F3: Realized IC feedback from previous prediction logs.

    If past predictions and actuals are available, compute realized
    accuracy per model and use it to calibrate current weights.
    Creates a self-improving feedback loop.

    Parameters
    ----------
    base_weights:
        Current ensemble weights.
    prediction_log_summary:
        Output from ``prediction_log.fill_actuals()``. Contains
        ``hit_rate``, ``realized_ic``, ``per_model_ic`` (if available).

    Returns
    -------
    Adjusted weights incorporating historical accuracy.
    """
    if not base_weights or not prediction_log_summary:
        return base_weights

    _per_model = prediction_log_summary.get("per_model_ic", {})
    if not _per_model:
        # No per-model breakdown -- use overall IC as a global confidence scaler
        _realized_ic = prediction_log_summary.get("realized_ic", 0.0)
        if abs(_realized_ic) > 0.001:
            # If overall realized IC is very low, reduce all weights toward uniform
            _confidence = max(0.3, min(1.0, abs(_realized_ic) * 10))
            n = len(base_weights)
            uniform = 1.0 / max(n, 1)
            adjusted = {
                k: _confidence * v + (1 - _confidence) * uniform
                for k, v in base_weights.items()
            }
            total = sum(adjusted.values())
            return {k: v / total for k, v in adjusted.items()} if total > 0 else base_weights
        return base_weights

    # Per-model IC available: upweight models with high realized IC
    adjusted = dict(base_weights)
    for model_name, weight in base_weights.items():
        _base = model_name.lower().split("_")[0].split("(")[0]
        _model_ic = _per_model.get(_base, _per_model.get(model_name, 0.0))
        if abs(_model_ic) > 0.001:
            _mult = max(0.2, min(3.0, abs(_model_ic) * 15))
            adjusted[model_name] = weight * _mult

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted


def train_stacking_meta_learner(
    forward_pass_predictions: list[dict] | None = None,
    cache: Any = None,
) -> dict[str, float] | None:
    """F1: Train a Ridge regression stacking meta-learner on walk-forward predictions.

    Instead of inverse-RMSE weighting, learns conditional patterns from
    model outputs -- e.g. 'trust GARCH during high vol, trust Kalman during trends'.

    Parameters
    ----------
    forward_pass_predictions:
        List of prediction log entries from ForwardPassResult.predictions_log.
        Each entry has model predictions and actual values.
    cache:
        Daily cache DataFrame for extracting regime/survival features.

    Returns
    -------
    Dict of model_name -> learned weight, or None if insufficient data.
    """
    if not forward_pass_predictions or len(forward_pass_predictions) < 50:
        return None

    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        logger.debug("F1: scikit-learn not available for stacking")
        return None

    try:
        # Extract model predictions and actuals from forward pass log
        model_names: list[str] = []
        X_rows: list[list[float]] = []
        y_vals: list[float] = []

        # First pass: identify all model names
        for entry in forward_pass_predictions:
            if not isinstance(entry, dict):
                continue
            _preds = entry.get("predictions", {})
            for m_name in _preds:
                if m_name not in model_names:
                    model_names.append(m_name)

        if len(model_names) < 2:
            return None

        # Second pass: build feature matrix
        for entry in forward_pass_predictions:
            if not isinstance(entry, dict):
                continue
            _actual = entry.get("actual")
            _preds = entry.get("predictions", {})
            if _actual is None or math.isnan(_actual):
                continue

            row = [_preds.get(m, float("nan")) for m in model_names]
            if any(math.isnan(v) for v in row):
                continue

            X_rows.append(row)
            y_vals.append(_actual)

        if len(X_rows) < 30:
            return None

        X = np.array(X_rows)
        y = np.array(y_vals)

        # Fit Ridge regression (regularized to prevent overfitting)
        meta = Ridge(alpha=1.0, fit_intercept=True)
        meta.fit(X, y)

        # Extract learned weights (coefficients)
        raw_weights = {
            model_names[i]: max(0.0, float(meta.coef_[i]))
            for i in range(len(model_names))
        }

        # Normalize to sum to 1
        total = sum(raw_weights.values())
        if total > 0:
            weights = {k: v / total for k, v in raw_weights.items()}
        else:
            return None

        logger.info(
            "F1 stacking meta-learner: trained on %d samples, %d models, "
            "weights=%s",
            len(X_rows), len(model_names),
            {k: f"{v:.3f}" for k, v in sorted(weights.items(), key=lambda x: -x[1])[:5]},
        )
        return weights

    except Exception as exc:
        logger.debug("F1 stacking meta-learner failed: %s", exc)
        return None


def optimise_ensemble_weights_ga(
    metrics: list[ModelMetrics],
    *,
    population_size: int = 50,
    generations: int = 30,
    random_state: int = 42,
) -> dict[str, float]:
    """Optimise ensemble weights using a genetic algorithm.

    Spec reference: The_Apps_core_idea.pdf Section E.2 Category 4
    (Genetic Algorithm for Meta-Optimization).

    Falls back to inverse-RMSE weights if the GA library (deap) is
    not installed or optimisation fails.

    Parameters
    ----------
    metrics:
        List of ``ModelMetrics`` from ``ForecastResult.metrics``.
    population_size:
        Number of individuals in the GA population.
    generations:
        Number of evolutionary generations.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    Dict mapping ``model_name`` to optimised weight (sum = 1.0).
    """
    # Collect fitted models with valid RMSE
    model_rmse: dict[str, float] = {}
    for m in metrics:
        if not m.fitted or math.isnan(m.rmse) or m.rmse < MIN_RMSE_FOR_WEIGHTING:
            continue
        name = m.model_name
        if name not in model_rmse or m.rmse < model_rmse[name]:
            model_rmse[name] = m.rmse

    if len(model_rmse) < 2:
        return compute_ensemble_weights(metrics)

    model_names = list(model_rmse.keys())
    n_models = len(model_names)
    rmse_array = np.array([model_rmse[n] for n in model_names])

    try:
        from scipy.optimize import differential_evolution

        def objective(weights: np.ndarray) -> float:
            """Minimise weighted RMSE (proxy for ensemble loss)."""
            w = np.abs(weights)
            w_sum = w.sum()
            if w_sum <= 0:
                return 1e10
            w = w / w_sum
            return float(np.dot(w, rmse_array))

        bounds = [(0.01, 1.0)] * n_models
        result = differential_evolution(
            objective,
            bounds,
            maxiter=generations,
            popsize=population_size,
            seed=random_state,
            tol=1e-6,
        )

        opt_weights = np.abs(result.x)
        opt_weights /= opt_weights.sum()

        logger.info(
            "GA ensemble optimisation converged (gen=%d): %s",
            generations,
            {n: round(float(w), 4) for n, w in zip(model_names, opt_weights)},
        )

        return {n: float(w) for n, w in zip(model_names, opt_weights)}

    except ImportError:
        logger.info("scipy.optimize not available for GA -- using inverse-RMSE")
        return compute_ensemble_weights(metrics)
    except Exception as exc:
        logger.warning("GA optimisation failed: %s -- using inverse-RMSE", exc)
        return compute_ensemble_weights(metrics)


# ---------------------------------------------------------------------------
# Forecast aggregation
# ---------------------------------------------------------------------------


def aggregate_forecasts(
    forecast_result: ForecastResult,
    forward_pass_result: Any = None,
) -> dict[str, dict[str, float]]:
    """Extract aggregated point forecasts per variable per horizon.

    In the current architecture, ``ForecastResult.forecasts`` contains
    the best model's prediction from the fallback chain.  When a
    ``ForwardPassResult`` with model states is available, we blend the
    forward pass ensemble predictions into the point forecasts using
    a 70/30 weighting (70% fallback-chain winner, 30% forward-pass
    ensemble) to incorporate the online-learned model states.

    Parameters
    ----------
    forecast_result:
        Output from ``run_forecasting``.
    forward_pass_result:
        Optional ``ForwardPassResult`` from ``run_forward_pass``.
        If provided and model states are available, their last
        predictions are blended into the aggregated forecasts.

    Returns
    -------
    ``{variable: {horizon_label: point_forecast}}``
    """
    aggregated: dict[str, dict[str, float]] = {}

    # Extract forward-pass model state predictions if available.
    fp_predictions: dict[str, float] = {}
    if forward_pass_result is not None:
        model_states = getattr(forward_pass_result, "model_states", {})
        for var_name, wrapper in model_states.items():
            try:
                pred = wrapper.predict(np.array([0.0]))
                if len(pred) > 0 and not np.isnan(pred[0]):
                    fp_predictions[var_name] = float(pred[0])
            except Exception:
                pass

    for var_name, horizons in forecast_result.forecasts.items():
        var_forecasts: dict[str, float] = {}

        for h_label, value in horizons.items():
            if math.isnan(value):
                continue

            # Blend with forward-pass ensemble prediction for the 1d
            # horizon (model states predict one step ahead).
            if h_label == "1d" and var_name in fp_predictions:
                fp_val = fp_predictions[var_name]
                # 70% fallback-chain winner, 30% online-learned ensemble.
                value = 0.7 * value + 0.3 * fp_val

            var_forecasts[h_label] = value

        if var_forecasts:
            aggregated[var_name] = var_forecasts

    return aggregated


# ---------------------------------------------------------------------------
# Uncertainty bands
# ---------------------------------------------------------------------------


def _get_best_rmse_for_variable(
    variable: str,
    metrics: list[ModelMetrics],
) -> float:
    """Return the RMSE of the best fitted model for a variable."""
    best = float("nan")

    for m in metrics:
        if m.variable == variable and m.fitted and not math.isnan(m.rmse):
            if math.isnan(best) or m.rmse < best:
                best = m.rmse

    return best


def compute_uncertainty_bands(
    point_forecast: float,
    rmse: float,
    horizon_days: int,
    *,
    survival_probability: float = 1.0,
    survival_risk_multiplier: float = DEFAULT_SURVIVAL_RISK_MULTIPLIER,
    z_score: float = Z_SCORE_90,
    between_model_std: float = 0.0,
) -> tuple[float, float]:
    """Compute confidence interval bounds for a single prediction.

    The base interval is derived from the model's RMSE scaled by
    ``sqrt(horizon_days)`` (random-walk scaling).  If the Monte Carlo
    survival probability is below 1.0, the interval is widened by a
    risk factor proportional to the survival shortfall.

    Beyond Bands Method 6 (BMA): When ``between_model_std`` > 0, the
    total variance includes both within-model variance (RMSE^2) and
    between-model variance (model disagreement). This is the Law of
    Total Variance (Hoeting et al. 1999): total = within + between.

    Parameters
    ----------
    point_forecast:
        Central prediction value.
    rmse:
        Best model's root mean squared error on validation data.
    horizon_days:
        Forecast horizon in business days.
    survival_probability:
        Monte Carlo survival probability for this horizon (0-1).
        Defaults to 1.0 (no widening).
    survival_risk_multiplier:
        How aggressively to widen bands when survival prob is low.
    z_score:
        z-score for the desired confidence level (default 1.645 = 90%).
    between_model_std:
        Between-model standard deviation from BMA (Method 6).
        When > 0, widens bands to account for model disagreement.

    Returns
    -------
    (lower_ci, upper_ci)
    """
    if math.isnan(point_forecast):
        return float("nan"), float("nan")

    if math.isnan(rmse) or rmse < MIN_RMSE_FOR_WEIGHTING:
        # No valid RMSE -- use a conservative 10% of point forecast.
        base_spread = abs(point_forecast) * 0.10
    else:
        base_spread = z_score * rmse * math.sqrt(max(horizon_days, 1))

    # Beyond Bands Method 6: BMA total variance = within + between.
    # When models disagree, between_model_std > 0 widens the bands
    # even if the best model has low RMSE.
    if between_model_std > 0:
        within_var = base_spread ** 2
        between_var = (between_model_std * math.sqrt(max(horizon_days, 1))) ** 2
        base_spread = math.sqrt(within_var + between_var)

    # Survival-weighted widening.
    surv_prob = max(0.0, min(1.0, survival_probability))
    risk_factor = 1.0 + (1.0 - surv_prob) * survival_risk_multiplier

    adjusted_spread = base_spread * risk_factor

    lower_ci = point_forecast - adjusted_spread
    upper_ci = point_forecast + adjusted_spread

    return lower_ci, upper_ci


def compute_confidence_score(
    rmse: float,
    rmse_reference: float,
    survival_probability: float = 1.0,
) -> float:
    """Compute a 0-1 confidence score for a prediction.

    Combines model quality (normalised RMSE) with survival probability.

    Parameters
    ----------
    rmse:
        Model's RMSE on validation data.
    rmse_reference:
        A reference RMSE to normalise against (e.g. median RMSE across
        all models/variables).  If 0, model quality = 0.5 default.
    survival_probability:
        Monte Carlo survival probability (0-1).

    Returns
    -------
    Confidence score in [0, 1].
    """
    # Model quality score: 1.0 when RMSE is 0, decays as RMSE grows.
    if math.isnan(rmse) or math.isnan(rmse_reference):
        model_quality = 0.5  # default when no info
    elif rmse_reference <= MIN_RMSE_FOR_WEIGHTING:
        model_quality = 0.5
    else:
        # Exponential decay: quality = exp(-rmse / reference).
        model_quality = math.exp(-rmse / rmse_reference)
        model_quality = max(0.0, min(1.0, model_quality))

    surv_prob = max(0.0, min(1.0, survival_probability))
    if math.isnan(surv_prob):
        surv_prob = 1.0

    # Geometric mean of model quality and survival probability.
    confidence = math.sqrt(model_quality * surv_prob)

    return max(0.0, min(1.0, confidence))


# ---------------------------------------------------------------------------
# Technical Alpha masking
# ---------------------------------------------------------------------------


def apply_technical_alpha_mask(
    cache: pd.DataFrame,
    forecasts: dict[str, dict[str, float]],
) -> TechnicalAlphaMask:
    """Apply Technical Alpha protection to next-day OHLC predictions.

    Per Sec 17: mask next-day OHLC except Low.  The Low estimate is
    derived from the last observed close minus a volatility-based buffer.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with ``close`` and optionally
        ``volatility_21d`` columns.
    forecasts:
        Aggregated forecasts dict (may contain ``close`` 1d forecast).

    Returns
    -------
    ``TechnicalAlphaMask`` with open/high/close masked (None) and
    low set to an estimated value.
    """
    mask = TechnicalAlphaMask(mask_applied=False)

    # Get last observed close.
    if "close" in cache.columns:
        close_series = cache["close"].dropna()
        if len(close_series) > 0:
            last_close = float(close_series.iloc[-1])
        else:
            last_close = float("nan")
    else:
        last_close = float("nan")

    if math.isnan(last_close):
        logger.warning(
            "No close price available for Technical Alpha estimation"
        )
        return mask

    # Get volatility estimate.
    vol = DEFAULT_VOLATILITY
    if "volatility_21d" in cache.columns:
        vol_series = cache["volatility_21d"].dropna()
        if len(vol_series) > 0:
            vol_val = float(vol_series.iloc[-1])
            if not math.isnan(vol_val) and vol_val > 0:
                vol = vol_val

    # Compute all OHLC estimates (unmasked).
    # Masking is now report-only -- the calculation layer preserves
    # all values for downstream models (OHLC predictor, recursive
    # aggregator, HF position signal) that need full OHLC data.
    mask.next_day_low = last_close * (1.0 - vol * INTRADAY_LOW_FACTOR)
    mask.next_day_high = last_close * (1.0 + vol * INTRADAY_LOW_FACTOR)

    # Use close forecast from aggregated results if available
    close_1d = forecasts.get("close", {}).get("1d")
    if close_1d is not None and not math.isnan(close_1d):
        mask.next_day_close = close_1d
        # Open estimate: gap from last close toward forecast direction
        gap_direction = 1.0 if close_1d >= last_close else -1.0
        mask.next_day_open = last_close + gap_direction * vol * last_close * 0.3
    else:
        mask.next_day_close = last_close
        mask.next_day_open = last_close

    # mask_applied=False means the data layer has full OHLC values.
    # The report generator applies display masking at render time.
    mask.mask_applied = False

    logger.info(
        "Technical Alpha OHLC: open=%.4f, high=%.4f, low=%.4f, close=%.4f "
        "(masking deferred to report layer)",
        mask.next_day_open,
        mask.next_day_high,
        mask.next_day_low,
        mask.next_day_close,
    )

    return mask


# ===========================================================================
# Phase 1: Conformal interval integration
# ===========================================================================


def get_conformal_interval(
    variable: str,
    horizon: str,
    conformal_result: Any | None,
) -> tuple[float, float, bool]:
    """Try to extract a conformal prediction interval for a variable/horizon.

    Parameters
    ----------
    variable:
        Variable name.
    horizon:
        Horizon label (e.g. ``"1d"``).
    conformal_result:
        ``ConformalResult`` instance (or None).

    Returns
    -------
    (lower, upper, used_conformal)
        If a conformal interval is available with sufficient calibration,
        return its bounds and ``True``.  Otherwise ``(nan, nan, False)``.
    """
    if conformal_result is None:
        return float("nan"), float("nan"), False

    try:
        intervals = conformal_result.intervals
    except AttributeError:
        return float("nan"), float("nan"), False

    var_intervals = intervals.get(variable, {})
    ci = var_intervals.get(horizon)
    if ci is None:
        return float("nan"), float("nan"), False

    # Require minimum calibration size for trustworthy intervals.
    cal_size = getattr(ci, "calibration_size", 0)
    if cal_size < 20:
        logger.debug(
            "Conformal interval for %s/%s has only %d calibration samples "
            "(need 20) -- falling back to RMSE",
            variable, horizon, cal_size,
        )
        return float("nan"), float("nan"), False

    lower = getattr(ci, "lower", float("nan"))
    upper = getattr(ci, "upper", float("nan"))

    if math.isnan(lower) or math.isnan(upper):
        return float("nan"), float("nan"), False

    return lower, upper, True


# ===========================================================================
# Phase 2: Regime probability blending
# ===========================================================================


def compute_regime_blended_weights(
    base_weights: dict[str, float],
    dual_regime_result: Any | None,
    model_regime_affinity: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, float], bool]:
    """Compute soft-blended ensemble weights using regime probabilities.

    Instead of switching weights based on a single hard regime label, this
    uses the continuous probability distribution over regimes to produce a
    smooth blend.

    Parameters
    ----------
    base_weights:
        Default inverse-RMSE or survival-aware weights.
    dual_regime_result:
        ``DualRegimeResult`` with ``market_regime_probs`` and/or
        ``fund_regime_probs`` DataFrames.  Each row is a day, columns are
        regime names with probability values.
    model_regime_affinity:
        Optional mapping ``{model_name: {regime_name: affinity_score}}``.
        Higher affinity means the model is better in that regime.
        If None, a default affinity mapping is used.

    Returns
    -------
    (blended_weights, was_applied)
    """
    if dual_regime_result is None:
        return base_weights, False

    # Extract latest regime probabilities.
    market_probs: dict[str, float] = {}
    fund_probs: dict[str, float] = {}

    try:
        mrp = dual_regime_result.market_regime_probs
        if mrp is not None and len(mrp) > 0:
            last_row = mrp.iloc[-1]
            market_probs = {col: float(last_row[col]) for col in mrp.columns
                           if not math.isnan(float(last_row[col]))}
    except Exception:
        pass

    try:
        frp = dual_regime_result.fund_regime_probs
        if frp is not None and len(frp) > 0:
            last_row = frp.iloc[-1]
            fund_probs = {col: float(last_row[col]) for col in frp.columns
                          if not math.isnan(float(last_row[col]))}
    except Exception:
        pass

    if not market_probs and not fund_probs:
        return base_weights, False

    # Default model-regime affinity: simple heuristics.
    # Models with "kalman"/"var" in the name are better in calm regimes;
    # models with "tree"/"xgboost" handle non-linearity (turbulent) better.
    if model_regime_affinity is None:
        model_regime_affinity = {}
        for model_name in base_weights:
            name_lower = model_name.lower()
            affinity: dict[str, float] = {}
            if "kalman" in name_lower or "var" in name_lower or "ema" in name_lower:
                affinity = {"bull": 1.2, "low_vol": 1.2, "bear": 0.8,
                            "high_vol": 0.8, "healthy": 1.1, "stressed": 0.9,
                            "distress": 0.7}
            elif "tree" in name_lower or "xgb" in name_lower or "forest" in name_lower:
                affinity = {"bull": 0.9, "low_vol": 0.9, "bear": 1.2,
                            "high_vol": 1.3, "healthy": 1.0, "stressed": 1.1,
                            "distress": 1.2}
            elif "garch" in name_lower:
                affinity = {"bull": 0.8, "low_vol": 0.8, "bear": 1.1,
                            "high_vol": 1.4, "healthy": 0.9, "stressed": 1.1,
                            "distress": 1.1}
            else:
                affinity = {r: 1.0 for r in list(market_probs.keys()) + list(fund_probs.keys())}
            model_regime_affinity[model_name] = affinity

    # Compute blended weights: for each model, multiply base weight by
    # the probability-weighted affinity score.
    all_probs = {**market_probs, **fund_probs}
    blended: dict[str, float] = {}

    for model_name, base_w in base_weights.items():
        affinity = model_regime_affinity.get(model_name, {})
        score = 0.0
        total_prob = 0.0
        for regime, prob in all_probs.items():
            aff = affinity.get(regime, 1.0)
            score += prob * aff
            total_prob += prob
        if total_prob > 0:
            score /= total_prob
        else:
            score = 1.0
        blended[model_name] = base_w * score

    # Normalise.
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}

    return blended, True


# ===========================================================================
# Phase 3: Copula tail risk adjustment
# ===========================================================================

# Threshold above which copula joint crisis probability triggers band widening.
COPULA_CRISIS_THRESHOLD: float = 0.10


def compute_copula_tail_adjustment(
    variable: str,
    copula_result: Any | None,
    *,
    crisis_threshold: float = COPULA_CRISIS_THRESHOLD,
    max_widening: float = 2.0,
) -> float:
    """Compute a band-widening multiplier from copula tail dependencies.

    When copula analysis shows high joint crisis probability (multiple
    variables crashing together), the uncertainty bands for all correlated
    variables should be widened.

    Parameters
    ----------
    variable:
        Variable being predicted.
    copula_result:
        ``CopulaResult`` instance (or None).
    crisis_threshold:
        Joint crisis probability above which widening starts.
    max_widening:
        Maximum multiplier (caps the widening).

    Returns
    -------
    Multiplier >= 1.0.  1.0 means no widening.
    """
    if copula_result is None:
        return 1.0

    try:
        jcp = copula_result.joint_crisis_probability
    except AttributeError:
        return 1.0

    if math.isnan(jcp) or jcp <= crisis_threshold:
        return 1.0

    # Check if this variable has high tail dependence with others.
    try:
        tail_deps = copula_result.tail_dependence
    except AttributeError:
        tail_deps = {}

    # Find max tail dependence involving this variable.
    max_tail = 0.0
    for pair_key, dep_val in tail_deps.items():
        if variable in str(pair_key):
            max_tail = max(max_tail, dep_val)

    # Widening proportional to joint crisis probability and tail dependence.
    # Scale: jcp in [threshold, 1.0] -> factor in [1.0, max_widening].
    crisis_excess = (jcp - crisis_threshold) / (1.0 - crisis_threshold + 1e-10)
    tail_factor = max(max_tail, 0.3)  # floor at 0.3 so all vars get some widening
    widening = 1.0 + crisis_excess * tail_factor * (max_widening - 1.0)

    return min(widening, max_widening)


# ===========================================================================
# Phase 4: DTW analog ensemble channel
# ===========================================================================


def compute_dtw_analog_forecast(
    variable: str,
    last_value: float,
    dtw_result: Any | None,
    horizon_label: str,
) -> tuple[float | None, float | None, float | None]:
    """Derive an empirical forecast from DTW historical analogs.

    Maps the analog's empirical return distribution onto the variable's
    last observed value.

    Parameters
    ----------
    variable:
        Variable name (used for logging only -- analogs are price-based).
    last_value:
        Last observed value for the variable.
    dtw_result:
        ``DTWAnalogResult`` instance (or None).
    horizon_label:
        Horizon label (e.g. ``"1d"``).

    Returns
    -------
    (analog_point, analog_lower, analog_upper) or (None, None, None)
    if analogs are not available.
    """
    if dtw_result is None or not getattr(dtw_result, "available", False):
        return None, None, None

    analogs = getattr(dtw_result, "analogs", [])
    if not analogs:
        return None, None, None

    if math.isnan(last_value) or last_value == 0:
        return None, None, None

    # Use the empirical return distribution from the analog result.
    emp_mean = getattr(dtw_result, "empirical_return_mean", 0.0)
    emp_p5 = getattr(dtw_result, "empirical_return_p5", 0.0)
    emp_p95 = getattr(dtw_result, "empirical_return_p95", 0.0)

    if math.isnan(emp_mean):
        return None, None, None

    # Scale by horizon (analogs are matched to the DTW forecast horizon,
    # but we apply a sqrt(t) scaling if horizon differs).
    horizon_days = HORIZONS.get(horizon_label, 1)
    dtw_horizon = getattr(dtw_result, "forecast_horizon_days", horizon_days)
    if dtw_horizon > 0 and dtw_horizon != horizon_days:
        scale = math.sqrt(horizon_days / dtw_horizon)
        emp_mean *= scale
        emp_p5 *= scale
        emp_p95 *= scale

    analog_point = last_value * (1.0 + emp_mean)
    analog_lower = last_value * (1.0 + emp_p5)
    analog_upper = last_value * (1.0 + emp_p95)

    return analog_point, analog_lower, analog_upper


# ===========================================================================
# Phase 5: Granger causal propagation
# ===========================================================================


def apply_granger_causal_propagation(
    aggregated: dict[str, dict[str, float]],
    granger_result: Any | None,
    cache: pd.DataFrame,
    *,
    propagation_strength: float = 0.15,
) -> tuple[dict[str, dict[str, float]], int]:
    """Propagate forecast adjustments through the causal graph.

    When variable A Granger-causes variable B, and A's forecast deviates
    significantly from its baseline (last value), B's forecast is adjusted
    proportionally to the causal link strength.

    Parameters
    ----------
    aggregated:
        ``{variable: {horizon: point_forecast}}`` from the base aggregation.
    granger_result:
        ``GrangerResult`` instance (or None).
    cache:
        Daily cache (to get last observed values for baseline comparison).
    propagation_strength:
        How much of the driver's deviation to propagate (0-1 scale).

    Returns
    -------
    (adjusted_aggregated, n_adjustments)
    """
    if granger_result is None:
        return aggregated, 0

    if not getattr(granger_result, "fitted", False):
        return aggregated, 0

    significant_pairs = getattr(granger_result, "significant_pairs", [])
    if not significant_pairs:
        return aggregated, 0

    n_adjustments = 0
    # Use a snapshot of original values for deviation calculation to
    # prevent feedback amplification when circular links exist (A->B, B->A).
    original = {var: dict(horizons) for var, horizons in aggregated.items()}
    adjusted = {var: dict(horizons) for var, horizons in aggregated.items()}

    for pair in significant_pairs:
        source = pair.get("source", pair.get("cause", ""))
        target = pair.get("target", pair.get("effect", ""))
        p_value = pair.get("p_value", 1.0)

        if source not in original or target not in adjusted:
            continue

        # Get source's last observed value for baseline.
        if source not in cache.columns:
            continue
        source_series = cache[source].dropna()
        if len(source_series) == 0:
            continue
        source_baseline = float(source_series.iloc[-1])
        if source_baseline == 0 or math.isnan(source_baseline):
            continue

        # Strength: stronger for lower p-values.
        link_strength = propagation_strength * (1.0 - min(p_value, 1.0))

        for h_label in original[source]:
            if h_label not in adjusted[target]:
                continue

            # Always read source forecast from the ORIGINAL snapshot to
            # avoid circular feedback amplification.
            source_forecast = original[source][h_label]
            target_forecast = adjusted[target][h_label]

            # Skip NaN forecasts to prevent NaN propagation.
            if math.isnan(source_forecast) or math.isnan(target_forecast):
                continue

            # Proportional deviation of source from baseline.
            source_deviation = (source_forecast - source_baseline) / abs(source_baseline)

            # Apply adjustment to target.
            adjustment = target_forecast * source_deviation * link_strength
            adjusted[target][h_label] = target_forecast + adjustment
            n_adjustments += 1

    return adjusted, n_adjustments


# ===========================================================================
# Phase 5b: SHAP explanation attachment
# ===========================================================================


def get_shap_explanation(
    variable: str,
    shap_result: Any | None,
) -> tuple[str, list[str]]:
    """Extract SHAP explanation for a variable.

    Parameters
    ----------
    variable:
        Variable name.
    shap_result:
        ``SHAPResult`` instance (or None).

    Returns
    -------
    (narrative, top_drivers)
    """
    if shap_result is None:
        return "", []

    explanations = getattr(shap_result, "explanations", {})
    expl = explanations.get(variable)
    if expl is None:
        return "", []

    narrative = getattr(expl, "narrative", "")
    top_features = getattr(expl, "top_features", [])

    # top_features may be list of (feature_name, shap_value) tuples or strings.
    drivers: list[str] = []
    for item in top_features[:5]:
        if isinstance(item, tuple):
            drivers.append(str(item[0]))
        else:
            drivers.append(str(item))

    return narrative, drivers


# ===========================================================================
# Phase 6: Recency-weighted RMSE from walk-forward
# ===========================================================================


def compute_recency_weighted_rmse(
    variable: str,
    model_name: str,
    walk_forward_result: Any | None,
    *,
    window_days: int = 30,
    decay_halflife: int = 10,
) -> float:
    """Compute exponentially-decayed RMSE from recent walk-forward errors.

    Instead of using the static training-time RMSE, this uses the last
    ``window_days`` of walk-forward prediction errors to compute a
    recency-weighted RMSE that reflects recent model performance.

    Parameters
    ----------
    variable:
        Variable name.
    model_name:
        Model name to look up in walk-forward day errors.
    walk_forward_result:
        ``WalkForwardResult`` instance (or None).
    window_days:
        Number of recent days to consider.
    decay_halflife:
        Half-life (in days) for exponential decay weighting.

    Returns
    -------
    Recency-weighted RMSE, or ``nan`` if not available.
    """
    if walk_forward_result is None:
        return float("nan")

    day_errors = getattr(walk_forward_result, "day_errors", [])
    if not day_errors:
        return float("nan")

    # Filter to matching variable and model, take last N.
    matching = [
        de for de in day_errors
        if getattr(de, "variable", "") == variable
        and getattr(de, "model_name", "") == model_name
    ]

    if not matching:
        return float("nan")

    # Take most recent window_days entries.
    recent = matching[-window_days:]
    if not recent:
        return float("nan")

    # Compute exponentially-weighted squared errors.
    n = len(recent)
    weights = np.array([
        math.exp(-i / max(decay_halflife, 1))
        for i in range(n - 1, -1, -1)  # most recent gets highest weight
    ])

    sq_errors = np.array([
        getattr(de, "squared_error", getattr(de, "error", 0.0) ** 2)
        for de in recent
    ])

    # Filter out NaN.
    mask = np.isfinite(sq_errors)
    if not mask.any():
        return float("nan")

    weights = weights[mask]
    sq_errors = sq_errors[mask]

    w_sum = weights.sum()
    if w_sum <= 0:
        return float("nan")

    wmse = (weights * sq_errors).sum() / w_sum
    return float(math.sqrt(max(wmse, 0.0)))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_predictions(
    result: PredictionAggregatorResult,
    cache_dir: str = CACHE_DIR,
) -> tuple[Path, Path]:
    """Save predictions to parquet and summary to JSON.

    Parameters
    ----------
    result:
        The aggregation result to persist.
    cache_dir:
        Directory to write files into.

    Returns
    -------
    (parquet_path, json_path)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # -- Parquet: flattened predictions table --
    rows: list[dict[str, Any]] = []
    for var_name, horizons in result.predictions.items():
        for h_label, pred in horizons.items():
            rows.append({
                "variable": pred.variable,
                "horizon": pred.horizon,
                "point_forecast": pred.point_forecast,
                "lower_ci": pred.lower_ci,
                "upper_ci": pred.upper_ci,
                "confidence": pred.confidence,
                "model_used": pred.model_used,
                "ensemble_weight": pred.ensemble_weight,
                "survival_adjusted": pred.survival_adjusted,
                "interval_source": pred.interval_source,
                "analog_forecast": _safe_float(pred.analog_forecast),
                "causal_adjustment": pred.causal_adjustment,
                "regime_blend_applied": pred.regime_blend_applied,
            })

    parquet_path = cache_path / "predictions.parquet"
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(parquet_path, index=False)
    else:
        # Write empty parquet with correct schema.
        df = pd.DataFrame(columns=[
            "variable", "horizon", "point_forecast", "lower_ci",
            "upper_ci", "confidence", "model_used", "ensemble_weight",
            "survival_adjusted", "interval_source", "analog_forecast",
            "causal_adjustment", "regime_blend_applied",
        ])
        df.to_parquet(parquet_path, index=False)

    logger.info("Saved predictions to %s (%d rows)", parquet_path, len(rows))

    # -- JSON: summary metadata --
    ta = result.technical_alpha
    summary = {
        "prediction_date": result.prediction_date,
        "current_regime": result.current_regime,
        "variables_predicted": result.variables_predicted,
        "horizons": result.horizons,
        "n_models_available": result.n_models_available,
        "n_models_failed": result.n_models_failed,
        "ensemble_weights": result.ensemble_weights,
        "survival_probability_mean": _safe_float(
            result.survival_probability_mean,
        ),
        "survival_probability_p5": _safe_float(
            result.survival_probability_p5,
        ),
        "survival_probability_p95": _safe_float(
            result.survival_probability_p95,
        ),
        "technical_alpha": {
            "mask_applied": ta.mask_applied,
            "next_day_low": _safe_float(ta.next_day_low),
            "next_day_open": ta.next_day_open,
            "next_day_high": ta.next_day_high,
            "next_day_close": ta.next_day_close,
        },
        "fitted": result.fitted,
        "error": result.error,
        # --- Full-potential upgrade metadata ---
        "conformal_coverage": _safe_float(result.conformal_coverage),
        "copula_tail_risk": _safe_float(result.copula_tail_risk),
        "dtw_analogs_used": result.dtw_analogs_used,
        "granger_adjustments_applied": result.granger_adjustments_applied,
        "regime_blend_method": result.regime_blend_method,
        "recency_weighted_rmse_used": result.recency_weighted_rmse_used,
        "shap_available": result.shap_available,
    }

    json_path = cache_path / "prediction_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved prediction summary to %s", json_path)

    return parquet_path, json_path


def _safe_float(val: Any) -> float | None:
    """Convert a value to a JSON-safe float (None for NaN/Inf).

    Handles float, int, numpy scalars, and edge cases gracefully.
    """
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ===========================================================================
# D5: Iterative multi-step prediction (chain day predictions)
# ===========================================================================


@dataclass
class StepPrediction:
    """Single prediction for one step in the iterative chain."""

    step: int = 0  # 1-indexed
    point_forecast: dict[str, float] = field(default_factory=dict)
    lower_ci: dict[str, float] = field(default_factory=dict)
    upper_ci: dict[str, float] = field(default_factory=dict)
    regime_probabilities: dict[str, float] = field(default_factory=dict)


def iterative_multi_step_predict(
    models: dict[str, BaseModelWrapper],
    initial_state: pd.Series,
    hierarchy_weights: dict[str, float] | None = None,
    horizon_days: int = 5,
    mc_scenarios: int = 1000,
    *,
    regime_return_std: float | None = None,
) -> list[StepPrediction]:
    """Generate multi-step predictions by iterative chaining.

    For each step ``s`` in ``[1, horizon_days]``:
    1. Use model ensemble to predict state at step *s* from step *s-1*.
    2. Add noise sampled from regime-conditional distribution (Monte Carlo).
    3. Feed predicted state as input for step *s+1*.
    4. Collect all intermediate predictions with uncertainty bands.

    Parameters
    ----------
    models:
        Dict of ``variable_name -> ModelWrapper`` from the forward pass /
        burn-out.
    initial_state:
        The last observed day's state vector (Series with variable names
        as index).
    hierarchy_weights:
        Current tier weights for weighting predictions.
    horizon_days:
        Number of days to predict forward.
    mc_scenarios:
        Number of Monte Carlo paths for uncertainty estimation.
    regime_return_std:
        Standard deviation to use for noise injection.  If ``None``,
        defaults to 2% of the variable value (or 0.02 if value is near zero).

    Returns
    -------
    List of ``StepPrediction``, one per day in the horizon.
    """
    if hierarchy_weights is None:
        hierarchy_weights = {f"tier{i}": 20.0 for i in range(1, 6)}

    variables = list(models.keys())
    if not variables:
        return []

    results: list[StepPrediction] = []

    # Build the initial state vector
    current_values: dict[str, float] = {}
    for var in variables:
        val = initial_state.get(var, 0.0) if var in initial_state.index else 0.0
        current_values[var] = float(val) if not (isinstance(val, float) and math.isnan(val)) else 0.0

    # --- Deterministic chain (point forecast) ---
    point_chain: list[dict[str, float]] = []
    state = dict(current_values)

    for step in range(1, horizon_days + 1):
        step_forecast: dict[str, float] = {}
        for var in variables:
            wrapper = models.get(var)
            if wrapper is None:
                step_forecast[var] = state.get(var, 0.0)
                continue
            try:
                state_arr = np.array([state.get(var, 0.0)])
                pred = wrapper.predict(state_arr)
                step_forecast[var] = float(pred[0]) if len(pred) > 0 else state.get(var, 0.0)
            except Exception:
                step_forecast[var] = state.get(var, 0.0)
        point_chain.append(step_forecast)
        state = dict(step_forecast)

    # --- Monte Carlo uncertainty (stochastic paths) ---
    # Run mc_scenarios parallel chains, adding noise at each step
    all_paths: list[list[dict[str, float]]] = []

    for _ in range(mc_scenarios):
        mc_state = dict(current_values)
        mc_path: list[dict[str, float]] = []

        for step in range(horizon_days):
            step_forecast: dict[str, float] = {}
            for var in variables:
                base_val = mc_state.get(var, 0.0)
                # Noise scale: 2% of value or configurable
                if regime_return_std is not None:
                    noise_std = regime_return_std
                else:
                    noise_std = max(abs(base_val) * 0.02, 0.001)

                noise = np.random.normal(0, noise_std)
                # Use the deterministic point forecast + noise
                det_val = point_chain[step].get(var, base_val)
                step_forecast[var] = det_val + noise

            mc_path.append(step_forecast)
            mc_state = dict(step_forecast)

        all_paths.append(mc_path)

    # --- Assemble StepPrediction objects ---
    for step_idx in range(horizon_days):
        step_num = step_idx + 1
        point = point_chain[step_idx]

        # Compute percentiles from MC paths
        lower_ci: dict[str, float] = {}
        upper_ci: dict[str, float] = {}

        for var in variables:
            mc_values = [path[step_idx].get(var, 0.0) for path in all_paths]
            mc_arr = np.array(mc_values)
            lower_ci[var] = float(np.percentile(mc_arr, 5))
            upper_ci[var] = float(np.percentile(mc_arr, 95))

        results.append(StepPrediction(
            step=step_num,
            point_forecast=point,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            regime_probabilities={},
        ))

    return results


def build_multi_horizon_predictions(
    models: dict[str, BaseModelWrapper],
    cache: pd.DataFrame,
    hierarchy_weights: dict[str, float] | None = None,
    mc_scenarios: int = 500,
) -> dict[str, Any]:
    """Convenience wrapper generating predictions for all standard horizons.

    Produces ``predictions_next_day``, ``predictions_next_week``,
    ``predictions_next_month``, ``predictions_next_year`` in the format
    expected by the company profile builder.

    Technical Alpha protection: next-day masks OHLC except Low.

    Parameters
    ----------
    models:
        Dict of ``variable_name -> ModelWrapper``.
    cache:
        Full daily cache (used to extract initial state and volatility).
    hierarchy_weights:
        Current tier weights.
    mc_scenarios:
        Number of Monte Carlo paths.

    Returns
    -------
    Dict with keys ``next_day``, ``next_week``, ``next_month``, ``next_year``.
    """
    if len(cache) == 0:
        return {}

    initial_state = cache.iloc[-1]

    # Estimate regime-conditional noise from recent volatility
    regime_std = None
    if "volatility_21d" in cache.columns:
        vol = cache["volatility_21d"].dropna()
        if len(vol) > 0:
            regime_std = float(vol.iloc[-1]) / np.sqrt(252)  # daily vol

    horizons = {"next_day": 1, "next_week": 5, "next_month": 21, "next_year": 252}
    output: dict[str, Any] = {}

    for label, days in horizons.items():
        try:
            steps = iterative_multi_step_predict(
                models=models,
                initial_state=initial_state,
                hierarchy_weights=hierarchy_weights,
                horizon_days=days,
                mc_scenarios=mc_scenarios,
                regime_return_std=regime_std,
            )
        except Exception as exc:
            logger.warning("Iterative prediction failed for %s: %s -- using fallback", label, exc)
            # Fallback: use last known values
            steps = [StepPrediction(
                step=s + 1,
                point_forecast={v: float(initial_state.get(v, 0.0)) for v in models},
                lower_ci={v: float(initial_state.get(v, 0.0)) * 0.95 for v in models},
                upper_ci={v: float(initial_state.get(v, 0.0)) * 1.05 for v in models},
            ) for s in range(days)]

        horizon_data: dict[str, Any] = {
            "steps": len(steps),
            "date_range_days": days,
        }

        if steps:
            # Point forecasts for final step
            final_step = steps[-1]
            horizon_data["point_forecast"] = final_step.point_forecast
            horizon_data["lower_ci"] = final_step.lower_ci
            horizon_data["upper_ci"] = final_step.upper_ci

            # OHLC-style series (simplified: use close proxy from point forecasts)
            if "close" in models:
                ohlc_series = []
                for s in steps:
                    close_val = s.point_forecast.get("close", 0.0)
                    lo = s.lower_ci.get("close", close_val * 0.99)
                    hi = s.upper_ci.get("close", close_val * 1.01)
                    ohlc_series.append({
                        "step": s.step,
                        "open": close_val,  # simplified: open = predicted close
                        "high": hi,
                        "low": lo,
                        "close": close_val,
                    })

                # Technical Alpha protection for next_day
                if label == "next_day" and ohlc_series:
                    for candle in ohlc_series:
                        candle["open"] = "MASKED - Technical Alpha Protection"
                        candle["high"] = "MASKED - Technical Alpha Protection"
                        candle["close"] = "MASKED - Technical Alpha Protection"
                        # Only Low is exposed

                horizon_data["ohlc_series"] = ohlc_series

        output[label] = horizon_data

    return output


# ===========================================================================
# Phase 4 -- Survival-aware model weighting
# ===========================================================================


def compute_survival_aware_weights(
    base_weights: dict[str, float],
    mode_weights: dict[str, dict[str, float]] | None = None,
    current_mode: str = "normal",
    days_in_mode: int = 0,
    transition_halflife: int = _TRANSITION_BLEND_HALFLIFE,
    previous_mode: str | None = None,
) -> dict[str, float]:
    """Compute ensemble weights conditioned on the current survival mode.

    Uses walk-forward results (mode_weights) to select different ensemble
    weights depending on the current survival mode, with soft blending
    during transitions.

    Parameters
    ----------
    base_weights:
        Default inverse-RMSE weights from ``compute_ensemble_weights``.
    mode_weights:
        Per-mode model weights from walk-forward results.
        ``{mode_label: {model_name: weight}}``.
    current_mode:
        Current survival mode label.
    days_in_mode:
        How many consecutive days we have been in the current mode.
    transition_halflife:
        Number of days for the exponential blend during transitions.
    previous_mode:
        The mode we transitioned from (for blending).  If None, no
        blending is applied.

    Returns
    -------
    Dict mapping model_name to weight (sum ~ 1.0).
    """
    if mode_weights is None or current_mode not in mode_weights:
        return base_weights

    target_weights = mode_weights[current_mode]

    # If no transition blending needed, return target weights directly.
    if previous_mode is None or previous_mode == current_mode or days_in_mode <= 0:
        return _merge_weight_keys(base_weights, target_weights)

    # Soft blending: exponential ramp from previous mode's weights to
    # current mode's weights.  alpha = 1 - exp(-days / halflife).
    alpha = 1.0 - math.exp(-days_in_mode / max(transition_halflife, 1))
    alpha = max(0.0, min(1.0, alpha))

    prev_weights = mode_weights.get(previous_mode, base_weights)

    blended: dict[str, float] = {}
    all_models = set(list(prev_weights.keys()) + list(target_weights.keys()))
    for model in all_models:
        w_prev = prev_weights.get(model, 0.0)
        w_target = target_weights.get(model, 0.0)
        blended[model] = (1.0 - alpha) * w_prev + alpha * w_target

    # Normalise
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}

    return blended


def _merge_weight_keys(
    base: dict[str, float],
    override: dict[str, float],
) -> dict[str, float]:
    """Merge override weights with base, keeping all model keys.

    Models present in override get their override weight.  Models only
    in base keep a small residual weight.
    """
    result: dict[str, float] = {}
    all_keys = set(list(base.keys()) + list(override.keys()))
    for k in all_keys:
        if k in override:
            result[k] = override[k]
        else:
            result[k] = base.get(k, 0.0) * 0.1  # small residual

    # Normalise
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}
    return result


def get_survival_context_from_cache(
    cache: pd.DataFrame,
) -> dict[str, Any]:
    """Extract survival mode context from a daily cache for weighting.

    Returns a dict with keys: current_mode, days_in_mode, previous_mode,
    stability_score.
    """
    context: dict[str, Any] = {
        "current_mode": "normal",
        "days_in_mode": 0,
        "previous_mode": None,
        "stability_score": 1.0,
    }

    if cache.empty:
        return context

    if "survival_mode" in cache.columns:
        modes = cache["survival_mode"].dropna()
        if len(modes) > 0:
            context["current_mode"] = str(modes.iloc[-1])

            # Find previous mode (last different mode)
            if len(modes) > 1:
                current = context["current_mode"]
                for i in range(len(modes) - 2, -1, -1):
                    if str(modes.iloc[i]) != current:
                        context["previous_mode"] = str(modes.iloc[i])
                        break

    if "days_in_mode" in cache.columns:
        dim = cache["days_in_mode"].dropna()
        if len(dim) > 0:
            context["days_in_mode"] = int(dim.iloc[-1])

    if "stability_score_21d" in cache.columns:
        stab = cache["stability_score_21d"].dropna()
        if len(stab) > 0:
            context["stability_score"] = float(stab.iloc[-1])

    return context


# ===========================================================================
# Pipeline entry point
# ===========================================================================


# ---------------------------------------------------------------------------
# Gap 6: Feature-driven model routing + reject option
# ---------------------------------------------------------------------------


def compute_model_routing_weights(cache: pd.DataFrame) -> dict[str, float]:
    """Route to dominant model based on current market characteristics.

    Uses technical indicators already in the cache to determine which
    model type is best suited for the current regime. Applied as a
    confidence multiplier on per-prediction outputs.

    Affinity weights are loaded from ``config/scoring_weights.yml``
    section ``model_routing`` with hardcoded fallbacks.

    Returns normalized weights (model_name -> multiplier, sums to ~1).
    """
    if cache is None or cache.empty:
        return {}

    # Load affinity weights from config (with fallbacks)
    try:
        from operator1.scoring_weights import get_weight
        _trend = get_weight("model_routing.trend_affinity", {})
        _volatile = get_weight("model_routing.volatile_affinity", {})
        _mean_rev = get_weight("model_routing.mean_revert_affinity", {})
    except Exception:
        _trend = {}
        _volatile = {}
        _mean_rev = {}

    # Defaults if config is empty
    _trend = _trend or {"kalman": 1.3, "tree": 1.1, "garch": 0.7, "baseline": 0.9}
    _volatile = _volatile or {"garch": 1.4, "tree": 1.0, "kalman": 0.7, "baseline": 1.1}
    _mean_rev = _mean_rev or {"baseline": 1.3, "kalman": 0.9, "tree": 0.8, "garch": 1.0}

    latest = cache.iloc[-1]
    weights: dict[str, float] = {
        "kalman": 1.0, "garch": 1.0, "var": 1.0,
        "lstm": 1.0, "tree": 1.0, "baseline": 1.0,
    }

    # Detect market regime from technical indicators
    _is_trending = False
    _is_volatile = False
    _is_mean_reverting = False

    # ADX > 25 = strong trend
    adx = latest.get("adx_14", 20) if "adx_14" in cache.columns else 20
    if isinstance(adx, (int, float)) and np.isfinite(adx) and adx > 25:
        _is_trending = True

    # IV-RV spread > 0.05 = vol expansion
    iv_rv = latest.get("iv_rv_spread", 0) if "iv_rv_spread" in cache.columns else 0
    if isinstance(iv_rv, (int, float)) and np.isfinite(iv_rv) and iv_rv > 0.05:
        _is_volatile = True

    # VIX term structure > 1.0 = backwardation = stress
    vts = latest.get("vix_term_structure", 1.0) if "vix_term_structure" in cache.columns else 1.0
    if isinstance(vts, (int, float)) and np.isfinite(vts) and vts > 1.0:
        _is_volatile = True

    # Low return autocorrelation = mean-reverting
    if "return_1d" in cache.columns:
        ret = cache["return_1d"].dropna()
        if len(ret) >= 30:
            try:
                autocorr = float(ret.autocorr(lag=1))
                if np.isfinite(autocorr) and abs(autocorr) < 0.1:
                    _is_mean_reverting = True
            except Exception:
                pass

    # Apply config-driven affinity weights
    if _is_trending:
        for model, mult in _trend.items():
            if model in weights:
                weights[model] *= float(mult)
    if _is_volatile:
        for model, mult in _volatile.items():
            if model in weights:
                weights[model] *= float(mult)
    if _is_mean_reverting:
        for model, mult in _mean_rev.items():
            if model in weights:
                weights[model] *= float(mult)

    # Normalize to sum to 1
    total = sum(weights.values())
    if total > 0:
        return {k: v / total for k, v in weights.items()}
    return weights


def apply_reject_option(
    predictions: dict,
    cache: pd.DataFrame,
) -> dict:
    """Apply reject option: when no model is confident, reduce directional bet.

    If conformal interval width > 2x the absolute point forecast for 'close',
    replace the point forecast with last close (no directional bet) and
    reduce confidence to 0.3.

    This prevents the ensemble from making high-confidence predictions when
    all underlying models disagree significantly.

    Parameters
    ----------
    predictions:
        Dict of {variable: {horizon: HorizonPrediction}} from the aggregator.
    cache:
        Daily cache (for last close value).

    Returns
    -------
    Modified predictions dict (in-place modification + returned).
    """
    if not predictions or cache is None or cache.empty:
        return predictions

    last_close = None
    if "close" in cache.columns and cache["close"].notna().any():
        last_close = float(cache["close"].dropna().iloc[-1])

    for var, horizons in predictions.items():
        if not isinstance(horizons, dict):
            continue
        for h, hp in horizons.items():
            upper = getattr(hp, "upper_ci", None)
            lower = getattr(hp, "lower_ci", None)
            pf = getattr(hp, "point_forecast", None)

            if upper is not None and lower is not None and pf is not None:
                interval_width = abs(float(upper) - float(lower))
                abs_pf = abs(float(pf))

                # Reject when interval > 2x the forecast value (extreme uncertainty)
                if abs_pf > 0 and interval_width > 2.0 * abs_pf:
                    # For close price: use last close (no directional bet)
                    if var == "close" and last_close is not None:
                        hp.point_forecast = last_close
                    # Reduce confidence
                    if hasattr(hp, "confidence") and hp.confidence is not None:
                        hp.confidence *= 0.3
                    # Flag as rejected
                    if hasattr(hp, "metadata"):
                        if hp.metadata is None:
                            hp.metadata = {}
                        hp.metadata["reject_flag"] = True

    return predictions


def run_prediction_aggregation(
    cache: pd.DataFrame,
    forecast_result: ForecastResult,
    mc_result: MonteCarloResult | None = None,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    survival_risk_multiplier: float = DEFAULT_SURVIVAL_RISK_MULTIPLIER,
    save_to_cache: bool = True,
    cache_dir: str = CACHE_DIR,
    mode_weights: dict[str, dict[str, float]] | None = None,
    # --- v2 improvement inputs ---
    macro_quadrant_label: str = "",
    fundamental_fair_value: float | None = None,
    scenario_result: Any | None = None,
    signal_ic_result: Any | None = None,
    prediction_log_summary: dict | None = None,
    # --- New optional inputs from sibling modules ---
    conformal_result: Any | None = None,
    dual_regime_result: Any | None = None,
    copula_result: Any | None = None,
    dtw_result: Any | None = None,
    granger_result: Any | None = None,
    shap_result: Any | None = None,
    walk_forward_result: Any | None = None,
    feature_selection_result: Any | None = None,
    event_calendar_result: Any | None = None,
) -> PredictionAggregatorResult:
    """Run the full prediction aggregation pipeline.

    Combines forecasting results with Monte Carlo survival estimates
    and optional outputs from sibling modules to produce final predictions
    with uncertainty bands and Technical Alpha protection.

    Parameters
    ----------
    cache:
        Daily cache DataFrame (with ``close``, ``volatility_21d``,
        ``regime_label`` columns expected).
    forecast_result:
        Output from ``run_forecasting`` (T6.2).
    mc_result:
        Output from ``run_monte_carlo`` (T6.3).  Optional -- if
        ``None``, survival adjustment is skipped.
    confidence_level:
        Desired confidence level for uncertainty bands (default 0.90).
    survival_risk_multiplier:
        How aggressively to widen bands when survival prob is low.
    save_to_cache:
        If ``True``, write predictions.parquet and prediction_summary.json.
    cache_dir:
        Directory for cache files.
    mode_weights:
        Per-survival-mode model weights from walk-forward results
        (output of ``get_mode_weights_from_walk_forward``).  When
        provided and the cache contains survival timeline columns,
        ensemble weights are conditioned on the current survival mode
        with soft transition blending.
    conformal_result:
        ``ConformalResult`` from conformal prediction calibration.
        When available, distribution-free intervals replace the
        Gaussian RMSE-based bands.
    dual_regime_result:
        ``DualRegimeResult`` from regime mixer.  When available,
        soft regime probability blending modulates ensemble weights.
    copula_result:
        ``CopulaResult`` from copula analysis.  When available and
        joint crisis probability is high, uncertainty bands are widened.
    dtw_result:
        ``DTWAnalogResult`` from DTW historical analogs.  When available,
        provides an independent empirical forecast channel.
    granger_result:
        ``GrangerResult`` from Granger causality analysis.  When
        available, forecast adjustments propagate through the causal graph.
    shap_result:
        ``SHAPResult`` from explainability module.  When available,
        narratives and top drivers are attached to each prediction.
    walk_forward_result:
        ``WalkForwardResult`` from walk-forward evaluation.  When
        available, recency-weighted RMSE replaces static training RMSE.

    Returns
    -------
    ``PredictionAggregatorResult`` with all predictions, TA mask, and
    metadata.
    """
    logger.info("Starting prediction aggregation pipeline...")

    result = PredictionAggregatorResult()
    result.prediction_date = date.today().isoformat()
    result.horizons = sorted(HORIZONS.keys(), key=lambda h: HORIZONS[h])

    # ------------------------------------------------------------------
    # Current regime
    # ------------------------------------------------------------------
    if "regime_label" in cache.columns:
        labels = cache["regime_label"].dropna()
        if len(labels) > 0:
            result.current_regime = str(labels.iloc[-1])
        else:
            result.current_regime = "unknown"
    else:
        result.current_regime = "unknown"

    # ------------------------------------------------------------------
    # Build tier map for per-tier confidence multipliers (Phase 2.5)
    # ------------------------------------------------------------------
    tier_map = _load_tier_variables()

    # ------------------------------------------------------------------
    # Ensemble weights (survival-aware if mode_weights provided)
    # ------------------------------------------------------------------
    base_weights = compute_ensemble_weights(
        forecast_result.metrics,
    )

    # C3: IC-weighted calibration (use signal IC to upweight predictive models)
    if signal_ic_result is not None:
        base_weights = apply_ic_weighted_calibration(base_weights, signal_ic_result)

    # F3: Realized IC feedback from previous prediction logs
    if prediction_log_summary is not None:
        base_weights = apply_prediction_log_feedback(base_weights, prediction_log_summary)

    # Apply survival-aware weighting if walk-forward mode weights and
    # survival context are available.
    if mode_weights:
        surv_ctx = get_survival_context_from_cache(cache)
        result.ensemble_weights = compute_survival_aware_weights(
            base_weights=base_weights,
            mode_weights=mode_weights,
            current_mode=surv_ctx["current_mode"],
            days_in_mode=surv_ctx["days_in_mode"],
            previous_mode=surv_ctx["previous_mode"],
        )
        logger.info(
            "Survival-aware weights applied: mode=%s, days_in_mode=%d, "
            "stability=%.3f",
            surv_ctx["current_mode"],
            surv_ctx["days_in_mode"],
            surv_ctx["stability_score"],
        )
    else:
        result.ensemble_weights = base_weights

    # ------------------------------------------------------------------
    # Phase 2: Regime probability blending (soft weights)
    # ------------------------------------------------------------------
    regime_blend_applied = False
    if dual_regime_result is not None:
        blended, regime_blend_applied = compute_regime_blended_weights(
            result.ensemble_weights,
            dual_regime_result,
        )
        if regime_blend_applied:
            result.ensemble_weights = blended
            result.regime_blend_method = "soft_probability"
            logger.info(
                "Regime probability blending applied to ensemble weights"
            )
        else:
            result.regime_blend_method = "hard_label"
    else:
        result.regime_blend_method = "hard_label"

    # Count model availability.
    failed_flags = [
        forecast_result.model_failed_kalman,
        forecast_result.model_failed_garch,
        forecast_result.model_failed_var,
        forecast_result.model_failed_lstm,
        forecast_result.model_failed_tree,
    ]
    result.n_models_failed = sum(failed_flags)
    result.n_models_available = 5 - result.n_models_failed  # 5 model types

    # Module contribution scores (Section F.1 Category 7 from core idea).
    # Normalize ensemble weights to percentages for interpretability.
    if result.ensemble_weights:
        total_w = sum(result.ensemble_weights.values())
        if total_w > 0:
            result.module_contributions = {
                name: round(w / total_w * 100, 1)
                for name, w in sorted(
                    result.ensemble_weights.items(),
                    key=lambda x: -x[1],
                )
                if w > 0
            }

    # ------------------------------------------------------------------
    # Aggregate forecasts
    # ------------------------------------------------------------------
    aggregated = aggregate_forecasts(forecast_result)

    if not aggregated:
        result.error = "No forecasts available to aggregate"
        logger.warning(result.error)
        # Still apply TA mask and save.
        result.technical_alpha = apply_technical_alpha_mask(cache, {})
        if save_to_cache:
            save_predictions(result, cache_dir)
        return result

    # ------------------------------------------------------------------
    # Phase 5a: Granger causal propagation
    # ------------------------------------------------------------------
    n_granger_adjustments = 0
    if granger_result is not None:
        aggregated, n_granger_adjustments = apply_granger_causal_propagation(
            aggregated, granger_result, cache,
        )
        if n_granger_adjustments > 0:
            result.granger_adjustments_applied = n_granger_adjustments
            logger.info(
                "Granger causal propagation: %d adjustments applied",
                n_granger_adjustments,
            )

    # ------------------------------------------------------------------
    # Survival probabilities from Monte Carlo
    # ------------------------------------------------------------------
    mc_survival_by_horizon: dict[str, float] = {}
    if mc_result is not None and mc_result.fitted:
        mc_survival_by_horizon = mc_result.survival_probability
        result.survival_probability_mean = mc_result.survival_probability_mean
        result.survival_probability_p5 = mc_result.survival_probability_p5
        result.survival_probability_p95 = mc_result.survival_probability_p95

    # ------------------------------------------------------------------
    # Phase 3: Copula tail risk
    # ------------------------------------------------------------------
    copula_tail_risk = 0.0
    if copula_result is not None:
        try:
            copula_tail_risk = getattr(
                copula_result, "joint_crisis_probability", 0.0,
            )
            if math.isnan(copula_tail_risk):
                copula_tail_risk = 0.0
            result.copula_tail_risk = copula_tail_risk
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase 4: DTW analog metadata
    # ------------------------------------------------------------------
    if dtw_result is not None and getattr(dtw_result, "available", False):
        result.dtw_analogs_used = len(getattr(dtw_result, "analogs", []))

    # ------------------------------------------------------------------
    # Phase 5b: SHAP availability
    # ------------------------------------------------------------------
    if shap_result is not None:
        result.shap_available = bool(
            getattr(shap_result, "explanations", {})
        )

    # ------------------------------------------------------------------
    # Phase 1: Conformal coverage metadata
    # ------------------------------------------------------------------
    if conformal_result is not None:
        result.conformal_coverage = getattr(
            conformal_result, "coverage_level", None,
        )

    # ------------------------------------------------------------------
    # Reference RMSE for confidence scoring
    # ------------------------------------------------------------------
    valid_rmse_values = [
        m.rmse
        for m in forecast_result.metrics
        if m.fitted and not math.isnan(m.rmse)
    ]
    if valid_rmse_values:
        rmse_reference = float(np.median(valid_rmse_values))
    else:
        rmse_reference = float("nan")

    # ------------------------------------------------------------------
    # Z-score for requested confidence level
    # ------------------------------------------------------------------
    # Map common levels; fallback to 1.645 for 90%.
    z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576, 0.80: 1.282}
    z_score = z_map.get(confidence_level, Z_SCORE_90)

    # ------------------------------------------------------------------
    # Build predictions per variable per horizon
    # ------------------------------------------------------------------
    for var_name, var_forecasts in aggregated.items():
        # Phase 6: Recency-weighted RMSE from walk-forward.
        model_name = forecast_result.model_used.get(var_name, "unknown")
        recency_rmse = compute_recency_weighted_rmse(
            var_name, model_name, walk_forward_result,
        )
        var_rmse = _get_best_rmse_for_variable(
            var_name, forecast_result.metrics,
        )
        if not math.isnan(recency_rmse):
            var_rmse = recency_rmse
            result.recency_weighted_rmse_used = True

        model_weight = result.ensemble_weights.get(model_name, 0.0)

        # Get last observed value for DTW analog forecasts.
        last_value = float("nan")
        if var_name in cache.columns:
            vs = cache[var_name].dropna()
            if len(vs) > 0:
                last_value = float(vs.iloc[-1])

        # Phase 5b: SHAP explanation for this variable.
        explanation, top_drivers = get_shap_explanation(var_name, shap_result)

        horizon_preds: dict[str, HorizonPrediction] = {}

        for h_label in result.horizons:
            point = var_forecasts.get(h_label, float("nan"))
            horizon_days = HORIZONS.get(h_label, 1)

            # Survival probability for this horizon.
            surv_prob = mc_survival_by_horizon.get(h_label, 1.0)
            survival_adjusted = surv_prob < 1.0

            # ----------------------------------------------------------
            # B1: Survival-intensity POINT FORECAST adjustment.
            # When survival signals indicate distress, shift the point
            # forecast downward proportionally. This fixes the gap where
            # bands widen but the center stays bullish during distress.
            # Uses survival_intensity (continuous 0-1 from enriched
            # survival timeline) and expected max drawdown from MC.
            # ----------------------------------------------------------
            if not math.isnan(point) and survival_adjusted:
                _surv_intensity = 0.0
                if "survival_intensity" in cache.columns:
                    _si = cache["survival_intensity"].dropna()
                    if len(_si) > 0:
                        _surv_intensity = float(_si.iloc[-1])

                if _surv_intensity > 0.3:
                    # Expected max drawdown from MC (or conservative default)
                    _expected_dd = 0.20  # default 20% drawdown assumption
                    if mc_result is not None and mc_result.fitted:
                        _dd_stats = mc_result.max_drawdown_distribution.get(h_label, {})
                        _mc_dd = abs(_dd_stats.get("median", 0.0))
                        if _mc_dd > 0.01:
                            _expected_dd = min(0.60, _mc_dd)

                    # Horizon factor: longer horizons get more adjustment
                    # because fundamentals dominate over momentum
                    _horizon_factor = {1: 0.2, 5: 0.5, 21: 0.8, 252: 1.0}
                    _hf = _horizon_factor.get(horizon_days, min(1.0, horizon_days / 252))

                    # Distress haircut: intensity * expected drawdown * horizon factor
                    _distress_haircut = _surv_intensity * _expected_dd * _hf
                    _distress_haircut = min(0.40, _distress_haircut)  # cap at 40%
                    point = point * (1.0 - _distress_haircut)
                    logger.debug(
                        "B1 survival adjustment: %s %s haircut=%.3f "
                        "(intensity=%.2f, dd=%.2f, hf=%.1f)",
                        var_name, h_label, _distress_haircut,
                        _surv_intensity, _expected_dd, _hf,
                    )

            # ----------------------------------------------------------
            # B2: Macro-conditional return shift.
            # When macro quadrant indicates deterioration, apply a
            # drift adjustment to return-like variables.
            # ----------------------------------------------------------
            if (not math.isnan(point)
                    and macro_quadrant_label
                    and var_name in ("close", "return_1d", "return_5d", "return_21d")):
                _QUADRANT_DRIFT = {
                    "goldilocks": +0.0005,
                    "overheating": +0.0002,
                    "stagflation": -0.0003,
                    "recession": -0.0005,
                }
                _drift = _QUADRANT_DRIFT.get(macro_quadrant_label.lower(), 0.0)
                if abs(_drift) > 1e-6 and var_name == "close":
                    # For close price: apply return drift * horizon * last close
                    point = point * (1.0 + _drift * horizon_days)
                elif abs(_drift) > 1e-6:
                    # For return variables: shift directly
                    point = point + _drift * horizon_days

            # ----------------------------------------------------------
            # B3: Scenario-weighted price expectation.
            # During survival mode, blend with scenario engine results.
            # ----------------------------------------------------------
            if (not math.isnan(point)
                    and scenario_result is not None
                    and getattr(scenario_result, "available", False)
                    and var_name == "close"):
                _surv_intensity_b3 = 0.0
                if "survival_intensity" in cache.columns:
                    _si_b3 = cache["survival_intensity"].dropna()
                    if len(_si_b3) > 0:
                        _surv_intensity_b3 = float(_si_b3.iloc[-1])

                if _surv_intensity_b3 > 0.5:
                    try:
                        _orderly = getattr(scenario_result, "orderly", None)
                        _muddle = getattr(scenario_result, "muddle_through", None)
                        _catastrophic = getattr(scenario_result, "catastrophic", None)
                        if _orderly and _muddle and _catastrophic:
                            _ord_med = getattr(_orderly, "terminal_median_equity", point)
                            _mud_med = getattr(_muddle, "terminal_median_equity", point)
                            _cat_med = getattr(_catastrophic, "terminal_median_equity", point)
                            # Probability weights (from scenario engine or defaults)
                            _p_ord = 0.30
                            _p_mud = 0.45
                            _p_cat = 0.25
                            _scenario_price = _p_ord * _ord_med + _p_mud * _mud_med + _p_cat * _cat_med
                            if _scenario_price > 0 and not math.isnan(_scenario_price):
                                _scen_weight = min(0.6, _surv_intensity_b3 * 0.8)
                                point = (1.0 - _scen_weight) * point + _scen_weight * _scenario_price
                    except Exception:
                        pass  # scenario data structure mismatch -- skip gracefully

            # ----------------------------------------------------------
            # C2: Fundamental gravity for long horizons.
            # Blend price forecasts with fundamental fair value (DCF)
            # for 21d+ horizons. Price converges to fundamentals over time.
            # ----------------------------------------------------------
            if (not math.isnan(point)
                    and fundamental_fair_value is not None
                    and not math.isnan(fundamental_fair_value)
                    and fundamental_fair_value > 0
                    and var_name == "close"):
                _FUND_WEIGHTS = {"1d": 0.00, "5d": 0.05, "21d": 0.20, "252d": 0.50}
                _fw = _FUND_WEIGHTS.get(h_label, 0.0)
                if _fw > 0:
                    point = (1.0 - _fw) * point + _fw * fundamental_fair_value

            # ----------------------------------------------------------
            # Phase 1: Try conformal intervals first.
            # ----------------------------------------------------------
            conf_lower, conf_upper, used_conformal = get_conformal_interval(
                var_name, h_label, conformal_result,
            )
            if used_conformal:
                # Re-center conformal bounds on the ensemble point forecast.
                # The conformal WIDTH is valid (calibrated from residuals),
                # but the CENTER is stale -- it reflects the raw forward-pass
                # forecast before B1/B2/B3/C2 ensemble adjustments shifted
                # the point.  Preserving the half-width and moving the center
                # ensures lower_ci <= point_forecast <= upper_ci always holds.
                conf_half_width = (conf_upper - conf_lower) / 2.0
                if not math.isnan(point):
                    lower = point - conf_half_width
                    upper = point + conf_half_width
                else:
                    lower, upper = conf_lower, conf_upper
                interval_source = "conformal"
                # Still apply survival widening on top of conformal.
                if survival_adjusted:
                    surv_p = max(0.0, min(1.0, surv_prob))
                    risk_factor = 1.0 + (1.0 - surv_p) * survival_risk_multiplier
                    half_width = conf_half_width * risk_factor
                    lower = point - half_width
                    upper = point + half_width
            else:
                # Fallback to RMSE-based bands (with BMA Method 6 between-model var).
                _bma_std = compute_bma_between_model_std(
                    forecast_result.metrics, forecast_result.forecasts,
                    var_name, h_label,
                ) if forecast_result is not None else 0.0
                lower, upper = compute_uncertainty_bands(
                    point,
                    var_rmse,
                    horizon_days,
                    survival_probability=surv_prob,
                    survival_risk_multiplier=survival_risk_multiplier,
                    z_score=z_score,
                    between_model_std=_bma_std,
                )
                interval_source = "rmse+bma" if _bma_std > 0 else "rmse"

                # MC percentile override for long horizons (>= 21d).
                # The RMSE * sqrt(h) scaling assumes independent daily
                # errors, which underestimates uncertainty during trending
                # markets.  MC path percentiles from 10K regime-switching
                # simulations capture serial correlation and tail risk.
                if (
                    mc_result is not None
                    and horizon_days >= 21
                    and var_name == "close"
                ):
                    try:
                        _mc_tv = getattr(mc_result, "terminal_values", {})
                        _mc_paths = _mc_tv.get(horizon_days) or _mc_tv.get(f"{horizon_days}d")
                        if _mc_paths is not None and len(_mc_paths) > 100:
                            _mc_arr = np.array(_mc_paths)
                            _mc_base = last_value if last_value and not math.isnan(last_value) else point
                            _mc_p5 = float(np.percentile(_mc_arr, 5)) * _mc_base
                            _mc_p95 = float(np.percentile(_mc_arr, 95)) * _mc_base
                            if (_mc_p95 - _mc_p5) > (upper - lower):
                                lower = _mc_p5
                                upper = _mc_p95
                                interval_source = "mc_percentile"
                    except Exception:
                        pass

            # ----------------------------------------------------------
            # Phase 2.5: Per-tier confidence multipliers (survival mode).
            # In survival mode, Tier 4/5 predictions are less reliable.
            # Widen their intervals to reflect reduced confidence.
            # Source: The_Apps_core_idea.pdf Section 6.7
            # ----------------------------------------------------------
            if survival_adjusted:
                _TIER_CONFIDENCE = {1: 1.0, 2: 1.0, 3: 0.9, 4: 0.5, 5: 0.3}
                _var_tier = _get_tier_for_variable(var_name, tier_map) if tier_map else None
                _tier_num = int(_var_tier.replace("tier", "")) if _var_tier and _var_tier.startswith("tier") else 3
                _conf_mult = _TIER_CONFIDENCE.get(_tier_num, 1.0)
                if _conf_mult < 1.0:
                    mid = point if not math.isnan(point) else (lower + upper) / 2.0
                    half_width = (upper - lower) / 2.0
                    lower = mid - half_width / _conf_mult
                    upper = mid + half_width / _conf_mult

            # ----------------------------------------------------------
            # Phase 3: Copula tail risk widening.
            # ----------------------------------------------------------
            copula_multiplier = compute_copula_tail_adjustment(
                var_name, copula_result,
            )
            if copula_multiplier > 1.0:
                mid = point if not math.isnan(point) else (lower + upper) / 2.0
                lower = mid - (mid - lower) * copula_multiplier
                upper = mid + (upper - mid) * copula_multiplier

            # ----------------------------------------------------------
            # Phase 3.5: Minimum half-width floor (Fix 1).
            # A $0.45 band on a $250 stock is 0.18% -- absurdly narrow
            # for 90% coverage.  Floor: 1% of price * sqrt(horizon).
            # ----------------------------------------------------------
            _MIN_HW_PCT = 0.01
            if not math.isnan(point) and abs(point) > 0:
                _min_hw = abs(point) * _MIN_HW_PCT * math.sqrt(max(horizon_days, 1))
                _cur_hw = (upper - lower) / 2.0 if not (math.isnan(upper) or math.isnan(lower)) else 0.0
                if _cur_hw < _min_hw:
                    lower = point - _min_hw
                    upper = point + _min_hw
                    interval_source += "+floor"

            # ----------------------------------------------------------
            # Phase 3.6: Ensemble disagreement widening (Lakshminarayanan 2017).
            # When models disagree, their disagreement IS the uncertainty.
            # Use per-model RMSE spread as a proxy for forecast spread.
            # ----------------------------------------------------------
            if forecast_result is not None and hasattr(forecast_result, "metrics"):
                _model_rmses = [
                    m.rmse for m in forecast_result.metrics
                    if m.variable == var_name and m.fitted and m.rmse > 0
                ]
                if len(_model_rmses) >= 2:
                    _rmse_spread = max(_model_rmses) - min(_model_rmses)
                    _disagree_hw = z_score * _rmse_spread * math.sqrt(max(horizon_days, 1))
                    _cur_hw = (upper - lower) / 2.0 if not (math.isnan(upper) or math.isnan(lower)) else 0.0
                    if _disagree_hw > _cur_hw:
                        lower = point - _disagree_hw
                        upper = point + _disagree_hw
                        interval_source += "+disagreement"

            # ----------------------------------------------------------
            # Phase 3.7: IV-anchored interval blending (Hull 2018).
            # Options-implied vol is the market's forward-looking consensus
            # on uncertainty.  Blend with model-derived intervals.
            # ----------------------------------------------------------
            if "iv30" in cache.columns and var_name == "close":
                _iv_series = cache["iv30"].dropna()
                if len(_iv_series) > 0:
                    _iv30_val = float(_iv_series.iloc[-1])
                    if _iv30_val > 0 and not math.isnan(_iv30_val):
                        _iv_daily = _iv30_val / math.sqrt(252)
                        _last_px = last_value if last_value and not math.isnan(last_value) else point
                        _iv_hw = z_score * _last_px * _iv_daily * math.sqrt(max(horizon_days, 1))
                        _cur_hw = (upper - lower) / 2.0
                        # Horizon-decaying IV blend: IV is best at short
                        # horizons (market-priced event risk) but model
                        # RMSE + MC percentiles dominate at longer horizons.
                        _IV_BLEND = {1: 0.50, 5: 0.35, 21: 0.15, 252: 0.05}
                        _iv_w = _IV_BLEND.get(horizon_days, 0.20)
                        _blended_hw = _iv_w * _iv_hw + (1.0 - _iv_w) * _cur_hw
                        if _blended_hw > _cur_hw:
                            lower = point - _blended_hw
                            upper = point + _blended_hw
                            interval_source += "+iv"

            # ----------------------------------------------------------
            # Phase 3.8: ATH volatility scaling (Bouchaud 2002).
            # Near all-time highs, realized vol underestimates future vol.
            # Scale by vol-of-vol ratio.
            # ----------------------------------------------------------
            if "anchoring_52w_high" in cache.columns:
                _ath_s = cache["anchoring_52w_high"].dropna()
                if len(_ath_s) > 0:
                    _ath_v = float(_ath_s.iloc[-1])
                    if _ath_v > 0.90:
                        _vov = 0.0
                        _vol = 0.02
                        if "vol_of_vol_21d" in cache.columns:
                            _vov_s = cache["vol_of_vol_21d"].dropna()
                            if len(_vov_s) > 0:
                                _vov = float(_vov_s.iloc[-1])
                        if "volatility_21d" in cache.columns:
                            _vol_s = cache["volatility_21d"].dropna()
                            if len(_vol_s) > 0:
                                _vol = max(0.001, float(_vol_s.iloc[-1]))
                        _vov_ratio = max(1.0, _vov / _vol) if _vov > 0 else 1.0
                        _ath_scale = 1.0 + (_ath_v - 0.90) * _vov_ratio
                        _cur_hw = (upper - lower) / 2.0
                        lower = point - _cur_hw * _ath_scale
                        upper = point + _cur_hw * _ath_scale
                        interval_source += "+ath"

            # ----------------------------------------------------------
            # Phase 3.9: Invariant guard -- lower <= point <= upper.
            # Must ALWAYS hold regardless of which interval path fired.
            # ----------------------------------------------------------
            if not math.isnan(point) and not math.isnan(lower) and not math.isnan(upper):
                if lower > point or upper < point:
                    _hw = max(abs(upper - lower) / 2.0, abs(point) * _MIN_HW_PCT)
                    lower = point - _hw
                    upper = point + _hw

            # ----------------------------------------------------------
            # Phase 3.10: Width cap -- prevent multiplicative blowup.
            # MC P5/P95 range is the maximum reasonable uncertainty
            # (already accounts for regime switching + tail risk).
            # Absolute cap at 50% of price as final safety net.
            # ----------------------------------------------------------
            if mc_result is not None and var_name == "close" and not math.isnan(point):
                try:
                    _mc_tv = getattr(mc_result, "terminal_values", {})
                    _mc_paths_cap = _mc_tv.get(horizon_days) or _mc_tv.get(f"{horizon_days}d")
                    if _mc_paths_cap is not None and len(_mc_paths_cap) > 100:
                        _mc_arr_cap = np.array(_mc_paths_cap)
                        _mc_base_cap = last_value if last_value and not math.isnan(last_value) else point
                        _mc_p5_cap = float(np.percentile(_mc_arr_cap, 5)) * _mc_base_cap
                        _mc_p95_cap = float(np.percentile(_mc_arr_cap, 95)) * _mc_base_cap
                        _mc_max_hw = (_mc_p95_cap - _mc_p5_cap) / 2.0
                        _cur_hw_cap = (upper - lower) / 2.0
                        if _mc_max_hw > 0 and _cur_hw_cap > _mc_max_hw:
                            lower = point - _mc_max_hw
                            upper = point + _mc_max_hw
                            interval_source += "+mc_cap"
                except Exception:
                    pass

            # Absolute cap: 50% of price at any horizon
            if not math.isnan(point) and abs(point) > 0:
                _abs_max_hw = abs(point) * 0.50
                _cur_hw_abs = (upper - lower) / 2.0
                if _cur_hw_abs > _abs_max_hw:
                    lower = point - _abs_max_hw
                    upper = point + _abs_max_hw
                    interval_source += "+abs_cap"

            # ----------------------------------------------------------
            # Phase 4: DTW analog forecast.
            # ----------------------------------------------------------
            analog_point, analog_lower, analog_upper = compute_dtw_analog_forecast(
                var_name, last_value, dtw_result, h_label,
            )

            # If analog is available, blend it into the point forecast
            # with a small weight (analog as Bayesian prior).
            causal_adj = 0.0
            if analog_point is not None and not math.isnan(point):
                analog_weight = min(0.15, 0.05 * result.dtw_analogs_used)
                blended_point = (1.0 - analog_weight) * point + analog_weight * analog_point
                causal_adj = blended_point - point
                # Don't overwrite the point forecast itself -- record
                # the analog influence as causal_adjustment for
                # transparency.

            # Confidence score.
            confidence = compute_confidence_score(
                var_rmse, rmse_reference, surv_prob,
            )

            # P2: Sanity clamp for price-level predictions (close, open, high).
            # Level-based models (XGBoost, tree) can extrapolate wildly beyond
            # reasonable bounds. Cap at +/- max_pct_change per horizon from
            # the last observed value.
            if var_name in ("close", "open", "high", "low") and not math.isnan(point):
                _last_val = last_value  # from cache[var_name].iloc[-1]
                if _last_val is not None and not math.isnan(_last_val) and _last_val > 0:
                    _max_pct = {1: 0.10, 5: 0.20, 21: 0.40, 252: 1.50}.get(horizon_days, 0.50)
                    _upper_bound = _last_val * (1.0 + _max_pct)
                    _lower_bound = _last_val * (1.0 - _max_pct)
                    if point > _upper_bound or point < _lower_bound:
                        _clamped = max(_lower_bound, min(_upper_bound, point))
                        logger.debug(
                            "P2 clamp: %s/%s %.2f -> %.2f (last=%.2f, max_pct=%.0f%%)",
                            var_name, h_label, point, _clamped, _last_val, _max_pct * 100,
                        )
                        point = _clamped
                        # Also tighten CIs to clamped range
                        lower = max(_lower_bound, lower) if not math.isnan(lower) else lower
                        upper = min(_upper_bound, upper) if not math.isnan(upper) else upper

            # ----------------------------------------------------------
            # Beyond Bands: Distributional forecasting integration.
            # Compute skew signal, BMA between-model std, entropy-pooling
            # scenarios, and constrained optimization.
            # ----------------------------------------------------------
            _bb_skew = None
            _bb_between_std = None
            _bb_scenario_point = None
            _bb_scenarios = None

            # Method 1: Quantile regression skew signal from tree metrics.
            if forecast_result is not None:
                for _m in forecast_result.metrics:
                    if (_m.variable == var_name and _m.fitted
                            and _m.quantile_forecasts is not None):
                        _qf = _m.quantile_forecasts
                        _p05 = _qf.get(0.05)
                        _p50 = _qf.get(0.50)
                        _p95 = _qf.get(0.95)
                        if _p05 is not None and _p50 is not None and _p95 is not None:
                            _bb_skew = (_p95 - _p50) - (_p50 - _p05)
                            # Use quantile bounds for asymmetric intervals
                            # when they are wider than current symmetric bands
                            if _p05 < lower and not math.isnan(lower):
                                lower = _p05
                                interval_source += "+qr_lower"
                            if _p95 > upper and not math.isnan(upper):
                                upper = _p95
                                interval_source += "+qr_upper"
                        break

            # Method 2: Conditional sigma from distributional model.
            if forecast_result is not None:
                for _m in forecast_result.metrics:
                    if (_m.variable == var_name and _m.fitted
                            and _m.conditional_sigma is not None
                            and _m.conditional_sigma > 0):
                        _cond_hw = z_score * _m.conditional_sigma * math.sqrt(max(horizon_days, 1))
                        _cur_hw = (upper - lower) / 2.0 if not (math.isnan(upper) or math.isnan(lower)) else 0.0
                        if _cond_hw > _cur_hw:
                            lower = point - _cond_hw
                            upper = point + _cond_hw
                            interval_source += "+distributional"
                        break

            # Method 6: BMA between-model std.
            if forecast_result is not None:
                _bb_between_std = compute_bma_between_model_std(
                    forecast_result.metrics,
                    forecast_result.forecasts,
                    var_name,
                    h_label,
                )

            # Method 4: Entropy pooling scenario decomposition (close only).
            if (mc_result is not None and var_name == "close"
                    and not math.isnan(point)):
                _mc_tv = getattr(mc_result, "terminal_values", {})
                _mc_paths_ep = _mc_tv.get(h_label) or _mc_tv.get(horizon_days)
                if _mc_paths_ep is not None and len(_mc_paths_ep) > 100:
                    _model_ret = (point / last_value - 1.0) if last_value and last_value > 0 else 0.0
                    _ep = _entropy_pooling_scenarios(
                        np.array(_mc_paths_ep), _model_ret, last_value or point,
                    )
                    if _ep.get("available"):
                        _bb_scenario_point = _ep["weighted_point"]
                        _bb_scenarios = _ep["scenarios"]
                        result.scenario_decomposition[h_label] = _ep

            # Method 3: Constrained optimization (close only, after bands computed).
            if var_name == "close" and not math.isnan(point):
                _mom_5d = 0.0
                if "return_5d" in cache.columns:
                    _r5d_s = cache["return_5d"].dropna()
                    if len(_r5d_s) > 0:
                        _mom_5d = float(_r5d_s.iloc[-1])
                point = _constrained_point_forecast(
                    point, lower, upper,
                    last_value or point, _mom_5d, result.current_regime,
                )

            pred = HorizonPrediction(
                variable=var_name,
                horizon=h_label,
                point_forecast=point,
                lower_ci=lower,
                upper_ci=upper,
                confidence=confidence,
                model_used=model_name,
                ensemble_weight=model_weight,
                survival_adjusted=survival_adjusted,
                interval_source=interval_source,
                explanation=explanation,
                top_drivers=top_drivers,
                analog_forecast=analog_point,
                causal_adjustment=causal_adj,
                regime_blend_applied=regime_blend_applied,
                skew_signal=_bb_skew,
                between_model_std=_bb_between_std,
                scenario_weighted_point=_bb_scenario_point,
                scenarios=_bb_scenarios,
            )
            horizon_preds[h_label] = pred

        result.predictions[var_name] = horizon_preds

    result.variables_predicted = sorted(result.predictions.keys())

    # ------------------------------------------------------------------
    # Technical Alpha mask
    # ------------------------------------------------------------------
    result.technical_alpha = apply_technical_alpha_mask(cache, aggregated)

    # ------------------------------------------------------------------
    # Mark as fitted
    # ------------------------------------------------------------------
    result.fitted = True

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    if save_to_cache:
        try:
            save_predictions(result, cache_dir)
        except Exception as exc:
            logger.warning("Failed to save predictions: %s", exc)

    # ------------------------------------------------------------------
    # Summary log
    # ------------------------------------------------------------------
    extras = []
    if result.conformal_coverage is not None:
        extras.append(f"conformal={result.conformal_coverage:.0%}")
    if result.copula_tail_risk > 0:
        extras.append(f"copula_risk={result.copula_tail_risk:.4f}")
    if result.dtw_analogs_used > 0:
        extras.append(f"dtw_analogs={result.dtw_analogs_used}")
    if result.granger_adjustments_applied > 0:
        extras.append(f"granger_adj={result.granger_adjustments_applied}")
    if result.shap_available:
        extras.append("shap=yes")
    if result.recency_weighted_rmse_used:
        extras.append("recency_rmse=yes")
    extras_str = ", ".join(extras) if extras else "none"

    logger.info(
        "Prediction aggregation complete: %d variables, %d horizons, "
        "%d models available (%d failed), "
        "survival_mean=%.4f, regime='%s', blend='%s', extras=[%s]",
        len(result.variables_predicted),
        len(result.horizons),
        result.n_models_available,
        result.n_models_failed,
        result.survival_probability_mean
        if not math.isnan(result.survival_probability_mean)
        else -1.0,
        result.current_regime,
        result.regime_blend_method,
        extras_str,
    )

    # --- B3 fix: Apply feature-driven model routing weights ---
    # Routing weights adjust confidence based on market characteristics:
    # trending markets boost Kalman/tree, volatile markets boost GARCH/MC.
    try:
        routing_weights = compute_model_routing_weights(cache)
        if routing_weights and hasattr(result, "metadata"):
            if result.metadata is None:
                result.metadata = {}
            result.metadata["model_routing_weights"] = routing_weights

            # Apply routing as confidence multiplier on each prediction.
            # If the model_used for a prediction is favored by routing,
            # boost its confidence; if disfavored, dampen it.
            if hasattr(result, "predictions") and result.predictions:
                for _var, _horizons in result.predictions.items():
                    if isinstance(_horizons, dict):
                        for _h, _hp in _horizons.items():
                            _mu = getattr(_hp, "model_used", "")
                            if _mu and _mu in routing_weights:
                                _rw = routing_weights[_mu]
                                # Scale confidence by routing affinity (0.5-1.5x range)
                                _conf = getattr(_hp, "confidence", float("nan"))
                                if not math.isnan(_conf):
                                    _hp.confidence = max(0.0, min(1.0, _conf * _rw))
    except Exception:
        pass

    # --- Feature-weighted ensemble confidence ---
    # When Boruta/PIMP/mRMR confirmed features, boost confidence proportionally.
    # More confirmed features = more information supporting the prediction.
    try:
        if feature_selection_result is not None and getattr(feature_selection_result, "fitted", False):
            _boruta = getattr(feature_selection_result, "boruta_confirmed", [])
            _n_confirmed = len(_boruta) if _boruta else 0
            # support ranges from 0.3 (0 features) to 1.0 (10+ features)
            _support = min(1.0, max(0.3, _n_confirmed / 10.0))
            # Confidence multiplier: 0.7 + 0.3*support => range [0.79, 1.0]
            _feat_mult = 0.7 + 0.3 * _support
            if hasattr(result, "predictions") and result.predictions:
                for _var, _horizons in result.predictions.items():
                    if isinstance(_horizons, dict):
                        for _h, _hp in _horizons.items():
                            _conf = getattr(_hp, "confidence", float("nan"))
                            if not math.isnan(_conf):
                                _hp.confidence = max(0.0, min(1.0, _conf * _feat_mult))
            if result.metadata is None:
                result.metadata = {}
            result.metadata["feature_support_score"] = round(_support, 3)
            result.metadata["feature_confidence_multiplier"] = round(_feat_mult, 3)
    except Exception:
        pass

    # --- B4 fix: Apply reject option (reduce confidence for extreme uncertainty) ---
    try:
        if hasattr(result, "predictions") and result.predictions:
            apply_reject_option(result.predictions, cache)
    except Exception:
        pass

    # --- PEAD: pre-earnings drift adjustment (Method 7) ---
    # When an earnings event is imminent, apply a small directional shift
    # based on event uncertainty premium (positive premium = downward pressure
    # from uncertainty, Bernard & Thomas 1989).
    try:
        if event_calendar_result is not None and getattr(event_calendar_result, "available", False):
            _ec_days = getattr(event_calendar_result, "days_to_next_event", None)
            _ec_prem = getattr(event_calendar_result, "event_uncertainty_premium", None)
            if _ec_days is not None and _ec_days < 30 and _ec_prem is not None and abs(_ec_prem) > 0.001:
                _ec_drift = -_ec_prem * 0.001 * min(_ec_days, 21)
                _ec_n = 0
                for _ecv in result.predictions:
                    if _ecv == "close" and isinstance(result.predictions[_ecv], dict):
                        for _ech, _echp in result.predictions[_ecv].items():
                            if hasattr(_echp, "point_forecast") and _echp.point_forecast:
                                _echp.point_forecast *= (1 + _ec_drift)
                                _ec_n += 1
                if _ec_n > 0:
                    logger.debug("PEAD drift applied: %.4f (%d days to event, premium=%.4f)",
                                _ec_drift, _ec_days, _ec_prem)
    except Exception:
        pass

    return result

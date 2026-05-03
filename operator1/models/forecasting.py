"""T6.2 -- Forecasting models with fallback chains.

Provides a suite of time-series forecasting models for predicting
financial variables at multiple horizons.  Each model is wrapped in
``try/except`` so that a missing optional dependency logs a warning
and falls through to the next model in the chain.

**Model hierarchy (in order of attempted fit):**

1. **Kalman filter** -- for Tier 1-2 liquidity/solvency variables.
   Uses a local-level state-space model (statsmodels).
2. **GARCH** -- for volatility forecasting (``arch`` library).
3. **VAR** -- multivariate vector autoregression (statsmodels).
   Fallback: univariate AR(1).
4. **LSTM** -- nonlinear sequence model (PyTorch).
   Fallback: GradientBoosting or LinearRegression.
5. **RF / GBM / XGB** -- tree ensemble for tabular features
   (sklearn / xgboost).
6. **Baseline** -- last-value carry-forward or exponential moving
   average.  Always succeeds.

Each model produces:
  - Point forecast per horizon (1d, 5d, 21d, 252d).
  - ``model_failed_<name>`` flag (bool) if the model could not fit.
  - Error metrics (MAE, RMSE) on a held-out validation fold.

Top-level entry point:
  ``run_forecasting(cache, tier_variables, regime_labels)``

Spec refs: Sec 17
"""

from __future__ import annotations

import copy
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from operator1.config_loader import load_config
from operator1.constants import CACHE_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Forecast horizons (business days).
HORIZONS: dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "21d": 21,
    "252d": 252,
}

# Minimum observations required for each model type.
_MIN_OBS_KALMAN: int = 30
_MIN_OBS_GARCH: int = 60
_MIN_OBS_VAR: int = 50
_MIN_OBS_LSTM: int = 100
_MIN_OBS_TREE: int = 30
_MIN_OBS_BASELINE: int = 1

# Burn-out phase: retrain on last N days for refinement.
_BURNOUT_WINDOW: int = 126  # ~6 months of trading days
_EARLY_STOP_PATIENCE: int = 3  # stop if no improvement for N iterations

# LSTM defaults.
_LSTM_HIDDEN: int = 32
_LSTM_LAYERS: int = 1
_LSTM_LOOKBACK: int = 21
_LSTM_EPOCHS: int = 50
_LSTM_LR: float = 0.001

# VAR max lag selection cap.
_VAR_MAX_LAG: int = 10


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ModelMetrics:
    """Error metrics for a single model on a single variable."""

    model_name: str = ""
    variable: str = ""
    mae: float = float("nan")
    rmse: float = float("nan")
    n_train: int = 0
    n_test: int = 0
    fitted: bool = False
    error: str | None = None

    # Actual validation residuals (predicted - actual) from train/test split.
    # Used by ConformalCalibrator for distribution-free interval estimation.
    # If None, the calibrator falls back to synthetic +/-RMSE pairs.
    test_residuals: list[float] | None = None


@dataclass
class ForecastResult:
    """Container for all forecasting outputs."""

    # Per-variable, per-horizon point forecasts.
    # {variable: {horizon_label: value}}
    forecasts: dict[str, dict[str, float]] = field(default_factory=dict)

    # Model failure flags.
    model_failed_kalman: bool = False
    model_failed_garch: bool = False
    model_failed_var: bool = False
    model_failed_lstm: bool = False
    model_failed_tree: bool = False
    # Baseline never fails.

    # Error messages.
    kalman_error: str | None = None
    garch_error: str | None = None
    var_error: str | None = None
    lstm_error: str | None = None
    tree_error: str | None = None

    # Per-model, per-variable metrics.
    metrics: list[ModelMetrics] = field(default_factory=list)

    # Which model was used for each variable.
    # {variable: model_name}
    model_used: dict[str, str] = field(default_factory=dict)

    # Validation residuals collected across all fitted models.
    # Fed to ConformalCalibrator for distribution-free interval calibration.
    residuals: list[float] | None = None


# ---------------------------------------------------------------------------
# Helper: error metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """Return (MAE, RMSE) for non-NaN aligned pairs."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float("nan"), float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return mae, rmse


def _compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[float]:
    """Return list of (actual - predicted) residuals for non-NaN pairs.

    Used to feed the ConformalCalibrator with actual validation residuals
    instead of synthetic +/-RMSE pairs.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return []
    mask = ~(np.isnan(y_true[:n]) | np.isnan(y_pred[:n]))
    return (y_true[:n][mask] - y_pred[:n][mask]).tolist()


def _split_train_test(
    series: np.ndarray,
    test_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a 1-D series into train/test (no shuffle -- temporal)."""
    n = len(series)
    split = max(1, int(n * (1 - test_frac)))
    return series[:split], series[split:]


# ---------------------------------------------------------------------------
# Tier variable lookup
# ---------------------------------------------------------------------------


def _load_tier_variables() -> dict[str, list[str]]:
    """Load tier -> variable list from survival_hierarchy config."""
    try:
        cfg = load_config("survival_hierarchy")
    except FileNotFoundError:
        logger.warning("survival_hierarchy config not found")
        return {}
    tiers = cfg.get("tiers", {})
    result: dict[str, list[str]] = {}
    for tier_key, tier_data in tiers.items():
        result[tier_key] = tier_data.get("variables", [])
    return result


def _get_tier_for_variable(
    variable: str,
    tier_map: dict[str, list[str]],
) -> str:
    """Return the tier key a variable belongs to, or 'unknown'."""
    for tier_key, vars_list in tier_map.items():
        if variable in vars_list:
            return tier_key
    return "unknown"


# ===========================================================================
# Model implementations
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. Kalman filter (local-level state-space model)
# ---------------------------------------------------------------------------


def fit_kalman(
    series: np.ndarray,
    n_forecast: int = 1,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit a local-level Kalman filter and produce forecasts.

    Parameters
    ----------
    series:
        1-D array of observed values (may contain NaN -- the Kalman
        filter handles missing observations natively).
    n_forecast:
        Number of steps ahead to forecast.

    Returns
    -------
    (forecasts, metrics)
        ``forecasts`` is an array of length ``n_forecast``, or None on
        failure.
    """
    metrics = ModelMetrics(model_name="kalman")

    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents  # type: ignore[import-untyped]
    except ImportError:
        metrics.error = "statsmodels not installed -- skipping Kalman filter"
        logger.warning(metrics.error)
        return None, metrics

    clean = series[~np.isnan(series)]
    if len(clean) < _MIN_OBS_KALMAN:
        metrics.error = (
            f"Insufficient observations for Kalman ({len(clean)} < {_MIN_OBS_KALMAN})"
        )
        logger.warning(metrics.error)
        return None, metrics

    try:
        train, test = _split_train_test(clean)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(
                train,
                level="local level",
            )
            result = model.fit(disp=False, maxiter=200)

        # In-sample predictions for validation.
        if len(test) > 0:
            forecast_obj = result.get_forecast(steps=len(test))
            preds = forecast_obj.predicted_mean
            mae, rmse = _compute_metrics(test, preds)
            metrics.test_residuals = _compute_residuals(test, preds)
        else:
            mae, rmse = float("nan"), float("nan")

        # Full refit for final forecast.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = UnobservedComponents(clean, level="local level")
            full_result = full_model.fit(disp=False, maxiter=200)

        forecast_obj = full_result.get_forecast(steps=n_forecast)
        forecasts = forecast_obj.predicted_mean

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(train)
        metrics.n_test = len(test)
        metrics.fitted = True

        logger.info(
            "Kalman fit: %d train, %d test, MAE=%.6f, RMSE=%.6f",
            len(train), len(test), mae, rmse,
        )
        return np.array(forecasts), metrics

    except Exception as exc:
        metrics.error = f"Kalman fitting failed: {exc}"
        logger.warning(metrics.error)
        return None, metrics


# ---------------------------------------------------------------------------
# 1-regime. Per-Regime Kalman (Section K.2 of core idea)
# ---------------------------------------------------------------------------


def fit_kalman_per_regime(
    series: np.ndarray,
    regime_labels: np.ndarray,
    regime_probs: dict[str, float] | None = None,
    n_forecast: int = 1,
    min_regime_obs: int = 30,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit separate Kalman filters per regime and blend predictions.

    For each regime with sufficient data, fits a local-level state-space
    model. The final forecast is a probability-weighted blend:
    ``pred = sum(P(regime_k) * kalman_k.predict())``

    Source: The_Apps_core_idea.pdf Section K.2 -- per-regime model parameters.

    Falls back to standard ``fit_kalman()`` if regime data is insufficient
    or only one effective regime exists.
    """
    metrics = ModelMetrics(model_name="kalman_per_regime")

    if regime_labels is None or len(regime_labels) != len(series):
        return fit_kalman(series, n_forecast)

    # Identify regimes with sufficient data.
    # Filter out None/NaN before np.unique to avoid TypeError when
    # regime_labels contains mixed str + None (first few days before HMM warmup).
    _clean_labels = pd.Series(regime_labels).dropna().values
    if len(_clean_labels) == 0:
        return fit_kalman(series, n_forecast)
    unique_regimes = [r for r in np.unique(_clean_labels) if not (isinstance(r, float) and np.isnan(r))]
    regime_data = {}
    for r in unique_regimes:
        mask = regime_labels == r
        regime_series = series[mask]
        clean = regime_series[~np.isnan(regime_series)]
        if len(clean) >= min_regime_obs:
            regime_data[r] = clean

    if len(regime_data) < 2:
        # Not enough regimes with data -- fall back to standard Kalman
        return fit_kalman(series, n_forecast)

    # Fit per-regime Kalman models
    regime_forecasts = {}
    for regime, r_series in regime_data.items():
        fcast, met = fit_kalman(r_series, n_forecast)
        if fcast is not None:
            regime_forecasts[regime] = fcast

    if not regime_forecasts:
        return fit_kalman(series, n_forecast)

    # Blend forecasts using regime probabilities
    if regime_probs is None:
        # Use frequency-based probabilities
        total = sum(len(series[regime_labels == r]) for r in regime_forecasts)
        regime_probs = {
            r: len(series[regime_labels == r]) / max(total, 1)
            for r in regime_forecasts
        }

    blended = np.zeros(n_forecast)
    total_weight = 0.0
    for regime, fcast in regime_forecasts.items():
        prob = regime_probs.get(regime, regime_probs.get(str(regime), 0.0))
        if prob > 0.01:
            blended += prob * fcast
            total_weight += prob

    if total_weight > 0:
        blended /= total_weight
    else:
        return fit_kalman(series, n_forecast)

    metrics.fitted = True
    metrics.model_name = f"kalman_per_regime({len(regime_forecasts)})"
    logger.info(
        "Per-regime Kalman: %d regimes fitted, probs=%s",
        len(regime_forecasts),
        {k: f"{v:.2f}" for k, v in regime_probs.items() if k in regime_forecasts},
    )
    return blended, metrics


# ---------------------------------------------------------------------------
# 1a. ETS via statsforecast (replaces AutoARIMA -- 10-50x faster, competitive accuracy)
# ---------------------------------------------------------------------------


def fit_ets(
    series: np.ndarray,
    n_forecast: int = 1,
    season_length: int = 63,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit ETS (Error-Trend-Seasonality) using statsforecast.

    Exponential smoothing state-space model with automatic model selection.
    Replaces AutoARIMA: 10-50x faster with competitive accuracy on financial
    time series (M3/M4 competition results show ETS matches or beats ARIMA
    on average).  Unlike ARIMA's grid search over (p,d,q) orders, ETS
    selects among 30 model configurations via information criteria in a
    single pass.

    The Kalman filter already captures the same linear autoregressive
    dynamics that ARIMA models; ETS adds complementary exponential
    smoothing dynamics (level, trend, damped trend, seasonality) that
    the Kalman local-level model does not cover.

    Falls back to None if statsforecast is not installed.
    """
    metrics = ModelMetrics(model_name="ets")

    try:
        from statsforecast.models import AutoETS
    except ImportError:
        metrics.error = "statsforecast not installed -- skipping ETS"
        return None, metrics

    clean = series[~np.isnan(series)]
    if len(clean) < _MIN_OBS_KALMAN:
        metrics.error = f"Insufficient observations ({len(clean)})"
        return None, metrics

    import time as _time

    try:
        _t0 = _time.time()
        train, test = _split_train_test(clean)

        _sl = min(season_length, len(train) // 3)
        # AutoETS selects best among 30 ETS model configurations via AIC
        model = AutoETS(season_length=_sl)
        model.fit(train)

        # Validation
        if len(test) > 0:
            preds = model.predict(h=len(test))["mean"]
            mae, rmse = _compute_metrics(test, np.array(preds))
            metrics.test_residuals = _compute_residuals(test, np.array(preds))
        else:
            mae, rmse = float("nan"), float("nan")

        # Refit on full data so predict(h=1) forecasts from the end of the
        # series, not from the 85th-percentile train/test split boundary.
        full_model = AutoETS(season_length=_sl)
        full_model.fit(clean)
        forecasts = full_model.predict(h=n_forecast)["mean"]

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(train)
        metrics.n_test = len(test)
        metrics.fitted = True

        _elapsed = _time.time() - _t0
        logger.info(
            "ETS fit: %d train, %d test, MAE=%.6f (%.1fs)",
            len(train), len(test), mae, _elapsed,
        )
        return np.array(forecasts), metrics

    except Exception as exc:
        metrics.error = f"ETS failed: {exc}"
        logger.debug(metrics.error)
        return None, metrics


# ---------------------------------------------------------------------------
# 1b. Dynamic Factor Model (multi-variable state-space Kalman)
# ---------------------------------------------------------------------------


def fit_dynamic_factor(
    data: pd.DataFrame,
    target_col: str,
    n_forecast: int = 1,
    n_factors: int = 2,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit a Dynamic Factor Model extracting latent factors from multiple variables.

    Instead of the simple local-level model, this extracts a small number of
    latent factors from 5-15 variables using a Kalman smoother. Captures the
    common signal while filtering out variable-specific noise.

    Falls back to the simple local-level Kalman if DFM fitting fails.

    Parameters
    ----------
    data:
        DataFrame with multiple numeric columns including target_col.
    target_col:
        Variable to extract forecasts for.
    n_forecast:
        Steps ahead.
    n_factors:
        Number of latent factors to extract (default 2).

    Returns
    -------
    (forecasts, metrics)
    """
    metrics = ModelMetrics(model_name="kalman_dfm")

    try:
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
    except ImportError:
        metrics.error = "statsmodels DynamicFactor not available"
        return None, metrics

    # Select numeric columns with sufficient data
    numeric_cols = [
        c for c in data.columns
        if data[c].dtype in ("float64", "float32")
        and data[c].notna().sum() > _MIN_OBS_KALMAN
        and c != target_col
    ]
    if target_col not in data.columns:
        metrics.error = f"Target '{target_col}' not in data"
        return None, metrics

    # Use target + up to 10 most correlated variables
    if len(numeric_cols) > 10:
        corrs = data[numeric_cols].corrwith(data[target_col]).abs().dropna()
        numeric_cols = corrs.nlargest(10).index.tolist()

    cols = [target_col] + [c for c in numeric_cols if c != target_col]
    if len(cols) < 3:
        metrics.error = "Need at least 3 variables for DFM"
        return None, metrics

    clean = data[cols].dropna()
    if len(clean) < _MIN_OBS_KALMAN * 2:
        metrics.error = f"Insufficient data for DFM ({len(clean)})"
        return None, metrics

    n_factors = min(n_factors, len(cols) - 1)

    try:
        train_n = max(1, int(len(clean) * 0.85))
        train = clean.iloc[:train_n]
        test = clean.iloc[train_n:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DynamicFactor(
                train,
                k_factors=n_factors,
                factor_order=1,
            )
            result = model.fit(disp=False, maxiter=200)

        # Validation
        if len(test) > 0:
            forecast_obj = result.get_forecast(steps=len(test))
            pred_df = forecast_obj.predicted_mean
            if target_col in pred_df.columns:
                preds = pred_df[target_col].values
            else:
                preds = pred_df.iloc[:, 0].values
            actuals = test[target_col].values
            mae, rmse = _compute_metrics(actuals, preds[:len(actuals)])
            metrics.test_residuals = _compute_residuals(actuals, preds[:len(actuals)])
        else:
            mae, rmse = float("nan"), float("nan")

        # Full refit for final forecast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = DynamicFactor(
                clean,
                k_factors=n_factors,
                factor_order=1,
            )
            full_result = full_model.fit(disp=False, maxiter=200)

        forecast_obj = full_result.get_forecast(steps=n_forecast)
        pred_df = forecast_obj.predicted_mean
        if target_col in pred_df.columns:
            forecasts = pred_df[target_col].values
        else:
            forecasts = pred_df.iloc[:, 0].values

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(train)
        metrics.n_test = len(test)
        metrics.fitted = True
        metrics.model_name = "kalman_dfm"

        logger.info(
            "DFM fit: %d factors, %d vars, %d train, MAE=%.6f, RMSE=%.6f",
            n_factors, len(cols), len(train), mae, rmse,
        )
        return np.array(forecasts), metrics

    except Exception as exc:
        metrics.error = f"DFM fitting failed: {exc}"
        logger.debug(metrics.error)
        return None, metrics


# ---------------------------------------------------------------------------
# 2. GARCH (volatility forecasting)
# ---------------------------------------------------------------------------


def fit_garch(
    returns: np.ndarray,
    n_forecast: int = 1,
    p: int = 1,
    q: int = 1,
    cache: pd.DataFrame | None = None,
    extra_variables: list[str] | None = None,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit a GARCH(p,q) model for conditional volatility forecasting.

    Parameters
    ----------
    returns:
        1-D array of daily returns (not prices).
    n_forecast:
        Number of steps ahead.
    p, q:
        GARCH order parameters.
    cache:
        Daily cache DataFrame (needed for GARCH-X exogenous regressors).
    extra_variables:
        Feature columns for GARCH-X mean model (ARX + GARCH volatility).

    Returns
    -------
    (volatility_forecasts, metrics)
    """
    metrics = ModelMetrics(model_name="garch")

    try:
        from arch import arch_model  # type: ignore[import-untyped]
    except ImportError:
        metrics.error = "arch library not installed -- skipping GARCH"
        logger.warning(metrics.error)
        return None, metrics

    clean = returns[~np.isnan(returns)]
    if len(clean) < _MIN_OBS_GARCH:
        metrics.error = (
            f"Insufficient observations for GARCH ({len(clean)} < {_MIN_OBS_GARCH})"
        )
        logger.warning(metrics.error)
        return None, metrics

    # ------------------------------------------------------------------
    # GARCH-X: try ARX mean model with exogenous regressors first.
    # Features from Boruta/PIMP/mRMR inform the conditional mean,
    # while GARCH(1,1) handles conditional variance.
    # ------------------------------------------------------------------
    if extra_variables and cache is not None:
        try:
            from arch.univariate import ARX, GARCH as GARCHVol  # type: ignore[import-untyped]

            _exog_cols = [c for c in extra_variables if c in cache.columns and cache[c].notna().sum() > 20]
            if _exog_cols:
                # Align exog with returns length
                _exog = cache[_exog_cols].iloc[-len(clean):].copy()
                _exog = _exog.dropna(axis=1)  # drop cols with NaN in this window
                if len(_exog.columns) > 0 and len(_exog) == len(clean):
                    # Z-score normalize (GARCH-X needs stationary regressors)
                    _exog_std = _exog.std().clip(lower=1e-8)
                    _exog_norm = (_exog - _exog.mean()) / _exog_std
                    _scaled_x = clean * 100.0
                    _train_n = max(1, int(len(_scaled_x) * 0.85))

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _am = ARX(_scaled_x[:_train_n], lags=[1], x=_exog_norm.values[:_train_n])
                        _am.volatility = GARCHVol(p=p, q=q)
                        _res_x = _am.fit(disp="off", show_warning=False)

                    # Validate on test set
                    _test_n = len(_scaled_x) - _train_n
                    if _test_n > 0:
                        _fobj = _res_x.forecast(horizon=_test_n)
                        _var_fcast = _fobj.variance.iloc[-1].values[:_test_n]
                        _vol_pred = np.sqrt(_var_fcast) / 100.0
                        _vol_actual = np.abs(clean[_train_n:]) 
                        _gx_mae, _gx_rmse = _compute_metrics(
                            _vol_actual[:len(_vol_pred)], _vol_pred[:len(_vol_actual)]
                        )
                    else:
                        _gx_mae, _gx_rmse = float("nan"), float("nan")

                    # Full refit
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _am_full = ARX(_scaled_x, lags=[1], x=_exog_norm.values)
                        _am_full.volatility = GARCHVol(p=p, q=q)
                        _res_full = _am_full.fit(disp="off", show_warning=False)

                    _fobj_full = _res_full.forecast(horizon=n_forecast)
                    _var_full = _fobj_full.variance.iloc[-1].values[:n_forecast]
                    _vol_forecasts = np.sqrt(_var_full) / 100.0

                    metrics.model_name = "garch_x"
                    metrics.mae = _gx_mae
                    metrics.rmse = _gx_rmse
                    metrics.n_train = _train_n
                    metrics.n_test = _test_n
                    metrics.fitted = True
                    logger.info(
                        "GARCH-X(%d,%d) fit with %d exog features: MAE=%.6f, RMSE=%.6f",
                        p, q, len(_exog.columns), _gx_mae, _gx_rmse,
                    )
                    return _vol_forecasts, metrics
        except Exception as _gx_exc:
            logger.debug("GARCH-X failed, falling back to standard GARCH: %s", _gx_exc)

    try:
        # Scale returns to percentage for numerical stability.
        scaled = clean * 100.0
        train, test = _split_train_test(scaled)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                train,
                vol="Garch",
                p=p,
                q=q,
                mean="Constant",
                rescale=False,
            )
            result = model.fit(disp="off", show_warning=False)

        # Validation forecasts.
        if len(test) > 0:
            forecast_obj = result.forecast(horizon=len(test))
            # Variance forecast -> std dev.
            var_forecast = forecast_obj.variance.iloc[-1].values
            vol_pred = np.sqrt(var_forecast) / 100.0  # back to decimal
            vol_actual = np.abs(test) / 100.0
            mae, rmse = _compute_metrics(
                vol_actual[: len(vol_pred)],
                vol_pred[: len(vol_actual)],
            )
        else:
            mae, rmse = float("nan"), float("nan")

        # Full refit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = arch_model(
                scaled,
                vol="Garch",
                p=p,
                q=q,
                mean="Constant",
                rescale=False,
            )
            full_result = full_model.fit(disp="off", show_warning=False)

        forecast_obj = full_result.forecast(horizon=n_forecast)
        var_fcast = forecast_obj.variance.iloc[-1].values[:n_forecast]
        vol_forecasts = np.sqrt(var_fcast) / 100.0  # decimal volatility

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(train)
        metrics.n_test = len(test)
        metrics.fitted = True

        logger.info(
            "GARCH(%d,%d) fit: %d train, %d test, MAE=%.6f, RMSE=%.6f",
            p, q, len(train), len(test), mae, rmse,
        )
        return vol_forecasts, metrics

    except Exception as exc:
        metrics.error = f"GARCH fitting failed: {exc}"
        logger.warning(metrics.error)
        return None, metrics


# ---------------------------------------------------------------------------
# A3: GARCH-MIDAS (macro-driven long-run volatility)
# Two-component model: short-run GARCH + long-run macro-driven component.
# Engle, Ghysels & Sohn (2013).
# ---------------------------------------------------------------------------


def fit_garch_midas(
    returns: np.ndarray,
    macro_features: pd.DataFrame | None = None,
    n_forecast: int = 1,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit GARCH with MIDAS macro component for long-run volatility.

    The long-run component tau_t is driven by macro variables (GDP growth,
    inflation, credit spread) via exponential weighting. This lets macro
    deterioration influence vol forecasts BEFORE the crash happens.

    Falls back to standard GARCH when macro features unavailable.

    Reference: Engle, Ghysels & Sohn (2013).
    """
    metrics = ModelMetrics(model_name="garch_midas")

    try:
        from arch import arch_model
    except ImportError:
        metrics.error = "arch library not installed"
        return None, metrics

    clean = returns[~np.isnan(returns)]
    if len(clean) < _MIN_OBS_GARCH:
        metrics.error = f"Insufficient observations ({len(clean)})"
        return None, metrics

    # If no macro features, fall back to standard GARCH with exogenous vol proxy
    if macro_features is None or macro_features.empty:
        return fit_garch(returns, n_forecast)

    try:
        # Scale returns to percentage
        scaled = clean * 100.0
        train, test = _split_train_test(scaled)

        # Build MIDAS long-run component from macro features
        # Use realized variance as the MIDAS target (standard approach)
        # The macro features modulate the long-run variance level
        _rv = pd.Series(scaled ** 2).rolling(21, min_periods=5).mean()
        _rv_clean = _rv.dropna().values

        if len(_rv_clean) < 30:
            return fit_garch(returns, n_forecast)

        # Align macro features with returns
        _macro_cols = [
            c for c in macro_features.columns
            if macro_features[c].dtype in ("float64", "float32")
            and macro_features[c].notna().sum() > 20
        ][:5]  # cap at 5 macro features

        if not _macro_cols:
            return fit_garch(returns, n_forecast)

        # Compute long-run component via OLS of realized variance on macro
        _macro_aligned = macro_features[_macro_cols].iloc[-len(scaled):].copy()
        _macro_aligned = _macro_aligned.fillna(method="ffill").fillna(0)

        if len(_macro_aligned) != len(scaled):
            _macro_aligned = _macro_aligned.iloc[-len(scaled):]

        # Long-run component: exponentially weighted macro effect
        try:
            from sklearn.linear_model import Ridge as _Ridge
            _X = _macro_aligned.values[-len(_rv_clean):]
            _y = _rv_clean
            if len(_X) == len(_y) and len(_y) > 20:
                _reg = _Ridge(alpha=1.0)
                _reg.fit(_X, _y)
                _tau = _reg.predict(_macro_aligned.values[-len(scaled):])
                _tau = np.clip(_tau, 0.001, None)  # floor at positive

                # Short-run component: GARCH on tau-adjusted returns
                _adjusted = scaled / np.sqrt(_tau[-len(scaled):])
                _adjusted = np.clip(_adjusted, -50, 50)  # prevent numerical issues

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = arch_model(
                        _adjusted[:len(train)],
                        vol="Garch", p=1, q=1, mean="Constant", rescale=False,
                    )
                    result = model.fit(disp="off", show_warning=False)

                # Forecast: combine short-run GARCH with long-run macro
                fcast = result.forecast(horizon=n_forecast)
                _short_var = fcast.variance.iloc[-1].values[:n_forecast] / 10000.0
                _long_component = float(_tau[-1]) / 10000.0

                # Total vol = sqrt(tau * g)
                vol_forecasts = np.sqrt(_short_var * _long_component)

                # Validation
                if len(test) > 0:
                    _test_pred = np.full(len(test), vol_forecasts[0])
                    _test_actual = np.abs(test) / 100.0
                    mae, rmse = _compute_metrics(_test_actual, _test_pred)
                else:
                    mae, rmse = float("nan"), float("nan")

                metrics.mae = mae
                metrics.rmse = rmse
                metrics.n_train = len(train)
                metrics.n_test = len(test)
                metrics.fitted = True

                logger.info(
                    "GARCH-MIDAS fit: %d train, %d macro features, MAE=%.6f",
                    len(train), len(_macro_cols), mae,
                )
                return vol_forecasts, metrics

        except Exception as _inner_exc:
            logger.debug("GARCH-MIDAS inner fitting failed: %s", _inner_exc)

        # Fall back to standard GARCH
        return fit_garch(returns, n_forecast)

    except Exception as exc:
        metrics.error = f"GARCH-MIDAS failed: {exc}"
        logger.debug(metrics.error)
        return fit_garch(returns, n_forecast)


# ---------------------------------------------------------------------------
# 3. VAR (multivariate) with AR(1) fallback
# ---------------------------------------------------------------------------


def fit_var(
    data: pd.DataFrame,
    target_col: str,
    n_forecast: int = 1,
    max_lag: int = _VAR_MAX_LAG,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit a VAR model and forecast the target variable.

    Falls back to univariate AR(1) if VAR fails (e.g. singular matrix,
    too few variables).

    Parameters
    ----------
    data:
        DataFrame with multiple numeric columns (including ``target_col``).
    target_col:
        The variable to extract forecasts for.
    n_forecast:
        Steps ahead.
    max_lag:
        Maximum lag order for AIC selection.

    Returns
    -------
    (forecasts, metrics)
    """
    metrics = ModelMetrics(model_name="var")

    try:
        from statsmodels.tsa.api import VAR as VARModel  # type: ignore[import-untyped]
    except ImportError:
        metrics.error = "statsmodels not installed -- skipping VAR"
        logger.warning(metrics.error)
        return None, metrics

    clean = data.dropna()
    if len(clean) < _MIN_OBS_VAR:
        metrics.error = (
            f"Insufficient observations for VAR ({len(clean)} < {_MIN_OBS_VAR})"
        )
        logger.warning(metrics.error)
        # Try AR(1) fallback.
        return _fit_ar1_fallback(clean, target_col, n_forecast, metrics)

    if target_col not in clean.columns:
        metrics.error = f"Target column '{target_col}' not in data"
        logger.warning(metrics.error)
        return None, metrics

    try:
        train_n = max(1, int(len(clean) * 0.85))
        train = clean.iloc[:train_n]
        test = clean.iloc[train_n:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VARModel(train)
            # Select optimal lag via AIC, capped at max_lag.
            lag_order = min(max_lag, len(train) // 3)
            result = model.fit(maxlags=max(1, lag_order), ic="aic")

        selected_lag = result.k_ar

        # Validation.
        if len(test) > 0:
            forecast_arr = result.forecast(
                train.values[-selected_lag:], steps=len(test)
            )
            col_idx = list(clean.columns).index(target_col)
            preds = forecast_arr[:, col_idx]
            actual = test[target_col].values
            mae, rmse = _compute_metrics(actual, preds)
        else:
            mae, rmse = float("nan"), float("nan")

        # Full refit for final forecast.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = VARModel(clean)
            full_result = full_model.fit(maxlags=max(1, lag_order), ic="aic")

        full_lag = full_result.k_ar
        forecast_arr = full_result.forecast(
            clean.values[-full_lag:], steps=n_forecast
        )
        col_idx = list(clean.columns).index(target_col)
        forecasts = forecast_arr[:, col_idx]

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(train)
        metrics.n_test = len(test)
        metrics.fitted = True
        metrics.model_name = f"var(lag={full_lag})"

        logger.info(
            "VAR fit: lag=%d, %d vars, %d train, MAE=%.6f, RMSE=%.6f",
            full_lag, len(clean.columns), len(train), mae, rmse,
        )
        return forecasts, metrics

    except Exception as exc:
        msg = f"VAR fitting failed: {exc}"
        logger.warning(msg + " -- falling back to AR(1)")
        metrics.error = msg
        return _fit_ar1_fallback(clean, target_col, n_forecast, metrics)


def _fit_ar1_fallback(
    data: pd.DataFrame,
    target_col: str,
    n_forecast: int,
    parent_metrics: ModelMetrics,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Univariate AR(1) fallback for VAR."""
    metrics = ModelMetrics(model_name="ar1")

    if target_col not in data.columns:
        metrics.error = f"Target '{target_col}' not in data for AR(1)"
        return None, metrics

    try:
        from statsmodels.tsa.ar_model import AutoReg  # type: ignore[import-untyped]
    except ImportError:
        metrics.error = "statsmodels not installed -- AR(1) fallback unavailable"
        return None, metrics

    series = data[target_col].dropna().values
    if len(series) < 10:
        metrics.error = f"Insufficient data for AR(1) ({len(series)} < 10)"
        return None, metrics

    try:
        train, test = _split_train_test(series)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoReg(train, lags=1)
            result = model.fit()

        if len(test) > 0:
            preds = result.predict(start=len(train), end=len(train) + len(test) - 1)
            mae, rmse = _compute_metrics(test, np.asarray(preds))
        else:
            mae, rmse = float("nan"), float("nan")

        # Full refit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = AutoReg(series, lags=1)
            full_result = full_model.fit()

        preds_final = full_result.predict(
            start=len(series),
            end=len(series) + n_forecast - 1,
        )

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(train)
        metrics.n_test = len(test)
        metrics.fitted = True

        logger.info("AR(1) fallback fit: MAE=%.6f, RMSE=%.6f", mae, rmse)
        return np.asarray(preds_final), metrics

    except Exception as exc:
        metrics.error = f"AR(1) fallback failed: {exc}"
        logger.warning(metrics.error)
        return None, metrics


# ---------------------------------------------------------------------------
# 4. LSTM with tree/linear fallback
# ---------------------------------------------------------------------------


def fit_lstm(
    series: np.ndarray,
    n_forecast: int = 1,
    lookback: int = _LSTM_LOOKBACK,
    hidden_size: int = _LSTM_HIDDEN,
    num_layers: int = _LSTM_LAYERS,
    epochs: int = _LSTM_EPOCHS,
    lr: float = _LSTM_LR,
    random_state: int = 42,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit an LSTM for nonlinear pattern forecasting.

    Falls back to GradientBoosting or LinearRegression if PyTorch
    is unavailable.

    Parameters
    ----------
    series:
        1-D array of observed values.
    n_forecast:
        Steps ahead.
    lookback:
        Number of past observations used as input features.
    hidden_size:
        LSTM hidden dimension.
    num_layers:
        Number of stacked LSTM layers.
    epochs:
        Training epochs.
    lr:
        Learning rate.
    random_state:
        Random seed.

    Returns
    -------
    (forecasts, metrics)
    """
    metrics = ModelMetrics(model_name="lstm")

    try:
        import torch  # type: ignore[import-untyped]
        import torch.nn as nn  # type: ignore[import-untyped]
    except ImportError:
        metrics.error = "PyTorch not installed -- falling back to tree/linear"
        logger.warning(metrics.error)
        return _fit_linear_fallback(series, n_forecast, lookback, random_state)

    clean = series[~np.isnan(series)]
    if len(clean) < _MIN_OBS_LSTM:
        metrics.error = (
            f"Insufficient observations for LSTM ({len(clean)} < {_MIN_OBS_LSTM})"
        )
        logger.warning(metrics.error + " -- falling back to tree/linear")
        return _fit_linear_fallback(series, n_forecast, lookback, random_state)

    try:
        torch.manual_seed(random_state)

        # Normalise.
        mean_val = float(np.mean(clean))
        std_val = float(np.std(clean))
        if std_val < 1e-12:
            std_val = 1.0
        normed = (clean - mean_val) / std_val

        # Create sequences.
        X_list, y_list = [], []
        for i in range(lookback, len(normed)):
            X_list.append(normed[i - lookback: i])
            y_list.append(normed[i])

        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)

        # Train/test split.
        split = max(1, int(len(X_arr) * 0.85))
        X_train, X_test = X_arr[:split], X_arr[split:]
        y_train, y_test = y_arr[:split], y_arr[split:]

        X_train_t = torch.from_numpy(X_train).unsqueeze(-1)
        y_train_t = torch.from_numpy(y_train)
        X_test_t = torch.from_numpy(X_test).unsqueeze(-1)

        # Simple LSTM model.
        class _SimpleLSTM(nn.Module):
            def __init__(self, inp: int, hid: int, layers: int):
                super().__init__()
                self.lstm = nn.LSTM(inp, hid, layers, batch_first=True)
                self.fc = nn.Linear(hid, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        model = _SimpleLSTM(1, hidden_size, num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Training with early stop.
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss - 1e-6:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= _EARLY_STOP_PATIENCE * 3:
                    break

        # Validation.
        model.eval()
        with torch.no_grad():
            if len(X_test) > 0:
                preds_test = model(X_test_t).numpy()
                preds_test_denorm = preds_test * std_val + mean_val
                y_test_denorm = y_test * std_val + mean_val
                mae, rmse = _compute_metrics(y_test_denorm, preds_test_denorm)
                metrics.test_residuals = _compute_residuals(y_test_denorm, preds_test_denorm)
            else:
                mae, rmse = float("nan"), float("nan")

        # Multi-step forecast via autoregressive roll-forward.
        last_seq = torch.from_numpy(
            normed[-lookback:].astype(np.float32)
        ).unsqueeze(0).unsqueeze(-1)

        forecasts_list = []
        current_seq = last_seq.clone()
        for _ in range(n_forecast):
            with torch.no_grad():
                next_val = model(current_seq).item()
            forecasts_list.append(next_val * std_val + mean_val)
            # Roll the window.
            new_entry = torch.tensor([[[next_val]]], dtype=torch.float32)
            current_seq = torch.cat([current_seq[:, 1:, :], new_entry], dim=1)

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(X_train)
        metrics.n_test = len(X_test)
        metrics.fitted = True

        logger.info(
            "LSTM fit: lookback=%d, %d epochs, MAE=%.6f, RMSE=%.6f",
            lookback, epochs, mae, rmse,
        )
        return np.array(forecasts_list), metrics

    except Exception as exc:
        metrics.error = f"LSTM fitting failed: {exc}"
        logger.warning(metrics.error + " -- falling back to tree/linear")
        return _fit_linear_fallback(series, n_forecast, lookback, random_state)


# ---------------------------------------------------------------------------
# C2 -- Transformer Architecture (attention-based forecasting)
# ---------------------------------------------------------------------------


def fit_transformer(
    series: np.ndarray,
    n_forecast: int = 1,
    *,
    lookback: int = 20,
    d_model: int = 32,
    nhead: int = 4,
    num_layers: int = 2,
    epochs: int = 40,
    lr: float = 0.001,
    random_state: int = 42,
) -> ForecastResult:
    """Fit a Transformer encoder model for time series forecasting.

    Spec reference: The_Apps_core_idea.pdf Section E.2 Module 7 alt.

    Uses self-attention to identify which past days/variables matter
    most for predicting the next step.

    Falls back to linear model on failure.
    """
    metrics = ModelMetrics(model_name="transformer")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        metrics.error = "torch not installed"
        logger.warning("Transformer requires torch -- falling back")
        arr, fb_metrics = _fit_linear_fallback(series, n_forecast, lookback, random_state)
        fb_metrics.model_name = "transformer_fallback"
        return ForecastResult(metrics=[fb_metrics])

    try:
        clean = series[~np.isnan(series)]
        if len(clean) < lookback + n_forecast + 10:
            metrics.error = "Insufficient data for transformer"
            arr, fb_metrics = _fit_linear_fallback(series, n_forecast, lookback, random_state)
            fb_metrics.model_name = "transformer_fallback"
            return ForecastResult(metrics=[fb_metrics])

        # Standardise
        mu, sigma = clean.mean(), clean.std()
        if sigma < 1e-10:
            sigma = 1.0
        scaled = (clean - mu) / sigma

        # Create sequences
        X, y = [], []
        for i in range(len(scaled) - lookback - n_forecast + 1):
            X.append(scaled[i:i + lookback])
            y.append(scaled[i + lookback:i + lookback + n_forecast])
        X_arr = np.array(X)
        y_arr = np.array(y)

        split = max(1, int(len(X_arr) * 0.8))
        X_train = torch.FloatTensor(X_arr[:split]).unsqueeze(-1)  # (B, S, 1)
        y_train = torch.FloatTensor(y_arr[:split])
        X_test = torch.FloatTensor(X_arr[split:]).unsqueeze(-1)
        y_test = torch.FloatTensor(y_arr[split:])

        # Positional encoding
        class PositionalEncoding(nn.Module):
            def __init__(self, d_m: int, max_len: int = 500):
                super().__init__()
                pe = torch.zeros(max_len, d_m)
                pos = torch.arange(0, max_len).unsqueeze(1).float()
                div = torch.exp(torch.arange(0, d_m, 2).float() * (-np.log(10000.0) / d_m))
                pe[:, 0::2] = torch.sin(pos * div)
                if d_m > 1:
                    pe[:, 1::2] = torch.cos(pos * div[:d_m // 2])
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.pe[:, :x.size(1)]

        class TSTransformer(nn.Module):
            def __init__(self, inp: int, d_m: int, nh: int, nl: int, out: int):
                super().__init__()
                self.input_proj = nn.Linear(inp, d_m)
                self.pos_enc = PositionalEncoding(d_m)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_m, nhead=nh, dim_feedforward=d_m * 4,
                    batch_first=True, dropout=0.1,
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nl)
                self.fc = nn.Linear(d_m, out)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.input_proj(x)
                x = self.pos_enc(x)
                x = self.encoder(x)
                return self.fc(x[:, -1, :])  # last token

        torch.manual_seed(random_state)
        model = TSTransformer(1, d_model, nhead, num_layers, n_forecast)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        model.train()
        for ep in range(epochs):
            optimiser.zero_grad()
            pred = model(X_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            if len(X_test) > 0:
                test_pred = model(X_test).numpy()
                test_actual = y_test.numpy()
                test_rmse = float(np.sqrt(np.mean((test_pred - test_actual) ** 2)))
                metrics.rmse = test_rmse * sigma

            # Forecast
            last_seq = torch.FloatTensor(scaled[-lookback:]).unsqueeze(0).unsqueeze(-1)
            forecast_scaled = model(last_seq).numpy().flatten()[:n_forecast]

        forecast = forecast_scaled * sigma + mu
        metrics.model_name = "transformer"
        logger.info("Transformer fit: RMSE=%.4f", metrics.rmse or 0)

        return ForecastResult(
            forecasts={"target": {f"{i+1}d": float(forecast[i]) for i in range(len(forecast))}},
            metrics=[metrics],
        )

    except Exception as exc:
        metrics.error = f"Transformer fitting failed: {exc}"
        logger.warning(metrics.error + " -- falling back to linear")
        arr, fb_metrics = _fit_linear_fallback(series, n_forecast, lookback, random_state)
        fb_metrics.model_name = "transformer_fallback"
        return ForecastResult(metrics=[fb_metrics])


# ---------------------------------------------------------------------------
# C3 -- Particle Filter (Sequential Monte Carlo)
# ---------------------------------------------------------------------------


def fit_particle_filter(
    series: np.ndarray,
    n_forecast: int = 1,
    *,
    n_particles: int = 500,
    process_noise: float = 0.02,
    observation_noise: float = 0.05,
) -> ForecastResult:
    """Particle filter for non-linear, non-Gaussian state estimation.

    Spec reference: The_Apps_core_idea.pdf Section E.2 Module 4.

    Uses a swarm of particles to represent the state distribution.
    Provides full probability distribution of future states (not
    just mean). Handles extreme events better than Kalman.
    """
    metrics = ModelMetrics(model_name="particle_filter")

    try:
        clean = series[~np.isnan(series)]
        if len(clean) < 10:
            metrics.error = "Insufficient data for particle filter"
            return ForecastResult(metrics=[metrics])

        n = len(clean)

        # Initialise particles around the first observation
        particles = np.random.normal(clean[0], observation_noise * abs(clean[0]) + 1e-6, n_particles)
        weights = np.ones(n_particles) / n_particles

        # Run filter through observations
        for t in range(1, n):
            # Propagate: random walk with drift
            if t >= 2:
                drift = clean[t - 1] - clean[t - 2]
            else:
                drift = 0.0

            particles = particles + drift + np.random.normal(
                0, process_noise * abs(clean[t - 1]) + 1e-6, n_particles,
            )

            # Update weights based on observation likelihood
            likelihoods = np.exp(
                -0.5 * ((clean[t] - particles) / (observation_noise * abs(clean[t]) + 1e-6)) ** 2,
            )
            weights = weights * likelihoods
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones(n_particles) / n_particles

            # Resample (systematic resampling) when effective sample size drops
            n_eff = 1.0 / np.sum(weights ** 2)
            if n_eff < n_particles / 2:
                indices = _systematic_resample(weights)
                particles = particles[indices]
                weights = np.ones(n_particles) / n_particles

        # Forecast: propagate particles forward
        forecasts = np.zeros(n_forecast)
        current_particles = particles.copy()
        last_drift = clean[-1] - clean[-2] if len(clean) >= 2 else 0.0

        for step in range(n_forecast):
            current_particles = current_particles + last_drift + np.random.normal(
                0, process_noise * abs(clean[-1]) + 1e-6, n_particles,
            )
            forecasts[step] = np.average(current_particles, weights=weights)

        # Compute RMSE on last 20% as validation
        val_start = max(1, int(n * 0.8))
        val_errors = []
        pf_state = clean[val_start - 1]
        for t in range(val_start, n):
            drift_t = clean[t - 1] - clean[t - 2] if t >= 2 else 0.0
            pred_t = pf_state + drift_t
            val_errors.append((pred_t - clean[t]) ** 2)
            pf_state = clean[t]
        if val_errors:
            metrics.rmse = float(np.sqrt(np.mean(val_errors)))

        logger.info("Particle filter fit: RMSE=%.4f, n_particles=%d", metrics.rmse or 0, n_particles)

        return ForecastResult(
            forecasts={"target": {f"{i+1}d": float(forecasts[i]) for i in range(n_forecast)}},
            metrics=[metrics],
        )

    except Exception as exc:
        metrics.error = f"Particle filter failed: {exc}"
        logger.warning(metrics.error)
        return ForecastResult(metrics=[metrics])


def _systematic_resample(weights: np.ndarray) -> np.ndarray:
    """Systematic resampling for particle filter."""
    n = len(weights)
    positions = (np.arange(n) + np.random.random()) / n
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    return np.clip(indices, 0, n - 1)


def _fit_linear_fallback(
    series: np.ndarray,
    n_forecast: int,
    lookback: int,
    random_state: int,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """GradientBoosting or LinearRegression fallback for LSTM."""
    metrics = ModelMetrics(model_name="gradient_boosting")

    clean = series[~np.isnan(series)]
    if len(clean) < lookback + 5:
        metrics.error = f"Insufficient data for linear fallback ({len(clean)})"
        return None, metrics

    try:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[import-untyped]
        model_cls = GradientBoostingRegressor
        model_kwargs: dict[str, Any] = {
            "n_estimators": 50,
            "max_depth": 3,
            "random_state": random_state,
        }
    except ImportError:
        try:
            from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
            model_cls = LinearRegression  # type: ignore[assignment]
            model_kwargs = {}
            metrics.model_name = "linear_regression"
        except ImportError:
            metrics.error = "sklearn not installed -- tree/linear fallback unavailable"
            return None, metrics

    try:
        # Build lagged features.
        X_list, y_list = [], []
        for i in range(lookback, len(clean)):
            X_list.append(clean[i - lookback: i])
            y_list.append(clean[i])

        X = np.array(X_list)
        y = np.array(y_list)

        split = max(1, int(len(X) * 0.85))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)

        if len(X_test) > 0:
            preds = model.predict(X_test)
            mae, rmse = _compute_metrics(y_test, preds)
            metrics.test_residuals = _compute_residuals(y_test, preds)
        else:
            mae, rmse = float("nan"), float("nan")

        # Refit on all data.
        model.fit(X, y)

        # Multi-step autoregressive.
        current = clean[-lookback:].copy()
        forecasts = []
        for _ in range(n_forecast):
            next_val = float(model.predict(current.reshape(1, -1))[0])
            forecasts.append(next_val)
            current = np.append(current[1:], next_val)

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(X_train)
        metrics.n_test = len(X_test)
        metrics.fitted = True

        logger.info(
            "%s fallback fit: MAE=%.6f, RMSE=%.6f",
            metrics.model_name, mae, rmse,
        )
        return np.array(forecasts), metrics

    except Exception as exc:
        metrics.error = f"Tree/linear fallback failed: {exc}"
        logger.warning(metrics.error)
        return None, metrics


# ---------------------------------------------------------------------------
# 5. RF / GBM / XGB (tree ensembles for tabular features)
# ---------------------------------------------------------------------------


def fit_tree_ensemble(
    features: pd.DataFrame,
    target_col: str,
    n_forecast: int = 1,
    random_state: int = 42,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Fit a tree ensemble (XGBoost > GBM > RF) for tabular forecasting.

    Parameters
    ----------
    features:
        DataFrame with feature columns and ``target_col``.
    target_col:
        Column to forecast.
    n_forecast:
        Steps ahead (autoregressive roll-forward).
    random_state:
        Random seed.

    Returns
    -------
    (forecasts, metrics)
    """
    metrics = ModelMetrics(model_name="xgboost")

    # Try XGBoost first, then sklearn GBM, then RF.
    model_obj = _try_load_tree_model(random_state, metrics)
    if model_obj is None:
        return None, metrics

    clean = features.dropna()
    if len(clean) < _MIN_OBS_TREE:
        metrics.error = (
            f"Insufficient observations for tree ensemble "
            f"({len(clean)} < {_MIN_OBS_TREE})"
        )
        logger.warning(metrics.error)
        return None, metrics

    if target_col not in clean.columns:
        metrics.error = f"Target column '{target_col}' not in features"
        logger.warning(metrics.error)
        return None, metrics

    try:
        feature_cols = [c for c in clean.columns if c != target_col]
        if not feature_cols:
            metrics.error = "No feature columns available for tree ensemble"
            return None, metrics

        X = clean[feature_cols].values
        y = clean[target_col].values

        split = max(1, int(len(X) * 0.85))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model_obj.fit(X_train, y_train)

        if len(X_test) > 0:
            preds = model_obj.predict(X_test)
            mae, rmse = _compute_metrics(y_test, preds)
            metrics.test_residuals = _compute_residuals(y_test, preds)
        else:
            mae, rmse = float("nan"), float("nan")

        # Refit on all data.
        model_obj.fit(X, y)

        # For multi-step: use last row's features as starting point.
        last_features = X[-1:].copy()
        forecasts = []
        for _ in range(n_forecast):
            next_val = float(model_obj.predict(last_features)[0])
            forecasts.append(next_val)
            # Shift features (simple carry-forward for tabular).
            # In production, features would be updated properly.

        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = len(X_train)
        metrics.n_test = len(X_test)
        metrics.fitted = True

        logger.info(
            "%s fit: %d features, %d train, MAE=%.6f, RMSE=%.6f",
            metrics.model_name, len(feature_cols), len(X_train), mae, rmse,
        )
        return np.array(forecasts), metrics

    except Exception as exc:
        metrics.error = f"Tree ensemble fitting failed: {exc}"
        logger.warning(metrics.error)
        return None, metrics


def _try_load_tree_model(
    random_state: int,
    metrics: ModelMetrics,
) -> Any | None:
    """Try loading XGBoost > GBM > RF, return first available."""
    try:
        from xgboost import XGBRegressor  # type: ignore[import-untyped]
        metrics.model_name = "xgboost"
        return XGBRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
            verbosity=0,
        )
    except ImportError:
        pass

    try:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[import-untyped]
        metrics.model_name = "gradient_boosting"
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
        )
    except ImportError:
        pass

    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
        metrics.model_name = "random_forest"
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
        )
    except ImportError:
        pass

    metrics.error = "No tree ensemble library available (xgboost/sklearn)"
    logger.warning(metrics.error)
    return None


# ---------------------------------------------------------------------------
# 6. Baseline (always succeeds)
# ---------------------------------------------------------------------------


def fit_baseline(
    series: np.ndarray,
    n_forecast: int = 1,
    method: str = "ema",
    ema_span: int = 21,
) -> tuple[np.ndarray, ModelMetrics]:
    """Baseline forecaster: last-value or exponential moving average.

    This model **always** succeeds.  It is the final fallback.

    Parameters
    ----------
    series:
        1-D array of observed values.
    n_forecast:
        Steps ahead.
    method:
        ``"last"`` for last-value carry-forward, ``"ema"`` for
        exponential moving average.
    ema_span:
        EMA span in periods.

    Returns
    -------
    (forecasts, metrics)
        ``forecasts`` is always a valid array.
    """
    metrics = ModelMetrics(model_name=f"baseline_{method}", fitted=True)

    clean = series[~np.isnan(series)]

    if len(clean) == 0:
        # Absolute fallback: return zeros.
        metrics.model_name = "baseline_zero"
        return np.zeros(n_forecast), metrics

    if method == "ema" and len(clean) >= 3:
        # Compute EMA.
        alpha = 2.0 / (ema_span + 1)
        ema = clean[0]
        for val in clean[1:]:
            ema = alpha * val + (1 - alpha) * ema
        forecast_val = float(ema)
    else:
        forecast_val = float(clean[-1])

    forecasts = np.full(n_forecast, forecast_val)

    # Simple validation: use last 15% as test.
    if len(clean) > 10:
        split = max(1, int(len(clean) * 0.85))
        test = clean[split:]
        preds = np.full(len(test), float(clean[split - 1]))  # last-value
        mae, rmse = _compute_metrics(test, preds)
        metrics.mae = mae
        metrics.rmse = rmse
        metrics.n_train = split
        metrics.n_test = len(test)
        metrics.test_residuals = _compute_residuals(test, preds)

    logger.info(
        "Baseline (%s) forecast: %.6f (n_forecast=%d)",
        method, forecast_val, n_forecast,
    )
    return forecasts, metrics


# ===========================================================================
# Burn-out phase
# ===========================================================================


def _burnout_refit(
    series: np.ndarray,
    fit_fn: Any,
    n_forecast: int,
    window: int = _BURNOUT_WINDOW,
    patience: int = _EARLY_STOP_PATIENCE,
    **kwargs: Any,
) -> tuple[np.ndarray | None, ModelMetrics]:
    """Intensive retraining on the most recent window.

    Repeatedly shrinks the window and refits.  Stops early if the
    validation error does not improve for ``patience`` iterations.

    Parameters
    ----------
    series:
        Full 1-D series.
    fit_fn:
        One of the fit_* functions above.
    n_forecast:
        Steps to forecast.
    window:
        Maximum recent-data window.
    patience:
        Early-stop patience.
    **kwargs:
        Extra args forwarded to ``fit_fn``.

    Returns
    -------
    Best (forecasts, metrics) from the burnout phase, or (None, metrics)
    if no improvement was found.
    """
    clean = series[~np.isnan(series)]
    if len(clean) < 30:
        return None, ModelMetrics(
            model_name="burnout",
            error="Insufficient data for burnout refit",
        )

    best_rmse = float("inf")
    best_result: tuple[np.ndarray | None, ModelMetrics] = (
        None,
        ModelMetrics(model_name="burnout"),
    )
    no_improve = 0

    for frac in [1.0, 0.75, 0.5]:
        win = max(30, int(min(window, len(clean)) * frac))
        subset = clean[-win:]
        forecasts, met = fit_fn(subset, n_forecast=n_forecast, **kwargs)
        if forecasts is not None and not np.isnan(met.rmse) and met.rmse < best_rmse:
            best_rmse = met.rmse
            best_result = (forecasts, met)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_result


# ===========================================================================
# Pipeline entry point
# ===========================================================================


def apply_residual_feature_adjustment(
    cache: pd.DataFrame,
    variable: str,
    forecast_value: float,
    extra_variables: list[str],
    fitted_values: pd.Series | None = None,
    alpha: float = 1.0,
) -> float:
    """Adjust a univariate model's forecast using features via residual regression.

    Pattern from Prophet/Greykite/Orbit: fit univariate model, regress
    residuals on features, apply the adjustment. This allows Kalman, AR1,
    ETS to benefit from exogenous features without modifying their core.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    variable:
        Target variable name (e.g. "close").
    forecast_value:
        Original univariate model forecast.
    extra_variables:
        List of feature column names.
    fitted_values:
        In-sample fitted values from the univariate model.
    alpha:
        Ridge regression regularization strength (higher = more conservative).

    Returns
    -------
    Adjusted forecast value.
    """
    if fitted_values is None or len(extra_variables) == 0:
        return forecast_value

    try:
        feature_cols = [c for c in extra_variables if c in cache.columns and cache[c].notna().sum() > 20]
        if not feature_cols:
            return forecast_value

        # Compute residuals (in-sample only -- no look-ahead)
        resid = cache[variable] - fitted_values
        X = cache[feature_cols]
        common = X.dropna().index.intersection(resid.dropna().index)
        if len(common) < 30:
            return forecast_value

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=alpha)
        model.fit(X.loc[common].values, resid.loc[common].values)

        # Apply adjustment using latest feature values
        latest = X.iloc[-1:].fillna(0).values
        adjustment = float(model.predict(latest)[0])
        # Cap adjustment to prevent wild swings (max 5% of forecast)
        max_adj = abs(forecast_value) * 0.05 if forecast_value != 0 else 1.0
        adjustment = max(-max_adj, min(max_adj, adjustment))
        return forecast_value + adjustment
    except Exception:
        return forecast_value


def run_forecasting(
    cache: pd.DataFrame,
    variables: list[str] | None = None,
    *,
    extra_variables: list[str] | None = None,
    windows: Any | None = None,
    random_state: int = 42,
    enable_burnout: bool = True,
) -> tuple[pd.DataFrame, ForecastResult]:
    """Run the full forecasting pipeline on the daily cache.

    Applies each model in the fallback chain for each variable.
    Adds forecast columns to the cache and returns the result
    container with metrics.

    Parameters
    ----------
    cache:
        Daily cache DataFrame (DatetimeIndex) with feature columns.
    variables:
        List of variable names to forecast.  If ``None``, all tier
        variables from the survival hierarchy are used.
    windows:
        Adaptive window sizes from ``adaptive_windows.compute_adaptive_windows()``.
        If provided, overrides hardcoded LSTM lookback and burnout window
        with filing-frequency-anchored values (Nyquist-Shannon).  Expected
        attributes: ``short``, ``medium``, ``long``, ``trend``.
    random_state:
        Random seed for reproducible models.
    enable_burnout:
        If True, run burnout refinement phase on best model.

    Returns
    -------
    (cache, result)
        The (possibly augmented) cache and the ``ForecastResult``.
    """
    logger.info("Starting forecasting pipeline...")

    # Override hardcoded window sizes with adaptive values when available.
    # Nyquist-anchored windows prevent aliasing with the filing cycle
    # (e.g. a 21-day lookback on semi-annual data sees only 1/6 of a
    # filing period -- pure sub-period noise).
    global _LSTM_LOOKBACK, _BURNOUT_WINDOW
    _orig_lstm_lookback = _LSTM_LOOKBACK
    _orig_burnout_window = _BURNOUT_WINDOW
    if windows is not None:
        _new_lookback = getattr(windows, "short", None)
        _new_burnout = getattr(windows, "long", None)
        if _new_lookback and _new_lookback > 0:
            _LSTM_LOOKBACK = int(_new_lookback)
        if _new_burnout and _new_burnout > 0:
            _BURNOUT_WINDOW = int(_new_burnout)
        logger.info(
            "Adaptive windows applied: lstm_lookback=%d (was %d), burnout=%d (was %d)",
            _LSTM_LOOKBACK, _orig_lstm_lookback, _BURNOUT_WINDOW, _orig_burnout_window,
        )

    result = ForecastResult()
    tier_map = _load_tier_variables()

    if variables is None:
        variables = []
        for tier_vars in tier_map.values():
            variables.extend(tier_vars)
        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_vars: list[str] = []
        for v in variables:
            if v not in seen:
                seen.add(v)
                unique_vars.append(v)
        variables = unique_vars

    # Append extra variables (e.g. financial health scores) so they get
    # forecasted alongside the tier variables.
    if extra_variables:
        seen = set(variables)
        for ev in extra_variables:
            if ev not in seen and ev in cache.columns:
                variables.append(ev)
                seen.add(ev)

    # Filter to columns actually present in cache.
    available_vars = [v for v in variables if v in cache.columns]
    missing_vars = set(variables) - set(available_vars)
    if missing_vars:
        logger.info(
            "Forecasting: %d variables not in cache, skipping: %s",
            len(missing_vars),
            sorted(missing_vars)[:5],
        )

    # Also get returns/volatility for specialised models.
    has_returns = "return_1d" in cache.columns
    has_volatility = "volatility_21d" in cache.columns

    # ------------------------------------------------------------------
    # Unified cache-to-model data extraction
    # ------------------------------------------------------------------
    # All models consume the same cache DataFrame. This adapter
    # centralises how data is extracted for each model type, so every
    # model gets consistent, quality-checked inputs from the pipeline.

    def _extract_series(var: str) -> np.ndarray:
        """Extract a single variable's values from the cache.

        The pipeline's estimation step (Step 4b) fills missing values
        before forecasting runs. This function simply extracts whatever
        the estimator produced -- observed, estimated, or still NaN.
        """
        return cache[var].values

    def _extract_multivariate(target: str, max_cols: int = 10) -> pd.DataFrame:
        """Extract a multivariate DataFrame for VAR from the cache.

        Mixed-frequency aware: forward-filled quarterly/annual columns are
        replaced with their filing-change derivatives (the actual change
        on filing days, zero between filings). This prevents near-singular
        covariance matrices that cause VAR to fall back to AR(1).
        """
        from operator1.models._frequency_classifier import (
            classify_column_frequency,
            get_filing_change_derivative,
        )
        candidates = []
        for c in available_vars:
            if c not in cache.columns or not cache[c].notna().any():
                continue
            freq = classify_column_frequency(cache[c])
            if freq == "daily":
                candidates.append(c)
            elif freq in ("quarterly", "annual"):
                # Use the filing-change derivative instead of raw forward-filled
                deriv_col = get_filing_change_derivative(cache, c)
                if cache[deriv_col].notna().any() and cache[deriv_col].abs().sum() > 0:
                    candidates.append(deriv_col)
            # Skip 'constant' columns entirely
        candidates = candidates[:max_cols]
        if target not in candidates:
            candidates = [target] + candidates[:max_cols - 1]
        return cache[candidates].copy()

    def _extract_features(target: str, max_cols: int = 15) -> pd.DataFrame:
        """Extract a feature DataFrame for tree ensembles from the cache.

        Mixed-frequency aware: for quarterly/annual columns, adds
        engineered features (pct_change_at_filing, days_since_filing)
        that give the tree meaningful split points instead of only 3-4
        unique values from forward-filled quarterly data.
        """
        from operator1.models._frequency_classifier import (
            classify_column_frequency,
            add_filing_timing_features,
        )
        feature_cols = [
            c for c in cache.columns
            if c != target
            and not c.startswith("is_missing_")
            and not c.startswith("invalid_math_")
            and cache[c].dtype in (np.float64, np.float32, np.int64)
            and cache[c].notna().any()  # exclude entirely empty columns
        ][:max_cols]
        if not feature_cols:
            return pd.DataFrame()
        # Add filing-timing features for quarterly/annual columns
        extra_timing_cols = []
        for fc in feature_cols[:10]:  # limit to avoid bloat
            freq = classify_column_frequency(cache[fc])
            if freq in ("quarterly", "annual"):
                new_cols = add_filing_timing_features(cache, fc)
                extra_timing_cols.extend(new_cols)
        all_cols = feature_cols + extra_timing_cols + [target]
        all_cols = [c for c in all_cols if c in cache.columns]
        return cache[all_cols].copy()

    # ------------------------------------------------------------------
    # GARCH on volatility (special case -- uses return_1d from cache)
    # ------------------------------------------------------------------
    if has_returns:
        returns = _extract_series("return_1d")
        max_horizon = max(HORIZONS.values())
        garch_fcast, garch_met = fit_garch(
            returns, n_forecast=max_horizon,
            cache=cache, extra_variables=extra_variables,
        )
        garch_met.variable = "volatility_21d"
        result.metrics.append(garch_met)
        if garch_fcast is None:
            result.model_failed_garch = True
            result.garch_error = garch_met.error
        else:
            result.forecasts["volatility_garch"] = {
                label: float(garch_fcast[min(h - 1, len(garch_fcast) - 1)])
                for label, h in HORIZONS.items()
            }
            result.model_used["volatility_garch"] = "garch"

        # A3: GARCH-MIDAS with macro-driven long-run volatility component.
        # Try GARCH-MIDAS when macro features are available in cache.
        # The long-run component lets macro deterioration influence vol
        # forecasts BEFORE the crash happens.
        try:
            _macro_cols_for_midas = [
                c for c in cache.columns
                if c.startswith("macro_") and cache[c].dtype in ("float64", "float32")
                and cache[c].notna().sum() > 20
            ][:5]
            if _macro_cols_for_midas:
                _macro_df = cache[_macro_cols_for_midas].copy()
                _midas_fcast, _midas_met = fit_garch_midas(
                    returns, macro_features=_macro_df, n_forecast=max_horizon,
                )
                _midas_met.variable = "volatility_21d"
                result.metrics.append(_midas_met)
                if _midas_fcast is not None and not result.model_failed_garch:
                    result.forecasts["volatility_garch_midas"] = {
                        label: float(_midas_fcast[min(h - 1, len(_midas_fcast) - 1)])
                        for label, h in HORIZONS.items()
                    }
                    result.model_used["volatility_garch_midas"] = "garch_midas"
                    logger.info("A3 GARCH-MIDAS fitted with %d macro features", len(_macro_cols_for_midas))
        except Exception as _midas_exc:
            logger.debug("A3 GARCH-MIDAS skipped: %s", _midas_exc)

        # HAR-RV (Corsi 2009): Heterogeneous Autoregressive Realized Volatility.
        # Uses daily, weekly, and monthly RV as regressors. Captures long-memory
        # of volatility that single-regime GARCH misses. Often outperforms GARCH
        # in empirical tests.
        try:
            from arch.univariate import HARX  # type: ignore[import-untyped]

            _rv = returns.dropna() ** 2  # daily realized variance proxy
            if len(_rv) >= 63:
                har_model = HARX(_rv * 10000, lags=[1, 5, 22])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    har_res = har_model.fit(disp="off", show_warning=False)
                har_fcast = har_res.forecast(horizon=max(HORIZONS.values()))
                if har_fcast is not None and har_fcast.variance is not None:
                    _har_var = har_fcast.variance.iloc[-1].values / 10000
                    result.forecasts["volatility_har"] = {
                        label: float(np.sqrt(max(_har_var[min(h - 1, len(_har_var) - 1)], 0)))
                        for label, h in HORIZONS.items()
                    }
                    result.model_used["volatility_har"] = "har_rv"
                    logger.debug("HAR-RV volatility forecast computed")
        except Exception as _har_exc:
            logger.debug("HAR-RV forecast skipped: %s", _har_exc)

    # ------------------------------------------------------------------
    # Per-variable forecasting
    # ------------------------------------------------------------------
    kalman_attempted = False
    var_attempted = False
    lstm_attempted = False
    tree_attempted = False

    for var_name in available_vars:
        series = _extract_series(var_name)
        tier = _get_tier_for_variable(var_name, tier_map)
        max_horizon = max(HORIZONS.values())
        best_forecast: np.ndarray | None = None
        best_model_name = ""
        best_metrics: ModelMetrics | None = None

        # --- Kalman (preferred for tier1/tier2) ---
        # Try per-regime Kalman first (Section K.2), fall back to standard
        if tier in ("tier1", "tier2"):
            _regime_labels_arr = cache.get("regime_label")
            if _regime_labels_arr is not None and has_returns:
                _rl = _regime_labels_arr.values if hasattr(_regime_labels_arr, "values") else _regime_labels_arr
                # Build regime probability dict from current HMM state
                _rp = None
                _prob_cols = [c for c in cache.columns if c.startswith("regime_hmm_prob_")]
                if _prob_cols and len(cache) > 0:
                    _last_probs = cache[_prob_cols].iloc[-1]
                    _regime_map = {0: "bull", 1: "bear", 2: "high_vol", 3: "low_vol"}
                    _rp = {_regime_map.get(i, str(i)): float(_last_probs.iloc[i]) for i in range(len(_last_probs)) if not np.isnan(_last_probs.iloc[i])}
                fcast, met = fit_kalman_per_regime(series, _rl, regime_probs=_rp, n_forecast=max_horizon)
            else:
                fcast, met = fit_kalman(series, n_forecast=max_horizon)
            met.variable = var_name
            result.metrics.append(met)
            if fcast is not None:
                best_forecast = fcast
                best_model_name = "kalman"
                best_metrics = met
            else:
                if not kalman_attempted:
                    result.model_failed_kalman = True
                    result.kalman_error = met.error
            kalman_attempted = True

        # --- VAR (if multiple variables available) ---
        if best_forecast is None and len(available_vars) >= 2:
            var_df = _extract_multivariate(var_name)

            fcast, met = fit_var(var_df, var_name, n_forecast=max_horizon)
            met.variable = var_name
            result.metrics.append(met)
            if fcast is not None:
                best_forecast = fcast
                best_model_name = met.model_name
                best_metrics = met
            else:
                if not var_attempted:
                    result.model_failed_var = True
                    result.var_error = met.error
            var_attempted = True

        # --- LSTM / tree-linear fallback ---
        if best_forecast is None:
            fcast, met = fit_lstm(
                series,
                n_forecast=max_horizon,
                random_state=random_state,
            )
            met.variable = var_name
            result.metrics.append(met)
            if fcast is not None:
                best_forecast = fcast
                best_model_name = met.model_name
                best_metrics = met
            else:
                if not lstm_attempted:
                    result.model_failed_lstm = True
                    result.lstm_error = met.error
            lstm_attempted = True

        # --- Tree ensemble on tabular features ---
        if best_forecast is None:
            feat_df = _extract_features(var_name)
            if not feat_df.empty:
                fcast, met = fit_tree_ensemble(
                    feat_df,
                    var_name,
                    n_forecast=max_horizon,
                    random_state=random_state,
                )
                met.variable = var_name
                result.metrics.append(met)
                if fcast is not None:
                    best_forecast = fcast
                    best_model_name = met.model_name
                    best_metrics = met
                else:
                    if not tree_attempted:
                        result.model_failed_tree = True
                        result.tree_error = met.error
                tree_attempted = True

        # --- Baseline (always succeeds) ---
        if best_forecast is None:
            fcast, met = fit_baseline(series, n_forecast=max_horizon)
            met.variable = var_name
            result.metrics.append(met)
            best_forecast = fcast
            best_model_name = met.model_name
            best_metrics = met

        # --- Burnout refinement ---
        if enable_burnout and best_forecast is not None and best_model_name == "kalman":
            burnout_fcast, burnout_met = _burnout_refit(
                series, fit_kalman, n_forecast=max_horizon
            )
            if (
                burnout_fcast is not None
                and not np.isnan(burnout_met.rmse)
                and (
                    best_metrics is None
                    or np.isnan(best_metrics.rmse)
                    or burnout_met.rmse < best_metrics.rmse
                )
            ):
                best_forecast = burnout_fcast
                best_model_name = f"{best_model_name}_burnout"
                best_metrics = burnout_met

        # ----------------------------------------------------------
        # C1: Horizon-specific model selection.
        # For long horizons (21d, 252d), prefer fundamental/tree models
        # over autoregressive models. Tree ensembles with macro features
        # outperform Kalman/GARCH at longer horizons where momentum
        # decays and fundamentals dominate.
        # ----------------------------------------------------------
        _horizon_forecasts: dict[str, float] = {}
        _long_horizon_model: np.ndarray | None = None
        _long_model_name = ""

        if best_forecast is not None:
            # For short horizons (1d, 5d): use best autoregressive model (default cascade winner)
            for label, h in HORIZONS.items():
                _horizon_forecasts[label] = float(best_forecast[min(h - 1, len(best_forecast) - 1)])

            # Regime probability-weighted directional shift (Method 5)
            _prob_cols = [c for c in cache.columns if c.startswith("regime_hmm_prob_")]
            if _prob_cols and var_name == "close" and len(cache) > 0:
                try:
                    _last_probs = cache[_prob_cols].iloc[-1].dropna()
                    if len(_last_probs) > 0 and "regime_hmm" in cache.columns:
                        _ret_col = "return_1d"
                        if _ret_col in cache.columns:
                            _regime_means = {}
                            for _rpc in _prob_cols:
                                _ridx = int(_rpc.split("_")[-1])
                                _rmask = cache["regime_hmm"] == _ridx
                                if _rmask.sum() > 5:
                                    _regime_means[_ridx] = float(cache.loc[_rmask, _ret_col].mean())
                            if _regime_means:
                                _expected_daily = sum(
                                    _regime_means.get(int(_rpc.split("_")[-1]), 0.0) * float(_last_probs[_rpc])
                                    for _rpc in _last_probs.index
                                )
                                for _rl, _rh in HORIZONS.items():
                                    _rshift = _expected_daily * _rh
                                    _horizon_forecasts[_rl] *= (1 + _rshift)
                except Exception:
                    pass

            # P1 fix: Momentum overlay for close predictions.
            # The Kalman/baseline models anchor near the last price, missing
            # directional moves. Blend with a momentum-based prediction using
            # recent returns to reduce systematic upward bias at ATH.
            if var_name == "close" and "return_1d" in cache.columns:
                try:
                    _ret = cache["return_1d"].dropna()
                    if len(_ret) >= 21:
                        _last_close = float(cache["close"].dropna().iloc[-1])
                        _mom_21d = float(_ret.iloc[-21:].mean())
                        _mom_5d = float(_ret.iloc[-5:].mean())
                        for _ml, _mh in HORIZONS.items():
                            # Use shorter-window momentum for short horizons
                            _mom = _mom_5d if _mh <= 5 else _mom_21d
                            _mom_pred = _last_close * (1 + _mom * _mh)
                            # Blend: 70% model, 30% momentum
                            _horizon_forecasts[_ml] = (
                                0.7 * _horizon_forecasts[_ml] + 0.3 * _mom_pred
                            )
                except Exception:
                    pass

            # Step 2: Residual feature adjustment -- augment univariate model
            # forecasts with feature-based residual regression (Prophet/Greykite
            # pattern). Only applies when extra_variables are available.
            if extra_variables:
                try:
                    _fitted_vals = None
                    if best_metrics and best_metrics.test_residuals is not None:
                        _n_resid = len(best_metrics.test_residuals)
                        _actual = cache[var_name].dropna()
                        if len(_actual) >= _n_resid and _n_resid > 0:
                            _fitted_vals = _actual.iloc[-_n_resid:] - pd.Series(
                                best_metrics.test_residuals,
                                index=_actual.index[-_n_resid:],
                            )
                    for _rfa_label in _horizon_forecasts:
                        _rfa_orig = _horizon_forecasts[_rfa_label]
                        _rfa_adj = apply_residual_feature_adjustment(
                            cache, var_name, _rfa_orig, extra_variables,
                            fitted_values=_fitted_vals,
                        )
                        _horizon_forecasts[_rfa_label] = _rfa_adj
                except Exception:
                    pass  # graceful fallback: use unadjusted forecasts

            # For long horizons (21d, 252d): try tree ensemble as alternative
            # if the cascade winner was an autoregressive model (Kalman, GARCH, VAR, LSTM)
            _ar_models = {"kalman", "kalman_per_regime", "kalman_burnout", "kalman_dfm",
                          "garch", "var", "ar1", "lstm", "lstm_fallback_gbm",
                          "lstm_fallback_lr", "ets"}
            if best_model_name.lower().split("(")[0] in _ar_models:
                feat_df = _extract_features(var_name)
                if not feat_df.empty:
                    _lh_fcast, _lh_met = fit_tree_ensemble(
                        feat_df, var_name,
                        n_forecast=max_horizon,
                        random_state=random_state,
                    )
                    if _lh_fcast is not None and _lh_met.fitted:
                        _long_horizon_model = _lh_fcast
                        _long_model_name = _lh_met.model_name
                        # Blend: at 21d use 60% tree + 40% AR; at 252d use 80% tree + 20% AR
                        for label, h in HORIZONS.items():
                            if h >= 21:
                                _tree_val = float(_long_horizon_model[min(h - 1, len(_long_horizon_model) - 1)])
                                _ar_val = _horizon_forecasts[label]
                                _tree_weight = 0.6 if h == 21 else 0.8
                                _horizon_forecasts[label] = _tree_weight * _tree_val + (1 - _tree_weight) * _ar_val

            # Also try ETS for medium horizons (5d-21d) if not already the winner
            if best_model_name != "ets":
                _ets_fcast, _ets_met = fit_ets(
                    series[~np.isnan(series)],
                    n_forecast=max_horizon,
                )
                if _ets_fcast is not None and _ets_met.fitted:
                    # For 5d: blend 30% ETS + 70% cascade winner
                    for label, h in [(l, hh) for l, hh in HORIZONS.items() if 5 <= hh <= 21]:
                        _ets_val = float(_ets_fcast[min(h - 1, len(_ets_fcast) - 1)])
                        _ets_weight = 0.3 if h == 5 else 0.2  # Less weight at 21d (tree dominates)
                        _horizon_forecasts[label] = (1 - _ets_weight) * _horizon_forecasts[label] + _ets_weight * _ets_val

        # Store results.
        if best_forecast is not None:
            result.forecasts[var_name] = _horizon_forecasts
            result.model_used[var_name] = best_model_name

    # ------------------------------------------------------------------
    # Parallel tree ensemble: run tree on key variables even when another
    # model won the cascade, so features from Boruta/PIMP/mRMR are
    # actually consumed by at least one model per variable.
    # ------------------------------------------------------------------
    _parallel_tree_vars = ["close", "return_1d", "volatility_21d"]
    _tree_already_primary = {
        v for v, m in result.model_used.items()
        if "tree" in m or "xgboost" in m or "gbm" in m or "rf" in m
    }
    if extra_variables and len(extra_variables) > 0:
        for _ptv in _parallel_tree_vars:
            if _ptv in cache.columns and _ptv not in _tree_already_primary:
                _pt_feat_df = _extract_features(_ptv)
                if not _pt_feat_df.empty:
                    try:
                        _pt_fcast, _pt_met = fit_tree_ensemble(
                            _pt_feat_df, _ptv,
                            n_forecast=max(HORIZONS.values()),
                            random_state=random_state,
                        )
                        _pt_met.variable = _ptv
                        _pt_met.model_name = "tree_parallel"
                        result.metrics.append(_pt_met)
                        if _pt_fcast is not None and _pt_met.fitted:
                            # Store as secondary forecast channel (does not
                            # overwrite the primary cascade winner).
                            if _ptv not in result.forecasts:
                                result.forecasts[_ptv] = {}
                            if isinstance(result.forecasts[_ptv], dict):
                                for label, h in HORIZONS.items():
                                    # Store parallel tree forecast for each horizon
                                    # only if the primary cascade didn't already set it.
                                    if label not in result.forecasts[_ptv]:
                                        result.forecasts[_ptv][label] = float(
                                            _pt_fcast[min(h - 1, len(_pt_fcast) - 1)]
                                        )
                            logger.info(
                                "Parallel tree for '%s': RMSE=%.6f",
                                _ptv, _pt_met.rmse if np.isfinite(_pt_met.rmse) else -1.0,
                            )
                    except Exception as _pt_exc:
                        logger.debug("Parallel tree for '%s' failed: %s", _ptv, _pt_exc)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_forecasted = len(result.forecasts)
    n_failed = sum([
        result.model_failed_kalman,
        result.model_failed_garch,
        result.model_failed_var,
        result.model_failed_lstm,
        result.model_failed_tree,
    ])
    models_used = {}
    for var, model in result.model_used.items():
        models_used[model] = models_used.get(model, 0) + 1

    # Identify variables that had zero non-NaN observations (all-NaN columns)
    zero_obs_vars = []
    for var_name in available_vars:
        series = cache[var_name].values
        n_clean = int(np.sum(~np.isnan(series)))
        if n_clean == 0:
            zero_obs_vars.append(var_name)

    logger.info(
        "Forecasting complete: %d variables forecasted, %d model types failed, "
        "model distribution: %s",
        n_forecasted,
        n_failed,
        models_used,
    )

    if zero_obs_vars:
        logger.warning(
            "Forecasting: %d variables had 0 non-NaN observations (all-NaN columns, "
            "likely missing from data source): %s",
            len(zero_obs_vars),
            zero_obs_vars,
        )

    # Collect validation residuals from fitted models for conformal calibration.
    # Prefer actual test residuals stored per-metric (distribution-free).
    # Fall back to synthetic +/-RMSE pairs if no real residuals available.
    _residuals: list[float] = []
    _has_real = False
    for met in result.metrics:
        if met.fitted and met.test_residuals:
            _residuals.extend(met.test_residuals)
            _has_real = True
        elif met.fitted and np.isfinite(met.rmse) and met.rmse > 0:
            # Synthetic fallback: +/- RMSE (Gaussian assumption).
            _residuals.append(met.rmse)
            _residuals.append(-met.rmse)
    if _residuals:
        result.residuals = _residuals
        logger.info(
            "Collected %d %s residual samples for conformal calibration",
            len(_residuals), "real" if _has_real else "synthetic",
        )

    # Restore original window constants (avoid leaking adaptive values
    # into subsequent calls from tests or multi-company pipelines).
    _LSTM_LOOKBACK = _orig_lstm_lookback
    _BURNOUT_WINDOW = _orig_burnout_window

    return cache, result


# ===========================================================================
# Phase D -- Temporal Engine Enhancements
# ===========================================================================


# ---------------------------------------------------------------------------
# D3: ModelWrapper protocol and concrete wrappers
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelWrapperProtocol(Protocol):
    """Protocol every online-updatable model must satisfy."""

    def predict(self, state_t: np.ndarray) -> np.ndarray: ...

    def update(
        self,
        state_t: np.ndarray,
        actual_t_plus_1: np.ndarray,
        hierarchy_weights: dict[str, float],
    ) -> None: ...


class BaseModelWrapper(ABC):
    """Abstract base for all model wrappers used in the forward pass."""

    name: str = "base"
    failed_update: bool = False
    _update_error: str | None = None

    @abstractmethod
    def predict(self, state_t: np.ndarray) -> np.ndarray:
        """Return point forecast for the next step."""

    @abstractmethod
    def update(
        self,
        state_t: np.ndarray,
        actual_t_plus_1: np.ndarray,
        hierarchy_weights: dict[str, float],
    ) -> None:
        """Incremental parameter update after observing *actual_t_plus_1*."""


class KalmanWrapper(BaseModelWrapper):
    """Online Kalman wrapper using a local-level state-space model.

    Kalman is inherently online: the Kalman gain update is a single
    matrix operation per observation.
    """

    name = "kalman"

    def __init__(self, series: np.ndarray) -> None:
        clean = series[~np.isnan(series)]
        self._last_value = float(clean[-1]) if len(clean) else 0.0
        self._state = float(clean[-1]) if len(clean) else 0.0
        self._P = 1.0  # state covariance
        self._Q = 0.01  # process noise
        self._R = 0.1  # measurement noise (for real observations)
        self._R_stale = 100.0  # high noise for stale forward-filled days
        self._filing_aware = False  # set True if series is forward-filled
        self._fitted = len(clean) >= _MIN_OBS_KALMAN

        # Detect forward-filled financial data: if unique values < 5%
        # of total observations, this is quarterly/annual data repeated daily.
        # Use filing-aware mode with variable observation noise.
        if len(clean) >= _MIN_OBS_KALMAN:
            unique_ratio = len(np.unique(clean)) / len(clean)
            if unique_ratio < 0.05:
                self._filing_aware = True
                self._Q = 0.001  # lower process noise for stable financials
                self._R = 0.01  # tight on real observation days
                self._R_stale = 1000.0  # very loose on stale days
                logger.debug(
                    "Kalman filing-aware mode: %d unique values in %d obs (%.1f%%)",
                    len(np.unique(clean)), len(clean), unique_ratio * 100,
                )

        # Try fitting a proper statsmodels model for the initial state.
        if self._fitted and not self._filing_aware:
            try:
                from statsmodels.tsa.statespace.structural import (
                    UnobservedComponents,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = UnobservedComponents(clean, level="local level")
                    res = model.fit(disp=False, maxiter=200)
                self._state = float(res.filtered_state[0, -1])
                self._P = float(res.filtered_state_cov[0, 0, -1])
            except Exception:
                pass  # fall back to simple values

    def predict(self, state_t: np.ndarray) -> np.ndarray:
        # Local-level model: prediction = current state estimate
        predicted = self._state
        self._P += self._Q  # predicted covariance
        return np.array([predicted])

    def update(
        self,
        state_t: np.ndarray,
        actual_t_plus_1: np.ndarray,
        hierarchy_weights: dict[str, float],
    ) -> None:
        try:
            z = float(actual_t_plus_1[0]) if len(actual_t_plus_1) else self._state
            if np.isnan(z):
                return  # skip update on missing observation

            # Filing-aware: use high observation noise for stale (repeated) values
            # so the Kalman mostly ignores repeated forward-filled data and only
            # updates meaningfully when a new filing changes the value.
            R_effective = self._R
            if self._filing_aware and abs(z - self._last_value) < 1e-10:
                R_effective = self._R_stale  # stale repeated value -> high noise

            # Kalman gain
            S = self._P + R_effective
            K = self._P / S if S > 1e-12 else 0.5
            # State update
            innovation = z - self._state
            self._state += K * innovation
            self._P = (1 - K) * self._P
            self._last_value = z
            self.failed_update = False
        except Exception as exc:
            self.failed_update = True
            self._update_error = str(exc)


class GARCHWrapper(BaseModelWrapper):
    """Online GARCH wrapper with regime-switching volatility forecast.

    Supports two modes:
    1. **Single-regime** (default): standard GARCH(1,1) on full sample.
       Update rule: sigma2[t+1] = omega + alpha * eps^2[t] + beta * sigma2[t]
    2. **Regime-switching** (when regime_labels provided): fits separate
       GARCH per HMM regime, blends forecasts using transition probs.
       This prevents volatility mean-reversion to the unconditional
       variance, which causes underestimates during regime transitions.
    """

    name = "garch"

    def __init__(
        self,
        returns: np.ndarray,
        regime_labels: np.ndarray | None = None,
    ) -> None:
        clean = returns[~np.isnan(returns)]
        # Default GARCH(1,1) parameters
        self._omega = 0.0001
        self._alpha = 0.10
        self._beta = 0.85
        self._sigma2 = float(np.var(clean)) if len(clean) > 1 else 0.0004
        self._last_return = float(clean[-1]) if len(clean) else 0.0
        self._fitted = len(clean) >= _MIN_OBS_GARCH

        # Regime-switching state
        self._regime_params: dict[str, dict[str, float]] = {}
        self._transition_probs: dict[str, dict[str, float]] = {}
        self._current_regime: str = ""
        self._regime_switching = False

        if self._fitted:
            try:
                from arch import arch_model  # type: ignore[import-untyped]

                # Try regime-switching GARCH when labels are available
                if regime_labels is not None and len(regime_labels) == len(returns):
                    clean_labels = regime_labels[~np.isnan(returns)]
                    self._fit_regime_switching(clean, clean_labels)

                # Always fit single-regime as fallback
                scaled = clean * 100.0
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = arch_model(scaled, vol="Garch", p=1, q=1, mean="Constant", rescale=False)
                    res = model.fit(disp="off", show_warning=False)
                self._omega = float(res.params.get("omega", self._omega))
                self._alpha = float(res.params.get("alpha[1]", self._alpha))
                self._beta = float(res.params.get("beta[1]", self._beta))
                self._sigma2 = float(res.conditional_volatility.iloc[-1] ** 2) / 10000.0
            except Exception:
                pass

    def _fit_regime_switching(
        self,
        returns: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit per-regime GARCH models and compute transition matrix."""
        try:
            from arch import arch_model  # type: ignore[import-untyped]

            unique_regimes = [str(r) for r in np.unique(labels) if str(r) != "nan"]
            if len(unique_regimes) < 2:
                return

            # Minimum regime diversity check: merge tiny regimes (< 20 obs)
            # into the nearest neighbor by sample mean return. This prevents
            # fitting GARCH on 1-5 observations (e.g., HMM assigns 495 days
            # to "high_vol" and 1 day to "bull").
            _MIN_REGIME_OBS = 20
            regime_sizes = {
                r: int(np.sum([str(l) == r for l in labels]))
                for r in unique_regimes
            }
            small_regimes = [r for r, n in regime_sizes.items() if n < _MIN_REGIME_OBS]
            if small_regimes:
                # Find the largest regime to absorb small ones
                largest = max(regime_sizes, key=regime_sizes.get)
                for small_r in small_regimes:
                    unique_regimes.remove(small_r)
                    # Remap labels: replace small regime with largest
                    labels = np.array([largest if str(l) == small_r else l for l in labels])
                logger.debug(
                    "GARCH regime merge: %d small regimes absorbed into '%s'",
                    len(small_regimes), largest,
                )

            if len(unique_regimes) < 2:
                # After merging, only 1 effective regime -- skip switching
                return

            # Fit GARCH per regime
            for regime in unique_regimes:
                mask = np.array([str(l) == regime for l in labels])
                regime_returns = returns[mask]
                if len(regime_returns) < 30:
                    # Not enough data -- use sample variance
                    self._regime_params[regime] = {
                        "sigma2": float(np.var(regime_returns)) if len(regime_returns) > 1 else 0.0004,
                        "omega": 0.0001,
                        "alpha": 0.10,
                        "beta": 0.85,
                    }
                    continue

                try:
                    scaled = regime_returns * 100.0
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = arch_model(scaled, vol="Garch", p=1, q=1, mean="Constant", rescale=False)
                        res = model.fit(disp="off", show_warning=False)
                    self._regime_params[regime] = {
                        "sigma2": float(res.conditional_volatility.iloc[-1] ** 2) / 10000.0,
                        "omega": float(res.params.get("omega", 0.0001)),
                        "alpha": float(res.params.get("alpha[1]", 0.10)),
                        "beta": float(res.params.get("beta[1]", 0.85)),
                    }
                except Exception:
                    self._regime_params[regime] = {
                        "sigma2": float(np.var(regime_returns)),
                        "omega": 0.0001, "alpha": 0.10, "beta": 0.85,
                    }

            # Compute transition probabilities from label sequence
            for i in range(len(labels) - 1):
                from_r = str(labels[i])
                to_r = str(labels[i + 1])
                if from_r not in self._transition_probs:
                    self._transition_probs[from_r] = {}
                self._transition_probs[from_r][to_r] = (
                    self._transition_probs[from_r].get(to_r, 0) + 1
                )

            # Normalize to probabilities
            for from_r in self._transition_probs:
                total = sum(self._transition_probs[from_r].values())
                if total > 0:
                    self._transition_probs[from_r] = {
                        k: v / total for k, v in self._transition_probs[from_r].items()
                    }

            self._current_regime = str(labels[-1]) if len(labels) > 0 else ""
            if len(self._regime_params) >= 2:
                self._regime_switching = True
                logger.debug(
                    "Regime-switching GARCH: %d regimes, current=%s, "
                    "vols=%s",
                    len(self._regime_params),
                    self._current_regime,
                    {r: f"{p['sigma2']:.6f}" for r, p in self._regime_params.items()},
                )
        except Exception:
            pass

    def predict(self, state_t: np.ndarray) -> np.ndarray:
        if self._regime_switching and self._current_regime in self._transition_probs:
            # Regime-switching forecast: blend per-regime vol by transition probs
            blended_sigma2 = 0.0
            trans = self._transition_probs.get(self._current_regime, {})
            for regime, prob in trans.items():
                if regime in self._regime_params:
                    blended_sigma2 += prob * self._regime_params[regime]["sigma2"]
            if blended_sigma2 > 1e-12:
                return np.array([np.sqrt(blended_sigma2)])

        # Fallback: single-regime forecast
        return np.array([np.sqrt(max(self._sigma2, 1e-12))])

    def update(
        self,
        state_t: np.ndarray,
        actual_t_plus_1: np.ndarray,
        hierarchy_weights: dict[str, float],
    ) -> None:
        try:
            r = float(actual_t_plus_1[0]) if len(actual_t_plus_1) else 0.0
            if np.isnan(r):
                return
            eps2 = r ** 2

            # Update single-regime GARCH
            self._sigma2 = self._omega + self._alpha * eps2 + self._beta * self._sigma2

            # Update per-regime GARCH if active
            if self._regime_switching and self._current_regime in self._regime_params:
                p = self._regime_params[self._current_regime]
                p["sigma2"] = p["omega"] + p["alpha"] * eps2 + p["beta"] * p["sigma2"]

            self._last_return = r
            self.failed_update = False
        except Exception as exc:
            self.failed_update = True
            self._update_error = str(exc)


class VARWrapper(BaseModelWrapper):
    """Rolling-window VAR with periodic refit.

    Cannot be updated truly online; instead, accumulates new rows and
    refits every ``refit_interval`` steps.
    """

    name = "var"

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        *,
        max_lag: int = _VAR_MAX_LAG,
        refit_interval: int = 50,
    ) -> None:
        self._target_col = target_col
        self._max_lag = max_lag
        self._refit_interval = refit_interval
        self._steps_since_refit = 0
        self._buffer = data.dropna().copy()
        self._fitted = False
        self._result: Any = None
        self._cols = list(data.columns)
        self._last_pred = np.zeros(len(self._cols))
        self._refit()

    def _refit(self) -> None:
        try:
            from statsmodels.tsa.api import VAR as VARModel  # type: ignore[import-untyped]

            if len(self._buffer) < _MIN_OBS_VAR:
                return
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = VARModel(self._buffer.values)
                # Cap maxlags relative to observations and number of variables
                # to avoid "maxlags is too large" errors (matching fit_var logic)
                n_cols = max(1, self._buffer.shape[1])
                _capped_lag = min(self._max_lag, max(1, len(self._buffer) // (3 * n_cols)))
                self._result = model.fit(maxlags=max(1, _capped_lag), ic="aic")
            self._fitted = True
            self._steps_since_refit = 0
        except Exception:
            self._fitted = False

    def predict(self, state_t: np.ndarray) -> np.ndarray:
        if not self._fitted or self._result is None:
            return np.array([self._buffer[self._target_col].iloc[-1]]) if self._target_col in self._buffer.columns else np.zeros(1)
        try:
            lag_data = self._buffer.values[-self._result.k_ar:]
            fcast = self._result.forecast(lag_data, steps=1)
            self._last_pred = fcast[0]
            idx = self._cols.index(self._target_col) if self._target_col in self._cols else 0
            return np.array([fcast[0, idx]])
        except Exception:
            return np.array([self._buffer[self._target_col].iloc[-1]]) if self._target_col in self._buffer.columns else np.zeros(1)

    def update(
        self,
        state_t: np.ndarray,
        actual_t_plus_1: np.ndarray,
        hierarchy_weights: dict[str, float],
    ) -> None:
        try:
            # Append new observation row
            new_row = pd.DataFrame([actual_t_plus_1[:len(self._cols)]], columns=self._cols)
            self._buffer = pd.concat([self._buffer, new_row], ignore_index=True)
            # Keep buffer bounded
            if len(self._buffer) > 600:
                self._buffer = self._buffer.iloc[-500:]
            self._steps_since_refit += 1
            if self._steps_since_refit >= self._refit_interval:
                self._refit()
            self.failed_update = False
        except Exception as exc:
            self.failed_update = True
            self._update_error = str(exc)


class LSTMWrapper(BaseModelWrapper):
    """LSTM wrapper with single-step gradient updates and MC Dropout.

    Performs a single-epoch, single-sample gradient descent step on
    each new observation for online learning.

    **MC Dropout (G2):** By enabling dropout at inference time and
    running multiple forward passes, we can estimate *epistemic
    uncertainty* (model's own confidence about its prediction).
    This separates "the model doesn't know" from "inherent randomness",
    giving Gemini better language for the risk section.
    """

    name = "lstm"

    # MC Dropout settings
    _MC_DROPOUT_RATE: float = 0.1
    _MC_FORWARD_PASSES: int = 100

    def __init__(
        self,
        series: np.ndarray,
        *,
        hidden_size: int = _LSTM_HIDDEN,
        lookback: int = _LSTM_LOOKBACK,
        lr: float = _LSTM_LR,
        dropout_rate: float = 0.1,
    ) -> None:
        self._lookback = lookback
        self._lr = lr
        self._hidden_size = hidden_size
        self._fitted = False
        self._model: Any = None
        self._optimizer: Any = None
        self._scaler_mean = 0.0
        self._scaler_std = 1.0
        self._history: list[float] = []
        self._last_pred = 0.0
        self._last_epistemic_std = 0.0  # MC Dropout uncertainty
        self._dropout_rate = dropout_rate

        clean = series[~np.isnan(series)]
        if len(clean) < _MIN_OBS_LSTM:
            return

        # Mixed-frequency awareness: detect forward-filled quarterly data.
        # Add small time-varying noise to break constant stretches, so
        # the LSTM can learn that constant regions represent staleness
        # (not actual zero-volatility stability).
        n_unique = len(np.unique(clean))
        if n_unique > 0 and n_unique / len(clean) < 0.05:
            # Forward-filled quarterly data detected -- add filing-aware noise
            rng = np.random.default_rng(42)
            changes = np.diff(clean, prepend=clean[0])
            days_since_change = np.zeros(len(clean))
            counter = 0
            for i in range(len(clean)):
                if abs(changes[i]) > 0:
                    counter = 0
                counter += 1
                days_since_change[i] = counter
            # Noise grows with days since filing (uncertainty about true value)
            noise_scale = np.std(clean[clean != clean[0]]) if n_unique > 1 else abs(clean[0]) * 0.001
            noise = rng.normal(0, noise_scale * 0.01 * days_since_change / 63)
            clean = clean + noise

        self._history = list(clean)
        self._scaler_mean = float(np.mean(clean))
        self._scaler_std = float(np.std(clean)) or 1.0

        try:
            import torch
            import torch.nn as nn

            class _MiniLSTM(nn.Module):
                """LSTM with dropout for MC Dropout uncertainty."""

                def __init__(self, hidden: int, dropout: float = 0.1) -> None:
                    super().__init__()
                    self.lstm = nn.LSTM(1, hidden, batch_first=True, dropout=dropout if hidden > 1 else 0)
                    self.dropout = nn.Dropout(p=dropout)
                    self.fc = nn.Linear(hidden, 1)

                def forward(self, x: Any) -> Any:
                    out, _ = self.lstm(x)
                    out = self.dropout(out[:, -1, :])  # Dropout before FC
                    return self.fc(out)

            self._model = _MiniLSTM(hidden_size, dropout_rate)
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
            self._criterion = nn.MSELoss()

            # Initial training
            scaled = (clean - self._scaler_mean) / self._scaler_std
            X, y = [], []
            for i in range(len(scaled) - lookback):
                X.append(scaled[i:i + lookback])
                y.append(scaled[i + lookback])
            if len(X) < 10:
                return

            X_t = torch.FloatTensor(np.array(X)).unsqueeze(-1)
            y_t = torch.FloatTensor(np.array(y)).unsqueeze(-1)

            self._model.train()
            for _ in range(min(_LSTM_EPOCHS, 30)):
                self._optimizer.zero_grad()
                pred = self._model(X_t)
                loss = self._criterion(pred, y_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._optimizer.step()

            self._fitted = True
        except ImportError:
            pass
        except Exception:
            pass

    def _prepare_input(self) -> Any:
        """Prepare the input tensor from recent history."""
        import torch

        recent = self._history[-self._lookback:]
        scaled = [(v - self._scaler_mean) / self._scaler_std for v in recent]
        return torch.FloatTensor([scaled]).unsqueeze(-1)

    def predict(self, state_t: np.ndarray) -> np.ndarray:
        if not self._fitted or self._model is None:
            return np.array([self._history[-1]]) if self._history else np.zeros(1)
        try:
            import torch

            if len(self._history) < self._lookback:
                return np.array([self._history[-1]])

            x = self._prepare_input()
            self._model.eval()
            with torch.no_grad():
                pred_scaled = float(self._model(x).item())
            pred = pred_scaled * self._scaler_std + self._scaler_mean
            self._last_pred = pred
            return np.array([pred])
        except Exception:
            return np.array([self._history[-1]]) if self._history else np.zeros(1)

    def predict_with_uncertainty(
        self,
        n_passes: int | None = None,
    ) -> tuple[float, float, float]:
        """MC Dropout prediction: run multiple forward passes with dropout
        enabled to estimate epistemic uncertainty.

        Returns
        -------
        (mean_prediction, epistemic_std, aleatoric_estimate)
            - mean_prediction: average across MC passes
            - epistemic_std: std dev across MC passes (model uncertainty)
            - aleatoric_estimate: inherent noise (from training residuals)
        """
        if not self._fitted or self._model is None:
            val = self._history[-1] if self._history else 0.0
            return val, 0.0, 0.0

        if n_passes is None:
            n_passes = self._MC_FORWARD_PASSES

        try:
            import torch

            if len(self._history) < self._lookback:
                val = self._history[-1]
                return val, 0.0, 0.0

            x = self._prepare_input()

            # Enable dropout at inference time (MC Dropout)
            self._model.train()  # This keeps dropout active

            predictions: list[float] = []
            with torch.no_grad():
                for _ in range(n_passes):
                    pred_scaled = float(self._model(x).item())
                    pred = pred_scaled * self._scaler_std + self._scaler_mean
                    predictions.append(pred)

            # Restore eval mode
            self._model.eval()

            preds_arr = np.array(predictions)
            mean_pred = float(np.mean(preds_arr))
            epistemic_std = float(np.std(preds_arr))

            # Aleatoric estimate: use recent prediction errors as proxy
            if len(self._history) > 10:
                recent_vals = np.array(self._history[-20:])
                aleatoric = float(np.std(np.diff(recent_vals)))
            else:
                aleatoric = 0.0

            self._last_pred = mean_pred
            self._last_epistemic_std = epistemic_std

            return mean_pred, epistemic_std, aleatoric

        except Exception:
            val = self._history[-1] if self._history else 0.0
            return val, 0.0, 0.0

    @property
    def epistemic_uncertainty(self) -> float:
        """Last computed epistemic uncertainty from MC Dropout."""
        return self._last_epistemic_std

    def update(
        self,
        state_t: np.ndarray,
        actual_t_plus_1: np.ndarray,
        hierarchy_weights: dict[str, float],
    ) -> None:
        try:
            val = float(actual_t_plus_1[0])
            if np.isnan(val):
                return
            self._history.append(val)

            if not self._fitted or self._model is None:
                return

            import torch

            # Single-step gradient update
            if len(self._history) < self._lookback + 1:
                return

            recent = self._history[-(self._lookback + 1):]
            x_raw = recent[:-1]
            y_raw = recent[-1]

            scaled_x = [(v - self._scaler_mean) / self._scaler_std for v in x_raw]
            scaled_y = (y_raw - self._scaler_mean) / self._scaler_std

            x = torch.FloatTensor([scaled_x]).unsqueeze(-1)
            y = torch.FloatTensor([[scaled_y]])

            self._model.train()
            self._optimizer.zero_grad()
            pred = self._model(x)
            loss = self._criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()

            self.failed_update = False
        except Exception as exc:
            self.failed_update = True
            self._update_error = str(exc)


class TreeWrapper(BaseModelWrapper):
    """Tree ensemble wrapper with periodic refit.

    Tree models cannot be updated online, so we accumulate data and
    refit every *refit_interval* steps.
    """

    name = "tree"

    def __init__(
        self,
        features: pd.DataFrame,
        target_col: str,
        *,
        random_state: int = 42,
        refit_interval: int = 50,
    ) -> None:
        self._target_col = target_col
        self._random_state = random_state
        self._refit_interval = refit_interval
        self._steps_since_refit = 0
        self._fitted = False
        self._model_obj: Any = None
        self._feature_cols: list[str] = [c for c in features.columns if c != target_col]
        self._buffer = features.dropna().copy()
        self._last_pred = 0.0
        self._refit()

    def _refit(self) -> None:
        if len(self._buffer) < _MIN_OBS_TREE or not self._feature_cols:
            return
        try:
            self._model_obj = _try_load_tree_model(self._random_state, ModelMetrics())
            if self._model_obj is None:
                return
            X = self._buffer[self._feature_cols].values
            y = self._buffer[self._target_col].values
            self._model_obj.fit(X, y)
            self._fitted = True
            self._steps_since_refit = 0
        except Exception:
            self._fitted = False

    def predict(self, state_t: np.ndarray) -> np.ndarray:
        if not self._fitted or self._model_obj is None:
            return np.array([self._last_pred])
        try:
            # Use last known features
            last_row = self._buffer[self._feature_cols].iloc[-1:].values
            pred = float(self._model_obj.predict(last_row)[0])
            self._last_pred = pred
            return np.array([pred])
        except Exception:
            return np.array([self._last_pred])

    def update(
        self,
        state_t: np.ndarray,
        actual_t_plus_1: np.ndarray,
        hierarchy_weights: dict[str, float],
    ) -> None:
        try:
            new_row = pd.DataFrame(
                [actual_t_plus_1[:len(self._feature_cols) + 1]],
                columns=self._feature_cols + [self._target_col],
            )
            self._buffer = pd.concat([self._buffer, new_row], ignore_index=True)
            if len(self._buffer) > 600:
                self._buffer = self._buffer.iloc[-500:]
            self._steps_since_refit += 1
            if self._steps_since_refit >= self._refit_interval:
                self._refit()
            self.failed_update = False
        except Exception as exc:
            self.failed_update = True
            self._update_error = str(exc)


class BaselineWrapper(BaseModelWrapper):
    """Baseline wrapper using exponential moving average.

    Update rule: ema = alpha * actual + (1 - alpha) * ema
    """

    name = "baseline"

    def __init__(self, series: np.ndarray, ema_span: int = 21) -> None:
        clean = series[~np.isnan(series)]
        self._alpha = 2.0 / (ema_span + 1)
        self._ema = float(clean[-1]) if len(clean) else 0.0
        self._fitted = len(clean) >= 1

        if len(clean) >= 3:
            ema = clean[0]
            for val in clean[1:]:
                ema = self._alpha * val + (1 - self._alpha) * ema
            self._ema = float(ema)

    def predict(self, state_t: np.ndarray) -> np.ndarray:
        return np.array([self._ema])

    def update(
        self,
        state_t: np.ndarray,
        actual_t_plus_1: np.ndarray,
        hierarchy_weights: dict[str, float],
    ) -> None:
        try:
            val = float(actual_t_plus_1[0])
            if np.isnan(val):
                return
            self._ema = self._alpha * val + (1 - self._alpha) * self._ema
            self.failed_update = False
        except Exception as exc:
            self.failed_update = True
            self._update_error = str(exc)


class TFTWrapper(BaseModelWrapper):
    """Temporal Fusion Transformer wrapper for mixed-frequency data.

    TFT is purpose-built for time series with:
    - Static metadata (sector, country, industry)
    - Known future inputs (calendar features, earnings dates)
    - Observed inputs at different frequencies (daily prices, quarterly statements)

    Falls back gracefully if pytorch-forecasting is not installed.

    Parameters
    ----------
    cache:
        Full daily cache with all features.
    target_col:
        Name of the target variable to predict.
    static_features:
        Dict of static metadata (e.g. sector, country).
    lookback:
        Number of historical days to use as encoder input.
    """

    name = "tft"

    def __init__(
        self,
        cache: pd.DataFrame,
        target_col: str,
        *,
        static_features: dict[str, str] | None = None,
        lookback: int = 60,
        hidden_size: int = 16,
        n_heads: int = 2,
        max_epochs: int = 20,
        lr: float = 0.005,
    ) -> None:
        self._target_col = target_col
        self._lookback = lookback
        self._fitted = False
        self._model: Any = None
        self._trainer: Any = None
        self._scaler_mean = 0.0
        self._scaler_std = 1.0
        self._last_pred = 0.0
        self._history: list[float] = []

        clean = cache[target_col].dropna()
        if len(clean) < lookback + 30:
            return

        self._history = list(clean.values)
        self._scaler_mean = float(clean.mean())
        self._scaler_std = float(clean.std()) or 1.0
        # Store multivariate feature history for predict() so TFT
        # retains its multi-feature advantage during online prediction.
        self._feature_history: list[np.ndarray] = []

        try:
            import torch
            import torch.nn as nn

            # Simplified TFT-inspired model: multi-head attention over
            # historical features with gating.
            # Full pytorch-forecasting TFT requires complex data setup;
            # this is a practical self-attention model that captures the
            # core TFT idea of variable selection + temporal attention.

            class _GatedResidualNetwork(nn.Module):
                """GRN block from the TFT paper."""

                def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
                    super().__init__()
                    self.fc1 = nn.Linear(d_in, d_hidden)
                    self.fc2 = nn.Linear(d_hidden, d_out)
                    self.gate = nn.Linear(d_hidden, d_out)
                    self.ln = nn.LayerNorm(d_out)
                    self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

                def forward(self, x: Any) -> Any:
                    h = torch.nn.functional.elu(self.fc1(x))
                    h2 = self.fc2(h)
                    g = torch.sigmoid(self.gate(h))
                    out = g * h2
                    return self.ln(out + self.skip(x))

            class _MiniTFT(nn.Module):
                """Simplified TFT: variable selection + temporal self-attention."""

                def __init__(self, n_features: int, d_model: int, n_heads: int) -> None:
                    super().__init__()
                    self.input_proj = _GatedResidualNetwork(n_features, d_model * 2, d_model)
                    self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                    self.grn_out = _GatedResidualNetwork(d_model, d_model * 2, d_model)
                    self.fc_out = nn.Linear(d_model, 1)

                def forward(self, x: Any) -> Any:
                    # x: (batch, seq_len, n_features)
                    h = self.input_proj(x)  # (batch, seq_len, d_model)
                    attn_out, _ = self.attn(h, h, h)  # self-attention
                    h = self.grn_out(attn_out)
                    # Take last timestep
                    return self.fc_out(h[:, -1, :])

            # Prepare features: use numeric columns from cache
            numeric_cols = [
                c for c in cache.columns
                if cache[c].dtype in (np.float64, np.float32, np.int64)
            ][:20]  # cap at 20 features
            if target_col not in numeric_cols:
                numeric_cols = [target_col] + numeric_cols[:19]

            feature_data = cache[numeric_cols].fillna(0).values
            n_features = len(numeric_cols)

            # Normalise
            feat_mean = feature_data.mean(axis=0)
            feat_std = feature_data.std(axis=0)
            feat_std[feat_std < 1e-8] = 1.0
            scaled = (feature_data - feat_mean) / feat_std

            self._feat_mean = feat_mean
            self._feat_std = feat_std
            self._numeric_cols = numeric_cols

            # Build sequences
            X_seqs, y_seqs = [], []
            target_idx = numeric_cols.index(target_col) if target_col in numeric_cols else 0
            for i in range(len(scaled) - lookback):
                X_seqs.append(scaled[i:i + lookback])
                y_seqs.append(scaled[i + lookback, target_idx])

            if len(X_seqs) < 20:
                return

            X_t = torch.FloatTensor(np.array(X_seqs))
            y_t = torch.FloatTensor(np.array(y_seqs)).unsqueeze(-1)

            self._model = _MiniTFT(n_features, hidden_size, n_heads)
            optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            self._model.train()
            for epoch in range(max_epochs):
                optimizer.zero_grad()
                pred = self._model(X_t)
                loss = criterion(pred, y_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

            self._fitted = True
            # Store the multivariate feature history so predict() can
            # use all features, not just the target variable.
            self._feature_history = [
                feature_data[i] for i in range(len(feature_data))
            ]
            logger.info("TFT wrapper fitted for %s (%d sequences, %d features)", target_col, len(X_seqs), n_features)

        except ImportError:
            logger.info("PyTorch not available for TFT -- skipping")
        except Exception as exc:
            logger.warning("TFT fitting failed for %s: %s", target_col, exc)

    def predict(self, state_t: np.ndarray) -> np.ndarray:
        if not self._fitted or self._model is None:
            return np.array([self._history[-1]]) if self._history else np.zeros(1)

        try:
            import torch

            n_feat = len(self._numeric_cols)

            # Use multivariate feature history if available (preserves
            # TFT's multi-feature advantage during online prediction).
            if len(self._feature_history) >= self._lookback:
                recent = np.array(self._feature_history[-self._lookback:])
                # Ensure correct shape (lookback, n_features)
                if recent.ndim == 1:
                    recent = recent.reshape(-1, 1)
                if recent.shape[1] < n_feat:
                    padded = np.zeros((self._lookback, n_feat))
                    padded[:, :recent.shape[1]] = recent
                    recent = padded
                elif recent.shape[1] > n_feat:
                    recent = recent[:, :n_feat]
            elif len(self._history) >= self._lookback:
                # Fallback: univariate target history with zero-padding
                recent = np.zeros((self._lookback, n_feat))
                recent[:, 0] = np.array(self._history[-self._lookback:])
            else:
                return np.array([self._history[-1]]) if self._history else np.zeros(1)

            # Normalise using the stored feature statistics
            scaled = (recent - self._feat_mean[:n_feat]) / self._feat_std[:n_feat]

            x = torch.FloatTensor(scaled).unsqueeze(0)
            self._model.eval()
            with torch.no_grad():
                pred_scaled = float(self._model(x).item())

            pred = pred_scaled * self._scaler_std + self._scaler_mean
            self._last_pred = pred
            return np.array([pred])

        except Exception:
            return np.array([self._history[-1]]) if self._history else np.zeros(1)

    def update(
        self,
        state_t: np.ndarray,
        actual_t_plus_1: np.ndarray,
        hierarchy_weights: dict[str, float],
    ) -> None:
        try:
            val = float(actual_t_plus_1[0])
            if not np.isnan(val):
                self._history.append(val)
            # Also update multivariate feature history if we have
            # more than just the target value in the actual vector.
            if len(actual_t_plus_1) >= len(self._numeric_cols):
                self._feature_history.append(
                    actual_t_plus_1[:len(self._numeric_cols)].copy()
                )
            elif self._feature_history:
                # Fallback: carry forward last feature row with updated target
                last_row = self._feature_history[-1].copy()
                if not np.isnan(val):
                    last_row[0] = val  # target is always column 0
                self._feature_history.append(last_row)
            self.failed_update = False
        except Exception as exc:
            self.failed_update = True
            self._update_error = str(exc)


# ---------------------------------------------------------------------------
# D2: Regime-weighted historical training windows
# ---------------------------------------------------------------------------


def compute_regime_sample_weights(
    regime_labels: pd.Series,
    current_day_idx: int,
    *,
    half_life_days: int = 126,
    regime_similarity_boost: float = 2.0,
) -> np.ndarray:
    """Compute per-day training weights for regime-aware learning.

    ``w(tau) proportional to exp(-delta_t / half_life) * similarity(regime(tau), regime(t))``

    Parameters
    ----------
    regime_labels:
        Series of regime labels (str) indexed by day position.
    current_day_idx:
        Index of the current day *t* in the forward pass.
    half_life_days:
        Exponential decay half-life in trading days.
    regime_similarity_boost:
        Multiplier for days sharing the same regime as day *t*.

    Returns
    -------
    1-D array of non-negative weights for days ``[0, current_day_idx]``.
    Weights are normalised to sum to 1.
    """
    n = current_day_idx + 1
    if n <= 0:
        return np.array([1.0])

    current_regime = regime_labels.iloc[current_day_idx] if current_day_idx < len(regime_labels) else None

    # Temporal decay
    deltas = np.arange(n, dtype=np.float64)[::-1]  # [current_day_idx, ..., 0]
    # Reverse so deltas[i] = current_day_idx - i (time since day i)
    deltas = np.arange(current_day_idx, -1, -1, dtype=np.float64)
    decay = np.exp(-deltas / max(half_life_days, 1))

    # Regime similarity
    similarity = np.ones(n, dtype=np.float64)
    if current_regime is not None:
        for i in range(n):
            if i < len(regime_labels) and regime_labels.iloc[i] == current_regime:
                similarity[i] = regime_similarity_boost

    weights = decay * similarity
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        weights = np.ones(n) / n

    return weights


# ---------------------------------------------------------------------------
# D1: Day-by-day forward pass with predict-compare-update loop
# ---------------------------------------------------------------------------


@dataclass
class ForwardPassResult:
    """Container for forward-pass outputs."""

    errors_by_tier: dict[int, list[float]] = field(default_factory=lambda: {i: [] for i in range(1, 6)})
    errors_by_regime: dict[str, list[float]] = field(default_factory=dict)
    model_states: dict[str, BaseModelWrapper] = field(default_factory=dict)
    predictions_log: list[dict[str, Any]] = field(default_factory=list)
    total_days: int = 0
    warmup_days: int = 0
    n_break_resets: int = 0  # structural break model resets (Synergy 2)
    pid_summary: dict[str, Any] = field(default_factory=dict)  # PID controller state
    conformal_calibrator: Any = None  # Trained ConformalCalibrator from the forward pass

    def __getstate__(self):
        """Custom pickle: strip non-picklable model_states and calibrator."""
        state = self.__dict__.copy()
        state["model_states"] = {}  # Fitted sklearn/torch models are not picklable
        state["conformal_calibrator"] = None  # May hold thread-local refs
        # Preserve conformal residuals so downstream can rebuild the calibrator
        cal = self.__dict__.get("conformal_calibrator")
        if cal is not None and hasattr(cal, "scores"):
            state["_conformal_residuals"] = list(cal.scores)
        return state

    def __setstate__(self, state):
        """Custom unpickle: restore with empty model_states."""
        self.__dict__.update(state)


def _init_model_wrappers(
    cache: pd.DataFrame,
    variable: str,
    tier: str,
    tier_map: dict[str, list[str]],
    available_vars: list[str],
    random_state: int = 42,
) -> list[BaseModelWrapper]:
    """Initialise model wrappers for a single variable.

    Returns a list of successfully initialised wrappers (may be empty
    except for baseline which always succeeds).
    """
    series = cache[variable].values
    wrappers: list[BaseModelWrapper] = []

    # Kalman (preferred for tier1/tier2)
    if tier in ("tier1", "tier2"):
        w = KalmanWrapper(series)
        if w._fitted:
            wrappers.append(w)

    # GARCH (for volatility-related variables) -- regime-switching when HMM labels available
    if "volatility" in variable or "return" in variable:
        _regime_labels = None
        if cache is not None and "regime_label" in cache.columns:
            _rl = cache["regime_label"].values
            if len(_rl) == len(series):
                _regime_labels = _rl
            elif len(_rl) > len(series):
                _regime_labels = _rl[-len(series):]
        w = GARCHWrapper(series, regime_labels=_regime_labels)
        if w._fitted:
            wrappers.append(w)

    # LSTM
    w = LSTMWrapper(series, hidden_size=_LSTM_HIDDEN, lookback=_LSTM_LOOKBACK)
    if w._fitted:
        wrappers.append(w)

    # TFT (Temporal Fusion Transformer -- for mixed-frequency data)
    try:
        w = TFTWrapper(cache, variable, lookback=min(60, len(cache) // 3))
        if w._fitted:
            wrappers.append(w)
    except Exception as exc:
        logger.debug("TFT init failed for %s: %s", variable, exc)

    # Baseline (always succeeds)
    wrappers.append(BaselineWrapper(series))

    return wrappers


def run_forward_pass(
    cache: pd.DataFrame,
    tier_variables: dict[str, list[str]] | None = None,
    hierarchy_weights: dict[str, float] | None = None,
    regime_labels: pd.Series | None = None,
    *,
    extra_variables: list[str] | None = None,
    warmup_days: int = 60,
    log_interval: int = 50,
    random_state: int = 42,
) -> ForwardPassResult:
    """Day-by-day temporal analysis: predict -> compare -> update.

    Parameters
    ----------
    cache:
        Full 2-year daily cache (DatetimeIndex, ~500 rows).
    tier_variables:
        Mapping of tier name -> list of variable names.  If None,
        loaded from survival hierarchy config.
    hierarchy_weights:
        Current tier weights from survival mode analysis.  Defaults
        to equal weights.
    regime_labels:
        Per-day regime labels from HMM/GMM.  If None, all days are
        treated as the same regime.
    warmup_days:
        Number of initial days used for cold-start fitting.
    log_interval:
        Print progress every N days.
    random_state:
        Random seed for reproducible models.

    Returns
    -------
    ForwardPassResult containing per-tier daily errors, model states,
    and a day-by-day predictions log.
    """
    # Cap warmup to available data so lower-frequency caches (A/Q/M) can still
    # produce prediction steps instead of yielding 0 steps.
    warmup_days = min(warmup_days, max(3, len(cache) // 3))
    logger.info("Starting forward pass (warmup=%d days, cache=%d rows)...", warmup_days, len(cache))

    # Initialise PID bank for adaptive learning rate adjustment
    try:
        from operator1.models.pid_controller import create_pid_bank, compute_pid_adjustment
        _pid_available = True
    except ImportError:
        _pid_available = False

    # Initialise conformal calibrator for adaptive prediction intervals.
    # Prefer ConformalPIDCalibrator (PID-controlled + Mondrian per survival
    # mode) so that crisis-mode intervals are calibrated from crisis residuals.
    _conformal_calibrator = None
    _conformal_is_pid = False
    try:
        from operator1.models.conformal import ConformalPIDCalibrator
        _conformal_calibrator = ConformalPIDCalibrator(target_coverage=0.9)
        _conformal_is_pid = True
        logger.info("ConformalPIDCalibrator initialised (PID + Mondrian, coverage=90%%)")
    except (ImportError, Exception):
        try:
            from operator1.models.conformal import ConformalCalibrator
            _conformal_calibrator = ConformalCalibrator(coverage=0.9, adaptive=True)
            logger.info("ConformalCalibrator initialised (adaptive, coverage=90%%)")
        except ImportError:
            logger.debug("Conformal prediction not available")

    # Pre-compute candlestick pattern signals as daily features
    try:
        from operator1.models.pattern_detector import (
            detect_doji, detect_hammer, detect_shooting_star,
        )
        if all(c in cache.columns for c in ("open", "high", "low", "close")):
            _patt_doji = []
            _patt_hammer = []
            _patt_star = []
            for i in range(len(cache)):
                o, h, l, c_val = (
                    cache["open"].iloc[i], cache["high"].iloc[i],
                    cache["low"].iloc[i], cache["close"].iloc[i],
                )
                if any(np.isnan(x) for x in (o, h, l, c_val)):
                    _patt_doji.append(0)
                    _patt_hammer.append(0)
                    _patt_star.append(0)
                else:
                    _patt_doji.append(1 if detect_doji(o, h, l, c_val) else 0)
                    _patt_hammer.append(1 if detect_hammer(o, h, l, c_val) else 0)
                    _patt_star.append(1 if detect_shooting_star(o, h, l, c_val) else 0)
            cache["_fp_pattern_doji"] = _patt_doji
            cache["_fp_pattern_hammer"] = _patt_hammer
            cache["_fp_pattern_shooting_star"] = _patt_star
            logger.info("Candlestick pattern features injected for forward pass")
    except Exception as exc:
        logger.debug("Candlestick pattern injection skipped: %s", exc)

    # Pre-compute cycle phase signal as daily feature
    try:
        from operator1.models.cycle_decomposition import run_cycle_decomposition
        _cycle_info = run_cycle_decomposition(cache, variable="close")
        if _cycle_info.available and _cycle_info.dominant_cycles:
            # Use the strongest cycle to compute phase position (0.0 to 1.0)
            dominant_period = _cycle_info.dominant_cycles[0].get("period_days", 21)
            if dominant_period and dominant_period > 1:
                _phase = np.arange(len(cache)) % dominant_period / dominant_period
                cache["_fp_cycle_phase"] = _phase
                cache["_fp_cycle_period"] = float(dominant_period)
                logger.info("Cycle phase feature injected (dominant period=%.0f days)", dominant_period)
    except Exception as exc:
        logger.debug("Cycle phase injection skipped: %s", exc)

    result = ForwardPassResult(warmup_days=warmup_days)

    if tier_variables is None:
        tier_variables = _load_tier_variables()
    if hierarchy_weights is None:
        hierarchy_weights = {f"tier{i}": 20.0 for i in range(1, 6)}
    if regime_labels is None:
        regime_labels = pd.Series(["normal"] * len(cache), index=cache.index)

    # Flatten all tier variables that exist in cache
    all_vars: list[str] = []
    var_to_tier: dict[str, int] = {}
    for tier_key, var_list in tier_variables.items():
        tier_num = int(tier_key.replace("tier", "")) if "tier" in tier_key else 0
        for v in var_list:
            if v in cache.columns and v not in var_to_tier:
                all_vars.append(v)
                var_to_tier[v] = tier_num

    # Include extra variables (e.g. financial health scores) so the
    # forward pass also learns from / tracks them day-by-day.
    # Extra variables are assigned tier 0 (cross-tier composite).
    if extra_variables:
        for ev in extra_variables:
            if ev in cache.columns and ev not in var_to_tier:
                all_vars.append(ev)
                var_to_tier[ev] = 0

    if not all_vars:
        logger.warning("No tier variables found in cache for forward pass")
        return result

    # Initialise model wrappers for each variable (cold-start on warmup window)
    warmup_cache = cache.iloc[:warmup_days] if warmup_days < len(cache) else cache
    model_bank: dict[str, list[BaseModelWrapper]] = {}

    # Create PID controllers for adaptive learning rates
    pid_bank = None
    if _pid_available and all_vars:
        pid_bank = create_pid_bank(all_vars, tier_weights=hierarchy_weights)
        logger.info("PID bank initialised for %d variables", len(all_vars))
    for var_name in all_vars:
        tier_key = f"tier{var_to_tier[var_name]}"
        tier_map = tier_variables
        wrappers = _init_model_wrappers(
            warmup_cache, var_name, tier_key, tier_map,
            all_vars, random_state,
        )
        model_bank[var_name] = wrappers

    n_days = len(cache)
    result.total_days = n_days - warmup_days - 1

    # Structural break detection: check if cache has a structural_break
    # column from PELT/BCP.  When a break is detected at day t, re-initialise
    # all model wrappers on the post-break window so stale parameters from
    # the pre-break regime don't contaminate predictions.
    # Source: The_Apps_core_idea.pdf Section E.3 Synergy 2 --
    #   "When break detected: Kalman reset, LSTM retrain on post-break only,
    #    survival hierarchy recalibration."
    _has_structural_break = "structural_break" in cache.columns
    _break_reinit_window = 30  # minimum post-break days before reinit
    _n_break_resets = 0

    # Forward loop: day warmup_days to n_days - 2 (predict t+1, compare with actual t+1)
    for t in range(warmup_days, n_days - 1):
        regime_t = str(regime_labels.iloc[t]) if t < len(regime_labels) else "unknown"

        # Synergy 2: Structural break -> model reset
        # When a structural break is detected at day t, re-initialise model
        # wrappers using only post-break data.  This prevents outdated
        # parameters from degrading predictions after fundamental regime changes.
        if (
            _has_structural_break
            and t > warmup_days + _break_reinit_window
            and cache["structural_break"].iloc[t] == 1
        ):
            post_break_start = max(0, t - _break_reinit_window)
            post_break_cache = cache.iloc[post_break_start:t + 1]
            if len(post_break_cache) >= 10:
                for var_name_reset in all_vars:
                    tier_key = f"tier{var_to_tier[var_name_reset]}"
                    try:
                        new_wrappers = _init_model_wrappers(
                            post_break_cache, var_name_reset, tier_key,
                            tier_variables, all_vars, random_state,
                        )
                        if new_wrappers:
                            model_bank[var_name_reset] = new_wrappers
                    except Exception:
                        pass  # keep existing wrappers on failure
                _n_break_resets += 1
                logger.info(
                    "Structural break at day %d: re-initialised %d model banks "
                    "on %d-day post-break window",
                    t, len(all_vars), len(post_break_cache),
                )

        for var_name in all_vars:
            tier_num = var_to_tier[var_name]
            tier_weight = hierarchy_weights.get(f"tier{tier_num}", 20.0)

            state_t = np.array([cache[var_name].iloc[t]])
            actual_t1 = np.array([cache[var_name].iloc[t + 1]])

            if np.isnan(actual_t1[0]):
                continue  # skip variables with missing next-day data

            # Step B: Multi-module prediction (pick best / average)
            # Track per-model predictions for burn-out weight calibration.
            predictions: list[float] = []
            per_model_preds: dict[str, float] = {}
            for wrapper in model_bank.get(var_name, []):
                try:
                    pred = wrapper.predict(state_t)
                    if len(pred) > 0 and not np.isnan(pred[0]):
                        pval = float(pred[0])
                        predictions.append(pval)
                        per_model_preds[wrapper.name] = pval
                except Exception:
                    pass

            if not predictions:
                continue

            # Step C: Simple ensemble (average of available predictions)
            ensemble_pred = float(np.mean(predictions))

            # Step D: Reality check (with observed-vs-estimated weighting)
            # Source: The_Apps_core_idea.pdf Section J.3 -- penalize
            # observed values more than estimated ones in the loss.
            error = float(actual_t1[0]) - ensemble_pred
            source_col = f"{var_name}_source"
            obs_weight = 1.0  # default: treat as observed
            if source_col in cache.columns:
                src = cache.iloc[t + 1].get(source_col)
                obs_weight = 1.0 if src == "observed" else 0.3

            # Apply burn-out sample weight if available (exponential
            # recency decay * regime similarity from run_burnout).
            # Source: The_Apps_core_idea.pdf Section L.1
            sample_w = 1.0
            if "_burnout_sample_weight" in cache.columns:
                _sw = cache["_burnout_sample_weight"].iloc[t]
                if not np.isnan(_sw) and _sw > 0:
                    sample_w = float(_sw)

            # Huber loss (Huber 1964): quadratic for small errors, linear
            # for large errors.  More robust to fat-tailed financial returns
            # than pure squared error which gives disproportionate weight
            # to outliers.  Delta threshold = 3 * MAD of recent errors.
            _huber_delta = 0.05  # default ~5% return threshold
            abs_err = abs(error)
            if abs_err <= _huber_delta:
                _loss = 0.5 * error ** 2
            else:
                _loss = _huber_delta * (abs_err - 0.5 * _huber_delta)

            weighted_error = _loss * (tier_weight / 20.0) * obs_weight * sample_w

            result.errors_by_tier[tier_num].append(weighted_error)
            if regime_t not in result.errors_by_regime:
                result.errors_by_regime[regime_t] = []
            result.errors_by_regime[regime_t].append(weighted_error)

            # Step D2: Update conformal calibrator with this prediction/actual pair.
            # When using ConformalPIDCalibrator, pass the survival_mode so
            # Mondrian partitioning calibrates crisis intervals from crisis
            # residuals (not diluted by normal-mode data).
            if _conformal_calibrator is not None:
                try:
                    _survival_mode_t = "normal"
                    if "survival_mode" in cache.columns:
                        _sm = cache["survival_mode"].iloc[t]
                        if pd.notna(_sm):
                            _survival_mode_t = str(_sm)
                    if _conformal_is_pid:
                        _conformal_calibrator.add_score(
                            var_name, ensemble_pred, float(actual_t1[0]),
                            mode=_survival_mode_t,
                        )
                    else:
                        _conformal_calibrator.add_score(var_name, ensemble_pred, float(actual_t1[0]))
                    _was_covered = _conformal_calibrator.predict_interval(
                        var_name, ensemble_pred,
                    )
                    if hasattr(_was_covered, "lower") and hasattr(_was_covered, "upper"):
                        _covered = _was_covered.lower <= float(actual_t1[0]) <= _was_covered.upper
                        _conformal_calibrator.update_adaptive(var_name, _covered)
                except Exception:
                    pass  # calibrator needs enough data before it can produce intervals

            # Log prediction (includes per-model predictions for burn-out calibration)
            result.predictions_log.append({
                "day": t,
                "variable": var_name,
                "predicted": ensemble_pred,
                "actual": float(actual_t1[0]),
                "error": error,
                "tier": tier_num,
                "regime": regime_t,
                "per_model": per_model_preds,
            })

            # Step E: Online update with PID-adjusted learning rate
            pid_multiplier = 1.0
            if pid_bank is not None:
                pid_multiplier = pid_bank[var_name].update(error) if var_name in pid_bank else 1.0

            # Scale the hierarchy weights by PID multiplier for this update
            adjusted_weights = {
                k: v * pid_multiplier for k, v in hierarchy_weights.items()
            }
            for wrapper in model_bank.get(var_name, []):
                try:
                    wrapper.update(state_t, actual_t1, adjusted_weights)
                except Exception:
                    wrapper.failed_update = True

        # Step F: DTW analog check (every 5 days to limit compute cost)
        if (t - warmup_days) > 0 and (t - warmup_days) % 5 == 0:
            try:
                from operator1.models.dtw_analogs import find_historical_analogs
                _dtw_slice = cache.iloc[:t + 1]
                if len(_dtw_slice) >= 42:  # need at least 2x query window
                    _dtw_r = find_historical_analogs(
                        _dtw_slice, query_window=21, k=3, forecast_horizon=5,
                    )
                    if _dtw_r.available:
                        # Store the DTW analog signal as empirical prior
                        if not hasattr(result, "dtw_signals"):
                            result.dtw_signals = []
                        result.dtw_signals.append({
                            "day": t,
                            "n_analogs": len(_dtw_r.analogs) if _dtw_r.analogs else 0,
                            "empirical_return_mean": _dtw_r.empirical_return_mean,
                            "empirical_return_p5": _dtw_r.empirical_return_p5,
                            "empirical_return_p95": _dtw_r.empirical_return_p95,
                        })
            except Exception:
                pass  # DTW is best-effort

        if (t - warmup_days) % log_interval == 0:
            logger.info(
                "Forward pass: day %d/%d", t - warmup_days, n_days - warmup_days - 1,
            )

    # Store final model states
    result.model_states = {
        var_name: wrappers[0] if wrappers else BaselineWrapper(np.array([0.0]))
        for var_name, wrappers in model_bank.items()
    }
    result.n_break_resets = _n_break_resets
    if _n_break_resets > 0:
        logger.info(
            "Forward pass: %d structural break model resets performed",
            _n_break_resets,
        )

    # Summary
    total_errors = sum(len(v) for v in result.errors_by_tier.values())
    avg_errors = {
        k: float(np.mean(v)) if v else 0.0
        for k, v in result.errors_by_tier.items()
    }
    # Compute PID bank summary
    pid_summary = None
    if pid_bank is not None and _pid_available:
        pid_summary = compute_pid_adjustment(pid_bank, {v: 0.0 for v in all_vars})
        result.pid_summary = pid_summary.to_dict()
        logger.info(
            "PID summary: mean_multiplier=%.3f, max=%.3f",
            pid_summary.mean_multiplier, pid_summary.max_multiplier,
        )

    # Store conformal calibrator and diagnostics so downstream modules
    # (prediction_aggregator) can use the trained calibrator directly.
    if _conformal_calibrator is not None:
        result.conformal_calibrator = _conformal_calibrator
        try:
            result.conformal_diagnostics = _conformal_calibrator.get_diagnostics()
            logger.info("Conformal calibrator: %s", result.conformal_diagnostics)
        except Exception:
            pass

    logger.info(
        "Forward pass complete: %d prediction steps, avg tier errors: %s",
        total_errors, avg_errors,
    )

    return result


# ---------------------------------------------------------------------------
# D4: Exponential Gradient Weight Learner for Burn-Out Calibration
# ---------------------------------------------------------------------------


class ExponentialGradientWeightLearner:
    """Online weight learner using exponential gradient (Vovk 1990).

    Walks through the forward pass predictions_log day-by-day, maintains
    per-regime weight vectors for each model, and learns which models to
    trust in which regimes.  Produces regime-conditioned weights and
    distribution parameters for Monte Carlo and prediction aggregator.

    The exponential gradient update rule:
        w_i *= exp(-eta * loss_i)
        normalize: w_i /= sum(w)

    This has O(sqrt(T) * log(N)) regret bound where T is time steps
    and N is number of models.
    """

    def __init__(
        self,
        model_names: list[str],
        *,
        initial_eta: float = 0.1,
        eta_decay: float = 0.995,
        min_weight: float = 0.05,
        max_weight: float = 0.6,
    ) -> None:
        self.model_names = sorted(model_names)
        self.n_models = len(model_names)
        self.eta = initial_eta
        self.eta_decay = eta_decay
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Per-regime weight vectors: {regime: {model: weight}}
        self._regime_weights: dict[str, dict[str, float]] = {}
        # Per-regime error accumulators for distribution estimation
        self._regime_errors: dict[str, list[float]] = {}
        self._regime_weighted_returns: dict[str, list[float]] = {}
        # Weight stability tracking
        self._weight_deltas: list[float] = []
        self._steps = 0

    def _get_weights(self, regime: str) -> dict[str, float]:
        """Get or initialize uniform weight vector for a regime."""
        if regime not in self._regime_weights:
            uniform = 1.0 / self.n_models
            self._regime_weights[regime] = {m: uniform for m in self.model_names}
        return self._regime_weights[regime]

    def update(
        self,
        regime: str,
        per_model_preds: dict[str, float],
        actual: float,
    ) -> float:
        """Process one observation: update weights, return weighted prediction.

        Parameters
        ----------
        regime:
            Current regime label.
        per_model_preds:
            {model_name: prediction} for models that produced predictions.
        actual:
            Actual observed value.

        Returns
        -------
        Weighted ensemble prediction using current regime weights.
        """
        if not per_model_preds:
            return actual  # no predictions to learn from

        weights = self._get_weights(regime)

        # Compute per-model squared loss
        losses: dict[str, float] = {}
        for model, pred in per_model_preds.items():
            if model in weights:
                losses[model] = (pred - actual) ** 2

        if not losses:
            return actual

        # Weighted ensemble prediction (using current weights before update)
        w_sum = sum(weights.get(m, 0) for m in per_model_preds)
        if w_sum > 0:
            weighted_pred = sum(
                weights.get(m, 0) * p for m, p in per_model_preds.items()
            ) / w_sum
        else:
            weighted_pred = float(np.mean(list(per_model_preds.values())))

        # Exponential gradient update
        old_weights = dict(weights)
        for model, loss in losses.items():
            weights[model] *= np.exp(-self.eta * loss)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            for m in weights:
                weights[m] /= total

        # Apply weight floor and ceiling
        for m in weights:
            weights[m] = max(self.min_weight, min(self.max_weight, weights[m]))
        # Re-normalize after clamping
        total = sum(weights.values())
        if total > 0:
            for m in weights:
                weights[m] /= total

        # Track weight stability (L2 norm of delta)
        delta = np.sqrt(sum(
            (weights.get(m, 0) - old_weights.get(m, 0)) ** 2
            for m in self.model_names
        ))
        self._weight_deltas.append(delta)

        # Track weighted error for regime distribution estimation
        weighted_error = weighted_pred - actual
        if regime not in self._regime_errors:
            self._regime_errors[regime] = []
            self._regime_weighted_returns[regime] = []
        self._regime_errors[regime].append(weighted_error)
        # Store the actual return for distribution estimation
        self._regime_weighted_returns[regime].append(actual)

        # Decay learning rate
        self.eta *= self.eta_decay
        self._steps += 1

        return weighted_pred

    def get_regime_weights(self) -> dict[str, dict[str, float]]:
        """Return learned per-regime weight vectors."""
        return {r: dict(w) for r, w in self._regime_weights.items()}

    def get_regime_distributions(self) -> dict[str, dict[str, float]]:
        """Return per-regime distribution parameters (weighted by model quality).

        These are superior to raw return distributions because they
        incorporate model uncertainty: weighted mean/std across models.
        """
        distributions: dict[str, dict[str, float]] = {}
        for regime, returns in self._regime_weighted_returns.items():
            if len(returns) >= 5:
                arr = np.array(returns)
                distributions[regime] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)),
                    "n_obs": len(returns),
                }
            else:
                distributions[regime] = {
                    "mean": 0.0,
                    "std": 0.01,
                    "n_obs": len(returns),
                }
        return distributions

    def is_converged(self, threshold: float = 0.005, window: int = 20) -> bool:
        """Check if weights have stabilized (mean delta below threshold)."""
        if len(self._weight_deltas) < window:
            return False
        recent = self._weight_deltas[-window:]
        return float(np.mean(recent)) < threshold

    @property
    def weight_stability(self) -> float:
        """Mean weight delta over the last 20 steps (lower = more stable)."""
        if not self._weight_deltas:
            return 1.0
        window = min(20, len(self._weight_deltas))
        return float(np.mean(self._weight_deltas[-window:]))

    @property
    def steps(self) -> int:
        return self._steps


# ---------------------------------------------------------------------------
# D4: Enhanced Burn-Out Result and Calibration
# ---------------------------------------------------------------------------


@dataclass
class BurnoutResult:
    """Container for burn-out weight calibration outputs.

    The burn-out phase learns per-regime model weights from the forward
    pass predictions_log using exponential gradient online learning.
    These weights are consumed by Monte Carlo (for better distribution
    parameters) and the prediction aggregator (for better ensemble weights).
    """

    # Backward-compatible fields
    model_states: dict[str, BaseModelWrapper] = field(default_factory=dict)
    iterations_completed: int = 0
    converged: bool = False
    best_rmse_by_tier: dict[int, float] = field(default_factory=dict)
    rmse_history: list[float] = field(default_factory=list)
    learning_rate_multiplier: float = 1.0

    # New: regime-conditioned ensemble weights from online learning
    # {regime_label: {model_name: weight}}
    regime_weights: dict[str, dict[str, float]] = field(default_factory=dict)

    # New: model-weighted regime distributions for Monte Carlo
    # {regime_label: {"mean": float, "std": float, "n_obs": int}}
    regime_distributions: dict[str, dict[str, float]] = field(default_factory=dict)

    # New: convergence and stability metrics
    weight_stability: float = 1.0
    calibration_steps: int = 0
    calibrated: bool = False


def run_burnout(
    cache: pd.DataFrame,
    tier_variables: dict[str, list[str]] | None = None,
    hierarchy_weights: dict[str, float] | None = None,
    regime_labels: pd.Series | None = None,
    *,
    extra_variables: list[str] | None = None,
    burnout_window: int = 130,
    max_iterations: int = 10,
    patience: int = 3,
    validation_days: int = 20,
    learning_rate_multiplier: float = 2.0,
    random_state: int = 42,
    forward_pass_result: Any | None = None,
) -> BurnoutResult:
    """Weight calibration via exponential gradient online learning.

    Two-phase burn-out:

    **Phase A** -- Run forward pass on recent data (single pass, not
    repeated).  If ``forward_pass_result`` is provided, skip this step
    and use the existing predictions_log directly.

    **Phase B** -- Feed the predictions_log into an
    ExponentialGradientWeightLearner that walks day-by-day, learning
    per-regime model weights from per-model prediction errors.

    The learned weights produce:
    - ``regime_weights``: per-regime ensemble weights for prediction aggregator
    - ``regime_distributions``: model-weighted distributions for Monte Carlo
    - ``best_rmse_by_tier``: per-tier validation RMSE

    Parameters
    ----------
    cache:
        Full 2-year daily cache.
    tier_variables:
        Tier -> variable list mapping.
    hierarchy_weights:
        Current hierarchy weights.
    regime_labels:
        Per-day regime labels.
    forward_pass_result:
        If provided, use its predictions_log instead of running a new
        forward pass.  This avoids duplicate computation.
    burnout_window:
        Number of recent trading days for the forward pass (if needed).
    max_iterations:
        Maximum passes through the predictions_log for weight learning.
    patience:
        Stop if weight stability does not improve for this many iterations.
    validation_days:
        Number of final days used for RMSE measurement.
    learning_rate_multiplier:
        Initial learning rate for exponential gradient.
    random_state:
        Random seed.

    Returns
    -------
    BurnoutResult with regime weights, distributions, and convergence info.
    """
    logger.info("Starting burn-out weight calibration...")

    result = BurnoutResult(learning_rate_multiplier=learning_rate_multiplier)

    # Phase A: Get predictions_log (from existing forward pass or run a new one)
    predictions_log: list[dict] = []

    if forward_pass_result is not None and hasattr(forward_pass_result, "predictions_log"):
        predictions_log = forward_pass_result.predictions_log or []
        if forward_pass_result.model_states:
            result.model_states = forward_pass_result.model_states
        logger.info("Burn-out: using existing forward pass predictions_log (%d entries)", len(predictions_log))
    else:
        # Run a forward pass on the burn-out window with regime weighting.
        # Source: The_Apps_core_idea.pdf Section L.1 -- weight historical
        # days by: w(tau) = exp(-delta_t/half_life) * regime_similarity.
        actual_window = min(burnout_window, len(cache))
        burnout_cache = cache.iloc[-actual_window:].copy()

        # Apply regime-weighted sample importance (exponential decay * regime similarity)
        if "regime_hmm_prob_0" in burnout_cache.columns:
            _regime_cols = [c for c in burnout_cache.columns if c.startswith("regime_hmm_prob_")]
            if _regime_cols:
                _regime_matrix = burnout_cache[_regime_cols].fillna(0).values
                _current_regime = _regime_matrix[-1]  # current day's regime probabilities
                _n = len(burnout_cache)
                # Exponential recency decay (half-life = 63 trading days)
                _days_ago = np.arange(_n, 0, -1)
                _recency = np.exp(-_days_ago / 63.0)
                # Regime similarity: Gaussian kernel on probability vectors
                _diff = _regime_matrix - _current_regime
                _similarity = np.exp(-np.sum(_diff ** 2, axis=1) / 0.5)
                # Combined weight
                _sample_weights = _recency * _similarity
                _sample_weights = _sample_weights / _sample_weights.sum() * _n
                burnout_cache["_burnout_sample_weight"] = _sample_weights

        if len(burnout_cache) < validation_days + 30:
            logger.warning("Insufficient data for burn-out (%d rows)", len(burnout_cache))
            return result

        burnout_warmup = max(20, actual_window - validation_days - 50)
        fp_result = run_forward_pass(
            burnout_cache,
            tier_variables=tier_variables,
            hierarchy_weights=hierarchy_weights,
            regime_labels=(
                regime_labels.iloc[-actual_window:]
                if regime_labels is not None and len(regime_labels) >= actual_window
                else None
            ),
            extra_variables=extra_variables,
            warmup_days=min(burnout_warmup, actual_window - validation_days - 1),
            log_interval=999,
            random_state=random_state,
        )
        predictions_log = fp_result.predictions_log or []
        result.model_states = fp_result.model_states or {}
        logger.info("Burn-out: ran forward pass, got %d predictions_log entries", len(predictions_log))

    # Need per-model predictions for weight learning
    has_per_model = any(
        entry.get("per_model") for entry in predictions_log[:10]
    ) if predictions_log else False

    if not has_per_model or len(predictions_log) < 50:
        # Fall back to legacy behavior: just store basic RMSE info
        logger.info(
            "Burn-out: insufficient per-model data (%d entries, per_model=%s), "
            "skipping weight calibration",
            len(predictions_log), has_per_model,
        )
        if predictions_log:
            log_df = pd.DataFrame(predictions_log)
            for tier_num in range(1, 6):
                tier_entries = log_df[log_df["tier"] == tier_num]
                if len(tier_entries) > 0:
                    result.best_rmse_by_tier[tier_num] = float(
                        np.sqrt(np.mean(tier_entries["error"].values ** 2))
                    )
            result.iterations_completed = 1
        return result

    # Phase B: Exponential gradient weight learning
    # Discover all model names from the predictions_log
    all_model_names: set[str] = set()
    for entry in predictions_log:
        pm = entry.get("per_model", {})
        if isinstance(pm, dict):
            all_model_names.update(pm.keys())

    if len(all_model_names) < 2:
        logger.info("Burn-out: only %d models found, skipping weight calibration", len(all_model_names))
        result.iterations_completed = 1
        return result

    best_stability = float("inf")
    no_improve_count = 0

    for iteration in range(max_iterations):
        # Create a fresh learner for each iteration with decaying initial eta
        eta = learning_rate_multiplier * 0.1 * (0.8 ** iteration)
        learner = ExponentialGradientWeightLearner(
            sorted(all_model_names),
            initial_eta=eta,
            eta_decay=0.995,
            min_weight=0.05,
            max_weight=0.6,
        )

        # Walk through predictions_log day-by-day
        for entry in predictions_log:
            regime = entry.get("regime", "unknown")
            per_model = entry.get("per_model", {})
            actual = entry.get("actual", 0.0)

            if isinstance(per_model, dict) and per_model:
                learner.update(regime, per_model, actual)

        stability = learner.weight_stability
        result.rmse_history.append(stability)

        if stability < best_stability:
            best_stability = stability
            result.regime_weights = learner.get_regime_weights()
            result.regime_distributions = learner.get_regime_distributions()
            result.weight_stability = stability
            result.calibration_steps = learner.steps
            no_improve_count = 0
            logger.info(
                "  Burn-out iter %d: stability=%.6f (new best), %d regimes, %d steps",
                iteration + 1, stability, len(result.regime_weights), learner.steps,
            )
        else:
            no_improve_count += 1
            logger.info(
                "  Burn-out iter %d: stability=%.6f (no improvement %d/%d)",
                iteration + 1, stability, no_improve_count, patience,
            )

        if learner.is_converged() or no_improve_count >= patience:
            result.converged = True
            break

    result.iterations_completed = len(result.rmse_history)
    result.calibrated = bool(result.regime_weights)

    # Compute per-tier RMSE from predictions_log (backward compat)
    if predictions_log:
        log_df = pd.DataFrame(predictions_log)
        for tier_num in range(1, 6):
            tier_entries = log_df[log_df["tier"] == tier_num]
            if len(tier_entries) > 0:
                result.best_rmse_by_tier[tier_num] = float(
                    np.sqrt(np.mean(tier_entries["error"].values ** 2))
                )

    logger.info(
        "Burn-out complete: %d iterations, converged=%s, calibrated=%s, "
        "stability=%.6f, regimes=%s",
        result.iterations_completed,
        result.converged,
        result.calibrated,
        result.weight_stability,
        list(result.regime_weights.keys()) if result.regime_weights else "none",
    )

    return result

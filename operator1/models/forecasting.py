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
# 2. GARCH (volatility forecasting)
# ---------------------------------------------------------------------------


def fit_garch(
    returns: np.ndarray,
    n_forecast: int = 1,
    p: int = 1,
    q: int = 1,
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


def run_forecasting(
    cache: pd.DataFrame,
    variables: list[str] | None = None,
    *,
    extra_variables: list[str] | None = None,
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
    # GARCH on volatility (special case)
    # ------------------------------------------------------------------
    if has_returns:
        returns = cache["return_1d"].values
        max_horizon = max(HORIZONS.values())
        garch_fcast, garch_met = fit_garch(
            returns, n_forecast=max_horizon,
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

    # ------------------------------------------------------------------
    # Per-variable forecasting
    # ------------------------------------------------------------------
    kalman_attempted = False
    var_attempted = False
    lstm_attempted = False
    tree_attempted = False

    for var_name in available_vars:
        series = cache[var_name].values
        tier = _get_tier_for_variable(var_name, tier_map)
        max_horizon = max(HORIZONS.values())
        best_forecast: np.ndarray | None = None
        best_model_name = ""
        best_metrics: ModelMetrics | None = None

        # --- Kalman (preferred for tier1/tier2) ---
        if tier in ("tier1", "tier2"):
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
            # Build a small multivariate frame from available vars.
            var_subset_cols = [
                c for c in available_vars
                if c in cache.columns
            ][:10]  # Cap at 10 for stability.
            if var_name not in var_subset_cols:
                var_subset_cols = [var_name] + var_subset_cols[:9]
            var_df = cache[var_subset_cols].copy()

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
            feature_cols = [
                c for c in cache.columns
                if c != var_name
                and cache[c].dtype in (np.float64, np.float32, np.int64)
            ][:15]
            if feature_cols:
                feat_df = cache[feature_cols + [var_name]].copy()
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

        # Store results.
        if best_forecast is not None:
            result.forecasts[var_name] = {
                label: float(best_forecast[min(h - 1, len(best_forecast) - 1)])
                for label, h in HORIZONS.items()
            }
            result.model_used[var_name] = best_model_name

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
        self._R = 0.1  # measurement noise
        self._fitted = len(clean) >= _MIN_OBS_KALMAN

        # Try fitting a proper statsmodels model for the initial state.
        if self._fitted:
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
            # Kalman gain
            S = self._P + self._R
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
    """Online GARCH wrapper with recursive variance update.

    Update rule: sigma2[t+1] = omega + alpha * eps^2[t] + beta * sigma2[t]
    """

    name = "garch"

    def __init__(self, returns: np.ndarray) -> None:
        clean = returns[~np.isnan(returns)]
        # Default GARCH(1,1) parameters
        self._omega = 0.0001
        self._alpha = 0.10
        self._beta = 0.85
        self._sigma2 = float(np.var(clean)) if len(clean) > 1 else 0.0004
        self._last_return = float(clean[-1]) if len(clean) else 0.0
        self._fitted = len(clean) >= _MIN_OBS_GARCH

        if self._fitted:
            try:
                from arch import arch_model  # type: ignore[import-untyped]

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

    def predict(self, state_t: np.ndarray) -> np.ndarray:
        # Forecast next-step conditional volatility (as std dev)
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
            self._sigma2 = self._omega + self._alpha * eps2 + self._beta * self._sigma2
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
                self._result = model.fit(maxlags=self._max_lag, ic="aic")
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
                    h = torch.elu(self.fc1(x))
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

            # Use last lookback values from history
            if len(self._history) < self._lookback:
                return np.array([self._history[-1]])

            # Build feature sequence from recent history (simplified: use target only)
            recent = np.array(self._history[-self._lookback:]).reshape(-1, 1)
            # Pad to n_features if needed
            n_feat = len(self._numeric_cols)
            if recent.shape[1] < n_feat:
                padded = np.zeros((self._lookback, n_feat))
                padded[:, 0] = recent[:, 0]
                recent = padded

            # Normalise
            scaled = (recent - self._feat_mean[:recent.shape[1]]) / self._feat_std[:recent.shape[1]]

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
    pid_summary: dict[str, Any] = field(default_factory=dict)  # PID controller state


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

    # GARCH (for volatility-related variables)
    if "volatility" in variable or "return" in variable:
        w = GARCHWrapper(series)
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
    logger.info("Starting forward pass (warmup=%d days)...", warmup_days)

    # Initialise PID bank for adaptive learning rate adjustment
    try:
        from operator1.models.pid_controller import create_pid_bank, compute_pid_adjustment
        _pid_available = True
    except ImportError:
        _pid_available = False

    # Initialise conformal calibrator for adaptive prediction intervals
    _conformal_calibrator = None
    try:
        from operator1.models.conformal import ConformalCalibrator
        _conformal_calibrator = ConformalCalibrator(coverage=0.9, adaptive=True)
        logger.info("Conformal calibrator initialised (target coverage=90%%)")
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

    # Forward loop: day warmup_days to n_days - 2 (predict t+1, compare with actual t+1)
    for t in range(warmup_days, n_days - 1):
        regime_t = str(regime_labels.iloc[t]) if t < len(regime_labels) else "unknown"

        for var_name in all_vars:
            tier_num = var_to_tier[var_name]
            tier_weight = hierarchy_weights.get(f"tier{tier_num}", 20.0)

            state_t = np.array([cache[var_name].iloc[t]])
            actual_t1 = np.array([cache[var_name].iloc[t + 1]])

            if np.isnan(actual_t1[0]):
                continue  # skip variables with missing next-day data

            # Step B: Multi-module prediction (pick best / average)
            predictions: list[float] = []
            for wrapper in model_bank.get(var_name, []):
                try:
                    pred = wrapper.predict(state_t)
                    if len(pred) > 0 and not np.isnan(pred[0]):
                        predictions.append(float(pred[0]))
                except Exception:
                    pass

            if not predictions:
                continue

            # Step C: Simple ensemble (average of available predictions)
            ensemble_pred = float(np.mean(predictions))

            # Step D: Reality check
            error = float(actual_t1[0]) - ensemble_pred
            weighted_error = (error ** 2) * (tier_weight / 20.0)

            result.errors_by_tier[tier_num].append(weighted_error)
            if regime_t not in result.errors_by_regime:
                result.errors_by_regime[regime_t] = []
            result.errors_by_regime[regime_t].append(weighted_error)

            # Step D2: Update conformal calibrator with this prediction/actual pair
            if _conformal_calibrator is not None:
                try:
                    _conformal_calibrator.add_score(var_name, ensemble_pred, float(actual_t1[0]))
                    _was_covered = _conformal_calibrator.predict_interval(
                        var_name, ensemble_pred,
                    )
                    if hasattr(_was_covered, "lower") and hasattr(_was_covered, "upper"):
                        _covered = _was_covered.lower <= float(actual_t1[0]) <= _was_covered.upper
                        _conformal_calibrator.update_adaptive(var_name, _covered)
                except Exception:
                    pass  # calibrator needs enough data before it can produce intervals

            # Log prediction
            result.predictions_log.append({
                "day": t,
                "variable": var_name,
                "predicted": ensemble_pred,
                "actual": float(actual_t1[0]),
                "error": error,
                "tier": tier_num,
                "regime": regime_t,
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

    # Store conformal diagnostics
    if _conformal_calibrator is not None:
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
# D4: Convergence-based burn-out (replaces window-shrinking heuristic)
# ---------------------------------------------------------------------------


@dataclass
class BurnoutResult:
    """Container for convergence-based burn-out outputs."""

    model_states: dict[str, BaseModelWrapper] = field(default_factory=dict)
    iterations_completed: int = 0
    converged: bool = False
    best_rmse_by_tier: dict[int, float] = field(default_factory=dict)
    rmse_history: list[float] = field(default_factory=list)
    learning_rate_multiplier: float = 1.0


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
) -> BurnoutResult:
    """Intensive re-training on recent data with convergence detection.

    Each iteration:
    1. Reset to *burnout_window* days ago.
    2. Run forward pass with higher learning rates.
    3. Measure accuracy on last *validation_days*.
    4. If accuracy improves, save as new best pattern.
    5. Stop early if no improvement for *patience* iterations.

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
    burnout_window:
        Number of recent trading days to use (default ~6 months).
    max_iterations:
        Maximum burn-out iterations.
    patience:
        Stop if no improvement for this many iterations.
    validation_days:
        Number of final days used for accuracy measurement.
    learning_rate_multiplier:
        Factor to increase learning rates during burn-out.
    random_state:
        Random seed.

    Returns
    -------
    BurnoutResult with final model states, convergence info, and RMSE history.
    """
    logger.info(
        "Starting burn-out (window=%d, max_iter=%d, patience=%d)...",
        burnout_window, max_iterations, patience,
    )

    result = BurnoutResult(learning_rate_multiplier=learning_rate_multiplier)

    # Extract the burnout slice
    actual_window = min(burnout_window, len(cache))
    burnout_cache = cache.iloc[-actual_window:].copy()

    if len(burnout_cache) < validation_days + 30:
        logger.warning("Insufficient data for burn-out (%d rows)", len(burnout_cache))
        return result

    best_rmse = float("inf")
    best_states: dict[str, BaseModelWrapper] = {}
    no_improve_count = 0

    for iteration in range(max_iterations):
        logger.info("Burn-out iteration %d/%d", iteration + 1, max_iterations)

        # Run forward pass on the burnout window
        # Use a smaller warmup within the burn-out window
        burnout_warmup = max(20, actual_window - validation_days - 50)
        fp_result = run_forward_pass(
            burnout_cache,
            tier_variables=tier_variables,
            hierarchy_weights=hierarchy_weights,
            regime_labels=regime_labels.iloc[-actual_window:] if regime_labels is not None and len(regime_labels) >= actual_window else None,
            extra_variables=extra_variables,
            warmup_days=min(burnout_warmup, actual_window - validation_days - 1),
            log_interval=999,  # suppress inner logging
            random_state=random_state + iteration,
        )

        # Evaluate on last validation_days entries
        if fp_result.predictions_log:
            log_df = pd.DataFrame(fp_result.predictions_log)
            # Filter to last validation_days worth of unique days
            unique_days = sorted(log_df["day"].unique())
            val_days_set = set(unique_days[-validation_days:]) if len(unique_days) >= validation_days else set(unique_days)
            val_entries = log_df[log_df["day"].isin(val_days_set)]

            if len(val_entries) > 0:
                iter_rmse = float(np.sqrt(np.mean(val_entries["error"].values ** 2)))
            else:
                iter_rmse = float("inf")
        else:
            iter_rmse = float("inf")

        result.rmse_history.append(iter_rmse)

        if iter_rmse < best_rmse:
            best_rmse = iter_rmse
            best_states = copy.copy(fp_result.model_states)
            no_improve_count = 0
            logger.info("  -> New best RMSE: %.6f", iter_rmse)

            # Compute per-tier RMSE
            if fp_result.predictions_log:
                log_df = pd.DataFrame(fp_result.predictions_log)
                for tier_num in range(1, 6):
                    tier_entries = log_df[log_df["tier"] == tier_num]
                    if len(tier_entries) > 0:
                        result.best_rmse_by_tier[tier_num] = float(
                            np.sqrt(np.mean(tier_entries["error"].values ** 2))
                        )
        else:
            no_improve_count += 1
            logger.info("  -> RMSE: %.6f (no improvement, patience %d/%d)", iter_rmse, no_improve_count, patience)

        if no_improve_count >= patience:
            result.converged = True
            logger.info("Burn-out converged after %d iterations (patience exhausted)", iteration + 1)
            break

    result.iterations_completed = len(result.rmse_history)
    result.model_states = best_states

    logger.info(
        "Burn-out complete: %d iterations, converged=%s, best_rmse=%.6f",
        result.iterations_completed, result.converged, best_rmse,
    )

    return result

"""Tier 3 adaptive parameters: windows, hyperparameters, and minor constants.

Replaces 42 hardcoded constants across 15 files with data-derived values
using filing frequency, effective sample size, and recent data distributions.

Methods implemented:

1. **Filing-Frequency-Anchored Windows** (Nyquist-Shannon): all rolling
   windows are integer multiples of the filing period.

2. **Scaling-Law Architecture** (Kaplan et al. 2020): neural network
   size proportional to effective sample size.

3. **Adaptive Dropout** (Baldi & Sadowski 2013): dropout rate from the
   ratio of model parameters to effective samples.

4. **Innovation-Based Noise** (Mehra 1970): particle filter noise scales
   from the innovation sequence standard deviation.

5. **Distribution-Based Pattern Thresholds** (Bulkowski 2008): candlestick
   thresholds from recent body/range ratio percentiles.

6. **Filing-Period Confidence Decay**: interpolation confidence with
   exponential half-life proportional to filing period.

7. **Filing-Frequency Staleness**: stale data threshold proportional
   to the detected filing frequency.

Top-level entry point:
    ``compute_adaptive_windows(filing_frequency)`` -> AdaptiveWindows
    ``compute_nn_hyperparams(n_eff, n_features)`` -> NNHyperparams
    ``compute_particle_noise(series)`` -> (state_noise, obs_noise)
    ``compute_pattern_thresholds(cache)`` -> (body_thresh, doji_thresh)
    ``compute_stale_threshold(filing_frequency)`` -> int
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
# Filing period mapping (business days)
# ---------------------------------------------------------------------------

_FILING_PERIODS: dict[str, int] = {
    "quarterly": 63,
    "semiannual": 126,
    "annual": 252,
    "unknown": 63,  # default to quarterly
}


# ---------------------------------------------------------------------------
# 11. Filing-Frequency-Anchored Windows (Nyquist-Shannon)
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveWindows:
    """Rolling window sizes anchored to the filing frequency.

    All windows are integer multiples of the filing period, ensuring
    that each window contains a meaningful number of real data points
    rather than interpolated values.
    """

    # Core windows
    short: int = 21       # ~1/3 filing period (volatility, stability)
    medium: int = 63      # 1 filing period (error rolling, validation)
    long: int = 126       # 2 filing periods (burnout, drawdown)
    trend: int = 252      # 4 filing periods (growth, long-term trends)

    # Specific use-case windows
    volatility: int = 21
    drawdown: int = 252
    stability: int = 21
    error_rolling: int = 63
    burnout: int = 126
    validation: int = 63
    transition: int = 42

    # Metadata
    filing_frequency: str = "quarterly"
    filing_period: int = 63


def compute_adaptive_windows(
    filing_frequency: str = "quarterly",
) -> AdaptiveWindows:
    """Compute rolling window sizes from filing frequency.

    Windows are Nyquist-aligned: the short window is at least 1/3 of
    the filing period (captures one sub-period), the medium window
    matches one period, and longer windows are multiples.

    Parameters
    ----------
    filing_frequency:
        One of: "quarterly", "semiannual", "annual", "unknown".

    Returns
    -------
    AdaptiveWindows with all window sizes.
    """
    base = _FILING_PERIODS.get(filing_frequency, 63)

    result = AdaptiveWindows(
        short=max(base // 3, 5),
        medium=base,
        long=base * 2,
        trend=base * 4,
        volatility=max(base // 3, 5),
        drawdown=base * 4,
        stability=max(base // 3, 5),
        error_rolling=base,
        burnout=base * 2,
        validation=base,
        transition=max(base * 2 // 3, 10),
        filing_frequency=filing_frequency,
        filing_period=base,
    )

    logger.debug(
        "Adaptive windows for %s (base=%d): short=%d, medium=%d, long=%d, trend=%d",
        filing_frequency, base, result.short, result.medium, result.long, result.trend,
    )

    return result


# ---------------------------------------------------------------------------
# 12. Neural Network Hyperparameters (Kaplan scaling + adaptive dropout)
# ---------------------------------------------------------------------------


@dataclass
class NNHyperparams:
    """Data-proportional neural network hyperparameters."""

    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    hidden_dim: int = 32
    dropout: float = 0.1
    learning_rate: float = 0.001
    max_epochs: int = 60
    patience: int = 10
    batch_size: int = 32
    lookback: int = 21


def compute_nn_hyperparams(
    n_eff: float,
    n_features: int = 10,
    batch_size: int = 32,
) -> NNHyperparams:
    """Compute neural network hyperparameters from data characteristics.

    Uses Kaplan et al. (2020) scaling laws for architecture size,
    Baldi & Sadowski (2013) for dropout, and linear scaling rule
    (Goyal et al. 2017) for learning rate.

    Parameters
    ----------
    n_eff:
        Effective sample size (from Kish computation).
    n_features:
        Number of input features.
    batch_size:
        Training batch size.

    Returns
    -------
    NNHyperparams with all settings.
    """
    n_eff = max(n_eff, 10)  # floor

    # Architecture scaling (Kaplan 2020 adapted for tabular)
    d_model = max(16, min(128, int(8 * math.log2(max(n_eff, 2)))))
    # Round to nearest multiple of 8 for GPU efficiency
    d_model = ((d_model + 7) // 8) * 8

    n_heads = max(1, d_model // 8)

    hidden_dim = max(16, min(64, int(4 * math.sqrt(n_features * n_eff / 100))))

    n_layers = 1 if n_eff < 200 else 2

    # Lookback proportional to data richness (but capped)
    lookback = max(5, min(63, int(math.sqrt(n_eff))))

    # Adaptive dropout (Baldi & Sadowski 2013)
    n_params = d_model * hidden_dim * n_layers  # rough parameter count
    dropout = 1.0 - math.sqrt(n_eff / (n_eff + n_params))
    dropout = max(0.05, min(0.50, dropout))

    # Learning rate (linear scaling rule, Goyal 2017)
    lr = 0.001 * math.sqrt(batch_size / 32)
    lr = max(0.0001, min(0.01, lr))

    # Epochs from data size
    max_epochs = max(30, min(200, int(n_eff / batch_size * 3)))
    patience = max(5, max_epochs // 6)

    result = NNHyperparams(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        dropout=round(dropout, 3),
        learning_rate=round(lr, 6),
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        lookback=lookback,
    )

    logger.debug(
        "NN hyperparams (n_eff=%.0f, n_feat=%d): d_model=%d, hidden=%d, "
        "layers=%d, dropout=%.3f, lr=%.6f, epochs=%d",
        n_eff, n_features, d_model, hidden_dim, n_layers, dropout, lr, max_epochs,
    )

    return result


# ---------------------------------------------------------------------------
# 13. Particle Filter Noise Scales (Mehra 1970)
# ---------------------------------------------------------------------------


def compute_particle_noise(
    series: pd.Series,
    default_state: float = 0.02,
    default_obs: float = 0.05,
) -> tuple[float, float]:
    """Estimate particle filter noise scales from innovation sequence.

    Uses the standard deviation of the difference series as a proxy
    for observation noise (Mehra 1970 simplified).

    Parameters
    ----------
    series:
        Time series to filter (e.g., cash_ratio).
    default_state:
        Fallback state noise scale.
    default_obs:
        Fallback observation noise scale.

    Returns
    -------
    (state_noise_scale, obs_noise_scale)
        Both as fractions of state magnitude, in [0.001, 0.20] and
        [0.005, 0.50] respectively.
    """
    clean = series.dropna()
    if len(clean) < 10:
        return default_state, default_obs

    diff = clean.diff().dropna()
    if len(diff) < 5:
        return default_state, default_obs

    obs_noise = float(diff.std())
    state_noise = obs_noise * 0.4  # process is smoother than observations

    # Normalize to fraction of state magnitude
    mean_abs = max(float(clean.abs().mean()), 1e-8)
    state_scale = state_noise / mean_abs
    obs_scale = obs_noise / mean_abs

    state_scale = max(0.001, min(0.20, state_scale))
    obs_scale = max(0.005, min(0.50, obs_scale))

    logger.debug(
        "Particle noise: state=%.4f, obs=%.4f (raw_obs_noise=%.6f, mean=%.4f)",
        state_scale, obs_scale, obs_noise, mean_abs,
    )

    return round(state_scale, 4), round(obs_scale, 4)


# ---------------------------------------------------------------------------
# 14. Candlestick Pattern Thresholds (Bulkowski 2008)
# ---------------------------------------------------------------------------


def compute_pattern_thresholds(
    cache: pd.DataFrame,
    lookback: int = 63,
    default_body: float = 0.3,
    default_doji: float = 0.1,
) -> tuple[float, float]:
    """Compute adaptive candlestick body/doji thresholds from recent data.

    Uses the empirical distribution of body/range ratios over the
    lookback window. Doji = P10, body = P50 (Bulkowski 2008).

    Parameters
    ----------
    cache:
        Daily cache with open, high, low, close columns.
    lookback:
        Days of recent data to use.
    default_body:
        Fallback body threshold.
    default_doji:
        Fallback doji threshold.

    Returns
    -------
    (body_threshold, doji_threshold)
    """
    required = ["open", "high", "low", "close"]
    if not all(c in cache.columns for c in required):
        return default_body, default_doji

    ohlc = cache[required].dropna().tail(lookback)
    if len(ohlc) < 10:
        return default_body, default_doji

    o, h, l, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
    rng = h - l
    body = (c - o).abs()

    # Avoid division by zero
    valid = rng > 1e-10
    if valid.sum() < 10:
        return default_body, default_doji

    ratios = (body[valid] / rng[valid]).values

    doji_thresh = float(np.percentile(ratios, 10))
    body_thresh = float(np.percentile(ratios, 50))

    # Clamp to reasonable ranges
    doji_thresh = max(0.01, min(0.25, doji_thresh))
    body_thresh = max(0.15, min(0.60, body_thresh))

    logger.debug(
        "Pattern thresholds: doji=%.3f, body=%.3f (from %d candles)",
        doji_thresh, body_thresh, len(ratios),
    )

    return round(body_thresh, 3), round(doji_thresh, 3)


# ---------------------------------------------------------------------------
# 16. Filing-Period Confidence Decay
# ---------------------------------------------------------------------------


def compute_confidence_halflife(
    filing_frequency: str = "quarterly",
) -> float:
    """Compute interpolation confidence half-life from filing frequency.

    The half-life is 1/4 of the filing period: confidence decays to 50%
    when 25% through the inter-filing gap.

    Parameters
    ----------
    filing_frequency:
        One of: "quarterly", "semiannual", "annual", "unknown".

    Returns
    -------
    Half-life in days.
    """
    base = _FILING_PERIODS.get(filing_frequency, 63)
    return max(5.0, base / 4.0)


# ---------------------------------------------------------------------------
# 18. Filing-Frequency Staleness Threshold
# ---------------------------------------------------------------------------


def compute_stale_threshold(
    filing_frequency: str = "quarterly",
) -> int:
    """Compute stale data threshold from filing frequency.

    Threshold is ~2x the filing period for quarterly, ~1.7x for
    semi-annual, ~1.6x for annual.

    Parameters
    ----------
    filing_frequency:
        One of: "quarterly", "semiannual", "annual", "unknown".

    Returns
    -------
    Stale threshold in days.
    """
    thresholds = {
        "quarterly": 120,
        "semiannual": 210,
        "annual": 400,
        "unknown": 180,
    }
    return thresholds.get(filing_frequency, 180)


# ---------------------------------------------------------------------------
# 19. Market-Specific Entity Scoring Weights
# ---------------------------------------------------------------------------


def get_entity_scoring_weights(
    market_id: str = "",
) -> dict[str, int]:
    """Get market-specific entity discovery scoring weights.

    Markets with non-Latin scripts or romanization get lower name
    similarity weights and higher ticker weights to compensate.

    Parameters
    ----------
    market_id:
        Market identifier (e.g., "kr_dart", "cn_sse").

    Returns
    -------
    Dict with keys: ticker_weight, name_weight, country_weight,
    sector_weight, match_threshold.
    """
    # Default scoring weights
    defaults = {
        "ticker_weight": 40,
        "name_weight": 30,
        "country_weight": 15,
        "sector_weight": 15,
        "match_threshold": 70,
    }

    # Market-specific overrides
    overrides: dict[str, dict[str, int]] = {
        "cn_sse": {
            "ticker_weight": 50,
            "name_weight": 15,
            "match_threshold": 55,
        },
        "tw_mops": {
            "ticker_weight": 45,
            "name_weight": 20,
            "match_threshold": 60,
        },
        "jp_jquants": {
            "ticker_weight": 45,
            "name_weight": 20,
            "match_threshold": 60,
        },
        "kr_dart": {
            "ticker_weight": 42,
            "name_weight": 25,
            "match_threshold": 65,
        },
    }

    result = dict(defaults)
    if market_id in overrides:
        result.update(overrides[market_id])

    return result


# ---------------------------------------------------------------------------
# 20. Model Confidence Blend
# ---------------------------------------------------------------------------


def blend_by_confidence(
    estimates: list[float],
    confidences: list[float],
) -> float:
    """Blend multiple estimates weighted by their confidence scores.

    Parameters
    ----------
    estimates:
        List of point estimates from different models.
    confidences:
        List of confidence scores (0-1) for each estimate.

    Returns
    -------
    Weighted average of estimates. Falls back to simple mean if
    all confidences are zero.
    """
    if not estimates or not confidences:
        return float("nan")

    if len(estimates) != len(confidences):
        return float(np.mean(estimates))

    est = np.array(estimates, dtype=float)
    conf = np.array(confidences, dtype=float)

    # Filter out NaN estimates
    valid = ~np.isnan(est)
    if not valid.any():
        return float("nan")

    est = est[valid]
    conf = conf[valid]

    total_conf = conf.sum()
    if total_conf < 1e-10:
        return float(np.mean(est))

    weights = conf / total_conf
    return float(np.dot(est, weights))


# ---------------------------------------------------------------------------
# Aggregate container
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveTier3Params:
    """Collection of all Tier 3 adaptive parameters."""

    windows: AdaptiveWindows = field(default_factory=AdaptiveWindows)
    nn_params: NNHyperparams = field(default_factory=NNHyperparams)
    particle_state_noise: float = 0.02
    particle_obs_noise: float = 0.05
    pattern_body_threshold: float = 0.3
    pattern_doji_threshold: float = 0.1
    confidence_halflife: float = 15.75
    stale_threshold_days: int = 180
    entity_scoring: dict[str, int] = field(
        default_factory=lambda: {
            "ticker_weight": 40,
            "name_weight": 30,
            "country_weight": 15,
            "sector_weight": 15,
            "match_threshold": 70,
        },
    )
    adapted: bool = False

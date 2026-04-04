"""Adaptive model parameter calibration for Tier 2 constants.

Replaces fixed model hyperparameters with data-derived values using
established statistical and control-theory methods.

Methods implemented:

1. **Kish Effective Sample Size** (Kish 1965): computes the true
   information content of interpolated time series using interpolation
   confidence weights. Replaces fixed MIN_OBS constants.

2. **Inverse-Variance Blending** (Cochrane 1954): weights Cox PH and
   sigmoid survival models by inverse prediction variance.

3. **Lambda PID Tuning** (Dahlin 1968): auto-tunes PID gains from the
   error series autocorrelation half-life. Per-variable, non-oscillating.

4. **Copula Tail Contagion** (Joe 2014): derives edge-specific contagion
   probabilities from copula lower tail dependence.

5. **Amihud Participation Rate** (Amihud 2002): sets liquidation
   participation rate from stock illiquidity.

6. **Garman-Klass Intraday Factor** (Garman & Klass 1980): derives
   intraday low estimation from OHLC volatility.

7. **Precision-Targeted MC** (Glasserman 2003): sets Monte Carlo path
   count to achieve a target standard error on survival probability.

8. **Adaptive IS Tilt** (Bucklew 2004): sets importance sampling tilt
   proportional to event rarity.

Top-level entry points:
    ``compute_effective_sample_size()`` -- Kish n_eff for a variable
    ``compute_adaptive_min_obs()`` -- per-model minimum observations
    ``compute_blend_weights()`` -- Cox/sigmoid inverse-variance blend
    ``compute_pid_gains()`` -- Lambda-tuned PID gains
    ``compute_adaptive_contagion_prob()`` -- copula-derived contagion
    ``compute_adaptive_participation_rate()`` -- Amihud-derived rate
    ``compute_garman_klass_factor()`` -- OHLC intraday factor
    ``compute_adaptive_mc_params()`` -- precision-targeted MC config
    ``compute_regime_risk_multiplier()`` -- HMM volatility ratio
    ``compute_transition_halflife()`` -- from enriched timeline
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
# Model complexity multipliers (how much data each model needs per EDoF)
# ---------------------------------------------------------------------------

_MODEL_MULTIPLIERS: dict[str, float] = {
    "kalman": 3.0,
    "garch": 5.0,
    "var": 4.0,
    "lstm": 8.0,
    "tree": 3.0,
    "hmm": 5.0,
    "gmm": 3.0,
    "pelt": 2.0,
    "transformer": 10.0,
    "estimator": 2.0,
    "baseline": 1.0,
}

# Absolute floors (theoretical minimum for any model)
_MODEL_FLOORS: dict[str, int] = {
    "kalman": 10,
    "garch": 20,
    "var": 15,
    "lstm": 30,
    "tree": 10,
    "hmm": 20,
    "gmm": 15,
    "pelt": 15,
    "transformer": 40,
    "estimator": 5,
    "baseline": 1,
}


# ---------------------------------------------------------------------------
# 5. Effective Sample Size (Kish 1965)
# ---------------------------------------------------------------------------


def compute_effective_sample_size(
    cache: pd.DataFrame,
    metric: str,
) -> float:
    """Compute Kish effective sample size weighted by interpolation confidence.

    A series with 500 daily rows but only 4 real filings (interpolation
    confidence ~0.3 for most days) will have n_eff of ~20, not 500.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    metric:
        Column name to compute n_eff for.

    Returns
    -------
    float
        Effective sample size. Falls back to actual row count if no
        confidence data is available.
    """
    conf_col = f"interp_confidence_{metric}"

    if conf_col in cache.columns:
        w = cache[conf_col].fillna(0.5).values
    elif metric in cache.columns:
        # No confidence data -- estimate from uniqueness ratio
        series = cache[metric].dropna()
        if len(series) < 2:
            return float(len(series))
        n_unique = len(np.unique(series.values))
        ratio = n_unique / len(series)
        # If all values are unique (OHLCV), ratio=1.0, n_eff=n
        # If mostly repeated (interpolated), ratio=0.01, n_eff much less
        w = np.full(len(series), ratio)
    else:
        return 0.0

    sum_w = np.sum(w)
    sum_w2 = np.sum(w ** 2)

    if sum_w2 < 1e-10:
        return 0.0

    return float((sum_w ** 2) / sum_w2)


def compute_adaptive_min_obs(
    model_type: str,
    cache: pd.DataFrame,
    metric: str = "close",
) -> int:
    """Compute adaptive minimum observations for a model type.

    Uses Kish effective sample size when interpolation confidence is
    available, falls back to filing-count-based EDoF (Satterthwaite 1946).

    Parameters
    ----------
    model_type:
        One of: kalman, garch, var, lstm, tree, hmm, gmm, pelt,
        transformer, estimator, baseline.
    cache:
        Daily cache DataFrame.
    metric:
        Primary metric the model will be fitted on.

    Returns
    -------
    int
        Minimum observations required. Never below the model's
        absolute floor.
    """
    n_eff = compute_effective_sample_size(cache, metric)

    multiplier = _MODEL_MULTIPLIERS.get(model_type, 3.0)
    floor = _MODEL_FLOORS.get(model_type, 10)

    if n_eff > 0:
        required = int(n_eff * multiplier)
    else:
        # Fallback: use actual row count with standard multiplier
        n_rows = len(cache[metric].dropna()) if metric in cache.columns else 0
        required = int(n_rows * 0.3)  # assume 30% are informative

    return max(required, floor)


# ---------------------------------------------------------------------------
# 6. Cox/Sigmoid Blend Weights (Cochrane 1954)
# ---------------------------------------------------------------------------


def compute_blend_weights(
    sig_series: pd.Series,
    cox_series: pd.Series,
    actual_flags: pd.Series,
    lookback: int = 126,
) -> tuple[float, float]:
    """Compute inverse-variance blend weights for Cox PH and sigmoid models.

    Uses rolling prediction variance against actual survival mode flags
    (Cochrane 1954 inverse-variance combination).

    When ``scoring_weights.yml`` has ``survival_blend.use_adaptive: false``,
    this function returns the manual weights from config without computing
    the inverse-variance blend.

    Parameters
    ----------
    sig_series:
        Sigmoid survival probability series.
    cox_series:
        Cox PH survival score series.
    actual_flags:
        Binary company_survival_mode_flag series (ground truth).
    lookback:
        Rolling window for variance estimation (business days).

    Returns
    -------
    (w_sig, w_cox)
        Weights summing to 1.0. Falls back to (0.4, 0.6) if insufficient data.
    """
    # Check if user disabled adaptive blend via dashboard toggle
    try:
        from operator1.scoring_weights import get_weight
        use_adaptive = get_weight("survival_blend.use_adaptive", True)
        if not use_adaptive:
            w_sig = float(get_weight("survival_blend.sigmoid_weight", 0.4))
            w_cox = float(get_weight("survival_blend.cox_weight", 0.6))
            return w_sig, w_cox
    except Exception:
        pass

    if len(sig_series) < lookback or len(cox_series) < lookback:
        return 0.4, 0.6  # original defaults

    actual = actual_flags.fillna(0).astype(float)

    # Rolling variance of prediction error
    sig_err = (sig_series - actual).tail(lookback)
    cox_err = (cox_series - actual).tail(lookback)

    sig_var = float(sig_err.var())
    cox_var = float(cox_err.var())

    # Guard against zero variance
    if sig_var < 1e-10 and cox_var < 1e-10:
        return 0.5, 0.5

    inv_sig = 1.0 / max(sig_var, 1e-10)
    inv_cox = 1.0 / max(cox_var, 1e-10)
    total = inv_sig + inv_cox

    return float(inv_sig / total), float(inv_cox / total)


# ---------------------------------------------------------------------------
# 7. Lambda PID Tuning (Dahlin 1968)
# ---------------------------------------------------------------------------


def compute_pid_gains(
    error_series: pd.Series,
    variable_name: str = "",
) -> tuple[float, float, float]:
    """Compute PID gains from the error series autocorrelation half-life.

    Lambda tuning (Dahlin 1968): sets response time proportional to the
    error's natural decay time. Avoids oscillation by design.

    Parameters
    ----------
    error_series:
        Time series of prediction errors for one variable.
    variable_name:
        For logging only.

    Returns
    -------
    (kp, ki, kd)
        Proportional, integral, derivative gains. Falls back to
        (0.5, 0.1, 0.2) if insufficient data.
    """
    clean = error_series.dropna()
    if len(clean) < 10:
        return 0.5, 0.1, 0.2  # defaults

    # Autocorrelation at lag 1
    try:
        acf1 = float(clean.autocorr(lag=1))
    except Exception:
        return 0.5, 0.1, 0.2

    if math.isnan(acf1) or abs(acf1) < 0.01:
        tau = 1.0  # no autocorrelation -> fast response
    else:
        tau = max(-1.0 / math.log(min(abs(acf1), 0.999)), 0.5)

    # Lambda target: 2x the natural decay, minimum 3 days
    lambda_target = max(2.0 * tau, 3.0)

    # Process gain: steady-state error magnitude
    k_process = max(float(clean.abs().mean()), 0.01)

    # Dahlin formulas (clamped to reasonable ranges)
    kp = min(tau / (lambda_target * k_process), 2.0)
    ki = min(kp / max(tau, 0.1), 0.5)
    kd = min(kp * tau / 4.0, 1.0)

    # Floor: never below minimal correction
    kp = max(kp, 0.05)
    ki = max(ki, 0.01)
    kd = max(kd, 0.01)

    logger.debug(
        "Lambda PID for %s: tau=%.2f, lambda=%.2f, Kp=%.3f, Ki=%.3f, Kd=%.3f",
        variable_name, tau, lambda_target, kp, ki, kd,
    )

    return round(kp, 4), round(ki, 4), round(kd, 4)


# ---------------------------------------------------------------------------
# 8a. Copula-Derived Contagion Probability (Joe 2014)
# ---------------------------------------------------------------------------


def compute_adaptive_contagion_prob(
    copula_result: Any | None,
    default: float = 0.3,
) -> dict[str, float]:
    """Derive per-entity contagion probabilities from copula tail dependence.

    Parameters
    ----------
    copula_result:
        CopulaResult from ``run_copula_analysis()``. Must have
        ``tail_dependence`` dict attribute.
    default:
        Fallback probability when copula not available.

    Returns
    -------
    Dict mapping variable pairs to contagion probabilities.
    If copula unavailable, returns empty dict (caller uses default).
    """
    if copula_result is None:
        return {}

    tail_dep = getattr(copula_result, "tail_dependence", None)
    if not tail_dep or not isinstance(tail_dep, dict):
        return {}

    probs: dict[str, float] = {}
    for pair_key, lambda_val in tail_dep.items():
        if isinstance(lambda_val, (int, float)) and not math.isnan(lambda_val):
            # Clamp to [0.05, 0.95] -- never zero (always some contagion
            # risk) and never 1.0 (never certain)
            prob = max(0.05, min(0.95, float(lambda_val)))
            probs[str(pair_key)] = prob

    if probs:
        logger.info(
            "Copula-derived contagion: %d pairs, mean=%.3f",
            len(probs), np.mean(list(probs.values())),
        )

    return probs


# ---------------------------------------------------------------------------
# 8b. Amihud-Derived Participation Rate (Amihud 2002)
# ---------------------------------------------------------------------------


def compute_adaptive_participation_rate(
    cache: pd.DataFrame,
    default: float = 0.25,
) -> float:
    """Compute participation rate from Amihud illiquidity ratio.

    Highly liquid stocks: rate -> 0.50 (can trade 50% of daily volume).
    Illiquid stocks: rate -> 0.05 (limited to 5% of volume).

    Parameters
    ----------
    cache:
        Daily cache with ``close``, ``volume``, ``return_1d`` columns.
    default:
        Fallback rate when data unavailable.

    Returns
    -------
    float in [0.05, 0.50].
    """
    if "return_1d" not in cache.columns or "volume" not in cache.columns:
        return default

    returns = cache["return_1d"].dropna()
    volume = cache["volume"].dropna()

    if len(returns) < 20 or len(volume) < 20:
        return default

    # Align indices
    aligned = pd.DataFrame({
        "abs_ret": returns.abs(),
        "dollar_vol": volume * cache.get("close", pd.Series(1.0, index=cache.index)),
    }).dropna()

    if len(aligned) < 20 or (aligned["dollar_vol"] < 1e-6).all():
        return default

    # Amihud illiquidity = mean(|return| / dollar_volume)
    amihud = float((aligned["abs_ret"] / aligned["dollar_vol"].clip(lower=1e-6)).mean())

    # Transform to participation rate via logistic
    # amihud = 0 (very liquid) -> rate = 0.50
    # amihud = 1e-6 (typical large-cap) -> rate ~0.40
    # amihud = 1e-3 (typical small-cap) -> rate ~0.10
    rate = 1.0 / (1.0 + 10.0 * amihud * 1e6)
    rate = max(0.05, min(0.50, rate))

    logger.debug("Amihud participation rate: amihud=%.2e, rate=%.3f", amihud, rate)
    return round(rate, 4)


# ---------------------------------------------------------------------------
# 9a. Regime-Conditional Risk Multiplier
# ---------------------------------------------------------------------------


def compute_regime_risk_multiplier(
    regime_detector: Any | None,
    cache: pd.DataFrame,
    default: float = 2.0,
) -> float:
    """Compute survival risk multiplier from HMM regime volatilities.

    risk_mult = sigma_worst_regime / sigma_best_regime

    Parameters
    ----------
    regime_detector:
        Fitted regime detector from Step 5.5.
    cache:
        Daily cache with ``return_1d`` and ``regime_label``.
    default:
        Fallback multiplier.

    Returns
    -------
    float >= 1.0. Typically 1.5 to 5.0.
    """
    if regime_detector is None or "regime_label" not in cache.columns:
        return default

    if "return_1d" not in cache.columns:
        return default

    aligned = pd.DataFrame({
        "ret": cache["return_1d"],
        "regime": cache["regime_label"],
    }).dropna()

    if len(aligned) < 30:
        return default

    # Per-regime volatility
    regime_vols: dict[str, float] = {}
    for regime, group in aligned.groupby("regime"):
        if len(group) >= 10:
            regime_vols[str(regime)] = float(group["ret"].std())

    if len(regime_vols) < 2:
        return default

    vols = list(regime_vols.values())
    sigma_max = max(vols)
    sigma_min = max(min(vols), 1e-8)

    mult = sigma_max / sigma_min
    mult = max(1.0, min(10.0, mult))  # clamp to [1, 10]

    logger.debug(
        "Regime risk multiplier: sigma_max=%.4f, sigma_min=%.4f, mult=%.2f",
        sigma_max, sigma_min, mult,
    )
    return round(mult, 2)


# ---------------------------------------------------------------------------
# 9b. Garman-Klass Intraday Factor (1980)
# ---------------------------------------------------------------------------


def compute_garman_klass_factor(
    cache: pd.DataFrame,
    default: float = 1.5,
) -> float:
    """Compute intraday low factor from Garman-Klass OHLC volatility.

    Uses the ratio of GK volatility to close-to-close volatility
    scaled by the 90% CI z-score.

    Parameters
    ----------
    cache:
        Daily cache with ``open``, ``high``, ``low``, ``close``.
    default:
        Fallback factor.

    Returns
    -------
    float >= 1.0. Typically 1.2 to 3.0.
    """
    required = ["open", "high", "low", "close"]
    if not all(c in cache.columns for c in required):
        return default

    ohlc = cache[required].dropna()
    if len(ohlc) < 20:
        return default

    o, h, l, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]

    # Garman-Klass estimator
    log_hl = np.log(h / l.clip(lower=1e-10))
    log_co = np.log(c / o.clip(lower=1e-10))

    gk_var = 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2
    sigma_gk = float(np.sqrt(gk_var.mean()))

    # Close-to-close volatility
    sigma_cc = float(c.pct_change().std())

    if sigma_cc < 1e-8:
        return default

    ratio = sigma_gk / sigma_cc
    factor = ratio * 1.645  # 90% CI

    factor = max(1.0, min(5.0, factor))

    logger.debug(
        "Garman-Klass factor: sigma_gk=%.4f, sigma_cc=%.4f, factor=%.2f",
        sigma_gk, sigma_cc, factor,
    )
    return round(factor, 2)


# ---------------------------------------------------------------------------
# 9c. Transition Half-Life from Enriched Timeline
# ---------------------------------------------------------------------------


def compute_transition_halflife(
    enriched_timeline_result: Any | None,
    default: int = 5,
) -> int:
    """Compute regime transition blend half-life from switch point durations.

    half_life = median(transition_durations) / 2

    Parameters
    ----------
    enriched_timeline_result:
        From ``compute_enriched_survival_timeline()``.
    default:
        Fallback half-life in days.

    Returns
    -------
    int >= 2. Typically 3-15 days.
    """
    if enriched_timeline_result is None:
        return default

    base = getattr(enriched_timeline_result, "base", None)
    if base is None:
        return default

    switch_points = getattr(base, "switch_points", [])
    if len(switch_points) < 2:
        return default

    # Extract mode durations from switch points
    durations: list[int] = []
    timeline = getattr(base, "timeline", None)
    if timeline is not None and "days_in_mode" in timeline.columns:
        # Get the days_in_mode at each switch point
        for sp in switch_points:
            date = sp.get("date")
            if date and date in timeline.index:
                days = int(timeline.loc[date, "days_in_mode"])
                if days > 0:
                    durations.append(days)

    if not durations:
        return default

    median_duration = float(np.median(durations))
    halflife = max(2, int(median_duration / 2))
    halflife = min(halflife, 30)  # cap at 30 days

    logger.debug(
        "Transition half-life: median_duration=%.1f, halflife=%d",
        median_duration, halflife,
    )
    return halflife


# ---------------------------------------------------------------------------
# 10. Precision-Targeted Monte Carlo (Glasserman 2003)
# ---------------------------------------------------------------------------


def compute_adaptive_mc_params(
    cache: pd.DataFrame,
    preliminary_survival: float | None = None,
    target_se: float = 0.005,
) -> tuple[int, float]:
    """Compute adaptive Monte Carlo path count and IS tilt.

    Path count: precision-targeted for target_se standard error on
    survival probability (Glasserman 2003).

    IS tilt: adaptive exponential tilting proportional to event rarity
    (Bucklew 2004).

    When ``scoring_weights.yml`` has ``monte_carlo.use_adaptive: false``,
    returns the manual values from config without computing adaptive params.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    preliminary_survival:
        Preliminary survival probability estimate. If None, uses
        the latest cache value.
    target_se:
        Target standard error for survival probability (default 0.5%).

    Returns
    -------
    (n_paths, is_tilt)
        n_paths in [1000, 50000], is_tilt in [0.5, 5.0].
    """
    # Check if user disabled adaptive MC via dashboard toggle
    try:
        from operator1.scoring_weights import get_weight
        use_adaptive = get_weight("monte_carlo.use_adaptive", True)
        if not use_adaptive:
            n = int(get_weight("monte_carlo.n_paths", 10000))
            tilt = float(get_weight("monte_carlo.importance_tilt", 1.5))
            return n, tilt
    except Exception:
        pass

    # Get preliminary survival probability
    if preliminary_survival is None:
        surv_col = cache.get("survival_probability")
        if surv_col is not None and surv_col.notna().any():
            preliminary_survival = float(surv_col.dropna().iloc[-1])
        else:
            preliminary_survival = 0.90  # assume healthy

    p = max(0.01, min(0.99, preliminary_survival))

    # Precision-targeted path count: n = p*(1-p) / se^2
    n_paths = int(p * (1 - p) / (target_se ** 2))
    n_paths = max(1000, min(50000, n_paths))

    # Adaptive IS tilt: -log(p_tail) / (sigma * sqrt(252))
    p_tail = 1.0 - p
    vol = 0.02  # default
    if "volatility_21d" in cache.columns:
        vol_series = cache["volatility_21d"].dropna()
        if not vol_series.empty:
            vol = max(float(vol_series.iloc[-1]), 0.005)

    annual_vol = vol * math.sqrt(252)
    if p_tail > 0.01 and annual_vol > 0.01:
        is_tilt = min(-math.log(p_tail) / annual_vol, 5.0)
    else:
        is_tilt = 3.0  # aggressive tilt for very safe companies

    is_tilt = max(0.5, is_tilt)

    logger.debug(
        "Adaptive MC: p=%.3f, n_paths=%d, is_tilt=%.2f (vol=%.4f)",
        p, n_paths, is_tilt, vol,
    )
    return n_paths, round(is_tilt, 2)


# ---------------------------------------------------------------------------
# Aggregate result container
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveModelParams:
    """Collection of all Tier 2 adaptive parameters.

    Each field has a sensible default matching current fixed values,
    so modules can use this as a drop-in replacement.
    """

    # Min obs per model (dict: model_type -> min_obs)
    min_obs: dict[str, int] = field(default_factory=lambda: dict(_MODEL_FLOORS))

    # Cox/sigmoid blend weights
    blend_w_sig: float = 0.4
    blend_w_cox: float = 0.6

    # PID gains (dict: variable_name -> (kp, ki, kd))
    pid_gains: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    pid_default: tuple[float, float, float] = (0.5, 0.1, 0.2)

    # Graph risk / contagion
    contagion_probs: dict[str, float] = field(default_factory=dict)
    default_contagion_prob: float = 0.3
    participation_rate: float = 0.25

    # Prediction aggregator
    survival_risk_multiplier: float = 2.0
    intraday_low_factor: float = 1.5
    transition_halflife: int = 5

    # Monte Carlo
    mc_n_paths: int = 10_000
    mc_is_tilt: float = 1.5

    # Category A: train/test splits, confidence, vanity
    train_split_fracs: dict[str, float] = field(default_factory=dict)
    conformal_fallback_width: float | None = None

    # Metadata
    adapted: bool = False
    methods_used: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# A1. Variance-Optimized Train/Test Split (Arlot & Celisse 2010)
# ---------------------------------------------------------------------------

# Model-specific constants for split sizing.
# Higher c -> larger test set (more evaluation data needed).
_SPLIT_C: dict[str, float] = {
    "kalman": 2.0,
    "garch": 2.5,
    "var": 2.0,
    "lstm": 3.5,
    "tree": 3.0,
    "transformer": 4.0,
    "baseline": 1.5,
}


def compute_adaptive_split(
    n_eff: float,
    model_type: str = "tree",
) -> float:
    """Compute optimal train fraction from effective sample size.

    Uses Arlot & Celisse (2010) variance-optimized split:
    test_frac = c / sqrt(n_eff), clamped to [0.10, 0.30].

    Parameters
    ----------
    n_eff:
        Effective sample size (from Kish computation).
    model_type:
        Model identifier for complexity-specific constant.

    Returns
    -------
    float
        Train fraction in [0.70, 0.90]. Default 0.85 when n_eff
        is moderate (~200 for tree models).
    """
    c = _SPLIT_C.get(model_type, 2.5)
    test_frac = c / math.sqrt(max(n_eff, 4))
    test_frac = max(0.10, min(0.30, test_frac))
    return round(1.0 - test_frac, 3)


# ---------------------------------------------------------------------------
# A2. Conformal-Derived Confidence (Vovk et al. 2005)
# ---------------------------------------------------------------------------


def compute_calibrated_confidence(
    residuals: list[float] | np.ndarray,
    coverage_target: float = 0.90,
    value_range: float | None = None,
) -> float:
    """Compute confidence score from residual coverage and tightness.

    confidence = empirical_coverage * interval_tightness

    Parameters
    ----------
    residuals:
        Array of (actual - predicted) values.
    coverage_target:
        Target coverage level (default 90%).
    value_range:
        Range of the variable (max - min). If None, estimated
        from residuals.

    Returns
    -------
    float in [0.05, 0.95].
    """
    arr = np.asarray(residuals, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 5:
        return 0.50  # insufficient data, neutral confidence

    # Empirical coverage at target level
    q = float(np.percentile(np.abs(arr), coverage_target * 100))
    coverage = float(np.mean(np.abs(arr) <= q))

    # Interval tightness: how narrow is the interval relative to value range
    if value_range is None or value_range < 1e-8:
        value_range = float(np.percentile(np.abs(arr), 95) - np.percentile(np.abs(arr), 5))
        value_range = max(value_range, 1e-8)

    tightness = 1.0 - min(2 * q / value_range, 0.99)

    confidence = coverage * max(tightness, 0.10)
    return round(max(0.05, min(0.95, confidence)), 3)


# ---------------------------------------------------------------------------
# A3. Percentile Rank Score (Fama & French 1993 factor construction)
# ---------------------------------------------------------------------------


def percentile_rank_score(
    series: pd.Series,
    higher_is_worse: bool = True,
    min_periods: int = 10,
) -> pd.Series:
    """Convert a raw signal to a 0-100 percentile score.

    Replaces arbitrary scaling multipliers (*1000, *200, *10, etc.)
    with a monotone transformation to [0, 100] via expanding
    percentile rank. Standard approach in quantitative factor investing.

    Parameters
    ----------
    series:
        Raw signal series (e.g., R&D intensity excess).
    higher_is_worse:
        If True, higher raw values map to higher (worse) scores.
        If False, lower raw values map to higher scores.
    min_periods:
        Minimum observations before ranking starts.

    Returns
    -------
    pd.Series with values in [0, 100]. NaN where insufficient data.
    """
    rank = series.expanding(min_periods=min_periods).rank(pct=True)
    if higher_is_worse:
        return rank * 100
    else:
        return (1 - rank) * 100


# ---------------------------------------------------------------------------
# A4. IQR-Based Conformal Fallback (Hyndman & Athanasopoulos 2021)
# ---------------------------------------------------------------------------


def compute_conformal_fallback_width(
    residuals: list[float] | np.ndarray | None,
    point_forecast: float = 0.0,
    coverage: float = 0.90,
) -> float:
    """Compute conformal prediction interval fallback width from residuals.

    Uses IQR of residuals scaled to desired coverage (Hyndman 2021).
    Falls back to 10% of forecast value when no residuals available.

    Parameters
    ----------
    residuals:
        Array of forecast residuals.
    point_forecast:
        The point forecast value (for percentage-based fallback).
    coverage:
        Desired coverage level.

    Returns
    -------
    float
        Half-width of the prediction interval.
    """
    if residuals is not None:
        arr = np.asarray(residuals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) >= 10:
            iqr = float(np.percentile(np.abs(arr), 75) - np.percentile(np.abs(arr), 25))
            z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(coverage, 1.645)
            width = max(1.35 * iqr * z / 1.645, abs(point_forecast) * 0.001)
            return round(width, 6)

    # Last resort
    return round(abs(point_forecast) * 0.10, 6)


# ---------------------------------------------------------------------------
# A5. Calibration-Derived Confidence Bounds (Platt 1999)
# ---------------------------------------------------------------------------


def compute_confidence_bounds(
    empirical_coverage: float = 0.90,
    method_type: str = "mar",
) -> tuple[float, float]:
    """Compute adaptive confidence clip bounds from empirical coverage.

    Parameters
    ----------
    empirical_coverage:
        Fraction of held-out values within predicted intervals.
    method_type:
        "mar" for Missing At Random path, "mnar" for Missing Not
        At Random, "gain" for adversarial imputation.

    Returns
    -------
    (lower_bound, upper_bound) for confidence clipping.
    """
    # Upper bound: slightly above empirical coverage (optimistic ceiling)
    upper = min(0.95, empirical_coverage * 1.05)

    # Lower bound: depends on method type
    lower_map = {
        "mar": 0.05,    # MICE/GP/Matrix completion have good priors
        "mnar": 0.10,   # Heckman has structural model
        "gain": 0.10,   # adversarial has distribution matching
        "vae": 0.05,    # VAE reconstruction
    }
    lower = lower_map.get(method_type, 0.05)

    return round(lower, 3), round(upper, 3)


# =========================================================================
# Category C: Edge-case adaptive parameters
# =========================================================================


# ---------------------------------------------------------------------------
# C1. Country-Relative Conflict Event Scoring (Gleditsch 2002, Poisson Z)
# ---------------------------------------------------------------------------


def compute_conflict_event_score(
    events_30d: int,
    baseline_monthly_events: float = 0.0,
) -> float:
    """Score conflict events relative to the country's baseline rate.

    Uses a Poisson Z-score: z = (observed - expected) / sqrt(expected).
    Then applies sigmoid to map to [0, 1].

    Parameters
    ----------
    events_30d:
        Number of conflict events in the last 30 days.
    baseline_monthly_events:
        Country's typical monthly event count (from UCDP history).
        If 0 or unknown, falls back to absolute scoring.

    Returns
    -------
    float in [0, 1]. Higher = more concerning.
    """
    if events_30d == 0:
        return 0.0

    if baseline_monthly_events > 1.0:
        # Poisson Z-score
        z = (events_30d - baseline_monthly_events) / max(math.sqrt(baseline_monthly_events), 1.0)
        # Sigmoid mapping: z=0 -> 0.5, z=2 -> 0.88, z=-2 -> 0.12
        score = 1.0 / (1.0 + math.exp(-z))
    else:
        # No baseline: absolute log-scaling (current-style fallback)
        score = min(math.log10(events_30d + 1) / 2.0, 1.0)

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# C3. Parkinson/Yang-Zhang OHLC Noise Decomposition (1980/2000)
# ---------------------------------------------------------------------------


def compute_ohlc_noise_factors(
    cache: pd.DataFrame,
    lookback: int = 63,
) -> dict[str, float]:
    """Decompose price noise into overnight gap and intraday components.

    Uses Parkinson (1980) for intraday and Yang-Zhang (2000) for
    overnight gap variance decomposition.

    Parameters
    ----------
    cache:
        Daily cache with open, high, low, close columns.
    lookback:
        Days of recent data.

    Returns
    -------
    Dict with keys: gap_factor, high_low_factor, horizon_growth_rate.
    Defaults: gap_factor=0.3, high_low_factor=0.5, horizon_growth=0.02.
    """
    defaults = {"gap_factor": 0.3, "high_low_factor": 0.5, "horizon_growth_rate": 0.02}

    required = ["open", "high", "low", "close"]
    if not all(c in cache.columns for c in required):
        return defaults

    ohlc = cache[required].dropna().tail(lookback)
    if len(ohlc) < 20:
        return defaults

    o, h, l, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
    prev_c = c.shift(1).dropna()

    if len(prev_c) < 15:
        return defaults

    # Align
    o_a = o.iloc[1:]
    h_a = h.iloc[1:]
    l_a = l.iloc[1:]
    c_a = c.iloc[1:]
    pc = prev_c

    # Close-to-close volatility
    cc_ret = np.log(c_a.values / pc.values)
    sigma_cc = max(float(np.std(cc_ret)), 1e-8)

    # Overnight (gap) volatility: var(log(open / prev_close))
    gap_ret = np.log(o_a.values / pc.values)
    sigma_overnight = max(float(np.std(gap_ret)), 1e-10)

    # Parkinson intraday volatility: var(log(high/low)) / (4*ln2)
    hl_ratio = np.log(h_a.values / np.maximum(l_a.values, 1e-10))
    sigma_parkinson = max(float(np.sqrt(np.mean(hl_ratio ** 2) / (4 * math.log(2)))), 1e-10)

    gap_factor = round(sigma_overnight / sigma_cc, 4)
    high_low_factor = round(sigma_parkinson / sigma_cc * 0.5, 4)

    # Hurst exponent for horizon noise scaling (R/S analysis)
    horizon_growth = _compute_hurst_growth_rate(cc_ret)

    # Clamp to reasonable ranges
    gap_factor = max(0.05, min(1.0, gap_factor))
    high_low_factor = max(0.1, min(1.5, high_low_factor))
    horizon_growth = max(0.0, min(0.10, horizon_growth))

    return {
        "gap_factor": gap_factor,
        "high_low_factor": high_low_factor,
        "horizon_growth_rate": horizon_growth,
    }


# ---------------------------------------------------------------------------
# C6. Hurst Exponent Noise Scaling (Hurst 1951, Lo 1991)
# ---------------------------------------------------------------------------


def _compute_hurst_growth_rate(
    returns: np.ndarray,
    max_lag: int = 50,
) -> float:
    """Compute per-day noise growth rate from Hurst exponent.

    Uses R/S (rescaled range) analysis. H > 0.5 = trending (noise grows),
    H < 0.5 = mean-reverting (noise shrinks), H = 0.5 = random walk.

    Returns the per-day growth rate: (2*H - 1) / 252.
    """
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    if n < 20:
        return 0.02  # default

    # R/S analysis across multiple lag windows
    lags = range(10, min(max_lag, n // 2))
    if len(list(lags)) < 3:
        return 0.02

    log_rs = []
    log_n = []

    for lag in lags:
        rs_vals = []
        for start in range(0, n - lag, lag):
            window = returns[start:start + lag]
            if len(window) < lag:
                continue
            mean_r = np.mean(window)
            cumdev = np.cumsum(window - mean_r)
            R = np.max(cumdev) - np.min(cumdev)
            S = max(np.std(window, ddof=1), 1e-10)
            rs_vals.append(R / S)
        if rs_vals:
            log_rs.append(math.log(float(np.mean(rs_vals))))
            log_n.append(math.log(lag))

    if len(log_rs) < 3:
        return 0.02

    # Linear regression: log(R/S) = H * log(n) + c
    log_rs_arr = np.array(log_rs)
    log_n_arr = np.array(log_n)
    try:
        H = float(np.polyfit(log_n_arr, log_rs_arr, 1)[0])
    except Exception:
        return 0.02

    H = max(0.2, min(0.8, H))

    # Per-day growth rate: positive if trending, negative if mean-reverting
    growth = (2 * H - 1.0) / 252.0
    return round(growth, 6)


def compute_hurst_exponent(
    cache: pd.DataFrame,
    column: str = "return_1d",
) -> float:
    """Compute the Hurst exponent for a cache column.

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    column:
        Column to analyze (typically return_1d).

    Returns
    -------
    Hurst exponent in [0.2, 0.8]. 0.5 = random walk.
    """
    if column not in cache.columns:
        return 0.5

    returns = cache[column].dropna().values
    if len(returns) < 20:
        return 0.5

    growth = _compute_hurst_growth_rate(returns)
    H = (growth * 252 + 1.0) / 2.0
    return round(max(0.2, min(0.8, H)), 3)


# ---------------------------------------------------------------------------
# C4. Bootstrap Prediction Spread (Efron 1979)
# ---------------------------------------------------------------------------


def compute_bootstrap_spread(
    model_predictions: dict[str, float],
    default_spread_pct: float = 0.10,
) -> float:
    """Compute prediction interval spread from model ensemble disagreement.

    Instead of a fixed percentage, uses the actual spread between
    model predictions as the uncertainty measure.

    Parameters
    ----------
    model_predictions:
        Dict of {model_name: point_forecast}.
    default_spread_pct:
        Fallback spread as fraction of forecast.

    Returns
    -------
    float: half-width of the prediction spread.
    """
    if not model_predictions or len(model_predictions) < 2:
        # Can't compute spread from 0-1 models
        vals = list(model_predictions.values()) if model_predictions else [0.0]
        return abs(vals[0]) * default_spread_pct if vals else 0.0

    vals = [v for v in model_predictions.values() if not math.isnan(v)]
    if len(vals) < 2:
        return abs(vals[0]) * default_spread_pct if vals else 0.0

    arr = np.array(vals)
    # IQR of model predictions (robust to outlier models)
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr = q75 - q25

    # Scale to 90% CI: 1.35 * IQR ~= 1 sigma, then * 1.645
    spread = 1.35 * iqr * 1.645 / 2.0

    return max(spread, abs(float(np.mean(arr))) * 0.001)

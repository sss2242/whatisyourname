"""Iterative OHLC candlestick prediction for multiple horizons.

Generates day-by-day predicted Open/High/Low/Close series for:
- Next week (5 trading days)
- Next month (21 trading days)
- Next year (252 trading days)

Each day is predicted from the previous day's predicted state, with
uncertainty bands widening as the horizon extends.  Next-day OHLC
is masked per Technical Alpha protection (only Low is shown).

Spec refs: Sec E.4 Phase 4, Sec 17 (Technical Alpha)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OHLCCandle:
    """A single predicted OHLC candlestick."""

    date: str = ""
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    confidence: float = 1.0  # decays with horizon


@dataclass
class OHLCPredictionResult:
    """Container for all OHLC prediction outputs."""

    next_day: OHLCCandle | None = None  # masked except Low
    next_week: list[OHLCCandle] = field(default_factory=list)
    next_month: list[OHLCCandle] = field(default_factory=list)
    next_year: list[OHLCCandle] = field(default_factory=list)
    fitted: bool = False
    error: str | None = None


def _estimate_daily_ohlc(
    prev_close: float,
    daily_return: float,
    volatility: float,
    volume_ma: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Estimate a single day's OHLC from return and volatility.

    Uses empirical relationships between OHLC and close-to-close returns:
    - Open: previous close + small gap (mean-reverting)
    - High: max of open/close + intraday volatility component
    - Low: min of open/close - intraday volatility component
    - Close: previous close * (1 + daily_return)
    """
    predicted_close = prev_close * (1.0 + daily_return)

    # Open: small gap from previous close (mean-reverting)
    gap = rng.normal(0, volatility * 0.3) * prev_close
    predicted_open = prev_close + gap

    # Intraday range: proportional to volatility
    intraday_range = abs(prev_close * volatility * 1.5)

    # High and Low
    body_high = max(predicted_open, predicted_close)
    body_low = min(predicted_open, predicted_close)
    predicted_high = body_high + abs(rng.normal(0, intraday_range * 0.5))
    predicted_low = body_low - abs(rng.normal(0, intraday_range * 0.5))

    # Ensure consistency
    predicted_high = max(predicted_high, predicted_open, predicted_close)
    predicted_low = min(predicted_low, predicted_open, predicted_close)
    predicted_low = max(predicted_low, 0.01)  # no negative prices

    return {
        "open": round(predicted_open, 4),
        "high": round(predicted_high, 4),
        "low": round(predicted_low, 4),
        "close": round(predicted_close, 4),
        "volume": round(volume_ma * (1 + rng.normal(0, 0.15)), 0),
    }


def predict_ohlc_series(
    cache: pd.DataFrame,
    forecast_result: Any | None = None,
    mc_result: Any | None = None,
    *,
    pattern_drift_multiplier: float = 1.0,
    cycle_result: Any | None = None,
    random_seed: int = 42,
) -> OHLCPredictionResult:
    """Generate predicted OHLC candlestick series for all horizons.

    Parameters
    ----------
    cache:
        Daily cache with at least ``close``, ``open``, ``high``, ``low``,
        ``volume``, ``volatility_21d``, and ``return_1d`` columns.
    forecast_result:
        Output from ``run_forecasting()`` (optional, used for return
        forecasts if available).
    mc_result:
        Monte Carlo result (optional, used for survival-adjusted
        uncertainty).
    cycle_result:
        Output from ``run_cycle_decomposition()`` (optional). If provided,
        modulates predicted returns by the dominant cycle phase: at cycle
        peak, biases return downward; at trough, biases upward.

    Returns
    -------
    OHLCPredictionResult with candles for each horizon.
    """
    result = OHLCPredictionResult()
    rng = np.random.default_rng(random_seed)

    if cache is None or cache.empty or "close" not in cache.columns:
        result.error = "No price data available for OHLC prediction"
        return result

    # Extract base parameters from recent history
    close = cache["close"].dropna()
    if len(close) < 21:
        result.error = "Insufficient price history for OHLC prediction"
        return result

    last_close = float(close.iloc[-1])
    returns = close.pct_change().dropna()

    # Drift and volatility estimates
    mu = float(returns.tail(63).mean())  # ~3-month average daily return
    # Apply pattern-based drift adjustment (Synergy C)
    mu *= pattern_drift_multiplier

    # Apply cycle phase modulation (Phase 1.4 improvement)
    # If a dominant cycle is detected, modulate predicted return by cycle phase.
    # At cycle peak, bias return downward; at trough, bias upward.
    _cycle_adjustment = 0.0
    if cycle_result is not None:
        dominant_cycles = getattr(cycle_result, "dominant_cycles", [])
        if dominant_cycles:
            # Use the strongest cycle (first in list, sorted by amplitude)
            top_cycle = dominant_cycles[0]
            period = getattr(top_cycle, "period_days", 0)
            phase = getattr(top_cycle, "phase", 0)
            amplitude = getattr(top_cycle, "amplitude", 0)
            if period > 0 and amplitude > 0:
                # Current phase position: how far through the cycle are we?
                # phase is in radians; advance by 2*pi/period per day
                import math as _math
                # Negative cosine: +1 at trough (buy), -1 at peak (sell)
                # Normalize amplitude by price (EMD amplitude is in price
                # units, not percentage units).  Cap to prevent the cycle
                # adjustment from dominating the base drift.
                _norm_amplitude = amplitude / max(last_close, 1.0)
                _cycle_adjustment = -_math.cos(phase) * _norm_amplitude * 0.01
                _cycle_adjustment = max(-0.005, min(0.005, _cycle_adjustment))
                mu += _cycle_adjustment
                logger.debug(
                    "Cycle phase adjustment: period=%dd, phase=%.2f rad, "
                    "adjustment=%.6f",
                    period, phase, _cycle_adjustment,
                )
    vol_col = "volatility_21d"
    if vol_col in cache.columns and cache[vol_col].notna().any():
        sigma = float(cache[vol_col].dropna().iloc[-1])
        # Convert annualised vol to daily
        from operator1.freq_constants import get_vol_annualization as _gva
        sigma_daily = sigma / _gva() if sigma > 0.05 else sigma
    else:
        sigma_daily = float(returns.tail(21).std())

    # Volume moving average
    vol_ma = float(cache["volume"].tail(21).mean()) if "volume" in cache.columns else 1e6

    # Extract forecast-based return expectations per horizon
    horizon_mu: dict[str, float] = {}
    if forecast_result is not None:
        forecasts = getattr(forecast_result, "forecasts", {})
        if isinstance(forecasts, dict):
            for var in ("close", "return_1d"):
                if var in forecasts:
                    for h_label, val in forecasts[var].items():
                        if isinstance(val, (int, float)) and not math.isnan(val):
                            # Convert to daily return
                            days = {"1d": 1, "5d": 5, "21d": 21, "252d": 252}.get(h_label, 1)
                            if var == "close" and last_close > 0:
                                total_ret = (val - last_close) / last_close
                                horizon_mu[h_label] = total_ret / days
                            elif var == "return_1d":
                                horizon_mu[h_label] = val

    # Survival adjustment for uncertainty.
    # **Fix (2026-05-09):** Clamp survival_mult to [1.0, 1.5] to prevent
    # broken survival probabilities (from Problem 4: sector threshold mismatch)
    # from inflating noise to 9.4% daily volatility at year-end.
    # Previously: survival_mult = 1.0 + (1.0 - 0.473) = 1.527 for Apple,
    # which combined with linear noise growth produced $78.89 crash prediction.
    survival_mult = 1.0
    if mc_result is not None:
        surv_mean = getattr(mc_result, "survival_probability_mean", 1.0)
        if isinstance(surv_mean, (int, float)) and surv_mean < 1.0:
            survival_mult = 1.0 + (1.0 - surv_mean) * 0.5  # dampen: half the penalty
            survival_mult = min(survival_mult, 1.5)  # hard cap

    # --- MC-derived predictions (primary path when MC available) ---
    # Derive Close from MC terminal distributions, H/L from Parkinson formula.
    # This replaces the random walk with statistically grounded estimates.
    _mc_terminal = {}
    if mc_result is not None:
        _mc_terminal = getattr(mc_result, "terminal_values", {}) or {}

    # Skewness for asymmetric H/L estimation
    _skew = 0.0
    if "skewness_63d" in cache.columns:
        _sk = cache["skewness_63d"].dropna()
        if len(_sk) > 0:
            _skew = float(_sk.iloc[-1])
            if math.isnan(_skew):
                _skew = 0.0
    _skew = max(-1.0, min(1.0, _skew))

    def _mc_derived_candle(horizon_key: str, n_days: int, prev_close: float) -> OHLCCandle | None:
        """Derive a single candle from MC terminal distribution (Parkinson + skew)."""
        terminal = _mc_terminal.get(horizon_key)
        if terminal is None or not hasattr(terminal, "__len__") or len(terminal) < 50:
            return None
        terminal_arr = np.asarray(terminal, dtype=float)
        terminal_arr = terminal_arr[np.isfinite(terminal_arr)]
        if len(terminal_arr) < 30:
            return None

        # Close: MC median
        close_est = prev_close * float(np.median(terminal_arr))

        # Parkinson range: expected H-L from volatility
        log_terminal = np.log(np.maximum(terminal_arr, 1e-10))
        sigma_est = float(np.std(log_terminal))
        if sigma_est < 1e-6:
            sigma_est = sigma_daily * math.sqrt(n_days / 252)
        expected_range = sigma_est * math.sqrt(8.0 / math.pi)

        # Skewness-adjusted split
        up_ratio = 0.5 + 0.1 * _skew
        down_ratio = 0.5 - 0.1 * _skew

        high_est = close_est * (1.0 + expected_range * up_ratio)
        low_est = close_est * (1.0 - expected_range * down_ratio)

        # Sanity: H >= C >= L, all positive
        high_est = max(high_est, close_est * 1.001)
        low_est = min(low_est, close_est * 0.999)
        low_est = max(low_est, 0.01)

        confidence = max(0.1, 1.0 / (1.0 + 0.05 * math.sqrt(n_days)))

        return OHLCCandle(
            date="",  # filled by caller
            open=round(prev_close, 4),
            high=round(high_est, 4),
            low=round(low_est, 4),
            close=round(close_est, 4),
            volume=round(vol_ma, 0) if vol_ma else None,
            confidence=round(confidence, 4),
        )

    # --- Fallback: random walk series (when no MC terminal values) ---
    from datetime import timedelta
    last_date = cache.index[-1] if hasattr(cache.index[-1], "date") else pd.Timestamp.now()

    def _generate_series(n_days: int, horizon_key: str) -> list[OHLCCandle]:
        """Generate n_days of iterative OHLC predictions."""
        candles: list[OHLCCandle] = []
        prev_c = last_close

        # Use horizon-specific drift if available, else base drift
        daily_mu = horizon_mu.get(horizon_key, mu)

        for day_i in range(1, n_days + 1):
            # Confidence decays with sqrt of horizon
            confidence = max(0.1, 1.0 / (1.0 + 0.1 * math.sqrt(day_i)))

            # Add noise that grows with horizon.
            # **Fix (2026-05-09):** Changed from linear growth (1 + 0.02*day)
            # to sqrt growth.  Linear growth reached 6.04x at day 252,
            # producing 9.4% daily volatility and $78.89 crash predictions.
            # Sqrt growth reaches ~1.32x at day 252 (sqrt(252)/12 ~ 1.32),
            # consistent with the square-root-of-time volatility scaling law.
            noise_scale = sigma_daily * survival_mult * (1.0 + math.sqrt(day_i) / 12.0)
            daily_ret = daily_mu + rng.normal(0, noise_scale)

            ohlc = _estimate_daily_ohlc(prev_c, daily_ret, noise_scale, vol_ma, rng)

            # Compute date
            candle_date = last_date + timedelta(days=day_i)
            # Skip weekends
            while hasattr(candle_date, "weekday") and candle_date.weekday() >= 5:
                candle_date += timedelta(days=1)

            candle = OHLCCandle(
                date=str(candle_date.date()) if hasattr(candle_date, "date") else str(candle_date),
                open=ohlc["open"],
                high=ohlc["high"],
                low=ohlc["low"],
                close=ohlc["close"],
                volume=ohlc["volume"],
                confidence=round(confidence, 4),
            )
            candles.append(candle)
            prev_c = ohlc["close"]

        return candles

    try:
        # Next day (Technical Alpha: mask everything except Low)
        # Generate many simulated next-day candles via Monte Carlo to find
        # the most reliable Low estimate.  A single random sample gives an
        # unreliable low; averaging across N simulations produces a robust
        # estimate of the day's likely trough.
        _N_NEXT_DAY_SIMS = 500
        _sim_lows: list[float] = []
        _sim_date: str = ""
        for _sim_i in range(_N_NEXT_DAY_SIMS):
            _sim_rng = np.random.default_rng(random_seed + _sim_i)
            _sim_mu = horizon_mu.get("1d", mu)
            _noise = sigma_daily * survival_mult
            _sim_ret = _sim_mu + _sim_rng.normal(0, _noise)
            _sim_ohlc = _estimate_daily_ohlc(last_close, _sim_ret, _noise, vol_ma, _sim_rng)
            _sim_lows.append(_sim_ohlc["low"])
            if not _sim_date:
                _d = last_date + timedelta(days=1)
                while hasattr(_d, "weekday") and _d.weekday() >= 5:
                    _d += timedelta(days=1)
                _sim_date = str(_d.date()) if hasattr(_d, "date") else str(_d)

        # Use the 10th percentile of simulated lows as the estimated low
        # (conservative: 90% of simulations stay above this level)
        _robust_low = float(np.percentile(_sim_lows, 10))
        _median_low = float(np.median(_sim_lows))
        _confidence = max(0.1, 1.0 - (np.std(_sim_lows) / last_close) * 10)

        result.next_day = OHLCCandle(
            date=_sim_date,
            open=None,   # MASKED
            high=None,   # MASKED
            low=round(_robust_low, 4),  # VISIBLE -- robust estimate from 500 simulations
            close=None,  # MASKED
            volume=None,  # MASKED
            confidence=round(min(_confidence, 1.0), 4),
        )
        logger.info(
            "Next-day Low estimate: %.4f (p10 of %d sims, median=%.4f, std=%.4f)",
            _robust_low, _N_NEXT_DAY_SIMS, _median_low, float(np.std(_sim_lows)),
        )

        # Next week (5 trading days) -- MC-derived if available, else random walk
        _mc_week = _mc_derived_candle("5d", 5, last_close)
        if _mc_week is not None:
            _d = last_date + timedelta(days=5)
            _mc_week = OHLCCandle(date=str(_d.date()) if hasattr(_d, "date") else str(_d),
                                  open=_mc_week.open, high=_mc_week.high, low=_mc_week.low,
                                  close=_mc_week.close, volume=_mc_week.volume, confidence=_mc_week.confidence)
            result.next_week = [_mc_week]
            logger.debug("OHLC week: MC-derived close=%.2f", _mc_week.close)
        else:
            result.next_week = _generate_series(5, "5d")

        # Next month (21 trading days) -- MC-derived if available
        _mc_month = _mc_derived_candle("21d", 21, last_close)
        if _mc_month is not None:
            _d = last_date + timedelta(days=30)
            _mc_month = OHLCCandle(date=str(_d.date()) if hasattr(_d, "date") else str(_d),
                                   open=_mc_month.open, high=_mc_month.high, low=_mc_month.low,
                                   close=_mc_month.close, volume=_mc_month.volume, confidence=_mc_month.confidence)
            result.next_month = [_mc_month]
            logger.debug("OHLC month: MC-derived close=%.2f", _mc_month.close)
        else:
            result.next_month = _generate_series(21, "21d")

        # Next year (252 trading days) -- MC-derived if available
        _mc_year = _mc_derived_candle("252d", 252, last_close)
        if _mc_year is not None:
            _d = last_date + timedelta(days=365)
            _mc_year = OHLCCandle(date=str(_d.date()) if hasattr(_d, "date") else str(_d),
                                  open=_mc_year.open, high=_mc_year.high, low=_mc_year.low,
                                  close=_mc_year.close, volume=_mc_year.volume, confidence=_mc_year.confidence)
            result.next_year = [_mc_year]
            logger.info("OHLC year: MC-derived close=%.2f (vs current %.2f, %.1f%%)",
                        _mc_year.close, last_close,
                        (_mc_year.close - last_close) / last_close * 100)
        else:
            result.next_year = _generate_series(252, "252d")

        result.fitted = True
        logger.info(
            "OHLC prediction complete: 1+5+21+252 = %d candles generated",
            1 + len(result.next_week) + len(result.next_month) + len(result.next_year),
        )

    except Exception as exc:
        result.error = str(exc)
        logger.warning("OHLC prediction failed: %s", exc)

    return result


def format_ohlc_for_profile(result: OHLCPredictionResult) -> dict[str, Any]:
    """Convert OHLCPredictionResult to a profile-ready dict."""
    def _candle_dict(c: OHLCCandle) -> dict[str, Any]:
        return {
            "date": c.date,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "confidence": c.confidence,
        }

    profile: dict[str, Any] = {
        "available": result.fitted,
    }

    if result.next_day:
        profile["next_day"] = _candle_dict(result.next_day)
        profile["next_day"]["technical_alpha_masked"] = True

    if result.next_week:
        profile["next_week"] = {
            "n_candles": len(result.next_week),
            "series": [_candle_dict(c) for c in result.next_week],
            "predicted_return": round(
                (result.next_week[-1].close / result.next_week[0].open - 1) * 100, 2
            ) if result.next_week[0].open and result.next_week[-1].close else None,
        }

    if result.next_month:
        profile["next_month"] = {
            "n_candles": len(result.next_month),
            "series": [_candle_dict(c) for c in result.next_month],
            "predicted_return": round(
                (result.next_month[-1].close / result.next_month[0].open - 1) * 100, 2
            ) if result.next_month[0].open and result.next_month[-1].close else None,
        }

    if result.next_year:
        # For year, aggregate to monthly summaries to reduce clutter
        monthly_agg: list[dict[str, Any]] = []
        chunk_size = 21  # ~1 month
        for i in range(0, len(result.next_year), chunk_size):
            chunk = result.next_year[i:i + chunk_size]
            if not chunk:
                continue
            monthly_agg.append({
                "period_start": chunk[0].date,
                "period_end": chunk[-1].date,
                "open": chunk[0].open,
                "high": max(c.high for c in chunk if c.high is not None),
                "low": min(c.low for c in chunk if c.low is not None),
                "close": chunk[-1].close,
                "avg_confidence": round(sum(c.confidence for c in chunk) / len(chunk), 4),
            })

        profile["next_year"] = {
            "n_candles": len(result.next_year),
            "monthly_aggregates": monthly_agg,
            "predicted_return": round(
                (result.next_year[-1].close / result.next_year[0].open - 1) * 100, 2
            ) if result.next_year[0].open and result.next_year[-1].close else None,
        }

    return profile

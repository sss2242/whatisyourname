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
    vol_col = "volatility_21d"
    if vol_col in cache.columns and cache[vol_col].notna().any():
        sigma = float(cache[vol_col].dropna().iloc[-1])
        # Convert annualised vol to daily
        sigma_daily = sigma / math.sqrt(252) if sigma > 0.05 else sigma
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

    # Survival adjustment for uncertainty
    survival_mult = 1.0
    if mc_result is not None:
        surv_mean = getattr(mc_result, "survival_probability_mean", 1.0)
        if isinstance(surv_mean, (int, float)) and surv_mean < 1.0:
            survival_mult = 1.0 + (1.0 - surv_mean)  # widen uncertainty

    # --- Generate series for each horizon ---
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

            # Add noise that grows with horizon
            noise_scale = sigma_daily * survival_mult * (1.0 + 0.02 * day_i)
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

        # Next week (5 trading days, full OHLC)
        result.next_week = _generate_series(5, "5d")

        # Next month (21 trading days, full OHLC)
        result.next_month = _generate_series(21, "21d")

        # Next year (252 trading days, full OHLC)
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

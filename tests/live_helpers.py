"""Shared helpers for live full-pipeline integration tests.

Each regional test file imports from here to avoid duplication.
These tests hit REAL APIs -- they are skipped when network is unavailable.
"""

from __future__ import annotations

import logging
import os
import socket
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Network check
# ---------------------------------------------------------------------------

def is_network_available(host: str = "8.8.8.8", port: int = 53, timeout: float = 3) -> bool:
    """Quick check if we have internet connectivity."""
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except OSError:
        return False


NETWORK_OK = is_network_available()

# ---------------------------------------------------------------------------
# Date window: exactly 7 calendar days ending today
# ---------------------------------------------------------------------------

WINDOW_END = date.today()
WINDOW_START = WINDOW_END - timedelta(days=7)

# For OHLCV we need a wider window to guarantee at least 5 business days
OHLCV_START = WINDOW_END - timedelta(days=14)


# ---------------------------------------------------------------------------
# Temporary cache directory (cleaned up after tests)
# ---------------------------------------------------------------------------

def make_temp_cache() -> str:
    """Create a temporary directory for test cache files."""
    d = tempfile.mkdtemp(prefix="op1_live_test_")
    return d


# ---------------------------------------------------------------------------
# OHLCV helpers
# ---------------------------------------------------------------------------

def fetch_ohlcv_safe(ticker: str, market_id: str, years: int = 1) -> pd.DataFrame:
    """Fetch OHLCV with graceful fallback."""
    try:
        from operator1.clients.ohlcv_provider import fetch_ohlcv
        df = fetch_ohlcv(ticker, market_id, years=years)
        if df is not None and len(df) > 0:
            return df
    except Exception as exc:
        logger.warning("OHLCV fetch failed for %s/%s: %s", ticker, market_id, exc)

    # yfinance fallback
    try:
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance
        return fetch_ohlcv_yfinance(ticker, years=years)
    except Exception as exc:
        logger.warning("yfinance fallback also failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Macro helpers
# ---------------------------------------------------------------------------

def fetch_macro_safe(country_iso2: str, market_id: str = "") -> dict[str, pd.Series]:
    """Fetch macro data with wbgapi fallback."""
    try:
        from operator1.clients.macro_provider import fetch_macro
        result = fetch_macro(country_iso2, market_id=market_id, years=2)
        if result:
            return result
    except Exception as exc:
        logger.warning("Primary macro failed for %s: %s", country_iso2, exc)

    # wbgapi fallback
    try:
        from operator1.clients.macro_wbgapi import fetch_macro_wbgapi
        return fetch_macro_wbgapi(country_iso2, years=2)
    except Exception as exc:
        logger.warning("wbgapi fallback also failed for %s: %s", country_iso2, exc)
        return {}


# ---------------------------------------------------------------------------
# Minimal cache builder (7-day synthetic from real OHLCV)
# ---------------------------------------------------------------------------

def build_minimal_cache(
    ohlcv: pd.DataFrame,
    macro: dict[str, pd.Series] | None = None,
    profile: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a minimal daily cache from OHLCV + macro for testing.

    Takes the last 7 business days of OHLCV and merges macro indicators.
    """
    if ohlcv is None or len(ohlcv) == 0:
        # Return a minimal synthetic cache
        idx = pd.bdate_range(end=WINDOW_END, periods=5, name="date")
        return pd.DataFrame({
            "close": [100.0] * 5,
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "volume": [1000000] * 5,
        }, index=idx)

    # Ensure date index
    df = ohlcv.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Use all available data (models need 60-100+ rows to fit).
    # The "7 day" window applies to the freshness of the test, but the
    # cache must contain enough history for regime detection (60+),
    # forecasting (100+), etc.  We keep up to 252 rows (1 trading year).
    df = df.sort_index().tail(252)

    # Standardize column names
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("close", "open", "high", "low", "volume"):
            col_map[col] = lc
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    for col in ("close", "open", "high", "low", "volume"):
        if col not in df.columns:
            if col == "volume":
                df[col] = 0
            elif "close" in df.columns:
                df[col] = df["close"]

    # Add derived columns needed by the pipeline
    if "close" in df.columns and len(df) > 1:
        df["return_1d"] = df["close"].pct_change()
        df["volatility_21d"] = df["return_1d"].rolling(min_periods=1, window=min(len(df), 5)).std() * np.sqrt(252)
    else:
        df["return_1d"] = 0.0
        df["volatility_21d"] = 0.02

    # Merge macro as constant columns (last known values)
    if macro:
        for indicator_name, series in macro.items():
            if series is not None and len(series) > 0:
                df[indicator_name] = float(series.iloc[-1])

    # Add profile fields as constants
    if profile:
        for key in ("ticker", "country", "sector", "industry", "isin"):
            if key in profile:
                df[key] = profile[key]

    # Placeholder financial ratios
    for col in ("current_ratio", "debt_to_equity_abs", "fcf_yield"):
        if col not in df.columns:
            df[col] = 1.5 if col == "current_ratio" else (1.0 if "debt" in col else 0.05)

    # Regime label placeholder
    df["regime_label"] = "unknown"

    return df


# ---------------------------------------------------------------------------
# Pipeline stage runners
# ---------------------------------------------------------------------------

def run_regime_detection(cache: pd.DataFrame) -> pd.DataFrame:
    """Run regime detection on the cache, returning updated cache."""
    try:
        from operator1.models.regime_detector import detect_regimes_and_breaks
        detect_regimes_and_breaks(cache)
    except Exception as exc:
        logger.warning("Regime detection failed (non-fatal): %s", exc)
        if "regime_label" not in cache.columns:
            cache["regime_label"] = "unknown"
    return cache


def run_forecasting_safe(cache: pd.DataFrame):
    """Run forecasting with graceful fallback.

    Returns (updated_cache, ForecastResult).  The caller should use
    the updated cache for subsequent stages.
    """
    from operator1.models.forecasting import ForecastResult

    try:
        from operator1.models.forecasting import run_forecasting

        # Only pass numeric columns to forecasting (string columns like
        # regime_label, ticker, country cause isnan errors).
        numeric_cols = cache.select_dtypes(include=["number"]).columns.tolist()

        # Filter to meaningful financial variables (not helper columns).
        skip_prefixes = ("is_missing_", "regime_", "structural_")
        variables = [
            c for c in numeric_cols
            if not any(c.startswith(p) for p in skip_prefixes)
            and c not in ("volume",)  # volume is huge and noisy
        ]

        if not variables:
            logger.warning("No numeric variables found for forecasting")
            return cache, ForecastResult()

        # run_forecasting returns (updated_cache, ForecastResult)
        updated_cache, result = run_forecasting(
            cache, variables, enable_burnout=False,
        )
        return updated_cache, result
    except Exception as exc:
        logger.warning("Forecasting failed: %s", exc)
        return cache, ForecastResult()


def run_monte_carlo_safe(cache: pd.DataFrame):
    """Run Monte Carlo with graceful fallback."""
    try:
        from operator1.models.monte_carlo import run_monte_carlo
        return run_monte_carlo(cache, n_paths=100)
    except Exception as exc:
        logger.warning("Monte Carlo failed: %s", exc)
        return None


def run_aggregation_safe(cache, forecast_result, mc_result, cache_dir: str = ""):
    """Run prediction aggregation."""
    from operator1.models.prediction_aggregator import run_prediction_aggregation
    return run_prediction_aggregation(
        cache,
        forecast_result,
        mc_result,
        save_to_cache=bool(cache_dir),
        cache_dir=cache_dir or "cache",
    )

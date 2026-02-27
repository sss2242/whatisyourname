"""OHLCV provider dispatcher -- routes to per-region primary or yfinance fallback.

Each market has a primary OHLCV source. If the primary fails or its library
is not installed, falls back to yfinance (global, no key needed).

Research: .roo/research/ohlcv-per-region-2026-02-25.md
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Market ID -> primary OHLCV fetcher function
# Markets not listed here use yfinance directly.
_PRIMARY_FETCHERS: dict[str, str] = {
    "kr_dart": "pykrx",
    "tw_mops": "twstock",
    "cn_sse": "baostock",  # China: baostock (free, no key, works globally)
    "in_bse": "nselib",  # India: nselib (NSE data, no key, no geo-blocking)
}


def fetch_ohlcv(
    ticker: str,
    market_id: str,
    years: int = 5,
) -> pd.DataFrame:
    """Fetch OHLCV data for a ticker, using per-region primary + yfinance fallback.

    Parameters
    ----------
    ticker:
        Raw ticker symbol.
    market_id:
        Market identifier (e.g. "us_sec_edgar", "kr_dart").
    years:
        Number of years of history.

    Returns
    -------
    DataFrame with columns: date, open, high, low, close, volume.
    """
    df = pd.DataFrame()

    # Try per-region primary first
    primary = _PRIMARY_FETCHERS.get(market_id)

    if primary == "pykrx":
        try:
            from operator1.clients.ohlcv_pykrx import fetch_ohlcv_pykrx
            df = fetch_ohlcv_pykrx(ticker, years=years)
        except Exception as exc:
            logger.debug("pykrx primary failed: %s", exc)

    elif primary == "twstock":
        try:
            from operator1.clients.ohlcv_twstock import fetch_ohlcv_twstock
            df = fetch_ohlcv_twstock(ticker, years=years)
        except Exception as exc:
            logger.debug("twstock primary failed: %s", exc)

    elif primary == "baostock":
        try:
            from operator1.clients.ohlcv_baostock import fetch_ohlcv_baostock
            df = fetch_ohlcv_baostock(ticker, years=years)
        except Exception as exc:
            logger.debug("baostock primary failed: %s", exc)

    elif primary == "nselib":
        try:
            from operator1.clients.ohlcv_nselib import fetch_ohlcv_nselib
            df = fetch_ohlcv_nselib(ticker, years=years)
        except Exception as exc:
            logger.debug("nselib primary failed: %s", exc)

    # J-Quants OHLCV is handled inside jp_jquants_wrapper.py get_quotes()
    # So jp_jquants is NOT listed here -- it goes through the PIT client path.

    # Fallback to yfinance if primary returned nothing
    if df.empty:
        if primary:
            logger.info(
                "Per-region OHLCV (%s) returned empty for %s; falling back to yfinance",
                primary, ticker,
            )
        try:
            from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance
            df = fetch_ohlcv_yfinance(ticker, market_id=market_id, years=years)
        except Exception as exc:
            logger.warning("yfinance fallback also failed for %s: %s", ticker, exc)

    if df.empty:
        logger.warning("No OHLCV data available for %s (market: %s)", ticker, market_id)

    return df

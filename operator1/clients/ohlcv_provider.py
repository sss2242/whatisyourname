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
    "ca_sedar": "tmx",  # Canada: TMX GraphQL (recent ~8 days, merged with yfinance)
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

    elif primary == "tmx":
        # Canada: TMX GraphQL provides ~8 days of recent OHLCV data.
        # We fetch the full yfinance history first, then overlay TMX
        # recent bars for more accurate recent data from the source
        # exchange.  TMX data replaces yfinance for overlapping dates.
        try:
            from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance
            from operator1.clients.ohlcv_tmx import fetch_ohlcv_tmx

            # Get 2-year history from yfinance
            yf_df = fetch_ohlcv_yfinance(ticker, market_id=market_id, years=years)

            # Get recent ~8 days from TMX (minute bars aggregated to daily)
            tmx_df = fetch_ohlcv_tmx(ticker, interval=1)

            if not tmx_df.empty and not yf_df.empty:
                # Normalize date columns for comparison
                yf_df["date"] = pd.to_datetime(yf_df["date"]).dt.normalize()
                tmx_df["date"] = pd.to_datetime(tmx_df["date"]).dt.normalize()

                # Remove yfinance rows that overlap with TMX data
                tmx_dates = set(tmx_df["date"])
                yf_no_overlap = yf_df[~yf_df["date"].isin(tmx_dates)]

                # Combine: yfinance historical + TMX recent
                df = pd.concat([yf_no_overlap, tmx_df], ignore_index=True)
                df = df.sort_values("date").reset_index(drop=True)
                logger.info(
                    "TMX+yfinance merged for %s: %d yf + %d tmx = %d total rows",
                    ticker, len(yf_no_overlap), len(tmx_df), len(df),
                )
            elif not tmx_df.empty:
                df = tmx_df
            elif not yf_df.empty:
                df = yf_df
        except Exception as exc:
            logger.debug("TMX+yfinance merge failed for %s: %s", ticker, exc)

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


# ---------------------------------------------------------------------------
# Benchmark index returns (for beta computation)
# ---------------------------------------------------------------------------

_benchmark_cache: dict[str, pd.Series] = {}


def _load_benchmarks() -> dict[str, str]:
    """Load market_id -> benchmark ticker from config/market_benchmarks.yml."""
    try:
        import yaml
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent.parent / "config" / "market_benchmarks.yml"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def fetch_benchmark_returns(
    market_id: str,
    years: int = 2,
) -> pd.Series:
    """Fetch daily returns for the market benchmark index.

    Uses yfinance to fetch the benchmark index OHLCV, then computes
    daily returns.  Results are cached per market_id to avoid re-fetching.

    Parameters
    ----------
    market_id:
        Market identifier (e.g. "us_sec_edgar").
    years:
        Years of history.

    Returns
    -------
    pd.Series with DatetimeIndex and name ``benchmark_return_1d``.
    Empty Series if benchmark unavailable.
    """
    if market_id in _benchmark_cache:
        return _benchmark_cache[market_id]

    benchmarks = _load_benchmarks()
    ticker = benchmarks.get(market_id)
    if not ticker:
        logger.debug("No benchmark ticker configured for %s", market_id)
        empty = pd.Series(dtype=float, name="benchmark_return_1d")
        _benchmark_cache[market_id] = empty
        return empty

    try:
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance
        df = fetch_ohlcv_yfinance(ticker, market_id="", years=years)
    except Exception as exc:
        logger.warning("Benchmark fetch failed for %s (%s): %s", market_id, ticker, exc)
        empty = pd.Series(dtype=float, name="benchmark_return_1d")
        _benchmark_cache[market_id] = empty
        return empty

    if df.empty or "close" not in df.columns:
        logger.debug("Benchmark OHLCV empty for %s (%s)", market_id, ticker)
        empty = pd.Series(dtype=float, name="benchmark_return_1d")
        _benchmark_cache[market_id] = empty
        return empty

    # Compute daily returns
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

    returns = df["close"].pct_change()
    returns.name = "benchmark_return_1d"

    logger.info(
        "Benchmark returns fetched: %s (%s), %d days",
        market_id, ticker, returns.notna().sum(),
    )

    _benchmark_cache[market_id] = returns
    return returns

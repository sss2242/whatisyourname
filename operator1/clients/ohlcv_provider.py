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


# ---------------------------------------------------------------------------
# D3: Options-implied volatility (forward-looking vol signal)
# ---------------------------------------------------------------------------

_iv_cache: dict[str, pd.Series] = {}


def fetch_implied_volatility(
    ticker: str,
    years: int = 1,
) -> pd.Series:
    """Fetch 30-day at-the-money implied volatility for a ticker.

    Uses yfinance options chain data. The IV-RV spread (implied minus
    realized vol) is the single best predictor of vol regime changes
    (Christensen & Prabhala 1998).

    Parameters
    ----------
    ticker:
        Stock ticker (e.g. "AAPL").
    years:
        Not used for options (only current chain available), but kept
        for API consistency.

    Returns
    -------
    pd.Series with single value (current IV30), or empty if unavailable.
    """
    if ticker in _iv_cache:
        return _iv_cache[ticker]

    empty = pd.Series(dtype=float, name="iv30")

    try:
        import yfinance as yf

        yticker = yf.Ticker(ticker)
        # Get nearest expiry options chain
        expirations = yticker.options
        if not expirations:
            _iv_cache[ticker] = empty
            return empty

        # Pick expiry closest to 30 days
        from datetime import datetime, timedelta
        target_date = datetime.now() + timedelta(days=30)
        _closest_exp = min(
            expirations,
            key=lambda x: abs(datetime.strptime(x, "%Y-%m-%d") - target_date),
        )

        chain = yticker.option_chain(_closest_exp)
        if chain is None or chain.calls is None or chain.calls.empty:
            _iv_cache[ticker] = empty
            return empty

        calls = chain.calls
        # Find ATM call (strike closest to current price)
        _info = yticker.fast_info
        _current_price = getattr(_info, "last_price", None)
        if _current_price is None:
            _iv_cache[ticker] = empty
            return empty

        calls["strike_dist"] = abs(calls["strike"] - _current_price)
        atm = calls.nsmallest(1, "strike_dist")

        if atm.empty or "impliedVolatility" not in atm.columns:
            _iv_cache[ticker] = empty
            return empty

        iv30 = float(atm["impliedVolatility"].iloc[0])
        result = pd.Series([iv30], index=[pd.Timestamp.now().normalize()], name="iv30")

        logger.info("IV30 fetched for %s: %.4f", ticker, iv30)
        _iv_cache[ticker] = result
        return result

    except Exception as exc:
        logger.debug("IV fetch failed for %s: %s", ticker, exc)
        _iv_cache[ticker] = empty
        return empty


# ---------------------------------------------------------------------------
# D4: Cross-asset sector leading indicators
# ---------------------------------------------------------------------------

_SECTOR_LEADERS: dict[str, list[str]] = {
    "technology": ["SMH", "SOXX", "QQQ"],
    "information technology": ["SMH", "SOXX", "QQQ"],
    "semiconductors": ["SMH", "SOXX"],
    "energy": ["XLE", "USO", "OIH"],
    "financials": ["XLF", "KRE", "KBE"],
    "healthcare": ["XLV", "IBB", "XBI"],
    "consumer discretionary": ["XLY", "AMZN"],
    "consumer staples": ["XLP"],
    "industrials": ["XLI"],
    "materials": ["XLB", "GLD"],
    "real estate": ["XLRE", "VNQ"],
    "utilities": ["XLU"],
    "communication services": ["XLC"],
}

_leader_cache: dict[str, pd.DataFrame] = {}


def fetch_sector_leading_indicators(
    sector: str,
    years: int = 2,
) -> pd.DataFrame:
    """Fetch daily returns for sector-relevant ETFs that historically lead.

    Parameters
    ----------
    sector:
        Company sector (e.g. "Technology", "Energy").
    years:
        Years of history to fetch.

    Returns
    -------
    DataFrame with columns = ETF tickers, values = daily returns.
    Empty DataFrame if no leaders configured or fetch fails.
    """
    _sector_key = sector.lower() if sector else ""
    if _sector_key in _leader_cache:
        return _leader_cache[_sector_key]

    etfs = _SECTOR_LEADERS.get(_sector_key, [])
    if not etfs:
        # Try partial match
        for k, v in _SECTOR_LEADERS.items():
            if k in _sector_key or _sector_key in k:
                etfs = v
                break

    if not etfs:
        empty = pd.DataFrame()
        _leader_cache[_sector_key] = empty
        return empty

    try:
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance

        returns_dict: dict[str, pd.Series] = {}
        for etf in etfs[:3]:  # cap at 3 to limit API calls
            try:
                df = fetch_ohlcv_yfinance(etf, market_id="", years=years)
                if not df.empty and "close" in df.columns:
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date").sort_index()
                    ret = df["close"].pct_change()
                    ret.name = f"leader_{etf}_return"
                    returns_dict[etf] = ret
            except Exception:
                continue

        if returns_dict:
            result = pd.DataFrame(returns_dict)
            logger.info(
                "Sector leaders fetched: sector=%s, %d ETFs (%s)",
                sector, len(result.columns), list(result.columns),
            )
            _leader_cache[_sector_key] = result
            return result

    except Exception as exc:
        logger.debug("Sector leader fetch failed: %s", exc)

    empty = pd.DataFrame()
    _leader_cache[_sector_key] = empty
    return empty

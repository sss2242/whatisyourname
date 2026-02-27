"""India OHLCV provider using jugaad-data.

Primary OHLCV source for Indian market (NSE/BSE).
Fallback: yfinance with .NS/.BO suffix.

No API key required. Direct NSE scraping.
Research: .roo/research/ohlcv-per-region-2026-02-25.md
Package: jugaad-data 0.29
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_ohlcv_jugaad(
    ticker: str,
    years: int = 5,
) -> pd.DataFrame:
    """Fetch Indian OHLCV via jugaad-data.

    Parameters
    ----------
    ticker:
        NSE symbol (e.g. "TCS", "RELIANCE").
    years:
        Number of years of history.

    Returns
    -------
    DataFrame with columns: date, open, high, low, close, volume.
    """
    try:
        from jugaad_data.nse import stock_df
    except ImportError:
        logger.debug("jugaad-data not installed; falling back to yfinance")
        return pd.DataFrame()

    try:
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("jugaad-data NSE request timed out (30s)")

        end_dt = date.today()
        start_dt = end_dt - timedelta(days=365 * years)

        # Set 30s timeout -- NSE blocks non-Indian IPs with hanging connections
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(30)
        try:
            # Verbatim from jugaad-data docs
            df = stock_df(
                symbol=ticker.upper(),
                from_date=start_dt,
                to_date=end_dt,
                series="EQ",
            )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        if df is None or df.empty:
            return pd.DataFrame()

        # Normalize columns
        df.columns = [c.lower().strip() for c in df.columns]

        # jugaad-data returns: DATE, OPEN, HIGH, LOW, CLOSE, LTP, VOLUME, etc.
        col_map = {
            "ch_opening_price": "open",
            "ch_trade_high_price": "high",
            "ch_trade_low_price": "low",
            "ch_closing_price": "close",
            "ch_tot_traded_qty": "volume",
        }
        df = df.rename(columns=col_map)

        keep = ["date", "open", "high", "low", "close", "volume"]
        available = [c for c in keep if c in df.columns]
        df = df[available]

        logger.info("jugaad-data fetched %d rows for %s", len(df), ticker)
        return df

    except Exception as exc:
        logger.warning("jugaad-data failed for %s: %s", ticker, exc)
        return pd.DataFrame()

"""Taiwan OHLCV provider using twstock.

Primary OHLCV source for Taiwan market (TWSE/TPEX).
Fallback: yfinance with .TW suffix.

No API key required. Direct TWSE data.
Research: .roo/research/ohlcv-per-region-2026-02-25.md
Package: twstock 1.4.0
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_ohlcv_twstock(
    ticker: str,
    years: int = 5,
) -> pd.DataFrame:
    """Fetch Taiwan OHLCV via twstock.

    Parameters
    ----------
    ticker:
        Taiwan stock code (e.g. "2330" for TSMC).
    years:
        Number of years of history.

    Returns
    -------
    DataFrame with columns: date, open, high, low, close, volume.
    """
    try:
        import twstock
        import twstock.stock as _ts_mod
        from collections import namedtuple
    except ImportError:
        logger.debug("twstock not installed; falling back to yfinance")
        return pd.DataFrame()

    try:
        # Monkey-patch: TWSE now returns an extra field (shares_traded or similar)
        # that twstock 1.4.0's DATATUPLE (9 fields) doesn't expect.
        _old_fields = _ts_mod.DATATUPLE._fields
        if len(_old_fields) == 9:
            _new_tuple = namedtuple("Data", list(_old_fields) + ["extra"])
            _ts_mod.DATATUPLE = _new_tuple
            logger.debug("Patched twstock DATATUPLE to accept 10 fields (TWSE format change)")

        start_year = date.today().year - years
        start_month = date.today().month

        # Verbatim from twstock README
        s = twstock.Stock(ticker)
        s.fetch_from(start_year, start_month)

        if not s.date:
            return pd.DataFrame()

        df = pd.DataFrame({
            "date": s.date,
            "open": s.open,
            "high": s.high,
            "low": s.low,
            "close": s.close,
            "volume": s.capacity,
        })

        logger.info("twstock fetched %d rows for %s", len(df), ticker)
        return df

    except Exception as exc:
        logger.warning("twstock failed for %s: %s", ticker, exc)
        return pd.DataFrame()

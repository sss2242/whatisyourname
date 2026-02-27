"""India OHLCV provider using nselib.

Primary OHLCV source for Indian market (NSE).
Fallback: yfinance with .NS suffix.

No API key required. Uses NSE public data via nselib.
Package: nselib (pip install nselib)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_ohlcv_nselib(
    ticker: str,
    years: int = 5,
) -> pd.DataFrame:
    """Fetch Indian OHLCV via nselib.

    Parameters
    ----------
    ticker:
        NSE symbol (e.g. "RELIANCE", "TCS", "INFY").
    years:
        Number of years of history.

    Returns
    -------
    DataFrame with columns: date, open, high, low, close, volume.
    """
    try:
        from nselib import capital_market
    except ImportError:
        logger.debug("nselib not installed; falling back to yfinance")
        return pd.DataFrame()

    try:
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=365 * years)

        # nselib uses dd-mm-yyyy format
        from_date = start_dt.strftime("%d-%m-%Y")
        to_date = end_dt.strftime("%d-%m-%Y")

        # Fetch price + volume data from NSE
        df = capital_market.price_volume_and_deliverable_position_data(
            symbol=ticker.upper(),
            from_date=from_date,
            to_date=to_date,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Normalize column names (nselib returns mixed-case with BOM artifacts)
        df.columns = [c.strip().strip('\ufeff"').lower() for c in df.columns]

        col_map = {
            "symbol": "symbol",
            "date": "date",
            "openprice": "open",
            "highprice": "high",
            "lowprice": "low",
            "closeprice": "close",
            "tottrdqty": "volume",
            # Alternative column names across nselib versions
            "open price": "open",
            "high price": "high",
            "low price": "low",
            "close price": "close",
            "total traded quantity": "volume",
            "prevclose": "prev_close",
        }
        df = df.rename(columns=col_map)

        # Parse date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

        keep = ["date", "open", "high", "low", "close", "volume"]
        available = [c for c in keep if c in df.columns]
        df = df[available]

        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        logger.info("nselib fetched %d rows for %s", len(df), ticker)
        return df

    except Exception as exc:
        logger.warning("nselib failed for %s: %s", ticker, exc)
        return pd.DataFrame()

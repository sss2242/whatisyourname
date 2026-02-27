"""China OHLCV provider using baostock.

Primary OHLCV source for Chinese market (SSE/SZSE).
Fallback: yfinance with .SS/.SZ suffix.

No API key required. Works globally (no geo-blocking).
Provides richer data than yfinance: turnover rate, percent change, amount.
Package: baostock (pip install baostock)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def _to_baostock_code(ticker: str) -> str:
    """Convert a plain ticker like '600519' to baostock format 'sh.600519'.

    Shanghai (SSE): codes starting with 6 -> sh.XXXXXX
    Shenzhen (SZSE): codes starting with 0 or 3 -> sz.XXXXXX
    """
    ticker = ticker.strip()

    # Already in baostock format
    if ticker.startswith(("sh.", "sz.")):
        return ticker

    # Strip any .SS / .SZ yfinance suffix
    if ticker.upper().endswith(".SS"):
        return f"sh.{ticker[:-3]}"
    if ticker.upper().endswith(".SZ"):
        return f"sz.{ticker[:-3]}"

    # Infer exchange from code prefix
    if ticker.startswith("6"):
        return f"sh.{ticker}"
    elif ticker.startswith(("0", "3")):
        return f"sz.{ticker}"
    else:
        # Default to Shanghai for unknown prefixes
        return f"sh.{ticker}"


def fetch_ohlcv_baostock(
    ticker: str,
    years: int = 5,
) -> pd.DataFrame:
    """Fetch Chinese OHLCV via baostock.

    Parameters
    ----------
    ticker:
        Chinese stock code (e.g. "600519" for Kweichow Moutai,
        "000001" for Ping An Bank). Accepts plain codes, baostock
        format (sh.600519), or yfinance format (600519.SS).
    years:
        Number of years of history.

    Returns
    -------
    DataFrame with columns: date, open, high, low, close, volume.
    Also includes bonus columns when available: amount, turn, pct_chg.
    """
    try:
        import baostock as bs
    except ImportError:
        logger.debug("baostock not installed; falling back to yfinance")
        return pd.DataFrame()

    bs_code = _to_baostock_code(ticker)
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=365 * years)

    try:
        lg = bs.login()
        if lg.error_code != "0":
            logger.warning("baostock login failed: %s", lg.error_msg)
            return pd.DataFrame()

        fields = "date,open,high,low,close,volume,amount,turn,pctChg"
        rs = bs.query_history_k_data_plus(
            bs_code,
            fields,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="3",  # no adjustment (use "2" for forward-adjusted)
        )

        if rs.error_code != "0":
            logger.warning(
                "baostock query failed for %s: %s", bs_code, rs.error_msg
            )
            bs.logout()
            return pd.DataFrame()

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        bs.logout()

        if not rows:
            logger.info("baostock returned 0 rows for %s", bs_code)
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=fields.split(","))

        # Parse date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Convert numeric columns (baostock returns strings)
        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Rename bonus columns to snake_case for consistency
        df = df.rename(columns={"pctChg": "pct_chg"})

        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        logger.info("baostock fetched %d rows for %s", len(df), bs_code)
        return df

    except Exception as exc:
        logger.warning("baostock failed for %s: %s", ticker, exc)
        try:
            bs.logout()
        except Exception:
            pass
        return pd.DataFrame()

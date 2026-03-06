"""Shared yfinance-backed helpers for Phase 2 PIT clients.

Markets without free government filing APIs (SEDAR+, HKEX, SGX, etc.)
use yfinance as a data source for profile and financial statements.

These helpers provide:
- Profile enrichment via yfinance Ticker.info
- Financial statements via yfinance Ticker.income_stmt / balance_sheet / cashflow
- Company search via yfinance search

IMPORTANT: These helpers do NOT provide OHLCV data. OHLCV is handled
exclusively by the OHLCV provider layer (ohlcv_provider.py).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def yf_get_profile(
    ticker: str,
    market_id: str,
    country: str,
    country_code: str,
    exchange: str,
    currency: str,
    yf_suffix: str = "",
) -> dict[str, Any]:
    """Fetch company profile via yfinance Ticker.info.

    Parameters
    ----------
    ticker:
        Raw ticker symbol (without exchange suffix).
    market_id:
        PIT market identifier.
    country, country_code, exchange, currency:
        Market-specific metadata to include in the profile.
    yf_suffix:
        yfinance ticker suffix (e.g. ".TO" for TSX, ".HK" for HKEX).

    Returns
    -------
    Profile dict with canonical fields.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.debug("yfinance not installed; returning skeleton profile")
        return {
            "name": "", "ticker": ticker, "isin": "", "country": country_code,
            "sector": "", "industry": "", "exchange": exchange,
            "currency": currency, "cik": ticker, "market_id": market_id,
        }

    yf_ticker = ticker if ticker.endswith(yf_suffix) else f"{ticker}{yf_suffix}"
    try:
        info = yf.Ticker(yf_ticker).info or {}
        raw = {
            "name": info.get("longName", info.get("shortName", "")),
            "ticker": ticker,
            "isin": info.get("isin", ""),
            "country": country_code,
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "exchange": info.get("exchange", exchange),
            "currency": info.get("currency", currency),
            "cik": ticker,
            "market_id": market_id,
            "market_cap": info.get("marketCap"),
            "employees": info.get("fullTimeEmployees"),
            "website": info.get("website", ""),
        }
        from operator1.clients.canonical_translator import translate_profile
        return translate_profile(raw, market_id)
    except Exception as exc:
        logger.debug("yfinance profile failed for %s: %s", yf_ticker, exc)
        return {
            "name": "", "ticker": ticker, "isin": "", "country": country_code,
            "sector": "", "industry": "", "exchange": exchange,
            "currency": currency, "cik": ticker, "market_id": market_id,
        }


def yf_get_financials(
    ticker: str,
    market_id: str,
    statement_type: str,
    yf_suffix: str = "",
) -> pd.DataFrame:
    """Fetch financial statements via yfinance.

    Parameters
    ----------
    ticker:
        Raw ticker symbol.
    market_id:
        PIT market identifier.
    statement_type:
        One of "income", "balance", "cashflow".
    yf_suffix:
        yfinance ticker suffix.

    Returns
    -------
    DataFrame in canonical long format (canonical_name, value, report_date).
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.debug("yfinance not installed; returning empty financials")
        return pd.DataFrame()

    yf_ticker = ticker if ticker.endswith(yf_suffix) else f"{ticker}{yf_suffix}"
    try:
        t = yf.Ticker(yf_ticker)

        if statement_type == "income":
            df = t.income_stmt
        elif statement_type == "balance":
            df = t.balance_sheet
        elif statement_type == "cashflow":
            df = t.cashflow
        else:
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # yfinance returns columns=dates, index=line items
        # Transpose to: rows=dates, columns=line items
        df = df.T

        # Convert to long format for canonical translation
        records = []
        for report_date, row in df.iterrows():
            for concept, value in row.items():
                if pd.notna(value):
                    records.append({
                        "canonical_name": str(concept),
                        "value": float(value),
                        "report_date": pd.Timestamp(report_date),
                        "filing_date": pd.Timestamp(report_date),
                        "source": "yfinance",
                    })

        if not records:
            return pd.DataFrame()

        long = pd.DataFrame(records)

        # Translate through canonical translator
        try:
            from operator1.clients.canonical_translator import translate_financials
            return translate_financials(long, market_id, statement_type)
        except Exception:
            return long

    except Exception as exc:
        logger.debug("yfinance %s failed for %s: %s", statement_type, yf_ticker, exc)
        return pd.DataFrame()


def yf_search(
    query: str,
    market_id: str,
    country_code: str,
    exchange: str,
    yf_suffix: str = "",
) -> list[dict[str, Any]]:
    """Search for companies via yfinance.

    Parameters
    ----------
    query:
        Company name or ticker to search for.
    market_id:
        PIT market identifier.
    country_code:
        ISO-2 country code for filtering.
    exchange:
        Exchange name for the result metadata.
    yf_suffix:
        yfinance ticker suffix.

    Returns
    -------
    List of company dicts with ticker, name, etc.
    """
    if not query:
        return []

    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        # Try direct ticker lookup first
        yf_ticker = query if query.endswith(yf_suffix) else f"{query}{yf_suffix}"
        t = yf.Ticker(yf_ticker)
        info = t.info or {}
        name = info.get("longName", info.get("shortName", ""))
        if name:
            return [{
                "ticker": query.upper().replace(yf_suffix, ""),
                "name": name,
                "cik": query.upper(),
                "exchange": info.get("exchange", exchange),
                "country": country_code,
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_id": market_id,
            }]
    except Exception:
        pass

    return []

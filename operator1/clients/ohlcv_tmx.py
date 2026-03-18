"""TMX GraphQL data provider for Canadian stocks (TSX / TSXV).

Fetches real-time and near-real-time data from the TMX Money GraphQL
API (app-money.tmx.com/graphql).  No authentication or API key needed.

**Capabilities:**

1. **Rich profile data** (getQuoteBySymbol) -- replaces yfinance for
   Canadian stock profiles with authoritative TSX data:
   - Identity: symbol, name, sector, industry, exchangeName
   - Valuation: MarketCap, eps, peRatio, priceToBook, priceToCashFlow
   - Fundamentals: returnOnAssets, returnOnEquity, totalDebtToEquity, beta
   - Dividends: dividendYield, dividendAmount, exDividendDate, dividendFrequency
   - Volume: averageVolume10D/20D/30D/50D, shareOutStanding
   - Range: weeks52high, weeks52low
   - Description: longDescription, shortDescription

2. **Minute-level OHLCV** (getChartDataBySymbol) -- last ~8 trading
   days of intraday bars (1min, 5min, 15min, 60min, or daily).
   Aggregated to daily bars to supplement the yfinance 2-year spine
   with more accurate recent data.

Usage:
    from operator1.clients.ohlcv_tmx import tmx_enrich_profile, fetch_ohlcv_tmx
    profile = tmx_enrich_profile("RY", existing_profile)
    recent_daily = fetch_ohlcv_tmx("RY", interval=1440)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_TMX_GQL_URL = "https://app-money.tmx.com/graphql"
_TMX_HEADERS = {
    "Content-Type": "application/json",
    "locale": "en",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------------------------
# GraphQL helpers
# ---------------------------------------------------------------------------

def _gql_query(query: str, timeout: int = 15) -> dict | None:
    """Execute a TMX GraphQL query and return the parsed data dict."""
    try:
        resp = requests.post(
            _TMX_GQL_URL,
            json={"query": query},
            headers=_TMX_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "data" in data:
            return data["data"]
        if "errors" in data:
            logger.debug("TMX GraphQL error: %s", data["errors"][0].get("message", ""))
    except Exception as exc:
        logger.debug("TMX GraphQL request failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Profile enrichment
# ---------------------------------------------------------------------------

# Fields available from getQuoteBySymbol (all confirmed working)
_QUOTE_FIELDS = (
    "symbol name price priceChange percentChange openPrice prevClose "
    "volume dayHigh dayLow weeks52high weeks52low "
    "MarketCap eps peRatio beta "
    "dividendYield dividendAmount exDividendDate "
    "dividend3Years dividend5Years dividendFrequency dividendCurrency dividendPayDate "
    "sector industry exchangeName website employees "
    "shareOutStanding totalSharesOutStanding "
    "averageVolume10D averageVolume20D averageVolume30D averageVolume50D "
    "returnOnAssets returnOnEquity totalDebtToEquity "
    "priceToBook priceToCashFlow "
    "longDescription shortDescription qmdescription"
)


def tmx_get_quote(symbol: str) -> dict[str, Any] | None:
    """Fetch the full TMX quote for a Canadian stock.

    Parameters
    ----------
    symbol:
        TSX/TSXV ticker (e.g. 'RY', 'SHOP', 'ENB').

    Returns
    -------
    Dict of all available TMX quote fields, or None on failure.
    """
    query = '{getQuoteBySymbol(symbol:"' + symbol.upper() + '"){' + _QUOTE_FIELDS + '}}'
    data = _gql_query(query)
    if data and data.get("getQuoteBySymbol"):
        return data["getQuoteBySymbol"]
    return None


def tmx_enrich_profile(
    symbol: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Enrich a Canadian stock profile with TMX GraphQL data.

    Fills in missing fields from the TMX quote API.  TMX data is
    preferred over yfinance for Canadian stocks because it comes
    directly from the Toronto Stock Exchange.

    Fields enriched:
    - sector, industry, market_cap, shares_outstanding, eps, pe_ratio
    - beta, dividend_yield, dividend_amount, ex_dividend_date
    - return_on_assets, return_on_equity, total_debt_to_equity
    - price_to_book, price_to_cash_flow
    - weeks_52_high, weeks_52_low, average_volume
    - description, website, employees

    Parameters
    ----------
    symbol:
        TSX/TSXV ticker symbol.
    profile:
        Existing profile dict to enrich (modified in place and returned).

    Returns
    -------
    The enriched profile dict.
    """
    quote = tmx_get_quote(symbol)
    if not quote:
        logger.debug("TMX quote not available for %s", symbol)
        return profile

    # Map TMX field names to our canonical profile field names
    enrichment_map: dict[str, str] = {
        "name": "name",
        "sector": "sector",
        "industry": "industry",
        "exchangeName": "exchange",
        "website": "website",
        "employees": "employees",
        "MarketCap": "market_cap",
        "shareOutStanding": "shares_outstanding",
        "eps": "eps",
        "peRatio": "pe_ratio",
        "beta": "beta",
        "dividendYield": "dividend_yield",
        "dividendAmount": "dividend_amount",
        "exDividendDate": "ex_dividend_date",
        "dividendFrequency": "dividend_frequency",
        "returnOnAssets": "return_on_assets",
        "returnOnEquity": "return_on_equity",
        "totalDebtToEquity": "total_debt_to_equity",
        "priceToBook": "price_to_book",
        "priceToCashFlow": "price_to_cash_flow",
        "weeks52high": "weeks_52_high",
        "weeks52low": "weeks_52_low",
        "averageVolume30D": "average_volume",
        "longDescription": "description",
        "shortDescription": "short_description",
        "qmdescription": "sub_industry",
    }

    # Price snapshot fields (always overwrite with latest TMX data)
    price_fields: dict[str, str] = {
        "price": "latest_price",
        "priceChange": "price_change",
        "percentChange": "percent_change",
        "openPrice": "open_price",
        "prevClose": "prev_close",
        "dayHigh": "day_high",
        "dayLow": "day_low",
        "volume": "latest_volume",
    }

    # Enrich: only fill fields that are missing or empty in the profile
    for tmx_key, profile_key in enrichment_map.items():
        tmx_val = quote.get(tmx_key)
        if tmx_val is not None and tmx_val != "":
            existing = profile.get(profile_key)
            if not existing or existing == "" or existing is None:
                profile[profile_key] = tmx_val

    # Price snapshot: always overwrite (TMX is more current than cached data)
    for tmx_key, profile_key in price_fields.items():
        tmx_val = quote.get(tmx_key)
        if tmx_val is not None:
            profile[profile_key] = tmx_val

    # Volume averages (store all available)
    for period in ("10D", "20D", "30D", "50D"):
        key = f"averageVolume{period}"
        val = quote.get(key)
        if val is not None:
            profile[f"average_volume_{period.lower()}"] = val

    profile["_tmx_enriched"] = True
    logger.info(
        "TMX enriched profile for %s: MarketCap=%s, PE=%s, sector=%s",
        symbol,
        quote.get("MarketCap", "?"),
        quote.get("peRatio", "?"),
        quote.get("sector", "?"),
    )
    return profile


# ---------------------------------------------------------------------------
# OHLCV data (minute-level, last ~8 trading days)
# ---------------------------------------------------------------------------

def fetch_ohlcv_tmx(
    symbol: str,
    interval: int = 1440,
) -> pd.DataFrame:
    """Fetch recent OHLCV data from TMX GraphQL.

    Returns up to ~8 trading days of intraday bars.  The default
    interval of 1440 (minutes per day) returns daily bars.

    Parameters
    ----------
    symbol:
        TSX/TSXV ticker (e.g. 'RY', 'SHOP').
    interval:
        Bar interval in minutes: 1, 5, 15, 60, or 1440 (daily).

    Returns
    -------
    DataFrame with columns: date, open, high, low, close, volume.
    Empty DataFrame on failure.
    """
    query = (
        '{getChartDataBySymbol(symbol:"' + symbol.upper() + '"'
        ',interval:' + str(interval) + ')'
        '{open high low close volume dateTime}}'
    )
    data = _gql_query(query)
    if not data or not data.get("getChartDataBySymbol"):
        return pd.DataFrame()

    bars = data["getChartDataBySymbol"]
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["dateTime"], utc=True).dt.tz_convert(None)

    if interval < 1440:
        # Aggregate intraday bars to daily OHLCV
        df["trade_date"] = df["date"].dt.date
        daily = df.groupby("trade_date").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).reset_index()
        daily = daily.rename(columns={"trade_date": "date"})
        daily["date"] = pd.to_datetime(daily["date"])
    else:
        daily = df[["date", "open", "high", "low", "close", "volume"]].copy()

    daily = daily.sort_values("date").reset_index(drop=True)
    logger.info("TMX OHLCV for %s: %d daily bars (interval=%d)", symbol, len(daily), interval)
    return daily

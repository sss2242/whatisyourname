"""India BSE/NSE PIT client -- uses yfinance + BSE API.

Primary: yfinance (profile, financials via .NS suffix -- works globally)
Fallback: Direct BSE India API (https://api.bseindia.com/BseIndiaAPI/api)

OHLCV: handled separately via ohlcv_nselib.py + ohlcv_provider.py

Note: jugaad-data was removed because it depends on NSE endpoints that
geo-block non-Indian IPs and frequently change their anti-scraping measures.

Coverage: ~5,500+ listed companies on BSE/NSE, ~$4T market cap.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_BSE_BASE = "https://api.bseindia.com/BseIndiaAPI/api"
_CACHE_DIR = Path("cache/in_bse")


class INBseClient:
    """PIT client for Indian BSE/NSE equities using yfinance + BSE API.

    Implements the ``PITClient`` protocol. Uses yfinance as primary
    data source for profile, financial data. BSE API for company search.
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Operator1/1.0",
            "Referer": "https://www.bseindia.com/",
        }

    def _cache_path(self, identifier: str, fn: str) -> Path:
        return self._cache_dir / identifier.upper() / fn

    def _read_cache(self, identifier: str, fn: str) -> dict | None:
        p = self._cache_path(identifier, fn)
        if not p.exists():
            return None
        try:
            if fn == "profile.json" and (date.today() - date.fromtimestamp(p.stat().st_mtime)).days > 7:
                return None
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, fn: str, data: dict) -> None:
        p = self._cache_path(identifier, fn)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    @property
    def market_id(self) -> str:
        return "in_bse"

    @property
    def market_name(self) -> str:
        return "India (BSE / NSE) -- yfinance"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Indian companies from BSE API search."""
        if not query:
            return []
        try:
            data = cached_get(
                f"{_BSE_BASE}/Suggest/Getstockdata/{query}",
                headers=self._headers,
            )
            items = data if isinstance(data, list) else []
            return [
                {
                    "ticker": i.get("scrip_cd", ""),
                    "name": i.get("scripname", ""),
                    "cik": i.get("scrip_cd", ""),
                    "exchange": "BSE",
                    "country": "IN",
                    "market_id": self.market_id,
                }
                for i in items
            ]
        except Exception:
            return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        raw: dict[str, Any] = {
            "name": "",
            "ticker": identifier,
            "isin": "",
            "country": "IN",
            "sector": "",
            "industry": "",
            "sub_industry": "",
            "exchange": "BSE",
            "currency": "INR",
            "cik": identifier,
            "market_cap": "",
            "shares_outstanding": "",
            "lei": "",
        }

        # Primary: yfinance for rich profile data
        self._enrich_from_yfinance(identifier, raw)

        # Fallback: BSE API for basic info
        if not raw.get("name"):
            self._enrich_from_bse_api(identifier, raw)

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def _enrich_from_yfinance(self, identifier: str, raw: dict) -> None:
        """Enrich profile with yfinance data (.NS suffix for NSE)."""
        try:
            import yfinance as yf

            # Try NSE (.NS) first, then BSE (.BO)
            for suffix in [".NS", ".BO"]:
                yf_ticker = f"{identifier}{suffix}"
                t = yf.Ticker(yf_ticker)
                info = t.info or {}

                if info.get("longName") or info.get("shortName"):
                    raw["name"] = info.get("longName") or info.get("shortName") or ""
                    raw["sector"] = info.get("sector", "")
                    raw["industry"] = info.get("industry", "")
                    mc = info.get("marketCap")
                    if mc:
                        raw["market_cap"] = str(mc)
                    shares = info.get("sharesOutstanding")
                    if shares:
                        raw["shares_outstanding"] = str(shares)
                    raw["isin"] = info.get("isin", "")
                    raw["exchange"] = info.get("exchange", "NSI")
                    logger.info("yfinance enriched %s via %s: %s", identifier, yf_ticker, raw["name"])
                    break

        except Exception as exc:
            logger.debug("yfinance profile failed for %s: %s", identifier, exc)

    def _enrich_from_bse_api(self, identifier: str, raw: dict) -> None:
        """Fallback: BSE API for basic profile."""
        try:
            data = cached_get(
                f"{_BSE_BASE}/StockReachGraph/stockData/{identifier}",
                headers=self._headers,
            )
            if isinstance(data, dict):
                raw["name"] = data.get("comp_name", "")
                raw["isin"] = data.get("isin_code", "")
                raw["sector"] = data.get("sector_name", "")
                raw["industry"] = data.get("industry", "")
        except Exception as exc:
            logger.debug("BSE profile failed for %s: %s", identifier, exc)

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch income statements via yfinance."""
        return self._fetch_financials_yf(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        """Fetch balance sheets via yfinance."""
        return self._fetch_financials_yf(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch cash flow statements via yfinance."""
        return self._fetch_financials_yf(identifier, "cashflow")

    def _fetch_financials_yf(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial statements via yfinance Ticker object."""
        try:
            import yfinance as yf
            t = yf.Ticker(f"{identifier}.NS")

            fetch_map = {
                "income": t.financials,
                "balance": t.balance_sheet,
                "cashflow": t.cashflow,
            }

            df = fetch_map.get(statement_type)
            if df is None or df.empty:
                return pd.DataFrame()

            # yfinance returns columns as dates, rows as concepts
            # Transpose to get: rows = periods, columns = concepts
            df = df.T
            df = df.reset_index()
            df = df.rename(columns={"index": "report_date"})
            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
            df["filing_date"] = df["report_date"] + pd.Timedelta(days=60)

            # Melt to long format for the translator
            id_cols = ["report_date", "filing_date"]
            value_cols = [c for c in df.columns if c not in id_cols]
            long = df.melt(id_vars=id_cols, value_vars=value_cols,
                           var_name="concept", value_name="value")

            from operator1.clients.canonical_translator import translate_financials
            return translate_financials(long, self.market_id, statement_type)

        except Exception as exc:
            logger.debug("yfinance financials failed for %s/%s: %s", identifier, statement_type, exc)
            return pd.DataFrame()

    # -- Price data -----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """BSE does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    # -- Peers / related entities --------------------------------------------

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

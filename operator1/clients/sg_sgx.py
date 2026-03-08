"""Singapore SGX PIT client -- SGX API + yfinance fallback.

Primary: SGX public APIs
  - Company search: https://api.sgx.com/companyinfo/v2.0
  - Announcements: https://api.sgx.com/announcements/v1.0
  - Preserves filing_date from SGX announcement date (true PIT)

Fallback: yfinance (profile, financials via .SI suffix)

OHLCV: handled separately via ohlcv_provider.py (yfinance .SI)

Coverage: ~700+ listed companies on SGX, ~$0.6T market cap.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)
_SGX_BASE = "https://api.sgx.com"
_CACHE_DIR = Path("cache/sg_sgx")
_SGX_HEADERS = {
    "User-Agent": "Operator1/1.0",
    "Accept": "application/json",
}


class SGSgxClient:
    """PIT client for Singapore SGX equities.

    Uses SGX public APIs for company search and announcement discovery.
    Financial statements come from either SGX announcements (via LLM
    extraction of PDF filings) or yfinance as fallback.
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)

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
        return "sg_sgx"

    @property
    def market_name(self) -> str:
        return "Singapore (SGX)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Search SGX companies via SGX API, yfinance fallback."""
        if not query:
            return []

        # Try SGX company info API first
        try:
            data = cached_get(
                f"{_SGX_BASE}/companyinfo/v2.0",
                params={"keyword": query, "pagesize": "20", "pagestart": "0"},
                headers=_SGX_HEADERS,
            )
            items = []
            if isinstance(data, dict):
                items = data.get("data", data.get("result", []))
            elif isinstance(data, list):
                items = data

            if items:
                results = []
                for item in items:
                    ticker = item.get("stockCode", item.get("code", ""))
                    name = item.get("companyName", item.get("name", ""))
                    if ticker or name:
                        results.append({
                            "ticker": ticker,
                            "name": name,
                            "cik": ticker,
                            "exchange": "SGX",
                            "country": "SG",
                            "market_id": self.market_id,
                        })
                if results:
                    return results
        except Exception as exc:
            logger.debug("SGX company search failed: %s", exc)

        # Fallback to yfinance
        from operator1.clients.yfinance_backed import yf_search
        return yf_search(query, self.market_id, "SG", "SGX", yf_suffix=".SI")

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        # Try SGX company info API
        try:
            data = cached_get(
                f"{_SGX_BASE}/companyinfo/v2.0",
                params={"keyword": identifier, "pagesize": "1", "pagestart": "0"},
                headers=_SGX_HEADERS,
            )
            items = []
            if isinstance(data, dict):
                items = data.get("data", data.get("result", []))
            elif isinstance(data, list):
                items = data

            if items:
                item = items[0]
                raw = {
                    "name": item.get("companyName", item.get("name", "")),
                    "ticker": identifier,
                    "isin": item.get("isin", ""),
                    "country": "SG",
                    "sector": item.get("sectorName", item.get("sector", "")),
                    "industry": item.get("industryName", item.get("industry", "")),
                    "exchange": "SGX",
                    "currency": "SGD",
                    "cik": identifier,
                    "market_id": self.market_id,
                }
                if raw.get("name"):
                    from operator1.clients.canonical_translator import translate_profile
                    profile = translate_profile(raw, self.market_id)
                    self._write_cache(identifier, "profile.json", profile)
                    return profile
        except Exception as exc:
            logger.debug("SGX profile API failed for %s: %s", identifier, exc)

        # Fallback to yfinance
        from operator1.clients.yfinance_backed import yf_get_profile
        profile = yf_get_profile(identifier, self.market_id, "Singapore", "SG", "SGX", "SGD", yf_suffix=".SI")
        self._write_cache(identifier, "profile.json", profile)
        return profile

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch income statements via SGX announcements, yfinance fallback."""
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        """Fetch balance sheets via SGX announcements, yfinance fallback."""
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch cash flow statements via SGX announcements, yfinance fallback."""
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Try SGX announcement-based filing discovery first, yfinance fallback.

        SGX announcements provide true PIT data (announcement date is
        the filing date). yfinance sets filing_date = report_date.
        """
        # Path 1: SGX filing discovery + LLM extraction
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
            df = try_filing_extraction(
                ticker=identifier,
                market_id=self.market_id,
                statement_type=statement_type,
                llm_client=None,
            )
            if df is not None and not df.empty:
                logger.info("SGX %s %s: %d rows from filing discovery",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("SGX filing discovery failed for %s: %s", identifier, exc)

        # Path 2: yfinance fallback
        from operator1.clients.yfinance_backed import yf_get_financials
        logger.debug("SGX %s: falling back to yfinance for %s", identifier, statement_type)
        return yf_get_financials(identifier, self.market_id, statement_type, yf_suffix=".SI")

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """SGX does not provide free OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

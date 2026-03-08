"""Saudi Arabia Tadawul PIT client -- Tadawul API + yfinance fallback.

Primary: Saudi Exchange (Tadawul) public APIs
  - Company directory: https://www.saudiexchange.sa/
  - Financial disclosures with announcement dates (true PIT)

Fallback: yfinance (profile, financials via .SR suffix)

OHLCV: handled separately via ohlcv_provider.py (yfinance .SR)

Coverage: ~200+ listed companies on Tadawul, ~$2.7T market cap.
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
_TADAWUL_BASE = "https://www.saudiexchange.sa"
_CACHE_DIR = Path("cache/sa_tadawul")
_TADAWUL_HEADERS = {
    "User-Agent": "Operator1/1.0",
    "Accept": "application/json",
}


class SATadawulClient:
    """PIT client for Saudi Tadawul equities.

    Attempts to fetch company data from Tadawul's public endpoints.
    Falls back to yfinance for profile and financial statements when
    the Tadawul API is unavailable.
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
        return "sa_tadawul"

    @property
    def market_name(self) -> str:
        return "Saudi Arabia (Tadawul)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Search Tadawul companies, yfinance fallback."""
        if not query:
            return []

        # Try Tadawul company search
        try:
            data = cached_get(
                f"{_TADAWUL_BASE}/wps/portal/tadawul/market-participants/issuers/issuers-directory",
                params={"searchText": query},
                headers=_TADAWUL_HEADERS,
            )
            if isinstance(data, dict):
                items = data.get("data", data.get("result", []))
                if items:
                    results = []
                    for item in items:
                        ticker = item.get("symbol", item.get("code", ""))
                        name = item.get("companyName", item.get("name", ""))
                        if ticker or name:
                            results.append({
                                "ticker": ticker,
                                "name": name,
                                "cik": ticker,
                                "exchange": "Tadawul",
                                "country": "SA",
                                "market_id": self.market_id,
                            })
                    if results:
                        return results
        except Exception as exc:
            logger.debug("Tadawul company search failed: %s", exc)

        from operator1.clients.yfinance_backed import yf_search
        return yf_search(query, self.market_id, "SA", "Tadawul", yf_suffix=".SR")

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached
        from operator1.clients.yfinance_backed import yf_get_profile
        profile = yf_get_profile(identifier, self.market_id, "Saudi Arabia", "SA", "Tadawul", "SAR", yf_suffix=".SR")
        self._write_cache(identifier, "profile.json", profile)
        return profile

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financials via Tadawul filing discovery only (PIT-compliant).

        yfinance is NOT used for financial statements because it does not
        provide true filing dates (sets filing_date = report_date).
        """
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
            df = try_filing_extraction(
                ticker=identifier,
                market_id=self.market_id,
                statement_type=statement_type,
                llm_client=None,
            )
            if df is not None and not df.empty:
                logger.info("Tadawul %s %s: %d rows from filing discovery",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("Tadawul filing discovery failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

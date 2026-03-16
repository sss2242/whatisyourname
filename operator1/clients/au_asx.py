"""Australia ASX PIT client -- uses ASX MarkitDigital API + filing discovery.

Primary profile: ASX MarkitDigital API (asx.api.markitdigital.com)
Primary financials: Filing discovery + LLM extraction from ASX PDFs
OHLCV: Handled by ohlcv_provider.py (yfinance .AX suffix)

Coverage: ~2,200+ listed companies, ~$1.8T market cap.

NOTE: The old ASX API (asx.com.au/asx/1/) returns 404 as of 2026.
The MarkitDigital API is the current ASX data provider.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ASX MarkitDigital API (same API used by the ASX website and filing discoverer)
_MARKIT_BASE = "https://asx.api.markitdigital.com/asx-research/1.0"
_MARKIT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Operator1/1.0",
}

_CACHE_DIR = Path("cache/au_asx")


class AUAsxClient:
    """PIT client for Australian ASX equities.

    Uses the ASX MarkitDigital API for profile and company search.
    Uses filing discovery + LLM extraction for financial statements.
    No yfinance dependency -- all data comes from ASX-native sources.
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
        return "au_asx"

    @property
    def market_name(self) -> str:
        return "Australia (ASX)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Search companies via ASX MarkitDigital directory API."""
        try:
            url = f"{_MARKIT_BASE}/companies/directory"
            resp = requests.get(url, headers=_MARKIT_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", []) if isinstance(data, dict) else []
            companies = [
                {
                    "ticker": i.get("symbol", ""),
                    "name": i.get("displayName", i.get("name", "")),
                    "cik": i.get("symbol", ""),
                    "exchange": "ASX",
                    "country": "AU",
                    "market_id": self.market_id,
                }
                for i in items
            ]
            if query:
                q = query.lower()
                companies = [
                    c for c in companies
                    if q in c["ticker"].lower() or q in c["name"].lower()
                ]
            return companies
        except Exception as exc:
            logger.debug("ASX company list failed: %s", exc)
            return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from ASX MarkitDigital header API.

        Returns sector, industry, market cap, and listing date directly
        from the exchange -- no yfinance dependency.
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        raw: dict[str, Any] = {
            "name": "",
            "ticker": identifier.upper(),
            "isin": "",
            "country": "AU",
            "sector": "",
            "industry": "",
            "exchange": "ASX",
            "currency": "AUD",
            "cik": identifier,
        }

        # Primary: ASX MarkitDigital header endpoint
        try:
            url = f"{_MARKIT_BASE}/companies/{identifier.upper()}/header"
            resp = requests.get(url, headers=_MARKIT_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            raw["name"] = data.get("displayName", "")
            raw["sector"] = data.get("sector", data.get("industryGroup", ""))
            raw["industry"] = data.get("industryGroup", "")
            if data.get("marketCap"):
                raw["market_cap"] = data["marketCap"]
            if data.get("dateListed"):
                raw["date_listed"] = data["dateListed"]
            logger.info(
                "ASX profile for %s: %s (sector=%s)",
                identifier, raw["name"], raw["sector"],
            )
        except Exception as exc:
            logger.info("ASX MarkitDigital profile failed for %s: %s", identifier, exc)

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financials via ASX filing discovery only (PIT-compliant).

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
                logger.info("ASX %s %s: %d rows from filing discovery",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("ASX filing discovery failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """ASX does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()
    def get_peers(self, identifier: str) -> list[str]: return []
    def get_executives(self, identifier: str) -> list[dict[str, Any]]: return []

"""Australia ASX PIT client -- uses ASX undocumented API.

Primary: ASX API (https://www.asx.com.au/asx/1/share/)
Coverage: ~2,200+ listed companies, ~$1.8T market cap.
"""

from __future__ import annotations

import json, logging, os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)
_ASX_BASE = "https://www.asx.com.au/asx/1"
_CACHE_DIR = Path("cache/au_asx")


class AUAsxClient:
    """PIT client for Australian ASX equities."""

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._headers = {"Accept": "application/json", "User-Agent": "Operator1/1.0"}

    def _cache_path(self, identifier: str, fn: str) -> Path:
        return self._cache_dir / identifier.upper() / fn

    def _read_cache(self, identifier: str, fn: str) -> dict | None:
        p = self._cache_path(identifier, fn)
        if not p.exists(): return None
        try:
            if fn == "profile.json" and (date.today() - date.fromtimestamp(p.stat().st_mtime)).days > 7: return None
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return None

    def _write_cache(self, identifier: str, fn: str, data: dict) -> None:
        p = self._cache_path(identifier, fn)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    @property
    def market_id(self) -> str: return "au_asx"
    @property
    def market_name(self) -> str: return "Australia (ASX) -- ASX API"

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        try:
            data = cached_get(f"{_ASX_BASE}/share/list", headers=self._headers)
            items = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
            companies = [{"ticker": i.get("code", ""), "name": i.get("name", i.get("display_name", "")),
                         "cik": i.get("code", ""), "exchange": "ASX", "country": "AU",
                         "market_id": self.market_id} for i in items]
            if query:
                q = query.lower()
                companies = [c for c in companies if q in c["ticker"].lower() or q in c["name"].lower()]
            return companies
        except Exception: return []

    def search_company(self, name: str) -> list[dict[str, Any]]: return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached: return cached
        try:
            data = cached_get(f"{_ASX_BASE}/share/{identifier}", headers=self._headers)
            raw = {"name": data.get("name", data.get("display_name", "")),
                   "ticker": identifier.upper(), "isin": "", "country": "AU",
                   "sector": data.get("industry_group_name", ""),
                   "industry": data.get("industry_group_name", ""),
                   "exchange": "ASX", "currency": "AUD", "cik": identifier}
        except Exception:
            raw = {"name": "", "ticker": identifier, "isin": "", "country": "AU",
                   "sector": "", "industry": "", "exchange": "ASX", "currency": "AUD", "cik": identifier}
        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        from operator1.clients.yfinance_backed import yf_get_financials
        return yf_get_financials(identifier, self.market_id, "income", yf_suffix=".AX")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        from operator1.clients.yfinance_backed import yf_get_financials
        return yf_get_financials(identifier, self.market_id, "balance", yf_suffix=".AX")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        from operator1.clients.yfinance_backed import yf_get_financials
        return yf_get_financials(identifier, self.market_id, "cashflow", yf_suffix=".AX")

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """ASX does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()
    def get_peers(self, identifier: str) -> list[str]: return []
    def get_executives(self, identifier: str) -> list[dict[str, Any]]: return []

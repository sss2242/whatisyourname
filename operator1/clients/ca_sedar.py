"""Canada SEDAR+ PIT client -- scraper-based.

Fetches Canadian company filings from SEDAR+ (the successor to SEDAR),
operated by the Canadian Securities Administrators (CSA).

Primary: SEDAR+ API (https://www.sedarplus.ca)
Coverage: ~4,000+ Canadian public companies, TSX/TSXV.
"""

from __future__ import annotations

import json, logging, os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)
_SEDAR_BASE = "https://www.sedarplus.ca/csa-party"
_CACHE_DIR = Path("cache/ca_sedar")


class CASedarClient:
    """PIT client for Canadian SEDAR+ filings."""

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
    def market_id(self) -> str: return "ca_sedar"
    @property
    def market_name(self) -> str: return "Canada (TSX / TSXV) -- SEDAR+"

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        if not query: return []
        try:
            data = cached_get(f"{_SEDAR_BASE}/searchCompany", params={"searchText": query}, headers=self._headers)
            items = data if isinstance(data, list) else data.get("results", []) if isinstance(data, dict) else []
            return [{"ticker": i.get("symbol", ""), "name": i.get("name", ""), "cik": i.get("sedarId", ""),
                     "exchange": "TSX", "country": "CA", "market_id": self.market_id} for i in items]
        except Exception: return []

    def search_company(self, name: str) -> list[dict[str, Any]]: return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached: return cached
        matches = self.search_company(identifier)
        m = matches[0] if matches else {"name": "", "ticker": identifier}
        raw = {"name": m.get("name", ""), "ticker": m.get("ticker", identifier), "isin": "", "country": "CA",
               "sector": "", "industry": "", "exchange": "TSX", "currency": "CAD", "cik": m.get("cik", "")}
        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        from operator1.clients.yfinance_backed import yf_get_financials
        return yf_get_financials(identifier, self.market_id, "income", yf_suffix=".TO")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        from operator1.clients.yfinance_backed import yf_get_financials
        return yf_get_financials(identifier, self.market_id, "balance", yf_suffix=".TO")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        from operator1.clients.yfinance_backed import yf_get_financials
        return yf_get_financials(identifier, self.market_id, "cashflow", yf_suffix=".TO")

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """SEDAR+ does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()
    def get_peers(self, identifier: str) -> list[str]: return []
    def get_executives(self, identifier: str) -> list[dict[str, Any]]: return []

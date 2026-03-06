"""Hong Kong HKEX PIT client -- uses HKEX API.

Primary: HKEX API (https://www.hkexnews.hk)
Coverage: ~2,500+ listed companies, ~$4.5T market cap.
"""
from __future__ import annotations
import json, logging
from datetime import date
from pathlib import Path
from typing import Any
import pandas as pd
from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)
_HKEX_BASE = "https://www.hkexnews.hk"
_CACHE_DIR = Path("cache/hk_hkex")

class HKHkexClient:
    """PIT client for Hong Kong HKEX equities."""
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
    def market_id(self) -> str: return "hk_hkex"
    @property
    def market_name(self) -> str: return "Hong Kong (HKEX)"

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        from operator1.clients.yfinance_backed import yf_search
        return yf_search(query, self.market_id, "HK", "HKEX", yf_suffix=".HK")

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached: return cached
        from operator1.clients.yfinance_backed import yf_get_profile
        profile = yf_get_profile(identifier, self.market_id, "Hong Kong", "HK", "HKEX", "HKD", yf_suffix=".HK")
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        from operator1.clients.yfinance_backed import yf_get_financials
        return yf_get_financials(identifier, self.market_id, "income", yf_suffix=".HK")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        from operator1.clients.yfinance_backed import yf_get_financials
        return yf_get_financials(identifier, self.market_id, "balance", yf_suffix=".HK")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        from operator1.clients.yfinance_backed import yf_get_financials
        return yf_get_financials(identifier, self.market_id, "cashflow", yf_suffix=".HK")

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """HKEX does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()
    def get_peers(self, identifier: str) -> list[str]: return []
    def get_executives(self, identifier: str) -> list[dict[str, Any]]: return []

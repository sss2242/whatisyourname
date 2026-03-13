"""UAE DFM/ADX PIT client.
Coverage: ~100+ listed companies on DFM/ADX, ~$0.8T market cap.
"""
from __future__ import annotations
import json, logging
from datetime import date
from pathlib import Path
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)
_CACHE_DIR = Path("cache/ae_dfm")

class AEDfmClient:
    """PIT client for UAE DFM/ADX equities."""
    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
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
    def market_id(self) -> str: return "ae_dfm"
    @property
    def market_name(self) -> str: return "UAE (DFM / ADX)"

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        from operator1.clients.yfinance_backed import yf_search
        return yf_search(query, self.market_id, "AE", "DFM", yf_suffix=".AE")

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached: return cached
        from operator1.clients.yfinance_backed import yf_get_profile
        profile = yf_get_profile(identifier, self.market_id, "UAE", "AE", "DFM", "AED", yf_suffix=".AE")
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financials via DFM/ADX filing discovery only (PIT-compliant).

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
                logger.info("DFM %s %s: %d rows from filing discovery",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("DFM filing discovery failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """DFM does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()
    def get_peers(self, identifier: str) -> list[str]: return []
    def get_executives(self, identifier: str) -> list[dict[str, Any]]: return []

"""Singapore SGX PIT client -- SGX securities + financial reports APIs.

Primary profile: SGX Securities API (api.sgx.com/securities/v1.1)
  - Returns ~561 listed stocks with name, ticker code, currency, board
  - Works globally without authentication
  - Used for company search and basic profile

Primary financials: SGX Financial Reports API (api.sgx.com/financialreports/v1.0)
  - 12,674 financial reports across 1,369 companies
  - Filter by company name (companyname param)
  - Returns filing metadata with document URLs
  - Document pages contain PDF download links
  - Works globally without authentication

Fallback profile: yfinance (.SI suffix)

OHLCV: handled separately via ohlcv_provider.py (yfinance .SI)

Coverage: ~700+ listed companies on SGX, ~$0.6T market cap.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from operator1.clients.canonical_translator import translate_profile

logger = logging.getLogger(__name__)

_SGX_BASE = "https://api.sgx.com"
_SGX_LINKS_BASE = "https://links.sgx.com"
_CACHE_DIR = Path("cache/sg_sgx")

_SGX_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Module-level cache for the SGX securities directory.
_securities_cache: list[dict[str, Any]] | None = None
_securities_cache_fetched_at: float = 0.0
_securities_cache_ttl_hours: int = 24


def _get_securities_directory() -> list[dict[str, Any]]:
    """Fetch and cache the full SGX stock directory.

    Uses the SGX Securities API which works globally without auth.
    Returns ~561 listed stocks with name, ticker code, currency, board.
    Cached in-memory for 24 hours.
    """
    global _securities_cache, _securities_cache_fetched_at

    now = time.time()
    if (
        _securities_cache is not None
        and (now - _securities_cache_fetched_at) < _securities_cache_ttl_hours * 3600
    ):
        return _securities_cache

    try:
        resp = requests.get(
            f"{_SGX_BASE}/securities/v1.1",
            params={"type": "stocks", "pagestart": "0", "pagesize": "2000"},
            headers=_SGX_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        prices = data.get("data", {}).get("prices", [])

        # Filter to actual stocks with names
        stocks = [p for p in prices if p.get("type") == "stocks" and p.get("n")]

        if len(stocks) > 50:
            _securities_cache = stocks
            _securities_cache_fetched_at = now
            logger.info("SGX securities directory loaded: %d stocks", len(stocks))
            return stocks
    except Exception as exc:
        logger.warning("Failed to fetch SGX securities directory: %s", exc)

    return _securities_cache or []


def _search_securities(
    query: str,
    directory: list[dict[str, Any]],
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Search the SGX securities directory by ticker code or name.

    Field mapping from the SGX API:
      n    = company name
      nc   = ticker code (e.g. D05 for DBS)
      cur  = currency (SGD)
      m    = board (MAINBOARD, CATALIST)
      issuer-name = short issuer name
    """
    if not query or not directory:
        return []

    q_lower = query.strip().lower()
    q_upper = query.strip().upper()

    # Phase 1: Exact ticker code match
    exact = [d for d in directory if d.get("nc", "").upper() == q_upper]
    if exact:
        return exact[:max_results]

    # Phase 2: Name/issuer substring match
    name_matches = []
    code_matches = []
    for d in directory:
        name = (d.get("n") or "").lower()
        issuer = (d.get("issuer-name") or "").lower()
        code = (d.get("nc") or "").lower()
        if q_lower in name or q_lower in issuer:
            name_matches.append(d)
        elif q_lower in code:
            code_matches.append(d)

    combined = name_matches + code_matches
    return combined[:max_results]


def _securities_to_company_list(
    items: list[dict[str, Any]],
    market_id: str = "sg_sgx",
) -> list[dict[str, Any]]:
    """Convert raw SGX securities API items to standard company dicts."""
    return [
        {
            "ticker": item.get("nc", ""),
            "name": item.get("n", ""),
            "cik": item.get("nc", ""),
            "exchange": "SGX",
            "country": "SG",
            "market_id": market_id,
            "board": item.get("m", ""),
            "currency": item.get("cur", "SGD"),
        }
        for item in items
        if item.get("nc")
    ]


class SGSgxClient:
    """PIT client for Singapore SGX equities.

    Uses the SGX Securities API for company search and profile.
    Uses the SGX Financial Reports API for filing discovery.
    Financial statement extraction uses filing discovery + LLM.
    No yfinance dependency for search or profile.
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
        """Search SGX companies via SGX Securities API.

        The securities API returns ~561 listed stocks. When a query is
        provided, the full directory is fetched (cached 24h) and filtered
        client-side by ticker code or company name.

        Returns an empty list if no query is provided (no browse-all mode).
        """
        if not query:
            return []

        directory = _get_securities_directory()
        if directory:
            matches = _search_securities(query, directory)
            if matches:
                return _securities_to_company_list(matches, self.market_id)

        # Fallback to yfinance search
        try:
            from operator1.clients.yfinance_backed import yf_search
            return yf_search(query, self.market_id, "SG", "SGX", yf_suffix=".SI")
        except Exception:
            return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from SGX Securities API.

        Looks up the ticker in the securities directory for name, currency,
        and board. Falls back to yfinance for sector/industry enrichment.
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        raw: dict[str, Any] = {
            "name": "",
            "ticker": identifier.upper(),
            "isin": "",
            "country": "SG",
            "sector": "",
            "industry": "",
            "exchange": "SGX",
            "currency": "SGD",
            "cik": identifier,
            "market_id": self.market_id,
        }

        # Primary: SGX Securities API lookup
        directory = _get_securities_directory()
        if directory:
            matches = _search_securities(identifier, directory, max_results=1)
            if matches:
                item = matches[0]
                raw["name"] = item.get("n", "")
                raw["currency"] = item.get("cur", "SGD")
                if item.get("m"):
                    raw["board"] = item["m"]

        # Supplement with yfinance for sector/industry (SGX API doesn't provide these)
        if not raw.get("sector"):
            try:
                from operator1.clients.yfinance_backed import yf_get_profile
                yf_profile = yf_get_profile(
                    identifier, self.market_id, "Singapore", "SG", "SGX", "SGD",
                    yf_suffix=".SI",
                )
                # Only take fields that SGX didn't provide
                for field in ("sector", "industry", "name", "market_cap"):
                    if yf_profile.get(field) and not raw.get(field):
                        raw[field] = yf_profile[field]
            except Exception as exc:
                logger.debug("yfinance profile supplement failed for %s: %s", identifier, exc)

        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch income statements via SGX filing discovery + LLM extraction."""
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        """Fetch balance sheets via SGX filing discovery + LLM extraction."""
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch cash flow statements via SGX filing discovery + LLM extraction."""
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financials via SGX filing discovery only (PIT-compliant).

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
                logger.info("SGX %s %s: %d rows from filing discovery",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("SGX filing discovery failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """SGX does not provide free historical OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

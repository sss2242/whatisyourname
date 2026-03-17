"""Canada SEDAR+ PIT client -- TMX directory + GraphQL profile.

SEDAR+ (www.sedarplus.ca) is entirely behind a PerfDrive/F5 WAF bot
challenge -- all REST endpoints redirect to validate.perfdrive.com.
The Catalyst form submission approach works but requires session
management and HTML parsing (used by the filing discoverer).

For company search and profiles, we use two TMX endpoints that work
globally without authentication:

1. TMX Company Directory (tsx.com/json/company-directory/search/tsx/)
   - Returns all ~2,900 TSX companies, paginated alphabetically
   - Cached in-memory for 24 hours after initial fetch
   - Client-side fuzzy search by name or symbol

2. TMX GraphQL (app-money.tmx.com/graphql)
   - Returns per-symbol details: name, sector, industry, price
   - Used for profile enrichment (no yfinance dependency)

Financial statements: SEDAR+ filing discovery + LLM extraction via
filing_discoverer.py (uses the Catalyst form POST mechanism to bypass
the WAF-blocked REST API).

OHLCV: handled separately via ohlcv_provider.py (yfinance .TO suffix).

Coverage: ~2,900 TSX + ~1,800 TSXV companies, ~$3.5T market cap.
"""

from __future__ import annotations

import json
import logging
import os
import string
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_TMX_DIRECTORY_URL = "https://www.tsx.com/json/company-directory/search"
_TMX_GRAPHQL_URL = "https://app-money.tmx.com/graphql"
_SEDAR_BASE = "https://www.sedarplus.ca/csa-party"
_CACHE_DIR = Path("cache/ca_sedar")

_TMX_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Module-level cache for the full TMX company directory.
_tmx_directory: list[dict[str, Any]] | None = None
_tmx_directory_fetched_at: float = 0.0
_tmx_directory_ttl_hours: int = 24


def _get_tmx_directory() -> list[dict[str, Any]]:
    """Fetch and cache the full TMX (TSX + TSXV) company directory.

    The TMX directory API works globally without authentication.
    Companies are paginated by first letter of symbol (A-Z).
    We fetch all letters for both TSX and TSXV exchanges.

    The directory is cached in-memory for 24 hours.
    """
    import requests

    global _tmx_directory, _tmx_directory_fetched_at

    now = time.time()
    if (
        _tmx_directory is not None
        and (now - _tmx_directory_fetched_at) < _tmx_directory_ttl_hours * 3600
    ):
        return _tmx_directory

    all_companies: list[dict[str, Any]] = []

    for exchange in ["tsx", "tsxv"]:
        for letter in string.ascii_uppercase:
            try:
                resp = requests.get(
                    f"{_TMX_DIRECTORY_URL}/{exchange}/%5E{letter}",
                    params={"start": "0", "end": "2000"},
                    headers=_TMX_HEADERS,
                    timeout=15,
                )
                if resp.status_code == 200 and resp.text.strip():
                    data = resp.json()
                    results = data.get("results", [])
                    for r in results:
                        r["_exchange"] = exchange.upper()
                    all_companies.extend(results)
            except Exception:
                continue

    if len(all_companies) > 100:
        _tmx_directory = all_companies
        _tmx_directory_fetched_at = now
        logger.info("TMX directory loaded: %d companies (TSX + TSXV)", len(all_companies))
    else:
        logger.warning("TMX directory fetch returned only %d companies", len(all_companies))

    return _tmx_directory or []


def _search_tmx_directory(
    query: str,
    directory: list[dict[str, Any]],
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Search the TMX directory by symbol or company name.

    Matches against:
    - ``symbol``: exact match (highest priority)
    - ``name``: case-insensitive substring match
    - ``symbol``: case-insensitive substring (lower priority)

    Returns results with exact symbol matches first.
    """
    if not query or not directory:
        return []

    q_lower = query.strip().lower()
    q_upper = query.strip().upper()

    # Phase 1: Exact symbol match (highest priority)
    exact = [d for d in directory if d.get("symbol", "") == q_upper]
    if exact:
        return exact[:max_results]

    # Phase 2: Name substring + symbol substring
    name_matches = []
    symbol_matches = []
    for d in directory:
        name = d.get("name", "").lower()
        sym = d.get("symbol", "").lower()
        if q_lower in name:
            name_matches.append(d)
        elif q_lower in sym:
            symbol_matches.append(d)

    # Name matches first (more relevant), then symbol matches
    combined = name_matches + symbol_matches
    return combined[:max_results]


def _tmx_graphql_profile(symbol: str) -> dict[str, Any] | None:
    """Fetch company profile from TMX GraphQL API.

    Returns sector, industry, name, and price for a given TSX/TSXV symbol.
    Works globally without authentication.
    """
    import requests

    query = (
        '{getQuoteBySymbol(symbol:"' + symbol + '")'
        '{symbol name price sector industry}}'
    )
    try:
        resp = requests.post(
            _TMX_GRAPHQL_URL,
            data=json.dumps({"query": query}),
            headers={
                **_TMX_HEADERS,
                "Content-Type": "application/json",
                "locale": "en",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("data", {}).get("getQuoteBySymbol")
        if result and result.get("name"):
            return result
    except Exception as exc:
        logger.debug("TMX GraphQL failed for %s: %s", symbol, exc)

    return None


class CASedarClient:
    """PIT client for Canadian SEDAR+ filings.

    Company search and profiles use the TMX exchange APIs (work globally).
    Financial statements use the SEDAR+ filing discoverer (Catalyst form
    POST mechanism, bypasses the WAF-blocked REST API).
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
        return "ca_sedar"

    @property
    def market_name(self) -> str:
        return "Canada (TSX / TSXV) -- SEDAR+"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Canadian companies from TMX company directory.

        Uses the TMX directory API (tsx.com) which works globally.
        The full directory (~4,700 TSX + TSXV companies) is fetched
        once and cached in-memory for 24 hours.  Client-side search
        matches by symbol (exact) or company name (substring).

        No yfinance or SEDAR+ dependency for search.
        """
        if not query:
            return []
        directory = _get_tmx_directory()
        if not directory:
            return []

        matches = _search_tmx_directory(query, directory)
        return [
            {
                "ticker": d.get("symbol", ""),
                "name": d.get("name", ""),
                "cik": "",
                "exchange": d.get("_exchange", "TSX"),
                "country": "CA",
                "market_id": self.market_id,
            }
            for d in matches
        ]

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from TMX GraphQL API.

        Uses the TMX GraphQL endpoint (app-money.tmx.com) which works
        globally and returns sector, industry, name, and price.
        Enriches with exchange info from the directory.

        No yfinance or SEDAR+ dependency for profiles.
        Profile is cached for 7 days.
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        raw: dict[str, Any] = {
            "name": "",
            "ticker": identifier.upper(),
            "isin": "",
            "country": "CA",
            "sector": "",
            "industry": "",
            "sub_industry": "",
            "exchange": "TSX",
            "currency": "CAD",
            "cik": "",
            "market_cap": "",
            "shares_outstanding": "",
            "lei": "",
        }

        # Try TMX GraphQL for rich profile data
        gql = _tmx_graphql_profile(identifier.upper())
        if gql:
            raw["name"] = gql.get("name", "")
            raw["sector"] = gql.get("sector", "")
            raw["industry"] = gql.get("industry", "")
            raw["ticker"] = gql.get("symbol", identifier.upper())
            logger.info(
                "TMX GraphQL enriched %s: %s (%s / %s)",
                identifier, raw["name"], raw["sector"], raw["industry"],
            )
        else:
            # Fall back to directory for at least the name
            directory = _get_tmx_directory()
            dir_matches = [
                d for d in directory
                if d.get("symbol", "").upper() == identifier.upper()
            ]
            if dir_matches:
                raw["name"] = dir_matches[0].get("name", "")
                raw["exchange"] = dir_matches[0].get("_exchange", "TSX")

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
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
        """Fetch financials via SEDAR+ filing discovery only (PIT-compliant).

        Uses the Catalyst form POST mechanism to bypass the WAF-blocked
        SEDAR+ REST API.  yfinance is NOT used for financial statements
        because it does not provide true filing dates.
        """
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
            df = try_filing_extraction(
                ticker=identifier, market_id=self.market_id,
                statement_type=statement_type, llm_client=None,
            )
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.debug("SEDAR+ filing extraction failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    # -- Price data -----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """SEDAR+ does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

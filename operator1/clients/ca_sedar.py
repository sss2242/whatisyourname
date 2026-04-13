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
        """Fetch company profile from TMX GraphQL API with full enrichment.

        Uses the TMX GraphQL endpoint (app-money.tmx.com) which works
        globally and returns sector, industry, name, price, valuation
        ratios, dividend data, fundamentals, and company description.

        The TMX quote API provides richer data than yfinance for
        Canadian stocks (MarketCap, PE, ROA, ROE, debt/equity, beta,
        dividends, volume averages, 52-week range, description).

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

        # Try TMX GraphQL for rich profile data (full quote enrichment)
        try:
            from operator1.clients.ohlcv_tmx import tmx_enrich_profile
            tmx_enrich_profile(identifier.upper(), raw)
        except Exception as exc:
            logger.debug("TMX enrichment failed for %s: %s", identifier, exc)

        # If TMX didn't provide a name, fall back to GraphQL basic query
        if not raw.get("name") or raw["name"] == identifier.upper():
            gql = _tmx_graphql_profile(identifier.upper())
            if gql:
                raw["name"] = gql.get("name", "")
                if not raw.get("sector"):
                    raw["sector"] = gql.get("sector", "")
                if not raw.get("industry"):
                    raw["industry"] = gql.get("industry", "")
                raw["ticker"] = gql.get("symbol", identifier.upper())

        # If still no name, fall back to directory
        if not raw.get("name") or raw["name"] == identifier.upper():
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

    # -- Institutional holders (TMX GraphQL for insider activity) -------------

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch insider activity summary from TMX GraphQL API.

        Uses the ``getCompanyInsidersActivities`` GraphQL query discovered
        by probing the TMX Money Next.js app JS bundles.  Returns
        aggregated insider buy/sell activity per period.

        For detailed insider transactions, see ``get_insider_transactions()``.

        Probing confirmed (2026-03-23):
          - SEDAR+ direct API: behind PerfDrive WAF (2.9KB challenge pages)
          - SEDAR+ Catalyst form: insider/earlyWarning types return 404
          - SEDI database: form-based, no REST API found
          - TMX GraphQL: ``getInsiderTransactions`` returns full SEDI data
          - TMX GraphQL: ``getCompanyInsidersActivities`` returns aggregated data
          - TMX REST API: /api/* endpoints return health check page, no data
          - TMX GraphQL introspection: blocked by Apollo Server
        """
        holders: list[dict[str, Any]] = []
        import requests as _requests

        try:
            # PRIMARY: getInsiderTransactions returns individual SEDI
            # transaction data with registeredholder names, amounts, and
            # market values.  We aggregate by unique holder name to produce
            # a holder list.  This is more reliable than
            # getCompanyInsidersActivities (whose schema changed in 2025+,
            # removing numberOfTransactions/averagePrice/totalValue fields).
            r = _requests.post(
                _TMX_GRAPHQL_URL,
                json={
                    "query": (
                        '{getInsiderTransactions(symbol:"'
                        + identifier.upper()
                        + '",monthDuration:12)}'
                    ),
                },
                headers={
                    **_TMX_HEADERS,
                    "Content-Type": "application/json",
                    "locale": "en",
                },
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                txn_list = data.get("data", {}).get("getInsiderTransactions", [])
                if isinstance(txn_list, list) and txn_list:
                    # Aggregate by unique holder name
                    holder_agg: dict[str, dict[str, Any]] = {}
                    for txn in txn_list:
                        if not isinstance(txn, dict):
                            continue
                        name = (txn.get("registeredholder") or "").strip()
                        if not name:
                            name = (txn.get("filer") or "").strip()
                        if not name or len(name) < 3:
                            continue
                        if name not in holder_agg:
                            holder_agg[name] = {
                                "shares": 0,
                                "value": 0.0,
                                "last_date": "",
                                "txn_count": 0,
                                "relationship": txn.get("relationship", ""),
                            }
                        agg = holder_agg[name]
                        agg["shares"] += int(txn.get("amount", 0) or 0)
                        agg["value"] += float(txn.get("marketvalue", 0) or 0)
                        agg["txn_count"] += 1
                        txn_date = txn.get("date", "")
                        if txn_date > agg["last_date"]:
                            agg["last_date"] = txn_date

                    for name, agg in holder_agg.items():
                        holders.append({
                            "name": name,
                            "shares": agg["shares"],
                            "value": round(agg["value"], 2),
                            "percentage": 0.0,
                            "holder_type": "insider",
                            "date_reported": agg["last_date"],
                            "source": "tmx_graphql_sedi",
                            "transactions": agg["txn_count"],
                            "relationship": agg["relationship"],
                        })

                    # Sort by absolute value (most active insiders first)
                    holders.sort(key=lambda h: abs(h.get("value", 0)), reverse=True)

                    logger.info(
                        "CA holders for %s: %d unique insiders from %d TMX SEDI transactions",
                        identifier, len(holders), len(txn_list),
                    )
        except Exception as exc:
            logger.debug("TMX GraphQL insider activity failed for %s: %s", identifier, exc)

        # Fallback: extract shareholders from filing PDFs
        if not holders:
            try:
                from operator1.clients.filing_discoverer import try_shareholding_extraction
                holders = try_shareholding_extraction(identifier, market_id=self.market_id)
                if holders:
                    logger.info("SEDAR holders from PDF shareholding extraction: %d", len(holders))
            except Exception as exc:
                logger.debug("SEDAR PDF shareholding fallback failed: %s", exc)


        # --- MarketScreener fallback (global institutional shareholder data) ---
        if not holders:
            try:
                from operator1.clients.marketscreener import fetch_shareholders
                company_name = ""
                if hasattr(self, '_read_cache'):
                    profile = self._read_cache(identifier, "profile.json")
                    if profile:
                        company_name = profile.get("name", "")
                if not company_name:
                    company_name = identifier
                ms_holders = fetch_shareholders(company_name)
                if ms_holders:
                    holders.extend(ms_holders)
                    logger.info(
                        "%s holders for %s: %d from MarketScreener (fallback)",
                        self.market_id, identifier, len(ms_holders),
                    )
            except Exception as exc:
                logger.debug("MarketScreener fallback failed for %s: %s", identifier, exc)

        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return ownership metrics from TMX insider activity data."""
        try:
            from datetime import date as _date
            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            total_buy_value = sum(
                h.get("value", 0) for h in holders if "buy" in h.get("holder_type", "")
            )
            total_sell_value = sum(
                h.get("value", 0) for h in holders if "sell" in h.get("holder_type", "")
            )
            net_insider = total_buy_value - total_sell_value

            return pd.DataFrame([{
                "date_reported": pd.Timestamp(_date.today()),
                "inst_ownership_pct": 0.0,
                "inst_top5_concentration": 0.0,
                "inst_holder_count": len(holders),
                "insider_net_value": round(net_insider, 2),
            }])
        except Exception as exc:
            logger.debug("CA holder history failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch individual insider transactions from TMX GraphQL (SEDI data).

        Uses the ``getInsiderTransactions`` GraphQL query which returns
        structured SEDI (System for Electronic Disclosure by Insiders) data
        including: registered holder name, transaction date, share count,
        price, market value, security designation, and transaction type.

        The ``monthDuration`` parameter controls the lookback window
        (3 = 3 months, 6 = 6 months, 12 = 12 months).
        """
        transactions: list[dict[str, Any]] = []
        import requests as _requests

        try:
            r = _requests.post(
                _TMX_GRAPHQL_URL,
                json={
                    "query": (
                        "query getInsiderTransactions($symbol: String!, $monthDuration: Int) {\n"
                        "  getInsiderTransactions(symbol: $symbol, monthDuration: $monthDuration)\n"
                        "}"
                    ),
                    "variables": {
                        "symbol": identifier.upper(),
                        "monthDuration": 12,
                    },
                },
                headers={
                    **_TMX_HEADERS,
                    "Content-Type": "application/json",
                    "locale": "en",
                },
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                txn_list = data.get("data", {}).get("getInsiderTransactions", [])
                if isinstance(txn_list, list):
                    for txn in txn_list:
                        if not isinstance(txn, dict):
                            continue
                        transactions.append({
                            "insider_name": txn.get("registeredholder", ""),
                            "position": txn.get("generalremarks", ""),
                            "date": txn.get("date", ""),
                            "transaction": txn.get("type", ""),
                            "shares": int(txn.get("amount", 0) or 0),
                            "value": float(txn.get("marketvalue", 0) or 0),
                            "price": float(txn.get("pricefrom", 0) or 0),
                            "security": txn.get("securitydesignation", ""),
                            "filing_date": txn.get("filingdate", ""),
                            "source": "tmx_graphql_sedi",
                        })

                if transactions:
                    logger.info(
                        "CA insider transactions for %s: %d from TMX GraphQL (SEDI)",
                        identifier, len(transactions),
                    )
        except Exception as exc:
            logger.debug("TMX GraphQL insider transactions failed for %s: %s", identifier, exc)

        return transactions

    def extract_segment_data(self, identifier: str) -> dict:
        """Extract product segment data from PDF filings via fuzzy parser.

        Uses the filing discovery + fuzzy_pdf_parser pipeline with
        market-specific segment keywords for ca_sedar.
        """
        try:
            from operator1.clients.filing_discoverer import try_segment_extraction
            return try_segment_extraction(identifier, "ca_sedar")
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                "CASedarClient segment extraction failed for %s: %s", identifier, exc,
            )
            return {"n_segments": 0, "segments": {}, "descriptions": {}}


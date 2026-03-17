"""India BSE/NSE PIT client -- BSE filing discovery + LLM extraction.

Primary financials: try_filing_extraction() from filing_discoverer.py
    - BSE AnnSubCategoryGetData API discovers filings (works globally)
    - BSEFilingDiscoverer downloads PDFs from bseindia.com
    - LLMFilingExtractor with Ind AS taxonomy hints extracts structured data
    - Smart page selection: scores pages for financial tables, skips noise
    - Per-ticker caching: income/balance/cashflow share one extraction call
    - Rate-limited staged extraction respects LLM RPM limits

Profile: yfinance (.NS/.BO) for sector/industry metadata only.
OHLCV: handled separately via ohlcv_provider.py (yfinance/nselib).

No date range limit: BSE announcements API returns all filings for the
full 2-year window in a single request (unlike HKEX which needs windowing).

Coverage: ~5,500+ listed companies on BSE/NSE, ~$4T market cap.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_BSE_BASE = "https://api.bseindia.com/BseIndiaAPI/api"
_CACHE_DIR = Path("cache/in_bse")

# Headers for BSE API requests
_BSE_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bseindia.com/",
}


class INBseClient:
    """PIT client for Indian BSE/NSE equities.

    Financial statements are extracted via the filing discoverer pipeline:
    BSE AnnSubCategoryGetData -> PDF download -> LLM extraction with
    Ind AS taxonomy hints -> canonical long-format DataFrame.

    The LLM extractor uses smart page selection (_select_top_pages)
    to find pages with financial tables and skip auditor reports,
    governance sections, and other noise.  The Ind AS taxonomy hints
    in llm_filing_extractor.py guide the LLM on what to look for
    (Revenue from Operations, Profit Before Tax, Total Assets, etc.)
    and warn about lakhs/crores unit conventions.

    When no LLM key is available, returns empty (the filing discoverer
    logs a clear message about which env var to set).
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
        return "in_bse"

    @property
    def market_name(self) -> str:
        return "India (BSE / NSE)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Indian companies from BSE API search.

        Uses the BSE Suggest endpoint which works globally.
        Falls back to yfinance search if BSE API is unreachable.
        """
        if not query:
            return []
        try:
            import requests
            resp = requests.get(
                f"{_BSE_BASE}/Suggest/Getstockdata/{query}",
                headers=_BSE_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            items = resp.json() if resp.text.strip().startswith("[") else []
            return [
                {
                    "ticker": str(i.get("scrip_cd", "")),
                    "name": i.get("scripname", ""),
                    "cik": str(i.get("scrip_cd", "")),
                    "exchange": "BSE",
                    "country": "IN",
                    "market_id": self.market_id,
                }
                for i in items
                if isinstance(i, dict)
            ]
        except Exception as exc:
            logger.debug("BSE company search failed: %s", exc)
            try:
                from operator1.clients.yfinance_backed import yf_search
                return yf_search(query, self.market_id, "IN", "BSE", yf_suffix=".NS")
            except Exception:
                return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile.

        Uses yfinance for rich metadata (sector, industry, market cap).
        Profile is cached for 7 days.
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        raw: dict[str, Any] = {
            "name": "",
            "ticker": identifier,
            "isin": "",
            "country": "IN",
            "sector": "",
            "industry": "",
            "sub_industry": "",
            "exchange": "BSE",
            "currency": "INR",
            "cik": identifier,
            "market_cap": "",
            "shares_outstanding": "",
            "lei": "",
        }

        self._enrich_from_yfinance(identifier, raw)

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def _enrich_from_yfinance(self, identifier: str, raw: dict) -> None:
        """Enrich profile with yfinance data (.NS/.BO suffix)."""
        try:
            import yfinance as yf

            for suffix in [".NS", ".BO"]:
                yf_ticker = f"{identifier}{suffix}"
                t = yf.Ticker(yf_ticker)
                info = t.info or {}

                if info.get("longName") or info.get("shortName"):
                    raw["name"] = info.get("longName") or info.get("shortName") or ""
                    raw["sector"] = info.get("sector", "")
                    raw["industry"] = info.get("industry", "")
                    mc = info.get("marketCap")
                    if mc:
                        raw["market_cap"] = str(mc)
                    shares = info.get("sharesOutstanding")
                    if shares:
                        raw["shares_outstanding"] = str(shares)
                    raw["isin"] = info.get("isin", "")
                    raw["exchange"] = info.get("exchange", "NSI")
                    logger.info("yfinance enriched %s via %s: %s", identifier, yf_ticker, raw["name"])
                    break

        except Exception as exc:
            logger.debug("yfinance profile failed for %s: %s", identifier, exc)

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch income statements via BSE filing discovery + LLM extraction."""
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        """Fetch balance sheets via BSE filing discovery + LLM extraction."""
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch cash flow statements via BSE filing discovery + LLM extraction."""
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial data via the filing discoverer pipeline.

        Delegates entirely to try_filing_extraction() which handles:
        1. BSE AnnSubCategoryGetData API for filing discovery (2yr window)
        2. PDF download from bseindia.com/xml-data/corpfiling/AttachLive/
        3. Smart page selection (scores pages for financial tables)
        4. LLM extraction with Ind AS taxonomy hints
        5. Canonical translation to long-format DataFrame
        6. Per-ticker caching (all 3 statement types share one extraction)
        7. Rate-limited staged extraction (respects LLM RPM limits)

        If no LLM key is set, the filing discoverer logs which env var
        to configure and returns empty.
        """
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
            df = try_filing_extraction(
                ticker=identifier,
                market_id=self.market_id,
                statement_type=statement_type,
            )
            if df is not None and not df.empty:
                logger.info(
                    "BSE %s/%s: %d rows from filing discoverer + LLM",
                    identifier, statement_type, len(df),
                )
                return df
        except Exception as exc:
            logger.warning("BSE filing extraction failed for %s/%s: %s", identifier, statement_type, exc)

        return pd.DataFrame()

    # -- Price data -----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """BSE does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

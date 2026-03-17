"""India BSE/NSE PIT client -- BSE filing discovery + LLM extraction.

Primary financials: try_filing_extraction() from filing_discoverer.py
    - BSE AnnSubCategoryGetData API discovers filings (works globally)
    - BSEFilingDiscoverer downloads PDFs from bseindia.com
    - LLMFilingExtractor with Ind AS taxonomy hints extracts structured data
    - Smart page selection: scores pages for financial tables, skips noise
    - Per-ticker caching: income/balance/cashflow share one extraction call
    - Rate-limited staged extraction respects LLM RPM limits

Company search: BSE ListofScripData API (works globally, returns all
    ~4,800 active equity scrips). Client-side fuzzy search by company
    name, scrip_id (ticker), or scrip code. No yfinance dependency.

Profile: BSE ComHeadernew API (works globally) for sector, industry,
    ISIN, EPS, PE, market cap. No yfinance dependency for core metadata.

OHLCV: handled separately via ohlcv_provider.py (yfinance/nselib).

No date range limit: BSE announcements API returns all filings for the
full 2-year window in a single request (unlike HKEX which needs windowing).

BSE API access notes:
    - Suggest/Getstockdata: BLOCKED globally (302 -> error_Bse.html)
    - StockQuote, EQPeerGp: BLOCKED globally
    - ListofScripData: WORKS globally (full company directory, ~1.7MB)
    - ComHeadernew: WORKS globally (per-scrip detail with ISIN, sector)
    - FinancialResult: WORKS globally (quarterly results HTML table)
    - AnnSubCategoryGetData: WORKS globally (filing announcements)

Coverage: ~4,800+ listed companies on BSE, ~$4T market cap.
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

# Module-level cache for the full BSE company directory.
# Fetched once per process from ListofScripData (works globally).
_scrip_directory: list[dict[str, Any]] | None = None
_scrip_directory_ttl_hours: int = 24
_scrip_directory_fetched_at: float = 0.0


def _get_scrip_directory() -> list[dict[str, Any]]:
    """Fetch and cache the full BSE equity scrip directory.

    Uses the ListofScripData endpoint which works globally (unlike
    Suggest/Getstockdata which is geo-blocked).  Returns ~4,800 active
    equity scrips with scrip code, name, ISIN, market cap, and scrip_id.

    The directory is cached in-memory for 24 hours.
    """
    import time
    global _scrip_directory, _scrip_directory_fetched_at

    now = time.time()
    if (
        _scrip_directory is not None
        and (now - _scrip_directory_fetched_at) < _scrip_directory_ttl_hours * 3600
    ):
        return _scrip_directory

    try:
        import requests
        resp = requests.get(
            f"{_BSE_BASE}/ListofScripData/w",
            params={
                "Group": "",
                "Atea": "",
                "segment": "Equity",
                "status": "Active",
            },
            headers=_BSE_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 100:
            _scrip_directory = data
            _scrip_directory_fetched_at = now
            logger.info("BSE scrip directory loaded: %d companies", len(data))
            return data
    except Exception as exc:
        logger.warning("Failed to fetch BSE scrip directory: %s", exc)

    return _scrip_directory or []


def _search_scrip_directory(
    query: str,
    directory: list[dict[str, Any]],
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Search the BSE scrip directory by name, scrip_id, or scrip code.

    Matches against:
    - ``scrip_id`` (ticker symbol, e.g. TCS, RELIANCE, INFY) -- exact match first
    - ``Scrip_Name`` (company name) -- case-insensitive substring
    - ``SCRIP_CD`` (scrip code, e.g. 500325) -- exact prefix match
    - ``Issuer_Name`` (full issuer name) -- case-insensitive substring

    Returns results sorted by market cap (largest first).
    """
    if not query or not directory:
        return []

    q_lower = query.strip().lower()
    q_stripped = query.strip()

    # Phase 1: Exact scrip_id match (highest priority)
    exact_id = [
        d for d in directory
        if d.get("scrip_id", "").lower() == q_lower
    ]
    if exact_id:
        return exact_id[:max_results]

    # Phase 2: Exact scrip code match
    exact_code = [
        d for d in directory
        if d.get("SCRIP_CD", "") == q_stripped
    ]
    if exact_code:
        return exact_code[:max_results]

    # Phase 3: Substring match on name/issuer + scrip_id contains
    matches = []
    for d in directory:
        name = d.get("Scrip_Name", "").lower()
        issuer = d.get("Issuer_Name", "").lower()
        sid = d.get("scrip_id", "").lower()
        if q_lower in name or q_lower in issuer or q_lower in sid:
            matches.append(d)

    # Sort by market cap (descending) for relevance
    def _mktcap(d: dict) -> float:
        try:
            return float(d.get("Mktcap", 0) or 0)
        except (ValueError, TypeError):
            return 0.0

    matches.sort(key=_mktcap, reverse=True)
    return matches[:max_results]


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
        """List Indian companies from BSE ListofScripData directory.

        Uses the BSE ListofScripData endpoint which works globally
        (unlike Suggest/Getstockdata which returns 302 from non-IN IPs).
        The full directory (~4,800 scrips) is fetched once and cached
        in-memory for 24 hours.  Client-side search matches by:
        - scrip_id (ticker): exact match (e.g. TCS, RELIANCE, INFY)
        - Scrip_Name / Issuer_Name: substring match (e.g. "Tata", "Infosys")
        - SCRIP_CD (scrip code): exact match (e.g. 500325)

        No yfinance dependency.  All data is PIT-sourced from BSE.
        """
        if not query:
            return []
        directory = _get_scrip_directory()
        if not directory:
            return []

        matches = _search_scrip_directory(query, directory)
        return [
            {
                "ticker": d.get("scrip_id", d.get("SCRIP_CD", "")),
                "name": d.get("Scrip_Name", ""),
                "cik": str(d.get("SCRIP_CD", "")),
                "isin": d.get("ISIN_NUMBER", ""),
                "exchange": "BSE",
                "country": "IN",
                "sector": "",
                "market_id": self.market_id,
            }
            for d in matches
        ]

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from BSE ComHeadernew API.

        Uses the ComHeadernew endpoint (works globally) for sector,
        industry, ISIN, EPS, PE, market cap.  Enriches with company
        name from the scrip directory.  No yfinance dependency.

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

        self._enrich_from_bse(identifier, raw)

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def _enrich_from_bse(self, identifier: str, raw: dict) -> None:
        """Enrich profile using BSE native APIs (no yfinance).

        Uses two BSE endpoints that work globally:
        1. ComHeadernew: sector, industry, ISIN, EPS, PE ratios
        2. ListofScripData (scrip directory): company name, market cap

        The scrip code may be provided as a numeric code (500325) or
        a ticker symbol (RELIANCE).  If a ticker is given, we resolve
        it to a scrip code via the directory first.
        """
        scrip_code = identifier.strip()

        # If identifier looks like a ticker (non-numeric), resolve via directory
        if not scrip_code.isdigit():
            directory = _get_scrip_directory()
            matches = [
                d for d in directory
                if d.get("scrip_id", "").lower() == scrip_code.lower()
            ]
            if matches:
                scrip_code = str(matches[0].get("SCRIP_CD", ""))
                raw["name"] = matches[0].get("Scrip_Name", "")
                raw["isin"] = matches[0].get("ISIN_NUMBER", "")
                raw["ticker"] = matches[0].get("scrip_id", identifier)
                try:
                    raw["market_cap"] = str(float(matches[0].get("Mktcap", 0) or 0) * 1e7)
                except (ValueError, TypeError):
                    pass
            else:
                logger.debug("BSE: ticker %s not found in scrip directory", identifier)
                return

        # Enrich name from scrip directory if we have a scrip code
        if not raw["name"]:
            directory = _get_scrip_directory()
            dir_match = [d for d in directory if d.get("SCRIP_CD", "") == scrip_code]
            if dir_match:
                raw["name"] = dir_match[0].get("Scrip_Name", "")
                raw["isin"] = dir_match[0].get("ISIN_NUMBER", "")
                raw["ticker"] = dir_match[0].get("scrip_id", identifier)
                try:
                    raw["market_cap"] = str(float(dir_match[0].get("Mktcap", 0) or 0) * 1e7)
                except (ValueError, TypeError):
                    pass

        # Fetch detailed info from ComHeadernew (works globally)
        try:
            import requests
            resp = requests.get(
                f"{_BSE_BASE}/ComHeadernew/w",
                params={"quotession": "", "scripcode": scrip_code},
                headers=_BSE_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            info = resp.json()
            if isinstance(info, dict) and info.get("SecurityCode"):
                raw["sector"] = info.get("Sector", "")
                raw["industry"] = info.get("IndustryNew", "")
                raw["sub_industry"] = info.get("ISubGroup", "")
                if info.get("ISIN"):
                    raw["isin"] = info["ISIN"]
                if info.get("SecurityId"):
                    raw["ticker"] = info["SecurityId"]
                raw["cik"] = str(info.get("SecurityCode", scrip_code))
                logger.info(
                    "BSE ComHeadernew enriched %s: %s (%s / %s)",
                    scrip_code, raw["name"], raw["sector"], raw["industry"],
                )
        except Exception as exc:
            logger.debug("BSE ComHeadernew failed for %s: %s", scrip_code, exc)

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
        # Path 1: Filing discoverer + LLM extraction (best quality)
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
            logger.debug("BSE LLM extraction failed for %s/%s: %s", identifier, statement_type, exc)

        # Path 2: Fuzzy PDF parser fallback (no LLM needed)
        logger.info(
            "BSE %s/%s: LLM unavailable, trying fuzzy PDF parser",
            identifier, statement_type,
        )
        return self._fetch_financials_fuzzy(identifier, statement_type)

    def _fetch_financials_fuzzy(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fallback: extract financials using fuzzy PDF parser (no LLM).

        Discovers filings via BSE announcements API, downloads PDFs,
        and uses fuzzy string matching to identify financial line items
        in pdfplumber table output.  Handles Indian number formats
        (lakhs, crores, parenthetical negatives).
        """
        try:
            from operator1.clients.fuzzy_pdf_parser import extract_financials_from_pdf
        except ImportError:
            logger.debug("fuzzy_pdf_parser not available")
            return pd.DataFrame()

        # Discover filings via BSE announcements API
        try:
            from operator1.clients.filing_discoverer import BSEFilingDiscoverer
            discoverer = BSEFilingDiscoverer()
            discovery = discoverer.discover_filings(identifier, years=2)
            if not discovery.has_filings:
                return pd.DataFrame()
        except Exception as exc:
            logger.debug("BSE filing discovery failed for fuzzy fallback: %s", exc)
            return pd.DataFrame()

        all_rows: list[dict] = []
        for filing in discovery.filings[:8]:
            try:
                pdf_bytes = discoverer.download_filing(filing)
                rows = extract_financials_from_pdf(
                    pdf_bytes,
                    filing_date=filing.filing_date or "",
                    report_date=filing.report_date or "",
                    statement_type=statement_type,
                )
                all_rows.extend(rows)
            except Exception as exc:
                logger.debug("Fuzzy extraction failed for filing %s: %s", filing.filing_date, exc)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
        if "report_date" in df.columns:
            mask = df["report_date"].notna() & (df["report_date"] >= cutoff)
            df = df[mask]

        if df.empty:
            return pd.DataFrame()

        from operator1.clients.canonical_translator import translate_financials
        result = translate_financials(df, self.market_id, statement_type)
        if not result.empty:
            logger.info(
                "BSE %s/%s: %d rows from fuzzy PDF parser (no LLM)",
                identifier, statement_type, len(result),
            )
        return result

    # -- Price data -----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """BSE does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

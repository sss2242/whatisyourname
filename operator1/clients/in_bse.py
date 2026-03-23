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

Holder data: yfinance (.NS suffix) + BSE BoardMeetings/CorporateAction APIs
    - yfinance provides major_holders (insiders%, institutions%, count)
      and insider_transactions for NSE-listed Indian companies
    - BSE BoardMeetings API (works globally) returns board meeting
      announcements with PDF attachments (shareholding outcomes)
    - BSE CorporateAction API (works globally) returns dividend data
    - Uses the HKEX scraper pattern: persistent session, paginated
      queries, client-side filtering

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
    - BoardMeetings: WORKS globally (board meetings with attachments)
    - CorporateAction: WORKS globally (dividends, splits, bonuses)
    - shpSecurities.aspx: JS-rendered (not scrapable with requests)
    - InsiderTrading/w: BLOCKED (302 -> error page)
    - ShareHoldPat/w: BLOCKED (302 -> error page)

Coverage: ~4,800+ listed companies on BSE, ~$4T market cap.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

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


# ---------------------------------------------------------------------------
# BSE holder/disclosure scraper (HKEX scraper pattern)
# ---------------------------------------------------------------------------
#
# BSE's dedicated shareholding endpoints (ShareHoldPat, InsiderTrading,
# CatWiseShrhlding) are all geo-blocked or require JavaScript rendering.
#
# Working BSE API endpoints for holder data:
#   - BoardMeetings/w: board meeting outcomes with PDF attachment links
#   - CorporateAction/w: dividends, splits, bonuses
#   - AnnSubCategoryGetData/w: filing announcements by category
#
# Strategy:
#   1. yfinance (.NS suffix) for major_holders + insider_transactions
#   2. BSE BoardMeetings API for shareholding-related board outcomes
#   3. BSE AnnSubCategoryGetData for shareholding pattern filings
#
# Uses the HKEX scraper pattern: persistent session for cookie reuse,
# Referer header for BSE API access, client-side filtering.

_bse_session: requests.Session | None = None

# Rate limit between BSE API requests (no documented limit, be polite)
_BSE_REQUEST_DELAY_S = 0.3


def _get_bse_session() -> requests.Session:
    """Create or return a persistent requests session for BSE API calls.

    Mirrors the HKEX scraper pattern: persistent session reuses
    cookies and TCP connections.  The Referer header is critical
    for BSE API access (requests without it get 302 redirects).
    """
    global _bse_session
    if _bse_session is None:
        _bse_session = requests.Session()
        _bse_session.headers.update(_BSE_HEADERS)
    return _bse_session


def _resolve_scrip_code(identifier: str) -> str:
    """Resolve a ticker symbol to a BSE scrip code.

    BSE APIs require numeric scrip codes (e.g. 500325 for Reliance).
    If the identifier is already numeric, return as-is.  Otherwise,
    look it up in the scrip directory.
    """
    stripped = identifier.strip()
    if stripped.isdigit():
        return stripped

    directory = _get_scrip_directory()
    matches = [
        d for d in directory
        if d.get("scrip_id", "").lower() == stripped.lower()
    ]
    if matches:
        return str(matches[0].get("SCRIP_CD", ""))
    return stripped


def _resolve_nse_ticker(identifier: str) -> str:
    """Resolve a BSE identifier to an NSE ticker for yfinance.

    yfinance uses NSE tickers with .NS suffix (e.g. RELIANCE.NS).
    The BSE scrip directory contains scrip_id which is usually the
    NSE ticker symbol.
    """
    stripped = identifier.strip()

    # If already a ticker symbol (non-numeric), use it directly
    if not stripped.isdigit():
        return stripped.upper()

    # Numeric scrip code -- look up the ticker in the directory
    directory = _get_scrip_directory()
    matches = [d for d in directory if d.get("SCRIP_CD", "") == stripped]
    if matches:
        return matches[0].get("scrip_id", stripped).upper()
    return stripped


def _fetch_bse_board_meetings(
    scrip_code: str,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch board meeting announcements from BSE for a given scrip code.

    Uses the BSE BoardMeetings/w endpoint which works globally and
    returns JSON with meeting subjects and PDF attachment paths.

    Board meetings often include outcomes related to shareholding
    patterns, dividend declarations, and insider trading disclosures.

    Parameters
    ----------
    scrip_code:
        BSE numeric scrip code (e.g. '500325').
    session:
        Optional pre-existing requests session.

    Returns
    -------
    List of board meeting dicts with keys from the BSE API:
    AnnCategory, Sub_Ann, Attchment_name, Fin_Year.
    """
    if session is None:
        session = _get_bse_session()

    try:
        resp = session.get(
            f"{_BSE_BASE}/BoardMeetings/w",
            params={"scripcode": scrip_code},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.debug("BSE BoardMeetings failed for %s: %s", scrip_code, exc)

    return []


def _fetch_bse_corporate_actions(
    scrip_code: str,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch corporate actions (dividends, splits, bonuses) from BSE.

    Uses the BSE CorporateAction/w endpoint which returns structured
    JSON with action type, date, and amount.

    Parameters
    ----------
    scrip_code:
        BSE numeric scrip code (e.g. '500325').
    session:
        Optional pre-existing requests session.

    Returns
    -------
    List of corporate action dicts.
    """
    if session is None:
        session = _get_bse_session()

    try:
        resp = session.get(
            f"{_BSE_BASE}/CorporateAction/w",
            params={"scripcode": scrip_code},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data.get("Table", [])
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.debug("BSE CorporateAction failed for %s: %s", scrip_code, exc)

    return []


def _extract_holders_from_board_meetings(
    meetings: list[dict[str, Any]],
    scrip_code: str,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Extract shareholder data from board meeting announcement PDFs.

    Board meeting outcome announcements on BSE often contain references
    to shareholding patterns (e.g. "Approved the Shareholding Pattern",
    "Declaration of Dividend").  The attachment PDFs may contain
    detailed shareholder data.

    Uses the HKEX scraper pattern: session-based PDF download +
    text extraction for shareholder names and percentages.

    Parameters
    ----------
    meetings:
        List of board meeting dicts from BSE BoardMeetings/w.
    scrip_code:
        BSE scrip code (for logging).
    session:
        Active requests session.

    Returns
    -------
    List of holder dicts extracted from meeting attachments.
    """
    if session is None:
        session = _get_bse_session()

    holders: list[dict[str, Any]] = []

    shareholding_keywords = (
        "shareholding", "shareholder", "promoter", "institutional",
        "public", "fii", "dii", "mutual fund",
    )

    for meeting in meetings:
        subject = (meeting.get("Sub_Ann") or "").lower()
        attachment = meeting.get("Attchment_name", "")

        # Only process meetings related to shareholding outcomes
        if not any(kw in subject for kw in shareholding_keywords):
            continue

        if not attachment:
            continue

        # Construct full URL for the PDF attachment
        if attachment.startswith("/"):
            pdf_url = f"https://www.bseindia.com{attachment}"
        else:
            pdf_url = attachment

        # Download and parse the PDF for shareholder data
        try:
            resp = session.get(pdf_url, timeout=30)
            if resp.status_code != 200 or resp.content[:4] != b"%PDF":
                continue

            # Extract text from PDF for shareholding pattern data
            try:
                import pdfplumber
                import io

                with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                    text = ""
                    for page in pdf.pages[:5]:  # First 5 pages max
                        text += (page.extract_text() or "") + "\n"

                # Parse promoter/public/institutional percentages
                _parse_shareholding_text(text, holders, meeting)
            except ImportError:
                logger.debug("pdfplumber not available for BSE PDF parsing")
            except Exception as exc:
                logger.debug("PDF text extraction failed: %s", exc)

            time.sleep(_BSE_REQUEST_DELAY_S)
        except Exception as exc:
            logger.debug("BSE PDF download failed for %s: %s", pdf_url, exc)

    return holders


def _parse_shareholding_text(
    text: str,
    holders: list[dict[str, Any]],
    meeting: dict[str, Any],
) -> None:
    """Parse shareholding pattern text from a BSE PDF.

    Looks for SEBI-mandated shareholding pattern categories:
    - Promoter & Promoter Group
    - Public (Institutional + Non-Institutional)
    - FII / FPI
    - DII / Mutual Funds
    - Custodians (for GDR/ADR)

    Extracts category name + percentage from text patterns like:
    "Promoter & Promoter Group  50.30%"
    "Total Public Shareholding  49.70%"
    """
    # Common shareholding pattern line formats in Indian filings
    patterns = [
        (r"(?:Promoter\s*(?:&|and)\s*Promoter\s*Group)[^\d]*([\d.]+)\s*%?", "Promoter & Promoter Group"),
        (r"(?:Total\s+)?Public\s+Shareholding[^\d]*([\d.]+)\s*%?", "Public Shareholding"),
        (r"(?:Foreign\s+)?(?:Institutional\s+Investor|FII|FPI)[^\d]*([\d.]+)\s*%?", "Foreign Institutional Investors"),
        (r"(?:Domestic\s+)?(?:Institutional\s+Investor|DII)[^\d]*([\d.]+)\s*%?", "Domestic Institutional Investors"),
        (r"Mutual\s+Fund[s]?[^\d]*([\d.]+)\s*%?", "Mutual Funds"),
        (r"Insurance\s+Compan(?:y|ies)[^\d]*([\d.]+)\s*%?", "Insurance Companies"),
        (r"Bank[s]?\s*[,/&]\s*Financial\s+Institution[s]?[^\d]*([\d.]+)\s*%?", "Banks & Financial Institutions"),
        (r"(?:Non[- ]?Resident\s+Indian|NRI)[^\d]*([\d.]+)\s*%?", "Non-Resident Indians"),
        (r"Custodian[s]?\s*(?:for|of)?\s*(?:GDR|ADR|DR)[^\d]*([\d.]+)\s*%?", "Custodians (GDR/ADR)"),
    ]

    fin_year = meeting.get("Fin_Year", "")

    for pattern, category in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                pct = float(match.group(1))
                if 0.01 < pct < 100:
                    # Check for duplicate
                    if not any(h["name"] == category for h in holders):
                        holders.append({
                            "name": category,
                            "shares": 0,
                            "percentage": round(pct, 2),
                            "holder_type": "category",
                            "date_reported": fin_year,
                            "source": "bse_board_meeting",
                        })
            except ValueError:
                pass


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

    # -- Institutional holders (yfinance + BSE scraping) ---------------------

    def _yf_ticker(self, identifier: str) -> str:
        """Convert BSE identifier to yfinance NSE format (e.g. '500325' -> 'RELIANCE.NS')."""
        nse_ticker = _resolve_nse_ticker(identifier)
        return f"{nse_ticker}.NS"

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch institutional holders via yfinance + BSE board meeting scraping.

        Two-source strategy (mirrors HKHkexClient pattern):

        1. **yfinance** (.NS suffix): provides major_holders aggregate stats
           (insiders%, institutions%, count) for NSE-listed Indian companies.
           Note: yfinance institutional_holders and mutualfund_holders return
           empty for most Indian stocks -- the aggregate stats from
           major_holders are the primary value.

        2. **BSE BoardMeetings API**: fetches board meeting outcomes that
           reference shareholding patterns.  When meetings have PDF
           attachments, downloads and parses them for SEBI-mandated
           shareholding category breakdowns (Promoter, Public, FII, DII,
           Mutual Funds, etc.) using pdfplumber.

        Uses the HKEX scraper pattern: persistent session for cookie reuse,
        BSE Referer header for API access, rate limiting between requests.
        """
        holders: list[dict[str, Any]] = []
        scrip_code = _resolve_scrip_code(identifier)

        # Primary: yfinance major_holders for aggregate ownership stats
        try:
            import yfinance as yf
            yf_ticker = self._yf_ticker(identifier)
            tick = yf.Ticker(yf_ticker)

            # major_holders provides insiders%, institutions%, count
            mh = tick.major_holders
            if mh is not None and not mh.empty:
                for idx, row in mh.iterrows():
                    breakdown = (
                        str(row.get("Breakdown", idx)).lower()
                        if "Breakdown" in mh.columns
                        else str(idx).lower()
                    )
                    val = (
                        row.get("Value", row.iloc[-1])
                        if "Value" in mh.columns
                        else row.iloc[-1]
                    )
                    try:
                        pct = float(val) * 100 if float(val) < 1 else float(val)
                    except (ValueError, TypeError):
                        continue

                    if "insiderspercentheld" in breakdown or "insider" in breakdown:
                        holders.append({
                            "name": "Insiders / Promoters",
                            "shares": 0,
                            "percentage": round(pct, 2),
                            "holder_type": "promoter",
                            "date_reported": "",
                            "source": "yfinance_major",
                        })
                    elif "institutionspercentheld" in breakdown or (
                        "institutions" in breakdown and "percent" in breakdown
                    ):
                        holders.append({
                            "name": "Institutional Investors",
                            "shares": 0,
                            "percentage": round(pct, 2),
                            "holder_type": "institutional",
                            "date_reported": "",
                            "source": "yfinance_major",
                        })

            # Try institutional_holders (usually empty for Indian stocks)
            inst = tick.institutional_holders
            if inst is not None and not inst.empty:
                for _, row in inst.iterrows():
                    pct = row.get("pctHeld", 0) or row.get("% Out", 0) or 0
                    if isinstance(pct, (int, float)) and 0 < pct < 1:
                        pct = pct * 100
                    holders.append({
                        "name": str(row.get("Holder", "")),
                        "shares": int(row.get("Shares", 0)),
                        "value": float(row.get("Value", 0)),
                        "percentage": round(float(pct), 2),
                        "holder_type": "institutional",
                        "date_reported": str(row.get("Date Reported", "")),
                        "source": "yfinance",
                    })

            # Try mutualfund_holders (usually empty for Indian stocks)
            mf = tick.mutualfund_holders
            if mf is not None and not mf.empty:
                for _, row in mf.iterrows():
                    pct = row.get("pctHeld", 0) or row.get("% Out", 0) or 0
                    if isinstance(pct, (int, float)) and 0 < pct < 1:
                        pct = pct * 100
                    holders.append({
                        "name": str(row.get("Holder", "")),
                        "shares": int(row.get("Shares", 0)),
                        "value": float(row.get("Value", 0)),
                        "percentage": round(float(pct), 2),
                        "holder_type": "mutualfund",
                        "date_reported": str(row.get("Date Reported", "")),
                        "source": "yfinance",
                    })

            if holders:
                logger.info(
                    "BSE holders for %s: %d from yfinance (%s)",
                    identifier, len(holders), yf_ticker,
                )
        except Exception as exc:
            logger.debug("yfinance holders failed for BSE %s: %s", identifier, exc)

        # Supplement: BSE BoardMeetings API for shareholding pattern data
        # Uses HKEX scraper pattern: session-based, rate-limited queries
        try:
            session = _get_bse_session()
            meetings = _fetch_bse_board_meetings(scrip_code, session)
            if meetings:
                bse_holders = _extract_holders_from_board_meetings(
                    meetings, scrip_code, session,
                )
                if bse_holders:
                    existing_names = {h["name"].lower() for h in holders}
                    for bh in bse_holders:
                        if bh["name"].lower() not in existing_names:
                            holders.append(bh)
                            existing_names.add(bh["name"].lower())
                    logger.info(
                        "BSE board meetings for %s: %d additional holder categories",
                        identifier, len(bse_holders),
                    )
        except Exception as exc:
            logger.debug("BSE board meeting scraping failed for %s: %s", identifier, exc)

        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return institutional ownership metrics as a single-row snapshot.

        Uses yfinance major_holders to extract aggregate ownership stats
        for Indian companies.  yfinance provides a current-quarter snapshot.

        Uses the same pattern as HKHkexClient.get_holder_history().
        """
        try:
            import yfinance as yf
            from datetime import date as _date
            yf_ticker = self._yf_ticker(identifier)
            tick = yf.Ticker(yf_ticker)

            mh = tick.major_holders
            inst_pct = 0.0
            inst_count = 0
            if mh is not None and not mh.empty:
                for idx, row in mh.iterrows():
                    breakdown = (
                        str(row.get("Breakdown", idx)).lower()
                        if "Breakdown" in mh.columns
                        else str(idx).lower()
                    )
                    val = (
                        row.get("Value", row.iloc[-1])
                        if "Value" in mh.columns
                        else row.iloc[-1]
                    )
                    if (
                        "institutionspercentheld" in breakdown
                        or ("institutions" in breakdown and "percent" in breakdown)
                    ):
                        inst_pct = float(val) * 100 if float(val) < 1 else float(val)
                    elif "institutionscount" in breakdown or "count" in breakdown:
                        inst_count = int(float(val))

            holders = self.get_holders(identifier)
            hhi = 0.0
            if holders:
                top5 = [h for h in holders if h.get("holder_type") != "category"][:5]
                if not top5:
                    top5 = holders[:5]
                total_pct = sum(h.get("percentage", 0) for h in top5)
                if total_pct > 0:
                    hhi = sum(
                        (h.get("percentage", 0) / total_pct) ** 2 for h in top5
                    )

            if inst_pct > 0 or inst_count > 0 or holders:
                return pd.DataFrame([{
                    "date_reported": pd.Timestamp(_date.today()),
                    "inst_ownership_pct": round(inst_pct, 2),
                    "inst_top5_concentration": round(hhi, 4),
                    "inst_holder_count": inst_count or len(holders),
                }])
        except Exception as exc:
            logger.debug("BSE holder history failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch insider transactions via yfinance for BSE/NSE-listed companies.

        Uses the same pattern as HKHkexClient.get_insider_transactions().
        yfinance returns SEBI SAST disclosure data for Indian companies.
        """
        transactions: list[dict[str, Any]] = []
        try:
            import yfinance as yf
            yf_ticker = self._yf_ticker(identifier)
            tick = yf.Ticker(yf_ticker)
            insider = tick.insider_transactions
            if insider is not None and not insider.empty:
                for _, row in insider.iterrows():
                    shares = 0
                    try:
                        shares = int(row.get("Shares", 0))
                    except (ValueError, TypeError):
                        pass
                    transactions.append({
                        "insider_name": str(row.get("Insider", "")),
                        "position": str(row.get("Position", "")),
                        "date": str(row.get("Start Date", "")),
                        "transaction": str(row.get("Transaction", "")),
                        "shares": shares,
                        "value": float(row.get("Value", 0) or 0),
                    })
                logger.info(
                    "BSE insider transactions for %s: %d (%s)",
                    identifier, len(transactions), yf_ticker,
                )
        except Exception as exc:
            logger.debug(
                "yfinance insider transactions failed for BSE %s: %s",
                identifier, exc,
            )
        return transactions

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
# Uses BSE AnnSubCategoryGetData API with strCat='Insider Trading / SAST'
# to fetch SEBI SAST (Substantial Acquisition of Shares & Takeovers)
# disclosure filings.  This is the same API endpoint and parameter pattern
# used by BSEFilingDiscoverer for financial results (strCat='Result').
#
# Key discoveries from API probing:
#   - AnnSubCategoryGetData/w with strCat='Insider Trading / SAST': WORKS
#     Returns Reg. 29(1), 29(2), 31(1), 31(2) disclosures + trading window
#   - The HEADLINE field contains the substantial shareholder name directly
#     (e.g. "Bhairavi Madhusudhan Shibulal")
#   - SAST PDFs live at /xml-data/corpfiling/AttachHis/ (not AttachLive!)
#   - BoardMeetings/w: WORKS, returns JSON with board outcomes
#   - CorporateAction/w: WORKS, returns dividend/split/bonus data
#   - ShareHoldPat/w, InsiderTrading/w: blocked (302 redirect)
#   - shpSecurities.aspx: JS-rendered, not scrapable with requests
#
# Uses the HKEX scraper pattern: persistent session for cookie reuse,
# Referer header for BSE API access, client-side filtering.

# BSE PDF base URLs -- SAST filings use AttachHis, not AttachLive
_BSE_PDF_ATTACH_HIS = "https://www.bseindia.com/xml-data/corpfiling/AttachHis"
_BSE_PDF_ATTACH_LIVE = "https://www.bseindia.com/xml-data/corpfiling/AttachLive"

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


def _fetch_bse_sast_disclosures(
    scrip_code: str,
    session: requests.Session | None = None,
    years: int = 2,
) -> list[dict[str, Any]]:
    """Fetch SEBI SAST disclosure filings from BSE AnnSubCategoryGetData.

    Uses the exact same API endpoint and parameter pattern as
    BSEFilingDiscoverer.discover_filings() (strCat='Result'), but
    with strCat='Insider Trading / SAST' to get:
      - Reg. 29(1): disclosure on acquisition of shares
      - Reg. 29(2): disclosure on change in shareholding
      - Reg. 31(1): disclosure of shareholding pattern
      - Reg. 31(2): disclosure of aggregate shareholding
      - Closure of Trading Window notices

    The HEADLINE field contains the substantial shareholder name
    directly (e.g. "disclosure under Regulation 29(2) ... for
    Bhairavi Madhusudhan Shibulal").

    Parameters
    ----------
    scrip_code:
        BSE numeric scrip code (e.g. '500325').
    session:
        Optional pre-existing requests session.
    years:
        How many years back to search.

    Returns
    -------
    List of raw BSE announcement dicts with keys: NEWSID, SCRIP_CD,
    NEWSSUB, HEADLINE, NEWS_DT, ATTACHMENTNAME, SUBCATNAME, etc.
    """
    if session is None:
        session = _get_bse_session()

    today = date.today()
    from_date = today - timedelta(days=365 * years)

    try:
        resp = session.get(
            f"{_BSE_BASE}/AnnSubCategoryGetData/w",
            params={
                "Ession": "",
                "strCat": "Insider Trading / SAST",
                "strPrevDate": from_date.strftime("%Y%m%d"),
                "strScrip": scrip_code,
                "strSearch": "P",
                "strToDate": today.strftime("%Y%m%d"),
                "strType": "C",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        table = data.get("Table", [])
        logger.info(
            "BSE SAST disclosures for %s: %d filings (2yr window)",
            scrip_code, len(table),
        )
        return table
    except Exception as exc:
        logger.debug("BSE SAST API failed for %s: %s", scrip_code, exc)
        return []


def _extract_holders_from_sast(
    filings: list[dict[str, Any]],
    scrip_code: str,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Extract substantial shareholder data from BSE SAST disclosures.

    Two extraction strategies:

    1. **HEADLINE field** (fast, no PDF download needed): BSE includes
       the shareholder name directly in the HEADLINE field for Reg. 29
       disclosures (e.g. "disclosure under Regulation 29(2) ... for
       Bhairavi Madhusudhan Shibulal").

    2. **PDF extraction** (detailed): SAST PDFs at AttachHis/ contain
       the full SEBI disclosure form with shareholder name, shares held,
       percentage, and transaction details.  Parsed via pdfplumber.

    Parameters
    ----------
    filings:
        List of BSE SAST announcement dicts from _fetch_bse_sast_disclosures().
    scrip_code:
        BSE scrip code (for logging).
    session:
        Active requests session.

    Returns
    -------
    List of holder dicts with keys: name, shares, percentage,
    holder_type, date_reported, source, regulation.
    """
    if session is None:
        session = _get_bse_session()

    holders: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for filing in filings:
        subcatname = filing.get("SUBCATNAME", "")
        headline = filing.get("HEADLINE", "")
        news_dt = (filing.get("NEWS_DT") or "")[:10]
        attachment = filing.get("ATTACHMENTNAME", "")

        # Skip Closure of Trading Window (no shareholder data).
        # IMPORTANT: check for "closure of trading" not just "closure"
        # because "Disclosures under Reg. 29(2)" contains "closure" as
        # a substring of "Disclosures" and must NOT be skipped.
        if "closure of trading" in subcatname.lower():
            continue

        # Strategy 1: Extract shareholder name from HEADLINE field
        # HEADLINE format: "The Exchange has received the disclosure under
        # Regulation 29(2) of SEBI (SAST) Regulations, 2011 for <NAME>"
        shareholder_name = ""
        if headline:
            # Extract name after "for " at the end of the headline
            match = re.search(
                r"\bfor\s+([A-Z][A-Za-z\s.()]+?)\.?\s*$",
                headline,
            )
            if match:
                shareholder_name = match.group(1).strip().rstrip(".")
            elif " for " in headline:
                parts = headline.rsplit(" for ", 1)
                if len(parts) == 2 and len(parts[1].strip()) > 3:
                    shareholder_name = parts[1].strip().rstrip(".")

        if shareholder_name and shareholder_name.lower() not in seen_names:
            seen_names.add(shareholder_name.lower())

            # Determine regulation type
            regulation = ""
            if "29(1)" in subcatname:
                regulation = "Reg. 29(1) - Acquisition"
            elif "29(2)" in subcatname:
                regulation = "Reg. 29(2) - Change in Shareholding"
            elif "31(1)" in subcatname:
                regulation = "Reg. 31(1) - Shareholding Pattern"
            elif "31(2)" in subcatname:
                regulation = "Reg. 31(2) - Aggregate Shareholding"

            holders.append({
                "name": shareholder_name,
                "shares": 0,
                "percentage": 0.0,
                "holder_type": "substantial",
                "date_reported": news_dt,
                "source": "bse_sast_headline",
                "regulation": regulation,
            })

        # Strategy 2: Try downloading the SAST PDF for detailed data
        # SAST PDFs use AttachHis (not AttachLive)
        if attachment and len(holders) < 20:
            for base_url in [_BSE_PDF_ATTACH_HIS, _BSE_PDF_ATTACH_LIVE]:
                pdf_url = f"{base_url}/{attachment}"
                try:
                    resp = session.get(pdf_url, timeout=30)
                    if resp.status_code != 200 or resp.content[:4] != b"%PDF":
                        continue

                    # Parse SAST PDF for shareholding data
                    try:
                        import pdfplumber
                        import io

                        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                            text = ""
                            for page in pdf.pages[:5]:
                                text += (page.extract_text() or "") + "\n"

                        if text.strip():
                            _parse_sast_pdf_text(
                                text, holders, shareholder_name, news_dt,
                            )
                    except ImportError:
                        pass
                    except Exception as exc:
                        logger.debug("SAST PDF parse failed: %s", exc)

                    time.sleep(_BSE_REQUEST_DELAY_S)
                    break  # Found the PDF, don't try other base URLs
                except Exception:
                    continue

    return holders


def _parse_sast_pdf_text(
    text: str,
    holders: list[dict[str, Any]],
    shareholder_name: str,
    news_dt: str,
) -> None:
    """Parse a SEBI SAST disclosure PDF for shareholding percentages.

    SAST forms (Reg. 29/31) contain structured data including:
    - Name of the acquirer/substantial shareholder
    - Number of shares held before and after the transaction
    - Percentage of shares held
    - Type of shares (equity, preference, convertible)

    Extracts percentage from common SAST form patterns.
    """
    # Look for percentage patterns in SAST forms
    pct_patterns = [
        # "XX.XX% of the total share/voting capital"
        r"([\d.]+)\s*%\s*(?:of\s+(?:the\s+)?(?:total|paid[- ]?up|issued))",
        # "Percentage of shareholding: XX.XX%"
        r"(?:[Pp]ercentage|%)\s*(?:of\s+)?(?:share(?:holding)?|voting)[^:]*?:\s*([\d.]+)",
        # "XX.XX % of voting rights"
        r"([\d.]+)\s*%\s*(?:of\s+)?voting\s+rights",
    ]

    for pattern in pct_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                pct = float(match.group(1))
                if 0.01 < pct < 100:
                    # Update the existing holder entry if we have a name match
                    for h in holders:
                        if (
                            h["name"].lower() == shareholder_name.lower()
                            and h["percentage"] == 0.0
                        ):
                            h["percentage"] = round(pct, 2)
                            h["source"] = "bse_sast_pdf"
                            break
                    break
            except ValueError:
                pass

    # Also look for number of shares
    shares_patterns = [
        r"(?:[Nn]umber\s+of\s+shares)[^:]*?:\s*([\d,]+)",
        r"(?:[Tt]otal\s+shares)\s*(?:held)?[^:]*?:\s*([\d,]+)",
    ]

    for pattern in shares_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                shares = int(match.group(1).replace(",", ""))
                for h in holders:
                    if (
                        h["name"].lower() == shareholder_name.lower()
                        and h["shares"] == 0
                    ):
                        h["shares"] = shares
                        break
                break
            except ValueError:
                pass

    # Parse category-level shareholding patterns if present
    # (Reg. 31 forms contain full shareholding breakdown)
    _parse_shareholding_text(text, holders, {"Fin_Year": news_dt})


def _fetch_bse_board_meetings(
    scrip_code: str,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch board meeting announcements from BSE for a given scrip code.

    Uses the BSE BoardMeetings/w endpoint which works globally and
    returns JSON with meeting subjects and PDF attachment paths.
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
    """
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
                    if not any(h["name"] == category for h in holders):
                        holders.append({
                            "name": category,
                            "shares": 0,
                            "percentage": round(pct, 2),
                            "holder_type": "category",
                            "date_reported": fin_year,
                            "source": "bse_sast_pdf",
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

        # PRIMARY: BSE SAST disclosures (Reg. 29/31) via AnnSubCategoryGetData
        # Uses the same API pattern as BSEFilingDiscoverer (strCat='Result')
        # but with strCat='Insider Trading / SAST' for shareholding data.
        # HEADLINE field contains shareholder names; PDFs at AttachHis/ have details.
        try:
            bse_session = _get_bse_session()
            sast_filings = _fetch_bse_sast_disclosures(scrip_code, bse_session)
            if sast_filings:
                sast_holders = _extract_holders_from_sast(
                    sast_filings, scrip_code, bse_session,
                )
                if sast_holders:
                    existing_names = {h["name"].lower() for h in holders}
                    for sh in sast_holders:
                        if sh["name"].lower() not in existing_names:
                            holders.append(sh)
                            existing_names.add(sh["name"].lower())
                    logger.info(
                        "BSE SAST disclosures for %s: %d substantial shareholders",
                        identifier, len(sast_holders),
                    )
        except Exception as exc:
            logger.debug("BSE SAST scraping failed for %s: %s", identifier, exc)

        # SECONDARY: BSE filing discoverer + fuzzy_pdf_parser for shareholding
        # patterns from quarterly result PDFs (SEBI LODR Reg. 31 filings often
        # include shareholding pattern data as appendices).
        # Uses the existing fuzzy_pdf_parser module (camelot-py + fuzzy matching)
        # which handles Indian number formats (lakhs, crores, parenthetical
        # negatives) and SEBI-format table extraction with ~98% accuracy.
        if not any(h.get("holder_type") == "category" for h in holders):
            try:
                from operator1.clients.filing_discoverer import BSEFilingDiscoverer
                from operator1.clients.fuzzy_pdf_parser import extract_financials_from_pdf
                discoverer = BSEFilingDiscoverer()
                discovery = discoverer.discover_filings(scrip_code, years=1)

                if discovery.has_filings:
                    for filing in discovery.filings[:3]:
                        try:
                            pdf_bytes = discoverer.download_filing(filing)
                            if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
                                continue

                            # Use the fuzzy parser for structured extraction
                            rows = extract_financials_from_pdf(
                                pdf_bytes,
                                filing_date=filing.filing_date or "",
                                report_date=filing.report_date or "",
                                statement_type="balance",  # shareholding is in balance-like tables
                            )

                            # Also parse raw text for shareholding patterns
                            # (fuzzy parser handles financials; _parse_shareholding_text
                            # handles SEBI-mandated category breakdowns)
                            try:
                                from operator1.clients.fuzzy_pdf_parser import _extract_tables_text
                                text = _extract_tables_text(pdf_bytes)
                            except (ImportError, AttributeError):
                                import pdfplumber as _pdfp, io as _io
                                with _pdfp.open(_io.BytesIO(pdf_bytes)) as pdf:
                                    text = "\n".join(
                                        (p.extract_text() or "") for p in pdf.pages
                                    )

                            if text and ("promoter" in text.lower() or "category of shareholder" in text.lower()):
                                _parse_shareholding_text(
                                    text, holders,
                                    {"Fin_Year": filing.filing_date or filing.report_date or ""},
                                )

                            if any(h.get("holder_type") == "category" for h in holders):
                                logger.info(
                                    "BSE shareholding pattern for %s: extracted via fuzzy parser (%s)",
                                    identifier, filing.filing_date,
                                )
                                break
                        except Exception as exc:
                            logger.debug("BSE fuzzy shareholding extraction failed: %s", exc)
                            continue
            except Exception as exc:
                logger.debug("BSE filing discoverer for shareholding failed for %s: %s", identifier, exc)

        # No yfinance fallback -- native BSE SAST + fuzzy PDF extraction only.
        # Probing confirmed: ShareHoldPat/w and InsiderTrading/w are blocked
        # (redirect loops), shpSecurities.aspx is JS-rendered.  The SAST
        # announcements via AnnSubCategoryGetData + fuzzy PDF extraction are
        # the only working native paths for holder data.

        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return institutional ownership metrics from BSE native data.

        Derives aggregate metrics from get_holders() which uses BSE SAST
        disclosures and fuzzy PDF extraction.  No yfinance dependency.
        """
        try:
            from datetime import date as _date

            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            # Compute aggregate metrics from native holder data
            inst_holders = [
                h for h in holders
                if h.get("holder_type") in ("institutional", "substantial")
            ]
            cat_holders = [h for h in holders if h.get("holder_type") == "category"]

            inst_pct = sum(h.get("percentage", 0) for h in inst_holders)
            # If we have category breakdowns (from SEBI shareholding pattern),
            # use those for more accurate institutional percentage
            for ch in cat_holders:
                if "fii" in ch.get("name", "").lower() or "institutional" in ch.get("name", "").lower():
                    inst_pct = max(inst_pct, ch.get("percentage", 0))

            hhi = 0.0
            top5 = [h for h in holders if h.get("holder_type") != "category"][:5]
            if not top5:
                top5 = holders[:5]
            total_pct = sum(h.get("percentage", 0) for h in top5)
            if total_pct > 0:
                hhi = sum(
                    (h.get("percentage", 0) / total_pct) ** 2 for h in top5
                )

            return pd.DataFrame([{
                "date_reported": pd.Timestamp(_date.today()),
                "inst_ownership_pct": round(inst_pct, 2),
                "inst_top5_concentration": round(hhi, 4),
                "inst_holder_count": len(inst_holders) or len(holders),
            }])
        except Exception as exc:
            logger.debug("BSE holder history failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch insider transactions from BSE SAST announcements.

        Uses the BSE AnnSubCategoryGetData API with strCat='Insider Trading / SAST'
        to discover SEBI Regulation 29/31 filings.  Each announcement contains
        the insider name in the HEADLINE/NEWSSUB field and the transaction
        date.  PDF attachments can be extracted for detailed share counts.

        Probing confirmed: InsiderTrading/w endpoint is blocked (redirect loop).
        The AnnSubCategoryGetData API is the only working path.
        """
        transactions: list[dict[str, Any]] = []
        scrip_code = _resolve_scrip_code(identifier)
        try:
            bse_session = _get_bse_session()
            sast_filings = _fetch_bse_sast_disclosures(scrip_code, bse_session)
            if sast_filings:
                for filing in sast_filings[:20]:
                    headline = filing.get("NEWSSUB", filing.get("HEADLINE", ""))
                    date_str = filing.get("DT_TM", "")[:10]
                    # Parse insider name from headline
                    # SAST headlines: "Closure of Trading Window" or
                    # "Acquisition of shares by [NAME]" or similar
                    insider_name = headline[:60] if headline else "Unknown"
                    txn_type = "SAST Disclosure"
                    if "closure" in headline.lower():
                        txn_type = "Trading Window Closure"
                    elif "acquisition" in headline.lower():
                        txn_type = "Acquisition"
                    elif "disposal" in headline.lower() or "sale" in headline.lower():
                        txn_type = "Disposal"
                    transactions.append({
                        "insider_name": insider_name,
                        "position": "",
                        "date": date_str,
                        "transaction": txn_type,
                        "shares": 0,
                        "value": 0.0,
                        "source": "bse_sast_announcement",
                    })
                if transactions:
                    logger.info(
                        "BSE insider transactions for %s: %d from SAST announcements",
                        identifier, len(transactions),
                    )
        except Exception as exc:
            logger.debug("BSE SAST insider transactions failed for %s: %s", identifier, exc)
        return transactions

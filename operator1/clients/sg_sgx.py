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

Holder data: SGX disclosure announcement scraping (native, no yfinance)
  - SGX Financial Reports API searched for Disclosure of Interest filings
  - Announcement pages scraped for shareholder names and percentages
  - Annual report PDFs parsed via fuzzy_pdf_parser for shareholding sections
  - Uses the HKEX scraper pattern: persistent session, paginated queries,
    client-side filtering by company name

OHLCV: handled separately via ohlcv_provider.py

Coverage: ~700+ listed companies on SGX, ~$0.6T market cap.
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


# ---------------------------------------------------------------------------
# SGX Disclosure of Interest scraper (HKEX scraper pattern)
# ---------------------------------------------------------------------------
#
# SGX's substantial shareholding disclosure data is not available via a
# free JSON API (the /securities/v1.1/substantial-shareholders endpoint
# returns 403).  Instead, we use the Financial Reports API to find
# disclosure announcements, then parse the announcement pages for
# shareholder names and percentages.
#
# This follows the same pattern as the HKEX scraper (hkex_scraper.py):
#   1. Persistent session for cookie reuse across requests
#   2. Date-windowed queries (SGX API has no date limit per query,
#      but we paginate to stay under the default pagesize)
#   3. Client-side filtering by company name
#   4. Rate limiting between requests

_sgx_session: requests.Session | None = None

# Maximum results per page for the financial reports API.
_SGX_REPORTS_PAGE_SIZE = 50

# Pause between API requests to avoid rate-limiting (SGX has no
# documented rate limit, but we stay polite).
_SGX_REQUEST_DELAY_S = 0.3

# Disclosure-related keywords in announcement titles.
_DOI_TITLE_KEYWORDS = (
    "disclosure",
    "substantial",
    "interest",
    "shareholding",
    "deemed interest",
    "notification of change",
)


def _get_sgx_session() -> requests.Session:
    """Create or return a persistent requests session for SGX API calls.

    Mirrors the HKEX scraper pattern: a persistent session reuses
    cookies and TCP connections across multiple API requests.
    """
    global _sgx_session
    if _sgx_session is None:
        _sgx_session = requests.Session()
        _sgx_session.headers.update(_SGX_HEADERS)
    return _sgx_session


def _resolve_company_name(identifier: str) -> str:
    """Resolve a ticker code to a company name for API filtering.

    The SGX financial reports API filters by ``companyname`` (substring
    match).  We look up the full company name from the securities
    directory so we can search for disclosure announcements.
    """
    directory = _get_securities_directory()
    if not directory:
        return identifier

    matches = _search_securities(identifier, directory, max_results=1)
    if matches:
        return matches[0].get("n", identifier)
    return identifier


def _fetch_sgx_disclosure_announcements(
    identifier: str,
    session: requests.Session | None = None,
    years: int = 2,
) -> list[dict[str, Any]]:
    """Search SGX Financial Reports API for Disclosure of Interest filings.

    Uses the HKEX scraper pattern:
      - Persistent session for cookie reuse
      - Paginated queries to the ``/financialreports/v1.0`` endpoint
      - Client-side filtering by company name + disclosure keywords
      - Rate limiting between requests

    Parameters
    ----------
    identifier:
        SGX ticker code (e.g. 'D05' for DBS).
    session:
        Optional pre-existing requests session.  If None, uses the
        module-level persistent session.
    years:
        How many years back to search.

    Returns
    -------
    List of holder dicts with keys: name, percentage, holder_type,
    date_reported, source, announcement_url.
    """
    if session is None:
        session = _get_sgx_session()

    company_name = _resolve_company_name(identifier)
    if not company_name or company_name == identifier:
        # Could not resolve -- try with the raw identifier
        company_name = identifier

    logger.info(
        "SGX DOI search: identifier=%s, company=%s, years=%d",
        identifier, company_name, years,
    )

    # Calculate the cutoff date for filtering by documentDate
    cutoff_ms = int(
        (date.today() - timedelta(days=365 * years)).strftime("%s")
    ) * 1000

    # Paginate through the financial reports API
    # (like HKEX's date-windowed iteration, but SGX uses page offsets)
    all_doi_reports: list[dict[str, Any]] = []
    page_start = 0
    request_count = 0

    while True:
        try:
            resp = session.get(
                f"{_SGX_BASE}/financialreports/v1.0",
                params={
                    "pagestart": str(page_start),
                    "pagesize": str(_SGX_REPORTS_PAGE_SIZE),
                    "companyname": company_name,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug(
                "SGX financial reports query failed at offset %d: %s",
                page_start, exc,
            )
            break

        reports = data.get("data", [])
        if not reports:
            break

        request_count += 1

        for report in reports:
            doc_date = report.get("documentDate", 0)
            if isinstance(doc_date, (int, float)) and doc_date < cutoff_ms:
                continue

            title = (report.get("title") or "").lower()
            # Check if this is a disclosure-related announcement
            if any(kw in title for kw in _DOI_TITLE_KEYWORDS):
                all_doi_reports.append(report)

        # Check if we've fetched all available reports
        meta = data.get("meta", {})
        total_items = meta.get("totalItems", 0)
        if page_start + _SGX_REPORTS_PAGE_SIZE >= total_items:
            break

        page_start += _SGX_REPORTS_PAGE_SIZE

        # Rate limiting (HKEX pattern)
        if request_count % 3 == 0:
            time.sleep(_SGX_REQUEST_DELAY_S)

    logger.info(
        "SGX DOI search for %s: %d disclosure filings found (%d API requests)",
        identifier, len(all_doi_reports), request_count,
    )

    if not all_doi_reports:
        return []

    # Parse disclosure announcements for shareholder data
    # Each announcement has a URL to the filing page on links.sgx.com
    holders: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for report in all_doi_reports:
        url = report.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        # Extract date from epoch milliseconds
        doc_date_ms = report.get("documentDate", 0)
        if isinstance(doc_date_ms, (int, float)) and doc_date_ms > 0:
            doc_date_str = date.fromtimestamp(doc_date_ms / 1000).isoformat()
        else:
            doc_date_str = ""

        title = report.get("title", "")
        company = report.get("companyName", "")

        # Try to scrape the announcement page for shareholder details
        # The links.sgx.com pages sometimes contain structured data
        # about the substantial shareholder (name, percentage, shares)
        try:
            page_holders = _scrape_sgx_announcement_page(
                url, session, doc_date_str, title,
            )
            holders.extend(page_holders)
        except Exception as exc:
            logger.debug("Failed to scrape SGX announcement %s: %s", url, exc)

        # Rate limiting between page scrapes
        time.sleep(_SGX_REQUEST_DELAY_S)

    # Dedup by shareholder name (keep the most recent)
    unique: dict[str, dict[str, Any]] = {}
    for h in holders:
        name_key = h["name"].lower().strip()
        existing = unique.get(name_key)
        if not existing or h.get("date_reported", "") > existing.get("date_reported", ""):
            unique[name_key] = h

    return list(unique.values())


def _scrape_sgx_announcement_page(
    url: str,
    session: requests.Session,
    doc_date: str,
    title: str,
) -> list[dict[str, Any]]:
    """Scrape a single SGX announcement page for shareholder data.

    SGX disclosure announcement pages on links.sgx.com contain
    metadata about the filing including the shareholder name and
    percentage.  Some pages also link to PDF documents with the
    full disclosure form.

    Uses the HKEX scraper pattern:
      - Same persistent session for cookie reuse
      - Parse HTML for structured data (tables, key-value pairs)
      - Extract shareholder names and percentages via regex

    Parameters
    ----------
    url:
        Full URL to the announcement page on links.sgx.com.
    session:
        Active requests session.
    doc_date:
        ISO date string for the filing date.
    title:
        Announcement title (used as fallback for shareholder name).

    Returns
    -------
    List of holder dicts extracted from the page.
    """
    holders: list[dict[str, Any]] = []

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            return []
        html = resp.text
    except Exception:
        return []

    # Strategy 1: Look for percentage patterns in the page text
    # SGX disclosure forms typically state:
    #   "Percentage of total number of voting shares: XX.XX%"
    #   "Name of Substantial Shareholder: XXXX"
    pct_patterns = [
        r"(?:percentage|percent)[^:]*?:\s*([\d.]+)\s*%",
        r"([\d.]+)\s*%\s*(?:of\s+(?:total|issued|voting))",
    ]
    name_patterns = [
        r"(?:name\s+of\s+(?:substantial\s+)?shareholder)[^:]*?:\s*([^\n<]{3,60})",
        r"(?:name\s+of\s+(?:registered\s+)?holder)[^:]*?:\s*([^\n<]{3,60})",
        r"(?:name\s+of\s+(?:notifying\s+)?(?:party|person))[^:]*?:\s*([^\n<]{3,60})",
    ]

    # Clean HTML tags for text pattern matching
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)

    found_names: list[str] = []
    found_pcts: list[float] = []

    for pattern in name_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            name = match.group(1).strip().strip(".,;:")
            if name and len(name) > 2 and not name.startswith("http"):
                found_names.append(name)

    for pattern in pct_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                pct = float(match.group(1))
                if 0.01 < pct < 100:
                    found_pcts.append(pct)
            except ValueError:
                pass

    # Strategy 2: Parse tables for structured shareholder data
    tables = re.findall(r"<table[^>]*>.*?</table>", html, re.DOTALL | re.IGNORECASE)
    for table in tables:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.DOTALL | re.IGNORECASE)
        for row in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
            cleaned = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            # Look for rows with shareholder name + percentage
            for i, cell in enumerate(cleaned):
                if "%" in cell:
                    try:
                        pct = float(cell.replace("%", "").strip())
                        if 0.01 < pct < 100 and i > 0:
                            name = cleaned[i - 1].strip()
                            if name and len(name) > 2:
                                holders.append({
                                    "name": name,
                                    "shares": 0,
                                    "percentage": round(pct, 2),
                                    "holder_type": "substantial",
                                    "date_reported": doc_date,
                                    "source": "sgx_doi",
                                    "announcement_url": url,
                                })
                    except ValueError:
                        pass

    # Combine regex-extracted names and percentages
    if found_names and found_pcts:
        # Pair up names with percentages (1:1 if counts match, else best effort)
        for i, name in enumerate(found_names):
            pct = found_pcts[i] if i < len(found_pcts) else 0.0
            # Skip if we already found this holder from table parsing
            if any(h["name"].lower() == name.lower() for h in holders):
                continue
            holders.append({
                "name": name,
                "shares": 0,
                "percentage": round(pct, 2),
                "holder_type": "substantial",
                "date_reported": doc_date,
                "source": "sgx_doi",
                "announcement_url": url,
            })
    elif found_names and not found_pcts:
        # Names found but no percentages -- still useful
        for name in found_names:
            if any(h["name"].lower() == name.lower() for h in holders):
                continue
            holders.append({
                "name": name,
                "shares": 0,
                "percentage": 0.0,
                "holder_type": "substantial",
                "date_reported": doc_date,
                "source": "sgx_doi",
                "announcement_url": url,
            })

    # Fallback: extract company name from announcement title
    if not holders and title:
        # Some disclosure titles include the shareholder name, e.g.:
        # "Disclosure of Interest - Temasek Holdings"
        title_parts = title.split(" - ")
        if len(title_parts) > 1:
            shareholder_name = title_parts[-1].strip()
            if shareholder_name and len(shareholder_name) > 2:
                holders.append({
                    "name": shareholder_name,
                    "shares": 0,
                    "percentage": 0.0,
                    "holder_type": "substantial",
                    "date_reported": doc_date,
                    "source": "sgx_doi_title",
                    "announcement_url": url,
                })

    return holders


def _parse_sgx_shareholding_text(
    text: str,
    holders: list[dict[str, Any]],
    date_reported: str = "",
) -> None:
    """Parse shareholding data from SGX annual report PDF text.

    SGX annual reports typically include "Statistics of Shareholdings"
    or "Analysis of Shareholdings" sections with:
    - Substantial shareholders (>5%) with name, shares, percentage
    - Shareholding distribution by size
    - Top 20 shareholders list

    IMPORTANT: This function first isolates the relevant shareholding
    section(s) before applying regex extraction.  Without section
    isolation, the loose regex patterns match random text throughout
    the 100+ page annual report (e.g. funding breakdowns, executive
    bios) producing garbage "holders" like "Wholesale funding 16%".
    """
    import re as _re

    # ------------------------------------------------------------------
    # Step 1: Isolate the shareholding section(s) from the full text.
    # We look for known section headings and extract a bounded window
    # of text (~3000 chars) after each heading.  This prevents the
    # regex from matching random percentages in unrelated sections.
    # ------------------------------------------------------------------
    section_headings = [
        r"substantial\s+shareholder",
        r"statistics\s+of\s+shareholding",
        r"analysis\s+of\s+shareholding",
        r"twenty\s+largest\s+shareholder",
        r"top\s+20\s+shareholder",
    ]

    sections: list[str] = []
    for heading in section_headings:
        for m in _re.finditer(heading, text, _re.IGNORECASE):
            start = m.start()
            # Extract up to 3000 chars after the heading -- enough for
            # the substantial shareholder table but not so much that we
            # bleed into the next unrelated section.
            end = min(start + 3000, len(text))
            sections.append(text[start:end])

    if not sections:
        # No recognised section headings found -- skip entirely rather
        # than matching against the full document (which produces garbage).
        logger.debug("SGX PDF: no shareholding section headings found")
        return

    section_text = "\n".join(sections)

    # ------------------------------------------------------------------
    # Step 2: Extract substantial shareholder data from isolated sections.
    # Patterns are tuned for SGX annual report format:
    #   "Maju Holdings Pte. Ltd.  484,789,855  –  484,789,855  17.08"
    #   "Temasek Holdings  312,559,831  490,413,019  802,972,850  28.30"
    # ------------------------------------------------------------------

    # Skip words that indicate table headers, labels, or non-name text
    _SKIP_WORDS = {
        "name", "shareholder", "interest", "total", "percentage",
        "direct", "deemed", "number", "class", "shares", "ordinary",
        "size", "shareholdings", "location", "singapore", "malaysia",
        "overseas", "voting", "rights", "treasury", "issued",
        "excluding", "based", "calculated", "pursuant", "section",
        "subsidiary", "wholly", "owned", "note", "the", "and",
        "for", "with", "from", "this", "that", "which", "are",
        "was", "were", "has", "have", "not", "but", "all",
    }

    def _is_valid_name(name: str) -> bool:
        """Check if extracted text looks like a company/person name."""
        name_stripped = name.strip()
        if len(name_stripped) < 4:
            return False
        # Must contain at least one uppercase letter
        if not any(c.isupper() for c in name_stripped):
            return False
        # First word should not be a skip word
        first_word = name_stripped.split()[0].lower().rstrip(".,;:()")
        if first_word in _SKIP_WORDS:
            return False
        # Should look like an entity name (contains letters, not just numbers)
        alpha_count = sum(1 for c in name_stripped if c.isalpha())
        if alpha_count < 3:
            return False
        return True

    # Pattern: Name followed by share numbers and percentage
    # Matches lines like: "Maju Holdings Pte. Ltd. 484,789,855 – 484,789,855 17.08"
    pat_shares_pct = _re.compile(
        r"^([A-Z][A-Za-z\s&.,()'\-/]+?)"  # company name (starts with uppercase)
        r"\s+"
        r"([\d,]+(?:\s+[\d,–\-]+)*)"       # one or more share number columns
        r"\s+"
        r"(\d{1,3}\.\d{1,2})"              # percentage (e.g. 17.08, 28.30)
        r"\s*$",
        _re.MULTILINE,
    )

    for m in pat_shares_pct.finditer(section_text):
        name = m.group(1).strip().rstrip(".,;:")
        pct_str = m.group(3)
        try:
            pct = float(pct_str)
        except ValueError:
            continue
        if not (0.5 < pct < 100):
            continue
        if not _is_valid_name(name):
            continue
        # Try to extract the last (total) share count
        shares = 0
        shares_parts = m.group(2).replace("–", "").replace("-", "").split()
        for sp in reversed(shares_parts):
            sp_clean = sp.replace(",", "").strip()
            if sp_clean.isdigit():
                shares = int(sp_clean)
                break
        if not any(h["name"].lower() == name.lower() for h in holders):
            holders.append({
                "name": name,
                "shares": shares,
                "percentage": round(pct, 2),
                "holder_type": "substantial",
                "date_reported": date_reported,
                "source": "sgx_filing_pdf",
            })

    # Pattern for "Twenty largest shareholders" section
    # Format: "1 CITIBANK NOMINEES SINGAPORE  1,234,567  12.34"
    pat_top20 = _re.compile(
        r"^\s*(\d{1,2})\s+"                # rank number
        r"([A-Z][A-Za-z\s&.,()'\-/]+?)"    # shareholder name
        r"\s+"
        r"([\d,]+)"                         # shares
        r"\s+"
        r"(\d{1,3}\.\d{1,2})"              # percentage
        r"\s*$",
        _re.MULTILINE,
    )

    for m in pat_top20.finditer(section_text):
        rank = int(m.group(1))
        name = m.group(2).strip().rstrip(".,;:")
        shares_str = m.group(3).replace(",", "")
        pct_str = m.group(4)
        if rank > 20:
            continue
        try:
            pct = float(pct_str)
            shares = int(shares_str)
        except ValueError:
            continue
        if not (0.01 < pct < 100):
            continue
        if not _is_valid_name(name):
            continue
        if not any(h["name"].lower() == name.lower() for h in holders):
            holders.append({
                "name": name,
                "shares": shares,
                "percentage": round(pct, 2),
                "holder_type": "top20",
                "date_reported": date_reported,
                "source": "sgx_filing_pdf",
            })


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

        # No fallback -- SGX Securities API covers all listed stocks
        return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from SGX Securities API.

        Looks up the ticker in the securities directory for name, currency,
        and board. Falls back to OpenFIGI for sector/industry enrichment.
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

        # Supplement with OpenFIGI for sector/industry (SGX API doesn't provide these)
        if not raw.get("sector"):
            try:
                from operator1.clients.supplement import enrich_profile
                enriched = enrich_profile(
                    market_id=self.market_id,
                    ticker=identifier,
                    existing_profile=raw,
                )
                # Only take fields that SGX didn't provide
                for field in ("sector", "industry", "name", "market_cap", "isin"):
                    if enriched.get(field) and not raw.get(field):
                        raw[field] = enriched[field]
            except Exception as exc:
                logger.debug("OpenFIGI profile supplement failed for %s: %s", identifier, exc)

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

    # -- Institutional holders (SGX native -- DOI announcements + PDF) --------

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch substantial shareholders from SGX native disclosure data.

        SGX's substantial shareholding data (Section 137 of the Securities
        and Futures Act) is published as announcement PDFs via the financial
        reports API.  No yfinance dependency.

        Two-source native strategy:
        1. SGX Financial Reports API -> find Disclosure of Interest filings
           -> scrape announcement pages for shareholder names + percentages
        2. Annual report PDFs -> fuzzy_pdf_parser for "Statistics of
           Shareholdings" sections with substantial shareholder tables

        Uses the same session-based pattern as the HKEX scraper: persistent
        session for cookie reuse, paginated queries.
        """
        holders: list[dict[str, Any]] = []

        # PRIMARY: SGX Financial Reports API disclosure announcements
        # Uses the HKEX scraper pattern: session-based, paginated queries
        # to find Disclosure of Interest filings via the financial reports API
        try:
            doi_holders = _fetch_sgx_disclosure_announcements(identifier)
            if doi_holders:
                # Merge DOI holders -- they have shareholder names + percentages
                existing_names = {h["name"].lower() for h in holders}
                for dh in doi_holders:
                    if dh["name"].lower() not in existing_names:
                        holders.append(dh)
                        existing_names.add(dh["name"].lower())
                logger.info(
                    "SGX DOI announcements for %s: %d additional holders",
                    identifier, len(doi_holders),
                )
        except Exception as exc:
            logger.debug("SGX DOI scraping failed for %s: %s", identifier, exc)

        # SECONDARY: SGX filing discoverer -- two-stage PDF extraction.
        # Stage 1: Run fuzzy_pdf_parser for financial statement extraction
        #          (income, balance, cashflow -- same as before).
        # Stage 2: Extract shareholding data from pages that contain
        #          shareholding keywords (page-level isolation).
        #
        # Stage 2 avoids feeding the entire 100+ page annual report to the
        # shareholding regex, which previously produced garbage matches from
        # unrelated sections (e.g. "Wholesale funding 16%").
        if not holders:
            try:
                from operator1.clients.filing_discoverer import SGXFilingDiscoverer
                from operator1.clients.fuzzy_pdf_parser import extract_financials_from_pdf
                discoverer = SGXFilingDiscoverer()
                discovery = discoverer.discover_filings(identifier, years=1)

                if discovery.has_filings:
                    for filing in discovery.filings[:3]:
                        try:
                            pdf_bytes = discoverer.download_filing(filing)
                            if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
                                continue

                            # --- Stage 1: Financial statement extraction ---
                            # Run the fuzzy PDF parser for structured financial
                            # data (income, balance, cashflow).  This is the
                            # same extraction that _fetch_financials() uses.
                            try:
                                extract_financials_from_pdf(
                                    pdf_bytes,
                                    filing_date=filing.filing_date or "",
                                    report_date=filing.report_date or "",
                                    statement_type="balance",
                                )
                            except Exception as exc:
                                logger.debug("SGX financial extraction from PDF failed: %s", exc)

                            # --- Stage 2: Shareholding extraction ---
                            # Primary: fuzzy parser's extract_shareholders_from_pdf
                            # (camelot table extraction + page scoring + column ID).
                            # Fallback: SGX-specific regex parser for formats the
                            # generic table parser may miss (substantial shareholder
                            # tables + top-20 lists in SGX annual reports).
                            try:
                                from operator1.clients.fuzzy_pdf_parser import extract_shareholders_from_pdf
                                _fuzzy_holders = extract_shareholders_from_pdf(
                                    pdf_bytes,
                                    filing_date=filing.filing_date or "",
                                    market_id="sg_sgx",
                                )
                                for fh in _fuzzy_holders:
                                    holders.append(fh)
                            except Exception as _fpe:
                                logger.debug("SGX fuzzy shareholder extraction failed: %s", _fpe)

                            # Fallback: SGX custom regex parser if fuzzy parser
                            # found nothing (handles SGX-specific text layouts).
                            if not holders:
                                try:
                                    import pdfplumber as _pdfp, io as _io
                                    _SH_KEYWORDS = (
                                        "substantial shareholder",
                                        "statistics of shareholding",
                                        "analysis of shareholding",
                                        "twenty largest shareholder",
                                        "top 20 shareholder",
                                    )
                                    relevant_pages: list[str] = []
                                    with _pdfp.open(_io.BytesIO(pdf_bytes)) as pdf:
                                        for page in pdf.pages:
                                            page_text = page.extract_text() or ""
                                            page_lower = page_text.lower()
                                            if any(kw in page_lower for kw in _SH_KEYWORDS):
                                                relevant_pages.append(page_text)

                                    if relevant_pages:
                                        text = "\n".join(relevant_pages)
                                        _parse_sgx_shareholding_text(
                                            text, holders, filing.filing_date or "",
                                        )
                                except Exception as _regex_exc:
                                    logger.debug("SGX regex shareholding fallback failed: %s", _regex_exc)

                            if holders:
                                logger.info(
                                    "SGX shareholding for %s: %d holders from %d relevant pages (%s)",
                                    identifier, len(holders), len(relevant_pages),
                                    filing.filing_date,
                                )
                                break
                        except Exception as exc:
                            logger.debug("SGX fuzzy shareholding extraction failed: %s", exc)
                            continue
            except Exception as exc:
                logger.debug("SGX filing discoverer for shareholding failed for %s: %s", identifier, exc)

        # Fallback: extract shareholders from filing PDFs
        if not holders:
            try:
                from operator1.clients.filing_discoverer import try_shareholding_extraction
                holders = try_shareholding_extraction(identifier, market_id=self.market_id)
                if holders:
                    logger.info("SGX holders from PDF shareholding extraction: %d", len(holders))
            except Exception as exc:
                logger.debug("SGX PDF shareholding fallback failed: %s", exc)

        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return institutional ownership metrics from native SGX data.

        Derives aggregate metrics from get_holders() which uses SGX DOI
        announcements and annual report PDF extraction.  No yfinance dependency.

        Returns a single-row snapshot (same pattern as HKHkexClient).
        """
        try:
            from datetime import date as _date

            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            inst_holders = [h for h in holders if h.get("holder_type") in ("substantial", "institutional", "top20")]
            inst_pct = sum(h.get("percentage", 0) for h in inst_holders)

            hhi = 0.0
            top5 = inst_holders[:5]
            total_pct = sum(h.get("percentage", 0) for h in top5)
            if total_pct > 0:
                hhi = sum((h.get("percentage", 0) / total_pct) ** 2 for h in top5)

            return pd.DataFrame([{
                "date_reported": pd.Timestamp(_date.today()),
                "inst_ownership_pct": round(inst_pct, 2),
                "inst_top5_concentration": round(hhi, 4),
                "inst_holder_count": len(inst_holders),
            }])
        except Exception as exc:
            logger.debug("SGX holder history failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch director/board announcements from SGX Financial Reports API.

        Uses the SGX Financial Reports API to find board-related
        announcements (director appointments, resignations, dealings).
        Returns basic filing metadata; detailed extraction requires LLM.

        No yfinance dependency.
        """
        transactions: list[dict[str, Any]] = []
        try:
            session = _get_sgx_session()
            company_name = _resolve_company_name(identifier)

            # Search for director-related reports via financial reports API
            resp = session.get(
                f"{_SGX_BASE}/financialreports/v1.0",
                params={
                    "pagestart": "0",
                    "pagesize": "50",
                    "companyname": company_name,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                return transactions

            data = resp.json()
            reports = data.get("data", [])

            _INSIDER_KEYWORDS = (
                "director", "appointment", "resignation", "cessation",
                "change of", "dealings", "interested person",
            )

            for report in reports:
                title = (report.get("title") or "").lower()
                if not any(kw in title for kw in _INSIDER_KEYWORDS):
                    continue

                doc_date_ms = report.get("documentDate", 0)
                if isinstance(doc_date_ms, (int, float)) and doc_date_ms > 0:
                    doc_date_str = date.fromtimestamp(doc_date_ms / 1000).isoformat()
                else:
                    doc_date_str = ""

                transactions.append({
                    "insider_name": report.get("title", "Unknown"),
                    "position": "",
                    "date": doc_date_str,
                    "transaction": "Disclosure",
                    "shares": 0,
                    "value": 0.0,
                    "source": "sgx_financial_reports",
                    "announcement_url": report.get("url", ""),
                })

            if transactions:
                logger.info(
                    "SGX insider/director announcements for %s: %d from native API",
                    identifier, len(transactions),
                )
        except Exception as exc:
            logger.debug(
                "SGX insider transaction search failed for %s: %s",
                identifier, exc,
            )
        return transactions

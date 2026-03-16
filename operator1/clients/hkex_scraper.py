"""HKEX filing scraper using date-windowed queries to the JSON API.

HKEX News (www1.hkexnews.hk) publishes all listed company announcements
including financial results.  The search page exposes an undocumented
JSON API endpoint (``titleSearchServlet.do``) that returns filing records.

**Key discovery:** The HKEX API imposes a server-side date range limit
of approximately 2 weeks per query.  Requests spanning more than ~15
days return ``recordCnt: 0`` and ``result: "null"``.  This is NOT a
geo-blocking issue -- the API works from any IP address as long as the
date window is kept small.

**How it works:**
    1. GET the search page to create a session (JSESSIONID cookie).
    2. Issue multiple GET requests to the JSON API endpoint, each
       covering a 2-week window, with ``searchType=1`` and a title
       filter (e.g. ``"results"``) to narrow results to financial
       announcements.
    3. Filter results client-side by stock code.
    4. Aggregate and deduplicate across all windows.

**No proxy required.  No JSF session binding required.**

**API quirks:**
    - ``stockId=-1`` fetches ALL stocks; filtering is done client-side.
    - ``searchType=1`` enables title search (much more efficient).
    - The ``result`` field in the response is a JSON *string* (not an
      array), so it needs ``json.loads()`` after ``response.json()``.
    - ``lang=E`` (not ``EN``) for the API endpoint.
    - Date format is ``YYYYMMDD``.

No extra dependencies required -- just ``requests`` (already in
requirements).

Usage:
    scraper = HKEXScraper()
    filings = scraper.search_filings("00700", years=2)
    pdf_bytes = scraper.download_pdf(filings[0].document_url)

References:
    github.com/simonplmak-cloud/hkex-filing-scraper (MIT license)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HKEX URL constants
# ---------------------------------------------------------------------------

_HKEX_BASE_URL = "https://www1.hkexnews.hk"
_HKEX_SEARCH_PAGE = f"{_HKEX_BASE_URL}/search/titlesearch.xhtml"
_HKEX_API_ENDPOINT = f"{_HKEX_BASE_URL}/search/titleSearchServlet.do"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Maximum date range (days) per API request.  The HKEX server silently
# rejects ranges wider than ~15 days by returning recordCnt=0.
_MAX_WINDOW_DAYS = 14

# Pause between API requests to avoid rate-limiting.
_REQUEST_DELAY_S = 0.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HKEXFiling:
    """A single filing extracted from HKEX search results."""
    title: str = ""
    stock_code: str = ""
    stock_name: str = ""
    release_date: str = ""      # YYYY-MM-DD
    document_url: str = ""
    document_format: str = "pdf"
    filing_type: str = ""       # annual, interim, quarterly
    report_date: str = ""       # fiscal period end date (parsed from title)


# ---------------------------------------------------------------------------
# Title parsing helpers
# ---------------------------------------------------------------------------

def _parse_report_date(title: str) -> str:
    """Extract fiscal period end date from filing title.

    HKEX filing titles typically include the period end date in one of
    these formats:
        "Annual Results for the Year Ended 31 December 2024"
        "Interim Results for the Six Months Ended June 30, 2024"
    """
    patterns = [
        r"[Ee]nded\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        r"[Ee]nded\s+(\w+)\s+(\d{1,2}),?\s+(\d{4})",
    ]
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            groups = match.groups()
            if groups[1].lower() in months:
                return f"{groups[2]}-{months[groups[1].lower()]}-{int(groups[0]):02d}"
            elif groups[0].lower() in months:
                return f"{groups[2]}-{months[groups[0].lower()]}-{int(groups[1]):02d}"
    return ""


def _classify_filing_type(title: str) -> str:
    """Classify filing as annual, interim, or quarterly from its title."""
    lower = title.lower()
    if "annual" in lower or "year ended" in lower:
        return "annual"
    if "interim" in lower or "half" in lower or "six months" in lower:
        return "interim"
    if "quarter" in lower or "three months" in lower:
        return "quarterly"
    return "annual"


def _parse_api_record(record: dict) -> HKEXFiling:
    """Convert a raw HKEX API JSON record into an HKEXFiling object.

    The API returns records with these fields (among others)::

        {
            "NEWS_ID": "12022263",
            "STOCK_NAME": "TENCENT",
            "STOCK_CODE": "00700",
            "TITLE": "Annual Results ...",
            "FILE_TYPE": "PDF",
            "DATE_TIME": "11/02/2026 19:10",
            "FILE_LINK": "/listedco/listconews/sehk/2026/0211/....pdf",
            "LONG_TEXT": "Announcements and Notices - [Results]",
            "TOTAL_COUNT": "14957"
        }
    """
    # Parse release date from DD/MM/YYYY HH:MM format
    date_time = record.get("DATE_TIME", "")
    date_part = date_time.split(" ")[0] if date_time else ""
    release_date = ""
    if date_part and "/" in date_part:
        parts = date_part.split("/")
        if len(parts) == 3:
            release_date = f"{parts[2]}-{parts[1]}-{parts[0]}"

    # Stock code may contain <br/> for multi-code filings
    raw_code = record.get("STOCK_CODE", "")
    raw_code = raw_code.split("<br/>")[0].strip()

    raw_name = record.get("STOCK_NAME", "")
    raw_name = raw_name.split("<br/>")[0].strip()

    # Build full document URL from relative path
    file_link = record.get("FILE_LINK", "")
    if file_link and file_link.startswith("/"):
        file_link = _HKEX_BASE_URL + file_link

    # Clean HTML entities from title
    title = record.get("TITLE", "")
    title = title.replace("&#x3b;", ";").replace("&amp;", "&")

    # Determine document format
    doc_format = "pdf"
    file_type = record.get("FILE_TYPE", "").upper()
    if file_type in ("HTM", "HTML"):
        doc_format = "html"
    elif file_link.endswith((".xlsx", ".xls")):
        doc_format = "excel"

    return HKEXFiling(
        title=title.strip(),
        stock_code=raw_code,
        stock_name=raw_name.strip(),
        release_date=release_date,
        document_url=file_link,
        document_format=doc_format,
        filing_type=_classify_filing_type(title),
        report_date=_parse_report_date(title),
    )


def _is_financial_filing(record: dict, filing: HKEXFiling) -> bool:
    """Check whether a record is a financial result filing."""
    long_text = record.get("LONG_TEXT", "").lower()
    title_lower = filing.title.lower()
    return bool(
        ("result" in title_lower
         or "financial" in title_lower
         or "annual report" in title_lower
         or "[results]" in long_text)
        and filing.document_url
    )


# ---------------------------------------------------------------------------
# Main scraper class
# ---------------------------------------------------------------------------

class HKEXScraper:
    """Scrapes HKEX filing search results using date-windowed API queries.

    The HKEX JSON API limits each request to approximately 2 weeks of
    data.  This scraper automatically splits a multi-year search into
    2-week windows, using title filters to keep result counts manageable.

    No proxy or JSF session binding is required.
    """

    def __init__(self) -> None:
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        """Create or return a persistent requests session.

        A persistent session reuses the JSESSIONID cookie from the
        initial search page load, which the API endpoint requires.
        """
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(_HEADERS)
            # Load the search page to initialize the server session
            try:
                self._session.get(_HKEX_SEARCH_PAGE, timeout=30)
            except Exception as exc:
                logger.warning("HKEX: failed to load search page: %s", exc)
        return self._session

    def _query_window(
        self,
        session: requests.Session,
        from_yyyymmdd: str,
        to_yyyymmdd: str,
        title_filter: str = "results",
    ) -> list[dict]:
        """Query a single date window and return raw API records.

        Parameters
        ----------
        session:
            Active requests session with JSESSIONID.
        from_yyyymmdd:
            Start date in YYYYMMDD format.
        to_yyyymmdd:
            End date in YYYYMMDD format.
        title_filter:
            Title search string to narrow results (e.g. "results",
            "annual results").  Use empty string for no filter.

        Returns
        -------
        List of raw record dicts from the HKEX API.
        """
        try:
            resp = session.get(
                _HKEX_API_ENDPOINT,
                params={
                    "sortDir": "0",
                    "sortByOptions": "DateTime",
                    "category": "0",
                    "market": "SEHK",
                    "stockId": "-1",
                    "documentType": "-1",
                    "fromDate": from_yyyymmdd,
                    "toDate": to_yyyymmdd,
                    "title": title_filter,
                    "searchType": "1" if title_filter else "0",
                    "t1code": "-2",
                    "t2Gcode": "-2",
                    "t2code": "-2",
                    "rowRange": "5000",
                    "lang": "E",
                },
                headers={
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "Referer": _HKEX_SEARCH_PAGE,
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug(
                "HKEX API query failed for %s-%s: %s",
                from_yyyymmdd, to_yyyymmdd, exc,
            )
            return []

        result_raw = data.get("result", "null")
        if result_raw is None or result_raw == "null":
            return []

        if isinstance(result_raw, str):
            try:
                parsed = json.loads(result_raw)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []

        if isinstance(result_raw, list):
            return result_raw

        return []

    def search_filings(
        self,
        stock_code: str,
        years: int = 2,
    ) -> list[HKEXFiling]:
        """Search HKEX for financial result filings for a given stock.

        Automatically paginates across 2-week date windows to cover the
        requested time range, using title filters to keep each window's
        result count within the API's limits.

        Parameters
        ----------
        stock_code:
            HKEX stock code (e.g. '700', '00700', '5').
            Internally normalized to 5-digit zero-padded format.
        years:
            How many years back to search.

        Returns
        -------
        List of HKEXFiling objects for financial result announcements,
        sorted by release date (newest first).
        """
        code = stock_code.split(".")[0].strip().zfill(5)

        session = self._get_session()

        today = date.today()
        search_start = today - timedelta(days=365 * years)

        # Iterate backward in 2-week windows with title filter
        all_filings: list[HKEXFiling] = []
        current_end = today
        window = timedelta(days=_MAX_WINDOW_DAYS)
        request_count = 0

        while current_end > search_start:
            current_start = max(current_end - window, search_start)
            from_str = current_start.strftime("%Y%m%d")
            to_str = current_end.strftime("%Y%m%d")

            records = self._query_window(session, from_str, to_str, "results")
            request_count += 1

            for record in records:
                raw_code = record.get("STOCK_CODE", "").split("<br/>")[0].strip()
                if raw_code != code:
                    continue
                filing = _parse_api_record(record)
                if _is_financial_filing(record, filing):
                    all_filings.append(filing)

            # Move to next window (1-day gap to avoid overlap)
            current_end = current_start - timedelta(days=1)

            # Rate limiting
            if request_count % 5 == 0:
                time.sleep(_REQUEST_DELAY_S)

        # Dedup by document URL
        seen: set[str] = set()
        unique: list[HKEXFiling] = []
        for f in all_filings:
            if f.document_url not in seen:
                seen.add(f.document_url)
                unique.append(f)

        # Sort by release date (newest first)
        unique.sort(key=lambda f: f.release_date, reverse=True)

        logger.info(
            "HKEX search for %s: %d financial result filings "
            "(%d API requests across %d-year window)",
            code, len(unique), request_count, years,
        )
        return unique

    def download_pdf(self, url: str) -> bytes:
        """Download a filing PDF from HKEX.

        HKEX documents are hosted as static files and do not require
        session authentication -- a direct GET with standard headers
        works from any IP address.

        Parameters
        ----------
        url:
            Full URL to the PDF document on www1.hkexnews.hk.

        Returns
        -------
        Raw PDF bytes.

        Raises
        ------
        ValueError:
            If the downloaded content is not a valid PDF.
        """
        session = self._get_session()
        resp = session.get(url, timeout=60)
        resp.raise_for_status()

        if resp.content[:4] != b"%PDF":
            raise ValueError(
                f"Expected PDF but got {resp.headers.get('Content-Type', 'unknown')}"
            )

        return resp.content

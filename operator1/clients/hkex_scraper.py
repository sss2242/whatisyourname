"""HKEX filing scraper using the undocumented JSON API.

HKEX's filing search (www1.hkexnews.hk/search/titlesearch.xhtml) is a
JSF (JavaServer Faces) application.  Direct API calls return empty
results unless the server session is initialized correctly.

This module uses the approach discovered by the hkex-filing-scraper
project (MIT, github.com/simonplmak-cloud/hkex-filing-scraper):

**Why this is necessary:**
    HKEX has no public REST API for filing search.  The titlesearch.xhtml
    page is a JSF app that renders results via JavaScript.  Direct GET/POST
    calls to the API endpoint return ``recordCnt: 0`` unless the JSF
    server session has been properly initialized with date range parameters.

**How the undocumented API works:**
    1. GET the search page to create a JSF session (gets a JSESSIONID cookie)
       and extract the ViewState token from the HTML.
    2. POST the JSF form with ViewState + date range to "bind" the search
       parameters to the server-side session.
    3. GET the JSON API endpoint (titleSearchServlet.do) with the date range
       params.  The server now recognizes the session and returns real results.

**Important API quirks (learned from hkex-filing-scraper):**
    - The API uses ``stockId=-1`` to fetch ALL stocks (not a specific code).
      Filtering by stock code is done client-side after retrieval.
    - The ``rowRange`` parameter is cumulative: to paginate, increase it
      (e.g., 5000, 10000, 15000) rather than using offset/limit.
    - Date format in the POST form is ``YYYYMMDD``, and ``from``/``to`` are
      the field names (not ``startDate``/``endDate``).
    - The response ``result`` field is a JSON string (not a JSON array),
      so it needs ``json.loads()`` after ``json()`` parsing.
    - ``lang=E`` (not ``EN``) for the API endpoint.

**No extra dependencies required** -- just ``requests`` and optionally
``beautifulsoup4`` (both already in requirements).

Usage:
    scraper = HKEXAPIScraper()
    filings = scraper.search_filings("00700", years=2)
    pdf_bytes = scraper.download_pdf(filings[0].document_url)

References:
    github.com/simonplmak-cloud/hkex-filing-scraper (MIT license)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _get_hkex_proxy() -> dict[str, str] | None:
    """Read HKEX proxy from environment.

    HKEX geo-blocks non-HK IPs.  Users can set ``HKEX_PROXY`` in their
    ``.env`` file to route HKEX requests through an HK-based proxy.

    Supported formats::

        HKEX_PROXY=http://host:port
        HKEX_PROXY=socks5://host:port
        HKEX_PROXY=socks5://user:pass@host:port

    Returns a ``requests``-compatible proxies dict, or None.
    """
    proxy = os.environ.get("HKEX_PROXY", "").strip()
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}

# ---------------------------------------------------------------------------
# HKEX URL constants
# ---------------------------------------------------------------------------
# The base URL for all HKEX News endpoints.
_HKEX_BASE_URL = "https://www1.hkexnews.hk"

# The JSF search page -- loading this creates the server-side session.
_HKEX_SEARCH_PAGE = f"{_HKEX_BASE_URL}/search/titlesearch.xhtml"

# The undocumented JSON API endpoint that returns filing records.
# Only works after the session has been initialized via the search page.
_HKEX_API_ENDPOINT = f"{_HKEX_BASE_URL}/search/titleSearchServlet.do"

# Browser-like headers to avoid being blocked by HKEX's WAF.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


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


# ---------------------------------------------------------------------------
# Main scraper class
# ---------------------------------------------------------------------------

class HKEXAPIScraper:
    """Scrapes HKEX filing search results using the undocumented JSON API.

    See module docstring for a detailed explanation of why this approach
    is necessary and how the three-step session initialization works.
    """

    def __init__(self) -> None:
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        """Create or return a persistent requests session.

        A persistent session is needed because the HKEX API requires
        the JSESSIONID cookie from the initial page load to be sent
        with subsequent API calls.

        If ``HKEX_PROXY`` is set in the environment, the session routes
        all traffic through that proxy (needed because HKEX geo-blocks
        non-HK IP addresses).

        If ``HKEX_PROXY_NO_VERIFY=1`` is also set, SSL certificate
        verification is disabled.  This is needed for transparent/
        intercepting proxies that replace the server certificate.
        Only use this for reading public filing data -- never for
        endpoints that transmit credentials.
        """
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(_HEADERS)
            proxies = _get_hkex_proxy()
            if proxies:
                self._session.proxies.update(proxies)
                logger.info("HKEX scraper using proxy: %s", proxies.get("https", ""))
                if os.environ.get("HKEX_PROXY_NO_VERIFY", "").strip() in ("1", "true", "yes"):
                    self._session.verify = False
                    logger.warning("HKEX scraper: SSL verification disabled (HKEX_PROXY_NO_VERIFY=1)")
                    # Suppress InsecureRequestWarning
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return self._session

    def _init_jsf_session(
        self,
        session: requests.Session,
        from_yyyymmdd: str,
        to_yyyymmdd: str,
    ) -> bool:
        """Initialize the HKEX JSF server session with date range.

        This is the critical step that makes the JSON API work:
        1. GET the search page (creates JSESSIONID cookie + ViewState)
        2. POST the JSF form (binds date range to the server session)

        After this, the JSON API endpoint recognizes the session and
        returns real results instead of empty arrays.

        Returns True if initialization succeeded, False otherwise.
        """
        # Step 1: GET the search page to create the JSF session.
        # The params here set default search options on the server side.
        try:
            page_resp = session.get(
                _HKEX_SEARCH_PAGE,
                params={
                    "sortDir": "0",
                    "sortByRecordDate": "on",
                    "searchType": "0",
                    "category": "0",
                    "t1code": "-2",
                    "t2Gcode": "-2",
                    "t2code": "-2",
                    "documentType": "-1",
                    "rowRange": "0",
                    "lang": "EN",
                },
                timeout=30,
            )
            page_resp.raise_for_status()
        except Exception as exc:
            logger.warning("HKEX: failed to load search page: %s", exc)
            return False

        # Extract the JSF ViewState token, form action URL, and form ID
        # from HTML.  ViewState is required for the JSF form POST to be
        # accepted.  The form ID is auto-generated by JSF (e.g. j_idt10,
        # j_idt15) and can change between server deployments, so we must
        # extract it dynamically rather than hardcoding it.
        view_state = ""
        form_action = ""
        form_id = ""

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(page_resp.text, "html.parser")
            vs_el = soup.find("input", {"name": "javax.faces.ViewState"})
            if vs_el:
                view_state = vs_el.get("value", "")
            form_el = soup.find("form")
            if form_el:
                form_action = form_el.get("action", "")
                form_id = form_el.get("id", "")
        except ImportError:
            # Fallback: regex extraction if beautifulsoup4 not available
            vs_match = re.search(
                r'javax\.faces\.ViewState.*?value="([^"]+)"', page_resp.text
            )
            if vs_match:
                view_state = vs_match.group(1)
            fa_match = re.search(r'<form[^>]*action="([^"]+)"', page_resp.text)
            if fa_match:
                form_action = fa_match.group(1)
            fid_match = re.search(r'<form[^>]*id="([^"]+)"', page_resp.text)
            if fid_match:
                form_id = fid_match.group(1)

        if not view_state:
            logger.warning("HKEX: could not extract ViewState from search page")
            return False

        if not form_id:
            logger.warning("HKEX: could not extract form ID from search page")
            return False

        # Step 2: POST the JSF form to bind the date range to the session.
        # The hkex-filing-scraper project discovered that the form POST
        # field names are "from" and "to" (YYYYMMDD format), and the
        # form ID must match the JSF-generated id attribute on the <form>
        # element (previously j_idt10, now dynamically extracted).
        submit_url = (
            f"{_HKEX_BASE_URL}{form_action}"
            if form_action.startswith("/")
            else _HKEX_SEARCH_PAGE
        )

        try:
            session.post(
                submit_url,
                data={
                    form_id: form_id,
                    f"{form_id}:loadMoreRange": "100",
                    "javax.faces.ViewState": view_state,
                    "from": from_yyyymmdd,
                    "to": to_yyyymmdd,
                },
                timeout=30,
            )
        except Exception as exc:
            logger.warning("HKEX: JSF form POST failed: %s", exc)
            return False

        return True

    def search_filings(
        self,
        stock_code: str,
        years: int = 2,
    ) -> list[HKEXFiling]:
        """Search HKEX for financial result filings for a given stock.

        Parameters
        ----------
        stock_code:
            HKEX stock code (e.g. '700', '00700', '5').
            Internally normalized to 5-digit zero-padded format.
        years:
            How many years back to search.

        Returns
        -------
        List of HKEXFiling objects for financial result announcements.
        """
        # Normalize stock code to 5-digit zero-padded (HKEX standard)
        code = stock_code.split(".")[0].strip().zfill(5)

        today = date.today()
        from_date = today - timedelta(days=365 * years)
        from_str = from_date.strftime("%Y%m%d")
        to_str = today.strftime("%Y%m%d")

        session = self._get_session()

        # Initialize the JSF session with our date range.
        # This is the step that was missing from our earlier direct API calls.
        if not self._init_jsf_session(session, from_str, to_str):
            return []

        # Step 3: Call the JSON API endpoint.
        # Key findings from hkex-filing-scraper:
        #   - Use stockId=-1 to fetch ALL stocks (client-side filter later)
        #   - Use lang=E (not EN) for the API endpoint
        #   - rowRange is cumulative (not page size)
        #   - The "result" field is a JSON *string*, not an array
        try:
            api_resp = session.get(
                _HKEX_API_ENDPOINT,
                params={
                    "sortDir": "0",
                    "sortByOptions": "DateTime",
                    "category": "0",
                    "market": "SEHK",
                    "stockId": "-1",       # All stocks, filter client-side
                    "documentType": "-1",
                    "fromDate": from_str,
                    "toDate": to_str,
                    "title": "",           # No title filter (get everything)
                    "searchType": "0",
                    "t1code": "-2",
                    "t2Gcode": "-2",
                    "t2code": "-2",
                    "rowRange": "5000",    # Fetch up to 5000 records
                    "lang": "E",           # Note: "E" not "EN"
                },
                headers={
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "Referer": _HKEX_SEARCH_PAGE,
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=120,
            )
            api_resp.raise_for_status()
            data = api_resp.json()
        except Exception as exc:
            logger.warning("HKEX: JSON API call failed: %s", exc)
            return []

        # Parse the result -- it's a JSON string inside the JSON response.
        # data["result"] is a string like '[{"STOCK_CODE":"00700",...},...]'
        # but may also be "null", None, or "[]" when no results found.
        result_raw = data.get("result", "[]")
        logger.debug("HKEX API raw keys: %s", list(data.keys()))
        logger.debug("HKEX API result type: %s, first 200 chars: %s",
                      type(result_raw).__name__,
                      str(result_raw)[:200] if result_raw else "None")

        records: list = []
        if result_raw is None or result_raw == "null":
            records = []
        elif isinstance(result_raw, str):
            try:
                parsed = json.loads(result_raw)
                records = parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                records = []
        elif isinstance(result_raw, list):
            records = result_raw

        record_count = data.get("recordCnt", 0)
        logger.info(
            "HKEX API returned %d records (server says %d total)",
            len(records), record_count,
        )

        # Filter to our target stock code and financial results only.
        # The API returns ALL stocks' filings; we filter client-side.
        all_filings: list[HKEXFiling] = []
        for record in records:
            raw_code = record.get("STOCK_CODE", "").split("<br/>")[0].strip()
            if raw_code != code:
                continue

            filing = _parse_api_record(record)

            # Keep only financial result filings (skip corporate actions, etc.)
            long_text = record.get("LONG_TEXT", "").lower()
            title_lower = filing.title.lower()
            is_financial = (
                "result" in title_lower
                or "financial" in title_lower
                or "annual report" in title_lower
                or "[results]" in long_text
            )
            if is_financial and filing.document_url:
                all_filings.append(filing)

        # Dedup by document URL
        seen: set[str] = set()
        unique: list[HKEXFiling] = []
        for f in all_filings:
            if f.document_url not in seen:
                seen.add(f.document_url)
                unique.append(f)

        logger.info(
            "HKEX search for %s: %d financial result filings (from %d total records)",
            code, len(unique), len(records),
        )
        return unique

    def download_pdf(self, url: str) -> bytes:
        """Download a filing PDF from HKEX.

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

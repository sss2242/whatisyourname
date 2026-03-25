"""Filing discoverer framework for Tier 2 markets.

Discovers financial filing URLs from exchange announcement systems so the
LLMFilingExtractor can download and extract structured data from them.

Each exchange has a different announcement API. The FilingDiscoverer protocol
defines the common interface; per-market implementations handle the specifics.

Currently implemented:
  - BSEFilingDiscoverer (India) -- full pipeline: discovery + PDF download
  - ASXFilingDiscoverer (Australia) -- announcement discovery via MarkitDigital
  - HKEXFilingDiscoverer (Hong Kong) -- HKEX News title search + PDF download
  - SGXFilingDiscoverer (Singapore) -- SGX announcements API + PDF download
  - TadawulFilingDiscoverer (Saudi Arabia) -- Tadawul disclosure API
  - SEDARFilingDiscoverer (Canada) -- SEDAR+ document search
  - JSEFilingDiscoverer (South Africa) -- JSE SENS announcement search

Markets without structured APIs (SIX) use EU ESEF crossover.
All other Tier 2 markets have filing discoverers wired in.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol, runtime_checkable

import requests

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FilingMetadata:
    """Metadata for a single discovered filing."""

    title: str
    filing_date: str = ""       # ISO date when published by the exchange
    report_date: str = ""       # fiscal period end date (if parseable)
    document_url: str = ""      # URL to download the document
    document_format: str = ""   # pdf, html, ixbrl
    filing_type: str = ""       # annual, interim, quarterly
    market_id: str = ""
    attachment_id: str = ""     # exchange-specific document key
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilingDiscovery:
    """Result of filing discovery for a single company."""

    ticker: str = ""
    market_id: str = ""
    filings: list[FilingMetadata] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_filings(self) -> bool:
        return len(self.filings) > 0

    def annual_filings(self) -> list[FilingMetadata]:
        return [f for f in self.filings if f.filing_type == "annual"]

    def quarterly_filings(self) -> list[FilingMetadata]:
        return [f for f in self.filings if f.filing_type in ("quarterly", "interim")]

    def shareholding_filings(self) -> list[FilingMetadata]:
        return [f for f in self.filings if f.filing_type == "shareholding"]

    def insider_filings(self) -> list[FilingMetadata]:
        return [f for f in self.filings if f.filing_type == "insider"]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FilingDiscoverer(Protocol):
    """Protocol for per-market filing discovery."""

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        """Discover financial filings for a company.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol.
        years:
            How many years of filings to search for.

        Returns
        -------
        FilingDiscovery with list of discovered filings.
        """
        ...

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download the filing document as raw bytes.

        Parameters
        ----------
        filing:
            A FilingMetadata instance with document_url populated.

        Returns
        -------
        Raw document bytes (PDF, HTML, etc.).
        """
        ...


# ---------------------------------------------------------------------------
# BSE India Filing Discoverer
# ---------------------------------------------------------------------------

_BSE_ANN_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
_BSE_PDF_BASE = "https://www.bseindia.com/xml-data/corpfiling/AttachLive"
_BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.bseindia.com/",
}


def _parse_bse_report_date(subject: str) -> str:
    """Extract fiscal period end date from BSE filing subject line.

    Examples:
      'Consolidated ... For The Quarter And Nine Months Ended December 31, 2025'
      'Financial Results For The Year Ended March 31, 2025'
    """
    # Pattern: "Ended Month DD, YYYY" or "Ended DD Month YYYY"
    patterns = [
        r"[Ee]nded\s+(\w+)\s+(\d{1,2}),?\s+(\d{4})",
        r"[Ee]nded\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        r"[Ee]nded\s+(\w+)\s+(\d{4})",
    ]

    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }

    for pattern in patterns:
        match = re.search(pattern, subject, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                # Try "Month DD, YYYY"
                month_str = groups[0].lower()
                if month_str in months:
                    return f"{groups[2]}-{months[month_str]}-{int(groups[1]):02d}"
                # Try "DD Month YYYY"
                month_str = groups[1].lower()
                if month_str in months:
                    return f"{groups[2]}-{months[month_str]}-{int(groups[0]):02d}"
            elif len(groups) == 2:
                month_str = groups[0].lower()
                if month_str in months:
                    return f"{groups[1]}-{months[month_str]}-01"

    return ""


def _classify_bse_filing_type(subject: str) -> str:
    """Classify BSE filing as annual, quarterly, or interim."""
    lower = subject.lower()
    # Check half year / six months BEFORE "year ended" to avoid false match
    if "half year" in lower or "six months" in lower:
        return "interim"
    if "year ended" in lower or "annual" in lower:
        return "annual"
    if "quarter" in lower:
        return "quarterly"
    return "quarterly"


class BSEFilingDiscoverer:
    """Discovers financial result and shareholding filings from BSE India API.

    Uses the BSE corporate announcements API to find financial result
    PDFs and shareholding pattern filings. Each result filing contains
    quarterly or annual financial statements; shareholding filings contain
    SEBI Regulation 31 shareholding pattern data.
    """

    # BSE announcement categories for different filing types
    _FINANCIAL_CATS = ["Result"]
    _SHAREHOLDING_CATS = ["Shareholding", "Corp. Governance"]

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
        categories: list[str] | None = None,
    ) -> FilingDiscovery:
        """Discover filings from BSE India.

        Parameters
        ----------
        ticker:
            BSE scrip code (e.g. '500325' for Reliance).
        years:
            Number of years to search back.
        categories:
            Filing categories to discover: ["financial"], ["shareholding"],
            or ["financial", "shareholding"]. Default: ["financial"].
        """
        if categories is None:
            categories = ["financial"]

        result = FilingDiscovery(ticker=ticker, market_id="in_bse")

        today = date.today()
        from_date = today - timedelta(days=365 * years)

        # Build list of BSE strCat values based on requested categories
        str_cats = []
        if "financial" in categories:
            str_cats.extend(self._FINANCIAL_CATS)
        if "shareholding" in categories:
            str_cats.extend(self._SHAREHOLDING_CATS)
        if not str_cats:
            str_cats = self._FINANCIAL_CATS

        for str_cat in str_cats:
            try:
                resp = requests.get(
                    _BSE_ANN_URL,
                    params={
                        "Ession": "",
                        "strCat": str_cat,
                        "strPrevDate": from_date.strftime("%Y%m%d"),
                        "strScrip": ticker,
                        "strSearch": "P",
                        "strToDate": today.strftime("%Y%m%d"),
                        "strType": "C",
                    },
                    headers=_BSE_HEADERS,
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                result.errors.append(f"BSE API request failed for {str_cat}: {exc}")
                continue

            table = data.get("Table", [])
            if not table:
                continue

            for item in table:
            subject = item.get("NEWSSUB", "")
            attachment = item.get("ATTACHMENTNAME", "")
            news_dt = item.get("NEWS_DT", "")

            if not attachment:
                continue

            # Parse filing date from NEWS_DT (ISO timestamp)
            filing_date = ""
            if news_dt:
                try:
                    filing_date = news_dt[:10]  # "2026-01-16T19:07:13" -> "2026-01-16"
                except Exception:
                    pass

            report_date = _parse_bse_report_date(subject)
            # Classify as shareholding if from a shareholding category
            subject_lower = subject.lower()
            if str_cat in self._SHAREHOLDING_CATS or "shareholding" in subject_lower:
                filing_type = "shareholding"
            else:
                filing_type = _classify_bse_filing_type(subject)

            filing = FilingMetadata(
                title=subject,
                filing_date=filing_date,
                report_date=report_date,
                document_url=f"{_BSE_PDF_BASE}/{attachment}",
                document_format="pdf",
                filing_type=filing_type,
                market_id="in_bse",
                attachment_id=attachment,
            )
            result.filings.append(filing)

        logger.info(
            "BSE discovery for %s: found %d filings (%d annual, %d quarterly)",
            ticker, len(result.filings),
            len(result.annual_filings()), len(result.quarterly_filings()),
        )
        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download a BSE filing PDF."""
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")

        resp = requests.get(
            filing.document_url,
            headers=_BSE_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()

        if resp.content[:4] != b"%PDF":
            raise ValueError(
                f"Expected PDF but got {resp.headers.get('Content-Type', 'unknown')}"
            )

        logger.info(
            "Downloaded BSE filing: %s (%d bytes)",
            filing.title[:60], len(resp.content),
        )
        return resp.content


# ---------------------------------------------------------------------------
# ASX Australia Filing Discoverer
# ---------------------------------------------------------------------------

_ASX_MARKIT_BASE = "https://asx.api.markitdigital.com/asx-research/1.0"
_ASX_CDN_BASE = "https://cdn-api.markitdigital.com/apiman-gateway/ASX/asx-research/1.0/file"
_ASX_CDN_TOKEN = "83ff96335c2d45a094df02a206a39ff4"
_ASX_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Operator1/1.0",
}

# Announcement types that contain financial data
_ASX_FINANCIAL_TYPES = {
    "PERIODIC REPORTS",
    "ANNUAL REPORT",
    "HALF YEARLY REPORT",
}
_ASX_SHAREHOLDING_TYPES = {
    "BECOMING A SUBSTANTIAL HOLDER",
    "CEASING TO BE A SUBSTANTIAL HOLDER",
    "CHANGE IN SUBSTANTIAL HOLDING",
}


class ASXFilingDiscoverer:
    """Discovers financial filing announcements from ASX via MarkitDigital API.

    The MarkitDigital API returns announcement metadata including document
    keys. Document download URL patterns are still under investigation;
    this discoverer provides announcement discovery which can be used to
    identify filing dates even when the PDF is not directly downloadable.
    """

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        """Discover financial announcements from ASX MarkitDigital API.

        Parameters
        ----------
        ticker:
            ASX ticker symbol (e.g. 'BHP', 'CBA').
        years:
            Number of years to search back.
        """
        result = FilingDiscovery(ticker=ticker, market_id="au_asx")

        try:
            resp = requests.get(
                f"{_ASX_MARKIT_BASE}/companies/{ticker}/announcements",
                params={"count": 100, "market_sensitive": "false"},
                headers=_ASX_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            result.errors.append(f"ASX MarkitDigital API failed: {exc}")
            return result

        items = data.get("data", {}).get("items", [])
        if not items:
            result.errors.append("No announcements found")
            return result

        cutoff = date.today() - timedelta(days=365 * years)

        for item in items:
            headline = item.get("headline", "")
            ann_type = item.get("announcementType", "")
            ann_date = item.get("date", "")
            doc_key = item.get("documentKey", "")

            # Filter to financial filings
            if ann_type not in _ASX_FINANCIAL_TYPES:
                # Also check headline keywords
                lower_headline = headline.lower()
                if not any(kw in lower_headline for kw in
                           ["annual report", "half year", "financial result",
                            "preliminary final", "appendix 4d", "appendix 4e"]):
                    continue

            # Parse date
            filing_date = ""
            if ann_date:
                try:
                    filing_date = ann_date[:10]
                except Exception:
                    pass

            if filing_date and filing_date < cutoff.isoformat():
                continue

            # Classify filing type
            lower = headline.lower()
            if "annual" in lower or "appendix 4e" in lower:
                filing_type = "annual"
            elif "half year" in lower or "appendix 4d" in lower or lower.startswith("hy"):
                filing_type = "interim"
            else:
                filing_type = "quarterly"

            # Build download URL from CDN + documentKey
            doc_url = ""
            if doc_key:
                doc_url = f"{_ASX_CDN_BASE}/{doc_key}?access_token={_ASX_CDN_TOKEN}"

            filing = FilingMetadata(
                title=headline,
                filing_date=filing_date,
                report_date="",  # ASX doesn't provide this in the announcement
                document_url=doc_url,
                document_format="pdf",
                filing_type=filing_type,
                market_id="au_asx",
                attachment_id=doc_key,
            )
            result.filings.append(filing)

        logger.info(
            "ASX discovery for %s: found %d filings (%d annual, %d interim)",
            ticker, len(result.filings),
            len(result.annual_filings()), len(result.quarterly_filings()),
        )
        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download an ASX filing document via the CDN endpoint.

        Uses the MarkitDigital CDN with access token to download PDFs
        identified by their documentKey.
        """
        # Prefer the pre-built document_url (CDN URL)
        url = filing.document_url
        if not url and filing.attachment_id:
            url = f"{_ASX_CDN_BASE}/{filing.attachment_id}?access_token={_ASX_CDN_TOKEN}"

        if not url:
            raise ValueError("No document URL or key in filing metadata")

        resp = requests.get(
            url,
            headers={"User-Agent": "Operator1/1.0", "Accept": "*/*"},
            timeout=60,
        )
        resp.raise_for_status()

        if resp.content[:4] != b"%PDF":
            raise ValueError(
                f"Expected PDF but got {resp.headers.get('Content-Type', 'unknown')}"
            )

        logger.info(
            "Downloaded ASX filing: %s (%d bytes)",
            filing.title[:60], len(resp.content),
        )
        return resp.content


# ---------------------------------------------------------------------------
# HKEX Hong Kong Filing Discoverer
# ---------------------------------------------------------------------------

_HKEX_SEARCH_URL = "https://www1.hkexnews.hk/search/titlesearch.xhtml"
_HKEX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json, text/html",
    "Referer": "https://www.hkexnews.hk/",
}


def _parse_hkex_report_date(title: str) -> str:
    """Extract fiscal period end date from HKEX filing title.

    Examples:
      'Annual Results for the Year Ended 31 December 2025'
      'Interim Results for the Six Months Ended 30 June 2025'
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
                # "31 December 2025"
                return f"{groups[2]}-{months[groups[1].lower()]}-{int(groups[0]):02d}"
            elif groups[0].lower() in months:
                # "December 31, 2025"
                return f"{groups[2]}-{months[groups[0].lower()]}-{int(groups[1]):02d}"
    return ""


def _classify_hkex_filing_type(title: str) -> str:
    """Classify HKEX filing as annual, interim, or quarterly."""
    lower = title.lower()
    if "annual" in lower or "year ended" in lower:
        return "annual"
    if "interim" in lower or "half" in lower or "six months" in lower:
        return "interim"
    if "quarter" in lower or "three months" in lower:
        return "quarterly"
    return "annual"


class HKEXFilingDiscoverer:
    """Discovers financial result filings from HKEX News.

    Uses date-windowed queries to the HKEX JSON API.  The API limits
    each request to ~2 weeks of data, so this discoverer automatically
    splits a multi-year search into 2-week windows with title filters.

    No proxy is required -- the API works from any IP address as long
    as individual date windows are kept under ~15 days.

    HKEX stock codes are zero-padded to 5 digits (e.g. 00700 for Tencent,
    00005 for HSBC).
    """

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        """Discover financial result filings from HKEX News.

        Parameters
        ----------
        ticker:
            HKEX stock code (e.g. '0700', '00700', '5').
        years:
            Number of years to search back.
        """
        result = FilingDiscovery(ticker=ticker, market_id="hk_hkex")

        # HKEX uses 5-digit zero-padded stock codes
        code = ticker.split(".")[0].strip().zfill(5)

        try:
            from operator1.clients.hkex_scraper import HKEXScraper
            scraper = HKEXScraper()
            filings = scraper.search_filings(code, years=years)
            if filings:
                for f in filings:
                    result.filings.append(FilingMetadata(
                        title=f.title,
                        filing_date=f.release_date,
                        report_date=f.report_date,
                        document_url=f.document_url,
                        document_format=f.document_format,
                        filing_type=f.filing_type,
                        market_id="hk_hkex",
                    ))
                logger.info(
                    "HKEX discovery for %s: %d filings (%d annual, %d interim)",
                    code, len(result.filings),
                    len(result.annual_filings()), len(result.quarterly_filings()),
                )
            else:
                logger.info("HKEX discovery for %s: no filings found", code)
        except ImportError:
            result.errors.append("HKEX scraper module not available")
            logger.warning("HKEX scraper not available (hkex_scraper.py missing)")
        except Exception as exc:
            result.errors.append(f"HKEX discovery failed: {exc}")
            logger.warning("HKEX discovery failed for %s: %s", code, exc)

        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download an HKEX filing document (PDF, HTML, or Excel).

        HKEX documents are hosted as static files and don't require
        the JSF session -- a direct GET with a browser User-Agent works.
        """
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")

        resp = requests.get(
            filing.document_url,
            headers=_HKEX_HEADERS,
            timeout=60,  # HKEX PDFs can be large
        )
        resp.raise_for_status()

        # Validate the content matches the expected format.
        # PDF filings start with %PDF; HTML filings start with < or <!;
        # Excel files start with PK (ZIP signature).
        fmt = (filing.document_format or "").lower()
        if fmt == "pdf" and resp.content[:4] != b"%PDF":
            raise ValueError(
                f"Expected PDF but got {resp.headers.get('Content-Type', 'unknown')}"
            )

        logger.info(
            "Downloaded HKEX filing: %s (%d bytes)",
            filing.title[:60], len(resp.content),
        )
        return resp.content


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SGX Singapore Filing Discoverer
# ---------------------------------------------------------------------------

_SGX_REPORTS_URL = "https://api.sgx.com/financialreports/v1.0"
_SGX_LINKS_BASE = "https://links.sgx.com"
_SGX_ANN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


class SGXFilingDiscoverer:
    """Discovers financial report filings from SGX Financial Reports API.

    Uses the public ``api.sgx.com/financialreports/v1.0`` endpoint which
    returns annual/sustainability reports with PDF download links.  Works
    globally without authentication.

    Flow:
        1. Query ``financialreports/v1.0?companyname={NAME}`` for filings.
        2. Each filing has a ``url`` field pointing to an HTML document page.
        3. The document page contains ``<a>`` links to actual PDF files on
           ``links.sgx.com``.
        4. Download the PDF directly.

    Coverage: 1,369 unique companies, 12,674+ total reports.
    """

    def _resolve_company_name(self, ticker: str) -> str:
        """Resolve a ticker code to the full company name used by SGX.

        The financial reports API filters by company name, not ticker.
        We use the securities directory to map ticker -> company name.
        """
        try:
            from operator1.clients.sg_sgx import _get_securities_directory, _search_securities
            directory = _get_securities_directory()
            if directory:
                matches = _search_securities(ticker, directory, max_results=1)
                if matches and matches[0].get("n"):
                    return matches[0]["n"]
        except Exception:
            pass
        return ticker

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        result = FilingDiscovery(ticker=ticker, market_id="sg_sgx")

        from datetime import datetime

        today = date.today()
        cutoff = today - timedelta(days=365 * years)

        # Resolve ticker to company name for the API query
        company_name = self._resolve_company_name(ticker)

        try:
            resp = requests.get(
                _SGX_REPORTS_URL,
                params={
                    "companyname": company_name,
                    "pagestart": "0",
                    "pagesize": "50",
                },
                headers=_SGX_ANN_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            result.errors.append(f"SGX Financial Reports API failed: {exc}")
            return result

        items = data.get("data", [])
        if not items:
            # Try with the raw ticker as company name
            if company_name != ticker:
                try:
                    resp2 = requests.get(
                        _SGX_REPORTS_URL,
                        params={
                            "companyname": ticker,
                            "pagestart": "0",
                            "pagesize": "50",
                        },
                        headers=_SGX_ANN_HEADERS,
                        timeout=30,
                    )
                    resp2.raise_for_status()
                    items = resp2.json().get("data", [])
                except Exception:
                    pass

        if not items:
            result.errors.append(f"No financial reports found on SGX for {company_name}")
            return result

        for item in items:
            title_text = item.get("title", "")
            company = item.get("companyName", "")
            doc_date_ms = item.get("documentDate", 0)
            doc_url = item.get("url", "")

            if not title_text or not doc_url:
                continue

            # Parse document date from epoch milliseconds
            filing_date = ""
            report_date = ""
            if doc_date_ms:
                try:
                    dt = datetime.fromtimestamp(doc_date_ms / 1000)
                    filing_date = dt.strftime("%Y-%m-%d")
                    report_date = filing_date  # SGX reports use fiscal year end
                except (ValueError, OSError):
                    pass

            # Filter by date range
            if report_date and report_date < cutoff.isoformat():
                continue

            # Classify filing type from title
            lower = title_text.lower()
            if "annual" in lower:
                filing_type = "annual"
            elif "sustainability" in lower:
                filing_type = "annual"  # sustainability reports align with annual
            elif "interim" in lower or "half" in lower:
                filing_type = "interim"
            elif "quarter" in lower:
                filing_type = "quarterly"
            else:
                filing_type = "annual"

            filing = FilingMetadata(
                title=f"{company} - {title_text}",
                filing_date=filing_date,
                report_date=report_date,
                document_url=doc_url,
                document_format="pdf",
                filing_type=filing_type,
                market_id="sg_sgx",
                attachment_id=item.get("id", ""),
            )
            result.filings.append(filing)

        logger.info(
            "SGX discovery for %s (%s): found %d filings (%d annual, %d interim)",
            ticker, company_name, len(result.filings),
            len(result.annual_filings()), len(result.quarterly_filings()),
        )
        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download an SGX filing PDF.

        SGX document URLs point to HTML pages containing links to the
        actual PDF files.  This method fetches the HTML page, extracts
        the first PDF link, and downloads the PDF.
        """
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")

        # Step 1: Fetch the document page (HTML with PDF links)
        resp = requests.get(
            filing.document_url,
            headers=_SGX_ANN_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()

        # If the response is already a PDF, return it directly
        if resp.content[:4] == b"%PDF":
            logger.info(
                "Downloaded SGX filing (direct PDF): %s (%d bytes)",
                filing.title[:60], len(resp.content),
            )
            return resp.content

        # Step 2: Parse HTML page for PDF links
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            pdf_links = [
                a["href"] for a in soup.find_all("a", href=True)
                if a["href"].endswith(".pdf")
            ]
        except ImportError:
            # Fallback: regex extraction if BeautifulSoup unavailable
            pdf_links = re.findall(r'href="([^"]+\.pdf)"', resp.text)

        if not pdf_links:
            raise ValueError(
                f"No PDF links found on document page: {filing.document_url}"
            )

        # Step 3: Download the first PDF
        pdf_url = pdf_links[0]
        if pdf_url.startswith("/"):
            pdf_url = _SGX_LINKS_BASE + pdf_url

        pdf_resp = requests.get(pdf_url, headers=_SGX_ANN_HEADERS, timeout=60)
        pdf_resp.raise_for_status()

        if pdf_resp.content[:4] != b"%PDF":
            raise ValueError(
                f"Expected PDF but got {pdf_resp.headers.get('Content-Type', 'unknown')}"
            )

        logger.info(
            "Downloaded SGX filing: %s (%d bytes, from %s)",
            filing.title[:60], len(pdf_resp.content), pdf_url.split("/")[-1],
        )
        return pdf_resp.content


# ---------------------------------------------------------------------------
# Tadawul Saudi Arabia Filing Discoverer
# ---------------------------------------------------------------------------

_TADAWUL_DISC_URL = "https://www.saudiexchange.sa/wps/portal/tadawul/market-participants/issuers/reports-statements/financial-statements"
_TADAWUL_HEADERS = {
    "User-Agent": "Operator1/1.0",
    "Accept": "application/json",
}


class TadawulFilingDiscoverer:
    """Discovers financial filings from Saudi Exchange (Tadawul).

    Searches the Tadawul disclosures portal for financial statements.
    Falls back gracefully if the API is unavailable.
    """

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        result = FilingDiscovery(ticker=ticker, market_id="sa_tadawul")

        today = date.today()
        from_date = today - timedelta(days=365 * years)

        try:
            data = cached_get(
                f"https://www.saudiexchange.sa/tadawul-api/resources/listed-companies/{ticker}/disclosures",
                params={
                    "fromDate": from_date.strftime("%Y-%m-%d"),
                    "toDate": today.strftime("%Y-%m-%d"),
                    "category": "FINANCIAL",
                    "pageSize": "20",
                },
                headers=_TADAWUL_HEADERS,
            )
        except Exception as exc:
            result.errors.append(f"Tadawul API failed: {exc}")
            return result

        items = []
        if isinstance(data, dict):
            items = data.get("data", data.get("disclosures", data.get("result", [])))
        elif isinstance(data, list):
            items = data

        for item in items:
            title = item.get("title", item.get("subject", ""))
            ann_date = item.get("publishDate", item.get("date", ""))
            doc_url = item.get("url", item.get("pdfUrl", ""))

            if not title:
                continue

            lower = title.lower()
            if "annual" in lower or "year" in lower:
                filing_type = "annual"
            elif "interim" in lower or "half" in lower or "six" in lower:
                filing_type = "interim"
            elif "quarter" in lower:
                filing_type = "quarterly"
            else:
                filing_type = "annual"

            filing_date = str(ann_date)[:10] if ann_date else ""

            filing = FilingMetadata(
                title=title,
                filing_date=filing_date,
                document_url=doc_url if doc_url and doc_url.startswith("http") else "",
                document_format="pdf",
                filing_type=filing_type,
                market_id="sa_tadawul",
            )
            result.filings.append(filing)

        logger.info("Tadawul discovery for %s: found %d filings", ticker, len(result.filings))
        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")
        resp = requests.get(filing.document_url, headers=_TADAWUL_HEADERS, timeout=30)
        resp.raise_for_status()
        if resp.content[:4] != b"%PDF":
            raise ValueError(f"Not a PDF: {resp.headers.get('Content-Type', 'unknown')}")
        return resp.content


# ---------------------------------------------------------------------------
# SEDAR+ Canada Filing Discoverer
# ---------------------------------------------------------------------------

_SEDAR_BASE = "https://www.sedarplus.ca/csa-party"
_SEDAR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html, application/pdf, */*",
}

# Delay between SEDAR+ requests to avoid rate-limiting.
# SEDAR+ throttles after ~5 rapid requests.
_SEDAR_REQUEST_DELAY_S = 5.0

# Download timeout -- SEDAR+ CDN is extremely slow.
_SEDAR_DOWNLOAD_TIMEOUT_S = 120


class SEDARFilingDiscoverer:
    """Discovers financial filings from SEDAR+ (Canada).

    Uses the Catalyst form POST mechanism to search for filings.
    The SEDAR+ REST API is WAF-blocked, but the HTML form submission
    works reliably.

    Flow:
        1. GET the search form page to establish a session and get
           the session-specific form action URL.
        2. POST to the form action with filing criteria (company name,
           category, date range).
        3. Parse the HTML result page with BeautifulSoup to extract
           filing metadata and document download links.
        4. Download PDFs via session-bound resource.html URLs.

    Rate limiting: 5s delay between requests. SEDAR+ CDN is slow
    (~60s for a 400KB PDF).
    """

    def __init__(self) -> None:
        self._session: requests.Session | None = None
        self._form_action: str = ""

    def _ensure_session(self) -> requests.Session:
        """Create a session and get the form action URL."""
        if self._session is not None and self._form_action:
            return self._session

        self._session = requests.Session()
        self._session.headers.update(_SEDAR_HEADERS)

        resp = self._session.get(
            f"{_SEDAR_BASE}/service/create.html",
            params={
                "targetAppCode": "csa-party",
                "service": "searchDocuments",
                "_locale": "en",
            },
            timeout=30,
        )
        resp.raise_for_status()

        # Extract session-specific form action URL
        import re as _re
        match = _re.search(
            r'action="(https://www\.sedarplus\.ca/csa-party/viewInstance/update\.html\?id=[^"]+)"',
            resp.text,
        )
        if not match:
            raise RuntimeError("Could not find SEDAR+ form action URL")

        self._form_action = match.group(1).replace("&amp;", "&")
        logger.debug("SEDAR+ session established, form action: %s", self._form_action[:80])
        return self._session

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        result = FilingDiscovery(ticker=ticker, market_id="ca_sedar")

        today = date.today()
        from_date = today - timedelta(days=365 * years)

        try:
            session = self._ensure_session()
        except Exception as exc:
            result.errors.append(f"SEDAR+ session failed: {exc}")
            return result

        time.sleep(_SEDAR_REQUEST_DELAY_S)

        # Search for annual + interim financial statements
        for filing_type_label in [
            "Annual financial statements",
            "Interim financial statements/report",
        ]:
            try:
                resp = session.post(
                    self._form_action,
                    data={
                        "FilingIdentifier": ticker,
                        "FilingCategory": "Continuous disclosure",
                        "FilingType": filing_type_label,
                        "SubmissionDate": from_date.strftime("%d/%m/%Y"),
                        "SubmissionDate2": today.strftime("%d/%m/%Y"),
                        "nodeW285ac": "search",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
            except Exception as exc:
                result.errors.append(f"SEDAR+ search failed for {filing_type_label}: {exc}")
                time.sleep(_SEDAR_REQUEST_DELAY_S)
                continue

            # Parse results with BeautifulSoup
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                self._parse_results(soup, result, filing_type_label)
            except ImportError:
                result.errors.append("beautifulsoup4 not installed")
                break
            except Exception as exc:
                result.errors.append(f"SEDAR+ parse failed: {exc}")

            time.sleep(_SEDAR_REQUEST_DELAY_S)

        logger.info(
            "SEDAR+ discovery for %s: found %d filings (%d annual, %d interim)",
            ticker, len(result.filings),
            len(result.annual_filings()), len(result.quarterly_filings()),
        )
        return result

    def _parse_results(
        self,
        soup,
        result: FilingDiscovery,
        filing_type_label: str,
    ) -> None:
        """Parse SEDAR+ Catalyst HTML results into FilingMetadata objects."""
        is_annual = "annual" in filing_type_label.lower()

        # Find document download links (resource.html URLs)
        doc_links = soup.find_all("a", href=lambda h: h and "resource.html" in h)

        for a_tag in doc_links:
            title = a_tag.get_text(strip=True) or ""
            href = a_tag.get("href", "")
            if not href or not title:
                continue

            # Only include PDF documents
            if ".pdf" not in title.lower():
                continue

            # Extract entity name and date from surrounding HTML
            entity_name = ""
            filing_date_str = ""
            parent_block = a_tag.find_parent(
                "div", class_=lambda c: c and "csaFilingDocuments" in str(c) and "page" in str(c)
            )
            if parent_block:
                entity_div = parent_block.find(
                    "div", class_=lambda c: c and "filingEntities" in str(c)
                )
                if entity_div:
                    entity_name = entity_div.get_text(strip=True)
                date_div = parent_block.find(
                    "div", class_=lambda c: c and "SubmissionDate" in str(c) and "Attribute" in str(c)
                )
                if date_div:
                    date_text = date_div.get_text(strip=True)
                    # Parse "16 Mar 2026 11:05 EDT" -> "2026-03-16"
                    import re as _re
                    dm = _re.search(r"(\d{1,2})\s+(\w{3})\s+(\d{4})", date_text)
                    if dm:
                        _months = {
                            "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                            "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                            "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
                        }
                        mon = _months.get(dm.group(2), "01")
                        filing_date_str = f"{dm.group(3)}-{mon}-{int(dm.group(1)):02d}"

            filing = FilingMetadata(
                title=f"{entity_name}: {title}" if entity_name else title,
                filing_date=filing_date_str,
                document_url=href,
                document_format="pdf",
                filing_type="annual" if is_annual else "interim",
                market_id="ca_sedar",
            )
            result.filings.append(filing)

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download a SEDAR+ filing PDF via session-bound resource URL.

        SEDAR+ CDN is slow -- downloads can take 60-120 seconds for
        a typical filing PDF. The resource.html URLs are session-bound
        and require the same session cookies used during discovery.
        """
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")

        session = self._ensure_session()
        time.sleep(_SEDAR_REQUEST_DELAY_S)

        resp = session.get(
            filing.document_url,
            timeout=_SEDAR_DOWNLOAD_TIMEOUT_S,
            allow_redirects=True,
        )
        resp.raise_for_status()

        ct = resp.headers.get("Content-Type", "")
        if "pdf" not in ct and resp.content[:4] != b"%PDF":
            raise ValueError(
                f"Expected PDF but got {ct} ({len(resp.content)} bytes)"
            )

        logger.info(
            "Downloaded SEDAR+ filing: %s (%d bytes)",
            filing.title[:60], len(resp.content),
        )
        return resp.content


# ---------------------------------------------------------------------------
# JSE South Africa Filing Discoverer
# ---------------------------------------------------------------------------

_JSE_PORTAL_BASE = "https://clientportal.jse.co.za"
_JSE_SENS_URL = f"{_JSE_PORTAL_BASE}/_vti_bin/JSE/SENSService.svc/GetSensAnnouncementForDates"
_JSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
}


def _parse_jse_report_date(title: str) -> str:
    """Extract fiscal period end date from JSE SENS filing title.

    Examples:
      'Results for the Year Ended 31 December 2025'
      'Interim Results for the Six Months Ended 30 June 2025'
      'Financial Results for Year Ending 28 February 2026'
    """
    patterns = [
        r"[Ee]nd(?:ed|ing)\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        r"[Ee]nd(?:ed|ing)\s+(\w+)\s+(\d{1,2}),?\s+(\d{4})",
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
                # "31 December 2025"
                return f"{groups[2]}-{months[groups[1].lower()]}-{int(groups[0]):02d}"
            elif groups[0].lower() in months:
                # "December 31, 2025"
                return f"{groups[2]}-{months[groups[0].lower()]}-{int(groups[1]):02d}"
    return ""


def _classify_jse_filing_type(title: str) -> str:
    """Classify JSE filing as annual, interim, or quarterly."""
    lower = title.lower()
    if "annual" in lower or "year ended" in lower or "year ending" in lower:
        return "annual"
    if "interim" in lower or "half" in lower or "six months" in lower:
        return "interim"
    if "quarter" in lower or "three months" in lower:
        return "quarterly"
    return "annual"


class JSEFilingDiscoverer:
    """Discovers financial result filings from JSE SENS via WCF API.

    Uses the JSE Client Portal WCF service to fetch SENS announcements
    **per issuer** via ``GetSensAnnouncementsByIssuerMasterId``.  This
    is a single API call that returns all recent announcements for one
    company (analogous to J-Quants' ``get_financials()``).

    Each announcement includes a ``PDFPath`` field with a direct URL to
    the filing PDF on ``senspdf.jse.co.za``.

    The fast path requires resolving the ticker to a MasterID first
    via the issuer directory (also a single API call, cached).
    """

    def _resolve_master_id(self, ticker: str) -> int | None:
        """Resolve a JSE ticker to a MasterID via the issuer directory."""
        try:
            resp = requests.post(
                f"{_JSE_PORTAL_BASE}/_vti_bin/JSE/CustomerRoleService.svc/GetAllIssuers",
                json={"filterLongName": "", "filterType": "Equity Issuer"},
                headers=_JSE_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            issuers = resp.json()
            if isinstance(issuers, list):
                ticker_upper = ticker.strip().upper()
                for issuer in issuers:
                    if (issuer.get("AlphaCode", "").upper() == ticker_upper
                            or issuer.get("CustomerAlphaCode", "").upper() == ticker_upper):
                        return issuer.get("MasterID")
        except Exception as exc:
            logger.debug("JSE issuer resolution failed for %s: %s", ticker, exc)
        return None

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        """Discover financial result filings from JSE SENS (fast, single API call).

        Parameters
        ----------
        ticker:
            JSE ticker symbol (e.g. 'NPN' for Naspers, 'SOL' for Sasol).
        years:
            Number of years to search back (used for date filtering).
        """
        result = FilingDiscovery(ticker=ticker, market_id="za_jse")

        # Step 1: Resolve ticker to MasterID
        master_id = self._resolve_master_id(ticker)
        if master_id is None:
            result.errors.append(f"Could not resolve JSE ticker '{ticker}' to MasterID")
            return result

        # Step 2: Get SENS announcements for this issuer (single API call)
        try:
            resp = requests.post(
                f"{_JSE_PORTAL_BASE}/_vti_bin/JSE/SENSService.svc/GetSensAnnouncementsByIssuerMasterId",
                json={"issuerMasterId": master_id},
                headers=_JSE_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            result.errors.append(f"JSE SENS API failed: {exc}")
            return result

        # Extract announcements from the WCF response
        result_key = next(iter(data.keys()), None) if isinstance(data, dict) else None
        announcements = data.get(result_key, data) if result_key else data
        if not isinstance(announcements, list):
            announcements = []

        cutoff = date.today() - timedelta(days=365 * years)

        for ann in announcements:
            headline = ann.get("FlashHeadline", "")
            pdf_path = ann.get("PDFPath", "")
            ann_id = ann.get("AnnouncementId", "")
            ref = ann.get("AnnouncementReferenceNumber", "")

            # Parse announcement date from .NET JSON date format
            # Format: "/Date(1773843357873+0200)/"
            filing_date = ""
            raw_date = ann.get("AcknowledgeDateTime", "")
            if raw_date and "/Date(" in str(raw_date):
                try:
                    ts_match = re.search(r"/Date\((\d+)", str(raw_date))
                    if ts_match:
                        ts_ms = int(ts_match.group(1))
                        from datetime import datetime as _dt, timezone as _tz
                        dt = _dt.fromtimestamp(ts_ms / 1000, tz=_tz.utc)
                        filing_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

            # Apply date cutoff
            if filing_date and filing_date < cutoff.isoformat():
                continue

            report_date = _parse_jse_report_date(headline)
            filing_type = _classify_jse_filing_type(headline)

            filing = FilingMetadata(
                title=headline,
                filing_date=filing_date,
                report_date=report_date,
                document_url=pdf_path,
                document_format="pdf",
                filing_type=filing_type,
                market_id="za_jse",
                attachment_id=ann_id or ref,
            )
            result.filings.append(filing)

        # Dedup by PDF URL
        seen_urls: set[str] = set()
        unique: list[FilingMetadata] = []
        for f in result.filings:
            key = f.document_url or f.attachment_id or f.title
            if key not in seen_urls:
                seen_urls.add(key)
                unique.append(f)
        result.filings = unique

        logger.info(
            "JSE SENS discovery for %s (MasterID=%d): %d filings (%d annual, %d interim)",
            ticker, master_id, len(result.filings),
            len(result.annual_filings()), len(result.quarterly_filings()),
        )
        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download a JSE SENS filing PDF.

        PDFs are served as static files from senspdf.jse.co.za.
        No authentication required.
        """
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")

        resp = requests.get(
            filing.document_url,
            headers={
                "User-Agent": _JSE_HEADERS["User-Agent"],
                "Accept": "application/pdf",
            },
            timeout=30,
        )
        resp.raise_for_status()

        if resp.content[:4] != b"%PDF":
            raise ValueError(
                f"Expected PDF but got {resp.headers.get('Content-Type', 'unknown')}"
            )

        logger.info(
            "Downloaded JSE filing: %s (%d bytes)",
            filing.title[:60], len(resp.content),
        )
        return resp.content


# ---------------------------------------------------------------------------
# BMV Mexico Filing Discoverer
# ---------------------------------------------------------------------------

_BMV_BASE_URL = "https://www.bmv.com.mx"
_BMV_TOKEN_URL = f"{_BMV_BASE_URL}/rest/tokenservice/token"
_BMV_SEARCH_URL = f"{_BMV_BASE_URL}/api/searchservice/v1"
_BMV_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


class BMVFilingDiscoverer:
    """Discovers financial filings from BMV (Mexico).

    Uses the BMV WSO2 API Gateway search endpoint to find financial
    filings and corporate documents. The API requires a Bearer token
    obtained from a public endpoint (no API key needed).

    The BMV search API returns an ElasticSearch-backed response with
    company instruments and associated documents (quarterly records,
    issuer events, corporate actions, etc.). Document PDFs are served
    from ``bmv.com.mx/docs-pub/`` and ``bmv.com.mx/docs-dig/`` paths.

    This is analogous to the HKEX approach: obtain a session token,
    then query a JSON search API, then download static PDFs.
    """

    def __init__(self) -> None:
        self._token: str = ""
        self._token_time: float = 0.0

    def _get_token(self) -> str:
        """Obtain or refresh the BMV API Bearer token."""
        import time as _time
        now = _time.time()
        if self._token and (now - self._token_time) < 3600:
            return self._token
        try:
            resp = requests.get(_BMV_TOKEN_URL, headers=_BMV_HEADERS, timeout=15)
            resp.raise_for_status()
            self._token = resp.json()["response"]["access_token"]
            self._token_time = now
        except Exception as exc:
            logger.warning("BMV token acquisition failed: %s", exc)
            if not self._token:
                raise
        return self._token

    def _search(self, term: str, search_type: str = "busquedaPanel", lang: str = "es") -> dict:
        """Execute a BMV search API call."""
        token = self._get_token()
        parts = term.strip().split(" ", 1)
        payload = {
            "lang": lang,
            "payload": {
                "term": parts[0],
                "term2": parts[1] if len(parts) > 1 else "",
                "termT": term.strip(),
                "searchType": search_type,
            },
        }
        try:
            resp = requests.post(
                _BMV_SEARCH_URL,
                json=payload,
                headers={
                    **_BMV_HEADERS,
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("BMV search failed for '%s': %s", term, exc)
            return {}

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        """Discover financial filings from BMV search API.

        Parameters
        ----------
        ticker:
            BMV ticker symbol (e.g. 'AMX', 'WALMEX', 'CEMEX', 'BIMBO').
        years:
            Number of years to search back (used for filtering).
        """
        result = FilingDiscovery(ticker=ticker, market_id="mx_bmv")

        today = date.today()
        cutoff = today - timedelta(days=365 * years)

        # Search using busquedaPanel (returns documents alongside instruments)
        try:
            data = self._search(ticker, "busquedaPanel", "es")
        except Exception as exc:
            result.errors.append(f"BMV API unavailable: {exc}")
            return result

        # Extract documents from the response
        panel = data.get("response", {}).get("busquedaPanel", {})
        if not isinstance(panel, dict):
            result.errors.append("BMV search returned unexpected format")
            return result

        # Navigate to documents section
        doc_hits: list[dict] = []
        try:
            doc_hits = (
                panel
                .get("busquedaGeneral", {})
                .get("instrumentosEmisoras", {})
                .get("instrumentos", {})
                .get("coincidenciaParcialInstrumentos", {})
                .get("documentos", {})
                .get("hits", [])
            )
        except (AttributeError, TypeError):
            pass

        for hit in doc_hits:
            src = hit.get("_source", {})
            doc_bin = src.get("documento_binario", {})

            # Build document URL
            doc_url = doc_bin.get("url_documento", "")
            if doc_url and not doc_url.startswith("http"):
                doc_url = f"{_BMV_BASE_URL}{doc_url}"

            # Extract title (strip HTML tags)
            title = src.get("descripccion_documento", "")
            title = re.sub(r"<[^>]+>", "", title).strip()

            tag = src.get("tag_en", src.get("tag_es", ""))
            company = src.get("cve_empresa", "")

            # Skip documents not related to this ticker
            if company and company.upper() != ticker.upper():
                # Allow partial match (e.g. search for "AMX" finds "AMX" company docs)
                if ticker.upper() not in company.upper():
                    continue

            # Classify filing type from tag and title
            tag_lower = tag.lower()
            title_lower = title.lower()
            if "annual" in tag_lower or "anual" in title_lower or "annual" in title_lower:
                filing_type = "annual"
            elif "quarterly" in tag_lower or "trimestral" in title_lower:
                filing_type = "quarterly"
            elif "interim" in tag_lower or "semestral" in title_lower:
                filing_type = "interim"
            else:
                filing_type = "quarterly"

            # Check if this is a financial filing (not just any corporate event)
            is_financial = any(kw in tag_lower for kw in (
                "quarterly", "annual", "financial", "results",
            )) or any(kw in title_lower for kw in (
                "trimestral", "anual", "financier", "resultados",
                "constancia", "estado de resultado", "balance",
            ))

            # Parse filing date from document metadata
            filing_date = str(src.get("fecha_publicacion", ""))[:10]

            # Parse report date from title if present
            report_date = ""
            # Look for period patterns like "4-2025" or "2025"
            period_match = re.search(r"(\d{1,2})\s*[-/]\s*(\d{4})", title)
            if period_match:
                quarter = int(period_match.group(1))
                year = int(period_match.group(2))
                # Map quarter to period end date
                quarter_ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
                report_date = f"{year}-{quarter_ends.get(quarter, '12-31')}"
            else:
                year_match = re.search(r"\b(20\d{2})\b", title)
                if year_match:
                    report_date = f"{year_match.group(1)}-12-31"

            # Determine document format
            doc_format = "pdf"
            if doc_url.endswith((".xlsx", ".xls")):
                doc_format = "excel"
            elif doc_url.endswith((".htm", ".html")):
                doc_format = "html"

            filing = FilingMetadata(
                title=title or f"{ticker} {tag}",
                filing_date=filing_date,
                report_date=report_date,
                document_url=doc_url,
                document_format=doc_format,
                filing_type=filing_type,
                market_id="mx_bmv",
            )
            result.filings.append(filing)

        # Also try English search for additional coverage
        if len(result.filings) < 3:
            try:
                data_en = self._search(ticker, "busquedaPanel", "en")
                panel_en = data_en.get("response", {}).get("busquedaPanel", {})
                if isinstance(panel_en, dict):
                    doc_hits_en = (
                        panel_en
                        .get("busquedaGeneral", {})
                        .get("instrumentosEmisoras", {})
                        .get("instrumentos", {})
                        .get("coincidenciaParcialInstrumentos", {})
                        .get("documentos", {})
                        .get("hits", [])
                    )
                    existing_urls = {f.document_url for f in result.filings}
                    for hit in doc_hits_en:
                        src = hit.get("_source", {})
                        doc_bin = src.get("documento_binario", {})
                        doc_url = doc_bin.get("url_documento", "")
                        if doc_url and not doc_url.startswith("http"):
                            doc_url = f"{_BMV_BASE_URL}{doc_url}"
                        if doc_url and doc_url not in existing_urls:
                            title = re.sub(r"<[^>]+>", "", src.get("descripccion_documento", "")).strip()
                            filing = FilingMetadata(
                                title=title or f"{ticker} filing",
                                document_url=doc_url,
                                document_format="pdf",
                                filing_type="quarterly",
                                market_id="mx_bmv",
                            )
                            result.filings.append(filing)
            except Exception:
                pass

        # Dedup by document URL
        seen: set[str] = set()
        unique: list[FilingMetadata] = []
        for f in result.filings:
            key = f.document_url or f.title
            if key not in seen:
                seen.add(key)
                unique.append(f)
        result.filings = unique

        logger.info("BMV discovery for %s: found %d filings", ticker, len(result.filings))
        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download a BMV filing document.

        BMV documents are served as static files from bmv.com.mx/docs-pub/
        and bmv.com.mx/docs-dig/ paths. No authentication required for
        document download.
        """
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")
        resp = requests.get(filing.document_url, headers=_BMV_HEADERS, timeout=30)
        resp.raise_for_status()
        if resp.content[:4] != b"%PDF":
            raise ValueError(f"Not a PDF: {resp.headers.get('Content-Type', 'unknown')}")
        return resp.content


# ---------------------------------------------------------------------------
# DFM/ADX UAE Filing Discoverer (eFsah API on api2.dfm.ae)
# ---------------------------------------------------------------------------

# Old endpoints (404 since ~2025): www.dfm.ae/api/DisclosureFilesApi, www.adx.ae/api/disclosures
# New endpoint: api2.dfm.ae/efsah/v1/prototype_efsah (discovered via Nuxt SSR config)
# PDF download: feeds.dfm.ae/documents/{r_path} (from cms_resources=true)
_DFM_EFSAH_API = "https://api2.dfm.ae/efsah/v1/prototype_efsah"
_DFM_FEEDS_BASE = "https://feeds.dfm.ae/documents"
_DFM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Origin": "https://www.dfm.ae",
    "Referer": "https://www.dfm.ae/",
}


class DFMFilingDiscoverer:
    """Discovers financial filings from DFM via the eFsah API.

    Uses the eFsah disclosure API on api2.dfm.ae (discovered by
    reverse-engineering the DFM Nuxt.js SPA). This replaces the old
    www.dfm.ae/api/DisclosureFilesApi endpoint which was decommissioned.

    The eFsah API returns disclosure metadata with PIT publication dates.
    PDF resources are available via cms_resources=true parameter, with
    PDFs hosted on feeds.dfm.ae/documents/{r_path}.

    UAE adopted IFRS for all listed companies, so standard IFRS
    canonical mapping applies.
    """

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        """Discover financial filings from DFM eFsah API.

        Parameters
        ----------
        ticker:
            DFM ticker symbol (e.g. 'EMAAR', 'DIB', 'DEWA').
        years:
            Number of years to search back.
        """
        result = FilingDiscovery(ticker=ticker, market_id="ae_dfm")

        try:
            import json as _json

            # Fetch disclosures with PDF resources
            resp = requests.get(
                _DFM_EFSAH_API,
                params={
                    "announcement_type": "Disclosure",
                    "symbol": ticker.upper(),
                    "take": "50",
                    "skip": "0",
                    "lang": "en",
                    "h7_datetime_format": "MMM dd, yyyy HH:mm:ss",
                    "cms_resources": "true",
                },
                headers=_DFM_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()

            # eFsah returns UTF-8 BOM -- decode with utf-8-sig
            data = _json.loads(resp.content.decode("utf-8-sig"))
            records = data.get("root", [])

            for item in records:
                headline = item.get("headline", "")
                if not headline:
                    continue

                # Filter to financial disclosures
                lower = headline.lower()
                report_type = (item.get("integrated_report_type") or "").lower()
                is_financial = any(kw in lower for kw in [
                    "financial", "annual", "interim", "quarter",
                    "result", "statement", "report",
                ]) or "financial" in report_type

                if not is_financial:
                    continue

                # Parse publication date (PIT filing date)
                pub_date = item.get("publication_date", "")
                filing_date = ""
                if pub_date:
                    # Format: "Mar 12, 2026 03:53:30 PM"
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(
                            pub_date.split(" AM")[0].split(" PM")[0].rsplit(" ", 1)[0],
                            "%b %d, %Y",
                        )
                        filing_date = dt.strftime("%Y-%m-%d")
                    except (ValueError, IndexError):
                        filing_date = pub_date[:10]

                # Classify filing type
                if "annual" in lower or "year" in lower or "annual" in report_type:
                    filing_type = "annual"
                elif "interim" in lower or "half" in lower or "six" in lower:
                    filing_type = "interim"
                elif "quarter" in lower or "qtr" in lower:
                    filing_type = "quarterly"
                else:
                    filing_type = "annual"

                # Extract PDF URL from resources
                doc_url = ""
                resources = item.get("resources", [])
                if resources:
                    for res in resources:
                        r_path = res.get("r_path", "")
                        if r_path:
                            doc_url = f"{_DFM_FEEDS_BASE}{r_path}"
                            break

                # Determine report date from integrated_period
                report_date = ""
                period = item.get("integrated_period")
                if period:
                    report_date = f"{period}-12-31"

                filing = FilingMetadata(
                    title=headline,
                    filing_date=filing_date,
                    report_date=report_date,
                    document_url=doc_url,
                    document_format="pdf",
                    filing_type=filing_type,
                    market_id="ae_dfm",
                )
                result.filings.append(filing)

        except Exception as exc:
            result.errors.append(f"DFM eFsah API failed: {exc}")
            logger.debug("DFM eFsah discovery failed for %s: %s", ticker, exc)

        # Dedup by document URL or title
        seen: set[str] = set()
        unique: list[FilingMetadata] = []
        for f in result.filings:
            key = f.document_url or f.title
            if key not in seen:
                seen.add(key)
                unique.append(f)
        result.filings = unique

        logger.info("DFM eFsah discovery for %s: found %d filings", ticker, len(result.filings))
        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download a DFM filing PDF from feeds.dfm.ae.

        The feeds.dfm.ae server returns 406 if Accept header includes
        application/json. Must use Accept: application/pdf or */*.
        """
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")
        download_headers = {
            "User-Agent": _DFM_HEADERS["User-Agent"],
            "Accept": "application/pdf, */*",
            "Referer": "https://www.dfm.ae/",
        }
        resp = requests.get(filing.document_url, headers=download_headers, timeout=60)
        resp.raise_for_status()
        if resp.content[:4] != b"%PDF":
            raise ValueError(f"Not a PDF: {resp.headers.get('Content-Type', 'unknown')}")
        return resp.content


# ---------------------------------------------------------------------------
# SIX Swiss Exchange Filing Discoverer
# ---------------------------------------------------------------------------

# SIX official notices JSON API (undocumented, discovered from React SPA).
# Base endpoint: /sheldon/official_notices/v2/find.json
# Detail endpoint: /sheldon/official_notices/v2/details/{noticeId}.json
#
# Supported query parameters:
#   firstDate, lastDate     -- YYYYMMDD date range
#   pageNumber, pageSize    -- pagination (0-indexed)
#   sortAttribute, sortDirection -- sorting (date, desc/asc)
#   showManual              -- M-type notices (issuer corporate actions)
#   showAutomatic           -- A-type notices (automated adjustments)
#   showExDividend          -- EX-type notices (ex-dividend)
#   showFirstListing        -- FL-type notices (first listings)
#   showDelisting           -- DE-type notices (delistings)
#   showProvisional         -- PZ-type notices (provisional)
#   issuerWords             -- search by issuer name (space-separated words)
#   valorIds                -- search by ISIN or valor number
#   linkDirectly            -- filter to notices directly linked to the security
#   linkUnderlying          -- filter to notices referencing the underlying
#
# No authentication required. No rate limit documented. Free and public.

_SIX_API_BASE = "https://www.six-group.com/sheldon/official_notices/v2"
_SIX_FIND_URL = f"{_SIX_API_BASE}/find.json"
_SIX_DETAIL_URL = f"{_SIX_API_BASE}/details"

_SIX_HEADERS = {
    "User-Agent": "Operator1/1.0 (financial-research)",
    "Accept": "application/json",
}

# Delay between SIX API requests (be polite -- no documented rate limit).
_SIX_REQUEST_DELAY_S = 0.3


class SIXFilingDiscoverer:
    """Filing discoverer for SIX Swiss Exchange using the official notices API.

    Discovers corporate action notices (capital changes, dividends, share
    count updates) for SIX-listed companies via the undocumented sheldon
    JSON API that powers the official notices React SPA on six-group.com.

    Discovery approach (HKEX-inspired):
      1. Search by ISIN using ``valorIds`` with ``linkDirectly=true``
         to get notices directly related to the security.
      2. Filter by notice types: M (manual/issuer), EX (ex-dividend),
         FL (first listing), DE (delisting).
      3. Fetch detail text for each notice via the detail endpoint.
      4. Parse structured text for shares outstanding, dividend amounts,
         and other corporate action data.

    Note: SIX official notices contain corporate actions, NOT financial
    statement filings. Financial results for Swiss companies are published
    via ad-hoc disclosures (requires auth) or company IR websites.
    The notices are still valuable for:
      - Shares outstanding changes (PIT-compliant dates)
      - Ex-dividend dates and amounts
      - Capital destruction (buyback) events
      - First listing / delisting events
    """

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
        isin: str = "",
    ) -> FilingDiscovery:
        """Discover SIX official notices for a company.

        Parameters
        ----------
        ticker:
            SIX ticker symbol (e.g. 'NESN', 'NOVN', 'ROG', 'UBSG').
        years:
            How many years of notices to search for.
        isin:
            ISIN code (e.g. 'CH0038863350' for Nestle). If provided,
            used for precise ISIN-based search. Otherwise falls back
            to issuer name search.

        Returns
        -------
        FilingDiscovery with list of discovered notices.
        """
        result = FilingDiscovery(ticker=ticker, market_id="ch_six")

        end_date = date.today()
        start_date = end_date - timedelta(days=365 * years)

        try:
            # Build query params
            params: dict[str, str] = {
                "firstDate": start_date.strftime("%Y%m%d"),
                "lastDate": end_date.strftime("%Y%m%d"),
                "pageNumber": "0",
                "pageSize": "50",
                "sortAttribute": "date",
                "sortDirection": "desc",
                "showManual": "true",
                "showAutomatic": "false",
                "showExDividend": "true",
                "showFirstListing": "false",
                "showDelisting": "false",
                "showProvisional": "false",
            }

            if isin:
                # ISIN-based search (most precise)
                params["valorIds"] = isin
                params["linkDirectly"] = "true"
                params["linkUnderlying"] = "false"
            else:
                # Issuer name search (fallback)
                params["issuerWords"] = ticker

            resp = requests.get(
                _SIX_FIND_URL,
                params=params,
                headers=_SIX_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "Ok":
                result.errors.append(f"SIX API error: {data.get('status', 'unknown')}")
                return result

            items = data.get("itemList", [])
            total = data.get("totalCount", 0)

            for item in items:
                notice_id = item.get("noticeId", "")
                notice_type = item.get("noticeType", "")
                raw_date = str(item.get("date", ""))
                contact = item.get("contact", "")
                title = item.get("title", "")
                item_isin = item.get("isin") or isin

                # Parse date from YYYYMMDD integer
                filing_date = ""
                if raw_date and len(raw_date) == 8:
                    filing_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"

                # Classify filing type from notice type
                filing_type = _six_classify_notice_type(notice_type, title)

                result.filings.append(FilingMetadata(
                    title=f"{contact}: {title}".strip(": "),
                    filing_date=filing_date,
                    report_date="",  # Parsed from detail text if needed
                    document_url=f"{_SIX_DETAIL_URL}/{notice_id}.json",
                    document_format="json",  # SIX notices are structured text, not PDFs
                    filing_type=filing_type,
                    market_id="ch_six",
                    attachment_id=str(notice_id),
                    extra={
                        "notice_type": notice_type,
                        "isin": item_isin,
                        "contact": contact,
                    },
                ))

            # Dedup by notice ID
            seen: set[str] = set()
            unique: list[FilingMetadata] = []
            for f in result.filings:
                if f.attachment_id not in seen:
                    seen.add(f.attachment_id)
                    unique.append(f)
            result.filings = unique

            logger.info(
                "SIX discovery for %s (ISIN=%s): %d notices (%d total in range)",
                ticker, isin or "N/A", len(result.filings), total,
            )

        except Exception as exc:
            result.errors.append(f"SIX discovery failed: {exc}")
            logger.warning("SIX discovery failed for %s: %s", ticker, exc)

        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download a SIX notice detail as UTF-8 text bytes.

        SIX notices are structured text (not PDFs). The detail endpoint
        returns JSON with a ``text`` field containing the full notice.
        We extract the text and return it as UTF-8 bytes.
        """
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")

        time.sleep(_SIX_REQUEST_DELAY_S)

        resp = requests.get(
            filing.document_url,
            headers=_SIX_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        items = data.get("itemList", [])
        if not items:
            raise ValueError(f"SIX detail returned no items for {filing.attachment_id}")

        text = items[0].get("text", "")
        if not text:
            raise ValueError(f"SIX notice {filing.attachment_id} has no text content")

        return text.encode("utf-8")

    def get_notice_text(self, notice_id: int | str) -> str:
        """Fetch the full text of a SIX official notice.

        Convenience method for extracting structured data from notices
        (shares outstanding, dividend amounts, etc.).

        Parameters
        ----------
        notice_id:
            The numeric notice ID from the find results.

        Returns
        -------
        Full notice text as a string.
        """
        url = f"{_SIX_DETAIL_URL}/{notice_id}.json"
        resp = requests.get(url, headers=_SIX_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("itemList", [])
        if not items:
            return ""
        return items[0].get("text", "")


def _six_classify_notice_type(notice_type: str, title: str) -> str:
    """Classify a SIX notice into a filing type category."""
    nt = notice_type.upper()
    title_lower = title.lower()

    if nt == "EX":
        return "dividend"
    if nt == "FL":
        return "first_listing"
    if nt == "DE":
        return "delisting"
    if nt == "PZ":
        return "provisional"

    # M-type (manual/issuer notices) -- classify by title content
    if nt == "M":
        if any(kw in title_lower for kw in ("kapitalvernichtung", "capital destruction",
                                              "kapitalherabsetzung", "capital reduction")):
            return "capital_action"
        if any(kw in title_lower for kw in ("ruckkauf", "rueckkauf", "buyback", "repurchase")):
            return "buyback"
        if any(kw in title_lower for kw in ("dividende", "dividend")):
            return "dividend"
        if any(kw in title_lower for kw in ("annual", "jahres")):
            return "annual"
        if any(kw in title_lower for kw in ("interim", "halbjahr", "half")):
            return "interim"
        return "corporate_action"

    # A-type (automatic) -- typically structured product adjustments
    if nt == "A":
        return "automatic"

    return "other"


DISCOVERER_REGISTRY: dict[str, type] = {
    "in_bse": BSEFilingDiscoverer,
    "au_asx": ASXFilingDiscoverer,
    "hk_hkex": HKEXFilingDiscoverer,
    "sg_sgx": SGXFilingDiscoverer,
    "sa_tadawul": TadawulFilingDiscoverer,
    "ca_sedar": SEDARFilingDiscoverer,
    "za_jse": JSEFilingDiscoverer,
    "mx_bmv": BMVFilingDiscoverer,
    "ae_dfm": DFMFilingDiscoverer,
    "ch_six": SIXFilingDiscoverer,
}


def get_discoverer(market_id: str) -> FilingDiscoverer | None:
    """Return a filing discoverer for the given market, or None."""
    cls = DISCOVERER_REGISTRY.get(market_id)
    if cls is None:
        return None
    return cls()


# Module-level extraction cache to avoid redundant API calls.
# Keyed by "market_id:ticker" -> DataFrame with ALL extracted fields.
_extraction_cache: dict[str, "pd.DataFrame"] = {}

# Canonical field names by statement type for filtering.
_INCOME_FIELDS = {
    "revenue", "cost_of_revenue", "gross_profit", "operating_income",
    "ebit", "ebitda", "net_income", "interest_expense", "taxes",
    "sga_expenses", "rd_expenses", "eps", "eps_diluted",
}
_BALANCE_FIELDS = {
    "total_assets", "total_liabilities", "total_equity",
    "current_assets", "current_liabilities", "cash_and_equivalents",
    "short_term_debt", "long_term_debt", "total_debt",
    "retained_earnings", "goodwill", "intangible_assets",
    "receivables", "inventory", "payables",
}
_CASHFLOW_FIELDS = {
    "operating_cash_flow", "capex", "free_cash_flow",
    "investing_cf", "financing_cf", "dividends_paid", "stock_buybacks",
}
_STATEMENT_FIELD_MAP = {
    "income": _INCOME_FIELDS,
    "balance": _BALANCE_FIELDS,
    "cashflow": _CASHFLOW_FIELDS,
}


import threading

# Thread lock for extraction cache -- ensures only one thread does the
# discovery + LLM extraction per ticker, while others wait and use the
# cached result.  This prevents parallel statement fetches (income,
# balance, cashflow) from sending concurrent LLM requests that would
# overwhelm the API rate limit.
_extraction_lock = threading.Lock()

# Shared LLM client instance for filing extraction.  Created once on
# first use and reused across all threads, so the PooledLLMClient's
# key rotation state is shared (not duplicated per thread).
_shared_llm_client: Any = None
_shared_llm_client_lock = threading.Lock()


def _get_shared_llm_client():
    """Get or create a shared LLM client for filing extraction.

    Thread-safe: uses a lock to ensure only one client is created.
    The shared client enables consistent key rotation across threads.
    """
    global _shared_llm_client
    if _shared_llm_client is not None:
        return _shared_llm_client
    with _shared_llm_client_lock:
        if _shared_llm_client is not None:
            return _shared_llm_client
        try:
            from operator1.clients.llm_factory import create_llm_client
            from operator1.secrets_loader import load_secrets
            _shared_llm_client = create_llm_client(load_secrets())
            if _shared_llm_client is not None:
                logger.debug("Created shared LLM client for filing extraction")
        except Exception as exc:
            logger.debug("Could not create shared LLM client: %s", exc)
    return _shared_llm_client


def _try_fuzzy_extraction(
    discoverer,
    discovery,
    cache_key: str,
    market_id: str,
    ticker: str,
) -> "pd.DataFrame":
    """Try fuzzy PDF parser on discovered filings (no LLM needed).

    Downloads PDFs and uses camelot-py + fuzzy string matching to
    extract financial data.  Returns a combined DataFrame in canonical
    long format, or empty DataFrame if extraction fails.

    This is called automatically when no LLM key is available, giving
    all 9 filing-discoverer-backed markets a no-LLM fallback.
    """
    import pandas as pd

    try:
        from operator1.clients.fuzzy_pdf_parser import extract_financials_from_pdf
    except ImportError:
        logger.debug("fuzzy_pdf_parser not available for fallback")
        return pd.DataFrame()

    _type_priority = {"annual": 0, "interim": 1, "quarterly": 2}

    def _sort_key(f):
        tp = _type_priority.get(f.filing_type, 3)
        fd = f.filing_date or "0000-00-00"
        return (tp, "".join(chr(255 - ord(c)) for c in fd))

    sorted_filings = sorted(discovery.filings, key=_sort_key)[:8]

    all_records = []
    for filing in sorted_filings:
        try:
            pdf_bytes = discoverer.download_filing(filing)
        except Exception as exc:
            logger.debug("PDF download failed for fuzzy fallback: %s", exc)
            continue

        rows = extract_financials_from_pdf(
            pdf_bytes,
            filing_date=filing.filing_date or "",
            report_date=filing.report_date or "",
            market_id=market_id,
        )
        if rows:
            df = pd.DataFrame(rows)
            # Add canonical_name column (same as concept for fuzzy parser)
            if "canonical_name" not in df.columns and "concept" in df.columns:
                df["canonical_name"] = df["concept"]
            df["market_id"] = market_id
            df["currency"] = ""  # Will be filled by translator
            all_records.append(df)
            logger.info(
                "Fuzzy extracted %d concepts from %s (%s, %s)",
                len(rows), filing.title[:40] if filing.title else "?",
                filing.filing_type, filing.report_date or "?",
            )

    if not all_records:
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    for col in ("filing_date", "report_date"):
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")

    logger.info(
        "Fuzzy PDF fallback for %s/%s: %d records from %d/%d filings",
        market_id, ticker, len(combined), len(all_records), len(sorted_filings),
    )
    return combined


def try_filing_extraction(
    ticker: str,
    market_id: str,
    statement_type: str = "income",
    llm_client: Any = None,
) -> "pd.DataFrame":
    """Attempt to extract financial data via filing discovery + LLM extraction.

    This is the main integration point for Tier 2 clients. It:
    1. Discovers filings via the exchange's announcement API
    2. Downloads the filing PDF
    3. Extracts structured data via LLMFilingExtractor
    4. Returns a canonical long-format DataFrame filtered by statement_type

    Uses a per-ticker cache so that multiple calls (income, balance,
    cashflow) only trigger one discovery + download cycle.  Thread-safe:
    when parallel threads request different statement types for the same
    ticker, only one thread does the extraction and the others wait.

    Falls back to empty DataFrame if any step fails.

    Parameters
    ----------
    ticker:
        Exchange ticker or scrip code.
    market_id:
        PIT market identifier.
    statement_type:
        One of 'income', 'balance', 'cashflow'. Used to filter the
        extracted data to only return fields relevant to the requested
        statement type.
    llm_client:
        Optional LLM client for PDF extraction.  If None, uses a
        shared module-level client (created once, reused across threads).
    """
    import pandas as pd

    cache_key = f"{market_id}:{ticker}"

    # Fast path: check cache without lock (safe for dict reads)
    if cache_key in _extraction_cache:
        combined = _extraction_cache[cache_key]
        if combined.empty:
            return pd.DataFrame()
        return _filter_by_statement_type(combined, statement_type)

    # Slow path: acquire lock so only one thread does discovery + extraction.
    # Other threads for the same ticker will wait here and then hit the cache.
    # IMPORTANT: The entire extraction pipeline must run inside the lock so
    # that parallel calls (income, balance, cashflow) share results via cache
    # instead of each running discovery + download + LLM extraction independently.
    with _extraction_lock:
        # Double-check after acquiring lock (another thread may have finished)
        if cache_key in _extraction_cache:
            combined = _extraction_cache[cache_key]
            if combined.empty:
                return pd.DataFrame()
            return _filter_by_statement_type(combined, statement_type)

        discoverer = get_discoverer(market_id)
        if discoverer is None:
            logger.info("No filing discoverer registered for %s", market_id)
            return pd.DataFrame()

        try:
            discovery = discoverer.discover_filings(ticker, years=2)
        except Exception as exc:
            logger.warning("Filing discovery failed for %s/%s: %s", market_id, ticker, exc)
            _extraction_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        if not discovery.has_filings:
            logger.info("No filings discovered for %s/%s", market_id, ticker)
            _extraction_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        logger.info(
            "Filing discovery for %s/%s: %d filings found, starting extraction",
            market_id, ticker, len(discovery.filings),
        )

        # Use shared LLM client when not explicitly provided.
        # The shared client is created once and reused across all threads,
        # ensuring consistent key rotation and avoiding redundant connections.
        if llm_client is None:
            llm_client = _get_shared_llm_client()

        if llm_client is None:
            logger.info(
                "No LLM client available for %s/%s -- trying fuzzy PDF parser fallback",
                market_id, ticker,
            )
            # Fuzzy PDF parser fallback: uses camelot-py + fuzzy string
            # matching to extract financial data from PDFs without an LLM.
            fuzzy_df = _try_fuzzy_extraction(discoverer, discovery, cache_key, market_id, ticker)
            if not fuzzy_df.empty:
                _extraction_cache[cache_key] = fuzzy_df
                return _filter_by_statement_type(fuzzy_df, statement_type)

            logger.info(
                "Fuzzy PDF parser also returned empty for %s/%s -- "
                "set GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY "
                "for LLM-based extraction",
                market_id, ticker,
            )
            _extraction_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        # Try to extract from the most recent filings
        try:
            from operator1.clients.llm_filing_extractor import LLMFilingExtractor
            extractor = LLMFilingExtractor(llm_client)
        except ImportError:
            logger.info("LLMFilingExtractor not available for %s/%s", market_id, ticker)
            _extraction_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        # Sort filings by priority: annual first, then interim, then quarterly.
        # Within each type, newest filing_date first.
        _type_priority = {"annual": 0, "interim": 1, "quarterly": 2}

        def _filing_sort_key(f):
            tp = _type_priority.get(f.filing_type, 3)
            # Sort date descending within each type using character complement
            fd = f.filing_date or "0000-00-00"
            inverted_date = "".join(chr(255 - ord(c)) for c in fd)
            return (tp, inverted_date)

        sorted_filings = sorted(discovery.filings, key=_filing_sort_key)

        # Load extraction stage settings from config
        from operator1.config_loader import get_global_config
        _cfg = get_global_config()
        _max_filings = _cfg.get("filing_extraction_max_filings", 8)
        _stage_size = _cfg.get("filing_extraction_stage_size", 2)
        _stage_pause = _cfg.get("filing_extraction_stage_pause_s", 15)

        filings_to_extract = sorted_filings[:_max_filings]

        # Split into stages to respect LLM rate limits.
        # Between stages, pause to let the rate limit window reset.
        stages = [
            filings_to_extract[i:i + _stage_size]
            for i in range(0, len(filings_to_extract), _stage_size)
        ]

        logger.info(
            "Filing extraction for %s/%s: processing %d filings in %d stages",
            market_id, ticker, len(filings_to_extract), len(stages),
        )

        all_records = []
        extracted_count = 0
        download_failures = 0
        extraction_failures = 0

        for stage_num, stage_filings in enumerate(stages):
            if stage_num > 0 and _stage_pause > 0:
                logger.info(
                    "Filing extraction stage %d/%d: pausing %.0fs for rate limit reset",
                    stage_num + 1, len(stages), _stage_pause,
                )
                time.sleep(_stage_pause)

            for filing in stage_filings:
                try:
                    pdf_bytes = discoverer.download_filing(filing)
                except Exception as exc:
                    download_failures += 1
                    logger.info(
                        "PDF download failed for %s/%s '%s': %s",
                        market_id, ticker, filing.title[:40], exc,
                    )
                    continue

                try:
                    extraction = extractor.extract_from_pdf(pdf_bytes, market_id=market_id)
                    if extraction.success:
                        df = extractor.to_canonical_dataframe(extraction, market_id=market_id)
                        if not df.empty:
                            # Override filing_date from discovery metadata (more reliable)
                            if filing.filing_date:
                                df["filing_date"] = pd.Timestamp(filing.filing_date)
                            if filing.report_date:
                                df["report_date"] = pd.Timestamp(filing.report_date)
                            all_records.append(df)
                            extracted_count += 1
                            logger.info(
                                "Extracted %s (%s, %s): %d records",
                                filing.title[:50], filing.filing_type,
                                filing.report_date or "unknown period",
                                len(df),
                            )
                    else:
                        extraction_failures += 1
                        logger.info(
                            "LLM extraction returned no data for %s/%s '%s'",
                            market_id, ticker, filing.title[:40],
                        )
                except Exception as exc:
                    extraction_failures += 1
                    logger.info(
                        "LLM extraction failed for %s/%s '%s': %s",
                        market_id, ticker, filing.title[:40], exc,
                    )
                    continue

        if not all_records:
            logger.info(
                "Filing extraction for %s/%s: 0 records extracted "
                "(%d download failures, %d extraction failures out of %d filings)",
                market_id, ticker, download_failures, extraction_failures,
                len(filings_to_extract),
            )
            _extraction_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        combined = pd.concat(all_records, ignore_index=True)
        _extraction_cache[cache_key] = combined
        logger.info(
            "Filing extraction for %s/%s: %d records from %d/%d filings "
            "(%d stages, %d download failures, %d extraction failures, cached)",
            market_id, ticker, len(combined), extracted_count,
            len(filings_to_extract), len(stages),
            download_failures, extraction_failures,
        )

    # Return filtered result (cache was populated inside the lock)
    combined = _extraction_cache.get(cache_key, pd.DataFrame())
    if combined.empty:
        return pd.DataFrame()
    return _filter_by_statement_type(combined, statement_type)


def _filter_by_statement_type(df: "pd.DataFrame", statement_type: str) -> "pd.DataFrame":
    """Filter extraction result to only include fields for the requested statement type."""
    if df.empty or "canonical_name" not in df.columns:
        return df

    target_fields = _STATEMENT_FIELD_MAP.get(statement_type)
    if target_fields is None:
        return df  # unknown type, return everything

    mask = df["canonical_name"].isin(target_fields)
    filtered = df[mask].copy()
    return filtered


# ---------------------------------------------------------------------------
# Shareholding extraction pipeline
# ---------------------------------------------------------------------------

def try_shareholding_extraction(
    ticker: str,
    market_id: str,
    max_filings: int = 3,
) -> list[dict]:
    """Discover shareholding filings and extract holder data from PDFs.

    Uses the filing discoverer framework to find shareholding pattern
    filings, download their PDFs, and extract holder data using the
    fuzzy_pdf_parser's extract_shareholders_from_pdf().

    Parameters
    ----------
    ticker:
        Exchange ticker symbol.
    market_id:
        Market identifier (e.g. 'in_bse', 'sg_sgx').
    max_filings:
        Maximum number of shareholding filings to process.

    Returns
    -------
    List of holder dicts with: name, shares, percentage, holder_type, source.
    """
    discoverer = _get_discoverer(market_id)
    if discoverer is None:
        return []

    try:
        # Discover shareholding filings
        if hasattr(discoverer, 'discover_filings'):
            import inspect
            sig = inspect.signature(discoverer.discover_filings)
            if 'categories' in sig.parameters:
                discovery = discoverer.discover_filings(
                    ticker, years=2, categories=["shareholding"],
                )
            else:
                discovery = discoverer.discover_filings(ticker, years=2)
        else:
            return []

        # Filter for shareholding filings
        sh_filings = discovery.shareholding_filings()
        if not sh_filings:
            # Fallback: try annual filings (annual reports often contain
            # shareholding sections)
            sh_filings = discovery.annual_filings()[:2]

        if not sh_filings:
            return []

        logger.info(
            "Shareholding discovery for %s/%s: %d filings found",
            market_id, ticker, len(sh_filings),
        )

        # Download and extract from each filing
        from operator1.clients.fuzzy_pdf_parser import extract_shareholders_from_pdf

        all_holders: list[dict] = []
        for filing in sh_filings[:max_filings]:
            try:
                pdf_bytes = discoverer.download_filing(filing)
                if not pdf_bytes:
                    continue

                holders = extract_shareholders_from_pdf(
                    pdf_bytes,
                    filing_date=filing.filing_date,
                    market_id=market_id,
                )
                if holders:
                    all_holders.extend(holders)
                    logger.info(
                        "Shareholding extraction from '%s': %d holders",
                        filing.title[:40], len(holders),
                    )
                    break  # One successful extraction is enough
            except Exception as exc:
                logger.debug(
                    "Shareholding extraction failed for %s: %s",
                    filing.title[:40], exc,
                )
                continue

        return all_holders

    except Exception as exc:
        logger.debug("Shareholding extraction pipeline failed for %s/%s: %s",
                     market_id, ticker, exc)
        return []

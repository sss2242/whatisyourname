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
    """Discovers financial result filings from BSE India API.

    Uses the BSE corporate announcements API to find financial result
    PDFs. Each result filing contains quarterly or annual financial
    statements that can be extracted by the LLMFilingExtractor.
    """

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        """Discover financial result filings from BSE India.

        Parameters
        ----------
        ticker:
            BSE scrip code (e.g. '500325' for Reliance).
        years:
            Number of years to search back.
        """
        result = FilingDiscovery(ticker=ticker, market_id="in_bse")

        today = date.today()
        from_date = today - timedelta(days=365 * years)

        try:
            resp = requests.get(
                _BSE_ANN_URL,
                params={
                    "Ession": "",
                    "strCat": "Result",
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
            result.errors.append(f"BSE API request failed: {exc}")
            return result

        table = data.get("Table", [])
        if not table:
            result.errors.append("No financial result filings found")
            return result

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

_SGX_ANN_URL = "https://api.sgx.com/announcements/v1.0"
_SGX_ANN_HEADERS = {
    "User-Agent": "Operator1/1.0",
    "Accept": "application/json",
}


class SGXFilingDiscoverer:
    """Discovers financial result filings from SGX announcements API.

    SGX provides a public announcements endpoint that returns JSON
    with filing metadata including PDF attachment URLs.
    """

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        result = FilingDiscovery(ticker=ticker, market_id="sg_sgx")

        today = date.today()
        from_date = today - timedelta(days=365 * years)

        try:
            data = cached_get(
                _SGX_ANN_URL,
                params={
                    "company": ticker,
                    "category": "FINANCIAL_RESULTS",
                    "pagesize": "20",
                    "pagestart": "0",
                    "from": from_date.strftime("%Y-%m-%d"),
                    "to": today.strftime("%Y-%m-%d"),
                },
                headers=_SGX_ANN_HEADERS,
            )
        except Exception as exc:
            result.errors.append(f"SGX API request failed: {exc}")
            return result

        items = []
        if isinstance(data, dict):
            items = data.get("data", data.get("result", []))
        elif isinstance(data, list):
            items = data

        if not items:
            result.errors.append("No financial result filings found on SGX")
            return result

        for item in items:
            title_text = item.get("title", item.get("headline", ""))
            ann_date = item.get("date", item.get("announcementDate", ""))
            attachments = item.get("attachments", [])

            if not title_text:
                continue

            # Determine filing type from title
            lower = title_text.lower()
            if "full year" in lower or "annual" in lower:
                filing_type = "annual"
            elif "half year" in lower or "six months" in lower:
                filing_type = "interim"
            elif "quarter" in lower or "three months" in lower:
                filing_type = "quarterly"
            else:
                filing_type = "annual"

            # Parse report date from title
            report_date = ""
            rd_match = re.search(r"[Ee]nded\s+(\d{1,2})\s+(\w+)\s+(\d{4})", title_text)
            if rd_match:
                months = {
                    "january": "01", "february": "02", "march": "03", "april": "04",
                    "may": "05", "june": "06", "july": "07", "august": "08",
                    "september": "09", "october": "10", "november": "11", "december": "12",
                }
                m = rd_match.group(2).lower()
                if m in months:
                    report_date = f"{rd_match.group(3)}-{months[m]}-{int(rd_match.group(1)):02d}"

            filing_date = str(ann_date)[:10] if ann_date else ""

            # Get PDF URL from attachments
            doc_url = ""
            for att in attachments if isinstance(attachments, list) else []:
                url = att.get("url", att.get("fileUrl", ""))
                if url and url.lower().endswith(".pdf"):
                    doc_url = url if url.startswith("http") else f"https://api.sgx.com{url}"
                    break

            if not doc_url and isinstance(item.get("url", ""), str):
                doc_url = item["url"]

            filing = FilingMetadata(
                title=title_text,
                filing_date=filing_date,
                report_date=report_date,
                document_url=doc_url,
                document_format="pdf",
                filing_type=filing_type,
                market_id="sg_sgx",
            )
            result.filings.append(filing)

        logger.info(
            "SGX discovery for %s: found %d filings (%d annual, %d interim)",
            ticker, len(result.filings),
            len(result.annual_filings()), len(result.quarterly_filings()),
        )
        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download an SGX filing document."""
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")

        resp = requests.get(
            filing.document_url,
            headers=_SGX_ANN_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()

        if resp.content[:4] != b"%PDF":
            raise ValueError(
                f"Expected PDF but got {resp.headers.get('Content-Type', 'unknown')}"
            )

        logger.info(
            "Downloaded SGX filing: %s (%d bytes)",
            filing.title[:60], len(resp.content),
        )
        return resp.content


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
    "User-Agent": "Operator1/1.0",
    "Accept": "application/json",
}


class SEDARFilingDiscoverer:
    """Discovers financial filings from SEDAR+ (Canada).

    Uses the SEDAR+ document search API to find financial statements.
    SEDAR+ is the official Canadian securities filing system operated
    by the Canadian Securities Administrators (CSA).
    """

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        result = FilingDiscovery(ticker=ticker, market_id="ca_sedar")

        # First resolve ticker to SEDAR entity ID
        entity_id = self._resolve_entity(ticker)
        if not entity_id:
            result.errors.append(f"Could not resolve {ticker} on SEDAR+")
            return result

        today = date.today()
        from_date = today - timedelta(days=365 * years)

        try:
            data = cached_get(
                f"{_SEDAR_BASE}/records/companyDocuments",
                params={
                    "entityId": entity_id,
                    "category": "Annual Financial Statements",
                    "fromDate": from_date.strftime("%Y-%m-%d"),
                    "toDate": today.strftime("%Y-%m-%d"),
                    "pageSize": "10",
                },
                headers=_SEDAR_HEADERS,
            )
        except Exception as exc:
            result.errors.append(f"SEDAR+ document search failed: {exc}")
            # Try interim too
            data = None

        for category in ["Annual Financial Statements", "Interim Financial Statements"]:
            try:
                if data is None or (category != "Annual Financial Statements"):
                    data = cached_get(
                        f"{_SEDAR_BASE}/records/companyDocuments",
                        params={
                            "entityId": entity_id,
                            "category": category,
                            "fromDate": from_date.strftime("%Y-%m-%d"),
                            "toDate": today.strftime("%Y-%m-%d"),
                            "pageSize": "10",
                        },
                        headers=_SEDAR_HEADERS,
                    )
            except Exception:
                continue

            items = []
            if isinstance(data, dict):
                items = data.get("documents", data.get("data", data.get("result", [])))
            elif isinstance(data, list):
                items = data

            for item in items:
                title = item.get("title", item.get("name", ""))
                filing_dt = item.get("filingDate", item.get("date", ""))
                doc_url = item.get("documentUrl", item.get("url", ""))
                doc_id = item.get("documentId", item.get("id", ""))

                if not title:
                    continue

                is_annual = "annual" in category.lower()

                filing = FilingMetadata(
                    title=title,
                    filing_date=str(filing_dt)[:10] if filing_dt else "",
                    document_url=doc_url if doc_url and doc_url.startswith("http") else "",
                    document_format="pdf",
                    filing_type="annual" if is_annual else "interim",
                    market_id="ca_sedar",
                    attachment_id=str(doc_id),
                )
                result.filings.append(filing)

            data = None  # Reset for next category

        logger.info("SEDAR+ discovery for %s: found %d filings", ticker, len(result.filings))
        return result

    def _resolve_entity(self, ticker: str) -> str:
        """Resolve a ticker to a SEDAR+ entity ID."""
        try:
            data = cached_get(
                f"{_SEDAR_BASE}/searchCompany",
                params={"searchText": ticker},
                headers=_SEDAR_HEADERS,
            )
            items = data if isinstance(data, list) else data.get("results", []) if isinstance(data, dict) else []
            if items:
                return str(items[0].get("sedarId", items[0].get("entityId", "")))
        except Exception:
            pass
        return ""

    def download_filing(self, filing: FilingMetadata) -> bytes:
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")
        resp = requests.get(filing.document_url, headers=_SEDAR_HEADERS, timeout=30)
        resp.raise_for_status()
        if resp.content[:4] != b"%PDF":
            raise ValueError(f"Not a PDF: {resp.headers.get('Content-Type', 'unknown')}")
        return resp.content


# ---------------------------------------------------------------------------
# JSE South Africa Filing Discoverer
# ---------------------------------------------------------------------------

_JSE_SENS_URL = "https://senspdf.jse.co.za/documents/sensnews"
_JSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json, text/html",
    "Referer": "https://www.jse.co.za/",
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
    """Discovers financial result filings from JSE SENS.

    JSE SENS (Stock Exchange News Service) publishes all company
    announcements including financial results. Announcements include
    PDF attachments with the actual financial statements.

    The SENS system is the official disclosure platform for all
    JSE-listed companies. Announcement dates are the true filing dates.
    """

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        """Discover financial result filings from JSE SENS.

        Parameters
        ----------
        ticker:
            JSE ticker symbol (e.g. 'NPN' for Naspers, 'SOL' for Sasol).
        years:
            Number of years to search back.
        """
        result = FilingDiscovery(ticker=ticker, market_id="za_jse")

        today = date.today()
        from_date = today - timedelta(days=365 * years)

        # Search SENS for financial results announcements
        for search_term in ["financial results", "annual results", "interim results"]:
            try:
                resp = requests.get(
                    _JSE_SENS_URL,
                    params={
                        "keyword": f"{ticker} {search_term}",
                        "fromDate": from_date.strftime("%Y-%m-%d"),
                        "toDate": today.strftime("%Y-%m-%d"),
                        "pageSize": "20",
                    },
                    headers=_JSE_HEADERS,
                    timeout=15,
                )
                resp.raise_for_status()
            except Exception as exc:
                result.errors.append(f"JSE SENS search failed for '{search_term}': {exc}")
                continue

            # Try JSON response first
            try:
                data = resp.json()
                records = []
                if isinstance(data, dict):
                    records = data.get("data", data.get("items", data.get("result", [])))
                elif isinstance(data, list):
                    records = data

                for item in records:
                    title_text = item.get("title", item.get("headline", item.get("subject", "")))
                    ann_date = item.get("publishDate", item.get("date", item.get("releaseDate", "")))
                    doc_url = item.get("pdfUrl", item.get("documentUrl", item.get("url", "")))
                    doc_id = item.get("id", item.get("sensId", ""))

                    if not title_text:
                        continue

                    # Filter to financial results only
                    lower_title = title_text.lower()
                    if not any(kw in lower_title for kw in [
                        "financial result", "annual result", "interim result",
                        "year ended", "year ending", "half year", "six months",
                        "condensed", "audited", "reviewed",
                    ]):
                        continue

                    filing_date = ""
                    if ann_date:
                        try:
                            filing_date = str(ann_date)[:10]
                        except Exception:
                            pass

                    # Build document URL
                    if doc_url and not doc_url.startswith("http"):
                        doc_url = f"https://senspdf.jse.co.za{doc_url}"

                    report_date = _parse_jse_report_date(title_text)
                    filing_type = _classify_jse_filing_type(title_text)

                    filing = FilingMetadata(
                        title=title_text,
                        filing_date=filing_date,
                        report_date=report_date,
                        document_url=doc_url,
                        document_format="pdf",
                        filing_type=filing_type,
                        market_id="za_jse",
                        attachment_id=str(doc_id),
                    )
                    result.filings.append(filing)

            except (ValueError, AttributeError):
                # Not JSON -- try HTML parsing
                html = resp.text
                link_pattern = re.compile(
                    r'href="([^"]*\.pdf)"',
                    re.IGNORECASE,
                )
                date_pattern = re.compile(
                    r'(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})',
                )
                title_pattern = re.compile(
                    r'class="[^"]*(?:title|headline|subject)[^"]*"[^>]*>([^<]+)<',
                    re.IGNORECASE,
                )

                links = link_pattern.findall(html)
                dates = date_pattern.findall(html)
                titles = title_pattern.findall(html)

                for i, link in enumerate(links[:10]):
                    doc_url = link if link.startswith("http") else f"https://senspdf.jse.co.za{link}"
                    title_text = titles[i].strip() if i < len(titles) else search_term
                    date_str = dates[i] if i < len(dates) else ""

                    filing_date = ""
                    if date_str:
                        if "/" in date_str:
                            try:
                                parts = date_str.split("/")
                                filing_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                            except Exception:
                                pass
                        else:
                            filing_date = date_str[:10]

                    report_date = _parse_jse_report_date(title_text)
                    filing_type = _classify_jse_filing_type(title_text)

                    filing = FilingMetadata(
                        title=title_text,
                        filing_date=filing_date,
                        report_date=report_date,
                        document_url=doc_url,
                        document_format="pdf",
                        filing_type=filing_type,
                        market_id="za_jse",
                    )
                    result.filings.append(filing)

        # Dedup by document URL
        seen_urls: set[str] = set()
        unique: list[FilingMetadata] = []
        for f in result.filings:
            if f.document_url and f.document_url not in seen_urls:
                seen_urls.add(f.document_url)
                unique.append(f)
            elif not f.document_url:
                unique.append(f)
        result.filings = unique

        logger.info(
            "JSE discovery for %s: found %d filings (%d annual, %d interim)",
            ticker, len(result.filings),
            len(result.annual_filings()), len(result.quarterly_filings()),
        )
        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download a JSE SENS filing document."""
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")

        resp = requests.get(
            filing.document_url,
            headers=_JSE_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()

        # Validate it's a PDF
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

_BMV_EMISNET_URL = "https://emisnet.bmv.com.mx/informacion-financiera"
_BMV_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json, text/html",
    "Referer": "https://www.bmv.com.mx/",
}


class BMVFilingDiscoverer:
    """Discovers financial filings from BMV/CNBV (Mexico).

    Uses the BMV EMISNET disclosure portal to find financial statement
    announcements. EMISNET is the official electronic disclosure system
    for all BMV-listed companies.

    The BMV API is undocumented, so this discoverer tries multiple
    endpoint patterns and falls back to HTML parsing.
    """

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        """Discover financial filings from BMV EMISNET.

        Parameters
        ----------
        ticker:
            BMV ticker symbol (e.g. 'AMXL', 'WALMEX', 'FEMSAUBD').
        years:
            Number of years to search back.
        """
        result = FilingDiscovery(ticker=ticker, market_id="mx_bmv")

        today = date.today()
        from_date = today - timedelta(days=365 * years)

        # Try the BMV issuer profile / financial info endpoints
        for endpoint in [
            f"https://www.bmv.com.mx/en/issuers/financial-information/{ticker}",
            f"https://emisnet.bmv.com.mx/2009/emisoras_reportes.html?cb_emisora={ticker}&cb_tipo_doc=Informacion+Financiera",
        ]:
            try:
                resp = requests.get(
                    endpoint,
                    headers=_BMV_HEADERS,
                    timeout=15,
                )
                resp.raise_for_status()
            except Exception as exc:
                result.errors.append(f"BMV endpoint failed: {exc}")
                continue

            # Try JSON first
            try:
                data = resp.json()
                records = []
                if isinstance(data, dict):
                    records = data.get("data", data.get("items", data.get("result", [])))
                elif isinstance(data, list):
                    records = data

                for item in records:
                    title_text = item.get("title", item.get("descripcion", item.get("nombre", "")))
                    ann_date = item.get("date", item.get("fecha", item.get("fechaPublicacion", "")))
                    doc_url = item.get("url", item.get("archivo", item.get("documentUrl", "")))

                    if not title_text:
                        continue

                    filing_date = str(ann_date)[:10] if ann_date else ""

                    lower = title_text.lower()
                    if "anual" in lower or "annual" in lower or "year" in lower:
                        filing_type = "annual"
                    elif "trimestral" in lower or "quarter" in lower:
                        filing_type = "quarterly"
                    elif "semestral" in lower or "interim" in lower:
                        filing_type = "interim"
                    else:
                        filing_type = "quarterly"

                    # Parse report date
                    report_date = ""
                    rd_match = re.search(
                        r"(\d{4})[/-](\d{2})[/-](\d{2})", str(ann_date)
                    )
                    if rd_match:
                        report_date = f"{rd_match.group(1)}-{rd_match.group(2)}-{rd_match.group(3)}"

                    if doc_url and not doc_url.startswith("http"):
                        doc_url = f"https://emisnet.bmv.com.mx{doc_url}"

                    filing = FilingMetadata(
                        title=title_text,
                        filing_date=filing_date,
                        report_date=report_date,
                        document_url=doc_url,
                        document_format="pdf",
                        filing_type=filing_type,
                        market_id="mx_bmv",
                    )
                    result.filings.append(filing)

            except (ValueError, AttributeError):
                # HTML response -- parse for PDF links
                html = resp.text
                link_pattern = re.compile(
                    r'href="([^"]*\.pdf)"',
                    re.IGNORECASE,
                )
                links = link_pattern.findall(html)
                for link in links[:10]:
                    doc_url = link if link.startswith("http") else f"https://emisnet.bmv.com.mx{link}"
                    filing = FilingMetadata(
                        title=f"{ticker} financial filing",
                        document_url=doc_url,
                        document_format="pdf",
                        filing_type="quarterly",
                        market_id="mx_bmv",
                    )
                    result.filings.append(filing)

        # Dedup
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
        """Download a BMV/EMISNET filing document."""
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")
        resp = requests.get(filing.document_url, headers=_BMV_HEADERS, timeout=30)
        resp.raise_for_status()
        if resp.content[:4] != b"%PDF":
            raise ValueError(f"Not a PDF: {resp.headers.get('Content-Type', 'unknown')}")
        return resp.content


# ---------------------------------------------------------------------------
# DFM/ADX UAE Filing Discoverer
# ---------------------------------------------------------------------------

_DFM_DISC_URL = "https://www.dfm.ae/api/DisclosureFilesApi/GetCompanyDisclosures"
_ADX_DISC_URL = "https://www.adx.ae/api/disclosures"
_DFM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.dfm.ae/",
}


class DFMFilingDiscoverer:
    """Discovers financial filings from DFM and ADX (UAE).

    Tries both Dubai Financial Market (DFM) and Abu Dhabi Securities
    Exchange (ADX) disclosure APIs. Both exchanges publish company
    financial statements through their disclosure portals.

    UAE adopted IFRS for all listed companies, so standard IFRS
    canonical mapping applies.
    """

    def discover_filings(
        self,
        ticker: str,
        years: int = 2,
    ) -> FilingDiscovery:
        """Discover financial filings from DFM/ADX.

        Parameters
        ----------
        ticker:
            DFM or ADX ticker symbol (e.g. 'EMAAR', 'ETISALAT').
        years:
            Number of years to search back.
        """
        result = FilingDiscovery(ticker=ticker, market_id="ae_dfm")

        today = date.today()
        from_date = today - timedelta(days=365 * years)

        # Try DFM disclosure API
        for endpoint, exchange_name, headers in [
            (_DFM_DISC_URL, "DFM", _DFM_HEADERS),
            (_ADX_DISC_URL, "ADX", {**_DFM_HEADERS, "Referer": "https://www.adx.ae/"}),
        ]:
            try:
                resp = requests.get(
                    endpoint,
                    params={
                        "symbol": ticker,
                        "companySymbol": ticker,
                        "fromDate": from_date.strftime("%Y-%m-%d"),
                        "toDate": today.strftime("%Y-%m-%d"),
                        "category": "Financial",
                        "pageSize": "20",
                    },
                    headers=headers,
                    timeout=15,
                )
                resp.raise_for_status()
            except Exception as exc:
                result.errors.append(f"{exchange_name} disclosure API failed: {exc}")
                continue

            try:
                data = resp.json()
                records = []
                if isinstance(data, dict):
                    records = data.get("data", data.get("items", data.get("disclosures", [])))
                elif isinstance(data, list):
                    records = data

                for item in records:
                    # Try both English and Arabic field names
                    title_text = item.get("titleEn", item.get("title", item.get("subject", "")))
                    ann_date = item.get("publishDate", item.get("date", item.get("disclosureDate", "")))
                    doc_url = item.get("pdfUrl", item.get("fileUrl", item.get("documentUrl", "")))

                    if not title_text:
                        title_text = item.get("titleAr", "")
                    if not title_text:
                        continue

                    # Filter financial disclosures
                    lower = title_text.lower()
                    if not any(kw in lower for kw in [
                        "financial", "annual", "interim", "quarter",
                        "result", "statement", "report",
                        # Arabic keywords
                        "مالي", "سنوي", "نتائج", "بيانات",
                    ]):
                        continue

                    filing_date = str(ann_date)[:10] if ann_date else ""

                    if "annual" in lower or "سنوي" in lower or "year" in lower:
                        filing_type = "annual"
                    elif "interim" in lower or "half" in lower or "six" in lower:
                        filing_type = "interim"
                    elif "quarter" in lower:
                        filing_type = "quarterly"
                    else:
                        filing_type = "annual"

                    if doc_url and not doc_url.startswith("http"):
                        base = "https://www.dfm.ae" if exchange_name == "DFM" else "https://www.adx.ae"
                        doc_url = f"{base}{doc_url}"

                    filing = FilingMetadata(
                        title=title_text,
                        filing_date=filing_date,
                        document_url=doc_url,
                        document_format="pdf",
                        filing_type=filing_type,
                        market_id="ae_dfm",
                    )
                    result.filings.append(filing)

            except (ValueError, AttributeError):
                # HTML response -- try to find PDF links
                html = resp.text
                link_pattern = re.compile(r'href="([^"]*\.pdf)"', re.IGNORECASE)
                for link in link_pattern.findall(html)[:10]:
                    base = "https://www.dfm.ae" if exchange_name == "DFM" else "https://www.adx.ae"
                    doc_url = link if link.startswith("http") else f"{base}{link}"
                    filing = FilingMetadata(
                        title=f"{ticker} financial disclosure ({exchange_name})",
                        document_url=doc_url,
                        document_format="pdf",
                        filing_type="annual",
                        market_id="ae_dfm",
                    )
                    result.filings.append(filing)

        # Dedup
        seen: set[str] = set()
        unique: list[FilingMetadata] = []
        for f in result.filings:
            key = f.document_url or f.title
            if key not in seen:
                seen.add(key)
                unique.append(f)
        result.filings = unique

        logger.info("DFM/ADX discovery for %s: found %d filings", ticker, len(result.filings))
        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download a DFM/ADX filing document."""
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")
        resp = requests.get(filing.document_url, headers=_DFM_HEADERS, timeout=30)
        resp.raise_for_status()
        if resp.content[:4] != b"%PDF":
            raise ValueError(f"Not a PDF: {resp.headers.get('Content-Type', 'unknown')}")
        return resp.content


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
    with _extraction_lock:
        # Double-check after acquiring lock (another thread may have finished)
        if cache_key in _extraction_cache:
            combined = _extraction_cache[cache_key]
            if combined.empty:
                return pd.DataFrame()
            return _filter_by_statement_type(combined, statement_type)

    discoverer = get_discoverer(market_id)
    if discoverer is None:
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

    # Use shared LLM client when not explicitly provided.
    # The shared client is created once and reused across all threads,
    # ensuring consistent key rotation and avoiding redundant connections.
    if llm_client is None:
        llm_client = _get_shared_llm_client()

    # Try to extract from the most recent filings
    try:
        from operator1.clients.llm_filing_extractor import LLMFilingExtractor
        extractor = LLMFilingExtractor(llm_client)
    except ImportError:
        logger.debug("LLMFilingExtractor not available")
        _extraction_cache[cache_key] = pd.DataFrame()
        return pd.DataFrame()

    all_records = []

    for filing in discovery.filings[:8]:  # Limit to 8 most recent
        try:
            pdf_bytes = discoverer.download_filing(filing)
        except Exception as exc:
            logger.debug("Download failed for %s: %s", filing.title[:40], exc)
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
        except Exception as exc:
            logger.debug("Extraction failed for %s: %s", filing.title[:40], exc)
            continue

    if not all_records:
        _extraction_cache[cache_key] = pd.DataFrame()
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    _extraction_cache[cache_key] = combined
    logger.info(
        "Filing extraction for %s/%s: %d records from %d filings (cached)",
        market_id, ticker, len(combined), len(all_records),
    )
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

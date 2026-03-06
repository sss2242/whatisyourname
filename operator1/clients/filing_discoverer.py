"""Filing discoverer framework for Tier 2 markets.

Discovers financial filing URLs from exchange announcement systems so the
LLMFilingExtractor can download and extract structured data from them.

Each exchange has a different announcement API. The FilingDiscoverer protocol
defines the common interface; per-market implementations handle the specifics.

Currently implemented:
  - BSEFilingDiscoverer (India) -- full pipeline: discovery + PDF download
  - ASXFilingDiscoverer (Australia) -- announcement discovery via MarkitDigital

Markets without structured APIs (HKEX, SGX, BMV, JSE, SIX, Tadawul, DFM,
SEDAR+) continue to use yfinance as fallback.
"""

from __future__ import annotations

import logging
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
            elif "half year" in lower or "appendix 4d" in lower:
                filing_type = "interim"
            else:
                filing_type = "quarterly"

            filing = FilingMetadata(
                title=headline,
                filing_date=filing_date,
                report_date="",  # ASX doesn't provide this in the announcement
                document_url="",  # Document download URL TBD
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
        """Download an ASX filing document.

        Currently raises NotImplementedError as the MarkitDigital
        document download URL pattern needs further investigation.
        """
        if not filing.attachment_id:
            raise ValueError("No document key in filing metadata")

        # Try the MarkitDigital document endpoint
        try:
            resp = requests.get(
                f"{_ASX_MARKIT_BASE}/documents/{filing.attachment_id}",
                headers=_ASX_HEADERS,
                timeout=30,
            )
            if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                return resp.content
        except Exception:
            pass

        raise NotImplementedError(
            f"ASX document download not yet implemented for key {filing.attachment_id}. "
            "The MarkitDigital document endpoint returns 404 for some documents."
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DISCOVERER_REGISTRY: dict[str, type] = {
    "in_bse": BSEFilingDiscoverer,
    "au_asx": ASXFilingDiscoverer,
}


def get_discoverer(market_id: str) -> FilingDiscoverer | None:
    """Return a filing discoverer for the given market, or None."""
    cls = DISCOVERER_REGISTRY.get(market_id)
    if cls is None:
        return None
    return cls()


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
    4. Returns a canonical long-format DataFrame

    Falls back to empty DataFrame if any step fails.

    Parameters
    ----------
    ticker:
        Exchange ticker or scrip code.
    market_id:
        PIT market identifier.
    statement_type:
        One of 'income', 'balance', 'cashflow'.
    llm_client:
        Optional LLM client for PDF extraction.
    """
    import pandas as pd

    discoverer = get_discoverer(market_id)
    if discoverer is None:
        return pd.DataFrame()

    try:
        discovery = discoverer.discover_filings(ticker, years=2)
    except Exception as exc:
        logger.warning("Filing discovery failed for %s/%s: %s", market_id, ticker, exc)
        return pd.DataFrame()

    if not discovery.has_filings:
        logger.info("No filings discovered for %s/%s", market_id, ticker)
        return pd.DataFrame()

    # Try to extract from the most recent filings
    try:
        from operator1.clients.llm_filing_extractor import LLMFilingExtractor
        extractor = LLMFilingExtractor(llm_client)
    except ImportError:
        logger.debug("LLMFilingExtractor not available")
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
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    logger.info(
        "Filing extraction for %s/%s: %d records from %d filings",
        market_id, ticker, len(combined), len(all_records),
    )
    return combined

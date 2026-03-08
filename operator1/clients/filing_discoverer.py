"""Filing discoverer framework for Tier 2 markets.

Discovers financial filing URLs from exchange announcement systems so the
LLMFilingExtractor can download and extract structured data from them.

Each exchange has a different announcement API. The FilingDiscoverer protocol
defines the common interface; per-market implementations handle the specifics.

Currently implemented:
  - BSEFilingDiscoverer (India) -- full pipeline: discovery + PDF download
  - ASXFilingDiscoverer (Australia) -- announcement discovery via MarkitDigital
  - HKEXFilingDiscoverer (Hong Kong) -- HKEX News title search + PDF download

Markets without structured APIs (SGX, BMV, JSE, SIX, Tadawul, DFM,
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

    Uses the HKEX News title search to find annual/interim result
    announcements, then extracts PDF document URLs for LLM extraction.

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

        today = date.today()
        from_date = today - timedelta(days=365 * years)

        # Search for annual and interim results
        for search_term in ["annual results", "interim results"]:
            try:
                resp = requests.get(
                    _HKEX_SEARCH_URL,
                    params={
                        "lang": "EN",
                        "category": "0",
                        "market": "SEHK",
                        "searchType": "0",
                        "documentType": "-1",
                        "t1code": "-2",
                        "t2Gcode": "-2",
                        "t2code": "-2",
                        "stockId": code,
                        "from": from_date.strftime("%Y%m%d"),
                        "to": today.strftime("%Y%m%d"),
                        "title": search_term,
                        "rowRange": "20",
                        "sortDir": "desc",
                        "sortByDate": "desc",
                    },
                    headers=_HKEX_HEADERS,
                    timeout=15,
                )
                resp.raise_for_status()
            except Exception as exc:
                result.errors.append(f"HKEX search failed for '{search_term}': {exc}")
                continue

            # HKEX returns HTML with the results.  Parse the title search
            # results which are in a structured table/JSON depending on the
            # response format.  We try JSON first, then fall back to HTML regex.
            try:
                data = resp.json()
                records = data.get("result", data.get("data", []))
                if isinstance(records, list):
                    for item in records:
                        title_text = item.get("title", item.get("TITLE", ""))
                        file_link = item.get("file_link", item.get("FILE_LINK", ""))
                        date_str = item.get("release_date", item.get("RELEASE_DATE", ""))

                        if not title_text:
                            continue

                        # Build document URL
                        doc_url = ""
                        if file_link:
                            if file_link.startswith("http"):
                                doc_url = file_link
                            else:
                                doc_url = f"https://www1.hkexnews.hk{file_link}"

                        filing_date = ""
                        if date_str:
                            try:
                                filing_date = str(date_str)[:10]
                            except Exception:
                                pass

                        report_date = _parse_hkex_report_date(title_text)
                        filing_type = _classify_hkex_filing_type(title_text)

                        filing = FilingMetadata(
                            title=title_text,
                            filing_date=filing_date,
                            report_date=report_date,
                            document_url=doc_url,
                            document_format="pdf",
                            filing_type=filing_type,
                            market_id="hk_hkex",
                        )
                        result.filings.append(filing)
            except (ValueError, AttributeError):
                # Not JSON -- try HTML parsing with regex
                html = resp.text
                # Pattern: look for links to PDF documents with dates
                link_pattern = re.compile(
                    r'href="([^"]*\.pdf)"[^>]*>.*?</a>',
                    re.IGNORECASE | re.DOTALL,
                )
                date_pattern = re.compile(
                    r'(\d{2}/\d{2}/\d{4})',
                )
                title_pattern = re.compile(
                    r'class="[^"]*title[^"]*"[^>]*>([^<]+)<',
                    re.IGNORECASE,
                )

                # Extract whatever structured info we can from HTML
                links = link_pattern.findall(html)
                dates = date_pattern.findall(html)
                titles = title_pattern.findall(html)

                for i, link in enumerate(links[:10]):
                    doc_url = link if link.startswith("http") else f"https://www1.hkexnews.hk{link}"
                    title_text = titles[i] if i < len(titles) else search_term
                    date_str = dates[i] if i < len(dates) else ""
                    filing_date = ""
                    if date_str:
                        try:
                            parts = date_str.split("/")
                            filing_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                        except Exception:
                            pass

                    report_date = _parse_hkex_report_date(title_text)
                    filing_type = _classify_hkex_filing_type(title_text)

                    filing = FilingMetadata(
                        title=title_text.strip(),
                        filing_date=filing_date,
                        report_date=report_date,
                        document_url=doc_url,
                        document_format="pdf",
                        filing_type=filing_type,
                        market_id="hk_hkex",
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
            "HKEX discovery for %s: found %d filings (%d annual, %d interim)",
            code, len(result.filings),
            len(result.annual_filings()), len(result.quarterly_filings()),
        )
        return result

    def download_filing(self, filing: FilingMetadata) -> bytes:
        """Download an HKEX filing document."""
        if not filing.document_url:
            raise ValueError("No document URL in filing metadata")

        resp = requests.get(
            filing.document_url,
            headers=_HKEX_HEADERS,
            timeout=60,  # HKEX PDFs can be large
        )
        resp.raise_for_status()

        # Validate it's a PDF
        if resp.content[:4] != b"%PDF":
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

DISCOVERER_REGISTRY: dict[str, type] = {
    "in_bse": BSEFilingDiscoverer,
    "au_asx": ASXFilingDiscoverer,
    "hk_hkex": HKEXFilingDiscoverer,
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
    cashflow) only trigger one discovery + download cycle.

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
        Optional LLM client for PDF extraction.
    """
    import pandas as pd

    cache_key = f"{market_id}:{ticker}"

    # Check extraction cache first (avoids redundant API calls)
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

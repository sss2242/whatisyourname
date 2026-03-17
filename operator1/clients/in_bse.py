"""India BSE/NSE PIT client -- BSE announcements API + pdfplumber extraction.

Primary financials: BSE AnnSubCategoryGetData API (discovers filings) +
    pdfplumber PDF table extraction (no LLM needed).
Secondary: Filing discoverer + LLM extraction for complex PDFs.
Profile: yfinance (.NS/.BO suffix) for sector/industry metadata.
OHLCV: handled separately via ohlcv_provider.py (yfinance/nselib).

The BSE announcements API works globally (no geo-blocking) and returns
filing metadata with PDF attachment UUIDs.  PDFs are downloaded from
bseindia.com and parsed with pdfplumber to extract SEBI-format
financial result tables.

Coverage: ~5,500+ listed companies on BSE/NSE, ~$4T market cap.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_BSE_BASE = "https://api.bseindia.com/BseIndiaAPI/api"
_BSE_ANN_BASE = f"{_BSE_BASE}/AnnSubCategoryGetData/w"
_BSE_PDF_BASE = "https://www.bseindia.com/xml-data/corpfiling/AttachLive"
_CACHE_DIR = Path("cache/in_bse")

# Headers needed for BSE API requests
_BSE_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bseindia.com/",
}

# Ind AS / IFRS concept mapping for BSE financial result PDFs.
# BSE SEBI-format results use standard Indian Accounting Standards
# line item labels.  Map these to canonical field names.
_INDAS_INCOME_MAP: dict[str, str] = {
    "revenue from operations": "revenue",
    "total income": "revenue",
    "value of sales & services": "revenue",
    "total expenses": "cost_of_revenue",
    "profit before tax": "ebit",
    "profit before exceptional items and tax": "ebit",
    "profit/(loss) before tax": "ebit",
    "profit/(loss) before exceptional items and tax": "ebit",
    "profit after tax": "net_income",
    "profit/(loss) after tax": "net_income",
    "net profit": "net_income",
    "net profit/(loss)": "net_income",
    "profit/(loss) for the period": "net_income",
    "total comprehensive income": "net_income",
    "tax expense": "taxes",
    "exceptional items": "interest_expense",
    "depreciation": "sga_expenses",
    "earnings per share": "eps",
    "basic eps": "eps",
    "diluted eps": "eps_diluted",
    "basic": "eps",
    "diluted": "eps_diluted",
}

_INDAS_BALANCE_MAP: dict[str, str] = {
    "total assets": "total_assets",
    "total equity": "total_equity",
    "total liabilities": "total_liabilities",
    "net worth": "total_equity",
    "paid up equity share capital": "shares_outstanding",
    "reserves": "retained_earnings",
    "reserves excluding revaluation reserves": "retained_earnings",
}


class INBseClient:
    """PIT client for Indian BSE/NSE equities.

    Uses the BSE announcements API to discover financial result filings,
    downloads PDFs, and extracts structured data using pdfplumber.
    No LLM required for standard SEBI-format results.
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
        """List Indian companies from BSE API search."""
        if not query:
            return []
        try:
            import requests
            resp = requests.get(
                f"{_BSE_BASE}/Suggest/Getstockdata/{query}",
                headers=_BSE_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            items = resp.json() if resp.text.strip().startswith("[") else []
            return [
                {
                    "ticker": str(i.get("scrip_cd", "")),
                    "name": i.get("scripname", ""),
                    "cik": str(i.get("scrip_cd", "")),
                    "exchange": "BSE",
                    "country": "IN",
                    "market_id": self.market_id,
                }
                for i in items
                if isinstance(i, dict)
            ]
        except Exception as exc:
            logger.debug("BSE company search failed: %s", exc)
            # Fallback: yfinance search
            try:
                from operator1.clients.yfinance_backed import yf_search
                return yf_search(query, self.market_id, "IN", "BSE", yf_suffix=".NS")
            except Exception:
                return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
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

        # Primary: yfinance for rich profile data (sector, industry, name)
        self._enrich_from_yfinance(identifier, raw)

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def _enrich_from_yfinance(self, identifier: str, raw: dict) -> None:
        """Enrich profile with yfinance data (.NS suffix for NSE)."""
        try:
            import yfinance as yf

            for suffix in [".NS", ".BO"]:
                yf_ticker = f"{identifier}{suffix}"
                t = yf.Ticker(yf_ticker)
                info = t.info or {}

                if info.get("longName") or info.get("shortName"):
                    raw["name"] = info.get("longName") or info.get("shortName") or ""
                    raw["sector"] = info.get("sector", "")
                    raw["industry"] = info.get("industry", "")
                    mc = info.get("marketCap")
                    if mc:
                        raw["market_cap"] = str(mc)
                    shares = info.get("sharesOutstanding")
                    if shares:
                        raw["shares_outstanding"] = str(shares)
                    raw["isin"] = info.get("isin", "")
                    raw["exchange"] = info.get("exchange", "NSI")
                    logger.info("yfinance enriched %s via %s: %s", identifier, yf_ticker, raw["name"])
                    break

        except Exception as exc:
            logger.debug("yfinance profile failed for %s: %s", identifier, exc)

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch income statements via BSE announcements + PDF extraction."""
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        """Fetch balance sheets via BSE announcements + PDF extraction."""
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch cash flow statements via BSE announcements + PDF extraction."""
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial data using BSE announcements API + pdfplumber.

        Strategy:
        1. Query BSE AnnSubCategoryGetData API for financial result filings
        2. Download the PDF attachments
        3. Extract tables with pdfplumber (no LLM needed)
        4. Parse SEBI-format financial line items
        5. Map to canonical names via Ind AS concept mapping

        Fallback: Filing discoverer + LLM extraction for complex PDFs.
        """
        # Step 1: Discover filings via BSE announcements API
        filings = self._discover_filings_via_api(identifier)
        if not filings:
            logger.debug("No BSE filings found for %s", identifier)
            return self._try_filing_discoverer_fallback(identifier, statement_type)

        # Step 2-4: Download PDFs and extract financial data
        all_rows: list[dict] = []
        for filing in filings[:8]:  # Cap at 8 filings (2 years of quarterly)
            try:
                pdf_bytes = self._download_pdf(filing["pdf_url"])
                if not pdf_bytes:
                    continue

                rows = self._extract_financials_from_pdf(
                    pdf_bytes, filing, statement_type,
                )
                all_rows.extend(rows)
            except Exception as exc:
                logger.debug(
                    "PDF extraction failed for %s filing %s: %s",
                    identifier, filing.get("filing_date", "?"), exc,
                )

        if not all_rows:
            logger.debug("No rows extracted from PDFs for %s/%s", identifier, statement_type)
            return self._try_filing_discoverer_fallback(identifier, statement_type)

        # Step 5: Build DataFrame and translate to canonical names
        df = pd.DataFrame(all_rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Filter to 2-year window
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
                "BSE %s/%s: %d rows extracted from %d PDF filings",
                identifier, statement_type, len(result), len(filings),
            )
        return result

    def _discover_filings_via_api(self, identifier: str) -> list[dict]:
        """Query BSE AnnSubCategoryGetData API for financial result filings.

        This endpoint works globally (no geo-blocking) and returns
        announcement metadata with PDF attachment UUIDs.
        """
        try:
            import requests
            end_date = date.today()
            start_date = end_date - timedelta(days=730)

            resp = requests.get(
                _BSE_ANN_BASE,
                params={
                    "strCat": "Result",
                    "strPrevDate": start_date.strftime("%Y%m%d"),
                    "strToDate": end_date.strftime("%Y%m%d"),
                    "strScrip": str(identifier),
                    "strSearch": "P",
                    "strType": "C",
                },
                headers=_BSE_HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get("Table", []) if isinstance(data, dict) else []
            filings = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                attachment = item.get("ATTACHMENTNAME", "")
                if not attachment:
                    continue

                # Parse filing date from the announcement timestamp
                news_dt = item.get("NEWS_DT", item.get("DT_TM", ""))
                filing_date = ""
                if news_dt:
                    try:
                        filing_date = pd.Timestamp(news_dt).strftime("%Y-%m-%d")
                    except Exception:
                        filing_date = str(news_dt)[:10]

                # Parse report period end date from the subject line
                subject = item.get("NEWSSUB", item.get("HEADLINE", ""))
                report_date = self._parse_report_date_from_subject(subject)

                filings.append({
                    "pdf_url": f"{_BSE_PDF_BASE}/{attachment}",
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "subject": subject,
                    "company": item.get("SLONGNAME", ""),
                })

            logger.info("BSE announcements for %s: %d filings found", identifier, len(filings))
            return filings

        except Exception as exc:
            logger.debug("BSE announcements API failed for %s: %s", identifier, exc)
            return []

    @staticmethod
    def _parse_report_date_from_subject(subject: str) -> str:
        """Extract the fiscal period end date from a BSE filing subject.

        Examples:
            'Financial Results For The Quarter Ended December 31, 2025'
            -> '2025-12-31'
            'Financial Results For The Quarter And Half Year Ended September 30, 2025'
            -> '2025-09-30'
        """
        if not subject:
            return ""
        # Look for patterns like "ended December 31, 2025" or "ended 31.12.2025"
        patterns = [
            r"ended?\s+(\w+)\s+(\d{1,2}),?\s+(\d{4})",
            r"ended?\s+(\d{1,2})[./](\d{1,2})[./](\d{4})",
            r"ended?\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        ]
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "jun": 6, "jul": 7, "aug": 8, "sep": 9,
            "oct": 10, "nov": 11, "dec": 12,
        }

        subject_lower = subject.lower()
        # Pattern 1: "ended December 31, 2025"
        m = re.search(r"ended?\s+(\w+)\s+(\d{1,2}),?\s+(\d{4})", subject_lower)
        if m:
            month_str, day, year = m.group(1), m.group(2), m.group(3)
            month = month_map.get(month_str)
            if month:
                return f"{year}-{month:02d}-{int(day):02d}"

        # Pattern 2: "ended 31 December 2025"
        m = re.search(r"ended?\s+(\d{1,2})\s+(\w+),?\s+(\d{4})", subject_lower)
        if m:
            day, month_str, year = m.group(1), m.group(2), m.group(3)
            month = month_map.get(month_str)
            if month:
                return f"{year}-{month:02d}-{int(day):02d}"

        return ""

    @staticmethod
    def _download_pdf(url: str) -> bytes | None:
        """Download a PDF from BSE."""
        try:
            import requests
            resp = requests.get(url, headers=_BSE_HEADERS, timeout=30)
            resp.raise_for_status()
            if resp.content[:4] != b"%PDF":
                logger.debug("Not a PDF from %s", url)
                return None
            return resp.content
        except Exception as exc:
            logger.debug("PDF download failed: %s", exc)
            return None

    def _extract_financials_from_pdf(
        self,
        pdf_bytes: bytes,
        filing: dict,
        statement_type: str,
    ) -> list[dict]:
        """Extract financial line items from a BSE SEBI-format PDF.

        Uses pdfplumber to find tables.  If no tables are found,
        falls back to text-based extraction of key line items.
        """
        rows: list[dict] = []
        filing_date = filing.get("filing_date", "")
        report_date = filing.get("report_date", "")

        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed -- cannot extract BSE PDFs")
            return rows

        concept_map = (
            _INDAS_INCOME_MAP if statement_type == "income"
            else _INDAS_BALANCE_MAP if statement_type == "balance"
            else {}  # cashflow uses text extraction
        )

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
            # Try table extraction first
            for page in doc.pages:
                tables = page.extract_tables()
                for table in tables:
                    rows.extend(
                        self._parse_table_rows(table, concept_map, filing_date, report_date)
                    )

            # If no tables found, try text-based extraction
            if not rows:
                full_text = "\n".join(
                    (page.extract_text() or "") for page in doc.pages
                )
                rows.extend(
                    self._extract_from_text(full_text, concept_map, filing_date, report_date)
                )

        return rows

    def _parse_table_rows(
        self,
        table: list[list],
        concept_map: dict[str, str],
        filing_date: str,
        report_date: str,
    ) -> list[dict]:
        """Parse a pdfplumber table into canonical financial rows."""
        rows: list[dict] = []
        if not table or len(table) < 2:
            return rows

        for row in table:
            if not row or not row[0]:
                continue
            label = str(row[0]).strip().lower()
            # Clean up multi-line cell content
            label = re.sub(r"\s+", " ", label).strip()

            # Check if this label maps to a canonical concept
            canonical = None
            for pattern, canon in concept_map.items():
                if pattern in label:
                    canonical = canon
                    break

            if not canonical:
                continue

            # Try to get the most recent quarter value (usually column 1 or 2)
            value = None
            for cell in row[1:4]:  # Check first 3 value columns
                if cell is None:
                    continue
                cell_str = str(cell).strip()
                # Clean number: remove commas, spaces, handle negatives
                cell_str = re.sub(r"[,\s]", "", cell_str)
                cell_str = cell_str.replace("(", "-").replace(")", "")
                # Extract first number from potentially multi-value cells
                num_match = re.search(r"-?[\d.]+", cell_str)
                if num_match:
                    try:
                        value = float(num_match.group())
                        break
                    except ValueError:
                        continue

            if value is not None:
                rows.append({
                    "concept": canonical,
                    "value": value,
                    "filing_date": filing_date,
                    "report_date": report_date,
                })

        return rows

    def _extract_from_text(
        self,
        text: str,
        concept_map: dict[str, str],
        filing_date: str,
        report_date: str,
    ) -> list[dict]:
        """Extract financial line items from raw PDF text.

        This handles PDFs where pdfplumber can't detect tables but
        the text still contains recognizable financial line items
        with numbers.
        """
        rows: list[dict] = []
        if not text:
            return rows

        lines = text.split("\n")
        for line in lines:
            line_lower = line.strip().lower()
            if not line_lower:
                continue

            for pattern, canonical in concept_map.items():
                if pattern not in line_lower:
                    continue

                # Already mapped this concept? Skip duplicates
                if any(r["concept"] == canonical for r in rows):
                    break

                # Find numbers on this line or nearby
                numbers = re.findall(r"-?[\d,]+\.?\d*", line)
                # Filter out tiny numbers (likely footnote refs) and years
                numbers = [
                    n for n in numbers
                    if len(n.replace(",", "").replace(".", "")) >= 2
                    and not (1900 < float(n.replace(",", "")) < 2100)
                ]
                if numbers:
                    try:
                        val = float(numbers[0].replace(",", ""))
                        rows.append({
                            "concept": canonical,
                            "value": val,
                            "filing_date": filing_date,
                            "report_date": report_date,
                        })
                    except ValueError:
                        pass
                break

        return rows

    def _try_filing_discoverer_fallback(
        self, identifier: str, statement_type: str,
    ) -> pd.DataFrame:
        """Fallback: use the filing discoverer + LLM extraction path."""
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
            df = try_filing_extraction(
                ticker=identifier,
                market_id=self.market_id,
                statement_type=statement_type,
            )
            if df is not None and not df.empty:
                logger.info(
                    "BSE filing discoverer fallback succeeded for %s/%s: %d rows",
                    identifier, statement_type, len(df),
                )
                return df
        except Exception as exc:
            logger.debug("BSE filing discoverer fallback failed for %s: %s", identifier, exc)

        return pd.DataFrame()

    # -- Price data -----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """BSE does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    # -- Peers / related entities --------------------------------------------

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

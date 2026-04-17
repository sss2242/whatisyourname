"""US SEC EDGAR PIT client -- powered by edgartools + sec-edgar-api.

Replaces the original sec_edgar.py with unofficial wrapper libraries
that provide richer, more reliable access to SEC EDGAR data.

Primary library: edgartools (https://github.com/dgunning/edgartools)
  - Company search, profile, financial statements, filings
  - XBRL parsing with structured DataFrames
  - Built-in rate limiting and caching

Fallback library: sec-edgar-api (https://github.com/jadchaar/sec-edgar-api)
  - Raw JSON access to SEC endpoints
  - companyfacts, submissions, company concepts

Coverage: ~10,000+ public companies, $50T+ market cap.
API: https://data.sec.gov (free, no key required)

Important: SEC requires a User-Agent header with contact info.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# SEC requires identifying User-Agent
_DEFAULT_USER_AGENT = "Operator1/1.0 (https://github.com/Abdu2024/OP-1)"

# Cache directory for this market
_CACHE_DIR = Path("cache/us_sec_edgar")

# US-GAAP concept -> canonical name mapping (for fallback raw XBRL extraction)
_USGAAP_INCOME_CONCEPTS: dict[str, str] = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "CostOfGoodsAndServicesSold": "cost_of_revenue",
    "CostOfRevenue": "cost_of_revenue",
    "GrossProfit": "gross_profit",
    "OperatingIncomeLoss": "operating_income",
    "NetIncomeLoss": "net_income",
    "EarningsPerShareBasic": "eps_basic",
    "EarningsPerShareDiluted": "eps_diluted",
    "InterestExpense": "interest_expense",
    "IncomeTaxExpenseBenefit": "taxes",
    "SellingGeneralAndAdministrativeExpense": "sga_expense",
    "ResearchAndDevelopmentExpense": "research_and_development",
}

_USGAAP_BALANCE_CONCEPTS: dict[str, str] = {
    "Assets": "total_assets",
    "Liabilities": "total_liabilities",
    "StockholdersEquity": "total_equity",
    "AssetsCurrent": "current_assets",
    "LiabilitiesCurrent": "current_liabilities",
    "CashAndCashEquivalentsAtCarryingValue": "cash_and_equivalents",
    "ShortTermBorrowings": "short_term_debt",
    "LongTermDebt": "long_term_debt",
    "LongTermDebtNoncurrent": "long_term_debt",
    "RetainedEarningsAccumulatedDeficit": "retained_earnings",
    "Goodwill": "goodwill",
    "IntangibleAssetsNetExcludingGoodwill": "intangible_assets",
    "AccountsReceivableNetCurrent": "receivables",
    "InventoryNet": "inventory",
    "AccountsPayableCurrent": "payables",
}

_USGAAP_CASHFLOW_CONCEPTS: dict[str, str] = {
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "NetCashProvidedByUsedInInvestingActivities": "investing_cf",
    "NetCashProvidedByUsedInFinancingActivities": "financing_cf",
    "PaymentsOfDividends": "dividends_paid",
    "PaymentsOfDividendsCommonStock": "dividends_paid",
    "PaymentsForRepurchaseOfCommonStock": "buybacks",
}


class USEdgarError(Exception):
    """Raised on US SEC EDGAR wrapper failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"US EDGAR error on {endpoint}: {detail}")


class USEdgarClient:
    """Point-in-time client for SEC EDGAR (US equities) using edgartools.

    Implements the ``PITClient`` protocol.  All financial statements
    include ``filing_date`` (the SEC acceptance date) alongside the
    fiscal period ``report_date``.

    Uses edgartools as primary library for rich Company/Financials API,
    with sec-edgar-api as fallback for raw XBRL data extraction.

    Parameters
    ----------
    user_agent:
        SEC requires a User-Agent with your name/email.
    cache_dir:
        Local cache directory for profile and filings JSON.
    """

    def __init__(
        self,
        user_agent: str = "",
        cache_dir: Path | str = _CACHE_DIR,
    ) -> None:
        # Prefer EDGAR_IDENTITY env var, then SEC_EDGAR_EMAIL, fall back to provided arg or default
        self._user_agent = (
            user_agent
            or os.environ.get("EDGAR_IDENTITY", "")
            or os.environ.get("SEC_EDGAR_EMAIL", "")
            or _DEFAULT_USER_AGENT
        )
        # SEC requires email in User-Agent; if EDGAR_IDENTITY is just an email, wrap it
        if "@" in self._user_agent and "/" not in self._user_agent:
            self._user_agent = f"Operator1/1.0 ({self._user_agent})"
        self._cache_dir = Path(cache_dir)
        self._edgar_initialized = False
        self._sec_api_client = None

        # In-memory caches
        self._company_cache: dict[str, Any] = {}
        self._company_list_cache: list[dict[str, Any]] | None = None

    # -- Lazy initialization --------------------------------------------------

    def _init_edgartools(self) -> None:
        """Initialize edgartools with required SEC identity."""
        if self._edgar_initialized:
            return
        try:
            import edgar
            edgar.set_identity(self._user_agent)
            self._edgar_initialized = True
            logger.info("edgartools initialized with identity: %s", self._user_agent)
        except ImportError:
            logger.warning("edgartools not installed; falling back to sec-edgar-api")
        except Exception as exc:
            logger.warning("edgartools init failed: %s; falling back to sec-edgar-api", exc)

    def _get_sec_api_client(self):
        """Get or create the sec-edgar-api fallback client."""
        if self._sec_api_client is None:
            try:
                from sec_edgar_api import EdgarClient
                self._sec_api_client = EdgarClient(user_agent=self._user_agent)
                logger.info("sec-edgar-api client initialized")
            except ImportError:
                logger.error("Neither edgartools nor sec-edgar-api is installed")
                raise USEdgarError("init", "No SEC EDGAR library available")
        return self._sec_api_client

    def _get_edgar_company(self, identifier: str):
        """Get an edgartools Company object, with caching.

        VERIFIED AGAINST OFFICIAL DOCS:
        - Date: 2026-02-24
        - Version: edgartools@5.17.1
        - Docs: https://github.com/dgunning/edgartools
        - Research Log: .roo/research/us-sec-edgar-2026-02-24.md
        - Breaking change: v5.16+ raises CompanyNotFoundError instead of
          setting company.not_found (Section 4 of research log)
        """
        if identifier in self._company_cache:
            return self._company_cache[identifier]

        self._init_edgartools()
        if not self._edgar_initialized:
            raise USEdgarError("Company", "edgartools not available (init failed)")
        import edgar
        try:
            company = edgar.Company(identifier)
        except Exception as exc:
            # edgartools v5.16+ raises CompanyNotFoundError for invalid identifiers
            raise USEdgarError("Company", f"Company not found: {identifier} ({exc})") from exc
        # Backward compat: older edgartools versions may still use not_found attribute
        if getattr(company, "not_found", False):
            raise USEdgarError("Company", f"Company not found: {identifier}")
        self._company_cache[identifier] = company
        return company

    # -- Cache helpers --------------------------------------------------------

    def _cache_path(self, identifier: str, filename: str) -> Path:
        """Build a cache file path: cache/us_sec_edgar/{identifier}/{filename}."""
        safe_id = identifier.replace("/", "_").replace("\\", "_").upper()
        return self._cache_dir / safe_id / filename

    def _read_cache(self, identifier: str, filename: str) -> dict | None:
        """Read a cached JSON file if it exists and is fresh (< 7 days for profile)."""
        path = self._cache_path(identifier, filename)
        if not path.exists():
            return None
        try:
            stat = path.stat()
            age_days = (date.today() - date.fromtimestamp(stat.st_mtime)).days
            # Profile: refresh weekly; filings: keep until explicitly replaced
            if filename == "profile.json" and age_days > 7:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Cache read failed for %s/%s: %s", identifier, filename, exc)
            return None

    def _write_cache(self, identifier: str, filename: str, data: dict) -> None:
        """Write data to cache as JSON."""
        path = self._cache_path(identifier, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    # -- Protocol properties -------------------------------------------------

    @property
    def market_id(self) -> str:
        return "us_sec_edgar"

    @property
    def market_name(self) -> str:
        return "United States (NYSE / NASDAQ) -- SEC EDGAR"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List all SEC-registered companies, optionally filtered by query.

        Uses edgartools' built-in ticker lookup or falls back to
        sec-edgar-api's submissions endpoint.
        """
        if self._company_list_cache is None:
            try:
                self._init_edgartools()
                import edgar
                tickers_data = edgar.get_company_tickers()
                if hasattr(tickers_data, "to_dict"):
                    # It's a DataFrame
                    self._company_list_cache = [
                        {
                            "ticker": row.get("ticker", ""),
                            "name": row.get("title", row.get("name", "")),
                            "cik": str(row.get("cik", row.get("cik_str", ""))),
                            "exchange": "",
                            "market_id": self.market_id,
                        }
                        for _, row in tickers_data.iterrows()
                    ]
                elif isinstance(tickers_data, dict):
                    self._company_list_cache = [
                        {
                            "ticker": v.get("ticker", ""),
                            "name": v.get("title", ""),
                            "cik": str(v.get("cik_str", "")),
                            "exchange": "",
                            "market_id": self.market_id,
                        }
                        for v in tickers_data.values()
                    ]
                else:
                    self._company_list_cache = []
            except Exception as exc:
                logger.warning("edgartools list_companies failed: %s; trying sec-edgar-api", exc)
                try:
                    client = self._get_sec_api_client()
                    # sec-edgar-api doesn't have a direct list; use the
                    # same SEC bulk file our old client used
                    import requests
                    resp = requests.get(
                        "https://www.sec.gov/files/company_tickers.json",
                        headers={"User-Agent": self._user_agent},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    self._company_list_cache = [
                        {
                            "ticker": v.get("ticker", ""),
                            "name": v.get("title", ""),
                            "cik": str(v.get("cik_str", "")),
                            "exchange": "",
                            "market_id": self.market_id,
                        }
                        for v in data.values()
                    ]
                except Exception as exc2:
                    logger.error("Both company list methods failed: %s", exc2)
                    self._company_list_cache = []

        if not query:
            return self._company_list_cache

        q = query.lower()
        return [
            c for c in self._company_list_cache
            if q in c["name"].lower() or q in c["ticker"].lower()
        ]

    def search_company(self, name: str) -> list[dict[str, Any]]:
        """Search for a company by name or ticker.

        Uses a direct HTTP request to the SEC company tickers JSON
        for reliable, fast search without depending on edgartools
        (which can hang on network calls in some environments).
        """
        # Fast path: direct SEC tickers search (no edgartools dependency)
        try:
            import requests
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": self._user_agent},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            query = name.lower()
            results = []
            if isinstance(data, dict):
                for entry in data.values():
                    ticker = str(entry.get("ticker", "")).upper()
                    title = str(entry.get("title", ""))
                    cik = str(entry.get("cik_str", ""))
                    if query in title.lower() or query in ticker.lower():
                        results.append({
                            "ticker": ticker,
                            "name": title,
                            "cik": cik,
                            "exchange": "",
                            "market_id": self.market_id,
                        })
            if results:
                return results[:50]  # Cap at 50 results
        except Exception as exc:
            logger.warning("SEC tickers search failed: %s", exc)

        # Fallback to cached company list
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from SEC EDGAR.

        Uses edgartools Company object as primary source, with cached
        profile data on disk.

        Parameters
        ----------
        identifier:
            Ticker symbol (e.g. "AAPL") or CIK number.
        """
        # Check cache first
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        raw_profile: dict[str, Any] = {}

        try:
            company = self._get_edgar_company(identifier)
            raw_profile = {
                "name": company.name or "",
                "ticker": company.get_ticker() or identifier,
                "cik": str(company.cik),
                "isin": "",  # SEC doesn't provide ISINs natively
                "country": "US",
                "sector": company.industry or "",
                "industry": company.industry or "",
                "exchange": company.get_exchanges()[0] if company.get_exchanges() else "",
                "currency": "USD",
                "fiscal_year_end": str(company.fiscal_year_end or ""),
                "sic": str(company.sic or ""),
                "shares_outstanding": company.shares_outstanding,
            }
        except Exception as exc:
            logger.warning("edgartools profile failed for %s: %s; trying sec-edgar-api", identifier, exc)
            try:
                client = self._get_sec_api_client()
                cik = self._resolve_cik_fallback(identifier)
                data = client.get_submissions(cik)
                raw_profile = {
                    "name": data.get("name", ""),
                    "ticker": data.get("tickers", [""])[0] if data.get("tickers") else identifier,
                    "cik": cik,
                    "isin": "",
                    "country": "US",
                    "sector": data.get("sicDescription", ""),
                    "industry": data.get("sicDescription", ""),
                    "exchange": data.get("exchanges", [""])[0] if data.get("exchanges") else "",
                    "currency": "USD",
                    "fiscal_year_end": data.get("fiscalYearEnd", ""),
                    "sic": data.get("sic", ""),
                }
            except Exception as exc2:
                # Third fallback: direct requests to SEC JSON endpoints (no library needed)
                try:
                    raw_profile = self._fetch_profile_direct_requests(identifier)
                except Exception as exc3:
                    raise USEdgarError(
                        "get_profile",
                        f"All three methods failed for {identifier}: "
                        f"edgartools={exc}, sec-edgar-api={exc2}, direct={exc3}",
                    ) from exc3

        # Translate to canonical profile format
        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw_profile, self.market_id)

        # Cache to disk
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def _fetch_profile_direct_requests(self, identifier: str) -> dict[str, Any]:
        """Fetch profile using direct requests to SEC JSON endpoints.

        This is the last-resort fallback when neither edgartools nor
        sec-edgar-api is available or working. Uses only the stdlib
        requests library with proper User-Agent headers.

        SEC endpoints used:
        - /files/company_tickers.json -- ticker-to-CIK mapping
        - /cgi-bin/browse-edgar?action=getcompany&CIK=... -- submissions
        - /api/xbrl/companyfacts/CIK{cik}.json -- company facts
        """
        import requests

        headers = {"User-Agent": self._user_agent, "Accept": "application/json"}

        # Step 1: Resolve ticker to CIK
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        tickers_data = resp.json()

        cik = ""
        company_name = ""
        ticker_upper = identifier.upper()
        for entry in tickers_data.values():
            if str(entry.get("ticker", "")).upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                company_name = entry.get("title", "")
                break
            if str(entry.get("cik_str", "")) == identifier:
                cik = str(entry["cik_str"]).zfill(10)
                company_name = entry.get("title", "")
                ticker_upper = str(entry.get("ticker", identifier)).upper()
                break

        if not cik:
            raise USEdgarError("get_profile", f"Ticker/CIK not found: {identifier}")

        # Step 2: Fetch company submissions for metadata
        subs_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(subs_url, headers=headers, timeout=15)

        raw_profile: dict[str, Any] = {
            "name": company_name,
            "ticker": ticker_upper,
            "cik": cik,
            "isin": "",
            "country": "US",
            "currency": "USD",
        }

        if resp.status_code == 200:
            data = resp.json()
            raw_profile.update({
                "name": data.get("name", company_name),
                "sector": data.get("sicDescription", ""),
                "industry": data.get("sicDescription", ""),
                "exchange": (data.get("exchanges") or [""])[0],
                "fiscal_year_end": data.get("fiscalYearEnd", ""),
                "sic": data.get("sic", ""),
            })

        logger.info("Direct requests profile for %s: %s", identifier, raw_profile.get("name"))
        return raw_profile

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch income statements with filing_date and report_date columns.

        Primary: edgartools Company.income_statement() with filing metadata.
        Fallback: sec-edgar-api raw XBRL companyfacts extraction.
        """
        # Try edgartools first
        try:
            df = self._fetch_statement_edgartools(identifier, "income")
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.warning("edgartools income_statement failed for %s: %s", identifier, exc)

        # Fallback to raw XBRL
        return self._fetch_statement_fallback(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        """Fetch balance sheets with filing_date and report_date columns."""
        try:
            df = self._fetch_statement_edgartools(identifier, "balance")
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.warning("edgartools balance_sheet failed for %s: %s", identifier, exc)

        return self._fetch_statement_fallback(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch cash flow statements with filing_date and report_date columns."""
        try:
            df = self._fetch_statement_edgartools(identifier, "cashflow")
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.warning("edgartools cashflow_statement failed for %s: %s", identifier, exc)

        return self._fetch_statement_fallback(identifier, "cashflow")

    # -- Segment / product data extraction -----------------------------------

    def extract_segment_data(self, identifier: str) -> dict[str, Any]:
        """Extract segment revenue and product descriptions from SEC EDGAR XBRL.

        US companies disclose segment data under ASC 280 using XBRL dimensions.
        The companyfacts JSON contains segment-level revenue facts with
        dimensional qualifiers (e.g., ``us-gaap:RevenueFromExternalCustomers``
        with ``srt:ProductOrServiceAxis`` dimension members).

        Three extraction paths:
        1. **companyfacts XBRL dimensions** -- structured segment revenue from
           SEC EDGAR companyfacts JSON with ASC 280 dimensional qualifiers
        2. **10-K filing text** -- parse segment notes from the annual report
           using fuzzy_pdf_parser text patterns
        3. **edgartools Financials** -- if dimensional data is exposed

        Returns
        -------
        Dict with ``segments``, ``descriptions``, ``has_revenue``,
        ``has_descriptions``, ``n_segments``.
        """
        empty: dict[str, Any] = {
            "segments": {}, "descriptions": {},
            "has_revenue": False, "has_descriptions": False, "n_segments": 0,
        }

        # --- Path 1: companyfacts XBRL dimensional segment data ---
        try:
            segments = self._extract_segments_from_companyfacts(identifier)
            if segments and segments.get("n_segments", 0) >= 2:
                segments["source"] = "sec_edgar_xbrl_dimensions"
                logger.info(
                    "US segment data for %s: %d segments from XBRL companyfacts",
                    identifier, segments["n_segments"],
                )
                return segments
        except Exception as exc:
            logger.debug("SEC XBRL segment extraction failed for %s: %s", identifier, exc)

        # --- Path 2: 10-K filing text parsing via fuzzy_pdf_parser ---
        try:
            segments = self._extract_segments_from_10k(identifier)
            if segments and segments.get("n_segments", 0) >= 2:
                segments["source"] = "sec_edgar_10k_text"
                logger.info(
                    "US segment data for %s: %d segments from 10-K text",
                    identifier, segments["n_segments"],
                )
                return segments
        except Exception as exc:
            logger.debug("SEC 10-K segment extraction failed for %s: %s", identifier, exc)

        return empty

    def _extract_segments_from_companyfacts(self, identifier: str) -> dict[str, Any]:
        """Extract ASC 280 segment revenue from SEC EDGAR companyfacts XBRL.

        SEC companyfacts JSON contains dimensional facts where the segment
        axis (``us-gaap:StatementBusinessSegmentsAxis`` or
        ``srt:ProductOrServiceAxis``) has dimension members that ARE the
        segment/product names.

        The structure is:
        ``facts -> us-gaap -> RevenueFromExternalCustomers -> units -> USD``
        where each entry has a ``dimensions`` dict containing the segment member.
        """
        import requests

        # Resolve CIK
        try:
            cik = self._resolve_cik_fallback(identifier)
        except Exception:
            return {}

        headers = {"User-Agent": self._user_agent, "Accept": "application/json"}

        # Fetch companyfacts
        try:
            facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            resp = requests.get(facts_url, headers=headers, timeout=30)
            resp.raise_for_status()
            facts = resp.json()
        except Exception as exc:
            logger.debug("companyfacts fetch failed: %s", exc)
            return {}

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        # ASC 280 segment revenue concepts
        segment_concepts = [
            "RevenueFromExternalCustomers",
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ]

        # Segment axis dimension keys
        segment_axes = [
            "us-gaap:StatementBusinessSegmentsAxis",
            "srt:ConsolidationItemsAxis",
            "us-gaap:StatementOperatingActivitiesSegmentAxis",
        ]

        # Product axis dimension keys
        product_axes = [
            "srt:ProductOrServiceAxis",
            "us-gaap:ProductOrServiceAxis",
        ]

        segments: dict[str, float] = {}
        products: dict[str, list[str]] = {}
        segment_details: dict[str, dict[str, float]] = {}

        for concept_name in segment_concepts:
            concept_data = us_gaap.get(concept_name, {})
            usd_entries = concept_data.get("units", {}).get("USD", [])

            for entry in usd_entries:
                form = entry.get("form", "")
                if form not in ("10-K", "20-F"):
                    continue

                # Check for dimensional qualifiers
                # SEC companyfacts v2 may embed dimensions in the fact itself
                # Look for segment member in the 'segment' or 'dimensions' field
                frame = entry.get("frame", "")
                val = entry.get("val")
                if val is None:
                    continue

                # Recent entries (last 2 years)
                fy = entry.get("fy", 0)
                if fy and fy < 2023:
                    continue

                # SEC EDGAR v2 companyfacts don't expose dimensions directly
                # in the bulk JSON. The dimensional data is in the company-concept
                # endpoint. We'll try that separately.

            if segments:
                break

        # If bulk companyfacts didn't yield segment dimensions, try the
        # company-concept endpoint which includes dimensional breakdowns
        if not segments:
            for concept_name in segment_concepts:
                try:
                    concept_url = (
                        f"https://data.sec.gov/api/xbrl/companyconcept/"
                        f"CIK{cik}/us-gaap/{concept_name}.json"
                    )
                    resp = requests.get(concept_url, headers=headers, timeout=15)
                    if resp.status_code != 200:
                        continue

                    concept_data = resp.json()
                    usd_entries = concept_data.get("units", {}).get("USD", [])

                    # Group by fiscal year, look for entries with same fy/fp
                    # but different fact values (indicates segment breakdown)
                    from collections import defaultdict
                    by_period: dict[str, list[dict]] = defaultdict(list)
                    for entry in usd_entries:
                        form = entry.get("form", "")
                        if form not in ("10-K", "20-F"):
                            continue
                        fy = entry.get("fy", 0)
                        if fy and fy < 2023:
                            continue
                        period_key = f"{entry.get('fy', '')}-{entry.get('fp', '')}"
                        by_period[period_key].append(entry)

                    # Find the most recent period with multiple entries
                    # (multiple entries for same concept = segment breakdown)
                    for period_key in sorted(by_period.keys(), reverse=True):
                        entries = by_period[period_key]
                        if len(entries) >= 3:
                            # Multiple revenue entries = likely segment breakdown
                            # Unfortunately, the SEC company-concept endpoint
                            # doesn't include the dimension member names in the
                            # JSON response. The segment names are only in the
                            # inline XBRL of the actual filing.
                            logger.debug(
                                "SEC %s has %d entries for %s in %s (likely segments)",
                                identifier, len(entries), concept_name, period_key,
                            )
                            break

                    if segments:
                        break
                except Exception:
                    continue

        # SEC EDGAR bulk APIs don't expose segment dimension member names.
        # Return what we have (may be empty, triggering fallback to 10-K text)
        n_segments = len(segments)
        return {
            "segments": segments,
            "products": products,
            "descriptions": {},
            "segment_details": segment_details,
            "has_revenue": n_segments >= 2,
            "has_descriptions": False,
            "n_segments": n_segments,
        }

    def _extract_segments_from_10k(self, identifier: str) -> dict[str, Any]:
        """Extract segment data from 10-K or 20-F filing text.

        Downloads the most recent annual filing HTML and uses the fuzzy
        PDF parser's text extraction patterns to find ASC 280 / IFRS 8
        segment disclosures.

        Tries 10-K first (US domestic filers), then 20-F (foreign filers
        like Toyota, Sony, Honda that have US ADR listings).
        """
        import requests

        try:
            self._init_edgartools()
            company = self._get_edgar_company(identifier)
            if company is None:
                return {}

            # Get most recent annual filing: 10-K (domestic) or 20-F (foreign ADR)
            latest_10k = None
            for form_type in ("10-K", "20-F"):
                filings = company.get_filings(form=form_type)
                if filings is not None and len(filings) > 0:
                    latest_10k = list(filings)[:1][0]
                    logger.debug("Found %s filing for %s: %s", form_type, identifier, latest_10k.filing_date)
                    break
            if latest_10k is None:
                return {}

            # Extract text from the filing
            text = ""
            try:
                text = latest_10k.text()[:50000] if hasattr(latest_10k, "text") else ""
            except Exception:
                pass

            if not text:
                return {}

            # Use text-based segment extraction patterns
            from operator1.clients.fuzzy_pdf_parser import (
                _extract_segments_from_text,
                _parse_segment_descriptions,
                _parse_segment_paragraphs,
            )

            segments = _extract_segments_from_text(text)
            descriptions: dict[str, str] = {}

            if len(segments) >= 2:
                # Try to extract descriptions from the same text
                descriptions = _parse_segment_descriptions(text)
                if len(descriptions) < 2:
                    descriptions = _parse_segment_paragraphs(text)

            n_segments = max(len(segments), len(descriptions))
            return {
                "segments": segments,
                "descriptions": descriptions,
                "has_revenue": len(segments) >= 2,
                "has_descriptions": len(descriptions) >= 1,
                "n_segments": n_segments,
            }

        except Exception as exc:
            logger.debug("10-K segment extraction failed for %s: %s", identifier, exc)
            return {}

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """SEC EDGAR does not provide OHLCV price data.

        SEC EDGAR does not provide OHLCV data.
        Returns an empty DataFrame.
        """
        logger.debug(
            "SEC EDGAR does not provide OHLCV data for %s.",
            identifier,
        )
        return pd.DataFrame()

    # -- Peers / executives --------------------------------------------------

    def get_peers(self, identifier: str) -> list[str]:
        """Return peer companies based on SIC code matching.

        Queries the SEC browse-edgar endpoint to find companies with the
        same 4-digit SIC code, then maps CIKs back to tickers using the
        company tickers list.  Returns up to 10 peer tickers.
        """
        profile = self.get_profile(identifier)
        sic = profile.get("sic", "")
        target_ticker = profile.get("ticker", identifier).upper()
        if not sic:
            return []

        sic_4 = str(sic).zfill(4)

        # Step 1: Build a CIK -> ticker lookup from the company tickers list
        try:
            import requests
            tickers_data = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": self._user_agent},
                timeout=30,
            ).json()
            cik_to_ticker: dict[str, str] = {}
            if isinstance(tickers_data, dict):
                for entry in tickers_data.values():
                    cik_str = str(entry.get("cik_str", "")).lstrip("0")
                    ticker = str(entry.get("ticker", "")).upper()
                    if cik_str and ticker:
                        cik_to_ticker[cik_str] = ticker
        except Exception:
            return []

        # Step 2: Query SEC browse-edgar for companies with the same SIC
        peers = self._query_peers_by_sic(sic_4, target_ticker, cik_to_ticker)

        # Fallback to 2-digit SIC (broad sector) if fewer than 5 exact peers
        if len(peers) < 5:
            sic_2 = sic_4[:2]
            broad_peers = self._query_peers_by_sic(
                sic_2, target_ticker, cik_to_ticker, exclude=set(peers),
            )
            remaining = 10 - len(peers)
            peers.extend(broad_peers[:remaining])

        logger.info(
            "SEC EDGAR peers for %s (SIC %s): %d returned",
            target_ticker, sic_4, len(peers),
        )
        return peers[:10]

    def _query_peers_by_sic(
        self,
        sic: str,
        target_ticker: str,
        cik_to_ticker: dict[str, str],
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Find peer companies matching a SIC code.

        Uses the SEC EFTS full-text search API (modern, reliable) instead
        of the deprecated cgi-bin/browse-edgar endpoint which returns 503.

        Falls back to filtering the company_tickers.json list by SIC prefix
        if the EFTS query fails.

        Returns a list of resolved ticker symbols (up to 10).
        """
        import requests

        exclude = exclude or set()
        peers: list[str] = []

        # Method 1: Use EFTS search API (modern, reliable)
        try:
            resp = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "q": f"SIC:{sic}",
                    "dateRange": "custom",
                    "startdt": "2024-01-01",
                    "forms": "10-K,10-Q",
                },
                headers={"User-Agent": self._user_agent},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                for hit in hits:
                    source = hit.get("_source", {})
                    entity_name = source.get("entity_name", "")
                    cik = str(source.get("entity_id", "")).lstrip("0")
                    ticker = cik_to_ticker.get(cik, "")
                    if ticker and ticker != target_ticker and ticker not in peers and ticker not in exclude:
                        peers.append(ticker)
                    if len(peers) >= 10:
                        return peers
                if peers:
                    return peers
        except Exception as exc:
            logger.debug("EFTS SIC search failed: %s", exc)

        # Method 2: Filter from already-loaded company tickers by SIC prefix
        # The cik_to_ticker map was built from company_tickers.json which
        # also contains SIC codes. Use edgartools if available.
        try:
            from edgar import get_companies
            sic_matches = get_companies(sic=int(sic))
            if sic_matches is not None:
                for _, row in sic_matches.iterrows() if hasattr(sic_matches, "iterrows") else []:
                    ticker = str(row.get("ticker", ""))
                    if ticker and ticker != target_ticker and ticker not in peers and ticker not in exclude:
                        peers.append(ticker)
                    if len(peers) >= 10:
                        break
        except Exception as exc:
            logger.debug("edgartools SIC lookup failed: %s", exc)

        return peers

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        """SEC doesn't have a direct executives endpoint; returns empty."""
        return []

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch beneficial owners from SEC EDGAR SC 13D/13G filings.

        Searches EDGAR for Schedule 13D/13G filings (required for >5%
        beneficial ownership) and parses filer names and ownership
        percentages.  Also extracts ownership data from the most recent
        DEF 14A proxy statement's "Security Ownership" section via the
        LLM filing extractor or fuzzy PDF parser.

        Returns list of dicts with: name, shares, percentage, value,
        holder_type, date_reported, source.
        """
        holders: list[dict[str, Any]] = []

        # --- Path 1: SC 13D/13G filings (>5% beneficial owners) ---
        # IMPORTANT: edgartools' filing.company attribute returns the ISSUER
        # (e.g. "Apple Inc."), NOT the filer (e.g. "Berkshire Hathaway Inc").
        # For SC 13G/13D filings, the filer IS the institutional holder.
        # We extract the filer name from the SEC-HEADER in the filing's
        # .txt file (first 3000 bytes), which contains:
        #   FILED BY:
        #     COMPANY CONFORMED NAME: BERKSHIRE HATHAWAY INC
        try:
            import re
            import requests as _requests

            self._init_edgartools()
            company = self._get_edgar_company(identifier)
            if company is not None:
                seen_filers: set[str] = set()
                for form_type in ("SC 13G/A", "SC 13G", "SC 13D/A", "SC 13D"):
                    try:
                        filings = company.get_filings(form=form_type)
                        if filings is None or len(filings) == 0:
                            continue
                        # Use list() to avoid edgartools pyarrow slicing bug
                        # (filings[:N] raises AttributeError on ChunkedArray.as_py)
                        for filing in list(filings)[:20]:
                            try:
                                filing_date = str(getattr(filing, "filing_date", ""))

                                # Extract filer name from SEC-HEADER via text_url
                                # This is the only reliable way to get the actual
                                # filing entity (not the issuer) for SC 13G/13D.
                                filer_name = ""
                                pct = 0.0
                                text_url = getattr(filing, "text_url", "")
                                if text_url:
                                    try:
                                        resp = _requests.get(
                                            text_url,
                                            headers={"User-Agent": self._user_agent},
                                            timeout=10,
                                            stream=True,
                                        )
                                        if resp.status_code == 200:
                                            # Read only first 4KB for the SEC-HEADER
                                            header_chunk = next(resp.iter_content(4096), b"")
                                            resp.close()
                                            header_text = header_chunk.decode("utf-8", errors="replace")

                                            # Extract FILED BY company name
                                            filer_match = re.search(
                                                r"FILED BY:.*?COMPANY CONFORMED NAME:\s*([^\n]+)",
                                                header_text,
                                                re.DOTALL | re.IGNORECASE,
                                            )
                                            if filer_match:
                                                filer_name = filer_match.group(1).strip()
                                        else:
                                            resp.close()
                                    except Exception:
                                        pass

                                # Fallback: use filing.company (issuer name)
                                if not filer_name:
                                    filer_name = str(getattr(filing, "company", ""))
                                if not filer_name:
                                    continue

                                # Skip if same filer already seen
                                filer_key = filer_name.upper().strip()
                                if filer_key in seen_filers:
                                    continue
                                # Skip if the filer is the issuer itself
                                issuer_name = str(getattr(company, "name", "")).upper()
                                if filer_key == issuer_name:
                                    continue
                                seen_filers.add(filer_key)

                                # Try to extract percentage from the filing HTML
                                try:
                                    text = filing.text()[:8000] if hasattr(filing, "text") else ""
                                    # SC 13G Item 11: Percent of Class
                                    pct_match = re.search(
                                        r"(?:percent\s+of\s+class|percent\s+of\s+shares)[:\s]*(\d{1,3}(?:\.\d+)?)\s*%",
                                        text, re.IGNORECASE,
                                    )
                                    if not pct_match:
                                        pct_match = re.search(
                                            r"(\d{1,3}\.\d+)\s*%",
                                            text,
                                        )
                                    if pct_match:
                                        pct = float(pct_match.group(1))
                                        if pct > 90:
                                            pct = 0.0  # Likely not ownership %
                                except Exception:
                                    pass

                                holders.append({
                                    "name": filer_name,
                                    "shares": 0,
                                    "value": 0.0,
                                    "percentage": round(pct, 2),
                                    "holder_type": "institutional",
                                    "date_reported": filing_date,
                                    "source": f"sec_edgar_{form_type.lower().replace(' ', '_').replace('/', '')}",
                                })
                            except Exception:
                                continue
                    except Exception:
                        continue
                if holders:
                    logger.info(
                        "US holders for %s: %d from SEC EDGAR SC 13D/13G",
                        identifier, len(holders),
                    )
        except Exception as exc:
            logger.debug("SEC EDGAR SC 13D/13G search failed for %s: %s", identifier, exc)

        # --- Path 2: DEF 14A proxy statement (beneficial ownership table) ---
        if len(holders) < 3:
            try:
                self._init_edgartools()
                company = self._get_edgar_company(identifier)
                if company is not None:
                    proxy_filings = company.get_filings(form="DEF 14A")
                    if proxy_filings is not None and len(proxy_filings) > 0:
                        latest_proxy = proxy_filings[0]
                        proxy_text = ""
                        try:
                            proxy_text = latest_proxy.text()[:20000] if hasattr(latest_proxy, "text") else ""
                        except Exception:
                            pass
                        if proxy_text:
                            proxy_holders = self._extract_holders_from_proxy_text(proxy_text)
                            filing_date = str(getattr(latest_proxy, "filing_date", ""))
                            for ph in proxy_holders:
                                ph["date_reported"] = filing_date
                                ph["source"] = "sec_edgar_def14a"
                            # Only add holders not already found via SC 13D/13G
                            existing_names = {h["name"].lower() for h in holders}
                            new_holders = [
                                h for h in proxy_holders
                                if h["name"].lower() not in existing_names
                            ]
                            holders.extend(new_holders)
                            if new_holders:
                                logger.info(
                                    "US holders for %s: +%d from DEF 14A proxy",
                                    identifier, len(new_holders),
                                )
            except Exception as exc:
                logger.debug("DEF 14A holder extraction failed for %s: %s", identifier, exc)


        # --- MarketScreener fallback (global institutional shareholder data) ---
        if not holders:
            try:
                from operator1.clients.marketscreener import fetch_shareholders
                company_name = ""
                if hasattr(self, '_read_cache'):
                    profile = self._read_cache(identifier, "profile.json")
                    if profile:
                        company_name = profile.get("name", "")
                if not company_name:
                    company_name = identifier
                ms_holders = fetch_shareholders(company_name)
                if ms_holders:
                    holders.extend(ms_holders)
                    logger.info(
                        "%s holders for %s: %d from MarketScreener (fallback)",
                        self.market_id, identifier, len(ms_holders),
                    )
            except Exception as exc:
                logger.debug("MarketScreener fallback failed for %s: %s", identifier, exc)

        return holders

    @staticmethod
    def _extract_holders_from_proxy_text(text: str) -> list[dict[str, Any]]:
        """Extract beneficial ownership data from DEF 14A proxy text.

        Searches for the 'Security Ownership' section and parses the
        tabular data using regex patterns common in SEC proxy statements.
        """
        import re
        holders: list[dict[str, Any]] = []

        # Find the beneficial ownership section
        ownership_section = ""
        markers = [
            r"security ownership of certain beneficial owners",
            r"beneficial ownership of common stock",
            r"principal stockholders",
            r"security ownership",
        ]
        for marker in markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                start = match.start()
                # Extract ~3000 chars after the marker
                ownership_section = text[start:start + 3000]
                break

        if not ownership_section:
            return holders

        # Pattern: Name followed by percentage (e.g. "The Vanguard Group... 8.2%")
        # or shares count followed by percentage
        lines = ownership_section.split("\n")
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue
            # Look for lines with a percentage
            pct_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", line)
            if not pct_match:
                continue
            pct = float(pct_match.group(1))
            if pct < 1.0 or pct > 99.0:
                continue
            # Extract name (text before the first number or percentage)
            name_match = re.match(r"^([A-Za-z][\w\s&.,\'-]+?)(?:\s{2,}|\s*\d)", line)
            if name_match:
                name = name_match.group(1).strip()
                if len(name) > 3 and name.lower() not in ("name", "title", "percent", "shares"):
                    # Try to extract shares count
                    shares = 0
                    shares_match = re.search(r"([\d,]+)\s+(?:shares|common)", line, re.IGNORECASE)
                    if shares_match:
                        try:
                            shares = int(shares_match.group(1).replace(",", ""))
                        except ValueError:
                            pass
                    holders.append({
                        "name": name,
                        "shares": shares,
                        "value": 0.0,
                        "percentage": round(pct, 2),
                        "holder_type": "institutional" if pct >= 5 else "insider",
                    })

        return holders

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch insider transactions from SEC EDGAR Form 4 filings.

        Form 4 is filed within 2 business days of an insider transaction.
        Uses edgartools to fetch recent Form 4 filings and extracts
        transaction details from the filing metadata and text.
        """
        transactions: list[dict[str, Any]] = []
        try:
            self._init_edgartools()
            company = self._get_edgar_company(identifier)
            if company is None:
                return transactions

            form4_filings = company.get_filings(form="4")
            if form4_filings is None or len(form4_filings) == 0:
                return transactions

            import re

            for filing in form4_filings[:20]:
                try:
                    insider_name = str(getattr(filing, "company", ""))
                    if not insider_name:
                        insider_name = str(getattr(filing, "filer", ""))
                    filing_date = str(getattr(filing, "filing_date", ""))

                    # Try to extract transaction details from filing text
                    shares = 0
                    value = 0.0
                    txn_type = "unknown"
                    position = ""
                    try:
                        text = filing.text()[:5000] if hasattr(filing, "text") else ""
                        # Look for transaction type
                        if re.search(r"\b(?:purchase|acquired|bought)\b", text, re.IGNORECASE):
                            txn_type = "Purchase"
                        elif re.search(r"\b(?:sale|sold|disposed)\b", text, re.IGNORECASE):
                            txn_type = "Sale"
                        elif re.search(r"\b(?:grant|award|exercise)\b", text, re.IGNORECASE):
                            txn_type = "Grant/Award"
                        # Try to extract shares
                        shares_match = re.search(r"(\d[\d,]+)\s*(?:shares|common)", text, re.IGNORECASE)
                        if shares_match:
                            shares = int(shares_match.group(1).replace(",", ""))
                        # Try to extract position/title
                        pos_match = re.search(
                            r"(?:title|position|officer)\s*[:=]\s*([^\n]+)",
                            text, re.IGNORECASE,
                        )
                        if pos_match:
                            position = pos_match.group(1).strip()[:50]
                    except Exception:
                        pass

                    transactions.append({
                        "insider_name": insider_name,
                        "position": position,
                        "date": filing_date,
                        "transaction": txn_type,
                        "shares": shares,
                        "value": value,
                        "source": "sec_edgar_form4",
                    })
                except Exception:
                    continue

            if transactions:
                logger.info(
                    "US insider transactions for %s: %d from SEC EDGAR Form 4",
                    identifier, len(transactions),
                )
        except Exception as exc:
            logger.debug("SEC EDGAR Form 4 parsing failed for %s: %s", identifier, exc)
        return transactions

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return institutional ownership metrics from SEC EDGAR filings.

        Derives ownership snapshot from SC 13D/13G filings and DEF 14A
        proxy statements.  Returns a single-row DataFrame with aggregate
        metrics computed from the holder data.

        Returns DataFrame with: date_reported, inst_ownership_pct,
        inst_top5_concentration, inst_holder_count.
        """
        try:
            from datetime import date as _date

            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            # Compute aggregate metrics from holder data
            inst_holders = [h for h in holders if h.get("holder_type") == "institutional"]
            inst_pct = sum(h.get("percentage", 0) for h in inst_holders)

            # HHI concentration from top 5
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
            logger.debug("US holder history failed for %s: %s", identifier, exc)

        return pd.DataFrame()

    # -- edgartools financial extraction -------------------------------------

    # Metadata columns returned by edgartools as_dataframe=True (v5.15+).
    # These are NOT period data and must be excluded when iterating columns.
    _EDGARTOOLS_META_COLS = frozenset({
        "label", "depth", "is_abstract", "is_total", "section", "confidence",
    })

    @staticmethod
    def _parse_edgartools_period(col_name: str, fiscal_year_end: str = "") -> str | None:
        """Convert edgartools period column names to ISO date strings.

        edgartools v5.15+ uses names like 'FY 2025', 'Q1 2025', etc.
        Older versions used actual date strings like '2025-09-30'.

        Returns an ISO date string (YYYY-MM-DD) or None if unparseable.
        """
        import re
        col = str(col_name).strip()

        # Already a date string?
        if re.match(r"^\d{4}-\d{2}-\d{2}$", col):
            return col

        # FY YYYY -> fiscal year end date
        fy_match = re.match(r"^FY\s+(\d{4})$", col)
        if fy_match:
            year = int(fy_match.group(1))
            # Use fiscal year end month if available, else default Dec
            if fiscal_year_end and "/" in fiscal_year_end:
                parts = fiscal_year_end.split("/")
                month, day = int(parts[0]), int(parts[1])
            else:
                month, day = 12, 31
            return f"{year}-{month:02d}-{day:02d}"

        # Q1/Q2/Q3/Q4 YYYY -> approximate quarter end
        q_match = re.match(r"^Q(\d)\s+(\d{4})$", col)
        if q_match:
            quarter = int(q_match.group(1))
            year = int(q_match.group(2))
            # Approximate quarter-end dates
            q_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
            month, day = q_ends.get(quarter, (12, 31))
            return f"{year}-{month:02d}-{day:02d}"

        return None

    def _fetch_statement_edgartools(
        self,
        identifier: str,
        statement_type: str,
    ) -> pd.DataFrame | None:
        """Extract financial statement using edgartools Company methods.

        Returns a DataFrame with filing_date, report_date, and canonical
        field names, or None if extraction fails.

        Handles edgartools v5.15+ DataFrame format where:
        - Index contains XBRL concept names (e.g. 'RevenueFromContractWithCustomerExcludingAssessedTax')
        - Metadata columns: label, depth, is_abstract, is_total, section, confidence
        - Period columns: 'FY 2025', 'Q1 2025', etc. (or date strings in older versions)
        """
        self._init_edgartools()
        company = self._get_edgar_company(identifier)

        # Get filings to map report_date -> filing_date (PIT critical)
        filing_date_map = self._build_filing_date_map(company)

        # Get fiscal year end for period date conversion
        fiscal_year_end = str(getattr(company, "fiscal_year_end", "") or "")

        # Build XBRL concept -> label mapping from the DataFrame's 'label' column
        # so we can use human-readable names for concept mapping.
        label_lookup: dict[str, str] = {}

        # Fetch both annual and quarterly for 2-year coverage
        rows: list[dict] = []
        for annual in (True, False):
            periods = 8 if annual else 12  # ~2 years annual, ~3 years quarterly
            try:
                if statement_type == "income":
                    result = company.income_statement(
                        periods=periods, annual=annual, as_dataframe=True,
                    )
                elif statement_type == "balance":
                    result = company.balance_sheet(
                        periods=periods, annual=annual, as_dataframe=True,
                    )
                elif statement_type == "cashflow":
                    # edgartools v5.15+ renamed cash_flow() to cashflow_statement()
                    if hasattr(company, "cashflow_statement"):
                        result = company.cashflow_statement(
                            periods=periods, annual=annual, as_dataframe=True,
                        )
                    else:
                        result = company.cash_flow(
                            periods=periods, annual=annual, as_dataframe=True,
                        )
                else:
                    continue

                if result is None or (isinstance(result, pd.DataFrame) and result.empty):
                    continue

                df = result if isinstance(result, pd.DataFrame) else pd.DataFrame()
                if df.empty:
                    continue

                # Build label lookup from the 'label' column if present
                if "label" in df.columns:
                    for xbrl_concept in df.index:
                        lbl = df.at[xbrl_concept, "label"]
                        if pd.notna(lbl):
                            label_lookup[str(xbrl_concept)] = str(lbl)

                # Identify period columns (exclude metadata columns)
                period_cols = [
                    c for c in df.columns
                    if str(c).lower() not in self._EDGARTOOLS_META_COLS
                    and self._parse_edgartools_period(str(c), fiscal_year_end) is not None
                ]

                if not period_cols:
                    logger.debug(
                        "No period columns found in %s (annual=%s). Columns: %s",
                        statement_type, annual, list(df.columns),
                    )
                    continue

                # Process each period column
                for col in period_cols:
                    period_end = self._parse_edgartools_period(str(col), fiscal_year_end)
                    if not period_end:
                        continue

                    for xbrl_concept, value in df[col].items():
                        if pd.isna(value):
                            continue
                        # Resolve concept label: prefer human-readable label
                        # from the DataFrame's 'label' column, fall back to
                        # the raw XBRL concept name (DataFrame index).
                        concept_label = label_lookup.get(
                            str(xbrl_concept), str(xbrl_concept),
                        )
                        # Try mapping via human-readable label first,
                        # then via raw XBRL concept name.
                        canonical = self._map_edgartools_concept(
                            concept_label, statement_type,
                        )
                        if not canonical:
                            canonical = self._map_xbrl_concept(
                                str(xbrl_concept), statement_type,
                            )
                        if not canonical:
                            # Log unmapped concepts once per label to help
                            # diagnose missing canonical field mappings.
                            if not hasattr(self, "_unmapped_concepts"):
                                self._unmapped_concepts: set[str] = set()
                            _key = f"{statement_type}:{concept_label}"
                            if _key not in self._unmapped_concepts:
                                self._unmapped_concepts.add(_key)
                                logger.debug(
                                    "Unmapped edgartools concept: %s (xbrl=%s, type=%s)",
                                    concept_label, xbrl_concept, statement_type,
                                )
                        if canonical:
                            # Look up filing date from our map
                            filing_dt = filing_date_map.get(period_end, period_end)
                            rows.append({
                                "concept": canonical,
                                "value": float(value) if value is not None else None,
                                "filing_date": filing_dt,
                                "report_date": period_end,
                                "period_type": "annual" if annual else "quarterly",
                                "form": "10-K" if annual else "10-Q",
                            })
            except Exception as exc:
                logger.debug(
                    "edgartools %s (%s, annual=%s) failed: %s",
                    statement_type, identifier, annual, exc,
                )

        if not rows:
            return None

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Filter to 2-year window
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
        if "report_date" in df.columns:
            df = df[df["report_date"] >= cutoff]

        # Translate through canonical_translator
        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def _llm_resolve_unmapped_concepts(
        self,
        unmapped_labels: list[str],
        statement_type: str,
        llm_client: Any = None,
    ) -> dict[str, str]:
        """Use LLM to resolve unmapped financial statement concept labels.

        When edgartools returns concept labels that don't match any static
        mapping (e.g. company-specific line item names), ask the LLM to
        identify which canonical field they map to.

        Parameters
        ----------
        unmapped_labels:
            List of unrecognized concept labels from edgartools.
        statement_type:
            One of "income", "balance", "cashflow".
        llm_client:
            An LLM client instance (from llm_factory). If None, returns
            empty dict (no-op).

        Returns
        -------
        Dict mapping unmapped label -> canonical field name.
        Labels the LLM can't resolve are omitted.
        """
        if not llm_client or not unmapped_labels:
            return {}

        # Canonical field names the LLM can choose from
        from operator1.types import STATEMENT_FIELDS
        canonical_list = ", ".join(STATEMENT_FIELDS)

        prompt = (
            f"I have a {statement_type} financial statement from SEC EDGAR with "
            f"the following unrecognized line item labels:\n\n"
            + "\n".join(f"  - {label}" for label in unmapped_labels[:30])
            + f"\n\nMap each label to one of these canonical field names "
            f"(or SKIP if it does not match any):\n{canonical_list}\n\n"
            f"Return ONLY a JSON object mapping label -> canonical name. "
            f"Example: {{\"Net Sales\": \"revenue\", \"Unknown Item\": \"SKIP\"}}"
        )

        try:
            response = llm_client.generate(prompt)
            if response:
                import json as _json
                # Extract JSON from response (may have markdown wrapping)
                _cleaned = response.strip()
                if "```" in _cleaned:
                    _cleaned = _cleaned.split("```")[1]
                    if _cleaned.startswith("json"):
                        _cleaned = _cleaned[4:]
                    _cleaned = _cleaned.strip()
                mappings = _json.loads(_cleaned)
                result = {}
                for label, canonical in mappings.items():
                    if canonical and canonical != "SKIP" and canonical in STATEMENT_FIELDS:
                        result[label.lower().strip()] = canonical
                if result:
                    logger.info(
                        "LLM resolved %d/%d unmapped %s concepts: %s",
                        len(result), len(unmapped_labels), statement_type,
                        list(result.values()),
                    )
                return result
        except Exception as exc:
            logger.debug("LLM concept resolution failed: %s", exc)

        return {}

    def _build_filing_date_map(self, company) -> dict[str, str]:
        """Build a mapping of report_date -> filing_date from SEC filings.

        This is critical for point-in-time correctness: we need the date
        the filing was actually submitted (not the period end date).
        """
        filing_map: dict[str, str] = {}
        try:
            filings = company.get_filings(form=["10-K", "10-Q", "20-F"])
            if filings:
                for f in filings:
                    # edgartools Filing has .filing_date and .period_of_report
                    filed = str(getattr(f, "filing_date", ""))
                    period_end = str(getattr(f, "period_of_report", ""))
                    if filed and period_end:
                        filing_map[period_end] = filed
        except Exception as exc:
            logger.debug("Could not build filing date map: %s", exc)
        return filing_map

    def _map_edgartools_concept(
        self, label: str, statement_type: str,
    ) -> str | None:
        """Map an edgartools concept label to canonical field name.

        edgartools returns human-readable labels like 'Revenue',
        'Net Income', 'Total Assets', etc. Map these to our canonical names.
        """
        label_lower = label.lower().strip()

        # Income statement
        if statement_type == "income":
            income_map = {
                "revenue": "revenue",
                "revenues": "revenue",
                "net sales": "revenue",
                "total revenue": "revenue",
                "cost of revenue": "cost_of_revenue",
                "cost of goods sold": "cost_of_revenue",
                "cost of sales": "cost_of_revenue",
                "gross profit": "gross_profit",
                "operating income": "operating_income",
                "operating income (loss)": "operating_income",
                "income from operations": "operating_income",
                "net income": "net_income",
                "net income (loss)": "net_income",
                "net income attributable to common stockholders": "net_income",
                "ebit": "ebit",
                "ebitda": "ebitda",
                "income tax expense": "taxes",
                "income tax expense (benefit)": "taxes",
                "provision for income taxes": "taxes",
                "interest expense": "interest_expense",
                "selling, general and administrative": "sga_expense",
                "selling, general & administrative": "sga_expense",
                "sg&a": "sga_expense",
                "research and development": "research_and_development",
                "r&d expense": "research_and_development",
                "earnings per share, basic": "eps_basic",
                "earnings per share basic": "eps_basic",
                "basic eps": "eps_basic",
                "earnings per share, diluted": "eps_diluted",
                "earnings per share diluted": "eps_diluted",
                "diluted eps": "eps_diluted",
            }
            return income_map.get(label_lower)

        # Balance sheet
        if statement_type == "balance":
            balance_map = {
                "total assets": "total_assets",
                "assets": "total_assets",
                "total liabilities": "total_liabilities",
                "liabilities": "total_liabilities",
                "total equity": "total_equity",
                "stockholders' equity": "total_equity",
                "shareholders' equity": "total_equity",
                "total stockholders' equity": "total_equity",
                "equity": "total_equity",
                "current assets": "current_assets",
                "total current assets": "current_assets",
                "assets, current": "current_assets",
                "assetscurrent": "current_assets",
                "current liabilities": "current_liabilities",
                "total current liabilities": "current_liabilities",
                "liabilities, current": "current_liabilities",
                "liabilitiescurrent": "current_liabilities",
                "cash and cash equivalents": "cash_and_equivalents",
                "cash and equivalents": "cash_and_equivalents",
                "cash, cash equivalents": "cash_and_equivalents",
                "cash, cash equivalents and restricted cash": "cash_and_equivalents",
                "cash and cash equivalents, at carrying value": "cash_and_equivalents",
                "cash & cash equivalents": "cash_and_equivalents",
                "cash & equivalents": "cash_and_equivalents",
                "cashandcashequivalentsatcarryingvalue": "cash_and_equivalents",
                "short-term debt": "short_term_debt",
                "short term borrowings": "short_term_debt",
                "current portion of long-term debt": "short_term_debt",
                "long-term debt": "long_term_debt",
                "long term debt": "long_term_debt",
                "total debt": "total_debt",
                "retained earnings": "retained_earnings",
                "retained earnings (accumulated deficit)": "retained_earnings",
                "goodwill": "goodwill",
                "intangible assets": "intangible_assets",
                "accounts receivable": "receivables",
                "accounts receivable, net": "receivables",
                "trade receivables": "receivables",
                "inventories": "inventory",
                "inventory": "inventory",
                "accounts payable": "payables",
            }
            exact = balance_map.get(label_lower)
            if exact:
                return exact
            # Fuzzy fallback: check substrings for common XBRL label variants
            _balance_substring_map = [
                ("current assets", "current_assets"),
                ("current liabilities", "current_liabilities"),
                ("cash and cash equivalents", "cash_and_equivalents"),
                ("cash equivalents", "cash_and_equivalents"),
                ("short-term debt", "short_term_debt"),
                ("short term borrowings", "short_term_debt"),
                ("long-term debt", "long_term_debt"),
                ("retained earnings", "retained_earnings"),
                ("accounts receivable", "receivables"),
                ("trade receivable", "receivables"),
                ("accounts payable", "payables"),
                ("inventories", "inventory"),
                ("goodwill", "goodwill"),
                ("intangible assets", "intangible_assets"),
                ("total assets", "total_assets"),
                ("total liabilities", "total_liabilities"),
                ("stockholders' equity", "total_equity"),
                ("shareholders' equity", "total_equity"),
            ]
            for substr, canonical in _balance_substring_map:
                if substr in label_lower:
                    return canonical
            return None

        # Cash flow
        if statement_type == "cashflow":
            cashflow_map = {
                "operating cash flow": "operating_cash_flow",
                "net cash from operating activities": "operating_cash_flow",
                "cash from operations": "operating_cash_flow",
                "net cash provided by operating activities": "operating_cash_flow",
                "net cash used in operating activities": "operating_cash_flow",
                "capital expenditure": "capex",
                "capital expenditures": "capex",
                "purchases of property and equipment": "capex",
                "payments to acquire property, plant and equipment": "capex",
                "investing cash flow": "investing_cf",
                "net cash from investing activities": "investing_cf",
                "net cash used in investing activities": "investing_cf",
                "net cash provided by investing activities": "investing_cf",
                "financing cash flow": "financing_cf",
                "net cash from financing activities": "financing_cf",
                "net cash used in financing activities": "financing_cf",
                "net cash provided by financing activities": "financing_cf",
                "dividends paid": "dividends_paid",
                "payments of dividends": "dividends_paid",
                "free cash flow": "free_cash_flow",
                "repurchase of common stock": "buybacks",
                "share repurchases": "buybacks",
            }
            return cashflow_map.get(label_lower)

        return None

    @staticmethod
    def _map_xbrl_concept(xbrl_name: str, statement_type: str) -> str | None:
        """Map raw XBRL US-GAAP concept names to canonical field names.

        edgartools v5.15+ uses XBRL concept names as DataFrame index,
        e.g. 'RevenueFromContractWithCustomerExcludingAssessedTax'.
        """
        xbrl_lower = xbrl_name.lower()

        if statement_type == "income":
            xbrl_income = {
                "revenuefromcontractwithcustomerexcludingassessedtax": "revenue",
                "revenues": "revenue",
                "salesrevenuenet": "revenue",
                "revenuesfromexternalcustomersandpremiums": "revenue",
                "costofgoodsandservicessold": "cost_of_revenue",
                "costofrevenue": "cost_of_revenue",
                "costofgoodssold": "cost_of_revenue",
                "grossprofit": "gross_profit",
                "operatingincomeloss": "operating_income",
                "netincomeloss": "net_income",
                "netincomelossdiluted": "net_income",
                "netincomelossavailabletocommonstockholdersbasic": "net_income",
                "incometaxexpensebenefit": "taxes",
                "interestexpense": "interest_expense",
                "sellinggeneralandadministrativeexpense": "sga_expense",
                "researchanddevelopmentexpense": "research_and_development",
                "earningspersharebasic": "eps_basic",
                "earningspersharediluted": "eps_diluted",
                "sellingandmarketingexpense": "sga_expense",
                "generalandadministrativeexpense": "sga_expense",
                "operatingexpenses": "operating_expenses",
                "depreciationandamortization": "depreciation_amortization",
            }
            return xbrl_income.get(xbrl_lower)

        if statement_type == "balance":
            xbrl_balance = {
                "assets": "total_assets",
                "liabilities": "total_liabilities",
                "stockholdersequity": "total_equity",
                "liabilitiesandstockholdersequity": "total_liabilities_and_equity",
                "assetscurrent": "current_assets",
                "liabilitiescurrent": "current_liabilities",
                "cashandcashequivalentsatcarryingvalue": "cash_and_equivalents",
                "cashcashequivalentsandshortterminvestments": "cash_and_equivalents",
                "longtermdebt": "long_term_debt",
                "longtermdebtnoncurrent": "long_term_debt",
                "shorttermborrowing": "short_term_debt",
                "commercialpaper": "short_term_debt",
                "retainedearningsaccumulateddeficit": "retained_earnings",
                "goodwill": "goodwill",
                "accountsreceivablenetcurrent": "receivables",
                "inventorynet": "inventory",
                "accountspayablecurrent": "payables",
                "commonstocksharesoutstanding": "shares_outstanding",
                "propertyplantandequipmentnet": "ppe_net",
            }
            return xbrl_balance.get(xbrl_lower)

        if statement_type == "cashflow":
            xbrl_cashflow = {
                "netcashprovidedbyusedinoperatingactivities": "operating_cash_flow",
                "netcashprovidedbyusedinoperatingactivitiescontinuingoperations": "operating_cash_flow",
                "netcashprovidedbyusedininvestingactivities": "investing_cf",
                "netcashprovidedbyusedininvestingactivitiescontinuingoperations": "investing_cf",
                "netcashprovidedbyusedinfinancingactivities": "financing_cf",
                "netcashprovidedbyusedinfinancingactivitiescontinuingoperations": "financing_cf",
                "paymentstoacquirepropertyplantandequipment": "capex",
                "depreciationdepletionandamortization": "depreciation_amortization",
                "paymentsofdividends": "dividends_paid",
                "paymentsofdividendscommonstock": "dividends_paid",
                "paymentsforrepurchaseofcommonstock": "buybacks",
                "stockbasedcompensation": "stock_based_comp",
            }
            return xbrl_cashflow.get(xbrl_lower)

        return None

    # -- sec-edgar-api fallback extraction ------------------------------------

    def _fetch_statement_fallback(
        self,
        identifier: str,
        statement_type: str,
    ) -> pd.DataFrame:
        """Extract financial statement from raw XBRL companyfacts (fallback).

        Uses sec-edgar-api to fetch the companyfacts JSON and extracts
        US-GAAP concepts directly, similar to the original sec_edgar.py.
        """
        try:
            client = self._get_sec_api_client()
            cik = self._resolve_cik_fallback(identifier)
            facts = client.get_company_facts(cik)
        except Exception as exc:
            logger.error("Fallback companyfacts failed for %s: %s", identifier, exc)
            # Third fallback: direct requests to companyfacts endpoint
            try:
                facts = self._fetch_companyfacts_direct(identifier)
            except Exception as exc2:
                logger.error("Direct companyfacts also failed for %s: %s", identifier, exc2)
                return pd.DataFrame()

        concept_map = {
            "income": _USGAAP_INCOME_CONCEPTS,
            "balance": _USGAAP_BALANCE_CONCEPTS,
            "cashflow": _USGAAP_CASHFLOW_CONCEPTS,
        }

        concepts = concept_map.get(statement_type, {})
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        rows: list[dict] = []
        for concept_name, canonical_name in concepts.items():
            concept_data = us_gaap.get(concept_name, {})
            units = concept_data.get("units", {})

            for unit_key in ("USD", "USD/shares", "shares"):
                entries = units.get(unit_key, [])
                for entry in entries:
                    form = entry.get("form", "")
                    if form not in ("10-K", "10-Q", "20-F"):
                        continue

                    rows.append({
                        "concept": canonical_name,
                        "value": entry.get("val"),
                        "filing_date": entry.get("filed", ""),
                        "report_date": entry.get("end", ""),
                        "period_start": entry.get("start", ""),
                        "form": form,
                        "fiscal_year": entry.get("fy"),
                        "fiscal_period": entry.get("fp", ""),
                        "unit": unit_key,
                        "period_type": "annual" if form == "10-K" else "quarterly",
                    })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date", "period_start"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Filter to 2-year window
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
        if "report_date" in df.columns:
            df = df[df["report_date"] >= cutoff]

        # Cache each filing period to disk
        self._cache_filings(identifier, df)

        # Translate to canonical format
        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def _resolve_cik_fallback(self, identifier: str) -> str:
        """Resolve a ticker symbol to a CIK using the company list."""
        if identifier.isdigit():
            return identifier.zfill(10)

        companies = self.list_companies()
        for c in companies:
            if c["ticker"].upper() == identifier.upper():
                return str(c["cik"]).zfill(10)

        raise USEdgarError(
            "resolve_cik",
            f"Could not resolve '{identifier}' to a CIK number",
        )

    def _cache_filings(self, identifier: str, df: pd.DataFrame) -> None:
        """Cache financial statement data as per-period JSON files."""
        if df.empty or "report_date" not in df.columns:
            return

        for period_end, group in df.groupby("report_date"):
            period_str = pd.Timestamp(period_end).strftime("%Y-%m-%d")
            filename = f"filings/{period_str}.json"
            records = group.to_dict(orient="records")
            # Convert timestamps to strings for JSON
            for r in records:
                for k, v in r.items():
                    if isinstance(v, pd.Timestamp):
                        r[k] = v.isoformat()
            self._write_cache(identifier, filename, {"period_end": period_str, "rows": records})

    def _fetch_companyfacts_direct(self, identifier: str) -> dict:
        """Fetch companyfacts JSON using direct requests (no library needed).

        This is the last-resort fallback when neither edgartools nor
        sec-editor-api is available. Uses the SEC EDGAR XBRL API directly.

        Endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
        """
        import requests

        headers = {"User-Agent": self._user_agent, "Accept": "application/json"}

        # Resolve ticker to CIK using the tickers JSON
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        tickers_data = resp.json()

        cik = ""
        ticker_upper = identifier.upper()
        for entry in tickers_data.values():
            if str(entry.get("ticker", "")).upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                break
            if str(entry.get("cik_str", "")) == identifier:
                cik = str(entry["cik_str"]).zfill(10)
                break

        if not cik:
            raise USEdgarError("companyfacts", f"CIK not found for: {identifier}")

        # Fetch companyfacts
        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(facts_url, headers=headers, timeout=30)
        resp.raise_for_status()

        logger.info("Direct companyfacts fetched for %s (CIK %s)", identifier, cik)
        return resp.json()

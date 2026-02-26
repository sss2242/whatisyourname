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
        # Prefer EDGAR_IDENTITY env var (email), fall back to provided arg or default
        self._user_agent = (
            user_agent
            or os.environ.get("EDGAR_IDENTITY", "")
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
        """Query SEC browse-edgar for companies matching a SIC code.

        Returns a list of resolved ticker symbols (up to 10).
        """
        import requests

        exclude = exclude or set()
        try:
            resp = requests.get(
                "https://www.sec.gov/cgi-bin/browse-edgar",
                params={
                    "action": "getcompany",
                    "SIC": sic,
                    "owner": "include",
                    "count": "100",
                    "output": "atom",
                },
                headers={"User-Agent": self._user_agent},
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("SEC browse-edgar SIC query failed: %s", exc)
            return []

        # Parse the Atom XML to extract CIKs
        peer_ciks: list[str] = []
        try:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(resp.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", ns):
                for el in entry.iter():
                    tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
                    if tag == "cik" and el.text:
                        peer_ciks.append(el.text.lstrip("0"))
                        break
        except Exception as exc:
            logger.warning("Failed to parse SEC browse-edgar XML: %s", exc)
            return []

        # Map CIKs to tickers, excluding the target and already-found peers
        peers: list[str] = []
        for cik in peer_ciks:
            ticker = cik_to_ticker.get(cik, "")
            if ticker and ticker != target_ticker and ticker not in peers and ticker not in exclude:
                peers.append(ticker)
            if len(peers) >= 10:
                break

        return peers

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        """SEC doesn't have a direct executives endpoint; returns empty."""
        return []

    # -- edgartools financial extraction -------------------------------------

    def _fetch_statement_edgartools(
        self,
        identifier: str,
        statement_type: str,
    ) -> pd.DataFrame | None:
        """Extract financial statement using edgartools Company methods.

        Returns a DataFrame with filing_date, report_date, and canonical
        field names, or None if extraction fails.
        """
        self._init_edgartools()
        company = self._get_edgar_company(identifier)

        # Get filings to map report_date -> filing_date (PIT critical)
        filing_date_map = self._build_filing_date_map(company)

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
                    # Docs: https://github.com/dgunning/edgartools CHANGELOG v5.15.0
                    # Research: .roo/research/us-sec-edgar-2026-02-24.md Section 4
                    if hasattr(company, "cashflow_statement"):
                        result = company.cashflow_statement(
                            periods=periods, annual=annual, as_dataframe=True,
                        )
                    else:
                        # Fallback for older edgartools versions (<5.15)
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

                # Process the DataFrame: columns are period end dates,
                # rows are concept names
                for col in df.columns:
                    period_end = str(col)
                    for concept_label, value in df[col].items():
                        if pd.isna(value):
                            continue
                        canonical = self._map_edgartools_concept(
                            str(concept_label), statement_type,
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
                "current liabilities": "current_liabilities",
                "total current liabilities": "current_liabilities",
                "cash and cash equivalents": "cash_and_equivalents",
                "cash and equivalents": "cash_and_equivalents",
                "cash, cash equivalents": "cash_and_equivalents",
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
            return balance_map.get(label_lower)

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

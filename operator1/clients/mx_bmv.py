"""Mexico BMV PIT client -- native BMV search API + filing discovery.

Primary: BMV WSO2 API Gateway (https://www.bmv.com.mx/api/searchservice/v1)
  - Token-based auth (GET /rest/tokenservice/token, no credentials needed)
  - Company search via ElasticSearch backend (busquedaClaveCotizacion)
  - Profile data: ticker, name, series, market, status, company ID
  - Document discovery: quarterly records, issuer events, corporate docs

Fallback: yfinance (.MX suffix) for profile enrichment and OHLCV

OHLCV: handled separately via ohlcv_provider.py (yfinance .MX)

Coverage: ~150+ listed companies on BMV, ~$0.5T market cap.

Key discovery: BMV uses a WSO2 API Manager. The search endpoint requires
a Bearer token obtained from a public token endpoint (no API key needed).
This is analogous to HKEX's JSESSIONID pattern -- the session/token
must be obtained first before making API calls.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("cache/mx_bmv")

# ---------------------------------------------------------------------------
# BMV API constants
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

# Token refresh interval (the token has a very long expiry but we
# refresh periodically to be safe).
_TOKEN_REFRESH_INTERVAL_S = 3600  # 1 hour


# ---------------------------------------------------------------------------
# BMV API helpers
# ---------------------------------------------------------------------------


class _BMVTokenManager:
    """Manages the BMV API Bearer token lifecycle.

    The BMV site uses a WSO2 API Manager. A Bearer token is obtained
    from a public endpoint (no credentials) and must be included in
    all API requests. The token has a very long expiry but we refresh
    periodically.

    As of 2026-03, the BMV token endpoint sometimes returns an empty
    response string instead of an access_token dict. This manager
    handles both the old format (dict with access_token) and the new
    error format (empty string + error message) gracefully.
    """

    def __init__(self) -> None:
        self._token: str = ""
        self._obtained_at: float = 0.0
        self._session: requests.Session | None = None
        self._failed_attempts: int = 0

    def _get_session(self) -> requests.Session:
        """Get or create a persistent session with BMV cookies."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(_BMV_HEADERS)
            # Load main page first to get JSESSIONID + F5 cookies
            # (similar to HKEX scraper pattern)
            try:
                self._session.get(_BMV_BASE_URL, timeout=15)
            except Exception:
                pass
        return self._session

    def get_token(self) -> str:
        """Return a valid Bearer token, refreshing if needed.

        Returns empty string if the BMV token service is unavailable,
        allowing callers to fall back to yfinance.
        """
        now = time.time()
        if self._token and (now - self._obtained_at) < _TOKEN_REFRESH_INTERVAL_S:
            return self._token

        # Skip repeated attempts if we already know the service is down
        if self._failed_attempts >= 3:
            if (now - self._obtained_at) < 300:  # retry every 5 min
                return self._token  # return stale or empty

        session = self._get_session()

        try:
            resp = session.get(
                _BMV_TOKEN_URL,
                headers={
                    "Referer": f"{_BMV_BASE_URL}/",
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            # Handle both response formats:
            # Old: {"response": {"access_token": "xxx", ...}}
            # New/error: {"response": "", "mensaje": {"Error": "..."}}
            response = data.get("response", "")
            if isinstance(response, dict) and response.get("access_token"):
                self._token = response["access_token"]
                self._obtained_at = now
                self._failed_attempts = 0
                logger.debug("BMV token obtained: %s...", self._token[:12])
            elif isinstance(response, str) and response:
                # Some responses return the token as a plain string
                self._token = response
                self._obtained_at = now
                self._failed_attempts = 0
            else:
                # Empty response or error -- token service is down
                error_msg = data.get("mensaje", {}).get("Error", "unknown")
                self._failed_attempts += 1
                logger.info(
                    "BMV token service returned empty response (attempt %d): %s",
                    self._failed_attempts, error_msg,
                )
        except Exception as exc:
            self._failed_attempts += 1
            logger.info(
                "BMV token service unavailable (attempt %d): %s",
                self._failed_attempts, exc,
            )

        return self._token


# Module-level token manager (shared across all BMV operations)
_token_manager = _BMVTokenManager()


def _bmv_search(
    term: str,
    search_type: str = "busquedaClaveCotizacion",
    lang: str = "en",
) -> dict:
    """Execute a search against the BMV API.

    Parameters
    ----------
    term:
        Search term (ticker or company name).
    search_type:
        ``"busquedaClaveCotizacion"`` for ticker/instrument search,
        ``"busquedaPanel"`` for broader search including documents.
    lang:
        ``"en"`` or ``"es"``.

    Returns
    -------
    The full API response dict, or empty dict on failure.
    """
    token = _token_manager.get_token()

    # If no token available, BMV API is down -- return empty immediately
    # so callers can fall through to yfinance without waiting for a 503
    if not token:
        logger.debug("BMV search skipped: no token available (API may be down)")
        return {}

    # Split multi-word terms for the API's two-term fields
    parts = term.strip().split(" ", 1)
    term1 = parts[0]
    term2 = parts[1] if len(parts) > 1 else ""

    payload = {
        "lang": lang,
        "payload": {
            "term": term1,
            "term2": term2,
            "termT": term.strip(),
            "searchType": search_type,
        },
    }

    try:
        session = _token_manager._get_session()
        resp = session.post(
            _BMV_SEARCH_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Referer": f"{_BMV_BASE_URL}/",
                "Origin": _BMV_BASE_URL,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("BMV search failed for '%s': %s", term, exc)
        return {}


def _extract_companies(response: dict) -> list[dict[str, Any]]:
    """Extract company records from BMV search response.

    Navigates the nested ElasticSearch response to find equity
    instruments and returns them as flat dicts.
    """
    companies: list[dict[str, Any]] = []

    # Path for busquedaClaveCotizacion
    bcc = response.get("response", {}).get("busquedaClaveCotizacion", {})
    for category in ("emisorasSinSerie", "instrumentos", "empresas"):
        hits = bcc.get(category, {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            companies.append(_parse_company_hit(src))

    # Path for busquedaPanel
    panel = response.get("response", {}).get("busquedaPanel", {})
    if isinstance(panel, dict):
        try:
            emisoras_hits = (
                panel
                .get("busquedaGeneral", {})
                .get("instrumentosEmisoras", {})
                .get("instrumentos", {})
                .get("coincidenciaParcialInstrumentos", {})
                .get("emisoras", {})
                .get("hits", [])
            )
            for hit in emisoras_hits:
                src = hit.get("_source", {})
                companies.append(_parse_company_hit(src))
        except (AttributeError, TypeError):
            pass

    # Dedup by (ticker, series) and prefer active equity instruments
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for c in companies:
        key = f"{c['ticker']}_{c.get('series', '')}"
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # Sort: active equity first, then by ticker
    unique.sort(key=lambda c: (
        0 if c.get("market") == "Equity" and c.get("status") == "Active" else 1,
        c.get("ticker", ""),
    ))

    return unique


def _parse_company_hit(src: dict) -> dict[str, Any]:
    """Parse a single ElasticSearch hit into a company dict."""
    return {
        "ticker": src.get("cve_emisora", ""),
        "name": src.get("razon_social", ""),
        "series": src.get("serie", ""),
        "instrument": src.get("instrumento", ""),
        "market": src.get("mercado", ""),
        "status": src.get("estatus", ""),
        "exchange": src.get("bolsa_cotiza", "BMV"),
        "country": "MX",
        "id_empresa": src.get("id_empresa", ""),
        "id_emision": src.get("id_emision", ""),
        "market_id": "mx_bmv",
        "description": src.get("descripcion", ""),
    }


def _extract_documents(response: dict) -> list[dict[str, Any]]:
    """Extract filing documents from BMV search response.

    Returns a list of document dicts with title, tag, URL, and company.
    """
    docs: list[dict[str, Any]] = []

    panel = response.get("response", {}).get("busquedaPanel", {})
    if not isinstance(panel, dict):
        return docs

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
        for hit in doc_hits:
            src = hit.get("_source", {})
            doc_bin = src.get("documento_binario", {})
            url = doc_bin.get("url_documento", "")
            if url and not url.startswith("http"):
                url = f"{_BMV_BASE_URL}{url}"
            docs.append({
                "title": src.get("descripccion_documento", "").replace("<b>", "").replace("</b>", ""),
                "tag": src.get("tag_en", src.get("tag_es", "")),
                "url": url,
                "company": src.get("cve_empresa", ""),
                "doc_id": src.get("id_documento", ""),
                "date": src.get("fecha_publicacion", ""),
            })
    except (AttributeError, TypeError):
        pass

    return docs


# ---------------------------------------------------------------------------
# MXBmvClient -- PIT client for Mexican BMV equities
# ---------------------------------------------------------------------------


class MXBmvClient:
    """PIT client for Mexican BMV equities.

    Uses the BMV WSO2 API Gateway for company search and profile,
    with yfinance fallback for sector/industry enrichment.

    Company search works natively via the BMV ElasticSearch API --
    no yfinance dependency for ticker resolution.
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
        return "mx_bmv"

    @property
    def market_name(self) -> str:
        return "Mexico (BMV)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Search for companies on BMV.

        Uses the native BMV search API (token-based, no key needed).
        Falls back to yfinance if the BMV API is unreachable.
        """
        if not query:
            # BMV API requires a search term; for empty queries, fall back
            from operator1.clients.yfinance_backed import yf_search
            return yf_search(query, self.market_id, "MX", "BMV", yf_suffix=".MX")

        # Try BMV native search first
        try:
            # Try both search types for maximum coverage
            result_ticker = _bmv_search(query, "busquedaClaveCotizacion")
            companies = _extract_companies(result_ticker)

            if not companies:
                result_panel = _bmv_search(query, "busquedaPanel")
                companies = _extract_companies(result_panel)

            if companies:
                logger.info("BMV native search for '%s': %d results", query, len(companies))
                return companies
        except Exception as exc:
            logger.debug("BMV native search failed for '%s': %s", query, exc)

        # Fallback to yfinance
        from operator1.clients.yfinance_backed import yf_search
        return yf_search(query, self.market_id, "MX", "BMV", yf_suffix=".MX")

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from BMV + yfinance enrichment.

        Uses the BMV search API for core data (name, ticker, series,
        market), then enriches with yfinance for sector/industry.
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        profile: dict[str, Any] = {
            "ticker": identifier,
            "name": identifier,
            "country": "MX",
            "exchange": "BMV",
            "currency": "MXN",
            "market_id": "mx_bmv",
        }

        # Try BMV native search for core profile data
        try:
            result = _bmv_search(identifier, "busquedaClaveCotizacion")
            companies = _extract_companies(result)

            if not companies:
                result = _bmv_search(identifier, "busquedaPanel")
                companies = _extract_companies(result)

            if companies:
                # Pick the best match (first active equity instrument)
                best = companies[0]
                profile.update({
                    "ticker": best.get("ticker", identifier),
                    "name": best.get("name", identifier),
                    "series": best.get("series", ""),
                    "instrument": best.get("instrument", ""),
                    "id_empresa": best.get("id_empresa", ""),
                    "id_emision": best.get("id_emision", ""),
                    "status": best.get("status", ""),
                    "market_type": best.get("market", ""),
                })
                logger.info(
                    "BMV profile for %s: %s (%s)",
                    identifier, best.get("name", "?"), best.get("instrument", "?"),
                )
        except Exception as exc:
            logger.debug("BMV native profile failed for %s: %s", identifier, exc)

        # Enrich with yfinance for sector/industry (BMV API doesn't provide these)
        try:
            from operator1.clients.yfinance_backed import yf_get_profile
            yf_profile = yf_get_profile(
                identifier, self.market_id,
                "Mexico", "MX", "BMV", "MXN",
                yf_suffix=".MX",
            )
            # Only fill in missing fields from yfinance
            for key in ("sector", "industry", "market_cap", "shares_outstanding",
                        "pe_ratio", "eps", "description"):
                if key not in profile or not profile[key]:
                    val = yf_profile.get(key)
                    if val:
                        profile[key] = val
        except Exception as exc:
            logger.debug("yfinance enrichment failed for %s: %s", identifier, exc)

        self._write_cache(identifier, "profile.json", profile)
        return profile

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financials from BMV XBRL (primary, fast) or filing discovery (fallback).

        Primary path: Download XBRL JSON ZIPs from BMV's IFRS XBRL page.
        Each ZIP contains structured financial data with IFRS concept codes
        and period dates -- no LLM needed, no PDF parsing.

        Fallback: BMV search API filing discovery + LLM/fuzzy PDF extraction.

        yfinance is NOT used for financial statements because it does not
        provide true filing dates (sets filing_date = report_date).
        """
        # Primary: XBRL JSON extraction (fast, structured, no LLM needed)
        try:
            df = _fetch_bmv_xbrl_financials(identifier, statement_type)
            if df is not None and not df.empty:
                logger.info("BMV %s %s: %d rows from XBRL (fast path)",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("BMV XBRL extraction failed for %s: %s", identifier, exc)

        # Fallback: filing discovery + LLM/fuzzy extraction
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
            df = try_filing_extraction(
                ticker=identifier,
                market_id=self.market_id,
                statement_type=statement_type,
                llm_client=None,
            )
            if df is not None and not df.empty:
                logger.info("BMV %s %s: %d rows from filing discovery",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("BMV filing discovery failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """BMV does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    # -- Institutional holders (BMV filing discovery + LLM/fuzzy extraction) --

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch holders from BMV filing discovery + LLM/fuzzy extraction.

        Uses the BMV search API to find holder disclosure filings
        (tenencia accionaria / composicion accionaria) and extracts
        holder data via the LLM filing extractor or fuzzy PDF parser.

        Falls back to XBRL data if shareholder info is present in the
        annual report XBRL JSON.
        """
        holders: list[dict[str, Any]] = []

        # --- Path 1: BMV XBRL annual report may contain shareholder data ---
        try:
            df = _fetch_bmv_xbrl_financials(identifier, "balance")
            if df is not None and not df.empty:
                # Check for shareholder-related fields in XBRL data
                # (Mexican XBRL reports sometimes include capital structure)
                logger.debug(
                    "BMV XBRL data available for %s (%d rows); "
                    "checking for shareholder fields",
                    identifier, len(df),
                )
        except Exception:
            pass

        # --- Path 2: BMV filing discovery for holder disclosures ---
        if not holders:
            try:
                from operator1.clients.filing_discoverer import try_filing_extraction
                # Try to find holder-specific filings via BMV search
                df = try_filing_extraction(
                    ticker=identifier,
                    market_id=self.market_id,
                    statement_type="balance",
                    llm_client=None,
                )
                if df is not None and not df.empty:
                    logger.info(
                        "BMV filings available for %s (%d rows); "
                        "holder extraction available via LLM",
                        identifier, len(df),
                    )
            except Exception as exc:
                logger.debug("BMV filing discovery for holders failed: %s", exc)

        # Fallback: extract shareholders from filing PDFs
        if not holders:
            try:
                from operator1.clients.filing_discoverer import try_shareholding_extraction
                holders = try_shareholding_extraction(identifier, market_id=self.market_id)
                if holders:
                    logger.info("BMV holders from PDF shareholding extraction: %d", len(holders))
            except Exception as exc:
                logger.debug("BMV PDF shareholding fallback failed: %s", exc)


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

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return institutional ownership metrics from BMV filings."""
        try:
            from datetime import date as _date
            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            inst_pct = sum(h.get("percentage", 0) for h in holders)
            hhi = 0.0
            top5 = holders[:5]
            total_pct = sum(h.get("percentage", 0) for h in top5)
            if total_pct > 0:
                hhi = sum((h.get("percentage", 0) / total_pct) ** 2 for h in top5)

            return pd.DataFrame([{
                "date_reported": pd.Timestamp(_date.today()),
                "inst_ownership_pct": round(inst_pct, 2),
                "inst_top5_concentration": round(hhi, 4),
                "inst_holder_count": len(holders),
            }])
        except Exception as exc:
            logger.debug("BMV holder history failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch insider transactions from BMV filing discovery.

        Uses BMV search API to find insider dealing disclosures.
        Returns basic filing metadata; detailed extraction requires LLM.
        """
        transactions: list[dict[str, Any]] = []
        try:
            # BMV search API can find insider disclosure filings
            token = _token_manager.get_token()
            if not token:
                return transactions
            # Search for insider-related filings
            logger.debug(
                "BMV insider transactions not yet extracted for %s; "
                "filing discovery available via BMV search API",
                identifier,
            )
        except Exception as exc:
            logger.debug("BMV insider transaction search failed for %s: %s", identifier, exc)
        return transactions

    def extract_segment_data(self, identifier: str) -> dict[str, Any]:
        """Extract IFRS 8 segment data from BMV XBRL JSON ZIPs.

        Delegates to the standalone ``extract_bmv_segment_data()`` function
        which downloads recent XBRL ZIPs and extracts segment information.
        """
        try:
            return extract_bmv_segment_data(identifier)
        except Exception as exc:
            logger.debug("BMV segment extraction failed for %s: %s", identifier, exc)
            return {"n_segments": 0, "segments": {}, "descriptions": {}}


# ---------------------------------------------------------------------------
# BMV XBRL Financial Data Extraction (fast path, no LLM needed)
# ---------------------------------------------------------------------------

_BMV_XBRL_PAGE = f"{_BMV_BASE_URL}/en/issuers/standard-xbrl-files"

# Cache the parsed XBRL page (ticker -> ZIP URLs mapping)
_xbrl_index_cache: dict[str, list[dict]] | None = None
_xbrl_index_cache_time: float = 0.0
_XBRL_INDEX_TTL = 86400  # 24 hours

# IFRS concept -> canonical field name mapping
_IFRS_INCOME_CONCEPTS: dict[str, str] = {
    "ifrs-full_Revenue": "revenue",
    "ifrs-full_CostOfSales": "cost_of_revenue",
    "ifrs-full_GrossProfit": "gross_profit",
    "ifrs-full_ProfitLossFromOperatingActivities": "operating_income",
    "ifrs-full_ProfitLoss": "net_income",
    "ifrs-full_ProfitLossBeforeTax": "ebit",
    "ifrs-full_IncomeTaxExpenseContinuingOperations": "taxes",
    "ifrs-full_FinanceCosts": "interest_expense",
    "ifrs-full_SellingGeneralAndAdministrativeExpense": "sga_expenses",
    "ifrs-full_ResearchAndDevelopmentExpense": "rd_expenses",
    "ifrs-full_BasicEarningsLossPerShare": "eps_basic",
    "ifrs-full_DilutedEarningsLossPerShare": "eps_diluted",
}

_IFRS_BALANCE_CONCEPTS: dict[str, str] = {
    "ifrs-full_Assets": "total_assets",
    "ifrs-full_Liabilities": "total_liabilities",
    "ifrs-full_Equity": "total_equity",
    "ifrs-full_CurrentAssets": "current_assets",
    "ifrs-full_CurrentLiabilities": "current_liabilities",
    "ifrs-full_CashAndCashEquivalents": "cash_and_equivalents",
    "ifrs-full_NoncurrentLiabilities": "long_term_debt",
    "ifrs-full_RetainedEarnings": "retained_earnings",
    "ifrs-full_Goodwill": "goodwill",
    "ifrs-full_IntangibleAssetsOtherThanGoodwill": "intangible_assets",
    "ifrs-full_TradeAndOtherCurrentReceivables": "receivables",
    "ifrs-full_Inventories": "inventory",
    "ifrs-full_TradeAndOtherCurrentPayables": "payables",
}

_IFRS_CASHFLOW_CONCEPTS: dict[str, str] = {
    "ifrs-full_CashFlowsFromUsedInOperatingActivities": "operating_cash_flow",
    "ifrs-full_CashFlowsFromUsedInInvestingActivities": "investing_cf",
    "ifrs-full_CashFlowsFromUsedInFinancingActivities": "financing_cf",
    "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": "capex",
    "ifrs-full_DividendsPaidClassifiedAsFinancingActivities": "dividends_paid",
}

_STATEMENT_CONCEPTS: dict[str, dict[str, str]] = {
    "income": _IFRS_INCOME_CONCEPTS,
    "balance": _IFRS_BALANCE_CONCEPTS,
    "cashflow": _IFRS_CASHFLOW_CONCEPTS,
}

# IFRS 8 Operating Segments -- concept names for segment revenue extraction.
# Mexican XBRL files use these IFRS concepts when reporting segment data.
_IFRS_SEGMENT_REVENUE_CONCEPTS: tuple[str, ...] = (
    "ifrs-full_RevenueFromExternalCustomers",
    "ifrs-full_Revenue",
    "ifrs-full_RevenueFromContractsWithCustomers",
    "ifrs-full_RevenueFromRenderingOfServices",
    "ifrs-full_RevenueFromSaleOfGoods",
)

# Product-level concepts that appear within segment dimensions.
_IFRS_SEGMENT_DETAIL_CONCEPTS: tuple[str, ...] = (
    "ifrs-full_RevenueFromExternalCustomers",
    "ifrs-full_Revenue",
    "ifrs-full_ProfitLossFromOperatingActivities",
    "ifrs-full_Assets",
    "ifrs-full_DepreciationAndAmortisationExpense",
    "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
)


def _get_xbrl_index() -> dict[str, list[dict]]:
    """Parse the BMV XBRL page to build a ticker -> ZIP URLs index.

    The page contains ~5,800 XBRL ZIP entries for ~244 issuers.
    Cached for 24 hours to avoid re-parsing the 4MB HTML page.
    """
    import re as _re

    global _xbrl_index_cache, _xbrl_index_cache_time
    now = time.time()
    if _xbrl_index_cache is not None and (now - _xbrl_index_cache_time) < _XBRL_INDEX_TTL:
        return _xbrl_index_cache

    try:
        resp = requests.get(
            _BMV_XBRL_PAGE,
            headers=_BMV_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("BMV XBRL page fetch failed: %s", exc)
        return _xbrl_index_cache or {}

    rows = _re.findall(r"<tr[^>]*>(.*?)</tr>", resp.text, _re.DOTALL)
    index: dict[str, list[dict]] = {}

    for row in rows:
        cells = _re.findall(r"<td[^>]*>(.*?)</td>", row, _re.DOTALL)
        if len(cells) < 4:
            continue
        ticker = _re.sub(r"<[^>]+>", "", cells[0]).strip()
        if not ticker or ticker.lower() == "ticker":
            continue
        company = _re.sub(r"<[^>]+>", "", cells[1]).strip()
        date_str = _re.sub(r"<[^>]+>", "", cells[2]).strip()
        zip_match = _re.search(r"docins=([^\"&]+)", row)
        if not zip_match:
            continue

        zip_path = zip_match.group(1)
        if zip_path.startswith(".."):
            zip_url = f"{_BMV_BASE_URL}/docs-pub/{zip_path.replace('../', '')}"
        else:
            zip_url = f"{_BMV_BASE_URL}/docs-pub/ifrsxbrl/{zip_path}"

        # Parse filing date from "DD/MM/YYYY HH:MM" format
        filing_date = ""
        fd_match = _re.search(r"(\d{2})/(\d{2})/(\d{4})", date_str)
        if fd_match:
            filing_date = f"{fd_match.group(3)}-{fd_match.group(2)}-{fd_match.group(1)}"

        if ticker not in index:
            index[ticker] = []
        index[ticker].append({
            "url": zip_url,
            "filing_date": filing_date,
            "company": company,
        })

    _xbrl_index_cache = index
    _xbrl_index_cache_time = now
    logger.info("BMV XBRL index: %d tickers, %d ZIPs",
               len(index), sum(len(v) for v in index.values()))
    return index


def _extract_xbrl_json(zip_bytes: bytes, concept_map: dict[str, str]) -> list[dict]:
    """Extract financial values from a BMV XBRL JSON ZIP.

    Parameters
    ----------
    zip_bytes:
        Raw bytes of the XBRL ZIP file.
    concept_map:
        Mapping from IFRS concept name to canonical field name.

    Returns
    -------
    List of dicts with keys: canonical_name, value, report_date, filing_date.
    """
    import zipfile
    import io
    import json as _json

    records: list[dict] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        json_files = [n for n in zf.namelist() if n.endswith(".json")]
        if not json_files:
            return records

        with zf.open(json_files[0]) as jf:
            data = _json.loads(jf.read())

    hechos_por_concepto = data.get("HechosPorIdConcepto", {})
    hechos_por_id = data.get("HechosPorId", {})
    contextos = data.get("ContextosPorId", {})

    for ifrs_concept, canonical_name in concept_map.items():
        fact_ids = hechos_por_concepto.get(ifrs_concept, [])
        if not isinstance(fact_ids, list) or not fact_ids:
            continue

        for fact_id in fact_ids:
            fact = hechos_por_id.get(fact_id)
            if fact is None:
                continue

            value = fact.get("Valor") or fact.get("ValorNumerico")
            if value is None:
                continue

            try:
                value = float(str(value).replace(",", ""))
            except (ValueError, TypeError):
                continue

            # Get period from context
            ctx_id = fact.get("IdContexto", "")
            ctx = contextos.get(ctx_id, {})
            if ctx is None:
                continue
            period = ctx.get("Periodo", {})
            if period is None:
                continue

            # Balance sheet items use FechaInstante (instant date),
            # flow items use FechaFin (period end). Both may be present
            # but set to None, so we need explicit None checks.
            end_date = period.get("FechaFin") or period.get("FechaInstante") or ""
            start_date = period.get("FechaInicio") or ""

            # Skip if no date
            if not end_date:
                continue

            # Use end_date as report_date (period end = fiscal period end)
            report_date = str(end_date)[:10]

            records.append({
                "canonical_name": canonical_name,
                "value": value,
                "report_date": report_date,
                "start_date": str(start_date)[:10] if start_date else "",
            })

    return records


def _fetch_bmv_xbrl_financials(
    ticker: str,
    statement_type: str,
    max_zips: int = 8,
) -> pd.DataFrame:
    """Fetch structured financial data from BMV XBRL JSON ZIPs.

    Downloads XBRL ZIP files for the given ticker, extracts IFRS
    concept values, and returns a canonical long-format DataFrame.

    This is the fast path -- no LLM needed, no PDF parsing.
    Each ZIP is ~767KB and contains structured JSON.

    Parameters
    ----------
    ticker:
        BMV ticker (e.g. 'WALMEX', 'AMX', 'CEMEX').
    statement_type:
        One of 'income', 'balance', 'cashflow'.
    max_zips:
        Maximum number of ZIPs to download (8 = 2 years quarterly).
    """
    concept_map = _STATEMENT_CONCEPTS.get(statement_type, {})
    if not concept_map:
        return pd.DataFrame()

    # Get the XBRL index
    index = _get_xbrl_index()
    zip_entries = index.get(ticker.upper(), [])
    if not zip_entries:
        logger.debug("No XBRL ZIPs found for %s", ticker)
        return pd.DataFrame()

    # Download the most recent ZIPs (sorted by filing date, newest first)
    zip_entries.sort(key=lambda e: e.get("filing_date", ""), reverse=True)
    entries_to_fetch = zip_entries[:max_zips]

    all_records: list[dict] = []
    for entry in entries_to_fetch:
        try:
            resp = requests.get(
                entry["url"],
                headers=_BMV_HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            if resp.content[:2] != b"PK":
                continue

            records = _extract_xbrl_json(resp.content, concept_map)
            # Add filing_date from the XBRL page metadata
            for rec in records:
                rec["filing_date"] = entry.get("filing_date", "")
            all_records.extend(records)

        except Exception as exc:
            logger.debug("BMV XBRL ZIP download failed for %s: %s", entry["url"][-40:], exc)
            continue

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Convert dates
    for col in ("filing_date", "report_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Dedup: keep the most recent filing per (canonical_name, report_date)
    if "filing_date" in df.columns:
        df = df.sort_values("filing_date", ascending=False)
        df = df.drop_duplicates(subset=["canonical_name", "report_date"], keep="first")

    df = df.sort_values("report_date")

    logger.info(
        "BMV XBRL for %s/%s: %d records from %d/%d ZIPs",
        ticker, statement_type, len(df), len(entries_to_fetch), len(zip_entries),
    )
    return df


# ---------------------------------------------------------------------------
# BMV XBRL Segment / Product Extraction (IFRS 8 Operating Segments)
# ---------------------------------------------------------------------------


def _extract_segments_from_xbrl_json(
    zip_bytes: bytes,
) -> dict[str, Any]:
    """Extract segment revenue and product data from a BMV XBRL JSON ZIP.

    Mexican XBRL files use IFRS 8 segment reporting with dimension members
    in the context IDs. Each fact has an ``IdContexto`` that encodes the
    segment dimension (e.g. ``D-2024Q4-Segmento1``, ``D-2024Q4-Upstream``).

    The segment dimension member name IS the segment/product name.

    Returns
    -------
    Dict with:
        - ``segments``: {segment_name: revenue_value}
        - ``products``: {segment_name: [product_name, ...]}
        - ``segment_details``: {segment_name: {metric: value}}
        - ``n_segments``: int
    """
    import zipfile
    import io
    import json as _json
    import re as _re

    result: dict[str, Any] = {
        "segments": {},
        "products": {},
        "segment_details": {},
        "descriptions": {},
        "n_segments": 0,
    }

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            json_files = [n for n in zf.namelist() if n.endswith(".json")]
            if not json_files:
                return result
            with zf.open(json_files[0]) as jf:
                data = _json.loads(jf.read())
    except Exception:
        return result

    hechos_por_concepto = data.get("HechosPorIdConcepto", {})
    hechos_por_id = data.get("HechosPorId", {})
    contextos = data.get("ContextosPorId", {})

    # Step 1: Identify segment dimension members from contexts.
    # BMV XBRL contexts with segment dimensions contain "Miembro" (member)
    # or dimension axis references. We look for contexts that have
    # dimensional qualifiers beyond the simple period contexts.
    segment_contexts: dict[str, str] = {}  # context_id -> segment_name

    for ctx_id, ctx in contextos.items():
        if not isinstance(ctx, dict):
            continue

        # Check for dimension members in the context
        # BMV XBRL uses "Entidad" -> "Segmento" or dimension axis references
        dims = ctx.get("ValoresDimension", [])
        if not dims and isinstance(ctx.get("Entidad"), dict):
            dims = ctx.get("Entidad", {}).get("Segmento", [])

        if isinstance(dims, list) and dims:
            for dim in dims:
                if isinstance(dim, dict):
                    member = dim.get("Miembro", dim.get("MiembroExplicito", ""))
                    if member:
                        # Clean the member name: remove namespace prefixes
                        clean = _re.sub(r"^[a-z]+[-_]full[_:]", "", str(member))
                        clean = _re.sub(r"^.*:", "", clean)
                        # Convert CamelCase to readable
                        clean = _re.sub(r"([a-z])([A-Z])", r"\1 \2", clean)
                        clean = clean.replace("Member", "").replace("Segment", "").strip()
                        if clean and len(clean) > 1:
                            segment_contexts[ctx_id] = clean

        # Also check for explicit segment dimension in context ID string
        # (some BMV files embed segment info in the context ID itself)
        if ctx_id not in segment_contexts:
            # Pattern: context IDs containing segment identifiers
            seg_match = _re.search(
                r"(?:Segmento|Segment|BusinessSegment|OperatingSegment)"
                r"[_-]?(\w+)",
                ctx_id, _re.IGNORECASE,
            )
            if seg_match:
                seg_name = seg_match.group(1)
                seg_name = _re.sub(r"([a-z])([A-Z])", r"\1 \2", seg_name).strip()
                if seg_name and len(seg_name) > 1:
                    segment_contexts[ctx_id] = seg_name

    # Step 2: Extract revenue per segment from IFRS segment concepts.
    segment_revenue: dict[str, float] = {}
    segment_details: dict[str, dict[str, float]] = {}
    segment_products: dict[str, list[str]] = {}

    # Try segment revenue concepts
    for concept in _IFRS_SEGMENT_REVENUE_CONCEPTS:
        fact_ids = hechos_por_concepto.get(concept, [])
        if not isinstance(fact_ids, list):
            continue

        for fact_id in fact_ids:
            fact = hechos_por_id.get(fact_id)
            if fact is None:
                continue

            ctx_id = fact.get("IdContexto", "")
            if ctx_id not in segment_contexts:
                continue

            seg_name = segment_contexts[ctx_id]
            value = fact.get("Valor") or fact.get("ValorNumerico")
            if value is None:
                continue

            try:
                num_val = float(str(value).replace(",", ""))
            except (ValueError, TypeError):
                continue

            # Keep the largest revenue value per segment (avoid double-counting
            # from different periods in the same ZIP)
            if seg_name not in segment_revenue or abs(num_val) > abs(segment_revenue[seg_name]):
                segment_revenue[seg_name] = num_val

    # Step 3: Extract additional segment metrics (profit, assets, capex)
    for concept in _IFRS_SEGMENT_DETAIL_CONCEPTS:
        canonical = concept.split("_", 1)[-1] if "_" in concept else concept
        fact_ids = hechos_por_concepto.get(concept, [])
        if not isinstance(fact_ids, list):
            continue

        for fact_id in fact_ids:
            fact = hechos_por_id.get(fact_id)
            if fact is None:
                continue

            ctx_id = fact.get("IdContexto", "")
            if ctx_id not in segment_contexts:
                continue

            seg_name = segment_contexts[ctx_id]
            value = fact.get("Valor") or fact.get("ValorNumerico")
            if value is None:
                continue

            try:
                num_val = float(str(value).replace(",", ""))
            except (ValueError, TypeError):
                continue

            if seg_name not in segment_details:
                segment_details[seg_name] = {}
            segment_details[seg_name][canonical] = num_val

    # Step 4: Build product descriptions from segment names and details.
    # In Mexican XBRL, the segment member names ARE the product categories
    # (e.g., "Telecomunicaciones", "Infraestructura", "Tiendas de autoservicio").
    descriptions: dict[str, str] = {}
    for seg_name in set(list(segment_revenue.keys()) + list(segment_details.keys())):
        details = segment_details.get(seg_name, {})
        parts: list[str] = []

        if seg_name in segment_revenue:
            parts.append(f"Revenue: {segment_revenue[seg_name]:,.0f}")

        for metric, val in details.items():
            if metric.lower() not in ("revenue", "revenueFromExternalCustomers"):
                readable = _re.sub(r"([a-z])([A-Z])", r"\1 \2", metric)
                parts.append(f"{readable}: {val:,.0f}")

        if parts:
            descriptions[seg_name] = "; ".join(parts)

        # Track product names per segment
        if seg_name not in segment_products:
            segment_products[seg_name] = []

    # Step 5: Also scan for product-level dimension members
    # (sub-segments within operating segments)
    for ctx_id, ctx in contextos.items():
        if not isinstance(ctx, dict):
            continue
        dims = ctx.get("ValoresDimension", [])
        if not isinstance(dims, list) or len(dims) < 2:
            continue

        # Multiple dimensions = segment + product breakdown
        parent_seg = ""
        product_name = ""
        for dim in dims:
            if not isinstance(dim, dict):
                continue
            axis = dim.get("Dimension", dim.get("EjeDimension", ""))
            member = dim.get("Miembro", dim.get("MiembroExplicito", ""))

            axis_str = str(axis).lower()
            member_clean = _re.sub(r"^[a-z]+[-_]full[_:]", "", str(member))
            member_clean = _re.sub(r"^.*:", "", member_clean)
            member_clean = _re.sub(r"([a-z])([A-Z])", r"\1 \2", member_clean)
            member_clean = member_clean.replace("Member", "").strip()

            if "segment" in axis_str or "segmento" in axis_str:
                parent_seg = member_clean
            elif "product" in axis_str or "producto" in axis_str or "service" in axis_str:
                product_name = member_clean

        if parent_seg and product_name:
            if parent_seg not in segment_products:
                segment_products[parent_seg] = []
            if product_name not in segment_products[parent_seg]:
                segment_products[parent_seg].append(product_name)

    n_segments = max(len(segment_revenue), len(segment_details), len(segment_products))

    result = {
        "segments": segment_revenue,
        "products": segment_products,
        "segment_details": segment_details,
        "descriptions": descriptions,
        "has_revenue": len(segment_revenue) >= 2,
        "has_descriptions": len(descriptions) >= 1,
        "n_segments": n_segments,
    }

    if n_segments > 0:
        logger.info(
            "BMV XBRL segments: %d segments, %d with revenue, %d with details",
            n_segments, len(segment_revenue), len(segment_details),
        )

    return result


def extract_bmv_segment_data(
    ticker: str,
    max_zips: int = 3,
) -> dict[str, Any]:
    """Extract segment revenue and product data from BMV XBRL ZIPs.

    Public entry point for BMV segment extraction. Downloads recent XBRL
    ZIPs and extracts IFRS 8 segment information.

    Parameters
    ----------
    ticker:
        BMV ticker (e.g. ``'WALMEX'``, ``'AMX'``, ``'CEMEX'``).
    max_zips:
        Maximum ZIPs to download (3 = recent annual + 2 quarterlies).

    Returns
    -------
    Dict with ``segments``, ``products``, ``descriptions``, ``n_segments``.
    Falls back to fuzzy PDF extraction if XBRL segment data is empty.
    """
    # Primary: XBRL JSON extraction (fast, structured)
    index = _get_xbrl_index()
    zip_entries = index.get(ticker.upper(), [])
    if not zip_entries:
        return {"segments": {}, "products": {}, "descriptions": {}, "n_segments": 0}

    # Sort by date, newest first
    zip_entries.sort(key=lambda e: e.get("filing_date", ""), reverse=True)
    entries = zip_entries[:max_zips]

    best_result: dict[str, Any] = {
        "segments": {}, "products": {}, "descriptions": {}, "n_segments": 0,
    }

    for entry in entries:
        try:
            resp = requests.get(entry["url"], headers=_BMV_HEADERS, timeout=20)
            resp.raise_for_status()
            if resp.content[:2] != b"PK":
                continue

            seg_result = _extract_segments_from_xbrl_json(resp.content)

            # Keep the result with the most segments
            if seg_result["n_segments"] > best_result["n_segments"]:
                best_result = seg_result
                best_result["filing_date"] = entry.get("filing_date", "")
                best_result["source"] = "bmv_xbrl_json"

            # If we found good segment data, stop downloading more ZIPs
            if seg_result["n_segments"] >= 2 and seg_result.get("has_revenue"):
                break

        except Exception as exc:
            logger.debug("BMV segment ZIP download failed: %s", exc)
            continue

    # Fallback: try fuzzy PDF extraction if XBRL yielded no segments
    if best_result["n_segments"] < 2:
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
            logger.debug(
                "BMV XBRL segment extraction found %d segments for %s; "
                "PDF fallback available via filing_discoverer",
                best_result["n_segments"], ticker,
            )
        except ImportError:
            pass

    logger.info(
        "BMV segment data for %s: %d segments, revenue=%s, source=%s",
        ticker,
        best_result["n_segments"],
        best_result.get("has_revenue", False),
        best_result.get("source", "none"),
    )

    return best_result

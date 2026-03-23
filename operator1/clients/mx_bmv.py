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

    # -- Institutional holders (yfinance backed) -----------------------------

    def _yf_ticker(self, identifier: str) -> str:
        """Convert BMV ticker to yfinance format.

        Mexican stocks have series suffixes (A, B, L, CPO, etc.).
        yfinance expects ``{ticker}.MX`` format.  If the identifier
        already has .MX, return as-is.  Otherwise try common series
        suffixes until one works.
        """
        if ".MX" in identifier.upper():
            return identifier
        # Try the identifier directly first, then with common series
        return f"{identifier}.MX"

    def _try_yf_tickers(self, identifier: str) -> "yf.Ticker | None":
        """Try multiple yfinance ticker variants for BMV stocks.

        Mexican stocks use series suffixes (AMXB, AMXL, WALMEX, etc.).
        Returns the first Ticker that has major_holders data.
        """
        try:
            import yfinance as yf
        except ImportError:
            return None

        # Strip .MX if present
        base = identifier.upper().replace(".MX", "").strip()

        # Try these variants in order
        variants = [
            f"{base}.MX",           # direct (WALMEX.MX)
            f"{base}B.MX",          # B-series (AMXB.MX)
            f"{base}A.MX",          # A-series
            f"{base}L.MX",          # L-series (AMXL.MX)
            f"{base}CPO.MX",        # CPO series (TLEVISACPO.MX)
        ]

        for variant in variants:
            try:
                tick = yf.Ticker(variant)
                mh = tick.major_holders
                if mh is not None and not mh.empty:
                    logger.debug("BMV yfinance ticker resolved: %s -> %s", identifier, variant)
                    return tick
            except Exception:
                continue

        return None

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch institutional + mutual fund holders via yfinance.

        BMV (EMISNET) does publish holder disclosures but the API
        requires CAPTCHA-solving or JavaScript rendering.  yfinance
        provides aggregated holder data for major BMV-listed companies.
        Series-aware ticker resolution handles the A/B/L/CPO suffix.
        """
        holders: list[dict[str, Any]] = []
        try:
            tick = self._try_yf_tickers(identifier)
            if tick is None:
                return holders

            # Institutional holders
            inst = tick.institutional_holders
            if inst is not None and not inst.empty:
                for _, row in inst.iterrows():
                    pct = row.get("pctHeld", 0) or row.get("% Out", 0) or 0
                    if isinstance(pct, (int, float)) and 0 < pct < 1:
                        pct = pct * 100
                    holders.append({
                        "name": str(row.get("Holder", "")),
                        "shares": int(row.get("Shares", 0)),
                        "value": float(row.get("Value", 0)),
                        "percentage": round(float(pct), 2),
                        "holder_type": "institutional",
                        "date_reported": str(row.get("Date Reported", "")),
                    })

            # Mutual fund holders
            mf = tick.mutualfund_holders
            if mf is not None and not mf.empty:
                for _, row in mf.iterrows():
                    pct = row.get("pctHeld", 0) or row.get("% Out", 0) or 0
                    if isinstance(pct, (int, float)) and 0 < pct < 1:
                        pct = pct * 100
                    holders.append({
                        "name": str(row.get("Holder", "")),
                        "shares": int(row.get("Shares", 0)),
                        "value": float(row.get("Value", 0)),
                        "percentage": round(float(pct), 2),
                        "holder_type": "mutualfund",
                        "date_reported": str(row.get("Date Reported", "")),
                    })

            if holders:
                logger.info("BMV holders for %s: %d from yfinance", identifier, len(holders))
        except Exception as exc:
            logger.debug("yfinance holders failed for BMV %s: %s", identifier, exc)
        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return institutional ownership metrics as a single-row snapshot."""
        try:
            from datetime import date as _date
            tick = self._try_yf_tickers(identifier)
            if tick is None:
                return pd.DataFrame()

            mh = tick.major_holders
            inst_pct = 0.0
            inst_count = 0
            if mh is not None and not mh.empty:
                for idx, row in mh.iterrows():
                    breakdown = str(row.get("Breakdown", idx)).lower() if "Breakdown" in mh.columns else str(idx).lower()
                    val = row.get("Value", row.iloc[-1]) if "Value" in mh.columns else row.iloc[-1]
                    if "institutionspercentheld" in breakdown or ("institutions" in breakdown and "percent" in breakdown):
                        inst_pct = float(val) * 100 if float(val) < 1 else float(val)
                    elif "institutionscount" in breakdown or "count" in breakdown:
                        inst_count = int(float(val))

            holders = self.get_holders(identifier)
            hhi = 0.0
            if holders:
                top5 = holders[:5]
                total_pct = sum(h.get("percentage", 0) for h in top5)
                if total_pct > 0:
                    hhi = sum((h.get("percentage", 0) / total_pct) ** 2 for h in top5)

            if inst_pct > 0 or inst_count > 0 or holders:
                return pd.DataFrame([{
                    "date_reported": pd.Timestamp(_date.today()),
                    "inst_ownership_pct": round(inst_pct, 2),
                    "inst_top5_concentration": round(hhi, 4),
                    "inst_holder_count": inst_count or len(holders),
                }])
        except Exception as exc:
            logger.debug("BMV holder history failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch insider transactions via yfinance for BMV-listed companies."""
        transactions: list[dict[str, Any]] = []
        try:
            tick = self._try_yf_tickers(identifier)
            if tick is None:
                return transactions

            insider = tick.insider_transactions
            if insider is not None and not insider.empty:
                for _, row in insider.iterrows():
                    shares = 0
                    try:
                        shares = int(row.get("Shares", 0))
                    except (ValueError, TypeError):
                        pass
                    transactions.append({
                        "insider_name": str(row.get("Insider", "")),
                        "position": str(row.get("Position", "")),
                        "date": str(row.get("Start Date", "")),
                        "transaction": str(row.get("Transaction", "")),
                        "shares": shares,
                        "value": float(row.get("Value", 0) or 0),
                    })
                logger.info("BMV insider transactions for %s: %d", identifier, len(transactions))
        except Exception as exc:
            logger.debug("yfinance insider transactions failed for BMV %s: %s", identifier, exc)
        return transactions


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

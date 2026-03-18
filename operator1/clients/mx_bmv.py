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
    """

    def __init__(self) -> None:
        self._token: str = ""
        self._obtained_at: float = 0.0

    def get_token(self) -> str:
        """Return a valid Bearer token, refreshing if needed."""
        now = time.time()
        if self._token and (now - self._obtained_at) < _TOKEN_REFRESH_INTERVAL_S:
            return self._token

        try:
            resp = requests.get(
                _BMV_TOKEN_URL,
                headers=_BMV_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["response"]["access_token"]
            self._obtained_at = now
            logger.debug("BMV token obtained: %s...", self._token[:12])
        except Exception as exc:
            logger.warning("Failed to obtain BMV token: %s", exc)
            # Return stale token if we have one
            if not self._token:
                raise

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
        logger.warning("BMV search failed for '%s': %s", term, exc)
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
        """Fetch financials via BMV/EMISNET filing discovery only (PIT-compliant).

        yfinance is NOT used for financial statements because it does not
        provide true filing dates (sets filing_date = report_date).
        """
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

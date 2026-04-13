"""UAE DFM/ADX PIT client -- native DFM APIs (no yfinance for search/profile).

Three API endpoints discovered by reverse-engineering the DFM Nuxt.js SPA:

1. **Market Data API** -- api2.dfm.ae/mw/v1/stocks
   - 461 securities (85+ equities) with real-time OHLCV, bid/ask, 52-week range
   - No authentication required
   - Powers the DFM marketwatch

2. **eFsah Disclosure API** -- api2.dfm.ae/efsah/v1/prototype_efsah
   - Company disclosures with PIT publication dates
   - Financial statements (annual, interim, quarterly)
   - PDF resources via cms_resources=true parameter
   - 746+ disclosures per major company (EMAAR)
   - UTF-8 BOM in responses (decode with utf-8-sig)

3. **Document Download** -- feeds.dfm.ae/documents/{r_path}
   - PDF download from eFsah resource paths
   - No authentication required
   - Validated: 17.2 MB Emaar Properties IR PDF confirmed

Company search: 510 DFM-listed companies extracted from the Nuxt SSR state
of the DFM disclosure page (SecuritySymbol, FullName, Exchange, Sector).

Coverage: ~100+ listed companies on DFM/ADX, ~$0.8T market cap.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)
_CACHE_DIR = Path("cache/ae_dfm")

# ---------------------------------------------------------------------------
# DFM API constants
# ---------------------------------------------------------------------------

_DFM_API2_BASE = "https://api2.dfm.ae"
_DFM_STOCKS_URL = f"{_DFM_API2_BASE}/mw/v1/stocks"
_DFM_EFSAH_URL = f"{_DFM_API2_BASE}/efsah/v1/prototype_efsah"
_DFM_EFSAH_COUNT_URL = f"{_DFM_API2_BASE}/efsah/v1/efsah_count"
_DFM_FEEDS_BASE = "https://feeds.dfm.ae/documents"
_DFM_DISCLOSURE_PAGE = "https://www.dfm.ae/the-exchange/news-disclosures/disclosures"

_DFM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Origin": "https://www.dfm.ae",
    "Referer": "https://www.dfm.ae/",
}


# ---------------------------------------------------------------------------
# Stock list helpers (api2.dfm.ae/mw/v1/stocks)
# ---------------------------------------------------------------------------

_stocks_cache: list[dict] | None = None
_stocks_cache_time: float = 0


def _fetch_stocks(force: bool = False) -> list[dict]:
    """Fetch all DFM-listed securities from the market data API.

    Returns 461 securities including equities, bonds, sukuk, and futures.
    Cached for 1 hour to avoid redundant API calls.
    """
    import time

    global _stocks_cache, _stocks_cache_time
    if not force and _stocks_cache and (time.time() - _stocks_cache_time) < 3600:
        return _stocks_cache

    try:
        resp = requests.get(_DFM_STOCKS_URL, headers=_DFM_HEADERS, timeout=15)
        resp.raise_for_status()
        stocks = resp.json()
        if isinstance(stocks, list):
            _stocks_cache = stocks
            _stocks_cache_time = time.time()
            logger.info("DFM stocks API: %d securities", len(stocks))
            return stocks
    except Exception as exc:
        logger.debug("DFM stocks API failed: %s", exc)

    return _stocks_cache or []


def _is_equity(stock: dict) -> bool:
    """Filter equities from bonds/derivatives in the stock list.

    Equities have short alphabetic IDs (e.g. EMAAR, DIB, DFM).
    Bonds/sukuk have IDs with digits (e.g. EMAAR0624USD, AGRBK0827USD).
    Futures have date suffixes (e.g. EMAARH26, EMAARJ26).
    """
    stock_id = stock.get("id", "")
    if not stock_id or len(stock_id) > 12:
        return False
    # Exclude IDs with 4+ trailing digits (bonds, futures)
    if re.search(r"\d{4}", stock_id):
        return False
    # Exclude single-letter month + 2-digit year suffixes (futures)
    if re.search(r"[A-Z][FGHJKMNQUVXZ]\d{2}$", stock_id):
        return False
    return True


def _find_stock(symbol: str) -> dict | None:
    """Find a stock by symbol in the DFM market data."""
    stocks = _fetch_stocks()
    symbol_upper = symbol.upper().strip()
    for s in stocks:
        if s.get("id", "").upper() == symbol_upper:
            return s
    return None


# ---------------------------------------------------------------------------
# Company list from Nuxt SSR state
# ---------------------------------------------------------------------------

_companies_cache: list[dict] | None = None


def _fetch_companies() -> list[dict]:
    """Extract company list from the DFM disclosure page's Nuxt SSR state.

    The Nuxt state contains 510 DFM-listed companies with
    SecuritySymbol, FullName, Exchange, Sector, SecurityType.
    """
    global _companies_cache
    if _companies_cache is not None:
        return _companies_cache

    try:
        resp = requests.get(
            _DFM_DISCLOSURE_PAGE,
            headers={"User-Agent": _DFM_HEADERS["User-Agent"]},
            timeout=15,
        )
        resp.raise_for_status()

        # Extract SecuritySymbol + FullName from Nuxt state
        companies = re.findall(
            r'SecuritySymbol:"([^"]+)",FullName:"([^"]+)"',
            resp.text,
        )
        if companies:
            _companies_cache = [
                {
                    "ticker": sym,
                    "name": name,
                    "country": "AE",
                    "exchange": "DFM",
                    "market_id": "ae_dfm",
                }
                for sym, name in companies
            ]
            logger.info("DFM company list: %d companies from Nuxt state", len(_companies_cache))
            return _companies_cache

    except Exception as exc:
        logger.debug("DFM company list fetch failed: %s", exc)

    return []


# ---------------------------------------------------------------------------
# eFsah disclosure API helpers
# ---------------------------------------------------------------------------

def _efsah_get(endpoint: str, params: dict) -> dict:
    """Query the eFsah API and handle UTF-8 BOM in response."""
    try:
        resp = requests.get(
            f"{_DFM_API2_BASE}/efsah/v1/{endpoint}",
            params=params,
            headers=_DFM_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        return json.loads(resp.content.decode("utf-8-sig"))
    except Exception as exc:
        logger.debug("eFsah %s failed: %s", endpoint, exc)
        return {}


def _fetch_disclosures(
    symbol: str,
    take: int = 20,
    skip: int = 0,
    with_resources: bool = False,
) -> list[dict]:
    """Fetch disclosures for a company from eFsah.

    Parameters
    ----------
    symbol:
        DFM ticker symbol (e.g. 'EMAAR', 'DIB').
    take:
        Number of results per page.
    skip:
        Number of results to skip (pagination).
    with_resources:
        If True, include PDF resource paths in response.

    Returns
    -------
    List of disclosure dicts with keys: id, publication_date, headline,
    issuer_symbol, issuer, announcement_type, integrated_report_type,
    and optionally resources[].
    """
    params: dict[str, str] = {
        "announcement_type": "Disclosure",
        "symbol": symbol,
        "take": str(take),
        "skip": str(skip),
        "lang": "en",
        "h7_datetime_format": "MMM dd, yyyy HH:mm:ss",
    }
    if with_resources:
        params["cms_resources"] = "true"

    data = _efsah_get("prototype_efsah", params)
    return data.get("root", [])


def _get_pdf_url(resource: dict) -> str:
    """Build full PDF URL from eFsah resource path."""
    r_path = resource.get("r_path", "")
    if r_path:
        return f"{_DFM_FEEDS_BASE}{r_path}"
    return ""


# ---------------------------------------------------------------------------
# AEDfmClient -- PIT client for UAE DFM/ADX equities
# ---------------------------------------------------------------------------


class AEDfmClient:
    """PIT client for UAE DFM/ADX equities.

    Uses three native DFM APIs (no auth required):
    1. api2.dfm.ae/mw/v1/stocks -- market data + company search
    2. api2.dfm.ae/efsah/v1 -- disclosure discovery (eFsah)
    3. feeds.dfm.ae/documents -- PDF download

    Falls back to yfinance for OHLCV and profile enrichment.
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
        return "ae_dfm"

    @property
    def market_name(self) -> str:
        return "UAE (DFM / ADX)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Search for DFM-listed companies.

        Uses the DFM Nuxt SSR state (510 companies) for name search,
        and the api2 market data (85+ equities) for ticker lookup.
        Falls back to yfinance if both fail.
        """
        if not query:
            # Browse mode: return equity list from market data API
            stocks = _fetch_stocks()
            equities = [s for s in stocks if _is_equity(s)]
            return [
                {
                    "ticker": s["id"],
                    "name": s.get("name") or s["id"],
                    "country": "AE",
                    "exchange": "DFM",
                    "market_id": "ae_dfm",
                }
                for s in equities[:50]
            ]

        query_upper = query.upper().strip()

        # Try exact ticker match from market data
        stock = _find_stock(query_upper)
        if stock:
            return [{
                "ticker": stock["id"],
                "name": stock.get("name") or stock["id"],
                "country": "AE",
                "exchange": "DFM",
                "market_id": "ae_dfm",
            }]

        # Search by name in the Nuxt company list
        companies = _fetch_companies()
        query_lower = query.lower()
        matches = [
            c for c in companies
            if query_lower in c.get("name", "").lower()
            or query_lower in c.get("ticker", "").lower()
        ]
        if matches:
            logger.info("DFM search for '%s': %d matches from Nuxt state", query, len(matches))
            return matches[:25]

        # Fallback: try yfinance search
        try:
            from operator1.clients.yfinance_backed import yf_search
            return yf_search(query, self.market_id, "AE", "DFM", yf_suffix=".AE")
        except Exception:
            return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from DFM market data + eFsah.

        Data sources (in priority order):
        1. api2.dfm.ae/mw/v1/stocks -- market data (price, 52w range, volume)
        2. Nuxt SSR company list -- company name
        3. yfinance enrichment -- sector, industry, market cap
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        profile: dict[str, Any] = {
            "ticker": identifier,
            "name": identifier,
            "country": "AE",
            "exchange": "DFM",
            "currency": "AED",
            "market_id": "ae_dfm",
        }

        # Step 1: Market data for price info
        stock = _find_stock(identifier)
        if stock:
            profile["latest_close"] = stock.get("closingprice")
            profile["previous_close"] = stock.get("previousclosingprice")
            profile["high_52w"] = stock.get("highestin52weeks")
            profile["low_52w"] = stock.get("lowestin52weeks")
            profile["volume"] = stock.get("totalvolume")
            profile["total_value"] = stock.get("totalvalue")
            if stock.get("name"):
                profile["name"] = stock["name"]

        # Step 2: Name from Nuxt company list
        companies = _fetch_companies()
        for c in companies:
            if c.get("ticker", "").upper() == identifier.upper():
                profile["name"] = c.get("name", profile["name"])
                break

        # Step 3: yfinance enrichment for sector/industry/market_cap
        try:
            from operator1.clients.yfinance_backed import yf_get_profile
            yf_profile = yf_get_profile(
                identifier, self.market_id, "UAE", "AE", "DFM", "AED",
                yf_suffix=".AE",
            )
            for key in ("sector", "industry", "market_cap", "isin", "website"):
                if yf_profile.get(key) and not profile.get(key):
                    profile[key] = yf_profile[key]
        except Exception as exc:
            logger.debug("DFM yfinance enrichment failed: %s", exc)

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
        """Fetch financials via DFM eFsah filing discovery + LLM extraction.

        Uses the eFsah API (api2.dfm.ae/efsah/v1) for discovery and
        feeds.dfm.ae for PDF download. PIT-compliant: publication_date
        from eFsah is the true filing date.

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
                logger.info(
                    "DFM %s %s: %d rows from filing discovery",
                    identifier, statement_type, len(df),
                )
                return df
        except Exception as exc:
            logger.debug("DFM filing discovery failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """DFM real-time data is snapshot only (no historical OHLCV).

        Historical OHLCV is handled by ohlcv_provider.py (yfinance .AE).
        """
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    # -- Holder data (eFsah disclosure API) ----------------------------------

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch holder-related disclosures from DFM eFsah API.

        The eFsah API at api2.dfm.ae/efsah/v1/prototype_efsah returns
        all disclosures for a company including AGM invitations (which
        contain shareholding structures), BOD meeting results, and
        governance reports.  PDF attachments can be extracted for
        detailed holder tables via the LLM/fuzzy parser.

        Probing confirmed (2026-03-23):
          - eFsah returns disclosures (announcement_type='Disclosure')
          - eFsah count shows 746 total for EMAAR
          - DFM stocks API has no ownership fields (only 'capital': None)
          - DFM Nuxt website routes return 404 for company-specific pages
          - ADX company page has 'holder' keyword (189KB)
        """
        holders: list[dict[str, Any]] = []
        try:
            disclosures = _fetch_disclosures(
                symbol=identifier,
                take=50,
                with_resources=True,
            )
            if not disclosures:
                return holders

            # Filter for holder/governance related disclosures
            holder_keywords = (
                "agm", "general meeting", "shareholder", "board",
                "director", "governance", "capital", "ownership",
            )
            for disc in disclosures:
                headline = str(disc.get("headline", "")).lower()
                if not any(kw in headline for kw in holder_keywords):
                    continue

                pub_date = str(disc.get("publication_date", ""))[:10]
                title = disc.get("headline", "")

                # Check for PDF resources that may contain holder tables
                resources = disc.get("resources", [])
                pdf_url = ""
                if resources:
                    for res in resources:
                        url = _get_pdf_url(res)
                        if url:
                            pdf_url = url
                            break

                holders.append({
                    "name": title[:60],
                    "shares": 0,
                    "value": 0.0,
                    "percentage": 0.0,
                    "holder_type": "disclosure",
                    "date_reported": pub_date,
                    "source": "dfm_efsah",
                    "pdf_url": pdf_url,
                })

            if holders:
                logger.info(
                    "DFM holders for %s: %d governance/holder disclosures from eFsah",
                    identifier, len(holders),
                )
        except Exception as exc:
            logger.debug("DFM eFsah holder search failed for %s: %s", identifier, exc)

        # Fallback: extract shareholders from filing PDFs
        if not holders:
            try:
                from operator1.clients.filing_discoverer import try_shareholding_extraction
                holders = try_shareholding_extraction(identifier, market_id=self.market_id)
                if holders:
                    logger.info("DFM holders from PDF shareholding extraction: %d", len(holders))
            except Exception as exc:
                logger.debug("DFM PDF shareholding fallback failed: %s", exc)


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
        """Return holder metrics from DFM eFsah disclosures."""
        try:
            from datetime import date as _date
            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            return pd.DataFrame([{
                "date_reported": pd.Timestamp(_date.today()),
                "inst_ownership_pct": 0.0,
                "inst_top5_concentration": 0.0,
                "inst_holder_count": len(holders),
            }])
        except Exception as exc:
            logger.debug("DFM holder history failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch BOD meeting results from DFM eFsah (contains director changes).

        BOD meeting disclosures contain director appointment/resignation
        information.  Full transaction details require PDF extraction.
        """
        transactions: list[dict[str, Any]] = []
        try:
            disclosures = _fetch_disclosures(symbol=identifier, take=50)
            for disc in disclosures:
                headline = str(disc.get("headline", "")).lower()
                if any(kw in headline for kw in ("bod meeting", "board", "director", "appointment")):
                    transactions.append({
                        "insider_name": disc.get("headline", "")[:60],
                        "position": "Board",
                        "date": str(disc.get("publication_date", ""))[:10],
                        "transaction": "Board Disclosure",
                        "shares": 0,
                        "value": 0.0,
                        "source": "dfm_efsah",
                    })
            if transactions:
                logger.info(
                    "DFM insider disclosures for %s: %d from eFsah",
                    identifier, len(transactions),
                )
        except Exception as exc:
            logger.debug("DFM eFsah insider search failed for %s: %s", identifier, exc)
        return transactions

    def extract_segment_data(self, identifier: str) -> dict:
        """Extract product segment data from PDF filings via fuzzy parser.

        Uses the filing discovery + fuzzy_pdf_parser pipeline with
        market-specific segment keywords for ae_dfm.
        """
        try:
            from operator1.clients.filing_discoverer import try_segment_extraction
            return try_segment_extraction(identifier, "ae_dfm")
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                "AEDfmClient segment extraction failed for %s: %s", identifier, exc,
            )
            return {"n_segments": 0, "segments": {}, "descriptions": {}}


"""Switzerland SIX PIT client -- FQS + share details + official notices APIs.

Three undocumented SIX APIs discovered by reverse-engineering the React SPAs
on six-group.com. All free, no authentication required.

1. **FQS (Financial Query Service)** -- /fqs/ref.json
   - 110K+ securities with ValorId, ISIN, ticker (ValorSymbol), name
   - Searchable by ticker, ISIN, or name
   - Powers the share explorer on six-group.com

2. **Share Details API** -- /sheldon/share_details/v3/{ValorId}/...
   - overview/info.json: issuer code, website, next GM date
   - share/info.json: numberInIssue, nominalValue, indices, regulatory standard
   - share/dividend.json: 18+ years of ex-dividend dates and amounts
   - issuer/financial_reports.json: auditor, accounting standard, closing date
   - issuer/capital_structure.json: share capital, conditional capital
   - issuer/contact.json: issuer contact info

3. **Official Notices API** -- /sheldon/official_notices/v2/...
   - Corporate actions (capital destruction, buybacks) with PIT dates
   - Ex-dividend notices
   - 349K+ notices

ValorId format: {ISIN}{currency}4 (e.g. CH0038863350CHF4 for Nestle)

Financial statements: EU ESEF crossover (Swiss blue chips file IFRS via ESEF).
SIX does not provide financial statement line items through these APIs --
only metadata (auditor, standard, closing date).

OHLCV: yfinance (.SW suffix) via ohlcv_provider.py. SIX provides ~5 months
of close+volume via /sheldon/market_data/v1/{ValorId}/historic.csv but lacks
OHLC and the 2-year window the pipeline requires.

Coverage: ~250+ listed companies on SIX, ~$1.8T market cap.
"""
from __future__ import annotations

import io
import json
import logging
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)
_CACHE_DIR = Path("cache/ch_six")

# ---------------------------------------------------------------------------
# SIX API constants
# ---------------------------------------------------------------------------

_SIX_BASE = "https://www.six-group.com"

# FQS (Financial Query Service) -- reference data for all SIX securities
_FQS_REF_URL = f"{_SIX_BASE}/fqs/ref.json"

# Share details -- per-security structured data
_SHARE_DETAILS_BASE = f"{_SIX_BASE}/sheldon/share_details/v3"

# Official notices -- corporate actions, dividends
_NOTICES_FIND_URL = f"{_SIX_BASE}/sheldon/official_notices/v2/find.json"
_NOTICES_DETAIL_URL = f"{_SIX_BASE}/sheldon/official_notices/v2/details"

# Market data -- historic price CSV
_MARKET_DATA_BASE = f"{_SIX_BASE}/sheldon/market_data/v1"

_SIX_HEADERS = {
    "User-Agent": "Operator1/1.0 (financial-research)",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# FQS reference data helpers
# ---------------------------------------------------------------------------

def _fqs_search(
    ticker: str = "",
    isin: str = "",
    name: str = "",
    page_size: int = 20,
) -> list[dict[str, Any]]:
    """Search the SIX FQS reference data for securities.

    The FQS ref endpoint covers 110K+ securities listed on SIX.
    Returns ValorId (needed for all other SIX APIs), ISIN, ticker
    (ValorSymbol), name (ShortName), and ValorNumber.

    Parameters
    ----------
    ticker:
        SIX ticker symbol (ValorSymbol), e.g. 'NESN', 'NOVN'.
    isin:
        ISIN code, e.g. 'CH0038863350'.
    name:
        Company name substring search.
    page_size:
        Max results to return.

    Returns
    -------
    List of dicts with keys: name, ticker, isin, valor_number, valor_id.
    """
    params: dict[str, str] = {
        "select": "ShortName,ValorSymbol,ISIN,ValorNumber,ValorId",
        "pagesize": str(page_size),
    }

    # Build the where clause
    if ticker:
        params["where"] = f"ValorSymbol={ticker.upper()}"
    elif isin:
        params["where"] = f"ISIN={isin}"
    elif name:
        params["where"] = f"ShortName~{name.upper()}"
    else:
        params["orderby"] = "ShortName"

    try:
        resp = requests.get(_FQS_REF_URL, params=params, headers=_SIX_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        col_names = data.get("colNames", [])
        rows = data.get("rowData", [])

        results: list[dict[str, Any]] = []
        for row in rows:
            record = dict(zip(col_names, row))
            results.append({
                "name": record.get("ShortName", ""),
                "ticker": record.get("ValorSymbol", ""),
                "isin": record.get("ISIN", ""),
                "valor_number": record.get("ValorNumber"),
                "valor_id": record.get("ValorId", ""),
                "country": "CH",
                "exchange": "SIX",
                "market_id": "ch_six",
            })
        return results

    except Exception as exc:
        logger.debug("FQS search failed: %s", exc)
        return []


def _get_valor_id(ticker: str = "", isin: str = "") -> str:
    """Resolve a ticker or ISIN to a SIX ValorId.

    ValorId format: {ISIN}{currency}4 (e.g. CH0038863350CHF4).
    This is needed by all share_details and market_data endpoints.
    """
    results = _fqs_search(ticker=ticker, isin=isin, page_size=1)
    if results:
        return results[0].get("valor_id", "")
    return ""


# ---------------------------------------------------------------------------
# Share details helpers
# ---------------------------------------------------------------------------

def _fetch_share_detail(valor_id: str, path: str) -> dict[str, Any]:
    """Fetch a share details endpoint for a given ValorId."""
    if not valor_id:
        return {}
    url = f"{_SHARE_DETAILS_BASE}/{valor_id}/{path}"
    try:
        resp = requests.get(url, headers=_SIX_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("itemList", [])
        return items[0] if items else {}
    except Exception as exc:
        logger.debug("SIX share_details %s failed for %s: %s", path, valor_id, exc)
        return {}


def _fetch_share_detail_list(valor_id: str, path: str) -> list[dict[str, Any]]:
    """Fetch a share details endpoint that returns a list."""
    if not valor_id:
        return []
    url = f"{_SHARE_DETAILS_BASE}/{valor_id}/{path}"
    try:
        resp = requests.get(url, headers=_SIX_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("itemList", [])
    except Exception as exc:
        logger.debug("SIX share_details %s failed for %s: %s", path, valor_id, exc)
        return []


# ---------------------------------------------------------------------------
# Official notices helpers
# ---------------------------------------------------------------------------

def _six_search_notices(
    isin: str = "",
    issuer: str = "",
    years: int = 2,
    notice_types: str = "M,EX",
    page_size: int = 50,
) -> list[dict[str, Any]]:
    """Search SIX official notices by ISIN or issuer name."""
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * years)
    types = set(notice_types.upper().split(","))

    params: dict[str, str] = {
        "firstDate": start_date.strftime("%Y%m%d"),
        "lastDate": end_date.strftime("%Y%m%d"),
        "pageNumber": "0",
        "pageSize": str(page_size),
        "sortAttribute": "date",
        "sortDirection": "desc",
        "showManual": "true" if "M" in types else "false",
        "showAutomatic": "true" if "A" in types else "false",
        "showExDividend": "true" if "EX" in types else "false",
        "showFirstListing": "true" if "FL" in types else "false",
        "showDelisting": "true" if "DE" in types else "false",
        "showProvisional": "true" if "PZ" in types else "false",
    }

    if isin:
        params["valorIds"] = isin
        params["linkDirectly"] = "true"
        params["linkUnderlying"] = "false"
    elif issuer:
        params["issuerWords"] = issuer

    try:
        resp = requests.get(_NOTICES_FIND_URL, params=params, headers=_SIX_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "Ok":
            return data.get("itemList", [])
    except Exception as exc:
        logger.debug("SIX notices search failed: %s", exc)
    return []


def _six_get_notice_text(notice_id: int | str) -> str:
    """Fetch the full text of a SIX official notice."""
    url = f"{_NOTICES_DETAIL_URL}/{notice_id}.json"
    try:
        resp = requests.get(url, headers=_SIX_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("itemList", [])
        if items:
            return items[0].get("text", "")
    except Exception as exc:
        logger.debug("SIX notice detail %s failed: %s", notice_id, exc)
    return ""


def _parse_shares_outstanding(text: str) -> int | None:
    """Extract shares outstanding from a SIX capital action notice."""
    patterns = [
        r"(?:ou?tstanding|ausstehende)\s+(?:shares|Aktien):\s*(\d[\d,. ]*)",
        r"shares:\s*(\d[\d,. ]*\d)",
        r"[Aa]nzahl[^:]*:\s*(\d[\d,. ]*\d)",
        r"[Nn]ew\s+number[^:]*:\s*(\d[\d,. ]*\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            raw = match.group(1).replace(",", "").replace(".", "").replace(" ", "").strip()
            try:
                return int(raw)
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# CHSixClient -- PIT client for Swiss SIX equities
# ---------------------------------------------------------------------------


class CHSixClient:
    """PIT client for Swiss SIX equities.

    Uses three undocumented SIX APIs (no auth required):
    1. FQS ref endpoint -- company search, ValorId resolution
    2. Share details -- profile, dividends, capital structure
    3. Official notices -- corporate actions with PIT dates

    Falls back to EU ESEF for financial statements and yfinance for
    profile fields not available from SIX (sector, industry).
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
        return "ch_six"

    @property
    def market_name(self) -> str:
        return "Switzerland (SIX)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Search for SIX-listed companies via FQS reference data.

        Uses the native SIX FQS API (110K+ securities, no auth).
        Falls back to yfinance only if FQS is unreachable.
        """
        if not query:
            from operator1.clients.yfinance_backed import yf_search
            return yf_search("", self.market_id, "CH", "SIX", yf_suffix=".SW")

        # Try FQS by ticker first (exact match)
        results = _fqs_search(ticker=query, page_size=10)
        if results:
            logger.info("SIX FQS search for '%s' (ticker): %d results", query, len(results))
            return results

        # Try FQS by ISIN
        if query.startswith("CH") and len(query) >= 12:
            results = _fqs_search(isin=query, page_size=10)
            if results:
                logger.info("SIX FQS search for '%s' (ISIN): %d results", query, len(results))
                return results

        # Try FQS by name (substring match)
        results = _fqs_search(name=query, page_size=20)
        if results:
            logger.info("SIX FQS search for '%s' (name): %d results", query, len(results))
            return results

        # Fallback to yfinance
        from operator1.clients.yfinance_backed import yf_search
        return yf_search(query, self.market_id, "CH", "SIX", yf_suffix=".SW")

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from SIX APIs + yfinance enrichment.

        Data sources (in priority order):
        1. FQS ref -- ValorId, ISIN, ticker, name
        2. Share details -- shares outstanding, dividends, indices, regulatory standard
        3. Capital structure -- share capital breakdown
        4. Official notices -- PIT-dated corporate actions
        5. yfinance -- sector, industry, market_cap (SIX doesn't provide these)
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        profile: dict[str, Any] = {
            "ticker": identifier,
            "name": identifier,
            "country": "CH",
            "exchange": "SIX",
            "currency": "CHF",
            "market_id": "ch_six",
        }

        # Step 1: Resolve ValorId via FQS
        valor_id = ""
        fqs_results = _fqs_search(ticker=identifier, page_size=1)
        if not fqs_results and identifier.startswith("CH"):
            fqs_results = _fqs_search(isin=identifier, page_size=1)

        if fqs_results:
            fqs = fqs_results[0]
            valor_id = fqs.get("valor_id", "")
            profile.update({
                "name": fqs.get("name", identifier),
                "ticker": fqs.get("ticker", identifier),
                "isin": fqs.get("isin", ""),
                "valor_number": fqs.get("valor_number"),
                "valor_id": valor_id,
            })
            logger.info("SIX FQS resolved %s -> ValorId=%s", identifier, valor_id)

        # Step 2: Share info (shares outstanding, indices, regulatory standard)
        if valor_id:
            share_info = _fetch_share_detail(valor_id, "share/info.json")
            if share_info:
                profile["shares_outstanding"] = share_info.get("numberInIssue")
                profile["nominal_value"] = share_info.get("nominalValue")
                profile["trading_currency"] = share_info.get("tradingCurrency")
                profile["security_type"] = share_info.get("securityType")
                profile["regulatory_standard"] = share_info.get("regulatoryStandard")
                profile["primary_listed"] = share_info.get("primaryListed")
                profile["first_trading_date"] = share_info.get("firstTradingDate")
                profile["index_memberships"] = share_info.get("indexSymbols", [])
                profile["issued_by"] = share_info.get("issuedBy", "")
                if share_info.get("issuedBy"):
                    profile["name"] = share_info["issuedBy"]
                logger.info(
                    "SIX share info for %s: %s shares, indices=%s",
                    identifier,
                    share_info.get("numberInIssue"),
                    share_info.get("indexSymbols"),
                )

            # Step 3: Overview (issuer code, website, next GM)
            overview = _fetch_share_detail(valor_id, "overview/info.json")
            if overview:
                profile["website"] = overview.get("website", "")
                profile["issuer_code"] = overview.get("issuerCode", "")
                profile["next_gm_date"] = overview.get("nextGeneralMeeting")

            # Step 4: Capital structure
            cap_items = _fetch_share_detail_list(valor_id, "issuer/capital_structure.json")
            if cap_items:
                for item in cap_items:
                    if item.get("category") == "REPORTED_CAPITAL":
                        for cap in item.get("capitals", []):
                            if cap.get("capitalType") == "SHARE":
                                profile["reported_share_capital"] = cap.get("capital")
                                profile["reported_unit_capital"] = cap.get("unitCapital")

            # Step 5: Financial reports metadata
            fin_meta = _fetch_share_detail(valor_id, "issuer/financial_reports.json")
            if fin_meta:
                profile["accounting_standard"] = fin_meta.get("accountingStandardCode")
                profile["auditor"] = fin_meta.get("auditor")
                profile["annual_closing_date"] = fin_meta.get("annualClosingDate")

            # Step 6: Dividend history (latest)
            dividends = _fetch_share_detail_list(valor_id, "share/dividend.json")
            if dividends:
                latest = dividends[0]
                profile["latest_dividend_amount"] = latest.get("value")
                profile["latest_dividend_currency"] = latest.get("currency")
                profile["latest_ex_dividend_date"] = latest.get("exDividendDate")
                profile["dividend_history_years"] = len(dividends)

        # Step 7: Enrich with yfinance for sector/industry (SIX doesn't provide these)
        try:
            from operator1.clients.yfinance_backed import yf_get_profile
            yf_profile = yf_get_profile(
                identifier, self.market_id,
                "Switzerland", "CH", "SIX", "CHF",
                yf_suffix=".SW",
            )
            for key in ("sector", "industry", "market_cap", "description"):
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
        """Fetch financials via EU ESEF crossover or SIX filing discovery + LLM.

        SIX APIs do not provide financial statement line items (only metadata
        like auditor and accounting standard). For actual financials:

        Path 1: EU ESEF wrapper (Swiss blue chips like Nestle, Novartis, Roche
        file ESEF XBRL reports). PIT-compliant with filing dates.

        Path 2: SIX filing discovery (via SIXFilingDiscoverer) provides
        corporate action text that can be extracted by LLM.

        yfinance is NOT used for financial statements (no filing dates = no
        PIT compliance).
        """
        # Path 1: EU ESEF wrapper (Swiss blue chips file ESEF)
        try:
            from operator1.clients.eu_esef_wrapper import EUEsefClient
            esef = EUEsefClient()
            if statement_type == "income":
                df = esef.get_income_statement(identifier)
            elif statement_type == "balance":
                df = esef.get_balance_sheet(identifier)
            else:
                df = esef.get_cashflow_statement(identifier)
            if df is not None and not df.empty:
                logger.info("SIX %s %s: %d rows from EU ESEF crossover",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("EU ESEF crossover failed for SIX %s: %s", identifier, exc)

        # Path 2: SIX filing discovery + LLM extraction
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
            df = try_filing_extraction(
                ticker=identifier,
                market_id=self.market_id,
                statement_type=statement_type,
                llm_client=None,
            )
            if df is not None and not df.empty:
                logger.info("SIX %s %s: %d rows from filing discovery",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("SIX filing discovery failed for %s: %s", identifier, exc)

        # No yfinance fallback -- return empty for PIT compliance
        return pd.DataFrame()

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """SIX does not provide sufficient OHLCV data for the 2-year window.

        The SIX historic CSV (/sheldon/market_data/v1/{ValorId}/historic.csv)
        only provides ~5 months of close+volume data (no open/high/low).
        The pipeline requires 2 years of full OHLCV, so this is handled
        by ohlcv_provider.py via yfinance (.SW suffix).
        """
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

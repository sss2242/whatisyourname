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

OHLCV: SIX provides ~5 months of close+volume via
/sheldon/market_data/v1/{ValorId}/historic.csv (no open/high/low).
The pipeline uses this PIT-compliant source directly. For the periods
beyond the ~5 month window, the six_derived_proxies module fills
analytical gaps using dividend history, capital structure, and official
notices data.

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
# Historic market data (close + volume CSV)
# ---------------------------------------------------------------------------

def _fetch_historic_csv(valor_id: str) -> pd.DataFrame:
    """Fetch ~5 months of daily close+volume from SIX historic CSV.

    Returns DataFrame with DatetimeIndex and columns: close, volume.
    This is the only PIT-compliant price source from SIX (exchange data).
    The CSV has no open/high/low -- only close and volume.
    """
    if not valor_id:
        return pd.DataFrame()

    url = f"{_MARKET_DATA_BASE}/{valor_id}/historic.csv"
    try:
        resp = requests.get(url, headers={
            "User-Agent": "Operator1/1.0 (financial-research)",
            "Accept": "text/csv",
        }, timeout=30)
        resp.raise_for_status()

        # SIX CSV format: 2 metadata lines, then "Date;Price;Volume" header.
        # Example:
        #   NESTLE N (Nestlé/CH0038863350/CHF)
        #           19.03.2026
        #           Date;Price;Volume
        #   19.03.2026;77.24;1514740
        lines = resp.text.strip().split("\n")

        # Find the header line (contains "Date" and ";")
        header_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ";" in stripped and ("date" in stripped.lower() or "datum" in stripped.lower()):
                header_idx = i
                break

        csv_text = "\n".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(csv_text), sep=";")

        # Normalize column names: map Price->close, Date->date, Volume->volume
        col_map: dict[str, str] = {}
        for col in df.columns:
            cl = col.strip().lower()
            if cl in ("date", "datum"):
                col_map[col] = "date"
            elif cl in ("price", "close", "schluss", "last", "closing price"):
                col_map[col] = "close"
            elif cl in ("volume", "volumen", "umsatz stk"):
                col_map[col] = "volume"

        if "date" not in col_map.values() or "close" not in col_map.values():
            # Positional fallback: first col = date, second = close
            cols = df.columns.tolist()
            if len(cols) >= 2:
                col_map = {cols[0]: "date", cols[1]: "close"}
                if len(cols) >= 3:
                    col_map[cols[2]] = "volume"

        df = df.rename(columns=col_map)

        if "date" not in df.columns or "close" not in df.columns:
            logger.debug("SIX CSV missing date/close columns: %s", list(df.columns))
            return pd.DataFrame()

        # SIX dates are DD.MM.YYYY format (e.g. "19.03.2026")
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date").sort_index()

        # Clean numeric columns
        for col in ("close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", ".").str.replace("'", ""),
                    errors="coerce",
                )

        # SIX CSV sometimes has close=0 rows -- drop them
        if "close" in df.columns:
            df = df[df["close"] > 0]

        logger.info("SIX historic CSV for %s: %d rows (%s to %s)",
                     valor_id, len(df),
                     df.index[0].date() if len(df) > 0 else "N/A",
                     df.index[-1].date() if len(df) > 0 else "N/A")
        return df

    except Exception as exc:
        logger.debug("SIX historic CSV failed for %s: %s", valor_id, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sector inference from SIX metadata (no yfinance)
# ---------------------------------------------------------------------------

# SMI constituents and their sectors (as of 2025)
_SMI_SECTOR_MAP: dict[str, str] = {
    "NESN": "Consumer Defensive", "NOVN": "Healthcare", "ROG": "Healthcare",
    "UBSG": "Financial Services", "ZURN": "Financial Services",
    "ABBN": "Industrials", "SREN": "Financial Services",
    "LONN": "Healthcare", "CSGN": "Financial Services",
    "HOLN": "Basic Materials", "SIKA": "Basic Materials",
    "GIVN": "Basic Materials", "SCMN": "Communication Services",
    "SLHN": "Financial Services", "PGHN": "Financial Services",
    "GEBN": "Industrials", "LOGN": "Technology",
    "BAER": "Financial Services", "SOON": "Industrials",
    "ALC": "Healthcare",
}


def _infer_sector_from_six(profile: dict[str, Any]) -> None:
    """Infer sector from SIX index membership and known constituent map.

    Falls back to regulatory standard-based classification.
    """
    ticker = profile.get("ticker", "")

    # Direct lookup from known SMI/SLI constituents
    if ticker in _SMI_SECTOR_MAP:
        profile["sector"] = _SMI_SECTOR_MAP[ticker]
        return

    # Infer from regulatory standard
    reg = profile.get("regulatory_standard", "")
    if reg:
        reg_lower = reg.lower()
        if "bank" in reg_lower or "finance" in reg_lower:
            profile["sector"] = "Financial Services"
            return
        if "insurance" in reg_lower:
            profile["sector"] = "Financial Services"
            return
        if "pharma" in reg_lower or "health" in reg_lower:
            profile["sector"] = "Healthcare"
            return

    # Infer from name keywords
    name = (profile.get("name", "") or "").lower()
    if any(kw in name for kw in ("bank", "credit", "finance", "asset")):
        profile["sector"] = "Financial Services"
    elif any(kw in name for kw in ("pharma", "biotech", "health", "medical")):
        profile["sector"] = "Healthcare"
    elif any(kw in name for kw in ("insurance", "versicherung", "reinsurance")):
        profile["sector"] = "Financial Services"
    elif any(kw in name for kw in ("tech", "software", "digital")):
        profile["sector"] = "Technology"
    elif any(kw in name for kw in ("food", "nestle", "lindt", "chocolate")):
        profile["sector"] = "Consumer Defensive"


# ---------------------------------------------------------------------------
# CHSixClient -- PIT client for Swiss SIX equities
# ---------------------------------------------------------------------------


class CHSixClient:
    """PIT client for Swiss SIX equities.

    Uses three undocumented SIX APIs (no auth required):
    1. FQS ref endpoint -- company search, ValorId resolution
    2. Share details -- profile, dividends, capital structure, historic CSV
    3. Official notices -- corporate actions with PIT dates

    No yfinance dependency. Sector/industry enrichment handled by
    supplement.py (OpenFIGI) in main.py, or inferred from SIX index
    membership and regulatory standard.
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
        Returns empty list if FQS is unreachable (graceful degradation).
        """
        if not query:
            # Browse mode: return first page of SIX-listed equities
            return _fqs_search(page_size=20)

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

        logger.info("SIX FQS search for '%s': no results", query)
        return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from SIX APIs (PIT-compliant, no yfinance).

        Data sources (in priority order):
        1. FQS ref -- ValorId, ISIN, ticker, name
        2. Share details -- shares outstanding, dividends, indices, regulatory standard
        3. Capital structure -- share capital breakdown
        4. Official notices -- PIT-dated corporate actions
        5. Historic CSV -- latest close for market_cap computation
        6. Sector inference from index membership + name heuristics
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

        # Step 7: Compute market_cap from SIX data (shares * latest close)
        if valor_id and profile.get("shares_outstanding"):
            try:
                csv_df = _fetch_historic_csv(valor_id)
                if not csv_df.empty and "close" in csv_df.columns:
                    latest_close = float(csv_df["close"].dropna().iloc[-1])
                    profile["market_cap"] = int(float(profile["shares_outstanding"]) * latest_close)
                    profile["latest_close"] = latest_close
                    profile["close_date"] = str(csv_df.index[-1].date())
                    logger.info("SIX market_cap for %s: %s (close=%.2f)", identifier, profile["market_cap"], latest_close)
            except Exception as exc:
                logger.debug("SIX historic CSV failed for market_cap: %s", exc)

        # Step 7b: Sector inference from index membership + regulatory standard
        # (OpenFIGI via supplement.py handles full enrichment in main.py)
        if not profile.get("sector"):
            _infer_sector_from_six(profile)

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
        """Fetch financials via synthetic generation from SIX data.

        SIX APIs do not provide financial statement line items directly.
        Instead, we derive 22 canonical fields mathematically from:
        - 18 years of dividend history (SIX share/dividend.json)
        - Capital structure (SIX issuer/capital_structure.json)
        - Buyback notices (SIX official_notices)
        - Sector-calibrated financial ratios

        The remaining 8 fields (receivables, inventory, payables, goodwill,
        intangibles, sga, rd, short_term_debt) are supplemented from yfinance.

        Path 1: Synthetic financials from SIX data + math (primary)
        Path 2: EU ESEF crossover (fallback, rarely works)
        Path 3: SIX filing discovery + LLM extraction (fallback)
        """
        # Path 1: Synthetic financials from SIX data
        try:
            from operator1.features.six_derived_proxies import generate_synthetic_financials

            # Use cached profile for this identifier
            profile = self._read_cache(identifier, "profile.json")
            if not profile:
                profile = self.get_profile(identifier)

            synthetics = generate_synthetic_financials(profile)
            df = synthetics.get(statement_type, pd.DataFrame())
            if df is not None and not df.empty:
                logger.info("SIX %s %s: %d rows from synthetic financials",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("SIX synthetic financials failed for %s: %s", identifier, exc)

        # Path 2: EU ESEF wrapper (Swiss companies rarely file ESEF)
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

        # Path 3: SIX filing discovery + LLM extraction
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

        return pd.DataFrame()

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """Fetch ~5 months of daily close+volume from SIX historic CSV.

        The SIX historic CSV provides close+volume only (no open/high/low).
        This covers ~5 months -- shorter than the 2-year pipeline window,
        but the six_derived_proxies module fills analytical gaps using
        dividend history, capital structure, and official notices.

        The close+volume data is PIT-compliant (exchange trade data).
        """
        valor_id = ""
        fqs = _fqs_search(ticker=identifier, page_size=1)
        if not fqs and identifier.startswith("CH"):
            fqs = _fqs_search(isin=identifier, page_size=1)
        if fqs:
            valor_id = fqs[0].get("valor_id", "")

        if not valor_id:
            # Try cached profile for valor_id
            cached = self._read_cache(identifier, "profile.json")
            if cached:
                valor_id = cached.get("valor_id", "")

        if not valor_id:
            return pd.DataFrame()

        df = _fetch_historic_csv(valor_id)
        if df.empty:
            return pd.DataFrame()

        # Normalize to pipeline format: columns date, close, volume
        result = pd.DataFrame(index=df.index)
        result.index.name = "date"
        if "close" in df.columns:
            result["close"] = df["close"]
            # SIX CSV has no open/high/low -- set to close for compatibility
            result["open"] = df["close"]
            result["high"] = df["close"]
            result["low"] = df["close"]
        if "volume" in df.columns:
            result["volume"] = df["volume"]
        else:
            result["volume"] = 0

        return result.reset_index()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    # -- Institutional holders (yfinance .SW) --------------------------------

    def _yf_ticker(self, identifier: str) -> str:
        """Convert SIX ticker to yfinance format (e.g. 'NESN' -> 'NESN.SW')."""
        code = identifier.split(".")[0].strip().upper()
        return f"{code}.SW"

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch institutional shareholders from MarketScreener (primary) or yfinance (fallback).

        Primary: MarketScreener/Zonebourse -- scrapes institutional holder data
        with names, shares, and percentages.  Covers major SIX-listed companies.

        Fallback: yfinance (.SW suffix) for companies not found on MarketScreener.

        Returns list of dicts with: name, shares, percentage, value,
        holder_type, date_reported, source.
        """
        # --- Primary: MarketScreener ---
        try:
            from operator1.clients.marketscreener import fetch_shareholders
            # Resolve company name from profile
            company_name = identifier
            cached = self._read_cache(identifier, "profile.json")
            if cached and cached.get("name"):
                company_name = cached["name"]
            holders = fetch_shareholders(company_name)
            if holders:
                logger.info("SIX holders for %s: %d from MarketScreener", identifier, len(holders))
                return holders
        except Exception as exc:
            logger.debug("MarketScreener holder lookup failed for SIX %s: %s", identifier, exc)

        # --- Fallback: yfinance ---
        holders: list[dict[str, Any]] = []
        try:
            import yfinance as yf
            tick = yf.Ticker(self._yf_ticker(identifier))

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
                        "source": "yfinance",
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
                        "source": "yfinance",
                    })

            # Major holders aggregate stats
            major = tick.major_holders
            if major is not None and not major.empty:
                for idx, row in major.iterrows():
                    breakdown = (
                        str(row.get("Breakdown", idx)).lower()
                        if "Breakdown" in major.columns
                        else str(idx).lower()
                    )
                    val = (
                        row.get("Value", row.iloc[-1])
                        if "Value" in major.columns
                        else row.iloc[-1]
                    )
                    try:
                        pct = float(val) * 100 if float(val) < 1 else float(val)
                    except (ValueError, TypeError):
                        continue

                    if "insider" in breakdown:
                        holders.append({
                            "name": "Insiders / Directors",
                            "shares": 0,
                            "percentage": round(pct, 2),
                            "holder_type": "insider_aggregate",
                            "date_reported": "",
                            "source": "yfinance_major",
                        })
                    elif "institution" in breakdown and "percent" in breakdown:
                        holders.append({
                            "name": "Institutional Investors",
                            "shares": 0,
                            "percentage": round(pct, 2),
                            "holder_type": "institutional_aggregate",
                            "date_reported": "",
                            "source": "yfinance_major",
                        })

            if holders:
                logger.info(
                    "SIX holders for %s: %d from yfinance (%s)",
                    identifier, len(holders), self._yf_ticker(identifier),
                )
        except Exception as exc:
            logger.debug("yfinance holders failed for SIX %s: %s", identifier, exc)

        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return institutional ownership metrics as a single-row snapshot.

        Uses yfinance major_holders for aggregate ownership percentages.
        SIX does not expose historical holder data via public APIs.
        """
        try:
            import yfinance as yf
            tick = yf.Ticker(self._yf_ticker(identifier))

            major = tick.major_holders
            if major is None or major.empty:
                return pd.DataFrame()

            inst_pct = 0.0
            holder_count = 0
            for idx, row in major.iterrows():
                breakdown = (
                    str(row.get("Breakdown", idx)).lower()
                    if "Breakdown" in major.columns
                    else str(idx).lower()
                )
                val = (
                    row.get("Value", row.iloc[-1])
                    if "Value" in major.columns
                    else row.iloc[-1]
                )
                try:
                    fval = float(val)
                except (ValueError, TypeError):
                    continue

                if "institution" in breakdown and "percent" in breakdown:
                    inst_pct = fval * 100 if fval < 1 else fval
                elif "institution" in breakdown and "count" in breakdown:
                    holder_count = int(fval)

            return pd.DataFrame([{
                "date_reported": pd.Timestamp.now(),
                "inst_ownership_pct": round(inst_pct, 2),
                "inst_top5_concentration": 0.0,
                "inst_holder_count": holder_count,
            }])
        except Exception as exc:
            logger.debug("yfinance holder history failed for SIX %s: %s", identifier, exc)
            return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch management transactions from the native SIX API.

        Uses the undocumented SIX management_transactions API discovered
        by reverse-engineering the Vue component at:
        ``/etc.clientlibs/ihcc/components/content/management_transactions_teaser/``

        Endpoint: ``/sheldon/management_transactions/v1/overview.json``

        This is the official SIX Exchange Regulation management transaction
        register.  Swiss listed companies must disclose board and management
        share dealings.  The API returns 6,700+ transactions across all
        SIX-listed companies.  We filter client-side by ISIN.

        Fields per transaction:
        - ``transactionDate``: YYYYMMDD format
        - ``transactionSize``: number of shares
        - ``transactionAmountCHF``: total value in CHF
        - ``transactionAmountPerSecurityCHF``: price per share
        - ``buySellIndicator``: 1=buy, 2=sell
        - ``obligorFunctionCode``: 1=board/management
        - ``notificationSubmitter``: company name
        - ``ISIN``: security ISIN

        Returns list of dicts with: insider_name, position, date,
        transaction, shares, value, price_per_share, source.
        """
        transactions: list[dict[str, Any]] = []

        # Resolve ISIN from ticker
        isin = ""
        fqs = _fqs_search(ticker=identifier, page_size=1)
        if fqs:
            isin = fqs[0].get("isin", "")
        if not isin:
            # Try from cached profile
            cached = self._read_cache(identifier, "profile.json")
            if cached:
                isin = cached.get("isin", "")
        if not isin:
            logger.debug("SIX: could not resolve ISIN for %s", identifier)
            return transactions

        # Fetch management transactions (client-side ISIN filter)
        # pageSize=200 covers ~2 weeks of all SIX transactions
        try:
            url = f"{_SIX_BASE}/sheldon/management_transactions/v1/overview.json"
            resp = requests.get(
                url,
                params={"pageSize": 200},
                headers=_SIX_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "Ok":
                return transactions

            all_items = data.get("itemList", [])
        except Exception as exc:
            logger.debug("SIX management transactions API failed: %s", exc)
            return transactions

        # Filter by ISIN
        _FUNCTION_MAP = {
            "1": "Board/Management",
            "2": "Executive Management",
            "3": "Related Party",
        }
        _BUYSELL_MAP = {
            "1": "Purchase",
            "2": "Sale",
            "3": "Subscription",
            "4": "Other",
        }

        for item in all_items:
            if item.get("ISIN") != isin:
                continue

            tx_date_raw = str(item.get("transactionDate", ""))
            tx_date = ""
            if len(tx_date_raw) == 8:
                tx_date = f"{tx_date_raw[:4]}-{tx_date_raw[4:6]}-{tx_date_raw[6:]}"

            bsi = str(item.get("buySellIndicator", ""))
            tx_type = _BUYSELL_MAP.get(bsi, "Unknown")
            func_code = str(item.get("obligorFunctionCode", ""))
            position = _FUNCTION_MAP.get(func_code, "Management")

            try:
                shares = int(item.get("transactionSize", 0))
            except (ValueError, TypeError):
                shares = 0

            value = float(item.get("transactionAmountCHF", 0) or 0)
            price = float(item.get("transactionAmountPerSecurityCHF", 0) or 0)
            submitter = item.get("notificationSubmitter", "")

            transactions.append({
                "insider_name": submitter,
                "position": position,
                "date": tx_date,
                "transaction": tx_type,
                "shares": abs(shares),
                "value": value,
                "price_per_share": price,
                "notification_id": item.get("notificationId", ""),
                "source": "six_management_transactions",
            })

        if transactions:
            logger.info(
                "SIX management transactions for %s (%s): %d from native API",
                identifier, isin, len(transactions),
            )
        return transactions

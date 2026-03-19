"""Switzerland SIX PIT client -- official notices API + yfinance + ESEF crossover.

Primary data source: SIX official notices JSON API (undocumented, no auth).
  - Discovers corporate actions (capital changes, dividends, share counts)
    with PIT-compliant announcement dates.
  - Search by ISIN (valorIds) or issuer name for company discovery.
  - Detail endpoint provides full structured text of each notice.

Profile: yfinance (.SW suffix) + SIX notices enrichment (shares outstanding
from capital action notices, ex-dividend dates).

Financial statements: EU ESEF crossover (Swiss blue chips file IFRS reports
via ESEF) + SIX filing discovery for LLM-based extraction of notice text.

OHLCV: handled separately via ohlcv_provider.py (yfinance .SW suffix).

Coverage: ~250+ listed companies on SIX, ~$1.8T market cap.

API reference (undocumented, reverse-engineered from six-group.com React SPA):
  Search:  GET /sheldon/official_notices/v2/find.json?{params}
  Detail:  GET /sheldon/official_notices/v2/details/{noticeId}.json
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
_CACHE_DIR = Path("cache/ch_six")

# ---------------------------------------------------------------------------
# SIX official notices API constants
# ---------------------------------------------------------------------------

_SIX_API_BASE = "https://www.six-group.com/sheldon/official_notices/v2"
_SIX_FIND_URL = f"{_SIX_API_BASE}/find.json"
_SIX_DETAIL_URL = f"{_SIX_API_BASE}/details"

_SIX_HEADERS = {
    "User-Agent": "Operator1/1.0 (financial-research)",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# SIX notices API helpers
# ---------------------------------------------------------------------------

def _six_search_notices(
    isin: str = "",
    issuer: str = "",
    years: int = 2,
    notice_types: str = "M,EX",
    page_size: int = 50,
) -> list[dict[str, Any]]:
    """Search SIX official notices by ISIN or issuer name.

    Parameters
    ----------
    isin:
        ISIN code for precise search (e.g. 'CH0038863350').
    issuer:
        Issuer name for text search (e.g. 'Nestle AG').
    years:
        How many years back to search.
    notice_types:
        Comma-separated notice types to include: M (manual/issuer),
        EX (ex-dividend), FL (first listing), DE (delisting), A (automatic).
    page_size:
        Max results per page.

    Returns
    -------
    List of notice dicts from the API.
    """
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
        resp = requests.get(_SIX_FIND_URL, params=params, headers=_SIX_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "Ok":
            return data.get("itemList", [])
    except Exception as exc:
        logger.debug("SIX notices search failed: %s", exc)

    return []


def _six_get_notice_text(notice_id: int | str) -> str:
    """Fetch the full text of a SIX official notice."""
    url = f"{_SIX_DETAIL_URL}/{notice_id}.json"
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
    """Extract shares outstanding from a SIX capital action notice.

    SIX capital destruction notices contain lines like:
        'New number of oustanding shares: 2576520000'
    """
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


def _parse_dividend_info(text: str) -> dict[str, Any]:
    """Extract dividend details from a SIX ex-dividend notice."""
    result: dict[str, Any] = {}

    # Ex-date pattern
    ex_match = re.search(
        r"[Ee]x[\s-]*[Dd](?:ividend|ate|atum).*?(\d{1,2}[./]\d{1,2}[./]\d{4})",
        text,
    )
    if ex_match:
        result["ex_date"] = ex_match.group(1)

    # Amount pattern
    amt_match = re.search(
        r"(?:CHF|EUR|USD)\s*([\d.,]+)",
        text,
    )
    if amt_match:
        raw = amt_match.group(1).replace(",", ".")
        try:
            result["amount"] = float(raw)
        except ValueError:
            pass

    return result


# ---------------------------------------------------------------------------
# CHSixClient -- PIT client for Swiss SIX equities
# ---------------------------------------------------------------------------


class CHSixClient:
    """PIT client for Swiss SIX equities.

    Uses the SIX official notices API (undocumented JSON endpoint) for:
      - Company search by ISIN or issuer name
      - Corporate action data (shares outstanding, dividends)

    Falls back to yfinance for profile and EU ESEF for financial statements.
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
        """Search for SIX-listed companies.

        First tries the SIX official notices API to find issuers matching
        the query, then falls back to yfinance search.
        """
        if not query:
            from operator1.clients.yfinance_backed import yf_search
            return yf_search("", self.market_id, "CH", "SIX", yf_suffix=".SW")

        # Try SIX notices API -- search by issuer name across all notice types
        results: list[dict[str, Any]] = []
        try:
            notices = _six_search_notices(
                issuer=query, years=5, notice_types="M,EX,FL", page_size=50,
            )
            # Extract unique issuers from notice contacts
            seen_contacts: set[str] = set()
            for notice in notices:
                contact = notice.get("contact", "")
                isin = notice.get("isin") or ""
                if contact and contact not in seen_contacts and contact != "SIX Swiss Exchange":
                    seen_contacts.add(contact)
                    results.append({
                        "name": contact,
                        "ticker": "",
                        "isin": isin,
                        "country": "CH",
                        "exchange": "SIX",
                        "market_id": "ch_six",
                    })

            if results:
                logger.info("SIX notices search for '%s': %d unique issuers", query, len(results))
                return results
        except Exception as exc:
            logger.debug("SIX notices search failed for '%s': %s", query, exc)

        # Fallback to yfinance
        from operator1.clients.yfinance_backed import yf_search
        return yf_search(query, self.market_id, "CH", "SIX", yf_suffix=".SW")

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from yfinance + SIX notices enrichment.

        The SIX notices API provides:
          - Latest shares outstanding (from capital action notices)
          - Ex-dividend history
          - Corporate action timeline
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        # Base profile from yfinance
        from operator1.clients.yfinance_backed import yf_get_profile
        profile = yf_get_profile(
            identifier, self.market_id,
            "Switzerland", "CH", "SIX", "CHF",
            yf_suffix=".SW",
        )

        # Enrich with SIX notices data
        isin = profile.get("isin", "")
        issuer_name = profile.get("name", identifier)

        try:
            # Search for corporate actions to get shares outstanding.
            # Strategy: try ISIN first (most precise), then issuer name,
            # then simplified name (first word without accents).
            notices = []

            if isin:
                notices = _six_search_notices(
                    isin=isin, years=2, notice_types="M,EX",
                )

            if not notices and issuer_name:
                # Try full issuer name
                notices = _six_search_notices(
                    issuer=issuer_name, years=2, notice_types="M,EX",
                )

            if not notices and issuer_name:
                # Try simplified name (first word, strip accents)
                import unicodedata
                simple_name = unicodedata.normalize("NFD", issuer_name)
                simple_name = "".join(c for c in simple_name if unicodedata.category(c) != "Mn")
                first_word = simple_name.split()[0] if simple_name.split() else ""
                if first_word and first_word.lower() != identifier.lower():
                    raw_notices = _six_search_notices(
                        issuer=first_word, years=2, notice_types="M,EX",
                    )
                    # Filter to only notices from the actual company (not
                    # structured product issuers like Bank Vontobel)
                    if raw_notices:
                        notices = [
                            n for n in raw_notices
                            if first_word.lower() in (n.get("contact", "")).lower()
                        ]
                        # If filtering removed everything, try extracting
                        # ISIN from the detail text of matching notices
                        if not notices:
                            for n in raw_notices:
                                if first_word.lower() in (n.get("contact", "")).lower():
                                    notices.append(n)
                                    break

            # If we found notices but still no ISIN, extract it from:
            # 1. The search result's isin field
            # 2. The notice detail text (ISIN Code: CHxxxxx)
            # Then re-search with ISIN for more precise results.
            if notices and not isin:
                for notice in notices:
                    # Check search result's isin field
                    notice_isin = notice.get("isin")
                    if notice_isin and notice_isin.startswith("CH"):
                        isin = notice_isin
                        break

                # If still no ISIN, try extracting from notice detail text
                if not isin:
                    for notice in notices[:3]:  # Check up to 3 notices
                        nid = notice.get("noticeId")
                        if nid:
                            text = _six_get_notice_text(nid)
                            isin_match = re.search(r"ISIN[^:]*:\s*(CH\d{10,})", text)
                            if isin_match:
                                isin = isin_match.group(1)
                                break

                if isin:
                    profile["isin"] = isin
                    # Re-search with ISIN for more complete results
                    isin_notices = _six_search_notices(
                        isin=isin, years=2, notice_types="M,EX",
                    )
                    if isin_notices:
                        notices = isin_notices

            if notices:
                profile["six_notices_found"] = len(notices)

                # Extract shares outstanding from capital action notices
                for notice in notices:
                    if notice.get("noticeType") == "M":
                        notice_id = notice.get("noticeId")
                        if notice_id:
                            text = _six_get_notice_text(notice_id)
                            if text:
                                shares = _parse_shares_outstanding(text)
                                if shares:
                                    profile["shares_outstanding"] = shares
                                    profile["shares_outstanding_source"] = "SIX official notice"
                                    profile["shares_outstanding_date"] = str(notice.get("date", ""))
                                    logger.info(
                                        "SIX shares outstanding for %s: %d (from notice %s)",
                                        identifier, shares, notice_id,
                                    )
                                    break  # Use the most recent

                logger.info(
                    "SIX profile enrichment for %s: %d notices found",
                    identifier, len(notices),
                )
        except Exception as exc:
            logger.debug("SIX profile enrichment failed for %s: %s", identifier, exc)

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
        """SIX does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

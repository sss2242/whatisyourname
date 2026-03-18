"""South Africa JSE PIT client -- native JSE WCF API + filing discovery.

Primary: JSE Client Portal WCF Services (https://clientportal.jse.co.za)
  - Company listing via CustomerRoleService.svc/GetAllIssuers (298+ equity issuers)
  - SENS announcements via SENSService.svc/GetSensAnnouncementForDates
  - PDF download from senspdf.jse.co.za/documents/SENS_*.pdf
  - No authentication required (public SharePoint WCF endpoints)

Fallback: yfinance (.JO suffix) for profile enrichment and OHLCV

OHLCV: handled separately via ohlcv_provider.py (yfinance .JO)

Coverage: ~350+ listed companies on JSE, ~$1T market cap.

Key discovery: The JSE client portal (SharePoint-based) exposes WCF
REST services that return JSON. The SensAnnouncements.js webpart
revealed the endpoint URLs. This is analogous to HKEX's JSESSIONID
pattern -- session-less WCF services that return structured data.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("cache/za_jse")

# ---------------------------------------------------------------------------
# JSE API constants
# ---------------------------------------------------------------------------

_JSE_PORTAL_BASE = "https://clientportal.jse.co.za"
_JSE_ISSUERS_URL = f"{_JSE_PORTAL_BASE}/_vti_bin/JSE/CustomerRoleService.svc/GetAllIssuers"
_JSE_ISSUERS_NOFILTER_URL = f"{_JSE_PORTAL_BASE}/_vti_bin/JSE/CustomerRoleService.svc/GetAllIssuersNoFilter"

_JSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
}

# Cache the issuer list (refreshed once per session)
_issuer_cache: list[dict] | None = None
_issuer_cache_time: float = 0.0
_ISSUER_CACHE_TTL = 3600  # 1 hour


def _get_all_issuers() -> list[dict]:
    """Fetch the full JSE equity issuer directory.

    Uses the JSE Client Portal WCF service. Returns a list of dicts
    with AlphaCode (ticker), LongName, MasterID, ExchangeCode, etc.
    Results are cached for 1 hour.
    """
    global _issuer_cache, _issuer_cache_time

    now = time.time()
    if _issuer_cache is not None and (now - _issuer_cache_time) < _ISSUER_CACHE_TTL:
        return _issuer_cache

    try:
        resp = requests.post(
            _JSE_ISSUERS_URL,
            json={"filterLongName": "", "filterType": "Equity Issuer"},
            headers=_JSE_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            _issuer_cache = data
            _issuer_cache_time = now
            logger.info("JSE issuer directory: %d equity issuers loaded", len(data))
            return data
    except Exception as exc:
        logger.warning("JSE issuer directory fetch failed: %s", exc)

    return _issuer_cache or []


def _search_issuers(query: str) -> list[dict[str, Any]]:
    """Search the JSE issuer directory by ticker or name.

    Performs client-side fuzzy matching against the cached issuer list.
    """
    issuers = _get_all_issuers()
    if not issuers:
        return []

    query_upper = query.strip().upper()
    query_lower = query.strip().lower()

    results: list[dict[str, Any]] = []
    for issuer in issuers:
        alpha = issuer.get("AlphaCode", "")
        customer_alpha = issuer.get("CustomerAlphaCode", "")
        long_name = issuer.get("LongName", "")

        # Exact ticker match (highest priority)
        if alpha.upper() == query_upper or customer_alpha.upper() == query_upper:
            results.insert(0, _parse_issuer(issuer))
            continue

        # Name contains match
        if query_lower in long_name.lower():
            results.append(_parse_issuer(issuer))
            continue

        # Partial ticker match
        if query_upper in alpha.upper():
            results.append(_parse_issuer(issuer))

    return results[:25]  # Cap results


def _parse_issuer(issuer: dict) -> dict[str, Any]:
    """Convert a JSE issuer dict to our standard company dict format."""
    return {
        "ticker": issuer.get("AlphaCode", ""),
        "name": issuer.get("LongName", ""),
        "country": "ZA",
        "exchange": issuer.get("ExchangeCode", "JSE"),
        "exchange_name": issuer.get("ExchangeName", "JSE Limited"),
        "status": issuer.get("Status", ""),
        "master_id": issuer.get("MasterID", ""),
        "registration_number": issuer.get("RegistrationNumber", ""),
        "email": issuer.get("EmailAddress", ""),
        "website": issuer.get("Website", ""),
        "phone": issuer.get("TelephoneNumber", ""),
        "role": issuer.get("RoleDescription", ""),
        "market_id": "za_jse",
    }


class ZAJseClient:
    """PIT client for South African JSE equities.

    Uses the JSE Client Portal WCF services for company search and
    listing, with yfinance fallback for sector/industry enrichment.

    Company search works natively via the JSE issuer directory --
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
        return "za_jse"

    @property
    def market_name(self) -> str:
        return "South Africa (JSE)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Search for companies on JSE.

        Uses the native JSE issuer directory (WCF service, no key needed).
        Falls back to yfinance only if JSE portal is unreachable.
        """
        # Try JSE native search first
        try:
            if query:
                results = _search_issuers(query)
            else:
                # No query: return all equity issuers
                issuers = _get_all_issuers()
                results = [_parse_issuer(i) for i in issuers[:50]]

            if results:
                logger.info("JSE native search for '%s': %d results", query, len(results))
                return results
        except Exception as exc:
            logger.debug("JSE native search failed for '%s': %s", query, exc)

        # Fallback to yfinance
        from operator1.clients.yfinance_backed import yf_search
        return yf_search(query, self.market_id, "ZA", "JSE", yf_suffix=".JO")

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from JSE + yfinance enrichment.

        Uses the JSE issuer directory for core data, then enriches
        with yfinance for sector/industry/market_cap.
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        profile: dict[str, Any] = {
            "ticker": identifier,
            "name": identifier,
            "country": "ZA",
            "exchange": "JSE",
            "currency": "ZAR",
            "market_id": "za_jse",
        }

        # Try JSE native directory for core data
        master_id = None
        try:
            matches = _search_issuers(identifier)
            if matches:
                best = matches[0]
                master_id = best.get("master_id")
                profile.update({
                    "ticker": best.get("ticker", identifier),
                    "name": best.get("name", identifier),
                    "master_id": master_id or "",
                    "registration_number": best.get("registration_number", ""),
                    "email": best.get("email", ""),
                    "website": best.get("website", ""),
                    "status": best.get("status", ""),
                })
                logger.info("JSE profile for %s: %s (MasterID=%s)",
                           identifier, best.get("name", "?"), master_id)
        except Exception as exc:
            logger.debug("JSE native profile failed for %s: %s", identifier, exc)

        # Enrich with JSE instruments API (ISIN, sector, industry, market cap, price)
        if master_id:
            try:
                resp = requests.post(
                    f"{_JSE_PORTAL_BASE}/_vti_bin/JSE/SharesService.svc/GetAllInstrumentsForIssuer",
                    json={"issuerMasterId": master_id},
                    headers=_JSE_HEADERS,
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                instruments = data.get("GetAllInstrumentsForIssuerResult", [])
                if instruments:
                    # Pick the first current equity instrument
                    inst = instruments[0]
                    for key, src_key in [
                        ("isin", "ISIN"),
                        ("sector", "Sector"),
                        ("industry", "Industry"),
                        ("instrument_type", "InstrumentType"),
                        ("board", "Board"),
                        ("short_name", "ShortName"),
                    ]:
                        val = inst.get(src_key)
                        if val and not profile.get(key):
                            profile[key] = val
                    mc = inst.get("MarketCapitalisation")
                    if mc and not profile.get("market_cap"):
                        profile["market_cap"] = mc
                    price = inst.get("Price")
                    if price:
                        profile["latest_price"] = price
                    listing_date = inst.get("ListingDate", "")
                    if listing_date and "/Date(" in str(listing_date):
                        import re as _re
                        ts_match = _re.search(r"/Date\((\d+)", str(listing_date))
                        if ts_match:
                            from datetime import datetime as _dt, timezone as _tz
                            dt = _dt.fromtimestamp(int(ts_match.group(1)) / 1000, tz=_tz.utc)
                            profile["listing_date"] = dt.strftime("%Y-%m-%d")
                    logger.info(
                        "JSE instruments for %s: ISIN=%s, sector=%s, market_cap=%s",
                        identifier, inst.get("ISIN", "?"),
                        inst.get("Sector", "?"),
                        inst.get("MarketCapitalisation", "?"),
                    )
            except Exception as exc:
                logger.debug("JSE instruments API failed for %s: %s", identifier, exc)

        # Enrich with yfinance for sector/industry
        try:
            from operator1.clients.yfinance_backed import yf_get_profile
            yf_profile = yf_get_profile(
                identifier, self.market_id,
                "South Africa", "ZA", "JSE", "ZAR",
                yf_suffix=".JO",
            )
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
        """Fetch financials via JSE SENS filing discovery only (PIT-compliant).

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
                logger.info("JSE %s %s: %d rows from filing discovery",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("JSE filing discovery failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """JSE does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

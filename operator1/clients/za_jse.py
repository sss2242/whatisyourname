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


def _parse_wcf_date(raw: str) -> str:
    """Parse a WCF /Date(timestamp+offset)/ string to ISO date."""
    if not raw or "/Date(" not in str(raw):
        return ""
    import re as _re
    ts_match = _re.search(r"/Date\((\d+)", str(raw))
    if ts_match:
        dt = datetime.fromtimestamp(int(ts_match.group(1)) / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    return ""


def _extract_pdf_text(pdf_bytes: bytes, max_pages: int = 4) -> str:
    """Extract text from a PDF for classification (shareholding vs dealing).

    Uses pdfplumber for quick text extraction.  For actual shareholder
    table parsing, use ``fuzzy_pdf_parser.extract_shareholders_from_pdf``
    instead -- it handles page scoring, table extraction, and column
    identification automatically.
    """
    try:
        import pdfplumber
        import io
        full_text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:max_pages]:
                text = page.extract_text() or ""
                full_text += text + "\n"
        return full_text
    except ImportError:
        logger.debug("pdfplumber not installed -- cannot parse PDFs")
        return ""
    except Exception as exc:
        logger.debug("PDF text extraction failed: %s", exc)
        return ""


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

    # -- Institutional holders (yfinance + SENS director dealings) -----------

    def _yf_ticker(self, identifier: str) -> str:
        """Convert JSE ticker to yfinance format (e.g. 'NPN' -> 'NPN.JO')."""
        code = identifier.split(".")[0].strip().upper()
        return f"{code}.JO"

    def _resolve_master_id(self, identifier: str) -> int | None:
        """Resolve a ticker to JSE MasterID via the issuer directory."""
        matches = _search_issuers(identifier)
        if matches:
            return matches[0].get("master_id")
        return None

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch holders from JSE SENS shareholding announcements + PDF extraction.

        Three-stage approach (no yfinance):

        1. **SENS headline parsing**: Filter the 15 most recent SENS
           announcements for shareholding/director dealing keywords.
           Extract holder name and percentage from headlines.  Download
           and parse SENS PDFs for structured data.

        2. **SENS PDF deep parse**: For shareholding-related SENS PDFs,
           extract structured holder data (name, shares, percentage)
           using pdfplumber + regex.

        3. **Filing discovery PDF fallback**: Use the annual report filing
           discovery path to find and parse shareholder pages from annual
           reports (two-stage: page scoring + table extraction).

        Probing confirmed (2026-03-26):
          - GetSensAnnouncementsByIssuerMasterId returns max 15 recent items
          - API field is ``FlashHeadline`` (not ``Headline``)
          - API date field is ``AcknowledgeDateTime`` (not ``DateTimePublished``)
          - No dedicated shareholder WCF endpoints exist (all return 404)
          - SENS announcements + filing PDFs are the ONLY native holder sources

        No yfinance dependency.
        """
        holders: list[dict[str, Any]] = []
        import re as _re

        # Get SENS announcements and filter for shareholding disclosures
        master_id = self._resolve_master_id(identifier)
        if not master_id:
            logger.debug("JSE: could not resolve MasterID for %s", identifier)
            # Skip to filing discovery fallback
            return self._holders_from_filing_pdfs(identifier)

        try:
            resp = requests.post(
                f"{_JSE_PORTAL_BASE}/_vti_bin/JSE/SENSService.svc/"
                "GetSensAnnouncementsByIssuerMasterId",
                json={"issuerMasterId": master_id},
                headers=_JSE_HEADERS,
                timeout=15,
            )
            if resp.status_code != 200:
                logger.debug("JSE SENS returned %d for MasterID %s", resp.status_code, master_id)
                return self._holders_from_filing_pdfs(identifier)

            data = resp.json()
            announcements = data.get("GetSensAnnouncementsByIssuerMasterIdResult", [])
            if not isinstance(announcements, list):
                return self._holders_from_filing_pdfs(identifier)

            # Filter for shareholding/holder related announcements.
            # BUG FIX: API returns FlashHeadline, not Headline.
            holder_keywords = (
                "shareholder", "beneficial", "section 122", "section 56",
                "interest in", "holding", "acquisition of",
                "disposal of", "stake",
            )
            dealing_keywords = (
                "dealing in securities", "dealings in securities",
                "director dealing", "directors dealing",
            )

            seen_holders: dict[str, dict] = {}  # deduplicate by name
            holder_pdfs: list[str] = []

            for ann in announcements:
                # BUG FIX: use FlashHeadline (the actual API field)
                headline = str(ann.get("FlashHeadline", "")).strip()
                headline_lower = headline.lower()

                is_holder = any(kw in headline_lower for kw in holder_keywords)
                is_dealing = any(kw in headline_lower for kw in dealing_keywords)

                if not is_holder and not is_dealing:
                    continue

                # BUG FIX: use AcknowledgeDateTime (the actual API field)
                date_str = _parse_wcf_date(ann.get("AcknowledgeDateTime", ""))

                pdf_path = ann.get("PDFPath", "")

                if is_holder:
                    # Try to extract shareholder name and percentage from headline
                    name = headline[:80]
                    pct = 0.0
                    pct_match = _re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", headline)
                    if pct_match:
                        pct = float(pct_match.group(1))

                    holder_type = "substantial"
                    if "beneficial" in headline_lower:
                        holder_type = "beneficial"

                    if name not in seen_holders or pct > seen_holders[name].get("percentage", 0):
                        seen_holders[name] = {
                            "name": name,
                            "shares": 0,
                            "value": 0.0,
                            "percentage": round(pct, 2),
                            "holder_type": holder_type,
                            "date_reported": date_str,
                            "source": "jse_sens",
                            "pdf_path": pdf_path,
                        }
                    if pdf_path:
                        holder_pdfs.append(pdf_path)

                elif is_dealing and pdf_path:
                    # Director dealings -- download PDF for structured data
                    holder_pdfs.append(pdf_path)

            holders = list(seen_holders.values())

            # Stage 2: Download and parse SENS PDFs for structured holder data
            _parsed_from_pdfs = self._parse_sens_holder_pdfs(holder_pdfs[:5])
            if _parsed_from_pdfs:
                # Merge PDF-parsed holders (deduplicate by name)
                for ph in _parsed_from_pdfs:
                    ph_name = ph.get("name", "")
                    if ph_name and ph_name not in seen_holders:
                        holders.append(ph)
                        seen_holders[ph_name] = ph

            if holders:
                logger.info(
                    "JSE holders for %s: %d from SENS announcements + PDFs",
                    identifier, len(holders),
                )
        except Exception as exc:
            logger.debug("JSE SENS holder search failed for %s: %s", identifier, exc)

        # Stage 3: Filing discovery PDF fallback (annual report shareholder pages)
        if not holders:
            holders = self._holders_from_filing_pdfs(identifier)

        return holders

    def _parse_sens_holder_pdfs(
        self, pdf_urls: list[str],
    ) -> list[dict[str, Any]]:
        """Download and parse SENS PDFs for structured holder/dealing data.

        Handles two JSE-mandated PDF formats:
        1. Shareholding disclosure (Section 122): name, shares, percentage
        2. Director dealing (para 6.77-6.85): name, shares, value, date
        """
        import re as _re

        holders: list[dict[str, Any]] = []

        for pdf_url in pdf_urls:
            if not pdf_url:
                continue
            try:
                resp = requests.get(
                    pdf_url,
                    headers={"User-Agent": _JSE_HEADERS["User-Agent"]},
                    timeout=15,
                )
                if resp.status_code != 200 or resp.content[:4] != b"%PDF":
                    continue

                text = _extract_pdf_text(resp.content, max_pages=4)
                if not text:
                    continue

                text_lower = text.lower()

                # Check if this is a shareholding disclosure
                if any(kw in text_lower for kw in (
                    "shareholder", "beneficial owner", "section 122",
                    "section 56", "shareholding",
                )):
                    # Use the fuzzy PDF parser's shareholding extractor
                    # (page scoring + table extraction + column identification)
                    try:
                        from operator1.clients.fuzzy_pdf_parser import extract_shareholders_from_pdf
                        parsed_holders = extract_shareholders_from_pdf(
                            resp.content,
                            filing_date="",
                            market_id="za_jse",
                        )
                        for ph in parsed_holders:
                            ph["source"] = "jse_sens_pdf"
                            ph["pdf_path"] = pdf_url
                            holders.append(ph)
                    except ImportError:
                        logger.debug("fuzzy_pdf_parser not available for shareholding extraction")

                # Check if this is a director dealing
                elif any(kw in text_lower for kw in (
                    "dealing in securities", "paragraph 6.77",
                    "paragraph 6.83", "director",
                )):
                    parsed = _parse_director_dealings_pdf(resp.content, "")
                    for tx in parsed:
                        name = tx.get("insider_name", "")
                        if name:
                            holders.append({
                                "name": name,
                                "shares": tx.get("shares", 0),
                                "value": tx.get("value", 0.0),
                                "percentage": 0.0,
                                "holder_type": "director",
                                "date_reported": tx.get("date", ""),
                                "source": "jse_sens_pdf",
                                "pdf_path": pdf_url,
                            })

                time.sleep(0.5)  # rate limit between PDFs
            except Exception as exc:
                logger.debug("JSE SENS PDF parse failed for %s: %s", pdf_url, exc)

        return holders

    def _holders_from_filing_pdfs(self, identifier: str) -> list[dict[str, Any]]:
        """Extract shareholders from annual report PDFs via filing discovery.

        Uses the two-stage PDF extraction pipeline:
        1. Filing discovery finds annual report PDFs
        2. Page-level shareholding extraction isolates shareholder pages
        3. Table extraction + regex parsing extracts holder data

        No yfinance dependency.
        """
        try:
            from operator1.clients.filing_discoverer import try_shareholding_extraction
            holders = try_shareholding_extraction(identifier, market_id=self.market_id)
            if holders:
                logger.info("JSE holders from PDF shareholding extraction: %d", len(holders))
                return holders
        except Exception as exc:
            logger.debug("JSE PDF shareholding fallback failed: %s", exc)
        return []

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return institutional ownership metrics from JSE SENS data.

        Derives aggregate metrics from get_holders() which uses JSE SENS
        shareholding announcements and filing PDF extraction.

        No yfinance dependency.  Returns empty DataFrame when no native
        holder data is available (same as AU ASX).

        Probing confirmed (2026-03-26): No dedicated shareholder WCF
        endpoints exist.  SENS + filing PDFs are the only native sources.
        """
        try:
            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            # Compute aggregate metrics from holder data
            substantial = [h for h in holders if h.get("percentage", 0) > 0]
            directors = [h for h in holders if h.get("holder_type") == "director"]
            inst_pct = sum(h.get("percentage", 0) for h in substantial)

            hhi = 0.0
            top5 = sorted(substantial, key=lambda h: h.get("percentage", 0), reverse=True)[:5]
            total_pct = sum(h.get("percentage", 0) for h in top5)
            if total_pct > 0:
                hhi = sum((h.get("percentage", 0) / total_pct) ** 2 for h in top5)

            # Use the most recent date_reported from holders, or today
            report_dates = [h.get("date_reported", "") for h in holders if h.get("date_reported")]
            report_date = max(report_dates) if report_dates else date.today().isoformat()

            return pd.DataFrame([{
                "date_reported": pd.Timestamp(report_date),
                "inst_ownership_pct": round(inst_pct, 2),
                "inst_top5_concentration": round(hhi, 4),
                "inst_holder_count": len(substantial),
                "director_count": len(directors),
            }])
        except Exception as exc:
            logger.debug("JSE holder history failed for %s: %s", identifier, exc)
            return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch director dealings from JSE SENS announcements.

        JSE Listings Requirements paragraphs 3.63-3.74 and 6.77-6.85
        mandate disclosure of director/prescribed officer share dealings
        via SENS (Stock Exchange News Service).

        This method:
        1. Fetches SENS announcements for the issuer via the WCF API
        2. Filters for "dealing" keywords in headlines
        3. Downloads the SENS PDFs
        4. Parses structured director dealing data using regex

        The SENS PDFs follow a consistent JSE-mandated format with:
        - Director name and designation
        - Transaction date
        - Number of shares
        - Total value (ZAR)
        - Nature of transaction (sale/purchase)
        - Nature of interest (direct/indirect beneficial)
        """
        transactions: list[dict[str, Any]] = []

        master_id = self._resolve_master_id(identifier)
        if not master_id:
            logger.debug("JSE: could not resolve MasterID for %s", identifier)
            return transactions

        # Fetch SENS announcements for this issuer
        try:
            resp = requests.post(
                f"{_JSE_PORTAL_BASE}/_vti_bin/JSE/SENSService.svc/GetSensAnnouncementsByIssuerMasterId",
                json={"issuerMasterId": master_id},
                headers=_JSE_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            result_key = next(
                (k for k in data if "Result" in k), None
            )
            announcements = data.get(result_key, []) if result_key else []
        except Exception as exc:
            logger.debug("JSE SENS fetch failed for %s: %s", identifier, exc)
            return transactions

        # Filter for director dealing announcements
        dealing_keywords = [
            "dealing in securities",
            "dealings in securities",
            "director dealing",
            "directors dealing",
            "prescribed officer",
        ]
        dealing_anns = [
            ann for ann in announcements
            if any(kw in (ann.get("FlashHeadline") or "").lower() for kw in dealing_keywords)
        ]

        if not dealing_anns:
            logger.debug("JSE: no director dealing SENS for %s", identifier)
            return transactions

        logger.info(
            "JSE insider transactions for %s: %d dealing announcements found",
            identifier, len(dealing_anns),
        )

        # Download and parse up to 5 most recent dealing PDFs
        for ann in dealing_anns[:5]:
            pdf_url = ann.get("PDFPath", "")
            if not pdf_url:
                continue

            # Parse announcement date from WCF /Date() format
            import re as _re
            ann_date = ""
            ts_raw = ann.get("AcknowledgeDateTime", "")
            ts_match = _re.search(r"/Date\((\d+)", str(ts_raw))
            if ts_match:
                dt = datetime.fromtimestamp(
                    int(ts_match.group(1)) / 1000, tz=timezone.utc
                )
                ann_date = dt.strftime("%Y-%m-%d")

            try:
                pdf_resp = requests.get(
                    pdf_url,
                    headers={"User-Agent": _JSE_HEADERS["User-Agent"]},
                    timeout=15,
                )
                if pdf_resp.status_code != 200 or pdf_resp.content[:4] != b"%PDF":
                    continue

                parsed = _parse_director_dealings_pdf(pdf_resp.content, ann_date)
                transactions.extend(parsed)
                time.sleep(0.5)  # rate limiting between PDF downloads
            except Exception as exc:
                logger.debug(
                    "JSE dealing PDF parse failed for %s: %s", pdf_url, exc
                )

        if transactions:
            logger.info(
                "JSE insider transactions for %s: %d transactions parsed from %d PDFs",
                identifier, len(transactions), min(len(dealing_anns), 5),
            )
        return transactions


def _parse_director_dealings_pdf(
    pdf_bytes: bytes, announcement_date: str = ""
) -> list[dict[str, Any]]:
    """Parse a JSE SENS director dealings PDF into structured transactions.

    JSE-mandated format includes fields like:
    - Director / Name of associate
    - Nature of transaction (sale/purchase)
    - Transaction date
    - Number of shares / securities
    - Total value
    - Price per share (VWAP or average weighted)

    Two common formats:
    1. Tabular (Sasol style): surname, company, date, shares, value in table rows
    2. Key-value (Naspers/Lighthouse style): labeled lines like "Director: X"
    """
    try:
        import pdfplumber
        import io
    except ImportError:
        logger.debug("pdfplumber not installed -- cannot parse JSE dealing PDFs")
        return []

    transactions: list[dict[str, Any]] = []

    try:
        full_text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:4]:
                text = page.extract_text() or ""
                full_text += text + "\n"
    except Exception as exc:
        logger.debug("PDF text extraction failed: %s", exc)
        return []

    if not full_text.strip():
        return []

    import re as _re

    # Strategy 1: Key-value format (most common for single-transaction PDFs)
    # Handles two JSE sub-formats:
    #   SOL style:  "Director: V D Kahla", "Transaction date: 26 February 2026"
    #   SBK style:  "Prescribed Officer Ms. FZ Montjane", "Date of Transaction 2025-11-18"
    director_patterns = [
        # "Director: Jacobus Petrus Bekker" -- must start with a capital letter name
        _re.compile(r"^Director[:\s]+([A-Z][a-zA-Z\s,.'()-]+?)(?:\n|$)", _re.IGNORECASE | _re.MULTILINE),
        _re.compile(r"Name of director[^:]*[:\s]+([A-Z][a-zA-Z\s,.'()-]+?)(?:\n|$)", _re.IGNORECASE),
        _re.compile(r"Name of associate[:\s]+([A-Z][a-zA-Z\s,.'()-]+?)(?:\n|$)", _re.IGNORECASE),
        # SBK style: "Prescribed Officer Ms. FZ Montjane" or "Prescribed Officer Mr AB Smith"
        _re.compile(r"Prescribed Officer\s+(?:Ms\.?|Mr\.?|Mrs\.?|Dr\.?|Adv\.?)?\s*([A-Z][a-zA-Z\s,.'()-]+?)(?:\n|$)", _re.IGNORECASE),
        # "Surname and initials" column header followed by name line
        _re.compile(r"Surname\s+.*?\n\s*(?:and initials.*?\n\s*)?([A-Z][a-z]+ [A-Z](?:\s[A-Z])?)", _re.IGNORECASE),
    ]
    date_patterns = [
        _re.compile(r"Transaction date[:\s]+(\d{1,2}\s+\w+\s+\d{4})", _re.IGNORECASE),
        _re.compile(r"Transaction date[:\s]+(\d{4}-\d{2}-\d{2})", _re.IGNORECASE),
        # SBK style: "Date of Transaction 2025-11-18" (no colon)
        _re.compile(r"Date of Transaction\s+(\d{4}-\d{2}-\d{2})", _re.IGNORECASE),
        _re.compile(r"Date of Transaction\s+(\d{1,2}\s+\w+\s+\d{4})", _re.IGNORECASE),
    ]
    shares_patterns = [
        _re.compile(r"Number of (?:shares|securities)[:\s]+([\d,\s]+)", _re.IGNORECASE),
        # SBK style: "sale of 7,000 Standard Bank" -> extract number
        _re.compile(r"(?:sale|purchase|acquisition|disposal)\s+of\s+([\d,\s]+)\s+\w+", _re.IGNORECASE),
    ]
    value_patterns = [
        _re.compile(r"Total value[:\s]+[rR]?\s?([\d,.\s]+)", _re.IGNORECASE),
        _re.compile(r"Total value of (?:the )?transaction[:\s]+[rR]?\s?([\d,.\s]+)", _re.IGNORECASE),
        # SBK: "Total Value of Transaction R1,913,595.60"
        _re.compile(r"Total Value of Transaction\s+[rR]?\s?([\d,.\s]+)", _re.IGNORECASE),
    ]
    nature_patterns = [
        _re.compile(r"Nature of Transaction\s+(.+?)(?:\n|$)", _re.IGNORECASE),
        _re.compile(r"Nature of transaction[:\s]+(.+?)(?:\n|$)", _re.IGNORECASE),
    ]
    interest_patterns = [
        _re.compile(r"Nature (?:and extent )?of (?:director.s )?interest[:\s]+(.+?)(?:\n|$)", _re.IGNORECASE),
        # SBK: "Nature of Interest Direct Beneficial"
        _re.compile(r"Nature of Interest\s+(.+?)(?:\n|$)", _re.IGNORECASE),
    ]
    price_patterns = [
        _re.compile(r"(?:Average weighted |Volume weighted average\s+)price per share[:\s]+[rR]?\s?([\d,.\s]+)", _re.IGNORECASE),
        _re.compile(r"Price per (?:share|security)[:\s]+[rR]?\s?([\d,.\s]+)", _re.IGNORECASE),
        # SBK: "Price per share R273.3708" (no colon)
        _re.compile(r"Price per share\s+[rR]?\s?([\d,.\s]+)", _re.IGNORECASE),
    ]

    def _extract_first(patterns: list, text: str) -> str:
        for pat in patterns:
            m = pat.search(text)
            if m:
                return m.group(1).strip()
        return ""

    def _parse_number(s: str) -> float:
        """Parse a number from JSE format (commas, spaces, R prefix)."""
        if not s:
            return 0.0
        cleaned = s.replace(",", "").replace(" ", "").replace("R", "").strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0

    def _parse_date(s: str) -> str:
        """Parse a date string to ISO format."""
        if not s:
            return ""
        months = {
            "january": "01", "february": "02", "march": "03", "april": "04",
            "may": "05", "june": "06", "july": "07", "august": "08",
            "september": "09", "october": "10", "november": "11", "december": "12",
        }
        # "26 February 2026" format
        m = _re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", s.strip())
        if m:
            day, month_name, year = m.groups()
            month_num = months.get(month_name.lower(), "")
            if month_num:
                return f"{year}-{month_num}-{int(day):02d}"
        # Already ISO
        if _re.match(r"\d{4}-\d{2}-\d{2}", s):
            return s[:10]
        return s

    # Check for multi-transaction blocks (Naspers style: repeated "Transaction date" lines)
    # Handles both "Transaction date: 26 February 2026" (SOL) and
    # "Date of Transaction 2025-11-18" (SBK) formats.
    tx_date_matches = list(_re.finditer(
        r"(?:Transaction date|Date of Transaction)[:\s]+(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2})",
        full_text, _re.IGNORECASE,
    ))

    director_name = _extract_first(director_patterns, full_text)
    nature = _extract_first(nature_patterns, full_text)
    interest_type = _extract_first(interest_patterns, full_text)

    # Classify transaction type
    transaction_type = "Unknown"
    nature_lower = nature.lower()
    if "sale" in nature_lower or "disposal" in nature_lower or "sold" in nature_lower:
        transaction_type = "Sale"
    elif "purchase" in nature_lower or "acquisition" in nature_lower or "bought" in nature_lower:
        transaction_type = "Purchase"
    elif "vesting" in nature_lower or "award" in nature_lower:
        transaction_type = "Vesting"

    if len(tx_date_matches) > 1:
        # Multi-transaction PDF: split text into blocks per transaction date
        for i, match in enumerate(tx_date_matches):
            start = match.start()
            end = tx_date_matches[i + 1].start() if i + 1 < len(tx_date_matches) else len(full_text)
            block = full_text[start:end]

            tx_date = _parse_date(match.group(1))
            shares = _parse_number(_extract_first(shares_patterns, block))
            value = _parse_number(_extract_first(value_patterns, block))
            price = _parse_number(_extract_first(price_patterns, block))

            if shares > 0 or value > 0:
                transactions.append({
                    "insider_name": director_name,
                    "position": "Director",
                    "date": tx_date or announcement_date,
                    "transaction": transaction_type,
                    "shares": int(shares),
                    "value": value,
                    "price_per_share": price,
                    "interest_type": interest_type,
                    "source": "jse_sens",
                })
    elif tx_date_matches:
        # Single transaction
        tx_date = _parse_date(tx_date_matches[0].group(1))
        shares = _parse_number(_extract_first(shares_patterns, full_text))
        value = _parse_number(_extract_first(value_patterns, full_text))
        price = _parse_number(_extract_first(price_patterns, full_text))

        if shares > 0 or value > 0:
            transactions.append({
                "insider_name": director_name,
                "position": "Director",
                "date": tx_date or announcement_date,
                "transaction": transaction_type,
                "shares": int(shares),
                "value": value,
                "price_per_share": price,
                "interest_type": interest_type,
                "source": "jse_sens",
            })

    # Extract price for use in both strategies
    price = _parse_number(_extract_first(price_patterns, full_text))

    # Strategy 2: Tabular format (Sasol style)
    # Format: "V D Kahla Sasol Limited: Director 26 February 2026 7 000 924 677,89"
    # Name format: initials before surname (e.g. "V D Kahla") or surname first
    # Numbers use spaces as thousands separators and commas as decimal separators
    if not transactions:
        # Split into lines and look for lines containing a date pattern
        for line in full_text.split("\n"):
            # Must contain a date like "26 February 2026"
            date_match = _re.search(r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})", line, _re.IGNORECASE)
            if not date_match:
                continue

            tx_date = _parse_date(date_match.group(1))

            # Extract name: initials + surname
            # "V D Kahla Sasol Limited: Director ..." -> "V D Kahla"
            # Pattern: optional single-letter initials + one capitalized surname
            name_match = _re.match(
                r"((?:[A-Z]\s+)+[A-Z][a-z]+)",
                line.strip(),
            )
            if not name_match:
                # Fallback: try surname + initials (e.g. "Kahla V D")
                name_match = _re.match(
                    r"([A-Z][a-z]+\s+[A-Z](?:\s+[A-Z])?)",
                    line.strip(),
                )
            name = name_match.group(1).strip() if name_match else ""

            # Extract numbers after the date: "7 000 924 677,89"
            # SA format: spaces as thousands sep, comma as decimal sep
            # Strategy: parse the total value using the comma as decimal
            # anchor, then compute shares = round(value / price_per_share)
            after_date = line[date_match.end():].strip()
            shares = 0
            value = 0.0

            # SA format uses spaces as thousands separators and comma as
            # decimal separator.  "7 000 924 677,89" means two numbers:
            # shares=7,000 and value=924,677.89 but they merge into one
            # string with no reliable delimiter.
            #
            # Strategy: parse the full blob as one number, then use the
            # price per share (from another line) to find the correct split.
            # value / price should give a round share count.
            val_match = _re.search(
                r"((?:\d[\d\s]*\d|\d)),(\d{2})\s*$", after_date
            )
            if val_match:
                full_int = val_match.group(1).replace(" ", "")
                full_dec = val_match.group(2)
                try:
                    full_number = float(f"{full_int}.{full_dec}")
                except ValueError:
                    full_number = 0.0

                if price > 0 and full_number > 0:
                    # Try splitting: the true value = shares * price
                    # Check if full_number itself is the value (shares before it)
                    candidate_shares = round(full_number / price)
                    # Verify: candidate_shares * price should be close to full_number
                    if candidate_shares > 0 and abs(candidate_shares * price - full_number) / full_number < 0.01:
                        # full_number is the value, but shares got eaten into it
                        # The digit groups before the decimal: try peeling off leading groups as shares
                        digit_groups = _re.findall(r"\d+", val_match.group(1))
                        # Try: first N groups = shares, rest = value
                        for split_at in range(1, len(digit_groups)):
                            try_shares_str = "".join(digit_groups[:split_at])
                            try_val_str = "".join(digit_groups[split_at:]) + "." + full_dec
                            try:
                                try_shares = int(try_shares_str)
                                try_val = float(try_val_str)
                            except ValueError:
                                continue
                            if try_shares > 0 and try_val > 0 and price > 0:
                                ratio = try_val / try_shares
                                if abs(ratio - price) / price < 0.05:
                                    shares = try_shares
                                    value = try_val
                                    break
                        if shares == 0:
                            # Fallback: full number is value, compute shares from price
                            value = full_number
                            shares = round(value / price)
                    else:
                        value = full_number
                        shares = round(value / price) if price > 0 else 0
                else:
                    value = full_number

            # Also try to get price from subsequent lines
            price = _parse_number(_extract_first(price_patterns, full_text))

            if (shares > 0 or value > 0) and name:
                transactions.append({
                    "insider_name": name,
                    "position": "Director",
                    "date": tx_date or announcement_date,
                    "transaction": transaction_type,
                    "shares": shares,
                    "value": value,
                    "price_per_share": price,
                    "interest_type": interest_type,
                    "source": "jse_sens",
                })

    return transactions

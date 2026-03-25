"""Australia ASX PIT client -- uses ASX MarkitDigital API + filing discovery.

Primary profile: ASX MarkitDigital API (asx.api.markitdigital.com)
Primary financials: Filing discovery + LLM extraction from ASX PDFs
OHLCV: Handled by ohlcv_provider.py (yfinance .AX suffix)

Coverage: ~2,200+ listed companies, ~$1.8T market cap.

NOTE: The old ASX API (asx.com.au/asx/1/) returns 404 as of 2026.
The MarkitDigital API is the current ASX data provider.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ASX MarkitDigital API (same API used by the ASX website and filing discoverer)
_MARKIT_BASE = "https://asx.api.markitdigital.com/asx-research/1.0"
_MARKIT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Operator1/1.0",
}

_CACHE_DIR = Path("cache/au_asx")


class AUAsxClient:
    """PIT client for Australian ASX equities.

    Uses the ASX MarkitDigital API for profile and company search.
    Uses filing discovery + LLM extraction for financial statements.
    No yfinance dependency -- all data comes from ASX-native sources.
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
        return "au_asx"

    @property
    def market_name(self) -> str:
        return "Australia (ASX)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Search companies via ASX MarkitDigital directory API.

        The directory endpoint returns paginated results with up to 1,841+
        companies.  When a query is provided, we fetch the full list and
        filter client-side (the API has no search parameter).
        """
        try:
            # Fetch all companies (API supports up to ~2000 per page)
            url = f"{_MARKIT_BASE}/companies/directory"
            params = {"page": 0, "itemsPerPage": 2500}
            resp = requests.get(url, headers=_MARKIT_HEADERS, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            # Structure: {"data": {"items": [...], "count": N}}
            items = payload.get("data", {}).get("items", [])
            companies = [
                {
                    "ticker": i.get("symbol", ""),
                    "name": i.get("displayName", ""),
                    "cik": i.get("symbol", ""),
                    "sector": i.get("industry", ""),
                    "exchange": "ASX",
                    "country": "AU",
                    "market_id": self.market_id,
                }
                for i in items
            ]
            if query:
                q = query.lower()
                companies = [
                    c for c in companies
                    if q in c["ticker"].lower() or q in c["name"].lower()
                ]
            return companies
        except Exception as exc:
            logger.debug("ASX company list failed: %s", exc)
            return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from ASX MarkitDigital header API.

        Returns sector, industry, market cap, and listing date directly
        from the exchange -- no yfinance dependency.
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        raw: dict[str, Any] = {
            "name": "",
            "ticker": identifier.upper(),
            "isin": "",
            "country": "AU",
            "sector": "",
            "industry": "",
            "exchange": "ASX",
            "currency": "AUD",
            "cik": identifier,
        }

        # Primary: ASX MarkitDigital header endpoint
        try:
            url = f"{_MARKIT_BASE}/companies/{identifier.upper()}/header"
            resp = requests.get(url, headers=_MARKIT_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            raw["name"] = data.get("displayName", "")
            raw["sector"] = data.get("sector", data.get("industryGroup", ""))
            raw["industry"] = data.get("industryGroup", "")
            if data.get("marketCap"):
                raw["market_cap"] = data["marketCap"]
            if data.get("dateListed"):
                raw["date_listed"] = data["dateListed"]
            logger.info(
                "ASX profile for %s: %s (sector=%s)",
                identifier, raw["name"], raw["sector"],
            )
        except Exception as exc:
            logger.info("ASX MarkitDigital profile failed for %s: %s", identifier, exc)

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financials via ASX filing discovery only (PIT-compliant).

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
                logger.info("ASX %s %s: %d rows from filing discovery",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("ASX filing discovery failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """ASX does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    # -- Institutional holders (ASX filing discovery) -------------------------

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch holders from ASX substantial holder notices via filing discovery.

        ASX requires substantial holder notices (>5% ownership) to be filed
        as announcements.  Uses the ASX MarkitDigital announcements API to
        find these filings, then extracts holder data via LLM/fuzzy PDF
        parsing.

        Returns list of dicts with: name, shares, percentage, holder_type,
        date_reported, source.
        """
        holders: list[dict[str, Any]] = []

        # --- ASX substantial holder notices via announcements API ---
        try:
            import re
            ticker = identifier.upper().strip()
            url = f"{_MARKIT_BASE}/companies/{ticker}/announcements"
            params = {
                "count": "50",
                "market_sensitive": "false",
            }
            resp = requests.get(url, headers=_MARKIT_HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", {}).get("items", [])

                for item in items:
                    headline = str(item.get("headline", "")).lower()
                    # Filter for substantial holder notices
                    if any(kw in headline for kw in (
                        "substantial", "holder", "shareholder", "ceasing",
                        "becoming", "change in", "interest",
                    )):
                        date_str = item.get("document_date", "")
                        title = item.get("headline", "")

                        # Try to extract holder name and percentage from headline
                        name = ""
                        pct = 0.0

                        # Pattern: "Becoming a substantial holder from XYZ"
                        # or "Change in substantial holding for ABC"
                        name_match = re.search(
                            r"(?:from|by|for|of)\s+([A-Z][\w\s&.,\'-]+?)(?:\s*-|\s*$)",
                            title, re.IGNORECASE,
                        )
                        if name_match:
                            name = name_match.group(1).strip()

                        pct_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", title)
                        if pct_match:
                            pct = float(pct_match.group(1))

                        holders.append({
                            "name": name or title[:60],
                            "shares": 0,
                            "value": 0.0,
                            "percentage": round(pct, 2),
                            "holder_type": "substantial",
                            "date_reported": date_str,
                            "source": "asx_announcement",
                        })

                if holders:
                    logger.info(
                        "ASX holders for %s: %d from substantial holder notices",
                        identifier, len(holders),
                    )
        except Exception as exc:
            logger.debug("ASX substantial holder search failed for %s: %s", identifier, exc)

        # --- Fallback: filing discovery + LLM/fuzzy extraction ---
        if not holders:
            try:
                from operator1.clients.filing_discoverer import try_filing_extraction
                df = try_filing_extraction(
                    ticker=identifier,
                    market_id=self.market_id,
                    statement_type="balance",
                    llm_client=None,
                )
                if df is not None and not df.empty:
                    logger.info(
                        "ASX filings available for %s (%d rows); "
                        "holder extraction available via LLM",
                        identifier, len(df),
                    )
            except Exception as exc:
                logger.debug("ASX filing discovery for holders failed: %s", exc)

        # Fallback: extract shareholders from filing PDFs
        if not holders:
            try:
                from operator1.clients.filing_discoverer import try_shareholding_extraction
                holders = try_shareholding_extraction(identifier, market_id=self.market_id)
                if holders:
                    logger.info("ASX holders from PDF shareholding extraction: %d", len(holders))
            except Exception as exc:
                logger.debug("ASX PDF shareholding fallback failed: %s", exc)

        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return institutional ownership metrics from ASX filing data.

        Derives aggregate metrics from the holder data returned by
        get_holders().  Returns a single-row snapshot.
        """
        try:
            from datetime import date as _date

            holders = self.get_holders(identifier)
            if not holders:
                return pd.DataFrame()

            substantial = [h for h in holders if h.get("holder_type") == "substantial"]
            inst_pct = sum(h.get("percentage", 0) for h in substantial)

            hhi = 0.0
            top5 = substantial[:5]
            total_pct = sum(h.get("percentage", 0) for h in top5)
            if total_pct > 0:
                hhi = sum((h.get("percentage", 0) / total_pct) ** 2 for h in top5)

            return pd.DataFrame([{
                "date_reported": pd.Timestamp(_date.today()),
                "inst_ownership_pct": round(inst_pct, 2),
                "inst_top5_concentration": round(hhi, 4),
                "inst_holder_count": len(substantial),
            }])
        except Exception as exc:
            logger.debug("ASX holder history failed for %s: %s", identifier, exc)
            return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch insider/director transactions from ASX announcements.

        ASX requires directors to lodge Appendix 3Y (Change of Director's
        Interest Notice) within 5 business days of any change.  Searches
        the ASX MarkitDigital announcements API for director interest
        change notices.
        """
        transactions: list[dict[str, Any]] = []
        try:
            import re
            ticker = identifier.upper().strip()
            url = f"{_MARKIT_BASE}/companies/{ticker}/announcements"
            params = {"count": "50"}
            resp = requests.get(url, headers=_MARKIT_HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", {}).get("items", [])

                for item in items:
                    headline = str(item.get("headline", "")).lower()
                    # Filter for director interest notices (Appendix 3Y)
                    if any(kw in headline for kw in (
                        "appendix 3y", "director", "interest notice",
                        "change of director", "3y",
                    )):
                        transactions.append({
                            "insider_name": item.get("headline", "")[:60],
                            "position": "Director",
                            "date": item.get("document_date", ""),
                            "transaction": "Director Interest Change",
                            "shares": 0,
                            "value": 0.0,
                            "source": "asx_announcement",
                        })

                if transactions:
                    logger.info(
                        "ASX insider transactions for %s: %d from announcements",
                        identifier, len(transactions),
                    )
        except Exception as exc:
            logger.debug("ASX insider transaction search failed for %s: %s", identifier, exc)

        return transactions

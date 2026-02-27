"""EU ESEF PIT client -- powered by pyesef + filings.xbrl.org.

Covers pan-EU filings: France (Euronext), Germany (Frankfurt/XETRA),
and all other EU member states under the ESEF regulation.

Primary library: pyesef (https://github.com/ggravlingen/pyesef)
  - ESEF XBRL file extraction and parsing using Arelle

Fallback: Direct filings.xbrl.org API (same as original esef.py)

Coverage: ~5,000+ EU-listed companies, $8-9T combined market cap.
API: https://filings.xbrl.org/api (free, no key required)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_XBRL_API_BASE = "https://filings.xbrl.org/api"
_CACHE_DIR = Path("cache/eu_esef")


class EUEsefError(Exception):
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"EU ESEF error on {endpoint}: {detail}")


class EUEsefClient:
    """Point-in-time client for ESEF filings (EU equities) using pyesef.

    Implements the ``PITClient`` protocol. Filters by country code to
    support per-country market entries (EU, FR, DE).

    Parameters
    ----------
    country_code:
        ISO-2 country code to filter filings. "" for all EU.
    market_id:
        Registry market ID (e.g. "eu_esef", "fr_esef", "de_esef").
    cache_dir:
        Local cache directory.
    """

    def __init__(
        self,
        country_code: str = "",
        market_id: str = "eu_esef",
        cache_dir: Path | str = _CACHE_DIR,
    ) -> None:
        self._country_code = country_code.upper()
        self._market_id = market_id
        self._cache_dir = Path(cache_dir)
        self._pyesef_available = False
        self._filing_cache: dict[str, list[dict]] = {}

        try:
            import pyesef
            self._pyesef_available = True
            logger.info("pyesef available for ESEF XBRL parsing")
        except ImportError:
            logger.info("pyesef not available; using filings.xbrl.org API only")

    def _cache_path(self, identifier: str, filename: str) -> Path:
        safe_id = identifier.replace("/", "_").replace("\\", "_").upper()
        return self._cache_dir / safe_id / filename

    def _read_cache(self, identifier: str, filename: str) -> dict | None:
        path = self._cache_path(identifier, filename)
        if not path.exists():
            return None
        try:
            age_days = (date.today() - date.fromtimestamp(path.stat().st_mtime)).days
            if filename == "profile.json" and age_days > 7:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, filename: str, data: dict) -> None:
        path = self._cache_path(identifier, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    def _get_xbrl(self, path: str, params: dict | None = None) -> Any:
        url = f"{_XBRL_API_BASE}{path}"
        headers = {"Accept": "application/json", "User-Agent": "Operator1/1.0"}
        try:
            return cached_get(url, params=params, headers=headers)
        except HTTPError as exc:
            raise EUEsefError(path, str(exc)) from exc

    @property
    def market_id(self) -> str:
        return self._market_id

    @property
    def market_name(self) -> str:
        labels = {
            "eu_esef": "European Union (ESEF -- all EU)",
            "fr_esef": "France (Paris / Euronext -- ESEF)",
            "de_esef": "Germany (Frankfurt / XETRA -- ESEF)",
        }
        return labels.get(self._market_id, f"ESEF ({self._country_code})")

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List EU companies from ESEF filings."""
        filings = self._get_recent_filings()
        seen: dict[str, dict] = {}
        for f in filings:
            # filings.xbrl.org returns JSON:API format:
            # { "type": "filing", "id": "...", "attributes": {...}, "relationships": {...} }
            # Research Log: .roo/research/eu-esef-2026-02-24.md (Section B1-B3)
            attrs = f.get("attributes", f)  # fallback to f itself if flat format
            entity = f.get("entity", {})    # may be embedded or empty

            # Entity info: try embedded entity first, then attributes
            lei = entity.get("lei", "") or attrs.get("lei", "")
            name = entity.get("name", "") or attrs.get("entity_name", "") or attrs.get("filer_name", "")
            country = entity.get("country", "") or attrs.get("country", "")

            if self._country_code and country != self._country_code:
                continue
            key = lei or name
            if key and key not in seen:
                seen[key] = {
                    "ticker": lei[:12] if lei else "",
                    "name": name,
                    "lei": lei,
                    "cik": lei,
                    "country": country,
                    "exchange": "ESEF",
                    "market_id": self.market_id,
                }

        results = list(seen.values())
        if query:
            q = query.lower()
            results = [c for c in results if q in c["name"].lower() or q in c.get("lei", "").lower()]
        return results

    def search_company(self, name: str) -> list[dict[str, Any]]:
        """Search for EU companies -- tries filings first, then entities API."""
        results = self.list_companies(query=name)
        if results:
            return results

        # Fallback: search the /entities endpoint directly
        try:
            data = self._get_xbrl("/entities", params={"filter[name]": name, "page[size]": 10})
            entities = data.get("data", []) if isinstance(data, dict) else []
            for ent in entities:
                attrs = ent.get("attributes", {})
                ent_name = attrs.get("name", "")
                if name.lower() in ent_name.lower():
                    results.append({
                        "ticker": attrs.get("lei", ent.get("id", "")),
                        "name": ent_name,
                        "lei": attrs.get("lei", ""),
                        "cik": attrs.get("lei", str(ent.get("id", ""))),
                        "country": attrs.get("country", "EU"),
                        "exchange": "ESEF",
                        "market_id": self.market_id,
                    })
        except Exception as exc:
            logger.debug("Entity search fallback failed: %s", exc)

        return results

    def _get_recent_filings(self) -> list[dict]:
        """Fetch recent ESEF filings from filings.xbrl.org."""
        cache_key = self._country_code or "ALL"
        if cache_key in self._filing_cache:
            return self._filing_cache[cache_key]

        params: dict[str, Any] = {"page_size": 100}
        if self._country_code:
            params["country"] = self._country_code

        try:
            data = self._get_xbrl("/filings", params=params)
            filings = data.get("data", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            self._filing_cache[cache_key] = filings
            return filings
        except Exception:
            return []

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        # Search filings for this entity
        matches = self.search_company(identifier)
        if not matches:
            raise EUEsefError("get_profile", f"Entity not found: {identifier}")

        m = matches[0]
        raw_profile = {
            "name": m.get("name", ""),
            "ticker": m.get("lei", identifier),
            "isin": "",
            "country": m.get("country", "EU"),
            "sector": "",
            "industry": "",
            "exchange": "ESEF",
            "currency": "EUR",
            "lei": m.get("lei", ""),
            "cik": m.get("lei", ""),
        }

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw_profile, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial data from ESEF XBRL filings.

        Searches filings.xbrl.org for this entity's filings, then
        extracts IFRS concepts from the XBRL data.
        """
        filings = self._get_recent_filings()
        # Match entity filings -- handle both JSON:API and flat formats
        entity_filings = []
        for f in filings:
            attrs = f.get("attributes", f)
            entity = f.get("entity", {})
            f_name = (entity.get("name", "") or attrs.get("entity_name", "") or attrs.get("filer_name", "")).lower()
            f_lei = (entity.get("lei", "") or attrs.get("lei", "")).lower()
            if identifier.lower() in f_name or identifier.lower() in f_lei:
                entity_filings.append(f)

        if not entity_filings:
            return pd.DataFrame()

        rows: list[dict] = []
        for filing in entity_filings[:8]:
            # JSON:API format: fields are inside "attributes"
            # Research: .roo/research/eu-esef-2026-02-24.md (Section B1)
            attrs = filing.get("attributes", filing)
            filing_date = attrs.get("date_added", attrs.get("processed", ""))
            period_end = attrs.get("period_end", "")
            if not period_end:
                period_end = attrs.get("period", {}).get("end_date", "") if isinstance(attrs.get("period"), dict) else ""
            period_start = attrs.get("period_start", "")
            if not period_start:
                period_start = attrs.get("period", {}).get("start_date", "") if isinstance(attrs.get("period"), dict) else ""

            # Extract XBRL facts from the filing
            facts = filing.get("facts", {})
            if not facts and filing.get("id"):
                # Try to fetch detailed facts
                try:
                    detail = self._get_xbrl(f"/filings/{filing['id']}/facts")
                    facts = detail if isinstance(detail, dict) else {}
                except Exception:
                    pass

            for concept_uri, values in facts.items() if isinstance(facts, dict) else []:
                from operator1.clients.canonical_translator import _IFRS_MAP
                canonical = _IFRS_MAP.get(concept_uri)
                if not canonical:
                    continue

                value = values if isinstance(values, (int, float)) else None
                if isinstance(values, dict):
                    value = values.get("value", values.get("amount"))
                if isinstance(values, list) and values:
                    value = values[0].get("value") if isinstance(values[0], dict) else values[0]

                if value is not None:
                    try:
                        rows.append({
                            "concept": canonical,
                            "value": float(value),
                            "filing_date": filing_date,
                            "report_date": period_end,
                            "period_start": period_start,
                            "period_type": "annual",
                        })
                    except (ValueError, TypeError):
                        continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
        if "report_date" in df.columns:
            df = df[df["report_date"] >= cutoff]

        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        all_companies = self.list_companies()
        return [c.get("lei", c.get("name", "")) for c in all_companies if c.get("name", "") != identifier][:10]

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

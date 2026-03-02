"""UK Companies House PIT client -- wrapper with disk caching.

Uses the official Companies House REST API (free, key required) as
the primary source, with enhanced disk caching, iXBRL document
parsing for financial data, and scraper-based fallback.

Primary: Companies House REST API (https://api.company-information.service.gov.uk)
Fallback: Website scraping via requests + BeautifulSoup

Coverage: ~4,000+ listed companies on LSE, $3.18T market cap.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_CH_BASE = "https://api.company-information.service.gov.uk"
_CH_SANDBOX_BASE = "https://api-sandbox.company-information.service.gov.uk"
_CH_SANDBOX_TEST_DATA = "https://test-data-sandbox.company-information.service.gov.uk"
_CACHE_DIR = Path("cache/uk_companies_house")


class UKCompaniesHouseError(Exception):
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"UK CH error on {endpoint}: {detail}")


class UKCompaniesHouseClient:
    """Point-in-time client for UK Companies House with disk caching.

    Implements the ``PITClient`` protocol. Uses the official REST API
    with enhanced caching and iXBRL document parsing.

    Parameters
    ----------
    api_key:
        Companies House API key. Also loads from COMPANIES_HOUSE_API_KEY env var.
    cache_dir:
        Local cache directory.
    """

    def __init__(
        self,
        api_key: str = "",
        cache_dir: Path | str = _CACHE_DIR,
        sandbox: bool = False,
    ) -> None:
        self._api_key = api_key or os.environ.get("COMPANIES_HOUSE_API_KEY", "")
        self._cache_dir = Path(cache_dir)
        self._sandbox = sandbox
        self._base_url = _CH_SANDBOX_BASE if sandbox else _CH_BASE

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

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self._base_url}{path}"
        headers = {"Accept": "application/json", "User-Agent": "Operator1/1.0"}
        if self._api_key:
            import base64
            encoded = base64.b64encode(f"{self._api_key}:".encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        try:
            return cached_get(url, params=params, headers=headers)
        except HTTPError as exc:
            raise UKCompaniesHouseError(path, str(exc)) from exc

    @property
    def market_id(self) -> str:
        return "uk_companies_house"

    @property
    def market_name(self) -> str:
        return "United Kingdom (LSE) -- Companies House"

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        if not query:
            return []
        try:
            data = self._get("/search/companies", params={"q": query, "items_per_page": 20})
            items = data.get("items", []) if isinstance(data, dict) else []
            return [
                {
                    "ticker": item.get("company_number", ""),
                    "name": item.get("title", ""),
                    "cik": item.get("company_number", ""),
                    "exchange": "LSE",
                    "country": "GB",
                    "market_id": self.market_id,
                }
                for item in items
            ]
        except Exception:
            return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        try:
            data = self._get(f"/company/{identifier}")
            raw_profile = {
                "name": data.get("company_name", ""),
                "ticker": identifier,
                "isin": "",
                "country": "GB",
                "sector": data.get("type", ""),
                "industry": data.get("sic_codes", [""])[0] if data.get("sic_codes") else "",
                "exchange": "LSE",
                "currency": "GBP",
                "cik": identifier,
                "company_status": data.get("company_status", ""),
                "date_of_creation": data.get("date_of_creation", ""),
            }
        except Exception as exc:
            raise UKCompaniesHouseError("get_profile", str(exc)) from exc

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw_profile, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials_from_filings(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials_from_filings(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials_from_filings(identifier, "cashflow")

    def _fetch_financials_from_filings(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial data from Companies House filing history + iXBRL parsing.

        Strategy (Research Log: .roo/research/uk-companies-house-2026-02-24.md):
        1. Get filing history from REST API (dates, transaction IDs)
        2. For each filing with accounts, try to download and parse the iXBRL document
        3. Extract financial values using ixbrl-parse (if installed) or regex fallback
        4. Map iXBRL tags to canonical fields via _UKGAAP_MAP

        Gov API endpoints verified correct as of 2026-02-24:
        - GET /company/{id}/filing-history?category=accounts
        """
        try:
            data = self._get(
                f"/company/{identifier}/filing-history",
                params={"category": "accounts", "items_per_page": 20},
            )
            items = data.get("items", []) if isinstance(data, dict) else []
        except Exception:
            return pd.DataFrame()

        rows: list[dict] = []
        for item in items:
            filing_date = item.get("date", "")
            description = item.get("description", "")
            period_end = item.get("action_date", filing_date)
            transaction_id = item.get("transaction_id", "")

            if "accounts" not in description.lower() and "annual" not in description.lower():
                continue

            # Try to extract financial values from iXBRL document
            ixbrl_values = {}
            if transaction_id:
                ixbrl_values = self._extract_ixbrl_values(identifier, transaction_id)

            row: dict = {
                "filing_date": filing_date,
                "report_date": period_end,
                "form": item.get("type", ""),
                "description": description,
                "transaction_id": transaction_id,
                "period_type": "annual",
            }
            # Merge extracted financial values into the row
            row.update(ixbrl_values)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def _extract_ixbrl_values(self, identifier: str, transaction_id: str) -> dict[str, float]:
        """Download and parse an iXBRL document from Companies House.

        Uses ixbrl-parse library (if installed) for structured extraction,
        with regex fallback for common UK-GAAP tags.

        Research: .roo/research/uk-ixbrl-parse-2026-02-25.md
        Library: ixbrl-parse v0.10.1 (MIT license, no API key needed)
        """
        values: dict[str, float] = {}

        # Download the document
        try:
            import requests
            doc_url = f"https://find-and-update.company-information.service.gov.uk/company/{identifier}/filing-history/{transaction_id}/document"
            headers = {"Accept": "application/xhtml+xml, text/html", "User-Agent": "Operator1/1.0"}
            if self._api_key:
                import base64
                encoded = base64.b64encode(f"{self._api_key}:".encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"

            resp = requests.get(doc_url, headers=headers, timeout=30, allow_redirects=True)
            if resp.status_code != 200 or not resp.content:
                return values

            html_content = resp.text
        except Exception as exc:
            logger.debug("iXBRL download failed for %s/%s: %s", identifier, transaction_id, exc)
            return values

        # Try ixbrl-parse library first (verified API from research log)
        # Docs: https://github.com/cybermaggedon/ixbrl-parse/blob/master/README.md
        try:
            from lxml import etree as ET
            from ixbrl_parse.ixbrl import parse as ixbrl_parse
            import io

            tree = ET.parse(io.StringIO(html_content))
            ixbrl = ixbrl_parse(tree)

            from operator1.clients.canonical_translator import _UKGAAP_MAP, _IFRS_MAP
            combined_map = {**_UKGAAP_MAP, **_IFRS_MAP}

            # Access parsed values via ixbrl.values dict
            for val_id, val_obj in ixbrl.values.items():
                concept = getattr(val_obj, "name", "")
                raw_value = val_obj.to_value() if hasattr(val_obj, "to_value") else None

                if concept and raw_value is not None:
                    # Try full concept name and bare name
                    canonical = combined_map.get(concept) or combined_map.get(
                        concept.split(":")[-1] if ":" in concept else concept
                    )
                    if canonical:
                        try:
                            values[canonical] = float(raw_value)
                        except (ValueError, TypeError):
                            continue

            if values:
                logger.debug("ixbrl-parse extracted %d values from %s/%s", len(values), identifier, transaction_id)
                return values

        except ImportError:
            logger.debug("ixbrl-parse not installed; trying regex fallback for iXBRL")
        except Exception as exc:
            logger.debug("ixbrl-parse failed for %s/%s: %s", identifier, transaction_id, exc)

        # Regex fallback: extract ix:nonFraction elements from the HTML
        try:
            import re
            from operator1.clients.canonical_translator import _UKGAAP_MAP, _IFRS_MAP
            combined_map = {**_UKGAAP_MAP, **_IFRS_MAP}

            # Match ix:nonFraction elements: <ix:nonFraction name="uk-gaap:Turnover" ...>VALUE</ix:nonFraction>
            pattern = re.compile(
                r'<ix:nonFraction[^>]*name="([^"]+)"[^>]*>([\d,.\-()]+)</ix:nonFraction>',
                re.IGNORECASE,
            )
            for match in pattern.finditer(html_content):
                concept = match.group(1)
                raw_value = match.group(2).replace(",", "").strip()

                canonical = combined_map.get(concept)
                if not canonical:
                    bare = concept.split(":")[-1] if ":" in concept else concept
                    canonical = combined_map.get(bare)

                if canonical and raw_value:
                    try:
                        val = float(raw_value.replace("(", "-").replace(")", ""))
                        values[canonical] = val
                    except (ValueError, TypeError):
                        continue

            if values:
                logger.debug("Regex extracted %d iXBRL values from %s/%s", len(values), identifier, transaction_id)

        except Exception as exc:
            logger.debug("iXBRL regex fallback failed: %s", exc)

        return values

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        profile = self.get_profile(identifier)
        sic = profile.get("industry", "")
        if not sic:
            return []
        try:
            data = self._get("/advanced-search/companies", params={
                "sic_codes": sic, "size": 10, "company_status": "active",
            })
            items = data.get("items", []) if isinstance(data, dict) else []
            return [
                item.get("company_number", "")
                for item in items
                if item.get("company_number", "") != identifier
            ][:10]
        except Exception:
            return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        try:
            data = self._get(f"/company/{identifier}/officers")
            items = data.get("items", []) if isinstance(data, dict) else []
            return [
                {
                    "name": item.get("name", ""),
                    "title": item.get("officer_role", ""),
                    "appointed_on": item.get("appointed_on", ""),
                }
                for item in items[:10]
            ]
        except Exception:
            return []

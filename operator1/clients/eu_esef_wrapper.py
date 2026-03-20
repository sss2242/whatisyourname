"""EU ESEF PIT client -- filings.xbrl.org XBRL JSON extraction.

Covers pan-EU filings: France (Euronext), Germany (Frankfurt/XETRA),
Netherlands, Spain, Italy, Sweden, and all other EU/EEA/UK member
states under the ESEF regulation.

Data source: filings.xbrl.org JSON:API (free, no key required)
  - /api/filings: 23,900+ filings with XBRL JSON download URLs
  - /api/entities: 7,200+ registered filers with LEI identifiers
  - XBRL JSON (OIM format): Full IFRS financial statements (revenue,
    profit, assets, equity, cash, liabilities, EPS, etc.)

Coverage by country (filing counts):
  NL: 599, ES: 542, IT: 754, SE: 1,415, FR: 1,042, GB: 2,565
  DE: 0 (Germany does not file via ESEF/filings.xbrl.org)

Key API quirks:
  - JSON:API format: use filter[country], page[size], not country=
  - Entity names are NOT in filing records -- must follow
    relationships.entity.links.related to get entity name
  - XBRL JSON download needs Accept: application/json (not vnd.api+json)
  - Facts use XBRL OIM format: {"f-1": {value, dimensions: {concept, period, unit}}}
  - IFRS concepts are prefixed: "ifrs-full:Revenue", "ifrs-full:Assets"

Total coverage: ~4,350+ EU-listed companies, $8-9T combined market cap.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_XBRL_API_BASE = "https://filings.xbrl.org/api"
_XBRL_FILES_BASE = "https://filings.xbrl.org"
_CACHE_DIR = Path("cache/eu_esef")

_HEADERS_API = {
    "User-Agent": "Operator1/1.0 (financial-research)",
    "Accept": "application/vnd.api+json",
}
_HEADERS_JSON = {
    "User-Agent": "Operator1/1.0 (financial-research)",
    "Accept": "application/json",
}

# IFRS concept -> canonical field name mapping
_IFRS_CONCEPT_MAP: dict[str, str] = {
    "ifrs-full:Revenue": "revenue",
    "ifrs-full:CostOfSales": "cost_of_revenue",
    "ifrs-full:GrossProfit": "gross_profit",
    "ifrs-full:ProfitLossFromOperatingActivities": "operating_income",
    "ifrs-full:ProfitLoss": "net_income",
    "ifrs-full:ProfitLossAttributableToOwnersOfParent": "net_income_attributable",
    "ifrs-full:IncomeTaxExpenseContinuingOperations": "taxes",
    "ifrs-full:FinanceCosts": "interest_expense",
    "ifrs-full:BasicEarningsLossPerShare": "eps",
    "ifrs-full:DilutedEarningsLossPerShare": "eps_diluted",
    "ifrs-full:Assets": "total_assets",
    "ifrs-full:CurrentAssets": "current_assets",
    "ifrs-full:NoncurrentAssets": "noncurrent_assets",
    "ifrs-full:Liabilities": "total_liabilities",
    "ifrs-full:CurrentLiabilities": "current_liabilities",
    "ifrs-full:NoncurrentLiabilities": "noncurrent_liabilities",
    "ifrs-full:Equity": "total_equity",
    "ifrs-full:CashAndCashEquivalents": "cash_and_equivalents",
    "ifrs-full:IssuedCapital": "share_capital",
    "ifrs-full:RetainedEarnings": "retained_earnings",
    "ifrs-full:Inventories": "inventory",
    "ifrs-full:TradeAndOtherCurrentReceivables": "receivables",
    "ifrs-full:TradeAndOtherCurrentPayables": "payables",
    "ifrs-full:PropertyPlantAndEquipment": "property_plant_equipment",
    "ifrs-full:Goodwill": "goodwill",
    "ifrs-full:IntangibleAssetsOtherThanGoodwill": "intangible_assets",
    "ifrs-full:CashFlowsFromUsedInOperatingActivities": "operating_cash_flow",
    "ifrs-full:CashFlowsFromUsedInInvestingActivities": "investing_cf",
    "ifrs-full:CashFlowsFromUsedInFinancingActivities": "financing_cf",
    "ifrs-full:DividendsPaid": "dividends_paid",
    "ifrs-full:DepreciationAndAmortisationExpense": "depreciation_amortization",
}

# Statement type -> set of canonical field names that belong to it
_INCOME_FIELDS = {
    "revenue", "cost_of_revenue", "gross_profit", "operating_income",
    "net_income", "net_income_attributable", "taxes", "interest_expense",
    "eps", "eps_diluted", "depreciation_amortization",
}
_BALANCE_FIELDS = {
    "total_assets", "current_assets", "noncurrent_assets",
    "total_liabilities", "current_liabilities", "noncurrent_liabilities",
    "total_equity", "cash_and_equivalents", "share_capital",
    "retained_earnings", "inventory", "receivables", "payables",
    "property_plant_equipment", "goodwill", "intangible_assets",
}
_CASHFLOW_FIELDS = {
    "operating_cash_flow", "investing_cf", "financing_cf", "dividends_paid",
}


class EUEsefError(Exception):
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"EU ESEF error on {endpoint}: {detail}")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_get(path: str, params: dict | None = None) -> dict:
    """GET from filings.xbrl.org JSON:API (fallback for when xbrl-filings-api unavailable)."""
    try:
        resp = requests.get(
            f"{_XBRL_API_BASE}{path}",
            params=params,
            headers=_HEADERS_API,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("ESEF API %s failed: %s", path, exc)
        return {}


def _fetch_entity_name(entity_link: str) -> str:
    """Resolve an entity relationship link to the entity name."""
    if not entity_link:
        return ""
    try:
        url = entity_link if entity_link.startswith("http") else f"{_XBRL_FILES_BASE}{entity_link}"
        resp = requests.get(url, headers=_HEADERS_API, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("attributes", {}).get("name", "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# xbrl-filings-api library integration
# ---------------------------------------------------------------------------

_XF_AVAILABLE = False
try:
    import xbrl_filings_api as _xf
    _XF_AVAILABLE = True
except ImportError:
    _xf = None  # type: ignore


def _xf_get_filings(
    country: str = "",
    entity_name: str = "",
    limit: int = 100,
) -> tuple[list, list]:
    """Fetch filings using xbrl-filings-api library.

    Returns (filings_list, entities_list) where each filing is a dict
    with keys: json_url, reporting_date, country, entity_name, entity_id.
    """
    if not _XF_AVAILABLE:
        return [], []

    import warnings
    filters: dict = {}
    if country:
        filters["country"] = country
    if entity_name:
        filters["entity.name"] = entity_name

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fs = _xf.get_filings(
                filters=filters,
                sort="-last_end_date",
                limit=limit,
                flags=_xf.GET_ENTITY,
            )

        filings = []
        for f in fs:
            filings.append({
                "json_url": f.json_url,
                "reporting_date": str(f.reporting_date) if f.reporting_date else "",
                "country": f.country or "",
                "entity_name": f.entity.name if f.entity else "",
                "entity_id": f.entity.identifier if f.entity else "",
                "added_time": str(f.added_time)[:10] if f.added_time else "",
            })

        entities = []
        for e in fs.entities:
            entities.append({
                "name": e.name or "",
                "identifier": e.identifier or "",
            })

        return filings, entities
    except Exception as exc:
        logger.debug("xbrl-filings-api query failed: %s", exc)
        return [], []


def _download_xbrl_json(json_url: str) -> dict:
    """Download and parse an XBRL JSON (OIM) file from filings.xbrl.org.

    The JSON files need Accept: application/json (not vnd.api+json).
    Files can be large (6+ MB for Unilever).
    """
    if not json_url:
        return {}
    url = json_url if json_url.startswith("http") else f"{_XBRL_FILES_BASE}{json_url}"
    try:
        resp = requests.get(url, headers=_HEADERS_JSON, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("XBRL JSON download failed for %s: %s", json_url, exc)
        return {}


def _extract_ifrs_facts(
    xbrl_data: dict,
    statement_type: str,
) -> list[dict]:
    """Extract IFRS financial facts from XBRL JSON (OIM format).

    Each fact in the OIM format:
    {
      "f-1": {
        "value": "50503000000.0",
        "dimensions": {
          "concept": "ifrs-full:Revenue",
          "entity": "scheme:549300MKFYEKVRWML317",
          "period": "2025-01-01T00:00:00/2026-01-01T00:00:00",
          "unit": "iso4217:EUR"
        }
      }
    }
    """
    # Select fields for requested statement type
    if statement_type == "income":
        target_fields = _INCOME_FIELDS
    elif statement_type == "balance":
        target_fields = _BALANCE_FIELDS
    elif statement_type == "cashflow":
        target_fields = _CASHFLOW_FIELDS
    else:
        target_fields = _INCOME_FIELDS | _BALANCE_FIELDS | _CASHFLOW_FIELDS

    facts = xbrl_data.get("facts", {})
    rows: list[dict] = []

    for fact_id, fact in facts.items():
        dims = fact.get("dimensions", {})
        concept = dims.get("concept", "")
        canonical = _IFRS_CONCEPT_MAP.get(concept)
        if not canonical or canonical not in target_fields:
            continue

        value_str = fact.get("value", "")
        if not value_str:
            continue

        try:
            value = float(value_str)
        except (ValueError, TypeError):
            continue

        # Parse period: instant ("2025-01-01T00:00:00") or
        # duration ("2025-01-01T00:00:00/2026-01-01T00:00:00")
        period = dims.get("period", "")
        if "/" in period:
            # Duration: use end date as report_date
            report_date = period.split("/")[1][:10]
        else:
            # Instant: use the date directly
            report_date = period[:10]

        # Skip facts with extra dimensional breakdowns (segments, etc.)
        # to avoid double-counting. Keep only facts with exactly the
        # standard 4 dimensions: concept, entity, period, unit.
        n_dims = len(dims)
        if n_dims > 4:
            continue

        rows.append({
            "canonical_name": canonical,
            "value": value,
            "report_date": report_date,
            "filing_date": "",  # Set by caller from filing metadata
        })

    return rows


# ---------------------------------------------------------------------------
# Company name cleaning (strips legal suffixes for fuzzy matching)
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES = [
    " s.p.a.", " s.p.a", " spa", " s.a.", " sa", " n.v.", " nv",
    " ab", " ab (publ)", " (publ)", " plc", " ltd", " limited",
    " se", " ag", " gmbh", " oyj", " asa", " a/s",
    " societa per azioni", " societe anonyme",
    " pjsc", " jsc", " inc", " corp", " co",
]


def _clean_company_name(name: str) -> str:
    """Strip common legal entity suffixes and quotes for fuzzy matching."""
    cleaned = name.strip().strip('"').strip("'").lower()
    for suffix in _LEGAL_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    cleaned = cleaned.strip('"').strip("'").strip()
    return cleaned


def _fuzzy_match_entity(
    query: str,
    entities: dict[str, dict],
    threshold: float = 65.0,
) -> list[tuple[str, dict, float]]:
    """Fuzzy match a company name against the entity directory.

    Uses rapidfuzz WRatio (best general-purpose scorer, combines ratio,
    partial_ratio, token_sort_ratio, and token_set_ratio with optimal
    weights). Falls back to difflib SequenceMatcher if rapidfuzz is
    unavailable.

    Pattern from: operator1/clients/fuzzy_pdf_parser.py::_fuzzy_match_concept

    Returns list of (entity_id, entity_dict, score) sorted by score desc.
    """
    if not entities or not query:
        return []

    query_clean = _clean_company_name(query.lower())

    # Build choices: entity_id -> cleaned name
    choices: dict[str, str] = {}
    for eid, info in entities.items():
        choices[eid] = _clean_company_name(info.get("name", "").lower())

    try:
        from rapidfuzz import process, fuzz
        # WRatio is the best general-purpose scorer -- it automatically
        # picks the best combination of ratio, partial_ratio,
        # token_sort_ratio, and token_set_ratio.
        matches = process.extract(
            query_clean,
            choices,
            scorer=fuzz.WRatio,
            score_cutoff=threshold,
            limit=10,
        )
        results = []
        for match_name, score, eid in matches:
            results.append((eid, entities[eid], score))
        return results

    except ImportError:
        # Fallback: difflib SequenceMatcher
        from difflib import SequenceMatcher
        results = []
        for eid, cleaned in choices.items():
            ratio = SequenceMatcher(None, query_clean, cleaned).ratio() * 100
            if ratio >= threshold:
                results.append((eid, entities[eid], ratio))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:10]


# ---------------------------------------------------------------------------
# Entity cache (resolves LEI -> entity name)
# ---------------------------------------------------------------------------

_entity_name_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# EUEsefClient
# ---------------------------------------------------------------------------


class EUEsefClient:
    """Point-in-time client for ESEF filings (EU equities).

    Uses the filings.xbrl.org JSON:API with proper filter syntax
    and XBRL JSON extraction for structured IFRS financial data.

    Implements the ``PITClient`` protocol. Filters by country code to
    support per-country market entries (EU, FR, DE, NL, ES, IT, SE).
    """

    # Country code -> primary currency.  Most EU ESEF filers use EUR,
    # but non-eurozone countries (SE, DK, NO, PL, CZ, HU, RO, BG)
    # report in their local currency.
    _COUNTRY_CURRENCY: dict[str, str] = {
        "SE": "SEK",
        "DK": "DKK",
        "NO": "NOK",
        "PL": "PLN",
        "CZ": "CZK",
        "HU": "HUF",
        "RO": "RON",
        "BG": "BGN",
        "GB": "GBP",
        "CH": "CHF",
    }

    def __init__(
        self,
        country_code: str = "",
        market_id: str = "eu_esef",
        cache_dir: Path | str = _CACHE_DIR,
    ) -> None:
        self._country_code = country_code.upper()
        self._market_id = market_id
        self._cache_dir = Path(cache_dir)

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
            if filename.endswith("_financials.json") and age_days > 30:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, filename: str, data: Any) -> None:
        path = self._cache_path(identifier, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    @property
    def market_id(self) -> str:
        return self._market_id

    @property
    def market_name(self) -> str:
        labels = {
            "eu_esef": "European Union (ESEF -- all EU)",
            "fr_esef": "France (Paris / Euronext -- ESEF)",
            "de_esef": "Germany (Frankfurt / XETRA -- ESEF)",
            "nl_esef": "Netherlands (Euronext Amsterdam -- ESEF)",
            "es_esef": "Spain (Madrid / BME -- ESEF)",
            "it_esef": "Italy (Borsa Italiana -- ESEF)",
            "se_esef": "Sweden (Stockholm / Nasdaq Nordic -- ESEF)",
        }
        return labels.get(self._market_id, f"ESEF ({self._country_code})")

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List companies from ESEF filings for this country.

        Fetches filings via JSON:API filter[country], resolves entity
        names via relationship links, and caches the full directory
        to avoid repeated API calls.
        """
        # Build entity directory (cached per country)
        directory = self._get_entity_directory()

        results = list(directory.values())
        if query:
            q = query.lower()
            # Exact substring match first (fast)
            exact = [
                c for c in results
                if q in c["name"].lower() or q in c.get("lei", "").lower()
            ]
            if exact:
                return exact

            # Cleaned substring match (strips legal suffixes)
            q_clean = _clean_company_name(q)
            if len(q_clean) >= 3:
                cleaned = [
                    c for c in results
                    if q_clean in _clean_company_name(c["name"].lower())
                ]
                if cleaned:
                    return cleaned

            # Fuzzy match with rapidfuzz WRatio (handles legal suffixes,
            # acronyms, partial names). Higher threshold for short queries
            # to avoid false positives (e.g., "BNP" matching "CNP").
            min_score = 80.0 if len(q_clean) < 6 else 70.0
            fuzzy_matches = _fuzzy_match_entity(query, directory, threshold=min_score)
            if fuzzy_matches:
                return [info for _, info, _ in fuzzy_matches]

        return results

    def search_company(self, name: str) -> list[dict[str, Any]]:
        """Search for companies by name in ESEF filings.

        Strategy:
        1. Try exact match via xbrl-filings-api entity.name filter
        2. Try exact match via raw /api/entities endpoint
        3. Fall back to substring search in cached entity directory
        """
        # Path 1: Exact match via xbrl-filings-api (fast, handles JSON:API)
        if _XF_AVAILABLE:
            filings, entities = _xf_get_filings(
                country=self._country_code,
                entity_name=name,
                limit=5,
            )
            if entities:
                results = []
                for e in entities:
                    results.append({
                        "ticker": e["identifier"][:20],
                        "name": e["name"],
                        "lei": e["identifier"],
                        "cik": e["identifier"],
                        "country": filings[0]["country"] if filings else "EU",
                        "exchange": "ESEF",
                        "market_id": self.market_id,
                    })
                return results
            # Also try without country filter for exact name match
            if self._country_code:
                filings2, entities2 = _xf_get_filings(entity_name=name, limit=5)
                if entities2:
                    return [{
                        "ticker": e["identifier"][:20],
                        "name": e["name"],
                        "lei": e["identifier"],
                        "cik": e["identifier"],
                        "country": "EU",
                        "exchange": "ESEF",
                        "market_id": self.market_id,
                    } for e in entities2]

        # Path 2: Raw API exact match
        try:
            data = _api_get("/entities", params={"filter[name]": name, "page[size]": "5"})
            entities = data.get("data", [])
            results = []
            for ent in entities:
                attrs = ent.get("attributes", {})
                ent_name = attrs.get("name", "")
                ent_id = str(ent.get("id", ""))
                if ent_name:
                    results.append({
                        "ticker": ent_id[:20],
                        "name": ent_name,
                        "lei": ent_id,
                        "cik": ent_id,
                        "country": "EU",
                        "exchange": "ESEF",
                        "market_id": self.market_id,
                    })
            if results:
                return results
        except Exception:
            pass

        # Path 3: Substring search in cached directory
        return self.list_companies(query=name)

    def _get_entity_directory(self) -> dict[str, dict]:
        """Build and cache a directory of entities from ESEF filings.

        Uses ``include=entity`` sideload to get entity names in a
        single API call (no N+1 entity resolution). Fetches up to
        250 filings per page to cover most listed companies.
        """
        cache_key = f"entity_directory_{self._country_code or 'ALL'}"

        # Check disk cache (24h TTL)
        cached = self._read_cache(cache_key, "directory.json")
        if cached and isinstance(cached, dict):
            for eid, info in cached.items():
                _entity_name_cache[eid] = info.get("name", "")
            return cached

        params: dict[str, str] = {
            "page[size]": "500",
            "sort": "-period_end",
            "include": "entity",
        }
        if self._country_code:
            params["filter[country]"] = self._country_code

        data = _api_get("/filings", params=params)
        filings = data.get("data", [])
        included = data.get("included", [])

        # Build entity ID -> name map from sideloaded entities
        entity_map: dict[str, str] = {}
        for inc in included:
            if inc.get("type") == "entity":
                eid = str(inc.get("id", ""))
                name = inc.get("attributes", {}).get("name", "")
                if eid and name:
                    entity_map[eid] = name
                    _entity_name_cache[eid] = name

        # Build directory from filings + sideloaded entity names
        directory: dict[str, dict] = {}
        for f in filings:
            attrs = f.get("attributes", {})
            country = attrs.get("country", "")
            entity_link = (
                f.get("relationships", {})
                .get("entity", {})
                .get("links", {})
                .get("related", "")
            )
            entity_id = entity_link.split("/")[-1] if entity_link else ""

            if not entity_id or entity_id in directory:
                continue

            name = entity_map.get(entity_id, _entity_name_cache.get(entity_id, ""))
            if name:
                directory[entity_id] = {
                    "ticker": entity_id[:20],
                    "name": name,
                    "lei": entity_id,
                    "cik": entity_id,
                    "country": country,
                    "exchange": "ESEF",
                    "market_id": self.market_id,
                }

        if directory:
            self._write_cache(cache_key, "directory.json", directory)
            logger.info("ESEF entity directory built: %d entities for %s",
                        len(directory), self._country_code or "ALL")

        return directory

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        matches = self.search_company(identifier)
        if not matches:
            raise EUEsefError("get_profile", f"Entity not found: {identifier}")

        m = matches[0]
        # Resolve currency from country code: non-eurozone ESEF filers
        # report in their local currency (SEK, DKK, NOK, PLN, etc.).
        _country = m.get("country", self._country_code or "EU").upper()
        _currency = self._COUNTRY_CURRENCY.get(_country, "EUR")
        profile = {
            "name": m.get("name", ""),
            "ticker": m.get("lei", identifier),
            "isin": "",
            "country": _country if _country != "EU" else m.get("country", "EU"),
            "sector": "",
            "industry": "",
            "exchange": "ESEF",
            "currency": _currency,
            "lei": m.get("lei", ""),
            "cik": m.get("lei", ""),
            "market_id": self.market_id,
        }

        self._write_cache(identifier, "profile.json", profile)
        return profile

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _find_entity_id(self, identifier: str) -> str:
        """Resolve a company name or LEI to the ESEF entity ID.

        Checks the cached entity directory first (fast), then falls
        back to exact-match entities API.
        """
        # If it looks like an LEI, use directly
        if len(identifier) >= 18 and identifier.isalnum():
            return identifier

        # Check cached directory (substring match)
        directory = self._get_entity_directory()
        q = identifier.lower()
        for eid, info in directory.items():
            if q in info.get("name", "").lower() or q == eid.lower():
                return eid

        # Fallback: exact match via entities API
        try:
            data = _api_get("/entities", params={"filter[name]": identifier, "page[size]": "1"})
            entities = data.get("data", [])
            if entities:
                return str(entities[0].get("id", ""))
        except Exception:
            pass

        return ""

    def _get_entity_filings(self, entity_id: str) -> list[dict]:
        """Fetch all filings for a specific entity.

        Uses xbrl-filings-api entity.identifier filter (fastest),
        then falls back to raw API entity filings link.
        """
        if not entity_id:
            return []

        # Path 1: xbrl-filings-api with entity.identifier filter
        if _XF_AVAILABLE:
            filings, _ = _xf_get_filings(
                entity_name=_entity_name_cache.get(entity_id, ""),
                limit=20,
            )
            if not filings:
                # Try identifier-based filter
                import warnings
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fs = _xf.get_filings(
                            filters={"entity.identifier": entity_id},
                            sort="-last_end_date",
                            limit=20,
                        )
                    filings = [{
                        "json_url": f.json_url,
                        "reporting_date": str(f.reporting_date) if f.reporting_date else "",
                        "country": f.country or "",
                        "entity_name": f.entity.name if f.entity else "",
                        "entity_id": f.entity.identifier if f.entity else "",
                        "added_time": str(f.added_time)[:10] if f.added_time else "",
                    } for f in fs]
                except Exception:
                    pass

            if filings:
                # Convert to the format expected by _fetch_financials
                api_filings = []
                for fl in filings:
                    api_filings.append({
                        "attributes": {
                            "json_url": fl["json_url"],
                            "period_end": fl["reporting_date"],
                            "date_added": fl["added_time"],
                            "country": fl["country"],
                        },
                        "relationships": {
                            "entity": {"links": {"related": f"/api/entities/{entity_id}"}}
                        },
                    })
                return api_filings

        # Path 2: Direct entity filings via raw API
        try:
            data = _api_get(f"/entities/{entity_id}/filings", params={
                "page[size]": "20",
                "sort": "-period_end",
            })
            filings = data.get("data", [])
            if filings:
                return filings
        except Exception:
            pass

        return []

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial data by downloading XBRL JSON from filings.

        This is the key fix: instead of looking for inline facts or
        a non-existent /facts endpoint, we download the actual XBRL
        JSON file from each filing's json_url attribute and extract
        IFRS concepts directly.
        """
        # Check cache
        cache_key = f"{statement_type}_financials.json"
        cached = self._read_cache(identifier, cache_key)
        if cached:
            try:
                df = pd.DataFrame(cached)
                if not df.empty:
                    for col in ("filing_date", "report_date"):
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                    return df
            except Exception:
                pass

        entity_id = self._find_entity_id(identifier)
        if not entity_id:
            logger.info("ESEF: entity not found for '%s'", identifier)
            return pd.DataFrame()

        entity_filings = self._get_entity_filings(entity_id)
        if not entity_filings:
            logger.info("ESEF: no filings found for entity %s", entity_id)
            return pd.DataFrame()

        all_rows: list[dict] = []

        for filing in entity_filings[:5]:  # Limit to 5 most recent
            attrs = filing.get("attributes", {})
            json_url = attrs.get("json_url", "")
            filing_date = attrs.get("date_added", attrs.get("processed", ""))[:10]
            period_end = attrs.get("period_end", "")

            if not json_url:
                continue

            logger.info(
                "ESEF: downloading XBRL JSON for %s (period_end=%s)",
                identifier, period_end,
            )

            xbrl_data = _download_xbrl_json(json_url)
            if not xbrl_data:
                continue

            rows = _extract_ifrs_facts(xbrl_data, statement_type)

            # Set filing_date from metadata
            for row in rows:
                if not row["filing_date"]:
                    row["filing_date"] = filing_date

            all_rows.extend(rows)

            # Rate limiting
            time.sleep(0.5)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # Deduplicate: keep one value per (canonical_name, report_date)
        # preferring the most recent filing
        df = df.sort_values("filing_date", ascending=False)
        df = df.drop_duplicates(subset=["canonical_name", "report_date"], keep="first")

        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Cache results
        self._write_cache(identifier, cache_key, df.to_dict(orient="records"))

        logger.info(
            "ESEF %s %s: %d facts across %d periods",
            identifier, statement_type, len(df),
            df["report_date"].nunique() if "report_date" in df.columns else 0,
        )
        return df

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """ESEF does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        all_companies = self.list_companies()
        return [
            c.get("lei", c.get("name", ""))
            for c in all_companies
            if c.get("name", "") != identifier
        ][:10]

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

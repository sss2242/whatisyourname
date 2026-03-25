"""GLEIF corporate structure client -- parent/subsidiary ownership chains.

Fetches corporate group structure from the GLEIF LEI API (free, no auth,
2.6M+ legal entities globally).  This is NOT shareholder/portfolio data --
it is the corporate control chain: who owns whom at the entity level.

Use cases in the pipeline:
  - Entity discovery enrichment: add parent_companies and subsidiaries
    to the relationships dict alongside LLM-discovered entities
  - Graph risk: parent->subsidiary edges have ~0.85 contagion probability
    (controlling ownership is the strongest contagion channel)
  - Survival mode: subsidiary of a strong parent has lower survival risk
  - Fuzzy protection: subsidiary inherits parent's government protection
  - Economic planes: conglomerate parents span multiple planes

API: https://api.gleif.org/api/v1/lei-records
Coverage: 2.6M+ LEI records across all 25 Operator 1 markets.
Rate limits: No documented limits (be respectful, ~1 req/s).
Auth: None required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

_GLEIF_API = "https://api.gleif.org/api/v1/lei-records"
_HEADERS = {
    "User-Agent": "Operator1/1.0 (financial-research)",
    "Accept": "application/json",
}
_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CorporateEntity:
    """A single entity in the corporate structure tree."""

    lei: str = ""
    name: str = ""
    country: str = ""
    relationship: str = ""  # "ultimate_parent", "direct_parent", "subsidiary", "target"


@dataclass
class CorporateStructureResult:
    """Complete corporate structure for a target entity."""

    target_lei: str = ""
    target_name: str = ""
    ultimate_parent: CorporateEntity | None = None
    direct_parent: CorporateEntity | None = None
    subsidiaries: list[CorporateEntity] = field(default_factory=list)
    available: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# LEI resolution
# ---------------------------------------------------------------------------


def resolve_lei(identifier: str) -> str:
    """Resolve a company name, ticker, or ISIN to a GLEIF LEI.

    Tries exact LEI format first (20 alphanumeric chars), then searches
    GLEIF by legal name.

    Parameters
    ----------
    identifier:
        Company name, ticker, ISIN, or LEI.

    Returns
    -------
    LEI string (20 chars) or empty string if not found.
    """
    clean = identifier.strip()

    # Already a LEI?
    if len(clean) == 20 and clean.isalnum():
        return clean

    # Search GLEIF by name
    try:
        resp = requests.get(
            _GLEIF_API,
            params={"filter[entity.legalName]": clean, "page[size]": "1"},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.ok:
            records = resp.json().get("data", [])
            if records:
                return records[0].get("attributes", {}).get("lei", "")
    except Exception as exc:
        logger.debug("GLEIF LEI resolution failed for '%s': %s", identifier, exc)

    return ""


# ---------------------------------------------------------------------------
# Corporate structure fetch
# ---------------------------------------------------------------------------


def _parse_entity(data: dict, relationship: str) -> CorporateEntity | None:
    """Parse a GLEIF API record into a CorporateEntity."""
    if not data or not isinstance(data, dict):
        return None

    attrs = data.get("attributes", {})
    entity = attrs.get("entity", {})
    name = entity.get("legalName", {}).get("name", "")
    lei = attrs.get("lei", "")
    country = entity.get("legalAddress", {}).get("country", "")

    if not name and not lei:
        return None

    return CorporateEntity(
        lei=lei,
        name=name,
        country=country,
        relationship=relationship,
    )


def fetch_corporate_structure(
    identifier: str,
    max_subsidiaries: int = 30,
) -> CorporateStructureResult:
    """Fetch the complete corporate structure for a company.

    Parameters
    ----------
    identifier:
        Company name, ticker, ISIN, or LEI.
    max_subsidiaries:
        Maximum number of subsidiaries to fetch (default 30).

    Returns
    -------
    CorporateStructureResult with parent chain and subsidiaries.
    """
    result = CorporateStructureResult()

    lei = resolve_lei(identifier)
    if not lei:
        result.error = f"Could not resolve LEI for '{identifier}'"
        return result

    result.target_lei = lei

    # Get target entity name
    try:
        resp = requests.get(
            f"{_GLEIF_API}/{lei}",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.ok:
            target_data = resp.json().get("data", {})
            target_entity = _parse_entity(target_data, "target")
            if target_entity:
                result.target_name = target_entity.name
    except Exception as exc:
        logger.debug("GLEIF target lookup failed for %s: %s", lei, exc)

    # Ultimate parent
    try:
        resp = requests.get(
            f"{_GLEIF_API}/{lei}/ultimate-parent",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.ok:
            parent_data = resp.json().get("data")
            if parent_data and isinstance(parent_data, dict):
                parent = _parse_entity(parent_data, "ultimate_parent")
                if parent and parent.lei != lei:
                    result.ultimate_parent = parent
    except Exception as exc:
        logger.debug("GLEIF ultimate parent failed for %s: %s", lei, exc)

    # Direct parent
    try:
        resp = requests.get(
            f"{_GLEIF_API}/{lei}/direct-parent",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.ok:
            parent_data = resp.json().get("data")
            if parent_data and isinstance(parent_data, dict):
                parent = _parse_entity(parent_data, "direct_parent")
                # Avoid duplicate with ultimate parent
                if (parent and parent.lei != lei
                        and (not result.ultimate_parent
                             or parent.lei != result.ultimate_parent.lei)):
                    result.direct_parent = parent
    except Exception as exc:
        logger.debug("GLEIF direct parent failed for %s: %s", lei, exc)

    # Direct children (subsidiaries)
    try:
        resp = requests.get(
            f"{_GLEIF_API}/{lei}/direct-children",
            params={"page[size]": str(min(max_subsidiaries, 100))},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.ok:
            children = resp.json().get("data", [])
            for child_data in children[:max_subsidiaries]:
                child = _parse_entity(child_data, "subsidiary")
                if child:
                    result.subsidiaries.append(child)
    except Exception as exc:
        logger.debug("GLEIF children failed for %s: %s", lei, exc)

    result.available = bool(
        result.ultimate_parent
        or result.direct_parent
        or result.subsidiaries
    )

    if result.available:
        n_parents = sum(1 for x in [result.ultimate_parent, result.direct_parent] if x)
        logger.info(
            "GLEIF corporate structure for %s (%s): %d parents, %d subsidiaries",
            result.target_name or identifier,
            lei,
            n_parents,
            len(result.subsidiaries),
        )

    return result

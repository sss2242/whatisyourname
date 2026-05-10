"""Steps A/B -- Linked entity discovery and resolution.

Uses Gemini to propose related entities (competitors, suppliers, customers,
etc.) and resolves each to a record via the PIT data provider using fuzzy
matching with scoring.  Tracks search budgets and checkpoints progress to disk.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from operator1.clients.pit_base import PITClientError
from operator1.clients.equity_provider import EquityProvider
from operator1.clients.llm_base import LLMClient
from operator1.config_loader import get_global_config
from operator1.constants import CACHE_DIR, MATCH_SCORE_THRESHOLD, SECTOR_PEER_FALLBACK_COUNT

logger = logging.getLogger(__name__)

_PROGRESS_PATH = os.path.join(CACHE_DIR, "progress.json")

# Relationship groups expected from Gemini
RELATIONSHIP_GROUPS = (
    "competitors",
    "suppliers",
    "customers",
    "financial_institutions",
    "logistics",
    "regulators",
)


@dataclass
class LinkedEntity:
    """A resolved linked entity with match metadata and temporal context."""

    isin: str
    ticker: str
    name: str
    country: str
    sector: str
    relationship_group: str
    match_score: int
    market_cap: float | None = None
    market_id: str = ""  # PIT wrapper that resolved this entity (for cross-region data fetch)
    # Temporal context (from Gemini discovery)
    relationship_start: str = "unknown"   # "YYYY", "ongoing", "unknown"
    relationship_end: str = "current"     # "current", "YYYY", "unknown"
    relationship_stability: str = "stable"  # "stable", "volatile", "new"


@dataclass
class DiscoveryResult:
    """Container for the full discovery output."""

    linked: dict[str, list[LinkedEntity]] = field(default_factory=dict)
    search_calls_used: int = 0
    dropped_low_score: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_match(
    query: str,
    candidate: dict[str, Any],
    target_country: str,
    target_sector: str,
) -> int:
    """Score a search candidate against the original query.

    Scoring weights (out of 100):
      - Ticker exact match:   40 pts
      - Name similarity:      30 pts (SequenceMatcher ratio * 30)
      - Country match:        15 pts
      - Sector match:         15 pts
    """
    score = 0

    # Ticker exact match
    cand_ticker = (candidate.get("ticker") or "").upper()
    if cand_ticker and cand_ticker == query.upper():
        score += 40

    # Name similarity
    cand_name = (candidate.get("name") or "").lower()
    ratio = SequenceMatcher(None, query.lower(), cand_name).ratio()
    score += int(ratio * 30)

    # Country match
    cand_country = (candidate.get("country") or "").upper()
    if cand_country and cand_country == target_country.upper():
        score += 15

    # Sector match
    cand_sector = (candidate.get("sector") or "").lower()
    if cand_sector and cand_sector == target_sector.lower():
        score += 15

    return score


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _load_progress() -> dict[str, Any]:
    """Load discovery progress from disk."""
    if os.path.exists(_PROGRESS_PATH):
        try:
            with open(_PROGRESS_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {"resolved": {}, "search_calls": 0}


def _save_progress(progress: dict[str, Any]) -> None:
    """Checkpoint discovery progress to disk."""
    os.makedirs(os.path.dirname(_PROGRESS_PATH), exist_ok=True)
    with open(_PROGRESS_PATH, "w", encoding="utf-8") as fh:
        json.dump(progress, fh, indent=2)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _resolve_entity(
    query: str,
    group: str,
    pit_client: EquityProvider,
    target_country: str,
    target_sector: str,
) -> LinkedEntity | None:
    """Search PIT provider for *query* and return the best match above threshold.

    Returns ``None`` if no match scores >= MATCH_SCORE_THRESHOLD.
    """
    try:
        # Use search_company if available, otherwise list_companies
        if hasattr(pit_client, "search_company"):
            candidates = pit_client.search_company(query)
        elif hasattr(pit_client, "search"):
            candidates = pit_client.search(query)
        else:
            candidates = pit_client.list_companies(query)
    except (PITClientError, Exception) as exc:
        logger.warning("Search failed for '%s': %s", query, exc)
        return None

    # Fallback: if full name returns nothing, try the first word
    # (many PIT clients work better with shorter, simpler queries)
    if not candidates and " " in query:
        short_query = query.split()[0]
        try:
            if hasattr(pit_client, "search_company"):
                candidates = pit_client.search_company(short_query)
            else:
                candidates = pit_client.list_companies(short_query)
        except Exception:
            pass

    if not candidates:
        logger.debug("No search results for '%s'", query)
        return None

    # Score and rank
    scored = []
    for c in candidates:
        s = _score_match(query, c, target_country, target_sector)
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best = scored[0]
    if best_score < MATCH_SCORE_THRESHOLD:
        logger.debug(
            "Best match for '%s' scored %d (< %d) -- dropped",
            query, best_score, MATCH_SCORE_THRESHOLD,
        )
        return None

    return LinkedEntity(
        isin=best.get("isin", ""),
        ticker=best.get("ticker", ""),
        name=best.get("name", ""),
        country=best.get("country", ""),
        sector=best.get("sector", ""),
        relationship_group=group,
        match_score=best_score,
        market_cap=best.get("market_cap"),
    )


# ---------------------------------------------------------------------------
# Fallback: sector peers
# ---------------------------------------------------------------------------

def _load_static_competitors(
    target_ticker: str,
    target_sector: str,
    target_industry: str,
) -> list[LinkedEntity]:
    """Load competitors from static whale_competitors.yml registry.

    Returns LinkedEntity list for the target's sector/industry, excluding
    the target itself. Zero API calls, zero LLM calls.
    """
    try:
        from operator1.config_loader import load_config
        registry = load_config("whale_competitors")
    except (FileNotFoundError, Exception):
        return []

    # Normalize sector/industry for matching
    sector_lower = (target_sector or "").lower().replace(" ", "_").replace("&", "and")
    industry_lower = (target_industry or "").lower().replace(" ", "_").replace("&", "and")
    ticker_upper = (target_ticker or "").upper()

    candidates: list[LinkedEntity] = []

    for sector_key, industries in registry.items():
        if not isinstance(industries, dict):
            continue
        # Match by sector name (fuzzy)
        sector_match = any(
            kw in sector_lower
            for kw in sector_key.lower().split("_")
        ) or any(
            kw in sector_key.lower()
            for kw in sector_lower.split("_") if len(kw) > 3
        )

        if not sector_match:
            continue

        for industry_key, companies in industries.items():
            if not isinstance(companies, list):
                continue
            for comp in companies:
                if not isinstance(comp, dict):
                    continue
                comp_ticker = str(comp.get("ticker", "")).upper()
                # Skip the target itself
                if comp_ticker == ticker_upper:
                    continue
                candidates.append(LinkedEntity(
                    isin="",
                    ticker=comp_ticker,
                    name=comp.get("name", comp_ticker),
                    country="",
                    sector=sector_key,
                    relationship_group="competitors",
                    match_score=85,  # high confidence (static registry)
                    market_cap=None,
                ))

    if candidates:
        logger.info(
            "Static competitor registry: %d competitors loaded for sector '%s'",
            len(candidates), target_sector,
        )
    return candidates[:10]  # cap at 10


def _fallback_sector_peers(
    target_isin: str,
    pit_client: EquityProvider,
    target_sector: str = "",
    count: int = SECTOR_PEER_FALLBACK_COUNT,
) -> list[LinkedEntity]:
    """Fallback when no competitors found: use PIT provider peers.

    Two-level fallback strategy:
      1. ``get_peers()`` -- direct peer list from the PIT API.
      2. ``list_companies(sector)`` -- if peers returns nothing,
         search the company listing by sector name.
    """
    logger.info("Competitor fallback: fetching sector peers via PIT provider ...")
    peers: list[LinkedEntity] = []

    # Level 1: direct peer list
    try:
        peer_isins = pit_client.get_peers(target_isin)
        for isin in peer_isins:
            if isin == target_isin:
                continue
            try:
                profile = pit_client.get_profile(isin)
                peers.append(LinkedEntity(
                    isin=isin,
                    ticker=profile.get("ticker", ""),
                    name=profile.get("name", ""),
                    country=profile.get("country", ""),
                    sector=profile.get("sector", ""),
                    relationship_group="competitors",
                    match_score=100,  # direct peer, full confidence
                    market_cap=profile.get("market_cap"),
                ))
            except (PITClientError, Exception):
                continue
            if len(peers) >= count:
                break
    except (PITClientError, Exception) as exc:
        logger.debug("get_peers() failed: %s", exc)

    # Level 2: sector-based company listing fallback
    if len(peers) < count and target_sector:
        logger.info(
            "Peer list yielded %d/%d -- trying sector listing for '%s'",
            len(peers), count, target_sector,
        )
        try:
            if hasattr(pit_client, "search_company"):
                candidates = pit_client.search_company(target_sector)
            elif hasattr(pit_client, "list_companies"):
                candidates = pit_client.list_companies(query=target_sector)
            else:
                candidates = []

            existing_ids = {p.isin for p in peers} | {
                p.ticker for p in peers
            } | {target_isin}

            for c in candidates:
                c_isin = c.get("isin", "")
                c_ticker = c.get("ticker", "")
                if c_isin in existing_ids or c_ticker in existing_ids:
                    continue
                peers.append(LinkedEntity(
                    isin=c_isin,
                    ticker=c_ticker,
                    name=c.get("name", ""),
                    country=c.get("country", ""),
                    sector=c.get("sector", target_sector),
                    relationship_group="competitors",
                    match_score=70,  # sector match, lower confidence
                    market_cap=c.get("market_cap"),
                ))
                existing_ids.add(c_isin or c_ticker)
                if len(peers) >= count:
                    break
        except (PITClientError, Exception) as exc:
            logger.debug("Sector listing fallback failed: %s", exc)

    logger.info("Peer fallback yielded %d competitors", len(peers))
    return peers


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------

def _resolve_entity_cross_region(
    query: str,
    group: str,
    primary_client: EquityProvider,
    all_clients: list[EquityProvider] | None,
    target_country: str,
    target_sector: str,
) -> "LinkedEntity | None":
    """Resolve an entity by searching the primary PIT client first, then
    all other region clients.

    This handles the case where Gemini suggests "Apple" as a competitor
    for a Japanese company -- Apple won't be in EDINET but will be in
    SEC EDGAR.

    Parameters
    ----------
    query:
        Company name to search for.
    group:
        Relationship group (competitors, suppliers, etc.).
    primary_client:
        The target company's PIT client (searched first).
    all_clients:
        List of all available PIT clients (searched as fallback).
    target_country:
        Country of the target company.
    target_sector:
        Sector of the target company.
    """
    # Try the primary (target's region) client first
    entity = _resolve_entity(query, group, primary_client, target_country, target_sector)
    if entity is not None:
        return entity

    # If not found in the primary region, search other regions
    if all_clients:
        for client in all_clients:
            # Skip the primary client (already tried)
            if hasattr(client, "market_id") and hasattr(primary_client, "market_id"):
                if client.market_id == primary_client.market_id:
                    continue
            try:
                entity = _resolve_entity(query, group, client, target_country, target_sector)
                if entity is not None:
                    logger.info(
                        "  Cross-region resolve: '%s' found via %s",
                        query,
                        getattr(client, "market_name", "unknown"),
                    )
                    return entity
            except Exception:
                continue

    return None


def _resolve_entity_direct(
    entity_dict: dict,
    group: str,
    secrets: dict[str, str] | None,
    target_country: str,
    target_sector: str,
) -> LinkedEntity | None:
    """Resolve entity using LLM-provided market_id + ticker (direct routing).

    Creates the exact PIT client needed and searches by ticker, then by
    name.  Falls back to None if the market_id is invalid or no match
    is found.  This replaces the brute-force 25-wrapper loop with a
    single targeted API call.
    """
    market_id = entity_dict.get("market_id", "")
    ticker = entity_dict.get("ticker", "")
    name = entity_dict.get("name", "")

    if not market_id or not name:
        return None

    # Validate market_id against the registry
    from operator1.clients.pit_registry import MARKETS
    if market_id not in MARKETS:
        logger.debug("LLM returned unknown market_id '%s' for '%s'", market_id, name)
        return None

    try:
        from operator1.clients.equity_provider import create_pit_client
        client = create_pit_client(market_id, secrets or {})

        # Search by ticker first (more precise), then by name
        query = ticker if ticker else name
        entity = _resolve_entity(query, group, client, target_country, target_sector)

        if entity is not None:
            entity.market_id = market_id
            logger.info(
                "  Direct-routed: '%s' -> %s via %s (1 API call)",
                name, entity.ticker or entity.isin, market_id,
            )
            return entity

        # Ticker failed, try name as fallback
        if ticker and ticker != name:
            entity = _resolve_entity(name, group, client, target_country, target_sector)
            if entity is not None:
                entity.market_id = market_id
                return entity

    except Exception as exc:
        logger.debug("Direct resolution failed for '%s' via %s: %s", name, market_id, exc)

    return None


def _build_all_pit_clients(secrets: dict[str, str] | None = None) -> list[EquityProvider]:
    """Instantiate a PIT client for every supported market.

    Used for cross-region entity resolution: when Gemini suggests a
    company from a different region, we need to search that region's API.
    """
    if secrets is None:
        secrets = {}

    from operator1.clients.equity_provider import create_pit_client
    from operator1.clients.pit_registry import MARKETS

    clients: list[EquityProvider] = []
    for market_id in MARKETS:
        try:
            client = create_pit_client(market_id, secrets)
            clients.append(client)
        except Exception as exc:
            logger.debug("Could not create client for %s: %s", market_id, exc)
    return clients


def discover_linked_entities(
    target_profile: dict[str, Any],
    pit_client: EquityProvider,
    llm_client: LLMClient | None = None,
    target_isin: str = "",
    force_rebuild: bool | None = None,
    secrets: dict[str, str] | None = None,
) -> DiscoveryResult:
    """Discover and resolve linked entities for the target company.

    Gemini proposes company names (no financial data).  Each name is
    resolved by searching the target's PIT client first, then all other
    regional PIT clients.  This handles cross-region relationships
    (e.g. a Japanese company's US competitor).

    Parameters
    ----------
    target_profile:
        Full profile dict from PIT provider (used for Gemini hints).
    pit_client:
        Initialised PIT data client for the target's market.
    llm_client:
        Optional LLM client.  If ``None``, skips LLM proposals and
        goes straight to peer fallback for competitors.
    target_isin:
        Identifier of the verified target.
    force_rebuild:
        Override FORCE_REBUILD config.
    secrets:
        API key dictionary for creating cross-region PIT clients.

    Returns
    -------
    DiscoveryResult
        Contains linked entities grouped by relationship type.
    """
    cfg = get_global_config()
    if force_rebuild is None:
        force_rebuild = cfg.get("FORCE_REBUILD", False)

    budget_per_group: int = cfg.get("search_budget_per_group", 10)
    budget_global: int = cfg.get("search_budget_global", 50)

    target_country = (target_profile.get("country") or "").upper()
    target_sector = (target_profile.get("sector") or "").lower()

    if not target_isin:
        target_isin = target_profile.get("isin", "") or target_profile.get("ticker", "")

    # Load checkpoint (resume if partially complete)
    progress = _load_progress() if not force_rebuild else {"resolved": {}, "search_calls": 0}
    global_calls = progress.get("search_calls", 0)

    result = DiscoveryResult(search_calls_used=global_calls)

    # Restore previously resolved entities
    for group, entities_raw in progress.get("resolved", {}).items():
        result.linked[group] = [
            LinkedEntity(**e) for e in entities_raw
        ]

    # ------------------------------------------------------------------
    # P10: Static competitor fallback for well-known companies.
    # When entity discovery fails (LLM timeout, API rate limit, no key),
    # provide a curated competitor list for major companies so peer
    # features, DTW analogs, and competitive pressure index still work.
    # ------------------------------------------------------------------
    _STATIC_COMPETITORS: dict[str, list[str]] = {
        "AAPL": ["MSFT", "GOOG", "AMZN", "META", "SSNLF"],
        "MSFT": ["AAPL", "GOOG", "AMZN", "META", "ORCL"],
        "GOOG": ["MSFT", "META", "AMZN", "AAPL", "SNAP"],
        "AMZN": ["MSFT", "GOOG", "WMT", "BABA", "SHOP"],
        "META": ["GOOG", "SNAP", "PINS", "RDDT", "MSFT"],
        "TSLA": ["F", "GM", "RIVN", "NIO", "BYD"],
        "NVDA": ["AMD", "INTC", "QCOM", "AVGO", "TSM"],
        "JPM": ["BAC", "GS", "MS", "C", "WFC"],
        "7203": ["7267", "7201", "7211", "7261", "STLA"],  # Toyota
        "005930": ["000660", "066570", "051910", "AAPL", "INTC"],  # Samsung
        "2330": ["NVDA", "INTC", "AMD", "UMC", "ASML"],  # TSMC
        "PETR4": ["PBR", "E", "COP", "CVX", "XOM"],  # Petrobras
        "600519": ["000858", "000568", "002304", "603369", "000596"],  # Moutai
    }

    # ------------------------------------------------------------------
    # 0. Build cross-region PIT clients for resolving entities from
    #    other markets (e.g. a JP company's US competitor)
    # ------------------------------------------------------------------
    all_clients: list[EquityProvider] | None = None
    if cfg.get("cross_region_discovery", True):
        try:
            all_clients = _build_all_pit_clients(secrets)
            if all_clients:
                logger.info(
                    "Cross-region discovery: %d PIT clients available",
                    len(all_clients),
                )
        except Exception as exc:
            logger.debug("Cross-region client build failed: %s", exc)

    # ------------------------------------------------------------------
    # 1. Get LLM proposals (names only -- no financial data)
    # Uses 3-call strategy (international + local + gap-fill) for
    # thicker entity caches, with single-call fallback.
    # ------------------------------------------------------------------
    proposals: dict[str, list[str]] = {}
    if llm_client is not None:
        sector_hints = f"{target_sector}, country={target_country}"

        # Build market summary for LLM prompt injection (Pattern P1 from
        # Claude Code: inject available capabilities into the prompt so
        # the LLM can route entities to the correct wrapper).
        _available_markets = ""
        try:
            from operator1.clients.pit_registry import get_market_summary_for_llm
            _available_markets = get_market_summary_for_llm()
        except Exception:
            pass

        # Use single-call discovery (more reliable with free-tier LLM providers
        # that have strict rate limits -- 3-call burns through the budget and
        # all 3 fail, leaving nothing for the fallback either).
        _used_3call = False
        if False and hasattr(llm_client, "propose_linked_entities_3call"):
            try:
                proposals = llm_client.propose_linked_entities_3call(
                    target_profile, sector_hints=sector_hints,
                    available_markets=_available_markets,
                )
                _used_3call = bool(proposals)
            except Exception as exc:
                logger.warning("3-call entity discovery failed: %s; falling back to single call", exc)

        # Fall back to single-call if 3-call is unavailable or failed
        if not _used_3call:
            proposals = llm_client.propose_linked_entities(
                target_profile, sector_hints=sector_hints,
                available_markets=_available_markets,
            )

        logger.info(
            "LLM proposed entities for %d groups (%s): %s",
            len(proposals),
            "3-call" if _used_3call else "single-call",
            {g: len(v) for g, v in proposals.items()},
        )
    else:
        logger.info("No LLM client -- skipping entity proposals")

    # P10: Static competitor fallback when LLM proposals are empty.
    # When entity discovery fails (LLM timeout, API rate limit, no key),
    # use curated competitor lists for well-known companies.
    _target_ticker = (target_profile.get("ticker") or "").upper()
    if not proposals.get("competitors") and _target_ticker in _STATIC_COMPETITORS:
        proposals["competitors"] = _STATIC_COMPETITORS[_target_ticker]
        logger.info(
            "Static competitor fallback for %s: %s",
            _target_ticker, proposals["competitors"],
        )

    # ------------------------------------------------------------------
    # 2. Resolve each proposal via PIT provider search
    # ------------------------------------------------------------------
    for group in RELATIONSHIP_GROUPS:
        if group in result.linked:
            logger.debug("Group '%s' already resolved from checkpoint", group)
            continue

        items = proposals.get(group, [])
        resolved: list[LinkedEntity] = []
        group_calls = 0

        for item in items:
            if group_calls >= budget_per_group:
                logger.info("Budget exhausted for group '%s'", group)
                break
            if global_calls >= budget_global:
                logger.info("Global search budget exhausted")
                break

            # Extract name and optional market routing info from LLM
            if isinstance(item, dict):
                name = item.get("name", "")
                _entity_market_id = item.get("market_id", "")
                _entity_ticker = item.get("ticker", "")
            else:
                name = str(item)
                _entity_market_id = ""
                _entity_ticker = ""

            if not name:
                continue

            entity = None

            # Path A: Direct routing when LLM provided market_id
            # (1 targeted API call instead of brute-force 25-wrapper loop)
            if _entity_market_id:
                entity = _resolve_entity_direct(
                    {"name": name, "market_id": _entity_market_id, "ticker": _entity_ticker},
                    group, secrets, target_country, target_sector,
                )

            # Path B: Fallback to cross-region search (original behavior)
            if entity is None:
                entity = _resolve_entity_cross_region(
                    name, group, pit_client, all_clients,
                    target_country, target_sector,
                )

            group_calls += 1
            global_calls += 1

            if entity is not None:
                # Deduplicate by ISIN
                if entity.isin and entity.isin != target_isin:
                    existing_isins = {e.isin for e in resolved}
                    if entity.isin not in existing_isins:
                        resolved.append(entity)
                        logger.info(
                            "  [%s] Resolved: %s (%s) score=%d",
                            group, entity.name, entity.isin, entity.match_score,
                        )
            else:
                result.dropped_low_score.append({
                    "query": name,
                    "group": group,
                    "reason": "below_threshold",
                })

        result.linked[group] = resolved

        # Checkpoint after each group
        progress["resolved"][group] = [
            {
                "isin": e.isin, "ticker": e.ticker, "name": e.name,
                "country": e.country, "sector": e.sector,
                "relationship_group": e.relationship_group,
                "match_score": e.match_score, "market_cap": e.market_cap,
                "market_id": e.market_id,
            }
            for e in resolved
        ]
        progress["search_calls"] = global_calls
        _save_progress(progress)

    result.search_calls_used = global_calls

    # ------------------------------------------------------------------
    # 3. Fallback: if no competitors found, use sector peers
    # ------------------------------------------------------------------
    competitors = result.linked.get("competitors", [])
    if not competitors:
        logger.warning("No competitors resolved -- triggering peer fallback")
        # Try static whale competitor registry first (zero API calls)
        peers = _load_static_competitors(
            target_ticker=target_profile.get("ticker", ""),
            target_sector=target_sector,
            target_industry=target_profile.get("industry", ""),
        )
        if not peers:
            # Fall back to SIC code sector peer search
            peers = _fallback_sector_peers(
                target_isin, pit_client, target_sector=target_sector,
            )
        result.linked["competitors"] = peers
        progress["resolved"]["competitors"] = [
            {
                "isin": e.isin, "ticker": e.ticker, "name": e.name,
                "country": e.country, "sector": e.sector,
                "relationship_group": e.relationship_group,
                "match_score": e.match_score, "market_cap": e.market_cap,
                "market_id": e.market_id,
            }
            for e in peers
        ]
        _save_progress(progress)

    # Summary
    total = sum(len(v) for v in result.linked.values())
    logger.info(
        "Discovery complete: %d entities across %d groups, %d search calls, %d dropped",
        total,
        sum(1 for v in result.linked.values() if v),
        result.search_calls_used,
        len(result.dropped_low_score),
    )

    return result


def get_all_linked_isins(result: DiscoveryResult) -> list[str]:
    """Extract a flat list of unique ISINs from a discovery result."""
    seen: set[str] = set()
    isins: list[str] = []
    for entities in result.linked.values():
        for e in entities:
            if e.isin and e.isin not in seen:
                seen.add(e.isin)
                isins.append(e.isin)
    return isins

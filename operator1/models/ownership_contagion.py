"""Ownership Contagion Scorer -- institutional overlap and crowded trade analysis.

Computes signals that the existing graph_risk module cannot produce
(graph_risk models *business* relationships; this models *ownership* overlap):

1. **MHHI Delta** (Azar, Schmalz & Tecu 2018): common ownership index
   between the target and its competitors.
2. **Bipartite Network Centrality** (Anton & Polk 2014): models the
   institution-company ownership graph using ``networkx.bipartite``.
3. **Crowded Trade Score** (Khandani & Lo 2011): ownership concentration
   * total institutional % / liquidation capacity.
4. **Liquidation Pressure**: estimated days for top-5 holders to exit
   at 25% participation rate, adjusted by Amihud illiquidity.

Academic refs:
  - Azar, Schmalz & Tecu 2018 (common ownership and competition)
  - Anton & Polk 2014 (connected stocks via institutional ownership)
  - Khandani & Lo 2011 (crowded trades)
  - Amihud 2002 (illiquidity ratio)

No external dependencies beyond numpy/pandas/networkx (already installed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PARTICIPATION_RATE: float = 0.25  # 25% of daily volume
CROWDING_THRESHOLD: float = 0.6           # above this -> crowded trade flag
LIQUIDATION_DAYS_CAP: float = 252.0       # cap at 1 year


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class OwnershipContagionResult:
    """Container for ownership contagion analysis results."""

    # MHHI delta (Azar et al 2018)
    mhhi_delta: float = 0.0
    mhhi_pairwise: dict[str, float] = field(default_factory=dict)
    shared_institutions: list[str] = field(default_factory=list)
    n_shared_institutions: int = 0

    # Bipartite network (Anton & Polk 2014)
    target_bipartite_centrality: float = 0.0
    ownership_network_density: float = 0.0
    most_connected_institution: str = ""
    institution_influence_scores: dict[str, float] = field(default_factory=dict)

    # Crowded trade detection (Khandani & Lo 2011)
    crowding_score: float = 0.0
    crowded_trade_flag: bool = False

    # Liquidation pressure
    liquidation_days: float = 0.0
    liquidation_risk: float = 0.0
    top5_shares_total: int = 0
    avg_daily_volume: float = 0.0
    participation_rate: float = DEFAULT_PARTICIPATION_RATE

    # Metadata
    n_target_holders: int = 0
    n_competitors_with_holders: int = 0
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "mhhi_delta": round(self.mhhi_delta, 4),
            "mhhi_pairwise": {k: round(v, 4) for k, v in self.mhhi_pairwise.items()},
            "shared_institutions": self.shared_institutions[:10],
            "n_shared_institutions": self.n_shared_institutions,
            "target_bipartite_centrality": round(self.target_bipartite_centrality, 4),
            "ownership_network_density": round(self.ownership_network_density, 4),
            "most_connected_institution": self.most_connected_institution,
            "crowding_score": round(self.crowding_score, 4),
            "crowded_trade_flag": self.crowded_trade_flag,
            "liquidation_days": round(self.liquidation_days, 1),
            "liquidation_risk": round(self.liquidation_risk, 4),
            "n_target_holders": self.n_target_holders,
            "n_competitors_with_holders": self.n_competitors_with_holders,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Holder name normalization (for cross-source matching)
# ---------------------------------------------------------------------------


def _normalize_holder_name(name: str) -> str:
    """Normalize holder name for fuzzy matching across data sources.

    Removes common suffixes (LLC, LP, Inc, etc.) and lowercases.
    """
    n = name.strip().lower()
    for suffix in (
        " llc", " lp", " l.p.", " inc", " inc.", " corp", " corp.",
        " co.", " ltd", " ltd.", " plc", " group", " management",
        " advisors", " advisory", " partners", " capital",
        " investment", " investments", " asset", " assets",
        " fund", " funds", " trust", " holdings",
        ", inc.", ", llc", ", lp",
    ):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    # Remove "the" prefix
    if n.startswith("the "):
        n = n[4:]
    return n.strip()


def _match_holders(
    target_holders: list[dict[str, Any]],
    comp_holders: list[dict[str, Any]],
    threshold: float = 0.75,
) -> list[tuple[str, str, float, float]]:
    """Match holders between target and competitor by name similarity.

    Returns list of (target_name, comp_name, target_pct, comp_pct)
    for matched pairs above the similarity threshold.
    """
    matches: list[tuple[str, str, float, float]] = []
    used_comp: set[int] = set()

    for th in target_holders:
        t_name = _normalize_holder_name(th.get("name", ""))
        t_pct = float(th.get("percentage", 0))
        if not t_name or t_pct <= 0:
            continue

        best_score = 0.0
        best_idx = -1
        best_ch: dict = {}

        for idx, ch in enumerate(comp_holders):
            if idx in used_comp:
                continue
            c_name = _normalize_holder_name(ch.get("name", ""))
            if not c_name:
                continue

            # Exact match first
            if t_name == c_name:
                score = 1.0
            else:
                score = SequenceMatcher(None, t_name, c_name).ratio()

            if score > best_score:
                best_score = score
                best_idx = idx
                best_ch = ch

        if best_score >= threshold and best_idx >= 0:
            c_pct = float(best_ch.get("percentage", 0))
            matches.append((
                th.get("name", ""),
                best_ch.get("name", ""),
                t_pct,
                c_pct,
            ))
            used_comp.add(best_idx)

    return matches


# ---------------------------------------------------------------------------
# 1. MHHI Delta (Azar, Schmalz & Tecu 2018)
# ---------------------------------------------------------------------------


def _compute_mhhi_delta(
    target_holders: list[dict[str, Any]],
    competitor_holders: dict[str, list[dict[str, Any]]],
) -> tuple[float, dict[str, float], list[str], int]:
    """Compute Modified Herfindahl-Hirschman Index delta.

    Simplified formula for passive institutional holders where
    control weight = profit weight = ownership percentage:

        MHHI_pair(i,j) = sum_k(s_ki * s_kj) / sum_k(s_ki^2)

    where s_ki = institution k's stake in firm i.

    Returns (overall_mhhi, per_competitor_mhhi, shared_institutions, n_shared).
    """
    if not target_holders or not competitor_holders:
        return 0.0, {}, [], 0

    # Build target institution map
    target_map: dict[str, float] = {}
    for h in target_holders:
        name = _normalize_holder_name(h.get("name", ""))
        pct = float(h.get("percentage", 0))
        if name and pct > 0:
            target_map[name] = pct

    if not target_map:
        return 0.0, {}, [], 0

    # Denominator: sum of s_ki^2 for target
    denom = sum(v ** 2 for v in target_map.values())
    if denom < 1e-12:
        return 0.0, {}, [], 0

    all_shared: set[str] = set()
    pairwise: dict[str, float] = {}

    for comp_id, comp_holders_list in competitor_holders.items():
        if not comp_holders_list:
            continue

        # Match holders between target and this competitor
        matched = _match_holders(target_holders, comp_holders_list)

        if not matched:
            pairwise[comp_id] = 0.0
            continue

        # MHHI numerator: sum of s_ki * s_kj for shared institutions
        numerator = 0.0
        for t_name, c_name, t_pct, c_pct in matched:
            numerator += t_pct * c_pct
            all_shared.add(t_name)

        mhhi_pair = numerator / denom
        pairwise[comp_id] = min(1.0, mhhi_pair)

    # Overall MHHI delta = average across competitor pairs
    if pairwise:
        overall = sum(pairwise.values()) / len(pairwise)
    else:
        overall = 0.0

    shared_list = sorted(all_shared)

    return overall, pairwise, shared_list, len(shared_list)


# ---------------------------------------------------------------------------
# 2. Bipartite Network Centrality (Anton & Polk 2014)
# ---------------------------------------------------------------------------


def _compute_bipartite_centrality(
    target_holders: list[dict[str, Any]],
    competitor_holders: dict[str, list[dict[str, Any]]],
) -> tuple[float, float, str, dict[str, float]]:
    """Build bipartite institution-company ownership network.

    Uses networkx.bipartite for centrality and projected graph density.

    Returns (target_centrality, network_density, most_connected_inst,
             institution_influence_scores).
    """
    try:
        import networkx as nx
    except ImportError:
        logger.debug("networkx not installed, skipping bipartite centrality")
        return 0.0, 0.0, "", {}

    G = nx.Graph()

    # Collect all institutions
    institutions: dict[str, set[str]] = {}  # inst_name -> set of companies

    def _add_edges(company_id: str, holders: list[dict[str, Any]]) -> None:
        G.add_node(company_id, bipartite=1)
        for h in holders:
            name = _normalize_holder_name(h.get("name", ""))
            pct = float(h.get("percentage", 0))
            if not name or pct <= 0:
                continue
            G.add_node(name, bipartite=0)
            G.add_edge(name, company_id, weight=pct)
            institutions.setdefault(name, set()).add(company_id)

    _add_edges("__target__", target_holders)
    for comp_id, holders in competitor_holders.items():
        _add_edges(comp_id, holders)

    if G.number_of_nodes() < 3:
        return 0.0, 0.0, "", {}

    comp_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 1}
    inst_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}

    if not comp_nodes or not inst_nodes:
        return 0.0, 0.0, "", {}

    # Degree centrality for companies
    try:
        deg_c = nx.bipartite.degree_centrality(G, comp_nodes)
        target_centrality = deg_c.get("__target__", 0.0)
    except Exception:
        target_centrality = 0.0

    # Network density (of the projected company graph)
    try:
        projected = nx.bipartite.weighted_projected_graph(G, comp_nodes)
        density = nx.density(projected) if projected.number_of_nodes() > 1 else 0.0
    except Exception:
        density = 0.0

    # Most connected institution (by number of companies held)
    most_connected = ""
    max_connections = 0
    influence_scores: dict[str, float] = {}
    for inst, companies in institutions.items():
        n = len(companies)
        influence_scores[inst] = n / max(len(comp_nodes), 1)
        if n > max_connections:
            max_connections = n
            most_connected = inst

    return target_centrality, density, most_connected, influence_scores


# ---------------------------------------------------------------------------
# 3. Crowded Trade Score (Khandani & Lo 2011)
# ---------------------------------------------------------------------------


def _compute_crowding_score(
    inst_ownership_pct: float | None,
    inst_top5_concentration: float | None,
    avg_daily_volume: float,
    close_price: float,
    top5_shares: int,
) -> tuple[float, bool]:
    """Compute crowded trade fragility score.

    crowding = (HHI * ownership_fraction) * log(1 + liquidation_days / 20)

    Returns (score, flag).
    """
    if inst_ownership_pct is None or inst_top5_concentration is None:
        return 0.0, False

    ownership_fraction = inst_ownership_pct / 100.0
    ownership_crowding = inst_top5_concentration * ownership_fraction

    # Liquidation days factor
    if avg_daily_volume > 0 and top5_shares > 0:
        liq_days = top5_shares / (avg_daily_volume * DEFAULT_PARTICIPATION_RATE)
        liq_days = min(liq_days, LIQUIDATION_DAYS_CAP)
    else:
        liq_days = 0.0

    # Combined score: ownership crowding * liquidation difficulty
    raw_score = ownership_crowding * np.log1p(liq_days / 20.0)

    # Sigmoid normalization to [0, 1]
    score = float(1.0 / (1.0 + np.exp(-5.0 * (raw_score - 0.3))))

    flag = score > CROWDING_THRESHOLD

    return score, flag


# ---------------------------------------------------------------------------
# 4. Liquidation Pressure
# ---------------------------------------------------------------------------


def _compute_liquidation_pressure(
    target_holders: list[dict[str, Any]],
    avg_daily_volume: float,
    participation_rate: float = DEFAULT_PARTICIPATION_RATE,
) -> tuple[float, float, int]:
    """Estimate days to liquidate top-5 institutional positions.

    Returns (liquidation_days, liquidation_risk, top5_shares_total).
    """
    if not target_holders or avg_daily_volume <= 0:
        return 0.0, 0.0, 0

    # Sort by shares, take top 5
    sorted_holders = sorted(
        target_holders,
        key=lambda h: float(h.get("shares", 0)),
        reverse=True,
    )[:5]

    top5_shares = sum(int(h.get("shares", 0)) for h in sorted_holders)
    if top5_shares <= 0:
        return 0.0, 0.0, 0

    # Days to liquidate at participation rate
    tradeable_per_day = avg_daily_volume * participation_rate
    if tradeable_per_day <= 0:
        return LIQUIDATION_DAYS_CAP, 1.0, top5_shares

    liq_days = float(top5_shares / tradeable_per_day)
    liq_days = min(liq_days, LIQUIDATION_DAYS_CAP)

    # Normalize to [0, 1] risk score
    # 0 days = 0 risk, 20 days = 0.5 risk, 100+ days = ~1.0 risk
    liq_risk = float(1.0 / (1.0 + np.exp(-0.05 * (liq_days - 20))))

    return liq_days, liq_risk, top5_shares


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ownership_contagion(
    target_holders: list[dict[str, Any]],
    competitor_holders: dict[str, list[dict[str, Any]]],
    cache: pd.DataFrame,
    inst_ownership_pct: float | None = None,
    inst_top5_concentration: float | None = None,
) -> OwnershipContagionResult:
    """Compute all ownership contagion metrics.

    Parameters
    ----------
    target_holders:
        List of holder dicts from ``pit_client.get_holders(identifier)``.
    competitor_holders:
        Dict mapping competitor_id -> list of holder dicts.
    cache:
        Daily cache DataFrame (needs ``volume``, ``close``).
    inst_ownership_pct:
        Override total institutional ownership %.  If None, extracted
        from cache or computed from holders.
    inst_top5_concentration:
        Override top-5 HHI.  If None, extracted from cache or computed.

    Returns
    -------
    OwnershipContagionResult with all metrics populated.
    """
    result = OwnershipContagionResult(
        n_target_holders=len(target_holders),
        n_competitors_with_holders=sum(
            1 for v in competitor_holders.values() if v
        ),
    )

    if not target_holders:
        result.error = "No target holder data available"
        logger.info("Ownership contagion: no target holders, returning defaults")
        return result

    try:
        # 1. MHHI Delta
        mhhi, pairwise, shared, n_shared = _compute_mhhi_delta(
            target_holders, competitor_holders,
        )
        result.mhhi_delta = mhhi
        result.mhhi_pairwise = pairwise
        result.shared_institutions = shared
        result.n_shared_institutions = n_shared

        # 2. Bipartite network centrality
        centrality, density, most_connected, influence = _compute_bipartite_centrality(
            target_holders, competitor_holders,
        )
        result.target_bipartite_centrality = centrality
        result.ownership_network_density = density
        result.most_connected_institution = most_connected
        result.institution_influence_scores = influence

        # 3. Extract cache values for crowding + liquidation
        avg_vol = 0.0
        close_price = 0.0
        if "volume" in cache.columns and cache["volume"].notna().any():
            avg_vol = float(cache["volume"].dropna().mean())
        if "close" in cache.columns and cache["close"].notna().any():
            close_price = float(cache["close"].dropna().iloc[-1])

        result.avg_daily_volume = avg_vol

        # Get ownership stats from cache if not provided
        if inst_ownership_pct is None:
            pct_col = cache.get("inst_ownership_pct")
            if pct_col is not None and pct_col.notna().any():
                inst_ownership_pct = float(pct_col.dropna().iloc[-1])
            elif target_holders:
                inst_ownership_pct = sum(
                    float(h.get("percentage", 0)) for h in target_holders
                )

        if inst_top5_concentration is None:
            hhi_col = cache.get("inst_top5_concentration")
            if hhi_col is not None and hhi_col.notna().any():
                inst_top5_concentration = float(hhi_col.dropna().iloc[-1])
            elif target_holders:
                top5 = sorted(target_holders, key=lambda h: float(h.get("percentage", 0)), reverse=True)[:5]
                total_pct = sum(float(h.get("percentage", 0)) for h in top5)
                if total_pct > 0:
                    inst_top5_concentration = sum(
                        (float(h.get("percentage", 0)) / total_pct) ** 2
                        for h in top5
                    )

        # Top-5 total shares
        sorted_holders = sorted(
            target_holders,
            key=lambda h: float(h.get("shares", 0)),
            reverse=True,
        )[:5]
        top5_shares = sum(int(h.get("shares", 0)) for h in sorted_holders)

        # 3. Crowding score
        score, flag = _compute_crowding_score(
            inst_ownership_pct,
            inst_top5_concentration,
            avg_vol,
            close_price,
            top5_shares,
        )
        result.crowding_score = score
        result.crowded_trade_flag = flag

        # 4. Liquidation pressure
        liq_days, liq_risk, top5_total = _compute_liquidation_pressure(
            target_holders, avg_vol,
        )
        result.liquidation_days = liq_days
        result.liquidation_risk = liq_risk
        result.top5_shares_total = top5_total

        logger.info(
            "Ownership contagion: MHHI=%.3f, shared=%d institutions, "
            "crowding=%.3f (flag=%s), liquidation=%.0f days (risk=%.3f), "
            "bipartite_centrality=%.3f, network_density=%.3f",
            mhhi, n_shared, score, flag, liq_days, liq_risk,
            centrality, density,
        )

    except Exception as exc:
        logger.warning("Ownership contagion computation failed: %s", exc)
        result.available = False
        result.error = str(exc)

    return result


def inject_contagion_into_cache(
    cache: pd.DataFrame,
    result: OwnershipContagionResult,
) -> pd.DataFrame:
    """Add ownership contagion columns to the daily cache.

    All columns are constant across the daily index (quarterly snapshot).
    """
    cache["inst_mhhi_delta"] = result.mhhi_delta
    cache["inst_bipartite_centrality"] = result.target_bipartite_centrality
    cache["inst_ownership_network_density"] = result.ownership_network_density
    cache["inst_crowding_score"] = result.crowding_score
    cache["inst_crowded_trade_flag"] = int(result.crowded_trade_flag)
    cache["inst_liquidation_days"] = result.liquidation_days
    cache["inst_liquidation_risk"] = result.liquidation_risk

    return cache


def get_ownership_edge_weights(
    result: OwnershipContagionResult,
) -> dict[str, float]:
    """Extract ownership overlap weights for graph_risk edge weighting.

    Returns a dict mapping competitor_id -> edge weight (0-1) where
    higher weight means more ownership overlap (higher contagion risk).
    """
    if not result.mhhi_pairwise:
        return {}

    weights: dict[str, float] = {}
    for comp_id, overlap in result.mhhi_pairwise.items():
        # Base 30% contagion + 70% scaled by ownership overlap
        weights[comp_id] = 0.3 + 0.7 * min(overlap, 1.0)

    return weights

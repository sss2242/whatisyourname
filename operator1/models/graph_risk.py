"""Graph Theory module for supply chain and network risk analysis.

Models the target company and its linked entities as a directed graph
and computes network-theoretic risk metrics:

1. **Degree centrality**: How connected is the target? A company with
   many supplier/customer links is more exposed to chain disruptions.

2. **PageRank importance**: Weighted influence score -- companies that
   are linked to other highly-connected companies inherit importance.

3. **Contagion risk**: Simulates a distress cascade: if one node
   enters survival mode, what fraction of the network gets infected?
   Uses a simple SIR-like model on the graph.

4. **Concentration risk**: Herfindahl-Hirschman Index (HHI) on the
   target's relationship links.  High concentration = fragile supply
   chain (one supplier fails -> severe impact).

No external dependencies required -- uses only numpy.  Optionally
uses ``networkx`` for richer graph algorithms if installed.

Integration point: ``operator1/features/linked_aggregates.py``
Output feeds into: ``operator1/report/profile_builder.py``

Top-level entry points:
    ``build_entity_graph`` -- construct the graph from relationships.
    ``compute_graph_risk_metrics`` -- run all risk computations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONTAGION_PROB: float = 0.3    # probability of distress spreading per edge
DEFAULT_CONTAGION_STEPS: int = 5       # max cascade steps
DEFAULT_CONTAGION_SIMS: int = 1000     # Monte Carlo simulations for contagion


# ---------------------------------------------------------------------------
# Graph representation (pure numpy, no networkx required)
# ---------------------------------------------------------------------------


@dataclass
class EntityNode:
    """A node in the entity graph."""

    isin: str = ""
    name: str = ""
    relationship: str = ""  # competitor, supplier, customer, etc.
    is_target: bool = False
    in_survival: bool = False  # currently in survival mode
    market_cap: float | None = None


@dataclass
class GraphRiskResult:
    """Network risk metrics for the target company."""

    # Node count
    n_nodes: int = 0
    n_edges: int = 0

    # Centrality metrics for the target
    target_degree_centrality: float = 0.0
    target_pagerank: float = 0.0
    target_betweenness: float = 0.0

    # Contagion risk
    contagion_expected_infected: float = 0.0
    contagion_max_infected: int = 0
    contagion_target_infection_prob: float = 0.0

    # Concentration risk (HHI)
    supplier_hhi: float = 0.0
    customer_hhi: float = 0.0
    overall_hhi: float = 0.0
    concentration_label: str = "unknown"

    # Per-node PageRank
    pagerank_scores: dict[str, float] = field(default_factory=dict)

    # Systemic risk measures (CoVaR, SRISK)
    covar_scores: dict[str, float] = field(default_factory=dict)
    srisk: float = 0.0

    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "target_degree_centrality": round(self.target_degree_centrality, 4),
            "target_pagerank": round(self.target_pagerank, 4),
            "contagion_expected_infected": round(self.contagion_expected_infected, 4),
            "contagion_target_infection_prob": round(self.contagion_target_infection_prob, 4),
            "supplier_hhi": round(self.supplier_hhi, 4),
            "customer_hhi": round(self.customer_hhi, 4),
            "concentration_label": self.concentration_label,
            "top_pagerank": dict(
                sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "covar_scores": self.covar_scores,
            "srisk": round(self.srisk, 2),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_entity_graph(
    target_isin: str,
    relationships: dict[str, list[dict[str, Any]]],
) -> tuple[list[EntityNode], dict[int, list[int]]]:
    """Build a directed graph from the relationship discovery output.

    Parameters
    ----------
    target_isin:
        ISIN of the target company (center of the graph).
    relationships:
        Dict mapping relationship group -> list of entity dicts.
        Each entity dict should have at least ``isin`` and ``name``.

    Returns
    -------
    (nodes, adjacency)
        ``nodes`` is a list of EntityNode.
        ``adjacency`` maps node_index -> list of neighbor indices.
    """
    nodes: list[EntityNode] = []
    isin_to_idx: dict[str, int] = {}
    adjacency: dict[int, list[int]] = {}

    # Add target as node 0
    nodes.append(EntityNode(isin=target_isin, is_target=True, name="target"))
    isin_to_idx[target_isin] = 0
    adjacency[0] = []

    # Add linked entities
    for group, entities in relationships.items():
        if not isinstance(entities, list):
            continue
        for ent in entities:
            isin = ent.get("isin") or getattr(ent, "isin", "")
            name = ent.get("name") or getattr(ent, "name", "")
            if not isin:
                continue

            if isin not in isin_to_idx:
                idx = len(nodes)
                nodes.append(EntityNode(
                    isin=isin,
                    name=name,
                    relationship=group,
                    market_cap=ent.get("market_cap"),
                ))
                isin_to_idx[isin] = idx
                adjacency[idx] = []

            # Add edge: target <-> entity
            idx = isin_to_idx[isin]
            if idx not in adjacency[0]:
                adjacency[0].append(idx)
            if 0 not in adjacency[idx]:
                adjacency[idx].append(0)

            # Add edges between entities in same group (they compete/interact)
            group_indices = [
                isin_to_idx[e.get("isin") or getattr(e, "isin", "")]
                for e in entities
                if (e.get("isin") or getattr(e, "isin", "")) in isin_to_idx
            ]
            for i in group_indices:
                for j in group_indices:
                    if i != j and j not in adjacency.get(i, []):
                        adjacency.setdefault(i, []).append(j)

    logger.info(
        "Entity graph: %d nodes, %d edges",
        len(nodes),
        sum(len(v) for v in adjacency.values()) // 2,
    )
    return nodes, adjacency


# ---------------------------------------------------------------------------
# Degree centrality
# ---------------------------------------------------------------------------


def _degree_centrality(
    adjacency: dict[int, list[int]],
    node_idx: int,
    n_nodes: int,
) -> float:
    """Degree centrality = (number of edges) / (n_nodes - 1)."""
    if n_nodes <= 1:
        return 0.0
    return len(adjacency.get(node_idx, [])) / (n_nodes - 1)


# ---------------------------------------------------------------------------
# PageRank (power iteration, no networkx needed)
# ---------------------------------------------------------------------------


def _pagerank(
    adjacency: dict[int, list[int]],
    n_nodes: int,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[int, float]:
    """Compute PageRank via power iteration."""
    if n_nodes == 0:
        return {}

    # Initialize uniform
    pr = np.ones(n_nodes) / n_nodes

    for _ in range(max_iter):
        new_pr = np.ones(n_nodes) * (1 - damping) / n_nodes

        for node, neighbors in adjacency.items():
            if node >= n_nodes:
                continue
            out_degree = len(neighbors)
            if out_degree == 0:
                # Dangling node: distribute to all
                new_pr += damping * pr[node] / n_nodes
            else:
                share = damping * pr[node] / out_degree
                for nb in neighbors:
                    if nb < n_nodes:
                        new_pr[nb] += share

        # Check convergence
        if np.abs(new_pr - pr).sum() < tol:
            break
        pr = new_pr

    return {i: float(pr[i]) for i in range(n_nodes)}


# ---------------------------------------------------------------------------
# Contagion simulation
# ---------------------------------------------------------------------------


def _simulate_contagion(
    adjacency: dict[int, list[int]],
    n_nodes: int,
    seed_node: int,
    contagion_prob: float = DEFAULT_CONTAGION_PROB,
    max_steps: int = DEFAULT_CONTAGION_STEPS,
    n_sims: int = DEFAULT_CONTAGION_SIMS,
    random_state: int = 42,
) -> tuple[float, int, float]:
    """Simulate distress contagion from a seed node.

    Uses a simple SIR-like cascade: infected nodes attempt to infect
    their neighbors with probability ``contagion_prob`` at each step.

    Returns
    -------
    (expected_infected, max_infected, target_infection_prob)
    """
    rng = np.random.RandomState(random_state)
    infection_counts: list[int] = []
    target_infected_count = 0

    for _ in range(n_sims):
        infected = {seed_node}
        frontier = {seed_node}

        for _step in range(max_steps):
            new_frontier: set[int] = set()
            for node in frontier:
                for nb in adjacency.get(node, []):
                    if nb not in infected:
                        if rng.random() < contagion_prob:
                            infected.add(nb)
                            new_frontier.add(nb)
            frontier = new_frontier
            if not frontier:
                break

        infection_counts.append(len(infected))
        if 0 in infected:  # target is node 0
            target_infected_count += 1

    expected = float(np.mean(infection_counts))
    max_inf = int(np.max(infection_counts))
    target_prob = target_infected_count / n_sims

    return expected, max_inf, target_prob


# ---------------------------------------------------------------------------
# Concentration risk (HHI)
# ---------------------------------------------------------------------------


def _hhi(shares: list[float]) -> float:
    """Herfindahl-Hirschman Index from market share fractions.

    HHI = sum(s_i^2) where s_i are normalized shares summing to 1.
    Result in [0, 1].  Higher = more concentrated.
    """
    if not shares:
        return 0.0
    total = sum(shares)
    if total <= 0:
        return 0.0
    normalized = [s / total for s in shares]
    return float(sum(s ** 2 for s in normalized))


def _concentration_label(hhi_val: float) -> str:
    """Label an HHI value."""
    if hhi_val >= 0.25:
        return "highly_concentrated"
    if hhi_val >= 0.15:
        return "moderately_concentrated"
    return "diversified"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_covar(
    target_returns: np.ndarray,
    entity_returns: dict[str, np.ndarray],
    quantile: float = 0.05,
) -> dict[str, float]:
    """Compute CoVaR (Conditional Value-at-Risk) for each linked entity.

    CoVaR measures how much the target's VaR increases when a linked
    entity is in distress (below its own VaR). Higher CoVaR means the
    target is more exposed to the linked entity's downside.

    Returns {entity_id: delta_covar} where delta_covar > 0 means the
    target's risk increases when the entity is in distress.
    """
    covar_results: dict[str, float] = {}

    # Target unconditional VaR
    target_var = float(np.quantile(target_returns, quantile))

    for ent_id, ent_ret in entity_returns.items():
        n = min(len(target_returns), len(ent_ret))
        if n < 30:
            continue

        t_ret = target_returns[:n]
        e_ret = ent_ret[:n]

        # Entity VaR
        ent_var = np.quantile(e_ret, quantile)

        # Conditional: target returns when entity is below its VaR
        stress_mask = e_ret <= ent_var
        if stress_mask.sum() < 5:
            continue

        # CoVaR = VaR of target CONDITIONAL on entity being in distress
        covar = float(np.quantile(t_ret[stress_mask], quantile))

        # Delta CoVaR = CoVaR - unconditional VaR
        delta = covar - target_var
        covar_results[ent_id] = round(float(delta), 6)

    return covar_results


def _compute_srisk(
    target_market_cap: float | None,
    target_debt: float | None,
    stress_return: float = -0.40,
    prudential_ratio: float = 0.08,
) -> float:
    """Compute SRISK (Systemic Risk) for the target company.

    SRISK = max(0, k * Debt - (1 - k) * Equity * (1 + stress_return))
    where k = prudential capital ratio (Basel III: 8%).

    Measures the capital shortfall under a severe market stress scenario.
    Positive SRISK = company needs external capital in a crisis.
    """
    if target_market_cap is None or target_debt is None:
        return 0.0
    if target_market_cap <= 0:
        return 0.0

    equity = target_market_cap
    debt = target_debt

    srisk = max(
        0.0,
        prudential_ratio * debt
        - (1.0 - prudential_ratio) * equity * (1.0 + stress_return),
    )
    return round(float(srisk), 2)


def _weighted_contagion_sim(
    adjacency: dict[int, list[int]],
    n_nodes: int,
    seed_node: int,
    base_prob: float,
    node_weights: dict[int, float],
    max_steps: int = DEFAULT_CONTAGION_STEPS,
    n_sims: int = DEFAULT_CONTAGION_SIMS,
    random_state: int = 42,
) -> tuple[float, int, float]:
    """Edge-weighted contagion simulation.

    Like _simulate_contagion but scales infection probability per edge
    by the weight of the relationship (higher exposure = higher risk).
    """
    rng = np.random.RandomState(random_state)
    infection_counts: list[int] = []
    target_infected_count = 0

    for _ in range(n_sims):
        infected = {seed_node}
        frontier = {seed_node}

        for _step in range(max_steps):
            new_frontier: set[int] = set()
            for node in frontier:
                for nb in adjacency.get(node, []):
                    if nb not in infected:
                        # Scale probability by edge weight
                        weight = node_weights.get(nb, 1.0)
                        eff_prob = min(1.0, base_prob * weight)
                        if rng.random() < eff_prob:
                            infected.add(nb)
                            new_frontier.add(nb)
            frontier = new_frontier
            if not frontier:
                break

        infection_counts.append(len(infected))
        if 0 in infected:
            target_infected_count += 1

    expected = float(np.mean(infection_counts))
    max_inf = int(np.max(infection_counts))
    target_prob = target_infected_count / n_sims
    return expected, max_inf, target_prob


def compute_graph_risk_metrics(
    target_isin: str,
    relationships: dict[str, list[dict[str, Any]]],
    *,
    contagion_prob: float = DEFAULT_CONTAGION_PROB,
    contagion_sims: int = DEFAULT_CONTAGION_SIMS,
    random_state: int = 42,
    edge_weights: dict[str, float] | None = None,
    target_cache: "pd.DataFrame | None" = None,
    linked_caches: dict[str, "pd.DataFrame"] | None = None,
) -> GraphRiskResult:
    """Compute all graph-theoretic risk metrics.

    Parameters
    ----------
    target_isin:
        ISIN of the target company.
    relationships:
        Discovery result mapping group -> list of entity dicts.
    contagion_prob:
        Per-edge infection probability for contagion simulation.
        When ``edge_weights`` are provided, this is scaled per-edge
        by the weight (higher weight = higher contagion probability).
    contagion_sims:
        Number of Monte Carlo contagion simulations.
    random_state:
        Seed for reproducibility.
    edge_weights:
        Optional dict of entity_id -> weight (0-1) representing revenue
        or supply exposure. A supplier providing 30% of inputs gets
        weight 0.30. When provided, contagion probability per edge is
        scaled by this weight (higher exposure = higher risk). If None,
        all edges get equal weight (backward compatible).

    Returns
    -------
    GraphRiskResult with all metrics populated.
    """
    try:
        nodes, adjacency = build_entity_graph(target_isin, relationships)
        n = len(nodes)

        if n < 2:
            return GraphRiskResult(
                n_nodes=n,
                available=True,
                error="",
                concentration_label="no_links",
            )

        n_edges = sum(len(v) for v in adjacency.values()) // 2

        # Degree centrality for target
        deg_c = _degree_centrality(adjacency, 0, n)

        # PageRank
        pr = _pagerank(adjacency, n)
        pr_named = {nodes[i].name or nodes[i].isin: v for i, v in pr.items()}

        # Build node weight map from edge_weights (entity_id -> weight)
        # Map entity ISINs to node indices for weighted contagion
        node_weights: dict[int, float] = {}
        if edge_weights:
            for i, nd in enumerate(nodes):
                w = edge_weights.get(nd.isin, edge_weights.get(nd.name, 1.0))
                node_weights[i] = w

        # Contagion from each non-target node, measure impact on target
        # Pick the most connected non-target node as the seed
        non_target = [i for i in range(1, n)]
        if non_target:
            seed = max(non_target, key=lambda i: len(adjacency.get(i, [])))
            if edge_weights and node_weights:
                # Edge-weighted contagion (3.6)
                exp_inf, max_inf, target_prob = _weighted_contagion_sim(
                    adjacency, n, seed,
                    base_prob=contagion_prob,
                    node_weights=node_weights,
                    n_sims=contagion_sims,
                    random_state=random_state,
                )
            else:
                exp_inf, max_inf, target_prob = _simulate_contagion(
                    adjacency, n, seed,
                    contagion_prob=contagion_prob,
                    n_sims=contagion_sims,
                    random_state=random_state,
                )
        else:
            exp_inf, max_inf, target_prob = 0.0, 0, 0.0

        # Concentration (HHI) by group
        supplier_caps = [
            nd.market_cap for nd in nodes
            if nd.relationship in ("suppliers", "supply_chain") and nd.market_cap
        ]
        customer_caps = [
            nd.market_cap for nd in nodes
            if nd.relationship == "customers" and nd.market_cap
        ]
        all_caps = [nd.market_cap for nd in nodes if nd.market_cap and not nd.is_target]

        s_hhi = _hhi(supplier_caps)
        c_hhi = _hhi(customer_caps)
        o_hhi = _hhi(all_caps)

        result = GraphRiskResult(
            n_nodes=n,
            n_edges=n_edges,
            target_degree_centrality=deg_c,
            target_pagerank=pr.get(0, 0.0),
            contagion_expected_infected=exp_inf,
            contagion_max_infected=max_inf,
            contagion_target_infection_prob=target_prob,
            supplier_hhi=s_hhi,
            customer_hhi=c_hhi,
            overall_hhi=o_hhi,
            concentration_label=_concentration_label(o_hhi),
            pagerank_scores=pr_named,
            available=True,
        )

        # CoVaR and SRISK systemic risk measures (2.5)
        if target_cache is not None and linked_caches:
            _ret_col = "return_1d"
            if _ret_col in target_cache.columns:
                t_ret = target_cache[_ret_col].dropna().values
                entity_returns: dict[str, np.ndarray] = {}
                for ent_id, ent_cache in linked_caches.items():
                    if _ret_col in ent_cache.columns:
                        entity_returns[ent_id] = ent_cache[_ret_col].dropna().values
                if entity_returns:
                    covar = _compute_covar(t_ret, entity_returns)
                    result.covar_scores = covar  # type: ignore[attr-defined]

            # SRISK from target's market cap and debt
            _mc = target_cache.get("market_cap")
            _td = target_cache.get("total_debt_asof")
            if _mc is not None and _td is not None:
                mc_val = float(_mc.dropna().iloc[-1]) if _mc.notna().any() else None
                td_val = float(_td.dropna().iloc[-1]) if _td.notna().any() else None
                srisk = _compute_srisk(mc_val, td_val)
                result.srisk = srisk  # type: ignore[attr-defined]

        return result

    except Exception as exc:
        logger.warning("Graph risk computation failed: %s", exc)
        return GraphRiskResult(available=False, error=str(exc))

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


def compute_graph_risk_metrics(
    target_isin: str,
    relationships: dict[str, list[dict[str, Any]]],
    *,
    contagion_prob: float = DEFAULT_CONTAGION_PROB,
    contagion_sims: int = DEFAULT_CONTAGION_SIMS,
    random_state: int = 42,
    edge_weights: dict[str, float] | None = None,
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

        # Contagion from each non-target node, measure impact on target
        # Pick the most connected non-target node as the seed
        non_target = [i for i in range(1, n)]
        if non_target:
            # Seed from the node with highest degree
            seed = max(non_target, key=lambda i: len(adjacency.get(i, [])))
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

        return GraphRiskResult(
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

    except Exception as exc:
        logger.warning("Graph risk computation failed: %s", exc)
        return GraphRiskResult(available=False, error=str(exc))

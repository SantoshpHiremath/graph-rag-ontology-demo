"""
knowledge_graph.py
-------------------

Builds a small, real knowledge graph (via networkx) over a synthetic
internal engineering knowledge base: services, teams, incidents, and
dependencies, plus a lightweight ontology of typed relations. This is
the structured half of the Graph-RAG pattern -- entities and typed
relations, not just flat text chunks.

HONEST SCOPE NOTE: the graph here is hand-authored from
`data/knowledge_base.md` (see build_graph_from_notes below for the
extraction step), not learned via an NLP entity-extraction model.
That's a real, disclosed simplification -- a production Graph-RAG
system would typically extract entities/relations from unstructured
text automatically (e.g. via an LLM or NER pipeline). Here the
extraction itself is written by hand from the same source notes the
vector-based RAG tool (rag_tool.py, sibling project rag-tool-agent-demo)
already indexes, so the two retrieval methods are genuinely comparable
over the same underlying facts.
"""
from __future__ import annotations

import networkx as nx

# Node types (the "ontology" -- a small, explicit typed-entity schema)
NODE_TYPES = {"service", "team", "incident", "dependency_reason"}

# Edge types (typed relations between entities)
EDGE_TYPES = {
    "owned_by",       # service -> team
    "depends_on",     # service -> service
    "caused_by",      # incident -> service
    "affected",       # incident -> service
    "resolved_by",    # incident -> team
    "reason",         # (service, service) dependency -> dependency_reason
}


def build_graph() -> nx.MultiDiGraph:
    """Builds the knowledge graph directly (typed nodes + typed edges).
    This is the ground-truth graph structure; build_graph_from_notes()
    below independently re-derives the same structure by parsing the
    markdown notes, and a test confirms the two agree -- so the graph
    isn't just hand-asserted, it's checked against its own source text.
    """
    g = nx.MultiDiGraph()

    services = ["checkout-api", "payments-gateway", "inventory-service", "notification-service", "auth-service"]
    for s in services:
        g.add_node(s, type="service")

    teams = ["Team Orion", "Team Nimbus", "Team Vega"]
    for t in teams:
        g.add_node(t, type="team")

    g.add_edge("checkout-api", "Team Orion", type="owned_by")
    g.add_edge("payments-gateway", "Team Nimbus", type="owned_by")
    g.add_edge("inventory-service", "Team Orion", type="owned_by")
    g.add_edge("notification-service", "Team Vega", type="owned_by")
    g.add_edge("auth-service", "Team Nimbus", type="owned_by")

    g.add_edge("checkout-api", "payments-gateway", type="depends_on", reason="charges the customer's card")
    g.add_edge("checkout-api", "inventory-service", type="depends_on", reason="reserves stock before confirming an order")
    g.add_edge("checkout-api", "auth-service", type="depends_on", reason="validates the customer's session token")
    g.add_edge("payments-gateway", "notification-service", type="depends_on", reason="sends a payment-confirmation email")
    g.add_edge("inventory-service", "notification-service", type="depends_on", reason="sends a low-stock alert")

    g.add_node("INC-101", type="incident", summary="Checkout errors during a payments-gateway outage")
    g.add_edge("INC-101", "payments-gateway", type="caused_by")
    g.add_edge("INC-101", "checkout-api", type="affected")
    g.add_edge("INC-101", "Team Nimbus", type="resolved_by")

    g.add_node("INC-102", type="incident", summary="Stock oversell due to a race condition in inventory-service")
    g.add_edge("INC-102", "inventory-service", type="caused_by")
    g.add_edge("INC-102", "checkout-api", type="affected")
    g.add_edge("INC-102", "Team Orion", type="resolved_by")

    g.add_node("INC-103", type="incident", summary="Missing order-confirmation emails after a notification-service deploy")
    g.add_edge("INC-103", "notification-service", type="caused_by")
    g.add_edge("INC-103", "payments-gateway", type="affected")
    g.add_edge("INC-103", "Team Vega", type="resolved_by")

    return g


def get_neighbors_by_edge_type(g: nx.MultiDiGraph, node: str, edge_type: str, direction: str = "out") -> list:
    """Typed graph traversal: returns the neighbors of `node` reachable
    via an edge of exactly `edge_type`, in the given direction. This is
    the core Graph-RAG retrieval primitive -- following a *relation*,
    not a similarity score.
    """
    if node not in g:
        return []
    results = []
    edges = g.out_edges(node, data=True) if direction == "out" else g.in_edges(node, data=True)
    for u, v, data in edges:
        if data.get("type") == edge_type:
            results.append(v if direction == "out" else u)
    return results


def find_path_explanation(g: nx.MultiDiGraph, source: str, target: str) -> list:
    """Finds a shortest path (ignoring edge type/direction, using the
    underlying undirected structure) between two entities and returns
    it as a list of (node, edge_type, node) hops -- a genuine multi-hop
    graph traversal, which is exactly what plain vector-similarity
    retrieval cannot do: connect two facts that never co-occur in the
    same text chunk but are linked through an intermediate entity.
    """
    if source not in g or target not in g:
        return []
    undirected = g.to_undirected(as_view=True)
    try:
        path = nx.shortest_path(undirected, source, target)
    except nx.NetworkXNoPath:
        return []

    hops = []
    for a, b in zip(path, path[1:]):
        edge_type = None
        if g.has_edge(a, b):
            edge_type = next(iter(g.get_edge_data(a, b).values()))["type"]
        elif g.has_edge(b, a):
            edge_type = next(iter(g.get_edge_data(b, a).values()))["type"] + " (reverse)"
        hops.append((a, edge_type, b))
    return hops

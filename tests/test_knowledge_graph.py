"""Tests for src/knowledge_graph.py -- the hand-built graph, typed
traversal primitives, and the cross-check against the independently
regex-parsed graph from notes_parser.py."""
from __future__ import annotations

from src.knowledge_graph import (
    build_graph,
    find_path_explanation,
    get_neighbors_by_edge_type,
)
from src.notes_parser import parse_graph_from_notes


class TestBuildGraph:
    def test_graph_contains_all_services(self):
        g = build_graph()
        services = {n for n, d in g.nodes(data=True) if d.get("type") == "service"}
        assert services == {
            "checkout-api", "payments-gateway", "inventory-service",
            "notification-service", "auth-service",
        }

    def test_graph_contains_all_teams(self):
        g = build_graph()
        teams = {n for n, d in g.nodes(data=True) if d.get("type") == "team"}
        assert teams == {"Team Orion", "Team Nimbus", "Team Vega"}

    def test_graph_contains_all_incidents(self):
        g = build_graph()
        incidents = {n for n, d in g.nodes(data=True) if d.get("type") == "incident"}
        assert incidents == {"INC-101", "INC-102", "INC-103"}

    def test_dependency_edges_have_reason_attribute(self):
        g = build_graph()
        edge_data = g.get_edge_data("checkout-api", "payments-gateway")
        reasons = [d["reason"] for d in edge_data.values() if d["type"] == "depends_on"]
        assert reasons and "charges the customer's card" in reasons[0]


class TestGetNeighborsByEdgeType:
    def test_outgoing_owned_by_edge(self):
        g = build_graph()
        result = get_neighbors_by_edge_type(g, "checkout-api", "owned_by", direction="out")
        assert result == ["Team Orion"]

    def test_outgoing_depends_on_edges(self):
        g = build_graph()
        result = get_neighbors_by_edge_type(g, "checkout-api", "depends_on", direction="out")
        assert set(result) == {"payments-gateway", "inventory-service", "auth-service"}

    def test_incoming_direction(self):
        g = build_graph()
        result = get_neighbors_by_edge_type(g, "payments-gateway", "depends_on", direction="in")
        assert result == ["checkout-api"]

    def test_unknown_node_returns_empty_list(self):
        g = build_graph()
        result = get_neighbors_by_edge_type(g, "nonexistent-service", "owned_by")
        assert result == []

    def test_no_matching_edge_type_returns_empty_list(self):
        g = build_graph()
        result = get_neighbors_by_edge_type(g, "notification-service", "depends_on", direction="out")
        assert result == []


class TestFindPathExplanation:
    def test_direct_two_hop_path(self):
        g = build_graph()
        path = find_path_explanation(g, "checkout-api", "payments-gateway")
        assert path == [("checkout-api", "depends_on", "payments-gateway")]

    def test_no_path_between_unrelated_nodes(self):
        g = build_graph()
        path = find_path_explanation(g, "checkout-api", "nonexistent-node")
        assert path == []

    def test_path_between_same_node_is_empty(self):
        g = build_graph()
        path = find_path_explanation(g, "checkout-api", "checkout-api")
        assert path == []


class TestParsedGraphMatchesHandBuiltGraph:
    """The core cross-check: the graph built by hand in build_graph()
    must exactly match the graph independently re-derived by parsing
    data/knowledge_base.md's actual text. This is what makes the graph
    trustworthy -- it's checked against its own stated source, not just
    asserted."""

    def test_parsed_graph_matches_hand_built_graph(self):
        with open("data/knowledge_base.md", "r", encoding="utf-8") as f:
            text = f.read()

        parsed = parse_graph_from_notes(text)
        hand_built = build_graph()

        assert set(parsed.nodes()) == set(hand_built.nodes())

        parsed_edges = {(u, v, d["type"]) for u, v, d in parsed.edges(data=True)}
        hand_edges = {(u, v, d["type"]) for u, v, d in hand_built.edges(data=True)}
        assert parsed_edges == hand_edges

    def test_parsed_graph_has_nonzero_edges(self):
        # Regression guard for the real bug found during development:
        # a line-anchored regex silently matched 0 wrapped bullets,
        # producing a graph with only 5 of 19 edges without raising
        # any error. This test would have caught that immediately.
        with open("data/knowledge_base.md", "r", encoding="utf-8") as f:
            text = f.read()
        parsed = parse_graph_from_notes(text)
        assert parsed.number_of_edges() == 19

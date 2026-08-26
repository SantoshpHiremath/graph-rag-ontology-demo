"""Tests for src/graph_retrieval.py -- typed graph traversal answering
recognized question patterns, including the path-explanation bug found
and fixed during development."""
from __future__ import annotations

import pytest

from src.graph_retrieval import (
    GraphAnswerNotFound,
    answer_incident_root_cause_owner,
    answer_owning_team_for_dependency,
    route_question,
)
from src.knowledge_graph import build_graph


@pytest.fixture
def graph():
    return build_graph()


class TestAnswerOwningTeamForDependency:
    def test_returns_all_dependencies_and_their_owning_teams(self, graph):
        result = answer_owning_team_for_dependency(graph, "checkout-api")
        deps = {r["dependency"]: r["owning_team"] for r in result["results"]}
        assert deps == {
            "payments-gateway": "Team Nimbus",
            "inventory-service": "Team Orion",
            "auth-service": "Team Nimbus",
        }

    def test_path_reflects_the_actual_traversal_not_a_shortest_path_search(self, graph):
        # Regression test for the real bug found during development:
        # the path previously came from a separate shortest_path()
        # search over the whole graph, which could return an unrelated
        # (and shorter) path through an incident edge instead of the
        # depends_on -> owned_by chain that actually produced the
        # answer. The path must now always be exactly these two hops.
        result = answer_owning_team_for_dependency(graph, "checkout-api")
        for r in result["results"]:
            assert r["path"] == [
                ("checkout-api", "depends_on", r["dependency"]),
                (r["dependency"], "owned_by", r["owning_team"]),
            ]

    def test_service_with_no_dependencies_raises(self, graph):
        with pytest.raises(GraphAnswerNotFound):
            answer_owning_team_for_dependency(graph, "notification-service")

    def test_unknown_service_raises(self, graph):
        with pytest.raises(GraphAnswerNotFound):
            answer_owning_team_for_dependency(graph, "nonexistent-service")


class TestAnswerIncidentRootCauseOwner:
    def test_derives_owning_team_via_caused_by_then_owned_by(self, graph):
        result = answer_incident_root_cause_owner(graph, "INC-103")
        assert result["root_cause_service"] == "notification-service"
        assert result["owning_team"] == "Team Vega"

    def test_path_reflects_caused_by_and_owned_by_hops(self, graph):
        result = answer_incident_root_cause_owner(graph, "INC-101")
        assert result["path"] == [
            ("INC-101", "caused_by", "payments-gateway"),
            ("payments-gateway", "owned_by", "Team Nimbus"),
        ]

    def test_unknown_incident_raises(self, graph):
        with pytest.raises(GraphAnswerNotFound):
            answer_incident_root_cause_owner(graph, "INC-999")


class TestRouteQuestion:
    def test_routes_dependency_team_question(self, graph):
        result = route_question(graph, "Which team should I contact if checkout-api is having payment problems?")
        assert result["service"] == "checkout-api"

    def test_routes_incident_question(self, graph):
        result = route_question(graph, "Who should I contact about INC-102?")
        assert result["incident"] == "INC-102"

    def test_unrecognized_question_raises(self, graph):
        with pytest.raises(GraphAnswerNotFound):
            route_question(graph, "What is the weather today?")

    def test_incident_pattern_takes_priority_when_both_could_match(self, graph):
        # A question mentioning both an incident ID and asking "which
        # team" should route to the incident answerer, since that's the
        # more specific match.
        result = route_question(graph, "For INC-101, which team should I contact?")
        assert "incident" in result

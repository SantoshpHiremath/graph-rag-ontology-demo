"""Tests for src/compare_retrieval.py -- the end-to-end comparison
between vector and graph retrieval on the same question."""
from __future__ import annotations

from src.compare_retrieval import compare


class TestCompare:
    def test_returns_both_vector_and_graph_results(self):
        result = compare("Which team should I contact if checkout-api is having payment problems?")
        assert result["vector_result"]["top_chunks"]
        assert result["graph_result"] is not None
        assert result["graph_error"] is None

    def test_graph_finds_correct_team_vector_finds_wrong_team(self):
        # The central demonstrated claim of this project.
        result = compare("Which team should I contact if checkout-api is having payment problems?")

        graph_teams = {r["owning_team"] for r in result["graph_result"]["results"]}
        assert "Team Nimbus" in graph_teams  # correct: owns payments-gateway, the actual dependency

        top_vector_chunk = result["vector_result"]["top_chunks"][0]["chunk"]
        assert "Team Orion" in top_vector_chunk  # checkout-api's own (less useful) owner

    def test_unrecognized_question_still_returns_vector_result(self):
        result = compare("What is the weather today?")
        assert result["vector_result"]["top_chunks"]
        assert result["graph_result"] is None
        assert result["graph_error"] is not None

    def test_hand_built_and_parsed_graphs_agree(self):
        result = compare("Who should I contact about INC-101?")
        assert result["graphs_agree"] is True

    def test_incident_question_returns_correct_graph_answer(self):
        result = compare("Who should I contact about INC-102?")
        assert result["graph_result"]["owning_team"] == "Team Orion"

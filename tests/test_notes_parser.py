"""Tests for src/notes_parser.py -- regex-based extraction of typed
graph entities/relations from the markdown knowledge base, including
the line-wrapping bug found and fixed during development."""
from __future__ import annotations

from src.notes_parser import _unwrap_bullets, parse_graph_from_notes


class TestUnwrapBullets:
    def test_single_line_bullet_stays_as_is(self):
        text = "- checkout-api is owned by Team Orion."
        assert _unwrap_bullets(text) == ["- checkout-api is owned by Team Orion."]

    def test_wrapped_bullet_is_joined_into_one_line(self):
        text = (
            "- checkout-api depends on payments-gateway because it charges the\n"
            "  customer's card during checkout."
        )
        result = _unwrap_bullets(text)
        assert len(result) == 1
        assert "charges the customer's card during checkout." in result[0]

    def test_multiple_bullets_each_unwrapped_independently(self):
        text = (
            "- first bullet wraps across\n"
            "  two lines here.\n"
            "- second bullet is short.\n"
        )
        result = _unwrap_bullets(text)
        assert len(result) == 2
        assert "wraps across two lines here." in result[0]
        assert result[1] == "- second bullet is short."

    def test_heading_lines_do_not_get_absorbed_into_a_bullet(self):
        text = (
            "- a bullet.\n"
            "\n"
            "## A Heading\n"
            "\n"
            "- another bullet.\n"
        )
        result = _unwrap_bullets(text)
        assert result == ["- a bullet.", "- another bullet."]


class TestParseGraphFromNotes:
    def test_parses_ownership_edges(self):
        text = "- checkout-api is owned by Team Orion.\n"
        g = parse_graph_from_notes(text)
        assert g.has_edge("checkout-api", "Team Orion")
        edge_data = next(iter(g.get_edge_data("checkout-api", "Team Orion").values()))
        assert edge_data["type"] == "owned_by"

    def test_parses_wrapped_dependency_edges(self):
        text = (
            "- checkout-api depends on payments-gateway because it charges the\n"
            "  customer's card during checkout.\n"
        )
        g = parse_graph_from_notes(text)
        assert g.has_edge("checkout-api", "payments-gateway")
        edge_data = next(iter(g.get_edge_data("checkout-api", "payments-gateway").values()))
        assert edge_data["type"] == "depends_on"
        assert "charges the customer's card" in edge_data["reason"]

    def test_parses_wrapped_incident_edges(self):
        text = (
            "- INC-101: Checkout errors during a payments-gateway outage. Caused by\n"
            "  payments-gateway. Affected checkout-api. Resolved by Team Nimbus.\n"
        )
        g = parse_graph_from_notes(text)
        assert g.nodes["INC-101"]["type"] == "incident"
        assert g.has_edge("INC-101", "payments-gateway")
        assert g.has_edge("INC-101", "checkout-api")
        assert g.has_edge("INC-101", "Team Nimbus")

        types = {d["type"] for _, _, d in g.out_edges("INC-101", data=True)}
        assert types == {"caused_by", "affected", "resolved_by"}

    def test_empty_text_produces_empty_graph(self):
        g = parse_graph_from_notes("")
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_full_knowledge_base_file_parses_without_error(self):
        with open("data/knowledge_base.md", "r", encoding="utf-8") as f:
            text = f.read()
        g = parse_graph_from_notes(text)
        assert g.number_of_nodes() > 0
        assert g.number_of_edges() == 19

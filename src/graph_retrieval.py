"""
graph_retrieval.py
--------------------

The graph-traversal half of the Graph-RAG comparison: answers a small,
fixed set of recognized question patterns by walking typed edges in the
knowledge graph, rather than by similarity search over text chunks.

This module deliberately does NOT use an LLM to decide which traversal
to run -- question routing is via a small keyword-pattern matcher, the
same honest, disclosed simplification pattern used for MockLLMClient's
keyword routing elsewhere in this portfolio. The graph traversal itself
(the actually-interesting part -- multi-hop lookups via typed edges) is
fully real, using the real networkx graph built in knowledge_graph.py.
"""
from __future__ import annotations

import re

from src.knowledge_graph import get_neighbors_by_edge_type


class GraphAnswerNotFound(Exception):
    pass


def answer_owning_team_for_dependency(g, service: str) -> dict:
    """Answers: 'which team should I contact about <service>'s
    dependency on X?' by a genuine two-hop traversal: service
    --depends_on--> dependency --owned_by--> team. This is the specific
    case a flat vector search over text chunks struggles with, since the
    ownership fact and the dependency fact live in different sections of
    the source notes and may not co-retrieve together.

    The returned 'path' is built directly from the two hops actually
    walked to compute the answer -- NOT from a separate shortest-path
    search over the whole graph. An earlier version called
    find_path_explanation(g, service, team) here, which could return a
    *different*, shorter path (e.g. through an unrelated incident edge)
    than the depends_on -> owned_by chain that actually produced the
    answer, making the "explanation" misleading. Found via manual
    testing (the path for checkout-api -> Team Nimbus came back via an
    incident hop, not the dependency chain) and fixed by constructing
    the path from the same two lookups used to derive the answer.
    """
    dependencies = get_neighbors_by_edge_type(g, service, "depends_on", direction="out")
    if not dependencies:
        raise GraphAnswerNotFound(f"No known dependencies for '{service}'.")

    answers = []
    for dep in dependencies:
        owning_teams = get_neighbors_by_edge_type(g, dep, "owned_by", direction="out")
        for team in owning_teams:
            path = [(service, "depends_on", dep), (dep, "owned_by", team)]
            answers.append({"dependency": dep, "owning_team": team, "path": path})
    return {"service": service, "results": answers}


def answer_incident_root_cause_owner(g, incident_id: str) -> dict:
    """Answers: 'who should I contact about incident X?' via
    incident --caused_by--> service --owned_by--> team, a different
    two-hop traversal than the dependency case above. The path is
    likewise built directly from the caused_by/owned_by hops actually
    walked, not from a separate shortest-path search -- for the same
    reason described in answer_owning_team_for_dependency's docstring
    (a shortest-path search could return the incident's own
    resolved_by edge instead, which answers a related but different
    question).
    """
    caused_by = get_neighbors_by_edge_type(g, incident_id, "caused_by", direction="out")
    if not caused_by:
        raise GraphAnswerNotFound(f"No known root cause for '{incident_id}'.")

    root_service = caused_by[0]
    owning_teams = get_neighbors_by_edge_type(g, root_service, "owned_by", direction="out")
    owning_team = owning_teams[0] if owning_teams else None
    path = (
        [(incident_id, "caused_by", root_service), (root_service, "owned_by", owning_team)]
        if owning_team
        else [(incident_id, "caused_by", root_service)]
    )
    return {
        "incident": incident_id,
        "root_cause_service": root_service,
        "owning_team": owning_team,
        "path": path,
    }


_DEPENDENCY_TEAM_PATTERN = re.compile(
    r"(?:which team|who).{0,40}contact.{0,60}\b([\w-]+-(?:api|service|gateway))\b", re.IGNORECASE
)
_INCIDENT_PATTERN = re.compile(r"\b(INC-\d+)\b")


def route_question(g, question: str) -> dict:
    """Keyword-pattern routing (disclosed simplification, see module
    docstring) to one of the two traversal answerers above. Raises
    GraphAnswerNotFound if the question doesn't match a recognized
    pattern or references an unknown entity -- callers (and tests)
    should expect this as a normal, honest outcome for out-of-scope
    questions, not something to silently swallow.
    """
    incident_match = _INCIDENT_PATTERN.search(question)
    if incident_match:
        return answer_incident_root_cause_owner(g, incident_match.group(1))

    dep_match = _DEPENDENCY_TEAM_PATTERN.search(question)
    if dep_match:
        return answer_owning_team_for_dependency(g, dep_match.group(1))

    raise GraphAnswerNotFound(
        f"Question does not match a recognized graph-traversal pattern: {question!r}"
    )

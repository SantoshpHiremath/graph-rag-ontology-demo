"""
notes_parser.py
-----------------

Parses data/knowledge_base.md and independently re-derives the same
typed entities and relations that knowledge_graph.build_graph()
hand-asserts. This exists so the knowledge graph isn't just an
unverified hand-written data structure -- a test
(tests/test_knowledge_graph.py::test_parsed_graph_matches_hand_built_graph)
confirms the two independently-produced graphs agree on every node and
edge, which is real evidence the graph accurately reflects its stated
source text rather than silently drifting from it.
"""
from __future__ import annotations

import re

import networkx as nx

_OWNERSHIP_RE = re.compile(r"^- (\S+) is owned by (Team \w+)\.$", re.MULTILINE)
_DEPENDENCY_RE = re.compile(
    r"^- (\S+) depends on (\S+) because it (.+?)\.$", re.MULTILINE
)
_INCIDENT_RE = re.compile(
    r"^- (INC-\d+): (.+?)\. Caused by (\S+)\. Affected (\S+)\. Resolved by (Team \w+)\.$",
    re.MULTILINE,
)


def _unwrap_bullets(markdown_text: str) -> list:
    """Markdown source wraps long bullet entries across multiple lines
    (each continuation line has no leading '- '). Re-joins each bullet
    back into a single logical line before regex matching -- without
    this, a line-anchored regex silently matches only bullets short
    enough to fit on one physical line, which is exactly the real bug
    this function was added to fix (found via
    test_parsed_graph_matches_hand_built_graph failing on first run:
    the parser silently produced only 5 of 19 edges, the wrapped
    dependency and incident bullets were dropped entirely).
    """
    lines = markdown_text.split("\n")
    bullets = []
    current = None
    for line in lines:
        if line.startswith("- "):
            if current is not None:
                bullets.append(current)
            current = line
        elif line.strip() and current is not None and not line.startswith("#"):
            current += " " + line.strip()
        else:
            if current is not None:
                bullets.append(current)
                current = None
    if current is not None:
        bullets.append(current)
    return bullets


def parse_graph_from_notes(markdown_text: str) -> nx.MultiDiGraph:
    """Builds a knowledge graph purely by regex-parsing the markdown
    notes -- no hand-asserted graph data, no shared state with
    knowledge_graph.build_graph(). Deliberately simple line-oriented
    parsing (not a general NLP/entity-extraction pipeline) since the
    source notes are structured specifically to support this; see the
    module docstring for the honest scope note on why a production
    Graph-RAG system would extract relations from unstructured text
    differently.
    """
    g = nx.MultiDiGraph()
    unwrapped = "\n".join(_unwrap_bullets(markdown_text))

    for service, team in _OWNERSHIP_RE.findall(unwrapped):
        g.add_node(service, type="service")
        g.add_node(team, type="team")
        g.add_edge(service, team, type="owned_by")

    for source, target, reason in _DEPENDENCY_RE.findall(unwrapped):
        g.add_node(source, type="service")
        g.add_node(target, type="service")
        g.add_edge(source, target, type="depends_on", reason=reason.strip())

    for inc_id, summary, caused_by, affected, resolved_by in _INCIDENT_RE.findall(unwrapped):
        g.add_node(inc_id, type="incident", summary=summary.strip())
        g.add_node(caused_by, type="service")
        g.add_node(affected, type="service")
        g.add_node(resolved_by, type="team")
        g.add_edge(inc_id, caused_by, type="caused_by")
        g.add_edge(inc_id, affected, type="affected")
        g.add_edge(inc_id, resolved_by, type="resolved_by")

    return g

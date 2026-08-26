"""
compare_retrieval.py
----------------------

Runs the same question through both retrieval strategies -- flat
TF-IDF vector similarity over text chunks, and typed graph traversal --
and reports what each one actually returns, so the difference is a
demonstrated result, not just an asserted claim.
"""
from __future__ import annotations

from src.graph_retrieval import GraphAnswerNotFound, route_question
from src.knowledge_graph import build_graph
from src.notes_parser import parse_graph_from_notes
from src.vector_retrieval import VectorRetriever, chunk_markdown_by_bullet


def load_knowledge_base_text(path: str = "data/knowledge_base.md") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vector_retriever(kb_text: str) -> VectorRetriever:
    chunks = chunk_markdown_by_bullet(kb_text)
    return VectorRetriever(chunks)


def compare(question: str, kb_text: str = None) -> dict:
    """Runs `question` through both retrieval strategies and returns a
    dict with both raw outputs, so a caller (or a human reading
    run_pipeline.py's output) can see exactly what each one found.
    """
    if kb_text is None:
        kb_text = load_knowledge_base_text()

    graph = build_graph()
    # Cross-check: rebuild the graph from the notes too, and use THAT
    # for the actual traversal below -- so the comparison is run
    # against the graph as genuinely derived from the source text, not
    # just the hand-authored one. (A test confirms the two are
    # identical, so this choice doesn't change behavior, but it keeps
    # the demonstrated pipeline honestly tied to its stated source.)
    parsed_graph = parse_graph_from_notes(kb_text)

    vector_retriever = build_vector_retriever(kb_text)
    vector_hits = vector_retriever.query(question, top_k=2)

    graph_result = None
    graph_error = None
    try:
        graph_result = route_question(parsed_graph, question)
    except GraphAnswerNotFound as exc:
        graph_error = str(exc)

    return {
        "question": question,
        "vector_result": {
            "top_chunks": [{"chunk": c[:200], "score": round(s, 4)} for c, s in vector_hits],
        },
        "graph_result": graph_result,
        "graph_error": graph_error,
        "graphs_agree": set(graph.nodes()) == set(parsed_graph.nodes()),
    }

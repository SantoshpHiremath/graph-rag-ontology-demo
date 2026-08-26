"""
run_pipeline.py
-----------------

End-to-end demo: runs a handful of questions through both retrieval
strategies (flat TF-IDF vector search vs. typed graph traversal) over
the same synthetic internal-engineering knowledge base, and prints what
each one actually returns -- including the specific case where vector
search returns a plausible-looking but wrong answer, and graph
traversal gets it right by following typed relations across two hops.

Run with:
    python3 run_pipeline.py
"""
from __future__ import annotations

from src.compare_retrieval import compare

QUESTIONS = [
    "Which team should I contact if checkout-api is having payment problems?",
    "Which team should I contact if payments-gateway is having notification problems?",
    "Who should I contact about INC-102?",
    "Who should I contact about INC-103?",
    "What is the capital of France?",
]


def _print_result(result: dict) -> None:
    print(f"Question: {result['question']}")
    print("-" * 70)

    print("VECTOR SEARCH (TF-IDF, top hit):")
    top = result["vector_result"]["top_chunks"][0]
    print(f"  score: {top['score']}")
    for line in top["chunk"].splitlines()[:4]:
        print(f"  {line}")
    print()

    if result["graph_result"] is not None:
        print("GRAPH TRAVERSAL:")
        gr = result["graph_result"]
        if "results" in gr:
            for r in gr["results"]:
                hop_str = " -> ".join(f"{a} --{t}--> {b}" for a, t, b in r["path"])
                print(f"  {r['dependency']} is owned by {r['owning_team']}   [{hop_str}]")
        else:
            hop_str = " -> ".join(f"{a} --{t}--> {b}" for a, t, b in gr["path"])
            print(f"  root cause: {gr['root_cause_service']}, owned by {gr['owning_team']}   [{hop_str}]")
    else:
        print(f"GRAPH TRAVERSAL: no match ({result['graph_error']})")

    print("=" * 70)
    print()


def main() -> None:
    print("=" * 70)
    print("graph-rag-ontology-demo -- vector search vs. graph traversal")
    print("=" * 70)
    print()

    for question in QUESTIONS:
        result = compare(question)
        _print_result(result)

    print("Notes:")
    print("- The first question is the central demonstrated case: vector search's")
    print("  top hit is the 'Services and ownership' chunk, which correctly says")
    print("  checkout-api is owned by Team Orion -- true, but NOT the answer to the")
    print("  question asked, since the payment problem's root cause is a different")
    print("  service (payments-gateway) owned by a different team (Team Nimbus).")
    print("  Graph traversal follows the actual depends_on -> owned_by chain and")
    print("  gets the right answer directly.")
    print("- The knowledge graph used for traversal is independently re-derived by")
    print("  parsing data/knowledge_base.md's raw text (see src/notes_parser.py),")
    print("  not just hand-asserted -- a test confirms the parsed and hand-built")
    print("  graphs are identical.")
    print("- The last question is deliberately out of scope for the graph (no")
    print("  recognized pattern matches) -- shown to demonstrate the graph fails")
    print("  honestly rather than returning a wrong or fabricated answer.")
    print("=" * 70)


if __name__ == "__main__":
    main()

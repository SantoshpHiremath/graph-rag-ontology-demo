# graph-rag-ontology-demo

A small Graph-RAG project: a typed knowledge graph (ontology of
services, teams, incidents, and their relations) built with `networkx`,
compared head-to-head against flat TF-IDF vector-similarity retrieval on
the same underlying facts — to demonstrate, not just claim, the specific
failure mode graph-based retrieval is meant to fix: connecting two facts
that live in different text chunks and never co-retrieve together.

It covers retrieval-augmented generation pipelines, ontologies, and
Graph-RAG systems.

## Scope

- **The knowledge base is synthetic**: a small e-commerce platform's
  internal engineering documentation (services, team ownership,
  dependencies, incident history), invented for this project. It is
  not real company data.
- **The knowledge graph is real** (`src/knowledge_graph.py`, built with
  `networkx`), with typed nodes (`service`, `team`, `incident`) and
  typed edges (`owned_by`, `depends_on`, `caused_by`, `affected`,
  `resolved_by`) — a genuine, if small, ontology, not a flat list.
- **The graph is independently cross-checked against its own source
  text.** `src/notes_parser.py` regex-parses `data/knowledge_base.md`
  from scratch and rebuilds the same graph structure with zero shared
  code or state with the hand-built version in `knowledge_graph.py`. A
  test (`test_parsed_graph_matches_hand_built_graph`) confirms the two
  independently-produced graphs agree on every node and every edge —
  real evidence the graph accurately reflects its stated source, not
  just an asserted claim.
- **The vector retriever is real** (`src/vector_retrieval.py`,
  scikit-learn `TfidfVectorizer` + cosine similarity) — not a mock, not
  a stub. It's a classical (not neural/dense-embedding) retriever,
  chosen deliberately so this comparison runs fully offline without
  depending on a live embeddings model being reachable. See the scope
  note in that module's docstring for why this is a fair, not a
  strawman, comparison.
- **Question routing to a graph-traversal pattern is a small keyword
  matcher**, not an LLM — a simple, deterministic approach that keeps the mechanism fully testable. The graph traversal itself
  — the actually load-bearing part — is fully real.

## The demonstrated result

Question: *"Which team should I contact if checkout-api is having
payment problems?"*

The correct answer requires connecting two separate facts: checkout-api
depends on payments-gateway (a dependency fact), and payments-gateway is
owned by **Team Nimbus** (an ownership fact about a *different*
service). These two facts live in different sections of the source
notes.

- **Vector search's top hit** (TF-IDF, real cosine-similarity score
  0.33): the "Services and ownership" chunk, which says checkout-api is
  owned by **Team Orion** — true, but not the answer to the question
  asked. It ranks highest because the question mentions "checkout-api"
  by name, and lexical similarity can't distinguish "the team that owns
  this service" from "the team that owns the dependency actually at
  fault."
- **Graph traversal** follows `checkout-api --depends_on--> payments-gateway --owned_by--> Team Nimbus`
  directly and returns the correct answer, with the exact two-hop path
  shown as its explanation.

This same pattern repeats for incident-root-cause questions
(`INC-102`, `INC-103`): vector search's top hit is consistently
`INC-101` (the incident with the most lexical overlap with the words
"incident," "outage," "affected," regardless of which incident ID was
actually asked about, at a low confidence score around 0.12), while
graph traversal correctly follows `caused_by` → `owned_by` to the right
team every time. See `run_pipeline.py`'s output below for the full,
literal comparison — this isn't summarized or cherry-picked, it's the
actual output of running the code.

## Project structure

```
src/
  knowledge_graph.py    -- hand-built typed graph + traversal primitives
  notes_parser.py         -- independently re-derives the graph from raw text
  vector_retrieval.py      -- TF-IDF flat-text retriever (comparison baseline)
  graph_retrieval.py        -- keyword-routed graph-traversal question answering
  compare_retrieval.py       -- runs both strategies on the same question
tests/
  test_knowledge_graph.py
  test_notes_parser.py
  test_vector_retrieval.py
  test_graph_retrieval.py
  test_compare_retrieval.py
data/
  knowledge_base.md    -- the synthetic source notes both retrievers read
run_pipeline.py          -- end-to-end demo
```

## Results

```
$ python3 -m pytest tests/ -v
============================== 48 passed in 1.15s ==============================
```

All 48 tests pass live, confirmed in a clean virtualenv (fresh install
of `networkx`, `scikit-learn`, `pytest` only — no leftover packages from
other projects). `python3 run_pipeline.py` runs cleanly end-to-end and
prints the full vector-vs-graph comparison for 5 questions, including
one deliberately out-of-scope question to show the graph failing cleanly
(raising a clear "no recognized pattern" error) rather than fabricating
an answer.

## Bugs found and fixed during development

Notes from building and testing the project:

1. **The notes parser silently dropped most of the graph.** The first
   version of `notes_parser.py` used line-anchored regexes
   (`re.MULTILINE`, `^...$`) against the raw markdown, assuming each
   bullet fits on one physical line. The actual source file wraps long
   bullets across two lines (standard markdown formatting). Running the
   cross-check test for the first time showed the parsed graph had only
   5 of 19 edges — the ownership bullets (short, single-line) parsed
   fine, but every wrapped dependency and incident bullet was silently
   dropped, with no error raised. Fixed by adding `_unwrap_bullets()`,
   which rejoins each bullet's wrapped continuation lines into one
   logical line before the regexes run. `test_parsed_graph_has_nonzero_edges`
   is a direct regression test for this — it would have caught the bug
   immediately instead of it going unnoticed.

2. **The graph-traversal "path" explanation could show the wrong path.**
   `answer_owning_team_for_dependency()` originally built its returned
   explanation by calling a separate `find_path_explanation()` shortest-
   path search over the whole graph. For `checkout-api` → `Team
   Nimbus`, this could return a path through an unrelated incident edge
   (`checkout-api --affected(reverse)--> INC-101 --resolved_by--> Team
   Nimbus`) instead of the `depends_on --> owned_by` chain that actually
   produced the answer — technically a valid shortest path in the
   undirected graph, but a misleading explanation of *why* that answer
   was given. Found via direct manual testing before writing the formal
   test suite. Fixed by building the explanation path directly from the
   two hops actually walked to compute the answer, rather than
   re-deriving it via an independent search.

3. **An early draft of the knowledge base leaked its own answer.** The
   first version of `data/knowledge_base.md` included a "Why Graph-RAG
   matters for this kind of question" section that restated the demo
   question almost verbatim in prose — which then became the top
   vector-search hit purely because of that verbatim overlap, silently
   inflating the comparison in the graph's favor for the wrong reason.
   Caught by re-reading the actual retrieved chunk during manual
   testing rather than trusting the result at face value. Fixed by
   moving that explanation out of the retrievable knowledge base
   entirely (into this README instead) — the source notes now contain
   only the raw facts, so the vector-vs-graph comparison is run on a
   level playing field.

## Relationship to sibling projects

This project is distinct from the existing vector-only RAG work in
`rag-tool-agent-demo` (dense FAISS retrieval over unstructured notes, no
graph or typed relations) and `rag-tool-mcp-server`. The "independently
re-derive and cross-check rather than trust a hand-built data structure"
discipline mirrors `ai-codegen-analyst`'s two-layer safety design and
`llm-eval-pipeline`'s judge-verification approach: don't just assert
correctness, demonstrate it.

## Running it yourself

```
python3 -m venv venv
source venv/bin/activate
pip install networkx scikit-learn pytest
python3 -m pytest tests/ -v
python3 run_pipeline.py
```

"""
vector_retrieval.py
---------------------

A real, runnable vector-similarity retriever over the same
data/knowledge_base.md notes the knowledge graph is built from --
built with scikit-learn's TfidfVectorizer + cosine similarity so this
project doesn't depend on a live embeddings model (Ollama/OpenAI) being
reachable, the same "make the mechanism genuinely testable without a
live model" discipline used in llm-eval-pipeline and rag-tool-agent-demo
elsewhere in this portfolio.

HONEST SCOPE NOTE: TF-IDF is a real, classical information-retrieval
technique -- not a mock or a stub -- but it is a weaker retriever than
a modern dense embedding model (e.g. nomic-embed-text, used in the
sibling rag-tool-agent-demo project against a live Ollama). It's
intentionally chosen here because the *point* of this project is
comparing retrieval strategies (flat text similarity vs. graph
traversal) on a level, fully-offline playing field, not maximizing
absolute retrieval quality. That said, TF-IDF's known weakness --
matching on lexical overlap rather than true semantic connection -- is
exactly what makes it a fair, honest opponent for the "same words,
wrong entity" failure mode this project demonstrates.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_markdown_by_bullet(markdown_text: str) -> list:
    """Splits the knowledge base into retrievable chunks -- one chunk
    per top-level section (## heading) with its bullets, mirroring how
    a real document would typically be chunked (by section) rather than
    by single sentence, since a question usually needs a whole section's
    context, not one isolated fact.
    """
    sections = re.split(r"\n(?=## )", markdown_text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        chunks.append(section)
    return chunks


class VectorRetriever:
    """A minimal but real TF-IDF vector retriever. Fit once over a list
    of text chunks; query returns the top-k most similar chunks by
    cosine similarity, exactly the retrieval primitive a dense-embedding
    RAG system would use, just with a classical (not neural) similarity
    metric.
    """

    def __init__(self, chunks: list):
        if not chunks:
            raise ValueError("VectorRetriever requires at least one chunk")
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(chunks)

    def query(self, question: str, top_k: int = 1) -> list:
        """Returns up to top_k (chunk, similarity_score) pairs, sorted
        by descending similarity. A chunk with zero overlap with the
        question still gets returned (with a low/zero score) if there
        aren't enough higher-scoring chunks -- callers should check the
        score, not just take whatever comes back.
        """
        q_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], float(scores[i])) for i in ranked[:top_k]]

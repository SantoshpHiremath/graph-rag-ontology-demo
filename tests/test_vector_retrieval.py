"""Tests for src/vector_retrieval.py -- the TF-IDF flat-text retriever
used as the comparison baseline against graph traversal."""
from __future__ import annotations

import pytest

from src.vector_retrieval import VectorRetriever, chunk_markdown_by_bullet


SAMPLE_KB = """# Heading

## Services and ownership

- checkout-api is owned by Team Orion.
- payments-gateway is owned by Team Nimbus.

## Service dependencies

- checkout-api depends on payments-gateway because it charges the
  customer's card.
"""


class TestChunkMarkdownByBullet:
    def test_splits_into_one_chunk_per_section(self):
        chunks = chunk_markdown_by_bullet(SAMPLE_KB)
        assert len(chunks) == 2

    def test_chunks_retain_their_heading(self):
        chunks = chunk_markdown_by_bullet(SAMPLE_KB)
        assert any(c.startswith("## Services and ownership") for c in chunks)
        assert any(c.startswith("## Service dependencies") for c in chunks)

    def test_top_level_title_is_not_its_own_chunk(self):
        chunks = chunk_markdown_by_bullet(SAMPLE_KB)
        assert not any(c.startswith("# Heading") for c in chunks)

    def test_real_knowledge_base_file_chunks_without_error(self):
        with open("data/knowledge_base.md", "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_markdown_by_bullet(text)
        assert len(chunks) == 3  # ownership, dependencies, incident history


class TestVectorRetriever:
    def test_raises_on_empty_chunk_list(self):
        with pytest.raises(ValueError):
            VectorRetriever([])

    def test_query_returns_requested_top_k(self):
        chunks = chunk_markdown_by_bullet(SAMPLE_KB)
        retriever = VectorRetriever(chunks)
        results = retriever.query("Who owns checkout-api?", top_k=1)
        assert len(results) == 1

    def test_query_returns_chunk_and_score_pairs(self):
        chunks = chunk_markdown_by_bullet(SAMPLE_KB)
        retriever = VectorRetriever(chunks)
        results = retriever.query("Who owns checkout-api?", top_k=2)
        for chunk, score in results:
            assert isinstance(chunk, str)
            assert isinstance(score, float)

    def test_relevant_chunk_scores_higher_than_irrelevant_chunk(self):
        chunks = chunk_markdown_by_bullet(SAMPLE_KB)
        retriever = VectorRetriever(chunks)
        results = retriever.query("ownership Team Orion", top_k=2)
        # The ownership chunk should rank above the dependency chunk
        # for an ownership-focused query.
        assert "Services and ownership" in results[0][0]

    def test_demonstrates_the_real_failure_mode_this_project_is_about(self):
        # The actual point of this project: a question about
        # checkout-api's payment dependency returns the checkout-api
        # OWNERSHIP chunk (Team Orion) as the top hit -- lexically
        # relevant (mentions "checkout-api"), but not the entity that
        # actually owns the at-fault dependency (Team Nimbus, who owns
        # payments-gateway). This is demonstrated, not just claimed.
        with open("data/knowledge_base.md", "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_markdown_by_bullet(text)
        retriever = VectorRetriever(chunks)
        results = retriever.query(
            "Which team should I contact if checkout-api is having payment problems?",
            top_k=1,
        )
        top_chunk = results[0][0]
        assert "checkout-api is owned by Team Orion" in top_chunk
        # The correct answer (Team Nimbus, via payments-gateway) is
        # NOT the top-ranked chunk's headline fact.
        assert "Team Nimbus" not in top_chunk.split("\n")[2]

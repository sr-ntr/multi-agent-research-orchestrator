"""
Tests for retrieval: RRF math, BM25 exact-token search, source filtering.

These tests use no API calls. The KnowledgeBase tests mock the embedder
with deterministic fake vectors.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.retrieval import tokenize, build_bm25, bm25_search, reciprocal_rank_fusion
from src.store import KnowledgeBase


# ---------------------------------------------------------------------------
# RRF math
# ---------------------------------------------------------------------------


def test_rrf_single_ranking():
    """RRF over one ranking assigns score 1/(k+rank)."""
    k = 60
    rankings = [["a", "b", "c"]]
    result = reciprocal_rank_fusion(rankings, k=k)
    result_map = dict(result)
    assert abs(result_map["a"] - 1 / (k + 1)) < 1e-9
    assert abs(result_map["b"] - 1 / (k + 2)) < 1e-9
    assert abs(result_map["c"] - 1 / (k + 3)) < 1e-9


def test_rrf_fusion_boosts_overlap():
    """Items appearing in multiple rankings get higher RRF scores."""
    k = 60
    rankings = [["a", "b", "c"], ["b", "c", "d"]]
    result = reciprocal_rank_fusion(rankings, k=k)
    result_map = dict(result)

    # 'b' appears in both: rank 2 in list 1, rank 1 in list 2
    expected_b = 1 / (k + 2) + 1 / (k + 1)
    assert abs(result_map["b"] - expected_b) < 1e-9

    # 'a' appears only in list 1 at rank 1
    expected_a = 1 / (k + 1)
    assert abs(result_map["a"] - expected_a) < 1e-9


def test_rrf_result_sorted_descending():
    """RRF result is sorted by score descending."""
    result = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60)
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_empty_rankings():
    """RRF over empty input returns empty result."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_deduplication():
    """Items in the same ranking appear only once in the output."""
    result = reciprocal_rank_fusion([["a", "b", "a"]], k=60)
    ids = [r[0] for r in result]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_lowercase():
    """Tokens are lowercased."""
    tokens = tokenize("HELLO World")
    assert all(t == t.lower() for t in tokens)


def test_tokenize_removes_stopwords():
    """Common stopwords are removed."""
    tokens = tokenize("the quick brown fox")
    assert "the" not in tokens


def test_tokenize_removes_single_chars():
    """Single-character tokens are removed."""
    tokens = tokenize("a b c hello")
    assert "a" not in tokens
    assert "b" not in tokens


def test_tokenize_handles_special_chars():
    """Special characters split tokens correctly."""
    tokens = tokenize("foo-bar baz_qux")
    assert "foo" in tokens
    assert "bar" in tokens


# ---------------------------------------------------------------------------
# BM25: rare token test
# ---------------------------------------------------------------------------


def _make_fake_embedder(dim: int = 4):
    """Return a patch target and side_effect that produces deterministic vectors."""
    import random

    def fake_embed_documents(texts):
        # Deterministic: use hash of text to seed
        result = []
        for t in texts:
            r = random.Random(hash(t))
            vec = [r.uniform(-1, 1) for _ in range(dim)]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            result.append([x / norm for x in vec])
        return result

    def fake_embed_query(text):
        return fake_embed_documents([text])[0]

    return fake_embed_documents, fake_embed_query


def test_bm25_finds_rare_token():
    """BM25 should rank a document with a rare exact token higher than vector search would.

    We build a tiny corpus where one document contains a unique product code
    that is not semantically close to the query in embedding space, but BM25
    matches it exactly.
    """
    corpus = [
        "The quarterly revenue figures look promising for the company.",
        "General overview of the market dynamics and competitive landscape.",
        "Product SKU-XZ9981 specifications: weight 2.3 kg, power 45W.",
    ]
    corpus_ids = ["chunk:0001", "chunk:0002", "chunk:0003"]

    index = build_bm25(corpus)
    results = bm25_search(index, corpus_ids, "SKU-XZ9981", top_k=3)

    assert results, "BM25 returned no results"
    top_id = results[0][0]
    assert top_id == "chunk:0003", (
        f"BM25 should rank the SKU document first, got {top_id!r}"
    )


def test_bm25_empty_query_returns_empty():
    """BM25 with an all-stopword query returns no results."""
    corpus = ["Some sample text about things.", "Another piece of content."]
    index = build_bm25(corpus)
    results = bm25_search(index, ["id:0", "id:1"], "the and or", top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# KnowledgeBase: source filter and search (offline, mock embeddings)
# ---------------------------------------------------------------------------


def _make_kb_with_chunks():
    """Build an ephemeral KnowledgeBase with two sources, using mock embeddings."""
    fake_embed_docs, fake_embed_query = _make_fake_embedder(dim=4)

    with (
        patch("src.store.embed_documents", side_effect=fake_embed_docs),
        patch("src.store.embed_query", side_effect=fake_embed_query),
        patch("src.embeddings.embed_documents", side_effect=fake_embed_docs),
        patch("src.embeddings.embed_query", side_effect=fake_embed_query),
    ):
        kb = KnowledgeBase()
        chunks_a = [
            {
                "chunk_id": "alpha:0001",
                "source": "alpha.md",
                "heading": "Introduction",
                "chunk_index": 1,
                "doc_hash": "aaa",
                "text": "Alpha company was founded in 2010.",
                "embed_text": "Introduction\nAlpha company was founded in 2010.",
            }
        ]
        chunks_b = [
            {
                "chunk_id": "beta:0001",
                "source": "beta.md",
                "heading": "Overview",
                "chunk_index": 1,
                "doc_hash": "bbb",
                "text": "Beta corporation operates in 12 countries.",
                "embed_text": "Overview\nBeta corporation operates in 12 countries.",
            }
        ]
        kb.add_chunks(chunks_a)
        kb.add_chunks(chunks_b)
    return kb, fake_embed_docs, fake_embed_query


def test_source_filter_restricts_results():
    """Searching with a source filter returns only chunks from that source."""
    fake_embed_docs, fake_embed_query = _make_fake_embedder(dim=4)

    with (
        patch("src.store.embed_documents", side_effect=fake_embed_docs),
        patch("src.store.embed_query", side_effect=fake_embed_query),
    ):
        kb = KnowledgeBase()
        kb.add_chunks([
            {
                "chunk_id": "alpha:0001",
                "source": "alpha.md",
                "heading": "Intro",
                "chunk_index": 1,
                "doc_hash": "aaa",
                "text": "Alpha company was founded in 2010.",
                "embed_text": "Intro\nAlpha company was founded in 2010.",
            },
            {
                "chunk_id": "beta:0001",
                "source": "beta.md",
                "heading": "Intro",
                "chunk_index": 1,
                "doc_hash": "bbb",
                "text": "Beta corporation operates in 12 countries.",
                "embed_text": "Intro\nBeta corporation operates in 12 countries.",
            },
        ])
        results = kb.search("company", top_k=10, source="alpha.md", mode="bm25")
        sources = {r["source"] for r in results}
        assert sources == {"alpha.md"}, f"Expected only alpha.md, got {sources}"


def test_get_neighbors_returns_adjacent(tmp_path):
    """get_neighbors returns chunks adjacent to the requested chunk."""
    fake_embed_docs, fake_embed_query = _make_fake_embedder(dim=4)

    chunks = [
        {
            "chunk_id": f"doc:000{i}",
            "source": "doc.md",
            "heading": "Section",
            "chunk_index": i,
            "doc_hash": "xxx",
            "text": f"This is chunk number {i} of the document.",
            "embed_text": f"Section\nThis is chunk number {i} of the document.",
        }
        for i in range(5)
    ]

    with (
        patch("src.store.embed_documents", side_effect=fake_embed_docs),
        patch("src.store.embed_query", side_effect=fake_embed_query),
    ):
        kb = KnowledgeBase()
        kb.add_chunks(chunks)
        neighbors = kb.get_neighbors("doc:0002", window=1)

    neighbor_ids = {n["chunk_id"] for n in neighbors}
    assert "doc:0001" in neighbor_ids
    assert "doc:0003" in neighbor_ids
    assert "doc:0002" not in neighbor_ids  # center chunk excluded


def test_list_documents_returns_correct_counts():
    """list_documents returns one entry per source with correct chunk counts."""
    fake_embed_docs, fake_embed_query = _make_fake_embedder(dim=4)

    with (
        patch("src.store.embed_documents", side_effect=fake_embed_docs),
        patch("src.store.embed_query", side_effect=fake_embed_query),
    ):
        kb = KnowledgeBase()
        kb.add_chunks([
            {
                "chunk_id": "a:0001",
                "source": "a.md",
                "heading": "",
                "chunk_index": 1,
                "doc_hash": "h1",
                "text": "Text from document A, first chunk.",
                "embed_text": "Text from document A, first chunk.",
            },
            {
                "chunk_id": "a:0002",
                "source": "a.md",
                "heading": "",
                "chunk_index": 2,
                "doc_hash": "h1",
                "text": "Text from document A, second chunk.",
                "embed_text": "Text from document A, second chunk.",
            },
            {
                "chunk_id": "b:0001",
                "source": "b.md",
                "heading": "",
                "chunk_index": 1,
                "doc_hash": "h2",
                "text": "Text from document B.",
                "embed_text": "Text from document B.",
            },
        ])
        docs = kb.list_documents()

    by_source = {d["source"]: d for d in docs}
    assert by_source["a.md"]["chunk_count"] == 2
    assert by_source["b.md"]["chunk_count"] == 1

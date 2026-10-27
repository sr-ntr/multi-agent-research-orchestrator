"""
Tests for tool functions: search, source filter, neighbors, web tool presence.

All tests are offline (no API calls). The KnowledgeBase is mocked or
built with deterministic fake embeddings.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.tools import make_tools, calculate_expr, _search_documents_fn, _list_documents_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_kb(chunks: list[dict] | None = None) -> MagicMock:
    """Return a mock KnowledgeBase with configurable search results."""
    kb = MagicMock()
    kb.search.return_value = chunks or [
        {
            "chunk_id": "alpha:0001",
            "source": "alpha.md",
            "heading": "Introduction",
            "text": "Alpha was founded in 2010.",
            "score": 0.9,
        }
    ]
    kb.list_documents.return_value = [
        {"source": "alpha.md", "chunk_count": 3, "doc_hash": "abc"},
        {"source": "beta.md", "chunk_count": 2, "doc_hash": "def"},
    ]
    kb.get_chunk.return_value = {
        "chunk_id": "alpha:0002",
        "source": "alpha.md",
        "heading": "Section",
        "chunk_index": 2,
        "doc_hash": "abc",
        "text": "The center chunk.",
    }
    kb.get_neighbors.return_value = [
        {
            "chunk_id": "alpha:0001",
            "source": "alpha.md",
            "heading": "Section",
            "chunk_index": 1,
            "doc_hash": "abc",
            "text": "The preceding chunk.",
        }
    ]
    return kb


# ---------------------------------------------------------------------------
# make_tools: tool list
# ---------------------------------------------------------------------------


def test_make_tools_without_web_has_four_tools():
    """make_tools without web returns exactly 4 tools."""
    kb = _make_fake_kb()
    tools = make_tools(kb, web_enabled=False)
    names = {t.name for t in tools}
    assert "search_documents" in names
    assert "get_chunk_context" in names
    assert "list_documents" in names
    assert "calculate" in names
    assert "web_search" not in names


def test_make_tools_with_web_has_five_tools():
    """make_tools with web_enabled=True includes web_search."""
    from src.web_search import FakeProvider
    kb = _make_fake_kb()
    tools = make_tools(kb, web_enabled=True, web_provider=FakeProvider())
    names = {t.name for t in tools}
    assert "web_search" in names
    assert len(tools) == 5


# ---------------------------------------------------------------------------
# search_documents: source filtering
# ---------------------------------------------------------------------------


def test_search_documents_passes_source_filter():
    """search_documents forwards the source filter to kb.search."""
    kb = _make_fake_kb()
    _search_documents_fn(kb, "query", top_k=5, source="alpha.md")
    kb.search.assert_called_once_with("query", top_k=5, source="alpha.md", mode="hybrid")


def test_search_documents_returns_json_array():
    """search_documents returns a valid JSON array string."""
    kb = _make_fake_kb()
    raw = _search_documents_fn(kb, "query")
    parsed = json.loads(raw)
    assert isinstance(parsed, list)
    assert parsed[0]["chunk_id"] == "alpha:0001"


def test_search_documents_caps_top_k():
    """search_documents caps top_k at 20 to avoid excessive retrieval."""
    kb = _make_fake_kb()
    _search_documents_fn(kb, "query", top_k=50)
    call_args = kb.search.call_args
    assert call_args[1]["top_k"] == 20 or call_args[0][1] == 20


# ---------------------------------------------------------------------------
# web_search tool: presence controlled by flag
# ---------------------------------------------------------------------------


def test_web_search_absent_when_disabled():
    """The web_search tool is not in the list when web_enabled=False."""
    kb = _make_fake_kb()
    tools = make_tools(kb, web_enabled=False)
    assert not any(t.name == "web_search" for t in tools)


def test_web_search_uses_fake_provider():
    """web_search tool invokes the provided FakeProvider."""
    from src.web_search import FakeProvider
    provider = FakeProvider()
    kb = _make_fake_kb()
    tools = make_tools(kb, web_enabled=True, web_provider=provider)
    web_tool = next(t for t in tools if t.name == "web_search")
    result_json = web_tool.invoke({"query": "test query", "max_results": 2})
    results = json.loads(result_json)
    assert len(results) <= 2
    assert all("id" in r for r in results)
    assert all(r["id"].startswith("web:") for r in results)


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


def test_list_documents_returns_json():
    """list_documents returns a valid JSON array of document summaries."""
    kb = _make_fake_kb()
    raw = _list_documents_fn(kb)
    docs = json.loads(raw)
    assert isinstance(docs, list)
    sources = {d["source"] for d in docs}
    assert "alpha.md" in sources

"""
Tests for the evidence ledger: citation parsing, invalid ID detection, dedup.
"""

import pytest

from src.ledger import (
    Evidence,
    extract_evidence,
    merge_ledger,
    build_ledger,
    parse_citations,
    check_citations,
)


# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------


def test_evidence_doc_creation():
    ev = Evidence(
        id="report:0001",
        kind="doc",
        source="report.md",
        heading="Risk Factors > Competition",
        text="The market is highly competitive.",
        score=0.85,
    )
    assert ev.id == "report:0001"
    assert ev.kind == "doc"
    assert ev.score == 0.85


def test_evidence_web_creation():
    ev = Evidence(
        id="web:abc12345",
        kind="web",
        source="https://example.com",
        url="https://example.com",
        text="A web snippet.",
    )
    assert ev.kind == "web"
    assert ev.url == "https://example.com"


def test_evidence_empty_id_raises():
    with pytest.raises(Exception):
        Evidence(id="  ", kind="doc", source="x.md", text="text")


# ---------------------------------------------------------------------------
# extract_evidence
# ---------------------------------------------------------------------------


def test_extract_evidence_from_doc_result_json():
    """extract_evidence parses a JSON doc result list."""
    import json
    data = json.dumps([
        {
            "chunk_id": "report:0001",
            "source": "report.md",
            "heading": "Summary",
            "text": "Revenue was 450M.",
            "score": 0.9,
        }
    ])
    items = extract_evidence(data)
    assert len(items) == 1
    assert items[0].id == "report:0001"
    assert items[0].kind == "doc"


def test_extract_evidence_from_web_result_json():
    """extract_evidence parses a JSON web result list."""
    import json
    data = json.dumps([
        {
            "id": "web:abc12345",
            "url": "https://example.com",
            "title": "Example",
            "text": "Web snippet.",
        }
    ])
    items = extract_evidence(data)
    assert len(items) == 1
    assert items[0].id == "web:abc12345"
    assert items[0].kind == "web"


def test_extract_evidence_ignores_bad_json():
    """extract_evidence returns empty list for non-JSON input."""
    items = extract_evidence("not valid json at all {{")
    assert items == []


def test_extract_evidence_skips_items_without_id():
    """extract_evidence skips result dicts that lack a chunk_id."""
    import json
    data = json.dumps([{"source": "x.md", "text": "no id here"}])
    items = extract_evidence(data)
    assert items == []


def test_extract_evidence_from_list():
    """extract_evidence accepts a pre-parsed list."""
    items = extract_evidence([
        {"chunk_id": "a:0001", "source": "a.md", "text": "hello"},
    ])
    assert len(items) == 1


# ---------------------------------------------------------------------------
# merge_ledger
# ---------------------------------------------------------------------------


def test_merge_ledger_adds_new_entries():
    ev1 = Evidence(id="a:0001", kind="doc", source="a.md", text="text a")
    ev2 = Evidence(id="b:0001", kind="doc", source="b.md", text="text b")
    base = {"a:0001": ev1}
    incoming = {"b:0001": ev2}
    merged = merge_ledger(base, incoming)
    assert "a:0001" in merged
    assert "b:0001" in merged


def test_merge_ledger_does_not_overwrite():
    """merge_ledger never replaces an existing entry."""
    ev_orig = Evidence(id="a:0001", kind="doc", source="a.md", text="original")
    ev_new = Evidence(id="a:0001", kind="doc", source="a.md", text="updated")
    base = {"a:0001": ev_orig}
    merge_ledger(base, {"a:0001": ev_new})
    assert base["a:0001"].text == "original"


# ---------------------------------------------------------------------------
# parse_citations
# ---------------------------------------------------------------------------


def test_parse_citations_single():
    ids = parse_citations("The revenue grew [report:0001].")
    assert ids == ["report:0001"]


def test_parse_citations_multiple_brackets():
    ids = parse_citations("Fact A [a:0001] and fact B [b:0002].")
    assert "a:0001" in ids
    assert "b:0002" in ids


def test_parse_citations_comma_separated():
    """[a, b] style citations are expanded into individual IDs."""
    ids = parse_citations("See [a:0001, b:0002] for details.")
    assert "a:0001" in ids
    assert "b:0002" in ids


def test_parse_citations_deduplication():
    """The same ID cited twice appears only once."""
    ids = parse_citations("[a:0001] and again [a:0001].")
    assert ids.count("a:0001") == 1


def test_parse_citations_web_id():
    ids = parse_citations("According to a web source [web:abc12345].")
    assert "web:abc12345" in ids


def test_parse_citations_no_citations():
    ids = parse_citations("No citations in this text at all.")
    assert ids == []


# ---------------------------------------------------------------------------
# check_citations
# ---------------------------------------------------------------------------


def test_check_citations_all_valid():
    ev = Evidence(id="a:0001", kind="doc", source="a.md", text="text")
    ledger = {"a:0001": ev}
    valid, invalid = check_citations("See [a:0001].", ledger)
    assert valid == ["a:0001"]
    assert invalid == []


def test_check_citations_invalid_id():
    ledger = {}
    valid, invalid = check_citations("See [fake:9999].", ledger)
    assert valid == []
    assert invalid == ["fake:9999"]


def test_check_citations_mixed():
    ev = Evidence(id="real:0001", kind="doc", source="x.md", text="text")
    ledger = {"real:0001": ev}
    valid, invalid = check_citations("See [real:0001] and [ghost:0002].", ledger)
    assert "real:0001" in valid
    assert "ghost:0002" in invalid

"""
Tests for document chunking: heading paths, overlap, minimum length, idempotency.

These tests are offline-only (no API calls).
"""

import hashlib
import pytest
from pathlib import Path

from src.ingest import chunk_markdown, chunk_text, chunk_file, _file_hash
from src.config import CHUNK_MIN_CHARS


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------


SAMPLE_MD = """\
# Annual Report 2024

This is the preamble before any section heading.

## Risk Factors

### Competition

The market is highly competitive with many established players.
New entrants face high barriers because of network effects and switching costs.

### Regulatory

Regulatory changes may adversely affect operations.
Compliance costs are expected to rise over the next fiscal year.

## Financial Summary

Revenue for the year was 450 million dollars.
Operating expenses totalled 310 million dollars.
"""


def test_markdown_heading_paths():
    """Heading paths are correctly built with ' > ' separators."""
    chunks = chunk_markdown(SAMPLE_MD, "report.md")
    headings = {c["heading"] for c in chunks}
    assert "Risk Factors > Competition" in headings
    assert "Risk Factors > Regulatory" in headings
    assert "Financial Summary" in headings


def test_markdown_preamble_chunk():
    """Text before the first heading is chunked with an empty heading."""
    chunks = chunk_markdown(SAMPLE_MD, "report.md")
    preamble_chunks = [c for c in chunks if c["heading"] == ""]
    assert len(preamble_chunks) >= 1
    assert "preamble" in preamble_chunks[0]["text"]


def test_markdown_chunk_ids():
    """Chunk IDs are '<slug>:<4-digit-index>' and are unique."""
    chunks = chunk_markdown(SAMPLE_MD, "report-2024.md")
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
    for cid in ids:
        slug, idx = cid.rsplit(":", 1)
        assert slug == "report-2024", f"Unexpected slug in {cid!r}"
        assert idx.isdigit() and len(idx) == 4, f"Index not zero-padded: {cid!r}"


def test_markdown_source_metadata():
    """Source metadata matches the filename argument."""
    chunks = chunk_markdown(SAMPLE_MD, "report.md")
    for c in chunks:
        assert c["source"] == "report.md"


def test_markdown_embed_text_includes_heading():
    """embed_text prepends the heading path to the chunk text."""
    chunks = chunk_markdown(SAMPLE_MD, "report.md")
    for c in chunks:
        if c["heading"]:
            assert c["embed_text"].startswith(c["heading"]), (
                f"embed_text should start with heading for chunk {c['chunk_id']!r}"
            )
        else:
            assert c["embed_text"] == c["text"]


def test_markdown_minimum_length():
    """No chunk shorter than CHUNK_MIN_CHARS is produced."""
    chunks = chunk_markdown(SAMPLE_MD, "report.md")
    for c in chunks:
        assert len(c["text"]) >= CHUNK_MIN_CHARS, (
            f"Chunk {c['chunk_id']!r} is too short: {len(c['text'])} chars"
        )


def test_markdown_drops_empty_sections():
    """Sections with no content produce no chunks."""
    md = "# Title\n\n## Empty Section\n\n## Real Section\n\nSome meaningful content here about things."
    chunks = chunk_markdown(md, "test.md")
    headings = [c["heading"] for c in chunks]
    assert "Empty Section" not in headings
    assert "Real Section" in headings


# ---------------------------------------------------------------------------
# Plain text chunking
# ---------------------------------------------------------------------------


SAMPLE_TXT = """\
This is the first paragraph. It contains a few sentences about the product.
It is quite important for the user to read this paragraph carefully.

This is the second paragraph. It describes the pricing model.
The base plan costs 10 dollars per month. The professional plan costs 25 dollars.

This is the third paragraph. It covers support policies.
Support tickets are answered within 24 hours on business days.
"""


def test_text_chunking_produces_chunks():
    """Plain text chunking produces at least one chunk."""
    chunks = chunk_text(SAMPLE_TXT, "doc.txt")
    assert len(chunks) >= 1


def test_text_chunking_heading_is_empty():
    """Plain text chunks have an empty heading."""
    chunks = chunk_text(SAMPLE_TXT, "doc.txt")
    for c in chunks:
        assert c["heading"] == ""


def test_text_chunking_minimum_length():
    """No plain text chunk is shorter than CHUNK_MIN_CHARS."""
    chunks = chunk_text(SAMPLE_TXT, "doc.txt")
    for c in chunks:
        assert len(c["text"]) >= CHUNK_MIN_CHARS


# ---------------------------------------------------------------------------
# Idempotency via doc_hash
# ---------------------------------------------------------------------------


def test_chunk_file_stamps_doc_hash(tmp_path):
    """chunk_file stamps the correct sha1 hash on every chunk."""
    f = tmp_path / "sample.md"
    f.write_text(SAMPLE_MD, encoding="utf-8")
    expected_hash = _file_hash(f.read_bytes())

    chunks = chunk_file(f)
    assert chunks, "Expected at least one chunk"
    for c in chunks:
        assert c["doc_hash"] == expected_hash, (
            f"Chunk {c['chunk_id']!r} has wrong hash"
        )


def test_chunk_file_same_content_same_hash(tmp_path):
    """Two files with identical content produce the same doc_hash."""
    content = SAMPLE_MD
    f1 = tmp_path / "a.md"
    f2 = tmp_path / "b.md"
    f1.write_text(content, encoding="utf-8")
    f2.write_text(content, encoding="utf-8")

    chunks1 = chunk_file(f1)
    chunks2 = chunk_file(f2)

    assert chunks1[0]["doc_hash"] == chunks2[0]["doc_hash"]


def test_chunk_file_rejects_unsupported_type(tmp_path):
    """chunk_file raises ValueError for unsupported file types."""
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF fake")
    with pytest.raises(ValueError, match="Unsupported"):
        chunk_file(f)


def test_chunk_file_rejects_oversized(tmp_path):
    """chunk_file raises ValueError for files exceeding the size limit."""
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (1_048_576 + 1))
    with pytest.raises(ValueError, match="1 MB"):
        chunk_file(f)

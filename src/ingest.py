"""
Document ingestion pipeline: load -> chunk -> embed -> upsert.

Also a CLI:
    python -m src.ingest <path> [--persist .chroma]

Handles .txt and .md files. Implements heading-aware chunking for markdown,
paragraph-boundary chunking for plain text and long sections. Idempotent:
same file with same hash is skipped; same filename with new hash triggers
delete + re-ingest.

Public API:
    chunk_markdown(text, source)    -- heading-aware chunking for .md files
    chunk_text(text, source)        -- paragraph-boundary chunking for .txt
    chunk_file(path)                -- dispatch to the right chunker
    ingest_file(path, kb)           -- chunk, embed, upsert with idempotency
    ingest_directory(path, kb)      -- ingest all .txt/.md files in a directory
"""

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from src.config import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    MAX_FILE_BYTES,
)

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _slug(name: str) -> str:
    """Convert a filename stem to a safe slug for use in chunk IDs."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _file_hash(content: bytes) -> str:
    """Return the sha1 hex digest of file content bytes."""
    return hashlib.sha1(content).hexdigest()


def _make_chunk_id(source_slug: str, index: int) -> str:
    """Format a chunk ID as '<slug>:<zero-padded-index>'."""
    return f"{source_slug}:{index:04d}"


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines, returning non-empty paragraphs."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _build_chunks_from_paragraphs(
    paragraphs: list[str],
    source: str,
    source_slug: str,
    heading: str,
    start_index: int,
) -> list[dict]:
    """Pack paragraphs into chunks respecting target/max size and overlap.

    Args:
        paragraphs: List of paragraph strings.
        source: Original filename (for metadata).
        source_slug: Slug for chunk ID generation.
        heading: Current heading path.
        start_index: Starting chunk index (for globally unique IDs).

    Returns:
        List of chunk dicts.
    """
    chunks: list[dict] = []
    current_parts: list[str] = []
    current_len = 0
    overlap_text = ""

    def flush(parts: list[str], idx: int) -> Optional[dict]:
        text = "\n\n".join(parts).strip()
        if len(text) < CHUNK_MIN_CHARS:
            return None
        embed_text = f"{heading}\n{text}" if heading else text
        return {
            "chunk_id": _make_chunk_id(source_slug, idx),
            "source": source,
            "heading": heading,
            "chunk_index": idx,
            "doc_hash": "",  # filled by caller
            "text": text,
            "embed_text": embed_text,
        }

    for para in paragraphs:
        # If a single paragraph exceeds max, split it at sentence boundaries
        if len(para) > CHUNK_MAX_CHARS:
            sub_paras = _split_long_paragraph(para)
        else:
            sub_paras = [para]

        for sub in sub_paras:
            if current_len + len(sub) > CHUNK_TARGET_CHARS and current_parts:
                chunk = flush(current_parts, start_index + len(chunks))
                if chunk:
                    chunks.append(chunk)
                # Keep overlap
                overlap_chars = 0
                overlap_parts: list[str] = []
                for p in reversed(current_parts):
                    if overlap_chars + len(p) <= CHUNK_OVERLAP_CHARS:
                        overlap_parts.insert(0, p)
                        overlap_chars += len(p)
                    else:
                        break
                current_parts = overlap_parts + [sub]
                current_len = sum(len(p) for p in current_parts)
            else:
                current_parts.append(sub)
                current_len += len(sub)

    if current_parts:
        chunk = flush(current_parts, start_index + len(chunks))
        if chunk:
            chunks.append(chunk)

    return chunks


def _split_long_paragraph(text: str) -> list[str]:
    """Split a paragraph that exceeds CHUNK_MAX_CHARS at sentence boundaries."""
    # Simple sentence split on '. ', '! ', '? '
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > CHUNK_TARGET_CHARS and current:
            parts.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current:
        parts.append(current.strip())
    return parts


# ---------------------------------------------------------------------------
# Markdown chunker
# ---------------------------------------------------------------------------


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Chunk a markdown document with heading-aware splitting.

    Splits at H1/H2/H3 headings and keeps the heading path in metadata.
    Sections longer than CHUNK_TARGET_CHARS are further split on paragraphs.

    Args:
        text: Full markdown file content.
        source: Filename (for metadata).

    Returns:
        List of chunk dicts (doc_hash not yet set).
    """
    source_slug = _slug(Path(source).stem)
    chunks: list[dict] = []
    heading_stack: list[str] = []

    # Split text at headings
    parts = _HEADING_RE.split(text)
    # parts alternates: preamble, hashes, heading_text, content, hashes, ...
    # Index 0 is pre-heading text, then groups of (hashes, heading, content)

    # Handle preamble (text before the first heading)
    preamble = parts[0].strip()
    if preamble:
        paras = _split_paragraphs(preamble)
        new_chunks = _build_chunks_from_paragraphs(
            paras, source, source_slug, "", len(chunks)
        )
        chunks.extend(new_chunks)

    # Walk through (hashes, heading_text, body) triples
    i = 1
    while i + 2 <= len(parts):
        hashes = parts[i]
        heading_text = parts[i + 1].strip()
        body = parts[i + 2].strip()
        level = len(hashes)  # 1, 2, or 3

        # Update heading stack
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(heading_text)
        heading_path = " > ".join(heading_stack)

        paras = _split_paragraphs(body) if body else []
        if paras:
            new_chunks = _build_chunks_from_paragraphs(
                paras, source, source_slug, heading_path, len(chunks)
            )
            chunks.extend(new_chunks)

        i += 3

    return chunks


# ---------------------------------------------------------------------------
# Plain text chunker
# ---------------------------------------------------------------------------


def chunk_text(text: str, source: str) -> list[dict]:
    """Chunk a plain text file on paragraph boundaries.

    Args:
        text: Full file content.
        source: Filename (for metadata).

    Returns:
        List of chunk dicts (doc_hash not yet set).
    """
    source_slug = _slug(Path(source).stem)
    paras = _split_paragraphs(text)
    return _build_chunks_from_paragraphs(paras, source, source_slug, "", 0)


# ---------------------------------------------------------------------------
# File-level dispatch
# ---------------------------------------------------------------------------


def chunk_file(path: Path) -> list[dict]:
    """Load and chunk a file, dispatching on extension.

    Args:
        path: Path to a .txt or .md file (<= MAX_FILE_BYTES).

    Returns:
        List of chunk dicts (doc_hash set).

    Raises:
        ValueError: If the file type is unsupported, too large, or unreadable.
    """
    if path.suffix.lower() not in (".txt", ".md"):
        raise ValueError(f"Unsupported file type: {path.suffix} (only .txt and .md)")

    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(
            f"File {path.name} exceeds the 1 MB limit "
            f"({path.stat().st_size / 1024:.1f} KB)"
        )

    content_bytes = path.read_bytes()
    doc_hash = _file_hash(content_bytes)
    text = content_bytes.decode("utf-8", errors="replace")
    source = path.name

    if path.suffix.lower() == ".md":
        chunks = chunk_markdown(text, source)
    else:
        chunks = chunk_text(text, source)

    # Stamp the doc_hash on every chunk
    for c in chunks:
        c["doc_hash"] = doc_hash

    return chunks


# ---------------------------------------------------------------------------
# Ingest with idempotency
# ---------------------------------------------------------------------------


def ingest_file(path: Path, kb) -> dict:  # type: ignore[no-untyped-def]
    """Ingest a single file into the knowledge base with idempotency.

    - If the file is already ingested with the same hash: skip.
    - If the file exists but the hash differs: delete old chunks and re-ingest.
    - Otherwise: ingest fresh.

    Args:
        path: Path to the file.
        kb: A KnowledgeBase instance.

    Returns:
        Dict with keys: source, chunks, status ("skipped", "ingested", "replaced").
    """
    source = path.name
    chunks = chunk_file(path)

    if not chunks:
        logger.warning("No chunks produced from %s", source)
        return {"source": source, "chunks": 0, "status": "empty"}

    new_hash = chunks[0]["doc_hash"]

    # Check existing documents
    existing_docs = {d["source"]: d for d in kb.list_documents()}

    if source in existing_docs:
        if existing_docs[source]["doc_hash"] == new_hash:
            logger.info("Skipping %s (unchanged, hash=%s)", source, new_hash[:8])
            return {"source": source, "chunks": existing_docs[source]["chunk_count"], "status": "skipped"}
        else:
            logger.info("Re-ingesting %s (hash changed)", source)
            kb.delete_source(source)
            status = "replaced"
    else:
        status = "ingested"

    n = kb.add_chunks(chunks)
    logger.info("Ingested %d chunks from %s [%s]", n, source, status)
    return {"source": source, "chunks": n, "status": status}


def ingest_directory(path: Path, kb) -> list[dict]:  # type: ignore[no-untyped-def]
    """Ingest all .txt and .md files in a directory (non-recursive).

    Args:
        path: Directory path.
        kb: A KnowledgeBase instance.

    Returns:
        List of per-file ingest result dicts.
    """
    results = []
    files = sorted(f for f in path.iterdir() if f.suffix.lower() in (".txt", ".md"))
    for f in files:
        try:
            result = ingest_file(f, kb)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", f.name, exc)
            result = {"source": f.name, "chunks": 0, "status": f"error: {exc}"}
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest .txt and .md files into the knowledge base."
    )
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument(
        "--persist",
        default=".chroma",
        help="Chroma persist directory (default: .chroma)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    from src.store import KnowledgeBase  # late import to avoid circular

    kb = KnowledgeBase(persist_dir=args.persist)
    target = Path(args.path)

    if target.is_dir():
        results = ingest_directory(target, kb)
    elif target.is_file():
        results = [ingest_file(target, kb)]
    else:
        print(f"ERROR: {target} does not exist", file=sys.stderr)
        sys.exit(1)

    print()
    print(f"{'Source':<40} {'Chunks':>6}  Status")
    print("-" * 60)
    for r in results:
        print(f"{r['source']:<40} {r['chunks']:>6}  {r['status']}")
    print()
    total = sum(r["chunks"] for r in results)
    print(f"Total chunks in collection: {total}")


if __name__ == "__main__":
    _main()

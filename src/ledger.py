"""
Evidence ledger: Evidence model, ledger building, citation parsing and checking.

Exports:
    Evidence                -- Pydantic model for a single evidence item
    extract_evidence()      -- parse tool output into Evidence items
    merge_ledger()          -- merge two ledger dicts without overwriting
    parse_citations()       -- extract [id] markers from text
    check_citations()       -- split cited IDs into valid and invalid sets

The ledger is a plain dict[str, Evidence] keyed by evidence ID.
IDs for document chunks look like "report-2024:0007".
IDs for web results look like "web:1a2b3c4d".

Evidence is captured from tool outputs by the researcher node; the writer
may only cite IDs that are present in the ledger and in the evidence it was
explicitly shown.
"""

import json
import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# Matches [id], [id1, id2], or [id1,id2] — IDs may contain letters, digits, -, :
_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_ID_RE = re.compile(r"[\w:\-]+")


# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """A single piece of evidence retrieved by a tool call.

    Attributes:
        id:      Unique identifier (chunk_id or "web:<hash>").
        kind:    Source type: "doc" for document chunks, "web" for web results.
        source:  Filename (doc) or URL (web).
        heading: Section heading within the source (doc only; empty for web).
        url:     Full URL (web only; empty for doc).
        text:    The retrieved passage text.
        score:   Retrieval relevance score (optional, 0.0 default).
    """

    id: str
    kind: Literal["doc", "web"]
    source: str
    heading: str = ""
    url: str = ""
    text: str
    score: float = 0.0

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        """Ensure the ID is not blank."""
        if not v.strip():
            raise ValueError("Evidence id must not be empty")
        return v


# ---------------------------------------------------------------------------
# Evidence extraction from tool outputs
# ---------------------------------------------------------------------------


def extract_evidence(tool_output: str | list | dict) -> list[Evidence]:
    """Parse tool output into a list of Evidence items.

    Tool outputs may be:
    - A JSON string (single dict or list of dicts)
    - A list of result dicts (already parsed)
    - A single result dict

    Document search results (from search_documents / get_chunk_context) have
    keys: chunk_id, source, heading, text, score.
    Web search results have keys: id, url, title, text.

    Args:
        tool_output: Raw tool output (string or parsed structure).

    Returns:
        List of Evidence items. Empty on parse failure.
    """
    if isinstance(tool_output, str):
        try:
            parsed = json.loads(tool_output)
        except (json.JSONDecodeError, ValueError):
            # Not JSON; return empty list
            logger.debug("extract_evidence: could not parse as JSON: %r", tool_output[:80])
            return []
    else:
        parsed = tool_output

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        return []

    items: list[Evidence] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            ev = _dict_to_evidence(item)
            if ev:
                items.append(ev)
        except Exception as exc:
            logger.debug("extract_evidence: skipped item: %s", exc)

    return items


def _dict_to_evidence(item: dict) -> Optional[Evidence]:
    """Convert a result dict from any tool to an Evidence object."""
    # Web search result
    if item.get("id", "").startswith("web:") or "url" in item:
        return Evidence(
            id=item.get("id", f"web:{hash(item.get('url','')) & 0xFFFFFFFF:08x}"),
            kind="web",
            source=item.get("url", ""),
            url=item.get("url", ""),
            heading="",
            text=item.get("text", item.get("body", "")),
            score=item.get("score", 0.0),
        )

    # Document chunk result
    chunk_id = item.get("chunk_id", item.get("id", ""))
    if not chunk_id:
        return None
    return Evidence(
        id=chunk_id,
        kind="doc",
        source=item.get("source", ""),
        heading=item.get("heading", ""),
        url="",
        text=item.get("text", ""),
        score=item.get("score", 0.0),
    )


# ---------------------------------------------------------------------------
# Ledger operations
# ---------------------------------------------------------------------------


def merge_ledger(
    base: dict[str, Evidence], incoming: dict[str, Evidence]
) -> dict[str, Evidence]:
    """Merge two ledger dicts without overwriting existing entries.

    Args:
        base: The existing ledger (modified in place and returned).
        incoming: New evidence to add.

    Returns:
        The merged ledger (same object as base).
    """
    for eid, ev in incoming.items():
        if eid not in base:
            base[eid] = ev
    return base


def build_ledger(evidence_list: list[Evidence]) -> dict[str, Evidence]:
    """Build a ledger dict from a list of Evidence items.

    Args:
        evidence_list: List of Evidence items (duplicates ignored by ID).

    Returns:
        Dict mapping evidence ID to Evidence object.
    """
    return {ev.id: ev for ev in evidence_list}


# ---------------------------------------------------------------------------
# Citation parsing and checking
# ---------------------------------------------------------------------------


def parse_citations(text: str) -> list[str]:
    """Extract all citation IDs from inline [id] markers in text.

    Handles:
    - Single: [report-2024:0007]
    - Multiple in one bracket: [a:0001, b:0002]
    - Adjacent brackets: [a:0001][b:0002]

    Args:
        text: Markdown text containing [id] citation markers.

    Returns:
        List of unique citation IDs in the order first encountered.
    """
    seen: set[str] = set()
    ids: list[str] = []
    for match in _CITATION_RE.finditer(text):
        raw = match.group(1)
        for sub_id in _ID_RE.findall(raw):
            if sub_id not in seen:
                seen.add(sub_id)
                ids.append(sub_id)
    return ids


def check_citations(
    text: str, ledger: dict[str, Evidence]
) -> tuple[list[str], list[str]]:
    """Split cited IDs in text into valid (in ledger) and invalid sets.

    Args:
        text: The answer text with [id] citation markers.
        ledger: The shared evidence ledger.

    Returns:
        Tuple of (valid_ids, invalid_ids).
    """
    cited = parse_citations(text)
    valid = [cid for cid in cited if cid in ledger]
    invalid = [cid for cid in cited if cid not in ledger]
    return valid, invalid

"""
Deterministic evaluation metrics and golden dataset validation.

Exports:
    normalize_text(s)                     -- Lowercase and collapse whitespace
    contains_gold_evidence(text, gold)    -- Check if normalized text contains gold substring
    compute_evidence_recall(gold, texts)  -- Fraction of gold evidence found in retrieved texts
    is_abstention(text)                   -- Check if text represents canonical abstention
    validate_golden_dataset(...)          -- Validate golden set schema and evidence in corpus
"""

import json
import re
from pathlib import Path
from typing import Any

from src.config import ABSTAIN_TEXT


def normalize_text(s: str) -> str:
    """Lowercase text, strip markdown formatting marks, and collapse whitespace."""
    if not s:
        return ""
    # Strip markdown bold/italic/code marks (*, _, `)
    s = re.sub(r"[*_`]", "", s)
    return re.sub(r"\s+", " ", s.lower()).strip()


def contains_gold_evidence(text: str, gold_evidence: str) -> bool:
    """Check if normalized text contains the normalized gold evidence substring."""
    norm_gold = normalize_text(gold_evidence)
    if not norm_gold:
        return False
    return norm_gold in normalize_text(text)


def compute_evidence_recall(gold_evidence_list: list[str], retrieved_texts: list[str]) -> float:
    """Compute fraction of gold evidence items found in any of retrieved texts.

    Args:
        gold_evidence_list: List of gold substring phrases.
        retrieved_texts: List of text passages (from ledger or search).

    Returns:
        Float between 0.0 and 1.0. Returns 1.0 if gold_evidence_list is empty.
    """
    if not gold_evidence_list:
        return 1.0

    combined_retrieved = " ".join(normalize_text(t) for t in retrieved_texts)
    matches = 0
    for gold in gold_evidence_list:
        if normalize_text(gold) in combined_retrieved:
            matches += 1

    return matches / len(gold_evidence_list)


def is_abstention(answer: str) -> bool:
    """Check if answer represents an abstention according to ABSTAIN_TEXT."""
    if not answer:
        return False
    norm_answer = normalize_text(answer)
    norm_abstain = normalize_text(ABSTAIN_TEXT)
    # Check full phrase or essential core phrase
    return norm_abstain in norm_answer or "do not contain enough information to answer" in norm_answer


def is_false_abstention(answer: str, answerable: bool) -> bool:
    """True if question is answerable but agent abstained."""
    return answerable and is_abstention(answer)


def check_planner_decomposition(n_subquestions: int, question_type: str, expected_min: int = 1) -> bool:
    """Check whether planner decomposition matches requirements."""
    if question_type == "lookup":
        return n_subquestions == 1
    elif question_type == "multi_hop":
        return n_subquestions >= expected_min
    return n_subquestions >= 1


def check_web_routing(actual_used_web: bool, expected_web: bool) -> bool:
    """Check whether web tool usage matched expectation."""
    return actual_used_web == expected_web


def validate_golden_dataset(corpus_dir: Path, golden_path: Path, is_web: bool = False) -> list[str]:
    """Validate a golden set JSONL file against corpus files and schema requirements.

    Args:
        corpus_dir: Directory containing markdown corpus files.
        golden_path: Path to golden JSONL file.
        is_web: True if validating golden_web.jsonl, False for golden_core.jsonl.

    Returns:
        List of error messages. Empty list if validation passes.
    """
    errors: list[str] = []
    if not golden_path.exists():
        return [f"Golden set file not found: {golden_path}"]

    lines = [line.strip() for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    
    expected_count = 5 if is_web else 18
    if len(lines) != expected_count:
        errors.append(f"Expected {expected_count} items in {golden_path.name}, got {len(lines)}")

    type_counts: dict[str, int] = {}
    ids_seen: set[str] = set()

    # Load corpus texts
    corpus_cache: dict[str, str] = {}
    if corpus_dir.exists():
        for f in corpus_dir.iterdir():
            if f.is_file():
                corpus_cache[f.name] = normalize_text(f.read_text(encoding="utf-8"))

    for idx, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
        except Exception as e:
            errors.append(f"Line {idx}: JSON decode error: {e}")
            continue

        item_id = item.get("id")
        if not item_id or item_id in ids_seen:
            errors.append(f"Line {idx}: Invalid or duplicate id: {item_id}")
        else:
            ids_seen.add(item_id)

        q_type = item.get("type")
        type_counts[q_type] = type_counts.get(q_type, 0) + 1

        # Check required fields
        for field in ("question", "reference_answer", "gold_sources", "gold_evidence", "expected_subquestions_min", "answerable", "verified"):
            if field not in item:
                errors.append(f"Item {item_id}: Missing field '{field}'")

        if is_web and "expected_web" not in item:
            errors.append(f"Item {item_id}: Missing field 'expected_web'")

        # Evidence validation
        gold_sources = item.get("gold_sources", [])
        gold_evidence = item.get("gold_evidence", [])
        answerable = item.get("answerable", True)

        if not answerable:
            if gold_sources or gold_evidence:
                errors.append(f"Item {item_id}: Unanswerable question must have empty gold_sources and gold_evidence")
        else:
            # If doc-based or mixed, verify gold evidence exists in gold sources
            if gold_sources:
                if len(gold_evidence) < len(gold_sources) and not is_web:
                    errors.append(f"Item {item_id}: Number of gold_evidence phrases should match or exceed gold_sources")

                for g_ev in gold_evidence:
                    norm_g_ev = normalize_text(g_ev)
                    found = False
                    for src in gold_sources:
                        src_text = corpus_cache.get(src, "")
                        if not src_text:
                            errors.append(f"Item {item_id}: Source file '{src}' not found in corpus")
                        elif norm_g_ev in src_text:
                            found = True
                            break
                    if not found:
                        errors.append(f"Item {item_id}: Gold evidence '{g_ev}' not found in sources {gold_sources}")

    # Check distribution of question types
    if not is_web:
        expected_distribution = {
            "lookup": 5,
            "multi_hop": 5,
            "numeric": 3,
            "changed": 2,
            "unanswerable": 3,
        }
        for t, count in expected_distribution.items():
            if type_counts.get(t, 0) != count:
                errors.append(f"Expected {count} of type '{t}', found {type_counts.get(t, 0)}")
    else:
        expected_distribution = {
            "web_only": 3,
            "doc_only": 2,
        }
        for t, count in expected_distribution.items():
            if type_counts.get(t, 0) != count:
                errors.append(f"Expected {count} of type '{t}', found {type_counts.get(t, 0)}")

    return errors

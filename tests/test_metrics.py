"""
Tests for deterministic evaluation metrics and golden set validation.

All tests are offline (no live API calls).
"""

from pathlib import Path
import pytest

from evals.metrics import (
    normalize_text,
    contains_gold_evidence,
    compute_evidence_recall,
    is_abstention,
    is_false_abstention,
    check_planner_decomposition,
    check_web_routing,
    validate_golden_dataset,
)


def test_normalize_text():
    raw = "   CloudSync   PRO \n\n  v4.0   "
    assert normalize_text(raw) == "cloudsync pro v4.0"


def test_contains_gold_evidence():
    text = "The company reported total revenue of 312 million dollars in 2023."
    gold = "total revenue of 312 million dollars"
    assert contains_gold_evidence(text, gold)
    assert not contains_gold_evidence(text, "500 million dollars")


def test_compute_evidence_recall():
    gold = ["phrase one", "phrase two"]
    retrieved = ["Here is phrase one in chunk A.", "Some irrelevant text."]
    recall = compute_evidence_recall(gold, retrieved)
    assert recall == 0.5

    retrieved.append("And here is phrase two in chunk B.")
    assert compute_evidence_recall(gold, retrieved) == 1.0


def test_is_abstention():
    abstain_text = "The available documents do not contain enough information to answer this question."
    assert is_abstention(abstain_text)
    assert is_abstention(f"Prefix. {abstain_text}")
    assert not is_abstention("Revenue was 312 million dollars.")


def test_is_false_abstention():
    abstain_text = "The available documents do not contain enough information to answer this question."
    # Answerable question where agent abstained -> false abstention
    assert is_false_abstention(abstain_text, answerable=True)
    # Unanswerable question where agent abstained -> true abstention (not false abstention)
    assert not is_false_abstention(abstain_text, answerable=False)
    # Answerable question where agent provided answer -> not false abstention
    assert not is_false_abstention("Answer provided", answerable=True)


def test_check_planner_decomposition():
    assert check_planner_decomposition(1, "lookup")
    assert not check_planner_decomposition(2, "lookup")
    assert check_planner_decomposition(2, "multi_hop", expected_min=2)
    assert not check_planner_decomposition(1, "multi_hop", expected_min=2)


def test_check_web_routing():
    assert check_web_routing(True, True)
    assert check_web_routing(False, False)
    assert not check_web_routing(True, False)


def test_golden_core_validation():
    root = Path(__file__).resolve().parent.parent
    corpus_dir = root / "evals" / "corpus"
    golden_core = root / "evals" / "golden_core.jsonl"

    errors = validate_golden_dataset(corpus_dir, golden_core, is_web=False)
    assert not errors, f"Golden core validation errors: {errors}"


def test_golden_web_validation():
    root = Path(__file__).resolve().parent.parent
    corpus_dir = root / "evals" / "corpus"
    golden_web = root / "evals" / "golden_web.jsonl"

    errors = validate_golden_dataset(corpus_dir, golden_web, is_web=True)
    assert not errors, f"Golden web validation errors: {errors}"

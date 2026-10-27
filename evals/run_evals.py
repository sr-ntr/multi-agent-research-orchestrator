"""
Agent evaluation and ablation runner.

Runs evaluation suites ('core' or 'web') across configurations ('single',
'multi_noverify', 'multi', or 'all').

Features:
- Checkpoints after every single question; resumes seamlessly if interrupted.
- Calculates both deterministic metrics and calls LLM judges (faithfulness, correctness).
- Tracks token usage and latency.
- Refuses to evaluate unverified golden items unless --allow-unverified is specified.
- Prints estimated call count before running.

Usage:
    python -m evals.run_evals --suite core --mode single --limit 3 --allow-unverified
    python -m evals.run_evals --suite core --mode all --allow-unverified
    python -m evals.run_evals --suite web --mode multi --allow-unverified
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from evals.judges import (
    judge_abstention_fallback,
    judge_correctness,
    judge_faithfulness,
)
from evals.metrics import (
    check_planner_decomposition,
    check_web_routing,
    compute_evidence_recall,
    is_abstention,
    is_false_abstention,
)
from src.agents.graph import get_graph_mode
from src.agents.prompts import format_evidence_block
from src.ledger import check_citations
from src.llm import UsageCallback
from src.store import KnowledgeBase

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "evals" / "results"
GOLDEN_CORE = ROOT_DIR / "evals" / "golden_core.jsonl"
GOLDEN_WEB = ROOT_DIR / "evals" / "golden_web.jsonl"


def estimate_calls(n_questions: int, mode: str) -> int:
    """Estimate total LLM calls for a run given number of questions and mode."""
    # Approximate call counts per question:
    # single: 1 researcher (1-2 tools + structured) + 1 writer = ~3-4 agent + 2 judge = ~5-6
    # multi_noverify: 1 planner + 2 researchers (parallel) + 1 writer = ~6-8 agent + 2 judge = ~8-10
    # multi: 1 planner + 2 researchers + 1 writer + 1 verifier = ~8-10 agent + 2 judge = ~10-12
    calls_per_mode = {
        "single": 6,
        "multi_noverify": 9,
        "multi": 12,
    }
    if mode == "all":
        return n_questions * sum(calls_per_mode.values())
    return n_questions * calls_per_mode.get(mode, 10)


def load_dataset(suite: str, ids: Optional[list[str]] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Load and filter golden questions from the specified suite."""
    file_path = GOLDEN_CORE if suite == "core" else GOLDEN_WEB
    if not file_path.exists():
        raise FileNotFoundError(f"Suite dataset not found: {file_path}")

    items = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if ids and item.get("id") not in ids:
            continue
        items.append(item)
        if limit and len(items) >= limit:
            break
    return items


def evaluate_single_question(
    item: dict[str, Any],
    graph: Any,
    kb: KnowledgeBase,
    web_enabled: bool,
    mode: str,
) -> dict[str, Any]:
    """Execute the agent pipeline on one question and run all evaluations."""
    q_id = item["id"]
    question = item["question"]
    q_type = item.get("type", "lookup")
    reference = item.get("reference_answer", "")
    gold_evidence = item.get("gold_evidence", [])
    answerable = item.get("answerable", True)
    expected_subq = item.get("expected_subquestions_min", 1)
    expected_web = item.get("expected_web", False)

    usage_cb = UsageCallback()

    initial_state = {
        "question": question,
        "web_enabled": web_enabled,
        "plan": None,
        "findings": [],
        "ledger": {},
        "draft": "",
        "revision_count": 0,
        "unsupported_claims": [],
        "final_answer": "",
        "trace": [],
        "tool_call_log": [],
        "token_usage": {},
    }

    t0 = time.time()
    try:
        final_state = graph.invoke(initial_state, config={"callbacks": [usage_cb]})
        elapsed = time.time() - t0
        error = None
    except Exception as exc:
        logger.error("Failed to run graph for question %s: %s", q_id, exc)
        elapsed = time.time() - t0
        return {
            "id": q_id,
            "error": str(exc),
            "latency": elapsed,
            "success": False,
        }

    answer = final_state.get("final_answer", "")
    trace = final_state.get("trace", [])
    ledger = final_state.get("ledger", {})
    tool_calls = final_state.get("tool_call_log", [])
    plan = final_state.get("plan")

    # --- 1. Deterministic Metrics ---
    # Evidence Recall
    ledger_texts = [ev.text for ev in ledger.values()]
    evidence_recall = compute_evidence_recall(gold_evidence, ledger_texts)

    # Citation Validity
    valid_cids, invalid_cids = check_citations(answer, ledger)
    total_cited = len(valid_cids) + len(invalid_cids)
    citation_validity = (len(valid_cids) / total_cited) if total_cited > 0 else 1.0

    # Abstention detection
    phrase_abstain = is_abstention(answer)
    if not answerable:
        if phrase_abstain:
            abstention_correct = True
        else:
            # Fallback to LLM judge if phrase check is inconclusive
            abstain_verdict = judge_abstention_fallback(question, answer)
            abstention_correct = abstain_verdict.is_abstention
    else:
        abstention_correct = not phrase_abstain

    false_abstention = is_false_abstention(answer, answerable)

    # Planner decomposition check
    if plan and getattr(plan, "subquestions", None):
        n_subq = len(plan.subquestions)
    else:
        n_subq = 1
    planner_ok = check_planner_decomposition(n_subq, q_type, expected_subq)

    # Web routing check
    web_calls = [tc for tc in tool_calls if tc.get("tool") == "web_search"]
    actual_used_web = len(web_calls) > 0
    web_routing_ok = check_web_routing(actual_used_web, expected_web)

    # --- 2. LLM Judge Metrics ---
    # Faithfulness
    cited_evidences = [ledger[cid] for cid in valid_cids if cid in ledger]
    evidence_block = format_evidence_block(cited_evidences)
    faithfulness_result = judge_faithfulness(answer, evidence_block)

    # Correctness (only evaluated if answerable)
    if answerable:
        correctness_result = judge_correctness(question, answer, reference)
        correctness_score = correctness_result.score
        correctness_reason = correctness_result.reason
    else:
        correctness_score = 1.0 if abstention_correct else 0.0
        correctness_reason = "Honest abstention on unanswerable question" if abstention_correct else "Failed to abstain"

    return {
        "id": q_id,
        "type": q_type,
        "question": question,
        "answer": answer,
        "reference_answer": reference,
        "answerable": answerable,
        "metrics": {
            "evidence_recall": round(evidence_recall, 4),
            "citation_validity": round(citation_validity, 4),
            "faithfulness": round(faithfulness_result.score, 4),
            "correctness": round(correctness_score, 4),
            "correctness_reason": correctness_reason,
            "abstention_correct": abstention_correct,
            "false_abstention": false_abstention,
            "planner_decomposition_ok": planner_ok,
            "n_subquestions": n_subq,
            "web_routing_ok": web_routing_ok,
            "used_web": actual_used_web,
            "latency_s": round(elapsed, 2),
            "tool_call_count": len(tool_calls),
            "prompt_tokens": usage_cb.prompt_tokens,
            "completion_tokens": usage_cb.completion_tokens,
            "total_tokens": usage_cb.total_tokens,
        },
        "trace": trace,
        "tool_calls": tool_calls,
        "valid_citations": valid_cids,
        "invalid_citations": invalid_cids,
        "success": True,
    }


def run_eval_suite_for_mode(
    suite: str,
    mode: str,
    questions: list[dict[str, Any]],
    kb: KnowledgeBase,
    allow_unverified: bool,
) -> Path:
    """Run an evaluation suite for a specific mode with incremental checkpointing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"{timestamp}_{suite}_{mode}.json"

    # Verify verified flag unless overridden
    for q in questions:
        if not q.get("verified", False) and not allow_unverified:
            raise ValueError(
                f"Question {q['id']} is marked 'verified: false'. "
                "Verify human reference answers or pass --allow-unverified to proceed."
            )

    web_enabled = (suite == "web")
    graph = get_graph_mode(mode, kb=kb, web_enabled=web_enabled)

    records: list[dict[str, Any]] = []

    print(f"\n--- Running Suite: {suite} | Mode: {mode} ({len(questions)} questions) ---")

    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Question {item['id']}: {item['question'][:65]}...", end="", flush=True)
        res = evaluate_single_question(item, graph, kb, web_enabled, mode)
        records.append(res)

        if res.get("success"):
            m = res["metrics"]
            print(f" Done ({res['metrics']['latency_s']}s | Faith: {m['faithfulness']} | Corr: {m['correctness']})")
        else:
            print(f" ERROR: {res.get('error')}")

        # Checkpoint to file after each question
        checkpoint_data = {
            "timestamp": timestamp,
            "suite": suite,
            "mode": mode,
            "total_questions": len(questions),
            "completed": len(records),
            "results": records,
        }
        out_file.write_text(json.dumps(checkpoint_data, indent=2), encoding="utf-8")

    print(f"Suite completed. Checkpoint written to: {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent evaluations and ablation study.")
    parser.add_argument("--suite", choices=["core", "web"], default="core", help="Evaluation suite")
    parser.add_argument("--mode", choices=["single", "multi_noverify", "multi", "all"], default="all", help="Agent mode")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for dry runs)")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated question IDs to evaluate")
    parser.add_argument("--allow-unverified", action="store_true", help="Allow running unverified golden items")
    parser.add_argument("--persist", default=".chroma", help="Path to Chroma persist directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    ids = [x.strip() for x in args.ids.split(",")] if args.ids else None
    questions = load_dataset(args.suite, ids=ids, limit=args.limit)

    modes = ["single", "multi_noverify", "multi"] if args.mode == "all" else [args.mode]

    # Rule 11: Print estimated call count
    estimated_calls = sum(estimate_calls(len(questions), m) for m in modes)
    print(f"\n[Evaluation Plan] Suite: '{args.suite}' | Questions: {len(questions)} | Modes: {modes}")
    print(f"[Cost Notice] Estimated model calls: ~{estimated_calls}")
    if len(questions) > 3 and not args.limit and args.mode == "all":
        print("[Recommendation] Run with '--limit 3' first as a dry run to verify pipeline integrity.")

    kb = KnowledgeBase(persist_dir=args.persist)
    docs = kb.list_documents()
    if not docs:
        print(f"Error: KnowledgeBase at {args.persist} has no documents. Ingest corpus first:")
        print("  python -m src.ingest evals/corpus --persist .chroma")
        return

    for mode in modes:
        run_eval_suite_for_mode(args.suite, mode, questions, kb, args.allow_unverified)


if __name__ == "__main__":
    main()

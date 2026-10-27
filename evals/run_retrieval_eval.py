"""
Retrieval evaluation runner: vector vs BM25 vs hybrid at k in {3, 6}.

Evaluates retrieval quality directly on the knowledge base without calling LLM agents:
- Hit rate at k: fraction of questions where at least one retrieved chunk
  contains at least one gold evidence string.
- MRR (Mean Reciprocal Rank): mean of 1 / rank of the first chunk containing gold evidence.

Saves results to evals/results/retrieval_eval.json and evals/results/retrieval_eval.md.

Usage:
    python -m evals.run_retrieval_eval [--persist .chroma] [--k 3 6]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from evals.metrics import contains_gold_evidence, normalize_text
from src.store import KnowledgeBase

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_CORE_PATH = ROOT_DIR / "evals" / "golden_core.jsonl"
RESULTS_DIR = ROOT_DIR / "evals" / "results"


def evaluate_retrieval(
    kb: KnowledgeBase,
    k_values: list[int] = [3, 6],
    golden_path: Path = GOLDEN_CORE_PATH,
) -> dict[str, Any]:
    """Run retrieval evaluation comparing vector, bm25, and hybrid search.

    Args:
        kb: Initialized KnowledgeBase instance.
        k_values: List of k thresholds (default: [3, 6]).
        golden_path: Path to golden_core.jsonl.

    Returns:
        Dictionary with overall metrics and detailed per-question results.
    """
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {golden_path}")

    # Load only answerable questions with gold evidence
    questions = []
    for line in golden_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("answerable", True) and item.get("gold_evidence"):
            questions.append(item)

    if not questions:
        raise ValueError("No answerable questions with gold evidence found.")

    modes = ["vector", "bm25", "hybrid"]
    max_k = max(k_values)

    # Structure to hold results: mode -> list of question results
    results: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}

    for item in questions:
        q_id = item["id"]
        q_text = item["question"]
        gold_evidence = item["gold_evidence"]

        for mode in modes:
            # Fetch up to max_k chunks
            retrieved = kb.search(q_text, top_k=max_k, mode=mode)

            # Determine rank of first chunk containing any gold evidence
            first_hit_rank = 0  # 1-based rank; 0 means no hit in top max_k
            chunk_matches = []

            for rank, chunk in enumerate(retrieved, start=1):
                chunk_text = chunk.get("text", "")
                has_gold = any(contains_gold_evidence(chunk_text, g) for g in gold_evidence)
                chunk_matches.append(has_gold)
                if has_gold and first_hit_rank == 0:
                    first_hit_rank = rank

            reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0

            # Hit indicators at each k
            hits_at_k = {k: (1 if 0 < first_hit_rank <= k else 0) for k in k_values}

            results[mode].append(
                {
                    "id": q_id,
                    "first_hit_rank": first_hit_rank,
                    "reciprocal_rank": reciprocal_rank,
                    "hits_at_k": hits_at_k,
                    "retrieved_count": len(retrieved),
                }
            )

    # Compute summary aggregates
    summary: dict[str, Any] = {"k_values": k_values, "n_questions": len(questions), "modes": {}}

    for mode in modes:
        mode_records = results[mode]
        mrr = sum(r["reciprocal_rank"] for r in mode_records) / len(mode_records)
        hit_rates = {}
        for k in k_values:
            hit_rates[f"hit_rate@{k}"] = sum(r["hits_at_k"][k] for r in mode_records) / len(mode_records)

        summary["modes"][mode] = {
            "mrr": round(mrr, 4),
            **{metric: round(val, 4) for metric, val in hit_rates.items()},
        }

    return {"summary": summary, "details": results}


def format_markdown_table(summary: dict[str, Any]) -> str:
    """Format retrieval evaluation summary into a markdown table."""
    k_vals = summary["k_values"]
    modes = summary["modes"]

    headers = ["Metric"] + list(modes.keys())
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(['---'] * len(headers))} |",
    ]

    for k in k_vals:
        metric_key = f"hit_rate@{k}"
        row = [f"Hit Rate @ {k}"] + [f"{modes[m][metric_key]:.2%}" for m in modes]
        lines.append(f"| {' | '.join(row)} |")

    mrr_row = ["MRR"] + [f"{modes[m]['mrr']:.4f}" for m in modes]
    lines.append(f"| {' | '.join(mrr_row)} |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation over golden set.")
    parser.add_argument("--persist", default=".chroma", help="Path to Chroma persist directory")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 6], help="k thresholds to evaluate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    kb = KnowledgeBase(persist_dir=args.persist)
    docs = kb.list_documents()
    if not docs:
        print(f"Error: KnowledgeBase at {args.persist} has no documents. Ingest corpus first:")
        print("  python -m src.ingest evals/corpus --persist .chroma")
        return

    print(f"Evaluating retrieval over {len(docs)} documents at k={args.k}...")
    output = evaluate_retrieval(kb, k_values=args.k)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "retrieval_eval.json"
    md_path = RESULTS_DIR / "retrieval_eval.md"

    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    md_table = format_markdown_table(output["summary"])
    md_report = f"# Retrieval Evaluation Results\n\nEvaluated on {output['summary']['n_questions']} answerable questions.\n\n{md_table}\n"
    md_path.write_text(md_report, encoding="utf-8")

    print("\n" + md_report)
    print(f"Results saved to {json_path} and {md_path}")


if __name__ == "__main__":
    main()

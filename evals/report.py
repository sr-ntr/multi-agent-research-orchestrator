"""
Report generator: builds ablation and evaluation markdown tables from result JSON files.

Finds the latest result runs in evals/results/ and outputs:
- evals/results/SUMMARY.md (Ablation table + Web evaluation table)
- evals/results/FAILURE_ANALYSIS.md (Draft analysis of low-scoring/failed cases)

Usage:
    python -m evals.report
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "evals" / "results"
SUMMARY_FILE = RESULTS_DIR / "SUMMARY.md"
FAILURE_FILE = RESULTS_DIR / "FAILURE_ANALYSIS.md"


def find_latest_result(suite: str, mode: str) -> Optional[Path]:
    """Find the most recent results JSON file matching the given suite and mode."""
    if not RESULTS_DIR.exists():
        return None
    matches = sorted(RESULTS_DIR.glob(f"*_{suite}_{mode}.json"))
    return matches[-1] if matches else None


def aggregate_run_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate evaluation metrics across all questions in a run."""
    successful = [r for r in results if r.get("success")]
    if not successful:
        return {}

    n = len(successful)

    # Core metrics
    recalls = [r["metrics"]["evidence_recall"] for r in successful]
    faithfulness = [r["metrics"]["faithfulness"] for r in successful]
    correctness_all = [r["metrics"]["correctness"] for r in successful]

    # Multi-hop correctness
    multihop_records = [r for r in successful if r.get("type") == "multi_hop"]
    correctness_multihop = (
        [r["metrics"]["correctness"] for r in multihop_records] if multihop_records else []
    )

    # Abstention metrics
    unanswerable = [r for r in successful if not r.get("answerable", True)]
    abstention_accuracy = (
        sum(1 for r in unanswerable if r["metrics"]["abstention_correct"]) / len(unanswerable)
        if unanswerable
        else 1.0
    )

    answerable = [r for r in successful if r.get("answerable", True)]
    false_abstention_rate = (
        sum(1 for r in answerable if r["metrics"]["false_abstention"]) / len(answerable)
        if answerable
        else 0.0
    )

    # Web routing
    web_routing_accuracy = sum(1 for r in successful if r["metrics"]["web_routing_ok"]) / n

    # Performance
    latencies = [r["metrics"]["latency_s"] for r in successful]
    tokens = [r["metrics"]["total_tokens"] for r in successful]

    return {
        "n_questions": n,
        "evidence_recall": round(sum(recalls) / n, 4),
        "faithfulness": round(sum(faithfulness) / n, 4),
        "correctness_overall": round(sum(correctness_all) / n, 4),
        "correctness_multihop": round(sum(correctness_multihop) / len(correctness_multihop), 4) if correctness_multihop else 0.0,
        "abstention_accuracy": round(abstention_accuracy, 4),
        "false_abstention_rate": round(false_abstention_rate, 4),
        "web_routing_accuracy": round(web_routing_accuracy, 4),
        "avg_latency_s": round(sum(latencies) / n, 2),
        "avg_tokens": round(sum(tokens) / n, 1),
    }


def generate_ablation_markdown(core_data: dict[str, dict[str, Any]]) -> str:
    """Generate the markdown comparison table across single, multi_noverify, and multi."""
    modes = ["single", "multi_noverify", "multi"]

    headers = ["Metric", "single", "multi_noverify", "multi"]
    rows = [
        ("Evidence recall", "evidence_recall", "{:.2%}"),
        ("Faithfulness", "faithfulness", "{:.2%}"),
        ("Correctness (overall)", "correctness_overall", "{:.2%}"),
        ("Correctness (multi-hop only)", "correctness_multihop", "{:.2%}"),
        ("Abstention accuracy", "abstention_accuracy", "{:.2%}"),
        ("False abstention rate", "false_abstention_rate", "{:.2%}"),
        ("Avg latency (s)", "avg_latency_s", "{:.1f}s"),
        ("Avg tokens per question", "avg_tokens", "{:.0f}"),
    ]

    lines = [
        "## Agent Architecture Ablation Study",
        "",
        "Comparison of single-agent, multi-agent without verifier, and full multi-agent pipeline with verifier on the core golden dataset.",
        "",
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(['---'] * len(headers))} |",
    ]

    for label, key, fmt in rows:
        row_vals = []
        for m in modes:
            m_data = core_data.get(m, {})
            val = m_data.get(key)
            if val is not None:
                row_vals.append(fmt.format(val))
            else:
                row_vals.append("Not run")
        lines.append(f"| {label} | {' | '.join(row_vals)} |")

    return "\n".join(lines)


def generate_web_markdown(web_data: dict[str, Any]) -> str:
    """Generate markdown table for web evaluation results."""
    lines = [
        "## Web Search Routing Evaluation",
        "",
        "Evaluation on golden web dataset testing search routing and factual accuracy.",
        "",
        "| Metric | Result |",
        "| --- | --- |",
    ]
    if not web_data:
        lines.append("| Status | Not run |")
        return "\n".join(lines)

    lines.append(f"| Web routing accuracy | {web_data.get('web_routing_accuracy', 0):.2%} |")
    lines.append(f"| Evidence recall | {web_data.get('evidence_recall', 0):.2%} |")
    lines.append(f"| Correctness (overall) | {web_data.get('correctness_overall', 0):.2%} |")
    lines.append(f"| Faithfulness | {web_data.get('faithfulness', 0):.2%} |")
    lines.append(f"| Avg latency (s) | {web_data.get('avg_latency_s', 0):.1f}s |")
    lines.append(f"| Avg tokens per question | {web_data.get('avg_tokens', 0):.0f} |")

    return "\n".join(lines)


def generate_failure_analysis_draft(all_runs: list[dict[str, Any]]) -> str:
    """Draft failure analysis for failing or low-scoring cases across runs."""
    lines = [
        "# Failure Analysis (Draft, Pending Human Review)",
        "",
        "Analysis of questions scoring below 1.0 or encountering errors during evaluation.",
        "",
    ]

    seen_issues = 0
    for run in all_runs:
        suite = run.get("suite", "")
        mode = run.get("mode", "")
        for res in run.get("results", []):
            if not res.get("success"):
                seen_issues += 1
                lines.append(f"### [{res.get('id')}] Mode: {mode} ({suite}) - Execution Error")
                lines.append(f"- **Question**: {res.get('question')}")
                lines.append(f"- **Error**: `{res.get('error')}`")
                lines.append("- **Likely cause**: Pipeline execution exception / network error.\n")
                continue

            m = res["metrics"]
            # Flag low correctness, low faithfulness, or false abstention
            if m["correctness"] < 1.0 or m["faithfulness"] < 1.0 or m["false_abstention"]:
                seen_issues += 1
                lines.append(f"### [{res['id']}] Mode: {mode} ({suite}) - Score: {m['correctness']}")
                lines.append(f"- **Question**: {res['question']}")
                lines.append(f"- **Reference**: {res.get('reference_answer')}")
                lines.append(f"- **Answer**: {res.get('answer')}")
                lines.append(f"- **Judge Reason**: {m.get('correctness_reason')}")
                
                # Derive candidate cause
                if m["false_abstention"]:
                    cause = "False abstention: retrieval missed relevant chunks or researcher returned found=False prematurely."
                elif m["evidence_recall"] < 1.0:
                    cause = "Retrieval missed: one or more gold evidence passages were not retrieved."
                elif m["faithfulness"] < 1.0:
                    cause = "Writer overstated: claim introduced in draft was not entailed by retrieved evidence."
                else:
                    cause = "Judge strictness or nuance in wording compared to gold reference."
                lines.append(f"- **Likely cause**: {cause}\n")

    if seen_issues == 0:
        lines.append("No failures or low-scoring cases found in the analyzed runs.\n")

    return "\n".join(lines)


def generate_reports() -> None:
    """Generate SUMMARY.md and FAILURE_ANALYSIS.md from latest result files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    core_modes = ["single", "multi_noverify", "multi"]
    core_summaries = {}
    all_runs = []

    for m in core_modes:
        path = find_latest_result("core", m)
        if path:
            data = json.loads(path.read_text(encoding="utf-8"))
            core_summaries[m] = aggregate_run_metrics(data.get("results", []))
            all_runs.append(data)

    web_path = find_latest_result("web", "multi")
    web_summary = {}
    if web_path:
        web_data = json.loads(web_path.read_text(encoding="utf-8"))
        web_summary = aggregate_run_metrics(web_data.get("results", []))
        all_runs.append(web_data)

    # Check for retrieval eval report
    retrieval_md = ""
    retrieval_path = RESULTS_DIR / "retrieval_eval.md"
    if retrieval_path.exists():
        retrieval_md = retrieval_path.read_text(encoding="utf-8") + "\n\n---\n\n"

    # Build SUMMARY.md
    summary_parts = [
        "# Evaluation Results Summary",
        "",
        retrieval_md,
        generate_ablation_markdown(core_summaries),
        "",
        "---",
        "",
        generate_web_markdown(web_summary),
        "",
    ]
    summary_content = "\n".join(summary_parts)
    SUMMARY_FILE.write_text(summary_content, encoding="utf-8")
    print(f"Summary written to: {SUMMARY_FILE}")

    # Build FAILURE_ANALYSIS.md
    failure_content = generate_failure_analysis_draft(all_runs)
    FAILURE_FILE.write_text(failure_content, encoding="utf-8")
    print(f"Failure analysis written to: {FAILURE_FILE}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_reports()


if __name__ == "__main__":
    main()

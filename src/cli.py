"""
CLI entry point for the Agentic Document Research Assistant.

Usage:
    python -m src.cli ask "What is the main product?" --persist .chroma
    python -m src.cli ask "..." --mode multi --web on --persist .chroma

Options:
    --mode    single | multi_noverify | multi  (default: multi)
    --web     on | off                          (default: off)
    --persist PATH                              (Chroma persist dir; default: .chroma)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from src.config import ABSTAIN_TEXT


def _run_ask(args: argparse.Namespace) -> None:
    """Execute the ask subcommand."""
    from src.store import KnowledgeBase
    from src.agents.graph import get_graph_mode
    from src.ledger import check_citations

    persist = args.persist if args.persist else None
    web_enabled = args.web.lower() in ("on", "true", "1")
    mode = args.mode

    print(f"\nMode: {mode}  Web: {'on' if web_enabled else 'off'}")
    print(f"Persist: {persist or '(ephemeral)'}")
    print()

    kb = KnowledgeBase(persist_dir=persist)

    docs = kb.list_documents()
    if not docs:
        print("WARNING: No documents in the knowledge base. Ingest some first:")
        print("  python -m src.ingest <path> --persist .chroma")
        print()

    graph = get_graph_mode(mode, kb=kb, web_enabled=web_enabled)

    initial_state = {
        "question": args.question,
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
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - t0

    # --- Print trace ---
    print("=== Trace ===")
    for line in final_state.get("trace", []):
        print(f"  {line}")
    print()

    # --- Print tool calls ---
    tool_calls = final_state.get("tool_call_log", [])
    if tool_calls:
        print(f"=== Tool calls ({len(tool_calls)}) ===")
        for tc in tool_calls:
            print(f"  [{tc['subquestion_id']}] {tc['tool']}({tc['args']})")
        print()

    # --- Print citations ---
    ledger = final_state.get("ledger", {})
    answer = final_state.get("final_answer", "")
    valid_ids, invalid_ids = check_citations(answer, ledger)

    if valid_ids:
        print("=== Citations ===")
        for cid in valid_ids:
            ev = ledger[cid]
            if ev.kind == "doc":
                print(f"  [{cid}] {ev.source} > {ev.heading or '(top level)'}")
            else:
                print(f"  [{cid}] {ev.url}")
        print()

    if invalid_ids:
        print(f"WARNING: Invalid citations found: {invalid_ids}")
        print()

    # --- Print answer ---
    print("=== Answer ===")
    print(answer)
    print()

    # --- Stats ---
    print(f"Latency: {elapsed:.1f}s  |  Tool calls: {len(tool_calls)}")


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic Document Research Assistant CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ask subcommand
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="The question to answer")
    ask_parser.add_argument(
        "--mode",
        choices=["single", "multi_noverify", "multi"],
        default="multi",
        help="Agent configuration (default: multi)",
    )
    ask_parser.add_argument(
        "--web", default="off", choices=["on", "off"], help="Enable web search"
    )
    ask_parser.add_argument(
        "--persist",
        default=".chroma",
        help="Chroma persist directory (default: .chroma)",
    )
    ask_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "ask":
        _run_ask(args)


if __name__ == "__main__":
    _main()

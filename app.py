"""
Streamlit UI for the Agentic Document Research Assistant.

Usage:
    streamlit run app.py

Features:
- Ephemeral in-memory KnowledgeBase per session.
- Document upload (.txt, .md) with chunking and hybrid search index.
- Mode selection: single, multi_noverify, multi.
- Optional web search routing.
- Live execution trace streaming using st.status.
- Interactive citation inspection cards for retrieved passages.
- Question limit per session (capped at MAX_QUESTIONS_PER_SESSION).
- Strictly no emojis.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src.agents.graph import get_graph_mode
from src.config import (
    GROQ_API_KEY,
    MAX_FILES_PER_SESSION,
    MAX_FILE_BYTES,
    MAX_QUESTIONS_PER_SESSION,
)
from src.ingest import ingest_file
from src.ledger import check_citations
from src.store import KnowledgeBase

st.set_page_config(
    page_title="Document Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "kb" not in st.session_state:
        # Create ephemeral KnowledgeBase for this session
        st.session_state.kb = KnowledgeBase()
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = {}


def reset_documents() -> None:
    """Clear all ingested documents and reinitialize ephemeral knowledge base."""
    st.session_state.kb = KnowledgeBase()
    st.session_state.ingested_files = {}
    st.session_state.messages = []
    st.session_state.question_count = 0


def render_sidebar() -> tuple[str, bool, str]:
    """Render sidebar controls and document management."""
    st.sidebar.title("Settings")

    # API Key Configuration
    api_key_input = st.sidebar.text_input(
        "Groq API Key",
        value="",
        type="password",
        help="Leave blank to use GROQ_API_KEY from environment or .env.",
    )
    effective_api_key = api_key_input.strip() or GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    if effective_api_key:
        os.environ["GROQ_API_KEY"] = effective_api_key

    # Agent Mode
    mode = st.sidebar.selectbox(
        "Orchestration Mode",
        options=["multi", "multi_noverify", "single"],
        index=0,
        help="multi: Planner + Parallel Researchers + Verifier\nmulti_noverify: Planner + Parallel Researchers\nsingle: Single Researcher direct answer",
    )

    # Web Search Toggle
    web_enabled = st.sidebar.checkbox(
        "Enable Web Search",
        value=False,
        help="Allow researchers to search the web for external information.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Document Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        f"Upload .txt or .md files (max {MAX_FILES_PER_SESSION})",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if len(uploaded_files) > MAX_FILES_PER_SESSION:
            st.sidebar.error(f"Maximum of {MAX_FILES_PER_SESSION} files allowed.")
        else:
            if st.sidebar.button("Ingest Files", use_container_width=True):
                with st.sidebar.status("Ingesting documents...", expanded=True) as status:
                    for uf in uploaded_files:
                        if uf.size > MAX_FILE_BYTES:
                            st.write(f"Skipped {uf.name}: Exceeds 1 MB limit.")
                            continue

                        # Write to temporary file for ingest_file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp:
                            tmp.write(uf.getvalue())
                            tmp_path = Path(tmp.name)

                        try:
                            # Use original filename for metadata
                            renamed_path = tmp_path.parent / uf.name
                            tmp_path.rename(renamed_path)
                            res = ingest_file(renamed_path, st.session_state.kb)
                            st.session_state.ingested_files[uf.name] = res.get("chunks", 0)
                            st.write(f"{uf.name}: {res.get('chunks', 0)} chunks ({res.get('status')})")
                            renamed_path.unlink(missing_ok=True)
                        except Exception as e:
                            st.write(f"Error ingesting {uf.name}: {e}")
                    status.update(label="Ingestion complete", state="complete", expanded=False)

    # Ingested documents table
    if st.session_state.ingested_files:
        st.sidebar.write("Ingested Documents:")
        docs_summary = [
            {"Document": name, "Chunks": count}
            for name, count in st.session_state.ingested_files.items()
        ]
        st.sidebar.dataframe(docs_summary, use_container_width=True, hide_index=True)

        if st.sidebar.button("Clear Documents", use_container_width=True):
            reset_documents()
            st.rerun()

    # Session limit indicator
    remaining_questions = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
    st.sidebar.caption(f"Session questions remaining: {max(0, remaining_questions)} of {MAX_QUESTIONS_PER_SESSION}")

    return mode, web_enabled, effective_api_key


def render_citations(valid_ids: list[str], ledger: dict[str, Any]) -> None:
    """Render citations as expandable cards."""
    if not valid_ids:
        return

    st.markdown("##### Verified Sources and Citations")
    for cid in valid_ids:
        ev = ledger.get(cid)
        if not ev:
            continue

        if ev.kind == "web":
            title = f"Web Source: [{cid}] {ev.source or ev.url}"
            with st.expander(title):
                if ev.url:
                    st.markdown(f"**URL**: [{ev.url}]({ev.url})")
                st.markdown(f"**Passage**:\n{ev.text}")
        else:
            heading_info = f" > {ev.heading}" if ev.heading else ""
            title = f"Document: [{cid}] {ev.source}{heading_info}"
            with st.expander(title):
                st.markdown(f"**Source File**: `{ev.source}`")
                if ev.heading:
                    st.markdown(f"**Section**: `{ev.heading}`")
                st.markdown(f"**Passage**:\n{ev.text}")


def main() -> None:
    init_session_state()
    mode, web_enabled, api_key = render_sidebar()

    st.title("Document Research Assistant")
    st.caption("Grounded document research with Groq LPU models, FastEmbed hybrid retrieval, and citation verification.")

    if not api_key:
        st.warning("Please provide a Groq API Key in the sidebar or set GROQ_API_KEY in your .env file.")

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and "ledger" in msg:
                render_citations(msg["citations"], msg["ledger"])
            if "trace" in msg and msg["trace"]:
                with st.expander("Execution Trace", expanded=False):
                    for line in msg["trace"]:
                        st.text(line)

    # Chat input
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.info("Session question limit reached. Clear documents or restart the session to ask more questions.")
        return

    user_query = st.chat_input("Ask a question about the uploaded documents...")
    if user_query:
        if not api_key:
            st.error("API key is required to process questions.")
            return

        st.session_state.question_count += 1
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant response container
        with st.chat_message("assistant"):
            status_container = st.status("Initializing research pipeline...", expanded=True)

            try:
                graph = get_graph_mode(
                    mode,
                    kb=st.session_state.kb,
                    web_enabled=web_enabled,
                )

                initial_state = {
                    "question": user_query,
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

                accumulated_trace = []
                final_state = {}

                # Stream updates through the LangGraph nodes
                for update in graph.stream(initial_state, stream_mode="updates"):
                    for node_name, node_output in update.items():
                        if isinstance(node_output, dict):
                            traces = node_output.get("trace", [])
                            for t in traces:
                                accumulated_trace.append(t)
                                status_container.write(f"`{t}`")
                        # Keep latest state output
                        if isinstance(node_output, dict):
                            final_state.update(node_output)

                # Fetch final result fields
                answer = final_state.get("final_answer", "")
                if not answer:
                    answer = final_state.get("draft", "No answer generated.")
                ledger = final_state.get("ledger", {})

                valid_ids, invalid_ids = check_citations(answer, ledger)

                status_container.update(label="Research complete", state="complete", expanded=False)

                st.markdown(answer)
                render_citations(valid_ids, ledger)

                # Store assistant turn
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": valid_ids,
                        "ledger": ledger,
                        "trace": accumulated_trace,
                    }
                )

            except Exception as exc:
                status_container.update(label="Error executing request", state="error", expanded=True)
                st.error(f"Execution failed: {exc}")


if __name__ == "__main__":
    main()

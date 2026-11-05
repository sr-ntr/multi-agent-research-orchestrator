"""
Streamlit UI for the Multi-Agent Research Assistant.

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
- Server-side API key (no user key input required).
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

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Research Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS: Computer Modern fonts + pastel academic theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Import Computer Modern web fonts */
@import url('https://cdn.jsdelivr.net/gh/dreampulse/computer-modern-web-font@master/font.css');

/* ------------------------------------------------------------------ */
/* Typography                                                           */
/* ------------------------------------------------------------------ */
html, body, [class*="css"], .stMarkdown, .stTextInput, .stSelectbox,
.stCheckbox, .stButton, .stChatMessage, .stExpander, .stDataFrame,
.stCaption, .stSidebar, p, span, div, label {
    font-family: "Computer Modern Serif", "CMU Serif", "Latin Modern", Georgia, serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: "Computer Modern Serif", "CMU Serif", serif !important;
    font-weight: normal !important;
    letter-spacing: -0.01em;
    color: #1A1A2E !important;
}

code, pre, .stCodeBlock, kbd {
    font-family: "Computer Modern Typewriter", "CMU Typewriter Text", "Courier New", monospace !important;
    font-size: 0.88em;
}

/* ------------------------------------------------------------------ */
/* Layout & spacing                                                    */
/* ------------------------------------------------------------------ */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 900px;
}

/* ------------------------------------------------------------------ */
/* Pastel color overrides                                              */
/* ------------------------------------------------------------------ */

/* Main background */
.stApp {
    background-color: #FAFAF6 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #F2EFE8 !important;
    border-right: 1px solid #D9D4CA;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #3D5A66 !important;
}

/* Divider */
hr {
    border-color: #D9D4CA !important;
}

/* ------------------------------------------------------------------ */
/* Chat messages                                                       */
/* ------------------------------------------------------------------ */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #EDE9DF !important;
    border-radius: 8px;
    padding: 0.75rem 1rem !important;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:not([data-testid*="user"]) {
    background-color: transparent !important;
}

/* ------------------------------------------------------------------ */
/* Chat input                                                          */
/* ------------------------------------------------------------------ */
[data-testid="stChatInput"] textarea {
    font-family: "Computer Modern Serif", "CMU Serif", serif !important;
    border: 1px solid #C9C4BA !important;
    border-radius: 6px !important;
    background-color: #FAFAF6 !important;
    color: #1A1A2E !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #6B8F9E !important;
    box-shadow: 0 0 0 2px rgba(107,143,158,0.15) !important;
}

/* ------------------------------------------------------------------ */
/* Buttons                                                             */
/* ------------------------------------------------------------------ */
.stButton > button {
    font-family: "Computer Modern Serif", "CMU Serif", serif !important;
    background-color: #FAFAF6 !important;
    border: 1px solid #BBBCB6 !important;
    color: #1A1A2E !important;
    border-radius: 5px !important;
    padding: 0.4rem 1rem !important;
    transition: background-color 0.15s ease, border-color 0.15s ease;
}

.stButton > button:hover {
    background-color: #EDE9DF !important;
    border-color: #6B8F9E !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background-color: #6B8F9E !important;
    border-color: #6B8F9E !important;
    color: #FAFAF6 !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #5A7A88 !important;
}

/* ------------------------------------------------------------------ */
/* Expander (citation cards)                                          */
/* ------------------------------------------------------------------ */
[data-testid="stExpander"] {
    border: 1px solid #D9D4CA !important;
    border-radius: 6px !important;
    background-color: #F7F4EE !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stExpander"] summary {
    font-family: "Computer Modern Serif", "CMU Serif", serif !important;
    font-size: 0.9em !important;
    color: #3D5A66 !important;
}

/* ------------------------------------------------------------------ */
/* Status widget (agent trace)                                        */
/* ------------------------------------------------------------------ */
[data-testid="stStatus"] {
    border: 1px solid #D9D4CA !important;
    border-radius: 6px !important;
    background-color: #F7F4EE !important;
    font-family: "Computer Modern Typewriter", "CMU Typewriter Text", monospace !important;
}

/* ------------------------------------------------------------------ */
/* Selectbox, checkboxes, and form elements                           */
/* ------------------------------------------------------------------ */
[data-testid="stSelectbox"] > div > div {
    background-color: #FAFAF6 !important;
    border: 1px solid #C9C4BA !important;
    border-radius: 5px !important;
}

[data-baseweb="select"] {
    font-family: "Computer Modern Serif", "CMU Serif", serif !important;
}

/* ------------------------------------------------------------------ */
/* File uploader                                                       */
/* ------------------------------------------------------------------ */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #BBBCB6 !important;
    border-radius: 6px !important;
    background-color: #F7F4EE !important;
    padding: 0.75rem !important;
}

/* ------------------------------------------------------------------ */
/* Dataframe (ingested documents table)                               */
/* ------------------------------------------------------------------ */
[data-testid="stDataFrame"] {
    border: 1px solid #D9D4CA !important;
    border-radius: 5px !important;
}

/* ------------------------------------------------------------------ */
/* Caption / small text                                               */
/* ------------------------------------------------------------------ */
[data-testid="stCaptionContainer"], .stCaption {
    color: #8A8680 !important;
    font-size: 0.82em !important;
}

/* ------------------------------------------------------------------ */
/* Alert / info / warning boxes                                       */
/* ------------------------------------------------------------------ */
[data-testid="stAlert"] {
    border-radius: 5px !important;
    border-left-width: 3px !important;
    font-family: "Computer Modern Serif", "CMU Serif", serif !important;
}

/* ------------------------------------------------------------------ */
/* Title section                                                       */
/* ------------------------------------------------------------------ */
.research-header {
    border-bottom: 1px solid #D9D4CA;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}

.research-header h1 {
    font-size: 1.9rem !important;
    margin-bottom: 0.2rem !important;
}

.research-header .subtitle {
    font-size: 0.92em;
    color: #6A6560;
    font-style: italic;
}

/* ------------------------------------------------------------------ */
/* Empty state                                                         */
/* ------------------------------------------------------------------ */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #8A8680;
}

.empty-state h3 {
    font-size: 1.1rem !important;
    color: #5A5550 !important;
    margin-bottom: 0.5rem !important;
}

.empty-state p {
    font-size: 0.9em;
    line-height: 1.6;
}

/* ------------------------------------------------------------------ */
/* Citation section header                                            */
/* ------------------------------------------------------------------ */
.citation-header {
    font-size: 0.85em !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7A7570 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.4rem !important;
    border-top: 1px solid #D9D4CA;
    padding-top: 0.75rem;
}

/* ------------------------------------------------------------------ */
/* Session progress bar / remaining count                             */
/* ------------------------------------------------------------------ */
.session-info {
    font-size: 0.8em;
    color: #8A8680;
    text-align: center;
    padding-top: 0.4rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = {}


def reset_session() -> None:
    """Clear all ingested documents and reinitialize the session."""
    st.session_state.kb = KnowledgeBase()
    st.session_state.ingested_files = {}
    st.session_state.messages = []
    st.session_state.question_count = 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[str, bool]:
    """Render sidebar controls. Returns (mode, web_enabled)."""
    with st.sidebar:
        st.markdown("## Research Assistant")
        st.markdown(
            "<span style='font-size:0.85em; color:#7A7570; font-style:italic;'>"
            "Multi-agent document research with citation verification"
            "</span>",
            unsafe_allow_html=True,
        )
        st.divider()

        # --- Mode ---
        st.markdown("**Orchestration Mode**")
        mode = st.selectbox(
            "Mode",
            options=["multi", "single", "multi_noverify"],
            index=0,
            format_func=lambda m: {
                "multi": "Full Pipeline (Planner + Verifier)",
                "single": "Single Researcher",
                "multi_noverify": "Planner, no Verification",
            }[m],
            label_visibility="collapsed",
            help=(
                "Full Pipeline: Planner decomposes the question, parallel Researchers gather "
                "evidence, Writer drafts an answer, and the Verifier audits every citation.\n\n"
                "Single Researcher: One agent searches and answers directly. Faster for simple lookups.\n\n"
                "Planner, no Verification: Parallel research but skips the verification step."
            ),
        )

        # --- Web search ---
        web_enabled = st.checkbox(
            "Enable web search",
            value=False,
            help="Allow researchers to search the web for information not in uploaded documents.",
        )

        st.divider()

        # --- Document ingestion ---
        st.markdown("**Upload Documents**")
        st.caption(
            f"Supported: .txt, .md   |   Max {MAX_FILES_PER_SESSION} files, 1 MB each"
        )

        uploaded_files = st.file_uploader(
            "Upload files",
            type=["txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            if len(uploaded_files) > MAX_FILES_PER_SESSION:
                st.error(f"Maximum {MAX_FILES_PER_SESSION} files allowed.")
            else:
                if st.button("Index Documents", use_container_width=True, type="primary"):
                    with st.status("Indexing documents...", expanded=True) as status:
                        for uf in uploaded_files:
                            if uf.size > MAX_FILE_BYTES:
                                st.write(f"Skipped {uf.name}: exceeds 1 MB limit.")
                                continue
                            if uf.name in st.session_state.ingested_files:
                                st.write(f"{uf.name}: already indexed, skipping.")
                                continue
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=f"_{uf.name}"
                            ) as tmp:
                                tmp.write(uf.getvalue())
                                tmp_path = Path(tmp.name)
                            try:
                                renamed = tmp_path.parent / uf.name
                                tmp_path.rename(renamed)
                                res = ingest_file(renamed, st.session_state.kb)
                                chunk_count = res.get("chunks", 0)
                                st.session_state.ingested_files[uf.name] = chunk_count
                                st.write(f"{uf.name}: {chunk_count} chunks indexed.")
                                renamed.unlink(missing_ok=True)
                            except Exception as exc:
                                st.write(f"Error indexing {uf.name}: {exc}")
                        status.update(
                            label="Indexing complete", state="complete", expanded=False
                        )
                    st.rerun()

        # --- Indexed documents table ---
        if st.session_state.ingested_files:
            st.markdown("**Indexed Documents**")
            st.dataframe(
                [
                    {"Document": name, "Chunks": count}
                    for name, count in st.session_state.ingested_files.items()
                ],
                use_container_width=True,
                hide_index=True,
            )
            if st.button("Clear All Documents", use_container_width=True):
                reset_session()
                st.rerun()

        st.divider()

        # --- Session quota ---
        remaining = max(0, MAX_QUESTIONS_PER_SESSION - st.session_state.question_count)
        st.markdown(
            f"<div class='session-info'>"
            f"{remaining} of {MAX_QUESTIONS_PER_SESSION} questions remaining this session"
            f"</div>",
            unsafe_allow_html=True,
        )

    return mode, web_enabled


# ---------------------------------------------------------------------------
# Citation cards
# ---------------------------------------------------------------------------


def render_citations(valid_ids: list[str], ledger: dict[str, Any]) -> None:
    """Render citation evidence as expandable cards."""
    if not valid_ids:
        return

    st.markdown(
        "<p class='citation-header'>Verified Sources</p>",
        unsafe_allow_html=True,
    )
    for cid in valid_ids:
        ev = ledger.get(cid)
        if not ev:
            continue
        if ev.kind == "web":
            label = f"[{cid}]  Web  |  {ev.url or ev.source}"
            with st.expander(label):
                if ev.url:
                    st.markdown(f"**URL**: [{ev.url}]({ev.url})")
                st.markdown(f"**Passage**\n\n{ev.text}")
        else:
            section = f"  >  {ev.heading}" if ev.heading else ""
            label = f"[{cid}]  {ev.source}{section}"
            with st.expander(label):
                st.markdown(f"**File**: `{ev.source}`")
                if ev.heading:
                    st.markdown(f"**Section**: {ev.heading}")
                st.markdown(f"**Passage**\n\n{ev.text}")


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


def render_empty_state() -> None:
    """Render a helpful prompt when no documents are loaded and no history exists."""
    st.markdown(
        """
<div class='empty-state'>
    <h3>No documents loaded yet</h3>
    <p>
        Upload <code>.txt</code> or <code>.md</code> files in the sidebar, then click
        <strong>Index Documents</strong>.<br>
        Once indexed, type a question below and the research pipeline will retrieve
        relevant passages, synthesize an answer, and cite every claim.
    </p>
    <p style='margin-top:1rem; font-size:0.85em;'>
        Try uploading an annual report, policy document, meeting notes, or any long-form text.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    init_session_state()
    mode, web_enabled = render_sidebar()

    # Header
    st.markdown(
        """
<div class='research-header'>
    <h1>Research Assistant</h1>
    <div class='subtitle'>
        Grounded answers with hybrid retrieval, parallel research agents,
        and citation verification
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Guard: API key must be present server-side
    if not GROQ_API_KEY and not os.environ.get("GROQ_API_KEY", ""):
        st.error(
            "The server is missing a Groq API key. "
            "Contact the administrator or set GROQ_API_KEY in the environment."
        )
        return

    # Render conversation history
    has_history = bool(st.session_state.messages)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations") and msg.get("ledger"):
                render_citations(msg["citations"], msg["ledger"])
            if msg.get("trace"):
                with st.expander("Execution trace", expanded=False):
                    for line in msg["trace"]:
                        st.text(line)

    # Empty state hint (shown only before first question)
    if not has_history and not st.session_state.ingested_files:
        render_empty_state()

    # Question limit check
    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.info(
            "Session question limit reached. "
            "Click 'Clear All Documents' in the sidebar to start a new session."
        )
        return

    # Chat input
    placeholder = (
        "Ask a question about your documents..."
        if st.session_state.ingested_files
        else "Upload and index documents first, then ask a question..."
    )
    user_query = st.chat_input(placeholder)

    if not user_query:
        return

    if not st.session_state.ingested_files:
        st.warning("Please upload and index at least one document before asking a question.")
        return

    # Append user message
    st.session_state.question_count += 1
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run research pipeline
    with st.chat_message("assistant"):
        status_container = st.status("Starting research pipeline...", expanded=True)
        try:
            graph = get_graph_mode(mode, kb=st.session_state.kb, web_enabled=web_enabled)
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

            accumulated_trace: list[str] = []
            final_state: dict[str, Any] = {}

            for update in graph.stream(initial_state, stream_mode="updates"):
                for _node, node_output in update.items():
                    if isinstance(node_output, dict):
                        for t in node_output.get("trace", []):
                            accumulated_trace.append(t)
                            status_container.write(f"`{t}`")
                        final_state.update(node_output)

            answer = final_state.get("final_answer") or final_state.get("draft") or "No answer generated."
            ledger = final_state.get("ledger", {})
            valid_ids, _invalid = check_citations(answer, ledger)

            status_container.update(label="Research complete", state="complete", expanded=False)

            st.markdown(answer)
            render_citations(valid_ids, ledger)

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
            status_container.update(label="Error", state="error", expanded=True)
            st.error(f"Research pipeline failed: {exc}")


if __name__ == "__main__":
    main()

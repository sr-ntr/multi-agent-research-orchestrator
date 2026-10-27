"""
LangGraph node functions: planner, researcher, writer, verifier, finalize.

Each node is a plain Python function that receives and returns a dict
compatible with AgentState. Nodes must not share mutable state outside
the AgentState dict.

The researcher is a hand-written tool loop (no prebuilt ReAct helper) for
full control over budgets, evidence capture, and error handling.
"""

import json
import logging
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.prompts import (
    PLANNER_SYSTEM,
    PLANNER_USER,
    RESEARCHER_BUDGET_EXHAUSTED,
    RESEARCHER_SYSTEM,
    RESEARCHER_USER,
    VERIFIER_SYSTEM,
    VERIFIER_USER,
    WRITER_REVISION_NOTE,
    WRITER_SYSTEM,
    WRITER_USER,
    format_evidence_block,
)
from src.agents.state import (
    AgentState,
    Claim,
    Plan,
    ResearchFinding,
    ResearchResult,
    SubQuestion,
    Verdict,
)
from src.config import (
    ABSTAIN_TEXT,
    MAX_REVISIONS,
    MAX_TOOL_CALLS_PER_RESEARCHER,
)
from src.ledger import Evidence, build_ledger, check_citations, extract_evidence, merge_ledger
from src.llm import get_llm, invoke_with_retry, text_of

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------


def planner_node(state: AgentState, *, kb=None, web_enabled: bool = False) -> dict:
    """Decompose the user question into sub-questions.

    Returns one sub-question for simple lookups; up to MAX_SUBQUESTIONS for
    complex questions. If web is disabled globally, forces scope to "documents".

    Args:
        state: Current AgentState.
        kb: KnowledgeBase (unused here, kept for uniform signature).
        web_enabled: Whether web search is globally enabled.

    Returns:
        Partial AgentState update with plan and trace.
    """
    question = state["question"]
    web_status = "enabled" if web_enabled else "disabled"

    llm = get_llm("fast")
    structured = llm.with_structured_output(Plan)

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=PLANNER_USER.format(question=question, web_status=web_status)),
    ]

    try:
        plan: Plan = structured.invoke(messages)
    except Exception as exc:
        logger.warning("Planner failed, falling back to single sub-question: %s", exc)
        plan = Plan(
            complexity="simple",
            subquestions=[SubQuestion(id="sq1", question=question, scope="documents")],
        )

    # If web is disabled, override any web scopes
    if not web_enabled:
        for sq in plan.subquestions:
            if sq.scope in ("web", "both"):
                sq.scope = "documents"

    trace_line = (
        f"[planner] complexity={plan.complexity} "
        f"subquestions={[sq.question[:60] for sq in plan.subquestions]}"
    )
    logger.info(trace_line)

    return {"plan": plan, "trace": [trace_line]}


# ---------------------------------------------------------------------------
# Researcher node (hand-written tool loop)
# ---------------------------------------------------------------------------


def researcher_node(
    state: AgentState,
    *,
    subquestion: SubQuestion,
    kb,
    web_enabled: bool = False,
) -> dict:
    """Research one sub-question using a bounded tool loop.

    Calls tools up to MAX_TOOL_CALLS_PER_RESEARCHER times. When the budget is
    exhausted, makes one final call (without tools) asking for a summary.
    Ends with a structured ResearchResult call.

    Every tool output is parsed into Evidence and added to the returned ledger.

    Args:
        state: Current AgentState.
        subquestion: The SubQuestion this researcher is assigned.
        kb: KnowledgeBase instance.
        web_enabled: Global web-search toggle.

    Returns:
        Partial AgentState update with findings, ledger, trace, tool_call_log.
    """
    from src.tools import make_tools  # late import to avoid circular

    # Build tool set based on sub-question scope
    scope = subquestion.scope
    sq_web = web_enabled and scope in ("web", "both")
    tools = make_tools(kb, web_enabled=sq_web)
    tool_map = {t.name: t for t in tools}

    llm_fast = get_llm("fast")
    llm_with_tools = llm_fast.bind_tools(tools)

    system_msg = SystemMessage(
        content=RESEARCHER_SYSTEM.format(max_tool_calls=MAX_TOOL_CALLS_PER_RESEARCHER)
    )
    user_msg = HumanMessage(
        content=RESEARCHER_USER.format(question=subquestion.question, scope=scope)
    )
    messages: list = [system_msg, user_msg]

    tool_calls_made = 0
    new_evidence: list[Evidence] = []
    tool_call_log: list[dict] = []

    while tool_calls_made < MAX_TOOL_CALLS_PER_RESEARCHER:
        response = invoke_with_retry(llm_with_tools, messages)
        messages.append(response)  # keep AIMessage unmodified

        if not response.tool_calls:
            # Model decided to stop calling tools
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", f"call_{tool_calls_made}")

            logger.debug(
                "[researcher:%s] tool=%s args=%s", subquestion.id, tool_name, tool_args
            )
            tool_call_log.append(
                {
                    "subquestion_id": subquestion.id,
                    "tool": tool_name,
                    "args": tool_args,
                }
            )

            if tool_name in tool_map:
                try:
                    tool_input = tool_args if tool_args else (tc.get("args") if isinstance(tc, dict) and "args" in tc else tc)
                    raw_result = tool_map[tool_name].invoke(tool_input)
                    # Extract evidence from the tool output
                    evs = extract_evidence(raw_result)
                    new_evidence.extend(evs)
                    tool_result_content = str(raw_result)
                except Exception as exc:
                    logger.warning("Tool %s failed: %s", tool_name, exc)
                    tool_result_content = json.dumps({"error": str(exc)})
            else:
                tool_result_content = json.dumps({"error": f"Unknown tool: {tool_name}"})

            messages.append(
                ToolMessage(content=tool_result_content, tool_call_id=tool_id)
            )
            tool_calls_made += 1

    # If budget exhausted, ask for a plain summary first
    if tool_calls_made >= MAX_TOOL_CALLS_PER_RESEARCHER and (
        not messages or getattr(messages[-1], "tool_calls", None)
    ):
        llm_no_tools = get_llm("fast")
        messages.append(HumanMessage(content=RESEARCHER_BUDGET_EXHAUSTED))
        budget_response = invoke_with_retry(llm_no_tools, messages)
        messages.append(budget_response)

    # Final structured call: get ResearchResult
    llm_structured = get_llm("fast").with_structured_output(ResearchResult)
    # Build a clean context for the structured call
    structured_prompt = (
        f"Based on your research above, return a ResearchResult for sub-question id "
        f"'{subquestion.id}'. Include only evidence IDs from the searches you performed."
    )
    structured_messages = messages + [HumanMessage(content=structured_prompt)]

    try:
        result: ResearchResult = llm_structured.invoke(structured_messages)
    except Exception as exc:
        logger.warning("ResearchResult structured call failed: %s", exc)
        result = ResearchResult(
            subquestion_id=subquestion.id,
            summary=text_of(messages[-1]) if messages else "Research failed.",
            relevant_ids=[ev.id for ev in new_evidence[:3]],
            found=bool(new_evidence),
        )

    # Validate: drop relevant_ids not in the newly captured ledger
    local_ledger = build_ledger(new_evidence)
    original_count = len(result.relevant_ids)
    result.relevant_ids = [rid for rid in result.relevant_ids if rid in local_ledger]
    dropped = original_count - len(result.relevant_ids)
    if dropped > 0:
        logger.info(
            "[researcher:%s] dropped %d relevant_ids not in ledger",
            subquestion.id,
            dropped,
        )

    finding = ResearchFinding(
        subquestion_id=subquestion.id,
        question=subquestion.question,
        summary=result.summary,
        relevant_ids=result.relevant_ids,
        found=result.found,
    )

    trace_line = (
        f"[researcher:{subquestion.id}] tool_calls={tool_calls_made} "
        f"found={result.found} evidence_items={len(new_evidence)} "
        f"relevant_ids={result.relevant_ids}"
    )
    logger.info(trace_line)

    return {
        "findings": [finding],
        "ledger": local_ledger,
        "trace": [trace_line],
        "tool_call_log": tool_call_log,
    }


# ---------------------------------------------------------------------------
# Writer node
# ---------------------------------------------------------------------------


def writer_node(state: AgentState) -> dict:
    """Compose a cited markdown answer from the researchers' findings.

    Uses only the evidence entries listed in the findings' relevant_ids.
    Every factual sentence must end with a citation marker [id].

    Args:
        state: Current AgentState (must have findings and ledger).

    Returns:
        Partial AgentState update with draft and trace.
    """
    question = state["question"]
    findings = state.get("findings", [])
    ledger = state.get("ledger", {})
    unsupported = state.get("unsupported_claims", [])
    revision_count = state.get("revision_count", 0)

    # Collect all relevant evidence IDs across findings
    all_relevant_ids: list[str] = []
    for finding in findings:
        all_relevant_ids.extend(finding.relevant_ids)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_ids = [rid for rid in all_relevant_ids if not (rid in seen or seen.add(rid))]  # type: ignore

    # Fetch evidence objects for the writer (only what it's allowed to cite)
    writer_evidence = [ledger[rid] for rid in unique_ids if rid in ledger]

    # Build findings summary for context
    findings_summary = "\n".join(
        f"- Sub-question: {f.question}\n  Summary: {f.summary}"
        for f in findings
    )

    evidence_block = format_evidence_block(writer_evidence)
    revision_note = (
        WRITER_REVISION_NOTE.format(
            unsupported_claims="\n".join(f"- {c}" for c in unsupported)
        )
        if unsupported
        else ""
    )

    llm = get_llm("strong")
    messages = [
        SystemMessage(content=WRITER_SYSTEM.format(abstain_text=ABSTAIN_TEXT)),
        HumanMessage(
            content=WRITER_USER.format(
                question=question,
                evidence_block=evidence_block,
                revision_note=revision_note,
            )
        ),
    ]

    response = invoke_with_retry(llm, messages)
    draft = text_of(response).strip()

    trace_line = (
        f"[writer] revision={revision_count} "
        f"evidence_items={len(writer_evidence)} "
        f"draft_len={len(draft)}"
    )
    logger.info(trace_line)

    return {
        "draft": draft,
        "trace": [trace_line],
        "revision_count": revision_count + 1,
    }


# ---------------------------------------------------------------------------
# Verifier node
# ---------------------------------------------------------------------------


def verifier_node(state: AgentState) -> dict:
    """Verify the writer's draft against cited evidence.

    First runs a deterministic citation check (free, no LLM). Then calls the
    strong model for a structured Verdict over atomic claims.

    Args:
        state: Current AgentState (must have draft and ledger).

    Returns:
        Partial AgentState update with unsupported_claims and trace.
    """
    draft = state.get("draft", "")
    ledger = state.get("ledger", {})
    findings = state.get("findings", [])

    # Collect IDs the writer was allowed to see
    allowed_ids: set[str] = set()
    for f in findings:
        allowed_ids.update(f.relevant_ids)

    # --- Deterministic check ---
    valid_ids, invalid_ids = check_citations(draft, ledger)
    det_unsupported: list[str] = []
    for cid in invalid_ids:
        det_unsupported.append(f"Citation [{cid}] does not exist in the evidence ledger.")

    for cid in valid_ids:
        if cid not in allowed_ids:
            det_unsupported.append(
                f"Citation [{cid}] is in the ledger but was not shown to the writer."
            )

    # --- LLM-based check ---
    cited_evidence = [ledger[cid] for cid in valid_ids if cid in ledger and cid in allowed_ids]
    evidence_block = format_evidence_block(cited_evidence)

    llm = get_llm("strong")
    structured = llm.with_structured_output(Verdict)
    messages = [
        SystemMessage(content=VERIFIER_SYSTEM),
        HumanMessage(
            content=VERIFIER_USER.format(draft=draft, evidence_block=evidence_block)
        ),
    ]

    try:
        verdict: Verdict = structured.invoke(messages)
    except Exception as exc:
        logger.warning("Verifier structured call failed: %s", exc)
        verdict = Verdict(claims=[], has_unsupported=bool(det_unsupported))

    llm_unsupported = [
        f"{c.text} (reason: {c.reason})"
        for c in verdict.claims
        if not c.supported
    ]

    all_unsupported = det_unsupported + llm_unsupported

    trace_line = (
        f"[verifier] deterministic_issues={len(det_unsupported)} "
        f"llm_unsupported={len(llm_unsupported)} "
        f"total_claims={len(verdict.claims)}"
    )
    logger.info(trace_line)

    return {
        "unsupported_claims": all_unsupported,
        "trace": [trace_line],
    }


# ---------------------------------------------------------------------------
# Finalize node
# ---------------------------------------------------------------------------


def finalize_node(state: AgentState) -> dict:
    """Produce the final answer with optional unverified-statements note.

    Rules:
    - If no finding has found=True: set final_answer to ABSTAIN_TEXT.
    - If there are remaining unsupported claims after max revisions: append
      a short "Unverified statements" note to the draft.
    - Otherwise: final_answer = draft as-is.

    Args:
        state: Current AgentState.

    Returns:
        Partial AgentState update with final_answer and trace.
    """
    findings = state.get("findings", [])
    draft = state.get("draft", "")
    unsupported = state.get("unsupported_claims", [])

    any_found = any(f.found for f in findings)

    if not any_found:
        final = ABSTAIN_TEXT
        trace_line = "[finalize] abstaining: no findings with found=True"
    elif unsupported:
        note = (
            "\n\n---\n**Unverified statements** (could not be confirmed against the "
            "provided evidence):\n"
            + "\n".join(f"- {u}" for u in unsupported)
        )
        final = draft + note
        trace_line = f"[finalize] appended unverified note ({len(unsupported)} items)"
    else:
        final = draft
        trace_line = "[finalize] answer accepted as-is"

    logger.info(trace_line)
    return {"final_answer": final, "trace": [trace_line]}

"""
LangGraph graph builder for the Agentic Document Research Assistant.

build_graph(use_planner, use_verifier, kb, web_enabled) is the main entry point.
Returns a compiled LangGraph graph.

Three named configurations:
    single          -- no planner, no verifier (one researcher, fast)
    multi_noverify  -- planner + parallel researchers, no verifier
    multi           -- planner + parallel researchers + verifier

Branching strategy:
    - After planner: fan-out via Send() -- one researcher per sub-question.
    - After all researchers join: writer.
    - After writer (if use_verifier): verifier.
    - After verifier: route_after_verify decides revise vs finalize.
    - Finalize produces the final answer.
"""

import logging
from functools import partial
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send

from src.agents.state import AgentState, Plan, SubQuestion
from src.agents.nodes import (
    finalize_node,
    planner_node,
    researcher_node,
    verifier_node,
    writer_node,
)
from src.config import MAX_REVISIONS, MAX_SUBQUESTIONS

logger = logging.getLogger(__name__)

GraphMode = Literal["single", "multi_noverify", "multi"]


# ---------------------------------------------------------------------------
# Fan-out edge
# ---------------------------------------------------------------------------


def fan_out(state: AgentState) -> list:
    """Return a Send() for each sub-question in the plan.

    Each Send targets the "researcher" node with the sub-question injected
    into the state payload. The researcher node extracts it from the payload.

    Args:
        state: Current AgentState (must have a plan).

    Returns:
        List of Send objects, one per sub-question.
    """
    plan: Plan = state.get("plan")
    if plan is None:
        # Fallback: single sub-question equal to the original question
        question = state.get("question", "")
        sq = SubQuestion(id="sq1", question=question, scope="documents")
        plan = Plan(complexity="simple", subquestions=[sq])

    sends = []
    for sq in plan.subquestions[: MAX_SUBQUESTIONS]:
        sends.append(
            Send(
                "researcher",
                {**state, "_subquestion": sq.model_dump()},
            )
        )
    return sends


# ---------------------------------------------------------------------------
# Routing after verifier
# ---------------------------------------------------------------------------


def route_after_verify(state: AgentState) -> Literal["writer", "finalize"]:
    """Decide whether to revise or finalize after the verifier runs.

    Args:
        state: Current AgentState.

    Returns:
        "writer" if there are unsupported claims and revisions remain;
        "finalize" otherwise.
    """
    unsupported = state.get("unsupported_claims", [])
    revision_count = state.get("revision_count", 0)

    if unsupported and revision_count <= MAX_REVISIONS:
        logger.info(
            "[router] routing to writer for revision %d (unsupported=%d)",
            revision_count,
            len(unsupported),
        )
        return "writer"

    return "finalize"


# ---------------------------------------------------------------------------
# Planner bypass (for single mode)
# ---------------------------------------------------------------------------


def trivial_planner(state: AgentState) -> dict:
    """Create a trivial single-sub-question plan (no LLM call).

    Used in 'single' mode where the user question becomes the only sub-question.

    Args:
        state: Current AgentState.

    Returns:
        State update with a single-subquestion Plan.
    """
    question = state.get("question", "")
    plan = Plan(
        complexity="simple",
        subquestions=[
            SubQuestion(id="sq1", question=question, scope="documents")
        ],
    )
    return {"plan": plan, "trace": ["[planner] trivial single-question plan (no LLM)"]}


# ---------------------------------------------------------------------------
# Researcher wrapper (pulls sub-question from state payload)
# ---------------------------------------------------------------------------


def _researcher_wrapper(state: AgentState, *, kb, web_enabled: bool) -> dict:
    """Extract _subquestion from state and call researcher_node.

    The fan-out Send() injects _subquestion as a serialised dict.
    """
    sq_dict = state.get("_subquestion")
    if sq_dict is None:
        # Fallback if no sub-question injected (single mode without fan-out)
        question = state.get("question", "")
        sq = SubQuestion(id="sq1", question=question, scope="documents")
    else:
        sq = SubQuestion(**sq_dict)

    return researcher_node(state, subquestion=sq, kb=kb, web_enabled=web_enabled)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    use_planner: bool = True,
    use_verifier: bool = True,
    kb=None,
    web_enabled: bool = False,
):
    """Build and compile a LangGraph graph for the configured mode.

    Args:
        use_planner: If True, use the LLM planner; otherwise use a trivial
            single-sub-question plan.
        use_verifier: If True, include the verifier and revision loop.
        kb: KnowledgeBase instance (required).
        web_enabled: Whether web search is globally enabled.

    Returns:
        A compiled LangGraph graph ready for .invoke() or .stream().

    Raises:
        ValueError: If kb is None.
    """
    if kb is None:
        raise ValueError("build_graph requires a KnowledgeBase instance (kb=...)")

    # Bind runtime dependencies into node callables
    researcher = partial(_researcher_wrapper, kb=kb, web_enabled=web_enabled)
    planner = (
        partial(planner_node, kb=kb, web_enabled=web_enabled)
        if use_planner
        else trivial_planner
    )

    # ---------------------------------------------------------------------------
    # Graph construction
    # ---------------------------------------------------------------------------

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer_node)
    builder.add_node("finalize", finalize_node)

    if use_verifier:
        builder.add_node("verifier", verifier_node)

    # Edges: START -> planner -> fan-out -> researcher(s) -> writer
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", fan_out, ["researcher"])
    builder.add_edge("researcher", "writer")

    if use_verifier:
        builder.add_edge("writer", "verifier")
        builder.add_conditional_edges(
            "verifier",
            route_after_verify,
            {"writer": "writer", "finalize": "finalize"},
        )
    else:
        builder.add_edge("writer", "finalize")

    builder.add_edge("finalize", END)

    return builder.compile()


def get_graph_mode(mode: GraphMode, **kwargs):
    """Convenience wrapper returning a compiled graph for a named mode.

    Args:
        mode: "single", "multi_noverify", or "multi".
        **kwargs: Forwarded to build_graph (must include kb=).

    Returns:
        Compiled graph.
    """
    if mode == "single":
        return build_graph(use_planner=False, use_verifier=False, **kwargs)
    elif mode == "multi_noverify":
        return build_graph(use_planner=True, use_verifier=False, **kwargs)
    elif mode == "multi":
        return build_graph(use_planner=True, use_verifier=True, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'single', 'multi_noverify', or 'multi'.")

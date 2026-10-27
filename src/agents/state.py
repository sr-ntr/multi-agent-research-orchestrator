"""
Pydantic schemas and LangGraph AgentState TypedDict for the research agent.

State design:
    - findings: list[ResearchFinding] -- accumulated per-subquestion results
    - ledger: dict[str, Evidence]     -- shared evidence store (all retrieved items)
    - trace: list[str]                -- human-readable trace lines
    - Other fields are non-list scalars (last-write-wins).

Reducers:
    findings  -> list concatenation (parallel researchers append independently)
    ledger    -> dict merge, no overwrite (merge_ledger)
    trace     -> list concatenation

Pydantic models (also used for structured LLM output):
    SubQuestion     -- one research task with id, question, scope
    Plan            -- planner output: complexity + list of SubQuestions
    ResearchResult  -- researcher output: summary, relevant_ids, found flag
    Claim           -- one atomic claim with a supported flag and reason
    Verdict         -- verifier output: list of Claims
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional

from langgraph.graph import add_messages  # type: ignore
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.ledger import Evidence


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """A single research sub-question produced by the planner.

    Attributes:
        id:       Short identifier, e.g. "sq1", "sq2".
        question: The self-contained sub-question string.
        scope:    Which tools the researcher for this sub-question may use.
    """

    id: str
    question: str
    scope: Literal["documents", "web", "both"] = "documents"


class Plan(BaseModel):
    """The planner's decomposition of the user question.

    Attributes:
        complexity: Whether the question needs one lookup or multiple sub-questions.
        subquestions: 1 to MAX_SUBQUESTIONS SubQuestion objects.
    """

    complexity: Literal["simple", "complex"]
    subquestions: list[SubQuestion] = Field(min_length=1)


class ResearchResult(BaseModel):
    """The researcher's findings for one sub-question.

    Attributes:
        subquestion_id: The id of the SubQuestion this researcher handled.
        summary:        Plain-language summary of what was found.
        relevant_ids:   Evidence IDs the researcher considers most relevant.
                        IDs not present in the ledger are dropped by the node.
        found:          True if the researcher found information answering the
                        sub-question; False if the sources did not contain it.
    """

    subquestion_id: str
    summary: str
    relevant_ids: list[str] = Field(default_factory=list)
    found: bool


class Claim(BaseModel):
    """One atomic factual claim extracted from the writer's draft.

    Attributes:
        text:      The claim as a short, self-contained sentence.
        citation:  The evidence ID the writer cited for this claim (if any).
        supported: True if the cited evidence supports the claim.
        reason:    One-line explanation from the verifier.
    """

    text: str
    citation: Optional[str] = None
    supported: bool
    reason: str = ""


class Verdict(BaseModel):
    """The verifier's structured output.

    Attributes:
        claims: List of atomic Claim objects covering the full draft.
        has_unsupported: True if any claim is not supported (convenience flag).
    """

    claims: list[Claim]
    has_unsupported: bool = False


class ResearchFinding(BaseModel):
    """Wrapper stored in AgentState.findings for one sub-question's research.

    Attributes:
        subquestion_id: Links back to a SubQuestion.
        question:       The sub-question text (for the writer's context).
        summary:        Researcher's plain-language summary.
        relevant_ids:   Validated evidence IDs (subset of the ledger).
        found:          Whether useful information was found.
    """

    subquestion_id: str
    question: str
    summary: str
    relevant_ids: list[str]
    found: bool


# ---------------------------------------------------------------------------
# Reducer helpers
# ---------------------------------------------------------------------------


def _merge_ledger(
    base: dict[str, Evidence], incoming: dict[str, Evidence]
) -> dict[str, Evidence]:
    """LangGraph reducer: merge ledger dicts without overwriting existing entries."""
    from src.ledger import merge_ledger
    return merge_ledger(base, incoming)


def _concat_list(a: list, b: list) -> list:
    """LangGraph reducer: concatenate two lists."""
    return a + b


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """The shared state object passed through every LangGraph node.

    Fields with Annotated reducers are safe for parallel branches to write
    simultaneously; LangGraph applies the reducer when merging branch outputs.
    """

    # Input
    question: str
    web_enabled: bool

    # Planner output
    plan: Optional[Plan]

    # Researcher outputs (one per sub-question, merged by concatenation)
    findings: Annotated[list[ResearchFinding], _concat_list]

    # Shared evidence ledger (merged by dict merge, no overwrite)
    ledger: Annotated[dict[str, Evidence], _merge_ledger]

    # Writer / verifier
    draft: str
    revision_count: int
    unsupported_claims: list[str]

    # Final answer
    final_answer: str

    # Observability
    trace: Annotated[list[str], _concat_list]
    tool_call_log: Annotated[list[dict], _concat_list]
    token_usage: dict[str, int]

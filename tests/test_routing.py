"""
Tests for graph routing logic: fan_out, scope-to-tools, route_after_verify,
abstention short-circuit, and ledger reducer correctness.

All tests are offline (no API calls). The graph nodes are tested in isolation
using mocks and small synthetic state dicts.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.state import (
    AgentState,
    Plan,
    ResearchFinding,
    SubQuestion,
)
from src.agents.graph import fan_out, route_after_verify, trivial_planner
from src.ledger import Evidence, merge_ledger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Create a minimal AgentState-compatible dict."""
    base = {
        "question": "What is the revenue?",
        "web_enabled": False,
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
    base.update(overrides)
    return base


def _make_plan(*questions, scopes=None) -> Plan:
    """Build a Plan with the given sub-questions."""
    scopes = scopes or ["documents"] * len(questions)
    subqs = [
        SubQuestion(id=f"sq{i+1}", question=q, scope=s)
        for i, (q, s) in enumerate(zip(questions, scopes))
    ]
    complexity = "simple" if len(subqs) == 1 else "complex"
    return Plan(complexity=complexity, subquestions=subqs)


# ---------------------------------------------------------------------------
# fan_out
# ---------------------------------------------------------------------------


def test_fan_out_single_subquestion():
    """fan_out with one sub-question returns exactly one Send."""
    plan = _make_plan("What is the revenue?")
    state = _make_state(plan=plan)
    sends = fan_out(state)
    assert len(sends) == 1


def test_fan_out_multiple_subquestions():
    """fan_out with three sub-questions returns three Sends."""
    plan = _make_plan("Revenue?", "Expenses?", "Profit?")
    state = _make_state(plan=plan)
    sends = fan_out(state)
    assert len(sends) == 3


def test_fan_out_caps_at_max_subquestions():
    """fan_out caps the number of Sends at MAX_SUBQUESTIONS (4)."""
    from src.config import MAX_SUBQUESTIONS
    # Create 6 sub-questions (more than the max)
    plan = _make_plan(*[f"Q{i}?" for i in range(6)])
    state = _make_state(plan=plan)
    sends = fan_out(state)
    assert len(sends) == MAX_SUBQUESTIONS


def test_fan_out_with_null_plan_falls_back():
    """fan_out with plan=None produces one Send using the original question."""
    state = _make_state(plan=None, question="What is the revenue?")
    sends = fan_out(state)
    assert len(sends) == 1


def test_fan_out_injects_subquestion_into_payload():
    """fan_out injects _subquestion into each Send's payload."""
    plan = _make_plan("What is the revenue?", "What are the expenses?")
    state = _make_state(plan=plan)
    sends = fan_out(state)
    for send in sends:
        # Send object has .node and .arg attributes
        assert hasattr(send, "node")
        assert send.node == "researcher"
        assert "_subquestion" in send.arg
        assert "question" in send.arg["_subquestion"]


# ---------------------------------------------------------------------------
# scope-to-tools mapping
# ---------------------------------------------------------------------------


def test_scope_documents_excludes_web_tool():
    """When scope='documents', the web_search tool is not included."""
    from src.tools import make_tools
    kb = MagicMock()
    kb.search.return_value = []
    kb.list_documents.return_value = []

    tools = make_tools(kb, web_enabled=False)
    names = {t.name for t in tools}
    assert "web_search" not in names


def test_scope_web_includes_web_tool():
    """When web_enabled=True, the web_search tool is present."""
    from src.tools import make_tools
    from src.web_search import FakeProvider
    kb = MagicMock()
    kb.search.return_value = []
    kb.list_documents.return_value = []

    tools = make_tools(kb, web_enabled=True, web_provider=FakeProvider())
    names = {t.name for t in tools}
    assert "web_search" in names


# ---------------------------------------------------------------------------
# route_after_verify
# ---------------------------------------------------------------------------


def test_route_after_verify_no_issues_goes_to_finalize():
    """route_after_verify sends to 'finalize' when there are no unsupported claims."""
    state = _make_state(unsupported_claims=[], revision_count=0)
    assert route_after_verify(state) == "finalize"


def test_route_after_verify_first_revision_goes_to_writer():
    """route_after_verify sends to 'writer' on first revision with unsupported claims."""
    state = _make_state(
        unsupported_claims=["Claim X is not in the evidence."],
        revision_count=0,
    )
    assert route_after_verify(state) == "writer"


def test_route_after_verify_max_revisions_goes_to_finalize():
    """After MAX_REVISIONS, route_after_verify sends to 'finalize' regardless."""
    from src.config import MAX_REVISIONS
    state = _make_state(
        unsupported_claims=["Still unsupported."],
        revision_count=MAX_REVISIONS + 1,
    )
    assert route_after_verify(state) == "finalize"


def test_route_after_verify_empty_unsupported_after_revision():
    """If unsupported becomes empty after a revision, route to finalize."""
    state = _make_state(unsupported_claims=[], revision_count=1)
    assert route_after_verify(state) == "finalize"


# ---------------------------------------------------------------------------
# Abstention short-circuit
# ---------------------------------------------------------------------------


def test_finalize_abstains_when_nothing_found():
    """finalize_node sets final_answer to ABSTAIN_TEXT when no finding has found=True."""
    from src.agents.nodes import finalize_node
    from src.config import ABSTAIN_TEXT

    findings = [
        ResearchFinding(
            subquestion_id="sq1",
            question="What is the revenue?",
            summary="Not found.",
            relevant_ids=[],
            found=False,
        )
    ]
    state = _make_state(findings=findings, draft="", unsupported_claims=[])
    result = finalize_node(state)
    assert result["final_answer"] == ABSTAIN_TEXT


def test_finalize_does_not_abstain_when_something_found():
    """finalize_node uses the draft when at least one finding has found=True."""
    from src.agents.nodes import finalize_node

    findings = [
        ResearchFinding(
            subquestion_id="sq1",
            question="Revenue?",
            summary="Revenue was 450M.",
            relevant_ids=["report:0001"],
            found=True,
        )
    ]
    draft = "Revenue was 450M [report:0001]."
    state = _make_state(findings=findings, draft=draft, unsupported_claims=[])
    result = finalize_node(state)
    assert result["final_answer"] == draft


# ---------------------------------------------------------------------------
# Ledger reducer: merge without overwriting
# ---------------------------------------------------------------------------


def test_ledger_reducer_adds_new_entries():
    """merge_ledger adds incoming entries not in base."""
    ev1 = Evidence(id="a:0001", kind="doc", source="a.md", text="text a")
    ev2 = Evidence(id="b:0001", kind="doc", source="b.md", text="text b")
    base = {"a:0001": ev1}
    result = merge_ledger(base, {"b:0001": ev2})
    assert "b:0001" in result


def test_ledger_reducer_does_not_overwrite():
    """merge_ledger never replaces an existing entry."""
    ev_orig = Evidence(id="a:0001", kind="doc", source="a.md", text="original")
    ev_new = Evidence(id="a:0001", kind="doc", source="a.md", text="replacement")
    base = {"a:0001": ev_orig}
    merge_ledger(base, {"a:0001": ev_new})
    assert base["a:0001"].text == "original"


def test_findings_concat_from_parallel_branches():
    """Findings from parallel researchers are concatenated, not overwritten."""
    from src.agents.state import _concat_list

    f1 = ResearchFinding(
        subquestion_id="sq1", question="Q1", summary="S1", relevant_ids=[], found=True
    )
    f2 = ResearchFinding(
        subquestion_id="sq2", question="Q2", summary="S2", relevant_ids=[], found=True
    )
    merged = _concat_list([f1], [f2])
    assert len(merged) == 2
    assert merged[0].subquestion_id == "sq1"
    assert merged[1].subquestion_id == "sq2"


# ---------------------------------------------------------------------------
# trivial_planner
# ---------------------------------------------------------------------------


def test_trivial_planner_returns_single_subquestion():
    """trivial_planner creates exactly one sub-question equal to the question."""
    state = _make_state(question="What is the main product?")
    result = trivial_planner(state)
    plan = result["plan"]
    assert len(plan.subquestions) == 1
    assert plan.subquestions[0].question == "What is the main product?"
    assert plan.complexity == "simple"

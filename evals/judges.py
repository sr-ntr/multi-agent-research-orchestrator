"""
LLM judge prompts and parsers for evaluation.

Exports:
    judge_faithfulness(answer, cited_evidence) -> FaithfulnessVerdict
    judge_correctness(question, answer, reference) -> CorrectnessVerdict
    judge_abstention_fallback(question, answer) -> AbstentionVerdict

Uses MODEL_JUDGE with structured Pydantic output and strict evaluation rubrics.
Notice: The faithfulness prompt is deliberately distinct from the pipeline verifier prompt
to avoid circularity and shared prompt bias.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from src.llm import get_llm, invoke_structured, invoke_with_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured Judge Output
# ---------------------------------------------------------------------------


class FaithfulnessClaim(BaseModel):
    """An atomic factual claim extracted from an answer during evaluation."""
    claim: str
    cited_id: Optional[str] = None
    supported: bool
    reason: str


class FaithfulnessVerdict(BaseModel):
    """Result of LLM faithfulness evaluation."""
    claims: list[FaithfulnessClaim] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class CorrectnessVerdict(BaseModel):
    """Result of LLM correctness evaluation against a human reference answer."""
    score: float = Field(ge=0.0, le=1.0)  # Allowed: 0.0, 0.5, 1.0
    reason: str


class AbstentionVerdict(BaseModel):
    """Result of LLM judge determining if an answer honestly abstained."""
    is_abstention: bool
    reason: str


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FAITHFULNESS_JUDGE_SYSTEM = """\
You are an impartial evaluation judge assessing the faithfulness of a generated answer.
Faithfulness measures whether every claim in the answer is grounded in and directly
supported by the cited evidence passages provided to the model.

Rubric:
1. Break down the provided answer into independent, atomic factual statements.
2. For each statement, determine whether it is cited and whether the corresponding
   evidence passage directly entails it.
   - If an atomic statement adds details, names, numbers, or conclusions not in
     the cited passage, it is NOT supported (supported=false).
   - If the statement is completely supported by the cited passage, supported=true.
3. Compute the score as: (number of supported claims) / (total number of claims).
   If there are zero factual claims (e.g., explicit abstention), score is 1.0.
4. Output the structured FaithfulnessVerdict.
"""

FAITHFULNESS_JUDGE_USER = """\
Answer to evaluate:
{answer}

Cited Evidence Passages:
{evidence_block}

Evaluate faithfulness now.
"""

CORRECTNESS_JUDGE_SYSTEM = """\
You are an impartial evaluation judge assessing answer correctness against a verified
human reference answer.

Rubric:
- 1.0 (Correct): The generated answer correctly answers the question and contains all key
  facts and numbers present in the reference answer, without stating contradictions.
- 0.5 (Partially Correct): The answer provides some correct information from the reference,
  but misses key figures/facts or includes minor inaccuracies.
- 0.0 (Incorrect): The answer contradicts the reference, provides wrong facts or numbers,
  or fails to answer the question.

Output the score (must be 0.0, 0.5, or 1.0) and a concise one-line reason.
"""

CORRECTNESS_JUDGE_USER = """\
Question: {question}

Human Reference Answer:
{reference_answer}

Generated Answer:
{generated_answer}

Evaluate correctness now.
"""

ABSTENTION_JUDGE_SYSTEM = """\
You are an evaluation judge assessing whether an answer correctly refused or abstained
from answering due to lack of evidence in the provided documents.

Return is_abstention=true if the answer clearly conveys that the documents do not
contain enough information or that the answer is not available in the corpus.
Return is_abstention=false if the answer attempts to answer the question.
"""

ABSTENTION_JUDGE_USER = """\
Question: {question}

Generated Answer:
{answer}

Evaluate whether this is an abstention.
"""


# ---------------------------------------------------------------------------
# Judge Functions
# ---------------------------------------------------------------------------


def judge_faithfulness(answer: str, cited_evidence_text: str) -> FaithfulnessVerdict:
    """Evaluate faithfulness of an answer given its cited evidence passages.

    Args:
        answer: The generated answer string.
        cited_evidence_text: Formatted string of cited evidence passages.

    Returns:
        FaithfulnessVerdict containing atomic claims and fractional score.
    """
    if not answer.strip():
        return FaithfulnessVerdict(claims=[], score=1.0, reason="Empty answer")

    llm = get_llm("judge")
    messages = [
        SystemMessage(content=FAITHFULNESS_JUDGE_SYSTEM),
        HumanMessage(
            content=FAITHFULNESS_JUDGE_USER.format(
                answer=answer, evidence_block=cited_evidence_text or "(No evidence cited)"
            )
        ),
    ]

    try:
        verdict = invoke_structured(llm, FaithfulnessVerdict, messages)
        # Ensure mathematical consistency if claims exist
        if verdict.claims:
            supported = sum(1 for c in verdict.claims if c.supported)
            verdict.score = round(supported / len(verdict.claims), 4)
        return verdict
    except Exception as exc:
        logger.warning("Faithfulness judge failed: %s", exc)
        return FaithfulnessVerdict(claims=[], score=0.0, reason=f"Judge failed: {exc}")


def judge_correctness(question: str, answer: str, reference: str) -> CorrectnessVerdict:
    """Evaluate correctness of an answer against the human reference answer.

    Args:
        question: The user question.
        answer: The generated answer.
        reference: The ground-truth reference answer.

    Returns:
        CorrectnessVerdict with score (0.0, 0.5, or 1.0) and justification.
    """
    if not answer.strip():
        return CorrectnessVerdict(score=0.0, reason="Empty answer")

    llm = get_llm("judge")
    messages = [
        SystemMessage(content=CORRECTNESS_JUDGE_SYSTEM),
        HumanMessage(
            content=CORRECTNESS_JUDGE_USER.format(
                question=question, reference_answer=reference, generated_answer=answer
            )
        ),
    ]

    try:
        verdict = invoke_structured(llm, CorrectnessVerdict, messages)
        # Snap score to closest of [0.0, 0.5, 1.0]
        allowed = [0.0, 0.5, 1.0]
        closest = min(allowed, key=lambda x: abs(x - verdict.score))
        verdict.score = closest
        return verdict
    except Exception as exc:
        logger.warning("Correctness judge failed: %s", exc)
        return CorrectnessVerdict(score=0.0, reason=f"Judge failed: {exc}")


def judge_abstention_fallback(question: str, answer: str) -> AbstentionVerdict:
    """Fallback judge when phrase-matching cannot determine if an answer abstained.

    Args:
        question: The question asked.
        answer: The generated answer.

    Returns:
        AbstentionVerdict indicating whether the answer is an abstention.
    """
    llm = get_llm("judge")
    messages = [
        SystemMessage(content=ABSTENTION_JUDGE_SYSTEM),
        HumanMessage(content=ABSTENTION_JUDGE_USER.format(question=question, answer=answer)),
    ]

    try:
        return invoke_structured(llm, AbstentionVerdict, messages)
    except Exception as exc:
        logger.warning("Abstention fallback judge failed: %s", exc)
        return AbstentionVerdict(is_abstention=False, reason=f"Judge failed: {exc}")

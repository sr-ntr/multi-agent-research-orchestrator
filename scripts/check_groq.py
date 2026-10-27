"""
Groq Cloud and FastEmbed smoke test.

Usage:
    python -m scripts.check_groq

Checks (prints PASS or FAIL for each):
    1. Groq Model API  -- verifies GROQ_API_KEY and model availability
    2. Chat            -- one plain chat call with fast model (openai/gpt-oss-20b)
    3. FastEmbed       -- local embedding call (prints vector dimension)
    4. Structured      -- one structured-output call with Pydantic schema
    5. Tool call       -- one tool-calling round trip with a dummy tool

Requires GROQ_API_KEY in .env or the environment.
"""

import logging
import pathlib
import sys

logging.basicConfig(level=logging.WARNING)

# Ensure project root is on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from src.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    MODEL_FAST,
    MODEL_JUDGE,
    MODEL_STRONG,
)
from src.llm import get_llm, text_of


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pass(label: str, detail: str = "") -> None:
    msg = f"  PASS  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def _fail(label: str, error: str) -> None:
    print(f"  FAIL  {label}  -- {error}")


# ---------------------------------------------------------------------------
# Check 1: Groq API key & models
# ---------------------------------------------------------------------------


def check_groq_api() -> bool:
    label = "groq connectivity"
    if not GROQ_API_KEY:
        _fail(label, "GROQ_API_KEY is not set in .env")
        return False
    try:
        from langchain_groq import ChatGroq  # type: ignore

        llm = ChatGroq(model_name=MODEL_FAST, groq_api_key=GROQ_API_KEY)
        _pass(label, f"API key configured, using fast model '{MODEL_FAST}'")
        return True
    except Exception as exc:
        _fail(label, str(exc))
        return False


# ---------------------------------------------------------------------------
# Check 2: Plain chat call
# ---------------------------------------------------------------------------


def check_chat() -> bool:
    label = "chat"
    try:
        llm = get_llm("fast")
        from langchain_core.messages import HumanMessage

        response = llm.invoke([HumanMessage(content="Reply with the single word: hello")])
        text = text_of(response).strip().lower()
        if "hello" not in text:
            _fail(label, f"unexpected response: {text!r}")
            return False
        _pass(label, f"response contains 'hello'")
        return True
    except Exception as exc:
        _fail(label, str(exc))
        return False


# ---------------------------------------------------------------------------
# Check 3: FastEmbed local embedding call
# ---------------------------------------------------------------------------


def check_fastembed() -> bool:
    label = "fastembed"
    try:
        from src.embeddings import embed_query

        vec = embed_query("This is a test sentence for FastEmbed.")
        dim = len(vec)
        _pass(label, f"dim={dim} (model={EMBEDDING_MODEL}, configured EMBEDDING_DIM={EMBEDDING_DIM})")
        if dim != EMBEDDING_DIM:
            print(
                f"  NOTE  embedding dimension {dim} differs from "
                f"EMBEDDING_DIM={EMBEDDING_DIM}; update .env if needed"
            )
        return True
    except Exception as exc:
        _fail(label, str(exc))
        return False


# ---------------------------------------------------------------------------
# Check 4: Structured output
# ---------------------------------------------------------------------------


class _Greeting(BaseModel):
    message: str
    language: str


def check_structured_output() -> bool:
    label = "structured output"
    try:
        llm = get_llm("fast")
        structured = llm.with_structured_output(_Greeting)
        result = structured.invoke(
            "Respond with a greeting in English. Use the schema provided."
        )
        if not isinstance(result, _Greeting):
            _fail(label, f"got {type(result).__name__} instead of _Greeting")
            return False
        _pass(label, f"message={result.message!r} language={result.language!r}")
        return True
    except Exception as exc:
        _fail(label, str(exc))
        return False


# ---------------------------------------------------------------------------
# Check 5: Tool-calling round trip
# ---------------------------------------------------------------------------


def check_tool_call() -> bool:
    label = "tool call"
    try:
        from langchain_core.messages import HumanMessage, ToolMessage
        from langchain_core.tools import tool

        @tool
        def get_weather(city: str) -> str:
            """Return a fake weather report for testing purposes."""
            return f"The weather in {city} is sunny and 22C."

        llm = get_llm("fast")
        llm_with_tools = llm.bind_tools([get_weather])

        messages = [HumanMessage(content="What is the weather in Paris?")]

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            _fail(label, f"model did not call the tool; response: {text_of(response)!r}")
            return False

        for tc in response.tool_calls:
            tool_args = tc.get("args") if isinstance(tc, dict) and "args" in tc else tc
            tool_result = get_weather.invoke(tool_args)
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc["id"],
                )
            )

        final = llm_with_tools.invoke(messages)
        final_text = text_of(final)

        if not final_text:
            _fail(label, "empty final response after tool result")
            return False

        _pass(label, f"tool called and answer received (len={len(final_text)})")
        return True
    except Exception as exc:
        _fail(label, str(exc))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY is not set. Add your Groq API key to .env:")
        print("  GROQ_API_KEY=gsk_...")
        sys.exit(1)

    print("\nGroq Cloud + FastEmbed Smoke Test")
    print(f"  MODEL_FAST    = {MODEL_FAST}")
    print(f"  MODEL_STRONG  = {MODEL_STRONG}")
    print(f"  MODEL_JUDGE   = {MODEL_JUDGE}")
    print(f"  EMBEDDING     = {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print()

    results = [
        check_groq_api(),
        check_chat(),
        check_fastembed(),
        check_structured_output(),
        check_tool_call(),
    ]

    passed = sum(results)
    total = len(results)
    print()
    print(f"{'All checks passed' if passed == total else f'{passed}/{total} checks passed'}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()

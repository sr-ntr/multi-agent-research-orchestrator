"""
Gemini smoke test for the Agentic Document Research Assistant.

Usage:
    python -m scripts.check_gemini

Checks (prints PASS or FAIL for each):
    1. Model list   -- lists available models, flags whether configured IDs exist
    2. Chat         -- one plain chat call
    3. Embedding    -- one embedding call, prints dimension
    4. Structured   -- one structured-output call with a Pydantic schema
    5. Tool call    -- one tool-calling round trip with a dummy tool

Requires GOOGLE_API_KEY in .env or the environment.
"""

import sys
import logging
from typing import Optional

logging.basicConfig(level=logging.WARNING)

# Ensure project root is on sys.path when run as a module
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    MODEL_FAST,
    MODEL_STRONG,
    MODEL_JUDGE,
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
# Check 1: model list
# ---------------------------------------------------------------------------


def check_model_list() -> bool:
    label = "model list"
    try:
        models = []
        try:
            from google import genai  # type: ignore
            client = genai.Client(api_key=GOOGLE_API_KEY)
            models = [m.name for m in client.models.list()]
        except ImportError:
            import google.generativeai as legacy_genai  # type: ignore
            legacy_genai.configure(api_key=GOOGLE_API_KEY)
            models = [m.name for m in legacy_genai.list_models()]

        wanted = {MODEL_FAST, MODEL_STRONG, MODEL_JUDGE, EMBEDDING_MODEL}
        # Model names in the API can be prefixed with "models/"
        available = {m.removeprefix("models/") for m in models}
        found = wanted & available
        missing = wanted - available
        detail = f"{len(models)} models available"
        if missing:
            embed_options = [m for m in available if "embed" in m.lower()]
            detail += f"; MISSING: {', '.join(sorted(missing))}"
            if any("embed" in m for m in missing) and embed_options:
                detail += f" (available embedding models: {', '.join(sorted(embed_options)[:4])})"
        else:
            detail += f"; all configured IDs found"
        _pass(label, detail)
        return True
    except Exception as exc:
        _fail(label, str(exc))
        return False


# ---------------------------------------------------------------------------
# Check 2: plain chat call
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
# Check 3: embedding call
# ---------------------------------------------------------------------------


def check_embedding() -> bool:
    label = "embedding"
    try:
        vec = None
        # Try google-genai
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
            client = genai.Client(api_key=GOOGLE_API_KEY)
            model_target = EMBEDDING_MODEL if not EMBEDDING_MODEL.startswith("models/") else EMBEDDING_MODEL
            config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT") if hasattr(types, "EmbedContentConfig") else None
            res = client.models.embed_content(
                model=model_target,
                contents="This is a test sentence.",
                config=config,
            )
            if hasattr(res, "embeddings") and res.embeddings:
                vec = list(res.embeddings[0].values)
            elif hasattr(res, "embedding") and res.embedding:
                vec = list(res.embedding.values)
        except (ImportError, Exception):
            pass

        # Try google-generativeai
        if vec is None:
            try:
                import google.generativeai as legacy_genai  # type: ignore
                legacy_genai.configure(api_key=GOOGLE_API_KEY)
                res = legacy_genai.embed_content(
                    model=f"models/{EMBEDDING_MODEL}",
                    content="This is a test sentence.",
                    task_type="RETRIEVAL_DOCUMENT",
                )
                vec = res["embedding"]
            except (ImportError, Exception):
                pass

        # Try langchain-google-genai fallback
        if vec is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
            embedder = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY,
            )
            vec = embedder.embed_query("This is a test sentence.")

        dim = len(vec)
        _pass(label, f"dim={dim} (configured EMBEDDING_DIM={EMBEDDING_DIM})")
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
# Check 4: structured output
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
            'Respond with a greeting in English. Use the schema provided.'
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
# Check 5: tool-calling round trip
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

        # First turn: model decides to call the tool
        response = llm_with_tools.invoke(messages)
        messages.append(response)  # keep AIMessage unmodified

        if not response.tool_calls:
            _fail(label, f"model did not call the tool; response: {text_of(response)!r}")
            return False

        # Execute the tool and append ToolMessages
        for tc in response.tool_calls:
            tool_result = get_weather.invoke(tc)
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc["id"],
                )
            )

        # Second turn: model uses tool result to answer
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
    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    print(f"\nGemini smoke test")
    print(f"  MODEL_FAST    = {MODEL_FAST}")
    print(f"  MODEL_STRONG  = {MODEL_STRONG}")
    print(f"  MODEL_JUDGE   = {MODEL_JUDGE}")
    print(f"  EMBEDDING     = {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print()

    results = [
        check_model_list(),
        check_chat(),
        check_embedding(),
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

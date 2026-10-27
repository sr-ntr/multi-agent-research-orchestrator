"""
LLM helpers for the Agentic Document Research Assistant (Groq Cloud).

Exports:
    get_llm(role)       -- create a ChatGroq instance for the given role
    text_of(message)    -- extract plain text from any LangChain message
    UsageCallback       -- BaseCallbackHandler that accumulates token usage
"""

import json
import logging
import re
from typing import Any, Literal, Optional

from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import BaseCallbackHandler
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config import (
    GROQ_API_KEY,
    GOOGLE_API_KEY,
    MODEL_FAST,
    MODEL_JUDGE,
    MODEL_STRONG,
)

logger = logging.getLogger(__name__)

Role = Literal["fast", "strong", "judge"]


# ---------------------------------------------------------------------------
# text_of: normalise message content to a plain string
# ---------------------------------------------------------------------------


def text_of(message: BaseMessage | Any) -> str:
    """Extract plain-text content from any LangChain message.

    Normalises string content and list-of-blocks content (text/thinking parts)
    so callers never need scattered isinstance checks.

    Args:
        message: A LangChain BaseMessage, or any object with a ``content``
            attribute, or a raw string.

    Returns:
        The concatenated text content as a single string.
    """
    if isinstance(message, str):
        return message

    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return "".join(parts)

    return str(content)


# ---------------------------------------------------------------------------
# Retry predicate
# ---------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient HTTP errors that warrant a retry."""
    msg = str(exc).lower()
    retryable_codes = ("429", "503", "rate limit", "resource exhausted", "service unavailable")
    return any(code in msg for code in retryable_codes)


# ---------------------------------------------------------------------------
# Usage callback
# ---------------------------------------------------------------------------


class UsageCallback(BaseCallbackHandler):
    """Accumulate token usage from Groq / LLM responses.

    Supports both LangChain usage_metadata and OpenAI/Groq token_usage structures.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.call_count: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract usage metadata from generation and accumulate totals."""
        for generations in response.generations:
            for gen in generations:
                usage = getattr(gen, "generation_info", None) or {}
                metadata = usage.get("usage_metadata") or usage.get("token_usage") or {}
                if not metadata:
                    llm_out = getattr(response, "llm_output", {}) or {}
                    metadata = llm_out.get("usage_metadata") or llm_out.get("token_usage") or {}

                prompt = (
                    metadata.get("prompt_token_count")
                    or metadata.get("prompt_tokens")
                    or 0
                )
                completion = (
                    metadata.get("candidates_token_count")
                    or metadata.get("completion_tokens")
                    or 0
                )
                total = (
                    metadata.get("total_token_count")
                    or metadata.get("total_tokens")
                    or (prompt + completion)
                )

                self.prompt_tokens += prompt
                self.completion_tokens += completion
                self.total_tokens += total
                self.call_count += 1

    def summary(self) -> str:
        """Return a one-line usage summary string."""
        return (
            f"calls={self.call_count} "
            f"prompt={self.prompt_tokens} "
            f"completion={self.completion_tokens} "
            f"total={self.total_tokens}"
        )


# ---------------------------------------------------------------------------
# LLM factory (Groq Cloud)
# ---------------------------------------------------------------------------


def get_llm(role: Role = "fast", **kwargs: Any) -> Any:
    """Create an LLM chat model for the given role using Groq Cloud.

    Roles:
        fast    -- planner and researcher; uses MODEL_FAST (openai/gpt-oss-20b)
        strong  -- writer and verifier; uses MODEL_STRONG (openai/gpt-oss-120b)
        judge   -- eval judges; uses MODEL_JUDGE (openai/gpt-oss-120b)

    Args:
        role: One of "fast", "strong", "judge".
        **kwargs: Forwarded to chat model constructor.

    Returns:
        Configured ChatGroq instance.
    """
    model_map: dict[str, str] = {
        "fast": MODEL_FAST,
        "strong": MODEL_STRONG,
        "judge": MODEL_JUDGE,
    }
    model_name = model_map[role]

    # Primary: langchain-groq
    try:
        from langchain_groq import ChatGroq  # type: ignore

        llm_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "groq_api_key": GROQ_API_KEY,
            "temperature": 0.0,
            "request_timeout": 30.0,
            "max_retries": 2,
        }
        llm_kwargs.update(kwargs)
        logger.debug("Creating ChatGroq: role=%s model=%s", role, model_name)
        return ChatGroq(**llm_kwargs)
    except ImportError:
        pass

    # Secondary: langchain-openai pointing to Groq's endpoint
    try:
        from langchain_openai import ChatOpenAI  # type: ignore

        llm_kwargs = {
            "model": model_name,
            "api_key": GROQ_API_KEY,
            "base_url": "https://api.groq.com/openai/v1",
            "temperature": 0.0,
            "timeout": 30.0,
            "max_retries": 2,
        }
        llm_kwargs.update(kwargs)
        logger.debug("Creating ChatOpenAI (Groq endpoint): role=%s model=%s", role, model_name)
        return ChatOpenAI(**llm_kwargs)
    except ImportError:
        pass

    # Fallback to Google GenAI if groq libraries are not yet installed
    if GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        fallback_model = "gemini-2.0-flash-lite" if role == "fast" else "gemini-2.0-flash"
        logger.warning(
            "langchain-groq not found; falling back to ChatGoogleGenerativeAI (%s)",
            fallback_model,
        )
        return ChatGoogleGenerativeAI(
            model=fallback_model,
            google_api_key=GOOGLE_API_KEY,
            request_timeout=30.0,
            **kwargs,
        )

    raise ImportError(
        "Could not initialize LLM. Please install langchain-groq:\n"
        "  pip install langchain-groq\n"
        "And set GROQ_API_KEY in your .env file."
    )


# ---------------------------------------------------------------------------
# Retry-wrapped invocation helper
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
    stop=stop_after_attempt(3),
    reraise=True,
)
def invoke_with_retry(llm: Any, messages: list, **kwargs: Any) -> Any:
    """Invoke an LLM with quick retry on transient 429/503 errors.

    Args:
        llm: A ChatGroq or compatible LangChain chat model.
        messages: The list of messages to send.
        **kwargs: Additional keyword arguments forwarded to ``llm.invoke``.

    Returns:
        The model response message.
    """
    return llm.invoke(messages, **kwargs)


# ---------------------------------------------------------------------------
# Structured output invocation helper (Groq Cloud resilience)
# ---------------------------------------------------------------------------


def _extract_from_failed_generation(error_msg: str, schema: type[Any]) -> Optional[Any]:
    """Recover schema instance from Groq tool-validation error containing failed_generation.

    When open-weights models (e.g. gpt-oss-20b/120b) generate tool names like 'planner'
    or 'verdict' instead of 'Plan' or 'Verdict', Groq rejects with HTTP 400 tool_use_failed,
    but includes the completed valid JSON arguments in the failed_generation payload.
    """
    patterns = [
        r"failed_generation['\"]\s*:\s*(['\"].*?['\"])\s*\}",
        r"failed_generation['\"]\s*:\s*(\{.*?\})\s*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_msg, re.DOTALL)
        if match:
            raw = match.group(1).strip()
            if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
                raw = raw[1:-1]
            raw = raw.replace('\\"', '"').replace("\\'", "'").replace("\\n", "\n").replace("\\\\", "\\")
            try:
                data = json.loads(raw)
            except Exception:
                continue

            args = data.get("arguments", data)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if isinstance(args, dict):
                try:
                    return schema.model_validate(args)
                except Exception as val_err:
                    logger.debug("Failed schema model_validate for %s: %s", schema.__name__, val_err)
    return None


def invoke_structured(llm: Any, schema: type[Any], messages: list, **kwargs: Any) -> Any:
    """Invoke an LLM returning a Pydantic schema instance using the most reliable method.

    Prioritizes Groq native JSON schema / JSON mode to avoid tool name casing mismatches,
    and automatically recovers valid payloads from failed_generation if a tool-calling
    mismatch occurs.

    Args:
        llm: A ChatGroq or compatible LangChain chat model.
        schema: Target Pydantic model class.
        messages: Prompt messages.
        **kwargs: Additional kwargs forwarded to with_structured_output.

    Returns:
        Instance of schema.
    """
    # 1. Try native json_schema (Groq LPU grammar constrained decoding)
    try:
        runnable = llm.with_structured_output(schema, method="json_schema", **kwargs)
        return runnable.invoke(messages)
    except Exception as exc:
        logger.debug("Structured output with json_schema failed: %s", exc)

    # 2. Try json_mode
    try:
        runnable = llm.with_structured_output(schema, method="json_mode", **kwargs)
        return runnable.invoke(messages)
    except Exception as exc:
        logger.debug("Structured output with json_mode failed: %s", exc)

    # 3. Try default method (function calling) with recovery from failed_generation
    try:
        runnable = llm.with_structured_output(schema, **kwargs)
        return runnable.invoke(messages)
    except Exception as exc:
        err_str = str(exc)
        if "failed_generation" in err_str:
            recovered = _extract_from_failed_generation(err_str, schema)
            if recovered is not None:
                logger.info("Successfully recovered %s from failed_generation", schema.__name__)
                return recovered
        raise


"""
LLM helpers for the Agentic Document Research Assistant.

Exports:
    get_llm(role)       -- create a ChatGoogleGenerativeAI for the given role
    text_of(message)    -- extract plain text from any LangChain message
    UsageCallback       -- BaseCallbackHandler that accumulates token usage
"""

import logging
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config import (
    GOOGLE_API_KEY,
    LLM_MAX_CONCURRENCY,
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
    """Extract the plain-text content from a LangChain message.

    Gemini responses can arrive with content as either a plain string or a
    list of typed blocks (e.g., a thinking block followed by a text block).
    This helper normalises both cases so callers never need isinstance checks.

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
                # Gemini typed block: {"type": "text", "text": "..."}
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
    retryable_codes = ("429", "503", "resource exhausted", "service unavailable")
    return any(code in msg for code in retryable_codes)


# ---------------------------------------------------------------------------
# Usage callback
# ---------------------------------------------------------------------------


class UsageCallback(BaseCallbackHandler):
    """Accumulate token usage from Gemini responses.

    Attach an instance to a chain via ``config={"callbacks": [cb]}`` and read
    ``cb.total_tokens``, ``cb.prompt_tokens``, ``cb.completion_tokens`` at the end.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.call_count: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract usage_metadata from each generation and accumulate totals."""
        for generations in response.generations:
            for gen in generations:
                usage = getattr(gen, "generation_info", None) or {}
                metadata = usage.get("usage_metadata", {})
                if not metadata:
                    # Some versions expose it directly on the response
                    metadata = getattr(response, "llm_output", {}) or {}
                    metadata = metadata.get("usage_metadata", {})

                prompt = metadata.get("prompt_token_count", 0) or 0
                completion = metadata.get("candidates_token_count", 0) or 0
                total = metadata.get("total_token_count", prompt + completion)

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
# LLM factory
# ---------------------------------------------------------------------------


def get_llm(role: Role = "fast", **kwargs: Any) -> ChatGoogleGenerativeAI:
    """Create a ChatGoogleGenerativeAI instance for the given role.

    Roles:
        fast    -- planner and researcher; uses MODEL_FAST
        strong  -- writer and verifier; uses MODEL_STRONG
        judge   -- eval judges; uses MODEL_JUDGE

    Temperature is intentionally left at the model default following
    current Gemini 2.0 guidance. Override via kwargs if needed.

    The returned LLM is NOT wrapped with retry here; use the
    ``with_retry`` helper below when you need call-level retry.

    Args:
        role: One of "fast", "strong", "judge".
        **kwargs: Additional keyword arguments forwarded to
            ChatGoogleGenerativeAI (e.g., temperature, max_tokens).

    Returns:
        A configured ChatGoogleGenerativeAI instance.
    """
    model_map: dict[str, str] = {
        "fast": MODEL_FAST,
        "strong": MODEL_STRONG,
        "judge": MODEL_JUDGE,
    }
    model_name = model_map[role]

    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "google_api_key": GOOGLE_API_KEY,
        "max_retries": 0,  # we handle retries ourselves via tenacity
    }
    llm_kwargs.update(kwargs)

    logger.debug("Creating LLM: role=%s model=%s", role, model_name)
    return ChatGoogleGenerativeAI(**llm_kwargs)


# ---------------------------------------------------------------------------
# Retry-wrapped invocation helper
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential_jitter(initial=2, max=60, jitter=4),
    stop=stop_after_attempt(5),
    reraise=True,
)
def invoke_with_retry(llm: ChatGoogleGenerativeAI, messages: list, **kwargs: Any) -> Any:
    """Invoke an LLM with tenacity retry on transient errors (429, 503).

    Uses exponential backoff with jitter. Reraises after 5 failed attempts.

    Args:
        llm: A ChatGoogleGenerativeAI (or any LangChain chat model).
        messages: The list of messages to send.
        **kwargs: Additional keyword arguments forwarded to ``llm.invoke``.

    Returns:
        The model response message.
    """
    return llm.invoke(messages, **kwargs)

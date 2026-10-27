"""
Web search provider interface and implementations.

Exports:
    WebSearchProvider       -- Protocol defining the provider interface
    WebSearchResult         -- TypedDict for a single search result
    DdgsProvider            -- DuckDuckGo implementation (no API key required)
    TavilyProvider          -- Tavily implementation (requires TAVILY_API_KEY)
    FakeProvider            -- Deterministic fake for tests
    get_provider()          -- Return the configured provider instance

Result schema:
    {
        "id":    str,   # "web:" + sha1(url)[:8]
        "url":   str,
        "title": str,
        "text":  str,   # snippet, truncated to WEB_SNIPPET_MAX_CHARS
    }

Error handling: providers return an error result dict instead of raising so
the agent can continue gracefully.
"""

import hashlib
import logging
from typing import Protocol, runtime_checkable

from src.config import (
    TAVILY_API_KEY,
    WEB_MAX_RESULTS,
    WEB_SEARCH_PROVIDER,
    WEB_SNIPPET_MAX_CHARS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def _web_id(url: str) -> str:
    """Generate a short deterministic ID from a URL."""
    return "web:" + hashlib.sha1(url.encode()).hexdigest()[:8]


def _truncate(text: str, max_chars: int = WEB_SNIPPET_MAX_CHARS) -> str:
    """Truncate text to max_chars, appending '...' if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _error_result(query: str, error: str) -> list[dict]:
    """Return a single error result when the provider fails."""
    return [
        {
            "id": "web:error",
            "url": "",
            "title": "Search error",
            "text": f"Web search failed for query '{query}': {error}",
        }
    ]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class WebSearchProvider(Protocol):
    """Interface for web search providers."""

    def search(self, query: str, max_results: int = WEB_MAX_RESULTS) -> list[dict]:
        """Search the web and return a list of result dicts.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of result dicts with keys: id, url, title, text.
            On failure, returns a list containing a single error result dict.
        """
        ...


# ---------------------------------------------------------------------------
# DuckDuckGo provider
# ---------------------------------------------------------------------------


class DdgsProvider:
    """Web search using the DuckDuckGo Search (ddgs) package.

    No API key required. Returns organic search snippets only (no full page).
    """

    def search(self, query: str, max_results: int = WEB_MAX_RESULTS) -> list[dict]:
        """Search DuckDuckGo and return deduplicated results.

        Args:
            query: The search query string.
            max_results: Maximum number of results.

        Returns:
            List of result dicts. Returns an error result on failure.
        """
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            return _error_result(query, "ddgs package not installed")

        try:
            results = []
            seen_urls: set[str] = set()
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results * 2):
                    url = r.get("href", r.get("link", ""))
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    title = r.get("title", "")
                    snippet = r.get("body", r.get("snippet", ""))
                    results.append(
                        {
                            "id": _web_id(url),
                            "url": url,
                            "title": title,
                            "text": _truncate(snippet),
                        }
                    )
                    if len(results) >= max_results:
                        break
            return results if results else _error_result(query, "no results returned")
        except Exception as exc:
            logger.warning("DdgsProvider.search failed: %s", exc)
            return _error_result(query, str(exc))


# ---------------------------------------------------------------------------
# Tavily provider (optional, lazy import)
# ---------------------------------------------------------------------------


class TavilyProvider:
    """Web search using the Tavily API.

    Requires TAVILY_API_KEY to be set. Only instantiated when that key exists.
    The tavily-python package is imported lazily.
    """

    def search(self, query: str, max_results: int = WEB_MAX_RESULTS) -> list[dict]:
        """Search Tavily and return results.

        Args:
            query: The search query string.
            max_results: Maximum number of results.

        Returns:
            List of result dicts. Returns an error result on failure.
        """
        if not TAVILY_API_KEY:
            return _error_result(query, "TAVILY_API_KEY not set")
        try:
            from tavily import TavilyClient  # type: ignore

            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(query, max_results=max_results)
            results = []
            seen_urls: set[str] = set()
            for r in response.get("results", []):
                url = r.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(
                    {
                        "id": _web_id(url),
                        "url": url,
                        "title": r.get("title", ""),
                        "text": _truncate(r.get("content", "")),
                    }
                )
            return results if results else _error_result(query, "no results returned")
        except Exception as exc:
            logger.warning("TavilyProvider.search failed: %s", exc)
            return _error_result(query, str(exc))


# ---------------------------------------------------------------------------
# Fake provider (for tests)
# ---------------------------------------------------------------------------


class FakeProvider:
    """Deterministic fake provider for offline tests.

    Returns pre-configured results or a default set if none are configured.
    """

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results or [
            {
                "id": "web:abc12345",
                "url": "https://example.com/page1",
                "title": "Example Page 1",
                "text": "This is a fake search result for testing purposes.",
            },
            {
                "id": "web:def67890",
                "url": "https://example.com/page2",
                "title": "Example Page 2",
                "text": "Another fake result with different content.",
            },
        ]

    def search(self, query: str, max_results: int = WEB_MAX_RESULTS) -> list[dict]:
        """Return preconfigured fake results."""
        return self._results[:max_results]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_provider() -> WebSearchProvider:
    """Return the configured web search provider.

    Uses WEB_SEARCH_PROVIDER env var to select ddgs or tavily.
    Falls back to ddgs if the configured provider is unavailable.

    Returns:
        An instance of the configured WebSearchProvider.
    """
    provider_name = WEB_SEARCH_PROVIDER.lower()
    if provider_name == "tavily" and TAVILY_API_KEY:
        logger.debug("Using TavilyProvider")
        return TavilyProvider()
    logger.debug("Using DdgsProvider")
    return DdgsProvider()

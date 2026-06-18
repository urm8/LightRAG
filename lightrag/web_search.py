"""
Web search utilities for LightRAG agent tools.

Provides DuckDuckGo search (no API key required) and a WebSearchResult
data class for structured result formatting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lightrag.utils import logger

try:
    from ddgs import DDGS

    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False
    DDGS = None  # type: ignore[assignment]


@dataclass
class WebSearchResult:
    """A single web search result with metadata."""

    title: str
    link: str
    snippet: str = ""
    source: str = "web"

    def __str__(self) -> str:
        return f"Title: {self.title}\nLink: {self.link}\nSnippet: {self.snippet}"

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "link": self.link,
            "snippet": self.snippet,
            "source": self.source,
        }


def duckduckgo_search(
    query: str,
    num_results: int = 5,
    region: str = "wt-wt",
) -> list[WebSearchResult]:
    """Search the web using DuckDuckGo (no API key required).

    Args:
        query: Search query string.
        num_results: Maximum number of results (default 5, max 20).
        region: Region code (default 'wt-wt' for worldwide).

    Returns:
        List of WebSearchResult objects.

    Raises:
        ImportError: If duckduckgo_search package is not installed.
        RuntimeError: If the search fails unexpectedly.
    """
    if not HAS_DUCKDUCKGO:
        raise ImportError(
            "duckduckgo_search is required for web search. "
            "Install it with: uv pip install duckduckgo_search"
            " or pip install duckduckgo_search"
        )

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=min(num_results, 20),
                    region=region,
                )
            )
    except Exception as e:
        logger.error("[web_search] DuckDuckGo search failed: %s", e)
        raise RuntimeError(f"DuckDuckGo search failed: {e}") from e

    if not results:
        return []

    parsed: list[WebSearchResult] = []
    for r in results:
        title = r.get("title", "")
        link = r.get("href", "")
        snippet = r.get("body", "")
        if title or link:
            parsed.append(
                WebSearchResult(
                    title=title,
                    link=link,
                    snippet=snippet,
                    source="duckduckgo",
                )
            )

    return parsed


def format_search_results(results: list[WebSearchResult]) -> str:
    """Format a list of WebSearchResult into a human-readable string.

    Suitable for returning to the LLM as a tool result.
    """
    if not results:
        return "No web search results found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        if r.snippet:
            lines.append(f"    Summary: {r.snippet[:300]}")
        lines.append(f"    URL: {r.link}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry of available search providers (extensible)
# ---------------------------------------------------------------------------

SEARCH_PROVIDERS: dict[str, Any] = {
    "duckduckgo": duckduckgo_search,
}


def get_search_provider(name: str = "duckduckgo"):
    """Get a search function by provider name.

    Args:
        name: Provider name ('duckduckgo').

    Returns:
        A search function with signature ``(query: str, num_results: int) -> list[WebSearchResult]``.

    Raises:
        ValueError: If the provider is not registered.
    """
    provider = SEARCH_PROVIDERS.get(name)
    if provider is None:
        available = ", ".join(sorted(SEARCH_PROVIDERS))
        raise ValueError(
            f"Unknown search provider '{name}'. Available providers: {available}"
        )
    return provider

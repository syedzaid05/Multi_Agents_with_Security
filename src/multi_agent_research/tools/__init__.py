"""External tools: Wikipedia and optional web search."""

from __future__ import annotations

from .web_search import (
    DuckDuckGoClient,
    TavilyClient,
    WebSearchClient,
    WebSearchUnavailable,
    build_web_search_client,
)
from .wikipedia import WikipediaClient, WikipediaError

__all__ = [
    "DuckDuckGoClient",
    "TavilyClient",
    "WebSearchClient",
    "WebSearchUnavailable",
    "WikipediaClient",
    "WikipediaError",
    "build_web_search_client",
]

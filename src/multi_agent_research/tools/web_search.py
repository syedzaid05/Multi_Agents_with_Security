"""Optional web-search backends.

* :class:`DuckDuckGoClient` - free, keyless (uses the ``ddgs`` package).
* :class:`TavilyClient` - higher quality; free tier then **PAID**.

The dependency is imported lazily so the rest of the system (and the tests) work
without the optional ``ddgs`` package installed.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import httpx

from ..citations import Document
from ..config import Settings

__all__ = [
    "DuckDuckGoClient",
    "TavilyClient",
    "WebSearchClient",
    "WebSearchUnavailable",
    "build_web_search_client",
]


class WebSearchUnavailable(RuntimeError):
    """Raised when web search is requested but not usable in this environment."""


class WebSearchClient(ABC):
    name: str

    @abstractmethod
    async def search(self, query: str, *, limit: int = 5) -> list[Document]:
        ...

    async def aclose(self) -> None:  # pragma: no cover - overridden if needed
        return None


class DuckDuckGoClient(WebSearchClient):
    name = "web:duckduckgo"

    async def search(self, query: str, *, limit: int = 5) -> list[Document]:
        def _run() -> list[dict]:
            try:
                from ddgs import DDGS  # type: ignore import-not-found
            except ImportError:  # pragma: no cover - exercised only without extra
                try:
                    from duckduckgo_search import DDGS  # type: ignore
                except ImportError as exc:
                    raise WebSearchUnavailable(
                        "Web search needs the optional dependency: run "
                        "`uv sync --extra web`."
                    ) from exc
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max(1, limit)))

        results = await asyncio.to_thread(_run)
        documents: list[Document] = []
        for item in results:
            url = item.get("href") or item.get("url") or ""
            if not url:
                continue
            documents.append(
                Document(
                    title=item.get("title") or url,
                    url=url,
                    text=(item.get("body") or "").strip(),
                    source="web",
                )
            )
        return documents


class TavilyClient(WebSearchClient):
    name = "web:tavily"

    def __init__(self, api_key: str, *, client: httpx.AsyncClient | None = None) -> None:
        if not api_key:
            raise WebSearchUnavailable("TAVILY_API_KEY is not set.")
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=20.0)
        return self._client

    async def search(self, query: str, *, limit: int = 5) -> list[Document]:
        client = self._ensure_client()
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max(1, limit),
            "include_answer": False,
        }
        response = await client.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
        data = response.json()
        documents: list[Document] = []
        for item in data.get("results", []):
            documents.append(
                Document(
                    title=item.get("title") or item.get("url", ""),
                    url=item.get("url", ""),
                    text=(item.get("content") or "").strip(),
                    source="web",
                    score=item.get("score"),
                )
            )
        return documents

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def build_web_search_client(settings: Settings) -> WebSearchClient | None:
    """Return the configured backend, or ``None`` when web search is disabled."""
    backend = settings.web_search_backend
    if backend in {"none", "off", "disabled", ""}:
        return None
    if backend == "tavily":
        if not settings.tavily_api_key:
            return None
        return TavilyClient(settings.tavily_api_key)
    # Default: DuckDuckGo (free, keyless).
    return DuckDuckGoClient()

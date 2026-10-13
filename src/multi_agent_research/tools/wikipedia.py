"""Wikipedia research client (free, no API key required).

Uses the public MediaWiki Action API for search and the REST summary endpoint
for article extracts. Both are unauthenticated and rate-friendly; a descriptive
User-Agent is sent as Wikimedia's policy requests.
"""

from __future__ import annotations

import asyncio
import html
import re
from urllib.parse import quote

import httpx

from ..citations import Document

__all__ = ["WikipediaClient", "WikipediaError"]

_TAG_RE = re.compile(r"<[^>]+>")
_USER_AGENT = (
    "MultiAgentResearchSystem/0.1 "
    "(https://github.com/multi-agent-research; educational project)"
)


class WikipediaError(RuntimeError):
    """Raised when Wikipedia cannot be reached or returns malformed data."""


def _clean(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text or "")).strip()


class WikipediaClient:
    def __init__(
        self,
        *,
        language: str = "en",
        client: httpx.AsyncClient | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.language = language
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    @property
    def api_url(self) -> str:
        return f"https://{self.language}.wikipedia.org/w/api.php"

    @property
    def rest_base(self) -> str:
        return f"https://{self.language}.wikipedia.org/api/rest_v1/page/summary"

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
                follow_redirects=True,
            )
        return self._client

    async def search_titles(self, query: str, *, limit: int = 3) -> list[str]:
        client = self._ensure_client()
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(max(1, limit)),
            "format": "json",
            "srprop": "snippet",
        }
        try:
            response = await client.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise WikipediaError(f"search failed: {exc}") from exc
        except ValueError as exc:
            raise WikipediaError(f"search returned invalid JSON: {exc}") from exc

        hits = (data.get("query") or {}).get("search") or []
        return [hit["title"] for hit in hits if hit.get("title")]

    async def get_summary(self, title: str) -> Document | None:
        client = self._ensure_client()
        url = f"{self.rest_base}/{quote(title.replace(' ', '_'), safe='')}"
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            raise WikipediaError(f"summary fetch failed: {exc}") from exc
        if response.status_code == 404:
            return None
        try:
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WikipediaError(f"summary fetch failed: {exc}") from exc

        extract = data.get("extract") or ""
        page_url = (
            (data.get("content_urls") or {}).get("desktop", {}).get("page")
            or f"https://{self.language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        )
        return Document(
            title=data.get("title") or title,
            url=page_url,
            text=_clean(extract),
            source="wikipedia",
            metadata={"description": _clean(data.get("description", ""))},
        )

    async def research(self, query: str, *, limit: int = 3) -> list[Document]:
        """Search then fetch summaries concurrently, dropping any misses."""
        titles = await self.search_titles(query, limit=limit)
        if not titles:
            return []
        summaries = await asyncio.gather(
            *(self.get_summary(title) for title in titles),
            return_exceptions=True,
        )
        documents: list[Document] = []
        for _title, result in zip(titles, summaries, strict=False):
            if isinstance(result, Document):
                documents.append(result)
            elif isinstance(result, Exception):  # noqa: PERF203 - small list
                # One bad article should not sink the whole research step.
                continue
        return documents

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

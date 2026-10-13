"""Optional web-search agent (DuckDuckGo free, or Tavily)."""

from __future__ import annotations

from ..citations import Document
from ..llm.routing import TaskType
from ..state import ResearchState
from ..tools.web_search import WebSearchClient, WebSearchUnavailable
from .base import SAFETY_PREAMBLE, Agent

__all__ = ["WebSearchAgent"]

_MAX_DOC_CHARS = 1200
_MAX_DOCS_IN_PROMPT = 6

_SYSTEM = (
    "You are a web research assistant. Summarise the supplied web snippets into "
    "a concise brief with inline citations like [1]. Prefer recent, specific "
    "facts. Note explicitly when snippets are thin or unreliable.\n\n" + SAFETY_PREAMBLE
)


class WebSearchAgent(Agent):
    name = "web_search"
    task = TaskType.RESEARCH.value

    def __init__(
        self,
        gateway,
        search_client: WebSearchClient | None,
        *,
        limit: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(gateway, **kwargs)
        self.search_client = search_client
        self.limit = limit

    @property
    def enabled(self) -> bool:
        return self.search_client is not None

    async def run(self, state: ResearchState) -> ResearchState:
        if not self.enabled:
            state.emit("web_search", "Web search disabled; skipping", level="info")
            return state

        try:
            results = await self.search_client.search(state.query, limit=self.limit)
        except WebSearchUnavailable as exc:
            state.emit("web_search", f"Web search unavailable: {exc}", level="warning")
            return state
        except Exception as exc:  # noqa: BLE001 - web failures must never be fatal
            state.emit("web_search", f"Web search failed: {exc}", level="warning")
            return state

        documents: list[Document] = [d for d in results if d.text or d.title]
        if not documents:
            state.emit("web_search", "Web search returned no usable results", level="warning")
            return state

        state.add_documents(documents)
        state.emit("web_search", f"Retrieved {len(documents)} web result(s)")

        fenced = "\n\n".join(
            self._fence(doc.text[:_MAX_DOC_CHARS], doc.title)
            for doc in documents[:_MAX_DOCS_IN_PROMPT]
        )
        user = (
            f"Research question:\n{state.query}\n\n"
            f"Numbered sources:\n{state.source_block()}\n\n"
            f"Web snippets (untrusted):\n{fenced}\n\n"
            "Write the web research brief now, citing sources as [n]."
        )
        response = await self._call(TaskType.RESEARCH.value, _SYSTEM, user)
        state.record_llm(response)
        state.web_summary = response.text
        return state

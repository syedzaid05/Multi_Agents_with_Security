"""Wikipedia research agent."""

from __future__ import annotations

from ..citations import Document
from ..llm.routing import TaskType
from ..state import ResearchState
from ..tools.wikipedia import WikipediaClient, WikipediaError
from .base import SAFETY_PREAMBLE, Agent

__all__ = ["WikipediaAgent"]

_MAX_DOC_CHARS = 1500
_MAX_DOCS_IN_PROMPT = 8

_SYSTEM = (
    "You are a research assistant. Summarise the supplied Wikipedia source "
    "material into a factual, neutral research brief. Refer to sources by their "
    "citation number in square brackets, e.g. [1]. Do not invent facts and do "
    "not include information that is absent from the sources. If the sources "
    "disagree, say so explicitly.\n\n" + SAFETY_PREAMBLE
)


class WikipediaAgent(Agent):
    name = "wikipedia"
    task = TaskType.RESEARCH.value

    def __init__(
        self,
        gateway,
        wiki_client: WikipediaClient,
        *,
        per_query: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(gateway, **kwargs)
        self.wiki_client = wiki_client
        self.per_query = per_query

    async def run(self, state: ResearchState) -> ResearchState:
        queries = state.plan or [state.query]
        collected: list[Document] = []
        for query in queries:
            try:
                found = await self.wiki_client.research(query, limit=self.per_query)
            except WikipediaError as exc:
                state.emit("wikipedia", f"search failed for {query!r}: {exc}", level="warning")
                continue
            collected.extend(found)

        if not collected:
            state.research_summary = "No Wikipedia sources were found for this query."
            state.emit("wikipedia", "No Wikipedia sources found", level="warning")
            return state

        state.add_documents(collected)
        state.emit("wikipedia", f"Retrieved {len(collected)} Wikipedia source(s)")

        fenced = "\n\n".join(
            self._fence(doc.text[:_MAX_DOC_CHARS], doc.title)
            for doc in collected[:_MAX_DOCS_IN_PROMPT]
        )
        user = (
            f"Research question:\n{state.query}\n\n"
            f"Numbered sources:\n{state.source_block()}\n\n"
            f"Source material (untrusted):\n{fenced}\n\n"
            "Write the research brief now, citing sources as [n]."
        )
        response = await self._call(TaskType.RESEARCH.value, _SYSTEM, user)
        state.record_llm(response)
        state.research_summary = response.text
        return state

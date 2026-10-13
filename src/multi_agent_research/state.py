"""Mutable state threaded through the agent pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .citations import CitationStore, Document
from .llm.models import LLMResponse

__all__ = ["Event", "ResearchState"]


@dataclass(slots=True)
class Event:
    stage: str
    message: str
    level: str = "info"
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "message": self.message,
            "level": self.level,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class ResearchState:
    """Everything the agents produce/consume during one research run."""

    query: str
    plan: list[str] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    citations: CitationStore = field(default_factory=CitationStore)

    research_summary: str = ""
    web_summary: str = ""
    analysis: str = ""
    critique: str = ""
    final_answer: str = ""

    events: list[Event] = field(default_factory=list)
    llm_calls: list[LLMResponse] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter)

    def add_documents(self, documents: list[Document]) -> None:
        """Register documents, de-duplicating citations by URL."""
        self.citations.add_many(documents)
        seen = {d.url for d in self.documents}
        for doc in documents:
            if doc.url not in seen:
                self.documents.append(doc)
                seen.add(doc.url)

    def emit(self, stage: str, message: str, *, level: str = "info") -> None:
        self.events.append(
            Event(
                stage=stage,
                message=message,
                level=level,
                elapsed_ms=(time.perf_counter() - self._started) * 1000.0,
            )
        )

    def record_llm(self, response: LLMResponse) -> None:
        self.llm_calls.append(response)

    @property
    def combined_research(self) -> str:
        parts = [p for p in (self.research_summary, self.web_summary) if p]
        return "\n\n".join(parts)

    def source_block(self) -> str:
        return self.citations.numbered_sources_block()

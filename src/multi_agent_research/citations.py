"""Source tracking and citation formatting.

A :class:`CitationStore` assigns stable ``[1]``-style numbers to retrieved
documents as they are added, deduplicating by URL. Agents refer to sources by
number; the store renders the final reference list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

__all__ = ["Citation", "CitationStore", "Document"]


@dataclass(slots=True)
class Document:
    """A retrieved piece of source material."""

    title: str
    url: str
    text: str
    source: str  # "wikipedia" | "web" | ...
    score: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Citation:
    number: int
    title: str
    url: str
    source: str
    accessed_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "accessed_at": self.accessed_at,
        }


class CitationStore:
    """Assigns citation numbers to documents, de-duplicating by URL."""

    def __init__(self) -> None:
        self._by_url: dict[str, Citation] = {}
        self._documents: list[Document] = []

    def add(self, document: Document, *, accessed_at: datetime | None = None) -> Citation:
        key = document.url.strip()
        if key in self._by_url:
            return self._by_url[key]
        number = len(self._by_url) + 1
        timestamp = (accessed_at or datetime.now(UTC)).astimezone(
            UTC
        ).strftime("%Y-%m-%d")
        citation = Citation(
            number=number,
            title=document.title or document.url,
            url=document.url,
            source=document.source,
            accessed_at=timestamp,
        )
        self._by_url[key] = citation
        self._documents.append(document)
        return citation

    def add_many(self, documents: list[Document]) -> list[Citation]:
        return [self.add(doc) for doc in documents]

    def number_for(self, url: str) -> int | None:
        citation = self._by_url.get(url.strip())
        return citation.number if citation else None

    def __len__(self) -> int:
        return len(self._by_url)

    @property
    def citations(self) -> list[Citation]:
        return sorted(self._by_url.values(), key=lambda c: c.number)

    def numbered_sources_block(self) -> str:
        """A compact ``[n] source`` block suitable for embedding in a prompt."""
        if not self._by_url:
            return "(no sources retrieved)"
        return "\n".join(
            f"[{c.number}] {c.title} ({c.source}) - {c.url}" for c in self.citations
        )

    def references_markdown(self) -> str:
        """A Markdown references section for the final answer."""
        if not self._by_url:
            return ""
        lines = ["", "---", "### References", ""]
        for c in self.citations:
            lines.append(
                f"[{c.number}] [{c.title}]({c.url}) — *{c.source}*, accessed {c.accessed_at}"
            )
        return "\n".join(lines)

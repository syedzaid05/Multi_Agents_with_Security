"""Transport-agnostic request/response models for the LLM layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .routing import TaskType

__all__ = ["ChatMessage", "LLMRequest", "LLMResponse", "TaskType", "Usage"]


@dataclass(slots=True)
class ChatMessage:
    """A single chat turn in OpenAI wire format."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls("assistant", content)


@dataclass(slots=True)
class Usage:
    """Token accounting, when the provider reports it."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, raw: dict[str, Any] | None) -> Usage:
        raw = raw or {}
        return cls(
            prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw.get("completion_tokens", 0) or 0),
            total_tokens=int(raw.get("total_tokens", 0) or 0),
        )


@dataclass(slots=True)
class LLMRequest:
    """A logical request handed to the gateway."""

    task: TaskType | str
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass(slots=True)
class ProviderResult:
    """What a concrete :class:`~.providers.Provider` returns for one call."""

    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponse:
    """A successful gateway response, annotated with routing metadata."""

    text: str
    provider: str
    model: str
    task: str
    latency_ms: float
    usage: Usage = field(default_factory=Usage)
    attempts: int = 1
    fallback_used: bool = False
    finish_reason: str | None = None

    @property
    def is_demo(self) -> bool:
        return self.provider == "echo"

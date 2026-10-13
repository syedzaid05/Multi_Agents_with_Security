"""Shared test doubles and factories (no network, no API keys)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from multi_agent_research.citations import Document
from multi_agent_research.config import Settings
from multi_agent_research.llm.gateway import LLMGateway
from multi_agent_research.llm.models import ChatMessage, ProviderResult
from multi_agent_research.llm.providers import Provider


async def no_sleep(_: float) -> None:
    return None


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "groq_api_key": "",
        "google_api_key": "",
        "openrouter_api_key": "",
        "openai_api_key": "",
        "together_api_key": "",
        "tavily_api_key": "",
        "max_retries": 3,
        "requests_per_minute": 600,
        "retry_base_delay_seconds": 0.0,
        "retry_max_delay_seconds": 0.0,
    }
    base.update(overrides)
    return Settings(**base)


class FakeProvider(Provider):
    """A provider whose behaviour is supplied by ``handler``.

    ``handler(model, messages, kwargs)`` returns a :class:`ProviderResult` or an
    exception instance (which is raised).
    """

    def __init__(
        self,
        name: str,
        handler: Callable[[str, list[ChatMessage], dict[str, Any]], Any],
    ) -> None:
        self.name = name
        self._handler = handler
        self.calls: list[str] = []

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float | None = None,
        task: str | None = None,
    ) -> ProviderResult:
        self.calls.append(model)
        outcome = self._handler(
            model, messages, {"temperature": temperature, "max_tokens": max_tokens, "task": task}
        )
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def text_provider(name: str, text: str = "ok", model: str = "m") -> FakeProvider:
    return FakeProvider(name, lambda m, *_: ProviderResult(text=text, model=model))


def make_gateway(
    providers: Provider | list[Provider] | dict[str, Provider],
    settings: Settings,
    routes: dict[str, list[str]] | None = None,
) -> LLMGateway:
    if isinstance(providers, Provider):
        providers = {providers.name: providers}
    elif isinstance(providers, list):
        providers = {p.name: p for p in providers}
    if routes is None:
        routes = {
            task: [f"{name}:m"]
            for name in providers
            for task in ("planner", "research", "analysis", "critique", "synthesis")
        }
    return LLMGateway(providers=providers, routes=routes, settings=settings, sleep=no_sleep)


class FakeWikipediaClient:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    async def research(self, query: str, *, limit: int = 3) -> list[Document]:
        self.queries.append(query)
        return self.documents[:limit]

    async def aclose(self) -> None:
        return None


class FakeWebClient:
    name = "fake-web"

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int = 5) -> list[Document]:
        self.queries.append(query)
        return self.documents[:limit]

    async def aclose(self) -> None:
        return None


def doc(title: str, url: str, text: str = "body", source: str = "wikipedia") -> Document:
    return Document(title=title, url=url, text=text, source=source)

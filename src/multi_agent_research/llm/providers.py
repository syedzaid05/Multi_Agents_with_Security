"""LLM provider adapters.

Two concrete providers ship with the system:

* :class:`OpenAICompatibleProvider` - talks to any endpoint that implements the
  OpenAI ``/chat/completions`` schema. That single class covers Groq, Google
  Gemini, OpenRouter, Together, OpenAI itself, and local Ollama.
* :class:`EchoProvider` - a deterministic, offline provider used as a
  last-resort fallback so the pipeline is fully exercisable without keys.

Errors are normalised into :class:`ProviderError` so the gateway can decide
whether a failure is retryable and/or should trigger a fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from ..config import ProviderSpec
from .models import ChatMessage, ProviderResult, Usage

__all__ = ["EchoProvider", "OpenAICompatibleProvider", "Provider", "ProviderError"]


@dataclass
class ProviderError(Exception):
    """Normalised provider failure.

    ``retryable`` means "the same provider might succeed if we try again"
    (timeouts, connection resets, 429s, 5xx). Non-retryable errors (bad request,
    auth failure, unknown model) move straight to the next fallback.
    """

    message: str
    provider: str = "unknown"
    model: str = ""
    status_code: int | None = None
    retryable: bool = True
    kind: str = "error"

    def __str__(self) -> str:  # pragma: no cover - trivial
        parts = [f"[{self.provider}] {self.kind}: {self.message}"]
        if self.status_code is not None:
            parts.append(f"(status={self.status_code})")
        return " ".join(parts)


class Provider(ABC):
    """Interface every LLM backend implements."""

    name: str

    @abstractmethod
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
        """Run a single chat completion."""

    async def aclose(self) -> None:  # pragma: no cover - overridden where needed
        return None


class OpenAICompatibleProvider(Provider):
    """Adapter for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        spec: ProviderSpec,
        *,
        client: httpx.AsyncClient | None = None,
        default_timeout: float = 45.0,
    ) -> None:
        self.spec = spec
        self.name = spec.name
        self._client = client
        self._owns_client = client is None
        self._default_timeout = default_timeout

    @property
    def base_url(self) -> str:
        return self.spec.base_url.rstrip("/") + "/"

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {
                "Authorization": f"Bearer {self.spec.api_key}",
                "Content-Type": "application/json",
                **self.spec.default_headers,
            }
            self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers)
        return self._client

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
        client = self._ensure_client()
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = await client.post(
                "chat/completions",
                json=payload,
                timeout=timeout or self._default_timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"request timed out: {exc}",
                provider=self.name,
                model=model,
                retryable=True,
                kind="timeout",
            ) from exc
        except httpx.ConnectError as exc:
            # A refused/unreachable endpoint is almost always deterministic
            # (e.g. local Ollama not running) -> fall back immediately.
            raise ProviderError(
                f"connection failed: {exc}",
                provider=self.name,
                model=model,
                retryable=False,
                kind="connect",
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                f"transport error: {exc}",
                provider=self.name,
                model=model,
                retryable=True,
                kind="transport",
            ) from exc

        if response.status_code >= 400:
            retryable = response.status_code == 408 or response.status_code == 429 or (
                response.status_code >= 500
            )
            raise ProviderError(
                _extract_error_message(response),
                provider=self.name,
                model=model,
                status_code=response.status_code,
                retryable=retryable,
                kind="http_error",
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"invalid JSON in response: {exc}",
                provider=self.name,
                model=model,
                status_code=response.status_code,
                retryable=True,
                kind="bad_response",
            ) from exc

        return _parse_chat_completion(data, provider=self.name, model=model)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


class EchoProvider(Provider):
    """Deterministic offline provider used for demos and fallback.

    It never touches the network and produces a stable, task-shaped answer so
    the whole pipeline - guardrails, citations, critic loop - can be exercised
    without any API keys.
    """

    name = "echo"

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str = "echo-1",
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float | None = None,
        task: str | None = None,
    ) -> ProviderResult:
        task = task or "synthesis"
        prompt = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        text = _echo_response(task, prompt)
        return ProviderResult(
            text=text,
            model=f"echo/{task}",
            usage=Usage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(text.split()),
                total_tokens=len(prompt.split()) + len(text.split()),
            ),
            finish_reason="stop",
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract_error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        if data.get("message"):
            return str(data["message"])
    return str(data)[:300]


def _parse_chat_completion(
    data: dict, *, provider: str, model: str
) -> ProviderResult:
    if not isinstance(data, dict):
        raise ProviderError(
            f"unexpected response type: {type(data).__name__}",
            provider=provider,
            model=model,
            retryable=True,
            kind="bad_response",
        )
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(
            "response contained no choices",
            provider=provider,
            model=model,
            retryable=True,
            kind="bad_response",
        )
    choice = choices[0]
    message = choice.get("message") or {}
    text = message.get("content")
    if text is None:
        # Some providers return content as a list of parts.
        if isinstance(message.get("content"), list):
            text = "".join(
                part.get("text", "") for part in message["content"] if isinstance(part, dict)
            )
        else:
            text = ""
    return ProviderResult(
        text=str(text).strip(),
        model=data.get("model") or model,
        usage=Usage.from_openai(data.get("usage")),
        finish_reason=choice.get("finish_reason"),
        raw=data,
    )


def _echo_response(task: str, prompt: str) -> str:
    excerpt = prompt.strip().replace("\n", " ")
    if len(excerpt) > 240:
        excerpt = excerpt[:237] + "..."
    if task == "planner":
        return (
            "[DEMO] Plan:\n"
            "1. Gather background from Wikipedia.\n"
            "2. Identify key subtopics and tensions.\n"
            "3. Analyse evidence and critique it.\n"
            "4. Synthesise a cited answer."
        )
    if task == "research":
        return (
            "[DEMO research summary] Offline echo model generated this summary "
            "because no LLM API key is configured. Source material gathered from "
            "Wikipedia is cited inline [1][2]. Key points: (a) the topic has a "
            "well-documented history, (b) multiple perspectives exist, (c) "
            "evidence quality varies by source. Query: "
            f"{excerpt}"
        )
    if task == "analysis":
        return (
            "[DEMO analysis]\n"
            "- Main finding: the retrieved sources agree on the core definition [1].\n"
            "- Caveat: coverage of recent developments is thin [2].\n"
            "- Confidence: medium, pending additional sources."
        )
    if task == "critique":
        return (
            "[DEMO critique]\n"
            "- Issue: analysis relies on a small number of sources.\n"
            "- Issue: no quantitative comparison was provided.\n"
            "- Suggestion: cite at least one primary reference.\n"
            "- Verdict: mostly sound; add sources before publishing."
        )
    return (
        "[DEMO final answer] This response was produced entirely offline by the "
        "echo fallback model, so treat it as a placeholder. Configure a free Groq "
        "or Google Gemini key to get real answers. The retrieved sources are "
        "listed below with citations [1][2]. Question received: "
        f"{excerpt}"
    )

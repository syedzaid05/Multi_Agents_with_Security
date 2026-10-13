"""The LLM gateway.

Responsibilities:

* **Task-based routing** - pick the candidate chain for a :class:`TaskType`.
* **Timeouts** - enforce a per-request deadline at the HTTP layer.
* **Retries** - retry *retryable* failures with exponential backoff + jitter.
* **Rate limiting** - one token bucket per provider.
* **Model fallback** - on exhausted/failed candidates, move to the next model in
  the chain; remember which models were tried for observability.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..config import Settings
from ..metrics import Metrics
from .models import ChatMessage, LLMRequest, LLMResponse, ProviderResult, TaskType
from .providers import Provider, ProviderError
from .ratelimit import RateLimiter
from .routing import parse_route

__all__ = ["AllProvidersFailed", "AttemptRecord", "LLMGateway"]

SleepFn = Callable[[float], Awaitable[None]]


@dataclass(slots=True)
class AttemptRecord:
    """One provider/model try, recorded for metrics and error reporting."""

    route: str
    provider: str
    model: str
    ok: bool
    attempts: int = 1
    latency_ms: float = 0.0
    error: str | None = None
    retryable: bool | None = None


class AllProvidersFailed(Exception):
    """Raised when every candidate in a task's chain has been exhausted."""

    def __init__(self, task: str, records: list[AttemptRecord]) -> None:
        self.task = task
        self.records = records
        detail = "; ".join(
            f"{r.route} -> {r.error or 'failed'} ({r.attempts} attempt(s))"
            for r in records
            if not r.ok
        )
        super().__init__(f"All providers failed for task '{task}'. Tried: {detail or 'none'}")


@dataclass
class LLMGateway:
    """Routes logical tasks to concrete provider/model pairs."""

    providers: dict[str, Provider]
    routes: dict[str, list[str]]
    settings: Settings
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("mar.llm"))
    metrics: Metrics | None = None
    sleep: SleepFn = asyncio.sleep

    def __post_init__(self) -> None:
        self._rng = random.Random(0xC0FFEE)
        self._rate_limiters: dict[str, RateLimiter] = {
            name: RateLimiter(self.settings.requests_per_minute) for name in self.providers
        }
        # Providers that returned a connection error are considered down for the
        # lifetime of this gateway, so we don't pay the connect penalty again on
        # every stage (e.g. local Ollama not running during a demo run).
        self._unavailable: set[str] = set()

    # -- public API ---------------------------------------------------------
    async def complete(
        self,
        task: TaskType | str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float | None = None,
        routes: list[str] | None = None,
        raise_on_empty: bool = True,
    ) -> LLMResponse:
        """Run a task through its fallback chain and return the first success."""
        task_name = str(getattr(task, "value", task))
        request = LLMRequest(
            task=task_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        chain = routes or self.routes.get(task_name) or self.routes.get("synthesis", [])
        deadline = timeout if timeout is not None else self.settings.request_timeout_seconds
        first_route = chain[0] if chain else None

        records: list[AttemptRecord] = []
        for route in chain:
            try:
                provider_name, model = parse_route(route)
            except ValueError as exc:
                self.logger.warning("Skipping malformed route %r: %s", route, exc)
                records.append(
                    AttemptRecord(route, "?", "?", ok=False, error=str(exc), retryable=False)
                )
                continue

            provider = self.providers.get(provider_name)
            if provider is None:
                self.logger.debug("Provider %r not configured; skipping %s", provider_name, route)
                records.append(
                    AttemptRecord(
                        route,
                        provider_name,
                        model,
                        ok=False,
                        error="provider not configured",
                        retryable=False,
                    )
                )
                continue
            if provider_name in self._unavailable:
                records.append(
                    AttemptRecord(
                        route,
                        provider_name,
                        model,
                        ok=False,
                        error="provider marked unavailable (earlier connection error)",
                        retryable=False,
                    )
                )
                continue

            record, result = await self._run_candidate(
                provider, route, provider_name, model, request, deadline
            )
            records.append(record)
            if record.ok and result is not None:
                return LLMResponse(
                    text=result.text,
                    provider=provider_name,
                    model=result.model or model,
                    task=task_name,
                    latency_ms=record.latency_ms,
                    usage=result.usage,
                    attempts=record.attempts,
                    fallback_used=route != first_route,
                    finish_reason=result.finish_reason,
                )

        if self.metrics is not None:
            self.metrics.increment("llm.all_failed")
        if raise_on_empty:
            raise AllProvidersFailed(task_name, records)
        return LLMResponse(
            text="",
            provider="none",
            model="none",
            task=task_name,
            latency_ms=0.0,
            attempts=len(records),
        )

    async def _run_candidate(
        self,
        provider: Provider,
        route: str,
        provider_name: str,
        model: str,
        request: LLMRequest,
        deadline: float,
    ) -> tuple[AttemptRecord, ProviderResult | None]:
        """Try one provider/model with retries; return a record either way."""
        max_attempts = max(1, int(self.settings.max_retries))
        limiter = self._rate_limiters.get(provider_name)
        last_error: ProviderError | None = None
        started = time.perf_counter()
        attempts_made = 0

        for attempt in range(1, max_attempts + 1):
            attempts_made = attempt
            if limiter is not None:
                await limiter.acquire()
            try:
                result = await provider.chat(
                    request.messages,
                    model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    timeout=deadline,
                    task=request.task,
                )
            except ProviderError as exc:
                last_error = exc
                self.logger.warning(
                    "provider %s model %s attempt %d/%d failed: %s",
                    provider_name,
                    model,
                    attempt,
                    max_attempts,
                    exc,
                )
                if self.metrics is not None:
                    self.metrics.increment(f"llm.retry.{provider_name}")
                if exc.kind == "connect":
                    self._unavailable.add(provider_name)
                if not exc.retryable or attempt >= max_attempts:
                    break
                await self._backoff(attempt)
                continue
            except Exception as exc:  # noqa: BLE001 - normalise unexpected failures
                last_error = ProviderError(
                    f"unexpected error: {exc}",
                    provider=provider_name,
                    model=model,
                    retryable=False,
                    kind="unexpected",
                )
                break
            else:
                total_latency = (time.perf_counter() - started) * 1000.0
                record = AttemptRecord(
                    route=route,
                    provider=provider_name,
                    model=result.model or model,
                    ok=True,
                    attempts=attempt,
                    latency_ms=total_latency,
                )
                if self.metrics is not None:
                    self.metrics.increment(f"llm.success.{provider_name}")
                    self.metrics.observe(f"llm.latency_ms.{request.task}", total_latency)
                self.logger.info(
                    "task=%s provider=%s model=%s latency_ms=%.1f attempts=%d",
                    request.task,
                    provider_name,
                    result.model or model,
                    total_latency,
                    attempt,
                )
                return record, result

        total_latency = (time.perf_counter() - started) * 1000.0
        return (
            AttemptRecord(
                route=route,
                provider=provider_name,
                model=model,
                ok=False,
                attempts=attempts_made,
                latency_ms=total_latency,
                error=str(last_error) if last_error else "unknown error",
                retryable=last_error.retryable if last_error else None,
            ),
            None,
        )

    async def _backoff(self, attempt: int) -> None:
        base = self.settings.retry_base_delay_seconds
        cap = self.settings.retry_max_delay_seconds
        delay = min(cap, base * (2 ** (attempt - 1)))
        jitter = delay * 0.25 * (self._rng.random() * 2 - 1)
        await self.sleep(max(0.0, delay + jitter))

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()

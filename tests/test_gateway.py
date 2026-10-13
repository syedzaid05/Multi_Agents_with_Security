from __future__ import annotations

import pytest
from helpers import FakeProvider, make_gateway, make_settings

from multi_agent_research.llm.gateway import AllProvidersFailed, LLMGateway
from multi_agent_research.llm.models import ChatMessage, ProviderResult
from multi_agent_research.llm.providers import ProviderError
from multi_agent_research.metrics import Metrics

MESSAGES = [ChatMessage.user("hello")]


async def test_successful_first_provider(test_settings):
    provider = FakeProvider("groq", lambda m, *_: ProviderResult(text="answer", model=m))
    gateway = make_gateway(provider, test_settings)
    response = await gateway.complete("analysis", MESSAGES)
    assert response.text == "answer"
    assert response.provider == "groq"
    assert response.fallback_used is False
    assert response.attempts == 1


async def test_fallback_to_next_model(test_settings):
    primary = FakeProvider(
        "groq", lambda *_: ProviderError("boom", provider="groq", retryable=True)
    )
    secondary = FakeProvider("google", lambda m, *_: ProviderResult(text="ok", model=m))
    gateway = make_gateway(
        [primary, secondary],
        test_settings,
        routes={"analysis": ["groq:g1", "google:g2"]},
    )
    response = await gateway.complete("analysis", MESSAGES)
    assert response.text == "ok"
    assert response.provider == "google"
    assert response.fallback_used is True
    # retryable failure -> retried max_retries (3) times before fallback
    assert len(primary.calls) == 3


async def test_retry_then_success(test_settings):
    state = {"n": 0}

    def handler(model, *_):
        state["n"] += 1
        if state["n"] == 1:
            return ProviderError("transient", provider="groq", retryable=True)
        return ProviderResult(text="recovered", model=model)

    provider = FakeProvider("groq", handler)
    gateway = make_gateway(provider, test_settings, routes={"analysis": ["groq:g1"]})
    response = await gateway.complete("analysis", MESSAGES)
    assert response.text == "recovered"
    assert response.attempts == 2
    assert len(provider.calls) == 2


async def test_non_retryable_fails_over_immediately(test_settings):
    primary = FakeProvider(
        "groq", lambda *_: ProviderError("bad model", provider="groq", retryable=False)
    )
    secondary = FakeProvider("google", lambda m, *_: ProviderResult(text="ok", model=m))
    gateway = make_gateway(
        [primary, secondary],
        test_settings,
        routes={"analysis": ["groq:g1", "google:g2"]},
    )
    response = await gateway.complete("analysis", MESSAGES)
    assert response.provider == "google"
    assert len(primary.calls) == 1


async def test_skips_unconfigured_provider(test_settings):
    provider = FakeProvider("echo", lambda m, *_: ProviderResult(text="ok", model=m))
    gateway = make_gateway(
        provider, test_settings, routes={"analysis": ["openrouter:missing", "echo:e"]}
    )
    response = await gateway.complete("analysis", MESSAGES)
    assert response.provider == "echo"
    # openrouter was never called (unconfigured), echo handled it
    assert provider.calls == ["e"]


async def test_all_providers_failed(test_settings):
    settings = make_settings(max_retries=1)
    provider = FakeProvider(
        "groq", lambda *_: ProviderError("down", provider="groq", retryable=True)
    )
    gateway = make_gateway(provider, settings, routes={"analysis": ["groq:g1"]})
    with pytest.raises(AllProvidersFailed) as excinfo:
        await gateway.complete("analysis", MESSAGES)
    assert excinfo.value.task == "analysis"
    assert len(excinfo.value.records) == 1


async def test_malformed_route_is_skipped(test_settings):
    provider = FakeProvider("echo", lambda m, *_: ProviderResult(text="ok", model=m))
    gateway = make_gateway(provider, test_settings, routes={"analysis": ["bad", "echo:e"]})
    response = await gateway.complete("analysis", MESSAGES)
    assert response.provider == "echo"


async def test_metrics_recorded(test_settings):
    provider = FakeProvider("echo", lambda m, *_: ProviderResult(text="ok", model=m))
    metrics = Metrics()
    gateway = make_gateway(provider, test_settings)
    gateway.metrics = metrics
    await gateway.complete("analysis", MESSAGES)
    assert metrics.counters.get("llm.success.echo") == 1
    assert "llm.latency_ms.analysis" in metrics.observations


async def test_echo_provider_demo_shape():
    from multi_agent_research.llm.providers import EchoProvider

    result = await EchoProvider().chat(MESSAGES, "echo-1", task="analysis")
    assert "DEMO" in result.text
    assert result.usage.total_tokens > 0


def test_gateway_is_dataclass_constructible(test_settings):
    gateway = LLMGateway(providers={}, routes={}, settings=test_settings)
    assert isinstance(gateway, LLMGateway)

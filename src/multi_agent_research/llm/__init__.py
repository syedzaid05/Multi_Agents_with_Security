"""LLM layer: models, providers, rate limiting, routing and the gateway."""

from __future__ import annotations

from .gateway import AllProvidersFailed, LLMGateway
from .models import ChatMessage, LLMRequest, LLMResponse, TaskType
from .providers import EchoProvider, OpenAICompatibleProvider, Provider, ProviderError
from .ratelimit import RateLimiter
from .routing import DEFAULT_ROUTES, parse_route

__all__ = [
    "AllProvidersFailed",
    "ChatMessage",
    "DEFAULT_ROUTES",
    "EchoProvider",
    "LLMGateway",
    "LLMRequest",
    "LLMResponse",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "RateLimiter",
    "TaskType",
    "parse_route",
]

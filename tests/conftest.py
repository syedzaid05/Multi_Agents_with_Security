"""Pytest fixtures shared across the suite."""

from __future__ import annotations

import pytest
from helpers import make_gateway, make_settings

from multi_agent_research.llm.gateway import LLMGateway
from multi_agent_research.llm.providers import EchoProvider


@pytest.fixture
def test_settings():
    return make_settings()


@pytest.fixture
def echo_gateway(test_settings) -> LLMGateway:
    """Gateway backed only by the offline echo provider (demo mode)."""
    return make_gateway({"echo": EchoProvider()}, test_settings, routes=test_settings.routes)

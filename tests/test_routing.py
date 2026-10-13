from __future__ import annotations

import pytest

from multi_agent_research.llm.routing import TaskType, parse_route


def test_parse_route_simple():
    assert parse_route("groq:llama-3.1-8b-instant") == ("groq", "llama-3.1-8b-instant")


def test_parse_route_keeps_colons_in_model():
    provider, model = parse_route("openrouter:meta-llama/llama-3.3-70b-instruct:free")
    assert provider == "openrouter"
    assert model == "meta-llama/llama-3.3-70b-instruct:free"


@pytest.mark.parametrize("bad", ["", "colonless", ":model", "provider:"])
def test_parse_route_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_route(bad)


def test_task_types_are_strings():
    assert str(TaskType.ANALYSIS) == "analysis"
    assert TaskType.PLANNER.value == "planner"

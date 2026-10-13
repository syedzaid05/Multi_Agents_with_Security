from __future__ import annotations

from helpers import make_settings

from multi_agent_research.config import (
    build_provider_specs,
    mask_secret,
    public_config_summary,
)


def test_default_routes_cover_all_tasks(test_settings):
    routes = test_settings.routes
    for task in ("planner", "research", "analysis", "critique", "synthesis"):
        assert task in routes
        assert routes[task], f"{task} has an empty chain"


def test_echo_is_last_resort_in_default_routes(test_settings):
    for chain in test_settings.routes.values():
        assert chain[-1].startswith("echo:")


def test_routes_json_override_merges():
    settings = make_settings(llm_routes_json='{"analysis": ["groq:custom-model"]}')
    assert settings.routes["analysis"] == ["groq:custom-model"]
    # untouched tasks keep their defaults
    assert settings.routes["planner"]


def test_invalid_routes_json_falls_back():
    settings = make_settings(llm_routes_json="{not valid json")
    assert settings.routes["analysis"]  # defaults survive


def test_demo_mode_when_no_keys():
    assert make_settings().is_demo is True
    assert make_settings(groq_api_key="abc").is_demo is False


def test_provider_specs_only_configured():
    settings = make_settings(groq_api_key="gsk-test")
    specs = build_provider_specs(settings)
    assert "groq" in specs
    assert "openrouter" not in specs
    assert specs["groq"].configured is True


def test_mask_secret():
    assert mask_secret("") == ""
    assert mask_secret("abcdefgh", keep=4) == "abcd****"
    assert mask_secret("abc", keep=4) == "***"


def test_public_config_summary_redacts():
    settings = make_settings(groq_api_key="secret-key-123")
    summary = public_config_summary(settings)
    assert summary["providers"]["groq"]["api_key"].startswith("secr")
    assert "secret-key-123" not in str(summary)

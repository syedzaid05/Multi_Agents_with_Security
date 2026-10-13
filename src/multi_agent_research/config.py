"""Application configuration.

All secrets and runtime knobs live in environment variables / ``.env``. This
module centralises them behind a single cached :class:`Settings` object and
exposes the provider + routing registries the LLM gateway consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view over the process environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Provider credentials -------------------------------------------------
    groq_api_key: str = ""
    google_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    together_api_key: str = ""
    tavily_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"

    # --- Behaviour ------------------------------------------------------------
    web_search_backend: str = "duckduckgo"  # "duckduckgo" | "tavily" | "none"
    log_level: str = "INFO"
    log_json: bool = True

    # Gateway policy
    request_timeout_seconds: float = 45.0
    max_retries: int = 3
    requests_per_minute: int = 30
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 8.0

    # Guardrails
    max_input_chars: int = 4000
    max_output_chars: int = 12000

    # Optional JSON override for routing: {"analysis": ["groq:llama-3.3-70b-versatile", ...]}
    llm_routes_json: str | None = None

    # Demo mode: when true (or when no provider keys are present) the gateway
    # includes a deterministic offline "echo" model as a last-resort fallback.
    demo_mode: bool | None = None

    # --- Derived helpers ------------------------------------------------------
    @field_validator("web_search_backend")
    @classmethod
    def _normalise_backend(cls, value: str) -> str:
        return (value or "none").strip().lower()

    @property
    def routes(self) -> dict[str, list[str]]:
        """Task -> ordered list of ``provider:model`` candidates (fallback chain)."""
        # Imported lazily to avoid a circular import (llm depends on config).
        from .llm.routing import DEFAULT_ROUTES

        if self.llm_routes_json:
            try:
                parsed = json.loads(self.llm_routes_json)
                if isinstance(parsed, dict) and all(
                    isinstance(v, list) for v in parsed.values()
                ):
                    merged = {k: list(v) for k, v in DEFAULT_ROUTES.items()}
                    merged.update({k: [str(m) for m in v] for k, v in parsed.items()})
                    return merged
            except json.JSONDecodeError:
                # Fall through to defaults rather than crash on a bad override.
                pass
        return {task: list(chain) for task, chain in DEFAULT_ROUTES.items()}

    @property
    def has_any_provider_key(self) -> bool:
        return any(
            [
                self.groq_api_key,
                self.google_api_key,
                self.openrouter_api_key,
                self.openai_api_key,
                self.together_api_key,
            ]
        )

    @property
    def is_demo(self) -> bool:
        if self.demo_mode is not None:
            return self.demo_mode
        return not self.has_any_provider_key


@dataclass(slots=True)
class ProviderSpec:
    """Static description of an OpenAI-compatible provider endpoint."""

    name: str
    base_url: str
    api_key: str
    default_headers: dict[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        """A provider is usable when it has a key, or is a local/echo endpoint."""
        if self.name in {"echo", "ollama"}:
            return True
        return bool(self.api_key.strip())


def build_provider_specs(settings: Settings) -> dict[str, ProviderSpec]:
    """Materialise the provider registry from settings.

    Only providers with credentials (plus the free local ``ollama`` and the
    offline ``echo`` provider) end up in the registry; the gateway then skips
    anything that is not present when walking a fallback chain.
    """
    specs: dict[str, ProviderSpec] = {
        "groq": ProviderSpec(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
        ),
        "google": ProviderSpec(
            name="google",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=settings.google_api_key,
        ),
        "openrouter": ProviderSpec(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            default_headers={
                # OpenRouter recommends these attribution headers.
                "HTTP-Referer": "https://github.com/multi-agent-research",
                "X-Title": "Multi-Agent Research System",
            },
        ),
        "openai": ProviderSpec(
            name="openai",
            base_url=settings.openai_base_url.rstrip("/"),
            api_key=settings.openai_api_key,
        ),
        "together": ProviderSpec(
            name="together",
            base_url="https://api.together.xyz/v1",
            api_key=settings.together_api_key,
        ),
        "ollama": ProviderSpec(
            name="ollama",
            base_url=settings.ollama_base_url.rstrip("/"),
            api_key=settings.ollama_api_key or "ollama",
        ),
    }
    return {name: spec for name, spec in specs.items() if spec.configured}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests)."""
    get_settings.cache_clear()


def mask_secret(value: str, *, keep: int = 4) -> str:
    """Return a safe-to-log representation of a secret."""
    if not value:
        return ""
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * (len(value) - keep)}"


def public_config_summary(settings: Settings) -> dict[str, Any]:
    """A redacted snapshot of configuration for logging / the UI."""
    return {
        "providers": {
            name: {"base_url": spec.base_url, "api_key": mask_secret(spec.api_key)}
            for name, spec in build_provider_specs(settings).items()
        },
        "demo_mode": settings.is_demo,
        "web_search_backend": settings.web_search_backend,
        "request_timeout_seconds": settings.request_timeout_seconds,
        "max_retries": settings.max_retries,
        "requests_per_minute": settings.requests_per_minute,
    }

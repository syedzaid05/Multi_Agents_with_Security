"""Task-based LLM routing table.

Each task maps to an *ordered* list of ``provider:model`` candidates. The
gateway walks the list left-to-right, so the first entry is the primary model
and the rest are fallbacks. Free-tier providers come first; ``ollama`` (local,
free) and ``echo`` (offline, deterministic) close the chain so the system never
hard-fails during a demo.

Provider names must match those in :mod:`multi_agent_research.config`.
Model ids may themselves contain ``:`` (e.g. OpenRouter's ``...:free`` suffix),
so the gateway only splits on the *first* colon.
"""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    """Logical LLM tasks; used as routing keys and metrics labels."""

    PLANNER = "planner"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"


# --- Individual model candidates ---------------------------------------------
# Free / local first, then a guaranteed-offline fallback.
_GROQ_FAST = "groq:llama-3.1-8b-instant"
_GROQ_SMART = "groq:llama-3.3-70b-versatile"
_GEMINI = "google:gemini-3.5-flash-lite"
_GEMINI_FALLBACK = "google:gemini-3.1-flash-lite"
_OPENROUTER_FAST = "openrouter:meta-llama/llama-3.1-8b-instruct:free"
_OPENROUTER_SMART = "openrouter:meta-llama/llama-3.3-70b-instruct:free"
_OLLAMA = "ollama:llama3.2"
_ECHO = "echo:echo-1"

_SMART_CHAIN = [
    _GROQ_SMART,
    _GEMINI,
    _GEMINI_FALLBACK,
    _OPENROUTER_SMART,
    _OLLAMA,
    _ECHO,
]

DEFAULT_ROUTES: dict[str, list[str]] = {
    # Cheap, latency-sensitive intent/planning step.
    TaskType.PLANNER.value: [
        _GROQ_FAST,
        _GEMINI,
        _GEMINI_FALLBACK,
        _OPENROUTER_FAST,
        _OLLAMA,
        _ECHO,
    ],
    # Summarise retrieved source material.
    TaskType.RESEARCH.value: _SMART_CHAIN,
    # Structured analysis of the gathered evidence.
    TaskType.ANALYSIS.value: _SMART_CHAIN,
    # Adversarial review of the analysis.
    TaskType.CRITIQUE.value: _SMART_CHAIN,
    # Final user-facing answer.
    TaskType.SYNTHESIS.value: _SMART_CHAIN,
}


def parse_route(route: str) -> tuple[str, str]:
    """Split ``"provider:model"`` on the **first** colon.

    >>> parse_route("openrouter:meta-llama/llama-3.3-70b-instruct:free")
    ('openrouter', 'meta-llama/llama-3.3-70b-instruct:free')
    """
    provider, _, model = route.partition(":")
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        raise ValueError(f"Malformed route entry: {route!r} (expected 'provider:model')")
    return provider, model

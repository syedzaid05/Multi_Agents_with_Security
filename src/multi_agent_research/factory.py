"""Composition root: assemble the whole system from settings.

Keeping construction here (rather than scattered across modules) makes the
system easy to build in tests, in the CLI, and in the Streamlit app alike, and
gives a single place to close all network clients.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agents import (
    AnalysisAgent,
    CriticAgent,
    PlannerAgent,
    SynthesisAgent,
    WebSearchAgent,
    WikipediaAgent,
)
from .config import Settings, build_provider_specs, get_settings
from .guardrails import InputGuard, OutputGuard
from .llm.gateway import LLMGateway
from .llm.providers import EchoProvider, OpenAICompatibleProvider, Provider
from .logging_config import get_logger
from .orchestration.pipeline import ResearchPipeline
from .tools.web_search import WebSearchClient, build_web_search_client
from .tools.wikipedia import WikipediaClient

__all__ = ["ResearchSystem", "build_gateway", "build_providers", "build_system"]


def build_providers(settings: Settings) -> dict[str, Provider]:
    """Instantiate every configured provider plus the offline echo fallback."""
    providers: dict[str, Provider] = {}
    for name, spec in build_provider_specs(settings).items():
        providers[name] = OpenAICompatibleProvider(
            spec, default_timeout=settings.request_timeout_seconds
        )
    providers["echo"] = EchoProvider()
    return providers


def build_gateway(settings: Settings) -> LLMGateway:
    logger = get_logger("llm")
    return LLMGateway(
        providers=build_providers(settings),
        routes=settings.routes,
        settings=settings,
        logger=logger,
    )


@dataclass
class ResearchSystem:
    """A fully wired system plus handles for cleanup."""

    settings: Settings
    gateway: LLMGateway
    wikipedia_client: WikipediaClient
    web_client: WebSearchClient | None
    pipeline: ResearchPipeline

    async def aclose(self) -> None:
        await self.gateway.aclose()
        await self.wikipedia_client.aclose()
        if self.web_client is not None:
            await self.web_client.aclose()


def build_system(
    settings: Settings | None = None,
    *,
    enable_web: bool = True,
) -> ResearchSystem:
    """Build the complete research system."""
    settings = settings or get_settings()
    logger = get_logger("pipeline")
    gateway = build_gateway(settings)
    wikipedia_client = WikipediaClient()
    web_client = build_web_search_client(settings) if enable_web else None

    pipeline = ResearchPipeline(
        gateway=gateway,
        wikipedia_agent=WikipediaAgent(
            gateway, wikipedia_client, logger=get_logger("agent.wikipedia")
        ),
        web_search_agent=WebSearchAgent(gateway, web_client, logger=get_logger("agent.web")),
        analysis_agent=AnalysisAgent(gateway, logger=get_logger("agent.analysis")),
        critic_agent=CriticAgent(gateway, logger=get_logger("agent.critic")),
        synthesis_agent=SynthesisAgent(gateway, logger=get_logger("agent.synthesis")),
        planner_agent=PlannerAgent(gateway, logger=get_logger("agent.planner")),
        settings=settings,
        logger=logger,
        input_guard=InputGuard(max_chars=settings.max_input_chars),
        output_guard=OutputGuard(max_chars=settings.max_output_chars),
    )

    return ResearchSystem(
        settings=settings,
        gateway=gateway,
        wikipedia_client=wikipedia_client,
        web_client=web_client,
        pipeline=pipeline,
    )

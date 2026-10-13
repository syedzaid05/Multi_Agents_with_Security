"""The research pipeline that coordinates all agents.

Flow:

    input guardrail -> planner -> Wikipedia agent -> (optional) web agent
        -> analysis agent -> critic agent -> synthesis agent -> output guardrail

Every stage is wrapped in a metrics span; a caller-supplied ``progress``
callback is invoked after each stage so a UI can stream status.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from ..agents import (
    AnalysisAgent,
    CriticAgent,
    PlannerAgent,
    SynthesisAgent,
    WebSearchAgent,
    WikipediaAgent,
)
from ..guardrails import GuardrailResult, InputGuard, OutputGuard
from ..llm.gateway import AllProvidersFailed, LLMGateway
from ..metrics import Metrics
from ..state import ResearchState

__all__ = ["PipelineResult", "ResearchPipeline"]

ProgressFn = Callable[[str, str], None]


@dataclass
class PipelineResult:
    state: ResearchState
    input_guard: GuardrailResult
    output_guard: GuardrailResult | None
    metrics: Metrics
    answer_markdown: str = ""

    @property
    def blocked(self) -> bool:
        return self.input_guard.blocked

    @property
    def references(self) -> str:
        return self.state.citations.references_markdown()


@dataclass
class ResearchPipeline:
    gateway: LLMGateway
    wikipedia_agent: WikipediaAgent
    web_search_agent: WebSearchAgent
    analysis_agent: AnalysisAgent
    critic_agent: CriticAgent
    synthesis_agent: SynthesisAgent
    planner_agent: PlannerAgent
    settings: object  # Settings; typed loosely to avoid an import cycle
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("mar.pipeline"))
    input_guard: InputGuard = field(default_factory=InputGuard)
    output_guard: OutputGuard = field(default_factory=OutputGuard)

    async def run(
        self,
        query: str,
        *,
        enable_web: bool = False,
        progress: ProgressFn | None = None,
    ) -> PipelineResult:
        metrics = Metrics()
        input_guard = self.input_guard
        output_guard = self.output_guard

        def emit(stage: str, message: str) -> None:
            if progress is not None:
                progress(stage, message)

        # --- input guardrail ------------------------------------------------
        with metrics.span("guardrail.input"):
            checked = input_guard.check(query)
        for violation in checked.violations:
            self.logger.warning("input guardrail violation: %s", violation.as_dict())
        if checked.blocked:
            state = ResearchState(query=query)
            state.emit("guardrail", "Query blocked by input guardrails", level="error")
            reason = checked.details.get("blocked_reason") or "invalid input"
            answer = (
                "Your query was blocked by the input guardrails "
                f"({reason}). Please rephrase it as a normal research question."
            )
            state.final_answer = answer
            emit("guardrail", "Input blocked")
            return PipelineResult(
                state=state,
                input_guard=checked,
                output_guard=None,
                metrics=metrics,
                answer_markdown=answer,
            )

        state = ResearchState(query=checked.text)

        # --- research --------------------------------------------------------
        try:
            with metrics.span("agent.planner"):
                await self.planner_agent.run(state)
            emit("planner", state.events[-1].message)

            with metrics.span("agent.wikipedia"):
                await self.wikipedia_agent.run(state)
            emit("wikipedia", _last_message(state, "wikipedia"))

            if enable_web and self.web_search_agent.enabled:
                with metrics.span("agent.web_search"):
                    await self.web_search_agent.run(state)
                emit("web_search", _last_message(state, "web_search"))
            elif enable_web:
                state.emit("web_search", "Web search requested but no backend available", "warning")

            # --- reasoning ---------------------------------------------------
            with metrics.span("agent.analysis"):
                await self.analysis_agent.run(state)
            emit("analysis", "Analysis complete")

            with metrics.span("agent.critic"):
                await self.critic_agent.run(state)
            emit("critic", "Critique complete")

            with metrics.span("agent.synthesis"):
                await self.synthesis_agent.run(state)
            emit("synthesis", "Final answer ready")
        except AllProvidersFailed as exc:
            self.logger.error("all providers failed: %s", exc)
            state.emit("error", str(exc), level="error")
            state.final_answer = (
                "The system could not reach any LLM provider. Check your API keys "
                "and network, then try again. Details: " + str(exc)
            )

        # --- output guardrail -----------------------------------------------
        with metrics.span("guardrail.output"):
            out = output_guard.check(state.final_answer)
        for violation in out.violations:
            self.logger.warning("output guardrail violation: %s", violation.as_dict())
        if out.blocked:
            state.emit("guardrail", "Final answer blocked by output guardrails", level="error")
            state.final_answer = (
                "The generated answer was withheld by the output guardrails "
                f"({out.details.get('blocked_reason')}). Please try a different query."
            )
        else:
            state.final_answer = out.text

        references = state.citations.references_markdown()
        answer_markdown = state.final_answer + (references if references else "")
        emit("done", "Pipeline finished")

        self.logger.info("pipeline finished", extra={"metrics": metrics.summary()})
        return PipelineResult(
            state=state,
            input_guard=checked,
            output_guard=out,
            metrics=metrics,
            answer_markdown=answer_markdown,
        )


def _last_message(state: ResearchState, stage: str) -> str:
    for event in reversed(state.events):
        if event.stage == stage:
            return event.message
    return ""

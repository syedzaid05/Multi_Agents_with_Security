"""Analysis agent: structure the gathered evidence into an analysis."""

from __future__ import annotations

from ..llm.routing import TaskType
from ..state import ResearchState
from .base import SAFETY_PREAMBLE, Agent

__all__ = ["AnalysisAgent"]

_SYSTEM = (
    "You are an analytical researcher. Using ONLY the supplied research brief "
    "and numbered sources, produce a structured analysis:\n"
    "- Key findings (bullet points, each citing [n]).\n"
    "- Points of agreement and disagreement across sources.\n"
    "- Explicit gaps or uncertainties.\n"
    "Do not fabricate facts. Keep citations attached to the claims they "
    "support.\n\n" + SAFETY_PREAMBLE
)


class AnalysisAgent(Agent):
    name = "analysis"
    task = TaskType.ANALYSIS.value

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("max_tokens", 1200)
        super().__init__(*args, **kwargs)

    async def run(self, state: ResearchState) -> ResearchState:
        research = state.combined_research or "(no research was gathered)"
        user = (
            f"Research question:\n{state.query}\n\n"
            f"Numbered sources:\n{state.source_block()}\n\n"
            f"Research brief:\n{research}\n\n"
            "Produce the structured analysis now."
        )
        response = await self._call(TaskType.ANALYSIS.value, _SYSTEM, user)
        state.record_llm(response)
        state.analysis = response.text
        state.emit("analysis", "Analysis complete")
        return state

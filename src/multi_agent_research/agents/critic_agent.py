"""Critic agent: adversarially review the analysis against the sources."""

from __future__ import annotations

from ..llm.routing import TaskType
from ..state import ResearchState
from .base import SAFETY_PREAMBLE, Agent

__all__ = ["CriticAgent"]

_SYSTEM = (
    "You are a skeptical fact-checker and critic. Review the analysis against "
    "the numbered sources and the research brief. Identify:\n"
    "- Unsupported or overreaching claims (say which).\n"
    "- Logical gaps, bias, or missing counter-evidence.\n"
    "- Citation problems (claims without [n], or [n] that don't match).\n"
    "End with a one-line verdict: SOUND / NEEDS REVISION, and concrete fixes. "
    "Be concise and specific; do not rewrite the analysis.\n\n" + SAFETY_PREAMBLE
)


class CriticAgent(Agent):
    name = "critic"
    task = TaskType.CRITIQUE.value

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("max_tokens", 800)
        super().__init__(*args, **kwargs)

    async def run(self, state: ResearchState) -> ResearchState:
        user = (
            f"Research question:\n{state.query}\n\n"
            f"Numbered sources:\n{state.source_block()}\n\n"
            f"Research brief:\n{state.combined_research or '(none)'}\n\n"
            f"Analysis under review:\n{state.analysis or '(none)'}\n\n"
            "Produce the critique now."
        )
        response = await self._call(TaskType.CRITIQUE.value, _SYSTEM, user)
        state.record_llm(response)
        state.critique = response.text
        state.emit("critic", "Critique complete")
        return state

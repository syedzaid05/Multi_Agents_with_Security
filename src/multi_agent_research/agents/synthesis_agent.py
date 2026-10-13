"""Synthesis agent: produce the final cited answer, folding in the critique."""

from __future__ import annotations

from ..llm.routing import TaskType
from ..state import ResearchState
from .base import SAFETY_PREAMBLE, Agent

__all__ = ["SynthesisAgent"]

_SYSTEM = (
    "You are a careful research writer. Produce a clear, well-organised final "
    "answer to the question. Rules:\n"
    "- Ground every factual claim in the sources using inline citations [n].\n"
    "- Address the critic's points; do not repeat flagged errors.\n"
    "- If evidence is insufficient or uncertain, say so plainly.\n"
    "- Use short paragraphs and bullet points; no filler.\n\n" + SAFETY_PREAMBLE
)


class SynthesisAgent(Agent):
    name = "synthesis"
    task = TaskType.SYNTHESIS.value

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("max_tokens", 1600)
        super().__init__(*args, **kwargs)

    async def run(self, state: ResearchState) -> ResearchState:
        user = (
            f"Research question:\n{state.query}\n\n"
            f"Numbered sources:\n{state.source_block()}\n\n"
            f"Research brief:\n{state.combined_research or '(none)'}\n\n"
            f"Analysis:\n{state.analysis or '(none)'}\n\n"
            f"Critique to address:\n{state.critique or '(none)'}\n\n"
            "Write the final answer now with inline [n] citations."
        )
        response = await self._call(TaskType.SYNTHESIS.value, _SYSTEM, user)
        state.record_llm(response)
        state.final_answer = response.text
        state.emit("synthesis", "Final answer ready")
        return state

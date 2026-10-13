"""Planner agent: turns a question into a few focused search queries."""

from __future__ import annotations

from ..llm.routing import TaskType
from ..state import ResearchState
from .base import SAFETY_PREAMBLE, Agent, extract_json_list

__all__ = ["PlannerAgent"]

_SYSTEM = (
    "You are a meticulous research planner. Given a question, produce 2 to 4 "
    "short, specific search queries that would gather the background needed to "
    "answer it well. Return ONLY a JSON array of strings, e.g. "
    '["query one", "query two"]. Do not add prose.\n\n' + SAFETY_PREAMBLE
)


class PlannerAgent(Agent):
    name = "planner"
    task = TaskType.PLANNER.value

    def __init__(self, *args, max_queries: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_queries = max_queries

    async def run(self, state: ResearchState) -> ResearchState:
        user = (
            f"Research question:\n{state.query}\n\n"
            f"Return a JSON array of at most {self.max_queries} search queries."
        )
        response = await self._call(
            TaskType.PLANNER.value, _SYSTEM, user, max_tokens=200, temperature=0.0
        )
        state.record_llm(response)

        queries = extract_json_list(response.text)
        if not queries:
            queries = [state.query]
        # De-duplicate while preserving order, then cap.
        seen: set[str] = set()
        plan = [q for q in queries if not (q.lower() in seen or seen.add(q.lower()))]
        state.plan = plan[: self.max_queries]
        state.emit("planner", f"Planned {len(state.plan)} search queries")
        return state

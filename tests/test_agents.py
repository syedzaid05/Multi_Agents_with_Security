from __future__ import annotations

from helpers import FakeWebClient, FakeWikipediaClient, doc

from multi_agent_research.agents import (
    AnalysisAgent,
    CriticAgent,
    PlannerAgent,
    SynthesisAgent,
    WebSearchAgent,
    WikipediaAgent,
    extract_json_list,
)
from multi_agent_research.state import ResearchState


# --- helpers -----------------------------------------------------------------
def test_extract_json_list_from_fenced_block():
    assert extract_json_list('```json\n["a", "b"]\n```') == ["a", "b"]


def test_extract_json_list_from_bare_array():
    assert extract_json_list('Here you go: ["x", "y"]') == ["x", "y"]


def test_extract_json_list_invalid():
    assert extract_json_list("no json here") == []


# --- planner -----------------------------------------------------------------
async def test_planner_falls_back_to_query(echo_gateway):
    state = ResearchState(query="What is entropy?")
    await PlannerAgent(echo_gateway).run(state)
    assert state.plan == ["What is entropy?"]


# --- wikipedia ---------------------------------------------------------------
async def test_wikipedia_agent_collects_sources(echo_gateway):
    wiki = FakeWikipediaClient(
        [doc("Alpha", "https://a"), doc("Beta", "https://b")]
    )
    state = ResearchState(query="topic")
    state.plan = ["topic"]
    await WikipediaAgent(echo_gateway, wiki).run(state)

    assert len(state.documents) == 2
    assert len(state.citations) == 2
    assert state.research_summary  # echo produced a summary
    assert wiki.queries == ["topic"]


async def test_wikipedia_agent_handles_no_results(echo_gateway):
    wiki = FakeWikipediaClient([])
    state = ResearchState(query="nothing")
    await WikipediaAgent(echo_gateway, wiki).run(state)
    assert "No Wikipedia sources" in state.research_summary
    assert len(state.documents) == 0


# --- web search --------------------------------------------------------------
async def test_web_agent_disabled(echo_gateway):
    state = ResearchState(query="topic")
    await WebSearchAgent(echo_gateway, None).run(state)
    assert state.web_summary == ""


async def test_web_agent_collects_results(echo_gateway):
    web = FakeWebClient([doc("Web A", "https://wa", source="web")])
    state = ResearchState(query="topic")
    await WebSearchAgent(echo_gateway, web).run(state)
    assert len(state.documents) == 1
    assert state.web_summary


# --- analysis / critic / synthesis -------------------------------------------
async def test_analysis_agent(echo_gateway):
    state = ResearchState(query="q")
    state.research_summary = "some research [1]"
    await AnalysisAgent(echo_gateway).run(state)
    assert state.analysis


async def test_critic_agent(echo_gateway):
    state = ResearchState(query="q")
    state.analysis = "an analysis [1]"
    await CriticAgent(echo_gateway).run(state)
    assert state.critique


async def test_synthesis_agent(echo_gateway):
    state = ResearchState(query="q")
    state.analysis = "analysis"
    state.critique = "critique"
    await SynthesisAgent(echo_gateway).run(state)
    assert state.final_answer


async def test_agents_record_llm_calls(echo_gateway):
    state = ResearchState(query="q")
    await PlannerAgent(echo_gateway).run(state)
    assert len(state.llm_calls) == 1
    assert state.llm_calls[0].is_demo is True

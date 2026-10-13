"""Streamlit user interface (root-level entrypoint).

Run with::

    uv run streamlit run app.py

This is a directly-accessible copy of
``src/multi_agent_research/app.py`` placed at the project root so it is easy
to find and edit.
"""

from __future__ import annotations

import asyncio

import streamlit as st

from multi_agent_research.config import get_settings, public_config_summary
from multi_agent_research.factory import build_system
from multi_agent_research.logging_config import configure_logging

st.set_page_config(page_title="Multi-Agent AI Research System", page_icon="🔎", layout="wide")

_SAMPLE_QUERIES = [
    "What is the current state of fusion energy research?",
    "How do large language models handle long-context reasoning?",
    "What caused the 2008 financial crisis and how was it resolved?",
]


async def _run_pipeline(query: str, *, enable_web: bool, status) -> object:
    system = build_system(enable_web=True)
    try:
        def progress(stage: str, message: str) -> None:
            status.write(f"**{stage}** — {message}")

        return await system.pipeline.run(query, enable_web=enable_web, progress=progress)
    finally:
        await system.aclose()


def _render_result(result) -> None:
    if result.blocked:
        st.error(f"Query blocked by input guardrails.\n\n{result.state.final_answer}")
        with st.expander("Guardrail details", expanded=True):
            st.json(result.input_guard.as_dict())
        return

    if result.state.llm_calls and all(call.is_demo for call in result.state.llm_calls):
        st.warning(
            "Demo mode: no LLM API key configured, so answers come from the offline "
            "echo model. Add a free Groq or Google Gemini key in `.env` for real answers."
        )

    st.markdown(result.answer_markdown)

    tabs = st.tabs(["Plan", "Sources", "Analysis", "Critique", "Guardrails", "Metrics & Logs"])

    with tabs[0]:
        for index, item in enumerate(result.state.plan, start=1):
            st.markdown(f"{index}. {item}")

    with tabs[1]:
        citations = result.state.citations.citations
        if not citations:
            st.info("No sources were retrieved.")
        for citation in citations:
            st.markdown(
                f"**[{citation.number}]** [{citation.title}]({citation.url}) "
                f"— *{citation.source}*"
            )

    with tabs[2]:
        st.markdown(result.state.analysis or "_No analysis produced._")

    with tabs[3]:
        st.markdown(result.state.critique or "_No critique produced._")

    with tabs[4]:
        st.markdown("**Input guardrail**")
        st.json(result.input_guard.as_dict())
        if result.output_guard is not None:
            st.markdown("**Output guardrail**")
            st.json(result.output_guard.as_dict())

    with tabs[5]:
        st.markdown("**Stage latencies (ms)**")
        st.json(result.metrics.stage_timings())
        st.markdown("**LLM calls**")
        st.dataframe(
            [
                {
                    "task": call.task,
                    "provider": call.provider,
                    "model": call.model,
                    "latency_ms": round(call.latency_ms, 1),
                    "attempts": call.attempts,
                    "fallback": call.fallback_used,
                    "tokens": call.usage.total_tokens,
                }
                for call in result.state.llm_calls
            ],
            use_container_width=True,
        )
        st.markdown("**Timeline**")
        st.dataframe(
            [event.as_dict() for event in result.state.events],
            use_container_width=True,
        )
        if result.metrics.counters:
            st.markdown("**Counters**")
            st.json(result.metrics.counters)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)

    st.title("🔎 Multi-Agent AI Research System")
    st.caption(
        "Planner → Wikipedia → web → analysis → critic → synthesis, with a routed "
        "LLM gateway, guardrails, citations, and latency tracking."
    )

    with st.sidebar:
        st.header("Configuration")
        summary = public_config_summary(settings)
        if summary["demo_mode"]:
            st.error("Demo mode (no API keys). Using offline echo model.")
        else:
            st.success("Live LLM providers configured.")
        st.markdown("**Providers**")
        for name, info in summary["providers"].items():
            st.markdown(f"- `{name}` — key: `{info['api_key'] or 'none'}`")

        st.divider()
        enable_web = st.checkbox(
            "Enable web-search agent",
            value=settings.web_search_backend not in {"none", ""},
            help="Uses DuckDuckGo (free) or Tavily if TAVILY_API_KEY is set.",
        )
        show_debug = st.checkbox("Show debug details", value=False)

        st.divider()
        st.markdown("**Task routing**")
        for task, chain in settings.routes.items():
            st.markdown(f"- **{task}**: " + " → ".join(f"`{c}`" for c in chain))

    query = st.text_area(
        "Research question",
        placeholder="Ask something you want researched with citations…",
        height=100,
    )

    with st.expander("Sample questions"):
        for sample in _SAMPLE_QUERIES:
            if st.button(sample, use_container_width=True):
                query = sample

    run_clicked = st.button("Run research", type="primary", disabled=not query.strip())

    if run_clicked:
        with st.status("Running research pipeline…", expanded=True) as status:
            try:
                result = asyncio.run(
                    _run_pipeline(query, enable_web=enable_web, status=status)
                )
                st.session_state["result"] = result
                status.update(label="Done", state="complete", expanded=False)
            except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
                status.update(label="Failed", state="error", expanded=True)
                st.exception(exc)

    result = st.session_state.get("result")
    if result is not None:
        _render_result(result)
        if show_debug:
            with st.expander("Raw state", expanded=False):
                st.json(
                    {
                        "query": result.state.query,
                        "documents": len(result.state.documents),
                        "events": [e.as_dict() for e in result.state.events],
                        "metrics": result.metrics.summary(),
                    }
                )


main()

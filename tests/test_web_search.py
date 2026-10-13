from __future__ import annotations

import httpx
import respx
from helpers import FakeWebClient, doc

from multi_agent_research.tools.web_search import (
    DuckDuckGoClient,
    TavilyClient,
    build_web_search_client,
)


def test_build_returns_none_when_disabled(test_settings):
    settings = test_settings.model_copy(update={"web_search_backend": "none"})
    assert build_web_search_client(settings) is None


def test_build_returns_duckduckgo_by_default(test_settings):
    settings = test_settings.model_copy(update={"web_search_backend": "duckduckgo"})
    assert isinstance(build_web_search_client(settings), DuckDuckGoClient)


def test_build_tavily_requires_key(test_settings):
    settings = test_settings.model_copy(
        update={"web_search_backend": "tavily", "tavily_api_key": ""}
    )
    assert build_web_search_client(settings) is None


@respx.mock
async def test_tavily_parses_results():
    respx.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Result A", "url": "https://a", "content": "text a", "score": 0.9},
                    {"title": "Result B", "url": "https://b", "content": "text b"},
                ]
            },
        )
    )
    client = TavilyClient("key", client=httpx.AsyncClient())
    documents = await client.search("query", limit=2)
    await client.aclose()
    assert [d.title for d in documents] == ["Result A", "Result B"]
    assert documents[0].source == "web"
    assert documents[0].score == 0.9


async def test_fake_web_client_contract():
    client = FakeWebClient([doc("A", "https://a", source="web")])
    results = await client.search("q", limit=5)
    assert results[0].url == "https://a"
    assert client.queries == ["q"]

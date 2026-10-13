from __future__ import annotations

import httpx
import pytest
import respx

from multi_agent_research.tools.wikipedia import WikipediaClient


def _summary_payload(title: str, extract: str) -> dict:
    return {
        "title": title,
        "extract": extract,
        "description": "a description",
        "content_urls": {
            "desktop": {"page": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"}
        },
    }


@respx.mock
async def test_search_titles():
    respx.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(
            200, json={"query": {"search": [{"title": "Alpha"}, {"title": "Beta"}]}}
        )
    )
    client = WikipediaClient(client=httpx.AsyncClient())
    titles = await client.search_titles("test", limit=2)
    assert titles == ["Alpha", "Beta"]
    await client.aclose()


@respx.mock
async def test_research_combines_search_and_summaries():
    respx.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(
            200, json={"query": {"search": [{"title": "Test Page"}]}}
        )
    )
    respx.get(url__regex=r"https://en\.wikipedia\.org/api/rest_v1/page/summary/.*").mock(
        return_value=httpx.Response(
            200, json=_summary_payload("Test Page", "<p>Hello</p> world")
        )
    )
    client = WikipediaClient(client=httpx.AsyncClient())
    documents = await client.research("test", limit=1)
    await client.aclose()

    assert len(documents) == 1
    document = documents[0]
    assert document.title == "Test Page"
    assert document.text == "Hello world"  # HTML stripped
    assert document.source == "wikipedia"
    assert document.url.endswith("/Test_Page")


@respx.mock
async def test_research_drops_missing_articles():
    respx.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(
            200, json={"query": {"search": [{"title": "Missing"}]}}
        )
    )
    respx.get(url__regex=r"https://en\.wikipedia\.org/api/rest_v1/page/summary/.*").mock(
        return_value=httpx.Response(404)
    )
    client = WikipediaClient(client=httpx.AsyncClient())
    documents = await client.research("test", limit=1)
    await client.aclose()
    assert documents == []


@respx.mock
async def test_search_http_error_raises():
    from multi_agent_research.tools.wikipedia import WikipediaError

    respx.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(500)
    )
    client = WikipediaClient(client=httpx.AsyncClient())
    with pytest.raises(WikipediaError):
        await client.search_titles("test")
    await client.aclose()

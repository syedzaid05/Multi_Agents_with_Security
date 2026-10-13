from __future__ import annotations

from helpers import doc

from multi_agent_research.citations import CitationStore, Document


def test_add_assigns_sequential_numbers():
    store = CitationStore()
    c1 = store.add(doc("A", "https://a"))
    c2 = store.add(doc("B", "https://b"))
    assert (c1.number, c2.number) == (1, 2)


def test_deduplicates_by_url():
    store = CitationStore()
    store.add(doc("A", "https://a"))
    again = store.add(doc("A again", "https://a"))
    assert again.number == 1
    assert len(store) == 1


def test_add_many_and_number_for():
    store = CitationStore()
    store.add_many([doc("A", "https://a"), doc("B", "https://b")])
    assert store.number_for("https://b") == 2
    assert store.number_for("https://missing") is None


def test_numbered_sources_block():
    store = CitationStore()
    store.add(doc("Alpha", "https://a", source="wikipedia"))
    block = store.numbered_sources_block()
    assert "[1] Alpha (wikipedia) - https://a" in block


def test_references_markdown():
    store = CitationStore()
    store.add(doc("Alpha", "https://a"))
    md = store.references_markdown()
    assert "### References" in md
    assert "[1] [Alpha](https://a)" in md


def test_empty_references_markdown():
    assert CitationStore().references_markdown() == ""
    assert CitationStore().numbered_sources_block() == "(no sources retrieved)"


def test_document_source_field():
    document = Document(title="T", url="u", text="x", source="web")
    assert document.source == "web"

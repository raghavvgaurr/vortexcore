import pytest
from langchain_core.documents import Document

from src.ingestion import build_chunks
from src.retrieval import _make_client, bm25_search, dense_search, hybrid_retrieval, index_chunks
from tests.conftest import make_pages

DOC_A = ["ChatGPT reached 100 million users two months after its launch in November 2022. " * 6,
         "The Dartmouth Conference in 1956 first used the term Artificial Intelligence. " * 6]
DOC_B = ["TSMC holds about 55 percent of the chip manufacturing market share worldwide. " * 6]


def build(docs: dict[str, list[str]]):
    parents, children = [], []
    for doc_id, pages in make_pages(docs):
        p, c = build_chunks(pages, document_id=doc_id)
        parents += p
        children += c
    return parents, children, index_chunks(parents, children)


def test_child_index_matches_list_position():
    _, children, index = build({"a": DOC_A})
    for position in range(len(children)):
        hit = dense_search(index, children[position].page_content, top_k=1)[0]
        assert index.children[hit["child_index"]].page_content.strip() == hit["text"]


def test_every_child_has_a_parent():
    _, children, index = build({"a": DOC_A, "b": DOC_B})
    for child in children:
        assert str(child.metadata["parent_id"]) in index.parent_lookup


def test_parent_ids_do_not_collide_across_documents():
    parents, _, _ = build({"a": DOC_A, "b": DOC_B})
    ids = [p.metadata["parent_id"] for p in parents]
    assert len(ids) == len(set(ids))


def test_empty_chunks_are_removed_by_chunking():
    pages = [Document(page_content="   \n  ", metadata={"page": 0}),
             Document(page_content="Real text about transformers. " * 5, metadata={"page": 1})]
    _, children = build_chunks(pages)
    assert all(c.page_content.strip() for c in children)


def test_indexing_an_empty_chunk_fails_loudly():
    with pytest.raises(ValueError):
        index_chunks([], [Document(page_content="  ", metadata={})])


def test_qdrant_cloud_requires_an_api_key(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://cluster.cloud.qdrant.io")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    with pytest.raises(ValueError, match="QDRANT_API_KEY is required"):
        _make_client()


def test_document_filter_applies_to_dense_bm25_and_hybrid():
    _, _, index = build({"a": DOC_A, "b": DOC_B})
    query = "TSMC chip market share"
    for search in (dense_search, bm25_search, hybrid_retrieval):
        hits = search(index, query, document_ids={"a"})
        assert hits, search.__name__
        assert all(index.children[h["child_index"]].metadata["document_id"] == "a" for h in hits)


def test_hybrid_finds_the_right_chunk():
    _, _, index = build({"a": DOC_A, "b": DOC_B})
    top = hybrid_retrieval(index, "Dartmouth Conference 1956", top_k=3)[0]
    assert "Dartmouth" in top["text"]
    assert top["retrieval_sources"] == ["dense", "bm25"] or "bm25" in top["retrieval_sources"]
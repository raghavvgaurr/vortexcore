from langchain_core.documents import Document

from src.pipeline import expand_parents
from src.retrieval import Index


def make_index():
    parent = Document(page_content="FULL PARENT TEXT", metadata={"parent_id": "parent_0", "page": 4, "source": "a.pdf"})
    return Index(children=[], parent_lookup={"parent_0": parent}, client=None, bm25=None)


def child(child_index, parent_id, score=1.0):
    return {"child_index": child_index, "parent_id": parent_id, "text": f"child {child_index}",
            "reranker_score": score, "page": 0, "source": "a.pdf"}


def test_children_of_one_parent_are_collapsed():
    out = expand_parents([child(1, "parent_0"), child(2, "parent_0")], make_index())
    assert len(out) == 1
    assert out[0]["parent_text"] == "FULL PARENT TEXT"
    assert [c["child_index"] for c in out[0]["supporting_children"]] == [1, 2]


def test_missing_parent_falls_back_to_child_text():
    out = expand_parents([child(5, "no_such_parent")], make_index())
    assert out[0]["parent_found"] is False
    assert out[0]["parent_text"] == "child 5"


def test_collapsing_can_be_turned_off():
    out = expand_parents([child(1, "parent_0"), child(2, "parent_0")], make_index(), deduplicate_parents=False)
    assert len(out) == 2
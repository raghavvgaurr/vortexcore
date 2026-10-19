from src.pipeline import deduplicate


def passage(child_index, text, rrf=0.1):
    return {"child_index": child_index, "parent_id": f"p{child_index}", "text": text,
            "page": child_index, "rrf_score": rrf}


def test_identical_text_is_merged_with_provenance():
    out = deduplicate([passage(1, "ChatGPT reached 100 million users."),
                       passage(2, "chatgpt  reached 100 million users.", rrf=0.5)])
    assert len(out) == 1
    assert out[0]["duplicate_count"] == 2
    assert out[0]["all_parent_ids"] == ["p1", "p2"]
    assert out[0]["child_index"] == 2  # higher RRF score becomes the representative


def test_different_text_is_kept():
    out = deduplicate([passage(1, "Dartmouth Conference 1956"), passage(2, "Attention Is All You Need")])
    assert len(out) == 2


def test_near_duplicate_above_threshold_is_merged():
    base = "The Turing Test is still used today to test a machine's intelligence " * 3
    out = deduplicate([passage(1, base), passage(2, base + "x")])
    assert len(out) == 1


def test_empty_text_is_dropped():
    assert deduplicate([passage(1, "   ")]) == []
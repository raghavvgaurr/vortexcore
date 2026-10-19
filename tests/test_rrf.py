from src.retrieval import reciprocal_rank_fusion


def hit(child_index, source, text="t"):
    return {"child_index": child_index, "text": text, "retrieval_sources": [source]}


def test_chunk_found_by_both_lists_wins():
    dense = [hit(1, "dense"), hit(2, "dense"), hit(3, "dense")]
    bm25 = [hit(3, "bm25"), hit(4, "bm25")]
    fused = reciprocal_rank_fusion([dense, bm25], k=60, top_k=10)
    assert fused[0]["child_index"] == 3  # rank 3 + rank 1 beats rank 1 alone


def test_score_is_sum_of_reciprocal_ranks():
    fused = reciprocal_rank_fusion([[hit(7, "dense")], [hit(8, "bm25"), hit(7, "bm25")]], k=60, top_k=5)
    seven = next(f for f in fused if f["child_index"] == 7)
    assert abs(seven["rrf_score"] - (1 / 61 + 1 / 62)) < 1e-12


def test_sources_are_merged_and_first_copy_kept():
    fused = reciprocal_rank_fusion([[hit(1, "dense", "from dense")], [hit(1, "bm25", "from bm25")]], top_k=5)
    assert fused[0]["retrieval_sources"] == ["dense", "bm25"]
    assert fused[0]["text"] == "from dense"


def test_top_k_limits_output():
    many = [hit(i, "dense") for i in range(30)]
    assert len(reciprocal_rank_fusion([many], top_k=20)) == 20
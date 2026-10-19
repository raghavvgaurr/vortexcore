"""Cross-encoder reranker that scores each query-passage pair and reorders candidates."""
from functools import cache

from src.config import RERANKER_MODEL, TOP_K


@cache
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K) -> list[dict]:
    if not candidates:
        return []

    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = get_reranker().predict(pairs)

    reranked = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["reranker_score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda item: item["reranker_score"], reverse=True)
    return reranked[:top_k]
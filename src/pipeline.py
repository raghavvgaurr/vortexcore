"""The full RAG flow: decompose -> (rewrite) -> retrieve -> rerank -> dedup -> expand parents -> answer."""
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

from src.config import CANDIDATE_K, NEAR_DUPLICATE_THRESHOLD, TOP_K
from src.generation import generate_answer, generate_multipart_answer
from src.ingestion import build_chunks
from src.loader import load_pdf
from src.query_processing import decompose_query, rewrite_query
from src.reranking import rerank
from src.retrieval import Index, hybrid_retrieval, index_chunks

log = logging.getLogger(__name__)

NO_EVIDENCE_ANSWER = "The information is not available in the provided context."


def norm(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return (
        text.replace("“", '"').replace("”", '"')
            .replace("’", "'").replace("‘", "'")
    )


def deduplicate(passages: list[dict], threshold: float = NEAR_DUPLICATE_THRESHOLD) -> list[dict]:
    """Merge passages with (nearly) identical text, keeping where each copy came from."""
    groups = []

    for item in passages:
        normalized = norm(item.get("text", ""))
        if not normalized:
            continue

        found = None
        for group in groups:
            ratio = SequenceMatcher(None, normalized, group["normalized_text"]).ratio()
            if normalized == group["normalized_text"] or (threshold < 1.0 and ratio >= threshold):
                found = group
                break

        provenance = {
            "child_index": item.get("child_index"),
            "parent_id": item.get("parent_id"),
            "source": item.get("source", ""),
            "page": item.get("page"),
            "retrieval_sources": item.get("retrieval_sources", []),
            "rrf_score": item.get("rrf_score"),
            "reranker_score": item.get("reranker_score"),
        }

        if found is None:
            groups.append({**item, "normalized_text": normalized, "provenance": [provenance]})
        else:
            found["provenance"].append(provenance)
            # The copy with the better RRF score becomes the representative one.
            if item.get("rrf_score", -1) > found.get("rrf_score", -1):
                kept = found["provenance"]
                found.update(item)
                found["normalized_text"] = normalized
                found["provenance"] = kept

    for group in groups:
        group["duplicate_count"] = len(group["provenance"])
        group["all_parent_ids"] = sorted({
            str(p["parent_id"]) for p in group["provenance"] if p["parent_id"] is not None
        })
        group["all_locations"] = [
            {"source": p["source"], "page": p["page"], "child_index": p["child_index"]}
            for p in group["provenance"]
        ]
    return groups


def expand_parents(ranked_passages: list[dict], index: Index, deduplicate_parents: bool = True) -> list[dict]:
    """Swap each winning child chunk for its larger parent chunk."""
    expanded = []
    by_parent = {}

    for child in ranked_passages:
        parent_id = child.get("parent_id")
        key = str(parent_id) if parent_id is not None else f"missing_parent_child_{child.get('child_index')}"
        parent = index.parent_lookup.get(key)

        child_info = {
            "child_index": child.get("child_index"),
            "reranker_score": child.get("reranker_score"),
            "child_text": child.get("text", ""),
        }

        if deduplicate_parents and key in by_parent:
            by_parent[key]["supporting_children"].append(child_info)
            continue

        meta = parent.metadata if parent is not None else {}
        item = {
            "parent_key": key,
            "parent_id": parent_id,
            "parent_text": parent.page_content if parent is not None else child.get("text", ""),
            "source": meta.get("source", meta.get("file_path", child.get("source", ""))),
            "page": meta.get("page", child.get("page")),
            "child_evidence": child.get("text", ""),
            "reranker_score": child.get("reranker_score"),
            "supporting_children": [child_info],
            "parent_found": parent is not None,
        }
        expanded.append(item)
        by_parent[key] = item

    return expanded


def multipart_retrieval(
    question: str,
    index: Index,
    sub_queries: list[str] | None = None,
    candidate_k: int = CANDIDATE_K,
    top_k: int = TOP_K,
    document_ids=None,
) -> dict:
    """Run retrieval separately for each sub-query."""
    if sub_queries is None:
        sub_queries = decompose_query(question)

    all_results = []
    for sub_query in sub_queries:
        candidates = hybrid_retrieval(index, sub_query, top_k=candidate_k, document_ids=document_ids)
        ranked = rerank(sub_query, candidates, top_k=top_k)
        deduped = deduplicate(ranked)
        expanded = expand_parents(deduped, index)
        all_results.append({
            "sub_query": sub_query,
            "candidates": candidates,
            "ranked": ranked,
            "deduped": deduped,
            "expanded_context": expanded,
        })

    return {"question": question, "sub_queries": sub_queries, "results": all_results}


def _best_score(ranked_lists: list[list[dict]]) -> float | None:
    scores = [r["reranker_score"] for ranked in ranked_lists for r in ranked]
    return max(scores) if scores else None


def _should_refuse(best: float | None, min_score: float | None) -> bool:
    if best is None:
        return True
    return min_score is not None and best < min_score


def rag_pipeline(
    question: str,
    index: Index,
    candidate_k: int = CANDIDATE_K,
    top_k: int = TOP_K,
    document_ids=None,
    min_score: float | None = None,
) -> dict:
    """Answer a question. Single-intent questions are rewritten; multi-part ones are split."""
    sub_queries = decompose_query(question)

    if len(sub_queries) == 1:
        rewritten = rewrite_query(sub_queries[0])
        candidates = hybrid_retrieval(index, rewritten, top_k=candidate_k, document_ids=document_ids)
        ranked = rerank(rewritten, candidates, top_k=top_k)
        deduped = deduplicate(ranked)
        expanded = expand_parents(deduped, index)

        best = _best_score([ranked])
        abstained = _should_refuse(best, min_score)
        answer = NO_EVIDENCE_ANSWER if abstained else generate_answer(question, expanded)

        return {
            "question": question,
            "mode": "single",
            "sub_queries": sub_queries,
            "rewritten_query": rewritten,
            "candidates": candidates,
            "ranked": ranked,
            "deduped": deduped,
            "expanded_context": expanded,
            "best_score": best,
            "abstained": abstained,
            "answer": answer,
        }

    multipart = multipart_retrieval(
        question, index, sub_queries=sub_queries,
        candidate_k=candidate_k, top_k=top_k, document_ids=document_ids,
    )
    best = _best_score([item["ranked"] for item in multipart["results"]])
    abstained = _should_refuse(best, min_score)
    answer = NO_EVIDENCE_ANSWER if abstained else generate_multipart_answer(question, multipart)

    return {
        "question": question,
        "mode": "multipart",
        "sub_queries": sub_queries,
        "results": multipart["results"],
        "best_score": best,
        "abstained": abstained,
        "answer": answer,
    }


def build_index(pdf_paths: list[str | Path], force_ocr: bool = False) -> Index:
    """Read, chunk and index one or more PDFs. Each file's name becomes its document_id."""
    parents, children, seen = [], [], set()

    for path in map(Path, pdf_paths):
        document_id = path.stem
        if document_id in seen:
            raise ValueError(f"Two files are called '{document_id}'. Rename one of them.")
        seen.add(document_id)

        pages = load_pdf(path, force_ocr=force_ocr)
        file_parents, file_children = build_chunks(pages, document_id=document_id)
        log.info("%s: %d pages, %d parents, %d children",
                 path.name, len(pages), len(file_parents), len(file_children))
        parents += file_parents
        children += file_children

    if not children:
        raise ValueError("No text could be read from the PDF(s).")
    return index_chunks(parents, children)
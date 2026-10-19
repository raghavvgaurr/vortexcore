"""Dense (Qdrant) + BM25 search over the child chunks, fused with Reciprocal Rank Fusion."""
import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from src import embeddings
from src.config import BM25_K, CANDIDATE_K, DENSE_K, RRF_K

COLLECTION = "children"


@dataclass(eq=False)
class Index:
    """Everything the search functions need. Build it once with index_chunks()."""
    children: list[Document]
    parent_lookup: dict[str, Document]
    client: QdrantClient
    bm25: BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def _make_client() -> QdrantClient:
    # Qdrant Cloud is used only when QDRANT_URL is set; otherwise everything stays in memory.
    url = os.getenv("QDRANT_URL")
    if url:
        api_key = os.getenv("QDRANT_API_KEY")
        hostname = urlsplit(url).hostname or ""
        if hostname.endswith(".cloud.qdrant.io") and not api_key:
            raise ValueError(
                "QDRANT_API_KEY is required when connecting to Qdrant Cloud. "
                "Add it to your local .env file or environment."
            )
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(":memory:")


def index_chunks(parents: list[Document], children: list[Document]) -> Index:
    """Embed the children into Qdrant and build the BM25 index."""
    if not children:
        raise ValueError("No child chunks to index.")
    if any(not child.page_content.strip() for child in children):
        # child_index is the list position, so an empty chunk would shift every id.
        raise ValueError("Empty child chunk found. Remove empty chunks before indexing.")

    texts = [child.page_content.strip() for child in children]
    vectors = embeddings.embed_documents(texts)

    client = _make_client()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )

    points = []
    for position, (child, text, vector) in enumerate(zip(children, texts, vectors)):
        meta = child.metadata or {}
        points.append(models.PointStruct(
            id=position,
            vector=vector,
            payload={
                "text": text,
                "page": meta.get("page"),
                "parent_id": meta.get("parent_id"),
                "document_id": meta.get("document_id"),
                "source": meta.get("source", meta.get("file_path", "")),
                "child_index": position,
            },
        ))
    client.upload_points(collection_name=COLLECTION, points=points, batch_size=64)

    parent_lookup = {
        str(p.metadata["parent_id"]): p
        for p in parents
        if p.metadata.get("parent_id") is not None
    }
    bm25 = BM25Okapi([tokenize(child.page_content) for child in children])
    return Index(children=children, parent_lookup=parent_lookup, client=client, bm25=bm25)


def dense_search(index: Index, query: str, top_k: int = DENSE_K, document_ids=None) -> list[dict]:
    query_filter = None
    if document_ids:
        query_filter = models.Filter(must=[models.FieldCondition(
            key="document_id", match=models.MatchAny(any=list(document_ids)),
        )])

    query_vector = embeddings.embed_query(query)
    query_points = getattr(index.client, "query_points", None)
    if query_points is not None:
        hits = query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points
    else:
        hits = index.client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append({
            "child_index": payload["child_index"],
            "text": payload["text"],
            "page": payload.get("page"),
            "parent_id": payload.get("parent_id"),
            "source": payload.get("source"),
            "dense_score": float(hit.score),
            "retrieval_sources": ["dense"],
        })
    return results


def bm25_search(index: Index, query: str, top_k: int = BM25_K, document_ids=None) -> list[dict]:
    scores = index.bm25.get_scores(tokenize(query))
    if document_ids:
        allowed = np.array([c.metadata.get("document_id") in document_ids for c in index.children])
        scores = np.where(allowed, scores, -np.inf)

    results = []
    for position in scores.argsort()[::-1][:top_k]:
        if scores[position] == -np.inf:
            continue
        chunk = index.children[int(position)]
        meta = chunk.metadata or {}
        results.append({
            "child_index": int(position),
            "text": chunk.page_content,
            "page": meta.get("page"),
            "parent_id": meta.get("parent_id"),
            "source": meta.get("source", meta.get("file_path", "")),
            "bm25_score": float(scores[position]),
            "retrieval_sources": ["bm25"],
        })
    return results


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = RRF_K, top_k: int = CANDIDATE_K) -> list[dict]:
    """Merge ranked lists. Each hit scores 1/(k+rank) per list; scores add up across lists."""
    rrf_scores = {}
    first_copy = {}
    found_by = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["child_index"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            first_copy.setdefault(chunk_id, result)
            for name in result.get("retrieval_sources", []):
                if name not in found_by.setdefault(chunk_id, []):
                    found_by[chunk_id].append(name)

    best_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    fused = []
    for chunk_id in best_ids:
        item = dict(first_copy[chunk_id])
        item["rrf_score"] = rrf_scores[chunk_id]
        item["retrieval_sources"] = found_by.get(chunk_id, [])
        fused.append(item)
    return fused


def hybrid_retrieval(index: Index, query: str, top_k: int = CANDIDATE_K, document_ids=None) -> list[dict]:
    dense_results = dense_search(index, query, top_k=DENSE_K, document_ids=document_ids)
    bm25_results = bm25_search(index, query, top_k=BM25_K, document_ids=document_ids)
    return reciprocal_rank_fusion([dense_results, bm25_results], k=RRF_K, top_k=top_k)
"""Shared test helpers: a fake embedder so tests never download a model."""
import re
import zlib

import numpy as np
import pytest
from langchain_core.documents import Document

from src import embeddings

DIM = 64


def _vector(text: str) -> list[float]:
    # Bag-of-words hashed into 64 buckets. Deterministic, and similar texts score close.
    vec = np.zeros(DIM)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[zlib.crc32(word.encode()) % DIM] += 1.0
    return vec.tolist()


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(embeddings, "embed_documents", lambda texts: [_vector(t) for t in texts])
    monkeypatch.setattr(embeddings, "embed_query", _vector)


@pytest.fixture(autouse=True)
def use_local_qdrant(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)


def make_pages(texts_by_doc: dict[str, list[str]]) -> list[tuple[str, list[Document]]]:
    return [
        (doc_id, [Document(page_content=t, metadata={"source": f"{doc_id}.pdf", "page": i})
                  for i, t in enumerate(texts)])
        for doc_id, texts in texts_by_doc.items()
    ]
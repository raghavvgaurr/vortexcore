"""The embedding model, loaded once and reused."""
from functools import cache

from src.config import EMBEDDING_MODEL


@cache
def get_model():
    # Imported here so tests and the UI start without loading torch.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed many texts in one call; sentence-transformers batches them itself."""
    return get_model().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return get_model().embed_query(text)
"""Split pages into parent chunks (context for the LLM) and child chunks (what we search)."""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
)


def build_chunks(
    pages: list[Document], document_id: str | None = None
) -> tuple[list[Document], list[Document]]:
    """Return (parents, children). Every child carries its parent's parent_id."""
    # With several uploaded files, ids must not collide, so they get a prefix.
    prefix = f"{document_id}_" if document_id else ""

    parents = parent_splitter.split_documents(pages)
    children = []
    for i, parent in enumerate(parents):
        parent.metadata["parent_id"] = f"{prefix}parent_{i}"
        if document_id:
            parent.metadata["document_id"] = document_id
        for child in child_splitter.split_documents([parent]):
            # Dropping empties here means child_index == list position everywhere else.
            if child.page_content.strip():
                children.append(child)
    return parents, children
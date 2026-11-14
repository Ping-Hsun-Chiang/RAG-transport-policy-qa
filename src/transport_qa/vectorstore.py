"""Local persistent ChromaDB vector store."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import COLLECTION_NAME, VECTORSTORE_DIR
from .embeddings import get_embeddings


def make_chunk_id(doc: Document) -> str:
    """Deterministic ID, e.g. 'K0040012-7-1-0', so re-indexing never duplicates."""
    m = doc.metadata
    return f"{m['pcode']}-{m['article']}-{m['chunk_index']}"


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Open (or create) the persistent collection under vectorstore/."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_vectorstore(
    docs: list[Document],
    reset: bool = False,
    batch_size: int = 64,
    on_progress: Callable[[int, int], None] | None = None,
) -> Chroma:
    """Embed documents and upsert them into ChromaDB."""
    store = get_vectorstore()
    if reset:
        store.reset_collection()

    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        store.add_documents(batch, ids=[make_chunk_id(d) for d in batch])  # upsert by ID
        if on_progress:
            on_progress(min(start + batch_size, len(docs)), len(docs))
    return store

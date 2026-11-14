"""Similarity search over the local ChromaDB index."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from .config import MIN_SIMILARITY, TOP_K
from .vectorstore import get_vectorstore


@dataclass
class SearchResult:
    document: Document
    similarity: float  # cosine similarity in [0, 1]; higher = more relevant

    @property
    def citation(self) -> str:
        m = self.document.metadata
        return f"{m['law_name']} 第 {m['article']} 條"


def search(
    query: str,
    k: int = TOP_K,
    law_name: str | None = None,
    min_similarity: float = MIN_SIMILARITY,
    use_mmr: bool = False,
) -> list[SearchResult]:
    """Return the top-k most relevant chunks for ``query``.

    Args:
        query: natural-language question.
        k: number of chunks to return.
        law_name: restrict results to one regulation (metadata filter).
        min_similarity: drop results whose cosine similarity is below this.
        use_mmr: use Maximal Marginal Relevance to diversify results
            (avoids returning several near-identical chunks of the same article).
    """
    store = get_vectorstore()
    where = {"law_name": law_name} if law_name else None

    if use_mmr:
        docs = store.max_marginal_relevance_search(query, k=k, fetch_k=k * 4, filter=where)
        # MMR does not return scores; compute them from the query embedding.
        q = store.embeddings.embed_query(query)
        vecs = store.embeddings.embed_documents([d.page_content for d in docs])
        scored = [(d, sum(a * b for a, b in zip(q, v))) for d, v in zip(docs, vecs)]
    else:
        # Chroma returns cosine *distance* (1 - similarity).
        pairs = store.similarity_search_with_score(query, k=k, filter=where)
        scored = [(d, 1.0 - dist) for d, dist in pairs]

    return [SearchResult(d, s) for d, s in scored if s >= min_similarity]


def format_context(results: list[SearchResult]) -> str:
    """Render retrieved chunks as a numbered context block for the LLM prompt."""
    return "\n\n".join(f"[{i}] {r.document.page_content}" for i, r in enumerate(results, 1))

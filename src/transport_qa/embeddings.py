"""HuggingFace embedding model factory."""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE, EMBEDDING_MODEL


def _resolve_device() -> str:
    if EMBEDDING_DEVICE:
        return EMBEDDING_DEVICE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once per process.

    The model is downloaded from the HuggingFace Hub on first use
    (~2.3 GB for bge-m3) and cached under ~/.cache/huggingface.
    Vectors are L2-normalized so cosine similarity == dot product.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": _resolve_device()},
        encode_kwargs={"normalize_embeddings": True, "batch_size": EMBEDDING_BATCH_SIZE},
    )

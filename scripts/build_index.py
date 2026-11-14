"""
Phase 3: embed all chunks with BAAI/bge-m3 and store them in a local ChromaDB.

Usage:
    python scripts/build_index.py           # upsert (safe to re-run)
    python scripts/build_index.py --reset   # drop the collection and rebuild
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tqdm import tqdm  # noqa: E402

from transport_qa.chunker import chunk_laws  # noqa: E402
from transport_qa.config import COLLECTION_NAME, EMBEDDING_MODEL, RAW_DIR, VECTORSTORE_DIR  # noqa: E402
from transport_qa.embeddings import get_embeddings  # noqa: E402
from transport_qa.pdf_loader import load_all  # noqa: E402
from transport_qa.vectorstore import build_vectorstore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local ChromaDB index.")
    parser.add_argument("--reset", action="store_true", help="Drop the existing collection first")
    args = parser.parse_args()

    docs = chunk_laws(load_all(RAW_DIR))
    print(f"Loaded {len(docs)} chunks from {RAW_DIR}")

    t0 = time.perf_counter()
    print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads ~2.3 GB) ...")
    emb = get_embeddings()
    print(f"Model ready on {emb.model_kwargs['device']} in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    with tqdm(total=len(docs), desc="Embedding", unit="chunk") as bar:
        store = build_vectorstore(docs, reset=args.reset,
                                  on_progress=lambda done, total: bar.update(done - bar.n))
    elapsed = time.perf_counter() - t1

    count = store._collection.count()
    sample = store.get(limit=1, include=["embeddings", "metadatas"])
    dim = len(sample["embeddings"][0]) if len(sample["embeddings"]) else 0

    print(f"\nEmbedding finished in {elapsed:.1f}s")
    print(f"Collection   : {COLLECTION_NAME}")
    print(f"Persisted to : {VECTORSTORE_DIR}")
    print(f"Vectors      : {count} (dim={dim})")
    if count != len(docs):
        print(f"[WARN] vector count ({count}) != chunk count ({len(docs)}); try --reset")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

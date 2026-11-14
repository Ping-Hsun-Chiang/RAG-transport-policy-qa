"""
Phase 2: parse regulation PDFs and chunk them by article.

Prints statistics and sample chunks, and writes all chunks to
data/processed/chunks.jsonl for inspection.

Usage:
    python scripts/build_chunks.py
    python scripts/build_chunks.py --show 5 --article 7-1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transport_qa.chunker import chunk_laws  # noqa: E402
from transport_qa.config import CHUNK_OVERLAP, CHUNK_SIZE, PROCESSED_DIR, RAW_DIR  # noqa: E402
from transport_qa.pdf_loader import load_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk regulation PDFs by article.")
    parser.add_argument("--show", type=int, default=3, help="Number of sample chunks to print")
    parser.add_argument("--article", help="Print all chunks of a specific article number, e.g. 7-1")
    args = parser.parse_args()

    laws = load_all(RAW_DIR)
    docs = chunk_laws(laws)

    print(f"chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}\n")
    print(f"{'法規':<16}{'修正日期':<22}{'條文數':>6}{'chunks':>8}")
    per_law = Counter(d.metadata["law_name"] for d in docs)
    for law in laws:
        n_articles = len({d.metadata["article"] for d in docs if d.metadata["pcode"] == law.pcode})
        print(f"{law.law_name:<16}{law.amended_date:<22}{n_articles:>6}{per_law[law.law_name]:>8}")

    lengths = [len(d.page_content) for d in docs]
    split_articles = sum(1 for d in docs if d.metadata["chunk_index"] == 1)
    print(f"\nTotal chunks: {len(docs)}")
    print(f"Chunk length (chars): min={min(lengths)}, median={statistics.median(lengths):.0f}, max={max(lengths)}")
    print(f"Articles split into multiple chunks: {split_articles}")

    samples = [d for d in docs if d.metadata["article"] == args.article] if args.article else docs[: args.show]
    for d in samples:
        print("\n" + "-" * 60)
        print(json.dumps(d.metadata, ensure_ascii=False))
        print(d.page_content)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"page_content": d.page_content, "metadata": d.metadata}, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(docs)} chunks to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

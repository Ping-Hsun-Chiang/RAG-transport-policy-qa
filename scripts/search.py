"""
Phase 4: retrieve the Top-K most relevant regulation chunks for a question.

Usage:
    python scripts/search.py "酒後駕車會被處罰多少錢？"
    python scripts/search.py "汽車運輸業分為哪幾種？" -k 3 --law 公路法
    python scripts/search.py "闖紅燈罰多少？" --mmr
    python scripts/search.py --eval            # run tests/retrieval_cases.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from transport_qa.config import MIN_SIMILARITY, TOP_K  # noqa: E402
from transport_qa.retriever import search  # noqa: E402

CASES_FILE = ROOT / "tests" / "retrieval_cases.json"


def print_results(query: str, k: int, law: str | None, mmr: bool, min_sim: float) -> None:
    results = search(query, k=k, law_name=law, use_mmr=mmr, min_similarity=min_sim)
    print(f"\nQ: {query}")
    print(f"Top-{k}{' (MMR)' if mmr else ''}{f' filter={law}' if law else ''}, min_similarity={min_sim}")
    if not results:
        print("  (no relevant chunks found)")
    for rank, r in enumerate(results, 1):
        print("\n" + "=" * 70)
        print(f"#{rank}  similarity={r.similarity:.4f}  {r.citation}  "
              f"(chunk {r.document.metadata['chunk_index'] + 1}/{r.document.metadata['chunk_total']})")
        print("-" * 70)
        print(r.document.page_content)


def run_eval(k: int, mmr: bool) -> int:
    """Hit@K and MRR over a small hand-labelled question set."""
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    hits, rr_sum = 0, 0.0
    print(f"Evaluating {len(cases)} cases, Top-{k}{' (MMR)' if mmr else ''}\n")
    for case in cases:
        results = search(case["question"], k=k, use_mmr=mmr, min_similarity=0.0)
        citations = [r.citation for r in results]
        rank = next((i for i, c in enumerate(citations, 1) if c in case["expected"]), None)
        hits += rank is not None
        rr_sum += 1 / rank if rank else 0
        mark = f"HIT @{rank}" if rank else "MISS  "
        print(f"[{mark}] {case['question']}\n         got: {', '.join(citations)}")
    print(f"\nHit@{k}: {hits}/{len(cases)} = {hits / len(cases):.2%}")
    print(f"MRR   : {rr_sum / len(cases):.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Top-K similarity search over the regulation index.")
    parser.add_argument("query", nargs="?", help="Question to search")
    parser.add_argument("-k", type=int, default=TOP_K, help=f"Number of results (default {TOP_K})")
    parser.add_argument("--law", help="Restrict to one regulation, e.g. 公路法")
    parser.add_argument("--mmr", action="store_true", help="Diversify results with MMR")
    parser.add_argument("--min-sim", type=float, default=MIN_SIMILARITY, help="Minimum cosine similarity")
    parser.add_argument("--eval", action="store_true", help="Run the retrieval evaluation set")
    args = parser.parse_args()

    if args.eval:
        return run_eval(args.k, args.mmr)
    if not args.query:
        parser.error("please provide a query or use --eval")
    print_results(args.query, args.k, args.law, args.mmr, args.min_sim)
    return 0


if __name__ == "__main__":
    sys.exit(main())

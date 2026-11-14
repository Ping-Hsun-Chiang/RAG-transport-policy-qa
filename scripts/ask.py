"""
Phase 5: ask questions about Taiwan transport regulations (fully local RAG).

Usage:
    python scripts/ask.py "闖紅燈罰多少？"          # single question
    python scripts/ask.py                           # interactive mode
    python scripts/ask.py "..." --model qwen2.5:7b --show-context
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx  # noqa: E402

from transport_qa.config import OLLAMA_BASE_URL, OLLAMA_MODEL, TOP_K  # noqa: E402
from transport_qa.prompts import DISCLAIMER  # noqa: E402
from transport_qa.rag_chain import RegulationQA  # noqa: E402


def check_ollama(model: str) -> None:
    """Fail fast with a helpful message if Ollama or the model is missing."""
    try:
        tags = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).json()
    except httpx.HTTPError:
        sys.exit(f"[ERROR] Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                 "Is Ollama running? Start it with: ollama serve")
    names = {m["name"] for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        sys.exit(f"[ERROR] Model '{model}' not found. Download it with: ollama pull {model}")


def answer_one(qa: RegulationQA, question: str, show_context: bool) -> None:
    t0 = time.perf_counter()
    sources, tokens = qa.stream(question)
    t_retrieve = time.perf_counter() - t0

    if show_context:
        print("\n----- 檢索到的條文 -----")
        for i, s in enumerate(sources, 1):
            print(f"[{i}] ({s.similarity:.3f}) {s.document.page_content}\n")

    print("\n答：", end="", flush=True)
    for tok in tokens:
        print(tok, end="", flush=True)
    t_total = time.perf_counter() - t0

    if sources:
        print(f"\n\n{DISCLAIMER}")
        print("\n參考來源：")
        seen = set()
        for s in sources:
            if s.citation in seen:
                continue
            seen.add(s.citation)
            print(f"  - {s.citation}（相似度 {s.similarity:.3f}）{s.document.metadata['source_url']}")
    print(f"\n(retrieval {t_retrieve:.1f}s, total {t_total:.1f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local RAG QA for Taiwan transport regulations.")
    parser.add_argument("question", nargs="?", help="Question (omit for interactive mode)")
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model (default {OLLAMA_MODEL})")
    parser.add_argument("-k", type=int, default=TOP_K, help=f"Chunks to retrieve (default {TOP_K})")
    parser.add_argument("--law", help="Restrict retrieval to one regulation, e.g. 公路法")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks")
    args = parser.parse_args()

    check_ollama(args.model)
    print(f"Loading models (embedding + {args.model}) ...")
    qa = RegulationQA(model=args.model, k=args.k, law_name=args.law)

    if args.question:
        answer_one(qa, args.question, args.show_context)
        return 0

    print("台灣運輸法規問答系統（輸入 exit 離開）")
    while True:
        try:
            question = input("\n問：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            answer_one(qa, question, args.show_context)
    return 0


if __name__ == "__main__":
    sys.exit(main())

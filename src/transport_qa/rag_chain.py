"""Retrieval-Augmented Generation pipeline: retrieve -> prompt -> local LLM."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from .config import LLM_NUM_CTX, LLM_TEMPERATURE, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_GPU, TOP_K
from .prompts import DISCLAIMER, NO_CONTEXT_ANSWER, QA_PROMPT
from .retriever import SearchResult, format_context, search
from .vectorstore import get_vectorstore


def get_llm(model: str = OLLAMA_MODEL) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,  # low temperature = faithful, less creative
        num_ctx=LLM_NUM_CTX,
        num_gpu=OLLAMA_NUM_GPU,
    )


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[SearchResult]


class RegulationQA:
    """Question answering over Taiwanese transport regulations."""

    def __init__(self, model: str = OLLAMA_MODEL, k: int = TOP_K, law_name: str | None = None):
        self.k = k
        self.law_name = law_name
        self.chain = QA_PROMPT | get_llm(model) | StrOutputParser()
        get_vectorstore()  # preload the embedding model so the first query is not slow

    def retrieve(self, question: str) -> list[SearchResult]:
        return search(question, k=self.k, law_name=self.law_name)

    def ask(self, question: str) -> RAGAnswer:
        sources = self.retrieve(question)
        if not sources:  # guardrail: never let the LLM answer without evidence
            return RAGAnswer(question, NO_CONTEXT_ANSWER, [])
        answer = self.chain.invoke({"context": format_context(sources), "question": question})
        return RAGAnswer(question, f"{answer.strip()}\n\n{DISCLAIMER}", sources)

    def stream(self, question: str) -> tuple[list[SearchResult], Iterator[str]]:
        """Like ask(), but yields the answer token by token (disclaimer not included)."""
        sources = self.retrieve(question)
        if not sources:
            return [], iter([NO_CONTEXT_ANSWER])
        return sources, self.chain.stream({"context": format_context(sources), "question": question})

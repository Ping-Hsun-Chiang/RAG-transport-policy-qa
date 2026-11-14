"""Article-aware chunking for Taiwanese legal texts.

Strategy
--------
1. Structural split: parse the text into articles (「第 N 條」 / 「第 N-1 條」),
   tracking the current chapter (章) and section (節) as metadata.
2. Line repair: PDF export wraps lines at a fixed width, cutting sentences in
   the middle. Wrapped lines are joined back together, while real paragraph
   boundaries (項 "1   ", 款 "一、", 目 "（一）") are kept as newlines.
3. Size control: an article that fits in CHUNK_SIZE becomes exactly one chunk,
   so a legal provision is never split unnecessarily. Longer articles are split
   with RecursiveCharacterTextSplitter, preferring paragraph > sentence > clause
   boundaries, with overlap.
4. Context header: every chunk starts with 「【法規名稱 第 N 條】(章節)」 so the
   embedding knows which law / article the text belongs to, even after splitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .pdf_loader import LawDocument

ARTICLE_RE = re.compile(r"^第\s*(\d+(?:-\d+)?)\s*條\s*$")
CHAPTER_RE = re.compile(r"^\s*第\s*\S+\s*章\s*(.*)$")
SECTION_RE = re.compile(r"^\s*第\s*\S+\s*節\s*(.*)$")
# Lines that start a new paragraph (項 / 款 / 目) and must not be merged upward.
PARAGRAPH_START_RE = re.compile(r"^(\d+\s{2,}|[一二三四五六七八九十百]+、|（[一二三四五六七八九十百]+）)")
DELETED_RE = re.compile(r"^（刪除）$")

CHINESE_SEPARATORS = ["\n", "。", "；", "，", ""]


@dataclass
class Article:
    number: str
    chapter: str = ""
    section: str = ""
    lines: list[str] = field(default_factory=list)


def _normalize_heading(raw: str) -> str:
    """'   第 一 章 總則' -> '第一章 總則'."""
    parts = raw.split()
    # parts like ['第', '一', '章', '總則']
    if len(parts) >= 3 and parts[0] == "第":
        return f"第{parts[1]}{parts[2]} {' '.join(parts[3:])}".strip()
    return raw.strip()


def _join_wrapped_lines(lines: list[str]) -> str:
    """Merge fixed-width wrapped lines; keep paragraph boundaries."""
    paragraphs: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not paragraphs or PARAGRAPH_START_RE.match(stripped):
            # Collapse "1   本法..." into "1 本法..."
            paragraphs.append(re.sub(r"^(\d+)\s{2,}", r"\1 ", stripped))
        else:
            paragraphs[-1] += stripped
    return "\n".join(paragraphs)


def parse_articles(law: LawDocument) -> list[Article]:
    """Split a regulation's raw text into articles with chapter/section info."""
    articles: list[Article] = []
    chapter = section = ""
    current: Article | None = None

    for line in law.text.splitlines():
        stripped = line.strip()
        if m := ARTICLE_RE.match(stripped):
            current = Article(number=m.group(1), chapter=chapter, section=section)
            articles.append(current)
        elif CHAPTER_RE.match(line):
            chapter, section = _normalize_heading(line), ""
        elif SECTION_RE.match(line):
            section = _normalize_heading(line)
        elif current is not None:
            current.lines.append(line)
        # Lines before the first article (title, dates, notes) are metadata only.
    return articles


def chunk_law(law: LawDocument, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """Convert one regulation into LangChain Documents ready for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
        keep_separator="end",  # keep 「。」 at the end of the sentence
    )

    docs: list[Document] = []
    for art in parse_articles(law):
        body = _join_wrapped_lines(art.lines)
        if not body or DELETED_RE.match(body):
            continue  # skip repealed articles 「（刪除）」

        location = " ".join(x for x in (art.chapter, art.section) if x)
        header = f"【{law.law_name} 第 {art.number} 條】" + (f"（{location}）" if location else "")

        pieces = [body] if len(body) <= chunk_size else splitter.split_text(body)
        for i, piece in enumerate(pieces):
            docs.append(
                Document(
                    page_content=f"{header}\n{piece.strip()}",
                    metadata={
                        "law_name": law.law_name,
                        "pcode": law.pcode,
                        "article": art.number,
                        "chapter": art.chapter,
                        "section": art.section,
                        "chunk_index": i,
                        "chunk_total": len(pieces),
                        "amended_date": law.amended_date,
                        "source_file": law.source_file,
                        "source_url": law.source_url,
                    },
                )
            )
    return docs


def chunk_laws(laws: list[LawDocument], **kwargs) -> list[Document]:
    return [doc for law in laws for doc in chunk_law(law, **kwargs)]

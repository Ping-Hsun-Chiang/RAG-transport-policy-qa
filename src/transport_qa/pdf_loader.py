"""Load regulation PDFs exported from law.moj.gov.tw into plain text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

_META_PATTERNS = {
    "law_name": re.compile(r"^法規名稱：(.+)$", re.M),
    "amended_date": re.compile(r"^修正日期：(.+)$", re.M),
}


@dataclass
class LawDocument:
    """Raw text of one regulation plus document-level metadata."""

    pcode: str
    law_name: str
    amended_date: str
    source_file: str
    source_url: str
    text: str


def load_law_pdf(path: Path) -> LawDocument:
    """Extract the full text of a regulation PDF.

    File names follow the downloader convention ``<PCODE>_<法規名稱>.pdf``.
    """
    with pymupdf.open(path) as doc:
        text = "\n".join(page.get_text("text") for page in doc)

    pcode = path.stem.split("_", 1)[0]
    meta = {k: (m.group(1).strip() if (m := p.search(text)) else "") for k, p in _META_PATTERNS.items()}

    return LawDocument(
        pcode=pcode,
        law_name=meta["law_name"] or path.stem.split("_", 1)[-1],
        amended_date=meta["amended_date"],
        source_file=path.name,
        source_url=f"https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={pcode}",
        text=text,
    )


def load_all(raw_dir: Path) -> list[LawDocument]:
    """Load every PDF in ``raw_dir`` (sorted for deterministic output)."""
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {raw_dir}. Run scripts/download_laws.py first.")
    return [load_law_pdf(p) for p in pdfs]

"""PDF ingestion: extract text page by page and split it into overlapping chunks.

Every chunk remembers which document and page it came from, so answers can
cite their sources precisely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


@dataclass
class Chunk:
    id: int
    doc: str          # source file name
    page: int         # 1-based page number
    text: str


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(source: str | Path | BinaryIO) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) for every non-empty page."""
    reader = PdfReader(source)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append((i, text))
    return pages


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into chunks of roughly `chunk_size` characters.

    Splits on sentence boundaries where possible and carries `overlap`
    characters of context from one chunk into the next.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks, current = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > chunk_size:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
            # start the new chunk on a word boundary
            current = current[current.find(" ") + 1:] if " " in current else current
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def build_chunks(
    source: str | Path | BinaryIO,
    doc_name: str,
    start_id: int = 0,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    next_id = start_id
    for page_no, text in read_pdf(source):
        for piece in split_text(text, chunk_size, overlap):
            chunks.append(Chunk(id=next_id, doc=doc_name, page=page_no, text=piece))
            next_id += 1
    return chunks

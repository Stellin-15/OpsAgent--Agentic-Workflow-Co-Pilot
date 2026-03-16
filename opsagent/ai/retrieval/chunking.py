"""Runbook chunker.

Splits markdown documents into overlapping chunks suitable for vector search.
Each chunk carries its source document path and chunk index as metadata so
retrieved context can be attributed back to the correct runbook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    content: str
    source: str          # file path relative to runbooks dir
    chunk_index: int
    total_chunks: int


class MarkdownChunker:
    """
    Simple fixed-size sliding-window chunker.

    Splits on token count (approximate: 1 token ≈ 4 chars) with overlap to
    avoid cutting reasoning mid-sentence. Falls back to the whole document if
    it is shorter than chunk_size.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str) -> list[Chunk]:
        """Split *text* (from *source*) into overlapping chunks."""
        if not text.strip():
            return []

        step = self.chunk_size - self.chunk_overlap
        raw_chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            raw_chunks.append(text[start:end])
            if end >= len(text):
                break
            start += step

        return [
            Chunk(
                content=c,
                source=source,
                chunk_index=i,
                total_chunks=len(raw_chunks),
            )
            for i, c in enumerate(raw_chunks)
        ]

    def chunk_file(self, path: Path, base_dir: Path | None = None) -> list[Chunk]:
        """Read *path* and chunk its contents."""
        text = path.read_text(encoding="utf-8")
        source = str(path.relative_to(base_dir)) if base_dir else path.name
        return self.chunk_text(text, source)

    def chunk_directory(self, directory: Path) -> list[Chunk]:
        """Chunk every ``*.md`` file found recursively under *directory*."""
        all_chunks: list[Chunk] = []
        for md_path in sorted(directory.rglob("*.md")):
            all_chunks.extend(self.chunk_file(md_path, base_dir=directory))
        return all_chunks

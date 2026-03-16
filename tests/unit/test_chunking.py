"""Unit tests for the markdown chunker."""

from pathlib import Path

import pytest

from opsagent.ai.retrieval.chunking import Chunk, MarkdownChunker


class TestMarkdownChunker:
    def test_short_text_single_chunk(self) -> None:
        chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk_text("Short text", source="test.md")
        assert len(chunks) == 1
        assert chunks[0].content == "Short text"
        assert chunks[0].source == "test.md"
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1

    def test_long_text_multiple_chunks(self) -> None:
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=20)
        text = "x" * 300
        chunks = chunker.chunk_text(text, source="long.md")
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_overlap(self) -> None:
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=20)
        text = "a" * 200
        chunks = chunker.chunk_text(text, source="x.md")
        # Second chunk should start 80 chars into first (100 - 20 = 80)
        assert len(chunks[0].content) <= 100
        assert len(chunks) >= 2

    def test_empty_text_returns_no_chunks(self) -> None:
        chunker = MarkdownChunker()
        chunks = chunker.chunk_text("   ", source="empty.md")
        assert chunks == []

    def test_chunk_file(self, tmp_path: Path) -> None:
        (tmp_path / "runbook.md").write_text("# High CPU\n\nCheck top, kill process.")
        chunker = MarkdownChunker()
        chunks = chunker.chunk_file(tmp_path / "runbook.md")
        assert len(chunks) == 1
        assert "High CPU" in chunks[0].content

    def test_chunk_directory(self, tmp_path: Path) -> None:
        (tmp_path / "rb1.md").write_text("Runbook 1 content")
        (tmp_path / "rb2.md").write_text("Runbook 2 content")
        chunker = MarkdownChunker()
        chunks = chunker.chunk_directory(tmp_path)
        sources = {c.source for c in chunks}
        assert "rb1.md" in sources
        assert "rb2.md" in sources

    def test_chunk_directory_ignores_non_md(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("This is a text file")
        (tmp_path / "notes.md").write_text("Markdown notes")
        chunker = MarkdownChunker()
        chunks = chunker.chunk_directory(tmp_path)
        sources = {c.source for c in chunks}
        assert not any("readme.txt" in s for s in sources)
        assert "notes.md" in sources

"""Unit tests for doc_loader and chunker modules."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.chunking.chunker import chunk_documents
from src.utils.exceptions import (
    DocumentParsingError,
    UnsupportedFileTypeError,
)


# ── Chunker tests ─────────────────────────────────────────────────────

class TestChunker:

    def test_basic_chunking(self):
        docs = [Document(page_content="Hello world. " * 100, metadata={"sha256": "abc12345", "page": 1})]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10, min_chunk_length=10)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.page_content) <= 150  # some slack for splitter boundaries
            assert "chunk_id" in c.metadata
            assert "chunk_index" in c.metadata

    def test_empty_input_returns_empty(self):
        assert chunk_documents([]) == []

    def test_short_chunks_are_filtered(self):
        docs = [Document(page_content="Hi.", metadata={"sha256": "abc12345", "page": 1})]
        chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=50, min_chunk_length=100)
        # "Hi." is 3 chars < 100 → should be filtered out
        assert len(chunks) == 0

    def test_chunk_ids_are_unique(self):
        content = "The quick brown fox jumps over the lazy dog. " * 50
        docs = [Document(page_content=content, metadata={"sha256": "deadbeef", "page": 1})]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_preserved(self):
        docs = [Document(
            page_content="Some content. " * 30,
            metadata={"sha256": "aabbccdd", "page": 2, "filename": "test.pdf"},
        )]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        for c in chunks:
            assert c.metadata["filename"] == "test.pdf"
            assert c.metadata["page"] == 2


# ── Doc loader tests ──────────────────────────────────────────────────

class TestDocLoader:

    def test_load_txt_file(self, tmp_path):
        from src.ingestion.doc_loader import load_document

        txt = tmp_path / "test.txt"
        txt.write_text("Hello, this is a test document.", encoding="utf-8")
        docs = load_document(txt)
        assert len(docs) == 1
        assert "Hello" in docs[0].page_content
        assert docs[0].metadata["extension"] == ".txt"

    def test_load_md_file(self, tmp_path):
        from src.ingestion.doc_loader import load_document

        md = tmp_path / "test.md"
        md.write_text("# Title\n\nSome markdown content.", encoding="utf-8")
        docs = load_document(md)
        assert len(docs) == 1
        assert "markdown" in docs[0].page_content

    def test_unsupported_extension_raises(self, tmp_path):
        from src.ingestion.doc_loader import load_document

        f = tmp_path / "test.csv"
        f.write_text("a,b,c")
        with pytest.raises(UnsupportedFileTypeError):
            load_document(f)

    def test_empty_txt_raises(self, tmp_path):
        from src.ingestion.doc_loader import load_document

        empty = tmp_path / "empty.txt"
        empty.write_text("   ", encoding="utf-8")  # whitespace only
        with pytest.raises(DocumentParsingError):
            load_document(empty)

    def test_metadata_includes_sha256(self, tmp_path):
        from src.ingestion.doc_loader import load_document

        txt = tmp_path / "doc.txt"
        txt.write_text("Content for hashing test.")
        docs = load_document(txt)
        assert "sha256" in docs[0].metadata
        assert len(docs[0].metadata["sha256"]) == 64  # SHA-256 hex length

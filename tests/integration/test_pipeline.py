"""
Integration test — exercises the full RAG pipeline end-to-end using
the sample HR policy document that ships with the repo.

These tests require:
  - No real OpenAI key (LLM call is mocked)
  - No GPU (runs on CPU)

Run with:
    pytest tests/integration/ -v
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Skip if running in a CI environment without the model cache
SKIP_HEAVY = os.getenv("CI_SKIP_HEAVY", "false").lower() == "true"


@pytest.fixture(scope="module")
def sample_txt_file(tmp_path_factory) -> Path:
    """Write a sample text document to a temp file."""
    content = """
    ACME Corporation — HR Policy

    Section 1: Remote Work
    Employees may work remotely up to 3 days per week after completing
    their 90-day probationary period. Core hours are 10 AM – 3 PM local time.

    Section 2: Expense Policy
    Business travel is reimbursable. Client entertainment is capped at $150
    per person. Receipts must be submitted within 30 days.

    Section 3: Equipment
    The company provides one laptop and one external monitor per employee.
    Personal devices require IT approval before accessing company systems.
    """
    tmp = tmp_path_factory.mktemp("docs") / "hr_policy.txt"
    tmp.write_text(content.strip(), encoding="utf-8")
    return tmp


@pytest.mark.skipif(SKIP_HEAVY, reason="Heavy model test skipped in CI")
class TestIngestionPipeline:
    """Test document ingestion: load → chunk → embed → store."""

    def test_load_txt_document(self, sample_txt_file):
        from src.ingestion.doc_loader import load_document

        docs = load_document(sample_txt_file)
        assert len(docs) >= 1
        assert "Remote Work" in docs[0].page_content
        assert docs[0].metadata["filename"] == "hr_policy.txt"
        assert docs[0].metadata["sha256"]  # non-empty

    def test_chunk_documents(self, sample_txt_file):
        from src.chunking.chunker import chunk_documents
        from src.ingestion.doc_loader import load_document

        docs = load_document(sample_txt_file)
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20, min_chunk_length=30)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.page_content) >= 30
            assert "chunk_id" in chunk.metadata


@pytest.mark.skipif(SKIP_HEAVY, reason="Heavy model test skipped in CI")
class TestFullPipelineWithMockedLLM:
    """
    Full pipeline test with a mocked LLM (no real API key needed).
    Validates that ingest → query flows work end-to-end.
    """

    @patch("src.generation.generator.AnswerGenerator._get_llm")
    def test_ingest_and_query(self, mock_get_llm, sample_txt_file, tmp_path):
        """End-to-end: ingest a doc, then query it."""
        # Mock the LLM response
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "Employees may work remotely up to 3 days per week [1]. "
            "Core hours are 10 AM to 3 PM local time [1].\n\nSources: [1] hr_policy.txt"
        )
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        # Point ChromaDB to a temp dir so tests don't pollute production data
        os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "test_chroma")

        from src.utils.config import get_settings
        get_settings.cache_clear()

        from src.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        pipeline.initialize()

        # Ingest
        result = pipeline.ingest_document(sample_txt_file)
        assert result["status"] == "success"
        assert result["chunk_count"] > 0

        # Query
        answer = pipeline.query("What is the remote work policy?")
        assert "answer" in answer
        assert len(answer["answer"]) > 10
        assert isinstance(answer["sources"], list)

        # Stats
        stats = pipeline.get_stats()
        assert stats["total_chunks"] > 0
        assert "hr_policy.txt" in stats["indexed_files"]

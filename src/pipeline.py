"""
DocuMind RAG Pipeline
----------------------
Orchestrates the full Retrieval-Augmented Generation pipeline:

  Document → Load → Chunk → Embed → Store
  Query → Embed → Retrieve → Rerank → Generate → Answer

This is the single entry point for both the API and the frontend.
All components are initialised here and wired together.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from src.chunking.chunker import chunk_documents
from src.embeddings.embedder import EmbeddingGenerator
from src.generation.generator import AnswerGenerator
from src.ingestion.doc_loader import load_document
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.vector_store import ChromaVectorStore, get_vector_store
from src.utils.config import get_settings
from src.utils.exceptions import DocuMindError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    Wires together all DocuMind components into a coherent pipeline.

    Components initialised once at startup:
      • EmbeddingGenerator  — sentence-transformers model
      • ChromaVectorStore   — persistent local vector DB
      • SemanticRetriever   — embeds queries + runs similarity search
      • CrossEncoderReranker — cross-encoder for precise re-scoring
      • AnswerGenerator     — LLM answer generation with citations

    Usage:
        pipeline = RAGPipeline()
        pipeline.initialize()

        # Ingest a document
        result = pipeline.ingest_document(Path("report.pdf"))

        # Query
        answer = pipeline.query("What is the vacation policy?")
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedder: Optional[EmbeddingGenerator] = None
        self._vector_store: Optional[ChromaVectorStore] = None
        self._retriever: Optional[SemanticRetriever] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._generator: Optional[AnswerGenerator] = None
        self._initialized = False

    def initialize(self) -> None:
        """
        Load all models and connect to the vector store.
        Call once at application startup (e.g. FastAPI lifespan).
        """
        if self._initialized:
            logger.warning("RAGPipeline.initialize() called more than once — skipping.")
            return

        cfg = self._settings
        logger.info("Initialising RAG pipeline…")

        # 1. Embedding model
        self._embedder = EmbeddingGenerator(
            model_name=cfg.embedding.embedding_model_name,
            cache_dir=cfg.embedding.embedding_cache_dir,
            batch_size=cfg.embedding.embedding_batch_size,
        )
        self._embedder.load()

        # 2. Vector store
        self._vector_store = get_vector_store(
            store_type=cfg.vector_store.vector_store_type,
            persist_dir=cfg.vector_store.chroma_persist_dir,
            collection_name=cfg.vector_store.chroma_collection_name,
        )

        # 3. Retriever
        self._retriever = SemanticRetriever(
            vector_store=self._vector_store,
            embedder=self._embedder,
            top_k=cfg.retrieval.retrieval_top_k,
            score_threshold=cfg.retrieval.similarity_threshold,
        )

        # 4. Re-ranker
        self._reranker = CrossEncoderReranker(
            model_name=cfg.retrieval.reranker_model_name,
            top_n=cfg.retrieval.reranking_top_n,
        )

        # 5. LLM generator
        self._generator = AnswerGenerator(
            llm_provider=cfg.llm.llm_provider,
            model_name=cfg.llm.openai_model,
            api_key=cfg.llm.openai_api_key,
            temperature=cfg.llm.openai_temperature,
            max_tokens=cfg.llm.openai_max_tokens,
        )

        self._initialized = True
        logger.info("RAG pipeline ready.")

    # ── Ingestion ─────────────────────────────────────────────────────

    def ingest_document(self, file_path: Path) -> Dict:
        """
        Load, chunk, embed, and store a document.

        Returns a summary dict with filename, sha256, chunk_count.
        """
        self._check_initialized()
        logger.info("Ingesting document", path=str(file_path))

        # Load
        docs = load_document(file_path)

        # Chunk
        cfg = self._settings.chunking
        chunks = chunk_documents(
            docs,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            min_chunk_length=cfg.min_chunk_length,
        )

        if not chunks:
            return {
                "filename": file_path.name,
                "sha256": docs[0].metadata.get("sha256", ""),
                "chunk_count": 0,
                "status": "no_content",
            }

        # Embed
        texts = [c.page_content for c in chunks]
        embeddings = self._embedder.embed_texts(texts)

        # Store
        ids = self._vector_store.add_documents(chunks, embeddings)

        sha256 = docs[0].metadata.get("sha256", "")
        logger.info(
            "Document ingested",
            filename=file_path.name,
            chunks=len(ids),
            sha256=sha256[:8],
        )

        return {
            "filename": file_path.name,
            "sha256": sha256,
            "chunk_count": len(ids),
            "status": "success",
        }

    # ── Query ─────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        use_reranker: bool = True,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> Dict:
        """
        Run the full RAG pipeline for a user question.

        Args:
            question:    Natural language question.
            use_reranker: Whether to apply cross-encoder re-ranking.
            top_k:       Override retrieval top_k.
            top_n:       Override reranking top_n.

        Returns:
            Dict with keys: answer, sources, model, provider, chunks_used.
        """
        self._check_initialized()
        logger.info("Processing query", question=question[:80])

        # Retrieve
        candidates = self._retriever.retrieve(question, top_k=top_k)

        if not candidates:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents.",
                "sources": [],
                "model": self._settings.llm.openai_model,
                "provider": self._settings.llm.llm_provider,
                "chunks_used": 0,
            }

        # Re-rank
        if use_reranker:
            final_chunks = self._reranker.rerank(question, candidates, top_n=top_n)
        else:
            n = top_n or self._settings.retrieval.reranking_top_n
            final_chunks = candidates[:n]

        # Generate
        result = self._generator.generate(question, final_chunks)
        result["chunks_used"] = len(final_chunks)

        return result

    # ── Utility ───────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return current vector store stats."""
        self._check_initialized()
        return {
            "total_chunks": self._vector_store.count(),
            "indexed_files": self._vector_store.list_sources(),
            "embedding_model": self._settings.embedding.embedding_model_name,
            "llm_model": self._settings.llm.openai_model,
            "vector_store": self._settings.vector_store.vector_store_type,
        }

    def delete_document(self, source: str) -> Dict:
        """Remove all chunks from a source document."""
        self._check_initialized()
        count = self._vector_store.delete_by_source(source)
        return {"deleted_chunks": count, "source": source}

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise DocuMindError(
                "RAGPipeline has not been initialised. Call pipeline.initialize() first."
            )


# ── Module-level singleton ────────────────────────────────────────────
_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Return the process-wide RAGPipeline singleton (initialised on first call)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
        _pipeline_instance.initialize()
    return _pipeline_instance

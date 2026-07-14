"""
DocuMind Vector Store
----------------------
Abstraction layer over ChromaDB (local dev) and Pinecone (production).

All external vector DB calls are funnelled through this module so the
rest of the pipeline is database-agnostic. Switch backends by changing
VECTOR_STORE_TYPE in .env — no code changes required.

Design notes:
  • ChromaDB: embeds are stored with their text content and metadata.
  • All IDs are the chunk_id assigned during chunking.
  • upsert semantics: add_documents is idempotent (same ID = overwrite).
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from sqlalchemy import select

from src.utils.exceptions import (
    CollectionNotFoundError,
    DocumentNotFoundError,
    VectorStoreConnectionError,
    VectorStoreError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── ChromaDB Backend ─────────────────────────────────────────────────

class ChromaVectorStore:
    """
    Local persistent ChromaDB vector store.

    Args:
        persist_dir:     Directory where Chroma stores its SQLite + data files.
        collection_name: Name of the collection (table equivalent).
    """

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def connect(self) -> None:
        """Initialise the ChromaDB client and get/create the collection."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                # Cosine similarity is appropriate for normalized embeddings
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB connected",
                collection=self.collection_name,
                persist_dir=self.persist_dir,
                count=self._collection.count(),
            )
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Cannot connect to ChromaDB: {exc}",
                details={"persist_dir": self.persist_dir},
            ) from exc

    def _ensure_connected(self):
        if self._collection is None:
            self.connect()
        return self._collection

    def add_documents(
        self,
        chunks: List[Document],
        embeddings: List[List[float]],
    ) -> List[str]:
        """
        Upsert chunks + embeddings into the collection.

        Returns list of stored IDs.
        """
        if not chunks:
            return []

        collection = self._ensure_connected()

        ids = [c.metadata.get("chunk_id", f"chunk-{i}") for i, c in enumerate(chunks)]
        texts = [c.page_content for c in chunks]
        metadatas = []
        for c in chunks:
            # Chroma metadata values must be str/int/float/bool
            meta = {
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in c.metadata.items()
            }
            metadatas.append(meta)

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info("Upserted chunks to ChromaDB", count=len(ids))
            return ids
        except Exception as exc:
            raise VectorStoreError(
                f"ChromaDB upsert failed: {exc}",
                details={"num_chunks": len(chunks)},
            ) from exc

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Return top_k most similar chunks as (Document, score) tuples.
        Score is cosine similarity (higher = more similar).
        """
        collection = self._ensure_connected()

        query_params: Dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            query_params["where"] = filter_metadata

        try:
            results = collection.query(**query_params)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

        docs_with_scores: List[Tuple[Document, float]] = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Chroma returns cosine *distance* (0 = identical); convert to similarity
            score = 1.0 - dist
            docs_with_scores.append((Document(page_content=text, metadata=meta), score))

        return docs_with_scores

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks whose metadata['source'] matches source."""
        collection = self._ensure_connected()
        try:
            results = collection.get(where={"source": source}, include=["metadatas"])
            ids_to_delete = results["ids"]
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info("Deleted chunks", source=source, count=len(ids_to_delete))
            return len(ids_to_delete)
        except Exception as exc:
            raise VectorStoreError(f"Delete failed: {exc}") from exc

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self._ensure_connected().count()

    def list_sources(self) -> List[str]:
        """Return unique source filenames stored in the collection."""
        collection = self._ensure_connected()
        try:
            all_meta = collection.get(include=["metadatas"])["metadatas"]
            return list({m.get("filename", "unknown") for m in all_meta})
        except Exception:
            return []


# ── pgvector Backend ─────────────────────────────────────────────────

class PgVectorStore:
    """
    Postgres + pgvector vector store, using the Document/Embedding tables.

    Unlike ChromaVectorStore, every chunk is tied to a Document row, which
    is tied to a User — so retrieval, ingestion, and deletion are all
    scoped by user_id. Each call opens its own short-lived DB session
    (the store itself is a stateless singleton held by the pipeline).
    """

    def connect(self) -> None:
        """No-op — kept for interface parity with ChromaVectorStore."""
        logger.info("PgVectorStore ready (connections opened per-call)")

    def add_documents(
        self,
        chunks: List[Document],
        embeddings: List[List[float]],
        *,
        user_id: uuid.UUID,
        filename: str,
        file_type: str,
        file_size_bytes: int,
    ) -> List[str]:
        """Create a Document row, then one Embedding row per chunk."""
        if not chunks:
            return []

        from src.db.connection import SessionLocal
        from src.db.models import Document as DocumentRow
        from src.db.models import Embedding as EmbeddingRow

        db = SessionLocal()
        try:
            doc_row = DocumentRow(
                user_id=user_id,
                filename=filename,
                file_type=file_type,
                file_size_bytes=file_size_bytes,
                status="ready",
            )
            db.add(doc_row)
            db.flush()  # assigns doc_row.id without committing yet

            ids: List[str] = []
            for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                emb_row = EmbeddingRow(
                    document_id=doc_row.id,
                    chunk_index=i,
                    chunk_text=chunk.page_content,
                    vector=vector,
                )
                db.add(emb_row)
                db.flush()
                ids.append(str(emb_row.id))

            db.commit()
            logger.info("Upserted chunks to pgvector", count=len(ids), document_id=str(doc_row.id))
            return ids
        except Exception as exc:
            db.rollback()
            raise VectorStoreError(
                f"pgvector insert failed: {exc}", details={"num_chunks": len(chunks)}
            ) from exc
        finally:
            db.close()

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filter_metadata: Optional[Dict] = None,
        *,
        user_id: Optional[uuid.UUID] = None,
    ) -> List[Tuple[Document, float]]:
        """Cosine-similarity search, optionally scoped to one user's documents."""
        from src.db.connection import SessionLocal
        from src.db.models import Document as DocumentRow
        from src.db.models import Embedding as EmbeddingRow

        db = SessionLocal()
        try:
            distance = EmbeddingRow.vector.cosine_distance(query_embedding).label("distance")
            stmt = (
                select(EmbeddingRow, DocumentRow.filename, distance)
                .join(DocumentRow, EmbeddingRow.document_id == DocumentRow.id)
            )
            if user_id is not None:
                stmt = stmt.where(DocumentRow.user_id == user_id)
            if filter_metadata and "filename" in filter_metadata:
                stmt = stmt.where(DocumentRow.filename == filter_metadata["filename"])
            stmt = stmt.order_by(distance).limit(top_k)

            rows = db.execute(stmt).all()
        except Exception as exc:
            raise VectorStoreError(f"pgvector query failed: {exc}") from exc
        finally:
            db.close()

        docs_with_scores: List[Tuple[Document, float]] = []
        for emb_row, filename, dist in rows:
            # pgvector's cosine_distance = 1 - cosine_similarity
            score = 1.0 - float(dist)
            meta = {
                "filename": filename,
                "source": filename,
                "chunk_id": str(emb_row.id),
                "chunk_index": emb_row.chunk_index,
            }
            docs_with_scores.append((Document(page_content=emb_row.chunk_text, metadata=meta), score))

        return docs_with_scores

    def delete_by_source(self, source: str, *, user_id: Optional[uuid.UUID] = None) -> int:
        """Delete a Document (and cascade its Embeddings) by filename, optionally scoped to a user."""
        from src.db.connection import SessionLocal
        from src.db.models import Document as DocumentRow

        db = SessionLocal()
        try:
            q = db.query(DocumentRow).filter(DocumentRow.filename == source)
            if user_id is not None:
                q = q.filter(DocumentRow.user_id == user_id)
            doc_rows = q.all()

            total_chunks = sum(len(d.embeddings) for d in doc_rows)
            for d in doc_rows:
                db.delete(d)  # cascades to embeddings via relationship cascade
            db.commit()

            logger.info("Deleted chunks from pgvector", source=source, count=total_chunks)
            return total_chunks
        except Exception as exc:
            db.rollback()
            raise VectorStoreError(f"pgvector delete failed: {exc}") from exc
        finally:
            db.close()

    def count(self, *, user_id: Optional[uuid.UUID] = None) -> int:
        """Total number of stored chunks, optionally scoped to a user."""
        from src.db.connection import SessionLocal
        from src.db.models import Document as DocumentRow
        from src.db.models import Embedding as EmbeddingRow

        db = SessionLocal()
        try:
            q = db.query(EmbeddingRow)
            if user_id is not None:
                q = q.join(DocumentRow, EmbeddingRow.document_id == DocumentRow.id).filter(
                    DocumentRow.user_id == user_id
                )
            return q.count()
        finally:
            db.close()

    def list_sources(self, *, user_id: Optional[uuid.UUID] = None) -> List[str]:
        """Unique filenames currently indexed, optionally scoped to a user."""
        from src.db.connection import SessionLocal
        from src.db.models import Document as DocumentRow

        db = SessionLocal()
        try:
            q = db.query(DocumentRow.filename).distinct()
            if user_id is not None:
                q = q.filter(DocumentRow.user_id == user_id)
            return [row[0] for row in q.all()]
        finally:
            db.close()


# ── Factory ──────────────────────────────────────────────────────────

def get_vector_store(
    store_type: str = "chroma",
    persist_dir: str = "./vector_store/chroma_db",
    collection_name: str = "documind_collection",
):
    """
    Factory function returning the appropriate vector store backend.

    Supports 'chroma' (local dev, file-based) and 'pgvector' (Postgres,
    the production backend — scoped by user_id).
    """
    if store_type == "chroma":
        store = ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
        store.connect()
        return store
    elif store_type == "pgvector":
        store = PgVectorStore()
        store.connect()
        return store
    else:
        raise VectorStoreError(
            f"Unsupported vector store type: '{store_type}'. Use 'chroma' or 'pgvector'.",
        )
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

from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

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


# ── Factory ──────────────────────────────────────────────────────────

def get_vector_store(
    store_type: str = "chroma",
    persist_dir: str = "./vector_store/chroma_db",
    collection_name: str = "documind_collection",
) -> ChromaVectorStore:
    """
    Factory function returning the appropriate vector store backend.

    Currently only ChromaDB is supported. Pinecone support can be added
    by extending this factory with a PineconeVectorStore class.
    """
    if store_type == "chroma":
        store = ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
        store.connect()
        return store
    else:
        raise VectorStoreError(
            f"Unsupported vector store type: '{store_type}'. Use 'chroma'.",
        )

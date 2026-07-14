"""
DocuMind Semantic Retriever
----------------------------
Combines embedding generation + vector store lookup to fetch the most
relevant document chunks for a user query.

The retriever returns more candidates (top_k) than the final answer needs
so the re-ranker can select the best subset (top_n).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.embeddings.embedder import EmbeddingGenerator
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticRetriever:
    """
    Retrieves the most relevant chunks for a query using cosine similarity.

    Args:
        vector_store:   Initialised ChromaVectorStore instance.
        embedder:       Loaded EmbeddingGenerator instance.
        top_k:          Number of candidates to retrieve from the vector store.
        score_threshold: Minimum similarity score (0–1) to include a chunk.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: EmbeddingGenerator,
        top_k: int = 20,
        score_threshold: float = 0.3,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
        user_id=None,
    ) -> List[Tuple[Document, float]]:
        """
        Embed the query and retrieve similar chunks from the vector store.

        Args:
            query:           User's natural-language question.
            top_k:           Override default top_k for this call.
            filter_metadata: Optional Chroma where-clause (e.g. {"filename": "x.pdf"}).

        Returns:
            List of (Document, similarity_score) sorted by score desc,
            filtered by score_threshold.

        Raises:
            RetrievalError: If embedding or lookup fails.
        """
        if not query.strip():
            raise RetrievalError("Query must not be empty.")

        k = top_k or self.top_k
        logger.info("Retrieving chunks", query=query[:80], top_k=k)

        try:
            query_vec = self.embedder.embed_query(query)
        except Exception as exc:
            raise RetrievalError(
                f"Query embedding failed: {exc}",
                details={"query": query[:80]},
            ) from exc

        try:
            search_kwargs = {}
            if user_id is not None:
                search_kwargs["user_id"] = user_id
            results = self.vector_store.similarity_search(
                query_embedding=query_vec,
                top_k=k,
                filter_metadata=filter_metadata,
                **search_kwargs,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Vector store lookup failed: {exc}",
                details={"top_k": k},
            ) from exc

        # Apply score threshold
        filtered = [(doc, score) for doc, score in results if score >= self.score_threshold]

        logger.info(
            "Retrieval complete",
            total_results=len(results),
            after_threshold=len(filtered),
            threshold=self.score_threshold,
        )
        return filtered
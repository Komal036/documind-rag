"""
DocuMind Cross-Encoder Re-Ranker
----------------------------------
Takes the top_k candidates from the semantic retriever and re-scores them
using a cross-encoder model, which reads the query and each passage jointly
(much more accurate than bi-encoder similarity alone).

Model default: cross-encoder/ms-marco-MiniLM-L-6-v2
  — fast (~50ms for 20 candidates on CPU), strong MRR on MS MARCO.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.utils.exceptions import RerankerError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks retrieval candidates using a cross-encoder model.

    The cross-encoder scores each (query, passage) pair jointly, giving
    much higher accuracy than cosine similarity at the cost of more compute.
    It is applied only to the small candidate set (top_k ≤ 20) so latency
    remains acceptable (~50–200 ms on CPU).

    Args:
        model_name: HuggingFace model identifier for a cross-encoder.
        top_n:      Number of chunks to return after re-ranking.
        device:     "cpu" | "cuda". Auto-detected if None.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self._device = device
        self._model = None

    def _load(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return

        import torch
        from sentence_transformers import CrossEncoder

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading cross-encoder", model=self.model_name, device=device)
        try:
            self._model = CrossEncoder(self.model_name, device=device)
            logger.info("Cross-encoder loaded", model=self.model_name)
        except Exception as exc:
            raise RerankerError(
                f"Failed to load cross-encoder '{self.model_name}': {exc}"
            ) from exc

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Re-score and sort candidates by cross-encoder relevance.

        Args:
            query:      User's original query string.
            candidates: List of (Document, bi-encoder-score) from retriever.
            top_n:      Override the instance-level top_n for this call.

        Returns:
            Sorted list of (Document, cross_encoder_score), length ≤ top_n.

        Raises:
            RerankerError: If model inference fails.
        """
        if not candidates:
            return []

        n = top_n or self.top_n
        self._load()

        pairs = [(query, doc.page_content) for doc, _ in candidates]

        try:
            scores = self._model.predict(pairs)  # returns numpy array
        except Exception as exc:
            raise RerankerError(
                f"Cross-encoder inference failed: {exc}",
                details={"num_candidates": len(candidates)},
            ) from exc

        # Zip scores with original docs, sort descending, take top_n
        scored = sorted(
            zip([c[0] for c in candidates], scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:n]

        # Annotate metadata with reranker score for transparency
        results = []
        for doc, score in scored:
            doc.metadata["reranker_score"] = round(float(score), 4)
            results.append((doc, float(score)))

        logger.info(
            "Re-ranking complete",
            input_candidates=len(candidates),
            output_chunks=len(results),
        )
        return results

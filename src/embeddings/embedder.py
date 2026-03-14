"""
DocuMind Embedding Generator
-----------------------------
Wraps sentence-transformers to produce dense vector embeddings for
document chunks and query strings.

The model is loaded once (singleton pattern) to avoid repeated disk I/O.
All heavy ML logic is isolated here so the rest of the codebase stays clean.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

from src.utils.exceptions import EmbeddingError, EmbeddingModelNotLoadedError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Singleton-style wrapper around a SentenceTransformer model.

    Usage:
        gen = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectors = gen.embed_texts(["Hello world", "Another sentence"])
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        # Auto-detect GPU; fall back to CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[SentenceTransformer] = None

    # ── Lifecycle ────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the SentenceTransformer model into memory."""
        logger.info(
            "Loading embedding model",
            model=self.model_name,
            device=self.device,
        )
        try:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device,
            )
            logger.info("Embedding model loaded", model=self.model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

    def _ensure_loaded(self) -> SentenceTransformer:
        """Lazily load the model if not already in memory."""
        if self._model is None:
            self.load()
        if self._model is None:  # pragma: no cover
            raise EmbeddingModelNotLoadedError("Model failed to load.")
        return self._model

    # ── Core API ─────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of strings into dense float vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors (one per input string).

        Raises:
            EmbeddingError: If encoding fails.
        """
        if not texts:
            return []

        model = self._ensure_loaded()
        logger.debug("Embedding texts", count=len(texts), batch_size=self.batch_size)

        try:
            # show_progress_bar=False keeps logs clean in production
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # cosine similarity via dot product
            )
            return embeddings.tolist()
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding generation failed: {exc}",
                details={"num_texts": len(texts)},
            ) from exc

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string. Returns one float vector.

        Slightly different prompt prefix is used by some models (e.g., E5)
        to distinguish queries from documents — handled transparently here.
        """
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        model = self._ensure_loaded()
        return model.get_sentence_embedding_dimension()


# ── Module-level singleton ────────────────────────────────────────────
# Instantiated lazily; call get_embedding_generator() throughout the app.

_generator_instance: Optional[EmbeddingGenerator] = None


def get_embedding_generator(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: Optional[str] = None,
    batch_size: int = 64,
) -> EmbeddingGenerator:
    """
    Return the process-wide EmbeddingGenerator singleton.
    On first call it creates and loads the model.
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = EmbeddingGenerator(
            model_name=model_name,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )
    return _generator_instance

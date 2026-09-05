"""
DocuMind Embedding Generator
-----------------------------
Wraps fastembed to produce dense vector embeddings for
document chunks and query strings without PyTorch.

The model is loaded once (singleton pattern) to avoid repeated disk I/O.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np

from fastembed import TextEmbedding
from src.utils.exceptions import EmbeddingError, EmbeddingModelNotLoadedError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Singleton-style wrapper around a FastEmbed model.

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
        self._model: Optional[TextEmbedding] = None

    def load(self) -> None:
        """Load the FastEmbed model into memory."""
        logger.info(
            "Loading ONNX embedding model",
            model=self.model_name,
        )
        try:
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=1,
            )
            logger.info("ONNX Embedding model loaded", model=self.model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

    def _ensure_loaded(self) -> TextEmbedding:
        """Lazily load the model if not already in memory."""
        if self._model is None:
            self.load()
        if self._model is None:
            raise EmbeddingModelNotLoadedError("Model failed to load.")
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        model = self._ensure_loaded()
        logger.debug("Embedding texts", count=len(texts), batch_size=self.batch_size)

        try:
            # fastembed returns a generator of numpy arrays
            embeddings_gen = model.embed(
                texts,
                batch_size=self.batch_size,
            )
            embeddings = list(embeddings_gen)
            
            # normalize embeddings to match sentence-transformers cosine similarity via dot product
            normalized = []
            for emb in embeddings:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    normalized.append((emb / norm).tolist())
                else:
                    normalized.append(emb.tolist())
            return normalized
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding generation failed: {exc}",
                details={"num_texts": len(texts)},
            ) from exc

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        return 384  # Hardcoded for all-MiniLM-L6-v2 as fastembed doesn't have a direct dimension property exposed


_generator_instance: Optional[EmbeddingGenerator] = None

def get_embedding_generator(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir: Optional[str] = None,
    batch_size: int = 64,
) -> EmbeddingGenerator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = EmbeddingGenerator(
            model_name=model_name,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )
    return _generator_instance

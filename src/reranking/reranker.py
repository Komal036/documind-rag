"""
DocuMind Cross-Encoder Re-Ranker
----------------------------------
Pure ONNX implementation of cross-encoder scoring to avoid PyTorch memory bloat.
Takes the top_k candidates from the semantic retriever and re-scores them
using a cross-encoder model.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple
import numpy as np

from huggingface_hub import snapshot_download
import onnxruntime as ort
from tokenizers import Tokenizer
from langchain_core.documents import Document

from src.utils.exceptions import RerankerError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks retrieval candidates using an ONNX cross-encoder model.

    Args:
        model_name: HuggingFace model identifier (defaults to ms-marco-MiniLM-L-6-v2)
        top_n:      Number of chunks to return after re-ranking.
        device:     Ignored in ONNX CPU mode, kept for compatibility.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        device: Optional[str] = None,
    ) -> None:
        # map sentence-transformers name to Xenova's ONNX repo
        if model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2":
            self.model_name = "Xenova/ms-marco-MiniLM-L-6-v2"
        else:
            self.model_name = model_name
            
        self.top_n = top_n
        self._session = None
        self._tokenizer = None

    def _load(self):
        """Lazy-load the ONNX cross-encoder model and tokenizer."""
        if self._session is not None:
            return

        logger.info("Loading ONNX cross-encoder", model=self.model_name)
        try:
            # Download weights and config from HF Hub
            model_path = snapshot_download(repo_id=self.model_name)
            
            onnx_path = os.path.join(model_path, "onnx", "model.onnx")
            if not os.path.exists(onnx_path):
                # Fallback to root directory if 'onnx' subdir doesn't exist
                onnx_path = os.path.join(model_path, "model.onnx")
                
            # Initialize ONNX Runtime session
            self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            
            # Initialize Rust-based tokenizer
            tokenizer_path = os.path.join(model_path, "tokenizer.json")
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_truncation(max_length=512)
            self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            
            logger.info("ONNX Cross-encoder loaded", model=self.model_name)
        except Exception as exc:
            raise RerankerError(
                f"Failed to load ONNX cross-encoder '{self.model_name}': {exc}"
            ) from exc

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Re-score and sort candidates by cross-encoder relevance using ONNX.
        """
        if not candidates:
            return []

        n = top_n or self.top_n
        self._load()

        # Cross-encoder evaluates (query, document) pairs jointly
        pairs = [(query, doc.page_content) for doc, _ in candidates]

        try:
            # Tokenize all pairs in a batch
            encoded = self._tokenizer.encode_batch(pairs)
            
            # Prepare inputs for ONNX format
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)
            
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            
            # Run inference
            ort_outs = self._session.run(None, ort_inputs)
            logits = ort_outs[0]
            
            # Flatten to get raw scores for each candidate
            scores = logits.flatten()
            
        except Exception as exc:
            raise RerankerError(
                f"Cross-encoder ONNX inference failed: {exc}",
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
            "Re-ranking complete (ONNX)",
            input_candidates=len(candidates),
            output_chunks=len(results),
        )
        return results

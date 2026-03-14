"""
DocuMind Text Chunking
-----------------------
Splits loaded Documents into overlapping chunks suitable for embedding.

Strategy used: RecursiveCharacterTextSplitter from LangChain, which respects
paragraph → sentence → word boundaries before doing hard character splits.

Each output chunk retains the full metadata of its source Document plus:
    chunk_index   — position of the chunk within the document
    chunk_id      — deterministic ID (sha256[:8]-chunk_index)
    char_start    — not tracked by LangChain; reserved for future use
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.exceptions import ChunkingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_length: int = 50,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks for embedding.

    Args:
        documents:        Raw documents (typically one per page or file).
        chunk_size:       Maximum characters per chunk.
        chunk_overlap:    Overlap between consecutive chunks to preserve context.
        min_chunk_length: Chunks shorter than this are discarded as noise.

    Returns:
        List[Document] with updated metadata (chunk_index, chunk_id).

    Raises:
        ChunkingError: If splitting fails for any reason.
    """
    if not documents:
        logger.warning("chunk_documents called with empty document list")
        return []

    logger.info(
        "Starting chunking",
        documents=len(documents),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators try paragraph → line → sentence → word → char
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    try:
        raw_chunks: List[Document] = splitter.split_documents(documents)
    except Exception as exc:
        raise ChunkingError(
            f"Text splitting failed: {exc}",
            details={"num_docs": len(documents)},
        ) from exc

    # Filter noise and assign deterministic IDs
    chunks: List[Document] = []
    for idx, chunk in enumerate(raw_chunks):
        text = chunk.page_content.strip()
        if len(text) < min_chunk_length:
            logger.debug("Discarding short chunk", length=len(text))
            continue

        sha_prefix = chunk.metadata.get("sha256", "unknown")[:8]
        chunk.page_content = text
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{sha_prefix}-{idx}"
        chunks.append(chunk)

    logger.info(
        "Chunking complete",
        raw_chunks=len(raw_chunks),
        kept_chunks=len(chunks),
    )
    return chunks

"""
DocuMind Custom Exception Hierarchy
-------------------------------------
Defining a clean exception hierarchy serves two purposes:
  1. API layer can catch specific exceptions and return the right HTTP status
  2. Callers know exactly what can go wrong without reading implementation

Convention: all DocuMind exceptions inherit from DocuMindError so you can
always catch the base class if you want a catch-all.
"""


class DocuMindError(Exception):
    """Base exception for all DocuMind errors."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


# ── Ingestion Errors ──────────────────────────────────────────────────
class DocumentLoadError(DocuMindError):
    """Raised when a document cannot be opened or read."""


class UnsupportedFileTypeError(DocuMindError):
    """Raised when the uploaded file type is not supported."""


class FileTooLargeError(DocuMindError):
    """Raised when an uploaded file exceeds the size limit."""


class DocumentParsingError(DocuMindError):
    """Raised when text extraction from a document fails."""


# ── Chunking Errors ───────────────────────────────────────────────────
class ChunkingError(DocuMindError):
    """Raised when the chunking pipeline encounters an error."""


# ── Embedding Errors ──────────────────────────────────────────────────
class EmbeddingError(DocuMindError):
    """Raised when embedding generation fails."""


class EmbeddingModelNotLoadedError(DocuMindError):
    """Raised when attempting to embed before the model is initialised."""


# ── Vector Store Errors ───────────────────────────────────────────────
class VectorStoreError(DocuMindError):
    """Base class for vector store errors."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when the vector store cannot be reached."""


class DocumentNotFoundError(VectorStoreError):
    """Raised when a requested document ID does not exist in the store."""


class CollectionNotFoundError(VectorStoreError):
    """Raised when a named collection does not exist."""


# ── Retrieval Errors ──────────────────────────────────────────────────
class RetrievalError(DocuMindError):
    """Raised when the semantic retrieval step fails."""


class RerankerError(DocuMindError):
    """Raised when the cross-encoder re-ranker fails."""


# ── Generation Errors ─────────────────────────────────────────────────
class LLMError(DocuMindError):
    """Raised when the LLM API call fails."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM API rate limit is exceeded."""


class LLMContextTooLongError(LLMError):
    """Raised when the assembled context exceeds the model's context window."""


class PromptRenderError(DocuMindError):
    """Raised when a prompt template cannot be rendered."""


# ── API Errors ────────────────────────────────────────────────────────
class AuthenticationError(DocuMindError):
    """Raised when an API request fails authentication."""


class RateLimitError(DocuMindError):
    """Raised when a client exceeds the API rate limit."""


class ValidationError(DocuMindError):
    """Raised when request validation fails beyond Pydantic's checks."""

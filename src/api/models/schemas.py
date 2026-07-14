"""
DocuMind API Data Models
-------------------------
Pydantic models for all FastAPI request/response schemas.
Separating models here keeps route handlers clean.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Shared ────────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    citation_number: int
    filename: str
    page: Any  # could be int or "?"
    chunk_id: str
    relevance_score: float
    preview: str


class ErrorResponse(BaseModel):
    error: str
    details: Dict = Field(default_factory=dict)
    request_id: Optional[str] = None


# ── Auth ──────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: str = Field(..., examples=["komal@example.com"])
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(default=None)


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    is_active: bool


# ── Ingest ────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    filename: str
    sha256: str
    chunk_count: int
    status: str
    message: str


# ── Query ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's question",
        examples=["What is the remote work policy?"],
    )
    use_reranker: bool = Field(
        default=True,
        description="Apply cross-encoder re-ranking for higher accuracy (slightly slower)",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Override number of retrieval candidates",
    )
    top_n: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Override number of re-ranked chunks passed to LLM",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Chat session ID for multi-turn memory. Create one via POST /api/v1/chat/sessions.",
    )


class ChatSessionResponse(BaseModel):
    session_id: str
    title: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    model: str
    provider: str
    chunks_used: int
    question: str


# ── Stats ─────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_chunks: int
    indexed_files: List[str]
    embedding_model: str
    llm_model: str
    vector_store: str


# ── Delete ────────────────────────────────────────────────────────────

class DeleteResponse(BaseModel):
    deleted_chunks: int
    source: str
    message: str


# ── Health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    environment: str
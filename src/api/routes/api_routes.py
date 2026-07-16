"""
DocuMind API Routes
--------------------
All REST endpoints for the DocuMind RAG system.

Endpoints:
  GET  /health                  — liveness probe
  POST /api/v1/ingest           — upload and index a document
  POST /api/v1/query            — ask a question against indexed docs
  GET  /api/v1/stats            — vector store statistics
  DELETE /api/v1/documents/{filename} — remove a document from the index
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.auth.dependencies import get_current_user
from src.db.connection import get_db
from src.db.models import ChatSession, Message, User
from src.memory.chat_memory import append_turn, get_recent_history
from src.api.models.schemas import (
    DeleteResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from src.pipeline import RAGPipeline
from src.utils.config import get_settings
from src.utils.exceptions import (
    DocuMindError,
    FileTooLargeError,
    LLMError,
    RetrievalError,
    UnsupportedFileTypeError,
)
from src.utils.file_utils import validate_file
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
settings = get_settings()


# ── Dependency: get pipeline from app state ───────────────────────────

def get_pipeline(request: Request) -> RAGPipeline:
    """FastAPI dependency that returns the pipeline from app state."""
    return request.app.state.pipeline


# ── Health ────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe — returns 200 if the service is running."""
    return HealthResponse(
        status="healthy",
        environment=settings.environment,
    )


# ── Ingest ────────────────────────────────────────────────────────────

@router.post(
    "/api/v1/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Upload and index a document",
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF, TXT, DOCX, or MD file"),
    pipeline: RAGPipeline = Depends(get_pipeline),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a document, chunk it, embed it, and store it in the vector DB.
    Duplicate uploads (same content) are handled gracefully via upsert.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info("Ingest request", filename=file.filename, request_id=request_id)

    # Validate file type upfront
    suffix = Path(file.filename).suffix.lower()
    allowed = settings.api.allowed_file_types_list
    if suffix.lstrip(".") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{suffix}' is not supported. Allowed: {allowed}",
        )

    # Stream upload to a temp file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Size check
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
        max_mb = settings.api.max_upload_size_mb
        if size_mb > max_mb:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size {size_mb:.1f} MB exceeds {max_mb} MB limit.",
            )

        try:
            result = pipeline.ingest_document(tmp_path, user_id=current_user.id)
        except (UnsupportedFileTypeError, FileTooLargeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except DocuMindError as exc:
            logger.error("Ingestion failed", error=str(exc), request_id=request_id)
            raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(
        filename=result["filename"],
        sha256=result["sha256"],
        chunk_count=result["chunk_count"],
        status=result["status"],
        message=f"Successfully indexed {result['chunk_count']} chunks from '{result['filename']}'.",
    )


# ── Query ─────────────────────────────────────────────────────────────

@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["Q&A"],
    summary="Ask a question against indexed documents",
)
async def query_documents(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve relevant passages and generate a grounded, cited answer.

    If session_id is provided, prior turns in that session are used as
    conversation context, and this exchange is appended to it.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info("Query request", question=body.question[:80], request_id=request_id)

    chat_history = None
    session_uuid = None
    if body.session_id:
        try:
            session_uuid = uuid.UUID(body.session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid session_id format.")

        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_uuid, ChatSession.user_id == current_user.id)
            .first()
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found.")

        chat_history = get_recent_history(session_uuid)

    try:
        if body.use_self_rag:
            # v2 path: confidence-gated retrieval loop. Chat history isn't
            # threaded into this path yet — it's evaluated as a single-shot
            # retrieval strategy for now, kept separate from v1 for a clean
            # RAGAS comparison.
            result = pipeline.self_rag_query(question=body.question, user_id=current_user.id)
        else:
            result = pipeline.query(
                question=body.question,
                use_reranker=body.use_reranker,
                top_k=body.top_k,
                top_n=body.top_n,
                user_id=current_user.id,
                chat_history=chat_history,
            )
    except RetrievalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except DocuMindError as exc:
        logger.error("Query failed", error=str(exc), request_id=request_id)
        raise HTTPException(status_code=500, detail=str(exc))

    if session_uuid is not None:
        append_turn(session_uuid, "user", body.question)
        append_turn(session_uuid, "assistant", result["answer"])
        db.add(Message(session_id=session_uuid, role="user", content=body.question))
        db.add(Message(session_id=session_uuid, role="assistant", content=result["answer"]))
        db.commit()

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        model=result["model"],
        provider=result["provider"],
        chunks_used=result["chunks_used"],
        question=body.question,
        self_rag_retries=result.get("self_rag_retries"),
        self_rag_reformulations=result.get("self_rag_reformulations"),
    )


# ── Stats ─────────────────────────────────────────────────────────────

@router.get(
    "/api/v1/stats",
    response_model=StatsResponse,
    tags=["System"],
    summary="Vector store and pipeline statistics",
)
async def get_stats(
    pipeline: RAGPipeline = Depends(get_pipeline),
    current_user: User = Depends(get_current_user),
):
    """Returns current statistics about indexed documents and the pipeline."""
    stats = pipeline.get_stats(user_id=current_user.id)
    return StatsResponse(**stats)


# ── Delete ────────────────────────────────────────────────────────────

@router.delete(
    "/api/v1/documents/{filename}",
    response_model=DeleteResponse,
    tags=["Documents"],
    summary="Remove a document from the index",
)
async def delete_document(
    filename: str,
    pipeline: RAGPipeline = Depends(get_pipeline),
    current_user: User = Depends(get_current_user),
):
    """Delete all chunks associated with a given filename from the vector DB."""
    try:
        result = pipeline.delete_document(source=filename, user_id=current_user.id)
    except DocuMindError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if result["deleted_chunks"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for '{filename}'. Is the file indexed?",
        )

    return DeleteResponse(
        deleted_chunks=result["deleted_chunks"],
        source=result["source"],
        message=f"Removed {result['deleted_chunks']} chunks for '{filename}'.",
    )
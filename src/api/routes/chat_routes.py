"""
DocuMind Chat Routes
----------------------
  POST /api/v1/chat/sessions — create a new chat session for the current user
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.models.schemas import ChatSessionResponse
from src.auth.dependencies import get_current_user
from src.db.connection import get_db
from src.db.models import ChatSession, User

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post("/sessions", response_model=ChatSessionResponse, status_code=201)
async def create_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new chat session. Pass the returned session_id to /api/v1/query for memory."""
    session = ChatSession(user_id=current_user.id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return ChatSessionResponse(session_id=str(session.id), title=session.title)
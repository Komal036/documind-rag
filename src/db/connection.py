"""
DocuMind Database Connection
-----------------------------
Engine and session management for PostgreSQL + pgvector.
Mirrors the pattern used by src/utils/config.py: settings-driven, no
hard-coded credentials.
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.utils.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database.database_url,
    pool_size=settings.database.db_pool_size,
    max_overflow=settings.database.db_max_overflow,
    echo=settings.database.db_echo,
    pool_pre_ping=True,  # avoids stale-connection errors after idle periods
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a DB session and guarantees it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
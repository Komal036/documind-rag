"""
DocuMind FastAPI Application
-----------------------------
Main application entry point. Configures:
  • CORS middleware
  • Global exception handlers
  • Startup / shutdown lifecycle (model loading)
  • Router registration
  • OpenAPI docs metadata

Run with:
    uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes.api_routes import router
from src.pipeline import RAGPipeline
from src.utils.config import get_settings
from src.utils.exceptions import DocuMindError
from src.utils.logger import get_logger, setup_logging

settings = get_settings()

# Initialise logging as early as possible
setup_logging(
    log_level=settings.logging.log_level,
    log_format=settings.logging.log_format,
    log_file=settings.logging.log_file,
    is_development=settings.is_development(),
)

logger = get_logger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Loads models and connects to the vector store before serving requests,
    and cleanly shuts down on exit.
    """
    logger.info("DocuMind API starting up…", environment=settings.environment)

    # Initialise the RAG pipeline once at startup
    pipeline = RAGPipeline()
    pipeline.initialize()
    app.state.pipeline = pipeline

    logger.info("DocuMind API is ready to serve requests.")
    yield

    # Shutdown
    logger.info("DocuMind API shutting down.")


# ── Application ───────────────────────────────────────────────────────

app = FastAPI(
    title="DocuMind — Enterprise RAG API",
    description=(
        "Upload documents and ask questions. "
        "Powered by sentence-transformers + ChromaDB + GPT-4o."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ─────────────────────────────────────────

@app.exception_handler(DocuMindError)
async def documind_exception_handler(request: Request, exc: DocuMindError):
    logger.error("Unhandled DocuMindError", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": exc.message, "details": exc.details},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An unexpected error occurred.", "details": {}},
    )


# ── Routes ────────────────────────────────────────────────────────────

app.include_router(router)


# ── Dev runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api.api_host,
        port=settings.api.api_port,
        reload=settings.api.api_reload,
    )

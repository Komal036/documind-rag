"""
DocuMind Structured Logging
----------------------------
Built on loguru for structured JSON (production) or
human-readable coloured output (development).

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Document indexed", doc_id="abc123", chunks=42)
"""
import sys
from pathlib import Path

from loguru import logger as _loguru_logger


def setup_logging(log_level: str = "INFO", log_format: str = "console",
                  log_file: str = "logs/documind.log", is_development: bool = True) -> None:
    """Configure loguru sinks. Called once at application startup."""
    _loguru_logger.remove()

    if log_format == "json":
        fmt = ('{"time":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
               '"level":"{level}","name":"{name}","message":"{message}"}')
    else:
        fmt = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>")

    _loguru_logger.add(
        sys.stderr,
        format=fmt,
        level=log_level,
        colorize=(log_format == "console"),
        backtrace=True,
        diagnose=is_development,
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _loguru_logger.add(
        str(log_path),
        format=("{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level: <8} | "
                "{name}:{line} | {message}"),
        level=log_level,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        backtrace=True,
        diagnose=False,  # Never dump locals to file — may contain secrets
    )


def get_logger(name: str):
    """
    Return a named logger. Call at module top-level:
        logger = get_logger(__name__)
    """
    return _loguru_logger.bind(name=name)


def get_request_logger(name: str, request_id: str, user_id: str = "anonymous"):
    """Pre-bind request context for full request tracing in API handlers."""
    return _loguru_logger.bind(name=name, request_id=request_id, user_id=user_id)

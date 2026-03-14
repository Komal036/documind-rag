"""
DocuMind File Utilities
------------------------
Helpers for safe file handling across the ingestion pipeline.
"""
import hashlib
import mimetypes
import os
import shutil
from pathlib import Path

from src.utils.exceptions import FileTooLargeError, UnsupportedFileTypeError
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MIME_TO_EXT = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "text/markdown": ".md",
}


def validate_file(file_path: Path, max_size_mb: int = 50) -> None:
    """
    Validate that a file exists, is of a supported type, and within size limit.
    Raises UnsupportedFileTypeError or FileTooLargeError on failure.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"File type '{ext}' is not supported.",
            details={"supported": list(SUPPORTED_EXTENSIONS), "received": ext},
        )

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise FileTooLargeError(
            f"File size {size_mb:.1f} MB exceeds the {max_size_mb} MB limit.",
            details={"size_mb": round(size_mb, 2), "limit_mb": max_size_mb},
        )

    logger.debug("File validated", path=str(file_path), size_mb=round(size_mb, 2))


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA-256 hash of a file.
    Used to detect duplicate uploads and enable cache invalidation.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def safe_copy(src: Path, dest_dir: Path, overwrite: bool = False) -> Path:
    """
    Copy a file to dest_dir safely, returning the destination path.
    Creates dest_dir if it does not exist.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists() and not overwrite:
        logger.warning("File already exists, skipping copy", dest=str(dest))
        return dest
    shutil.copy2(src, dest)
    logger.info("File copied", src=str(src), dest=str(dest))
    return dest


def get_file_metadata(file_path: Path) -> dict:
    """Return a dict of useful file metadata."""
    stat = file_path.stat()
    return {
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 3),
        "sha256": compute_file_hash(file_path),
    }


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path

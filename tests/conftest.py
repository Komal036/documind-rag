"""
Shared pytest fixtures for DocuMind tests.
Fixtures here are available to ALL tests without importing.
"""
import os
import tempfile
from pathlib import Path

import pytest

# ── Ensure test environment never hits real APIs ──────────────────────
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("VECTOR_STORE_TYPE", "chroma")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def sample_docs_dir(project_root) -> Path:
    return project_root / "data" / "samples"


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_pdf_content() -> str:
    return """
    Employee Handbook — Remote Work Policy

    Section 3.2: Remote Work Guidelines

    All full-time employees may work remotely up to 3 days per week.
    Contractors must submit a remote work request form (Form RW-01) at
    least 5 business days in advance. Remote work is subject to manager
    approval and department requirements.

    Section 3.3: Equipment Policy

    The company provides a laptop and one external monitor to all remote
    workers. Employees are responsible for maintaining a secure and
    ergonomic home workspace.
    """


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "text": "All full-time employees may work remotely up to 3 days per week.",
            "metadata": {
                "source_file": "employee_handbook.pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
        },
        {
            "text": "Contractors must submit a remote work request form (Form RW-01).",
            "metadata": {
                "source_file": "employee_handbook.pdf",
                "page_number": 1,
                "chunk_index": 1,
            },
        },
    ]

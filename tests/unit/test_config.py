"""Tests for configuration management."""
import os
import pytest
from src.utils.config import get_settings, Settings


def test_settings_loads_defaults():
    """Settings should have sensible defaults without any .env file."""
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.environment == "development"
    assert settings.chunking.chunk_size == 512
    assert settings.chunking.chunk_overlap == 50
    assert settings.retrieval.retrieval_top_k == 20
    assert settings.retrieval.reranking_top_n == 5


def test_settings_is_development():
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.is_development() is True
    assert settings.is_production() is False


def test_chunking_overlap_must_be_less_than_chunk_size():
    from src.utils.config import ChunkingSettings
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ChunkingSettings(chunk_size=100, chunk_overlap=100)


def test_api_cors_origins_list():
    from src.utils.config import APISettings
    api = APISettings(cors_origins="http://localhost:8501,http://localhost:3000")
    assert len(api.cors_origins_list) == 2
    assert "http://localhost:8501" in api.cors_origins_list


def test_api_allowed_file_types_list():
    from src.utils.config import APISettings
    api = APISettings()
    assert "pdf" in api.allowed_file_types_list
    assert "docx" in api.allowed_file_types_list

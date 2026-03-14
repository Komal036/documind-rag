"""Tests for file utility functions."""
from pathlib import Path
import pytest
from src.utils.file_utils import compute_file_hash, get_file_metadata, ensure_dir
from src.utils.exceptions import UnsupportedFileTypeError, FileTooLargeError


def test_compute_file_hash_is_deterministic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    hash1 = compute_file_hash(f)
    hash2 = compute_file_hash(f)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex = 64 chars


def test_compute_file_hash_differs_for_different_content(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("content A")
    f2.write_text("content B")
    assert compute_file_hash(f1) != compute_file_hash(f2)


def test_get_file_metadata(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("some text content")
    meta = get_file_metadata(f)
    assert meta["filename"] == "sample.txt"
    assert meta["extension"] == ".txt"
    assert "sha256" in meta
    assert meta["size_bytes"] > 0


def test_ensure_dir_creates_nested(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    result = ensure_dir(nested)
    assert nested.exists()
    assert result == nested

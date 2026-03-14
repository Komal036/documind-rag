"""Tests for custom exception hierarchy."""
from src.utils.exceptions import (
    DocuMindError,
    DocumentLoadError,
    UnsupportedFileTypeError,
    FileTooLargeError,
    LLMRateLimitError,
    LLMError,
)


def test_all_exceptions_inherit_from_base():
    errors = [
        DocumentLoadError("test"),
        UnsupportedFileTypeError("test"),
        FileTooLargeError("test"),
        LLMRateLimitError("test"),
    ]
    for err in errors:
        assert isinstance(err, DocuMindError)
        assert isinstance(err, Exception)


def test_exception_with_details():
    err = UnsupportedFileTypeError(
        "Bad file type",
        details={"supported": [".pdf"], "received": ".exe"},
    )
    assert err.details["received"] == ".exe"
    assert "Bad file type" in str(err)


def test_llm_rate_limit_inherits_llm_error():
    err = LLMRateLimitError("Rate limit hit")
    assert isinstance(err, LLMError)
    assert isinstance(err, DocuMindError)

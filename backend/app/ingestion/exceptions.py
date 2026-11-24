"""Domain exception hierarchy for the document ingestion pipeline.

Why this module exists:
    Decouples domain validation, extraction, deduplication, and external API
    failures from FastAPI HTTP responses. Route handlers catch `IngestionError`
    and serialize its `status_code`, `error_code`, `message`, and `details`
    directly into the standardized JSON error envelope without embedding
    business logic in API routes.
"""

from __future__ import annotations

from typing import Any


class IngestionError(Exception):
    """Base domain exception carrying structured HTTP contract metadata.

    Attributes:
        status_code: HTTP status code matching the API contract (400/409/413/415/503).
        error_code: Machine-readable uppercase error code for clients and logs.
        message: Human-readable explanation safe to return to callers.
        details: Optional dictionary of structured debugging context.
    """

    status_code: int = 400
    error_code: str = "INGESTION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code
        self.message = message
        self.details: dict[str, Any] = details or {}


class InvalidDocumentError(IngestionError):
    """Raised when a document is empty, corrupted, scanned-only, or header-only (HTTP 400)."""

    status_code = 400
    error_code = "INVALID_DOCUMENT"


class DuplicateDocumentError(IngestionError):
    """Raised when an identical SHA-256 content hash already exists for the user (HTTP 409)."""

    status_code = 409
    error_code = "DUPLICATE_DOCUMENT"


class PayloadTooLargeError(IngestionError):
    """Raised when an uploaded file exceeds the configured byte size cap (HTTP 413)."""

    status_code = 413
    error_code = "PAYLOAD_TOO_LARGE"


class UnsupportedMediaTypeError(IngestionError):
    """Raised when file extension or content signature fails verification (HTTP 415)."""

    status_code = 415
    error_code = "UNSUPPORTED_MEDIA_TYPE"


class ExternalServiceError(IngestionError):
    """Raised when an external dependency (Cohere, Groq, Neon DB) times out or fails (HTTP 503)."""

    status_code = 503
    error_code = "EXTERNAL_SERVICE_UNAVAILABLE"

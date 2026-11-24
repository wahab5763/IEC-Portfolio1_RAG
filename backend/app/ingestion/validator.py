"""Multi-format file validator enforcing size limits, content signatures, and SHA-256 hashing.

Why this module exists:
    Prevents malicious, spoofed, oversized, or empty uploads from ever reaching
    document parsers or external embedding APIs. Verifies both binary formats
    (`.pdf` magic bytes; `.docx`/`.pptx` OOXML ZIP structure) and text formats
    (`.txt`/`.md`/`.csv` null-byte & encoding checks) before computing a
    deterministic SHA-256 content digest for per-user deduplication.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import chardet

from backend.app.config import get_settings
from backend.app.ingestion.exceptions import (
    InvalidDocumentError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({"pdf", "docx", "txt", "md", "pptx", "csv"})

# Required internal XML member paths distinguishing OOXML Office archives
OOXML_REQUIRED_MEMBERS: dict[str, str] = {
    "docx": "word/document.xml",
    "pptx": "ppt/presentation.xml",
}


@dataclass(frozen=True)
class ValidatedFile:
    """Immutable container for a verified uploaded document."""

    filename: str
    file_type: str
    file_hash: str
    file_size_bytes: int
    raw_bytes: bytes


def _verify_pdf_signature(raw_bytes: bytes, filename: str) -> None:
    """Verify that a `.pdf` payload starts with the PDF magic header `%PDF-`."""
    # Allow small leading whitespace/BOM before %PDF- per PDF spec, within first 1024 bytes.
    if b"%PDF-" not in raw_bytes[:1024]:
        raise UnsupportedMediaTypeError(
            "File content signature mismatch: expected a valid PDF header.",
            details={"filename": filename, "expected_type": "pdf"},
        )


def _verify_ooxml_signature(raw_bytes: bytes, file_type: str, filename: str) -> None:
    """Verify that `.docx`/`.pptx` payloads are valid ZIP archives containing OOXML entries."""
    if not raw_bytes.startswith(b"PK\x03\x04"):
        raise UnsupportedMediaTypeError(
            f"File content signature mismatch: expected a valid {file_type.upper()} archive.",
            details={"filename": filename, "expected_type": file_type},
        )
    required_member = OOXML_REQUIRED_MEMBERS[file_type]
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            if required_member not in archive.namelist():
                raise UnsupportedMediaTypeError(
                    f"Archive does not contain required {file_type.upper()} structure ({required_member}).",
                    details={"filename": filename, "expected_type": file_type},
                )
    except zipfile.BadZipFile as exc:
        raise InvalidDocumentError(
            f"Corrupted {file_type.upper()} archive cannot be opened.",
            details={"filename": filename},
        ) from exc


def _verify_text_signature(raw_bytes: bytes, file_type: str, filename: str) -> None:
    """Reject disguised binaries (e.g., `.exe` renamed to `.csv`/`.txt`) via byte inspection."""
    sample = raw_bytes[:8192]
    # Null bytes are a reliable indicator of binary payloads disguised as plain text/CSV.
    if b"\x00" in sample:
        raise UnsupportedMediaTypeError(
            f"File content signature mismatch: binary bytes detected in {file_type.upper()} upload.",
            details={"filename": filename, "expected_type": file_type},
        )

    # Reject payloads with excessive non-printable control characters (> 10% of sample).
    allowed_ctrl = {9, 10, 12, 13}  # \t, \n, \f, \r
    non_printable = sum(1 for b in sample if b < 32 and b not in allowed_ctrl)
    if sample and (non_printable / len(sample)) > 0.10:
        raise UnsupportedMediaTypeError(
            f"File content signature mismatch: non-text control bytes detected in {file_type.upper()}.",
            details={"filename": filename, "expected_type": file_type},
        )

    # Confirm bytes are decodable as UTF-8 or a detected text encoding.
    try:
        sample.decode("utf-8-sig")
    except UnicodeDecodeError:
        detected = chardet.detect(sample)
        if not detected.get("encoding") or (detected.get("confidence") or 0.0) < 0.35:
            raise UnsupportedMediaTypeError(
                f"Unable to decode {file_type.upper()} file as valid text.",
                details={"filename": filename, "expected_type": file_type},
            ) from None


def validate_upload(
    filename: str,
    raw_bytes: bytes,
    *,
    max_file_size_bytes: int | None = None,
) -> ValidatedFile:
    """Validate an uploaded document's extension, size, signature, and compute SHA-256.

    Args:
        filename: Original uploaded filename including extension.
        raw_bytes: Complete binary content of the uploaded file.
        max_file_size_bytes: Optional override for maximum allowed byte size
            (defaults to `Settings.max_file_size_bytes`).

    Returns:
        `ValidatedFile` containing normalized `file_type`, `file_hash`, and bytes.

    Raises:
        InvalidDocumentError: If filename is missing or file content is empty (HTTP 400).
        PayloadTooLargeError: If file exceeds `max_file_size_bytes` (HTTP 413).
        UnsupportedMediaTypeError: If extension is disallowed or magic bytes /
            text heuristics do not match the declared format (HTTP 415).
    """
    clean_name = Path(filename or "").name.strip()
    if not clean_name or "." not in clean_name:
        raise InvalidDocumentError("Uploaded file must have a valid filename and extension.")

    file_size = len(raw_bytes)
    if file_size == 0:
        raise InvalidDocumentError(
            "Uploaded file is empty (0 bytes).",
            details={"filename": clean_name},
        )

    limit_bytes = (
        max_file_size_bytes
        if max_file_size_bytes is not None
        else get_settings().max_file_size_bytes
    )
    if file_size > limit_bytes:
        raise PayloadTooLargeError(
            f"File exceeds maximum allowed size of {limit_bytes // (1024 * 1024)} MB.",
            details={
                "filename": clean_name,
                "file_size_bytes": file_size,
                "max_file_size_bytes": limit_bytes,
            },
        )

    ext = Path(clean_name).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedMediaTypeError(
            f"Unsupported file type '.{ext}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}.",
            details={"filename": clean_name, "extension": ext},
        )

    # Dispatch format-specific content signature verification.
    if ext == "pdf":
        _verify_pdf_signature(raw_bytes, clean_name)
    elif ext in OOXML_REQUIRED_MEMBERS:
        _verify_ooxml_signature(raw_bytes, ext, clean_name)
    else:
        _verify_text_signature(raw_bytes, ext, clean_name)

    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    return ValidatedFile(
        filename=clean_name,
        file_type=ext,
        file_hash=file_hash,
        file_size_bytes=file_size,
        raw_bytes=raw_bytes,
    )

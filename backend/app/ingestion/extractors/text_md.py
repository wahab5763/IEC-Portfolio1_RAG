"""Plain-text (`.txt`) and Markdown (`.md`) extractors with encoding auto-detection.

Why this module exists:
    Handles UTF-8 and legacy encodings cleanly via `chardet` fallback while
    preserving Markdown heading breadcrumb hierarchies (`# > ## > ###`) and
    avoiding false heading splits inside fenced code blocks.
"""

from __future__ import annotations

import re

import chardet

from backend.app.ingestion.exceptions import InvalidDocumentError
from backend.app.ingestion.extractors.base import (
    BaseExtractor,
    ExtractedBlock,
    ExtractedDocument,
)

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")


def _decode_text(raw_bytes: bytes) -> str:
    """Decode raw bytes using UTF-8 fast path with `chardet` fallback and CRLF cleanup."""
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "latin-1"
        text = raw_bytes.decode(encoding, errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


class TextExtractor(BaseExtractor):
    """Extracts normalized plain-text content from `.txt` documents."""

    file_types = ("txt",)

    def extract(self, raw_bytes: bytes, filename: str) -> ExtractedDocument:
        """Decode plain text bytes into a single canonical block."""
        text = _decode_text(raw_bytes)
        if not text:
            raise InvalidDocumentError(
                "Text document is empty or contains only whitespace.",
                details={"filename": filename},
            )
        return ExtractedDocument(
            filename=filename,
            file_type="txt",
            page_count=1,
            blocks=[
                ExtractedBlock(
                    block_type="text",
                    content=text,
                    page_number=1,
                    section_title="Document Body",
                    metadata={"unit_type": "section"},
                )
            ],
        )


class MarkdownExtractor(BaseExtractor):
    """Extracts heading-scoped sections from `.md` documents."""

    file_types = ("md",)

    def extract(self, raw_bytes: bytes, filename: str) -> ExtractedDocument:
        """Parse Markdown into heading-breadcrumb blocks while respecting code fences."""
        text = _decode_text(raw_bytes)
        if not text:
            raise InvalidDocumentError(
                "Markdown document is empty or contains only whitespace.",
                details={"filename": filename},
            )

        blocks: list[ExtractedBlock] = []
        heading_stack: list[tuple[int, str]] = []
        buffer: list[str] = []
        in_code_fence = False

        def _flush() -> None:
            body = "\n".join(buffer).strip()
            if body:
                breadcrumb = " > ".join(title for _, title in heading_stack) or "Document Body"
                blocks.append(
                    ExtractedBlock(
                        block_type="text",
                        content=body,
                        page_number=1,
                        section_title=breadcrumb[:180],
                        metadata={"unit_type": "section"},
                    )
                )
            buffer.clear()

        for line in text.splitlines():
            if line.strip().startswith("```"):
                in_code_fence = not in_code_fence
                buffer.append(line)
                continue

            match = _HEADING_RE.match(line) if not in_code_fence else None
            if match:
                _flush()
                level = len(match.group(1))
                title = match.group(2).strip()
                heading_stack = [(lvl, t) for lvl, t in heading_stack if lvl < level]
                heading_stack.append((level, title))
            else:
                buffer.append(line)

        _flush()

        if not blocks:
            # Handle heading-only Markdown files gracefully by keeping raw text.
            blocks.append(
                ExtractedBlock(
                    block_type="text",
                    content=text,
                    page_number=1,
                    section_title="Document Body",
                    metadata={"unit_type": "section"},
                )
            )

        return ExtractedDocument(
            filename=filename,
            file_type="md",
            page_count=1,
            blocks=blocks,
        )

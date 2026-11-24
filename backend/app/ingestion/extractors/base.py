"""Base extractor contract and canonical intermediate representation models.

Why this module exists:
    Defines the format-agnostic contract (`BaseExtractor`, `ExtractedDocument`,
    `ExtractedBlock`) so downstream chunking, embedding, and database storage
    never need format-specific conditionals. Adding a new document type only
    requires implementing `BaseExtractor` and registering it in `ExtractorRegistry`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedBlock:
    """Single normalized structural unit extracted from any supported document.

    Attributes:
        block_type: One of `"text"`, `"table"`, `"slide_notes"`, or `"csv_table"`.
        content: Plain or Markdown-formatted text (empty for `"csv_table"`, which
            carries structured `headers` and `rows` in `metadata`).
        page_number: 1-based page (PDF), slide (PPTX), batch index (CSV), or 1.
        section_title: Nearest heading breadcrumb, slide title, or CSV section label.
        metadata: Format-specific provenance dictionary stored in PostgreSQL JSONB.
    """

    block_type: str
    content: str
    page_number: int = 1
    section_title: str = "General"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    """Canonical format-agnostic representation produced by every extractor."""

    filename: str
    file_type: str
    page_count: int
    blocks: list[ExtractedBlock]


class BaseExtractor(ABC):
    """Abstract base class implemented by all document format extractors."""

    file_types: tuple[str, ...] = ()

    @abstractmethod
    def extract(self, raw_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract structured text and tables from raw file bytes.

        Args:
            raw_bytes: Validated raw binary payload of the document.
            filename: Original sanitized filename for metadata provenance.

        Returns:
            Canonical `ExtractedDocument` containing one or more `ExtractedBlock`s.

        Raises:
            InvalidDocumentError: If the document cannot be parsed or contains
                no extractable text or data rows.
        """


def rows_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Serialize tabular headers and rows into a clean GitHub-Flavored Markdown table.

    Args:
        headers: Column header labels.
        rows: List of data rows aligned with `headers`.

    Returns:
        Markdown table string preserving column/row relationships for LLMs.
    """
    clean_headers = [
        str(h or "").replace("|", "\\|").replace("\n", " ").strip() or f"col_{i + 1}"
        for i, h in enumerate(headers)
    ]
    lines = [
        "| " + " | ".join(clean_headers) + " |",
        "| " + " | ".join(["---"] * len(clean_headers)) + " |",
    ]
    col_count = len(clean_headers)
    for row in rows:
        padded = list(row) + [""] * max(0, col_count - len(row))
        vals = [
            str(v or "").replace("|", "\\|").replace("\n", " ").strip()
            for v in padded[:col_count]
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

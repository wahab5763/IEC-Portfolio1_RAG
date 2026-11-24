"""DOCX extractor preserving true reading order across paragraphs, headings, and tables.

Why this module exists:
    `python-docx` exposes `doc.paragraphs` and `doc.tables` as separate lists,
    which loses their relative reading order. Iterating the underlying OOXML
    body children sequentially keeps each table attached to its exact heading
    section so retrieved chunks carry accurate `section_title` provenance.
"""

from __future__ import annotations

import io
from collections.abc import Iterator

from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from backend.app.ingestion.exceptions import InvalidDocumentError
from backend.app.ingestion.extractors.base import (
    BaseExtractor,
    ExtractedBlock,
    ExtractedDocument,
    rows_to_markdown,
)


def _iter_block_items(doc: DocxDocument) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables from a Word document in exact XML reading order."""
    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


class DocxExtractor(BaseExtractor):
    """Extracts heading-scoped prose and Markdown tables from `.docx` files."""

    file_types = ("docx",)

    def extract(self, raw_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract reading-order paragraphs and tables from DOCX bytes.

        Args:
            raw_bytes: Validated DOCX archive bytes.
            filename: Original filename for metadata provenance.

        Returns:
            `ExtractedDocument` with heading-scoped `text` and `table` blocks.

        Raises:
            InvalidDocumentError: If the DOCX cannot be opened or has no text.
        """
        try:
            doc = DocxDocument(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise InvalidDocumentError(
                f"Failed to parse DOCX document: {exc}",
                details={"filename": filename},
            ) from exc

        blocks: list[ExtractedBlock] = []
        current_section = "General"
        para_buffer: list[str] = []

        def _flush_paragraphs() -> None:
            if para_buffer:
                blocks.append(
                    ExtractedBlock(
                        block_type="text",
                        content="\n\n".join(para_buffer),
                        page_number=1,
                        section_title=current_section,
                        metadata={"unit_type": "section"},
                    )
                )
                para_buffer.clear()

        for item in _iter_block_items(doc):
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                style_name = (item.style.name or "").lower() if item.style else ""
                if style_name.startswith("heading"):
                    _flush_paragraphs()
                    current_section = text[:160]
                else:
                    para_buffer.append(text)
            elif isinstance(item, Table):
                _flush_paragraphs()
                rows = [
                    [cell.text.strip() for cell in row.cells]
                    for row in item.rows
                    if any(cell.text.strip() for cell in row.cells)
                ]
                if len(rows) >= 2:
                    blocks.append(
                        ExtractedBlock(
                            block_type="table",
                            content=rows_to_markdown(rows[0], rows[1:]),
                            page_number=1,
                            section_title=current_section,
                            metadata={"unit_type": "section"},
                        )
                    )

        _flush_paragraphs()

        if not blocks:
            raise InvalidDocumentError(
                "DOCX document contains no extractable text or tables.",
                details={"filename": filename},
            )

        return ExtractedDocument(
            filename=filename,
            file_type="docx",
            page_count=1,
            blocks=blocks,
        )

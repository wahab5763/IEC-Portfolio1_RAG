"""PDF extractor combining PyMuPDF layout text extraction with pdfplumber tables.

Why this module exists:
    `pymupdf` provides fast, accurate layout-ordered text extraction with
    coordinate-based header/footer clipping, while `pdfplumber` excels at
    recovering structured tables as Markdown. Also enforces the scanned/image-only
    PDF guardrail (`< min_pdf_chars_per_page` -> HTTP 400).
"""

from __future__ import annotations

import io

import pdfplumber
import pymupdf

from backend.app.config import get_settings
from backend.app.ingestion.exceptions import InvalidDocumentError
from backend.app.ingestion.extractors.base import (
    BaseExtractor,
    ExtractedBlock,
    ExtractedDocument,
    rows_to_markdown,
)


class PDFExtractor(BaseExtractor):
    """Extracts text and Markdown tables from native text PDF documents."""

    file_types = ("pdf",)

    def __init__(self, min_chars_per_page: int | None = None, margin_clip_pt: float = 36.0) -> None:
        """Initialize the PDF extractor.

        Args:
            min_chars_per_page: Minimum average characters per page required to
                distinguish native text PDFs from scanned image-only PDFs.
            margin_clip_pt: Vertical header/footer exclusion margin in PDF points.
        """
        self._min_chars_per_page = (
            min_chars_per_page
            if min_chars_per_page is not None
            else get_settings().min_pdf_chars_per_page
        )
        self._margin_clip_pt = margin_clip_pt

    def extract(self, raw_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract page-level text and Markdown tables from PDF bytes.

        Args:
            raw_bytes: Validated PDF bytes.
            filename: Original filename for provenance metadata.

        Returns:
            `ExtractedDocument` with page-numbered `text` and `table` blocks.

        Raises:
            InvalidDocumentError: If the PDF is corrupted, has 0 pages, or
                yields fewer than `min_chars_per_page` average characters.
        """
        blocks: list[ExtractedBlock] = []
        total_chars = 0

        try:
            with pymupdf.open(stream=raw_bytes, filetype="pdf") as pdf_doc:
                page_count = len(pdf_doc)
                if page_count == 0:
                    raise InvalidDocumentError(
                        "PDF contains zero pages.",
                        details={"filename": filename},
                    )

                for page_idx, page in enumerate(pdf_doc, start=1):
                    rect = page.rect
                    # Clip top & bottom margins when page height allows to strip running headers/footers.
                    if rect.height > (self._margin_clip_pt * 3):
                        clip = pymupdf.Rect(
                            rect.x0,
                            rect.y0 + self._margin_clip_pt,
                            rect.x1,
                            rect.y1 - self._margin_clip_pt,
                        )
                        page_text = page.get_text("text", clip=clip).strip()
                    else:
                        page_text = page.get_text("text").strip()

                    total_chars += len(page_text)
                    if page_text:
                        first_line = page_text.splitlines()[0].strip()[:120]
                        blocks.append(
                            ExtractedBlock(
                                block_type="text",
                                content=page_text,
                                page_number=page_idx,
                                section_title=first_line or f"Page {page_idx}",
                                metadata={"unit_type": "page"},
                            )
                        )
        except InvalidDocumentError:
            raise
        except Exception as exc:
            raise InvalidDocumentError(
                f"Failed to parse PDF document: {exc}",
                details={"filename": filename},
            ) from exc

        avg_chars = total_chars / max(page_count, 1)
        if avg_chars < self._min_chars_per_page:
            raise InvalidDocumentError(
                "PDF lacks extractable text and appears to be scanned or image-only. "
                "OCR preprocessing is required.",
                details={
                    "filename": filename,
                    "page_count": page_count,
                    "avg_chars_per_page": round(avg_chars, 2),
                },
            )

        # Extract structured tables via pdfplumber and serialize to Markdown.
        try:
            with pdfplumber.open(io.BytesIO(raw_bytes)) as plumber_doc:
                for page_idx, page in enumerate(plumber_doc.pages, start=1):
                    for tbl in page.extract_tables() or []:
                        cleaned_rows = [
                            [cell or "" for cell in row]
                            for row in tbl
                            if row and any((cell or "").strip() for cell in row)
                        ]
                        if len(cleaned_rows) >= 2:
                            blocks.append(
                                ExtractedBlock(
                                    block_type="table",
                                    content=rows_to_markdown(cleaned_rows[0], cleaned_rows[1:]),
                                    page_number=page_idx,
                                    section_title=f"Table (Page {page_idx})",
                                    metadata={"unit_type": "page"},
                                )
                            )
        except Exception:
            # Table extraction is best-effort enrichment; native text blocks are already captured.
            pass

        return ExtractedDocument(
            filename=filename,
            file_type="pdf",
            page_count=page_count,
            blocks=blocks,
        )

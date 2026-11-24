"""PowerPoint (`.pptx`) extractor capturing slide titles, text, tables, and speaker notes.

Why this module exists:
    Slide decks frequently store critical figures inside embedded tables or
    presenter speaker notes rather than body bullets. Extracting all three with
    `slide_number` mapped to `page_number` enables accurate slide-level citations.
"""

from __future__ import annotations

import io

from pptx import Presentation

from backend.app.ingestion.exceptions import InvalidDocumentError
from backend.app.ingestion.extractors.base import (
    BaseExtractor,
    ExtractedBlock,
    ExtractedDocument,
    rows_to_markdown,
)


class PptxExtractor(BaseExtractor):
    """Extracts slide text, tables, and speaker notes from `.pptx` presentations."""

    file_types = ("pptx",)

    def extract(self, raw_bytes: bytes, filename: str) -> ExtractedDocument:
        """Parse PPTX bytes into slide-numbered canonical blocks.

        Args:
            raw_bytes: Validated PPTX archive bytes.
            filename: Original filename for metadata provenance.

        Returns:
            `ExtractedDocument` where `page_count` equals slide count and each
            block carries `metadata={"unit_type": "slide"}`.

        Raises:
            InvalidDocumentError: If the deck cannot be opened or contains zero
                extractable text across all slides.
        """
        try:
            prs = Presentation(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise InvalidDocumentError(
                f"Failed to parse PPTX presentation: {exc}",
                details={"filename": filename},
            ) from exc

        blocks: list[ExtractedBlock] = []
        slide_count = len(prs.slides)

        for slide_num, slide in enumerate(prs.slides, start=1):
            title_shape = slide.shapes.title
            slide_title = (
                title_shape.text.strip()
                if title_shape is not None and title_shape.text
                else f"Slide {slide_num}"
            )

            body_parts: list[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame and shape != title_shape:
                    txt = shape.text_frame.text.strip()
                    if txt:
                        body_parts.append(txt)
                if shape.has_table:
                    tbl_rows = [
                        [cell.text.strip() for cell in row.cells]
                        for row in shape.table.rows
                        if any(cell.text.strip() for cell in row.cells)
                    ]
                    if len(tbl_rows) >= 2:
                        blocks.append(
                            ExtractedBlock(
                                block_type="table",
                                content=rows_to_markdown(tbl_rows[0], tbl_rows[1:]),
                                page_number=slide_num,
                                section_title=slide_title[:160],
                                metadata={"unit_type": "slide"},
                            )
                        )

            if body_parts:
                blocks.append(
                    ExtractedBlock(
                        block_type="text",
                        content="\n".join(body_parts),
                        page_number=slide_num,
                        section_title=slide_title[:160],
                        metadata={"unit_type": "slide"},
                    )
                )

            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes_text = slide.notes_slide.notes_text_frame.text.strip()
                if notes_text:
                    blocks.append(
                        ExtractedBlock(
                            block_type="slide_notes",
                            content=notes_text,
                            page_number=slide_num,
                            section_title=f"{slide_title[:140]} (Notes)",
                            metadata={"unit_type": "slide"},
                        )
                    )

        if not blocks:
            raise InvalidDocumentError(
                "PPTX presentation contains no extractable slide text, tables, or notes.",
                details={"filename": filename, "slide_count": slide_count},
            )

        return ExtractedDocument(
            filename=filename,
            file_type="pptx",
            page_count=max(slide_count, 1),
            blocks=blocks,
        )

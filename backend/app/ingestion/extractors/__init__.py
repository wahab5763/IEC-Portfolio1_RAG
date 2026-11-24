"""Pluggable format-agnostic document extractor registry.

Why this module exists:
    Decouples the ingestion pipeline from concrete file-format libraries.
    Adding a new document type (e.g., `.xlsx`, `.html`, `.json`) only requires
    registering a `BaseExtractor` instance on `ExtractorRegistry`.
"""

from __future__ import annotations

from backend.app.ingestion.exceptions import UnsupportedMediaTypeError
from backend.app.ingestion.extractors.base import BaseExtractor, ExtractedBlock, ExtractedDocument
from backend.app.ingestion.extractors.csv_extractor import CsvExtractor
from backend.app.ingestion.extractors.docx import DocxExtractor
from backend.app.ingestion.extractors.pdf import PDFExtractor
from backend.app.ingestion.extractors.pptx import PptxExtractor
from backend.app.ingestion.extractors.text_md import MarkdownExtractor, TextExtractor


class ExtractorRegistry:
    """Strategy registry mapping normalized file extensions to `BaseExtractor` instances."""

    def __init__(self) -> None:
        self._extractors: dict[str, BaseExtractor] = {}

    def register(self, extractor: BaseExtractor) -> None:
        """Register an extractor for all of its declared `file_types`.

        Args:
            extractor: Concrete `BaseExtractor` implementation.
        """
        for ft in extractor.file_types:
            self._extractors[ft.lower().lstrip(".")] = extractor

    def get(self, file_type: str) -> BaseExtractor:
        """Retrieve the registered extractor for a normalized file type.

        Args:
            file_type: Normalized lowercase extension without leading dot.

        Returns:
            Matching `BaseExtractor` instance.

        Raises:
            UnsupportedMediaTypeError: If no extractor is registered for `file_type`.
        """
        key = file_type.lower().lstrip(".")
        extractor = self._extractors.get(key)
        if extractor is None:
            raise UnsupportedMediaTypeError(
                f"No extractor registered for file type '{key}'.",
                details={"file_type": key, "supported_types": self.supported_types()},
            )
        return extractor

    def supported_types(self) -> list[str]:
        """Return sorted list of all registered file extensions."""
        return sorted(self._extractors.keys())


def build_default_registry() -> ExtractorRegistry:
    """Construct an `ExtractorRegistry` pre-loaded with all 6 Phase 1 core formats."""
    registry = ExtractorRegistry()
    for extractor in (
        PDFExtractor(),
        DocxExtractor(),
        TextExtractor(),
        MarkdownExtractor(),
        PptxExtractor(),
        CsvExtractor(),
    ):
        registry.register(extractor)
    return registry


__all__ = [
    "BaseExtractor",
    "CsvExtractor",
    "DocxExtractor",
    "ExtractedBlock",
    "ExtractedDocument",
    "ExtractorRegistry",
    "MarkdownExtractor",
    "PDFExtractor",
    "PptxExtractor",
    "TextExtractor",
    "build_default_registry",
]

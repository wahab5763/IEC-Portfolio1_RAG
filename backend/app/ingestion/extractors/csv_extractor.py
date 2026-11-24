"""CSV tabular extractor with encoding/delimiter sniffing and schema preservation.

Why this module exists:
    Naive line-splitting on CSV files destroys column meaning on every chunk
    after chunk 0. This extractor decodes the CSV, sniffs the delimiter
    (`,`, `;`, `\\t`, `|`), strips empty rows, validates that at least one data
    row exists, and emits a structured `csv_table` block so the format-aware
    chunker can batch rows with repeated column headers.
"""

from __future__ import annotations

import csv
import io

import chardet

from backend.app.ingestion.exceptions import InvalidDocumentError
from backend.app.ingestion.extractors.base import (
    BaseExtractor,
    ExtractedBlock,
    ExtractedDocument,
)


class CsvExtractor(BaseExtractor):
    """Extracts structured header + data rows from `.csv` files."""

    file_types = ("csv",)

    def extract(self, raw_bytes: bytes, filename: str) -> ExtractedDocument:
        """Sniff encoding/delimiter and parse CSV bytes into a structured block.

        Args:
            raw_bytes: Validated CSV bytes.
            filename: Original filename for metadata provenance.

        Returns:
            `ExtractedDocument` containing a single `csv_table` block carrying
            `headers` and `rows` in `metadata`.

        Raises:
            InvalidDocumentError: If the CSV is empty, malformed, or header-only
                (zero data rows).
        """
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            encoding = chardet.detect(raw_bytes).get("encoding") or "latin-1"
            text = raw_bytes.decode(encoding, errors="replace")

        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            raise InvalidDocumentError(
                "CSV file is empty.",
                details={"filename": filename},
            )

        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel

        try:
            parsed_rows = list(csv.reader(io.StringIO(text), dialect))
        except csv.Error as exc:
            raise InvalidDocumentError(
                f"Malformed CSV file: {exc}",
                details={"filename": filename},
            ) from exc

        non_empty_rows = [
            [cell.strip() for cell in row]
            for row in parsed_rows
            if any(cell.strip() for cell in row)
        ]

        if len(non_empty_rows) < 2:
            raise InvalidDocumentError(
                "CSV file contains no data rows (empty or header-only).",
                details={"filename": filename, "row_count": len(non_empty_rows)},
            )

        headers = [col or f"column_{i + 1}" for i, col in enumerate(non_empty_rows[0])]
        data_rows = non_empty_rows[1:]

        return ExtractedDocument(
            filename=filename,
            file_type="csv",
            page_count=1,
            blocks=[
                ExtractedBlock(
                    block_type="csv_table",
                    content="",
                    page_number=1,
                    section_title="CSV Dataset",
                    metadata={
                        "unit_type": "rows",
                        "headers": headers,
                        "rows": data_rows,
                        "total_rows": len(data_rows),
                    },
                )
            ],
        )

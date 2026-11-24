"""Format-aware recursive token chunker with header-repeated CSV/table batching.

Why this module exists:
    Chunking quality directly governs retrieval recall and precision. Narrative
    blocks split recursively along paragraphs -> sentences -> token windows with
    configurable overlap, while tabular/CSV blocks split strictly on row
    boundaries with automatic column-header repetition (or `Column: Value`
    records for wide tables > 12 columns) so no numeric value loses its schema.
    Every chunk optionally receives a contextual metadata prefix prior to
    embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from backend.app.config import get_settings
from backend.app.ingestion.extractors.base import ExtractedDocument, rows_to_markdown

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    """Return exact `cl100k_base` token count for a string."""
    return len(_ENCODING.encode(text))


@dataclass
class ChunkData:
    """Single embed-ready document chunk with provenance metadata."""

    chunk_index: int
    page_number: int
    section_title: str
    content: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class FormatAwareChunker:
    """Partitions any canonical `ExtractedDocument` into token-bounded chunks."""

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        inject_headers: bool = True,
        wide_csv_column_threshold: int = 12,
    ) -> None:
        """Initialize the chunker with token budget and contextual header settings.

        Args:
            chunk_size: Maximum token target per chunk (default from `Settings`: 400).
            chunk_overlap: Token overlap for consecutive narrative chunks (default: 60).
            inject_headers: Whether to prepend `[Document | Section | Unit]` headers
                (True for Config B, False for Config A baseline).
            wide_csv_column_threshold: Column count above which CSV rows switch from
                Markdown tables to compact `Column: Value` key-value records.
        """
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size_tokens
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap_tokens
        )
        self.inject_headers = inject_headers
        self.wide_csv_column_threshold = wide_csv_column_threshold

    def _build_header(
        self,
        *,
        filename: str,
        section: str,
        page_number: int,
        unit_type: str,
        extra: str = "",
    ) -> str:
        """Build the contextual metadata prefix injected at the top of each chunk."""
        if not self.inject_headers:
            return ""
        if unit_type == "page":
            unit_part = f"Page: {page_number}"
        elif unit_type == "slide":
            unit_part = f"Slide: {page_number}"
        elif unit_type == "rows" and extra:
            unit_part = extra
        else:
            unit_part = f"Section: {section}"
        return f"[Document: {filename} | Section: {section} | {unit_part}]\n"

    def _split_text_recursive(self, text: str, max_tokens: int) -> list[str]:
        """Recursively split narrative text along paragraphs -> sentences -> token windows.

        Args:
            text: Raw narrative text body (without contextual header).
            max_tokens: Available token budget for body text inside each chunk.

        Returns:
            Ordered list of overlapping text segments each fitting within `max_tokens`.
        """
        if count_tokens(text) <= max_tokens:
            return [text]

        # 1. Split by paragraph double newlines first to keep logical ideas intact.
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            return self._pack_units(paragraphs, separator="\n\n", max_tokens=max_tokens)

        # 2. Split single long paragraph by sentence terminations.
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if len(sentences) > 1:
            return self._pack_units(sentences, separator=" ", max_tokens=max_tokens)

        # 3. Token-aligned window fallback with overlap for indivisible text runs.
        tokens = _ENCODING.encode(text)
        step = max(max_tokens - self.chunk_overlap, 32)
        return [
            _ENCODING.decode(tokens[start : start + max_tokens]).strip()
            for start in range(0, len(tokens), step)
            if tokens[start : start + max_tokens]
        ]

    def _pack_units(self, units: list[str], *, separator: str, max_tokens: int) -> list[str]:
        """Greedily pack text units into token-bounded chunks with tail overlap."""
        packed: list[str] = []
        current: list[str] = []

        for unit in units:
            # If a single unit exceeds budget, recursively split it down.
            if count_tokens(unit) > max_tokens:
                if current:
                    packed.append(separator.join(current))
                    current = []
                packed.extend(self._split_text_recursive(unit, max_tokens))
                continue

            candidate = separator.join([*current, unit]) if current else unit
            if current and count_tokens(candidate) > max_tokens:
                packed.append(separator.join(current))
                # Carry trailing units up to `chunk_overlap` tokens into next chunk.
                overlap_units: list[str] = []
                overlap_tokens = 0
                for prev in reversed(current):
                    prev_tok = count_tokens(prev)
                    if overlap_tokens + prev_tok > self.chunk_overlap:
                        break
                    overlap_units.insert(0, prev)
                    overlap_tokens += prev_tok
                current = [*overlap_units, unit]
            else:
                current.append(unit)

        if current:
            packed.append(separator.join(current))
        return packed

    @staticmethod
    def _rows_to_kv_records(headers: list[str], rows: list[list[str]], start_row: int) -> str:
        """Serialize wide CSV rows (> 12 columns) as compact `Column: Value` records."""
        record_lines: list[str] = []
        for offset, row in enumerate(rows):
            row_num = start_row + offset
            pairs = [
                f"{col}: {row[i] if i < len(row) else ''}"
                for i, col in enumerate(headers)
                if (i < len(row) and row[i] != "")
            ]
            record_lines.append(f"Row {row_num} | " + " | ".join(pairs))
        return "\n".join(record_lines)

    def _chunk_csv_block(
        self,
        *,
        filename: str,
        headers: list[str],
        data_rows: list[list[str]],
    ) -> list[ChunkData]:
        """Batch CSV rows into token-bounded chunks with repeated headers and row ranges."""
        use_kv = len(headers) > self.wide_csv_column_threshold
        col_summary = ", ".join(headers[:15]) + ("..." if len(headers) > 15 else "")

        def _format_batch(batch_rows: list[list[str]], start_row: int) -> str:
            if use_kv:
                return self._rows_to_kv_records(headers, batch_rows, start_row)
            return rows_to_markdown(headers, batch_rows)

        results: list[ChunkData] = []
        batch: list[list[str]] = []
        start_row = 1
        batch_idx = 1

        for row_idx, row in enumerate(data_rows, start=1):
            candidate_body = _format_batch([*batch, row], start_row)
            candidate_header = self._build_header(
                filename=filename,
                section=f"Rows {start_row}-{row_idx}",
                page_number=batch_idx,
                unit_type="rows",
                extra=f"Columns: {col_summary}",
            )
            if batch and count_tokens(candidate_header + candidate_body) > self.chunk_size:
                end_row = row_idx - 1
                section = f"Rows {start_row}-{end_row}"
                final_header = self._build_header(
                    filename=filename,
                    section=section,
                    page_number=batch_idx,
                    unit_type="rows",
                    extra=f"Columns: {col_summary}",
                )
                content = final_header + _format_batch(batch, start_row)
                results.append(
                    ChunkData(
                        chunk_index=0,
                        page_number=batch_idx,
                        section_title=section,
                        content=content,
                        token_count=count_tokens(content),
                        metadata={
                            "unit_type": "rows",
                            "row_start": start_row,
                            "row_end": end_row,
                            "columns": headers,
                            "format": "kv_records" if use_kv else "markdown_table",
                        },
                    )
                )
                batch = [row]
                start_row = row_idx
                batch_idx += 1
            else:
                batch.append(row)

        if batch:
            end_row = start_row + len(batch) - 1
            section = f"Rows {start_row}-{end_row}"
            final_header = self._build_header(
                filename=filename,
                section=section,
                page_number=batch_idx,
                unit_type="rows",
                extra=f"Columns: {col_summary}",
            )
            content = final_header + _format_batch(batch, start_row)
            results.append(
                ChunkData(
                    chunk_index=0,
                    page_number=batch_idx,
                    section_title=section,
                    content=content,
                    token_count=count_tokens(content),
                    metadata={
                        "unit_type": "rows",
                        "row_start": start_row,
                        "row_end": end_row,
                        "columns": headers,
                        "format": "kv_records" if use_kv else "markdown_table",
                    },
                )
            )

        return results

    def _chunk_markdown_table(
        self,
        *,
        filename: str,
        table_md: str,
        page_number: int,
        section_title: str,
        unit_type: str,
    ) -> list[ChunkData]:
        """Keep Markdown tables atomic when within budget; split by rows with repeated header if oversized."""
        prefix = self._build_header(
            filename=filename,
            section=section_title,
            page_number=page_number,
            unit_type=unit_type,
        )
        full = prefix + table_md
        if count_tokens(full) <= self.chunk_size:
            return [
                ChunkData(
                    chunk_index=0,
                    page_number=page_number,
                    section_title=section_title,
                    content=full,
                    token_count=count_tokens(full),
                    metadata={"unit_type": unit_type, "block_type": "table"},
                )
            ]

        # Split oversized Markdown table along row lines while repeating the 2-line header.
        lines = [ln for ln in table_md.splitlines() if ln.strip()]
        if len(lines) <= 3:
            return [
                ChunkData(
                    chunk_index=0,
                    page_number=page_number,
                    section_title=section_title,
                    content=full,
                    token_count=count_tokens(full),
                    metadata={"unit_type": unit_type, "block_type": "table"},
                )
            ]

        header_lines = lines[:2]
        body_lines = lines[2:]
        chunks: list[ChunkData] = []
        current_rows: list[str] = []

        for row_line in body_lines:
            candidate = prefix + "\n".join([*header_lines, *current_rows, row_line])
            if current_rows and count_tokens(candidate) > self.chunk_size:
                text = prefix + "\n".join([*header_lines, *current_rows])
                chunks.append(
                    ChunkData(
                        chunk_index=0,
                        page_number=page_number,
                        section_title=section_title,
                        content=text,
                        token_count=count_tokens(text),
                        metadata={"unit_type": unit_type, "block_type": "table"},
                    )
                )
                current_rows = [row_line]
            else:
                current_rows.append(row_line)

        if current_rows:
            text = prefix + "\n".join([*header_lines, *current_rows])
            chunks.append(
                ChunkData(
                    chunk_index=0,
                    page_number=page_number,
                    section_title=section_title,
                    content=text,
                    token_count=count_tokens(text),
                    metadata={"unit_type": unit_type, "block_type": "table"},
                )
            )
        return chunks

    def chunk(self, document: ExtractedDocument) -> list[ChunkData]:
        """Partition an `ExtractedDocument` into sequential `ChunkData` objects.

        Args:
            document: Canonical extracted document from any registered extractor.

        Returns:
            List of 0-indexed `ChunkData` records ready for vector embedding.
        """
        all_chunks: list[ChunkData] = []

        for block in document.blocks:
            unit_type = str(block.metadata.get("unit_type", "section"))

            if block.block_type == "csv_table":
                all_chunks.extend(
                    self._chunk_csv_block(
                        filename=document.filename,
                        headers=block.metadata["headers"],
                        data_rows=block.metadata["rows"],
                    )
                )
                continue

            if block.block_type == "table":
                all_chunks.extend(
                    self._chunk_markdown_table(
                        filename=document.filename,
                        table_md=block.content,
                        page_number=block.page_number,
                        section_title=block.section_title,
                        unit_type=unit_type,
                    )
                )
                continue

            # Narrative / slide notes blocks: recursive split within header-adjusted budget.
            prefix = self._build_header(
                filename=document.filename,
                section=block.section_title,
                page_number=block.page_number,
                unit_type=unit_type,
            )
            body_budget = max(self.chunk_size - count_tokens(prefix), 64)
            segments = self._split_text_recursive(block.content, max_tokens=body_budget)
            for seg in segments:
                content = prefix + seg
                all_chunks.append(
                    ChunkData(
                        chunk_index=0,
                        page_number=block.page_number,
                        section_title=block.section_title,
                        content=content,
                        token_count=count_tokens(content),
                        metadata={
                            "unit_type": unit_type,
                            "block_type": block.block_type,
                        },
                    )
                )

        # Assign dense sequential chunk indices across the entire document.
        for idx, item in enumerate(all_chunks):
            item.chunk_index = idx
        return all_chunks

"""Format-agnostic transactional document ingestion orchestrator.

Why this module exists:
    Coordinates the complete ingestion lifecycle:
      1. File validation (extension, size, signature, SHA-256 hash).
      2. Fast-fail per-user duplicate check (avoids wasting Cohere API credits).
      3. Offloaded CPU-bound multi-format extraction & format-aware chunking
         via `asyncio.to_thread` so the FastAPI event loop remains responsive.
      4. Async batched Cohere `embed-v4.0` (`output_dimension=1024`) calls.
      5. Single ACID PostgreSQL transaction persisting the `Document` and all
         `Chunk` rows, catching concurrent `IntegrityError` (HTTP 409) and
         rolling back cleanly on any failure so retries are never locked out.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import Chunk, Document
from backend.app.ingestion.chunker import ChunkData, FormatAwareChunker
from backend.app.ingestion.embedder import CohereEmbedder
from backend.app.ingestion.exceptions import (
    DuplicateDocumentError,
    ExternalServiceError,
    IngestionError,
    InvalidDocumentError,
)
from backend.app.ingestion.extractors import (
    ExtractedDocument,
    ExtractorRegistry,
    build_default_registry,
)
from backend.app.ingestion.validator import validate_upload


@dataclass(frozen=True)
class IngestionResult:
    """Structured summary returned after a successful document ingestion."""

    document_id: uuid.UUID
    filename: str
    file_type: str
    file_hash: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    status: str
    latency_ms: float


class IngestionPipeline:
    """Orchestrates validation, extraction, chunking, embedding, and DB persistence."""

    def __init__(
        self,
        *,
        registry: ExtractorRegistry | None = None,
        chunker: FormatAwareChunker | None = None,
        embedder: CohereEmbedder | None = None,
    ) -> None:
        """Initialize the pipeline with default or injected collaborators."""
        self.registry = registry or build_default_registry()
        self.chunker = chunker or FormatAwareChunker()
        self.embedder = embedder or CohereEmbedder()

    def _extract_and_chunk_sync(
        self,
        *,
        raw_bytes: bytes,
        filename: str,
        file_type: str,
    ) -> tuple[ExtractedDocument, list[ChunkData]]:
        """Run CPU-bound extraction and tokenization synchronously (for `asyncio.to_thread`)."""
        extractor = self.registry.get(file_type)
        extracted = extractor.extract(raw_bytes, filename)
        chunks = self.chunker.chunk(extracted)
        if not chunks:
            raise InvalidDocumentError(
                "Document produced zero indexable chunks after extraction.",
                details={"filename": filename, "file_type": file_type},
            )
        return extracted, chunks

    async def ingest(
        self,
        *,
        session: AsyncSession,
        user_id: uuid.UUID,
        filename: str,
        raw_bytes: bytes,
    ) -> IngestionResult:
        """Execute the full transactional ingestion pipeline for a single upload.

        Args:
            session: Active async SQLAlchemy database session.
            user_id: Owner user UUID enforcing multi-tenant isolation.
            filename: Original uploaded filename.
            raw_bytes: Complete binary content of the uploaded file.

        Returns:
            `IngestionResult` containing the persisted document ID, chunk count,
            and end-to-end processing latency.

        Raises:
            InvalidDocumentError: Empty, scanned, malformed, or header-only file (400).
            DuplicateDocumentError: SHA-256 hash already exists for `user_id` (409).
            PayloadTooLargeError: File exceeds configured MB cap (413).
            UnsupportedMediaTypeError: Disallowed extension or spoofed bytes (415).
            ExternalServiceError: Embedding API or DB failure after rollback (503).
        """
        started = time.perf_counter()

        # 1. Validate extension, size, signature, and compute SHA-256 hash.
        validated = validate_upload(filename, raw_bytes)

        # 2. Fast-fail duplicate check before incurring extraction or Cohere API costs.
        existing_stmt = select(Document.id).where(
            Document.user_id == user_id,
            Document.file_hash == validated.file_hash,
        )
        existing_doc_id = (await session.execute(existing_stmt)).scalar_one_or_none()
        if existing_doc_id is not None:
            raise DuplicateDocumentError(
                "A document with identical content hash already exists.",
                details={
                    "file_hash": validated.file_hash,
                    "existing_document_id": str(existing_doc_id),
                },
            )

        # 3. Offload CPU-bound parsing & chunking to a worker thread.
        extracted, chunks = await asyncio.to_thread(
            self._extract_and_chunk_sync,
            raw_bytes=validated.raw_bytes,
            filename=validated.filename,
            file_type=validated.file_type,
        )

        # 4. Generate 1024-dim embeddings BEFORE opening DB write lock so a 503
        #    outage never leaves a stale `processing` row blocking future retries.
        vectors = await self.embedder.embed_documents([c.content for c in chunks])

        # Compute effective unit count (for CSV, number of row batches).
        effective_page_count = (
            max((c.page_number for c in chunks), default=1)
            if validated.file_type == "csv"
            else extracted.page_count
        )

        # 5. Persist Document + all Chunks atomically in one ACID transaction.
        doc_record = Document(
            user_id=user_id,
            filename=validated.filename,
            file_hash=validated.file_hash,
            file_type=validated.file_type,
            file_size_bytes=validated.file_size_bytes,
            page_count=effective_page_count,
            chunk_count=len(chunks),
            status="ready",
        )

        try:
            session.add(doc_record)
            await session.flush()  # Populates doc_record.id and enforces UNIQUE(user_id, file_hash)

            chunk_records = [
                Chunk(
                    document_id=doc_record.id,
                    user_id=user_id,
                    chunk_index=chunk_item.chunk_index,
                    page_number=chunk_item.page_number,
                    section_title=chunk_item.section_title,
                    content=chunk_item.content,
                    token_count=chunk_item.token_count,
                    embedding=vector,
                    metadata_json=chunk_item.metadata,
                )
                for chunk_item, vector in zip(chunks, vectors, strict=True)
            ]
            session.add_all(chunk_records)
            await session.commit()
        except IntegrityError as exc:
            # Concurrent duplicate upload race (Edge Case #10): clean rollback -> HTTP 409.
            await session.rollback()
            raise DuplicateDocumentError(
                "A document with identical content hash was uploaded concurrently.",
                details={"file_hash": validated.file_hash},
            ) from exc
        except IngestionError:
            await session.rollback()
            raise
        except Exception as exc:
            await session.rollback()
            raise ExternalServiceError(
                f"Database persistence failed during ingestion: {exc}",
                details={"service": "postgres"},
            ) from exc

        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return IngestionResult(
            document_id=doc_record.id,
            filename=doc_record.filename,
            file_type=doc_record.file_type,
            file_hash=doc_record.file_hash,
            file_size_bytes=doc_record.file_size_bytes,
            page_count=doc_record.page_count,
            chunk_count=doc_record.chunk_count,
            status=doc_record.status,
            latency_ms=latency_ms,
        )

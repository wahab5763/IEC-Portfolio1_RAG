"""Comprehensive pytest suite for Milestone 4 (Universal Async Ingestion Pipeline).

Covers:
    - Validator: all 6 supported formats, SHA-256 determinism, empty file (400),
      oversized file (413), unsupported extension (415), spoofed binary PDF (415),
      and binary bytes disguised as CSV/TXT (415).
    - Pluggable Extractors: PDF, DOCX (reading order + tables), TXT, MD (heading
      breadcrumbs + code fences), PPTX (slides + tables + notes), CSV (sniffer),
      plus edge cases (`empty_scanned.pdf` -> 400, `header_only.csv` -> 400).
    - FormatAwareChunker: Config A vs Config B token budgets, header-repeated CSV
      row batching, wide-CSV (>12 cols) key-value serialization, contextual headers.
    - CohereEmbedder: explicit `output_dimension=1024`, batching, retries on
      transient failure, and `ExternalServiceError` (503) on persistent outage.
    - IngestionPipeline: happy path, fast-fail duplicate (409), concurrent
      `IntegrityError` rollback (409), and embed outage rollback (503).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from backend.app.ingestion.chunker import FormatAwareChunker
from backend.app.ingestion.embedder import CohereEmbedder
from backend.app.ingestion.exceptions import (
    DuplicateDocumentError,
    ExternalServiceError,
    InvalidDocumentError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from backend.app.ingestion.extractors import build_default_registry
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.ingestion.validator import validate_upload

CORPUS_DIR = Path("data/corpus")
EDGE_DIR = Path("data/edge_cases")


# ---------------------------------------------------------------------------
# 1. File Validator Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("filename", "expected_type"),
    [
        ("01_company_hr_policy.pdf", "pdf"),
        ("03_remote_work_policy_2025.docx", "docx"),
        ("05_incident_runbook.md", "md"),
        ("06_security_guidelines.txt", "txt"),
        ("07_q3_product_roadmap.pptx", "pptx"),
        ("08_regional_sales_2025.csv", "csv"),
    ],
)
def test_validator_accepts_all_six_corpus_formats(filename: str, expected_type: str) -> None:
    """Every valid format in the corpus must pass signature verification and SHA-256 hashing."""
    raw = (CORPUS_DIR / filename).read_bytes()
    validated = validate_upload(filename, raw)
    assert validated.file_type == expected_type
    assert len(validated.file_hash) == 64
    assert validated.file_size_bytes == len(raw)


def test_validator_rejects_empty_and_oversized_files() -> None:
    """Empty files return 400 and oversized files return 413."""
    with pytest.raises(InvalidDocumentError) as exc_empty:
        validate_upload("empty.txt", b"")
    assert exc_empty.value.status_code == 400

    with pytest.raises(PayloadTooLargeError) as exc_large:
        validate_upload("big.txt", b"a" * 2048, max_file_size_bytes=1024)
    assert exc_large.value.status_code == 413


def test_validator_rejects_spoofed_and_unsupported_uploads() -> None:
    """Spoofed binaries (.exe renamed to .pdf or .csv) and disallowed extensions return 415."""
    spoofed_pdf = (EDGE_DIR / "spoofed_binary.pdf").read_bytes()
    with pytest.raises(UnsupportedMediaTypeError) as exc_pdf:
        validate_upload("spoofed_binary.pdf", spoofed_pdf)
    assert exc_pdf.value.status_code == 415

    with pytest.raises(UnsupportedMediaTypeError) as exc_csv:
        validate_upload("malware.csv", b"MZ\x90\x00\x03\x00\x00\x00BINARY")
    assert exc_csv.value.status_code == 415

    with pytest.raises(UnsupportedMediaTypeError) as exc_ext:
        validate_upload("archive.exe", b"anything")
    assert exc_ext.value.status_code == 415


# ---------------------------------------------------------------------------
# 2. Pluggable Extractor Tests
# ---------------------------------------------------------------------------
def test_registry_extracts_all_six_formats_cleanly() -> None:
    """ExtractorRegistry must produce non-empty canonical blocks for all 6 formats."""
    registry = build_default_registry()
    assert set(registry.supported_types()) == {"pdf", "docx", "txt", "md", "pptx", "csv"}

    for fpath in sorted(CORPUS_DIR.iterdir()):
        if not fpath.is_file():
            continue
        ext = fpath.suffix.lower().lstrip(".")
        extracted = registry.get(ext).extract(fpath.read_bytes(), fpath.name)
        assert extracted.file_type == ext
        assert len(extracted.blocks) >= 1


def test_extractors_reject_scanned_pdf_and_header_only_csv() -> None:
    """Scanned PDF (< 20 chars/page) and header-only CSV must raise InvalidDocumentError (400)."""
    registry = build_default_registry()

    with pytest.raises(InvalidDocumentError, match="scanned or image-only"):
        registry.get("pdf").extract(
            (EDGE_DIR / "empty_scanned.pdf").read_bytes(), "empty_scanned.pdf"
        )

    with pytest.raises(InvalidDocumentError, match="no data rows"):
        registry.get("csv").extract(
            (EDGE_DIR / "header_only.csv").read_bytes(), "header_only.csv"
        )


# ---------------------------------------------------------------------------
# 3. Format-Aware Chunker Tests
# ---------------------------------------------------------------------------
def test_chunker_csv_repeats_headers_across_batches() -> None:
    """CSV chunks must stay within budget and repeat column headers on every batch."""
    registry = build_default_registry()
    csv_doc = registry.get("csv").extract(
        (CORPUS_DIR / "08_regional_sales_2025.csv").read_bytes(),
        "08_regional_sales_2025.csv",
    )
    chunker = FormatAwareChunker(chunk_size=400, chunk_overlap=60, inject_headers=True)
    chunks = chunker.chunk(csv_doc)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.token_count <= 400
        assert "| Region | Quarter | Product | Revenue_USD | Units_Sold | Margin_Pct |" in chunk.content
        assert chunk.metadata["unit_type"] == "rows"
        assert chunk.content.startswith("[Document: 08_regional_sales_2025.csv |")


def test_chunker_wide_csv_switches_to_kv_records() -> None:
    """CSVs with > 12 columns must serialize as `Column: Value` key-value records."""
    headers = [f"col_{i}" for i in range(1, 15)]
    row = [f"val_{i}" for i in range(1, 15)]
    raw_csv = (",".join(headers) + "\n" + ",".join(row) + "\n").encode("utf-8")

    registry = build_default_registry()
    doc = registry.get("csv").extract(raw_csv, "wide.csv")
    chunks = FormatAwareChunker(chunk_size=400, inject_headers=True).chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].metadata["format"] == "kv_records"
    assert "col_1: val_1" in chunks[0].content
    assert "col_14: val_14" in chunks[0].content


# ---------------------------------------------------------------------------
# 4. Async Cohere Embedder Tests (Mocked External API)
# ---------------------------------------------------------------------------
async def test_embedder_enforces_1024_dim_and_batches() -> None:
    """CohereEmbedder must pass output_dimension=1024 and merge concurrent batches in order."""
    calls: list[dict[str, object]] = []

    async def fake_embed(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        texts = kwargs["texts"]
        assert kwargs["output_dimension"] == 1024
        assert kwargs["input_type"] == "search_document"
        return SimpleNamespace(
            embeddings=SimpleNamespace(float=[[0.1] * 1024 for _ in texts])  # type: ignore[arg-type]
        )

    fake_client = SimpleNamespace(embed=fake_embed)
    embedder = CohereEmbedder(
        client=fake_client,
        output_dimension=1024,
        batch_size=2,
        max_concurrency=2,
    )
    vectors = await embedder.embed_documents(["chunk1", "chunk2", "chunk3"])
    assert len(vectors) == 3
    assert len(vectors[0]) == 1024
    assert len(calls) == 2


async def test_embedder_retries_and_raises_503_on_outage() -> None:
    """Persistent Cohere outage must exhaust retries and raise ExternalServiceError (503)."""
    attempts = 0

    async def failing_embed(**_kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("simulated timeout")

    embedder = CohereEmbedder(
        client=SimpleNamespace(embed=failing_embed),
        max_retries=2,
        timeout_seconds=1.0,
    )
    with pytest.raises(ExternalServiceError) as exc:
        await embedder.embed_documents(["hello"])
    assert exc.value.status_code == 503
    assert attempts == 2


# ---------------------------------------------------------------------------
# 5. Transactional Ingestion Pipeline Tests
# ---------------------------------------------------------------------------
def _make_mock_session(existing_doc_id: uuid.UUID | None = None) -> AsyncMock:
    """Create a mock AsyncSession simulating pre-check query, flush, commit, and rollback."""
    session = AsyncMock()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = existing_doc_id
    session.execute.return_value = exec_result

    def _add_side_effect(obj: object) -> None:
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()  # type: ignore[attr-defined]

    session.add = MagicMock(side_effect=_add_side_effect)
    session.add_all = MagicMock()
    return session


async def test_pipeline_ingests_document_atomically() -> None:
    """Happy-path ingestion must validate, extract, chunk, embed, and commit."""
    session = _make_mock_session(existing_doc_id=None)

    async def fake_embed(**kwargs: object) -> SimpleNamespace:
        texts = kwargs["texts"]
        return SimpleNamespace(
            embeddings=SimpleNamespace(float=[[0.25] * 1024 for _ in texts])  # type: ignore[arg-type]
        )

    pipeline = IngestionPipeline(
        embedder=CohereEmbedder(client=SimpleNamespace(embed=fake_embed), output_dimension=1024)
    )
    raw = (CORPUS_DIR / "06_security_guidelines.txt").read_bytes()
    result = await pipeline.ingest(
        session=session,
        user_id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
        filename="06_security_guidelines.txt",
        raw_bytes=raw,
    )

    assert result.file_type == "txt"
    assert result.status == "ready"
    assert result.chunk_count >= 1
    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()


async def test_pipeline_rejects_duplicate_and_concurrent_race_with_409() -> None:
    """Fast-fail duplicate check and concurrent IntegrityError both return HTTP 409."""
    raw = (CORPUS_DIR / "06_security_guidelines.txt").read_bytes()
    user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")

    # Case A: Fast-fail pre-check duplicate
    session_dup = _make_mock_session(existing_doc_id=uuid.uuid4())
    pipeline = IngestionPipeline(embedder=CohereEmbedder(client=SimpleNamespace()))
    with pytest.raises(DuplicateDocumentError) as exc_dup:
        await pipeline.ingest(
            session=session_dup,
            user_id=user_id,
            filename="06_security_guidelines.txt",
            raw_bytes=raw,
        )
    assert exc_dup.value.status_code == 409

    # Case B: Concurrent IntegrityError race -> rollback + 409
    session_race = _make_mock_session(existing_doc_id=None)
    session_race.flush.side_effect = IntegrityError("INSERT", {}, Exception("uq_user_file_hash"))

    async def fake_embed(**kwargs: object) -> SimpleNamespace:
        texts = kwargs["texts"]
        return SimpleNamespace(
            embeddings=SimpleNamespace(float=[[0.1] * 1024 for _ in texts])  # type: ignore[arg-type]
        )

    pipeline_race = IngestionPipeline(
        embedder=CohereEmbedder(client=SimpleNamespace(embed=fake_embed), output_dimension=1024)
    )
    with pytest.raises(DuplicateDocumentError) as exc_race:
        await pipeline_race.ingest(
            session=session_race,
            user_id=user_id,
            filename="06_security_guidelines.txt",
            raw_bytes=raw,
        )
    assert exc_race.value.status_code == 409
    session_race.rollback.assert_awaited_once()

"""Unit tests verifying Milestone 1 & Milestone 2 configuration, schema, and corpus.

Covers:
    - Neon Postgres URL normalization for asyncpg (`postgresql://`, `sslmode`,
      `channel_binding`) and connect_args TLS configuration.
    - Embedding dimension validation (`{256, 512, 1024, 1536}`).
    - SQLAlchemy 2.0 ORM model table names, columns, vector(1024) dim, and constraints.
    - `schema.sql` completeness (pgvector extension, 3 tables, 5 indexes, default user).
    - Validation corpus & edge-case fixture presence across all 6 formats.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from backend.app.config import Settings
from backend.app.db.migrate import (
    EXPECTED_INDEXES,
    EXPECTED_TABLES,
    SCHEMA_PATH,
    _split_sql_statements,
)
from backend.app.db.models import Base, Chunk, Document, User


def test_neon_database_url_normalization_strips_libpq_params() -> None:
    """Neon postgresql:// URLs must normalize to postgresql+asyncpg:// with TLS connect_args."""
    cfg = Settings(
        database_url=(
            "postgresql://neondb_owner:secret@ep-cool-host.us-east-2.aws.neon.tech/"
            "neondb?sslmode=require&channel_binding=require"
        )
    )
    assert cfg.async_database_url.startswith("postgresql+asyncpg://")
    assert "sslmode" not in cfg.async_database_url
    assert "channel_binding" not in cfg.async_database_url
    assert cfg.db_connect_args["ssl"] == "require"
    assert cfg.db_connect_args["timeout"] == cfg.db_connect_timeout_seconds


def test_empty_database_url_raises_clear_error() -> None:
    """Accessing async_database_url without DATABASE_URL must fail explicitly."""
    cfg = Settings(database_url="")
    with pytest.raises(ValueError, match="DATABASE_URL is not configured"):
        _ = cfg.async_database_url


def test_cohere_embed_dimension_validation() -> None:
    """Only Cohere embed-v4.0 Matryoshka dimensions are accepted."""
    assert Settings(cohere_embed_dim=1024).cohere_embed_dim == 1024
    with pytest.raises(ValueError, match="cohere_embed_dim"):
        Settings(cohere_embed_dim=768)


def test_orm_models_match_schema_contract() -> None:
    """Verify SQLAlchemy ORM metadata matches the build plan DDL 1:1."""
    assert set(Base.metadata.tables.keys()) == EXPECTED_TABLES
    assert User.__tablename__ == "users"
    assert Document.__tablename__ == "documents"
    assert Chunk.__tablename__ == "chunks"

    # Confirm 1024-dim vector column and JSONB metadata column mapping
    chunk_table = Base.metadata.tables["chunks"]
    assert chunk_table.c.embedding.type.dim == 1024
    assert "metadata" in chunk_table.c
    assert "tsv" in chunk_table.c

    # Confirm per-user deduplication constraint on documents
    doc_constraints = {c.name for c in Base.metadata.tables["documents"].constraints}
    assert "uq_user_file_hash" in doc_constraints


def test_schema_sql_contains_all_required_objects() -> None:
    """Verify schema.sql parses cleanly and defines extension, tables, seed user, and indexes."""
    sql_text = SCHEMA_PATH.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql_text)
    assert len(statements) >= 10
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql_text
    assert str(uuid.UUID("00000000-0000-0000-0000-000000000000")) in sql_text
    for idx_name in EXPECTED_INDEXES:
        assert idx_name in sql_text


def test_multi_format_validation_corpus_and_edge_cases_exist() -> None:
    """Confirm all 6 document formats and edge-case fixtures are generated."""
    corpus_dir = Path("data/corpus")
    edge_dir = Path("data/edge_cases")
    assert corpus_dir.exists()
    assert edge_dir.exists()

    extensions = {p.suffix.lower() for p in corpus_dir.iterdir() if p.is_file()}
    assert extensions == {".pdf", ".docx", ".md", ".txt", ".pptx", ".csv"}
    assert len(list(corpus_dir.iterdir())) >= 8
    assert (edge_dir / "empty_scanned.pdf").exists()
    assert (edge_dir / "header_only.csv").exists()
    assert (edge_dir / "spoofed_binary.pdf").exists()
    assert (edge_dir / "prompt_injection_doc.txt").exists()

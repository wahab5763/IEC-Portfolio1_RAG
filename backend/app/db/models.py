"""SQLAlchemy 2.0 async ORM models mapping the Neon Postgres + pgvector schema.

Why this module exists:
    Provides strongly typed relational and vector mappings (`User`, `Document`,
    `Chunk`) that mirror `schema.sql` 1:1. Keeping cascade delete rules on both
    the database foreign keys and ORM relationships guarantees that deleting a
    document atomically removes every associated chunk and vector embedding.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CHAR,
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base class for all ORM models in the RAG service."""


class User(Base):
    """Tenant / user account model.

    In Phase 1 single-tenant mode, all documents are owned by the seeded system
    user (`00000000-0000-0000-0000-000000000000`). In Phase 2, JWT auth isolates
    each registered user's documents and chunks by `user_id`.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )

    # Relationships with database-backed cascade deletes
    documents: Mapped[list[Document]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Document(Base):
    """Uploaded document metadata and deduplication record.

    Enforces per-user deduplication via `UNIQUE (user_id, file_hash)` so
    concurrent or repeated uploads of identical content fail cleanly with
    `IntegrityError` (mapped to HTTP 409 Conflict at the API layer).
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("user_id", "file_hash", name="uq_user_file_hash"),
        Index("idx_documents_user_id", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    file_type: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=1, server_default=text("1"))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="processing",
        server_default=text("'processing'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )

    user: Mapped[User] = relationship(back_populates="documents")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(Base):
    """Indexed document chunk storing text, dense vector, and sparse tsvector.

    Stores:
        - `embedding`: 1024-dim float vector from Cohere `embed-v4.0`
          (`output_dimension=1024`), indexed with HNSW (`vector_cosine_ops`).
        - `tsv`: PostgreSQL generated stored `tsvector` using `'simple'` config
          (avoids English stemming corruption on product codes & multilingual
          text), indexed with GIN.
        - `metadata_json`: Format-specific provenance (`unit_type`, `row_start`,
          `row_end`, `columns`) mapped to the `metadata` JSONB column.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        Index(
            "idx_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
        Index("idx_chunks_tsv_gin", "tsv", postgresql_using="gin"),
        Index("idx_chunks_document_id", "document_id"),
        Index("idx_chunks_user_id", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, default=1, server_default=text("1"))
    section_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', content)", persisted=True),
        nullable=True,
    )
    # Attribute named `metadata_json` to avoid shadowing DeclarativeBase.metadata
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
    user: Mapped[User] = relationship(back_populates="chunks")

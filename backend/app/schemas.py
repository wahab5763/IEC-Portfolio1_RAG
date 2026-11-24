"""Strict Pydantic v2 request and response schemas for the API v1 contract.

Why this module exists:
    Centralizes request validation, OpenAPI Swagger documentation examples, and
    response serialization across all endpoints, guaranteeing every success and
    error payload conforms strictly to the documented API contract.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Standardized JSON error envelope returned by all non-2xx responses."""

    request_id: str = Field(..., description="Unique request correlation identifier.")
    error_code: str = Field(..., description="Machine-readable uppercase error code.")
    message: str = Field(..., description="Human-readable error description.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured context for debugging.",
    )


class HealthResponse(BaseModel):
    """Response model for `GET /api/v1/health`."""

    status: Literal["healthy"] = "healthy"
    database: str = Field(..., examples=["connected"])
    pgvector_enabled: bool = Field(..., examples=[True])
    pgvector_version: str = Field(..., examples=["0.8.0"])
    request_id: str


class DocumentUploadResponse(BaseModel):
    """Response model for `POST /api/v1/documents` (HTTP 201 Created)."""

    document_id: uuid.UUID
    filename: str
    file_type: str = Field(..., examples=["pdf", "docx", "txt", "md", "pptx", "csv"])
    file_hash: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    status: str = Field(..., examples=["ready"])
    latency_ms: float
    request_id: str


class DocumentSummary(BaseModel):
    """Single document item returned in paginated `GET /api/v1/documents` listings."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    status: str
    created_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated response model for `GET /api/v1/documents`."""

    items: list[DocumentSummary]
    page: int
    limit: int
    total: int
    request_id: str


class AskRequest(BaseModel):
    """Request payload for `POST /api/v1/ask`."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Plain-language question to answer strictly from uploaded documents.",
        examples=["How many days of paid annual leave do full-time employees receive?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional override for number of chunks retrieved (default: 5).",
    )
    mode: Literal["hybrid_rerank", "dense_only"] = Field(
        default="hybrid_rerank",
        description="Retrieval pipeline mode: Config B (`hybrid_rerank`) or Config A (`dense_only`).",
    )


class CitationItem(BaseModel):
    """Verified inline citation referencing an actually retrieved chunk."""

    chunk_id: str
    filename: str
    page_number: int
    section_title: str | None = None


class RetrievedChunkItem(BaseModel):
    """Full chunk inspection record displayed in the frontend Retrieved Chunks Inspector."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section_title: str | None = None
    content: str
    similarity_score: float = Field(..., description="Dense cosine similarity score.")
    rrf_score: float | None = Field(default=None, description="Reciprocal Rank Fusion score.")
    rerank_score: float | None = Field(
        default=None,
        description="Calibrated Cohere rerank-v5.0-fast relevance score.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """Response model for `POST /api/v1/ask`."""

    request_id: str
    answer: str
    citations: list[CitationItem]
    retrieved_chunks: list[RetrievedChunkItem]
    retrieved_count: int
    refused: bool
    refusal_reason: str | None = None
    guardrail_violations: list[str] = Field(default_factory=list)
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    model: str

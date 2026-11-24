"""Document lifecycle routes (`POST`, `GET`, `DELETE /api/v1/documents`).

Business logic resides strictly in `IngestionPipeline`; this router maps HTTP
requests/responses and enforces multi-tenant `user_id` scoping.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.v1.dependencies import get_current_user_id, get_ingestion_pipeline
from backend.app.db.models import Document
from backend.app.db.session import get_db
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.observability import get_request_id
from backend.app.schemas import (
    DocumentListResponse,
    DocumentSummary,
    DocumentUploadResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Empty, corrupted, scanned PDF, or header-only CSV"},
        409: {"model": ErrorResponse, "description": "Duplicate document (SHA-256 conflict)"},
        413: {"model": ErrorResponse, "description": "File exceeds maximum size limit"},
        415: {"model": ErrorResponse, "description": "Unsupported format or spoofed content signature"},
        503: {"model": ErrorResponse, "description": "Embedding API or database unavailable"},
    },
    summary="Upload & Ingest Document",
    description="Validates, extracts, chunks, embeds, and indexes `.pdf`, `.docx`, `.txt`, `.md`, `.pptx`, or `.csv` files.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to ingest."),
    session: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> DocumentUploadResponse:
    """Ingest a single uploaded file into Neon Postgres + pgvector."""
    raw_bytes = await file.read()
    result = await pipeline.ingest(
        session=session,
        user_id=user_id,
        filename=file.filename or "",
        raw_bytes=raw_bytes,
    )
    return DocumentUploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        file_type=result.file_type,
        file_hash=result.file_hash,
        file_size_bytes=result.file_size_bytes,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        status=result.status,
        latency_ms=result.latency_ms,
        request_id=get_request_id(),
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List Uploaded Documents",
    description="Returns a paginated list of documents owned by the active user.",
)
async def list_documents(
    page: int = Query(default=1, ge=1, description="1-based page index."),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page."),
    session: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
) -> DocumentListResponse:
    """Return paginated document metadata for `user_id`."""
    total_stmt = select(func.count(Document.id)).where(Document.user_id == user_id)
    total = int((await session.execute(total_stmt)).scalar_one())

    offset = (page - 1) * limit
    items_stmt = (
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    docs = (await session.execute(items_stmt)).scalars().all()

    return DocumentListResponse(
        items=[DocumentSummary.model_validate(d) for d in docs],
        page=page,
        limit=limit,
        total=total,
        request_id=get_request_id(),
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
    summary="Delete Document & Associated Vectors",
    description="Permanently deletes the document and cascades removal of all vector chunks.",
)
async def delete_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
) -> Response:
    """Delete a document owned by `user_id` via `ON DELETE CASCADE`."""
    stmt = select(Document).where(Document.id == document_id, Document.user_id == user_id)
    doc = (await session.execute(stmt)).scalar_one_or_none()
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    await session.delete(doc)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

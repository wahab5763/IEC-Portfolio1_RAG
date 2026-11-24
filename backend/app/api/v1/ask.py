"""Question-answering route (`POST /api/v1/ask`) with rate limiting and guardrails."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.v1.dependencies import (
    get_current_user_id,
    get_generator,
    get_retriever,
    limiter,
)
from backend.app.config import get_settings
from backend.app.db.session import get_db
from backend.app.generation import GroundedGenerator
from backend.app.guardrails import check_query_input
from backend.app.ingestion.exceptions import IngestionError
from backend.app.observability import get_request_id
from backend.app.retrieval import HybridRetriever
from backend.app.schemas import (
    AskRequest,
    AskResponse,
    CitationItem,
    ErrorResponse,
    RetrievedChunkItem,
)

router = APIRouter(tags=["Question Answering"])


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Prompt injection or jailbreak blocked"},
        422: {"model": ErrorResponse, "description": "Query validation error (empty or >1000 chars)"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Cohere, Groq, or Neon DB unavailable"},
    },
    summary="Ask Question Over Uploaded Documents",
    description=(
        "Screens input guardrails, executes Hybrid HNSW + FTS + RRF + Cohere Rerank retrieval, "
        "and generates a strictly grounded, cited answer via Groq LLM."
    ),
)
@limiter.limit(f"{get_settings().rate_limit_per_minute}/minute")
async def ask_question(
    request: Request,
    payload: AskRequest,
    session: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id),
    retriever: HybridRetriever = Depends(get_retriever),
    generator: GroundedGenerator = Depends(get_generator),
) -> AskResponse:
    """Run input guardrails, hybrid retrieval, and grounded generation."""
    started = time.perf_counter()
    settings = get_settings()

    # Layer 1: Input Guardrails (prompt injection / jailbreak screening)
    input_check = check_query_input(payload.question, max_length=settings.max_query_length)
    if not input_check.is_safe:
        raise IngestionError(
            input_check.reason or "Query rejected by input guardrails.",
            status_code=400,
            error_code="PROMPT_INJECTION_DETECTED",
            details={"matched_rule": input_check.matched_rule},
        )

    # Layer 2: Hybrid Dense + Sparse Retrieval + Rerank
    retrieval = await retriever.retrieve(
        session=session,
        query=payload.question,
        user_id=user_id,
        top_k=payload.top_k,
        mode=payload.mode,
    )

    # Layer 3: XML-Fenced Grounded Generation + Output Guardrails
    generation = await generator.generate(
        question=payload.question,
        retrieval=retrieval,
    )

    total_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return AskResponse(
        request_id=get_request_id(),
        answer=generation.answer,
        citations=[
            CitationItem(
                chunk_id=c.chunk_id,
                filename=c.filename,
                page_number=c.page_number,
                section_title=c.section_title,
            )
            for c in generation.citations
        ],
        retrieved_chunks=[
            RetrievedChunkItem(
                chunk_id=rc.chunk_id,
                document_id=rc.document_id,
                filename=rc.filename,
                page_number=rc.page_number,
                section_title=rc.section_title,
                content=rc.content,
                similarity_score=rc.similarity_score,
                rrf_score=rc.rrf_score,
                rerank_score=rc.rerank_score,
                metadata=rc.metadata,
            )
            for rc in retrieval.chunks
        ],
        retrieved_count=len(retrieval.chunks),
        refused=generation.refused,
        refusal_reason=generation.refusal_reason,
        guardrail_violations=generation.guardrail_violations,
        retrieval_latency_ms=retrieval.latency_ms,
        generation_latency_ms=generation.latency_ms,
        total_latency_ms=total_ms,
        model=generation.model,
    )

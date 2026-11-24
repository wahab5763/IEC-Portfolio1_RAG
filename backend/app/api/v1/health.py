"""System health check route (`GET /api/v1/health`)."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.db.session import check_db_health
from backend.app.ingestion.exceptions import ExternalServiceError
from backend.app.observability import get_request_id
from backend.app.schemas import ErrorResponse, HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse, "description": "Database or pgvector unavailable"}},
    summary="System Health Check",
    description="Verifies Neon Postgres connectivity and confirms the `pgvector` extension is enabled.",
)
async def health_check() -> HealthResponse:
    """Ping PostgreSQL and verify `pgvector` availability."""
    try:
        status = await check_db_health(timeout_seconds=5.0)
    except Exception as exc:
        raise ExternalServiceError(
            f"Database health check failed: {exc}",
            details={"service": "postgres"},
        ) from exc

    return HealthResponse(
        status="healthy",
        database=str(status["database"]),
        pgvector_enabled=bool(status["pgvector_enabled"]),
        pgvector_version=str(status["pgvector_version"]),
        request_id=get_request_id(),
    )

"""FastAPI application entry point, middleware stack, and global error handlers.

Why this module exists:
    Assembles the HTTP application without any domain business logic: configures
    structured JSON logging, request correlation IDs, CORS, rate limiting
    (`slowapi`), standardized error envelopes across all HTTP status codes, and
    optional static serving of the Phase 1 HTML/CSS/JS frontend.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

from backend.app.api.v1.dependencies import limiter
from backend.app.api.v1.router import api_v1_router
from backend.app.config import get_settings
from backend.app.db.session import dispose_engine
from backend.app.ingestion.exceptions import IngestionError
from backend.app.observability import (
    RequestContextMiddleware,
    configure_logging,
    get_request_id,
)
from backend.app.schemas import ErrorResponse

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize logging on startup and close async DB pools cleanly on shutdown."""
    configure_logging()
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Production-grade Retrieval-Augmented Generation API supporting universal "
            "multi-format ingestion (PDF, DOCX, TXT, MD, PPTX, CSV), Neon Postgres + "
            "pgvector hybrid search with RRF & Cohere Rerank, grounded Groq LLM "
            "citations, and multi-layer security guardrails."
        ),
        lifespan=lifespan,
    )

    # Attach slowapi limiter state
    app.state.limiter = limiter

    # Middleware: CORS + Request ID / Structured Latency Logging
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    # --- Global Standardized Error Envelope Handlers ---
    @app.exception_handler(IngestionError)
    async def handle_domain_error(_request: Request, exc: IngestionError) -> JSONResponse:
        body = ErrorResponse(
            request_id=get_request_id(),
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        body = ErrorResponse(
            request_id=get_request_id(),
            error_code="VALIDATION_ERROR",
            message="Request payload failed schema validation.",
            details={"errors": exc.errors()},
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(RateLimitExceeded)
    async def handle_rate_limit(_request: Request, exc: RateLimitExceeded) -> JSONResponse:
        body = ErrorResponse(
            request_id=get_request_id(),
            error_code="RATE_LIMIT_EXCEEDED",
            message="Too many requests. Please slow down and try again shortly.",
            details={"limit": str(exc.detail)},
        )
        return JSONResponse(status_code=429, content=body.model_dump())

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        code_map = {401: "UNAUTHORIZED", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}
        body = ErrorResponse(
            request_id=get_request_id(),
            error_code=code_map.get(exc.status_code, f"HTTP_{exc.status_code}"),
            message=str(exc.detail),
            details={},
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("rag_service").exception("unhandled_server_error", exc_info=exc)
        body = ErrorResponse(
            request_id=get_request_id(),
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected internal error occurred.",
            details={},
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    # Mount versioned API routes
    app.include_router(api_v1_router)

    # Serve Phase 1 static HTML/CSS/JS frontend at `/` when `frontend/index.html` exists
    if (FRONTEND_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()

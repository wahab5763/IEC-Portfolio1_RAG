"""Structured JSON logging and per-request correlation ID middleware.

Why this module exists:
    Attaches a unique `request_id` (`req_<10hex>`) to every incoming HTTP
    request via `contextvars.ContextVar`, logs structured request latency and
    status codes without ever emitting secrets, and returns `X-Request-ID` in
    every HTTP response header for end-to-end traceability.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from fastapi import Request, Response
from pythonjsonlogger.json import JsonFormatter
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from backend.app.config import get_settings

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="req_system")


def get_request_id() -> str:
    """Return the active correlation ID for the current async execution context."""
    return request_id_ctx.get()


class _RequestIdFilter(logging.Filter):
    """Inject the current `request_id` into every structured log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def configure_logging() -> logging.Logger:
    """Configure root JSON logging once at application startup."""
    settings = get_settings()
    logger = logging.getLogger("rag_service")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)
    handler.addFilter(_RequestIdFilter())

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False
    return logger


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign `X-Request-ID` and log structured request duration and status."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:10]}"
        token = request_id_ctx.set(req_id)
        logger = logging.getLogger("rag_service")
        started = time.perf_counter()

        try:
            response = await call_next(request)
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            response.headers["X-Request-ID"] = req_id
            logger.info(
                "http_request_completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                },
            )
            return response
        finally:
            request_id_ctx.reset(token)

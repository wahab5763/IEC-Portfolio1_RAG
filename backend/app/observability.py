"""Structured JSON logging, request correlation IDs, rolling p50/p95 metrics, and Langfuse tracing.

Why this module exists:
    Provides full operational observability (Phase 2 Bonus C):
      1. `RequestContextMiddleware` assigns `X-Request-ID`, records HTTP latency
         and error rates into an in-memory rolling telemetry tracker, and emits
         structured JSON logs without ever logging secrets.
      2. `metrics_tracker` computes live p50/p95 latency, token consumption,
         estimated cost per query, and error rate exposed at `GET /api/v1/metrics`.
      3. `record_rag_trace` forwards retrieval, rerank, and generation spans to
         Langfuse whenever `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set,
         and degrades cleanly to a no-op when unconfigured.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections import deque
from contextvars import ContextVar
from typing import Any

from fastapi import Request, Response
from pythonjsonlogger.json import JsonFormatter
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from backend.app.config import get_settings

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="req_system")


def get_request_id() -> str:
    """Return the active correlation ID for the current async execution context."""
    return request_id_ctx.get()


class MetricsTracker:
    """Thread-safe rolling operational metrics collector for latency, tokens, cost, and errors."""

    def __init__(self, max_samples: int = 1000) -> None:
        self._latencies_ms: deque[float] = deque(maxlen=max_samples)
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.ask_requests: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    def record_http(self, *, status_code: int, latency_ms: float) -> None:
        """Record an HTTP request completion and error flag."""
        self.total_requests += 1
        if status_code >= 400:
            self.total_errors += 1
        self._latencies_ms.append(latency_ms)

    def record_ask(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        """Record LLM token usage for a `/api/v1/ask` request."""
        self.ask_requests += 1
        self.total_prompt_tokens += max(prompt_tokens, 0)
        self.total_completion_tokens += max(completion_tokens, 0)

    def _percentile(self, pct: float) -> float:
        if not self._latencies_ms:
            return 0.0
        ordered = sorted(self._latencies_ms)
        idx = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
        return round(ordered[idx], 2)

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable operational telemetry summary."""
        error_rate = (
            round(self.total_errors / self.total_requests, 4) if self.total_requests > 0 else 0.0
        )
        # Approximate blended Groq + Cohere API cost ($0.15/1M prompt + $0.60/1M completion)
        est_cost = (
            (self.total_prompt_tokens / 1_000_000.0) * 0.15
            + (self.total_completion_tokens / 1_000_000.0) * 0.60
        )
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": error_rate,
            "ask_requests": self.ask_requests,
            "p50_latency_ms": self._percentile(50.0),
            "p95_latency_ms": self._percentile(95.0),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "estimated_cost_usd": round(est_cost, 6),
            "langfuse_enabled": is_langfuse_enabled(),
        }


metrics_tracker = MetricsTracker()


def is_langfuse_enabled() -> bool:
    """Return True when both Langfuse API keys are configured in `.env`."""
    settings = get_settings()
    return bool(settings.langfuse_public_key.strip() and settings.langfuse_secret_key.strip())


def record_rag_trace(
    *,
    request_id: str,
    user_id: str,
    question: str,
    answer: str,
    model: str,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    retrieved_count: int,
    refused: bool,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Emit structured RAG trace spans to Langfuse when keys are configured.

    Degrades silently to a no-op if Langfuse is unconfigured or unreachable so
    observability never breaks user-facing `/api/v1/ask` responses.
    """
    metrics_tracker.record_ask(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    if not is_langfuse_enabled():
        return

    try:
        from langfuse import Langfuse

        settings = get_settings()
        lf = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        with lf.start_as_current_span(
            name="rag_ask_request",
            input={"question": question, "user_id": user_id, "request_id": request_id},
            metadata={
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "retrieved_count": retrieved_count,
                "refused": refused,
            },
        ) as root_span:
            root_span.update(
                output={"answer": answer, "model": model, "refused": refused},
            )
            with lf.start_as_current_observation(
                as_type="generation",
                name="groq_grounded_generation",
                model=model,
                input=question,
                output=answer,
                usage_details={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                },
            ):
                pass
        lf.flush()
    except Exception as exc:
        logging.getLogger("rag_service").warning(
            "langfuse_trace_failed",
            extra={"error": str(exc)},
        )


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
    """Assign `X-Request-ID`, record telemetry, and log structured request duration."""

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
            if request.url.path.startswith("/api/"):
                metrics_tracker.record_http(
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                )
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

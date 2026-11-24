"""FastAPI dependency injection providers, JWT Bearer auth guard, and rate limiter."""

from __future__ import annotations

import uuid
from functools import lru_cache

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.auth import AuthenticationError, decode_access_token
from backend.app.config import get_settings
from backend.app.generation import GroundedGenerator
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.retrieval import HybridRetriever

# Per-IP sliding window rate limiter attached to FastAPI app state in main.py
limiter = Limiter(key_func=get_remote_address)

# Optional Bearer scheme so we return our standardized JSON 401 envelope via AuthenticationError
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> uuid.UUID:
    """Resolve authenticated `user_id` from JWT Bearer token (or default user when auth disabled).

    Args:
        credentials: Parsed `Authorization: Bearer <jwt>` header if provided.

    Returns:
        Authenticated user UUID enforcing per-tenant DB isolation.

    Raises:
        AuthenticationError: If `auth_required=True` and token is missing/invalid/expired (401).
    """
    settings = get_settings()
    if credentials is not None and credentials.credentials:
        return decode_access_token(credentials.credentials)

    if settings.auth_required:
        raise AuthenticationError(
            "Authentication required. Provide a valid Bearer JWT token from /api/v1/auth/login."
        )
    return settings.default_user_id


@lru_cache(maxsize=1)
def get_ingestion_pipeline() -> IngestionPipeline:
    """Return a cached `IngestionPipeline` instance for document uploads."""
    return IngestionPipeline()


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Return a cached `HybridRetriever` instance for Q&A retrieval."""
    return HybridRetriever()


@lru_cache(maxsize=1)
def get_generator() -> GroundedGenerator:
    """Return a cached `GroundedGenerator` instance for Groq LLM inference."""
    return GroundedGenerator()

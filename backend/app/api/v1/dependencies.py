"""FastAPI dependency injection providers and rate limiter instance.

Why this module exists:
    Keeps route handlers thin and 100% testable via `app.dependency_overrides`.
    In Phase 1, `get_current_user_id` resolves to the seeded single-tenant
    system user; in Phase 2, swapping this single dependency enables full JWT
    multi-tenancy without modifying any route handler.
"""

from __future__ import annotations

import uuid
from functools import lru_cache

from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.config import get_settings
from backend.app.generation import GroundedGenerator
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.retrieval import HybridRetriever

# Per-IP sliding window rate limiter attached to FastAPI app state in main.py
limiter = Limiter(key_func=get_remote_address)


def get_current_user_id() -> uuid.UUID:
    """Return the active user UUID (default system user in Phase 1 single-tenant mode)."""
    return get_settings().default_user_id


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

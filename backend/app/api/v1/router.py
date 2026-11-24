"""Aggregated API v1 router mounting health, document, and Q&A endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.v1.ask import router as ask_router
from backend.app.api.v1.documents import router as documents_router
from backend.app.api.v1.health import router as health_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router)
api_v1_router.include_router(documents_router)
api_v1_router.include_router(ask_router)

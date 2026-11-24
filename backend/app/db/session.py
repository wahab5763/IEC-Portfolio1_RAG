"""Async SQLAlchemy engine, connection pool, session dependency, and DB health check.

Why this module exists:
    Centralizes Neon Serverless Postgres connection pooling with `pool_pre_ping`
    and `pool_recycle` so idle serverless scale-to-zero disconnects recover
    automatically without failing in-flight FastAPI requests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.app.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Create and cache the process-wide SQLAlchemy async engine.

    Returns:
        Configured `AsyncEngine` connected via `asyncpg`.

    Raises:
        ValueError: If `DATABASE_URL` is missing from environment configuration.
    """
    settings = get_settings()
    return create_async_engine(
        settings.async_database_url,
        echo=settings.debug,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        # Neon Serverless suspends idle compute; pre_ping & recycle prevent stale sockets.
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args=settings.db_connect_args,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return a cached async session factory bound to the shared engine.

    Returns:
        `async_sessionmaker` configured with `expire_on_commit=False` so ORM
        objects remain readable after transaction commit in async handlers.
    """
    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a scoped async database session.

    Yields:
        Active `AsyncSession` instance that rolls back on unhandled exceptions
        and closes cleanly when the request finishes.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def check_db_health(timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Verify database connectivity and confirm the `pgvector` extension is active.

    Args:
        timeout_seconds: Maximum time to wait for the health probe before failing.

    Returns:
        Dictionary containing `database`, `pgvector_enabled`, and
        `pgvector_version` status fields.

    Raises:
        RuntimeError: If the database is unreachable or `pgvector` is not installed.
    """

    async def _probe() -> dict[str, Any]:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            result = await conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            vector_version = result.scalar_one_or_none()
            if not vector_version:
                raise RuntimeError("pgvector extension is not enabled in the target database.")
            return {
                "database": "connected",
                "pgvector_enabled": True,
                "pgvector_version": str(vector_version),
            }

    try:
        return await asyncio.wait_for(_probe(), timeout=timeout_seconds)
    except TimeoutError as exc:
        raise RuntimeError(
            f"Database health check timed out after {timeout_seconds}s."
        ) from exc


async def dispose_engine() -> None:
    """Close all pooled database connections cleanly during application shutdown."""
    if get_engine.cache_info().currsize > 0:
        await get_engine().dispose()
        get_session_factory.cache_clear()
        get_engine.cache_clear()

"""Shared pytest fixtures for async FastAPI integration tests (with Phase 2 JWT Auth)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.auth import create_access_token
from backend.app.db.session import get_db
from backend.app.main import create_app


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Provide an isolated AsyncMock SQLAlchemy session."""
    return AsyncMock()


@pytest.fixture
async def api_client(mock_db_session: AsyncMock) -> AsyncIterator[tuple[AsyncClient, object]]:
    """Yield an authenticated `(httpx.AsyncClient, FastAPI)` tuple with a valid JWT Bearer header."""
    app = create_app()

    async def _override_get_db() -> AsyncIterator[AsyncMock]:
        yield mock_db_session

    app.dependency_overrides[get_db] = _override_get_db

    test_user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    token, _ = create_access_token(test_user_id, "tester@acme.internal")

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        yield client, app

    app.dependency_overrides.clear()


@pytest.fixture
async def unauthenticated_client(
    mock_db_session: AsyncMock,
) -> AsyncIterator[tuple[AsyncClient, object]]:
    """Yield an unauthenticated `(httpx.AsyncClient, FastAPI)` tuple for testing 401 & signup/login."""
    app = create_app()

    async def _override_get_db() -> AsyncIterator[AsyncMock]:
        yield mock_db_session

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, app

    app.dependency_overrides.clear()

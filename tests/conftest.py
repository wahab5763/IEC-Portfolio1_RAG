"""Shared pytest fixtures for async FastAPI integration tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.db.session import get_db
from backend.app.main import create_app


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Provide an isolated AsyncMock SQLAlchemy session."""
    return AsyncMock()


@pytest.fixture
async def api_client(mock_db_session: AsyncMock) -> AsyncIterator[tuple[AsyncClient, object]]:
    """Yield an `(httpx.AsyncClient, FastAPI)` tuple with DB dependency overridden."""
    app = create_app()

    async def _override_get_db() -> AsyncIterator[AsyncMock]:
        yield mock_db_session

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, app

    app.dependency_overrides.clear()

"""Phase 2 pytest suite: JWT Auth, Multi-Tenancy, Observability Metrics & Edge Cases."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.auth import (
    AuthenticationError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.app.guardrails import wrap_retrieved_context
from backend.app.observability import MetricsTracker
from backend.app.retrieval import RetrievedChunk


def test_argon2_password_hashing_and_jwt_roundtrip() -> None:
    """Passwords hash via Argon2 and HS256 JWT tokens encode/decode cleanly."""
    hashed = hash_password("SuperSecurePass123!")
    assert hashed != "SuperSecurePass123!"
    assert verify_password("SuperSecurePass123!", hashed) is True
    assert verify_password("WrongPassword", hashed) is False

    uid = uuid.uuid4()
    token, expires_in = create_access_token(uid, "alice@acme.internal")
    assert expires_in > 0
    assert decode_access_token(token) == uid

    with pytest.raises(AuthenticationError) as exc:
        decode_access_token("not.a.valid.jwt")
    assert exc.value.status_code == 401


async def test_protected_routes_require_valid_jwt_401(unauthenticated_client) -> None:
    """Documents and Ask routes must return 401 Unauthorized when no Bearer token is provided."""
    client, _app = unauthenticated_client

    docs_resp = await client.get("/api/v1/documents")
    assert docs_resp.status_code == 401
    assert docs_resp.json()["error_code"] == "UNAUTHORIZED"

    ask_resp = await client.post("/api/v1/ask", json={"question": "Any leave days?"})
    assert ask_resp.status_code == 401
    assert ask_resp.json()["error_code"] == "UNAUTHORIZED"


async def test_signup_and_login_flow_and_duplicate_email_409(
    unauthenticated_client,
    mock_db_session: AsyncMock,
) -> None:
    """POST /auth/signup returns 201 + JWT; duplicate returns 409; /login verifies Argon2."""
    client, _app = unauthenticated_client

    # 1. Signup success -> 201 Created
    not_found_res = MagicMock()
    not_found_res.scalar_one_or_none.return_value = None
    mock_db_session.execute.return_value = not_found_res

    def _add_user(obj: object) -> None:
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()  # type: ignore[attr-defined]
        if getattr(obj, "created_at", None) is None:
            obj.created_at = datetime.now(UTC)  # type: ignore[attr-defined]

    mock_db_session.add = MagicMock(side_effect=_add_user)

    signup_resp = await client.post(
        "/api/v1/auth/signup",
        json={"email": "alice@acme.internal", "password": "StrongPassword99!"},
    )
    assert signup_resp.status_code == 201
    data = signup_resp.json()
    assert data["token_type"] == "bearer"
    assert data["email"] == "alice@acme.internal"
    assert decode_access_token(data["access_token"]) == uuid.UUID(data["user_id"])

    # 2. Duplicate signup -> 409 Conflict
    dup_res = MagicMock()
    dup_res.scalar_one_or_none.return_value = uuid.uuid4()
    mock_db_session.execute.return_value = dup_res
    dup_resp = await client.post(
        "/api/v1/auth/signup",
        json={"email": "alice@acme.internal", "password": "StrongPassword99!"},
    )
    assert dup_resp.status_code == 409
    assert dup_resp.json()["error_code"] == "EMAIL_ALREADY_REGISTERED"

    # 3. Login success -> 200 OK & invalid password -> 401
    stored_user = MagicMock(
        id=uuid.uuid4(),
        email="alice@acme.internal",
        password_hash=hash_password("StrongPassword99!"),
    )
    found_user_res = MagicMock()
    found_user_res.scalar_one_or_none.return_value = stored_user
    mock_db_session.execute.return_value = found_user_res

    login_ok = await client.post(
        "/api/v1/auth/login",
        json={"email": "alice@acme.internal", "password": "StrongPassword99!"},
    )
    assert login_ok.status_code == 200
    assert decode_access_token(login_ok.json()["access_token"]) == stored_user.id

    login_bad = await client.post(
        "/api/v1/auth/login",
        json={"email": "alice@acme.internal", "password": "WrongPassword!"},
    )
    assert login_bad.status_code == 401


async def test_metrics_endpoint_and_tracker_percentiles(api_client) -> None:
    """GET /api/v1/metrics returns rolling p50/p95 latency, error rate, tokens, and cost."""
    tracker = MetricsTracker()
    for lat in (10.0, 20.0, 30.0, 40.0, 100.0):
        tracker.record_http(status_code=200, latency_ms=lat)
    tracker.record_http(status_code=503, latency_ms=250.0)
    tracker.record_ask(prompt_tokens=1000, completion_tokens=200)
    snap = tracker.snapshot()
    assert snap["total_requests"] == 6
    assert snap["total_errors"] == 1
    assert snap["p50_latency_ms"] == 30.0
    assert snap["p95_latency_ms"] == 250.0
    assert snap["total_prompt_tokens"] == 1000
    assert snap["estimated_cost_usd"] > 0

    client, _app = api_client
    resp = await client.get("/api/v1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "p50_latency_ms" in body
    assert "p95_latency_ms" in body
    assert "langfuse_enabled" in body


def test_edge_case_12_indirect_prompt_injection_fixture_fenced() -> None:
    """Indirect prompt injection file must remain strictly isolated inside <retrieved_context>."""
    poison_text = Path("data/edge_cases/prompt_injection_doc.txt").read_text(encoding="utf-8")
    chunk = RetrievedChunk(
        chunk_id="chk_inj",
        document_id="doc_inj",
        filename="prompt_injection_doc.txt",
        page_number=1,
        section_title="Body",
        content=poison_text,
        similarity_score=0.78,
        rerank_score=0.81,
    )
    fenced = wrap_retrieved_context([chunk])
    assert "<retrieved_context>" in fenced
    assert "</retrieved_context>" in fenced
    assert "Net-45 days" in fenced

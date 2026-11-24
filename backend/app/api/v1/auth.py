"""Authentication routes (`POST /api/v1/auth/signup`, `/login`, `GET /me`)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.v1.dependencies import get_current_user_id
from backend.app.auth import (
    AuthenticationError,
    create_access_token,
    hash_password,
    verify_password,
)
from backend.app.db.models import User
from backend.app.db.session import get_db
from backend.app.ingestion.exceptions import DuplicateDocumentError
from backend.app.observability import get_request_id
from backend.app.schemas import (
    AuthTokenResponse,
    ErrorResponse,
    LoginRequest,
    SignupRequest,
    UserProfileResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/signup",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse, "description": "Email already registered"}},
    summary="Register New User Account",
    description="Creates a new multi-tenant user with Argon2 password hashing and returns a JWT Bearer token.",
)
async def signup(
    payload: SignupRequest,
    session: AsyncSession = Depends(get_db),
) -> AuthTokenResponse:
    """Register a user and issue an immediate JWT access token."""
    normalized_email = payload.email.strip().lower()
    existing = (
        await session.execute(select(User.id).where(User.email == normalized_email))
    ).scalar_one_or_none()
    if existing is not None:
        raise DuplicateDocumentError(
            "An account with this email already exists.",
            error_code="EMAIL_ALREADY_REGISTERED",
            details={"email": normalized_email},
        )

    user = User(
        email=normalized_email,
        password_hash=hash_password(payload.password),
    )
    try:
        session.add(user)
        await session.commit()
        await session.refresh(user)
    except IntegrityError as exc:
        await session.rollback()
        raise DuplicateDocumentError(
            "An account with this email already exists.",
            error_code="EMAIL_ALREADY_REGISTERED",
            details={"email": normalized_email},
        ) from exc

    token, expires_in = create_access_token(user.id, user.email)
    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user_id=user.id,
        email=user.email,
        request_id=get_request_id(),
    )


@router.post(
    "/login",
    response_model=AuthTokenResponse,
    responses={401: {"model": ErrorResponse, "description": "Invalid email or password"}},
    summary="Authenticate & Issue JWT Token",
    description="Verifies Argon2 password hash and returns a signed HS256 JWT Bearer token.",
)
async def login(
    payload: LoginRequest,
    session: AsyncSession = Depends(get_db),
) -> AuthTokenResponse:
    """Authenticate an existing user and return a JWT access token."""
    normalized_email = payload.email.strip().lower()
    user = (
        await session.execute(select(User).where(User.email == normalized_email))
    ).scalar_one_or_none()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise AuthenticationError("Invalid email or password.")

    token, expires_in = create_access_token(user.id, user.email)
    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user_id=user.id,
        email=user.email,
        request_id=get_request_id(),
    )


@router.get(
    "/me",
    response_model=UserProfileResponse,
    responses={401: {"model": ErrorResponse, "description": "Missing or invalid JWT token"}},
    summary="Get Current Authenticated User",
)
async def get_me(
    user_id: uuid.UUID = Depends(get_current_user_id),
    session: AsyncSession = Depends(get_db),
) -> UserProfileResponse:
    """Return profile metadata for the authenticated JWT owner."""
    user = (await session.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise AuthenticationError("Authenticated user account no longer exists.")
    return UserProfileResponse(
        user_id=user.id,
        email=user.email,
        created_at=user.created_at,
        request_id=get_request_id(),
    )

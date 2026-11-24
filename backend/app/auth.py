"""JWT authentication and Argon2 password hashing security utilities (Phase 2 Bonus A).

Why this module exists:
    Isolates cryptographic password hashing (`pwdlib[argon2]`) and HS256 JWT
    token creation/verification (`PyJWT`) from HTTP routers. Passwords are never
    stored or logged in plaintext, and every protected endpoint authenticates
    a scoped `user_id` UUID from the Bearer token.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash

from backend.app.config import get_settings
from backend.app.ingestion.exceptions import IngestionError

_PASSWORD_HASHER = PasswordHash.recommended()


class AuthenticationError(IngestionError):
    """Raised on missing, invalid, expired JWT tokens or bad credentials (HTTP 401)."""

    status_code = 401
    error_code = "UNAUTHORIZED"


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using Argon2id."""
    return _PASSWORD_HASHER.hash(plain_password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verify a plaintext password against its stored Argon2 hash in constant time."""
    try:
        return _PASSWORD_HASHER.verify(plain_password, password_hash)
    except Exception:
        return False


def _get_jwt_secret() -> str:
    """Return configured JWT signing secret (requires non-empty value in production)."""
    settings = get_settings()
    secret = settings.jwt_secret_key.strip()
    if not secret:
        # Deterministic dev fallback only when explicitly running outside production
        if settings.app_env.lower() == "production":
            raise RuntimeError("JWT_SECRET_KEY must be set in production.")
        return "dev-only-insecure-jwt-secret-change-in-env-32bytes"
    return secret


def create_access_token(user_id: uuid.UUID, email: str) -> tuple[str, int]:
    """Create a signed HS256 JWT access token carrying `sub` (`user_id`) and `email`.

    Args:
        user_id: Authenticated user's UUID.
        email: Authenticated user's email address.

    Returns:
        Tuple of `(encoded_jwt_token, expires_in_seconds)`.
    """
    settings = get_settings()
    expires_seconds = settings.jwt_expire_minutes * 60
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_seconds)).timestamp()),
    }
    token = jwt.encode(payload, _get_jwt_secret(), algorithm=settings.jwt_algorithm)
    return token, expires_seconds


def decode_access_token(token: str) -> uuid.UUID:
    """Validate an HS256 JWT Bearer token and extract the owner `user_id` UUID.

    Args:
        token: Raw JWT string (without the `Bearer ` prefix).

    Returns:
        Parsed `uuid.UUID` of the authenticated user.

    Raises:
        AuthenticationError: If token is expired, malformed, or signature fails (HTTP 401).
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            _get_jwt_secret(),
            algorithms=[settings.jwt_algorithm],
        )
        sub = payload.get("sub")
        if not sub:
            raise AuthenticationError("Token payload is missing subject claim.")
        return uuid.UUID(str(sub))
    except AuthenticationError:
        raise
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Access token has expired. Please log in again.") from exc
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise AuthenticationError("Invalid or malformed access token.") from exc

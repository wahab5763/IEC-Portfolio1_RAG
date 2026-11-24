"""Centralized application configuration loaded strictly from environment variables.

Why this module exists:
    Enforces the zero-hardcoded-secrets engineering principle by validating all
    runtime settings, API keys, timeouts, and retrieval hyperparameters through
    Pydantic BaseSettings at startup. Also normalizes Neon Postgres connection
    URIs so standard libpq URLs (`postgresql://...?sslmode=require`) work
    seamlessly with SQLAlchemy's `asyncpg` driver.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration model for the Production RAG Service.

    All fields can be overridden via environment variables or a local `.env`
    file. Sensitive values (database URLs, API keys, JWT secrets) have empty
    defaults and are validated before live external calls so unit tests can
    import settings without requiring production credentials.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application Metadata ---
    app_name: str = Field(default="Production RAG Service", description="Service display name.")
    app_env: str = Field(default="development", description="Runtime environment identifier.")
    debug: bool = Field(default=False, description="Enable verbose debug behaviors.")
    log_level: str = Field(default="INFO", description="Structured JSON log level.")

    # --- Database (Neon Serverless Postgres + pgvector) ---
    database_url: str = Field(
        default="",
        description="PostgreSQL connection string (Neon Serverless or local Postgres).",
    )
    db_pool_size: int = Field(default=10, ge=1, le=50, description="SQLAlchemy async pool size.")
    db_max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections.")
    db_connect_timeout_seconds: float = Field(
        default=15.0,
        gt=0,
        description="Connection establishment timeout in seconds for asyncpg.",
    )

    # --- Cohere API (Embeddings & Cross-Encoder Reranking) ---
    cohere_api_key: str = Field(default="", description="Cohere v2 API key.")
    cohere_embed_model: str = Field(default="embed-v4.0", description="Cohere embedding model ID.")
    cohere_embed_dim: int = Field(
        default=1024,
        description="Matryoshka embedding dimension matching pgvector vector(1024).",
    )
    cohere_rerank_model: str = Field(
        default="rerank-v5.0-fast",
        description="Cohere cross-encoder rerank model ID.",
    )

    # --- Groq API (Grounded LLM Generation) ---
    groq_api_key: str = Field(default="", description="Groq API key.")
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq LLM model ID for grounded answer generation.",
    )
    groq_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deterministic sampling temperature for grounded RAG generation.",
    )
    groq_max_tokens: int = Field(
        default=1024,
        ge=64,
        le=8192,
        description="Maximum completion tokens per answer.",
    )

    # --- External API Resilience (Timeouts & Retries) ---
    api_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Per-request HTTP timeout in seconds for Cohere and Groq calls.",
    )
    api_max_retries: int = Field(
        default=3,
        ge=1,
        le=6,
        description="Maximum exponential-backoff retry attempts for transient external errors.",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=96,
        description="Chunks per Cohere embed API batch request.",
    )
    embedding_max_concurrency: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum concurrent embedding API requests via asyncio.Semaphore.",
    )

    # --- Chunking & Retrieval Hyperparameters ---
    chunk_size_tokens: int = Field(
        default=400,
        ge=100,
        le=2000,
        description="Target token budget per chunk (Config B default: 400).",
    )
    chunk_overlap_tokens: int = Field(
        default=60,
        ge=0,
        le=400,
        description="Token overlap between consecutive narrative chunks (15% default).",
    )
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Final number of chunks passed to the LLM after retrieval/reranking.",
    )
    retrieval_candidate_pool: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Candidate pool size per retriever branch before RRF + reranking.",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion smoothing constant k (standard value: 60).",
    )
    similarity_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum dense cosine similarity gate for Config A (vector-only) refusal.",
    )
    rerank_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum calibrated Cohere rerank relevance_score gate for Config B refusal.",
    )

    # --- Guardrails & Input Limits ---
    max_file_size_mb: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum allowed upload file size in megabytes (returns 413 if exceeded).",
    )
    max_query_length: int = Field(
        default=1000,
        ge=10,
        le=4000,
        description="Maximum user question character length (returns 422 if exceeded).",
    )
    min_pdf_chars_per_page: int = Field(
        default=20,
        ge=1,
        description="Minimum average extracted characters per PDF page before flagging scanned PDF.",
    )
    rate_limit_per_minute: int = Field(
        default=30,
        ge=1,
        description="Per-IP/user sliding window request limit for /api/v1/ask.",
    )

    # --- Single-Tenant Default User & Phase 2 JWT Auth ---
    default_user_id: uuid.UUID = Field(
        default=uuid.UUID("00000000-0000-0000-0000-000000000000"),
        description="System user UUID used in Phase 1 single-tenant mode.",
    )
    jwt_secret_key: str = Field(default="", description="HS256 signing secret for Phase 2 JWT.")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm.")
    jwt_expire_minutes: int = Field(default=60, ge=5, description="JWT access token lifetime.")

    # --- Phase 2 Observability (Langfuse) ---
    langfuse_public_key: str = Field(default="", description="Optional Langfuse public key.")
    langfuse_secret_key: str = Field(default="", description="Optional Langfuse secret key.")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse ingestion endpoint.",
    )

    @field_validator("cohere_embed_dim")
    @classmethod
    def validate_embed_dim(cls, value: int) -> int:
        """Ensure embedding dimension matches Cohere embed-v4.0 Matryoshka options.

        Args:
            value: Configured vector dimension.

        Returns:
            Validated integer dimension.

        Raises:
            ValueError: If dimension is not one of {256, 512, 1024, 1536}.
        """
        allowed = {256, 512, 1024, 1536}
        if value not in allowed:
            raise ValueError(f"cohere_embed_dim must be one of {sorted(allowed)}, got {value}.")
        return value

    @property
    def max_file_size_bytes(self) -> int:
        """Return the configured maximum upload size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def async_database_url(self) -> str:
        """Normalize DATABASE_URL into an asyncpg-compatible SQLAlchemy URL.

        Why:
            Neon dashboard connection strings use `postgresql://...` and libpq
            parameters (`sslmode=require`, `channel_binding=require`). The
            `asyncpg` driver rejects `sslmode` and `channel_binding` as keyword
            arguments, so we switch the scheme to `postgresql+asyncpg://` and
            strip those query parameters here (passing SSL via `db_connect_args`).

        Returns:
            Normalized SQLAlchemy async database URL string.

        Raises:
            ValueError: If `DATABASE_URL` is empty.
        """
        raw = self.database_url.strip()
        if not raw:
            raise ValueError("DATABASE_URL is not configured. Set it in your .env file.")

        # Convert standard Postgres schemes to SQLAlchemy's asyncpg dialect.
        if raw.startswith("postgres://"):
            raw = "postgresql+asyncpg://" + raw[len("postgres://") :]
        elif raw.startswith("postgresql://"):
            raw = "postgresql+asyncpg://" + raw[len("postgresql://") :]

        parts = urlsplit(raw)
        # Filter out libpq-only query keys that cause asyncpg TypeError on connect.
        filtered_query = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in {"sslmode", "channel_binding"}
        ]
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(filtered_query), parts.fragment)
        )

    @property
    def db_connect_args(self) -> dict[str, Any]:
        """Build driver-level connection arguments for `asyncpg`.

        Returns:
            Dictionary passed as `connect_args` to `create_async_engine`,
            including connection timeout and TLS (`ssl="require"`) whenever
            requested by the URL or targeting Neon cloud hosts.
        """
        connect_args: dict[str, Any] = {"timeout": self.db_connect_timeout_seconds}
        raw_lower = self.database_url.lower()

        # Enable TLS automatically if sslmode is present or connecting to Neon.
        if "sslmode=require" in raw_lower or "ssl=require" in raw_lower or "neon.tech" in raw_lower:
            connect_args["ssl"] = "require"

        return connect_args


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance.

    Returns:
        Validated `Settings` singleton loaded from environment / `.env`.
    """
    return Settings()

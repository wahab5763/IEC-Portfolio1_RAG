"""Asynchronous batched Cohere Embed v4.0 client with retries and concurrency caps.

Why this module exists:
    Encapsulates all embedding API calls behind a resilient async client that:
      1. Explicitly passes `output_dimension=1024` (preventing Cohere's default
         1536-dim output from colliding with PostgreSQL `vector(1024)`).
      2. Uses asymmetric `input_type="search_document"` vs `"search_query"`.
      3. Bounds concurrent batch requests via `asyncio.Semaphore` for large files.
      4. Retries transient network/rate-limit errors via `tenacity` with strict
         timeouts and maps persistent outages to `ExternalServiceError` (HTTP 503).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import cohere
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from backend.app.config import get_settings
from backend.app.ingestion.exceptions import ExternalServiceError

InputType = Literal["search_document", "search_query"]


class CohereEmbedder:
    """Async batched embedding generator backed by Cohere v2 API (`embed-v4.0`)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        output_dimension: int | None = None,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize the embedder from settings or explicit dependency injection."""
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.cohere_api_key
        self.model = model or settings.cohere_embed_model
        self.output_dimension = output_dimension or settings.cohere_embed_dim
        self.batch_size = batch_size or settings.embedding_batch_size
        self.max_concurrency = max_concurrency or settings.embedding_max_concurrency
        self.timeout_seconds = timeout_seconds or settings.api_timeout_seconds
        self.max_retries = max_retries or settings.api_max_retries
        self._injected_client = client

    def _get_client(self) -> Any:
        """Return injected client (for testing) or a configured `cohere.AsyncClientV2`."""
        if self._injected_client is not None:
            return self._injected_client
        if not self.api_key:
            raise ExternalServiceError(
                "COHERE_API_KEY is not configured.",
                details={"service": "cohere_embed"},
            )
        return cohere.AsyncClientV2(api_key=self.api_key, timeout=self.timeout_seconds)

    async def _embed_batch(
        self,
        client: Any,
        texts: list[str],
        input_type: InputType,
        semaphore: asyncio.Semaphore,
    ) -> list[list[float]]:
        """Embed one bounded batch with concurrency limiting, timeout, and exponential backoff."""
        async with semaphore:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
                    reraise=True,
                ):
                    with attempt:
                        response = await asyncio.wait_for(
                            client.embed(
                                model=self.model,
                                input_type=input_type,
                                texts=texts,
                                output_dimension=self.output_dimension,
                                embedding_types=["float"],
                            ),
                            timeout=self.timeout_seconds,
                        )
                        vectors = list(response.embeddings.float)
                        if len(vectors) != len(texts):
                            raise RuntimeError(
                                f"Cohere returned {len(vectors)} embeddings for {len(texts)} texts."
                            )
                        if vectors and len(vectors[0]) != self.output_dimension:
                            raise RuntimeError(
                                f"Expected {self.output_dimension}-dim vector, got {len(vectors[0])}."
                            )
                        return [list(map(float, vec)) for vec in vectors]
            except Exception as exc:
                raise ExternalServiceError(
                    f"Cohere embedding service failed after {self.max_retries} attempts: {exc}",
                    details={"service": "cohere_embed", "model": self.model},
                ) from exc

        raise ExternalServiceError("Cohere embedding call exited without returning vectors.")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate 1024-dim document embeddings in bounded concurrent batches.

        Args:
            texts: Ordered list of chunk strings to embed.

        Returns:
            List of float vectors aligned 1:1 with `texts`.

        Raises:
            ExternalServiceError: If Cohere API is unreachable, rate-limited, or
                times out across all retry attempts (HTTP 503).
        """
        if not texts:
            return []

        client = self._get_client()
        semaphore = asyncio.Semaphore(self.max_concurrency)
        batches = [
            texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)
        ]

        # Run batches concurrently while preserving original order via asyncio.gather.
        batch_results = await asyncio.gather(
            *(self._embed_batch(client, batch, "search_document", semaphore) for batch in batches)
        )
        return [vec for batch_vecs in batch_results for vec in batch_vecs]

    async def embed_query(self, query: str) -> list[float]:
        """Generate a single asymmetric `search_query` embedding vector.

        Args:
            query: User search question string.

        Returns:
            Single 1024-dim float vector.

        Raises:
            ExternalServiceError: If Cohere API fails after retries (HTTP 503).
        """
        client = self._get_client()
        semaphore = asyncio.Semaphore(1)
        vectors = await self._embed_batch(client, [query], "search_query", semaphore)
        return vectors[0]

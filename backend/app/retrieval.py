"""Hybrid Dense-Sparse retrieval engine with Reciprocal Rank Fusion (RRF) and Cohere Rerank.

Why this module exists:
    Dense semantic search (`pgvector` HNSW cosine similarity) captures conceptual
    meaning across 100+ languages, while sparse full-text search (PostgreSQL
    `tsvector` GIN with `'simple'` config) captures exact product codes, policy
    IDs, and alphanumerics. Merging both via RRF (`k=60`) and reranking the top
    20 candidates with Cohere `rerank-v5.0-fast` maximizes MRR and Recall@5
    while enforcing strict per-user tenant isolation (`WHERE user_id = :user_id`).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import cohere
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from backend.app.config import get_settings
from backend.app.db.models import Document
from backend.app.ingestion.embedder import CohereEmbedder
from backend.app.ingestion.exceptions import ExternalServiceError

RetrievalMode = Literal["dense_only", "hybrid_rerank"]


@dataclass
class RetrievedChunk:
    """Ranked chunk returned by retrieval with full scoring transparency."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section_title: str | None
    content: str
    similarity_score: float
    rrf_score: float | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalOutcome:
    """Complete retrieval result including corpus-empty and threshold-gate flags."""

    chunks: list[RetrievedChunk]
    corpus_empty: bool
    below_threshold: bool
    mode: RetrievalMode
    latency_ms: float


def reciprocal_rank_fusion(
    dense_ids: list[str],
    sparse_ids: list[str],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse two ranked chunk-ID lists via Reciprocal Rank Fusion.

    Formula:
        RRF_Score(d) = sum_{m in {dense, sparse}} 1 / (k + rank_m(d))

    Args:
        dense_ids: Ordered chunk IDs from dense HNSW cosine search (1-based rank).
        sparse_ids: Ordered chunk IDs from sparse `tsvector` search (1-based rank).
        k: Smoothing constant mitigating outlier dominance (standard: 60).

    Returns:
        List of `(chunk_id, rrf_score)` sorted descending by fused score.
    """
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
    for rank, cid in enumerate(sparse_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class CohereReranker:
    """Async cross-encoder reranker backed by Cohere v2 API (`rerank-v5.0-fast`)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.cohere_api_key
        self.model = model or settings.cohere_rerank_model
        self.timeout_seconds = timeout_seconds or settings.api_timeout_seconds
        self.max_retries = max_retries or settings.api_max_retries
        self._injected_client = client

    def _get_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        if not self.api_key:
            raise ExternalServiceError(
                "COHERE_API_KEY is not configured.",
                details={"service": "cohere_rerank"},
            )
        return cohere.AsyncClientV2(api_key=self.api_key, timeout=self.timeout_seconds)

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int = 5,
    ) -> list[RetrievedChunk]:
        """Score `(query, chunk)` pairs jointly and return top-N reranked chunks.

        Args:
            query: User search question.
            chunks: Candidate pool (typically Top-20 from RRF).
            top_n: Maximum number of chunks to return.

        Returns:
            Top-N `RetrievedChunk` objects ordered by descending `rerank_score`.

        Raises:
            ExternalServiceError: If Cohere Rerank fails after retries (HTTP 503).
        """
        if not chunks:
            return []

        client = self._get_client()
        documents = [c.content for c in chunks]
        bounded_top_n = min(top_n, len(chunks))

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
                reraise=True,
            ):
                with attempt:
                    resp = await asyncio.wait_for(
                        client.rerank(
                            model=self.model,
                            query=query,
                            documents=documents,
                            top_n=bounded_top_n,
                        ),
                        timeout=self.timeout_seconds,
                    )
                    reranked: list[RetrievedChunk] = []
                    for item in resp.results:
                        chunk = chunks[item.index]
                        chunk.rerank_score = round(float(item.relevance_score), 4)
                        reranked.append(chunk)
                    return reranked
        except Exception as exc:
            raise ExternalServiceError(
                f"Cohere rerank service failed after {self.max_retries} attempts: {exc}",
                details={"service": "cohere_rerank", "model": self.model},
            ) from exc

        return []


class HybridRetriever:
    """Executes Config A (dense-only) or Config B (Hybrid HNSW + FTS + RRF + Rerank)."""

    def __init__(
        self,
        *,
        embedder: CohereEmbedder | None = None,
        reranker: CohereReranker | None = None,
    ) -> None:
        self.embedder = embedder or CohereEmbedder()
        self.reranker = reranker or CohereReranker()

    async def _has_user_documents(self, session: AsyncSession, user_id: uuid.UUID) -> bool:
        """Check whether the user has at least one ready document (Edge Case #4)."""
        stmt = select(func.count(Document.id)).where(
            Document.user_id == user_id,
            Document.status == "ready",
        )
        count = (await session.execute(stmt)).scalar_one()
        return int(count) > 0

    async def _dense_search(
        self,
        session: AsyncSession,
        *,
        user_id: uuid.UUID,
        query_vector: list[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Run pgvector HNSW cosine similarity search scoped strictly to `user_id`."""
        vec_literal = "[" + ",".join(f"{v:.8f}" for v in query_vector) + "]"
        sql = text(
            """
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text AS document_id,
                d.filename AS filename,
                c.page_number AS page_number,
                c.section_title AS section_title,
                c.content AS content,
                c.metadata AS metadata,
                1.0 - (c.embedding <=> CAST(:qvec AS vector)) AS similarity_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.user_id = :user_id
              AND d.status = 'ready'
              AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> CAST(:qvec AS vector) ASC
            LIMIT :lim
            """
        )
        result = await session.execute(
            sql,
            {"qvec": vec_literal, "user_id": user_id, "lim": limit},
        )
        return [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                filename=row.filename,
                page_number=int(row.page_number or 1),
                section_title=row.section_title,
                content=row.content,
                similarity_score=round(float(row.similarity_score), 4),
                metadata=dict(row.metadata or {}),
            )
            for row in result.fetchall()
        ]

    async def _sparse_search(
        self,
        session: AsyncSession,
        *,
        user_id: uuid.UUID,
        query: str,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Run PostgreSQL `tsvector` GIN keyword search scoped strictly to `user_id`."""
        sql = text(
            """
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text AS document_id,
                d.filename AS filename,
                c.page_number AS page_number,
                c.section_title AS section_title,
                c.content AS content,
                c.metadata AS metadata,
                ts_rank_cd(c.tsv, plainto_tsquery('simple', :query)) AS sparse_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.user_id = :user_id
              AND d.status = 'ready'
              AND c.tsv @@ plainto_tsquery('simple', :query)
            ORDER BY sparse_score DESC
            LIMIT :lim
            """
        )
        result = await session.execute(sql, {"query": query, "user_id": user_id, "lim": limit})
        return [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                filename=row.filename,
                page_number=int(row.page_number or 1),
                section_title=row.section_title,
                content=row.content,
                similarity_score=0.0,
                metadata=dict(row.metadata or {}),
            )
            for row in result.fetchall()
        ]

    async def retrieve(
        self,
        *,
        session: AsyncSession,
        query: str,
        user_id: uuid.UUID,
        top_k: int | None = None,
        mode: RetrievalMode = "hybrid_rerank",
        similarity_threshold: float | None = None,
        rerank_threshold: float | None = None,
    ) -> RetrievalOutcome:
        """Retrieve ranked, threshold-gated chunks for a user query.

        Args:
            session: Active async SQLAlchemy database session.
            query: Validated user question string.
            user_id: Authenticated user UUID for multi-tenant isolation.
            top_k: Number of final chunks to return (defaults to `Settings.retrieval_top_k`).
            mode: `"dense_only"` (Config A baseline) or `"hybrid_rerank"` (Config B).
            similarity_threshold: Cosine gate for Config A (defaults to `0.35`).
            rerank_threshold: Calibrated Cohere rerank gate for Config B (defaults to `0.25`).

        Returns:
            `RetrievalOutcome` containing ranked chunks and refusal-gate flags.
        """
        started = time.perf_counter()
        settings = get_settings()
        final_k = top_k or settings.retrieval_top_k
        pool_size = settings.retrieval_candidate_pool
        sim_gate = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )
        rr_gate = (
            rerank_threshold if rerank_threshold is not None else settings.rerank_threshold
        )

        # Edge Case #4: Short-circuit cleanly when user has uploaded 0 documents.
        if not await self._has_user_documents(session, user_id):
            return RetrievalOutcome(
                chunks=[],
                corpus_empty=True,
                below_threshold=False,
                mode=mode,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        query_vec = await self.embedder.embed_query(query)

        # Config A: Vector-only dense search gated on cosine similarity.
        if mode == "dense_only":
            dense_hits = await self._dense_search(
                session, user_id=user_id, query_vector=query_vec, limit=final_k
            )
            max_sim = max((c.similarity_score for c in dense_hits), default=0.0)
            below_threshold = (not dense_hits) or (max_sim < sim_gate)
            return RetrievalOutcome(
                chunks=[] if below_threshold else dense_hits,
                corpus_empty=False,
                below_threshold=below_threshold,
                mode=mode,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )

        # Config B: Dense HNSW + Sparse tsvector -> RRF (k=60) -> Cohere Rerank.
        dense_pool = await self._dense_search(
            session, user_id=user_id, query_vector=query_vec, limit=pool_size
        )
        sparse_pool = await self._sparse_search(
            session, user_id=user_id, query=query, limit=pool_size
        )

        chunk_map: dict[str, RetrievedChunk] = {}
        for item in sparse_pool:
            chunk_map[item.chunk_id] = item
        for item in dense_pool:
            # Prefer dense record so accurate dense cosine `similarity_score` is preserved.
            chunk_map[item.chunk_id] = item

        fused = reciprocal_rank_fusion(
            [c.chunk_id for c in dense_pool],
            [c.chunk_id for c in sparse_pool],
            k=settings.rrf_k,
        )

        candidates: list[RetrievedChunk] = []
        for cid, rrf_val in fused[:pool_size]:
            chunk = chunk_map[cid]
            chunk.rrf_score = round(rrf_val, 5)
            candidates.append(chunk)

        reranked = await self.reranker.rerank(query, candidates, top_n=final_k)
        max_rerank = max(((c.rerank_score or 0.0) for c in reranked), default=0.0)
        below_threshold = (not reranked) or (max_rerank < rr_gate)

        return RetrievalOutcome(
            chunks=[] if below_threshold else reranked,
            corpus_empty=False,
            below_threshold=below_threshold,
            mode=mode,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )

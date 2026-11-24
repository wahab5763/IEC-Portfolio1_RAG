"""Unit & async tests for Milestone 5 (Hybrid Retrieval, RRF, Rerank & Grounded Generation)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.generation import (
    EMPTY_CORPUS_MESSAGE,
    STANDARD_REFUSAL_MESSAGE,
    GroundedGenerator,
)
from backend.app.ingestion.embedder import CohereEmbedder
from backend.app.ingestion.exceptions import ExternalServiceError
from backend.app.retrieval import (
    CohereReranker,
    HybridRetriever,
    RetrievalOutcome,
    RetrievedChunk,
    reciprocal_rank_fusion,
)


def test_reciprocal_rank_fusion_rewards_dual_branch_agreement() -> None:
    """Chunks appearing high in both dense and sparse lists must outrank single-branch hits."""
    fused = reciprocal_rank_fusion(
        dense_ids=["c1", "c2", "c3"],
        sparse_ids=["c2", "c4", "c1"],
        k=60,
    )
    top_ids = [cid for cid, _ in fused]
    # c1 (rank 1 + rank 3) and c2 (rank 2 + rank 1) must be the top two
    assert set(top_ids[:2]) == {"c1", "c2"}
    assert top_ids[0] == "c2"  # 1/62 + 1/61 > 1/61 + 1/63


async def test_reranker_orders_by_relevance_and_handles_outage_with_503() -> None:
    """CohereReranker must attach rerank_score and raise 503 after retry exhaustion."""
    chunks = [
        RetrievedChunk("c1", "d1", "a.pdf", 1, "S1", "first text", 0.7),
        RetrievedChunk("c2", "d1", "a.pdf", 2, "S2", "best matching text", 0.65),
    ]

    async def fake_rerank(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=1, relevance_score=0.92),
                SimpleNamespace(index=0, relevance_score=0.41),
            ]
        )

    reranker = CohereReranker(client=SimpleNamespace(rerank=fake_rerank))
    ranked = await reranker.rerank("best matching", chunks, top_n=2)
    assert [c.chunk_id for c in ranked] == ["c2", "c1"]
    assert ranked[0].rerank_score == 0.92

    # Outage path -> HTTP 503
    async def failing_rerank(**_kwargs: object) -> None:
        raise TimeoutError("rerank timeout")

    failing = CohereReranker(client=SimpleNamespace(rerank=failing_rerank), max_retries=2)
    with pytest.raises(ExternalServiceError) as exc:
        await failing.rerank("q", chunks, top_n=2)
    assert exc.value.status_code == 503


async def test_retriever_empty_corpus_short_circuits_cleanly() -> None:
    """When user has 0 documents, retriever returns corpus_empty=True without calling Cohere."""
    session = AsyncMock()
    count_res = MagicMock()
    count_res.scalar_one.return_value = 0
    session.execute.return_value = count_res

    retriever = HybridRetriever(
        embedder=CohereEmbedder(client=SimpleNamespace()),
        reranker=CohereReranker(client=SimpleNamespace()),
    )
    outcome = await retriever.retrieve(
        session=session,
        query="What is the leave policy?",
        user_id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
    )
    assert outcome.corpus_empty is True
    assert outcome.chunks == []


async def test_generator_handles_empty_corpus_refusal_and_cited_answer() -> None:
    """Generator must short-circuit empty/below-threshold cases and verify valid citations."""
    user_chunk = RetrievedChunk(
        chunk_id="chk_1",
        document_id="doc_1",
        filename="01_company_hr_policy.pdf",
        page_number=1,
        section_title="Paid Annual Leave",
        content="Employees receive 22 days of paid annual leave.",
        similarity_score=0.88,
        rerank_score=0.95,
    )

    async def fake_create(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "Full-time employees get 22 days of paid annual leave "
                            "[Chunk ID: chk_1 | 01_company_hr_policy.pdf, p. 1]."
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=24),
        )

    fake_groq = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    gen = GroundedGenerator(client=fake_groq, model="llama-3.3-70b-versatile")

    # 1. Empty corpus short-circuit
    empty_res = await gen.generate(
        question="Anything?",
        retrieval=RetrievalOutcome([], corpus_empty=True, below_threshold=False, mode="hybrid_rerank", latency_ms=1.0),
    )
    assert empty_res.refused is True
    assert empty_res.answer == EMPTY_CORPUS_MESSAGE

    # 2. Below threshold short-circuit
    low_res = await gen.generate(
        question="Capital of Mars?",
        retrieval=RetrievalOutcome([], corpus_empty=False, below_threshold=True, mode="hybrid_rerank", latency_ms=2.0),
    )
    assert low_res.refused is True
    assert low_res.answer == STANDARD_REFUSAL_MESSAGE

    # 3. Grounded cited answer
    ok_res = await gen.generate(
        question="How many annual leave days?",
        retrieval=RetrievalOutcome([user_chunk], corpus_empty=False, below_threshold=False, mode="hybrid_rerank", latency_ms=5.0),
    )
    assert ok_res.refused is False
    assert len(ok_res.citations) == 1
    assert ok_res.citations[0].chunk_id == "chk_1"
    assert ok_res.prompt_tokens == 120

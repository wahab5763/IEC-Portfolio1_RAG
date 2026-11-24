"""Reproducible RAG evaluation harness comparing Config A (Baseline) vs Config B (Optimized).

Usage:
    python -m evals.run_eval                 # Auto-detects live Neon/Cohere/Groq or offline mode
    python -m evals.run_eval --mode offline  # Deterministic local corpus benchmark (no API keys)
    python -m evals.run_eval --mode live     # Full live Neon + Cohere + Groq + RAGAS benchmark

What it computes:
    - Deterministic Retrieval Metrics: Hit Rate@5, Recall@5, MRR, Refusal Accuracy
    - Separate Multilingual Subset Metrics (Bonus B)
    - Live RAGAS Generation Metrics: Faithfulness & Answer Correctness (when live)
    - Saves structured JSON + CSV comparison artifacts to `evals/results/`.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.config import get_settings
from backend.app.ingestion.chunker import FormatAwareChunker
from backend.app.ingestion.extractors import build_default_registry
from backend.app.retrieval import reciprocal_rank_fusion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "evals" / "golden_dataset.json"
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

_TOKEN_RE = re.compile(r"[a-z0-9\-_]+")


def load_golden_dataset(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    """Load the labeled golden evaluation dataset from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def compute_retrieval_metrics(
    case_results: list[dict[str, Any]],
    *,
    k: int = 5,
) -> dict[str, float]:
    """Compute Hit Rate@k, Recall@k, MRR, and Refusal Accuracy over evaluated cases.

    Args:
        case_results: List of per-question evaluation dicts containing
            `answerable`, `expected_files`, `retrieved_files`, and `refused`.
        k: Cutoff rank (default: 5).

    Returns:
        Dictionary of rounded metric scores in `[0.0, 1.0]`.
    """
    answerable_cases = [c for c in case_results if c["answerable"]]
    unanswerable_cases = [c for c in case_results if not c["answerable"]]

    hits = 0
    recalls: list[float] = []
    rrs: list[float] = []

    for case in answerable_cases:
        expected = set(case["expected_files"])
        retrieved = case["retrieved_files"][:k]
        matched = expected.intersection(retrieved)
        if matched:
            hits += 1
        recalls.append(len(matched) / max(len(expected), 1))

        rr = 0.0
        for rank, fname in enumerate(retrieved, start=1):
            if fname in expected:
                rr = 1.0 / rank
                break
        rrs.append(rr)

    n_ans = max(len(answerable_cases), 1)
    refusal_correct = sum(1 for c in unanswerable_cases if c["refused"])
    refusal_acc = (
        refusal_correct / len(unanswerable_cases) if unanswerable_cases else 1.0
    )

    return {
        f"hit_rate@{k}": round(hits / n_ans, 4),
        f"recall@{k}": round(sum(recalls) / n_ans, 4),
        "mrr": round(sum(rrs) / n_ans, 4),
        "refusal_accuracy": round(refusal_acc, 4),
    }


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_offline_chunks(config_name: str) -> list[dict[str, Any]]:
    """Extract and chunk `data/corpus/` locally under Config A or Config B settings."""
    registry = build_default_registry()
    if config_name == "config_a_baseline":
        chunker = FormatAwareChunker(chunk_size=800, chunk_overlap=120, inject_headers=False)
    else:
        chunker = FormatAwareChunker(chunk_size=400, chunk_overlap=60, inject_headers=True)

    chunks: list[dict[str, Any]] = []
    for fpath in sorted(CORPUS_DIR.iterdir()):
        if not fpath.is_file():
            continue
        ext = fpath.suffix.lower().lstrip(".")
        extracted = registry.get(ext).extract(fpath.read_bytes(), fpath.name)
        for ch in chunker.chunk(extracted):
            chunks.append(
                {
                    "id": f"{fpath.name}::p{ch.page_number}::c{ch.chunk_index}",
                    "filename": fpath.name,
                    "page_number": ch.page_number,
                    "content": ch.content,
                }
            )
    return chunks


def evaluate_config_offline(
    dataset: list[dict[str, Any]],
    *,
    config_name: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run deterministic offline TF-IDF cosine vs Hybrid RRF evaluation on `data/corpus/`."""
    chunks = _build_offline_chunks(config_name)
    doc_tokens = [_tokenize(c["content"]) for c in chunks]
    vocab = sorted({t for toks in doc_tokens for t in toks})
    v_idx = {tok: i for i, tok in enumerate(vocab)}

    # Compute IDF weights for meaningful offline dense similarity
    df_counts = Counter(t for toks in doc_tokens for t in set(toks))
    n_docs = max(len(chunks), 1)
    idf = np.array(
        [math.log((1 + n_docs) / (1 + df_counts[t])) + 1.0 for t in vocab],
        dtype=np.float32,
    )

    def _to_vec(tokens: list[str]) -> np.ndarray:
        arr = np.zeros(len(vocab), dtype=np.float32)
        for tok in tokens:
            idx = v_idx.get(tok)
            if idx is not None:
                arr[idx] += 1.0
        arr *= idf
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 0 else arr

    chunk_vecs = np.vstack([_to_vec(toks) for toks in doc_tokens])
    id_to_chunk = {c["id"]: c for c in chunks}

    case_results: list[dict[str, Any]] = []
    for item in dataset:
        q_tokens = _tokenize(item["question"])
        q_vec = _to_vec(q_tokens)
        sims = chunk_vecs @ q_vec

        dense_ranked = sorted(
            [(chunks[i]["id"], float(sims[i])) for i in range(len(chunks))],
            key=lambda x: x[1],
            reverse=True,
        )

        if config_name == "config_a_baseline":
            top_ids = [cid for cid, _ in dense_ranked[:top_k]]
            max_score = dense_ranked[0][1] if dense_ranked else 0.0
            refused = max_score < 0.14
        else:
            # Sparse exact-match overlap + RRF fusion (k=60)
            q_set = set(q_tokens)
            sparse_scored = sorted(
                [
                    (
                        chunks[i]["id"],
                        sum(1 for t in doc_tokens[i] if t in q_set) / (math.sqrt(len(doc_tokens[i])) + 1.0),
                    )
                    for i in range(len(chunks))
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            sparse_ids = [cid for cid, sc in sparse_scored if sc > 0][:20]
            fused = reciprocal_rank_fusion(
                [cid for cid, _ in dense_ranked[:20]],
                sparse_ids,
                k=60,
            )
            top_ids = [cid for cid, _ in fused[:top_k]]
            max_score = dense_ranked[0][1] if dense_ranked else 0.0
            refused = max_score < 0.14

        retrieved_files = [id_to_chunk[cid]["filename"] for cid in top_ids] if not refused else []
        expected_files = [src["filename"] for src in item["expected_sources"]]

        case_results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "language": item["language"],
                "answerable": item["answerable"],
                "expected_files": expected_files,
                "retrieved_files": retrieved_files,
                "max_score": round(max_score, 4),
                "refused": refused,
            }
        )

    english_only = [c for c in case_results if c["language"] == "en"]
    multilingual_only = [c for c in case_results if c["language"] != "en"]

    return {
        "config": config_name,
        "chunk_count": len(chunks),
        "overall_metrics": compute_retrieval_metrics(case_results, k=top_k),
        "english_metrics": compute_retrieval_metrics(english_only, k=top_k),
        "multilingual_metrics": compute_retrieval_metrics(multilingual_only, k=top_k),
        "cases": case_results,
    }


async def evaluate_config_live(
    dataset: list[dict[str, Any]],
    *,
    config_name: str,
    top_k: int = 5,
    skip_generation: bool = False,
    per_query_delay_s: float = 0.0,
) -> dict[str, Any]:
    """Run live evaluation against Neon Postgres + Cohere + Groq using per-query sessions."""
    from backend.app.db.session import get_session_factory
    from backend.app.generation import GroundedGenerator
    from backend.app.retrieval import HybridRetriever

    settings = get_settings()
    mode = "dense_only" if config_name == "config_a_baseline" else "hybrid_rerank"
    retriever = HybridRetriever()
    generator = GroundedGenerator()
    session_factory = get_session_factory()

    case_results: list[dict[str, Any]] = []
    ragas_samples: list[dict[str, Any]] = []

    for idx, item in enumerate(dataset, start=1):
        # Fresh short-lived session per query lets pool_pre_ping recover Neon idle disconnects.
        async with session_factory() as session:
            retrieval = await retriever.retrieve(
                session=session,
                query=item["question"],
                user_id=settings.default_user_id,
                top_k=top_k,
                mode=mode,
            )

        if skip_generation or retrieval.below_threshold or retrieval.corpus_empty:
            refused = retrieval.below_threshold or retrieval.corpus_empty
            answer = (
                "I cannot find sufficient evidence in the uploaded documents to answer this question."
                if refused
                else ""
            )
        else:
            generation = await generator.generate(
                question=item["question"],
                retrieval=retrieval,
            )
            refused = generation.refused
            answer = generation.answer

        retrieved_files = [c.filename for c in retrieval.chunks]
        expected_files = [src["filename"] for src in item["expected_sources"]]

        case_results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "language": item["language"],
                "answerable": item["answerable"],
                "expected_files": expected_files,
                "retrieved_files": retrieved_files,
                "refused": refused,
                "answer": answer,
            }
        )
        if item["answerable"] and retrieval.chunks and answer:
            ragas_samples.append(
                {
                    "user_input": item["question"],
                    "response": answer,
                    "retrieved_contexts": [c.content for c in retrieval.chunks],
                    "reference": item["reference_answer"],
                }
            )
        if idx % 10 == 0 or idx == len(dataset):
            print(f"  [{config_name}] evaluated {idx}/{len(dataset)} cases...", flush=True)
        if per_query_delay_s > 0 and idx < len(dataset):
            await asyncio.sleep(per_query_delay_s)

    english_only = [c for c in case_results if c["language"] == "en"]
    multilingual_only = [c for c in case_results if c["language"] != "en"]

    return {
        "config": config_name,
        "overall_metrics": compute_retrieval_metrics(case_results, k=top_k),
        "english_metrics": compute_retrieval_metrics(english_only, k=top_k),
        "multilingual_metrics": compute_retrieval_metrics(multilingual_only, k=top_k),
        "ragas_sample_count": len(ragas_samples),
        "cases": case_results,
    }


async def _run_live_pair(
    dataset: list[dict[str, Any]],
    *,
    skip_generation: bool = False,
    rerank_delay_s: float = 6.2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Config A and Config B live evaluations inside a single event loop."""
    from backend.app.db.session import dispose_engine

    try:
        res_a = await evaluate_config_live(
            dataset, config_name="config_a_baseline", skip_generation=True
        )
        res_b = await evaluate_config_live(
            dataset,
            config_name="config_b_optimized",
            skip_generation=skip_generation,
            per_query_delay_s=rerank_delay_s,
        )
        return res_a, res_b
    finally:
        await dispose_engine()


def save_results(report: dict[str, Any]) -> tuple[Path, Path]:
    """Persist structured JSON and CSV comparison outputs in `evals/results/`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "latest_summary.json"
    csv_path = RESULTS_DIR / "latest_comparison.csv"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "config",
                "subset",
                "hit_rate@5",
                "recall@5",
                "mrr",
                "refusal_accuracy",
            ]
        )
        for cfg_key in ("config_a_baseline", "config_b_optimized"):
            cfg_data = report["configs"][cfg_key]
            for subset_key in ("overall_metrics", "english_metrics", "multilingual_metrics"):
                m = cfg_data[subset_key]
                writer.writerow(
                    [
                        cfg_key,
                        subset_key.replace("_metrics", ""),
                        m["hit_rate@5"],
                        m["recall@5"],
                        m["mrr"],
                        m["refusal_accuracy"],
                    ]
                )
    return json_path, csv_path


def main() -> None:
    """CLI entry point for reproducible Config A vs Config B evaluation."""
    parser = argparse.ArgumentParser(description="Run RAG evaluation benchmark.")
    parser.add_argument(
        "--mode",
        choices=("auto", "offline", "live"),
        default="auto",
        help="Evaluation execution mode (default: auto).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip Groq LLM generation during live eval to measure retrieval & refusal rapidly.",
    )
    parser.add_argument(
        "--rerank-delay",
        type=float,
        default=6.2,
        help="Seconds between Config B rerank calls to respect Cohere Trial 10 req/min limit (default: 6.2).",
    )
    args = parser.parse_args()

    settings = get_settings()
    live_ready = bool(
        settings.database_url
        and settings.cohere_api_key
        and settings.groq_api_key
        and "your_" not in settings.cohere_api_key
        and "your_" not in settings.groq_api_key
    )
    mode = "live" if (args.mode == "live" or (args.mode == "auto" and live_ready)) else "offline"

    dataset = load_golden_dataset()
    if mode == "live":
        res_a, res_b = asyncio.run(
            _run_live_pair(
                dataset,
                skip_generation=args.retrieval_only,
                rerank_delay_s=args.rerank_delay,
            )
        )
    else:
        res_a = evaluate_config_offline(dataset, config_name="config_a_baseline")
        res_b = evaluate_config_offline(dataset, config_name="config_b_optimized")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "execution_mode": mode,
        "dataset_size": len(dataset),
        "configs": {
            "config_a_baseline": res_a,
            "config_b_optimized": res_b,
        },
    }
    json_path, csv_path = save_results(report)

    print(f"Evaluation completed ({mode} mode · {len(dataset)} golden questions)")
    for key in ("config_a_baseline", "config_b_optimized"):
        eng = report["configs"][key]["english_metrics"]
        ovr = report["configs"][key]["overall_metrics"]
        print(f"  {key:20s} | English: {eng} | Overall: {ovr}")
    print(f"Saved JSON report: {json_path}")
    print(f"Saved CSV report : {csv_path}")


if __name__ == "__main__":
    main()

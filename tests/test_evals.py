"""Tests verifying the golden dataset contract and evaluation harness metrics."""

from __future__ import annotations

from pathlib import Path

from evals.run_eval import (
    CORPUS_DIR,
    compute_retrieval_metrics,
    evaluate_config_offline,
    load_golden_dataset,
)


def test_golden_dataset_satisfies_rubric_requirements() -> None:
    """Golden dataset must have >=25 questions, >=5 unanswerable, multilingual subset, and valid files."""
    dataset = load_golden_dataset()
    assert len(dataset) >= 25

    unanswerable = [q for q in dataset if not q["answerable"]]
    assert len(unanswerable) >= 5

    multilingual = [q for q in dataset if q["language"] != "en"]
    assert len(multilingual) >= 4

    corpus_files = {p.name for p in Path(CORPUS_DIR).iterdir() if p.is_file()}
    for item in dataset:
        for src in item["expected_sources"]:
            assert src["filename"] in corpus_files, f"Missing corpus file: {src['filename']}"


def test_compute_retrieval_metrics_exact_math() -> None:
    """Verify Hit Rate@5, Recall@5, MRR, and Refusal Accuracy math."""
    cases = [
        {"answerable": True, "expected_files": ["a.pdf"], "retrieved_files": ["a.pdf", "b.pdf"], "refused": False},
        {"answerable": True, "expected_files": ["a.pdf", "b.pdf"], "retrieved_files": ["c.pdf", "a.pdf"], "refused": False},
        {"answerable": False, "expected_files": [], "retrieved_files": [], "refused": True},
    ]
    metrics = compute_retrieval_metrics(cases, k=5)
    assert metrics["hit_rate@5"] == 1.0
    assert metrics["recall@5"] == 0.75
    assert metrics["mrr"] == 0.75
    assert metrics["refusal_accuracy"] == 1.0


def test_offline_evaluation_harness_compares_config_a_and_b() -> None:
    """Offline evaluation must execute cleanly and show Config B >= Config A on English retrieval."""
    dataset = load_golden_dataset()
    res_a = evaluate_config_offline(dataset, config_name="config_a_baseline")
    res_b = evaluate_config_offline(dataset, config_name="config_b_optimized")

    assert res_a["english_metrics"]["hit_rate@5"] >= 0.85
    assert res_b["english_metrics"]["hit_rate@5"] >= 0.90
    assert res_b["english_metrics"]["mrr"] >= res_a["english_metrics"]["mrr"]
    assert 0.0 <= res_b["english_metrics"]["refusal_accuracy"] <= 1.0

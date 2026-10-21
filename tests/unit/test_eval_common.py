"""Tests for golden data validation and source-based retrieval metrics."""

import json
from types import SimpleNamespace

import pytest

from eval.common import load_golden_dataset, retrieval_metrics


def test_load_golden_dataset_validates_jsonl(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "What is RAG?",
                "ground_truth_answer": "Retrieval-augmented generation.",
                "ground_truth_sources": [{"source": "paper.pdf", "pages": [2]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_golden_dataset(path)

    assert len(cases) == 1
    assert cases[0]["question"] == "What is RAG?"


def test_load_golden_dataset_rejects_missing_sources(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps({"question": "Q?", "ground_truth_answer": "A", "ground_truth_sources": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one"):
        load_golden_dataset(path)


def test_retrieval_metrics_calculates_recall_and_mrr_from_source_pages():
    retrieved = [
        SimpleNamespace(source="other.pdf", page=2),
        SimpleNamespace(source="paper.pdf", page=4),
        SimpleNamespace(source="paper.pdf", page=5),
    ]

    scores = retrieval_metrics(
        retrieved,
        [{"source": "paper.pdf", "pages": [4, 5]}],
    )

    assert scores["recall_at_k"] == 1.0
    assert scores["mrr"] == 0.5


def test_retrieval_metrics_returns_zero_for_no_matching_source():
    scores = retrieval_metrics(
        [SimpleNamespace(source="other.pdf", page=1)],
        [{"source": "paper.pdf", "pages": [1]}],
    )

    assert scores == {"recall_at_k": 0.0, "mrr": 0.0}

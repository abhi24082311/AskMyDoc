"""Tests for cross-encoder result ordering."""

from __future__ import annotations

import numpy as np

from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store.base import SearchResult


def test_reranker_orders_by_cross_encoder_score() -> None:
    class FakeModel:
        def predict(self, pairs, show_progress_bar):
            assert pairs == [("question", "first passage"), ("question", "second passage")]
            assert show_progress_bar is False
            return np.asarray([0.1, 0.9])

    reranker = CrossEncoderReranker("unused-in-test")
    reranker._model = FakeModel()
    candidates = [
        SearchResult("first", "first passage", "a.md", 0, "A", 0.5, 1),
        SearchResult("second", "second passage", "b.md", 0, "B", 0.4, 2),
    ]

    results = reranker.rerank("question", candidates, top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "second"
    assert results[0].score == 0.9
    assert results[0].rank == 1

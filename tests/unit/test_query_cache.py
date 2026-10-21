"""Tests for completed ask-response caching."""

from __future__ import annotations

from app.query_cache import AskResponseCache
from app.schemas import AskResponse, Citation, LatencyBreakdown


def _response(answer: str) -> AskResponse:
    return AskResponse(
        answer=answer,
        citations=[
            Citation(
                source="guide.md",
                page=0,
                chunk_id="abc",
                section_heading="Intro",
                snippet="A supporting passage",
            )
        ],
        latency=LatencyBreakdown(retrieve_ms=1, rerank_ms=0, generate_ms=2, total_ms=3),
        retrieval_mode="vector_only",
        chunks_retrieved=1,
    )


def test_cache_returns_independent_response_copy() -> None:
    cache = AskResponseCache(max_size=2, ttl_seconds=60)
    key = ("question", 4, "vector_only")
    cache.put(key, _response("answer"))

    cached = cache.get(key)
    assert cached is not None
    cached.answer = "changed by caller"

    assert cache.get(key).answer == "answer"


def test_cache_evicts_oldest_entry_at_capacity() -> None:
    cache = AskResponseCache(max_size=1, ttl_seconds=60)
    first = ("first", 4, "vector_only")
    second = ("second", 4, "vector_only")
    cache.put(first, _response("first"))
    cache.put(second, _response("second"))

    assert cache.get(first) is None
    assert cache.get(second).answer == "second"


def test_zero_size_disables_cache() -> None:
    cache = AskResponseCache(max_size=0, ttl_seconds=60)
    key = ("question", 4, "vector_only")
    cache.put(key, _response("answer"))

    assert cache.get(key) is None

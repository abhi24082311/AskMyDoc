"""Unit tests for vector retrieval orchestration."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from retrieval.search_service import SearchService
from retrieval.vector_store.base import SearchResult


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_single(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self) -> None:
        self.top_k: int | None = None

    def query(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        assert query_embedding == [0.1, 0.2]
        self.top_k = top_k
        return [SearchResult("chunk-1", "answer context", "guide.md", 0, "Intro", 0.9, 1)]


class FakeBM25Index:
    def __init__(self) -> None:
        self.refreshed = False

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return [SearchResult("chunk-2", "exact keyword match", "notes.md", 1, "Details", 2.0, 1)]

    def refresh(self) -> bool:
        self.refreshed = True
        return True


class FakeReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        self.calls += 1
        return [
            SearchResult(
                chunk_id=c.chunk_id,
                text=c.text,
                source=c.source,
                page=c.page,
                section_heading=c.section_heading,
                score=1.0 / rank,
                rank=rank,
            )
            for rank, c in enumerate(list(reversed(candidates))[:top_k], start=1)
        ]


def test_search_uses_configured_top_k(tmp_path: Path) -> None:
    settings = Settings(
        docs_dir=tmp_path / "docs",
        chroma_persist_dir=tmp_path / "chroma",
        top_k=6,
        retrieval_mode="vector_only",
    )
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    service = SearchService(store, embedder, settings)

    results, elapsed_ms, rerank_ms = service.search("What is retrieval?")

    assert embedder.queries == ["What is retrieval?"]
    assert store.top_k == 6
    assert results[0].chunk_id == "chunk-1"
    assert elapsed_ms >= 0
    assert rerank_ms == 0


def test_search_honors_top_k_override(tmp_path: Path) -> None:
    settings = Settings(
        docs_dir=tmp_path / "docs",
        chroma_persist_dir=tmp_path / "chroma",
        top_k=6,
        retrieval_mode="vector_only",
    )
    store = FakeVectorStore()
    service = SearchService(store, FakeEmbedder(), settings)

    service.search("Question", top_k=2)

    assert store.top_k == 2


def test_hybrid_mode_fuses_vector_and_keyword_results(tmp_path: Path) -> None:
    settings = Settings(
        docs_dir=tmp_path / "docs",
        chroma_persist_dir=tmp_path / "chroma",
        bm25_index_dir=tmp_path / "bm25",
        top_k=2,
        retrieval_candidate_k=5,
        retrieval_mode="hybrid",
        query_cache_size=0,
    )
    embedder = FakeEmbedder()
    service = SearchService(FakeVectorStore(), embedder, settings, bm25_index=FakeBM25Index())

    results, retrieve_ms, rerank_ms = service.search("exact phrase")

    assert {result.chunk_id for result in results} == {"chunk-1", "chunk-2"}
    assert retrieve_ms >= 0
    assert rerank_ms == 0


def test_hybrid_rerank_calls_reranker_and_keeps_stage_latency(tmp_path: Path) -> None:
    settings = Settings(
        docs_dir=tmp_path / "docs",
        chroma_persist_dir=tmp_path / "chroma",
        bm25_index_dir=tmp_path / "bm25",
        top_k=1,
        retrieval_candidate_k=5,
        retrieval_mode="hybrid_rerank",
        query_cache_size=0,
    )
    reranker = FakeReranker()
    service = SearchService(
        FakeVectorStore(),
        FakeEmbedder(),
        settings,
        bm25_index=FakeBM25Index(),
        reranker=reranker,
    )

    results, retrieve_ms, rerank_ms = service.search("exact phrase")

    assert reranker.calls == 1
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-2"
    assert retrieve_ms >= 0
    assert rerank_ms >= 0

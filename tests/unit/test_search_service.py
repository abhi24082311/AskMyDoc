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


def test_search_uses_configured_top_k(tmp_path: Path) -> None:
    settings = Settings(
        docs_dir=tmp_path / "docs",
        chroma_persist_dir=tmp_path / "chroma",
        top_k=6,
    )
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    service = SearchService(store, embedder, settings)

    results, elapsed_ms = service.search("What is retrieval?")

    assert embedder.queries == ["What is retrieval?"]
    assert store.top_k == 6
    assert results[0].chunk_id == "chunk-1"
    assert elapsed_ms >= 0


def test_search_honors_top_k_override(tmp_path: Path) -> None:
    settings = Settings(
        docs_dir=tmp_path / "docs",
        chroma_persist_dir=tmp_path / "chroma",
        top_k=6,
    )
    store = FakeVectorStore()
    service = SearchService(store, FakeEmbedder(), settings)

    service.search("Question", top_k=2)

    assert store.top_k == 2

"""Tests for persistent BM25 indexing over stored chunks."""

from __future__ import annotations

from pathlib import Path

from retrieval.bm25_index import BM25Index
from retrieval.vector_store.base import SearchResult


class MemoryStore:
    def __init__(self, chunks: list[SearchResult]) -> None:
        self.chunks = chunks

    def get_all_chunks(self) -> list[SearchResult]:
        return list(self.chunks)


def _chunks() -> list[SearchResult]:
    return [
        SearchResult("a", "Atlas uses local storage", "a.md", 0, "A", 0.0),
        SearchResult("b", "Another document about retrieval", "b.md", 1, "B", 0.0),
        SearchResult("c", "Atlas retrieves selected passages", "c.md", 2, "C", 0.0),
    ]


def test_bm25_finds_exact_terms_and_preserves_metadata(tmp_path: Path) -> None:
    index = BM25Index(MemoryStore(_chunks()), tmp_path)

    results = index.search("Atlas", top_k=2)

    assert {result.chunk_id for result in results} == {"a", "c"}
    assert results[0].source in {"a.md", "c.md"}
    assert results[0].rank == 1
    assert (tmp_path / "bm25_index.json").exists()


def test_bm25_cache_load_avoids_rebuilding_index(monkeypatch, tmp_path: Path) -> None:
    store = MemoryStore(_chunks())
    BM25Index(store, tmp_path)

    def unexpected_rebuild(_documents):
        raise AssertionError("a matching persisted index should be loaded")

    monkeypatch.setattr(BM25Index, "_build_index", staticmethod(unexpected_rebuild))
    reloaded = BM25Index(store, tmp_path)

    assert [result.chunk_id for result in reloaded.search("Atlas", top_k=2)] == ["a", "c"]


def test_bm25_refresh_rebuilds_after_document_changes(tmp_path: Path) -> None:
    store = MemoryStore(_chunks())
    index = BM25Index(store, tmp_path)
    store.chunks = [
        SearchResult("z", "Zephyr appears in this document", "z.md", 0, "Z", 0.0)
    ]

    assert index.refresh() is True
    assert [result.chunk_id for result in index.search("Zephyr", top_k=2)] == ["z"]

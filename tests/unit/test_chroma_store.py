"""Tests for the Chroma vector-store adapter."""

from __future__ import annotations

from pathlib import Path

from retrieval.vector_store.chroma_store import ChromaVectorStore


def test_empty_collection_returns_no_search_results(tmp_path: Path) -> None:
    store = ChromaVectorStore(str(tmp_path / "chroma"), "empty-test")

    assert store.query([0.1, 0.2], top_k=4) == []
    assert store.get_all_chunks() == []


def test_get_all_chunks_returns_text_and_metadata(tmp_path: Path) -> None:
    store = ChromaVectorStore(str(tmp_path / "chroma"), "stored-test")
    store.upsert(
        chunk_ids=["chunk-a", "chunk-b"],
        texts=["Atlas is a project", "Chroma stores vectors"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {"source": "a.md", "page": 0, "section_heading": "Intro", "chunk_index": 0},
            {"source": "b.pdf", "page": 2, "section_heading": "Storage", "chunk_index": 1},
        ],
    )

    chunks = {chunk.chunk_id: chunk for chunk in store.get_all_chunks()}

    assert chunks["chunk-a"].text == "Atlas is a project"
    assert chunks["chunk-a"].source == "a.md"
    assert chunks["chunk-b"].page == 2
    assert chunks["chunk-b"].section_heading == "Storage"

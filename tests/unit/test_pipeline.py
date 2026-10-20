"""Unit tests for the ingestion pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ingestion import pipeline


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(index)] for index, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: dict[str, tuple[str, list[float], dict]] = {}

    def upsert(self, chunk_ids, texts, embeddings, metadatas) -> None:
        for values in zip(chunk_ids, texts, embeddings, metadatas):
            chunk_id, text, embedding, metadata = values
            self.upserted[chunk_id] = (text, embedding, metadata)

    def count(self) -> int:
        return len(self.upserted)


class FakeChunker:
    def chunk_documents(self, raw_docs):
        return [
            SimpleNamespace(
                text="Document text",
                chunk_id="stable-id",
                source="notes/guide.md",
                page=0,
                section_heading="Overview",
                chunk_index=0,
            )
        ]


def test_run_ingestion_embeds_and_upserts_chunks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "load_documents", lambda _docs_dir: [object()])
    store = FakeVectorStore()

    documents, chunks = pipeline.run_ingestion(
        docs_dir=tmp_path,
        vector_store=store,
        embedding_model=FakeEmbeddingModel(),
        chunker=FakeChunker(),
    )

    assert (documents, chunks) == (1, 1)
    assert store.upserted["stable-id"] == (
        "Document text",
        [0.0],
        {
            "source": "notes/guide.md",
            "page": 0,
            "section_heading": "Overview",
            "chunk_index": 0,
        },
    )


def test_run_ingestion_empty_directory_skips_embedding(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "load_documents", lambda _docs_dir: [])

    class UnexpectedEmbedder:
        def encode(self, texts):
            raise AssertionError("embedding should be skipped when there are no documents")

    store = FakeVectorStore()
    documents, chunks = pipeline.run_ingestion(
        docs_dir=tmp_path,
        vector_store=store,
        embedding_model=UnexpectedEmbedder(),
        chunker=FakeChunker(),
    )

    assert (documents, chunks) == (0, 0)
    assert store.count() == 0

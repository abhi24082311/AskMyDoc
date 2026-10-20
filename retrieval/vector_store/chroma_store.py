"""ChromaDB implementation of the VectorStore interface.

ChromaDB is an embedded vector database that persists to disk.  We use the
PersistentClient so data survives process restarts.  Embeddings are stored
pre-computed (we pass them in) so Chroma does not run its own embedding model.

Idempotency: Chroma's collection.upsert() replaces documents with matching IDs
in-place, satisfying our "re-ingest safely" requirement.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from retrieval.vector_store.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)

# Maximum batch size for a single Chroma upsert call to avoid memory spikes
_BATCH_SIZE = 512


class ChromaVectorStore(VectorStore):
    """Chroma-backed persistent vector store.

    Args:
        persist_dir: Directory where ChromaDB stores its SQLite + index files.
        collection_name: Name of the Chroma collection (table equivalent).
    """

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # get_or_create is idempotent on restart
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        logger.info(
            "ChromaVectorStore ready: collection='%s', persist_dir='%s', count=%d",
            collection_name,
            persist_dir,
            self._collection.count(),
        )

    def upsert(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Batch-upsert chunks into the Chroma collection.

        Splits into batches of _BATCH_SIZE to prevent OOM on large ingestions.
        """
        total = len(chunk_ids)
        for start in range(0, total, _BATCH_SIZE):
            end = min(start + _BATCH_SIZE, total)
            self._collection.upsert(
                ids=chunk_ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
        logger.info("Upserted %d chunks into Chroma.", total)

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """Query the collection for the nearest neighbours of query_embedding."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self._collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            # Chroma cosine distance is in [0, 2]; convert to similarity [0, 1]
            score = 1.0 - (dist / 2.0)
            search_results.append(
                SearchResult(
                    chunk_id=cid,
                    text=doc,
                    source=meta.get("source", ""),
                    page=int(meta.get("page", 0)),
                    section_heading=meta.get("section_heading", ""),
                    score=score,
                    rank=rank,
                )
            )

        return search_results

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self._collection.count()

    def get_all_texts_and_ids(self) -> tuple[list[str], list[str]]:
        """Retrieve all documents from the collection for BM25 index building."""
        if self._collection.count() == 0:
            return [], []
        result = self._collection.get(include=["documents"])
        return result.get("documents", []), result.get("ids", [])

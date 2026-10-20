"""Vector-search-only retrieval service for Phase 1.

Phase 2 will extend this with BM25 + RRF + CrossEncoder via the
SearchService class. For now we expose a clean interface so the
FastAPI endpoint doesn't need to change when we add hybrid retrieval.
"""

from __future__ import annotations

import logging
import time

from app.config import Settings
from retrieval.embedder import EmbeddingModel
from retrieval.vector_store.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class SearchService:
    """Coordinates embedding + vector retrieval (Phase 1: vector_only).

    Args:
        vector_store: Any VectorStore implementation (Chroma, Weaviate, …).
        embedding_model: Loaded EmbeddingModel instance.
        settings: Application settings (top_k, retrieval_mode).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        settings: Settings,
    ) -> None:
        self._store = vector_store
        self._embedder = embedding_model
        self._settings = settings

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[SearchResult], float]:
        """Run a retrieval query and return results with latency.

        Args:
            query: Natural-language question string.
            top_k: Override for number of results (defaults to settings.top_k).

        Returns:
            Tuple of (results, retrieve_ms) where retrieve_ms is the wall-clock
            time in milliseconds spent on embedding + vector lookup.
        """
        k = top_k or self._settings.top_k
        t0 = time.perf_counter()

        query_emb = self._embedder.encode_single(query)
        results = self._store.query(query_embedding=query_emb, top_k=k)

        retrieve_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Vector search returned %d results in %.1f ms",
            len(results),
            retrieve_ms,
        )
        return results, retrieve_ms

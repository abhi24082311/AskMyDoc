"""Coordinates vector, BM25, fusion, reranking, and query caching."""

from __future__ import annotations

import logging
import time
from dataclasses import replace

from app.config import Settings
from retrieval.bm25_index import BM25Index
from retrieval.embedder import EmbeddingModel
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class SearchService:
    """Coordinate the configured retrieval pipeline.

    Vector-only mode needs only Chroma. Hybrid modes also use a persisted BM25
    index and Reciprocal Rank Fusion; hybrid_rerank applies a CrossEncoder to
    the fused candidate set before selecting the context chunks.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        settings: Settings,
        *,
        bm25_index: BM25Index | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._store = vector_store
        self._embedder = embedding_model
        self._settings = settings
        self._bm25 = bm25_index
        if settings.retrieval_mode != "vector_only" and self._bm25 is None:
            self._bm25 = BM25Index(vector_store, settings.bm25_index_dir)
        self._reranker = reranker or CrossEncoderReranker(settings.reranker_model_name)

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[SearchResult], float, float]:
        """Run retrieval and return results, retrieval latency, and rerank latency."""
        k = top_k if top_k is not None else self._settings.top_k
        if k < 1:
            raise ValueError("top_k must be at least 1")
        started = time.perf_counter()
        query_embedding = self._embedder.encode_single(query)
        if self._settings.retrieval_mode == "vector_only":
            results = self._store.query(query_embedding=query_embedding, top_k=k)
            rerank_ms = 0.0
        else:
            candidate_k = max(k, self._settings.retrieval_candidate_k)
            vector_results = self._store.query(
                query_embedding=query_embedding,
                top_k=candidate_k,
            )
            assert self._bm25 is not None
            bm25_results = self._bm25.search(query, top_k=candidate_k)
            fused = reciprocal_rank_fusion(
                [vector_results, bm25_results],
                rrf_k=self._settings.rrf_k,
                top_k=candidate_k,
            )

            retrieve_ms = (time.perf_counter() - started) * 1000
            if self._settings.retrieval_mode == "hybrid_rerank":
                rerank_started = time.perf_counter()
                results = self._reranker.rerank(query, fused, top_k=k)
                rerank_ms = (time.perf_counter() - rerank_started) * 1000
            else:
                results = fused[:k]
                rerank_ms = 0.0
            logger.info(
                "Hybrid search returned %d results in %.1fms; rerank=%.1fms; mode=%s",
                len(results),
                retrieve_ms,
                rerank_ms,
                self._settings.retrieval_mode,
            )
            return results, retrieve_ms, rerank_ms

        retrieve_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Vector search returned %d results in %.1fms",
            len(results),
            retrieve_ms,
        )
        return results, retrieve_ms, rerank_ms

    def refresh_bm25_index(self) -> bool:
        """Refresh the lexical index after ingestion."""
        return self._bm25.refresh() if self._bm25 is not None else False


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Merge ranked result lists using ``sum(1 / (rrf_k + rank))``."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")
    scores: dict[str, float] = {}
    records: dict[str, SearchResult] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            if result.chunk_id not in records:
                records[result.chunk_id] = result
                first_seen[result.chunk_id] = order
                order += 1
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], first_seen[chunk_id]))
    if top_k is not None:
        ranked_ids = ranked_ids[:top_k]
    return [
        replace(records[chunk_id], score=scores[chunk_id], rank=rank)
        for rank, chunk_id in enumerate(ranked_ids, start=1)
    ]

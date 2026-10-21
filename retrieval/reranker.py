"""Cross-encoder reranking for a small set of retrieved passages."""

from __future__ import annotations

import logging

from retrieval.vector_store.base import SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Score query/passage pairs with a lazily loaded sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Return the highest-scoring candidates, ordered by cross-encoder score."""
        if not candidates or top_k <= 0:
            return []
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for hybrid_rerank retrieval"
                ) from exc
            logger.info("Loading cross-encoder model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: (-float(item[1]), item[0].rank),
        )[:top_k]
        return [
            SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                source=result.source,
                page=result.page,
                section_heading=result.section_heading,
                score=float(score),
                rank=rank,
            )
            for rank, (result, score) in enumerate(ranked, start=1)
        ]

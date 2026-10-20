"""Embedding model wrapper using sentence-transformers.

Design decision: We wrap sentence-transformers in a simple class rather than
using Chroma's built-in embedding functions. This gives us:
1. Control over batching and device placement.
2. The ability to swap models without touching Chroma config.
3. Testability: we can mock this class in unit tests without needing a GPU.

The embedding model is loaded once at startup (expensive) and reused across
all ingestion and query calls (cheap).
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wraps a SentenceTransformer model for encoding texts into dense vectors.

    Args:
        model_name: HuggingFace model identifier, e.g. 'BAAI/bge-small-en-v1.5'.
        device: PyTorch device string ('cpu', 'cuda', 'mps').
            Defaults to None, which lets sentence-transformers auto-detect.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model ready. Dimension: %d", self._dim)

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality."""
        return self._dim  # type: ignore[return-value]

    def encode(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Encode a list of texts into embedding vectors.

        Args:
            texts: List of strings to encode.
            batch_size: Number of texts to encode per forward pass.

        Returns:
            List of embedding vectors as plain Python float lists.
        """
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        """Encode a single query string.

        Args:
            text: The query text.

        Returns:
            Embedding vector as a Python float list.
        """
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> EmbeddingModel:
    """Return a cached singleton EmbeddingModel instance.

    Using lru_cache ensures the model is loaded only once per process even
    if called from multiple places (e.g. ingestion CLI + FastAPI startup).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded EmbeddingModel instance.
    """
    return EmbeddingModel(model_name)

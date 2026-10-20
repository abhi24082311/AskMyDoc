"""Abstract VectorStore interface.

Why an interface here?  The Dependency Inversion Principle tells us that
high-level modules (retrieval, ingestion) should depend on abstractions, not
concretions.  Every method that touches storage goes through this ABC, making
the codebase testable with a simple in-memory mock and swappable to any vector
database without touching retrieval logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    """A single retrieved chunk with its relevance score.

    Attributes:
        chunk_id: The stable deterministic chunk identifier.
        text: The chunk's text content.
        source: Relative file path of originating document.
        page: Page number (0 for non-paginated sources).
        section_heading: Section heading context.
        score: Relevance score (higher = more relevant). Interpretation is
            backend-dependent (cosine similarity, distance, etc.).
        rank: 1-based ordinal rank within the result list (set by retrieval).
    """

    chunk_id: str
    text: str
    source: str
    page: int
    section_heading: str
    score: float
    rank: int = 0


class VectorStore(ABC):
    """Abstract vector store contract.

    All vector store implementations must satisfy this interface.
    """

    @abstractmethod
    def upsert(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Insert or update chunks by their stable chunk IDs.

        Idempotency contract: if a chunk_id already exists in the store, its
        embedding and metadata are updated in-place.  Callers may safely call
        this on every ingestion run without creating duplicate entries.

        Args:
            chunk_ids: Parallel list of stable chunk IDs.
            texts: Parallel list of chunk text content.
            embeddings: Parallel list of embedding vectors.
            metadatas: Parallel list of metadata dicts (source, page, etc.).
        """
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """Retrieve the top-k most similar chunks for a query embedding.

        Args:
            query_embedding: Embedded representation of the query.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by descending relevance.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks currently in the store.

        Returns:
            Integer count of stored chunks.
        """
        ...

    @abstractmethod
    def get_all_texts_and_ids(self) -> tuple[list[str], list[str]]:
        """Retrieve all stored texts and their IDs for BM25 index building.

        Returns:
            Tuple of (texts, chunk_ids) lists in matching order.
        """
        ...

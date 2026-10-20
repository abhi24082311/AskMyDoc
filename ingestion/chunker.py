"""Token-aware sliding-window document chunker.

Design decision: We use tiktoken (OpenAI's tokenizer library) to count tokens
against a standard BPE encoding (cl100k_base) rather than character counts.
This ensures chunk sizes are meaningful w.r.t. the embedding model's context
window and avoids silent truncation of long chunks.

Chunk IDs are deterministic SHA-256 hashes derived from (source, page, index,
text). Hashing ensures idempotent ingestion: re-running on the same document
produces the same IDs and Chroma's upsert simply no-ops.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

import tiktoken

from ingestion.loader import RawDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk ready for embedding and storage.

    Attributes:
        chunk_id: Stable deterministic identifier (hex SHA-256).
        text: The chunk text used for embedding.
        source: Relative path to originating document file.
        page: Page number (1-based for PDFs, 0 for Markdown).
        section_heading: Section heading context from the source document.
        chunk_index: 0-based position of this chunk within its parent document.
    """

    chunk_id: str
    text: str
    source: str
    page: int
    section_heading: str
    chunk_index: int


def _make_chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    """Derive a stable, collision-resistant chunk ID.

    Using a content hash ensures idempotency: the same logical chunk always
    maps to the same ID, so re-ingesting a document is a no-op in Chroma.

    Args:
        source: Source file path string.
        page: Page number.
        chunk_index: Position of chunk in document.
        text: Chunk text (first 200 chars used to avoid pathological inputs).

    Returns:
        16-character hex prefix of SHA-256 hash (sufficient for uniqueness
        given typical document collection sizes; extend to 32 chars if needed).
    """
    raw = f"{source}||{page}||{chunk_index}||{text[:200]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class TokenChunker:
    """Sliding-window chunker that counts tokens with tiktoken.

    Args:
        chunk_size: Target chunk size in tokens.
        overlap: Number of overlap tokens between consecutive chunks.
        encoding_name: tiktoken encoding name (e.g. 'cl100k_base').
    """

    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 90,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
        self._chunk_size = chunk_size
        self._overlap = overlap
        try:
            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load tiktoken encoding '{encoding_name}': {exc}") from exc

    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        """Split a single RawDocument into overlapping token-bounded chunks.

        The algorithm:
        1. Tokenize the full document text.
        2. Slide a window of size ``chunk_size`` over the token list, advancing
           by (chunk_size - overlap) tokens each step.
        3. Decode each window slice back to text for embedding.
        4. Assign a deterministic ID to each chunk.

        Args:
            doc: The raw document to chunk.

        Returns:
            Ordered list of Chunk objects.  Empty if doc.text is blank.
        """
        text = doc.text.strip()
        if not text:
            return []

        tokens = self._enc.encode(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        step = self._chunk_size - self._overlap
        chunks: list[Chunk] = []
        chunk_index = 0
        start = 0

        while start < total_tokens:
            end = min(start + self._chunk_size, total_tokens)
            token_slice = tokens[start:end]
            chunk_text = self._enc.decode(token_slice).strip()

            if chunk_text:
                chunk_id = _make_chunk_id(doc.source, doc.page, chunk_index, chunk_text)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=doc.source,
                        page=doc.page,
                        section_heading=doc.section_heading,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            if end == total_tokens:
                break
            start += step

        logger.debug(
            "Chunked '%s' (page %d): %d tokens → %d chunks",
            doc.source,
            doc.page,
            total_tokens,
            len(chunks),
        )
        return chunks

    def chunk_documents(self, docs: list[RawDocument]) -> list[Chunk]:
        """Chunk a list of RawDocuments.

        Args:
            docs: Raw documents to process.

        Returns:
            Flat list of all Chunk objects across all documents.
        """
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Total chunks produced: %d from %d documents", len(all_chunks), len(docs))
        return all_chunks

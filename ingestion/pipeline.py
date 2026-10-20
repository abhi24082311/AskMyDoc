"""Ingestion pipeline orchestrator.

Wires together loader → chunker → embedder → vector store.
Can be called from the CLI (make ingest) or the POST /ingest API endpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ingestion.chunker import TokenChunker
from ingestion.loader import load_documents
from retrieval.embedder import EmbeddingModel
from retrieval.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


def run_ingestion(
    docs_dir: Path,
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
    chunker: TokenChunker,
) -> tuple[int, int]:
    """Load, chunk, embed, and index all documents in docs_dir.

    This function is idempotent: re-running it on the same documents will
    upsert chunks with identical IDs, resulting in no duplicate entries in
    the vector store.

    Args:
        docs_dir: Directory containing raw PDF and Markdown files.
        vector_store: VectorStore implementation to upsert into.
        embedding_model: EmbeddingModel for generating embeddings.
        chunker: TokenChunker for splitting documents.

    Returns:
        Tuple of (documents_loaded, chunks_indexed).
    """
    logger.info("Starting ingestion from: %s", docs_dir)

    # Step 1: Load raw documents
    raw_docs = load_documents(docs_dir)
    if not raw_docs:
        logger.warning("No documents found in %s. Nothing to ingest.", docs_dir)
        return 0, 0

    logger.info("Loaded %d document pages/sections.", len(raw_docs))

    # Step 2: Chunk documents
    chunks = chunker.chunk_documents(raw_docs)
    if not chunks:
        logger.warning("Chunking produced 0 chunks. Check document content.")
        return len(raw_docs), 0

    logger.info("Produced %d chunks. Generating embeddings …", len(chunks))

    # Step 3: Embed all chunks
    texts = [c.text for c in chunks]
    embeddings = embedding_model.encode(texts)

    # Step 4: Prepare metadata and upsert
    chunk_ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "source": c.source,
            "page": c.page,
            "section_heading": c.section_heading,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    vector_store.upsert(
        chunk_ids=chunk_ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    total = vector_store.count()
    logger.info(
        "Ingestion complete. Documents: %d, chunks: %d, total in store: %d",
        len(raw_docs),
        len(chunks),
        total,
    )
    return len(raw_docs), len(chunks)

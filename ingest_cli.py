"""CLI entrypoint for ingestion.

Usage:
    python ingest_cli.py
    python ingest_cli.py --docs-dir /path/to/docs

This script is equivalent to calling POST /ingest but doesn't require the API
server to be running. Useful for offline batch ingestion and CI pipelines.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from ingestion.pipeline import run_ingestion
from retrieval.embedder import get_embedding_model
from retrieval.vector_store.chroma_store import ChromaVectorStore
from ingestion.chunker import TokenChunker

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=settings.docs_dir,
        help=f"Directory containing PDF/Markdown files (default: {settings.docs_dir})",
    )
    args = parser.parse_args()

    logger.info("=== Ask My Docs – Ingestion CLI ===")
    logger.info("docs_dir:     %s", args.docs_dir)
    logger.info("chroma_dir:   %s", settings.chroma_persist_dir)
    logger.info("embed_model:  %s", settings.embedding_model_name)
    logger.info("chunk_size:   %d tokens", settings.chunk_size_tokens)
    logger.info("chunk_overlap:%d tokens", settings.chunk_overlap_tokens)

    vector_store = ChromaVectorStore(
        persist_dir=str(settings.chroma_persist_dir),
        collection_name=settings.chroma_collection_name,
    )
    embedding_model = get_embedding_model(settings.embedding_model_name)
    chunker = TokenChunker(
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
        encoding_name=settings.tokenizer_encoding,
    )

    n_docs, n_chunks = run_ingestion(
        docs_dir=args.docs_dir,
        vector_store=vector_store,
        embedding_model=embedding_model,
        chunker=chunker,
    )

    total = vector_store.count()
    print(f"\n✓ Ingestion complete.")
    print(f"  Documents loaded : {n_docs}")
    print(f"  Chunks indexed   : {n_chunks}")
    print(f"  Total in store   : {total}")


if __name__ == "__main__":
    main()

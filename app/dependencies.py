"""FastAPI dependency injection wiring.

We use FastAPI's dependency injection system (Depends) to:
1. Keep app startup logic out of route handlers.
2. Make services testable: tests can override dependencies with mocks.
3. Ensure singletons (embedding model, vector store) are initialised once.

Design note: We could use lifespan context managers (FastAPI 0.93+) instead of
module-level singletons. We chose module-level globals here because Chroma's
PersistentClient is not safely picklable for multiprocessing, and module-level
init is straightforward and explicit for a single-worker setup.
"""

from __future__ import annotations

import logging

from app.config import Settings, settings
from ingestion.chunker import TokenChunker
from llm.base import LLMProvider
from llm.factory import create_llm_provider
from retrieval.embedder import EmbeddingModel, get_embedding_model
from retrieval.search_service import SearchService
from retrieval.vector_store.base import VectorStore
from retrieval.vector_store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# ─── Singletons (initialised once at module import time) ─────────────────────

_vector_store: VectorStore | None = None
_embedding_model: EmbeddingModel | None = None
_llm_provider: LLMProvider | None = None
_search_service: SearchService | None = None
_chunker: TokenChunker | None = None


def _init_services(app_settings: Settings = settings) -> None:
    """Initialise all heavyweight singleton services.

    Called once from the FastAPI lifespan or on first request.
    """
    global _vector_store, _embedding_model, _llm_provider, _search_service, _chunker

    if _vector_store is None:
        logger.info("Initialising ChromaVectorStore …")
        _vector_store = ChromaVectorStore(
            persist_dir=str(app_settings.chroma_persist_dir),
            collection_name=app_settings.chroma_collection_name,
        )

    if _embedding_model is None:
        logger.info("Loading embedding model: %s", app_settings.embedding_model_name)
        _embedding_model = get_embedding_model(app_settings.embedding_model_name)

    if _llm_provider is None:
        logger.info("Initialising Ollama provider: %s", app_settings.ollama_model)
        _llm_provider = create_llm_provider(app_settings)

    if _search_service is None:
        _search_service = SearchService(
            vector_store=_vector_store,
            embedding_model=_embedding_model,
            settings=app_settings,
        )

    if _chunker is None:
        _chunker = TokenChunker(
            chunk_size=app_settings.chunk_size_tokens,
            overlap=app_settings.chunk_overlap_tokens,
            encoding_name=app_settings.tokenizer_encoding,
        )


# ─── FastAPI Dependency Functions ────────────────────────────────────────────

def get_settings() -> Settings:
    """Return the application settings singleton."""
    return settings


def get_vector_store() -> VectorStore:
    """Return the VectorStore singleton, initialising if necessary."""
    _init_services()
    assert _vector_store is not None
    return _vector_store


def get_embedding_model_dep() -> EmbeddingModel:
    """Return the EmbeddingModel singleton."""
    _init_services()
    assert _embedding_model is not None
    return _embedding_model


def get_llm_provider() -> LLMProvider:
    """Return the LLM provider singleton."""
    _init_services()
    assert _llm_provider is not None
    return _llm_provider


def get_search_service() -> SearchService:
    """Return the SearchService singleton."""
    _init_services()
    assert _search_service is not None
    return _search_service


def get_chunker() -> TokenChunker:
    """Return the TokenChunker singleton."""
    _init_services()
    assert _chunker is not None
    return _chunker

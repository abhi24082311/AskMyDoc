"""FastAPI application: routes, startup, and error handling.

Route summary:
  GET  /health      – liveness + system info
  POST /ingest      – trigger document ingestion
  POST /ask         – question-answering with citations
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.query_cache import AskResponseCache
from app.dependencies import (
    _init_services,
    get_chunker,
    get_embedding_model_dep,
    get_llm_provider,
    get_search_service,
    get_settings,
    get_vector_store,
)
from app.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LatencyBreakdown,
)
from ingestion.pipeline import run_ingestion
from llm.prompt import generate_answer

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)
_ask_cache = AskResponseCache(
    max_size=settings.query_cache_size,
    ttl_seconds=settings.query_cache_ttl_seconds,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all services on startup, clean up on shutdown."""
    logger.info("Starting up Ask My Docs RAG API …")
    _init_services()
    logger.info("All services initialised. Ready to serve requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Ask My Docs RAG API",
    description=(
        "Production-grade Retrieval-Augmented Generation system. "
        "Ingest documents, then ask questions grounded in their content."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and system info",
    tags=["System"],
)
def health(
    cfg=Depends(get_settings),
    store=Depends(get_vector_store),
):
    """Return system health and current index statistics."""
    return HealthResponse(
        status="ok",
        chunks_in_store=store.count(),
        llm_provider="ollama",
        embedding_model=cfg.embedding_model_name,
        retrieval_mode=cfg.retrieval_mode,
    )


# ─── Ingest ──────────────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents into the vector store",
    tags=["Ingestion"],
)
def ingest(
    body: IngestRequest = IngestRequest(),
    cfg=Depends(get_settings),
    store=Depends(get_vector_store),
    embedder=Depends(get_embedding_model_dep),
    chunker=Depends(get_chunker),
    search_svc=Depends(get_search_service),
):
    """Load all PDFs and Markdown files from the docs directory, chunk them,
    embed them, and upsert into the vector store.

    This endpoint is **idempotent**: re-running it on the same documents
    produces no duplicates (existing chunks are updated in-place).
    """
    docs_dir = Path(body.docs_dir) if body.docs_dir else cfg.docs_dir

    if not docs_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"docs_dir '{docs_dir}' does not exist.",
        )

    try:
        n_docs, n_chunks = run_ingestion(
            docs_dir=docs_dir,
            vector_store=store,
            embedding_model=embedder,
            chunker=chunker,
        )
        search_svc.refresh_bm25_index()
        _ask_cache.clear()
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        status="ok",
        documents_loaded=n_docs,
        chunks_created=n_chunks,
        total_chunks_in_store=store.count(),
        message=f"Ingested {n_docs} document sections into {n_chunks} chunks.",
    )


# ─── Ask ─────────────────────────────────────────────────────────────────────

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question grounded in the indexed documents",
    tags=["QA"],
)
def ask(
    body: AskRequest,
    cfg=Depends(get_settings),
    search_svc=Depends(get_search_service),
    llm=Depends(get_llm_provider),
):
    """Retrieve relevant document chunks and generate a cited answer.

    Returns the LLM answer, citations (source + page + chunk_id + snippet),
    and per-stage latency breakdown in milliseconds.
    """
    total_start = time.perf_counter()

    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    effective_top_k = body.top_k if body.top_k is not None else cfg.top_k
    cache_key = (
        " ".join(body.question.casefold().split()),
        effective_top_k,
        cfg.retrieval_mode,
    )
    cached = _ask_cache.get(cache_key)
    if cached is not None:
        total_ms = (time.perf_counter() - total_start) * 1000
        logger.info("Ask response cache hit: total=%.1fms", total_ms)
        return cached.model_copy(
            update={
                "latency": LatencyBreakdown(
                    retrieve_ms=0.0,
                    rerank_ms=0.0,
                    generate_ms=0.0,
                    total_ms=round(total_ms, 2),
                )
            }
        )

    # ── Retrieve ──────────────────────────────────────────────────────────────
    try:
        results, retrieve_ms, rerank_ms = search_svc.search(
            query=body.question,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("Retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}") from exc

    if not results:
        # No documents indexed yet or query out-of-distribution
        response = AskResponse(
            answer="I don't know based on the provided documents.",
            citations=[],
            latency=LatencyBreakdown(
                retrieve_ms=retrieve_ms,
                rerank_ms=rerank_ms,
                generate_ms=0.0,
                total_ms=(time.perf_counter() - total_start) * 1000,
            ),
            retrieval_mode=cfg.retrieval_mode,
            chunks_retrieved=0,
        )
        _ask_cache.put(cache_key, response)
        return response

    # ── Generate ──────────────────────────────────────────────────────────────
    try:
        answer, generate_ms = generate_answer(
            query=body.question,
            results=results,
            llm=llm,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("LLM generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

    # ── Build citations ───────────────────────────────────────────────────────
    citations = [
        Citation(
            source=r.source,
            page=r.page,
            chunk_id=r.chunk_id,
            section_heading=r.section_heading,
            snippet=r.text[:200],
        )
        for r in results
    ]

    total_ms = (time.perf_counter() - total_start) * 1000
    logger.info(
        "ask: retrieve=%.1fms rerank=%.1fms generate=%.1fms total=%.1fms",
        retrieve_ms,
        rerank_ms,
        generate_ms,
        total_ms,
    )

    response = AskResponse(
        answer=answer,
        citations=citations,
        latency=LatencyBreakdown(
            retrieve_ms=round(retrieve_ms, 2),
            rerank_ms=round(rerank_ms, 2),
            generate_ms=round(generate_ms, 2),
            total_ms=round(total_ms, 2),
        ),
        retrieval_mode=cfg.retrieval_mode,
        chunks_retrieved=len(results),
    )
    _ask_cache.put(cache_key, response)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_env == "development",
    )

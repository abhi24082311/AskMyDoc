"""Pydantic request/response schemas for the FastAPI endpoints.

Keeping schemas in a dedicated module (separate from routes) allows them to be
imported by tests and the OpenAPI spec without importing FastAPI app state.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ─── Request Models ──────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question to answer from the documents.",
        examples=["What is the main contribution of the paper?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the number of context chunks to retrieve. Defaults to config value.",
    )


class IngestRequest(BaseModel):
    """Optional request body for POST /ingest (all fields optional)."""

    docs_dir: str | None = Field(
        default=None,
        description="Override the docs directory path. Defaults to config DOCS_DIR.",
    )


# ─── Response Models ─────────────────────────────────────────────────────────

class Citation(BaseModel):
    """A single document citation backing a claim in the answer."""

    source: str = Field(description="Relative file path of the source document.")
    page: int = Field(description="Page number (0 for non-paginated sources like Markdown).")
    chunk_id: str = Field(description="Stable deterministic chunk identifier.")
    section_heading: str = Field(description="Section heading context (empty string if none).")
    snippet: str = Field(description="First 200 characters of the chunk text.")


class LatencyBreakdown(BaseModel):
    """Per-stage pipeline latency in milliseconds."""

    retrieve_ms: float = Field(description="Time spent embedding and retrieving/fusing candidate chunks.")
    rerank_ms: float = Field(description="Time spent reranking retrieval candidates (0 when disabled).")
    generate_ms: float = Field(description="Time spent waiting for the LLM response.")
    total_ms: float = Field(description="Total wall-clock time for the /ask request.")


class AskResponse(BaseModel):
    """Response body for POST /ask."""

    answer: str = Field(description="LLM-generated answer grounded in retrieved documents.")
    citations: list[Citation] = Field(description="Source chunks used to generate the answer.")
    latency: LatencyBreakdown = Field(description="Per-stage latency breakdown.")
    retrieval_mode: str = Field(description="Active retrieval mode used for this request.")
    chunks_retrieved: int = Field(description="Number of context chunks passed to the LLM.")


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    status: str = Field(description="'ok' on success.")
    documents_loaded: int = Field(description="Number of raw document pages/sections loaded.")
    chunks_created: int = Field(description="Number of chunks embedded and indexed.")
    total_chunks_in_store: int = Field(description="Total chunks now in the vector store.")
    message: str = Field(description="Human-readable summary.")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(description="'ok' when all systems are operational.")
    chunks_in_store: int = Field(description="Total chunks currently indexed.")
    llm_provider: str = Field(description="Active LLM provider name.")
    embedding_model: str = Field(description="Active embedding model name.")
    retrieval_mode: str = Field(description="Active retrieval mode.")

from pathlib import Path
from typing import Literal
# pyrefly: ignore[missing-import]
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Ask My Docs RAG", description="Application name")
    app_env: str = Field(default="development", description="Environment: development, test, production")
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    log_level: str = Field(default="INFO", description="Logging level")

    # Storage Paths
    docs_dir: Path = Field(default=Path("data/docs"), description="Directory containing raw documents")
    chroma_persist_dir: Path = Field(default=Path("data/chroma_db"), description="ChromaDB persistence directory")
    chroma_collection_name: str = Field(default="ask_my_docs", description="ChromaDB collection name")
    bm25_index_dir: Path = Field(default=Path("data/bm25_index"), description="Persistent BM25 index directory")

    # Document Chunking
    chunk_size_tokens: int = Field(default=600, ge=100, le=2000, description="Chunk size in tokens")
    chunk_overlap_tokens: int = Field(default=90, ge=0, le=500, description="Chunk overlap in tokens (~15%)")
    tokenizer_encoding: str = Field(default="cl100k_base", description="tiktoken encoding for token counts")

    # Embedding & Retrieval
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model name for embeddings",
    )
    top_k: int = Field(default=4, ge=1, le=50, description="Number of context chunks to feed into LLM")
    retrieval_mode: Literal["vector_only", "hybrid", "hybrid_rerank"] = Field(
        default="vector_only",
        description="Retrieval mode: vector search, hybrid search, or hybrid with reranking",
    )
    retrieval_candidate_k: int = Field(default=20, ge=1, le=200, description="Candidates per retrieval channel")
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal Rank Fusion rank constant")
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        description="Sentence Transformers CrossEncoder model used by hybrid_rerank",
    )
    query_cache_size: int = Field(default=128, ge=0, le=10000, description="Maximum cached query results; 0 disables caching")
    query_cache_ttl_seconds: float = Field(default=300.0, ge=0.0, le=86400.0, description="Cached query lifetime")

    # Evaluation
    eval_golden_path: Path = Field(default=Path("eval/golden.jsonl"), description="Reviewed evaluation questions")
    eval_results_dir: Path = Field(default=Path("eval"), description="Evaluation output directory")
    eval_candidate_count: int = Field(default=50, ge=1, le=100, description="Number of draft Q&A pairs to generate")
    eval_ollama_model: str = Field(default="llama3.2", description="Local Ollama model used as the Ragas evaluator")
    eval_llm_timeout_seconds: int = Field(default=180, ge=15, le=900, description="Ragas evaluator request timeout")
    eval_max_tokens: int = Field(default=1024, ge=256, le=8192, description="Maximum tokens generated per Ragas judging response")
    eval_min_faithfulness: float = Field(default=0.30, ge=0.0, le=1.0)
    eval_min_answer_relevancy: float = Field(default=0.30, ge=0.0, le=1.0)
    eval_min_context_precision: float = Field(default=0.30, ge=0.0, le=1.0)
    eval_min_context_recall: float = Field(default=0.30, ge=0.0, le=1.0)
    eval_min_recall_at_k: float = Field(default=0.50, ge=0.0, le=1.0)
    eval_min_mrr: float = Field(default=0.50, ge=0.0, le=1.0)

    # LLM Settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    ollama_model: str = Field(default="llama3.2", description="Ollama model tag")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    llm_timeout_seconds: float = Field(default=60.0, ge=5.0, le=300.0, description="LLM call timeout")

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        return self

    def ensure_directories(self) -> None:
        """Create storage directories if they do not exist."""
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()

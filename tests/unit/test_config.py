"""Unit tests for application configuration invariants."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_reject_overlap_larger_than_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        Settings(
            docs_dir=tmp_path / "docs",
            chroma_persist_dir=tmp_path / "chroma",
            chunk_size_tokens=100,
            chunk_overlap_tokens=100,
        )


@pytest.mark.parametrize("mode", ["vector_only", "hybrid", "hybrid_rerank"])
def test_settings_accept_supported_retrieval_modes(tmp_path: Path, mode: str) -> None:
    settings = Settings(
        docs_dir=tmp_path / "docs",
        chroma_persist_dir=tmp_path / "chroma",
        bm25_index_dir=tmp_path / "bm25",
        retrieval_mode=mode,
    )

    assert settings.retrieval_mode == mode


def test_settings_reject_unknown_retrieval_mode(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(
            docs_dir=tmp_path / "docs",
            chroma_persist_dir=tmp_path / "chroma",
            retrieval_mode="bm25_only",
        )

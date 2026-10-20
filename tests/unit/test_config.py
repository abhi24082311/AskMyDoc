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


def test_phase_one_only_allows_vector_search(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(
            docs_dir=tmp_path / "docs",
            chroma_persist_dir=tmp_path / "chroma",
            retrieval_mode="hybrid",
        )

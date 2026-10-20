"""Tests for constructing the configured Ollama provider."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from llm.factory import create_llm_provider
from llm.ollama_provider import OllamaProvider


def test_factory_returns_ollama_using_configured_model(tmp_path: Path) -> None:
    settings = Settings(
        docs_dir=tmp_path / "docs",
        chroma_persist_dir=tmp_path / "chroma",
        ollama_model="qwen2.5:3b",
    )

    provider = create_llm_provider(settings)

    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "qwen2.5:3b"

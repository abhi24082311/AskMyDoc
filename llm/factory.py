"""Construct the configured LLM implementation."""

from __future__ import annotations

from app.config import Settings
from llm.base import LLMProvider
from llm.ollama_provider import OllamaProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Build the Ollama provider using application configuration."""
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
    )

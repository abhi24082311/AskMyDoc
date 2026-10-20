"""Abstract LLM provider interface.

Every concrete provider must implement this ABC. The current application uses Ollama.
The RAG pipeline depends only on this interface, keeping provider details out
of retrieval and answer-generation code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Structured response from an LLM provider.

    Attributes:
        text: The generated answer text.
        model: The model identifier that produced the response.
        generate_ms: Wall-clock time spent waiting for the provider in ms.
    """

    text: str
    model: str
    generate_ms: float


class LLMProvider(ABC):
    """Abstract interface that every LLM provider must satisfy."""

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response for the given prompt.

        Args:
            prompt: The complete, formatted prompt string.

        Returns:
            LLMResponse with the generated text and timing metadata.

        Raises:
            TimeoutError: If the provider does not respond within the configured timeout.
            RuntimeError: For non-retryable provider errors (e.g., invalid model name).
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the human-readable model identifier for logging and citations."""
        ...

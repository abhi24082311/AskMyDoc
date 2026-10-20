"""Ollama LLM provider implementation.

Ollama exposes a local REST API at http://localhost:11434 that is compatible
with a subset of the OpenAI chat API. We use httpx for synchronous HTTP calls
with a configurable timeout.

Design decision: We call the /api/generate endpoint (non-streaming) rather
than /api/chat because it requires no conversation history management and keeps
the provider stateless. For Phase 2 we can add streaming if needed.
"""

from __future__ import annotations

import logging
import time

import httpx

from llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a locally-running Ollama server.

    Args:
        base_url: Ollama API base URL (default: http://localhost:11434).
        model: Ollama model tag to use (e.g. 'llama3.2', 'mistral').
        temperature: Sampling temperature (0.0 = deterministic).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        logger.info("OllamaProvider initialised: model=%s, base_url=%s", model, base_url)

    @property
    def model_name(self) -> str:
        """Return the Ollama model tag."""
        return self._model

    def generate(self, prompt: str) -> LLMResponse:
        """Send a prompt to Ollama and return the generated text.

        Args:
            prompt: Fully-formatted prompt string.

        Returns:
            LLMResponse with the generated text and latency.

        Raises:
            TimeoutError: If Ollama does not respond within the timeout.
            RuntimeError: For HTTP errors or malformed responses.
        """
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": 1024,
            },
        }

        t0 = time.perf_counter()
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Ollama request timed out after {self._timeout}s for model '{self._model}'"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama HTTP error {exc.response.status_code}: {exc.response.text}"
            ) from exc

        generate_ms = (time.perf_counter() - t0) * 1000

        data = response.json()
        text = data.get("response", "").strip()
        if not text:
            raise RuntimeError(f"Ollama returned empty response. Full payload: {data}")

        logger.info("Ollama generated response in %.1f ms (%d chars)", generate_ms, len(text))
        return LLMResponse(text=text, model=self._model, generate_ms=generate_ms)

    def __del__(self) -> None:
        """Close the HTTP client on garbage collection."""
        try:
            self._client.close()
        except Exception:
            pass

"""Unit tests for the Ollama HTTP adapter."""

from __future__ import annotations

import httpx
import pytest
import json

from llm.ollama_provider import OllamaProvider


def test_generate_sends_model_prompt_and_returns_response(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        payload = json.loads(request.content)
        assert payload["model"] == "llama3.2"
        assert payload["prompt"] == "Answer from this context."
        assert payload["stream"] is False
        return httpx.Response(200, json={"response": "Grounded answer"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("llm.ollama_provider.httpx.Client", lambda timeout: client)
    provider = OllamaProvider(model="llama3.2")

    result = provider.generate("Answer from this context.")

    assert result.text == "Grounded answer"
    assert result.model == "llama3.2"
    assert result.generate_ms >= 0


def test_generate_converts_http_timeout(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("llm.ollama_provider.httpx.Client", lambda timeout: client)
    provider = OllamaProvider(timeout=5)

    with pytest.raises(TimeoutError, match="timed out"):
        provider.generate("Question")

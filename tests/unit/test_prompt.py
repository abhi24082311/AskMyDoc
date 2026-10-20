"""Unit tests for grounded prompt construction and answer generation."""

from __future__ import annotations

from llm.base import LLMProvider, LLMResponse
from llm.prompt import build_prompt, generate_answer
from retrieval.vector_store.base import SearchResult


def _result() -> SearchResult:
    return SearchResult(
        chunk_id="abc123",
        text="Retrieval finds relevant document passages.",
        source="guide.md",
        page=0,
        section_heading="Overview",
        score=0.9,
        rank=1,
    )


def test_prompt_includes_context_and_citation_identity() -> None:
    prompt = build_prompt("What does retrieval do?", [_result()])

    assert "Retrieval finds relevant document passages." in prompt
    assert "source: guide.md" in prompt
    assert "CHUNK abc123" in prompt
    assert "I don't know based on the provided documents." in prompt


def test_prompt_handles_empty_context() -> None:
    prompt = build_prompt("Question", [])

    assert "No relevant documents were retrieved." in prompt


def test_generate_answer_returns_provider_text_and_latency() -> None:
    class FakeProvider(LLMProvider):
        @property
        def model_name(self) -> str:
            return "fake"

        def generate(self, prompt: str) -> LLMResponse:
            assert "What does retrieval do?" in prompt
            return LLMResponse("It finds passages.", "fake", 12.5)

    answer, latency_ms = generate_answer("What does retrieval do?", [_result()], FakeProvider())

    assert answer == "It finds passages."
    assert latency_ms == 12.5

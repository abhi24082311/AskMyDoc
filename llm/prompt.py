"""Prompt builder and RAG answer generator.

The grounding prompt is the most critical piece of a RAG system. We enforce
three hard constraints via explicit prompt instructions:
1. Answer ONLY from the provided context.
2. Cite every factual claim with [source:page:chunk_id] markers.
3. Output "I don't know based on the provided documents" when context is
   insufficient — never hallucinate.

Design decision: We use a single-turn prompt (system + user combined) because
Ollama's /api/generate endpoint is stateless and keeps the provider interface
simple.
"""

from __future__ import annotations

import logging
import time

from llm.base import LLMProvider, LLMResponse
from retrieval.vector_store.base import SearchResult

logger = logging.getLogger(__name__)

# Separator used between context chunks in the prompt
_CHUNK_SEPARATOR = "\n---\n"


def build_prompt(query: str, results: list[SearchResult]) -> str:
    """Construct the grounded RAG prompt from a query and retrieved chunks.

    The prompt instructs the LLM to:
    - Use ONLY the provided context.
    - Inline-cite every claim with [source:page:chunk_id].
    - Admit ignorance when context is insufficient.

    Args:
        query: The user's natural-language question.
        results: Retrieved and optionally re-ranked context chunks.

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    if not results:
        context_section = "No relevant documents were retrieved."
    else:
        chunks = []
        for r in results:
            page_info = f"page {r.page}" if r.page > 0 else "no page"
            heading_info = f", section: {r.section_heading}" if r.section_heading else ""
            header = f"[CHUNK {r.chunk_id} | source: {r.source} | {page_info}{heading_info}]"
            chunks.append(f"{header}\n{r.text}")
        context_section = _CHUNK_SEPARATOR.join(chunks)

    prompt = f"""You are a precise, citation-focused document assistant. Your job is to answer
questions STRICTLY based on the provided context documents.

STRICT RULES:
1. Answer ONLY using information explicitly present in the context below.
2. After every factual claim, add an inline citation in this exact format: [source_file:page:chunk_id]
   Example: "The system uses cosine similarity for retrieval [docs/paper.pdf:3:a1b2c3d4]"
3. If the context does not contain sufficient information to answer the question, respond with
   exactly: "I don't know based on the provided documents."
4. Do NOT speculate, infer beyond what is stated, or use any external knowledge.
5. Be concise. Prefer one clear paragraph with citations over lengthy prose.

=== CONTEXT DOCUMENTS ===
{context_section}
=== END CONTEXT ===

QUESTION: {query}

ANSWER:"""

    return prompt


def generate_answer(
    query: str,
    results: list[SearchResult],
    llm: LLMProvider,
) -> tuple[str, float]:
    """Build the prompt, call the LLM, and return the answer with latency.

    Args:
        query: User question string.
        results: Retrieved context chunks.
        llm: Configured LLM provider instance.

    Returns:
        Tuple of (answer_text, generate_ms).
    """
    prompt = build_prompt(query, results)
    logger.debug("Prompt length: %d chars", len(prompt))

    response: LLMResponse = llm.generate(prompt)
    logger.info("LLM generate_ms=%.1f, answer_length=%d", response.generate_ms, len(response.text))
    return response.text, response.generate_ms

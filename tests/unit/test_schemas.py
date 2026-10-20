"""Unit tests for schemas (validation rules)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import AskRequest, Citation, IngestRequest


def test_ask_request_valid():
    req = AskRequest(question="What is RAG?")
    assert req.question == "What is RAG?"
    assert req.top_k is None


def test_ask_request_with_top_k():
    req = AskRequest(question="Q?", top_k=5)
    assert req.top_k == 5


def test_ask_request_empty_question_fails():
    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_ask_request_top_k_out_of_range():
    with pytest.raises(ValidationError):
        AskRequest(question="Q?", top_k=0)
    with pytest.raises(ValidationError):
        AskRequest(question="Q?", top_k=21)


def test_ingest_request_defaults():
    req = IngestRequest()
    assert req.docs_dir is None


def test_citation_snippet_field():
    c = Citation(
        source="paper.pdf",
        page=1,
        chunk_id="abc123",
        section_heading="Introduction",
        snippet="First 200 chars ...",
    )
    assert c.source == "paper.pdf"

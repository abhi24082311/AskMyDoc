"""Unit tests for TokenChunker."""

from __future__ import annotations

import pytest

from ingestion.chunker import TokenChunker, _make_chunk_id
from ingestion.loader import RawDocument


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def chunker() -> TokenChunker:
    return TokenChunker(chunk_size=100, overlap=15)


def _make_doc(text: str, source: str = "test.md", page: int = 0) -> RawDocument:
    return RawDocument(text=text, source=source, page=page, section_heading="Test Section")


# ─── chunk_id determinism ────────────────────────────────────────────────────

def test_chunk_id_is_deterministic():
    """Same inputs must always produce the same chunk ID."""
    id1 = _make_chunk_id("source.pdf", 1, 0, "hello world")
    id2 = _make_chunk_id("source.pdf", 1, 0, "hello world")
    assert id1 == id2


def test_chunk_id_differs_on_source():
    """Chunk IDs must differ when the source file changes."""
    id1 = _make_chunk_id("a.pdf", 1, 0, "text")
    id2 = _make_chunk_id("b.pdf", 1, 0, "text")
    assert id1 != id2


def test_chunk_id_differs_on_page():
    """Chunk IDs must differ when the page number changes."""
    id1 = _make_chunk_id("a.pdf", 1, 0, "text")
    id2 = _make_chunk_id("a.pdf", 2, 0, "text")
    assert id1 != id2


def test_chunk_id_is_16_hex_chars():
    """Chunk IDs must be 16 hexadecimal characters."""
    cid = _make_chunk_id("x.pdf", 0, 0, "hello")
    assert len(cid) == 16
    assert all(c in "0123456789abcdef" for c in cid)


# ─── TokenChunker behaviour ──────────────────────────────────────────────────

def test_empty_text_returns_no_chunks(chunker: TokenChunker):
    """Empty and whitespace-only text must produce zero chunks."""
    doc = _make_doc("")
    assert chunker.chunk_document(doc) == []

    doc_ws = _make_doc("   \n\t  ")
    assert chunker.chunk_document(doc_ws) == []


def test_short_text_produces_single_chunk(chunker: TokenChunker):
    """Text shorter than chunk_size must produce exactly one chunk."""
    doc = _make_doc("This is a short document.")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].source == "test.md"


def test_long_text_produces_multiple_chunks():
    """Text much longer than chunk_size must produce multiple chunks."""
    chunker = TokenChunker(chunk_size=50, overlap=10)
    # Generate ~500 tokens of text
    text = " ".join(["word"] * 500)
    doc = _make_doc(text)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1


def test_chunks_have_unique_ids(chunker: TokenChunker):
    """All chunks from a single document must have distinct IDs."""
    text = " ".join(["word"] * 500)
    chunker_small = TokenChunker(chunk_size=50, overlap=10)
    doc = _make_doc(text)
    chunks = chunker_small.chunk_document(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs are not unique"


def test_chunk_metadata_preserved(chunker: TokenChunker):
    """Each chunk must carry the source, page, and section_heading from the document."""
    doc = RawDocument(
        text="Some test content for metadata verification.",
        source="docs/test.pdf",
        page=3,
        section_heading="Methods",
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c.source == "docs/test.pdf"
    assert c.page == 3
    assert c.section_heading == "Methods"


def test_overlap_invalid_raises():
    """overlap >= chunk_size must raise ValueError."""
    with pytest.raises(ValueError, match="overlap"):
        TokenChunker(chunk_size=100, overlap=100)


def test_chunk_documents_aggregates(chunker: TokenChunker):
    """chunk_documents() must aggregate chunks across multiple documents."""
    docs = [_make_doc("Short text one."), _make_doc("Short text two.")]
    all_chunks = chunker.chunk_documents(docs)
    assert len(all_chunks) == 2  # one chunk each for short texts

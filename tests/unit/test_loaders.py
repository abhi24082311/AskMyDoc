"""Unit tests for document loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.loader import RawDocument, _clean_markdown, load_markdown


# ─── _clean_markdown ─────────────────────────────────────────────────────────

def test_clean_markdown_removes_code_fences():
    text = "Before\n```python\ncode here\n```\nAfter"
    result = _clean_markdown(text)
    assert "```" not in result
    assert "Before" in result
    assert "After" in result


def test_clean_markdown_removes_inline_code():
    text = "Use the `print()` function."
    result = _clean_markdown(text)
    assert "`" not in result
    assert "Use the" in result


def test_clean_markdown_keeps_link_text():
    text = "See [FastAPI docs](https://fastapi.tiangolo.com) for details."
    result = _clean_markdown(text)
    assert "FastAPI docs" in result
    assert "https://" not in result


def test_clean_markdown_removes_html():
    text = "Hello <b>world</b>!"
    result = _clean_markdown(text)
    assert "<b>" not in result
    assert "Hello" in result
    assert "world" in result


# ─── load_markdown ────────────────────────────────────────────────────────────

def test_load_markdown_sections(tmp_path: Path):
    """Markdown with headings produces one RawDocument per section."""
    md = tmp_path / "doc.md"
    md.write_text(
        "# Introduction\n\nThis is the intro.\n\n"
        "## Methods\n\nThis is methods.\n",
        encoding="utf-8",
    )
    docs = load_markdown(md, tmp_path)
    assert len(docs) == 2
    assert docs[0].section_heading == "Introduction"
    assert docs[1].section_heading == "Methods"
    assert "intro" in docs[0].text.lower()


def test_load_markdown_no_headings(tmp_path: Path):
    """Markdown without headings produces a single RawDocument."""
    md = tmp_path / "flat.md"
    md.write_text("Just a paragraph with no headings.", encoding="utf-8")
    docs = load_markdown(md, tmp_path)
    assert len(docs) == 1
    assert docs[0].section_heading == ""


def test_load_markdown_source_is_relative(tmp_path: Path):
    """source field must be a relative path, not an absolute one."""
    md = tmp_path / "sub" / "file.md"
    md.parent.mkdir()
    md.write_text("# Heading\nContent.", encoding="utf-8")
    docs = load_markdown(md, tmp_path)
    assert docs[0].source == str(Path("sub") / "file.md")
    assert not docs[0].source.startswith(str(tmp_path))


def test_load_markdown_empty_file(tmp_path: Path):
    """Empty Markdown file must produce no documents."""
    md = tmp_path / "empty.md"
    md.write_text("", encoding="utf-8")
    docs = load_markdown(md, tmp_path)
    assert docs == []


def test_load_markdown_page_is_zero(tmp_path: Path):
    """Markdown documents must have page=0 (no page concept)."""
    md = tmp_path / "doc.md"
    md.write_text("# H\ntext", encoding="utf-8")
    docs = load_markdown(md, tmp_path)
    assert all(d.page == 0 for d in docs)

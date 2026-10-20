"""Document loaders for PDF and Markdown files.

Design decision: We extract text with pypdf (pure-Python, no system deps) for
PDFs instead of pymupdf/pdfplumber to avoid C-library build requirements. For
Markdown we strip HTML/code fences and extract heading context per chunk so the
metadata field ``section_heading`` is always populated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    """A single page or logical section extracted from a source file.

    Attributes:
        text: Raw text content of the page/section.
        source: Relative path string of the source file (used in citations).
        page: 1-based page number (PDFs) or 0 when not applicable (Markdown).
        section_heading: Nearest heading above this content, empty if none.
    """

    text: str
    source: str
    page: int = 0
    section_heading: str = ""


def load_pdf(path: Path, docs_root: Path) -> list[RawDocument]:
    """Load a PDF and return one RawDocument per page with metadata.

    Args:
        path: Absolute path to the PDF file.
        docs_root: Root directory used to compute relative source paths.

    Returns:
        List of RawDocument objects, one per page that contains text.
    """
    try:
        import pypdf  # local import so the module can be tested without pypdf
    except ImportError:
        raise ImportError("pypdf is required for PDF loading. Run: pip install pypdf")

    source = str(path.relative_to(docs_root))
    docs: list[RawDocument] = []

    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as exc:
        logger.warning("Could not open PDF %s: %s", path, exc)
        return docs

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            logger.debug("PDF page %d of %s has no extractable text – skipping.", page_num, source)
            continue
        docs.append(
            RawDocument(
                text=text,
                source=source,
                page=page_num,
                section_heading="",  # PDFs don't reliably expose heading structure
            )
        )

    logger.info("Loaded PDF %s: %d pages with text", source, len(docs))
    return docs


def load_markdown(path: Path, docs_root: Path) -> list[RawDocument]:
    """Load a Markdown file and return one RawDocument per heading section.

    The file is split on ATX headings (# / ## / ###). Each section carries its
    nearest ancestor heading in ``section_heading``. If a file has no headings,
    the whole content is returned as one document.

    Args:
        path: Absolute path to the Markdown file.
        docs_root: Root directory used to compute relative source paths.

    Returns:
        List of RawDocument objects, one per section.
    """
    source = str(path.relative_to(docs_root))
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read Markdown %s: %s", path, exc)
        return []

    # Split on ATX headings (# Heading / ## Heading / etc.)
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []  # (heading_text, content_text)

    matches = list(heading_pattern.finditer(raw))
    if not matches:
        # No headings — treat entire file as one section
        text = _clean_markdown(raw).strip()
        if text:
            sections.append(("", text))
    else:
        # Content before first heading
        preamble = raw[: matches[0].start()].strip()
        if preamble:
            sections.append(("", _clean_markdown(preamble)))

        for i, match in enumerate(matches):
            heading_text = match.group(2).strip()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            body = raw[match.end() : end].strip()
            text = _clean_markdown(body).strip()
            if text:
                sections.append((heading_text, text))

    docs = [
        RawDocument(
            text=content,
            source=source,
            page=0,  # Markdown has no page concept
            section_heading=heading,
        )
        for heading, content in sections
        if content.strip()
    ]

    logger.info("Loaded Markdown %s: %d sections", source, len(docs))
    return docs


def _clean_markdown(text: str) -> str:
    """Strip code fences, HTML tags, and link syntax from Markdown text.

    We preserve the prose text for embedding. Code blocks are replaced with a
    single space so they don't contaminate chunk boundaries with noise tokens.

    Args:
        text: Raw Markdown string.

    Returns:
        Plain-text string with Markdown syntax removed.
    """
    # Remove fenced code blocks
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove images but keep alt text
    text = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", text)
    # Convert links to just their display text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_documents(docs_dir: Path) -> list[RawDocument]:
    """Recursively load all PDF and Markdown files from a directory.

    Args:
        docs_dir: Root directory to scan for documents.

    Returns:
        Flat list of RawDocument objects from all discovered files.
    """
    if not docs_dir.exists():
        logger.warning("docs_dir %s does not exist – returning empty list.", docs_dir)
        return []

    all_docs: list[RawDocument] = []
    pdf_files = list(docs_dir.rglob("*.pdf"))
    md_files = list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.markdown"))

    logger.info("Discovered %d PDF(s) and %d Markdown file(s).", len(pdf_files), len(md_files))

    for pdf_path in pdf_files:
        all_docs.extend(load_pdf(pdf_path, docs_dir))

    for md_path in md_files:
        all_docs.extend(load_markdown(md_path, docs_dir))

    return all_docs

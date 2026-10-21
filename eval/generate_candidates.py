"""Draft an unreviewed JSONL set of question/answer pairs using local Ollama."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import settings
from llm.factory import create_llm_provider
from retrieval.vector_store.chroma_store import ChromaVectorStore

logger = logging.getLogger("eval.candidates")


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract model-produced JSON array, tolerating a surrounding code fence."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("Ollama did not return a JSON array.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, list):
        raise ValueError("Candidate output must be a JSON array.")
    return [item for item in payload if isinstance(item, dict)]


def generate_candidates(count: int, output_path: Path) -> int:
    """Ask Ollama to draft grounded candidates from indexed chunks."""
    store = ChromaVectorStore(
        persist_dir=str(settings.chroma_persist_dir),
        collection_name=settings.chroma_collection_name,
    )
    chunks = store.get_all_chunks()
    if not chunks:
        raise RuntimeError("Chroma is empty. Ingest documents before drafting evaluation questions.")
    provider = create_llm_provider(settings)
    candidates: list[dict[str, Any]] = []
    batch_size = 5
    target = count

    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        source_text = "\n\n".join(
            f"[SOURCE {i + 1}: {item.source}, page {item.page}]\n{item.text}"
            for i, item in enumerate(batch)
        )
        prompt = f"""Create at most {min(batch_size, target - len(candidates))} useful evaluation question-answer pairs from the passages below.
Return only a JSON array. Each item must have exactly these fields:
{{"question":"...","ground_truth_answer":"...","ground_truth_sources":[{{"source":"filename","pages":[1]}}]}}
Questions must be answerable from the supplied passages. Answers must be concise and factual. Use the exact source filename and page shown. Do not invent facts or sources.

PASSAGES:
{source_text}
"""
        if len(candidates) >= target:
            break
        try:
            generated = _extract_json_array(provider.generate(prompt).text)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Skipping malformed candidate batch at chunk %d: %s", offset, exc)
            continue
        for item in generated:
            if all(key in item for key in ("question", "ground_truth_answer", "ground_truth_sources")):
                candidates.append(item)
                if len(candidates) >= target:
                    break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in candidates),
        encoding="utf-8",
    )
    logger.info("Wrote %d draft candidates to %s; review and copy accepted rows into golden.jsonl", len(candidates), output_path)
    return len(candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=settings.eval_candidate_count)
    parser.add_argument("--output", type=Path, default=Path("eval/candidates.jsonl"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    logging.basicConfig(level=settings.log_level)
    try:
        total = generate_candidates(args.count, args.output)
    except Exception as exc:
        logger.exception("Candidate generation failed: %s", exc)
        return 2
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())

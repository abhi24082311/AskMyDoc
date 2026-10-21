"""Shared dataset parsing and source-based retrieval metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_golden_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and validate JSONL questions used to score retrieval and answers."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        required = {"question", "ground_truth_answer", "ground_truth_sources"}
        if not required.issubset(record):
            raise ValueError(f"{path}:{line_number} must include {sorted(required)}")
        if not isinstance(record["question"], str) or not record["question"].strip():
            raise ValueError(f"{path}:{line_number} has an empty question")
        if not isinstance(record["ground_truth_answer"], str) or not record["ground_truth_answer"].strip():
            raise ValueError(f"{path}:{line_number} has an empty ground_truth_answer")
        sources = record["ground_truth_sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{path}:{line_number} needs at least one ground_truth_source")
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("source"), str):
                raise ValueError(f"{path}:{line_number} sources need a source filename")
            pages = source.get("pages", [])
            if not isinstance(pages, list) or any(not isinstance(page, int) for page in pages):
                raise ValueError(f"{path}:{line_number} source pages must be a list of integers")
        records.append(record)
    if not records:
        raise ValueError(f"No evaluation questions found in {path}")
    return records


def retrieval_metrics(
    results: list[Any],
    ground_truth_sources: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute recall@k and reciprocal rank from source filename/page labels.

    A retrieved chunk matches a gold source when the filename matches and,
    when gold pages are supplied, its page is in that page list.
    """
    relevant_total = sum(max(1, len(source.get("pages", []))) for source in ground_truth_sources)
    relevant_hits: set[tuple[str, int | None]] = set()
    first_relevant_rank: int | None = None

    for rank, result in enumerate(results, start=1):
        result_source = Path(str(result.source)).as_posix().casefold()
        for source in ground_truth_sources:
            gold_source = Path(source["source"]).as_posix().casefold()
            pages = source.get("pages", [])
            if result_source != gold_source:
                continue
            if pages:
                if result.page in pages:
                    relevant_hits.add((gold_source, result.page))
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
            else:
                relevant_hits.add((gold_source, None))
                if first_relevant_rank is None:
                    first_relevant_rank = rank

    return {
        "recall_at_k": len(relevant_hits) / relevant_total,
        "mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
    }

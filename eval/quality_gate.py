"""Fail evaluation when a metric falls below its configured minimum."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from app.config import settings

THRESHOLDS = {
    "faithfulness": settings.eval_min_faithfulness,
    "answer_relevancy": settings.eval_min_answer_relevancy,
    "context_precision": settings.eval_min_context_precision,
    "context_recall": settings.eval_min_context_recall,
    "recall_at_k": settings.eval_min_recall_at_k,
    "mrr": settings.eval_min_mrr,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=settings.eval_results_dir / "results.json")
    args = parser.parse_args()
    if not args.results.exists():
        print(f"Evaluation results not found: {args.results}", file=sys.stderr)
        return 2

    report = json.loads(args.results.read_text(encoding="utf-8"))
    failures: list[str] = []
    for mode, mode_data in report.get("modes", {}).items():
        metrics = mode_data.get("metrics", {})
        for name, threshold in THRESHOLDS.items():
            value = metrics.get(name)
            if value is None or not math.isfinite(float(value)) or float(value) < threshold:
                failures.append(f"{mode}.{name}: {value!r} < {threshold:.2f}")

    if failures:
        print("Quality gate failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Quality gate passed for all retrieval modes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

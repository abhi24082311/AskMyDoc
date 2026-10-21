"""Run the golden evaluation over vector, hybrid, and hybrid-rerank modes."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings

from app.config import settings
from eval.common import load_golden_dataset, retrieval_metrics
from llm.prompt import generate_answer
from retrieval.embedder import get_embedding_model
from retrieval.search_service import SearchService
from retrieval.vector_store.chroma_store import ChromaVectorStore

logger = logging.getLogger("eval")
MODES = ("vector_only", "hybrid", "hybrid_rerank")
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class OllamaStructuredEvaluator:
    """Ragas structured-output adapter that calls Ollama's native chat API."""

    def __init__(self, base_url: str, model: str, timeout: int, num_predict: int = 768) -> None:
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model
        self._timeout = timeout
        self._num_predict = num_predict
        self.is_async = False

    def _request_payload(self, prompt: str, response_model: type) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": response_model.model_json_schema(),
            "options": {
                "temperature": 0,
                "num_predict": self._num_predict,
                "repeat_penalty": 1.12,
            },
        }

    def generate(self, prompt: str, response_model: type) -> Any:
        """Generate and validate a response against Ragas' Pydantic schema."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._url, json=self._request_payload(prompt, response_model))
            response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        return response_model.model_validate_json(content)

    async def agenerate(self, prompt: str, response_model: type) -> Any:
        """Asynchronously generate and validate a schema-constrained response."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._url, json=self._request_payload(prompt, response_model))
            response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        return response_model.model_validate_json(content)


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(numeric) if numeric else None


def _evaluate_with_ragas(samples: list[dict[str, Any]], cfg: Any) -> dict[str, float | None]:
    """Use local Ollama and the project's embeddings as Ragas judges."""
    # Disable Ragas usage telemetry; evaluation inputs stay local to this project.
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms.base import InstructorBaseRagasLLM
        from ragas.metrics._faithfulness import StatementGeneratorPrompt
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation dependencies are missing. Install the project requirements with "
            "'.venv\\Scripts\\python.exe -m pip install -r requirements.txt'."
        ) from exc

    class RagasOllamaStructuredEvaluator(OllamaStructuredEvaluator, InstructorBaseRagasLLM):
        """Combine the native Ollama client with Ragas' structured LLM contract."""

    evaluator_llm = RagasOllamaStructuredEvaluator(
        cfg.ollama_base_url,
        cfg.eval_ollama_model,
        cfg.eval_llm_timeout_seconds,
        num_predict=cfg.eval_max_tokens,
    )
    embedding_model = get_embedding_model(cfg.embedding_model_name)

    class ProjectEmbeddingAdapter(Embeddings):
        """LangChain adapter backed by the project's local embedding model."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embedding_model.encode(texts)

        def embed_query(self, text: str) -> list[float]:
            return embedding_model.encode_single(text)

    evaluator_embeddings = LangchainEmbeddingsWrapper(ProjectEmbeddingAdapter())

    class BoundedStatementGeneratorPrompt(StatementGeneratorPrompt):
        """Keep a small local model's faithfulness output short and non-repetitive."""

        instruction = (
            "Read the answer and extract its factual claims as concise, independently "
            "understandable statements. Return at most 5 statements, with one statement "
            "per distinct claim. Do not repeat or expand claims. Do not add facts. "
            "Return an empty list when the answer has no factual claims."
        )

    ragas_dataset = Dataset.from_list(
        [
            {
                "user_input": sample["question"],
                "response": sample["answer"],
                "retrieved_contexts": sample["contexts"],
                "reference": sample["ground_truth"],
            }
            for sample in samples
        ]
    )
    metric_objects = [
        Faithfulness(
            llm=evaluator_llm,
            statement_generator_prompt=BoundedStatementGeneratorPrompt(),
        ),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1),
        LLMContextPrecisionWithReference(llm=evaluator_llm, name="context_precision"),
        LLMContextRecall(llm=evaluator_llm),
    ]
    scored = evaluate(
        dataset=ragas_dataset,
        metrics=metric_objects,
        run_config=RunConfig(
            max_workers=1,
            timeout=cfg.eval_llm_timeout_seconds,
            max_retries=2,
            max_wait=10,
        ),
        raise_exceptions=True,
        show_progress=True,
    )
    frame = scored.to_pandas()
    return {name: _mean(frame[name].tolist()) if name in frame else None for name in RAGAS_METRICS}


def _write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    columns = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "recall_at_k", "mrr"]
    lines = [
        "# Retrieval Evaluation Results",
        "",
        f"Generated: {report['generated_at']}",
        f"Questions: {report['question_count']} · Ollama model: `{report['ollama_model']}` · top-k: {report['top_k']}",
        "",
        "| Retrieval mode | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for mode in MODES:
        metrics = report["modes"].get(mode, {}).get("metrics", {})
        values = [metrics.get(column) for column in columns]
        display = ["N/A" if value is None else f"{float(value):.3f}" for value in values]
        lines.append(f"| `{mode}` | " + " | ".join(display) + " |")
    lines.extend(
        [
            "",
            "Ragas scores use the configured local Ollama model as the judge. Retrieval metrics compare the retrieved source/page pairs with `ground_truth_sources` in `golden.jsonl`.",
            "",
        ]
    )
    markdown_path = output_dir / "results.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s and %s", results_path, markdown_path)


def run_evaluation(dataset_path: Path, output_dir: Path, limit: int | None = None) -> dict[str, Any]:
    """Retrieve and answer every golden question under each configured mode."""
    golden = load_golden_dataset(dataset_path)
    if limit is not None:
        golden = golden[:limit]
    if not golden:
        raise ValueError("The evaluation limit selected zero questions.")

    store = ChromaVectorStore(
        persist_dir=str(settings.chroma_persist_dir),
        collection_name=settings.chroma_collection_name,
    )
    if store.count() == 0:
        raise RuntimeError("Chroma is empty. Ingest the documents before running evaluation.")
    embedder = get_embedding_model(settings.embedding_model_name)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "question_count": len(golden),
        "ollama_model": settings.ollama_model,
        "evaluator_model": settings.eval_ollama_model,
        "top_k": settings.top_k,
        "modes": {},
    }
    for mode in MODES:
        logger.info("Evaluating mode %s (%d questions)", mode, len(golden))
        mode_settings = settings.model_copy(update={"retrieval_mode": mode})
        search = SearchService(store, embedder, mode_settings)
        ragas_samples: list[dict[str, Any]] = []
        retrieval_scores: list[dict[str, float]] = []
        per_question: list[dict[str, Any]] = []

        # A fresh provider is created per mode; the LLM answer cache is bypassed.
        from llm.factory import create_llm_provider

        llm = create_llm_provider(settings)
        for case in golden:
            results, _, _ = search.search(case["question"])
            answer = (
                generate_answer(case["question"], results, llm)[0]
                if results
                else "I don't know based on the provided documents."
            )
            scores = retrieval_metrics(results, case["ground_truth_sources"])
            retrieval_scores.append(scores)
            per_question.append(
                {
                    "question": case["question"],
                    "recall_at_k": scores["recall_at_k"],
                    "mrr": scores["mrr"],
                    "retrieved_sources": [
                        {"source": item.source, "page": item.page, "chunk_id": item.chunk_id}
                        for item in results
                    ],
                }
            )
            ragas_samples.append(
                {
                    "question": case["question"],
                    "answer": answer,
                    "contexts": [item.text for item in results] or ["No context retrieved."],
                    "ground_truth": case["ground_truth_answer"],
                }
            )
        metric_scores = _evaluate_with_ragas(ragas_samples, settings)
        metric_scores["recall_at_k"] = _mean([score["recall_at_k"] for score in retrieval_scores])
        metric_scores["mrr"] = _mean([score["mrr"] for score in retrieval_scores])
        report["modes"][mode] = {"metrics": metric_scores, "questions": per_question}
        logger.info("%s metrics: %s", mode, metric_scores)

    _write_report(report, output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=settings.eval_golden_path)
    parser.add_argument("--output-dir", type=Path, default=settings.eval_results_dir)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions (useful in CI).")
    args = parser.parse_args()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s – %(message)s")
    try:
        run_evaluation(args.dataset, args.output_dir, args.limit)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

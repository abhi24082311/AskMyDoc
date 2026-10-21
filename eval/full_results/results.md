# Retrieval Evaluation Results

Generated: 2026-10-21T21:21:14.415027+00:00
Questions: 10 · Ollama model: `llama3.2` · top-k: 4

| Retrieval mode | faithfulness | answer_relevancy | context_precision | context_recall | recall_at_k | mrr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vector_only` | 0.620 | 0.860 | 0.883 | 0.617 | 1.000 | 0.767 |
| `hybrid` | 0.850 | 0.888 | 0.867 | 0.583 | 1.000 | 0.933 |
| `hybrid_rerank` | 0.715 | 0.931 | 0.950 | 0.457 | 0.900 | 0.667 |

Ragas scores use the configured local Ollama model as the judge. Retrieval metrics compare the retrieved source/page pairs with `ground_truth_sources` in `golden.jsonl`.

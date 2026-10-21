"""Persistent lexical search over Chroma chunks using rank_bm25."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path

from retrieval.vector_store.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_FORMAT_VERSION = 1
_EMPTY_INDEX = object()


def tokenize(text: str) -> list[str]:
    """Normalize text into lowercase Unicode word tokens."""
    return _TOKEN_PATTERN.findall(text.casefold())


class BM25Index:
    """Build or load a JSON-persisted BM25Okapi index for stored chunks.

    The cache stores BM25's scoring statistics as JSON rather than using
    pickle, so loading an index file cannot execute code. A content signature
    triggers a rebuild when indexed text changes.
    """

    def __init__(self, vector_store: VectorStore, index_dir: Path) -> None:
        self._store = vector_store
        self._cache_path = index_dir / "bm25_index.json"
        self._lock = threading.RLock()
        self._records: list[SearchResult] = []
        self._bm25 = None
        self._signature = ""
        self.refresh()

    def refresh(self) -> bool:
        """Synchronize the lexical index with Chroma; return whether it changed."""
        with self._lock:
            records = sorted(self._store.get_all_chunks(), key=lambda item: item.chunk_id)
            indexed = [(record, tokenize(record.text)) for record in records]
            indexed = [(record, tokens) for record, tokens in indexed if tokens]
            ids = [record.chunk_id for record, _ in indexed]
            signature = _content_signature(indexed)

            if signature == self._signature:
                self._records = [record for record, _ in indexed]
                return False

            self._records = [record for record, _ in indexed]
            cached = self._read_cache(signature, ids)
            if cached is _EMPTY_INDEX:
                self._bm25 = None
                self._signature = signature
                return False
            if cached is not None:
                self._bm25 = cached
                self._signature = signature
                logger.info("Loaded persisted BM25 index for %d chunks", len(ids))
                return False

            self._bm25 = self._build_index([tokens for _, tokens in indexed])
            self._signature = signature
            self._write_cache(signature, ids, self._bm25)
            logger.info("Built BM25 index for %d chunks", len(ids))
            return True

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return the highest-scoring lexical matches for a query."""
        query_tokens = tokenize(query)
        with self._lock:
            if self._bm25 is None or not query_tokens:
                return []
            scores = self._bm25.get_scores(query_tokens)
            ranked = sorted(
                enumerate(scores),
                key=lambda pair: (-float(pair[1]), self._records[pair[0]].chunk_id),
            )
            results: list[SearchResult] = []
            for index, raw_score in ranked:
                score = float(raw_score)
                if not any(query_token in self._bm25.doc_freqs[index] for query_token in query_tokens):
                    continue
                record = self._records[index]
                results.append(
                    SearchResult(
                        chunk_id=record.chunk_id,
                        text=record.text,
                        source=record.source,
                        page=record.page,
                        section_heading=record.section_heading,
                        score=score,
                        rank=len(results) + 1,
                    )
                )
                if len(results) >= top_k:
                    break
            return results

    @staticmethod
    def _build_index(tokenized_documents: list[list[str]]):
        if not tokenized_documents:
            return None
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("rank-bm25 is required for hybrid retrieval. Install requirements.txt") from exc
        return BM25Okapi(tokenized_documents)

    def _read_cache(self, signature: str, ids: list[str]):
        try:
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if (
                payload.get("format_version") != _FORMAT_VERSION
                or payload.get("signature") != signature
                or payload.get("chunk_ids") != ids
            ):
                return None
            if payload.get("empty") and not ids:
                return _EMPTY_INDEX

            from rank_bm25 import BM25Okapi

            index = BM25Okapi.__new__(BM25Okapi)
            index.k1 = float(payload["k1"])
            index.b = float(payload["b"])
            index.epsilon = float(payload["epsilon"])
            index.corpus_size = int(payload["corpus_size"])
            index.avgdl = float(payload["avgdl"])
            index.doc_len = [int(value) for value in payload["doc_len"]]
            index.doc_freqs = [
                {str(term): int(freq) for term, freq in frequencies.items()}
                for frequencies in payload["doc_freqs"]
            ]
            index.idf = {str(term): float(value) for term, value in payload["idf"].items()}
            index.average_idf = float(payload["average_idf"])
            index.tokenizer = None
            if (
                index.corpus_size != len(ids)
                or len(index.doc_freqs) != len(ids)
                or len(index.doc_len) != len(ids)
            ):
                return None
            return index
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError):
            logger.warning("BM25 cache is invalid; rebuilding it")
            return None

    def _write_cache(self, signature: str, ids: list[str], index) -> None:
        if index is None:
            payload = {
                "format_version": _FORMAT_VERSION,
                "signature": signature,
                "chunk_ids": ids,
                "empty": True,
            }
        else:
            payload = {
                "format_version": _FORMAT_VERSION,
                "signature": signature,
                "chunk_ids": ids,
                "corpus_size": index.corpus_size,
                "avgdl": index.avgdl,
                "doc_len": index.doc_len,
                "doc_freqs": index.doc_freqs,
                "idf": index.idf,
                "average_idf": index.average_idf,
                "k1": index.k1,
                "b": index.b,
                "epsilon": index.epsilon,
            }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._cache_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temp_path.replace(self._cache_path)


def _content_signature(indexed: list[tuple[SearchResult, list[str]]]) -> str:
    digest = hashlib.sha256()
    for record, _tokens in indexed:
        digest.update(record.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()

"""BM25 sparse vectors with corpus DF statistics (same role as reference Bm25SparseVectorizer)."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Iterable

from app.text_tokenizer import TextTokenizer


class Bm25SparseVectorizer:
    def __init__(
        self,
        storage_path: str | None,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        self.storage_path = storage_path
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or TextTokenizer()
        self._lock = threading.Lock()
        self._df: dict[int, int] = {}
        self._total_documents = 0
        self._total_document_length = 0
        self._load()

    def _load(self) -> None:
        if not self.storage_path:
            return
        path = Path(self.storage_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._total_documents = int(data.get("total_documents", 0))
            self._total_document_length = int(data.get("total_document_length", 0))
            self._df = {int(k): int(v) for k, v in data.get("frequencies", {}).items()}
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    def save(self) -> None:
        if not self.storage_path:
            return
        path = Path(self.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "total_documents": self._total_documents,
                "total_document_length": self._total_document_length,
                "frequencies": {str(k): v for k, v in self._df.items()},
            }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def add_document(self, text: str) -> None:
        tf, _token_count = self.tokenizer.get_term_frequencies(text)
        if not tf:
            return
        unique = set(tf.keys())
        with self._lock:
            self._total_documents += 1
            self._total_document_length += sum(tf.values())
            for h in unique:
                self._df[h] = self._df.get(h, 0) + 1

    def add_documents(self, texts: Iterable[str]) -> None:
        for t in texts:
            self.add_document(t)
        self.save()

    def clear(self) -> None:
        with self._lock:
            self._df.clear()
            self._total_documents = 0
            self._total_document_length = 0
        self.save()

    @property
    def average_document_length(self) -> float:
        if self._total_documents <= 0:
            return 1.0
        return self._total_document_length / self._total_documents

    def compute_sparse_vector(self, text: str) -> tuple[list[int], list[float]]:
        term_frequencies, token_count = self.tokenizer.get_term_frequencies(text)
        if token_count == 0:
            return [], []
        avg_doc_len = self.average_document_length
        total_docs = max(1, self._total_documents)
        sparse: dict[int, float] = {}
        with self._lock:
            df_snapshot = dict(self._df)
        for index, tf in term_frequencies.items():
            df = df_snapshot.get(index, 0)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * (token_count / avg_doc_len))
            )
            sparse[index] = float(idf * tf_norm)
        ordered = sorted(sparse.items(), key=lambda x: x[0])
        return [k for k, _ in ordered], [v for _, v in ordered]

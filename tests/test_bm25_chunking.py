"""Regression: BM25 sparse vectorizer + chunking invariants."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.bm25 import Bm25SparseVectorizer
from app.chunking import ChunkingService
from app.text_tokenizer import TextTokenizer, hash_term_to_index


def test_hash_term_stable():
    h1 = hash_term_to_index("kubernetes")
    h2 = hash_term_to_index("kubernetes")
    assert h1 == h2


def test_bm25_sparse_non_empty_after_corpus():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "df.json"
        b = Bm25SparseVectorizer(str(p))
        b.add_documents(["hello world", "world of warcraft"])
        idx, vals = b.compute_sparse_vector("hello world")
        assert len(idx) == len(vals) and len(idx) > 0


def test_chunking_splits_long_text():
    ch = ChunkingService()
    long_text = "tokenbulk " * 8000
    sections = [{"text": long_text, "section_path": "", "section": "s", "start_page": 0, "end_page": 0}]
    chunks = ch.sections_to_chunks("doc.txt", sections)
    assert len(chunks) >= 2
    assert all(c.chunk_total == len(chunks) for c in chunks)

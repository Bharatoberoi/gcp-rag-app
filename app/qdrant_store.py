"""Qdrant hybrid store: dense + BM25 sparse with RRF fusion (reference architecture)."""

from __future__ import annotations

import json
from typing import Any

from qdrant_client import QdrantClient, models

from app.bm25 import Bm25SparseVectorizer
from app.config import settings
from app.schemas import DocumentChunk


DENSE = "dense"
SPARSE = "sparse"


def _payload(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "chunkId": chunk.id,
        "text": chunk.text,
        "sourceDocument": chunk.source_document,
        "chunkIndex": chunk.chunk_index,
        "chunkTotal": chunk.chunk_total,
        "startPage": chunk.start_page,
        "endPage": chunk.end_page,
        "section": chunk.section,
        "sectionPath": chunk.section_path,
        "metadata": json.dumps(chunk.metadata),
    }


def _from_payload(payload: dict[str, Any]) -> DocumentChunk:
    meta_raw = payload.get("metadata") or "{}"
    try:
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
    except json.JSONDecodeError:
        meta = {}
    return DocumentChunk(
        id=str(payload.get("chunkId", "")),
        text=str(payload.get("text", "")),
        source_document=str(payload.get("sourceDocument", "")),
        chunk_index=int(payload.get("chunkIndex", 0)),
        chunk_total=int(payload.get("chunkTotal", 0)),
        start_page=int(payload.get("startPage", 0)),
        end_page=int(payload.get("endPage", 0)),
        section=str(payload.get("section", "")),
        section_path=str(payload.get("sectionPath", "")),
        metadata={str(k): str(v) for k, v in meta.items()},
    )


class HybridQdrantStore:
    def __init__(self) -> None:
        url = (settings.qdrant_url or "").strip()
        if url.lower() in {"memory", ":memory:", "inmem"}:
            self.client = QdrantClient(":memory:")
        else:
            kwargs: dict[str, Any] = {"url": url}
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key
            self.client = QdrantClient(**kwargs)
        self.collection = settings.qdrant_collection
        self.vector_size = settings.embedding_dimensions
        self.dense_weight = settings.dense_vector_weight
        self.sparse_weight = max(0.0, 1.0 - self.dense_weight)
        self.bm25 = Bm25SparseVectorizer(settings.bm25_state_path)
        self._collection_initialized = False

    def _ensure_collection(self) -> None:
        if self._collection_initialized:
            return
        cols = self.client.get_collections().collections
        names = {c.name for c in cols}
        if self.collection in names:
            self._collection_initialized = True
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams()},
        )
        indexes = [
            ("sourceDocument", models.PayloadSchemaType.KEYWORD),
            ("sectionPath", models.PayloadSchemaType.KEYWORD),
            ("chunkIndex", models.PayloadSchemaType.INTEGER),
        ]
        for field, schema in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass
        self._collection_initialized = True

    def upsert_chunks(self, chunks: list[DocumentChunk], dense_vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._ensure_collection()
        self.bm25.add_documents(c.embedding_input() for c in chunks)
        points: list[models.PointStruct] = []
        for chunk, vec in zip(chunks, dense_vectors, strict=True):
            idx, vals = self.bm25.compute_sparse_vector(chunk.embedding_input())
            points.append(
                models.PointStruct(
                    id=chunk.id,
                    vector={
                        DENSE: vec,
                        SPARSE: models.SparseVector(indices=idx, values=vals),
                    },
                    payload=_payload(chunk),
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def delete_by_source(self, source_document: str) -> None:
        self._ensure_collection()
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="sourceDocument",
                            match=models.MatchValue(value=source_document),
                        )
                    ]
                )
            ),
        )

    def _prefetch_limits(self, top_k: int) -> tuple[int, int]:
        total_w = self.dense_weight + self.sparse_weight
        if total_w <= 0:
            total_w = 1.0
        dense_prefetch = max(top_k, int(top_k * 4 * (self.dense_weight / total_w)))
        sparse_prefetch = max(top_k, int(top_k * 4 * (self.sparse_weight / total_w)))
        return dense_prefetch, sparse_prefetch

    def search_hybrid(self, query_text: str, query_dense: list[float], top_k: int) -> list[DocumentChunk]:
        self._ensure_collection()
        idx, vals = self.bm25.compute_sparse_vector(query_text)
        dlim, slim = self._prefetch_limits(top_k)
        res = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(query=query_dense, using=DENSE, limit=dlim),
                models.Prefetch(
                    query=models.SparseVector(indices=idx, values=vals),
                    using=SPARSE,
                    limit=slim,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        chunks: list[DocumentChunk] = []
        for p in res.points:
            if p.payload:
                chunks.append(_from_payload(p.payload))
        return chunks

    def scroll_chunk(self, source_document: str, chunk_index: int) -> DocumentChunk | None:
        self._ensure_collection()
        must = [
            models.FieldCondition(key="sourceDocument", match=models.MatchValue(value=source_document)),
            models.FieldCondition(key="chunkIndex", match=models.MatchValue(value=chunk_index)),
        ]
        res, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(must=must),
            limit=1,
            with_payload=True,
        )
        if not res:
            return None
        pl = res[0].payload or {}
        return _from_payload(pl)

    def expand_adjacent(self, results: list[DocumentChunk], adjacent: int) -> list[DocumentChunk]:
        if adjacent <= 0 or not results:
            return results
        have = {(c.source_document, c.chunk_index) for c in results}
        extra: list[DocumentChunk] = []
        for c in results:
            for off in range(-adjacent, adjacent + 1):
                if off == 0:
                    continue
                ni = c.chunk_index + off
                if ni < 0 or ni >= c.chunk_total:
                    continue
                key = (c.source_document, ni)
                if key in have:
                    continue
                adj = self.scroll_chunk(c.source_document, ni)
                if adj:
                    extra.append(adj)
                    have.add(key)
        return results + extra

"""HTTP cross-encoder reranker (same API as reference reranker service)."""

from __future__ import annotations

import httpx

from app.config import settings
from app.schemas import DocumentChunk


async def rerank_chunks(query: str, chunks: list[DocumentChunk], top_k: int) -> list[DocumentChunk]:
    if not settings.reranker_enabled or not settings.reranker_url or not chunks:
        return chunks[:top_k]
    docs = [c.text for c in chunks]
    url = settings.reranker_url.rstrip("/") + "/rerank"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json={"query": query, "documents": docs})
            r.raise_for_status()
            data = r.json()
            scores = data.get("scores") or []
    except Exception:
        return chunks[:top_k]
    if len(scores) != len(chunks):
        return chunks[:top_k]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in order[:top_k]]

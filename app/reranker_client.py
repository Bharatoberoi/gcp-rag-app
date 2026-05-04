"""Reranking: score and re-order retrieved chunks by relevance.

Supports two modes:
1. Built-in Gemini reranker (free, no external service needed)
2. External HTTP cross-encoder (e.g., Cohere Rerank, BGE)
"""

from __future__ import annotations

import json

import httpx

from app.config import settings
from app.schemas import DocumentChunk

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


async def rerank_chunks(query: str, chunks: list[DocumentChunk], top_k: int) -> list[DocumentChunk]:
    if not chunks:
        return []
    if not settings.reranker_enabled:
        return chunks[:top_k]
    if settings.reranker_url:
        return await _rerank_external(query, chunks, top_k)
    return _rerank_with_gemini(query, chunks, top_k)


async def _rerank_external(query: str, chunks: list[DocumentChunk], top_k: int) -> list[DocumentChunk]:
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


def _rerank_with_gemini(query: str, chunks: list[DocumentChunk], top_k: int) -> list[DocumentChunk]:
    """Use Gemini to score relevance of each chunk to the query."""
    if not settings.gemini_api_key:
        return chunks[:top_k]

    passages = ""
    for i, c in enumerate(chunks):
        excerpt = c.text[:500]
        passages += f"[{i}] {excerpt}\n\n"

    prompt = (
        "You are a relevance scoring system. Given a query and numbered passages, "
        "rate each passage's relevance to the query on a scale of 1-10.\n"
        "Return ONLY a JSON array of objects with 'index' and 'score' fields.\n\n"
        f"Query: {query}\n\n"
        f"Passages:\n{passages}\n"
        'Output format: [{"index": 0, "score": 8}, ...]\n'
        "JSON array:"
    )

    url = f"{BASE_URL}/models/{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024},
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return chunks[:top_k]
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        text = text.replace("```json", "").replace("```", "").strip()
        scores = json.loads(text)
        score_map = {item["index"]: item["score"] for item in scores}
        indexed = [(i, score_map.get(i, 0)) for i in range(len(chunks))]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [chunks[i] for i, _ in indexed[:top_k]]
    except Exception:
        return chunks[:top_k]

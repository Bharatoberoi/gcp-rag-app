"""Reranking: score and re-order retrieved chunks by relevance.

Top-k retrieval returns 10-20 candidates. A reranker then picks the best 3-5.
This alone can double answer quality because embedding similarity is a rough
proxy for relevance — reranking applies deeper semantic understanding.

This module supports two modes:
1. Built-in Gemini reranker (free, no external service needed)
2. External HTTP cross-encoder (e.g., Cohere Rerank, BGE) for production
"""

from __future__ import annotations

import json

import google.generativeai as genai
import httpx

from app.config import settings
from app.schemas import DocumentChunk


async def rerank_chunks(query: str, chunks: list[DocumentChunk], top_k: int) -> list[DocumentChunk]:
    """Rerank chunks using the configured method."""
    if not chunks:
        return []
    if not settings.reranker_enabled:
        return chunks[:top_k]

    if settings.reranker_url:
        return await _rerank_external(query, chunks, top_k)
    return _rerank_with_gemini(query, chunks, top_k)


async def _rerank_external(query: str, chunks: list[DocumentChunk], top_k: int) -> list[DocumentChunk]:
    """Rerank using an external HTTP cross-encoder service."""
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
    """Use Gemini to score relevance of each chunk to the query.

    Asks the LLM to rate each passage's relevance on a 1-10 scale,
    then sorts by score. This is slower than a dedicated cross-encoder
    but works without any additional service.
    """
    if not settings.gemini_api_key:
        return chunks[:top_k]

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model)

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
        "Output format: [{\"index\": 0, \"score\": 8}, ...]\n"
        "JSON array:"
    )

    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=1024),
        )
        text = (resp.text or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        scores = json.loads(text)
        score_map = {item["index"]: item["score"] for item in scores}
        indexed = [(i, score_map.get(i, 0)) for i in range(len(chunks))]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [chunks[i] for i, _ in indexed[:top_k]]
    except Exception:
        return chunks[:top_k]

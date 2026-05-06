"""Hybrid client: local embeddings (fastembed/ONNX) + Groq for generation.

Embeddings run locally with no API key needed. Groq is only called once per
query for answer generation. Groq free tier: 30 RPM, 14,400 RPD.
"""

from __future__ import annotations

from typing import Sequence

import httpx

from app.config import settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir="/app/models")
    return _embed_model


class GeminiClient:
    """Despite the class name, now uses Groq for generation and fastembed for embeddings."""

    def __init__(self) -> None:
        self._ready = False

    def init(self) -> None:
        if self._ready:
            return
        _get_embed_model()
        self._ready = True

    async def embed_texts_async(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts locally using fastembed ONNX (no API key needed)."""
        model = _get_embed_model()
        embeddings = list(model.embed(list(texts)))
        return [emb.tolist() for emb in embeddings]

    async def generate_answer_async(self, system_prompt: str, user_prompt: str) -> str:
        """Generate answer using Groq API (single call per query)."""
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required for answer generation")
        body = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(GROQ_URL, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "No response from model."
        return choices[0].get("message", {}).get("content", "").strip() or "No text in model response."

    async def generate_short_async(self, prompt: str, temperature: float = 0.1, max_tokens: int = 300) -> str:
        """Generate a short response for query planning, rewriting, etc."""
        if not settings.groq_api_key:
            return ""
        body = {
            "model": settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(GROQ_URL, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "").strip()

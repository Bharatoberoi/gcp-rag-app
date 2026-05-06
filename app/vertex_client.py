"""Hybrid client: local embeddings (sentence-transformers) + Gemini for generation.

Embeddings run locally with no API key needed. Gemini is only called once per
query for answer generation, minimizing rate limit issues.
"""

from __future__ import annotations

from typing import Sequence

import httpx
from sentence_transformers import SentenceTransformer

from app.config import settings

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        import os
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/app/models")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


class GeminiClient:
    def __init__(self) -> None:
        self._ready = False

    def init(self) -> None:
        if self._ready:
            return
        _get_embed_model()
        self._ready = True

    async def embed_texts_async(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts locally using sentence-transformers (no API key needed)."""
        model = _get_embed_model()
        embeddings = model.encode(list(texts), normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    async def generate_answer_async(self, system_prompt: str, user_prompt: str) -> str:
        """Generate answer using Gemini API (single call per query)."""
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for answer generation")
        url = f"{BASE_URL}/models/{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
        body = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
        }
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "No response from model."
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p["text"] for p in parts if "text" in p]
        return "\n".join(texts).strip() or "No text in model response."

    async def generate_short_async(self, prompt: str, temperature: float = 0.1, max_tokens: int = 300) -> str:
        """Generate a short response for query planning, rewriting, etc."""
        if not settings.gemini_api_key:
            return ""
        url = f"{BASE_URL}/models/{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()

"""Hybrid client: lightweight local embeddings + Groq for generation.

Embeddings use a hashing vectorizer (zero memory, no model download).
BM25 sparse search handles semantic matching; dense vectors provide diversity.
Groq is called once per query for answer generation.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import httpx
import numpy as np

from app.config import settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DIMS = 384
_WORD_SPLIT = re.compile(r"\w+", re.UNICODE)


def _hash_embed(text: str) -> list[float]:
    """Create a dense vector via feature hashing (hashing trick).

    Each word is hashed to multiple bucket indices in a fixed-size vector.
    This is memory-free (no model) and deterministic.
    """
    vec = np.zeros(DIMS, dtype=np.float32)
    words = _WORD_SPLIT.findall(text.lower())
    if not words:
        return vec.tolist()

    for word in words:
        h1 = int(hashlib.md5(word.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(word.encode()).hexdigest(), 16)
        idx1 = h1 % DIMS
        idx2 = h2 % DIMS
        sign1 = 1.0 if (h1 // DIMS) % 2 == 0 else -1.0
        sign2 = 1.0 if (h2 // DIMS) % 2 == 0 else -1.0
        vec[idx1] += sign1
        vec[idx2] += sign2

    # Also hash bigrams for better semantic capture
    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
        idx = h % DIMS
        sign = 1.0 if (h // DIMS) % 2 == 0 else -1.0
        vec[idx] += sign * 0.5

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


class GeminiClient:
    """Uses hashing vectorizer for embeddings and Groq for generation."""

    def __init__(self) -> None:
        self._ready = False

    def init(self) -> None:
        self._ready = True

    async def embed_texts_async(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts using feature hashing (instant, zero memory)."""
        return [_hash_embed(t) for t in texts]

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

"""Gemini API client using async httpx for all calls."""

from __future__ import annotations

from typing import Sequence

import httpx

from app.config import settings

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiClient:
    def __init__(self) -> None:
        self._ready = False

    def init(self) -> None:
        if self._ready:
            return
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required")
        self._ready = True

    def _url(self, model: str, method: str) -> str:
        return f"{BASE_URL}/models/{model}:{method}?key={settings.gemini_api_key}"

    async def embed_texts_async(self, texts: Sequence[str]) -> list[list[float]]:
        self.init()
        out: list[list[float]] = []
        batch_size = 16
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i : i + batch_size])
                requests = [
                    {"model": f"models/{settings.embedding_model}", "content": {"parts": [{"text": t}]}}
                    for t in batch
                ]
                url = f"{BASE_URL}/models/{settings.embedding_model}:batchEmbedContents?key={settings.gemini_api_key}"
                resp = await client.post(url, json={"requests": requests})
                resp.raise_for_status()
                data = resp.json()
                for emb in data["embeddings"]:
                    out.append(emb["values"])
        return out

    async def generate_answer_async(self, system_prompt: str, user_prompt: str) -> str:
        self.init()
        url = self._url(settings.gemini_model, "generateContent")
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
        """Generate a short response (for query rewriting, planning, etc.)."""
        self.init()
        url = self._url(settings.gemini_model, "generateContent")
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

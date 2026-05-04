"""Gemini API client using direct REST calls (no SDK version issues)."""

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

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.init()
        out: list[list[float]] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            requests = [
                {"model": f"models/{settings.embedding_model}", "content": {"parts": [{"text": t}]}}
                for t in batch
            ]
            url = f"{BASE_URL}/models/{settings.embedding_model}:batchEmbedContents?key={settings.gemini_api_key}"
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json={"requests": requests})
                resp.raise_for_status()
                data = resp.json()
            for emb in data["embeddings"]:
                out.append(emb["values"])
        return out

    def embed_query(self, text: str) -> list[float]:
        self.init()
        url = self._url(settings.embedding_model, "embedContent")
        body = {
            "model": f"models/{settings.embedding_model}",
            "content": {"parts": [{"text": text}]},
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
        return data["embedding"]["values"]

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        self.init()
        url = self._url(settings.gemini_model, "generateContent")
        body = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "No response from model."
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p["text"] for p in parts if "text" in p]
        return "\n".join(texts).strip() or "No text in model response."

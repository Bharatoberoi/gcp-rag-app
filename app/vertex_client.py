"""Gemini API embeddings and generation (free tier via AI Studio)."""

from __future__ import annotations

from typing import Sequence

import google.generativeai as genai

from app.config import settings


class VertexRagClient:
    """Uses the Gemini API (google-generativeai SDK) for embeddings and generation."""

    def __init__(self) -> None:
        self._ready = False
        self._gen_model = None

    def init(self) -> None:
        if self._ready:
            return
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required")
        genai.configure(api_key=settings.gemini_api_key)
        self._gen_model = genai.GenerativeModel(settings.gemini_model)
        self._ready = True

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.init()
        batch_size = 16
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            result = genai.embed_content(
                model=f"models/{settings.embedding_model}",
                content=batch,
                task_type="retrieval_document",
                output_dimensionality=settings.embedding_dimensions,
            )
            if isinstance(result["embedding"], list) and isinstance(result["embedding"][0], list):
                out.extend(result["embedding"])
            else:
                out.append(result["embedding"])
        return out

    def embed_query(self, text: str) -> list[float]:
        self.init()
        result = genai.embed_content(
            model=f"models/{settings.embedding_model}",
            content=text,
            task_type="retrieval_query",
            output_dimensionality=settings.embedding_dimensions,
        )
        return result["embedding"]

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        self.init()
        assert self._gen_model is not None
        config = genai.GenerationConfig(temperature=0.2, max_output_tokens=2048)
        prompt = f"{system_prompt}\n\n{user_prompt}"
        resp = self._gen_model.generate_content(prompt, generation_config=config)
        if getattr(resp, "text", None):
            return str(resp.text).strip()
        if not resp.candidates:
            return "No response from model."
        parts = []
        for c in resp.candidates:
            if c.content and c.content.parts:
                for p in c.content.parts:
                    if hasattr(p, "text") and p.text:
                        parts.append(p.text)
        return "\n".join(parts).strip() or "No text in model response."

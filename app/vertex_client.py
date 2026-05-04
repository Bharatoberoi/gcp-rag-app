"""Vertex AI embeddings and Gemini generation (GCP-native)."""

from __future__ import annotations

from typing import Sequence

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingModel

from app.config import settings


class VertexRagClient:
    def __init__(self) -> None:
        self._embed_model: TextEmbeddingModel | None = None
        self._gen_model: GenerativeModel | None = None
        self._ready = False

    def init(self) -> None:
        if self._ready:
            return
        if not settings.gcp_project:
            raise RuntimeError("GCP_PROJECT is required for Vertex AI")
        vertexai.init(project=settings.gcp_project, location=settings.gcp_location)
        self._embed_model = TextEmbeddingModel.from_pretrained(settings.embedding_model)
        self._gen_model = GenerativeModel(settings.gemini_model)
        self._ready = True

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.init()
        assert self._embed_model is not None
        batch_size = 16
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            preds = self._embed_model.get_embeddings(batch)
            for p in preds:
                vals = getattr(p, "values", None) or getattr(p, "value", None)
                if vals is None:
                    raise RuntimeError("Unexpected embedding response shape")
                out.append(list(vals))
        return out

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        self.init()
        assert self._gen_model is not None
        cfg = GenerationConfig(temperature=0.2, max_output_tokens=2048)
        prompt = f"{system_prompt}\n\n{user_prompt}"
        resp = self._gen_model.generate_content(prompt, generation_config=cfg)
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

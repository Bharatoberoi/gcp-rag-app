"""Query Rewriting & HyDE (Hypothetical Document Embeddings).

HyDE generates a hypothetical answer to the user's query using the LLM,
then embeds THAT instead of the raw query. This retrieves far more relevant
chunks because the hypothetical answer uses the same vocabulary and style
as the actual documents.

Query Rewriting reformulates ambiguous user queries into clearer, more
specific search queries before retrieval.
"""

from __future__ import annotations

import google.generativeai as genai

from app.config import settings


class QueryEnhancer:
    def __init__(self) -> None:
        self._ready = False
        self._model = None

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)
        self._ready = True

    def rewrite_query(self, question: str) -> str:
        """Rewrite an ambiguous user query into a clear, specific search query."""
        self._ensure_ready()
        assert self._model is not None
        prompt = (
            "You are a search query optimizer. Rewrite the following user question "
            "into a clear, specific search query that will retrieve the most relevant "
            "document passages. Output ONLY the rewritten query, nothing else.\n\n"
            f"User question: {question}\n\nRewritten query:"
        )
        try:
            resp = self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=200),
            )
            rewritten = (resp.text or "").strip()
            return rewritten if rewritten else question
        except Exception:
            return question

    def generate_hypothetical_answer(self, question: str) -> str:
        """Generate a hypothetical document passage that would answer the question (HyDE).

        The hypothetical answer uses terminology and phrasing similar to real documents,
        making its embedding closer to relevant passages than the raw query embedding.
        """
        self._ensure_ready()
        assert self._model is not None
        prompt = (
            "You are a document passage generator. Given the question below, write a "
            "short paragraph (3-5 sentences) that would be a perfect passage from a "
            "document answering this question. Write it as if it's FROM the document, "
            "not as an answer to the user. Output ONLY the passage.\n\n"
            f"Question: {question}\n\nHypothetical document passage:"
        )
        try:
            resp = self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=300),
            )
            hyde = (resp.text or "").strip()
            return hyde if hyde else question
        except Exception:
            return question

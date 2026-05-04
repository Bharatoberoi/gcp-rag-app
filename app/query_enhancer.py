"""Query Rewriting & HyDE (Hypothetical Document Embeddings).

HyDE generates a hypothetical answer to the user's query using the LLM,
then embeds THAT instead of the raw query. This retrieves far more relevant
chunks because the hypothetical answer uses the same vocabulary and style
as the actual documents.

Query Rewriting reformulates ambiguous user queries into clearer, more
specific search queries before retrieval.
"""

from __future__ import annotations

import httpx

from app.config import settings

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _generate(prompt: str, temperature: float = 0.1, max_tokens: int = 300) -> str:
    url = f"{BASE_URL}/models/{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


class QueryEnhancer:
    def rewrite_query(self, question: str) -> str:
        """Rewrite an ambiguous user query into a clear, specific search query."""
        prompt = (
            "You are a search query optimizer. Rewrite the following user question "
            "into a clear, specific search query that will retrieve the most relevant "
            "document passages. Output ONLY the rewritten query, nothing else.\n\n"
            f"User question: {question}\n\nRewritten query:"
        )
        try:
            result = _generate(prompt, temperature=0.1, max_tokens=200)
            return result if result else question
        except Exception:
            return question

    def generate_hypothetical_answer(self, question: str) -> str:
        """Generate a hypothetical document passage that would answer the question (HyDE)."""
        prompt = (
            "You are a document passage generator. Given the question below, write a "
            "short paragraph (3-5 sentences) that would be a perfect passage from a "
            "document answering this question. Write it as if it's FROM the document, "
            "not as an answer to the user. Output ONLY the passage.\n\n"
            f"Question: {question}\n\nHypothetical document passage:"
        )
        try:
            result = _generate(prompt, temperature=0.3, max_tokens=300)
            return result if result else question
        except Exception:
            return question

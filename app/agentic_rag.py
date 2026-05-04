"""Agentic RAG: a reasoning loop that decides when to retrieve, what to query,
and when it has enough information to answer.

Unlike single-shot RAG (query -> retrieve -> answer), Agentic RAG:
1. Analyzes the question to determine what information is needed
2. Formulates targeted sub-queries
3. Retrieves and evaluates results
4. Decides if more retrieval is needed or if it can answer
5. Synthesizes the final answer from all gathered context
"""

from __future__ import annotations

import json

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


class AgenticRag:
    def plan_queries(self, question: str) -> list[str]:
        """Decompose a complex question into targeted sub-queries for retrieval."""
        prompt = (
            "You are a research planner. Given a user question, decompose it into "
            "1-3 specific search queries that together will gather all information "
            "needed to answer comprehensively.\n\n"
            "Rules:\n"
            "- If the question is simple and specific, return just 1 query\n"
            "- If it's complex or multi-part, return 2-3 targeted sub-queries\n"
            "- Each query should target a different aspect of the question\n"
            "- Return ONLY a JSON array of query strings\n\n"
            f"Question: {question}\n\n"
            "JSON array of queries:"
        )
        try:
            text = _generate(prompt, temperature=0.1, max_tokens=300)
            text = text.replace("```json", "").replace("```", "").strip()
            queries = json.loads(text)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:3]
        except Exception:
            pass
        return [question]

    def evaluate_sufficiency(self, question: str, context: str) -> dict:
        """Evaluate if gathered context is sufficient to answer the question."""
        prompt = (
            "You are evaluating whether retrieved context is sufficient to answer a question.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context[:3000]}\n\n"
            "Evaluate:\n"
            "1. Is the context sufficient to provide a complete, accurate answer?\n"
            "2. If not, what specific information is missing?\n"
            "3. What follow-up search query would find the missing information?\n\n"
            'Return ONLY JSON: {"sufficient": true/false, "missing": "...", "follow_up_query": "..."}\n'
            "JSON:"
        )
        try:
            text = _generate(prompt, temperature=0.0, max_tokens=200)
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            return {
                "sufficient": bool(result.get("sufficient", True)),
                "missing": str(result.get("missing", "")),
                "follow_up_query": str(result.get("follow_up_query", "")),
            }
        except Exception:
            return {"sufficient": True, "missing": "", "follow_up_query": ""}

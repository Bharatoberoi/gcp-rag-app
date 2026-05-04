"""Agentic RAG: a reasoning loop that decides when to retrieve, what to query,
and when it has enough information to answer.

Unlike single-shot RAG (query → retrieve → answer), Agentic RAG:
1. Analyzes the question to determine what information is needed
2. Formulates targeted sub-queries
3. Retrieves and evaluates results
4. Decides if more retrieval is needed or if it can answer
5. Synthesizes the final answer from all gathered context

This is essentially ReAct (Reasoning + Acting) applied to RAG.
"""

from __future__ import annotations

import json

import google.generativeai as genai

from app.config import settings
from app.schemas import DocumentChunk


MAX_ITERATIONS = 3


class AgenticRag:
    def __init__(self) -> None:
        self._model = None

    def _ensure_model(self) -> None:
        if self._model:
            return
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)

    def plan_queries(self, question: str) -> list[str]:
        """Decompose a complex question into targeted sub-queries for retrieval."""
        self._ensure_model()
        assert self._model is not None
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
            resp = self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=300),
            )
            text = (resp.text or "").strip()
            text = text.replace("```json", "").replace("```", "").strip()
            queries = json.loads(text)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:3]
        except Exception:
            pass
        return [question]

    def evaluate_sufficiency(self, question: str, context: str) -> dict:
        """Evaluate if gathered context is sufficient to answer the question.

        Returns:
            {"sufficient": bool, "missing": str, "follow_up_query": str}
        """
        self._ensure_model()
        assert self._model is not None
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
            resp = self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=200),
            )
            text = (resp.text or "").strip()
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            return {
                "sufficient": bool(result.get("sufficient", True)),
                "missing": str(result.get("missing", "")),
                "follow_up_query": str(result.get("follow_up_query", "")),
            }
        except Exception:
            return {"sufficient": True, "missing": "", "follow_up_query": ""}

"""End-to-end async RAG pipeline with production techniques."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from app.chunking import ChunkingService
from app.config import settings
from app.document_loaders import load_document
from app.qdrant_store import HybridQdrantStore
from app.schemas import DocumentChunk, SourceCitation
from app.vertex_client import GeminiClient

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class RagPipeline:
    def __init__(self) -> None:
        self.chunker = ChunkingService()
        self.store = HybridQdrantStore()
        self.gemini = GeminiClient()

    async def ingest_bytes_async(self, filename: str, raw: bytes) -> int:
        sections = load_document(filename, raw)
        if not sections:
            return 0
        chunks = self.chunker.sections_to_chunks(Path(filename).name, sections)
        if not chunks:
            return 0
        texts = [c.embedding_input() for c in chunks]
        vectors = await self.gemini.embed_texts_async(texts)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding count mismatch")
        self.store.upsert_chunks(chunks, vectors)
        return len(chunks)

    def delete_document(self, source_name: str) -> None:
        self.store.delete_by_source(source_name)

    def _format_context(self, chunks: list[DocumentChunk]) -> str:
        if not chunks:
            return "No relevant information found."
        parts: list[str] = ["Context from documents:", ""]
        for c in chunks:
            src = c.source_document
            sec = c.section_path or c.section
            line = f"[Source: {src}"
            if sec and sec != src:
                line += f", Section: {sec}"
            line += f", Pages: {c.start_page}-{c.end_page}]"
            parts.append(line)
            parts.append(c.text)
            parts.append("")
        return "\n".join(parts)

    def _extractive_answer(self, question: str, chunks: list[DocumentChunk]) -> str:
        """Fallback answer for local/dev runs without an LLM API key."""
        if not chunks:
            return (
                "I could not find relevant indexed context for that question. "
                "Upload a document first, or try a more specific question."
            )

        parts = [
            "GROQ_API_KEY is not configured, so I cannot generate a polished LLM answer.",
            "Here are the most relevant passages I found:",
            "",
        ]
        for i, c in enumerate(chunks[:3], start=1):
            excerpt = " ".join(c.text.split())
            if len(excerpt) > 700:
                excerpt = excerpt[:700].rstrip() + "..."
            parts.append(f"{i}. {excerpt}")
            parts.append(f"   Source: {c.source_document}, chunk {c.chunk_index}")
        return "\n".join(parts)

    async def _rewrite_query(self, question: str) -> str:
        prompt = (
            "Rewrite this user question into a clear, specific search query. "
            "Output ONLY the rewritten query.\n\n"
            f"Question: {question}\n\nRewritten:"
        )
        try:
            result = await self.gemini.generate_short_async(prompt, temperature=0.1, max_tokens=200)
            return result if result else question
        except Exception:
            return question

    async def _generate_hyde(self, question: str) -> str:
        prompt = (
            "Write a short paragraph (3-5 sentences) that would be a perfect document "
            "passage answering this question. Write as if FROM the document.\n\n"
            f"Question: {question}\n\nPassage:"
        )
        try:
            result = await self.gemini.generate_short_async(prompt, temperature=0.3, max_tokens=300)
            return result if result else question
        except Exception:
            return question

    async def _plan_queries(self, question: str) -> list[str]:
        prompt = (
            "Decompose this question into 1-3 specific search queries. "
            "Return ONLY a JSON array of strings.\n\n"
            f"Question: {question}\n\nJSON:"
        )
        try:
            text = await self.gemini.generate_short_async(prompt, temperature=0.1)
            text = text.replace("```json", "").replace("```", "").strip()
            queries = json.loads(text)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:3]
        except Exception:
            pass
        return [question]

    async def _rerank(self, query: str, chunks: list[DocumentChunk], top_k: int) -> list[DocumentChunk]:
        if not settings.reranker_enabled or not chunks:
            return chunks[:top_k]
        passages = ""
        for i, c in enumerate(chunks[:20]):
            passages += f"[{i}] {c.text[:400]}\n\n"
        prompt = (
            "Rate each passage's relevance to the query (1-10). "
            'Return ONLY JSON: [{"index": 0, "score": 8}, ...]\n\n'
            f"Query: {query}\n\nPassages:\n{passages}\nJSON:"
        )
        try:
            text = await self.gemini.generate_short_async(prompt, temperature=0.0, max_tokens=1024)
            text = text.replace("```json", "").replace("```", "").strip()
            scores = json.loads(text)
            score_map = {item["index"]: item["score"] for item in scores}
            indexed = [(i, score_map.get(i, 0)) for i in range(len(chunks[:20]))]
            indexed.sort(key=lambda x: x[1], reverse=True)
            return [chunks[i] for i, _ in indexed[:top_k]]
        except Exception:
            return chunks[:top_k]

    async def _retrieve(self, query: str, top_k: int) -> list[DocumentChunk]:
        rewritten = await self._rewrite_query(query)
        hyde_passage = await self._generate_hyde(rewritten)
        hyde_embedding = (await self.gemini.embed_texts_async([hyde_passage]))[0]
        retrieve_k = max(top_k, top_k * settings.retrieval_multiplier)
        hits = self.store.search_hybrid(rewritten, hyde_embedding, retrieve_k)
        hits = self.store.expand_adjacent(hits, settings.adjacent_chunk_count)
        return hits

    async def answer_async(self, question: str, top_k: int = 5) -> tuple[str, list[SourceCitation]]:
        """RAG pipeline. Uses agentic features if enabled, otherwise simple retrieval."""
        self.gemini.init()

        if settings.agentic_mode:
            all_chunks = await self._agentic_retrieve(question, top_k)
            ranked = await self._rerank(question, all_chunks, top_k)
        else:
            q_embedding = (await self.gemini.embed_texts_async([question]))[0]
            retrieve_k = max(top_k, top_k * settings.retrieval_multiplier)
            hits = self.store.search_hybrid(question, q_embedding, retrieve_k)
            hits = self.store.expand_adjacent(hits, settings.adjacent_chunk_count)
            ranked = hits[:top_k]

        if settings.groq_api_key:
            context = self._format_context(ranked)
            system = (
                "You are a careful assistant. Answer using ONLY the provided context. "
                "If the context is insufficient, say so. Cite sources by document name when possible."
            )
            user = f"Question: {question}\n\n{context}"
            answer = await self.gemini.generate_answer_async(system, user)
        else:
            answer = self._extractive_answer(question, ranked)

        sources = [
            SourceCitation(
                source_document=c.source_document,
                section_path=c.section_path,
                chunk_index=c.chunk_index,
                excerpt=c.text[:400] + ("..." if len(c.text) > 400 else ""),
            )
            for c in ranked
        ]
        return answer, sources

    async def _agentic_retrieve(self, question: str, top_k: int) -> list[DocumentChunk]:
        """Full agentic retrieval: plan, HyDE, multi-query, rerank."""
        sub_queries = await self._plan_queries(question)
        all_chunks: list[DocumentChunk] = []
        seen_ids: set[str] = set()
        for sub_q in sub_queries:
            hits = await self._retrieve(sub_q, top_k)
            for chunk in hits:
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)
        return all_chunks

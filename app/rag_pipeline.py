"""End-to-end RAG orchestration with production techniques:

1. Hybrid Search (Dense + Sparse BM25 with RRF fusion)
2. Query Rewriting & HyDE (Hypothetical Document Embeddings)
3. Contextual Chunking (semantic splitting + chunk headers)
4. Reranking (Gemini-based or external cross-encoder)
5. Agentic RAG (multi-step reasoning loop)
6. Adjacent chunk expansion
"""

from __future__ import annotations

from pathlib import Path

from app.agentic_rag import AgenticRag
from app.chunking import ChunkingService
from app.config import settings
from app.document_loaders import load_document
from app.qdrant_store import HybridQdrantStore
from app.query_enhancer import QueryEnhancer
from app.reranker_client import rerank_chunks_sync
from app.schemas import DocumentChunk, SourceCitation
from app.vertex_client import GeminiClient


class RagPipeline:
    def __init__(self) -> None:
        self.chunker = ChunkingService()
        self.store = HybridQdrantStore()
        self.gemini = GeminiClient()
        self.enhancer = QueryEnhancer()
        self.agent = AgenticRag()

    def ingest_bytes(self, filename: str, raw: bytes) -> int:
        sections = load_document(filename, raw)
        if not sections:
            return 0
        chunks = self.chunker.sections_to_chunks(Path(filename).name, sections)
        if not chunks:
            return 0
        texts = [c.embedding_input() for c in chunks]
        vectors = self.gemini.embed_texts(texts)
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

    def _retrieve_for_query(self, query: str, top_k: int) -> list[DocumentChunk]:
        """Run the full retrieval pipeline for a single query."""
        rewritten = self.enhancer.rewrite_query(query)
        hyde_passage = self.enhancer.generate_hypothetical_answer(rewritten)
        hyde_embedding = self.gemini.embed_texts([hyde_passage])[0]
        retrieve_k = max(top_k, top_k * settings.retrieval_multiplier)
        hits = self.store.search_hybrid(rewritten, hyde_embedding, retrieve_k)
        hits = self.store.expand_adjacent(hits, settings.adjacent_chunk_count)
        return hits

    def answer_sync(self, question: str, top_k: int = 5) -> tuple[str, list[SourceCitation]]:
        """Agentic RAG: plan queries, retrieve iteratively, rerank, generate answer."""
        self.gemini.init()

        sub_queries = self.agent.plan_queries(question)

        all_chunks: list[DocumentChunk] = []
        seen_ids: set[str] = set()

        for sub_q in sub_queries:
            hits = self._retrieve_for_query(sub_q, top_k)
            for chunk in hits:
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)

        context_text = self._format_context(all_chunks)
        eval_result = self.agent.evaluate_sufficiency(question, context_text)

        if not eval_result["sufficient"] and eval_result["follow_up_query"]:
            follow_up_hits = self._retrieve_for_query(eval_result["follow_up_query"], top_k)
            for chunk in follow_up_hits:
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)

        ranked = rerank_chunks_sync(question, all_chunks, top_k)
        context = self._format_context(ranked)

        system = (
            "You are a careful assistant. Answer using ONLY the provided context. "
            "If the context is insufficient, say so. Cite sources by document name when possible."
        )
        user = f"Question: {question}\n\n{context}"
        answer = self.gemini.generate_answer(system, user)

        sources = [
            SourceCitation(
                source_document=c.source_document,
                section_path=c.section_path,
                chunk_index=c.chunk_index,
                excerpt=c.text[:400] + ("…" if len(c.text) > 400 else ""),
            )
            for c in ranked
        ]
        return answer, sources

"""End-to-end RAG orchestration."""

from __future__ import annotations

from pathlib import Path

from app.chunking import ChunkingService
from app.config import settings
from app.document_loaders import load_document
from app.qdrant_store import HybridQdrantStore
from app.reranker_client import rerank_chunks
from app.schemas import DocumentChunk, SourceCitation
from app.vertex_client import GeminiClient


class RagPipeline:
    def __init__(self) -> None:
        self.chunker = ChunkingService()
        self.store = HybridQdrantStore()
        self.gemini = GeminiClient()

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

    async def answer(self, question: str, top_k: int = 5) -> tuple[str, list[SourceCitation]]:
        self.gemini.init()
        q_emb = self.gemini.embed_texts([question])[0]
        retrieve_k = max(top_k, top_k * settings.retrieval_multiplier)
        hits = self.store.search_hybrid(question, q_emb, retrieve_k)
        hits = self.store.expand_adjacent(hits, settings.adjacent_chunk_count)
        ranked = await rerank_chunks(question, hits, top_k)
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

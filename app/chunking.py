"""Token-based chunking using tiktoken (same idea as reference DocumentExtractor)."""

from __future__ import annotations

import uuid

import tiktoken

from app.config import settings
from app.schemas import DocumentChunk


class ChunkingService:
    def __init__(self) -> None:
        self._enc = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text or ""))

    def split_text(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        if not text.strip():
            return []
        tokens = self._enc.encode(text)
        if len(tokens) <= max_tokens:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            piece = self._enc.decode(tokens[start:end])
            chunks.append(piece)
            if end >= len(tokens):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def sections_to_chunks(
        self,
        source_document: str,
        sections: list[dict],
    ) -> list[DocumentChunk]:
        """sections: {text, section_path, section, start_page, end_page, metadata}"""
        max_t = settings.max_chunk_tokens
        overlap = settings.chunk_overlap_tokens
        flat: list[DocumentChunk] = []
        for sec in sections:
            parts = self.split_text(sec["text"], max_t, overlap)
            for part in parts:
                flat.append(
                    DocumentChunk(
                        id="",
                        text=part.strip(),
                        source_document=source_document,
                        chunk_index=0,
                        chunk_total=0,
                        start_page=int(sec.get("start_page", 0)),
                        end_page=int(sec.get("end_page", 0)),
                        section=str(sec.get("section", "")),
                        section_path=str(sec.get("section_path", "")),
                        metadata=dict(sec.get("metadata", {})),
                    )
                )
        total = len(flat)
        out: list[DocumentChunk] = []
        for idx, c in enumerate(flat):
            c.chunk_index = idx
            c.chunk_total = total
            c.id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{c.source_document}::{c.section_path}::{c.chunk_index}",
                )
            )
            out.append(c)
        return out

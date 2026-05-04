"""Contextual Chunking: semantic paragraph-based splitting with chunk headers.

Instead of naive token-count splitting, this chunker:
1. Splits by paragraphs/sentences first (semantic boundaries)
2. Merges small paragraphs together up to the token limit
3. Prepends a "chunk header" with document name + section path so each
   chunk makes sense in isolation without surrounding context
"""

from __future__ import annotations

import re
import uuid

import tiktoken

from app.config import settings
from app.schemas import DocumentChunk

PARAGRAPH_SEPARATORS = re.compile(r"\n\s*\n")
SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")


class ChunkingService:
    def __init__(self) -> None:
        self._enc = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text or ""))

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text at paragraph boundaries (double newlines)."""
        paragraphs = PARAGRAPH_SEPARATORS.split(text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_long_paragraph(self, paragraph: str, max_tokens: int) -> list[str]:
        """Split a single long paragraph at sentence boundaries."""
        sentences = SENTENCE_ENDINGS.split(paragraph)
        if not sentences:
            return [paragraph]

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = self.count_tokens(sentence)
            if current and (current_tokens + s_tokens) > max_tokens:
                chunks.append(" ".join(current))
                current = [sentence]
                current_tokens = s_tokens
            else:
                current.append(sentence)
                current_tokens += s_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _semantic_split(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Split text semantically: by paragraphs first, then merge small ones."""
        if not text.strip():
            return []

        if self.count_tokens(text) <= max_tokens:
            return [text]

        paragraphs = self._split_into_paragraphs(text)
        if not paragraphs:
            return [text]

        segments: list[str] = []
        for para in paragraphs:
            if self.count_tokens(para) > max_tokens:
                segments.extend(self._split_long_paragraph(para, max_tokens))
            else:
                segments.append(para)

        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for segment in segments:
            seg_tokens = self.count_tokens(segment)
            if current_parts and (current_tokens + seg_tokens) > max_tokens:
                chunks.append("\n\n".join(current_parts))
                overlap_parts: list[str] = []
                overlap_tokens = 0
                for p in reversed(current_parts):
                    p_tokens = self.count_tokens(p)
                    if overlap_tokens + p_tokens > overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_tokens += p_tokens
                current_parts = overlap_parts + [segment]
                current_tokens = overlap_tokens + seg_tokens
            else:
                current_parts.append(segment)
                current_tokens += seg_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    def _build_chunk_header(self, source_document: str, section_path: str, section: str) -> str:
        """Build a contextual header prepended to each chunk so it's self-contained."""
        parts: list[str] = []
        if source_document:
            parts.append(f"Document: {source_document}")
        path = section_path or section
        if path and path != source_document:
            parts.append(f"Section: {path}")
        return " | ".join(parts)

    def sections_to_chunks(
        self,
        source_document: str,
        sections: list[dict],
    ) -> list[DocumentChunk]:
        """Convert document sections into contextual chunks.

        Each chunk gets a header prepended so it can stand alone without
        needing surrounding context to make sense.
        """
        max_t = settings.max_chunk_tokens
        overlap = settings.chunk_overlap_tokens
        flat: list[DocumentChunk] = []

        for sec in sections:
            header = self._build_chunk_header(
                source_document,
                str(sec.get("section_path", "")),
                str(sec.get("section", "")),
            )
            header_tokens = self.count_tokens(header + "\n\n") if header else 0
            effective_max = max(50, max_t - header_tokens)

            parts = self._semantic_split(sec["text"], effective_max, overlap)
            for part in parts:
                text_with_context = f"{header}\n\n{part}" if header else part
                flat.append(
                    DocumentChunk(
                        id="",
                        text=text_with_context.strip(),
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

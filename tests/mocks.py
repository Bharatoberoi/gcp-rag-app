"""Test doubles for RAG pipeline."""

from __future__ import annotations

from app.schemas import SourceCitation


class MockRagPipeline:
    def ingest_bytes(self, filename: str, raw: bytes) -> int:
        if not raw:
            return 0
        return 2

    def delete_document(self, source_name: str) -> None:
        self._deleted = source_name

    async def answer(self, question: str, top_k: int = 5):
        src = [
            SourceCitation(
                source_document="mock.pdf",
                section_path="sec",
                chunk_index=0,
                excerpt="excerpt",
            )
        ]
        return f"Echo: {question[:80]}", src
